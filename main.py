from tinygrad import Tensor, TinyJit, nn, GlobalCounters
import sys
from tinygrad.helpers import trange
import youtokentome as yttm
from pathlib import Path
import math
import random
from typing import Tuple

N_CTX = 256 
D_emb = 128 
D_head = 128 
d_ff = 4 * D_emb
n_layers = 4

NL = "<|nl|>"

tiny_shakes_text_raw = (Path(__file__).parent / "shakespeare.txt").read_text("utf-8")
# since shakespeare.txt already uses NL tokens, use raw text directly
tiny_shakes_text = tiny_shakes_text_raw
model_path = Path(__file__).parent / "shakes.model"
data_path = Path(__file__).parent / "shakespeare.txt"
if not model_path.exists():
  yttm.BPE.train(data=str(data_path), model=str(model_path), vocab_size=1024)
bpe = yttm.BPE(model=str(model_path))

class MLP:
  def __init__(self):
    self.w1 = nn.Linear(D_emb, d_ff, bias=False)
    self.w2 = nn.Linear(D_emb, d_ff, bias=False)
    self.w3 = nn.Linear(d_ff, D_emb, bias=False)
  def __call__(self, x: Tensor) -> Tensor:
    return self.w3(self.w1(x) * self.w2(x).sigmoid()).dropout(p=0.1)

class SelfAttention:
  def __init__(self):
    self.q = nn.Linear(D_emb, D_head, bias=False)
    self.k = nn.Linear(D_emb, D_head, bias=False)
    self.v = nn.Linear(D_emb, D_head, bias=False)
    self.o = nn.Linear(D_head, D_emb, bias=False)
  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    q = self.q(x)
    k = self.k(x)
    v = self.v(x) # all of shape (B, T, D_head)

    scores: Tensor = (q @ k.transpose(1,2)) * (1.0 / math.sqrt(D_head)) #  (B, T, T)
    weights = scores.masked_fill(mask == 0, -1e9).softmax().dropout(p=0.1) # (B, T, T)

    return self.o(weights @ v) # (B, T, D_emb) same as input, so we can chain to next module

class Block:
  def __init__(self):
    self.ln1 = nn.RMSNorm(D_emb)
    self.attn = SelfAttention()
    self.ln2 = nn.RMSNorm(D_emb)
    self.mlp = MLP()
  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    x += self.attn(self.ln1(x), mask).dropout(p=0.1)
    x += self.mlp(self.ln2(x)).dropout(p=0.1)
    return x

class Transformer: 
  def __init__(self):
    self.wte = nn.Embedding(bpe.vocab_size(), D_emb) # token embeddings 
    self.wpe = nn.Embedding(N_CTX, D_emb) # positional embeddings (learned, added to wte)
    self.blocks = [Block() for _ in range(n_layers)]
    self.ln_f = nn.RMSNorm(D_emb)
    self.head_bias = Tensor.zeros((bpe.vocab_size(),), requires_grad=True)
  def __call__(self, token_ids: Tensor) -> Tensor:
    _, T = token_ids.shape
    pos = Tensor.arange(T).unsqueeze(0)
    x = (self.wte(token_ids) + self.wpe(pos)).dropout(p=0.1)
    mask = Tensor.tril(Tensor.ones((T,T))).unsqueeze(0) # (1, T, T)
    for blk in self.blocks: 
      x = blk(x, mask)
    x = self.ln_f(x) 
    logits = (x @ self.wte.weight.transpose(1,0)) + self.head_bias
    return logits

def make_blocks(token_ids, block_size=256, test_frac=0.1, stride=64):
  L = len(token_ids)
  starts = list(range(0, L - block_size - 1, stride))
  X = [token_ids[i:i+block_size] for i in starts]
  y = [token_ids[i+1:i+block_size+1] for i in starts]
  pairs = list(zip(X, y))
  random.shuffle(pairs)
  X, y = zip(*pairs)
  split = int(len(X) * (1.0 - test_frac))
  return Tensor(list(X[:split])), Tensor(list(y[:split])), Tensor(list(X[split:])), Tensor(list(y[split:]))


def generate(epoch: int, prompt: str = "", max_tokens:int = 256, temperature: float = 1.0, top_k: int = 50):
  state_dict = nn.state.safe_load(f"checkpoints/model{epoch}.safetensors")
  model = Transformer()
  nn.state.load_state_dict(model, state_dict=state_dict)
  ctx = list(bpe.encode([prompt], output_type=yttm.OutputType.ID)[0])

  def _sample_next(next_token_logits: Tensor, temperature: float = 0.7, top_k: int = 50) -> int:
    import numpy as np

    if temperature is None or temperature <= 0:
      return int(next_token_logits.argmax().item())

    logits = next_token_logits.realize().numpy().astype(np.float64)

    logits = logits / max(1e-8, float(temperature))

    V = logits.shape[0]

    if top_k is not None and top_k > 0 and top_k < V:
      kth = top_k - 1
      idx = np.argpartition(-logits, kth)[:top_k]      
      sub_logits = logits[idx]

      sub_logits = sub_logits - sub_logits.max()
      probs = np.exp(sub_logits)
      probs /= probs.sum()

      choice_in_subset = np.random.choice(top_k, p=probs)
      return int(idx[choice_in_subset])
    else:
      logits = logits - logits.max()
      probs = np.exp(logits)
      probs /= probs.sum()
      return int(np.random.choice(V, p=probs))

  if prompt:
    sys.stdout.write(prompt)
    sys.stdout.flush()

  def _infer(input_ids: Tensor):
    logits = model(input_ids)            
    next_token_logits = logits[0, -1]   
    return _sample_next(next_token_logits, temperature=temperature, top_k=top_k)

  generated_tokens = []
  last_decoded_len = 0
  
  try:
    for _ in range(max_tokens):
      input_ids = Tensor([ctx[-N_CTX:]])
      next_id = int(_infer(input_ids))
      ctx.append(next_id)
      generated_tokens.append(next_id)
      
      full_text = ''.join(bpe.decode([generated_tokens])).replace(NL, "\n")
      
      new_text = full_text[last_decoded_len:]
      if new_text:
        sys.stdout.write(new_text)
        sys.stdout.flush()
        last_decoded_len = len(full_text)

  except KeyboardInterrupt:
    print("")
    pass
    
if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "generate":
    generate(epoch=999, prompt="to be or not to be ")
    exit()

  model = Transformer()
  
  # print param count for fun
  print("parameter count:", sum([p.numel() for p in nn.state.get_parameters(model)]))
  
  # need to skip weight decay on some parameters
  all_params = nn.state.get_parameters(model)
  no_wd = [v for k,v in nn.state.get_state_dict(model).items() if ("bias" in k) or ("ln" in k) or ("wte" in k) or ("wpe" in k)]
  wd_params = [p for p in all_params if p not in set(no_wd)]
  print(len(no_wd), len(wd_params))

  opt_wd = nn.optim.AdamW(wd_params, lr=1e-4, b1=0.9, b2=0.95, weight_decay=0.01)
  opt_no_wd = nn.optim.Adam(no_wd, lr=1e-4, b1=0.9, b2=0.95)
  optimizer = nn.optim.OptimizerGroup(opt_wd, opt_no_wd)

  token_ids = bpe.encode([tiny_shakes_text], output_type=yttm.OutputType.ID)[0]
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids)
  losses = []

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    optimizer.zero_grad()
    samples = Tensor.randint(128, high=X_train.shape[0])
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).mean().backward()
    return loss.realize(*optimizer.schedule_step())

  @TinyJit
  def get_test_acc() -> Tensor:
    val_loss = model(X_test).sparse_categorical_crossentropy(Y_test).mean()
    val_ppl = val_loss.exp()
    return val_ppl

  test_acc = float('nan')
  for i in (t:=trange(1000)):
    GlobalCounters.reset()
    loss = train_step() 
    losses.append(loss.item())
    if i%10 == 9:
      test_acc = get_test_acc().item()
    if i%100 == 99:
      state_dict = nn.state.get_state_dict(model)
      nn.state.safe_save(state_dict, f"checkpoints/model{i}.safetensors")
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy(ppl): {test_acc:.2f}")

  open("losses.txt", "w").write(','.join([str(x) for x in losses]))