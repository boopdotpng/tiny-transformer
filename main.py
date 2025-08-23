from tinygrad import Tensor, TinyJit, nn, GlobalCounters, dtypes
import sys
from tinygrad.helpers import trange, getenv
import youtokentome as yttm
from pathlib import Path
import math
import random

D_emb = 192 
N_HEADS = 6 
D_head =  D_emb // N_HEADS
d_ff = 4 * D_emb
n_layers = 6 

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
  def __init__(self, d_ff=d_ff, dropout_p=0.0):
    self.fc1 = nn.Linear(D_emb, d_ff, bias=True)
    self.fc2 = nn.Linear(d_ff, D_emb, bias=True)
    self.dropout_p = dropout_p

  def __call__(self, x: Tensor) -> Tensor:
    x = self.fc1(x)     # (B, T, d_ff)
    x = x.gelu()        # nonlinearity
    x = self.fc2(x)     # (B, T, D_emb)
    if self.dropout_p and self.dropout_p > 0.0:
      x = x.dropout(p=self.dropout_p)
    return x

class SelfAttention:
  def __init__(self):
    self.qkv = nn.Linear(D_emb, 3*D_emb, bias=True)
    self.proj_out = nn.Linear(D_emb, D_emb, bias=True)
  def __call__(self, x: Tensor, mask: Tensor) -> Tensor: 
    B, T, _ = x.shape
    # (B, T, 3*D) → (B, T, 3, H, D_HEAD) → (3, B, H, T, D_HEAD)
    qkv = self.qkv(x).reshape(B, T, 3, N_HEADS, D_head).permute(2,0,3,1,4)
    q, k, v = qkv[0], qkv[1], qkv[2]                       # each (B, H, T, D_HEAD)

    # scaled dot-product
    att = (q @ k.transpose(2,3)) * (1.0 / math.sqrt(D_head))   # (B, H, T, T)
    att = att.masked_fill(mask == 0, -1e9).softmax()

    # (B, H, T, T) @ (B, H, T, D_HEAD) → (B, H, T, D_HEAD)
    out = att @ v
    out = out.transpose(1,2).reshape(B, T, D_emb)              # concat heads

    return self.proj_out(out)          # (B, T, D_emb)

class Block:
  def __init__(self):
    self.ln1 = nn.RMSNorm(D_emb)
    self.attn = SelfAttention()
    self.ln2 = nn.RMSNorm(D_emb)
    self.mlp = MLP()
  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    x += self.attn(self.ln1(x), mask)
    x += self.mlp(self.ln2(x))
    return x

class Transformer: 
  def __init__(self):
    self.wte = nn.Embedding(bpe.vocab_size(), D_emb) # token embeddings 
    self.wpe = nn.Embedding(N_CTX, D_emb) # positional embeddings (learned, added to wte)
    self.blocks = [Block() for _ in range(n_layers)]
    self.ln_f = nn.RMSNorm(D_emb)
  def __call__(self, token_ids: Tensor) -> Tensor:
    _, T = token_ids.shape
    pos = Tensor.arange(T).unsqueeze(0)
    x = (self.wte(token_ids) + self.wpe(pos))
    mask = Tensor.tril(Tensor.ones((T,T))).unsqueeze(0).unsqueeze(1) # (1, T, T)
    for blk in self.blocks: 
      x = blk(x, mask)
    x = self.ln_f(x) 
    logits = (x @ self.wte.weight.transpose(1,0)) 
    return logits

def make_blocks(token_ids, block_size=256, test_frac=0.1, stride=192):
  L = len(token_ids)
  starts = list(range(0, L - block_size - 1, stride))
  X = [token_ids[i:i+block_size] for i in starts]
  y = [token_ids[i+1:i+block_size+1] for i in starts]
  pairs = list(zip(X, y))
  random.shuffle(pairs)
  X, y = zip(*pairs)
  split = int(len(X) * (1.0 - test_frac))
  return Tensor(list(X[:split]), dtype=dtypes.int32), Tensor(list(y[:split]), dtype=dtypes.int32), Tensor(list(X[split:]), dtype=dtypes.int32), Tensor(list(y[split:]), dtype=dtypes.int32)


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

  @Tensor.train(False)
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
    generate(epoch=4999, prompt="to be or not to be ")
    exit()
  
  Path("checkpoints").mkdir(exist_ok=True)

  model = Transformer()
  
  opt = nn.optim.Adam(nn.state.get_parameters(model), lr=3e-4)

  token_ids = bpe.encode([tiny_shakes_text], output_type=yttm.OutputType.ID)[0]
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids)
  losses = []

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(getenv("BS", 128), high=int(X_train.shape[0]))
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  @Tensor.train(False)
  def get_test_acc() -> Tensor:
    val_loss = model(X_test).sparse_categorical_crossentropy(Y_test)
    val_ppl = val_loss.exp()
    return val_ppl

  test_acc = float('nan')
  for i in (t:=trange(5000)):
    GlobalCounters.reset()
    loss = train_step() 
    losses.append(loss.item())
    if i%10 == 9:
      test_acc = get_test_acc().item()
    if i%500 == 499:
      state_dict = nn.state.get_state_dict(model)
      nn.state.safe_save(state_dict, f"checkpoints/model{i}.safetensors")
    t.set_description(f"loss: {loss.item():6.2f} val_ppl: {test_acc:.2f}")

  open("losses.txt", "w").write(','.join([str(x) for x in losses]))