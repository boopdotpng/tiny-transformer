from tinygrad import Tensor, TinyJit, nn, GlobalCounters
import sys
from tinygrad.helpers import trange
import youtokentome as yttm
from pathlib import Path
import math

N_CTX = 256 
D_emb = 256
D_head = 256
d_ff = 4 * D_emb
n_layers = 6

NL = "<|nl|>"

text_path = Path(__file__).parent / "shakespeare.txt"
transformed_text_path = Path(__file__).parent / "shakespeare_transformed.txt"

tiny_shakes_text_raw = (Path(__file__).parent / "shakespeare.txt").read_text("utf-8")
tiny_shakes_text = tiny_shakes_text_raw.replace("\r\n", "\n").replace("\n", f" {NL} ")

# train tokenizer on transformed text
if not (Path(__file__).parent / "shakes.model").exists():
  with open(transformed_text_path, "w", encoding="utf-8") as f:
    f.write(tiny_shakes_text)
  yttm.BPE.train(data=str(transformed_text_path), model="shakes.model", vocab_size=512)
bpe = yttm.BPE(model=str(Path(__file__).parent / "shakes.model"))

class MLP:
  def __init__(self):
    self.fc1 = nn.Linear(D_emb, d_ff)
    self.fc2 = nn.Linear(d_ff, D_emb)
  def __call__(self, x: Tensor) -> Tensor:
    return self.fc2(self.fc1(x).gelu())

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
    weights = scores.masked_fill(mask == 0, -1e9).softmax() # (B, T, T)

    return self.o(weights @ v) # (B, T, D_emb) same as input, so we can chain to next module

class Block:
  def __init__(self):
    self.ln1 = nn.RMSNorm(D_emb)
    self.attn = SelfAttention()
    self.ln2 = nn.RMSNorm(D_emb)
    self.mlp = MLP()
  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    x = x + self.attn(self.ln1(x), mask)
    x = x + self.mlp(self.ln2(x)) 
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
    x = self.wte(token_ids) + self.wpe(pos)
    mask = Tensor.tril(Tensor.ones((T,T))).unsqueeze(0) # (1, T, T)
    for blk in self.blocks: 
      x = blk(x, mask)
    x = self.ln_f(x) 
    logits = x @ self.wte.weight.transpose(1,0)
    return logits

def make_blocks(token_ids, block_size: int = 256, stride: int = 64) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  X, y = [], []
  for i in range(0, len(token_ids) - block_size -1, stride):
    x = token_ids[i : i+block_size]
    Y = token_ids[i+1 : i+block_size+1]
    X.append(x)
    y.append(Y)
  num_samples = len(X)
  train_size = int(num_samples * 0.9)
  X_train, Y_train = X[:train_size], y[:train_size]
  X_test,  Y_test  = X[train_size:], y[train_size:]
  return Tensor(X_train), Tensor(Y_train), Tensor(X_test), Tensor(Y_test)

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
      
      full_text = bpe.decode([generated_tokens])[0]
      
      full_text = full_text.replace(NL, "\n")
      
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
    generate(epoch=899, prompt="to be or not to be ")
    exit()

  model = Transformer()
  opt = nn.optim.AdamW(nn.state.get_parameters(model))
  token_ids = bpe.encode([tiny_shakes_text], output_type=yttm.OutputType.ID)[0]
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids)

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(128, high=X_train.shape[0])
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).mean().backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  def get_test_acc() -> Tensor:
    val_loss = model(X_test).sparse_categorical_crossentropy(Y_test).mean()
    val_ppl = val_loss.exp()
    return val_ppl

  test_acc = float('nan')
  for i in (t:=trange(1000)):
    GlobalCounters.reset()
    loss = train_step() 
    t.set_description(f"loss: {loss.item():6.2f} test_accuracy(ppl): {test_acc:.2f}")
    if i%10 == 9:
      test_acc = get_test_acc().item()
      state_dict = nn.state.get_state_dict(model)
    if i%500 == 499:
      nn.state.safe_save(state_dict, f"checkpoints/model{i}.safetensors")
