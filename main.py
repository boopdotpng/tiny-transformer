from tinygrad import Tensor, TinyJit, nn, GlobalCounters
import sys, os
from tinygrad.helpers import trange
import sentencepiece as spm
from pathlib import Path
import math

N_CTX  = 512 
D_emb  = 256
D_head = 256          
d_ff   = 4 * D_emb
n_layers = 6
VOCAB_SIZE = 1024 # tokenizer size

TEXT_PATH = Path(__file__).parent / "shakespeare.txt"
SPM_MODEL = Path(__file__).parent / "shakes_sp.model"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

if not SPM_MODEL.exists():
  spm.SentencePieceTrainer.Train(
      input=str(TEXT_PATH),
      model_prefix=str(SPM_MODEL.with_suffix("")),
      vocab_size=VOCAB_SIZE,
      model_type="bpe",
      byte_fallback=True,
      normalization_rule_name="identity",
      add_dummy_prefix=False,
      remove_extra_whitespaces=False,
  )

sp = spm.SentencePieceProcessor(model_file=str(SPM_MODEL))

def sp_encode_ids(text: str) -> list[int]: return sp.encode(text.replace("\r\n", "\n"), out_type=int)
def sp_decode_ids(ids: list[int]) -> str: return sp.decode(ids)

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
    v = self.v(x)  # (B, T, D_head)
    scores: Tensor = (q @ k.transpose(1,2)) * (1.0 / math.sqrt(D_head))  # (B, T, T)
    weights = scores.masked_fill(mask == 0, -1e9).softmax()              # (B, T, T)
    return self.o(weights @ v)  # (B, T, D_emb)

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
    self.wte = nn.Embedding(sp.get_piece_size(), D_emb)  
    self.wpe = nn.Embedding(N_CTX, D_emb)
    self.blocks = [Block() for _ in range(n_layers)]
    self.ln_f = nn.RMSNorm(D_emb)
  def __call__(self, token_ids: Tensor) -> Tensor:
    _, T = token_ids.shape
    pos = Tensor.arange(T).unsqueeze(0)
    x = self.wte(token_ids) + self.wpe(pos)
    mask = Tensor.tril(Tensor.ones((T, T))).unsqueeze(0)  # (1, T, T)
    for blk in self.blocks:
      x = blk(x, mask)
    x = self.ln_f(x)
    logits = x @ self.wte.weight.transpose(1, 0)
    return logits

def make_blocks(token_ids, block_size: int = 256, stride: int = 64):
  X, y = [], []
  for i in range(0, len(token_ids) - block_size - 1, stride):
    x = token_ids[i : i + block_size]
    Y = token_ids[i + 1 : i + block_size + 1]
    X.append(x)
    y.append(Y)
  num_samples = len(X)
  train_size = int(num_samples * 0.9)
  X_train, Y_train = X[:train_size], y[:train_size]
  X_test,  Y_test  = X[train_size:], y[train_size:]
  return Tensor(X_train), Tensor(Y_train), Tensor(X_test), Tensor(Y_test)

def generate(epoch: int, prompt: str = "", max_tokens: int = 256, temperature: float = 1.0, top_k: int = 50):
  state_dict = nn.state.safe_load(str(CHECKPOINT_DIR / f"model{epoch}.safetensors"))
  model = Transformer()
  nn.state.load_state_dict(model, state_dict=state_dict)

  ctx = sp_encode_ids(prompt)

  def _sample_next(next_token_logits: Tensor, temperature: float = 0.7, top_k: int = 50) -> int:
    import numpy as np
    if temperature is None or temperature <= 0:
      return int(next_token_logits.argmax().item())
    logits = next_token_logits.realize().numpy().astype(np.float64)
    logits = logits / max(1e-8, float(temperature))
    V = logits.shape[0]

    if top_k is not None and 0 < top_k < V:
      kth = top_k - 1
      idx = np.argpartition(-logits, kth)[:top_k]
      sub_logits = logits[idx]
      sub_logits = sub_logits - sub_logits.max()
      probs = np.exp(sub_logits); probs /= probs.sum()
      choice_in_subset = np.random.choice(top_k, p=probs)
      return int(idx[choice_in_subset])
    else:
      logits = logits - logits.max()
      probs = np.exp(logits); probs /= probs.sum()
      return int(np.random.choice(V, p=probs))

  if prompt:
    sys.stdout.write(prompt)
    sys.stdout.flush()

  def _infer(input_ids: Tensor):
    logits = model(input_ids)
    next_token_logits = logits[0, -1]
    return _sample_next(next_token_logits, temperature=temperature, top_k=top_k)

  generated_ids = []
  last_printed_text_len = 0

  try:
    for _ in range(max_tokens):
      input_ids = Tensor([ctx[-N_CTX:]])
      next_id = int(_infer(input_ids))
      ctx.append(next_id)
      generated_ids.append(next_id)

      full_text = sp_decode_ids(generated_ids)
      new_text = full_text[last_printed_text_len:]
      if new_text:
        sys.stdout.write(new_text)
        sys.stdout.flush()
        last_printed_text_len = len(full_text)
  except KeyboardInterrupt:
    print("")
    pass

if __name__ == "__main__":
  if len(sys.argv) > 2 and sys.argv[1] == "generate":
    generate(epoch=int(sys.argv[2]), prompt="to be or not to be ")
    sys.exit(0)

  raw_text = TEXT_PATH.read_text(encoding="utf-8")
  token_ids = sp_encode_ids(raw_text)

  X_train, Y_train, X_test, Y_test = make_blocks(token_ids, block_size=N_CTX, stride=128)

  model = Transformer()
  opt = nn.optim.AdamW(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(128, high=X_train.shape[0])
    loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).mean().backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  def get_val_ppl() -> Tensor:
    val_loss = model(X_test).sparse_categorical_crossentropy(Y_test).mean()
    return val_loss.exp()

  test_ppl = float('nan')
  for i in (t := trange(2000)):
    GlobalCounters.reset()
    loss = train_step()
    t.set_description(f"loss: {loss.item():6.2f} val_ppl: {test_ppl:.2f}")
    if i % 50 == 49:
      test_ppl = get_val_ppl().item()
    if i % 500 == 499:
      state_dict = nn.state.get_state_dict(model)
      nn.state.safe_save(state_dict, str(CHECKPOINT_DIR / f"model{i}.safetensors"))
