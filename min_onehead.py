# min_onehead_char.py
from tinygrad import Tensor, nn, dtypes
from tinygrad.helpers import trange
from pathlib import Path
import math

# ----------------------------
# Hyperparams (tiny & stable)
# ----------------------------
N_CTX = 128
D_EMB = 64           # single head => head_dim == D_EMB
N_LAYERS = 1         # keep 1 block for the first sanity check
BATCH_SIZE = 64
STEPS = 1500
LR = 3e-4

# ----------------------------
# Data: char-level Shakespeare
# ----------------------------
tiny_shakes_text = (Path(__file__).parent / "shakespeare.txt").read_text("utf-8")

UNK_CHAR = '\ufffd'
chars = sorted(set(tiny_shakes_text))
if UNK_CHAR not in chars:
  chars.append(UNK_CHAR)

char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for c,i in char2idx.items()}
VOCAB_SIZE = len(chars)

def text_to_token_ids(text: str):
  unk_idx = char2idx[UNK_CHAR]
  return [char2idx.get(ch, unk_idx) for ch in text]

def make_blocks(token_ids, block_size=N_CTX, test_frac=0.1, stride=N_CTX//2):
  L = len(token_ids)
  starts = list(range(0, L - block_size - 1, stride))
  X = [token_ids[i:i+block_size] for i in starts]
  y = [token_ids[i+1:i+block_size+1] for i in starts]
  split = int(len(X) * (1.0 - test_frac))
  return (Tensor(X[:split], dtype=dtypes.int32),
          Tensor(y[:split], dtype=dtypes.int32),
          Tensor(X[split:], dtype=dtypes.int32),
          Tensor(y[split:], dtype=dtypes.int32))

# ----------------------------
# Single-head self-attention
# ----------------------------
class OneHeadSelfAttention:
  def __init__(self):
    # three simple projections (no bias keeps it minimal)
    self.q = nn.Linear(D_EMB, D_EMB, bias=False)
    self.k = nn.Linear(D_EMB, D_EMB, bias=False)
    self.v = nn.Linear(D_EMB, D_EMB, bias=False)
    # no output projection — keep truly minimal

  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    # x: (B, T, D)
    q = self.q(x)                            # (B, T, D)
    k = self.k(x)                            # (B, T, D)
    v = self.v(x)                            # (B, T, D)

    att = (q @ k.transpose(1, 2)) * (1.0 / math.sqrt(D_EMB))  # (B, T, T)
    # robust numerics: cast before masking/softmax
    att = att.cast(dtypes.float32).masked_fill(mask == 0, -1e9).softmax(axis=-1)
    out = att @ v                             # (B, T, D)
    return out

# ----------------------------
# One-block attention-only LM
# ----------------------------
class OneHeadBlock:
  def __init__(self):
    self.ln = nn.RMSNorm(D_EMB)
    self.attn = OneHeadSelfAttention()

  def __call__(self, x: Tensor, mask: Tensor) -> Tensor:
    x = x + self.attn(self.ln(x), mask)       # pre-norm + residual
    return x

class TinyOneHeadLM:
  def __init__(self):
    self.wte = nn.Embedding(VOCAB_SIZE, D_EMB)
    self.wpe = nn.Embedding(N_CTX, D_EMB)
    self.blocks = [OneHeadBlock() for _ in range(N_LAYERS)]
    self.ln_f = nn.RMSNorm(D_EMB)
    self.head = nn.Linear(D_EMB, VOCAB_SIZE, bias=True)  # separate head (simpler than weight tying)

  def __call__(self, token_ids: Tensor) -> Tensor:
    # token_ids: (B, T) int32
    B, T = token_ids.shape
    pos = Tensor.arange(T, dtype=dtypes.int32).unsqueeze(0)    # (1, T)
    x = self.wte(token_ids) + self.wpe(pos)                    # (B, T, D)

    # causal mask for (B, T, T)
    mask = Tensor.tril(Tensor.ones((T, T))).unsqueeze(0)       # (1, T, T) → broadcast to (B, T, T)

    for blk in self.blocks:
      x = blk(x, mask)

    x = self.ln_f(x)                                           # (B, T, D)
    logits = self.head(x)                                      # (B, T, V)
    return logits

# --- add this next to your other helpers ---
def token_ids_to_text(ids):
  return ''.join(idx2char[int(i)] for i in ids)

def _sample_from_logits(next_token_logits: Tensor, temperature: float = 1.0, top_k: int = 50) -> int:
  """
  next_token_logits: (V,) tinygrad Tensor
  returns: python int (token id)
  """
  import numpy as np
  # deterministic at temperature<=0
  if temperature is None or temperature <= 0:
    return int(next_token_logits.argmax().item())

  logits = next_token_logits.realize().numpy().astype(np.float64)
  logits = logits / max(1e-8, float(temperature))

  V = logits.shape[0]
  if top_k is not None and 0 < top_k < V:
    # top-k filter
    kth = top_k - 1
    idx = np.argpartition(-logits, kth)[:top_k]
    sub_logits = logits[idx]
    sub_logits = sub_logits - sub_logits.max()
    probs = np.exp(sub_logits)
    probs /= probs.sum()
    choice = np.random.choice(top_k, p=probs)
    return int(idx[choice])
  else:
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    return int(np.random.choice(V, p=probs))


def generate(prompt: str = "",
             max_tokens: int = 256,
             temperature: float = 1.0,
             top_k: int = 50,
             checkpoint: str | None = None,
             stream: bool = True) -> str:
  """
  Text generation for the TinyOneHeadLM.
  If `checkpoint` is provided, loads weights from it. Otherwise uses fresh init (likely gibberish).
  """
  # build model and optionally load weights
  model = TinyOneHeadLM()
  if checkpoint is not None:
    state = nn.state.safe_load(checkpoint)
    nn.state.load_state_dict(model, state)

  # encode prompt
  ctx = text_to_token_ids(prompt)

  # single-step inference: returns logits for the last position (shape (V,))
  @Tensor.train(False)
  def _step(input_ids: Tensor) -> Tensor:
    # model(input_ids): (1, t, V) → take last position
    return model(input_ids)[0, -1]

  out_tokens = []
  if stream and prompt:
    import sys
    sys.stdout.write(prompt)
    sys.stdout.flush()

  for _ in range(max_tokens):
    # feed last N_CTX tokens
    inp = Tensor([ctx[-N_CTX:]], dtype=dtypes.int32)   # (1, t<=N_CTX)
    next_logits = _step(inp)
    next_id = _sample_from_logits(next_logits, temperature=temperature, top_k=top_k)
    ctx.append(next_id)
    out_tokens.append(next_id)

    if stream:
      import sys
      sys.stdout.write(token_ids_to_text([next_id]))
      sys.stdout.flush()

  if stream:
    # ensure newline after streaming
    import sys
    sys.stdout.write("\n")
    sys.stdout.flush()

  return token_ids_to_text(out_tokens)


# ----------------------------
# Train: overfit sanity first
# ----------------------------
if __name__ == "__main__":
  import sys
  from pathlib import Path

  # CLI: `python min_onehead_char.py generate "prompt here" checkpoints/model.safetensors`
  if len(sys.argv) > 1 and sys.argv[1] == "generate":
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    ckpt = sys.argv[3] if len(sys.argv) > 3 else None
    _ = generate(prompt=prompt, max_tokens=256, temperature=1.0, top_k=50, checkpoint=ckpt, stream=True)
    raise SystemExit(0)

  # ------- training (unchanged) -------
  token_ids = text_to_token_ids(tiny_shakes_text)
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids)

  model = TinyOneHeadLM()
  params = nn.state.get_parameters(model)
  opt = nn.optim.AdamW(params, lr=LR, b1=0.90, b2=0.95, weight_decay=0.1)

  from tinygrad import Tensor
  from tinygrad.helpers import trange

  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    idx = Tensor.randint(BATCH_SIZE, high=int(X_train.shape[0]))
    x = X_train[idx]
    y = Y_train[idx]
    opt.zero_grad()
    logits = model(x)
    loss = logits.sparse_categorical_crossentropy(y).mean()
    loss.backward()
    opt.step()
    return loss

  @Tensor.train(False)
  def eval_ppl() -> Tensor:
    logits = model(X_test)
    nll = logits.sparse_categorical_crossentropy(Y_test).mean()
    return nll.exp()

  val_ppl = float("nan")
  for i in (t := trange(STEPS)):
    loss = train_step().item()
    if i % 200 == 199:
      val_ppl = eval_ppl().item()
      nn.state.safe_save(nn.state.get_state_dict(model), f"checkpoints/sm_model{i}.safetensors")
    t.set_description(f"loss: {loss:6.3f}  val_ppl: {val_ppl:6.2f}")
