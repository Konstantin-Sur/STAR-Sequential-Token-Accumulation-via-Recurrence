``` python
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm.auto import tqdm

# ─── Hyperparams — identical to STARLLM.py ──────────────────────────────────
D_MODEL    = 512
N_HEADS    = 8          
N_LAYERS   = 4
FFN_MULT   = 4
SEQ_LEN    = 256
BATCH_SIZE = 16
LR         = 3e-4
EPOCHS     = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ─── CausalSelfAttention (standart MHA with Mask) ───────────────────
class CausalSelfAttention(nn.Module):
    """
    Precise copy of algorithm that used in GPT2
    """

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = SEQ_LEN):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = self.d_head ** -0.5
        self.qkv      = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model,     bias=False)

        # Causal маска — нижний треугольник
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
              .view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)                                    # (B, T, 3D)
        q, k, v = qkv.split(D, dim=2)                        # (B, T, D) × 3

        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (B, H, T, d)

        attn = (q @ k.transpose(-2, -1)) * self.scale        # (B, H, T, T)
        attn = attn.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v                                        # (B, H, T, d)
        out = out.transpose(1, 2).contiguous().view(B, T, D) # (B, T, D)

        return self.out_proj(out)


# ─── MHABlock ─────────────────────────────────────────────────────────────────
class MHABlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = FFN_MULT):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = CausalSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # attention sub-layer
        x = x + self.ffn(self.norm2(x))    # ffn sub-layer
        return x


# ─── BaselineLM ───────────────────────────────────────────────────────────────
class BaselineLM(nn.Module):
    def __init__(
        self,
        vocab_size:  int,
        d_model:     int = D_MODEL,
        n_heads:     int = N_HEADS,
        n_layers:    int = N_LAYERS,
        max_seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks  = nn.ModuleList(
            [MHABlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=idx.device)
        x   = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        return self.head(self.norm(x))                       # (B, T, vocab_size)

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]
            logits   = self(idx_cond)[:, -1, :] / temperature
            probs    = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx


# ─── Dataset ──────────────────────────────────────────────────────────────────
class TokenDataset(Dataset):
    def __init__(self, tokens: list, seq_len: int):
        n            = (len(tokens) - 1) // seq_len * seq_len
        self.tokens  = tokens[:n + 1]
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, i: int):
        s = i * self.seq_len
        x = torch.tensor(self.tokens[s     : s + self.seq_len],        dtype=torch.long)
        y = torch.tensor(self.tokens[s + 1 : s + self.seq_len + 1],    dtype=torch.long)
        return x, y


# ─── Utilities ──────────────────────────────────────────────────────────────────
def tokenize_split(split, tokenizer) -> list:
    text = "\n\n".join(t for t in split["text"] if t.strip())
    return tokenizer.encode(text)


def ppl(loss: float) -> float:
    return math.exp(min(loss, 20))


def count_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n/1e6:.2f}M"


# ─── Train ─────────────────────────────────────────────────────────────────
def train():
    print("=" * 50)
    print("  BASELINE: Multi-Head Attention (GPT-2 style)")
    print("=" * 50)
    print(f"Device : {DEVICE}")

    print("Loading GPT-2 tokenizer...")
    tokenizer  = GPT2TokenizerFast.from_pretrained("gpt2")
    VOCAB_SIZE = tokenizer.vocab_size

    print("Loading WikiText-2-raw...")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    print("Tokenization...")
    train_ids = tokenize_split(raw["train"],      tokenizer)
    val_ids   = tokenize_split(raw["validation"], tokenizer)
    test_ids  = tokenize_split(raw["test"],       tokenizer)
    print(f"  train : {len(train_ids):,} tokens")
    print(f"  val   : {len(val_ids):,} tokens")
    print(f"  test  : {len(test_ids):,} tokens")

    train_ds = TokenDataset(train_ids, SEQ_LEN)
    val_ds   = TokenDataset(val_ids,   SEQ_LEN)
    test_ds  = TokenDataset(test_ids,  SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = BaselineLM(VOCAB_SIZE).to(DEVICE)
    print(f"Params: {count_params(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    total_steps = EPOCHS * len(train_dl)
    scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=LR / 10
    )

    criterion     = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}  train")
        for step, (x, y) in enumerate(pbar, 1):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits = model(x)
            loss   = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            avg           = running_loss / step
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                avg=f"{avg:.3f}",
                ppl=f"{ppl(avg):.1f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        avg_train = running_loss / len(train_dl)

        # ── Validation ─────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x, y in tqdm(val_dl, desc=f"Epoch {epoch}/{EPOCHS}  val  "):
                x, y      = x.to(DEVICE), y.to(DEVICE)
                logits    = model(x)
                val_loss += criterion(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

        avg_val = val_loss / len(val_dl)

        print(
            f"\n── Epoch {epoch} ──────────────────────────────\n"
            f"  train  loss {avg_train:.3f}  ppl {ppl(avg_train):.1f}\n"
            f"  val    loss {avg_val:.3f}  ppl {ppl(avg_val):.1f}\n"
        )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "baseline_mha_best.pt")
            print("  ✓ best model saved → baseline_mha_best.pt\n")

    # ── Test ───────────────────────────────────────────────────────────────
    print("Loading best model for test...")
    model.load_state_dict(torch.load("baseline_mha_best.pt", map_location=DEVICE))
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(test_dl, desc="Test"):
            x, y       = x.to(DEVICE), y.to(DEVICE)
            logits     = model(x)
            test_loss += criterion(logits.view(-1, VOCAB_SIZE), y.view(-1)).item()

    avg_test = test_loss / len(test_dl)

    print(f"\nTest loss {avg_test:.3f}  |  Test perplexity {ppl(avg_test):.1f}")

    print("\n" + "=" * 50)
    print("  Results  MHA Baseline")
    print("=" * 50)
    print(f"  Params  : {count_params(model)}")
    print(f"  Val loss    : {best_val_loss:.3f}")
    print(f"  Val ppl     : {ppl(best_val_loss):.1f}")
    print(f"  Test loss   : {avg_test:.3f}")
    print(f"  Test ppl    : {ppl(avg_test):.1f}")
    print("=" * 50)

    print("\n── Generation ──────────────────────────────────")
    prompts = [
        "The history of science",
        "In the beginning of the century",
    ]
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], device=DEVICE)
        out = model.generate(ids, max_new_tokens=60, temperature=0.8)
        print(f"\nPrompt: {prompt!r}")
        print(tokenizer.decode(out[0].tolist()))

    print("\nReady.")


if __name__ == "__main__":
    train()

```
