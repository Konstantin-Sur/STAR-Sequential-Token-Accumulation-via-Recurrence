# STAR — Sequential Token Accumulation via Recurrence

> *A recurrent sequence mechanism that outperforms baseline Multi-Head Attention on WikiText-2 in a single epoch.*

---

## Overview

Hello, everyone who will see this paper and maybe even use it in their own LLMs.

I developed an algorithm that — to my knowledge — has no exact prior art, and that outperforms a standard MHA baseline in a controlled, fair comparison.

---

## Core Idea

The central question was:

> **What if tokens, instead of attending to each other, contributed differences to a shared "scene" vector that accumulates a representation of what the text describes?**

Each token computes a `delta` — a change it wants to apply to the **scene vector**. The scene vector absorbs these deltas one by one, causally building up a compressed representation of the context seen so far. Every token then reads from the current scene to enrich itself.

### Example

Take the sentence: `"Bird was flying around"`

| Step | Token | Scene vector knows… |
|------|-------|----------------------|
| 1 | `Bird` | There is a bird |
| 2 | `was` | Something was |
| 3 | `flying` | A bird was flying ✦ |
| 4 | `around` | A bird was flying around |

At step 3, the scene vector has effectively *imagined* a flying bird — not by attending over all past tokens quadratically, but by accumulating structured differences recurrently.

---

## Mechanism

```
scene₀  =  0

for each token t:
    δ     =  tanh( tok_proj(t)  ×  (1 + scene_proj(scene)) )
    scene =  LayerNorm( scene + δ )
    out_t =  t + W_out( scene )
```

Where:
- **`tok_proj`** — 2-layer MLP that nonlinearly encodes *what the token carries*
- **`scene_proj`** — 2-layer MLP that nonlinearly reads *what is currently active in the scene*
- **`×`** — elementwise (Hadamard) product: the token modulates exactly the active dimensions of the scene
- **`(1 + …)`** — ensures the first token can write into an empty scene and gradients flow from step 0
- **`tanh`** — bounds each write to prevent explosion over long sequences
- **`LayerNorm`** — keeps the scene magnitude stable across all 256+ steps

---

## Why This Works

A single linear matrix is just rotation and scaling — it cannot nonlinearly separate `"bird"` from `"motion"`. The 2-layer MLPs inside `tok_proj` and `scene_proj` are what allow the mechanism to extract and combine complex features.

The Hadamard product is the key interaction: `"motion"` gets applied not into the void, but *onto the active bird dimensions* of the scene. If the scene held `"stone"` instead, the same `"motion"` token would produce a rolling stone.

---

## Complexity

| Mechanism | Sequence complexity |
|-----------|-------------------|
| Multi-Head Attention | O(n²) |
| **STAR (ours)** | **O(n)** |

STAR is inherently sequential (recurrent), but contains no quadratic attention matrix. A parallelized CUDA scan kernel (as in Mamba) would make it significantly faster than attention at long context lengths.

---

## Results

Trained on **WikiText-2-raw** for **1 epoch**, `d_model=512`, 4 layers, `seq_len=256`, batch size 16. All hyperparameters and training conditions are identical between models.

| Model | Params | Train loss | Val loss | Val perplexity |
|-------|--------|------------|----------|----------------|
| MHA Baseline | ~39M | 6.40 | 6.00 | 403 |
| **STAR (ours)** | **~39M** | **6.20** | **5.83** | **340** |

**STAR achieves 15.6% lower perplexity** than the MHA baseline in the same number of steps, with the same parameter count, data, optimizer, and learning rate schedule.

---

## Architecture

```
Embedding + Positional Embedding
        ↓
[ STARBlock × 4 ]
   ├── LayerNorm → STARLayer → residual
   └── LayerNorm → FFN (GELU) → residual
        ↓
LayerNorm → Linear head (weight-tied with embedding)
```

---

## Code

```python
class STARLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.tok_proj   = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.scene_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.W_out      = nn.Linear(d_model, d_model, bias=False)
        self.scene_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        scene   = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        out     = []
        for i in range(T):
            tok   = x[:, i, :]
            delta = torch.tanh(self.tok_proj(tok) * (1.0 + self.scene_proj(scene)))
            scene = self.scene_norm(scene + delta)
            out.append(tok + self.W_out(scene))
        return torch.stack(out, dim=1)
```

---

## Status

This is an early experimental result on a small dataset. Next steps:

- [ ] Train on WikiText-103 for statistical robustness
- [ ] Compare against Mamba and RWKV at the same scale
- [ ] Implement parallel CUDA scan for O(n) wall-clock speed
- [ ] Visualize what the scene vector learns over a sequence
- [ ] Scale to larger model sizes

---

## Citation

If you use this work, please cite it

---

*Built from scratch. Mechanism, theory, and experiments by a single maybe not good but researcher.*
