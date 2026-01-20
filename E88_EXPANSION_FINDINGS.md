# E88 FLA Hybrid: Square State vs Rectangular State

**Date:** 2026-01-20
**Benchmark:** 10 minutes training, ~95M parameters, depth=20

## Summary

E88 combines FLA-GDN's design elements (Mamba2-style exponential decay, output gating, short convolutions) with a nonlinear matrix state: `S = tanh(decay * S + outer(delta, k_norm))`.

Key finding: **Square state with many small heads dramatically outperforms rectangular state with fewer large heads.**

## Results Table

| Model | Loss | tok/s | Dim | State Shape | Notes |
|-------|------|-------|-----|-------------|-------|
| fla-gdn | **1.39** | 58,824 | 768 | [64×384] | Parallel scan baseline |
| **E88sh16n32** | **1.40** | 33,498 | 1792 | [32×32]×16 | Best E88 variant |
| mamba2 | 1.55 | 54,756 | 896 | d_state=128 | SSM baseline |
| E88sh8n64 | 1.60 | 26,114 | 1792 | [64×64]×8 | Square, 8 heads |
| E88th16n64 | 1.67 | 15,527 | 1152 | [64×64]×16 | Tied k/v |
| E88th8n64 | 1.84 | 25,500 | 2176 | [64×64]×8 | Tied k/v |
| E88sh8n96 | 1.98 | 11,675 | 1280 | [96×96]×8 | Large state |
| E88h8n64_exp2 | 2.00 | 11,993 | 1152 | [64×128]×8 | Rectangular |
| E88th8n96 | 2.00 | 11,879 | 1536 | [96×96]×8 | Tied + large |

## Key Findings

### 1. Square State Beats Rectangular (expansion=1.0 > expansion=2.0)

Same n_state=64, same ~95M params:
- **E88sh8n64** (square [64×64]): **1.60 loss**
- **E88h8n64_exp2** (rect [64×128]): **2.00 loss**

The rectangular state with expansion=2.0 wastes capacity. The value dimension expansion doesn't help and may hurt gradient flow through the nonlinear tanh.

### 2. More Heads > Larger State (16×32² ≈ 8×64²)

Both have same total state size (16,384 floats):
- **E88sh16n32** (16 heads, 32×32): **1.40 loss**
- **E88sh8n64** (8 heads, 64×64): **1.60 loss**

**0.20 nats improvement** from using more smaller heads. This suggests:
- Each head learns independent retrieval patterns
- Diversity of small memories > capacity of large memories
- Similar to attention: more heads = more parallel subspaces

### 3. tie_kv Hurts Quality

Skipping v_proj (setting v=k after conv+silu):
- **E88sh8n64** (separate k,v): **1.60 loss**
- **E88th8n64** (tied k=v): **1.84 loss**

**0.24 nats worse**. The key and value need independent projections for the delta rule to work well. When v=k, the model can only store information along the key direction.

### 4. Large State (n_state=96) Underperforms

Despite more capacity:
- **E88sh8n64** (64×64, 4K per head): **1.60 loss**
- **E88sh8n96** (96×96, 9K per head): **1.98 loss**

Larger states are harder to train with sequential updates. The tanh nonlinearity may saturate more easily with larger matrices.

### 5. E88sh16n32 Nearly Matches fla-gdn

- **fla-gdn**: 1.39 loss (parallel scan, FLA library)
- **E88sh16n32**: 1.40 loss (sequential, pure CUDA)

Only **0.01 nats worse** than the parallel scan baseline! This is remarkable for a fully sequential model. The combination of:
- Many heads (16)
- Small square states (32×32)
- Nonlinear tanh update
- FLA-GDN-style decay and gating

...produces near-optimal performance.

## Architecture Details

### E88sh16n32 (Best E88)
```
dim = 1792
n_heads = 16
n_state = 32
expansion = 1.0 (square state)

Per head:
  k, q: [32] L2-normalized
  v: [32]
  S: [32×32] matrix state

Update: S = tanh(decay * S + outer(v - S@k, k))
Output: S @ q (then gated by sigmoid(g_proj(x)))
```

### State Efficiency Comparison

| Model | Total State | Loss | State/Loss Ratio |
|-------|-------------|------|------------------|
| mamba2 | 229K | 1.55 | 148K per nat |
| fla-gdn | 98K | 1.39 | 71K per nat |
| E88sh16n32 | 16K | 1.40 | 11K per nat |

E88sh16n32 uses **6× less state** than mamba2 for similar quality.

## Recommendations

1. **Use E88sh16n32** as the default E88 configuration
2. **Always use expansion=1.0** (square state) for E88
3. **Prefer more heads with smaller states** over fewer heads with larger states
4. **Don't tie k/v** - independent projections are worth the parameters
5. **Try E88sh32n16** next - even more heads with smaller states

## Code

Models available in `elman/models/e88_fla_hybrid.py`:
- `E88sh16n32`: Best performer (16 heads, 32×32 state, expansion=1.0)
- `E88sh8n64`: Good alternative (8 heads, 64×64 state)
- `E88th*`: Tied k/v variants (not recommended)

CUDA kernel in `elman/cuda/lib/e88_fla_hybrid_gpu.cu.cc` supports:
- n_state: {32, 64, 96}
- head_v_dim: {32, 64, 96, 128}
- Square states (n_state = head_v_dim) fully supported

## Next Steps

1. Test E88sh32n16 (32 heads, 16×16) - even more head diversity
2. Test longer training (1 hour) to see if findings hold
3. Scale to 1B parameters
4. Compare inference latency vs fla-gdn
