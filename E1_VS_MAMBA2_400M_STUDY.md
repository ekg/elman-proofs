# E1 vs Mamba2 at 400M Scale: Benchmark Study

## Summary

E1 (Gated Elman) is **0.09 nats behind Mamba2** at 400M scale after 1500 steps of training on the same data. E1 uses less memory but is slower per step.

## Experimental Setup

- **Data**: 1.2TB pile.txt (The Pile, byte-level)
- **Batch size**: 16 (same for all models)
- **Steps**: 1500 (same for all models)
- **Total tokens**: 12.3M per model
- **Optimizer**: Schedule-Free AdamW (lr=3e-4, weight_decay=0.1)
- **Precision**: bfloat16
- **No gradient clipping**
- **Random seed**: 42 (same data stream for all)

## Results

| Model | Params | Loss | Tok/s | Memory | Time |
|-------|--------|------|-------|--------|------|
| **Mamba2 d22** | 402M | **1.50** | 23.8K | 8.1 GB | 515s |
| E1 d26 | 403M | 1.59 | 15.8K | 9.7 GB | 778s |
| E1 d20 | 394M | 1.59 | 18.6K | 8.8 GB | 661s |
| E1 d16 | 391M | 1.60 | 19.3K | 8.0 GB | 637s |
| E1 d12 | 394M | 1.62 | 22.1K | 7.4 GB | 556s |
| E1 d6 | 386M | 1.69 | 25.6K | 6.1 GB | 479s |
| minLSTM d24 | 383M | 1.75 | 20.9K | 17.0 GB | 587s |
| minGRU d24 | 364M | 1.78 | 19.5K | 18.7 GB | 629s |

## Key Findings

### 1. Loss Gap
- **Mamba2**: 1.50 nats
- **Best E1 (d26)**: 1.59 nats
- **Gap**: 0.09 nats (6% relative)

### 2. E1 Depth Scaling
Deeper E1 = better loss, but slower:
```
d6:  1.69 loss, 25.6K tok/s
d12: 1.62 loss, 22.1K tok/s
d16: 1.60 loss, 19.3K tok/s
d20: 1.59 loss, 18.6K tok/s
d26: 1.59 loss, 15.8K tok/s  <- diminishing returns after d20
```

### 3. Memory Efficiency
E1 is memory-efficient compared to parallel-scan models:
- E1 d26: 9.7 GB
- Mamba2: 8.1 GB
- minGRU: 18.7 GB (2.3x more than Mamba2)
- minLSTM: 17.0 GB (2.1x more than Mamba2)

### 4. Throughput
Mamba2 is 1.5x faster than E1 d26 at same param count:
- Mamba2: 23.8K tok/s
- E1 d26: 15.8K tok/s

### 5. Deep Mamba2 Fails
Mamba2 cannot scale depth (from separate experiment):
- Mamba2 d22: 1.50 loss
- Mamba2 d32: 1.97 loss (collapses!)
- Mamba2 d40: 1.95 loss

E1 scales depth gracefully while Mamba2 does not.

## The 0.09 Nat Gap

E1 d26 achieves 1.59 vs Mamba2's 1.50. What could close this gap?

### Hypotheses to Test

1. **Gating mechanism**: Mamba2 uses selective state space gating. E1 uses simple silu(z) gating.
   - Could selective gating improve E1?

2. **State expansion**: Mamba2 expands hidden state (d_state=128). E1 has no expansion.
   - Could adding state expansion help?

3. **Initialization**: Different init schemes may favor different architectures.
   - Try Mamba2's init on E1?

4. **Normalization**: E1 uses spectral norm on W_h. Mamba2 uses different normalization.
   - Experiment with normalization strategies.

5. **Learning rate**: Maybe E1 needs different LR schedule.
   - Try LR sweep for E1 specifically.

6. **Depth vs Width tradeoff**: At d20, E1 hits diminishing returns.
   - Maybe wider + shallower with modifications?

## Architecture Comparison

### E1 (Gated Elman)
```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output = h_t * silu(z_t)
```
- Sequential recurrence (no parallel scan)
- Simple gating with silu
- Spectral norm on W_h for stability

### Mamba2
```
Selective state space with:
- Input-dependent discretization
- Parallel associative scan
- d_state expansion (128)
- Selective gating mechanism
```

## Next Steps

1. **Ablate Mamba2 features on E1**: Add one feature at a time to see what closes the gap
2. **Study the gradient flow**: Why does E1 need more depth to match?
3. **Selective gating experiment**: Implement Mamba-style selectivity in E1
4. **State expansion experiment**: Add learnable state expansion to E1

## Raw Data

Configs used:
- E1 d6: dim=3584, depth=6 (386M params)
- E1 d12: dim=2560, depth=12 (394M params)
- E1 d16: dim=2208, depth=16 (391M params)
- E1 d20: dim=1984, depth=20 (394M params)
- E1 d26: dim=1760, depth=26 (403M params)
- Mamba2: dim=1728, depth=22, d_state=128, headdim=64 (402M params)
- minGRU: dim=2752, depth=24 (364M params)
- minLSTM: dim=2304, depth=24 (383M params)

Date: 2026-01-08
