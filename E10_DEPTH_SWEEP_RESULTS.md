# E10 (MultiScale EMA) Depth Sweep Results

## Setup

- **Model**: E10 with k=4 EMA memory banks
- **Batch size**: 16
- **Steps**: 1500
- **Optimizer**: Schedule-Free AdamW (lr=3e-4, weight_decay=0.1)
- **Data**: pile.txt (byte-level)

## Results

| Model | Params | Loss | Tok/s | Time |
|-------|--------|------|-------|------|
| E10 d26 | 726M | **1.67** | 9.4K | 1312s |
| E10 d20 | 709M | 1.80 | 10.9K | 1122s |
| E10 d16 | 703M | 1.89 | 11.4K | 1076s |
| E10 d12 | 709M | 1.98 | 13.3K | 925s |
| E10 d6  | 695M | 2.14 | 14.7K | 836s |

## Key Findings

### 1. Deeper is Better for E10
Like E1, E10 benefits from more depth:
- d26 achieves 1.67 loss (best)
- d6 achieves 2.14 loss (worst)
- Clear monotonic improvement with depth

### 2. Parameter Count Issue
All configs have ~700M params, not the target 400M. This is because E10's EMA memory banks add significant parameters beyond the basic E1 architecture.

### 3. Comparison with E1 and Mamba2 (from previous study)

| Model | Params | Loss | Tok/s | Memory |
|-------|--------|------|-------|--------|
| Mamba2 | 402M | **1.50** | 23.8K | 8.1 GB |
| E1 d26 | 403M | 1.59 | 15.8K | 9.7 GB |
| E10 d26 | 726M | 1.67 | 9.4K | 20.8 GB |

**E10 is NOT competitive:**
- Worse loss (1.67 vs 1.50/1.59) despite 80% more parameters
- Much slower (9.4K vs 15.8K/23.8K tok/s)
- Uses 2x memory (20.8GB vs 8-10GB)

### 4. Why E10 Underperforms

1. **Memory overhead**: The EMA banks require storing separate memory states per timestep
2. **Extra computation**: Reading from and writing to memory banks adds overhead
3. **Parameter inefficiency**: Most extra params go to memory, not core recurrence

## Conclusion

E10's multi-scale EMA approach does not improve over E1 at similar or higher parameter counts. The additional memory banks hurt throughput without improving loss.

**Recommendation**: Focus on E1 optimizations rather than architectural additions like memory banks.

Date: 2026-01-08
