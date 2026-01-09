# E12: Selective Gating Experiment Results

## Hypothesis

Mamba2's advantage over E1 might come from its "selective" gating mechanism where the gate depends on the hidden state, not just the input. E1 uses `output = h * silu(z)` while Mamba2 uses input-dependent gating via selective state space mechanisms.

## E12 Architecture

Minimal modification to E1:
```
E1:  output = h * silu(z)              # gate = silu(z)
E12: output = h * sigmoid(z + W_g @ h) # gate = sigmoid(z + W_g @ h)
```

This adds one extra GEMM per timestep (W_g @ h) to make the gate depend on the hidden state.

## Results (batch=16, 500 steps)

### 400M Scale Comparison
| Model | Params | Loss | Tok/s | Memory |
|-------|--------|------|-------|--------|
| E1    | 403M   | 1.73 | 15.8K | 9.7 GB |
| E12   | 484M   | 2.28 | 9.3K  | 10.9 GB |

### 50M Scale Comparison
| Model | Params | Loss | Tok/s | Memory |
|-------|--------|------|-------|--------|
| E1    | 50M    | 1.73 | 72.1K | 1.6 GB |
| E12   | 59M    | 1.89 | 52.4K | 1.8 GB |

## Analysis

**E12 is WORSE than E1 in every metric:**

1. **Higher Loss**: E12 achieves 2.28 vs E1's 1.73 (0.55 nats worse at 400M scale)
2. **Lower Throughput**: E12 is 41% slower (9.3K vs 15.8K tok/s)
3. **More Memory**: E12 uses 12% more memory
4. **More Parameters**: The extra W_g matrix adds ~20% more params

## Why It Failed

1. **The extra GEMM is expensive**: W_g @ h adds one full dim×dim GEMM per timestep
2. **sigmoid vs silu**: Sigmoid saturates more, potentially hurting gradient flow
3. **E1's gating is already sufficient**: silu(z) provides good gating without needing h

## Conclusion

**Selective gating does NOT explain the gap between E1 and Mamba2.**

The hypothesis that Mamba2's advantage comes from input-dependent gating is incorrect, or the specific form of selectivity matters greatly. E1's simple silu(z) gating is already effective.

## Next Steps to Close the Gap

Other hypotheses from the original study:
1. **State expansion**: Mamba2 expands hidden state (d_state=128). Try adding state expansion to E1.
2. **Initialization**: Different init schemes may favor different architectures.
3. **Normalization**: E1 uses spectral norm. Try Mamba2's normalization.
4. **Learning rate**: Maybe E1 needs different LR schedule.

Date: 2026-01-08
