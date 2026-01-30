# Open Question: Temporal Nonlinearity vs Depth

**Date:** 2026-01-29
**Context:** E88 documentation work revealed a gap in our expressivity analysis

## The Key Insight

Mamba2 and FLA-GDN have **linear temporal dynamics** within each layer:
```
h_T = Σ α^(T-t) · f(x_t)      # Mamba2
S_T = Σ α^(T-t) · k_t v_t^T   # FLA-GDN
```

The nonlinearities (SiLU, gating, projections) operate *within* each timestep. Information flows forward through time via purely linear operations.

**"Nonlinearity flows down (through layers), not forward (through time)."**

E88 has **nonlinear temporal dynamics**:
```
S := tanh(αS + δk^T)
```

The tanh compounds across timesteps, making S_T a nonlinear function of the entire history.

## What LinearLimitations.lean Proves

- Linear RNNs (h := Ah + Bx, y = Ch) cannot compute threshold functions
- Linear RNNs cannot compute XOR
- These proofs apply to the **temporal dimension** of Mamba2/FLA-GDN within each layer

## Open Questions

### Q1: Does depth compensate for linear temporal dynamics?

Can a D-layer Mamba2 compute functions that require temporal nonlinearity?

**Hypothesis:** No. Each layer's output at position t can only depend linearly on positions 0..t within that layer. Stacking layers gives nonlinear mixing of *features*, but each layer still has the same linear-temporal limitation.

**Counter-argument:** The features computed by layer L at position t include nonlinearly-mixed information from positions 0..t via layers 1..L-1. Maybe this indirection provides temporal expressivity?

**Needed:** Formal analysis of multi-layer SSMs.

### Q2: Separation theorem?

Is there a function computable by 1-layer E88 (with temporal tanh) that cannot be computed by any D-layer Mamba2?

**Intuition:** Consider a threshold over the *running sum* of inputs: output 1 when Σ_{i≤t} x_i > θ. This requires comparing accumulated information at each timestep. E88's tanh creates decision boundaries that track this. Mamba2's linear accumulation can track the sum, but can it threshold it temporally (not just at the final output)?

### Q3: Practical implications

Even if there's a theoretical separation, does it matter for language modeling?

- LLMs seem to work well with linear temporal dynamics (Mamba2 matches Transformers)
- Maybe the functions needed for language don't require temporal nonlinearity?
- Or maybe depth is sufficient in practice?

## Proposed Work

1. **Theorem (multi-layer limitation):** Prove or disprove that D-layer linear-temporal models have the same per-layer limitations.

2. **Separation example:** Find a concrete function family that:
   - 1-layer E88 can compute
   - D-layer Mamba2 cannot (for any D)

3. **Empirical test:** Design a synthetic task requiring temporal nonlinearity:
   - Running threshold detection
   - Temporal XOR (output x_t XOR x_{t-k})
   - Compare E88 vs deep Mamba2

## Connection to Existing Proofs

The current `LinearLimitations.lean` proves single-layer results. Extensions needed:

```lean
-- Current: single layer
theorem linear_cannot_threshold : ¬ LinearlyComputable (thresholdFunction τ T)

-- Needed: multi-layer
theorem multilayer_linear_cannot_threshold (D : ℕ) :
  ¬ MultiLayerLinearComputable D (thresholdFunction τ T)
```

Where `MultiLayerLinearComputable D f` means f is computable by D stacked layers, each with linear temporal dynamics but nonlinear inter-layer connections.

## Summary

The E88 documentation work clarified that:
1. LinearLimitations proofs apply to Mamba2/FLA-GDN's temporal dimension
2. But we haven't formally analyzed whether depth compensates
3. This is a real gap in our understanding of SSM expressivity
