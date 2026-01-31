# Sorry Catalog: Broken Theorem Statements Analysis

This document catalogs all `sorry` statements in the codebase, categorized by **WHY** they are present and what makes each theorem unprovable or incomplete.

## Summary Statistics

- **Total files with sorries**: 19 project files
- **Total sorry occurrences**: 78+ instances
- **Categories**: 7 distinct root causes

---

## Category 1: Numerical Bound Verification (16 instances)

These sorries require verifying tight numerical inequalities involving transcendental functions (tanh, exp, etc.).

### Files Affected:
- `ElmanProofs/Expressivity/TanhSaturation.lean`
- `ElmanProofs/Architectures/E42_GradientFlow.lean`
- `ElmanProofs/Expressivity/ExactCounting.lean`

### Representative Examples:

| File | Line | Statement | WHY it's broken |
|------|------|-----------|-----------------|
| TanhSaturation.lean | 279 | `tanh(0.575) > 0.5` | Requires proving `exp(23/20) > 3`, tight numerical bound |
| E42_GradientFlow.lean | 75 | `tanh_deriv x < 0.15` for `\|x\| > 2` | Needs `tanh(2) ≈ 0.964` numerical verification |
| ExactCounting.lean | 561 | `latched_threshold_persists` | Numerical bounds `tanh(arg) > 0.8` require `arg > 1.099` |

### Root Cause:
Lean 4's native numerical computation can't directly verify transcendental bounds. These require:
- Taylor series expansions with error bounds
- Monotonicity arguments via derivatives
- Interval arithmetic libraries (not in standard Mathlib)

---

## Category 2: Theorem Statements Are Mathematically FALSE (12 instances)

These theorems have incorrect statements that cannot be proven because they are false for some valid inputs.

### Files Affected:
- `ElmanProofs/Expressivity/TanhSaturation.lean`
- `ElmanProofs/Expressivity/AttentionPersistence.lean`

### Critical Examples:

| File | Line | Statement | WHY it's FALSE |
|------|------|-----------|----------------|
| TanhSaturation.lean | 1051, 1122 | `alertBasin_nonempty` | **FALSE for α ≤ 1**: When `α ≤ 1`, iterates converge to 0, so no `S_init` can keep `\|iter\| > θ` for all t. Theorem needs `α > 1` hypothesis. |
| TanhSaturation.lean | 1148, 1160 | `alert_state_robust` | **FALSE for small α**: The bound `\|δ\| < (1 - tanh θ)/2` allows cases where `θ - δ ≤ 0` or `arg ≈ 0`, making `tanh(arg) < θ` possible. |
| AttentionPersistence.lean | 1167, 1182, 1363, 1382 | `strong_input_persists` | **FALSE for large θ**: For `θ > 0.96`, `tanh(θ + 1) < θ`, so the claimed bound fails. Theorem needs `θ < ~0.76` constraint. |
| AttentionPersistence.lean | 1804, 1847 | `strong_input_persists` branches | Both positive and negative `arg` cases fail without additional constraints on `(α, θ, S)` relationships. |

### Root Cause:
The theorem statements are too general. They claim results for parameter ranges where the conclusions are actually false. The fix requires:
- Adding hypothesis `α > 1` to fixed-point existence theorems
- Adding hypothesis `θ < S*(α)` where `S*` is the nonzero fixed point
- Restricting `θ < 0.76` (or `θ < artanh(θ)`) for persistence theorems

---

## Category 3: Missing Algebraic/Sum Manipulation Lemmas (15 instances)

Straightforward algebraic manipulations that require tedious but routine proof steps.

### Files Affected:
- `ElmanProofs/Dynamics/GradientFlow.lean`
- `ElmanProofs/DeltaRule.lean`
- `ElmanProofs/Architectures/E23_DualMemory.lean`
- `ElmanProofs/Architectures/E26_ParallelDualMemory.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| GradientFlow.lean | 105 | `linearRNN_closed_form` | Index juggling in sum: `∑ k ∈ range (T+1), W^{T-k} · x_k` expansion |
| GradientFlow.lean | 256 | `powerlaw_condition_number` | Algebraic: `σ₁/σᵣ = 1^{-α} / r^{-α} = r^α` |
| DeltaRule.lean | 155 | `delta_rule_exact_retrieval_unit` | Matrix algebra: `v_i * (∑ j, k_j²) = v_i * 1` |
| DeltaRule.lean | 292 | `orthogonal_keys_capacity` | Sum over orthonormal basis: `∑ i, v_i * δ_{ij} = v_j` |
| E23_DualMemory.lean | 167 | `replacement_write_bounded` | Convex combination boundedness |

### Root Cause:
These proofs are mathematically straightforward but require:
- Careful handling of Finset sums and index manipulation
- Distributivity of matrix operations over sums
- Properties of orthonormal bases

---

## Category 4: Derivative/Calculus Infrastructure (8 instances)

Proofs requiring differentiation and calculus machinery.

### Files Affected:
- `ElmanProofs/Dynamics/GradientFlow.lean`
- `ElmanProofs/Architectures/E1_GatedElman.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| GradientFlow.lean | 115 | `jacobian_linear_rnn` | Need `fderiv ℝ (affine map) = linear part` |
| E1_GatedElman.lean | 168 | `sigmoid_deriv_max` | Need calculus: max of `t(1-t)` on `(0,1)` is `1/4` at `t=0.5` |

### Root Cause:
Requires Mathlib's calculus imports and careful handling of:
- `fderiv` for affine maps
- Mean value theorem applications
- Critical point analysis

---

## Category 5: Softmax/Attention Properties (6 instances)

Properties of softmax and attention mechanisms.

### Files Affected:
- `ElmanProofs/Architectures/E23_DualMemory.lean`
- `ElmanProofs/Architectures/E42_SpectralAnalysis.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| E23_DualMemory.lean | 123 | `softmax_sums_to_one` | Standard: `∑ i, exp(x_i) / (∑ j, exp(x_j)) = 1` |
| E23_DualMemory.lean | 128 | `softmax_nonneg` | `exp > 0` and denominator > 0 |
| E23_DualMemory.lean | 133 | `softmax_le_one` | Each term ≤ sum |
| E23_DualMemory.lean | 221 | `attention_read_bounded` | Convex combination of bounded values |

### Root Cause:
Require positivity of exponentials and division properties.

---

## Category 6: Spectral/Eigenvalue Analysis (6 instances)

Linear algebra requiring spectral theory.

### Files Affected:
- `ElmanProofs/Architectures/E42_SpectralAnalysis.lean`
- `ElmanProofs/Architectures/ResidualRecurrence.lean`
- `ElmanProofs/Architectures/Mamba2Verified.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| E42_SpectralAnalysis.lean | 109 | Eigenvalue computation | Requires SVD/spectral decomposition |
| E42_SpectralAnalysis.lean | 113 | SVD bounds | Matrix norm infrastructure |
| ResidualRecurrence.lean | 115 | `pow_lt_pow_succ` | Requires Mathlib lemmas on power decay |

### Root Cause:
Mathlib has spectral theory but connecting it to matrix norms requires careful setup.

---

## Category 7: Fixed Point Existence/IVT (8 instances)

Proofs requiring intermediate value theorem or fixed point arguments.

### Files Affected:
- `ElmanProofs/Expressivity/TanhSaturation.lean`
- `ElmanProofs/Expressivity/ExactCounting.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| TanhSaturation.lean | 515 | `tanh_multiple_fixed_points` | IVT: `g(x) = tanh(αx) - x` changes sign |
| TanhSaturation.lean | 522 | `tanh_basin_of_attraction` | Contraction mapping near fixed point |
| ExactCounting.lean | 485 | `e88_count_mod_3_existence` | Basin construction requires IVT |

### Root Cause:
Need careful application of `Topology.MetricSpace.Contracting` or `Analysis.Calculus.MeanValue`.

---

## Category 8: Structure/Definitional Issues (7 instances)

Proofs blocked by structure mismatches or definitional issues.

### Files Affected:
- `ElmanProofs/Architectures/GatedDeltaRule.lean`
- `ElmanProofs/Architectures/E31_SparseGatedElman.lean`

### Examples:

| File | Line | Statement | WHY incomplete |
|------|------|-----------|----------------|
| GatedDeltaRule.lean | 193-194 | Nested sorry in update | Let-binding unfolding issues |
| E31_SparseGatedElman.lean | 90, 94 | SiLU bounds | Need `silu(x) = x * sigmoid(x)` with `0 < sigmoid < 1` |
| E31_SparseGatedElman.lean | 319 | Sparse update | Top-k selection implementation |

---

## Priority Ranking for Fixes

### High Priority (Blocking other proofs):
1. **Category 2**: Fix theorem statements - these are mathematically false
2. **Category 3**: Algebraic lemmas - many proofs depend on these
3. **Category 5**: Softmax properties - used throughout attention code

### Medium Priority (Self-contained):
4. **Category 1**: Numerical bounds - can use `native_decide` for some
5. **Category 4**: Calculus infrastructure
6. **Category 7**: Fixed point theorems

### Low Priority (Nice to have):
7. **Category 6**: Spectral analysis
8. **Category 8**: Structural issues

---

## Recommended Actions

### For Category 2 (FALSE statements):
```lean
-- BAD (current):
theorem alertBasin_nonempty (α θ : ℝ) (hα : 0 < α) (hα_lt : α < 2) ...

-- GOOD (fixed):
theorem alertBasin_nonempty (α θ : ℝ) (hα : 1 < α) (hα_lt : α < 2)
    (hθ_small : θ < nonzeroFixedPoint α) ...
```

### For Category 1 (Numerical):
Consider using:
- `norm_num` extensions for transcendental bounds
- Interval arithmetic via `Mathlib.Analysis.SpecialFunctions.Pow.NNReal`
- Direct `native_decide` for computable bounds

### For Category 3 (Algebraic):
Use standard Mathlib tactics:
- `simp [Finset.sum_add_distrib, ...]`
- `ext` for function/matrix extensionality
- `congr 1` for careful argument matching
