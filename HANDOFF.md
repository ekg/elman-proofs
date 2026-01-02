# Elman-Proofs Handoff Document

## Project Overview

This Lean 4 project contains formal proofs for gradient descent convergence, Lyapunov stability, and related optimization/dynamics theory. The proofs are designed to establish rigorous foundations for neural network training analysis, particularly for Elman-style recurrent networks.

## Current State (2026-01-02)

### New Research Direction: Expressivity Bounds

A new research direction has been established to formally prove expressivity and computational tradeoffs between:
- **Linear recurrence** (Mamba2, S4D, minGRU): Enables parallel scan via associativity
- **Nonlinear recurrence** (Elman, GRU): Requires sequential computation but has more expressivity
- **Log-polynomial** (novel): A middle ground being explored in ~/elman

See `RESEARCH_ROADMAP.md` for the full research plan.

### Associativity Separation (Mostly Complete)

**New file**: `ElmanProofs/Expressivity/Associativity.lean`

Core theorems proven:
- `LinearScanElement.instMonoid`: Linear RNN state transitions form a monoid (key for parallel scan!)
- `polynomial_composition_structure`: Pure power functions compose nicely: `(|a * |b|^α|)^α = |a|^α * |b|^(α²)`
- `tanh_strictMono`: Tanh is strictly monotone
- `tanh_injective`: Tanh is injective

Theorems with sorries (need numerical calculus bounds):
- `tanh_composition_not_linear`: Tanh RNN cannot be reduced to single affine step
- `polynomial_rnn_not_associative`: Polynomial RNN is also non-associative

### Linear State Capacity (Complete)

**File**: `ElmanProofs/Expressivity/LinearCapacity.lean`

All core theorems proven:
- `linear_state_is_sum`: State = Σ A^{T-1-t} B x_t (explicit sum formula!)
- `state_additive`, `state_scalar`: State function is linear in inputs
- `output_determined_by_state`: Output depends only on current state
- `same_state_same_future`: Same state → same future outputs (indistinguishability!)
- `reachable_is_subspace`: Reachable states form a vector subspace
- `reachable_dim_bound`: dim(reachable) ≤ n (information capacity bound)
- `not_linearly_computable_if_state_independent`: Key limitation theorem

### Previously Completed Proofs

**Gradient Descent Convergence** (`ElmanProofs/Gradient/Flow.lean`):
- `convex_convergence_rate`: O(1/k) convergence for smooth convex functions
- `strong_smooth_interpolation`: Key interpolation bound for μ-strongly convex AND L-smooth functions

**Lyapunov Stability** (`ElmanProofs/Dynamics/Lyapunov.lean`):
- `subseq_limit_eq_equilibrium`: Subsequential limits of Lyapunov-decreasing sequences are equilibria
- `strict_lyapunov_implies_asymptotic`: Strict Lyapunov functions imply asymptotic stability

**Spectral Radius** (`ElmanProofs/Stability/SpectralRadius.lean`):
- `powers_tendsto_zero`: Matrix powers tend to zero when spectral radius < 1
- `diagonal_spectral_radius`: For diagonal matrices, spectral radius = max|eigenvalue|

### Proof Chain (All Complete)

1. `lsmooth_fundamental_ineq` - Fundamental inequality for L-smooth functions
2. `lsmooth_cocoercivity` - Baillon-Haddad theorem: ‖∇f(x)‖² ≤ L⟨∇f(x), x - x*⟩
3. `strong_convex_gradient_monotonicity` - Gradient monotonicity for strongly convex functions
4. `strong_smooth_interpolation` - The key interpolation bound combining both properties
5. `convex_convergence_rate` - O(1/k) convergence rate

## Key Files

| File | Purpose |
|------|---------|
| `ElmanProofs/Expressivity/Associativity.lean` | **NEW** Associativity separation proofs |
| `ElmanProofs/Gradient/Flow.lean` | Gradient descent convergence proofs |
| `ElmanProofs/Dynamics/Lyapunov.lean` | Lyapunov stability theory |
| `ElmanProofs/Stability/SpectralRadius.lean` | Spectral radius bounds |
| `ElmanProofs/Activations/Lipschitz.lean` | Lipschitz bounds for activation functions |
| `ElmanProofs/RNN/Recurrence.lean` | Recurrence relation properties |
| `ElmanProofs/LogSpace/Stability.lean` | Log-space stability analysis |

## Build Instructions

```bash
cd /home/erikg/elman-proofs
~/.elan/bin/lake build
```

## Key Mathlib Dependencies

- `Mathlib.Analysis.Calculus.Gradient.Basic` - Gradient definitions
- `Mathlib.Analysis.InnerProductSpace.Dual` - Fréchet-Riesz theorem, `toDual`, `innerSL`
- `Mathlib.Analysis.Convex.Function` - Convexity definitions
- `Mathlib.Topology.MetricSpace.Lipschitz` - Lipschitz continuity
- `Mathlib.Analysis.SpecialFunctions.Pow.Real` - Real power functions (rpow)
- `Mathlib.Topology.Order.LeftRightNhds` - Filter tendsto lemmas

## Common Patterns Used

### Gradient Manipulation
```lean
-- Convert between gradient and fderiv
hasGradientAt_iff_hasFDerivAt
-- Apply gradient at a point
(hDiff z).hasGradientAt
-- Inner product with gradient
@inner ℝ E _ (gradient f x) v
```

### Inner Product Identities
```lean
-- Real inner product symmetry
real_inner_comm x y
-- Expansion formulas
norm_add_sq_real, norm_sub_sq_real
-- Bilinearity
inner_add_left, inner_sub_left, inner_smul_left
```

### Convexity First-Order Conditions
```lean
-- For convex f with ∇f(x*) = 0, x* is a minimizer
convex_first_order_optimality f hConvex hDiff x_star h_grad_zero
-- Gradient lower bound for convex functions
ConvexOn.inner_gradient_le_sub_of_mem_interior
```

### Filter/Tendsto for Limits
```lean
-- Nat cast tends to infinity
tendsto_natCast_atTop_atTop
-- f + const tends to infinity
h_nat.atTop_add tendsto_const_nhds
-- Inverse tends to zero
tendsto_inv_atTop_zero.comp h1
-- Continuity composition
h_cont.tendsto.comp h_exp_tendsto
-- Epsilon characterization of ≤
le_iff_forall_pos_lt_add
```

### Frobenius Norm
```lean
-- Custom frobNorm equals Mathlib's under frobeniusSeminormedAddCommGroup
letI := Matrix.frobeniusSeminormedAddCommGroup (α := ℝ) (m := Fin n) (n := Fin n)
-- Submultiplicativity
Matrix.frobenius_norm_mul M N
```

## Possible Next Steps

### Expressivity Research (Priority)
1. **Complete counterexample proofs**: Prove `tanh(x)/x` is strictly decreasing for x > 0
2. **Linear state capacity**: Prove linear RNN state is linear combination of inputs (`LinearCapacity.lean`)
3. **Linear limitations**: Prove linear RNNs cannot compute threshold/XOR functions (`LinearLimitations.lean`)

### Convergence Theory
4. **Linear convergence for strongly convex**: Use `strong_smooth_interpolation` to prove (1-μ/L)^k contraction
5. **Connect to RNN analysis**: Use Lyapunov and spectral radius results for Elman network stability
6. **Extend gradient descent**: Accelerated methods (Nesterov), stochastic gradient descent

## Tips for Working on This Codebase

1. **Circularity Warning**: When proving smoothness-related bounds, be careful of circular dependencies. The `strong_smooth_interpolation` proof specifically avoids calling `lsmooth_cocoercivity` on h because that would require proving h is L-smooth first.

2. **linarith limitations**: `linarith` doesn't handle expressions with inner products well. Pre-compute algebraic identities with `field_simp; ring` when inner products appear in inequalities.

3. **Build time**: Full build takes ~2 minutes. Use `lake build ElmanProofs.Gradient.Flow` to build specific files.

4. **Notation**: The project uses `∇ f x` for `gradient f x` and standard inner product notation `⟨·,·⟩` via `@inner ℝ E _`.

5. **Frobenius norm**: Custom `frobNorm` definition matches Mathlib's under `Matrix.frobeniusSeminormedAddCommGroup`. Use this to access `Matrix.frobenius_norm_mul` for submultiplicativity.
