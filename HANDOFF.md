# Elman-Proofs Handoff Document

## Project Overview

This Lean 4 project contains formal proofs for gradient descent convergence, Lyapunov stability, and related optimization/dynamics theory. The proofs are designed to establish rigorous foundations for neural network training analysis, particularly for Elman-style recurrent networks.

## Current State (2026-01-03)

**STATUS: SORRY-FREE** - The entire codebase compiles with no sorries.

### New Research Direction: Expressivity Bounds

A new research direction has been established to formally prove expressivity and computational tradeoffs between:
- **Linear recurrence** (Mamba2, S4D, minGRU): Enables parallel scan via associativity
- **Nonlinear recurrence** (Elman, GRU): Requires sequential computation but has more expressivity
- **Log-polynomial** (novel): A middle ground being explored in ~/elman

See `RESEARCH_ROADMAP.md` for the full research plan.

### Associativity Separation (Complete)

**File**: `ElmanProofs/Expressivity/Associativity.lean`

All theorems proven:
- `LinearScanElement.instMonoid`: Linear RNN state transitions form a monoid (key for parallel scan!)
- `polynomial_composition_structure`: Pure power functions compose nicely: `(|a * |b|^α|)^α = |a|^α * |b|^(α²)`
- `tanh_strictMono`: Tanh is strictly monotone
- `tanh_injective`: Tanh is injective
- `sinh_gt_id`: sinh(x) > x for x > 0 (key calculus lemma)
- `two_tanh_one_gt_tanh_two`: 2*tanh(1) > tanh(2) (proves tanh(x)/x is decreasing)
- `tanh_composition_not_linear`: Tanh RNN cannot be reduced to single affine step
- `polynomial_rnn_not_associative`: Polynomial RNN with |x|^α activation is non-associative
  - Uses counterexample (w=1, x₁=1, x₂=1) with case analysis showing 2^α ∈ {2,4} contradiction

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

**Gradient Vanishing** (`ElmanProofs/Activations/Lipschitz.lean`):
- `tanh_deriv_lt_one_of_ne_zero`: |tanh'(x)| < 1 for x ≠ 0
- `tanh_saturation`: Derivative vanishes at infinity
- `tanh_deriv_uniform_bound`: Uniform bound |tanh'(x)| ≤ 1 - tanh²(δ) when |x| ≥ δ
- `deep_tanh_gradient_vanishing`: Product ∏|tanh'(x_t)| ≤ r^T for r < 1
- `tendsto_tanh_atTop`: tanh(x) → 1 as x → ∞ (limit at infinity)

### Gradient Dynamics: Mamba2 vs Elman (Complete)

**File**: `ElmanProofs/Expressivity/GradientDynamics.lean`

Formalizes WHY Mamba2 learns slightly better than Elman - the key is what the gradient depends on:

| Architecture      | Gradient ∂h'/∂h depends on |
|-------------------|----------------------------|
| Pure Linear (S4)  | Nothing (fixed A)          |
| Mamba2 (Selective)| x only (via A(x))          |
| Elman             | x AND h (via tanh)         |

Key theorems proven:
- `tanh_deriv_strict`: |tanh'(x)| < 1 for x ≠ 0 (gradient factor strictly less than 1)
- `linear_gradient_deterministic`: Linear RNN gradient is input-independent
- `nonlinear_gradient_varies`: Tanh gradient varies with different inputs
- `elman_gradient_h_dependent`: Elman gradient factor depends on hidden state h
- `elman_gradient_varies_with_h`: Different h values → different gradient factors
- `mamba2_gradient_h_independent`: Mamba2 gradient is h-independent (same x → same gradient)
- `tanh_gradient_in_unit_interval`: Gradient factor ∈ [0, 1]
- `selective_gradient_simpler`: SSM gradient is just a diagonal matrix
- `stock_elman_principle`: Elman has minimal nonlinearity (1 tanh vs 3-4 for GRU/LSTM)
- `mamba2_tradeoff`: Mamba2 has better gradient quality score than Elman

### Expressivity-Gradient Tradeoff Analysis (Complete)

**File**: `ElmanProofs/Expressivity/ExpressivityGradientTradeoff.lean`

Formalizes TWO INDEPENDENT DIMENSIONS for RNN architecture evaluation:

#### Learning Efficiency: Gradient Condition Number κ
The ratio of max/min gradient magnitude. Lower κ = more stable training.

| Architecture | Gradient Bounds | κ (Condition Number) |
|--------------|-----------------|---------------------|
| Stock Elman | [0, 1] | ∞ (can vanish) |
| Residual Elman | [1, 2] | 2 |
| Linear RNN | constant | 1 (but limited expressivity) |

Key theorems:
- `stock_elman_gradient_bounds`: Gradient factor ∈ [0, 1]
- `residual_elman_gradient_bounds_full`: 1 ≤ 1 + tanh'(x) ≤ 2
- `residual_finite_condition_number`: κ = 2 for residual
- `stock_infinite_condition_number`: κ = ∞ for stock Elman

#### Expressivity: Structural Property
Can the architecture break linear structure?

| Architecture | Expressivity Class |
|--------------|-------------------|
| Linear RNN | Linear (limited to n-dim subspace) |
| Stock Elman | Nonlinear (breaks linear structure) |
| Residual Elman | Nonlinear (same as stock!) |

Key theorems:
- `LinearExpressivity`, `NonlinearExpressivity`: Formal definitions
- `nonlinear_implies_not_linear`: Structural separation
- `tanh_compresses_ratio`: tanh(100)/tanh(1) < 100 (compression proves nonlinearity)
- `tanh_one_gt_hundredth`: tanh(1) > 1/100 (via exp bounds)

#### Summary Theorem
- `residual_elman_optimal_tradeoff`: Residual achieves κ=2 AND nonlinear expressivity

#### Experimental Validation (batch=256)
| Model | Tok/s | Loss | Analysis |
|-------|-------|------|----------|
| X-Gated Elman | 166k | 1.89 | κ~2, nonlinear |
| Mamba2 | 96k | 1.84 | κ=1, linear in h |

X-Gated is 1.73x FASTER. h-dependence in gates hurts; h-dependence in core computation helps.

### Proof Chain (All Complete)

1. `lsmooth_fundamental_ineq` - Fundamental inequality for L-smooth functions
2. `lsmooth_cocoercivity` - Baillon-Haddad theorem: ‖∇f(x)‖² ≤ L⟨∇f(x), x - x*⟩
3. `strong_convex_gradient_monotonicity` - Gradient monotonicity for strongly convex functions
4. `strong_smooth_interpolation` - The key interpolation bound combining both properties
5. `convex_convergence_rate` - O(1/k) convergence rate

## Key Files

| File | Purpose |
|------|---------|
| `ElmanProofs/Expressivity/Associativity.lean` | Associativity separation proofs |
| `ElmanProofs/Expressivity/GradientDynamics.lean` | Mamba2 vs Elman gradient analysis |
| `ElmanProofs/Expressivity/ExpressivityGradientTradeoff.lean` | **NEW** Two-layer expressivity-gradient tradeoff |
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

### Expressivity Research
1. **Log-polynomial gradient bounds**: Prove gradient bounds for |x|^α activation (`LogPolynomialGradient.lean`)
2. **Nonlinear universality**: Prove tanh RNN is a universal approximator (`Universality.lean`)
3. **Memory capacity**: Formalize effective memory capacity of different architectures

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
