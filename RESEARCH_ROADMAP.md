# Expressivity & Information-Theoretic Bounds: Research Roadmap

## Overview

This roadmap tracks formal proofs establishing the theoretical foundations for comparing:
- **Linear recurrence** (Mamba2, S4D, minGRU): `h_{t+1} = A h_t + B x_t`
- **Nonlinear recurrence** (Elman, GRU): `h_{t+1} = σ(W_h h_t + W_x x_t)`
- **Log-polynomial** (novel): `log|h_{t+1}| = α log|v_t|` where `v = W_h h + W_x x`

The goal is to rigorously establish what each architecture class can and cannot do.

---

## Phase 1: Foundational Separations

### 1.1 Associativity Separation [MOSTLY COMPLETE]

**Goal**: Prove linear recurrence is associative (enabling parallel scan) while nonlinear is not.

**Theorems proven**:

- [x] `LinearScanElement.instMonoid`: Linear recurrence forms a monoid under composition
  ```lean
  structure LinearScanElement (n : ℕ) where
    A : Matrix (Fin n) (Fin n) ℝ
    b : Fin n → ℝ

  instance instMonoid (n : ℕ) : Monoid (LinearScanElement n) where
    mul := fun e₂ e₁ => ⟨e₂.A * e₁.A, e₂.A.mulVec e₁.b + e₂.b⟩
    one := ⟨1, 0⟩
    mul_assoc := mul_assoc  -- proven!
    one_mul := one_mul      -- proven!
    mul_one := mul_one      -- proven!
  ```

- [x] `polynomial_composition_structure`: Pure power functions compose nicely
  ```lean
  theorem polynomial_composition_structure :
    ∀ (α : ℝ), α > 0 → ∀ (a b : ℝ),
    (|a * |b|.rpow α|).rpow α = |a|.rpow α * |b|.rpow (α * α)
  ```

- [x] `tanh_strictMono`: Tanh is strictly monotone (helper for counterexamples)
- [x] `tanh_injective`: Tanh is injective (follows from strict monotonicity)

**Theorems with numerical sorries** (structure proven, needs calculus bounds):

- [ ] `tanh_composition_not_linear`: Tanh RNN cannot be reduced to single affine step
  - Proof structure complete: shows tanh(tanh(h)) ≠ tanh(a*h + b)
  - Needs: proof that tanh(x)/x is strictly decreasing for x > 0

- [ ] `polynomial_rnn_not_associative`: Polynomial RNN is also non-associative
  - Proof structure complete: shows ||h+1|^α + 1|^α ≠ |a*h + b|^α
  - Needs: similar numerical bounds

**Key insight**: The associativity of linear recurrence is WHY Mamba2 can use parallel scan. Nonlinear models pay a computational cost (sequential processing) for expressivity.

**Status**: Core monoid structure COMPLETE, counterexamples need numerical bounds
**Location**: `ElmanProofs/Expressivity/Associativity.lean`

---

### 1.2 Linear State Capacity Bound [COMPLETE]

**Goal**: Prove that d-dimensional linear RNN state carries at most O(d) bits of information about the input history.

**All theorems proven**:

- [x] `linear_state_is_sum`: State = Σ A^{T-1-t} B x_t (explicit sum formula!)
- [x] `state_additive`: state(x₁ + x₂) = state(x₁) + state(x₂)
- [x] `state_scalar`: state(c • x) = c • state(x)
- [x] `output_determined_by_state`: Output depends only on current state
- [x] `same_state_same_future`: If two sequences produce same state, all future outputs are identical
- [x] `reachable_is_subspace`: Reachable states form a vector subspace (closed under +, •)
- [x] `reachable_dim_bound`: dim(reachable states) ≤ n
- [x] `not_linearly_computable_if_state_independent`: Key limitation theorem

**Key insight**: Linear systems cannot "compute" on their state - they can only linearly combine past inputs. This fundamentally limits what patterns they can detect.

**Status**: COMPLETE - All proofs finished
**Location**: `ElmanProofs/Expressivity/LinearCapacity.lean`

---

## Phase 2: Gradient-Expressivity Tradeoff

### 2.1 Tanh Gradient Vanishing [DONE - partial]

**Goal**: Formalize why tanh causes gradient vanishing.

**Theorems**:

- [x] `tanh_lipschitz`: tanh is 1-Lipschitz (in `Activations/Lipschitz.lean`)
- [ ] `tanh_derivative_bound`: |tanh'(x)| ≤ 1, with equality only at x = 0
- [ ] `tanh_saturation`: For |x| > c, |tanh'(x)| < ε (saturation)
- [ ] `deep_tanh_gradient_vanishing`: Product of derivatives → 0 exponentially

**Status**: Partially complete
**Location**: `ElmanProofs/Activations/Lipschitz.lean`

---

### 2.2 Log-Polynomial Gradient Bounds [NOT STARTED]

**Goal**: Prove the log-polynomial activation has bounded gradients.

**Theorems to prove**:

- [ ] `polynomial_gradient_factor`: Gradient of |x|^α is α|x|^{α-1}
  ```lean
  theorem polynomial_gradient_factor (α : ℝ) (x : ℝ) (hx : x ≠ 0) :
    deriv (fun x => |x|^α) x = α * |x|^(α - 1) * SignType.sign x
  ```

- [ ] `log_polynomial_gradient_constant`: In log-space, gradient is constant
  ```lean
  theorem log_polynomial_gradient_constant (α : ℝ) (log_x : ℝ) :
    deriv (fun log_x => α * log_x) log_x = α
  ```

- [ ] `log_polynomial_no_vanishing`: Unlike tanh, gradient doesn't vanish
  ```lean
  theorem log_polynomial_no_vanishing (α : ℝ) (hα : α > 0) :
    ∀ log_x, deriv (fun log_x => α * log_x) log_x = α  -- constant!
  ```

- [ ] `log_polynomial_chain_rule`: Through T layers, gradient is α^T (controllable)
  ```lean
  theorem log_polynomial_chain_rule (α : ℝ) (T : ℕ) :
    gradient_through_T_layers α T = α^T
  ```

**Key insight**: The log-polynomial gets bounded gradients NOT by saturating (like tanh) but by operating in log-space where multiplication becomes addition.

**Status**: Not started
**Location**: `ElmanProofs/Expressivity/LogPolynomialGradient.lean`

---

## Phase 3: Expressivity Separations

### 3.1 Functions Linear RNNs Cannot Compute [IN PROGRESS]

**Goal**: Exhibit specific functions that linear RNNs provably cannot compute.

**Theorems proven**:

- [x] `xor_not_affine`: XOR function is not affine (key lemma)
- [x] `linear_cannot_xor`: Linear RNN cannot compute XOR over history
- [x] `linear_output_as_sum`: Output is weighted sum of inputs
- [x] `linear_output_additive`: Output is additive in inputs
- [x] `linear_output_scalar`: Output is homogeneous in inputs

**Theorems with sorries**:

- [~] `linear_cannot_threshold`: Linear RNN cannot compute step function (needs continuity argument)
- [~] `linear_rnn_affine_on_binary`: Output is affine on binary inputs (technical lemma)

- [ ] `linear_cannot_count_mod_k`: Linear RNN cannot count modulo k (for k > dim)

**Status**: Core XOR impossibility proof complete
**Location**: `ElmanProofs/Expressivity/LinearLimitations.lean`

---

### 3.2 Nonlinear Universality [NOT STARTED]

**Goal**: Prove RNNs with nonlinear activations are universal approximators.

**Theorems to prove**:

- [ ] `tanh_rnn_universal`: Tanh RNN can approximate any continuous function on sequences
- [ ] `polynomial_rnn_universal`: Log-polynomial RNN is also universal (for α ≠ 1)

**Note**: This requires Stone-Weierstrass style arguments. May be complex.

**Status**: Not started
**Location**: `ElmanProofs/Expressivity/Universality.lean`

---

## Phase 4: Memory and Attention Comparisons

### 4.1 Effective Memory Capacity [NOT STARTED]

**Goal**: Formalize how much of the input history each architecture retains.

- [ ] `linear_memory_decay`: Linear RNN forgets exponentially (spectral radius)
- [ ] `nonlinear_memory_selective`: Nonlinear RNN can selectively remember
- [ ] `attention_memory_unbounded`: Attention has O(T) memory but O(T²) cost

**Status**: Not started

---

### 4.2 Computational vs Statistical Efficiency [NOT STARTED]

**Goal**: Characterize sample complexity vs computational cost tradeoffs.

- [ ] `linear_sample_efficient`: Linear models need fewer samples for linear targets
- [ ] `nonlinear_compute_expensive`: Nonlinear models require sequential computation

**Status**: Not started

---

## Summary Table

| Theorem | File | Status | Difficulty |
|---------|------|--------|------------|
| `LinearScanElement.instMonoid` | Associativity.lean | [x] DONE | Medium |
| `polynomial_composition_structure` | Associativity.lean | [x] DONE | Medium |
| `tanh_strictMono` | Associativity.lean | [x] DONE | Easy |
| `tanh_injective` | Associativity.lean | [x] DONE | Easy |
| `tanh_composition_not_linear` | Associativity.lean | [~] sorry | Medium |
| `polynomial_rnn_not_associative` | Associativity.lean | [~] sorry | Medium |
| `output_determined_by_state` | LinearCapacity.lean | [x] DONE | Easy |
| `same_state_same_future` | LinearCapacity.lean | [x] DONE | Medium |
| `not_linearly_computable_if_state_independent` | LinearCapacity.lean | [x] DONE | Medium |
| `linear_state_is_sum` | LinearCapacity.lean | [x] DONE | Medium |
| `state_additive` | LinearCapacity.lean | [x] DONE | Easy |
| `state_scalar` | LinearCapacity.lean | [x] DONE | Easy |
| `reachable_dim_bound` | LinearCapacity.lean | [x] DONE | Medium |
| `polynomial_gradient_factor` | LogPolynomialGradient.lean | [ ] | Easy |
| `log_polynomial_gradient_constant` | LogPolynomialGradient.lean | [ ] | Easy |
| `log_polynomial_chain_rule` | LogPolynomialGradient.lean | [ ] | Medium |
| `xor_not_affine` | LinearLimitations.lean | [x] DONE | Easy |
| `linear_cannot_xor` | LinearLimitations.lean | [x] DONE | Medium |
| `linear_cannot_threshold` | LinearLimitations.lean | [~] sorry | Medium |
| `linear_rnn_affine_on_binary` | LinearLimitations.lean | [~] sorry | Medium |

---

## Dependencies

```
Associativity.lean
    ↓
LinearCapacity.lean ← depends on associativity for state composition
    ↓
LinearLimitations.lean ← uses capacity bounds
    ↓
LogPolynomialGradient.lean ← independent, can be done in parallel
    ↓
Universality.lean ← depends on all above
```

---

## Next Actions

1. ~~Create `ElmanProofs/Expressivity/` directory~~ DONE
2. ~~Start with `Associativity.lean` - prove linear recurrence forms a monoid~~ DONE
3. ~~Complete `LinearCapacity.lean` - prove state is linear combination of inputs~~ DONE
4. Complete counterexample proofs (need numerical bounds for tanh(x)/x decreasing)
5. Start `LinearLimitations.lean` - prove linear RNNs cannot compute threshold functions

---

*Last updated: 2026-01-02*
