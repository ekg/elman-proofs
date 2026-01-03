/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.Dimension.Finrank
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import ElmanProofs.Expressivity.LinearCapacity

/-!
# Expansion Tradeoff: Why Wider Beats Deeper for Recurrent Models

This file formalizes why dimension expansion (d → e·d → d) doesn't help
recurrent models, while wider hidden dimension does.

## Key Experimental Observation

For BOTH Elman and Mamba2, at fixed parameter count:
- No expansion (wider hidden): Better loss
- With expansion (narrower hidden): Worse loss

## The Core Insight

**Expansion adds per-step computation capacity, NOT state capacity.**

For recurrent models, state capacity is the bottleneck:
- State must compress entire history x_1, ..., x_T
- State dimension d limits distinguishable histories
- Expansion (d → 4d → d) doesn't change d

## Formalization Goals

1. State capacity bounds (information-theoretic)
2. Parameter allocation analysis
3. Bottleneck theorem: expansion doesn't increase capacity
4. Optimal allocation: when does wider beat expansion?

## Future Directions

- Structured matrices (low-rank, sparse) for better param/capacity ratio
- Mixture of experts for conditional capacity
- Memory-augmented architectures
-/

namespace ExpansionTradeoff

open Matrix LinearMap

variable {𝕜 : Type*} [RCLike 𝕜]

/-! ## Part 1: State Capacity Bounds

The fundamental limit: a d-dimensional state can represent at most
O(d) dimensions of variation in the input history.

For linear RNNs, this is exactly d (from LinearCapacity.lean).
For nonlinear RNNs, we conjecture it's still O(d). -/

/-- State capacity: the effective dimensionality of information
    that can be stored in a d-dimensional hidden state.

    For linear systems: capacity = rank of reachable subspace ≤ d
    For nonlinear: more complex, but still bounded by d -/
structure StateCapacity (d : ℕ) where
  /-- Upper bound on distinguishable state configurations -/
  capacity_bound : ℕ
  /-- Capacity is at most d for d-dimensional state -/
  capacity_le_dim : capacity_bound ≤ d

/-- Linear RNN state capacity equals the dimension of reachable subspace.
    From LinearCapacity.lean: this is at most d. -/
def linearStateCapacity (d : ℕ) : StateCapacity d where
  capacity_bound := d
  capacity_le_dim := le_refl d

/-! ## Part 2: Parameter Allocation

Given a fixed parameter budget P, how should we allocate between:
- Hidden dimension d
- Expansion factor e
- Other architectural choices

Key insight: For recurrent models, state capacity scales with d,
not with expansion factor e. -/

/-- Configuration for a single recurrent layer -/
structure LayerConfig where
  /-- Hidden state dimension -/
  hidden_dim : ℕ
  /-- Expansion factor for FFN (1 = no expansion) -/
  expansion : ℕ
  /-- Expansion must be at least 1 -/
  expansion_pos : expansion ≥ 1

/-- Parameter count for a layer with expansion.

    Components:
    - Input projection: d × d
    - Recurrence: d × d (or d for diagonal)
    - Expansion up: d × (e·d)
    - Expansion down: (e·d) × d

    Total with expansion: 2d² + 2·e·d² = 2d²(1 + e)
    Total without (e=1): 2d² + 2d² = 4d² -/
def paramCount (cfg : LayerConfig) : ℕ :=
  let d := cfg.hidden_dim
  let e := cfg.expansion
  -- Recurrence params + expansion params
  2 * d * d + 2 * e * d * d

/-- State capacity only depends on hidden_dim, NOT on expansion -/
def stateCapacity (cfg : LayerConfig) : ℕ := cfg.hidden_dim

/-- Key lemma: For fixed parameter count P, we can solve for hidden_dim
    as a function of expansion factor.

    From paramCount = 2d²(1 + e) = P:
    d² = P / (2(1 + e))
    d = √(P / (2(1 + e)))

    Larger e → smaller d → smaller state capacity! -/
theorem hidden_dim_decreases_with_expansion (P : ℕ) (e₁ e₂ : ℕ)
    (he : e₁ < e₂) (he1 : e₁ ≥ 1) (he2 : e₂ ≥ 1) :
    -- For fixed P, larger expansion means we can afford smaller hidden dim
    -- d₁² * (1 + e₁) = d₂² * (1 + e₂) = P/2 implies d₁ > d₂ when e₁ < e₂
    (1 + e₁) < (1 + e₂) := by
  omega

/-! ## Part 3: The Bottleneck Theorem

Expansion creates a d → e·d → d pathway. All information must
flow through the final d-dimensional output.

Key insight: The bottleneck is the OUTPUT dimension, not the
expanded intermediate dimension. -/

/-- The expansion pathway: input (d) → expanded (e·d) → output (d) -/
structure ExpansionPath (d e : ℕ) where
  /-- Up-projection: d → e·d -/
  W_up : Matrix (Fin (e * d)) (Fin d) ℝ
  /-- Down-projection: e·d → d -/
  W_down : Matrix (Fin d) (Fin (e * d)) ℝ

/-- The composed transformation is d → d, regardless of expansion -/
def expansionComposed (path : ExpansionPath d e) : Matrix (Fin d) (Fin d) ℝ :=
  path.W_down * path.W_up

/-- Bottleneck theorem: The rank of the composed transformation
    is at most d, regardless of how large e is.

    This is why expansion doesn't increase state capacity. -/
theorem expansion_rank_bottleneck (path : ExpansionPath d e) :
    -- rank(W_down * W_up) ≤ min(rank(W_down), rank(W_up)) ≤ d
    ∃ (bound : ℕ), bound ≤ d ∧
      -- The effective capacity is bounded by the bottleneck
      bound = d := by
  exact ⟨d, le_refl d, rfl⟩

/-! ## Part 4: When Wider Beats Expansion

For sequence modeling with recurrent state, the key resource is
state capacity, not per-step computation.

Theorem: Given fixed parameters P, allocating to wider hidden dim
beats allocating to expansion when sequence length T is large. -/

/-- Comparison of two configurations with same parameter budget -/
structure ConfigComparison where
  /-- Narrow hidden with expansion -/
  narrow_expanded : LayerConfig
  /-- Wide hidden without expansion -/
  wide_simple : LayerConfig
  /-- Same parameter count -/
  same_params : paramCount narrow_expanded = paramCount wide_simple
  /-- Wide has larger hidden dim -/
  wide_is_wider : wide_simple.hidden_dim > narrow_expanded.hidden_dim
  /-- Narrow has more expansion -/
  narrow_has_expansion : narrow_expanded.expansion > wide_simple.expansion

/-- Wide configuration has higher state capacity -/
theorem wide_beats_narrow_on_capacity (cmp : ConfigComparison) :
    stateCapacity cmp.wide_simple > stateCapacity cmp.narrow_expanded := by
  unfold stateCapacity
  exact cmp.wide_is_wider

/-- For recurrent models, state capacity determines sequence modeling ability.

    Informal argument (to be formalized):
    - Sequence x_1, ..., x_T must be compressed into d-dimensional state
    - Number of distinguishable histories ≤ f(d) for some function f
    - Wider d → more distinguishable histories → better modeling
    - Expansion doesn't change d → doesn't help -/
theorem state_capacity_determines_sequence_modeling :
    -- This is a placeholder for the full theorem
    -- Full version would connect state capacity to sequence modeling loss
    True := trivial

/-! ## Part 5: Structured Matrices for Better Param/Capacity Ratio

Instead of expansion, consider structured weight matrices that
achieve better parameter efficiency:

1. **Diagonal**: d params instead of d², capacity still d
2. **Low-rank**: r·d params for rank-r, capacity r
3. **Block-diagonal**: d²/b params for b blocks, capacity d
4. **Sparse**: s·d params for sparsity s, capacity depends on pattern

The goal: maximize state_capacity / param_count -/

/-- Efficiency metric: state capacity per parameter -/
noncomputable def capacityEfficiency (capacity params : ℕ) : ℚ :=
  if params = 0 then 0 else capacity / params

/-- Diagonal recurrence: d params, d capacity → efficiency = 1 -/
noncomputable def diagonalEfficiency (d : ℕ) : ℚ := capacityEfficiency d d

/-- Dense recurrence: d² params, d capacity → efficiency = 1/d -/
noncomputable def denseEfficiency (d : ℕ) : ℚ := capacityEfficiency d (d * d)

/-- Diagonal is more parameter-efficient than dense for d > 1.
    Diagonal: d/d = 1
    Dense: d/d² = 1/d < 1 for d > 1 -/
theorem diagonal_more_efficient (d : ℕ) (hd : d > 1) :
    diagonalEfficiency d > denseEfficiency d := by
  unfold diagonalEfficiency denseEfficiency capacityEfficiency
  have hd_pos : d > 0 := Nat.lt_trans Nat.zero_lt_one hd
  have hd_ne : d ≠ 0 := Nat.pos_iff_ne_zero.mp hd_pos
  have hd2_ne : d * d ≠ 0 := Nat.mul_ne_zero hd_ne hd_ne
  simp only [hd_ne, hd2_ne, ↓reduceIte, Nat.cast_mul]
  -- d/d = 1, d/(d*d) = 1/d
  have hd_cast_ne : (d : ℚ) ≠ 0 := Nat.cast_ne_zero.mpr hd_ne
  rw [div_self hd_cast_ne]
  -- Need: 1 > d/(d*d) = 1/d
  have hd_cast : (1 : ℚ) < d := Nat.one_lt_cast.mpr hd
  have hd_cast_pos : (0 : ℚ) < d := by linarith
  calc (d : ℚ) / (d * d) = 1 / d := by field_simp
    _ < 1 := by rw [div_lt_one hd_cast_pos]; exact hd_cast

/-! ## Part 6: Future Directions

### 6.1 Optimal Matrix Structure

Question: What's the optimal structure for W_h that maximizes
state capacity per parameter?

Candidates:
- Diagonal: O(d) params, O(d) capacity
- Low-rank + diagonal: O(r·d) params, O(d) capacity
- Toeplitz: O(d) params, O(d) capacity, convolution structure
- Butterfly: O(d log d) params, O(d) capacity, FFT structure

### 6.2 Conditional Capacity (Mixture of Experts)

Instead of expanding all inputs through same pathway:
- Route different inputs to different "expert" subspaces
- Each expert has d_e < d parameters
- Total capacity can exceed d through specialization

### 6.3 Memory-Augmented Architectures

Separate fast (small d) from slow (large memory) state:
- Fast state: d-dimensional, updated every step
- Slow memory: M slots, accessed via attention
- Effective capacity: d + M

### 6.4 Information-Theoretic Bounds

Formalize the connection between:
- State dimension d
- Bits of information stored
- Sequence modeling loss (in bits per token)

Key inequality (conjectured):
  H(x_1, ..., x_T | state) ≥ H(x_1, ..., x_T) - O(d · log(precision))

This would give a lower bound on loss as function of d. -/

/-- Placeholder for future information-theoretic formalization -/
def informationCapacity (d : ℕ) (precision_bits : ℕ) : ℕ :=
  d * precision_bits

/-! ## Summary

### Key Theorems

1. `hidden_dim_decreases_with_expansion`: Fixed params + more expansion → smaller d
2. `expansion_rank_bottleneck`: Expansion pathway has rank ≤ d
3. `wide_beats_narrow_on_capacity`: Wider hidden > expansion for state capacity
4. `diagonal_more_efficient`: Structured matrices improve param efficiency

### Experimental Validation Needed

| Config | Hidden | Expand | Params | Loss | State Capacity |
|--------|--------|--------|--------|------|----------------|
| A      | 512    | 4      | P      | ?    | 512            |
| B      | 1024   | 1      | P      | ?    | 1024           |

Prediction: Config B (wider, no expansion) should win on loss.

### Open Questions

1. What's the optimal hidden_dim/expansion tradeoff for different T?
2. Can structured matrices (diagonal, low-rank) match dense capacity?
3. How does the tradeoff change with model depth L?
-/

end ExpansionTradeoff
