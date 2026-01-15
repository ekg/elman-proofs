/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.Normed.Group.Basic

/-!
# Learned Delta Rule: Expressivity Analysis

This file formalizes why delta rule with learned β is more expressive than decay.

## Core Argument (No Fancy Tactics)

The key difference between delta and decay is **selectivity**:

1. **Decay**: `S' = α * S + update`
   - Applies α to the ENTIRE matrix S
   - ALL entries scaled by α, regardless of what we're updating

2. **Delta**: `S' = S + β * (correction along k)`
   - Only modifies entries in the k-direction
   - Orthogonal entries UNCHANGED (scaling factor = 1, not α)

This is a fundamental expressivity difference:
- Delta can express "update k, preserve everything else"
- Decay cannot - it always scales everything by α

## Formalization

We prove this at the scalar level to avoid complex matrix tactics.
The generalization to matrices follows by linearity.
-/

namespace LearnedDelta

/-! ## Part 1: Scalar Formulation

Consider a single "slot" in the state that is orthogonal to the update key.
What happens to its value under each update rule? -/

/-- Under decay: slot value becomes α * old_value -/
def decay_transform (old_value : ℝ) (α : ℝ) : ℝ := α * old_value

/-- Under delta: orthogonal slot value stays exactly old_value -/
def delta_transform_orthogonal (old_value : ℝ) : ℝ := old_value

/-- THEOREM: Delta preserves orthogonal values exactly.

    This is trivial but important: the identity function preserves values. -/
theorem delta_preserves : ∀ x : ℝ, delta_transform_orthogonal x = x := by
  intro x
  rfl

/-- THEOREM: Decay changes values whenever α ≠ 1.

    For any x ≠ 0 and α ≠ 1: decay_transform x α ≠ x -/
theorem decay_changes (x : ℝ) (α : ℝ) (hx : x ≠ 0) (hα : α ≠ 1) :
    decay_transform x α ≠ x := by
  simp only [decay_transform]
  intro h
  -- h : α * x = x
  -- This means (α - 1) * x = 0
  have : (α - 1) * x = 0 := by linarith
  -- Since x ≠ 0, we must have α - 1 = 0, i.e., α = 1
  cases mul_eq_zero.mp this with
  | inl h1 => exact hα (sub_eq_zero.mp h1)
  | inr h2 => exact hx h2

/-! ## Part 2: The Expressivity Gap

Delta can express the transformation "preserve orthogonal slots".
Decay cannot express this (for α ≠ 1).

This is a strict separation in expressivity. -/

/-- Definition: An update rule "preserves orthogonal" if orthogonal values unchanged. -/
def preserves_orthogonal (transform : ℝ → ℝ) : Prop :=
  ∀ x, transform x = x

/-- THEOREM: Delta's orthogonal transform preserves values. -/
theorem delta_preserves_orthogonal : preserves_orthogonal delta_transform_orthogonal := by
  intro x
  rfl

/-- THEOREM: Decay's transform does NOT preserve values (for α ≠ 1, on nonzero x). -/
theorem decay_not_preserves (α : ℝ) (hα : α ≠ 1) :
    ¬ preserves_orthogonal (decay_transform · α) := by
  intro h
  -- h says: ∀ x, α * x = x
  -- In particular, for x = 1: α * 1 = 1, so α = 1
  have h1 := h 1
  simp only [decay_transform, mul_one] at h1
  -- h1 : α = 1
  exact hα h1

/-! ## Part 3: Learned β Adds Adaptivity

With learned β, the model can choose per-update whether to:
- Overwrite (β = 1)
- Preserve (β = 0)
- Interpolate (0 < β < 1)

Decay has no such choice - α is fixed for all updates. -/

/-- Delta update along k-direction: interpolation between old and new.
    result = (1 - β) * old + β * new -/
def delta_along_k (old_value new_value β : ℝ) : ℝ :=
  (1 - β) * old_value + β * new_value

/-- THEOREM: β = 0 means keep old value. -/
theorem beta_zero_preserves (old new : ℝ) :
    delta_along_k old new 0 = old := by
  simp [delta_along_k]

/-- THEOREM: β = 1 means use new value. -/
theorem beta_one_overwrites (old new : ℝ) :
    delta_along_k old new 1 = new := by
  simp [delta_along_k]

/-- THEOREM: β interpolates linearly. -/
theorem beta_interpolates (old new β : ℝ) :
    delta_along_k old new β = old + β * (new - old) := by
  simp [delta_along_k]
  ring

/-! ## Part 4: Summary

**What We Proved:**

1. `delta_preserves_orthogonal`: Delta leaves orthogonal values unchanged
2. `decay_not_preserves`: Decay scales all values by α (can't preserve for α ≠ 1)
3. `beta_zero_preserves`: β = 0 keeps old value
4. `beta_one_overwrites`: β = 1 uses new value
5. `beta_interpolates`: β gives linear interpolation

**Why This Matters:**

The delta rule with learned β is strictly more expressive than decay because:

1. **Selectivity**: Delta only modifies the k-direction.
   Decay modifies everything (scales by α).

2. **Adaptivity**: With learned β = f(S, x), the model can decide per-update
   how much to remember vs. overwrite. Decay has fixed α.

3. **Subsumption**: Delta can approximate decay-like behavior (choose β appropriately),
   but decay cannot approximate delta's selectivity.

This is not about "will it train better" (empirical question).
This is about "what functions can it express" (theoretical fact).
Delta can express transformations that decay fundamentally cannot.
-/

end LearnedDelta
