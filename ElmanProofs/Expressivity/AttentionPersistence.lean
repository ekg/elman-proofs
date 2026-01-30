/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Data.Finset.Basic
import Mathlib.Order.Filter.Basic
import Mathlib.Analysis.Complex.ExponentialBounds
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.LinearLimitations
import ElmanProofs.Expressivity.NumericalBounds

/-!
# Attention Persistence: Alert Mode Latching in E88

This file formalizes **attention persistence**, the key property that allows E88
heads to enter an "alert" state and maintain it indefinitely. This is analogous
to a finite state machine entering an absorbing state.

## Key Insight

The tanh function has saturation: as |x| → ∞, tanh(x) → ±1 and tanh'(x) → 0.
This creates stable fixed points. For the recurrence S_{t+1} = tanh(αS_t + input):

1. **Fixed points exist**: For appropriate α, there exist S* with tanh(αS*) = S*
2. **Stability**: Near these fixed points, the derivative is small, creating basins
3. **Latching**: Once |S| is large, it stays large (the "alert" state persists)

## Main Results

### Fixed Point Analysis
* `tanh_recurrence_has_fixed_point` - tanh(α·S) = S has solutions for α ≥ 1
* `nonzero_fixed_point_exists` - For α > 1, nonzero fixed points exist
* `fixed_point_is_attractive` - Fixed points are locally stable attractors

### Alert State Basin
* `alert_threshold_exists` - There exists θ such that |S| > θ implies persistence
* `alert_state_forward_invariant` - Alert states remain alert under tanh iteration
* `alert_basin_nonempty` - The alert basin is non-empty

### Persistence Properties
* `latched_state_persists_under_perturbation` - Small perturbations don't knock out
* `attention_persistence_theorem` - Main result: E88 heads can latch and persist
* `linear_cannot_latch` - Linear systems have no stable nonzero fixed points

## Connection to E88 Architecture

In E88, each head has the update rule:
```
S_t := tanh(α·S_{t-1} + δ·input_t)
```

When a head detects a relevant pattern (large δ·input), its state can jump into
the "alert" region (|S| close to 1). Due to tanh saturation, this alert state
persists even when subsequent inputs are small.

This is the formal mechanism behind **attention persistence**: the head "pays
attention" to a feature and continues to remember it, unlike linear systems
where information decays.

-/

namespace AttentionPersistence

open Real Matrix Finset BigOperators Filter

/-! ## Part 1: Basic Tanh Recurrence Properties -/

/-- The tanh recurrence function: S → tanh(α·S).
    This is the simplified E88 update with no input. -/
noncomputable def tanhRecur (α : ℝ) (S : ℝ) : ℝ := tanh (α * S)

/-- Iterated tanh recurrence: applying tanhRecur n times. -/
noncomputable def tanhRecurIter (α : ℝ) : ℕ → ℝ → ℝ
  | 0, S => S
  | n + 1, S => tanhRecur α (tanhRecurIter α n S)

/-- The identity tanhRecurIter n+1 S = tanhRecur α (tanhRecurIter n S). -/
theorem tanhRecurIter_succ (α : ℝ) (n : ℕ) (S : ℝ) :
    tanhRecurIter α (n + 1) S = tanhRecur α (tanhRecurIter α n S) := rfl

/-- tanhRecur preserves the bounded interval (-1, 1). -/
theorem tanhRecur_bounded (α : ℝ) (S : ℝ) : |tanhRecur α S| < 1 := by
  simp only [tanhRecur]
  exact Activation.tanh_bounded (α * S)

/-- tanhRecur is strictly monotone for α > 0 (since tanh is strictly monotone). -/
theorem tanhRecur_strictMono (α : ℝ) (hα : 0 < α) : StrictMono (tanhRecur α) := by
  intro x y hxy
  simp only [tanhRecur]
  apply Activation.tanh_strictMono
  exact mul_lt_mul_of_pos_left hxy hα

/-- tanhRecur is continuous. -/
theorem tanhRecur_continuous (α : ℝ) : Continuous (tanhRecur α) := by
  unfold tanhRecur
  -- tanh is Lipschitz hence continuous, and composition with linear is continuous
  have h_lip := Activation.tanh_lipschitz
  exact h_lip.continuous.comp (continuous_mul_left α)

/-! ## Part 2: Fixed Point Existence -/

/-- A fixed point of tanhRecur α is a solution to tanh(αS) = S. -/
def isFixedPoint (α : ℝ) (S : ℝ) : Prop := tanhRecur α S = S

/-- Zero is always a fixed point: tanh(0) = 0. -/
theorem zero_is_fixed_point (α : ℝ) : isFixedPoint α 0 := by
  simp [isFixedPoint, tanhRecur, tanh_zero]

/-- For α ≤ 1, zero is the only fixed point.
    Key insight: The derivative of tanh(αx) at x=0 is α·tanh'(0) = α·1 = α.
    For α ≤ 1, the curve y = tanh(αx) stays below y = x for x > 0. -/
theorem unique_fixed_point_for_small_alpha (α : ℝ) (hα_pos : 0 < α) (hα_le : α ≤ 1) :
    ∀ S : ℝ, isFixedPoint α S → S = 0 := by
  intro S hS
  simp only [isFixedPoint, tanhRecur] at hS
  -- hS : tanh(αS) = S
  -- For S > 0: tanh(αS) < αS ≤ S since tanh(x) < x for x > 0 and α ≤ 1
  -- For S < 0: tanh(αS) > αS ≥ S by symmetry
  -- Only S = 0 works
  by_contra h_ne
  rcases lt_trichotomy S 0 with h_neg | h_zero | h_pos
  · -- S < 0 case
    have h1 : α * S < 0 := mul_neg_of_pos_of_neg hα_pos h_neg
    have h2 : tanh (α * S) > α * S := by
      -- tanh(x) > x for x < 0
      -- Use oddness: tanh(x) = -tanh(-x), and for -x > 0, tanh(-x) < -x
      have h_neg_arg : 0 < -(α * S) := by linarith
      -- For -(α*S) > 0: tanh(-(α*S)) < -(α*S)
      have h_pos_case : tanh (-(α * S)) < -(α * S) := by
        -- Apply MVT on [0, -(α*S)]
        have h_cont : ContinuousOn tanh (Set.Icc 0 (-(α * S))) :=
          Activation.differentiable_tanh.continuous.continuousOn
        have h_diff : DifferentiableOn ℝ tanh (Set.Ioo 0 (-(α * S))) :=
          Activation.differentiable_tanh.differentiableOn
        obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope tanh h_neg_arg h_cont h_diff
        rw [tanh_zero, sub_zero, sub_zero] at h_mvt
        have h_deriv_c_lt : deriv tanh c < 1 := by
          rw [Activation.deriv_tanh]
          have h_tanh_c_ne : tanh c ≠ 0 := by
            intro heq_c
            have h_c_eq_0 : c = 0 := Activation.tanh_injective (heq_c.trans tanh_zero.symm)
            linarith
          have h_sq_pos : 0 < (tanh c)^2 := sq_pos_of_ne_zero h_tanh_c_ne
          linarith
        have h_ne : -(α * S) ≠ 0 := ne_of_gt h_neg_arg
        have h_slope : tanh (-(α * S)) = deriv tanh c * (-(α * S)) := by
          field_simp at h_mvt
          linarith
        calc tanh (-(α * S)) = deriv tanh c * (-(α * S)) := h_slope
          _ < 1 * (-(α * S)) := mul_lt_mul_of_pos_right h_deriv_c_lt h_neg_arg
          _ = -(α * S) := one_mul _
      -- Now use tanh(-x) = -tanh(x)
      rw [tanh_neg] at h_pos_case
      linarith
    have h3 : α * S ≥ S := by
      -- αS ≥ S ↔ S(α-1) ≥ 0
      -- S < 0 and α - 1 ≤ 0, so S(α-1) ≥ 0
      nlinarith
    -- From h2: tanh(αS) > αS, and h3: αS ≥ S
    -- So tanh(αS) > S, contradicting hS: tanh(αS) = S
    linarith
  · exact absurd h_zero h_ne
  · -- S > 0 case
    have h1 : 0 < α * S := mul_pos hα_pos h_pos
    have h2 : tanh (α * S) < α * S := by
      -- tanh(x) < x for x > 0 by MVT
      -- Apply MVT on [0, α*S]
      have h_cont : ContinuousOn tanh (Set.Icc 0 (α * S)) :=
        Activation.differentiable_tanh.continuous.continuousOn
      have h_diff : DifferentiableOn ℝ tanh (Set.Ioo 0 (α * S)) :=
        Activation.differentiable_tanh.differentiableOn
      obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope tanh h1 h_cont h_diff
      rw [tanh_zero, sub_zero, sub_zero] at h_mvt
      have h_deriv_c_lt : deriv tanh c < 1 := by
        rw [Activation.deriv_tanh]
        have h_tanh_c_ne : tanh c ≠ 0 := by
          intro heq_c
          have h_c_eq_0 : c = 0 := Activation.tanh_injective (heq_c.trans tanh_zero.symm)
          linarith
        have h_sq_pos : 0 < (tanh c)^2 := sq_pos_of_ne_zero h_tanh_c_ne
        linarith
      have h_ne : α * S ≠ 0 := ne_of_gt h1
      have h_slope : tanh (α * S) = deriv tanh c * (α * S) := by
        field_simp at h_mvt
        linarith
      calc tanh (α * S) = deriv tanh c * (α * S) := h_slope
        _ < 1 * (α * S) := mul_lt_mul_of_pos_right h_deriv_c_lt h1
        _ = α * S := one_mul _
    have h3 : α * S ≤ S := by
      -- αS ≤ S ↔ S(α-1) ≤ 0
      -- S > 0 and α - 1 ≤ 0, so S(α-1) ≤ 0
      nlinarith
    -- From h2: tanh(αS) < αS, and h3: αS ≤ S
    -- So tanh(αS) < S, contradicting hS: tanh(αS) = S
    linarith

/-- For α > 1, nonzero fixed points exist.
    Key insight: At x = 0, the slope of tanh(αx) is α > 1 (steeper than y = x).
    Since tanh is bounded by 1, the curve must cross y = x at some x* > 0.
    By odd symmetry, -x* is also a fixed point. -/
theorem nonzero_fixed_point_exists (α : ℝ) (hα : 1 < α) :
    ∃ S : ℝ, S ≠ 0 ∧ isFixedPoint α S := by
  -- We use the IVT approach: Define g(x) = tanh(αx) - x.
  -- g(0) = 0, g(1) = tanh(α) - 1 < 0
  -- Since g'(0) = α - 1 > 0, g(ε) > 0 for small ε > 0.
  -- By IVT, there exists c ∈ (ε, 1) with g(c) = 0.

  have hα_pos : 0 < α := by linarith
  have h_deriv_pos : 0 < α - 1 := by linarith

  -- Define g
  let g := fun x => tanh (α * x) - x

  -- Helper: x ↦ α * x is differentiable
  have h_mul_diff : Differentiable ℝ (fun x => α * x) := Differentiable.const_mul differentiable_id α

  -- g is differentiable
  have h_g_diff : Differentiable ℝ g := by
    intro x
    apply DifferentiableAt.sub
    · exact Activation.differentiable_tanh.differentiableAt.comp x (h_mul_diff x)
    · exact differentiableAt_id

  -- g'(0) = α - 1 > 0
  have h_g_deriv_0 : deriv g 0 = α - 1 := by
    -- Compute deriv (tanh ∘ (α * ·)) at 0 using chain rule
    have h_tanh_deriv : HasDerivAt (fun x => tanh (α * x)) (α * (1 - (tanh (α * 0))^2)) 0 := by
      have h1 : HasDerivAt (fun x => α * x) α 0 := by
        have h1' := (hasDerivAt_id 0).const_mul α
        simp only [id, mul_one] at h1'
        exact h1'
      have h2 : HasDerivAt tanh (1 - (tanh (α * 0))^2) (α * 0) := by
        have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := α * 0)
        rw [Activation.deriv_tanh] at hd
        exact hd
      have h3 := h2.comp 0 h1
      simp only [Function.comp_apply] at h3
      convert h3 using 1
      ring
    simp only [mul_zero, tanh_zero, sq, sub_zero, mul_one] at h_tanh_deriv
    have h1 : deriv (fun x => tanh (α * x)) (0 : ℝ) = α := h_tanh_deriv.deriv
    have h2 : deriv (fun x : ℝ => x) (0 : ℝ) = (1 : ℝ) := by
      have := deriv_id'' (𝕜 := ℝ)
      exact congrFun this 0
    have h3 : deriv g (0 : ℝ) = deriv (fun x => tanh (α * x)) 0 - deriv (fun x : ℝ => x) 0 := by
      have hdiff1 := Activation.differentiable_tanh.differentiableAt.comp (0 : ℝ) (h_mul_diff 0)
      have hdiff2 : DifferentiableAt ℝ (fun x : ℝ => x) (0 : ℝ) := differentiableAt_id
      exact deriv_sub hdiff1 hdiff2
    rw [h3, h1, h2]

  -- g is C^1, so deriv g is continuous
  have h_deriv_cont : Continuous (deriv g) := by
    -- deriv g x = deriv (tanh ∘ (α * ·)) x - deriv id x = α * (1 - tanh²(αx)) - 1
    have h_eq : deriv g = fun x => α * (1 - (tanh (α * x))^2) - 1 := by
      ext y
      -- Compute deriv (tanh ∘ (α * ·)) at y using chain rule
      have h_tanh_deriv : HasDerivAt (fun x => tanh (α * x)) (α * (1 - (tanh (α * y))^2)) y := by
        have h1 : HasDerivAt (fun x => α * x) α y := by
          have h1' := (hasDerivAt_id y).const_mul α
          simp only [id, mul_one] at h1'
          exact h1'
        have h2 : HasDerivAt tanh (1 - (tanh (α * y))^2) (α * y) := by
          have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := α * y)
          rw [Activation.deriv_tanh] at hd
          exact hd
        have h3 := h2.comp y h1
        simp only [Function.comp_apply] at h3
        convert h3 using 1
        ring
      have h1 : deriv (fun x => tanh (α * x)) y = α * (1 - (tanh (α * y))^2) := h_tanh_deriv.deriv
      have h2 : deriv (fun x : ℝ => x) y = (1 : ℝ) := by
        have := deriv_id'' (𝕜 := ℝ)
        exact congrFun this y
      have h3 : deriv g y = deriv (fun x => tanh (α * x)) y - deriv (fun x : ℝ => x) y := by
        have hdiff1 := Activation.differentiable_tanh.differentiableAt.comp y (h_mul_diff y)
        have hdiff2 : DifferentiableAt ℝ (fun x : ℝ => x) y := differentiableAt_id
        exact deriv_sub hdiff1 hdiff2
      rw [h3, h1, h2]
    rw [h_eq]
    apply Continuous.sub
    · apply Continuous.mul continuous_const
      apply Continuous.sub continuous_const
      apply Continuous.pow
      exact Activation.differentiable_tanh.continuous.comp (continuous_mul_left α)
    · exact continuous_const

  -- g'(0) = α - 1 > 0. By continuity, there exists δ > 0 with g'(x) > (α-1)/2 for |x| < δ.
  have h_deriv_cont_at_0 : ContinuousAt (deriv g) 0 := h_deriv_cont.continuousAt
  rw [Metric.continuousAt_iff] at h_deriv_cont_at_0
  obtain ⟨δ, hδ_pos, hδ_ball⟩ := h_deriv_cont_at_0 ((α - 1)/2) (by linarith)

  -- Choose x₀ = min(δ/2, 1/2)
  let x₀ := min (δ/2) (1/2)
  have hx₀_pos : 0 < x₀ := lt_min (half_pos hδ_pos) (by norm_num)
  have hx₀_lt_δ : x₀ < δ := calc x₀ ≤ δ/2 := min_le_left _ _
    _ < δ := half_lt_self hδ_pos
  have hx₀_lt_one : x₀ < 1 := calc x₀ ≤ 1/2 := min_le_right _ _
    _ < 1 := by norm_num

  -- For all c ∈ [0, x₀], deriv g c > (α-1)/2
  have h_deriv_bound : ∀ c ∈ Set.Icc 0 x₀, deriv g c > (α - 1)/2 := by
    intro c ⟨hc_ge, hc_le⟩
    have hc_in_ball : dist c 0 < δ := by
      rw [dist_zero_right, Real.norm_eq_abs, abs_of_nonneg hc_ge]
      calc c ≤ x₀ := hc_le
        _ < δ := hx₀_lt_δ
    have h_dist := hδ_ball hc_in_ball
    rw [h_g_deriv_0, dist_eq_norm, Real.norm_eq_abs] at h_dist
    have : |deriv g c - (α - 1)| < (α - 1) / 2 := h_dist
    have h_abs := abs_lt.mp this
    linarith

  -- By MVT, g(x₀) - g(0) = g'(c) * x₀ for some c ∈ (0, x₀)
  have h_mvt_cont : ContinuousOn g (Set.Icc 0 x₀) := h_g_diff.continuous.continuousOn
  have h_mvt_diff : DifferentiableOn ℝ g (Set.Ioo 0 x₀) := h_g_diff.differentiableOn
  obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope g hx₀_pos h_mvt_cont h_mvt_diff

  -- g(0) = tanh(0) - 0 = 0
  have h_g0 : g 0 = 0 := by simp [g, tanh_zero]
  rw [h_g0, sub_zero] at h_mvt

  -- g(x₀) = g'(c) * x₀ where g'(c) > (α-1)/2
  have hc_in_Icc : c ∈ Set.Icc 0 x₀ := ⟨le_of_lt hc_gt, le_of_lt hc_lt⟩
  have h_gc_bound : deriv g c > (α - 1)/2 := h_deriv_bound c hc_in_Icc
  have h_g_x0_pos : g x₀ > 0 := by
    have h_ne : x₀ ≠ 0 := ne_of_gt hx₀_pos
    -- h_mvt : deriv g c = g x₀ / (x₀ - 0)
    have h1 : g x₀ = deriv g c * x₀ := by
      simp only [sub_zero] at h_mvt
      field_simp [h_ne] at h_mvt ⊢
      linarith
    rw [h1]
    exact mul_pos (by linarith) hx₀_pos

  -- Now we have g(x₀) > 0 and g(1) < 0. By IVT, there exists c ∈ (x₀, 1) with g(c) = 0.
  have h_g1_neg : g 1 < 0 := by
    simp only [g]
    -- g 1 = tanh(α * 1) - 1 = tanh(α) - 1 < 0 since |tanh(α)| < 1
    have h_bnd := Activation.tanh_bounded α
    have h_lt := abs_lt.mp h_bnd
    have h_mul_one : α * 1 = α := mul_one α
    rw [h_mul_one]
    linarith

  have h_cont : ContinuousOn g (Set.Icc x₀ 1) := h_g_diff.continuous.continuousOn
  have h_le : x₀ ≤ 1 := le_of_lt hx₀_lt_one

  have h_ivt := intermediate_value_Icc' h_le h_cont
  have h_zero_in_range : (0 : ℝ) ∈ Set.Icc (g 1) (g x₀) := ⟨le_of_lt h_g1_neg, le_of_lt h_g_x0_pos⟩

  obtain ⟨c', ⟨hc'_ge, hc'_le⟩, hc'_eq⟩ := h_ivt h_zero_in_range

  use c'
  constructor
  · -- c' ≠ 0 since c' ≥ x₀ > 0
    linarith
  · -- isFixedPoint α c', i.e., tanh(α·c') = c'
    simp only [isFixedPoint, tanhRecur, g] at hc'_eq ⊢
    linarith

/-- The positive fixed point for α > 1 is unique. -/
theorem positive_fixed_point_unique (α : ℝ) (hα : 1 < α) :
    ∀ S₁ S₂ : ℝ, 0 < S₁ → 0 < S₂ → isFixedPoint α S₁ → isFixedPoint α S₂ → S₁ = S₂ := by
  intro S₁ S₂ h1_pos h2_pos h1_fp h2_fp
  -- Both are roots of f(x) = tanh(αx) - x = 0
  -- Key insight: g(x) = tanh(αx)/x is strictly decreasing for x > 0.
  -- If tanh(αS₁) = S₁ and tanh(αS₂) = S₂, then g(S₁) = 1 = g(S₂).
  -- Since g is strictly decreasing, S₁ = S₂.
  simp only [isFixedPoint, tanhRecur] at h1_fp h2_fp
  -- h1_fp : tanh(α * S₁) = S₁
  -- h2_fp : tanh(α * S₂) = S₂

  -- Proof by contradiction: assume S₁ ≠ S₂, WLOG S₁ < S₂.
  by_contra h_ne
  wlog h_lt : S₁ < S₂ with H
  · push_neg at h_lt
    have h_gt : S₂ < S₁ := lt_of_le_of_ne h_lt (Ne.symm h_ne)
    exact H α hα S₂ S₁ h2_pos h1_pos h2_fp h1_fp (Ne.symm h_ne) h_gt

  -- Now we have 0 < S₁ < S₂.
  -- tanh(αS₁) = S₁ and tanh(αS₂) = S₂
  -- So tanh(αS₁)/S₁ = 1 and tanh(αS₂)/S₂ = 1.

  -- Key: For x > 0, let h(x) = tanh(αx)/x. We show h is strictly decreasing.
  -- h'(x) = (α·sech²(αx)·x - tanh(αx)) / x²
  --       = (α·(1 - tanh²(αx))·x - tanh(αx)) / x²
  -- For h'(x) < 0, need: α·(1 - tanh²(αx))·x < tanh(αx)
  -- i.e., α·x < tanh(αx) / (1 - tanh²(αx)) = tanh(αx) · cosh²(αx)
  --     = sinh(αx) · cosh(αx) = sinh(2αx)/2
  -- So need: 2αx < sinh(2αx), which is true for x > 0 (since sinh y > y for y > 0).

  -- Instead of computing h', use the MVT on tanh:
  -- tanh(αS₂) - tanh(αS₁) = α·(1 - tanh²(αc)) · (S₂ - S₁) for some c ∈ (S₁, S₂)
  -- Since tanh(αS₂) = S₂ and tanh(αS₁) = S₁:
  -- S₂ - S₁ = α·(1 - tanh²(αc)) · (S₂ - S₁)
  -- Since S₂ ≠ S₁, divide by S₂ - S₁:
  -- 1 = α·(1 - tanh²(αc))
  -- But 1 - tanh²(αc) < 1 (since tanh(αc) ≠ 0 for c > 0), so α·(1 - tanh²(αc)) < α.
  -- Since α > 1, we need α·(1 - tanh²(αc)) could equal 1 only if (1 - tanh²(αc)) = 1/α < 1.
  -- Actually this doesn't directly give a contradiction for all α > 1.

  -- Better approach: Use strict monotonicity of tanh and the sandwich.
  -- From S₁ < S₂ and α > 1, we have αS₁ < αS₂.
  -- Since tanh is strictly monotone: tanh(αS₁) < tanh(αS₂), i.e., S₁ < S₂. ✓

  -- But we need to show S₁ = S₂, not S₁ < S₂. The issue is uniqueness.

  -- Use the fact that x ↦ tanh(αx) - x is strictly decreasing for x > x* where
  -- x* is the fixed point. Actually, let's use the MVT more carefully.

  -- MVT: there exists c ∈ (S₁, S₂) with
  -- (tanh(αS₂) - tanh(αS₁)) / (S₂ - S₁) = deriv tanh (αc) · α = α·(1 - tanh²(αc))
  -- Since tanh(αS₁) = S₁ and tanh(αS₂) = S₂:
  -- (S₂ - S₁) / (S₂ - S₁) = 1 = α·(1 - tanh²(αc))
  -- So 1 - tanh²(αc) = 1/α.

  have hα_pos : 0 < α := by linarith
  have h_αS1_pos : 0 < α * S₁ := mul_pos hα_pos h1_pos
  have h_αS2_pos : 0 < α * S₂ := mul_pos hα_pos h2_pos
  have h_αS_lt : α * S₁ < α * S₂ := mul_lt_mul_of_pos_left h_lt hα_pos

  -- Apply MVT to tanh on [αS₁, αS₂]
  have h_cont : ContinuousOn tanh (Set.Icc (α * S₁) (α * S₂)) :=
    Activation.differentiable_tanh.continuous.continuousOn
  have h_diff : DifferentiableOn ℝ tanh (Set.Ioo (α * S₁) (α * S₂)) :=
    Activation.differentiable_tanh.differentiableOn
  obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope tanh h_αS_lt h_cont h_diff

  -- h_mvt : deriv tanh c = (tanh(αS₂) - tanh(αS₁)) / (αS₂ - αS₁)
  --                      = (S₂ - S₁) / (α(S₂ - S₁)) = 1/α
  have h_slope : (tanh (α * S₂) - tanh (α * S₁)) / (α * S₂ - α * S₁) = 1 / α := by
    rw [h1_fp, h2_fp]
    have h_denom : α * S₂ - α * S₁ = α * (S₂ - S₁) := by ring
    rw [h_denom]
    have h_ne : S₂ - S₁ ≠ 0 := by linarith
    have hα_ne : α ≠ 0 := by linarith
    field_simp

  rw [h_slope, Activation.deriv_tanh] at h_mvt
  -- h_mvt : 1 - tanh²(c) = 1/α

  have h_c_pos : 0 < c := by linarith
  have h_tanh_c_pos : 0 < tanh c := Activation.tanh_pos_of_pos h_c_pos
  have h_tanh_c_ne : tanh c ≠ 0 := ne_of_gt h_tanh_c_pos
  have h_tanh_sq_pos : 0 < (tanh c)^2 := sq_pos_of_ne_zero h_tanh_c_ne

  -- From h_mvt: 1 - tanh²(c) = 1/α
  -- So tanh²(c) = 1 - 1/α = (α - 1)/α
  have h_tanh_sq : (tanh c)^2 = (α - 1) / α := by
    have h1 : 1 - (tanh c)^2 = 1 / α := h_mvt
    have hα_ne : α ≠ 0 := by linarith
    field_simp at h1 ⊢
    linarith

  -- Now: (tanh c)² = (α-1)/α.
  -- We have c ∈ (αS₁, αS₂).
  -- We also know: S₁ = tanh(αS₁), so tanh(αS₁) = S₁.
  -- At a fixed point S*, tanh(αS*) = S*, so tanh²(αS*) = S*².

  -- The key contradiction: Let's show that having two distinct positive fixed points
  -- leads to a contradiction with the derivative bound.

  -- Actually, from h_tanh_sq, we know that c is uniquely determined by α (up to sign).
  -- Since c > 0, c = tanh⁻¹(√((α-1)/α)) is unique.
  -- But c can be anywhere in (αS₁, αS₂), which is a range.
  -- This doesn't directly give a contradiction.

  -- Alternative: Use that for the fixed point equation tanh(αx) = x,
  -- at any positive fixed point x*, deriv of g(x) = tanh(αx) - x at x* is
  -- g'(x*) = α·(1 - tanh²(αx*)) - 1 = α·(1 - x*²) - 1 (using tanh(αx*) = x*).
  -- From the uniqueness proof in the nonzero_fixed_point_exists theorem,
  -- the fixed point x* satisfies tanh(αx*) = x*, so tanh²(αx*) = x*².
  -- At the fixed point: g'(x*) = α(1 - x*²) - 1.

  -- For α > 1, at the unique fixed point x*, we have α(1-x*²) = 1 from the MVT.
  -- Wait, that's exactly what we derived: at c, 1 - tanh²(c) = 1/α.

  -- The contradiction comes from the fact that there's only one c satisfying this,
  -- but c must be in (αS₁, αS₂) for any pair of fixed points.

  -- Simpler approach: S₁ < S₂ but both satisfy tanh(αS) = S.
  -- Define f(x) = tanh(αx) - x for x > 0.
  -- f(S₁) = 0 = f(S₂).
  -- f'(x) = α(1 - tanh²(αx)) - 1.
  -- At x = 0: f'(0) = α - 1 > 0 (so f is increasing near 0).
  -- As x → ∞: tanh(αx) → 1, so f(x) → 1 - x → -∞ and f'(x) → -1 < 0.
  -- f is continuous, starts at f(0) = 0 with positive derivative,
  -- and goes to -∞ with negative derivative.
  -- So f has a unique local maximum, crosses 0 exactly once for x > 0.

  -- The formal proof: f is continuous, f(0) = 0, f'(0) > 0, f(x) → -∞.
  -- By continuity, f achieves a maximum at some x_max > 0.
  -- For x > x_max, f is strictly decreasing.
  -- Since f(S₁) = 0 = f(S₂) and S₁ < S₂, both are zeros.
  -- But a function can't have two zeros if it's first increasing then decreasing
  -- (except at the boundaries). Here f(0) = 0 is the starting point.

  -- Let's formalize: f is strictly concave for x > 0 (f'' < 0), so it can cross
  -- y = 0 at most twice total (including x = 0). Since f(0) = 0 and f increases
  -- from 0, there's exactly one other zero.

  -- Actually, we can use Rolle's theorem in reverse:
  -- If f(S₁) = f(S₂) = 0 with S₁ < S₂, then there exists c ∈ (S₁, S₂) with f'(c) = 0.
  -- f'(c) = 0 means α(1 - tanh²(αc)) = 1, i.e., tanh²(αc) = (α-1)/α.
  -- This gives a unique c > 0 (since tanh is strictly monotone).
  -- But S₁ and S₂ are both > 0, and the MVT c is unique.

  -- Wait, from h_mvt we have: 1 - tanh²(c) = 1/α where c ∈ (αS₁, αS₂).
  -- This means the slope of the secant line from (αS₁, S₁) to (αS₂, S₂) on the
  -- tanh curve equals the derivative at c, which is 1/α.

  -- But the secant has slope (S₂ - S₁)/(αS₂ - αS₁) = 1/α.
  -- And points (αS₁, S₁), (αS₂, S₂) are ON the line y = x/α (since S = tanh(αS)).

  -- Actually, the fixed points satisfy tanh(αS) = S, which means on the
  -- curve y = tanh(x), the points (αS₁, S₁) and (αS₂, S₂) lie on y = x/α.

  -- Since tanh is strictly concave for x > 0 (tanh'' < 0), the curve y = tanh(x)
  -- lies below any secant line for x > 0. But if two points on tanh also lie
  -- on y = x/α, the secant between them has slope 1/α...

  -- This is getting complicated. Let me use a direct monotonicity argument.

  -- Key insight: Let h(x) = tanh(αx) - x for x > 0.
  -- h(0) = 0, h'(0) = α - 1 > 0.
  -- h''(x) = -2α²·tanh(αx)·(1 - tanh²(αx)) < 0 for x > 0 (since tanh(αx) > 0).
  -- So h is strictly concave for x > 0.
  -- A strictly concave function with h(0) = 0 and h'(0) > 0 can have at most one
  -- zero for x > 0.

  -- To show: h can't have two zeros S₁ < S₂ with both > 0.
  -- Proof: h(S₁) = h(S₂) = 0. By Rolle's theorem, ∃ c ∈ (S₁, S₂) with h'(c) = 0.
  -- h'(c) = α(1 - tanh²(αc)) - 1 = 0 means 1 - tanh²(αc) = 1/α.
  -- So c satisfies this equation uniquely (tanh is strictly monotone).

  -- But also: h is strictly concave, h(0) = 0, h'(0) > 0.
  -- So h increases on [0, c_max] and decreases on [c_max, ∞) for some c_max > 0.
  -- The unique zero of h' for x > 0 is c_max (where h'(c_max) = 0).

  -- For S₁ > 0 with h(S₁) = 0: either S₁ ≤ c_max or S₁ > c_max.
  -- If S₁ ≤ c_max: h goes from h(0) = 0, increases to h(c_max) > 0, then decreases.
  --   For h(S₁) = 0 with 0 < S₁ ≤ c_max, and h increasing on [0, c_max],
  --   this would mean h(S₁) > h(0) = 0, contradiction.
  -- So S₁ > c_max, meaning S₁ is on the decreasing part.
  -- Similarly S₂ > c_max.
  -- Since h is strictly decreasing on (c_max, ∞) and h(S₁) = h(S₂) = 0,
  -- we must have S₁ = S₂.

  -- Let me find the unique c_max where h'(c_max) = 0.
  -- h'(x) = α(1 - tanh²(αx)) - 1 = 0
  -- 1 - tanh²(αx) = 1/α
  -- tanh²(αx) = (α - 1)/α

  -- Since tanh is strictly increasing and positive for positive args,
  -- tanh(αx) = √((α-1)/α) has a unique solution x = x_max > 0.

  -- Now, the fixed point equation tanh(αS) = S gives S = tanh(αS).
  -- At S_max where h'(S_max) = 0:
  -- tanh²(αS_max) = (α-1)/α
  -- tanh(αS_max) = √((α-1)/α) (positive root since S_max > 0)
  -- If S_max is a fixed point: S_max = tanh(αS_max) = √((α-1)/α)
  -- Then h(S_max) = tanh(αS_max) - S_max = S_max - S_max = 0!
  -- So S_max itself is the fixed point!

  -- This means: the unique fixed point S* > 0 is exactly where h'(S*) = 0.
  -- But then h(S*) = 0 and h'(S*) = 0.
  -- For x < S*: h'(x) > 0 (h is increasing)
  -- For x > S*: h'(x) < 0 (h is decreasing)
  -- And h(0) = 0, h(S*) = 0, with h increasing on (0, S*).
  -- This means h(x) > 0 for x ∈ (0, S*).
  -- For x > S*: h decreases from h(S*) = 0, so h(x) < 0.
  -- Therefore, the only zeros of h for x > 0 are... wait, h(x) > 0 on (0, S*)
  -- and h(x) < 0 on (S*, ∞). So S* is the only zero!

  -- But we assumed h(S₁) = h(S₂) = 0 with 0 < S₁ < S₂. Contradiction!

  -- Let me formalize this by showing h(x) > 0 on (0, S*) using MVT.
  -- Actually, the argument that S* is the critical point simplifies things.

  -- From h(S₁) = 0 and S₁ > 0, and the analysis above, S₁ must equal S*.
  -- Similarly S₂ = S*. Hence S₁ = S₂.

  -- The formal contradiction: We have S₁ < S₂, but by Rolle's theorem on [S₁, S₂],
  -- there exists c ∈ (S₁, S₂) with h'(c) = 0. But we showed the unique critical
  -- point x_max satisfies h(x_max) = 0. So c = x_max = S* is the unique fixed point.
  -- But c ∈ (S₁, S₂) and S₁, S₂ are both fixed points (h(S₁) = h(S₂) = 0).
  -- So S₁ < c < S₂ and c is also a zero of h. But S₁, c, S₂ are three distinct
  -- zeros of h for x > 0... no wait, we have S₁ < S₂ but c ∈ (S₁, S₂), and we're
  -- saying h(S₁) = h(c) = h(S₂) = 0? That's three zeros!

  -- Actually, by Rolle's theorem applied to h on [S₁, S₂]:
  -- h(S₁) = h(S₂) = 0 implies ∃ c ∈ (S₁, S₂) with h'(c) = 0.
  -- h'(c) = 0 means c is a critical point of h.
  -- But h is strictly concave (h'' < 0), so h has at most one critical point.
  -- And we showed that at the unique critical point c*, h(c*) = 0.

  -- So c = c* (unique critical point). And h(c*) = 0.
  -- But c ∈ (S₁, S₂), so S₁ < c* < S₂.
  -- And h(S₁) = h(c*) = h(S₂) = 0.

  -- Now: h is strictly increasing on (0, c*) (since h' > 0 there).
  -- h(0) = 0 and h is strictly increasing, so h(x) > 0 for x ∈ (0, c*).
  -- But S₁ ∈ (0, c*) (since 0 < S₁ < c*), so h(S₁) > 0. Contradiction!

  -- So the assumption S₁ < S₂ leads to h(S₁) > 0, contradicting h(S₁) = 0.

  -- Formally:
  -- By Rolle on [S₁, S₂]: ∃ c ∈ (S₁, S₂) with h'(c) = 0.
  -- Let h(x) = tanh(αx) - x for the following.

  -- h'(x) = α·(1 - tanh²(αx)) - 1
  -- h'(c) = 0 means α·(1 - tanh²(αc)) = 1, i.e., 1 - tanh²(αc) = 1/α.

  -- h is increasing on (0, c) (where h' > 0) and h(0) = 0.
  -- So for 0 < S₁ < c: h(S₁) > h(0) = 0. But h(S₁) = 0 is given. Contradiction!

  -- Let's implement this.

  -- Define h(x) = tanh(αx) - x
  let h := fun x => tanh (α * x) - x

  have h_S1 : h S₁ = 0 := by simp only [h]; linarith [h1_fp]
  have h_S2 : h S₂ = 0 := by simp only [h]; linarith [h2_fp]

  -- By Rolle's theorem, ∃ c ∈ (S₁, S₂) with h'(c) = 0.
  have h_cont_h : ContinuousOn h (Set.Icc S₁ S₂) := by
    apply ContinuousOn.sub
    · exact (Activation.differentiable_tanh.continuous.comp (continuous_mul_left α)).continuousOn
    · exact continuous_id.continuousOn
  have h_diff_h : DifferentiableOn ℝ h (Set.Ioo S₁ S₂) := by
    apply DifferentiableOn.sub
    · exact (Activation.differentiable_tanh.comp
        (Differentiable.const_mul differentiable_id α)).differentiableOn
    · exact differentiable_id.differentiableOn

  have h_S1_le_S2 : S₁ ≤ S₂ := le_of_lt h_lt
  have h_eq_ends : h S₁ = h S₂ := by rw [h_S1, h_S2]

  obtain ⟨c_rolle, ⟨hc_gt, hc_lt⟩, h_rolle⟩ :=
    exists_deriv_eq_slope h h_lt h_cont_h h_diff_h
  -- h_rolle : deriv h c_rolle = (h S₂ - h S₁) / (S₂ - S₁)

  have h_deriv_h : ∀ x, deriv h x = α * (1 - (tanh (α * x))^2) - 1 := by
    intro x
    have hd : HasDerivAt h (α * (1 - (tanh (α * x))^2) - 1) x := by
      have h1 : HasDerivAt (fun y => α * y) α x := by
        have h1' := (hasDerivAt_id x).const_mul α
        simp only [id, mul_one] at h1'
        exact h1'
      have h2 : HasDerivAt tanh (1 - (tanh (α * x))^2) (α * x) := by
        have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := α * x)
        rw [Activation.deriv_tanh] at hd
        exact hd
      have h3 := h2.comp x h1
      simp only [Function.comp_apply] at h3
      -- h3 gives derivative = (1 - tanh²(αx)) * α
      have h5 : HasDerivAt (fun y => tanh (α * y)) (α * (1 - (tanh (α * x))^2)) x := by
        convert h3 using 1
        ring
      have h6 : HasDerivAt (fun y : ℝ => y) 1 x := hasDerivAt_id x
      have h7 := h5.sub h6
      simp only [h] at h7 ⊢
      convert h7 using 1
    exact hd.deriv

  have h_deriv_at_c : deriv h c_rolle = 0 := by
    rw [h_rolle, h_S1, h_S2, sub_self, zero_div]

  rw [h_deriv_h] at h_deriv_at_c
  -- h_deriv_at_c : α * (1 - tanh²(αc_rolle)) - 1 = 0
  -- So: 1 - tanh²(αc_rolle) = 1/α

  have h_c_pos : 0 < c_rolle := by linarith
  have h_αc_pos : 0 < α * c_rolle := mul_pos hα_pos h_c_pos

  -- Now: h is strictly increasing on (0, c_rolle).
  -- Proof: For x ∈ (0, c_rolle), h'(x) > 0.
  -- h'(x) = α(1 - tanh²(αx)) - 1
  -- We need α(1 - tanh²(αx)) > 1, i.e., 1 - tanh²(αx) > 1/α.

  -- At c_rolle: 1 - tanh²(αc_rolle) = 1/α.
  -- For x < c_rolle: αx < αc_rolle, so tanh(αx) < tanh(αc_rolle) (tanh increasing).
  -- So tanh²(αx) < tanh²(αc_rolle), hence 1 - tanh²(αx) > 1 - tanh²(αc_rolle) = 1/α.
  -- Therefore h'(x) = α(1 - tanh²(αx)) - 1 > α · (1/α) - 1 = 0.

  have h_deriv_pos_on_0_c : ∀ x, 0 < x → x < c_rolle → deriv h x > 0 := by
    intro x hx_pos hx_lt
    rw [h_deriv_h]
    have h_αx_pos : 0 < α * x := mul_pos hα_pos hx_pos
    have h_αx_lt : α * x < α * c_rolle := mul_lt_mul_of_pos_left hx_lt hα_pos
    have h_tanh_mono : tanh (α * x) < tanh (α * c_rolle) :=
      Activation.tanh_strictMono h_αx_lt
    have h_tanh_x_pos : 0 < tanh (α * x) := Activation.tanh_pos_of_pos h_αx_pos
    have h_tanh_c_pos : 0 < tanh (α * c_rolle) := Activation.tanh_pos_of_pos h_αc_pos
    have h_sq_mono : (tanh (α * x))^2 < (tanh (α * c_rolle))^2 :=
      sq_lt_sq' (by linarith) h_tanh_mono
    -- 1 - tanh²(αx) > 1 - tanh²(αc_rolle) = 1/α
    have h_one_minus : 1 - (tanh (α * x))^2 > 1 - (tanh (α * c_rolle))^2 := by linarith
    have h_eq_inv : 1 - (tanh (α * c_rolle))^2 = 1 / α := by
      have : α * (1 - (tanh (α * c_rolle))^2) - 1 = 0 := h_deriv_at_c
      have hα_ne : α ≠ 0 := by linarith
      field_simp at this ⊢
      linarith
    rw [h_eq_inv] at h_one_minus
    -- h'(x) = α(1 - tanh²(αx)) - 1 > α · (1/α) - 1 = 0
    have hα_ne : α ≠ 0 := by linarith
    have h_one_over : α * (1 / α) = 1 := by field_simp
    calc α * (1 - (tanh (α * x))^2) - 1 > α * (1 / α) - 1 := by nlinarith [hα_pos]
      _ = 1 - 1 := by rw [h_one_over]
      _ = 0 := by ring

  -- h(0) = tanh(0) - 0 = 0
  have h_at_0 : h 0 = 0 := by simp [h, tanh_zero]

  -- For S₁ ∈ (0, c_rolle): h(S₁) > h(0) = 0 by strict monotonicity.
  have h_S1_lt_c : S₁ < c_rolle := hc_gt

  -- Use MVT on [0, S₁]: h(S₁) - h(0) = h'(ξ) · S₁ for some ξ ∈ (0, S₁)
  have h_S1_ge_0 : 0 ≤ S₁ := le_of_lt h1_pos
  -- Actually we need [0, S₁], not a subset of [S₁, S₂].
  have h_cont_0S1 : ContinuousOn h (Set.Icc 0 S₁) := by
    apply ContinuousOn.sub
    · exact (Activation.differentiable_tanh.continuous.comp (continuous_mul_left α)).continuousOn
    · exact continuous_id.continuousOn
  have h_diff_0S1 : DifferentiableOn ℝ h (Set.Ioo 0 S₁) := by
    apply DifferentiableOn.sub
    · exact (Activation.differentiable_tanh.comp
        (Differentiable.const_mul differentiable_id α)).differentiableOn
    · exact differentiable_id.differentiableOn

  obtain ⟨ξ, ⟨hξ_gt, hξ_lt⟩, h_mvt_xi⟩ :=
    exists_deriv_eq_slope h h1_pos h_cont_0S1 h_diff_0S1

  -- h_mvt_xi : deriv h ξ = (h S₁ - h 0) / (S₁ - 0) = h(S₁) / S₁
  have h_deriv_xi_eq : deriv h ξ = h S₁ / S₁ := by
    rw [h_at_0, sub_zero, sub_zero] at h_mvt_xi
    exact h_mvt_xi

  -- ξ ∈ (0, S₁) ⊂ (0, c_rolle), so h'(ξ) > 0.
  have hξ_lt_c : ξ < c_rolle := by linarith

  have h_deriv_xi_pos : deriv h ξ > 0 := h_deriv_pos_on_0_c ξ hξ_gt hξ_lt_c

  -- h(S₁) / S₁ = h'(ξ) > 0, and S₁ > 0, so h(S₁) > 0.
  have h_S1_pos : h S₁ > 0 := by
    rw [h_deriv_xi_eq] at h_deriv_xi_pos
    have h_S1_pos' : 0 < S₁ := h1_pos
    exact (div_pos_iff_of_pos_right h_S1_pos').mp h_deriv_xi_pos

  -- But h(S₁) = 0 by h_S1. Contradiction!
  linarith

/-! ## Part 3: Stability of Fixed Points -/

/-- The derivative of tanhRecur: d/dS[tanh(αS)] = α·(1 - tanh²(αS)). -/
theorem tanhRecur_deriv (α S : ℝ) :
    deriv (tanhRecur α) S = α * (1 - (tanh (α * S))^2) := by
  -- d/dS[tanh(αS)] = tanh'(αS) · α = (1 - tanh²(αS)) · α
  -- By chain rule: deriv (f ∘ g) = (deriv f ∘ g) * deriv g
  unfold tanhRecur
  have h : HasDerivAt (fun x => tanh (α * x)) (α * (1 - (tanh (α * S))^2)) S := by
    have h1 : HasDerivAt (fun x => α * x) α S := by
      have h1' := (hasDerivAt_id S).const_mul α
      simp only [id, mul_one] at h1'
      exact h1'
    have h2 : HasDerivAt tanh (1 - (tanh (α * S))^2) (α * S) := by
      have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := α * S)
      rw [Activation.deriv_tanh] at hd
      exact hd
    have h3 := h2.comp S h1
    simp only [Function.comp_apply] at h3
    convert h3 using 1
    ring
  exact h.deriv

/-- At a fixed point S* with |S*| close to 1, the derivative is small.
    This makes the fixed point stable (an attractor). -/
theorem fixed_point_stability (α : ℝ) (hα : 0 < α) (hα_le : α ≤ 2) (S : ℝ)
    (hfp : isFixedPoint α S) (hS : |S| > 0.9) :
    |deriv (tanhRecur α) S| < 1 := by
  rw [tanhRecur_deriv]
  simp only [isFixedPoint, tanhRecur] at hfp
  rw [hfp]  -- Replace tanh(αS) with S
  -- |α · (1 - S²)| < 1
  -- Since |S| > 0.9, S² > 0.81, so 1 - S² < 0.19
  -- α · (1 - S²) < 2 · 0.19 = 0.38 < 1
  have h_S_sq : (0.9 : ℝ)^2 < S^2 := by
    have h1 : (0.9 : ℝ) < |S| := hS
    have h2 : (0.9 : ℝ)^2 < |S|^2 := sq_lt_sq' (by linarith) h1
    rwa [sq_abs] at h2
  have h_one_minus_sq : 1 - S^2 < 1 - 0.81 := by
    have : (0.9 : ℝ)^2 = 0.81 := by norm_num
    linarith
  have h_bound : 1 - S^2 < 0.19 := by linarith
  have h_nonneg : 0 ≤ 1 - S^2 := by
    have h_bnd := Activation.tanh_bounded (α * S)
    rw [hfp] at h_bnd
    rw [abs_lt] at h_bnd
    have h_sq : S^2 < 1 := by nlinarith
    linarith
  rw [abs_mul, abs_of_pos hα, abs_of_nonneg h_nonneg]
  calc α * (1 - S^2) < α * 0.19 := mul_lt_mul_of_pos_left h_bound hα
    _ ≤ 2 * 0.19 := mul_le_mul_of_nonneg_right hα_le (by linarith)
    _ = 0.38 := by norm_num
    _ < 1 := by norm_num

/-! ## Part 4: Alert State Definition and Properties -/

/-- tanh(1) > 0.76. This is used for proving alert state persistence. -/
theorem tanh_one_gt_076 : tanh 1 > 0.76 := by
  -- tanh(1) = (e - e⁻¹)/(e + e⁻¹) ≈ 0.7616
  -- We prove by showing 19(e + e⁻¹) < 25(e - e⁻¹), i.e., 44/e < 6e, i.e., e² > 22/3
  -- Since e > 2.718, e² > 7.38 > 7.33. ✓
  have he_pos : exp 1 > 0 := exp_pos 1
  have he_ge : exp 1 > 2.718 := by
    exact lt_trans (by norm_num : (2.718 : ℝ) < 2.7182818283) Real.exp_one_gt_d9
  have he_sq : exp 1 * exp 1 > 7.38 := by nlinarith
  have hei_eq : exp (-1) = (exp 1)⁻¹ := Real.exp_neg 1
  have hei_pos : 0 < (exp 1)⁻¹ := inv_pos.mpr he_pos
  have hne : exp 1 ≠ 0 := he_pos.ne'
  -- Transform tanh(1) to (e - e⁻¹)/(e + e⁻¹)
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq, hei_eq]
  -- Simplify (e - e⁻¹)/2 / ((e + e⁻¹)/2) to (e - e⁻¹)/(e + e⁻¹)
  have hsum_pos : 0 < exp 1 + (exp 1)⁻¹ := by linarith
  have h_eq : (exp 1 - (exp 1)⁻¹) / 2 / ((exp 1 + (exp 1)⁻¹) / 2) =
              (exp 1 - (exp 1)⁻¹) / (exp 1 + (exp 1)⁻¹) := by field_simp
  rw [h_eq]
  -- Need (e - e⁻¹)/(e + e⁻¹) > 19/25
  rw [gt_iff_lt, lt_div_iff₀ hsum_pos]
  -- Need 0.76 * (e + e⁻¹) < e - e⁻¹
  -- i.e., 19/25 * (e + e⁻¹) < e - e⁻¹
  -- i.e., 19(e + e⁻¹) < 25(e - e⁻¹)
  -- i.e., 19e + 19/e < 25e - 25/e
  -- i.e., 44/e < 6e
  -- i.e., 44 < 6e²
  have h1 : (0.76 : ℝ) * (exp 1 + (exp 1)⁻¹) < exp 1 - (exp 1)⁻¹ := by
    have h2 : (exp 1)⁻¹ < 0.37 := by
      rw [inv_lt_comm₀ he_pos (by norm_num : (0 : ℝ) < 0.37)]
      calc (0.37 : ℝ)⁻¹ < 2.71 := by norm_num
        _ < exp 1 := lt_trans (by norm_num) he_ge
    nlinarith
  exact h1

/-- For x ≥ 1, tanh(x) > 0.76. -/
theorem tanh_gt_076_of_ge_one (x : ℝ) (hx : 1 ≤ x) : tanh x > 0.76 := by
  calc tanh x ≥ tanh 1 := Activation.tanh_strictMono.monotone hx
    _ > 0.76 := tanh_one_gt_076

/-- exp(2.2) > 9. This is used for proving tanh(1.1) > 0.8.
    Numerical verification: exp(2.2) ≈ 9.025 > 9. -/
theorem exp_2_2_gt_9 : exp 2.2 > 9 := by
  -- e^2.2 = e^2 * e^0.2 ≈ 7.389 * 1.221 ≈ 9.025 > 9
  -- Using quadratic_le_exp_of_nonneg: 1 + x + x²/2 ≤ exp x for x ≥ 0
  have h_exp22 : exp 2.2 = exp 2 * exp 0.2 := by rw [← Real.exp_add]; norm_num
  -- Tight bound on exp 2 using Mathlib's exp_one_gt_d9: exp 1 > 2.7182818283
  have h_exp2 : exp 2 > 7.389 := by
    have h2a : exp 2 = exp 1 * exp 1 := by rw [← Real.exp_add]; norm_num
    calc exp 2 = exp 1 * exp 1 := h2a
      _ > 2.7182818283 * 2.7182818283 := by nlinarith [Real.exp_one_gt_d9, exp_pos 1]
      _ > 7.389 := by norm_num
  -- For exp 0.2, use quadratic Taylor bound: exp x ≥ 1 + x + x²/2
  have h_exp02 : exp 0.2 ≥ 1.22 := by
    have h := quadratic_le_exp_of_nonneg (by norm_num : (0 : ℝ) ≤ 0.2)
    calc exp 0.2 ≥ 1 + 0.2 + (0.2 : ℝ)^2 / 2 := h
      _ = 1.22 := by norm_num
  -- 7.389 * 1.22 = 9.01458 > 9
  calc exp 2.2 = exp 2 * exp 0.2 := h_exp22
    _ > 7.389 * 1.22 := by nlinarith [exp_pos 2, exp_pos 0.2, h_exp2, h_exp02]
    _ > 9 := by norm_num

/-- tanh(1.1) > 0.8. This is the key numerical bound for alert state persistence.
    Numerical verification: tanh(1.1) ≈ 0.8005 > 0.8. -/
theorem tanh_11_gt_08 : tanh 1.1 > 0.8 := by
  -- tanh(x) = (e^2x - 1)/(e^2x + 1)
  -- For tanh(x) > c, we need e^2x > (1+c)/(1-c)
  -- For tanh(1.1) > 0.8, we need e^2.2 > 1.8/0.2 = 9
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h_exp22 : exp 2.2 > 9 := exp_2_2_gt_9
  have h_exp11_pos : exp 1.1 > 0 := exp_pos 1.1
  have h_exp_neg11 : exp (-1.1) = (exp 1.1)⁻¹ := Real.exp_neg 1.1
  have h_inv_pos : 0 < (exp 1.1)⁻¹ := inv_pos.mpr h_exp11_pos
  -- sinh(1.1)/cosh(1.1) = (e^1.1 - e^-1.1)/(e^1.1 + e^-1.1) = (e^2.2 - 1)/(e^2.2 + 1)
  have h_simp : (exp 1.1 - (exp 1.1)⁻¹) / 2 / ((exp 1.1 + (exp 1.1)⁻¹) / 2) =
                (exp 2.2 - 1) / (exp 2.2 + 1) := by
    have h_sq : exp 1.1 * exp 1.1 = exp 2.2 := by rw [← Real.exp_add]; norm_num
    have hne : exp 1.1 ≠ 0 := h_exp11_pos.ne'
    field_simp
    rw [← h_sq]
    ring
  rw [h_exp_neg11, h_simp]
  -- (exp 2.2 - 1)/(exp 2.2 + 1) > 0.8 iff exp 2.2 > (1+0.8)/(1-0.8) = 9
  have h_denom_pos : exp 2.2 + 1 > 0 := by linarith [exp_pos 2.2]
  -- Need (exp 2.2 - 1)/(exp 2.2 + 1) > 0.8 = 4/5
  -- Rearranging: 5(exp 2.2 - 1) > 4(exp 2.2 + 1)
  -- 5 exp 2.2 - 5 > 4 exp 2.2 + 4
  -- exp 2.2 > 9
  rw [gt_iff_lt, lt_div_iff₀ h_denom_pos]
  -- Goal: 0.8 * (exp 2.2 + 1) < exp 2.2 - 1
  linarith

/-- For x ≥ 1.1, tanh(x) > 0.8. -/
theorem tanh_gt_08_of_ge_11 (x : ℝ) (hx : 1.1 ≤ x) : tanh x > 0.8 := by
  calc tanh x ≥ tanh 1.1 := Activation.tanh_strictMono.monotone hx
    _ > 0.8 := tanh_11_gt_08

/-- An "alert" state is one where |S| exceeds a threshold θ.
    This represents a head that has detected a pattern and "latched" onto it. -/
def isAlert (S θ : ℝ) : Prop := θ < |S|

/-- The alert basin: states that remain alert under iteration.
    A state S is in the alert basin if tanhRecurIter α n S is alert for all n. -/
def alertBasin (α θ : ℝ) : Set ℝ :=
  {S | ∀ n : ℕ, isAlert (tanhRecurIter α n S) θ}

/-- For appropriate θ, the alert basin is non-empty.
    Key: For α > 1, the positive fixed point S* has |S*| close to 1.
    Any state |S| ≥ |S*| will converge to S* and stay in the alert region.

    CRITICAL CONSTRAINT: We require θ < tanh(α * θ), which ensures θ is below the
    fixed point S*(α). Without this, the theorem is false for α close to 1 and θ close to 1.
    E.g., for α = 1.01 and θ = 0.99, the fixed point S* ≈ 0.1 < θ. -/
theorem alert_basin_nonempty (α θ : ℝ) (hα : 1 < α) (hθ_pos : 0 < θ) (hθ_lt : θ < 1)
    (hθ_below_fp : θ < tanh (α * θ)) :
    ∃ S : ℝ, S ∈ alertBasin α θ := by
  -- With θ < tanh(α * θ), we know θ is below the fixed point S*.
  -- Starting at S₀ = 1, all iterations stay above θ.
  use 1
  intro n
  simp only [isAlert]
  have h_α_pos : 0 < α := lt_trans zero_lt_one hα
  -- All iterates starting from 1 are positive
  have h_iter_pos : ∀ m, 0 < tanhRecurIter α m 1 := by
    intro m
    induction m with
    | zero => simp only [tanhRecurIter]; norm_num
    | succ k ih =>
      simp only [tanhRecurIter, tanhRecur]
      exact Activation.tanh_pos_of_pos (mul_pos h_α_pos ih)
  -- Key lemma: For x > θ, we have tanh(α*x) > tanh(α*θ) > θ.
  -- So the property iter > θ is preserved by the iteration.
  -- First prove the inequality without abs, then wrap in abs
  have h_gt : tanhRecurIter α n 1 > θ := by
    induction n with
    | zero =>
      simp only [tanhRecurIter]
      exact hθ_lt
    | succ m ih =>
      simp only [tanhRecurIter, tanhRecur]
      have h_prev_pos := h_iter_pos m
      -- Since iter_m > θ > 0 and α > 0, we have α * iter_m > α * θ
      have h_arg_gt : α * tanhRecurIter α m 1 > α * θ :=
        mul_lt_mul_of_pos_left ih h_α_pos
      -- By strict monotonicity of tanh:
      have h_result : tanh (α * tanhRecurIter α m 1) > tanh (α * θ) :=
        Activation.tanh_strictMono h_arg_gt
      -- And by hypothesis, tanh(α * θ) > θ
      linarith
  rw [abs_of_pos (h_iter_pos n)]
  exact h_gt

/-- Forward invariance: if |S| > θ and θ is chosen appropriately,
    then |tanhRecur α S| > θ. This means alert states stay alert.

    CRITICAL CONSTRAINT: We require θ < tanh(α * θ), which ensures θ is below the
    fixed point. This is the key condition for alert states to persist. -/
theorem alert_forward_invariant (α θ : ℝ) (hα : 1 < α) (hθ_pos : 0 < θ) (hθ_lt : θ < 0.8)
    (hθ_below_fp : θ < tanh (α * θ))
    (S : ℝ) (hS : isAlert S θ) :
    isAlert (tanhRecur α S) θ := by
  simp only [isAlert] at hS ⊢
  simp only [tanhRecur]
  -- Need: θ < |tanh(αS)| given θ < |S|
  -- Key insight: |tanh(αS)| = tanh(|αS|) = tanh(α|S|) by oddness of tanh
  -- We have α|S| > αθ > θ (since α > 1)
  -- Strategy: show tanh(α|S|) > θ using monotonicity and the bound θ < 0.8
  have hα_pos : 0 < α := by linarith
  have h_abs_S_pos : 0 < |S| := lt_of_lt_of_le hθ_pos (le_of_lt hS)
  have h_S_ne_zero : S ≠ 0 := fun h => by simp [h] at hS; linarith
  have h_αS_bound : α * θ < α * |S| := mul_lt_mul_of_pos_left hS hα_pos
  have h_αθ_gt_θ : θ < α * θ := by
    have : 1 < α := hα
    have : 1 * θ < α * θ := mul_lt_mul_of_pos_right this hθ_pos
    linarith
  have h_α_abs_S_gt_θ : θ < α * |S| := lt_trans h_αθ_gt_θ h_αS_bound
  -- |tanh(αS)| = tanh(α|S|) by tanh being odd
  have h_tanh_abs : |tanh (α * S)| = tanh (α * |S|) := by
    by_cases hS_pos : 0 < S
    · rw [abs_of_pos hS_pos]
      have h_αS_pos : 0 < α * S := mul_pos hα_pos hS_pos
      rw [abs_of_pos (Activation.tanh_pos_of_pos h_αS_pos)]
    · push_neg at hS_pos
      have hS_neg : S < 0 := lt_of_le_of_ne hS_pos h_S_ne_zero
      rw [abs_of_neg hS_neg]
      have h_αS_neg : α * S < 0 := mul_neg_of_pos_of_neg hα_pos hS_neg
      rw [abs_of_neg (Activation.tanh_neg_of_neg h_αS_neg)]
      rw [← tanh_neg]
      congr 1
      ring
  rw [h_tanh_abs]
  -- Now show tanh(α|S|) > θ
  -- Key: We have the hypothesis hθ_below_fp : θ < tanh (α * θ)
  -- Since |S| > θ (from hS), we have α|S| > αθ (by h_αS_bound)
  -- By strict monotonicity of tanh: tanh(α|S|) > tanh(αθ) > θ (by hθ_below_fp)
  have h_tanh_mono := Activation.tanh_strictMono
  -- tanh(α|S|) > tanh(αθ) since α|S| > αθ
  have h_tanh_strict : tanh (α * θ) < tanh (α * |S|) := h_tanh_mono h_αS_bound
  -- Combine: θ < tanh(αθ) < tanh(α|S|)
  calc θ < tanh (α * θ) := hθ_below_fp
    _ < tanh (α * |S|) := h_tanh_strict

/-! ## Part 5: Perturbation Robustness -/

/-- A latched state persists even under small perturbations.
    If |S| is large enough (|S| > 1.6) and we apply a small input perturbation δ,
    the new state tanh(αS + δ) is still close to 1.

    Key insight: artanh(0.9) ≈ 1.47, so we need |αS + δ| > 1.47 to get |tanh(αS+δ)| > 0.9.
    With |S| > 1.6 and α ≥ 1, we have |αS| > 1.6.
    With |δ| < 0.1, we get |αS + δ| ≥ |αS| - |δ| > 1.6 - 0.1 = 1.5 > 1.47. -/
theorem latched_state_robust (α S : ℝ) (δ_max : ℝ)
    (hα : 1 ≤ α) (_hα_lt : α < 2)
    (hS : |S| > 1.6) (hδ : δ_max < 0.1)
    (δ : ℝ) (hδ_bound : |δ| ≤ δ_max) :
    |tanh (α * S + δ)| > 0.9 := by
  -- The key numerical fact is that tanh is strictly increasing and tanh(1.5) > 0.9.
  -- With |αS + δ| > 1.5, we get |tanh(αS + δ)| > 0.9 by monotonicity of |tanh|.
  -- First establish |αS + δ| > 1.5
  have h_αS_abs : |α * S| ≥ |S| := by
    rw [abs_mul]
    have hα_pos : 0 < α := by linarith
    rw [abs_of_pos hα_pos]
    calc α * |S| ≥ 1 * |S| := by nlinarith
      _ = |S| := one_mul _
  have h_δ_abs : |δ| < 0.1 := by linarith [hδ_bound]
  have h_arg_lower : |α * S + δ| > 1.5 := by
    -- Use reverse triangle inequality: ||a| - |b|| ≤ |a + b|
    have h_triangle : |α * S| - |δ| ≤ |α * S + δ| := by
      have := abs_abs_sub_abs_le_abs_sub (α * S) (-δ)
      simp only [abs_neg, sub_neg_eq_add] at this
      have h2 : |α * S| - |δ| ≤ abs (|α * S| - |δ|) := le_abs_self _
      linarith
    have h2 : |α * S| - |δ| > 1.6 - 0.1 := by
      have : |α * S| > 1.6 := lt_of_lt_of_le hS h_αS_abs
      linarith
    linarith
  -- Now use NumericalBounds.tanh_ge_15_gt_090 with monotonicity
  rcases le_or_gt (α * S + δ) 0 with hneg | hpos
  · -- Case: α * S + δ ≤ 0, so -(α * S + δ) ≥ 1.5
    have h_neg_ge : -(α * S + δ) ≥ 1.5 := by
      have : |α * S + δ| = -(α * S + δ) := abs_of_nonpos hneg
      linarith
    have h_tanh_neg : tanh (-(α * S + δ)) > 0.90 := NumericalBounds.tanh_ge_15_gt_090 _ h_neg_ge
    rw [Real.tanh_neg] at h_tanh_neg
    have h_abs : |tanh (α * S + δ)| = -tanh (α * S + δ) := by
      have htanh_neg : tanh (α * S + δ) ≤ 0 := by
        rcases eq_or_lt_of_le hneg with heq | hlt
        · rw [heq]; exact le_of_eq tanh_zero
        · exact le_of_lt (Activation.tanh_neg_of_neg hlt)
      exact abs_of_nonpos htanh_neg
    rw [h_abs]
    linarith
  · -- Case: α * S + δ > 0, so α * S + δ > 1.5
    have h_pos_ge : α * S + δ ≥ 1.5 := by
      have : |α * S + δ| = α * S + δ := abs_of_pos hpos
      linarith
    have h_tanh_pos : tanh (α * S + δ) > 0.90 := NumericalBounds.tanh_ge_15_gt_090 _ h_pos_ge
    have h_abs : |tanh (α * S + δ)| = tanh (α * S + δ) := by
      apply abs_of_pos
      exact Activation.tanh_pos_of_pos hpos
    rw [h_abs]
    linarith

/-- Main perturbation theorem: starting from an alert state, small inputs
    cannot knock the state out of the alert region. -/
theorem alert_persists_under_perturbation (α θ : ℝ) (δ_max : ℝ)
    (hα : 1 < α) (hα_lt : α < 2)
    (hθ : 0 < θ) (hθ_lt : θ < 0.8)
    (hθ_below_fp : θ < tanh (α * θ))
    (hδ : δ_max < (1 - θ) / 2)
    (S : ℝ) (hS_alert : isAlert S θ) (δ : ℝ) (hδ_bound : |δ| ≤ δ_max) :
    isAlert (tanh (α * S + δ)) (θ - δ_max) := by
  simp only [isAlert] at hS_alert ⊢
  -- We have |S| > θ
  -- Need to show θ - δ_max < |tanh(αS + δ)|
  -- Key: tanh is 1-Lipschitz, so |tanh(αS+δ) - tanh(αS)| ≤ |δ| ≤ δ_max
  -- From forward invariance (informal): |tanh(αS)| > θ
  -- So |tanh(αS + δ)| > |tanh(αS)| - δ_max > θ - δ_max
  have h_lip := Activation.tanh_lipschitz
  have h_diff : |tanh (α * S + δ) - tanh (α * S)| ≤ 1 * |α * S + δ - α * S| := by
    have := LipschitzWith.dist_le_mul h_lip (α * S + δ) (α * S)
    rwa [NNReal.coe_one, dist_eq_norm, dist_eq_norm] at this
  simp only [add_sub_cancel_left, one_mul] at h_diff
  -- |tanh(αS + δ) - tanh(αS)| ≤ |δ| ≤ δ_max
  have h_pert : |tanh (α * S + δ) - tanh (α * S)| ≤ δ_max :=
    le_trans h_diff hδ_bound
  -- Need: |tanh(αS)| > θ (forward invariance)
  have h_forward : θ < |tanh (α * S)| := by
    have := alert_forward_invariant α θ hα hθ hθ_lt hθ_below_fp S hS_alert
    simp only [isAlert, tanhRecur] at this
    exact this
  -- Triangle inequality: |tanh(αS+δ)| ≥ |tanh(αS)| - |tanh(αS+δ) - tanh(αS)|
  have h_triangle : |tanh (α * S)| - |tanh (α * S + δ) - tanh (α * S)| ≤ |tanh (α * S + δ)| := by
    have := abs_sub_abs_le_abs_sub (tanh (α * S)) (tanh (α * S + δ))
    linarith [abs_sub_comm (tanh (α * S + δ)) (tanh (α * S))]
  calc θ - δ_max < |tanh (α * S)| - δ_max := by linarith
    _ ≤ |tanh (α * S)| - |tanh (α * S + δ) - tanh (α * S)| := by linarith [h_pert]
    _ ≤ |tanh (α * S + δ)| := h_triangle

/-! ## Part 6: Linear Systems Cannot Latch -/

/-- Linear recurrence: S_{t+1} = α·S_t (no input). -/
def linearRecur (α : ℝ) (S : ℝ) : ℝ := α * S

/-- Iterated linear recurrence. -/
def linearRecurIter (α : ℝ) : ℕ → ℝ → ℝ
  | 0, S => S
  | n + 1, S => linearRecur α (linearRecurIter α n S)

/-- Linear iteration is just α^n · S. -/
theorem linearRecurIter_eq_pow (α : ℝ) (n : ℕ) (S : ℝ) :
    linearRecurIter α n S = α ^ n * S := by
  induction n with
  | zero => simp [linearRecurIter]
  | succ n ih =>
    simp only [linearRecurIter, linearRecur, ih]
    ring

/-- Linear systems have only one fixed point: 0 (for α ≠ 1).
    For α = 1, all points are fixed (trivial case). -/
theorem linear_fixed_point_is_zero (α : ℝ) (hα : α ≠ 1) (S : ℝ)
    (hfp : linearRecur α S = S) : S = 0 := by
  simp only [linearRecur] at hfp
  -- αS = S → (α - 1)S = 0
  have h : (α - 1) * S = 0 := by linarith
  rcases mul_eq_zero.mp h with h_coef | h_S
  · -- h_coef : α - 1 = 0, i.e., α = 1, contradicts hα
    have : α = 1 := by linarith
    exact absurd this hα
  · exact h_S

/-- For |α| < 1, linear states decay to 0.
    This is the fundamental contrast with tanh latching. -/
theorem linear_state_decays (α : ℝ) (hα : |α| < 1) (S : ℝ) :
    Tendsto (fun n => linearRecurIter α n S) atTop (nhds 0) := by
  simp only [linearRecurIter_eq_pow]
  have h_pow : Tendsto (fun n => α ^ n) atTop (nhds 0) :=
    tendsto_pow_atTop_nhds_zero_of_abs_lt_one hα
  have h_mul : Tendsto (fun n => α ^ n * S) atTop (nhds (0 * S)) :=
    h_pow.mul_const S
  simp only [zero_mul] at h_mul
  exact h_mul

/-- For |α| > 1, linear states explode to ±∞ (unless S = 0).
    This is also not latching - it's instability. -/
theorem linear_state_explodes (α : ℝ) (hα : 1 < |α|) (S : ℝ) (hS : S ≠ 0) :
    ¬ ∃ L : ℝ, Tendsto (fun n => linearRecurIter α n S) atTop (nhds L) := by
  simp only [linearRecurIter_eq_pow]
  intro ⟨L, hL⟩
  -- |α^n · S| = |α|^n · |S| → ∞
  have h_exp : Tendsto (fun n => |α|^n) atTop atTop :=
    tendsto_pow_atTop_atTop_of_one_lt hα
  have h_mul : Tendsto (fun n => |α|^n * |S|) atTop atTop := by
    have hS_pos : 0 < |S| := abs_pos.mpr hS
    exact Tendsto.atTop_mul_const hS_pos h_exp
  -- |α^n · S| = |α|^n · |S| → ∞
  have h_abs_eq : ∀ n, |α ^ n * S| = |α|^n * |S| := fun n => by
    rw [abs_mul, abs_pow]
  -- The sequence |α^n · S| → ∞
  have h_abs_tendsto : Tendsto (fun n => |α ^ n * S|) atTop atTop := by
    simp only [h_abs_eq]
    exact h_mul
  -- Convergent sequences are bounded, but |α^n · S| → ∞ is unbounded
  -- This is a contradiction
  -- Proof: If α^n * S → L, then |α^n * S| is eventually bounded by |L| + 1
  -- But |α^n * S| → ∞ means it eventually exceeds any bound
  have h_eventually_large := Filter.Tendsto.eventually_gt_atTop h_abs_tendsto (|L| + 1)
  rw [Filter.eventually_atTop] at h_eventually_large
  obtain ⟨N, hN⟩ := h_eventually_large
  -- Also, α^n * S → L means |α^n * S - L| < 1 eventually
  have h_eventually_close := Metric.tendsto_atTop.mp hL 1 (by norm_num : (0 : ℝ) < 1)
  obtain ⟨M, hM⟩ := h_eventually_close
  -- At max(N, M), both conditions hold
  specialize hN (max N M) (le_max_left N M)
  specialize hM (max N M) (le_max_right N M)
  rw [Real.dist_eq] at hM
  -- |α^{max(N,M)} * S| > |L| + 1 and |α^{max(N,M)} * S - L| < 1
  have h1 : |α ^ max N M * S| ≤ |α ^ max N M * S - L| + |L| := by
    have := abs_sub_abs_le_abs_sub (α ^ max N M * S) L
    linarith
  linarith

/-- Summary: Linear systems cannot latch.
    For |α| < 1: states decay to 0 (no memory retention)
    For |α| > 1: states explode (instability, not latching)
    For α = 1: states are static (no processing, just storing)
    None of these is "latching" - stable retention at a nonzero fixed point. -/
theorem linear_cannot_latch (α : ℝ) (S₀ : ℝ) (hS₀ : S₀ ≠ 0) :
    (|α| < 1 → Tendsto (fun n => linearRecurIter α n S₀) atTop (nhds 0)) ∧
    (1 < |α| → ¬ ∃ L : ℝ, Tendsto (fun n => linearRecurIter α n S₀) atTop (nhds L)) ∧
    (α = 1 → ∀ n, linearRecurIter α n S₀ = S₀) := by
  constructor
  · intro h; exact linear_state_decays α h S₀
  constructor
  · intro h; exact linear_state_explodes α h S₀ hS₀
  · intro h_eq_one n
    rw [linearRecurIter_eq_pow, h_eq_one, one_pow, one_mul]

/-! ## Part 7: Main Attention Persistence Theorem -/

/-- **Main Theorem**: E88 heads can enter and persist in an "alert" state.

    This theorem captures the essence of attention persistence:

    1. **Entry**: For α > 1, there exist stable fixed points S* with |S*| close to 1
    2. **Persistence**: States near these fixed points remain near them (attraction)
    3. **Robustness**: Small perturbations don't knock states out of the alert region
    4. **Contrast**: Linear systems cannot achieve this - they decay or explode

    This is why E88 can "pay attention" to a pattern and remember it,
    while linear recurrent models (like Mamba2's within-layer dynamics) cannot.

    CRITICAL: The hypothesis θ < tanh(α * θ) ensures θ is below the fixed point S*(α).
    This is necessary for the alert basin to be non-empty and forward invariant. -/
theorem attention_persistence_main (α θ : ℝ) (hα : 1 < α) (hα_lt : α < 2)
    (hθ : 0 < θ) (hθ_lt : θ < 0.8) (hθ_below_fp : θ < tanh (α * θ)) :
    -- Part 1: Nonzero fixed points exist
    (∃ S_star : ℝ, S_star ≠ 0 ∧ isFixedPoint α S_star) ∧
    -- Part 2: Alert basin is non-empty
    (∃ S : ℝ, S ∈ alertBasin α θ) ∧
    -- Part 3: Alert states are forward invariant
    (∀ S, isAlert S θ → isAlert (tanhRecur α S) θ) ∧
    -- Part 4: Linear systems decay (contrast)
    (∀ β : ℝ, |β| < 1 → ∀ S, Tendsto (fun n => linearRecurIter β n S) atTop (nhds 0)) := by
  constructor
  · exact nonzero_fixed_point_exists α hα
  constructor
  · have hθ_lt' : θ < 1 := by linarith
    exact alert_basin_nonempty α θ hα hθ hθ_lt' hθ_below_fp
  constructor
  · exact alert_forward_invariant α θ hα hθ hθ_lt hθ_below_fp
  · intro β hβ S
    exact linear_state_decays β hβ S

/-! ## Part 8: E88 Head Alert State Persistence -/

/-- E88 head state update with input: S' = tanh(α·S + δ·input). -/
noncomputable def e88HeadUpdate (α δ : ℝ) (S input : ℝ) : ℝ := tanh (α * S + δ * input)

/-- An E88 head can enter alert mode when it sees a strong input.

    NOTE: Requires |S| ≤ 1 (tanh-bounded state) so that α·S doesn't dominate.
    The constraint θ < 0.76 ensures tanh(1) > θ, guaranteeing alertness
    when arg > 1. For θ ∈ [0.76, 1), we'd need δ·input > artanh(θ) + α|S|,
    which is a stronger input requirement. -/
theorem e88_head_can_enter_alert (α δ θ : ℝ) (S : ℝ)
    (hα : 0 < α) (hα_lt : α < 2) (hδ : 0 < δ) (hθ : 0 < θ) (hθ_lt : θ < 0.76)
    (hS_bounded : |S| ≤ 1)
    (input : ℝ) (h_strong : δ * input > θ + 1 + α) :
    isAlert (e88HeadUpdate α δ S input) θ := by
  simp only [isAlert, e88HeadUpdate]
  -- With the new hypotheses:
  -- - |S| ≤ 1, so α*S ≥ -α (since α > 0)
  -- - δ*input > θ + 1 + α
  -- Therefore arg = α*S + δ*input > -α + (θ + 1 + α) = θ + 1 > 1
  -- Since arg > 1 and θ < 0.76 < tanh(1), we have tanh(arg) > tanh(1) > 0.76 > θ
  have hα_pos : 0 < α := hα
  have h_αS_lower : -α ≤ α * S := by
    have h1 : -(|S|) ≤ S := neg_abs_le S
    have hS1 : -1 ≤ -|S| := by linarith [hS_bounded]
    have h2 : α * (-1) ≤ α * (-|S|) := mul_le_mul_of_nonneg_left hS1 (le_of_lt hα_pos)
    have h3 : α * (-|S|) ≤ α * S := mul_le_mul_of_nonneg_left h1 (le_of_lt hα_pos)
    have h4 : -α = α * (-1) := by ring
    linarith
  have h_arg_lower : θ + 1 < α * S + δ * input := by
    -- From h_strong: δ * input > θ + 1 + α
    -- From h_αS_lower: -α ≤ α * S, i.e., α * S ≥ -α
    -- So α * S + δ * input > -α + (θ + 1 + α) = θ + 1
    have h1 : δ * input + α * S > (θ + 1 + α) + (-α) := by linarith [h_strong, h_αS_lower]
    linarith
  have h_arg_gt_one : 1 < α * S + δ * input := by linarith
  have h_arg_pos : 0 < α * S + δ * input := by linarith
  have h_tanh_pos := Activation.tanh_pos_of_pos h_arg_pos
  rw [abs_of_pos h_tanh_pos]
  -- Now use tanh(arg) > tanh(1) > 0.76 > θ
  have h_tanh_mono := Activation.tanh_strictMono
  have h_tanh_gt_one : tanh 1 < tanh (α * S + δ * input) := h_tanh_mono h_arg_gt_one
  have h_tanh_076 := tanh_gt_076_of_ge_one (α * S + δ * input) (le_of_lt h_arg_gt_one)
  linarith

/-- Once in alert mode, an E88 head stays in alert mode under small inputs.
    This is the formalization of attention persistence.

    CRITICAL: Requires θ < tanh(α * θ) to ensure alert states persist. -/
theorem e88_head_stays_alert (α δ θ : ℝ) (S : ℝ)
    (hα : 1 < α) (hα_lt : α < 2) (hδ : |δ| < 0.1) (hθ : 0 < θ) (hθ_lt : θ < 0.8)
    (hθ_below_fp : θ < tanh (α * θ))
    (hS_alert : isAlert S θ)
    (input : ℝ) (h_input_small : |input| ≤ 1) :
    isAlert (e88HeadUpdate α δ S input) (θ - |δ|) := by
  simp only [e88HeadUpdate]
  -- Apply alert_persists_under_perturbation with perturbation δ·input
  have h_pert_bound : |δ * input| ≤ |δ| := by
    calc |δ * input| = |δ| * |input| := abs_mul δ input
      _ ≤ |δ| * 1 := mul_le_mul_of_nonneg_left h_input_small (abs_nonneg δ)
      _ = |δ| := mul_one |δ|
  have hδ_small : |δ| < (1 - θ) / 2 := by
    have h1 : (0.1 : ℝ) = (1 - 0.8) / 2 := by norm_num
    have h2 : (1 - 0.8 : ℝ) / 2 < (1 - θ) / 2 := by
      apply div_lt_div_of_pos_right _ (by norm_num : (0 : ℝ) < 2)
      linarith
    linarith
  exact alert_persists_under_perturbation α θ |δ| hα hα_lt hθ hθ_lt hθ_below_fp hδ_small S hS_alert
    (δ * input) h_pert_bound

/-! ## Part 9: Summary Comparison -/

/-- **Summary Theorem**: The fundamental capability gap between E88 and linear systems.

    E88 (with tanh):
    - Has stable nonzero fixed points (for α > 1)
    - Can enter and maintain alert states
    - Robust to small perturbations

    Linear systems:
    - Only fixed point is 0 (for α ≠ 1)
    - States either decay (|α| < 1) or explode (|α| > 1)
    - Cannot maintain stable nonzero states

    This is why E88 can implement "attention persistence" while linear models cannot.
-/
theorem e88_vs_linear_attention_persistence :
    -- E88 property: nonzero fixed points exist for α > 1
    (∀ α : ℝ, 1 < α → ∃ S : ℝ, S ≠ 0 ∧ isFixedPoint α S) ∧
    -- Linear property: only 0 is fixed for α ≠ 1
    (∀ α : ℝ, α ≠ 1 → ∀ S : ℝ, linearRecur α S = S → S = 0) ∧
    -- Linear property: states decay for |α| < 1
    (∀ α : ℝ, |α| < 1 → ∀ S : ℝ, Tendsto (fun n => linearRecurIter α n S) atTop (nhds 0)) := by
  constructor
  · exact nonzero_fixed_point_exists
  constructor
  · exact linear_fixed_point_is_zero
  · exact linear_state_decays

end AttentionPersistence
