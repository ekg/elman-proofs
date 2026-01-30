/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

/-!
# Numerical Bounds for Tanh and Exp

This file provides tight numerical bounds on exp and tanh that are needed
for the expressivity proofs. These bounds are proven from Mathlib's
`Real.exp_one_gt_d9` which states `exp 1 > 2.7182818283`.

## Main Results

* `exp_2_gt_7` : exp 2 > 7
* `exp_3_gt_20` : exp 3 > 20
* `exp_4_gt_54` : exp 4 > 54
* `tanh_1_gt_076` : tanh 1 > 0.76
* `tanh_15_gt_090` : tanh 1.5 > 0.90
* `tanh_2_gt_096` : tanh 2 > 0.96

-/

namespace NumericalBounds

open Real

/-! ## Exponential Bounds -/

/-- exp 1 > 2.718 from Mathlib -/
lemma exp_1_gt_2718 : exp 1 > 2.718 := by
  have h := Real.exp_one_gt_d9
  linarith

/-- exp 1 > 2.5 (weaker but easier to use) -/
lemma exp_1_gt_25 : exp 1 > 2.5 := by linarith [exp_1_gt_2718]

/-- exp 2 > 7 because exp 2 = (exp 1)² > 2.718² > 7.38 > 7 -/
theorem exp_2_gt_7 : exp 2 > 7 := by
  have h1 : exp 2 = exp 1 * exp 1 := by rw [← Real.exp_add]; norm_num
  have h2 : exp 1 > 2.718 := exp_1_gt_2718
  have h3 : exp 1 > 0 := exp_pos 1
  calc exp 2 = exp 1 * exp 1 := h1
    _ > 2.718 * 2.718 := by nlinarith
    _ > 7 := by norm_num

/-- exp 3 > 20 because exp 3 = exp 1 * exp 2 > 2.718 * 7.38 > 20 -/
theorem exp_3_gt_20 : exp 3 > 20 := by
  have h1 : exp 3 = exp 1 * exp 2 := by rw [← Real.exp_add]; norm_num
  have h2 : exp 1 > 2.718 := exp_1_gt_2718
  have h3 : exp 2 > 7.38 := by
    have h3a : exp 2 = exp 1 * exp 1 := by rw [← Real.exp_add]; norm_num
    have h3b : exp 1 > 2.718 := exp_1_gt_2718
    have h3c : exp 1 > 0 := exp_pos 1
    calc exp 2 = exp 1 * exp 1 := h3a
      _ > 2.718 * 2.718 := by nlinarith
      _ = 7.387524 := by norm_num
      _ > 7.38 := by norm_num
  have h4 : exp 1 > 0 := exp_pos 1
  have h5 : exp 2 > 0 := exp_pos 2
  calc exp 3 = exp 1 * exp 2 := h1
    _ > 2.718 * 7.38 := by nlinarith
    _ > 20 := by norm_num

/-- exp 4 > 54 because exp 4 = (exp 2)² > 7² = 49, actually > 54 -/
theorem exp_4_gt_54 : exp 4 > 54 := by
  have h1 : exp 4 = exp 2 * exp 2 := by rw [← Real.exp_add]; norm_num
  have h2 : exp 2 > 7.38 := by
    have h2a : exp 2 = exp 1 * exp 1 := by rw [← Real.exp_add]; norm_num
    have h2b : exp 1 > 2.718 := exp_1_gt_2718
    have h2c : exp 1 > 0 := exp_pos 1
    calc exp 2 = exp 1 * exp 1 := h2a
      _ > 2.718 * 2.718 := by nlinarith
      _ = 7.387524 := by norm_num
      _ > 7.38 := by norm_num
  have h3 : exp 2 > 0 := exp_pos 2
  calc exp 4 = exp 2 * exp 2 := h1
    _ > 7.38 * 7.38 := by nlinarith
    _ = 54.4644 := by norm_num
    _ > 54 := by norm_num

/-! ## Tanh Bounds

tanh x = (exp(2x) - 1) / (exp(2x) + 1)

For tanh x > c, we need exp(2x) > (1+c)/(1-c).
- tanh x > 0.76 needs exp(2x) > 7.33, so x > 1 works (exp 2 > 7.38)
- tanh x > 0.9 needs exp(2x) > 19, so x > 1.5 works (exp 3 > 20)
- tanh x > 0.96 needs exp(2x) > 49, so x > 2 works (exp 4 > 54)
-/

/-- tanh 1 > 0.76 -/
theorem tanh_1_gt_076 : tanh 1 > 0.76 := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h_exp2 : exp 2 > 7.38 := by
    have h2a : exp 2 = exp 1 * exp 1 := by rw [← Real.exp_add]; norm_num
    have h2b : exp 1 > 2.718 := exp_1_gt_2718
    have h2c : exp 1 > 0 := exp_pos 1
    calc exp 2 = exp 1 * exp 1 := h2a
      _ > 2.718 * 2.718 := by nlinarith
      _ = 7.387524 := by norm_num
      _ > 7.38 := by norm_num
  have h_exp1_pos : exp 1 > 0 := exp_pos 1
  have h_exp_neg1 : exp (-1) = (exp 1)⁻¹ := Real.exp_neg 1
  have h_inv_pos : 0 < (exp 1)⁻¹ := inv_pos.mpr h_exp1_pos
  -- sinh(1)/cosh(1) = (e - 1/e)/(e + 1/e) = (e² - 1)/(e² + 1)
  have h_simp : (exp 1 - (exp 1)⁻¹) / 2 / ((exp 1 + (exp 1)⁻¹) / 2) =
                (exp 2 - 1) / (exp 2 + 1) := by
    have h_sq : exp 1 * exp 1 = exp 2 := by rw [← Real.exp_add]; norm_num
    have hne : exp 1 ≠ 0 := h_exp1_pos.ne'
    field_simp
    rw [← h_sq]
    ring
  rw [h_exp_neg1, h_simp]
  -- (exp 2 - 1)/(exp 2 + 1) > 0.76 iff exp 2 > (1+0.76)/(1-0.76) = 7.33...
  have h_denom_pos : exp 2 + 1 > 0 := by linarith [exp_pos 2]
  -- Want: 0.76 < (exp 2 - 1) / (exp 2 + 1)
  -- i.e., 0.76 * (exp 2 + 1) < exp 2 - 1
  -- i.e., 19/25 * (exp 2 + 1) < exp 2 - 1
  -- Multiply through: 19 * (exp 2 + 1) < 25 * (exp 2 - 1)
  -- 19 exp 2 + 19 < 25 exp 2 - 25
  -- 44 < 6 exp 2
  -- exp 2 > 44/6 = 7.333...
  rw [gt_iff_lt, lt_div_iff₀ h_denom_pos]
  -- 0.76 * (exp 2 + 1) < exp 2 - 1
  linarith

/-- tanh 1.5 > 0.90 -/
theorem tanh_15_gt_090 : tanh 1.5 > 0.90 := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h_exp3 : exp 3 > 20 := exp_3_gt_20
  have h_exp15_pos : exp 1.5 > 0 := exp_pos 1.5
  have h_exp_neg15 : exp (-1.5) = (exp 1.5)⁻¹ := Real.exp_neg 1.5
  have h_inv_pos : 0 < (exp 1.5)⁻¹ := inv_pos.mpr h_exp15_pos
  have h_simp : (exp 1.5 - (exp 1.5)⁻¹) / 2 / ((exp 1.5 + (exp 1.5)⁻¹) / 2) =
                (exp 3 - 1) / (exp 3 + 1) := by
    have h_sq : exp 1.5 * exp 1.5 = exp 3 := by rw [← Real.exp_add]; norm_num
    have hne : exp 1.5 ≠ 0 := h_exp15_pos.ne'
    field_simp
    rw [← h_sq]
    ring
  rw [h_exp_neg15, h_simp]
  -- (exp 3 - 1)/(exp 3 + 1) > 0.90 iff exp 3 > (1+0.9)/(1-0.9) = 19
  have h_denom_pos : exp 3 + 1 > 0 := by linarith [exp_pos 3]
  rw [gt_iff_lt, lt_div_iff₀ h_denom_pos]
  -- 0.9 * (exp 3 + 1) < exp 3 - 1
  -- 9/10 * (exp 3 + 1) < exp 3 - 1
  -- 9 (exp 3 + 1) < 10 (exp 3 - 1)
  -- 9 exp 3 + 9 < 10 exp 3 - 10
  -- 19 < exp 3
  linarith

/-- tanh 2 > 0.96 -/
theorem tanh_2_gt_096 : tanh 2 > 0.96 := by
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h_exp4 : exp 4 > 54 := exp_4_gt_54
  have h_exp2_pos : exp 2 > 0 := exp_pos 2
  have h_exp_neg2 : exp (-2) = (exp 2)⁻¹ := Real.exp_neg 2
  have h_inv_pos : 0 < (exp 2)⁻¹ := inv_pos.mpr h_exp2_pos
  have h_simp : (exp 2 - (exp 2)⁻¹) / 2 / ((exp 2 + (exp 2)⁻¹) / 2) =
                (exp 4 - 1) / (exp 4 + 1) := by
    have h_sq : exp 2 * exp 2 = exp 4 := by rw [← Real.exp_add]; norm_num
    have hne : exp 2 ≠ 0 := h_exp2_pos.ne'
    field_simp
    rw [← h_sq]
    ring
  rw [h_exp_neg2, h_simp]
  -- (exp 4 - 1)/(exp 4 + 1) > 0.96 iff exp 4 > (1+0.96)/(1-0.96) = 49
  have h_denom_pos : exp 4 + 1 > 0 := by linarith [exp_pos 4]
  rw [gt_iff_lt, lt_div_iff₀ h_denom_pos]
  -- 0.96 * (exp 4 + 1) < exp 4 - 1
  -- 24/25 * (exp 4 + 1) < exp 4 - 1
  -- 24 (exp 4 + 1) < 25 (exp 4 - 1)
  -- 24 exp 4 + 24 < 25 exp 4 - 25
  -- 49 < exp 4
  linarith

/-! ## Additional Useful Bounds -/

/-- tanh is strictly monotone -/
theorem tanh_strictMono : StrictMono tanh := by
  intro x y hxy
  rw [Real.tanh_eq_sinh_div_cosh, Real.tanh_eq_sinh_div_cosh]
  have hcx : 0 < cosh x := cosh_pos x
  have hcy : 0 < cosh y := cosh_pos y
  rw [div_lt_div_iff₀ hcx hcy]
  -- sinh x * cosh y < sinh y * cosh x
  -- -sinh(y-x) < 0
  -- sinh(y-x) > 0
  -- y - x > 0
  have h : sinh x * cosh y - sinh y * cosh x = -sinh (y - x) := by
    rw [Real.sinh_sub]
    ring
  have h2 : sinh x * cosh y < sinh y * cosh x ↔ sinh (y - x) > 0 := by
    constructor
    · intro hlt
      have hsub : sinh x * cosh y - sinh y * cosh x < 0 := sub_neg_of_lt hlt
      rw [h] at hsub
      linarith
    · intro hpos
      have hsub : sinh x * cosh y - sinh y * cosh x < 0 := by rw [h]; linarith
      linarith
  rw [h2, gt_iff_lt, Real.sinh_pos_iff]
  linarith

/-- For x ≥ 2, tanh x > 0.96 -/
theorem tanh_ge_2_gt_096 (x : ℝ) (hx : x ≥ 2) : tanh x > 0.96 := by
  have h := tanh_2_gt_096
  have h_mono := tanh_strictMono
  calc tanh x ≥ tanh 2 := h_mono.monotone hx
    _ > 0.96 := h

/-- For x ≥ 1.5, tanh x > 0.90 -/
theorem tanh_ge_15_gt_090 (x : ℝ) (hx : x ≥ 1.5) : tanh x > 0.90 := by
  have h := tanh_15_gt_090
  have h_mono := tanh_strictMono
  calc tanh x ≥ tanh 1.5 := h_mono.monotone hx
    _ > 0.90 := h

/-- For x ≥ 1, tanh x > 0.76 -/
theorem tanh_ge_1_gt_076 (x : ℝ) (hx : x ≥ 1) : tanh x > 0.76 := by
  have h := tanh_1_gt_076
  have h_mono := tanh_strictMono
  calc tanh x ≥ tanh 1 := h_mono.monotone hx
    _ > 0.76 := h

/-- tanh x > 0 when x > 0 -/
theorem tanh_pos_of_pos {x : ℝ} (hx : 0 < x) : 0 < tanh x := by
  have h := tanh_strictMono
  calc tanh x > tanh 0 := h hx
    _ = 0 := tanh_zero

/-- |tanh x| > 0.76 when |x| ≥ 1 -/
theorem abs_tanh_ge_1_gt_076 (x : ℝ) (hx : |x| ≥ 1) : |tanh x| > 0.76 := by
  rcases le_or_gt 0 x with hpos | hneg
  · -- x ≥ 0 case
    have hx' : x ≥ 1 := by
      rw [abs_of_nonneg hpos] at hx
      exact hx
    have h := tanh_ge_1_gt_076 x hx'
    have h_pos : tanh x > 0 := tanh_pos_of_pos (by linarith)
    rw [abs_of_pos h_pos]
    exact h
  · -- x < 0 case
    have hx' : -x ≥ 1 := by
      rw [abs_of_neg hneg] at hx
      exact hx
    have h := tanh_ge_1_gt_076 (-x) hx'
    have h_pos : tanh (-x) > 0 := tanh_pos_of_pos (by linarith)
    -- tanh x = -tanh(-x), so |tanh x| = |tanh(-x)| = tanh(-x) since tanh(-x) > 0
    have h_eq : tanh x = -tanh (-x) := by rw [Real.tanh_neg]; ring
    rw [h_eq, abs_neg, abs_of_pos h_pos]
    exact h

end NumericalBounds
