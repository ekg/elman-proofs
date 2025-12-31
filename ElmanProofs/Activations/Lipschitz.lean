/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Lipschitz Properties of Activation Functions

This file proves Lipschitz constants for common activation functions used in RNNs.
The Lipschitz constant is critical for contraction analysis.

## Main Results

* `tanh_lipschitz`: tanh is 1-Lipschitz
* `sigmoid_lipschitz`: sigmoid is 1/4-Lipschitz
* `relu_lipschitz`: ReLU is 1-Lipschitz

## Implications for RNNs

If an activation σ is L-Lipschitz and ‖R_h‖ < 1/L, then the RNN is a contraction.
For tanh (L=1), we need ‖R_h‖ < 1.
For sigmoid (L=1/4), we need ‖R_h‖ < 4.

-/

namespace Activation

open Real

/-- Bounded activation functions have bounded outputs.
    tanh = sinh/cosh ∈ (-1, 1) since cosh² - sinh² = 1 and cosh > 0.

    Proof: From cosh² - sinh² = 1 and cosh > 0, we have sinh² < cosh²,
    hence |sinh| < cosh, so |tanh| = |sinh|/cosh < 1. -/
theorem tanh_bounded (x : ℝ) : |Real.tanh x| < 1 := by
  rw [Real.tanh_eq_sinh_div_cosh]
  rw [abs_div]
  have hcosh_pos : 0 < Real.cosh x := Real.cosh_pos x
  rw [div_lt_one (abs_pos.mpr hcosh_pos.ne')]
  rw [abs_of_pos hcosh_pos]
  -- From cosh² - sinh² = 1 and cosh > 0, sinh² < cosh²
  have h := Real.cosh_sq_sub_sinh_sq x
  have hsinh_sq : Real.sinh x ^ 2 < Real.cosh x ^ 2 := by linarith [sq_nonneg (Real.cosh x)]
  -- |sinh x| < cosh x follows from sinh² < cosh²
  rw [abs_lt]
  constructor
  · -- -cosh x < sinh x
    nlinarith [sq_abs (Real.sinh x), sq_abs (Real.cosh x)]
  · -- sinh x < cosh x
    nlinarith [sq_abs (Real.sinh x), sq_abs (Real.cosh x)]

/-- tanh is differentiable everywhere (composition of exp functions). -/
theorem differentiable_tanh : Differentiable ℝ Real.tanh := by
  have h_eq : Real.tanh = Real.sinh / Real.cosh := by
    ext y
    exact Real.tanh_eq_sinh_div_cosh y
  rw [h_eq]
  exact Real.differentiable_sinh.div Real.differentiable_cosh (fun x => (Real.cosh_pos x).ne')

/-- The derivative of tanh is 1 - tanh².
    Proof: tanh = sinh/cosh, so by quotient rule:
    tanh' = (sinh' · cosh - sinh · cosh') / cosh²
          = (cosh · cosh - sinh · sinh) / cosh²
          = (cosh² - sinh²) / cosh²
          = 1 / cosh² = sech² = 1 - tanh² -/
theorem deriv_tanh (x : ℝ) : deriv Real.tanh x = 1 - (Real.tanh x)^2 := by
  have h_eq : Real.tanh = Real.sinh / Real.cosh := by
    ext y
    exact Real.tanh_eq_sinh_div_cosh y
  conv_lhs => rw [h_eq]
  conv_rhs => rw [Real.tanh_eq_sinh_div_cosh]
  have hcosh_pos : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  simp only [deriv_div Real.differentiable_sinh.differentiableAt
      Real.differentiable_cosh.differentiableAt hcosh_pos]
  rw [Real.deriv_sinh, Real.deriv_cosh]
  -- (cosh * cosh - sinh * sinh) / cosh² = 1 - sinh²/cosh²
  have h := Real.cosh_sq_sub_sinh_sq x
  field_simp [h]

/-- The derivative of tanh is bounded by 1.
    Proof: tanh'(x) = 1 - tanh²(x) ∈ (0, 1] since |tanh x| < 1. -/
theorem tanh_deriv_bound (x : ℝ) : |deriv Real.tanh x| ≤ 1 := by
  rw [deriv_tanh]
  have h := tanh_bounded x
  -- |tanh x| < 1 implies tanh² < 1
  have h_sq : (Real.tanh x)^2 < 1 := by
    rw [sq_lt_one_iff_abs_lt_one]
    exact h
  have h_nonneg : 0 ≤ 1 - (Real.tanh x)^2 := by linarith
  rw [abs_of_nonneg h_nonneg]
  have h_sq_nonneg : 0 ≤ (Real.tanh x)^2 := sq_nonneg _
  linarith

/-- tanh is 1-Lipschitz: |tanh(x) - tanh(y)| ≤ |x - y|.
    Proof: tanh' = 1 - tanh² ∈ (0, 1], so by MVT |tanh x - tanh y| ≤ |x - y|. -/
theorem tanh_lipschitz : LipschitzWith 1 Real.tanh := by
  apply lipschitzWith_of_nnnorm_deriv_le differentiable_tanh
  intro x
  rw [← NNReal.coe_le_coe, NNReal.coe_one, coe_nnnorm]
  exact tanh_deriv_bound x

/-- sigmoid(x) = 1 / (1 + exp(-x)). -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- sigmoid is bounded in (0, 1). -/
theorem sigmoid_bounded (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  constructor
  · simp only [sigmoid, one_div]
    apply inv_pos.mpr
    linarith [exp_pos (-x)]
  · simp only [sigmoid, one_div]
    have h : 1 < 1 + exp (-x) := by linarith [exp_pos (-x)]
    exact inv_lt_one_of_one_lt₀ h

/-- sigmoid is continuous everywhere. -/
theorem continuous_sigmoid : Continuous sigmoid := by
  unfold sigmoid
  exact continuous_one.div (continuous_one.add (continuous_exp.comp continuous_neg))
    fun x => by linarith [exp_pos (-x)]

/-- Helper: exp ∘ (-id) is differentiable. -/
theorem differentiableAt_exp_neg (x : ℝ) : DifferentiableAt ℝ (fun y => exp (-y)) x := by
  exact Real.differentiable_exp.differentiableAt.comp x differentiable_neg.differentiableAt

/-- sigmoid is differentiable everywhere.
    Proof: sigmoid = 1 / (1 + exp(-x)) is a quotient of differentiable functions.
    The denominator 1 + exp(-x) > 1 > 0 is never zero. -/
theorem differentiable_sigmoid : Differentiable ℝ sigmoid := by
  intro x
  unfold sigmoid
  have h_denom_ne : 1 + exp (-x) ≠ 0 := by linarith [exp_pos (-x)]
  have h_num : DifferentiableAt ℝ (fun _ : ℝ => (1 : ℝ)) x := differentiableAt_const 1
  have h_exp_neg : DifferentiableAt ℝ (fun y => exp (-y)) x := differentiableAt_exp_neg x
  have h_denom : DifferentiableAt ℝ (fun y => 1 + exp (-y)) x :=
    (differentiableAt_const 1).add h_exp_neg
  exact DifferentiableAt.div h_num h_denom h_denom_ne

/-- The derivative of sigmoid equals sigmoid(x) * (1 - sigmoid(x)).
    Proof via quotient rule on 1 / (1 + exp(-x)). -/
theorem deriv_sigmoid (x : ℝ) : deriv sigmoid x = sigmoid x * (1 - sigmoid x) := by
  unfold sigmoid
  have h_denom_ne : 1 + exp (-x) ≠ 0 := by linarith [exp_pos (-x)]
  have h_denom_pos : 0 < 1 + exp (-x) := by linarith [exp_pos (-x)]
  have h_num : DifferentiableAt ℝ (fun _ : ℝ => (1 : ℝ)) x := differentiableAt_const 1
  have h_exp_neg : DifferentiableAt ℝ (fun y => exp (-y)) x := differentiableAt_exp_neg x
  have h_denom : DifferentiableAt ℝ (fun y => 1 + exp (-y)) x :=
    (differentiableAt_const 1).add h_exp_neg
  simp only [deriv_fun_div h_num h_denom h_denom_ne]
  simp only [deriv_const, zero_mul, zero_sub]
  -- deriv of 1 + exp(-y) = deriv of exp(-y) = -exp(-x)
  have h_denom_deriv : deriv (fun y => 1 + exp (-y)) x = -exp (-x) := by
    have h2 : (fun y => 1 + exp (-y)) = (fun y => (1 : ℝ) + exp (-y)) := rfl
    rw [h2, deriv_const_add]
    -- deriv of exp(-y) at x = exp(-x) * (-1) = -exp(-x)
    -- Use HasDerivAt for the chain rule
    have h_neg : HasDerivAt Neg.neg (-1 : ℝ) x := hasDerivAt_neg x
    have h_exp : HasDerivAt exp (exp (-x)) (-x) := Real.hasDerivAt_exp (-x)
    have h_comp : HasDerivAt (fun y => exp (-y)) (exp (-x) * (-1)) x :=
      h_exp.comp x h_neg
    simp only [mul_neg_one] at h_comp
    exact h_comp.deriv
  rw [h_denom_deriv]
  -- Simplify: -(1 * (-exp(-x))) / (1 + exp(-x))^2 = exp(-x) / (1 + exp(-x))^2
  field_simp
  ring

/-- The derivative of sigmoid is bounded by 1/4.
    Since sigmoid(x) ∈ (0, 1), the product sigmoid(x)(1 - sigmoid(x)) is maximized
    when sigmoid(x) = 1/2, giving maximum derivative value 1/4.

    Proof: sigmoid'(x) = sigmoid(x) · (1 - sigmoid(x)).
    For s ∈ (0,1), s(1-s) ≤ 1/4 by AM-GM: s(1-s) ≤ ((s + (1-s))/2)² = 1/4. -/
theorem sigmoid_deriv_bound (x : ℝ) : |deriv sigmoid x| ≤ 1/4 := by
  rw [deriv_sigmoid]
  have hs := sigmoid_bounded x
  set s := sigmoid x with hs_def
  have h_prod_nonneg : 0 ≤ s * (1 - s) := mul_nonneg (le_of_lt hs.1) (by linarith [hs.2])
  have h_prod_bound : s * (1 - s) ≤ 1/4 := by nlinarith [hs.1, hs.2, sq_nonneg (s - 1/2)]
  rw [abs_of_nonneg h_prod_nonneg]
  exact h_prod_bound

/-- sigmoid is 1/4-Lipschitz.
    Proof: By the mean value theorem, for any x, y with x ≠ y,
    there exists c between x and y such that
    sigmoid(x) - sigmoid(y) = sigmoid'(c) * (x - y).
    Since |sigmoid'(c)| ≤ 1/4, we get |sigmoid(x) - sigmoid(y)| ≤ 1/4 * |x - y|. -/
theorem sigmoid_lipschitz : ∀ x y : ℝ, |sigmoid x - sigmoid y| ≤ (1/4) * |x - y| := by
  intro x y
  by_cases hxy : x = y
  · simp [hxy]
  · by_cases h : x < y
    · -- Case: x < y, apply MVT
      have hab : x < y := h
      have hcont : ContinuousOn sigmoid (Set.Icc x y) :=
        continuous_sigmoid.continuousOn
      have hdiff : DifferentiableOn ℝ sigmoid (Set.Ioo x y) :=
        (differentiable_sigmoid).differentiableOn
      obtain ⟨c, _hc_mem, hc_deriv⟩ :=
        exists_deriv_eq_slope sigmoid hab hcont hdiff
      have mvt_eq : sigmoid y - sigmoid x = deriv sigmoid c * (y - x) := by
        have h_ne : y - x ≠ 0 := by linarith
        field_simp [h_ne] at hc_deriv
        field_simp
        exact hc_deriv.symm
      calc |sigmoid x - sigmoid y|
          = |-(sigmoid y - sigmoid x)| := by simp [neg_sub]
        _ = |sigmoid y - sigmoid x| := abs_neg _
        _ = |deriv sigmoid c * (y - x)| := by rw [mvt_eq]
        _ = |deriv sigmoid c| * |y - x| := abs_mul _ _
        _ ≤ (1/4) * |y - x| := mul_le_mul_of_nonneg_right (sigmoid_deriv_bound c) (abs_nonneg _)
        _ = (1/4) * |x - y| := by rw [abs_sub_comm x y]
    · -- Case: y < x (since x ≠ y and ¬(x < y))
      push_neg at h
      have hyx : y < x := Ne.lt_of_le (Ne.symm hxy) h
      have hab : y < x := hyx
      have hcont : ContinuousOn sigmoid (Set.Icc y x) :=
        continuous_sigmoid.continuousOn
      have hdiff : DifferentiableOn ℝ sigmoid (Set.Ioo y x) :=
        (differentiable_sigmoid).differentiableOn
      obtain ⟨c, _hc_mem, hc_deriv⟩ :=
        exists_deriv_eq_slope sigmoid hab hcont hdiff
      have mvt_eq : sigmoid x - sigmoid y = deriv sigmoid c * (x - y) := by
        have h_ne : x - y ≠ 0 := by linarith
        field_simp [h_ne] at hc_deriv
        field_simp
        exact hc_deriv.symm
      calc |sigmoid x - sigmoid y|
          = |deriv sigmoid c * (x - y)| := by rw [mvt_eq]
        _ = |deriv sigmoid c| * |x - y| := abs_mul _ _
        _ ≤ (1/4) * |x - y| := mul_le_mul_of_nonneg_right (sigmoid_deriv_bound c) (abs_nonneg _)

/-- ReLU is 1-Lipschitz. -/
def relu (x : ℝ) : ℝ := max 0 x

theorem relu_lipschitz : LipschitzWith 1 relu := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simp only [relu, Real.dist_eq, NNReal.coe_one, one_mul]
  -- |max 0 x - max 0 y| ≤ |x - y|
  -- Case analysis on signs of x and y
  rcases le_or_gt x 0 with hx | hx <;> rcases le_or_gt y 0 with hy | hy
  · -- x ≤ 0, y ≤ 0: max 0 x = 0, max 0 y = 0
    simp only [max_eq_left hx, max_eq_left hy, sub_self, abs_zero, abs_nonneg]
  · -- x ≤ 0, y > 0: max 0 x = 0, max 0 y = y
    simp only [max_eq_left hx, max_eq_right (le_of_lt hy)]
    rw [zero_sub, abs_neg, abs_of_pos hy]
    have : y ≤ |x - y| := by
      rw [abs_sub_comm]
      calc y = y - x + x := by ring
        _ ≤ y - x + 0 := by linarith
        _ = y - x := by ring
        _ ≤ |y - x| := le_abs_self _
    exact this
  · -- x > 0, y ≤ 0: max 0 x = x, max 0 y = 0
    simp only [max_eq_right (le_of_lt hx), max_eq_left hy, sub_zero]
    rw [abs_of_pos hx]
    calc x = x - y + y := by ring
      _ ≤ x - y + 0 := by linarith
      _ = x - y := by ring
      _ ≤ |x - y| := le_abs_self _
  · -- x > 0, y > 0: max 0 x = x, max 0 y = y
    simp only [max_eq_right (le_of_lt hx), max_eq_right (le_of_lt hy)]
    exact le_refl _

/-- SiLU(x) = x * sigmoid(x). -/
noncomputable def silu (x : ℝ) : ℝ := x * sigmoid x

/-- Monotonicity of relu. -/
theorem relu_monotone : Monotone relu := by
  intro x y hxy
  simp only [relu]
  exact max_le_max le_rfl hxy

end Activation
