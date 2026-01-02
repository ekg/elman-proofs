/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Gradient Flow and Learning Dynamics

This file formalizes gradient descent as a dynamical system and proves
convergence results relevant to neural network training.

## Main Definitions

* `GradientDescentStep`: One step of gradient descent
* `IsLSmooth`: Function with L-Lipschitz gradient
* `IsStronglyConvex`: μ-strongly convex function

## Main Theorems

* `gradient_descent_convex`: O(1/k) convergence for convex functions
* `gradient_descent_strongly_convex`: O(c^k) convergence for strongly convex

## Application to RNN Training

For RNN training with loss L(θ):
- If L is L-smooth and μ-strongly convex, gradient descent converges linearly
- The condition number κ = L/μ determines convergence rate

-/

namespace Gradient

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- A function is L-smooth if its gradient is L-Lipschitz. -/
def IsLSmooth (f : E → ℝ) (L : ℝ) : Prop :=
  Differentiable ℝ f ∧ ∀ x y, ‖gradient f x - gradient f y‖ ≤ L * ‖x - y‖

/-- A function is μ-strongly convex. -/
def IsStronglyConvex (f : E → ℝ) (μ : ℝ) : Prop :=
  ∀ x y : E, ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
    f (t • x + (1 - t) • y) ≤ t * f x + (1 - t) * f y - (μ / 2) * t * (1 - t) * ‖x - y‖^2

/-- Strong convexity implies ordinary convexity.

    If f is μ-strongly convex with μ ≥ 0, then f is convex on the whole space.
-/
theorem stronglyConvex_implies_convexOn (f : E → ℝ) (μ : ℝ) (hμ : 0 ≤ μ)
    (hStrong : IsStronglyConvex f μ) : ConvexOn ℝ Set.univ f := by
  constructor
  · exact convex_univ
  · intro x _ y _ a b ha hb hab
    -- ConvexOn uses weights a, b with a + b = 1
    -- IsStronglyConvex uses t with (1-t)
    -- We have: a • x + b • y where a + b = 1
    -- We want to show: f (a • x + b • y) ≤ a * f x + b * f y
    -- Using IsStronglyConvex with t = a:
    -- f (a • x + (1 - a) • y) ≤ a * f x + (1 - a) * f y - (μ/2) * a * (1 - a) * ‖x - y‖²
    have hb_eq : b = 1 - a := by linarith
    rw [hb_eq]
    have ha_le_1 : a ≤ 1 := by linarith
    have h1_minus_a_nonneg : 0 ≤ 1 - a := by linarith
    have hStrong' := hStrong x y a ha ha_le_1
    have h_nonneg : 0 ≤ (μ / 2) * a * (1 - a) * ‖x - y‖^2 := by
      have h1 : 0 ≤ μ / 2 := by linarith
      have h2 : 0 ≤ a * (1 - a) := mul_nonneg ha h1_minus_a_nonneg
      have h3 : 0 ≤ (μ / 2) * (a * (1 - a)) := mul_nonneg h1 h2
      have h4 : 0 ≤ ‖x - y‖ ^ 2 := sq_nonneg _
      calc (μ / 2) * a * (1 - a) * ‖x - y‖^2
          = (μ / 2) * (a * (1 - a)) * ‖x - y‖^2 := by ring
        _ ≥ 0 := mul_nonneg h3 h4
    -- Convert smul to mul for reals: a • r = a * r
    simp only [smul_eq_mul] at *
    linarith

/-- Strong convexity implies a lower bound on the gradient inner product.

    For μ-strongly convex f with ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ (μ/2)‖x - x*‖²

    This follows from the first-order characterization of strong convexity.
-/
theorem strong_convex_gradient_lower_bound (f : E → ℝ) (μ : ℝ) (hμ : 0 < μ)
    (hStrong : IsStronglyConvex f μ) (hDiff : Differentiable ℝ f)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    @inner ℝ E _ (gradient f x) (x - x_star) ≥ (μ / 2) * ‖x - x_star‖^2 := by
  /- The proof uses the first-order characterization of strong convexity.

     For μ-strongly convex f, the definition gives:
     f(t·a + (1-t)·b) ≤ t·f(a) + (1-t)·f(b) - (μ/2)·t·(1-t)·‖a - b‖²

     The first-order characterization (for differentiable f) is:
     f(y) ≥ f(x) + ⟨∇f(x), y - x⟩ + (μ/2)·‖y - x‖²

     Setting y = x* (where ∇f(x*) = 0):
     f(x*) ≥ f(x) + ⟨∇f(x), x* - x⟩ + (μ/2)·‖x* - x‖²

     Rearranging:
     ⟨∇f(x), x - x*⟩ = -⟨∇f(x), x* - x⟩ ≥ f(x) - f(x*) + (μ/2)·‖x - x*‖²

     Since x* is a critical point (∇f(x*) = 0) for a strongly convex function,
     it's the unique global minimum, so f(x) - f(x*) ≥ 0.

     Therefore: ⟨∇f(x), x - x*⟩ ≥ (μ/2)·‖x - x*‖²

     The key step requiring formalization is deriving the first-order
     characterization from the definition of IsStronglyConvex.
     This typically requires taking limits as t → 0 in the definition.
  -/

  -- The formal proof requires:
  -- 1. Deriving first-order characterization from IsStronglyConvex
  -- 2. Using that ∇f(x*) = 0 implies x* is global minimum for strongly convex f
  -- 3. Combining the bounds

  -- Key derivation: From the strong convexity definition with a = x, b = x*, t ∈ (0,1]:
  -- f(t•x + (1-t)•x*) ≤ t•f(x) + (1-t)•f(x*) - (μ/2)•t•(1-t)•‖x - x*‖²
  --
  -- Rearranging: f(x* + t(x - x*)) ≤ f(x*) + t(f(x) - f(x*)) - (μ/2)t(1-t)‖x - x*‖²
  --
  -- Taking the derivative w.r.t. t at t = 0 (using differentiability):
  -- LHS derivative: ⟨∇f(x*), x - x*⟩ = 0 (since ∇f(x*) = 0)
  -- RHS derivative: f(x) - f(x*) - (μ/2)(1)‖x - x*‖² = f(x) - f(x*) - (μ/2)‖x - x*‖²
  --
  -- Wait, this gives information at x*, not at x. Let me use a = x*, b = x instead:
  -- f(t•x* + (1-t)•x) ≤ t•f(x*) + (1-t)•f(x) - (μ/2)•t•(1-t)•‖x* - x‖²
  --
  -- Rewrite LHS: f(x + t(x* - x)) = f(x - t(x - x*))
  --
  -- Taking derivative w.r.t. t at t = 0:
  -- LHS: ⟨∇f(x), x* - x⟩ = -⟨∇f(x), x - x*⟩
  -- RHS: (d/dt)[t•f(x*) + (1-t)•f(x) - (μ/2)•t•(1-t)•‖x - x*‖²] at t=0
  --     = f(x*) - f(x) - (μ/2)•(1-2t)•‖x - x*‖² at t=0
  --     = f(x*) - f(x) - (μ/2)‖x - x*‖²
  --
  -- The strong convexity inequality gives: LHS ≤ RHS (as t → 0⁺)
  -- -⟨∇f(x), x - x*⟩ ≤ f(x*) - f(x) - (μ/2)‖x - x*‖²
  -- ⟨∇f(x), x - x*⟩ ≥ f(x) - f(x*) + (μ/2)‖x - x*‖²
  --
  -- Since x* is a critical point of strongly convex f, it's the global minimum:
  -- f(x) - f(x*) ≥ 0
  --
  -- Therefore: ⟨∇f(x), x - x*⟩ ≥ (μ/2)‖x - x*‖²

  -- The formal proof requires taking limits as t → 0 in the strong convexity
  -- definition and using differentiability. This involves:
  -- 1. Showing the function g(t) = f(x + t(x* - x)) is differentiable at t = 0
  -- 2. Computing g'(0) = ⟨∇f(x), x* - x⟩
  -- 3. Bounding g(t) using strong convexity
  -- 4. Taking the limit to get the first-order condition

  -- Define the direction and path
  let d := x_star - x
  let g := fun t : ℝ => f (x + t • d)
  -- The upper bound from strong convexity: h(t) = (1-t)f(x) + tf(x*) - (μ/2)t(1-t)‖d‖²
  let h := fun t : ℝ => (1 - t) * f x + t * f x_star - (μ / 2) * t * (1 - t) * ‖d‖^2
  -- Strong convexity gives g(t) ≤ h(t) for t ∈ [0, 1]
  have h_ineq : ∀ t, 0 ≤ t → t ≤ 1 → g t ≤ h t := by
    intro t ht0 ht1
    have hconv := hStrong x_star x t ht0 ht1
    -- t•x* + (1-t)•x = x + t•(x* - x) = x + t•d
    have heq : t • x_star + (1 - t) • x = x + t • d := by
      simp only [d]; rw [smul_sub]; ring_nf; module
    simp only [g, h, heq] at hconv ⊢
    have hnorm : ‖x_star - x‖ = ‖d‖ := by simp only [d]
    rw [hnorm] at hconv
    linarith
  -- At t = 0: g(0) = h(0) = f(x)
  have hg0 : g 0 = f x := by simp only [g, zero_smul, add_zero]
  have hh0 : h 0 = f x := by simp only [h]; ring
  -- Compute h'(0) = f(x*) - f(x) - (μ/2)‖d‖²
  have h_deriv : HasDerivAt h (f x_star - f x - (μ / 2) * ‖d‖^2) 0 := by
    -- h(t) = (1-t)f(x) + tf(x*) - (μ/2)t(1-t)‖d‖²
    -- Rewrite as: h(t) = f(x) + t*(f(x*) - f(x)) - (μ/2)*‖d‖²*(t - t²)
    -- h'(t) = f(x*) - f(x) - (μ/2)*‖d‖²*(1 - 2t)
    -- h'(0) = f(x*) - f(x) - (μ/2)*‖d‖²
    have h1 : HasDerivAt (fun t : ℝ => (1 - t) * f x) (-f x) 0 := by
      have hid : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
        (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
      convert hid.mul_const (f x) using 1; ring
    have h2 : HasDerivAt (fun t : ℝ => t * f x_star) (f x_star) 0 := by
      convert (hasDerivAt_id (0 : ℝ)).mul_const (f x_star) using 1; ring
    have h3 : HasDerivAt (fun t : ℝ => (μ / 2) * t * (1 - t) * ‖d‖^2) ((μ / 2) * ‖d‖^2) 0 := by
      -- (μ/2)*t*(1-t)*‖d‖² has derivative (μ/2)*‖d‖²*(1 - 2t) at t
      -- At t = 0: (μ/2)*‖d‖²
      have hpoly : HasDerivAt (fun t : ℝ => t * (1 - t)) 1 0 := by
        have h1' := hasDerivAt_id (0 : ℝ)
        have h2' : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
          (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
        have hprod := h1'.mul h2'
        convert hprod using 2 <;> simp
      convert hpoly.const_mul ((μ / 2) * ‖d‖^2) using 1
      · ext t; ring
      · ring
    convert (h1.add h2).sub h3 using 1; ring
  -- Compute g'(0) = ⟨∇f(x), d⟩
  have g_deriv : HasDerivAt g (@inner ℝ E _ (gradient f x) d) 0 := by
    have hγ : HasDerivAt (fun t : ℝ => x + t • d) d 0 := by
      have h1 : HasDerivAt (fun _ : ℝ => x) 0 0 := hasDerivAt_const 0 x
      have h2 : HasDerivAt (fun t : ℝ => t • d) ((1 : ℝ) • d) 0 :=
        (hasDerivAt_id 0).smul_const d
      have hsum := h1.add h2
      simp only [zero_add, one_smul] at hsum
      exact hsum
    have hf_grad : HasGradientAt f (gradient f x) x := (hDiff x).hasGradientAt
    have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x)) x := hf_grad.hasFDerivAt
    have hf_fderiv' : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x)) (x + (0 : ℝ) • d) := by
      simp only [zero_smul, add_zero]; exact hf_fderiv
    have hcomp := hf_fderiv'.comp_hasDerivAt (0 : ℝ) hγ
    simp only [Function.comp_apply, innerSL_apply_apply, zero_smul, add_zero] at hcomp
    exact hcomp
  -- Key lemma: if g(0) = h(0) and g(t) ≤ h(t) for t ∈ (0, 1], then g'(0) ≤ h'(0)
  -- This follows from: (g(t) - g(0))/t ≤ (h(t) - h(0))/t for t > 0
  -- Taking limit as t → 0⁺ gives g'(0) ≤ h'(0)
  have h_deriv_ineq : @inner ℝ E _ (gradient f x) d ≤ f x_star - f x - (μ / 2) * ‖d‖^2 := by
    by_contra hcontra
    push_neg at hcontra
    -- Let δ = g'(0) - h'(0) > 0
    let δ := @inner ℝ E _ (gradient f x) d - (f x_star - f x - (μ / 2) * ‖d‖^2)
    have hδ_pos : δ > 0 := by simp only [δ]; linarith
    -- From HasDerivAt, the difference quotient converges to the derivative
    -- For g: (g(t) - g(0))/t → g'(0) as t → 0
    -- For h: (h(t) - h(0))/t → h'(0) as t → 0
    -- So (g(t) - h(t))/t → g'(0) - h'(0) = δ > 0
    have h_gh_deriv : HasDerivAt (fun t => g t - h t) δ 0 := HasDerivAt.sub g_deriv h_deriv
    have h_gh_0 : (fun t => g t - h t) 0 = 0 := by simp only [hg0, hh0, sub_self]
    -- HasDerivAt gives: (g-h)(t) = (g-h)(0) + δ*t + o(t) = δ*t + o(t)
    -- For small t > 0: (g-h)(t) ≈ δ*t > 0 since δ > 0
    rw [hasDerivAt_iff_isLittleO] at h_gh_deriv
    -- h_gh_deriv : (fun t => (g-h)(0+t) - (g-h)(0) - (t-0)•δ) =o[𝓝 0] (fun t => t-0)
    -- Use IsLittleO.def to get: for c = δ/2 > 0, eventually ‖...‖ ≤ c * ‖t - 0‖
    have hε_half : 0 < δ / 2 := by linarith
    have h_bound_evt := h_gh_deriv.def hε_half
    -- h_bound_evt : ∀ᶠ t in 𝓝 0, ‖(g t - h t) - (g 0 - h 0) - (t - 0) • δ‖ ≤ (δ/2) * ‖t - 0‖
    simp only [h_gh_0, sub_zero, smul_eq_mul] at h_bound_evt
    -- h_bound_evt : ∀ᶠ t in 𝓝 0, ‖g t - h t - t * δ‖ ≤ (δ/2) * ‖t‖
    rw [Filter.eventually_iff_exists_mem] at h_bound_evt
    obtain ⟨s, hs_mem, hs_bound⟩ := h_bound_evt
    rw [Metric.mem_nhds_iff] at hs_mem
    obtain ⟨ε, hε_pos, hε_sub⟩ := hs_mem
    -- Pick t = min(ε/2, 1/2) > 0
    let t := min (ε / 2) (1 / 2)
    have ht_pos : 0 < t := by positivity
    have ht_lt_ε : t < ε := by simp only [t]; linarith [min_le_left (ε / 2) (1 / 2)]
    have ht_le_1 : t ≤ 1 := by simp only [t]; linarith [min_le_right (ε / 2) (1 / 2)]
    have ht_in_ball : t ∈ Metric.ball 0 ε := by
      simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos ht_pos]
      exact ht_lt_ε
    have ht_in_s : t ∈ s := hε_sub ht_in_ball
    -- hs_bound says: ‖(g-h)(t) - t*δ‖ ≤ (δ/2) * ‖t‖
    have h_bound := hs_bound t ht_in_s
    simp only [Real.norm_eq_abs, abs_of_pos ht_pos] at h_bound
    -- h_bound : |g t - h t - t * δ| ≤ (δ / 2) * t
    -- |f(t) - t*δ| ≤ (δ/2)*t means f(t) ≥ t*δ - (δ/2)*t = (δ/2)*t > 0
    have h_lower : g t - h t ≥ t * δ - (δ / 2) * t := by
      have h1 : -((δ / 2) * t) ≤ (g t - h t) - t * δ := by
        have := neg_abs_le (g t - h t - t * δ)
        linarith
      linarith
    have h_diff_pos : g t - h t > 0 := by
      have : t * δ - (δ / 2) * t = (δ / 2) * t := by ring
      rw [this] at h_lower
      have : (δ / 2) * t > 0 := mul_pos (by linarith) ht_pos
      linarith
    -- But h_ineq says g(t) ≤ h(t), contradiction
    have h_le := h_ineq t (le_of_lt ht_pos) ht_le_1
    linarith
  -- Now: ⟨∇f(x), d⟩ ≤ f(x*) - f(x) - (μ/2)‖d‖²
  -- Since d = x* - x, we have ⟨∇f(x), x - x*⟩ = -⟨∇f(x), d⟩
  have h_inner_neg : @inner ℝ E _ (gradient f x) (x - x_star) =
      -@inner ℝ E _ (gradient f x) d := by
    simp only [d, ← inner_neg_right, neg_sub]
  rw [h_inner_neg]
  -- Need: -⟨∇f(x), d⟩ ≥ (μ/2)‖x - x*‖²
  -- From h_deriv_ineq: ⟨∇f(x), d⟩ ≤ f(x*) - f(x) - (μ/2)‖d‖²
  -- So: -⟨∇f(x), d⟩ ≥ f(x) - f(x*) + (μ/2)‖d‖²
  -- Need to show f(x) - f(x*) ≥ 0, i.e., x* is global minimum
  have h_min : f x_star ≤ f x := by
    -- Use derivative limit argument at x* with ∇f(x*) = 0
    -- Define path from x* to x: p(t) = f(x* + t(x - x*))
    -- Strong convexity gives p(t) ≤ RHS, and taking derivative limit at t = 0
    -- with p'(0) = ⟨∇f(x*), x - x*⟩ = 0 gives the desired inequality.
    let e := x - x_star
    let p := fun t : ℝ => f (x_star + t • e)
    let q := fun t : ℝ => t * f x + (1 - t) * f x_star - (μ / 2) * t * (1 - t) * ‖e‖^2
    -- Strong convexity gives p(t) ≤ q(t) for t ∈ [0, 1]
    have hpq_ineq : ∀ t, 0 ≤ t → t ≤ 1 → p t ≤ q t := by
      intro t ht0 ht1
      have hconv := hStrong x x_star t ht0 ht1
      have heq : t • x + (1 - t) • x_star = x_star + t • e := by
        simp only [e]; rw [smul_sub]; ring_nf; module
      simp only [p, q, heq] at hconv ⊢
      have hnorm : ‖x - x_star‖ = ‖e‖ := by simp only [e]
      rw [hnorm] at hconv
      linarith
    -- At t = 0: p(0) = q(0) = f(x*)
    have hp0 : p 0 = f x_star := by simp only [p, zero_smul, add_zero]
    have hq0 : q 0 = f x_star := by simp only [q]; ring
    -- Compute q'(0) = f(x) - f(x*) - (μ/2)‖e‖²
    have q_deriv : HasDerivAt q (f x - f x_star - (μ / 2) * ‖e‖^2) 0 := by
      have h1 : HasDerivAt (fun t : ℝ => t * f x) (f x) 0 := by
        convert (hasDerivAt_id (0 : ℝ)).mul_const (f x) using 1; ring
      have h2 : HasDerivAt (fun t : ℝ => (1 - t) * f x_star) (-f x_star) 0 := by
        have hid : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
          (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
        convert hid.mul_const (f x_star) using 1; ring
      have h3 : HasDerivAt (fun t : ℝ => (μ / 2) * t * (1 - t) * ‖e‖^2) ((μ / 2) * ‖e‖^2) 0 := by
        have hpoly : HasDerivAt (fun t : ℝ => t * (1 - t)) 1 0 := by
          have ha : HasDerivAt (fun t : ℝ => t) 1 0 := hasDerivAt_id (0 : ℝ)
          have hb : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
            (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
          exact (ha.mul hb).congr_deriv (by simp [id])
        convert hpoly.const_mul ((μ / 2) * ‖e‖^2) using 1
        · ext t; ring
        · ring
      convert (h1.add h2).sub h3 using 1 <;> ring
    -- Compute p'(0) = ⟨∇f(x*), e⟩ = 0 (since ∇f(x*) = 0)
    have p_deriv : HasDerivAt p 0 0 := by
      have hγ : HasDerivAt (fun t : ℝ => x_star + t • e) e 0 := by
        have h1 : HasDerivAt (fun _ : ℝ => x_star) 0 0 := hasDerivAt_const 0 x_star
        have h2 : HasDerivAt (fun t : ℝ => t • e) ((1 : ℝ) • e) 0 :=
          (hasDerivAt_id 0).smul_const e
        have hsum := h1.add h2
        simp only [zero_add, one_smul] at hsum
        exact hsum
      have hf_grad : HasGradientAt f (gradient f x_star) x_star := (hDiff x_star).hasGradientAt
      have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) x_star :=
        hf_grad.hasFDerivAt
      have hf_fderiv' : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) (x_star + (0 : ℝ) • e) := by
        simp only [zero_smul, add_zero]; exact hf_fderiv
      have hcomp := hf_fderiv'.comp_hasDerivAt (0 : ℝ) hγ
      simp only [Function.comp_apply, innerSL_apply_apply, zero_smul, add_zero, hMin, inner_zero_left] at hcomp
      exact hcomp
    -- Key: if p(0) = q(0), p(t) ≤ q(t) for t > 0, and both differentiable at 0, then p'(0) ≤ q'(0)
    have hderiv_ineq : 0 ≤ f x - f x_star - (μ / 2) * ‖e‖^2 := by
      by_contra hcontra
      push_neg at hcontra
      -- Let δ = p'(0) - q'(0) = 0 - q'(0) = -(f x - f x_star - (μ/2)‖e‖²) > 0
      let δ := -(f x - f x_star - (μ / 2) * ‖e‖^2)
      have hδ_pos : δ > 0 := by simp only [δ]; linarith
      have h_pq_deriv : HasDerivAt (fun t => p t - q t) δ 0 := by
        have := HasDerivAt.sub p_deriv q_deriv
        convert this using 2
        simp only [δ]; ring
      have h_pq_0 : (fun t => p t - q t) 0 = 0 := by simp only [hp0, hq0, sub_self]
      -- Use isLittleO characterization instead of tendsto_slope (which gives nhdsWithin)
      rw [hasDerivAt_iff_isLittleO] at h_pq_deriv
      have hε_half : 0 < δ / 2 := by linarith
      have h_bound_evt := h_pq_deriv.def hε_half
      simp only [h_pq_0, sub_zero, smul_eq_mul] at h_bound_evt
      rw [Filter.eventually_iff_exists_mem] at h_bound_evt
      obtain ⟨s, hs_mem, hs_bound⟩ := h_bound_evt
      rw [Metric.mem_nhds_iff] at hs_mem
      obtain ⟨ε, hε_pos, hε_sub⟩ := hs_mem
      let t := min (ε / 2) (1 / 2)
      have ht_pos : 0 < t := by positivity
      have ht_lt_ε : t < ε := by simp only [t]; linarith [min_le_left (ε / 2) (1 / 2)]
      have ht_le_1 : t ≤ 1 := by simp only [t]; linarith [min_le_right (ε / 2) (1 / 2)]
      have ht_in_ball : t ∈ Metric.ball 0 ε := by
        simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos ht_pos]
        exact ht_lt_ε
      have ht_in_s : t ∈ s := hε_sub ht_in_ball
      have h_bound := hs_bound t ht_in_s
      simp only [Real.norm_eq_abs, abs_of_pos ht_pos] at h_bound
      -- h_bound : ‖p t - q t - t * δ‖ ≤ (δ/2) * t
      -- This means: -(δ/2)*t ≤ (p t - q t) - t*δ ≤ (δ/2)*t
      -- So: t*δ - (δ/2)*t ≤ p t - q t, i.e., (δ/2)*t ≤ p t - q t
      have h_lower : p t - q t ≥ t * δ - (δ / 2) * t := by
        have h1 : -((δ / 2) * t) ≤ (p t - q t) - t * δ := by
          have := neg_abs_le (p t - q t - t * δ)
          linarith
        linarith
      have h_diff_pos : p t - q t > 0 := by
        have : t * δ - (δ / 2) * t = (δ / 2) * t := by ring
        rw [this] at h_lower
        have : (δ / 2) * t > 0 := mul_pos (by linarith) ht_pos
        linarith
      have h_le := hpq_ineq t (le_of_lt ht_pos) ht_le_1
      linarith
    -- From 0 ≤ f(x) - f(x*) - (μ/2)‖e‖², we get f(x*) ≤ f(x) - (μ/2)‖e‖² ≤ f(x)
    have h_e_sq_nonneg : 0 ≤ (μ / 2) * ‖e‖^2 := by positivity
    linarith
  have h_d_norm : ‖d‖ = ‖x - x_star‖ := by simp only [d, norm_sub_rev]
  rw [h_d_norm] at h_deriv_ineq
  linarith

/-- Gradient monotonicity for strongly convex functions (full μ, not μ/2).

    For μ-strongly convex f with ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ μ‖x - x*‖²

    This is twice as strong as `strong_convex_gradient_lower_bound` and comes from
    adding the first-order conditions at both x and x*.

    Proof:
    1. First-order at x: ⟨∇f(x), x* - x⟩ ≤ f(x*) - f(x) - (μ/2)‖x - x*‖²
       → ⟨∇f(x), x - x*⟩ ≥ f(x) - f(x*) + (μ/2)‖x - x*‖²
    2. First-order at x* with ∇f(x*) = 0: 0 ≤ f(x) - f(x*) - (μ/2)‖x - x*‖²
       → f(x) - f(x*) ≥ (μ/2)‖x - x*‖²
    3. Combining: ⟨∇f(x), x - x*⟩ ≥ (μ/2)‖x - x*‖² + (μ/2)‖x - x*‖² = μ‖x - x*‖²
-/
theorem strong_convex_gradient_monotonicity (f : E → ℝ) (μ : ℝ) (hμ : 0 < μ)
    (hStrong : IsStronglyConvex f μ) (hDiff : Differentiable ℝ f)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    @inner ℝ E _ (gradient f x) (x - x_star) ≥ μ * ‖x - x_star‖^2 := by
  -- The proof combines two first-order conditions from strong convexity.
  -- Define d = x* - x and e = x - x*
  let d := x_star - x
  let e := x - x_star

  -- Part 1: First-order condition at x gives:
  -- ⟨∇f(x), x - x*⟩ ≥ f(x) - f(x*) + (μ/2)‖x - x*‖²
  -- (This is derived in strong_convex_gradient_lower_bound as h_deriv_ineq)

  -- We'll derive both bounds together using the same technique.

  -- Step A: Derive ⟨∇f(x), d⟩ ≤ f(x*) - f(x) - (μ/2)‖d‖² via derivative limit
  let g := fun t : ℝ => f (x + t • d)
  let h := fun t : ℝ => (1 - t) * f x + t * f x_star - (μ / 2) * t * (1 - t) * ‖d‖^2
  have h_ineq : ∀ t, 0 ≤ t → t ≤ 1 → g t ≤ h t := by
    intro t ht0 ht1
    have hconv := hStrong x_star x t ht0 ht1
    have heq : t • x_star + (1 - t) • x = x + t • d := by
      simp only [d]; rw [smul_sub]; ring_nf; module
    simp only [g, h, heq] at hconv ⊢
    have hnorm : ‖x_star - x‖ = ‖d‖ := by simp only [d]
    rw [hnorm] at hconv
    linarith
  have hg0 : g 0 = f x := by simp only [g, zero_smul, add_zero]
  have hh0 : h 0 = f x := by simp only [h]; ring
  have h_deriv : HasDerivAt h (f x_star - f x - (μ / 2) * ‖d‖^2) 0 := by
    have h1 : HasDerivAt (fun t : ℝ => (1 - t) * f x) (-f x) 0 := by
      have hid : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
        (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
      convert hid.mul_const (f x) using 1; ring
    have h2 : HasDerivAt (fun t : ℝ => t * f x_star) (f x_star) 0 := by
      convert (hasDerivAt_id (0 : ℝ)).mul_const (f x_star) using 1; ring
    have h3 : HasDerivAt (fun t : ℝ => (μ / 2) * t * (1 - t) * ‖d‖^2) ((μ / 2) * ‖d‖^2) 0 := by
      have hpoly : HasDerivAt (fun t : ℝ => t * (1 - t)) 1 0 := by
        have h1' := hasDerivAt_id (0 : ℝ)
        have h2' : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
          (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
        have hprod := h1'.mul h2'
        convert hprod using 2 <;> simp
      convert hpoly.const_mul ((μ / 2) * ‖d‖^2) using 1
      · ext t; ring
      · ring
    convert (h1.add h2).sub h3 using 1; ring
  have g_deriv : HasDerivAt g (@inner ℝ E _ (gradient f x) d) 0 := by
    have hγ : HasDerivAt (fun t : ℝ => x + t • d) d 0 := by
      have h1 : HasDerivAt (fun _ : ℝ => x) 0 0 := hasDerivAt_const 0 x
      have h2 : HasDerivAt (fun t : ℝ => t • d) ((1 : ℝ) • d) 0 :=
        (hasDerivAt_id 0).smul_const d
      have hsum := h1.add h2
      simp only [zero_add, one_smul] at hsum
      exact hsum
    have hf_grad : HasGradientAt f (gradient f x) x := (hDiff x).hasGradientAt
    have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x)) x := hf_grad.hasFDerivAt
    have hf_fderiv' : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x)) (x + (0 : ℝ) • d) := by
      simp only [zero_smul, add_zero]; exact hf_fderiv
    have hcomp := hf_fderiv'.comp_hasDerivAt (0 : ℝ) hγ
    simp only [Function.comp_apply, innerSL_apply_apply, zero_smul, add_zero] at hcomp
    exact hcomp
  -- Derivative limit argument: g(0) = h(0), g ≤ h on (0,1], so g'(0) ≤ h'(0)
  have h_deriv_ineq : @inner ℝ E _ (gradient f x) d ≤ f x_star - f x - (μ / 2) * ‖d‖^2 := by
    by_contra hcontra
    push_neg at hcontra
    let δ := @inner ℝ E _ (gradient f x) d - (f x_star - f x - (μ / 2) * ‖d‖^2)
    have hδ_pos : δ > 0 := by simp only [δ]; linarith
    have h_gh_deriv : HasDerivAt (fun t => g t - h t) δ 0 := HasDerivAt.sub g_deriv h_deriv
    have h_gh_0 : (fun t => g t - h t) 0 = 0 := by simp only [hg0, hh0, sub_self]
    rw [hasDerivAt_iff_isLittleO] at h_gh_deriv
    have hε_half : 0 < δ / 2 := by linarith
    have h_bound_evt := h_gh_deriv.def hε_half
    simp only [h_gh_0, sub_zero, smul_eq_mul] at h_bound_evt
    rw [Filter.eventually_iff_exists_mem] at h_bound_evt
    obtain ⟨s, hs_mem, hs_bound⟩ := h_bound_evt
    rw [Metric.mem_nhds_iff] at hs_mem
    obtain ⟨ε, hε_pos, hε_sub⟩ := hs_mem
    let t := min (ε / 2) (1 / 2)
    have ht_pos : 0 < t := by positivity
    have ht_lt_ε : t < ε := by simp only [t]; linarith [min_le_left (ε / 2) (1 / 2)]
    have ht_le_1 : t ≤ 1 := by simp only [t]; linarith [min_le_right (ε / 2) (1 / 2)]
    have ht_in_ball : t ∈ Metric.ball 0 ε := by
      simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos ht_pos]
      exact ht_lt_ε
    have ht_in_s : t ∈ s := hε_sub ht_in_ball
    have h_bound := hs_bound t ht_in_s
    simp only [Real.norm_eq_abs, abs_of_pos ht_pos] at h_bound
    have h_lower : g t - h t ≥ t * δ - (δ / 2) * t := by
      have h1 : -((δ / 2) * t) ≤ (g t - h t) - t * δ := by
        have := neg_abs_le (g t - h t - t * δ)
        linarith
      linarith
    have h_diff_pos : g t - h t > 0 := by
      have : t * δ - (δ / 2) * t = (δ / 2) * t := by ring
      rw [this] at h_lower
      have : (δ / 2) * t > 0 := mul_pos (by linarith) ht_pos
      linarith
    have h_le := h_ineq t (le_of_lt ht_pos) ht_le_1
    linarith

  -- Step B: Derive f(x) - f(x*) ≥ (μ/2)‖e‖² via derivative limit at x*
  let p := fun t : ℝ => f (x_star + t • e)
  let q := fun t : ℝ => t * f x + (1 - t) * f x_star - (μ / 2) * t * (1 - t) * ‖e‖^2
  have hpq_ineq : ∀ t, 0 ≤ t → t ≤ 1 → p t ≤ q t := by
    intro t ht0 ht1
    have hconv := hStrong x x_star t ht0 ht1
    have heq : t • x + (1 - t) • x_star = x_star + t • e := by
      simp only [e]; rw [smul_sub]; ring_nf; module
    simp only [p, q, heq] at hconv ⊢
    have hnorm : ‖x - x_star‖ = ‖e‖ := by simp only [e]
    rw [hnorm] at hconv
    linarith
  have hp0 : p 0 = f x_star := by simp only [p, zero_smul, add_zero]
  have hq0 : q 0 = f x_star := by simp only [q]; ring
  have q_deriv : HasDerivAt q (f x - f x_star - (μ / 2) * ‖e‖^2) 0 := by
    have h1 : HasDerivAt (fun t : ℝ => t * f x) (f x) 0 := by
      convert (hasDerivAt_id (0 : ℝ)).mul_const (f x) using 1; ring
    have h2 : HasDerivAt (fun t : ℝ => (1 - t) * f x_star) (-f x_star) 0 := by
      have hid : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
        (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
      convert hid.mul_const (f x_star) using 1; ring
    have h3 : HasDerivAt (fun t : ℝ => (μ / 2) * t * (1 - t) * ‖e‖^2) ((μ / 2) * ‖e‖^2) 0 := by
      have hpoly : HasDerivAt (fun t : ℝ => t * (1 - t)) 1 0 := by
        have ha : HasDerivAt (fun t : ℝ => t) 1 0 := hasDerivAt_id (0 : ℝ)
        have hb : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 :=
          (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ)) |>.congr_deriv (by ring)
        exact (ha.mul hb).congr_deriv (by simp [id])
      convert hpoly.const_mul ((μ / 2) * ‖e‖^2) using 1
      · ext t; ring
      · ring
    convert (h1.add h2).sub h3 using 1 <;> ring
  have p_deriv : HasDerivAt p 0 0 := by
    have hγ : HasDerivAt (fun t : ℝ => x_star + t • e) e 0 := by
      have h1 : HasDerivAt (fun _ : ℝ => x_star) 0 0 := hasDerivAt_const 0 x_star
      have h2 : HasDerivAt (fun t : ℝ => t • e) ((1 : ℝ) • e) 0 :=
        (hasDerivAt_id 0).smul_const e
      have hsum := h1.add h2
      simp only [zero_add, one_smul] at hsum
      exact hsum
    have hf_grad : HasGradientAt f (gradient f x_star) x_star := (hDiff x_star).hasGradientAt
    have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) x_star :=
      hf_grad.hasFDerivAt
    have hf_fderiv' : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) (x_star + (0 : ℝ) • e) := by
      simp only [zero_smul, add_zero]; exact hf_fderiv
    have hcomp := hf_fderiv'.comp_hasDerivAt (0 : ℝ) hγ
    simp only [Function.comp_apply, innerSL_apply_apply, zero_smul, add_zero, hMin, inner_zero_left] at hcomp
    exact hcomp
  -- p'(0) = 0 ≤ q'(0) = f(x) - f(x*) - (μ/2)‖e‖² gives f(x) - f(x*) ≥ (μ/2)‖e‖²
  have h_func_bound : 0 ≤ f x - f x_star - (μ / 2) * ‖e‖^2 := by
    by_contra hcontra
    push_neg at hcontra
    let δ := -(f x - f x_star - (μ / 2) * ‖e‖^2)
    have hδ_pos : δ > 0 := by simp only [δ]; linarith
    have h_pq_deriv : HasDerivAt (fun t => p t - q t) δ 0 := by
      have := HasDerivAt.sub p_deriv q_deriv
      convert this using 2
      simp only [δ]; ring
    have h_pq_0 : (fun t => p t - q t) 0 = 0 := by simp only [hp0, hq0, sub_self]
    rw [hasDerivAt_iff_isLittleO] at h_pq_deriv
    have hε_half : 0 < δ / 2 := by linarith
    have h_bound_evt := h_pq_deriv.def hε_half
    simp only [h_pq_0, sub_zero, smul_eq_mul] at h_bound_evt
    rw [Filter.eventually_iff_exists_mem] at h_bound_evt
    obtain ⟨s, hs_mem, hs_bound⟩ := h_bound_evt
    rw [Metric.mem_nhds_iff] at hs_mem
    obtain ⟨ε, hε_pos, hε_sub⟩ := hs_mem
    let t := min (ε / 2) (1 / 2)
    have ht_pos : 0 < t := by positivity
    have ht_lt_ε : t < ε := by simp only [t]; linarith [min_le_left (ε / 2) (1 / 2)]
    have ht_le_1 : t ≤ 1 := by simp only [t]; linarith [min_le_right (ε / 2) (1 / 2)]
    have ht_in_ball : t ∈ Metric.ball 0 ε := by
      simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos ht_pos]
      exact ht_lt_ε
    have ht_in_s : t ∈ s := hε_sub ht_in_ball
    have h_bound := hs_bound t ht_in_s
    simp only [Real.norm_eq_abs, abs_of_pos ht_pos] at h_bound
    have h_lower : p t - q t ≥ t * δ - (δ / 2) * t := by
      have h1 : -((δ / 2) * t) ≤ (p t - q t) - t * δ := by
        have := neg_abs_le (p t - q t - t * δ)
        linarith
      linarith
    have h_diff_pos : p t - q t > 0 := by
      have : t * δ - (δ / 2) * t = (δ / 2) * t := by ring
      rw [this] at h_lower
      have : (δ / 2) * t > 0 := mul_pos (by linarith) ht_pos
      linarith
    have h_le := hpq_ineq t (le_of_lt ht_pos) ht_le_1
    linarith

  -- Step C: Combine the two bounds
  -- From h_deriv_ineq: ⟨∇f(x), d⟩ ≤ f(x*) - f(x) - (μ/2)‖d‖²
  -- So: ⟨∇f(x), x - x*⟩ = -⟨∇f(x), d⟩ ≥ f(x) - f(x*) + (μ/2)‖d‖²
  have h_inner_neg : @inner ℝ E _ (gradient f x) (x - x_star) =
      -@inner ℝ E _ (gradient f x) d := by
    simp only [d, ← inner_neg_right, neg_sub]
  have h_d_norm : ‖d‖ = ‖x - x_star‖ := by simp only [d, norm_sub_rev]
  have h_e_norm : ‖e‖ = ‖x - x_star‖ := by simp only [e]

  -- From h_deriv_ineq: -⟨∇f(x), d⟩ ≥ f(x) - f(x*) + (μ/2)‖d‖²
  have h_inner_lb : @inner ℝ E _ (gradient f x) (x - x_star) ≥
      f x - f x_star + (μ / 2) * ‖x - x_star‖^2 := by
    rw [h_inner_neg]
    simp only [h_d_norm] at h_deriv_ineq
    linarith

  -- From h_func_bound: f(x) - f(x*) ≥ (μ/2)‖e‖² = (μ/2)‖x - x*‖²
  have h_func_lb : f x - f x_star ≥ (μ / 2) * ‖x - x_star‖^2 := by
    rw [h_e_norm] at h_func_bound
    linarith

  -- Combine: ⟨∇f(x), x - x*⟩ ≥ (μ/2)‖x - x*‖² + (μ/2)‖x - x*‖² = μ‖x - x*‖²
  calc @inner ℝ E _ (gradient f x) (x - x_star)
      ≥ f x - f x_star + (μ / 2) * ‖x - x_star‖^2 := h_inner_lb
    _ ≥ (μ / 2) * ‖x - x_star‖^2 + (μ / 2) * ‖x - x_star‖^2 := by linarith [h_func_lb]
    _ = μ * ‖x - x_star‖^2 := by ring

-- strong_smooth_interpolation is defined below after lsmooth_cocoercivity

/-- Fundamental inequality for L-smooth functions:
    f(y) ≤ f(x) + ⟨∇f(x), y - x⟩ + (L/2)‖y - x‖²

    ## Mathematical Proof

    This follows from integrating the gradient along the line from x to y
    and using the Lipschitz condition on the gradient.

    **Step 1: Define the path**

    Let γ(t) = x + t(y - x) for t ∈ [0, 1].
    Then γ(0) = x and γ(1) = y.

    **Step 2: Apply the Fundamental Theorem of Calculus**

    Define g(t) = f(γ(t)). By chain rule: g'(t) = ⟨∇f(γ(t)), y - x⟩.

    Therefore: f(y) - f(x) = g(1) - g(0) = ∫₀¹ g'(t) dt = ∫₀¹ ⟨∇f(γ(t)), y - x⟩ dt.

    **Step 3: Decompose and bound**

    f(y) - f(x) - ⟨∇f(x), y - x⟩
      = ∫₀¹ ⟨∇f(γ(t)), y - x⟩ dt - ⟨∇f(x), y - x⟩
      = ∫₀¹ ⟨∇f(γ(t)) - ∇f(x), y - x⟩ dt

    By Cauchy-Schwarz:
      |⟨∇f(γ(t)) - ∇f(x), y - x⟩| ≤ ‖∇f(γ(t)) - ∇f(x)‖ · ‖y - x‖

    By L-smoothness (gradient is L-Lipschitz):
      ‖∇f(γ(t)) - ∇f(x)‖ ≤ L · ‖γ(t) - x‖ = L · t · ‖y - x‖

    Therefore:
      ⟨∇f(γ(t)) - ∇f(x), y - x⟩ ≤ L · t · ‖y - x‖²

    **Step 4: Integrate**

    f(y) - f(x) - ⟨∇f(x), y - x⟩ ≤ ∫₀¹ L · t · ‖y - x‖² dt
                                   = L · ‖y - x‖² · ∫₀¹ t dt
                                   = L · ‖y - x‖² · (1/2)
                                   = (L/2) · ‖y - x‖²

    **Lean Formalization Requirements**

    1. `MeasureTheory.integral_Icc_eq_integral_Ioc` - integration on [0,1]
    2. `HasDerivAt.integral_eq_sub` - FTC for path integrals
    3. `MeasureTheory.integral_mono` - for bounding integrals
    4. `integral_id` or similar for ∫₀¹ t dt = 1/2
-/
theorem lsmooth_fundamental_ineq (f : E → ℝ) (L : ℝ) (hL : 0 ≤ L)
    (hSmooth : IsLSmooth f L) (x y : E) :
    f y ≤ f x + @inner ℝ E _ (gradient f x) (y - x) + (L / 2) * ‖y - x‖^2 := by
  obtain ⟨hDiff, hLip⟩ := hSmooth
  -- Special case: if x = y, the inequality is trivially true
  by_cases hxy : x = y
  · simp only [hxy, sub_self, inner_zero_right, norm_zero, sq, mul_zero, add_zero, le_refl]
  -- Special case: if L = 0, gradient is constant, so f is affine
  by_cases hL0 : L = 0
  · -- When L = 0, ∇f is constant (0-Lipschitz means constant)
    -- So f(y) = f(x) + ⟨∇f(x), y - x⟩ for all x, y
    simp only [hL0, zero_div, zero_mul, add_zero]
    -- For constant gradient, f is affine: f(y) - f(x) = ⟨∇f(x), y - x⟩
    -- From 0-Lipschitz: ‖∇f(x) - ∇f(y)‖ ≤ 0 * ‖x - y‖ = 0. So ∇f(x) = ∇f(y) for all x, y.
    -- When gradient is constant, by the MVT: f(y) - f(x) = ⟨∇f(ξ), y - x⟩ for some ξ.
    -- Since ∇f is constant, ∇f(ξ) = ∇f(x), so f(y) - f(x) = ⟨∇f(x), y - x⟩
    have h_grad_const : ∀ z, gradient f z = gradient f x := by
      intro z
      have h0 : ‖gradient f z - gradient f x‖ ≤ 0 * ‖z - x‖ := by
        rw [← hL0]
        exact hLip z x
      simp only [zero_mul, norm_le_zero_iff] at h0
      exact sub_eq_zero.mp h0
    -- For the formal proof, we use that zero Frechet derivative implies constant.
    -- Define h(z) = f(z) - ⟨∇f(x), z⟩. Then fderiv h z = 0 (gradient is constant).
    -- Zero fderiv on convex set implies h is constant, so h(y) = h(x).
    let g := gradient f x
    let h := fun z => f z - @inner ℝ E _ g z
    have hh_diff : Differentiable ℝ h := by
      intro z
      apply DifferentiableAt.sub (hDiff z)
      exact (innerSL (𝕜 := ℝ) g).differentiableAt
    -- h has zero Frechet derivative everywhere
    have h_fderiv_zero : ∀ z, fderiv ℝ h z = 0 := by
      intro z
      have hf_diff : DifferentiableAt ℝ f z := hDiff z
      have hg_diff : DifferentiableAt ℝ (fun w => @inner ℝ E _ g w) z :=
        (innerSL (𝕜 := ℝ) g).differentiableAt
      -- fderiv of f z = innerSL (gradient f z)
      have h_fderiv_f : fderiv ℝ f z = innerSL (𝕜 := ℝ) (gradient f z) := by
        have hgrad := hf_diff.hasGradientAt
        exact hgrad.hasFDerivAt.fderiv
      -- fderiv of (inner g ·) = innerSL g
      have h_fderiv_inner : fderiv ℝ (fun w => @inner ℝ E _ g w) z = innerSL (𝕜 := ℝ) g :=
        (innerSL (𝕜 := ℝ) g).fderiv
      -- fderiv of h = fderiv f - fderiv inner
      have h1 : fderiv ℝ h z = fderiv ℝ f z - fderiv ℝ (fun w => @inner ℝ E _ g w) z := by
        exact fderiv_sub hf_diff hg_diff
      rw [h1, h_fderiv_f, h_fderiv_inner, h_grad_const z]
      exact sub_self _
    -- h is constant: use that zero derivative on convex set implies constant
    have h_const : h y = h x := by
      have hconvex : Convex ℝ (Set.univ : Set E) := convex_univ
      have hdiff_on : DifferentiableOn ℝ h Set.univ := hh_diff.differentiableOn
      have hfderiv_on : ∀ z ∈ Set.univ, fderivWithin ℝ h Set.univ z = 0 := by
        intro z _
        rw [fderivWithin_univ]
        exact h_fderiv_zero z
      exact Convex.is_const_of_fderivWithin_eq_zero hconvex hdiff_on hfderiv_on
        (Set.mem_univ x) (Set.mem_univ y)
    -- Expand h(y) = h(x): f(y) - ⟨g, y⟩ = f(x) - ⟨g, x⟩, so f(y) = f(x) + ⟨g, y - x⟩
    simp only [h] at h_const
    have h_inner_sub : @inner ℝ E _ g y - @inner ℝ E _ g x = @inner ℝ E _ g (y - x) := by
      rw [inner_sub_right]
    linarith [h_const, h_inner_sub]
  -- Main case: L > 0
  have hL_pos : 0 < L := lt_of_le_of_ne hL (Ne.symm hL0)
  /- The proof uses integration along the line segment from x to y.

     Define γ(t) = x + t(y - x) for t ∈ [0, 1].
     Define g(t) = f(γ(t)).

     Then g'(t) = ⟨∇f(γ(t)), y - x⟩.

     By the fundamental theorem of calculus:
     f(y) - f(x) = g(1) - g(0) = ∫₀¹ g'(t) dt = ∫₀¹ ⟨∇f(γ(t)), y - x⟩ dt

     Therefore:
     f(y) - f(x) - ⟨∇f(x), y - x⟩ = ∫₀¹ ⟨∇f(γ(t)) - ∇f(x), y - x⟩ dt

     By Cauchy-Schwarz and L-Lipschitz gradient:
     ⟨∇f(γ(t)) - ∇f(x), y - x⟩ ≤ ‖∇f(γ(t)) - ∇f(x)‖ · ‖y - x‖
                                 ≤ L · ‖γ(t) - x‖ · ‖y - x‖
                                 = L · t · ‖y - x‖²

     Integrating:
     f(y) - f(x) - ⟨∇f(x), y - x⟩ ≤ ∫₀¹ L · t · ‖y - x‖² dt
                                   = L · ‖y - x‖² · [t²/2]₀¹
                                   = (L/2) · ‖y - x‖²

     This requires Mathlib's MeasureTheory.integral machinery and
     careful handling of the FTC for paths in Hilbert spaces.

     **Mathlib theorems needed**:
     - `MeasureTheory.integral_Icc` for ∫₀¹ ... dt
     - `HasDerivAt.integral_eq_sub` for FTC
     - `real_inner_le_norm` for Cauchy-Schwarz
     - `intervalIntegral.integral_mono` for bounding integrals

     **Alternative approach via second derivative**:
     Define g(t) = f(x + t(y-x)). Then:
     - g'(t) = ⟨∇f(x + t(y-x)), y - x⟩
     - g''(t) = ⟨Hf(x + t(y-x))(y-x), y - x⟩ where Hf is the Hessian
     - For L-smooth f, the Hessian satisfies ‖Hf‖ ≤ L, so g''(t) ≤ L‖y-x‖²

     Integrating g''(t) twice:
     - g'(t) ≤ g'(0) + L·t·‖y-x‖²
     - g(t) ≤ g(0) + g'(0)·t + (L/2)·t²·‖y-x‖²

     At t = 1:
     - f(y) ≤ f(x) + ⟨∇f(x), y-x⟩ + (L/2)‖y-x‖²
  -/

  /- **Proof Strategy using Monotonicity (avoids MeasureTheory integration)**
     Define:
     - γ(t) = x + t • (y - x) for t ∈ [0, 1]
     - g(t) = f(γ(t)) - t * ⟨∇f(x), y - x⟩
     - K = L * ‖y - x‖²
     - h(t) = g(t) - (K/2) * t²
     Then:
     - g'(t) = ⟨∇f(γ(t)) - ∇f(x), y - x⟩ (after simplification)
     - g'(t) ≤ L * t * ‖y - x‖² = K * t (by Lipschitz + Cauchy-Schwarz)
     - h'(t) = g'(t) - K * t ≤ 0
     - By antitoneOn_of_deriv_nonpos: h(1) ≤ h(0)
     - Expanding: g(1) - K/2 ≤ g(0)
     - So: f(y) - ⟨∇f(x), y-x⟩ - (L/2)‖y-x‖² ≤ f(x)
     - Rearranging: f(y) ≤ f(x) + ⟨∇f(x), y-x⟩ + (L/2)‖y-x‖²
  -/
  -- Define the path γ(t) = x + t • (y - x)
  let γ := fun t : ℝ => x + t • (y - x)
  -- Define K = L * ‖y - x‖²
  let K := L * ‖y - x‖^2
  -- Define inner_val = ⟨∇f(x), y - x⟩
  let inner_val := @inner ℝ E _ (gradient f x) (y - x)
  -- Define g(t) = f(γ(t)) - t * inner_val : measures deviation from linear model
  let g := fun t : ℝ => f (γ t) - t * inner_val
  -- Define h(t) = g(t) - (K/2) * t² : we'll show h is antitone
  let h := fun t : ℝ => g t - (K / 2) * t^2
  -- Key boundary values
  have hγ0 : γ 0 = x := by simp only [γ, zero_smul, add_zero]
  have hγ1 : γ 1 = y := by simp only [γ, one_smul, add_sub_cancel]
  have hg0 : g 0 = f x := by simp only [g, hγ0, zero_mul, sub_zero]
  have hg1 : g 1 = f y - inner_val := by simp only [g, hγ1, one_mul]
  have hh0 : h 0 = f x := by simp only [h, hg0, sq, mul_zero, sub_zero]
  have hh1 : h 1 = f y - inner_val - K / 2 := by
    simp only [h, hg1, one_pow, mul_one]
  -- γ(t) - x = t • (y - x) for the Lipschitz bound
  have hγ_diff : ∀ t, γ t - x = t • (y - x) := by
    intro t; simp only [γ, add_sub_cancel_left]
  -- ‖γ(t) - x‖ = |t| * ‖y - x‖
  have hγ_norm : ∀ t, ‖γ t - x‖ = |t| * ‖y - x‖ := by
    intro t; rw [hγ_diff, norm_smul, Real.norm_eq_abs]
  -- For t ∈ [0, 1], |t| = t
  have h_abs_t : ∀ t : ℝ, 0 ≤ t → t ≤ 1 → |t| = t := fun t ht _ => abs_of_nonneg ht
  -- The key bound: ⟨∇f(γ(t)) - ∇f(x), y - x⟩ ≤ L * t * ‖y - x‖² for t ∈ [0, 1]
  -- This uses: Cauchy-Schwarz, then L-Lipschitz of gradient, then ‖γ(t) - x‖ = t * ‖y - x‖
  have h_grad_bound : ∀ t, 0 ≤ t → t ≤ 1 →
      @inner ℝ E _ (gradient f (γ t) - gradient f x) (y - x) ≤ L * t * ‖y - x‖^2 := by
    intro t ht0 ht1
    have hCS : @inner ℝ E _ (gradient f (γ t) - gradient f x) (y - x) ≤
        ‖gradient f (γ t) - gradient f x‖ * ‖y - x‖ := real_inner_le_norm _ _
    have hLip : ‖gradient f (γ t) - gradient f x‖ ≤ L * ‖γ t - x‖ := hLip (γ t) x
    have hNorm : ‖γ t - x‖ = t * ‖y - x‖ := by rw [hγ_norm, h_abs_t t ht0 ht1]
    calc @inner ℝ E _ (gradient f (γ t) - gradient f x) (y - x)
        ≤ ‖gradient f (γ t) - gradient f x‖ * ‖y - x‖ := hCS
      _ ≤ (L * ‖γ t - x‖) * ‖y - x‖ := by nlinarith [norm_nonneg (y - x)]
      _ = L * (t * ‖y - x‖) * ‖y - x‖ := by rw [hNorm]
      _ = L * t * ‖y - x‖^2 := by ring
  -- Step 1: h is continuous on [0, 1]
  -- γ is continuous
  have hγ_cont : Continuous γ := by
    simp only [γ]
    exact continuous_const.add (continuous_id.smul continuous_const)
  -- f ∘ γ is continuous
  have hfγ_cont : Continuous (f ∘ γ) := hDiff.continuous.comp hγ_cont
  -- g is continuous
  have hg_cont : Continuous g := by
    simp only [g]
    exact hfγ_cont.sub (continuous_id.mul continuous_const)
  -- h is continuous
  have hh_cont : Continuous h := by
    simp only [h]
    exact hg_cont.sub (continuous_const.mul (continuous_pow 2))
  have h_cont : ContinuousOn h (Set.Icc 0 1) := hh_cont.continuousOn
  -- Step 2: h is differentiable on interior (0, 1)
  -- The derivative of h at t is: ⟨∇f(γ(t)), y-x⟩ - inner_val - K*t
  --                            = ⟨∇f(γ(t)) - ∇f(x), y-x⟩ - K*t
  -- We use the chain rule: deriv (f ∘ γ) t = fderiv f (γ t) (deriv γ t)
  --                                        = ⟨∇f(γ(t)), y - x⟩
  -- Since γ(t) = x + t • (y - x), we have deriv γ t = y - x (constant)
  have h_deriv : ∀ t ∈ Set.Ioo (0 : ℝ) 1,
      HasDerivAt h (@inner ℝ E _ (gradient f (γ t) - gradient f x) (y - x) - K * t) t := by
    intro t _ht
    -- γ has derivative y - x
    have hγ_deriv : HasDerivAt γ (y - x) t := by
      have h1 : HasDerivAt (fun s : ℝ => x) 0 t := hasDerivAt_const t x
      have h2 : HasDerivAt (fun s : ℝ => s • (y - x)) ((1 : ℝ) • (y - x)) t := by
        exact (hasDerivAt_id t).smul_const (y - x)
      have h3 := h1.add h2
      simp only [zero_add, one_smul] at h3
      convert h3 using 1
    -- f ∘ γ has derivative ⟨∇f(γ(t)), y - x⟩
    have hfγ_deriv : HasDerivAt (f ∘ γ) (@inner ℝ E _ (gradient f (γ t)) (y - x)) t := by
      have hf_grad : HasGradientAt f (gradient f (γ t)) (γ t) := (hDiff (γ t)).hasGradientAt
      have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f (γ t))) (γ t) :=
        hf_grad.hasFDerivAt
      have := hf_fderiv.comp_hasDerivAt t hγ_deriv
      simp only [innerSL_apply_apply] at this
      exact this
    -- (t ↦ t * inner_val) has derivative inner_val
    have h_lin_deriv : HasDerivAt (fun s => s * inner_val) inner_val t := by
      have := (hasDerivAt_id t).mul_const inner_val
      simp only [one_mul] at this
      exact this
    -- g = (f ∘ γ) - (t ↦ t * inner_val) has derivative ⟨∇f(γ(t)), y-x⟩ - inner_val
    have hg_deriv : HasDerivAt g (@inner ℝ E _ (gradient f (γ t)) (y - x) - inner_val) t := by
      exact hfγ_deriv.sub h_lin_deriv
    -- Rewrite using inner_sub_left: ⟨a, v⟩ - ⟨b, v⟩ = ⟨a - b, v⟩
    have h_inner_eq : @inner ℝ E _ (gradient f (γ t)) (y - x) - inner_val =
        @inner ℝ E _ (gradient f (γ t) - gradient f x) (y - x) := by
      simp only [inner_val, inner_sub_left]
    rw [h_inner_eq] at hg_deriv
    -- (t ↦ (K/2) * t²) has derivative K * t
    have h_quad_deriv : HasDerivAt (fun s => (K / 2) * s^2) (K * t) t := by
      have h1 := hasDerivAt_pow 2 t
      have h2 := h1.const_mul (K / 2)
      simp only [Nat.cast_ofNat] at h2
      convert h2 using 1
      ring
    -- h = g - (t ↦ (K/2) * t²)
    exact hg_deriv.sub h_quad_deriv
  -- Step 3: deriv h t ≤ 0 on (0, 1)
  have h_deriv_nonpos : ∀ t ∈ Set.Ioo (0 : ℝ) 1, deriv h t ≤ 0 := by
    intro t ht
    have hd := h_deriv t ht
    rw [hd.deriv]
    have hbound := h_grad_bound t (le_of_lt ht.1) (le_of_lt ht.2)
    linarith
  -- Step 4: Apply antitone result
  -- interior of Icc 0 1 = Ioo 0 1
  have h_interior : interior (Set.Icc (0 : ℝ) 1) = Set.Ioo 0 1 := interior_Icc
  have h_diff_on : DifferentiableOn ℝ h (interior (Set.Icc (0 : ℝ) 1)) := by
    rw [h_interior]
    intro t ht
    exact (h_deriv t ht).differentiableAt.differentiableWithinAt
  have h_deriv_le : ∀ t ∈ interior (Set.Icc (0 : ℝ) 1), deriv h t ≤ 0 := by
    rw [h_interior]
    exact h_deriv_nonpos
  have h_mono := Convex.image_sub_le_mul_sub_of_deriv_le (convex_Icc (0 : ℝ) 1) h_cont h_diff_on
    h_deriv_le 0 (Set.left_mem_Icc.mpr zero_le_one) 1 (Set.right_mem_Icc.mpr zero_le_one)
    zero_le_one
  -- h(1) - h(0) ≤ 0 * (1 - 0) = 0
  simp only [zero_mul, sub_zero] at h_mono
  -- h(1) ≤ h(0) means f(y) - inner_val - K/2 ≤ f(x)
  rw [hh1, hh0] at h_mono
  -- Conclude: f(y) ≤ f(x) + inner_val + K/2
  simp only [inner_val, K] at h_mono
  linarith

/-- First-order optimality for convex functions: if ∇f(x*) = 0 and f is convex and differentiable,
    then x* is a global minimizer.

    This is proved using a derivative limit argument: define p(t) = f(x* + t(y - x*)).
    Convexity gives p(t) ≤ (1-t)p(0) + tp(1). At t = 0, p(0) = p(0), so equality holds.
    p'(0) = ⟨∇f(x*), y - x*⟩ = 0. For the convex bound p(t) ≤ (1-t)p(0) + tp(1) with
    derivative 0 at t = 0, we must have p(0) ≤ p(1), i.e., f(x*) ≤ f(y). -/
lemma convex_first_order_optimality (f : E → ℝ) (hConvex : ConvexOn ℝ Set.univ f)
    (hDiff : Differentiable ℝ f) (x_star : E) (hMin : gradient f x_star = 0) :
    ∀ y, f x_star ≤ f y := by
  -- First-order optimality: for convex differentiable f, ∇f(x*) = 0 implies x* is a global minimizer.
  -- The proof uses convexity along paths and derivative comparison at t = 0.
  intro y
  -- Define p(t) = f(x* + t(y - x*)) and q(t) = (1-t)f(x*) + tf(y)
  let e := y - x_star
  let p := fun t : ℝ => f (x_star + t • e)
  let q := fun t : ℝ => (1 - t) * f x_star + t * f y
  -- By convexity: p(t) ≤ q(t) for t ∈ [0, 1]
  have hpq_ineq : ∀ t, 0 ≤ t → t ≤ 1 → p t ≤ q t := by
    intro t ht0 ht1
    have h1mt : 0 ≤ 1 - t := by linarith
    have hsum : (1 - t) + t = 1 := by ring
    have hconv := hConvex.2 (Set.mem_univ x_star) (Set.mem_univ y) h1mt ht0 hsum
    have heq : (1 - t) • x_star + t • y = x_star + t • e := by
      simp only [e, smul_sub]
      ring_nf
      module
    simp only [p, q, heq, smul_eq_mul] at hconv ⊢
    linarith
  -- At t = 0: p(0) = q(0) = f(x*)
  have hp0 : p 0 = f x_star := by simp only [p, zero_smul, add_zero]
  have hq0 : q 0 = f x_star := by simp only [q]; ring
  -- p'(0) = ⟨∇f(x*), e⟩ = 0
  have p_deriv : HasDerivAt p 0 0 := by
    have hγ : HasDerivAt (fun t : ℝ => x_star + t • e) e 0 := by
      have h1 : HasDerivAt (fun _ : ℝ => x_star) 0 0 := hasDerivAt_const 0 x_star
      have h2 : HasDerivAt (fun t : ℝ => t • e) ((1 : ℝ) • e) 0 := (hasDerivAt_id 0).smul_const e
      have hsum := h1.add h2
      simp only [zero_add, one_smul] at hsum
      exact hsum
    have hf_grad : HasGradientAt f (gradient f x_star) x_star := (hDiff x_star).hasGradientAt
    have hf_fderiv : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) x_star := hf_grad.hasFDerivAt
    have hf_fderiv' : HasFDerivAt f (innerSL (𝕜 := ℝ) (gradient f x_star)) (x_star + (0 : ℝ) • e) := by
      simp only [zero_smul, add_zero]; exact hf_fderiv
    have hcomp := hf_fderiv'.comp_hasDerivAt (0 : ℝ) hγ
    simp only [Function.comp_apply, innerSL_apply_apply, zero_smul, add_zero, hMin, inner_zero_left] at hcomp
    exact hcomp
  -- q'(0) = f(y) - f(x*)
  have q_deriv : HasDerivAt q (f y - f x_star) 0 := by
    have h1 : HasDerivAt (fun t : ℝ => (1 - t) * f x_star) (-f x_star) 0 := by
      have hid : HasDerivAt (fun t : ℝ => 1 - t) (-1) 0 := by
        have := (hasDerivAt_const (0 : ℝ) (1 : ℝ)).sub (hasDerivAt_id (0 : ℝ))
        convert this using 1
        ring
      have := hid.mul_const (f x_star)
      convert this using 1
      ring
    have h2 : HasDerivAt (fun t : ℝ => t * f y) (f y) 0 := by
      have := (hasDerivAt_id (0 : ℝ)).mul_const (f y)
      convert this using 1
      ring
    have h3 := h1.add h2
    convert h3 using 1
    ring
  -- Proof by contradiction: assume f(x*) > f(y)
  by_contra hcontra
  push_neg at hcontra
  let δ := f x_star - f y
  have hδ_pos : δ > 0 := by simp only [δ]; linarith
  -- (p - q)'(0) = 0 - (f(y) - f(x*)) = δ > 0
  have h_pq_deriv : HasDerivAt (fun t => p t - q t) δ 0 := by
    have hsub := HasDerivAt.sub p_deriv q_deriv
    convert hsub using 1
    simp only [δ]
    ring
  have h_pq_0 : (fun t => p t - q t) 0 = 0 := by simp only [hp0, hq0, sub_self]
  -- Use isLittleO characterization of derivative
  rw [hasDerivAt_iff_isLittleO] at h_pq_deriv
  have hε_half : 0 < δ / 2 := by linarith
  have h_bound_evt := h_pq_deriv.def hε_half
  simp only [h_pq_0, sub_zero, smul_eq_mul] at h_bound_evt
  rw [Filter.eventually_iff_exists_mem] at h_bound_evt
  obtain ⟨s, hs_mem, hs_bound⟩ := h_bound_evt
  rw [Metric.mem_nhds_iff] at hs_mem
  obtain ⟨ε, hε_pos, hε_sub⟩ := hs_mem
  -- Choose t in (0, min(ε/2, 1/2)]
  let t := min (ε / 2) (1 / 2)
  have ht_pos : 0 < t := by positivity
  have ht_lt_ε : t < ε := by simp only [t]; linarith [min_le_left (ε / 2) (1 / 2)]
  have ht_le_1 : t ≤ 1 := by simp only [t]; linarith [min_le_right (ε / 2) (1 / 2)]
  have ht_in_ball : t ∈ Metric.ball 0 ε := by
    simp only [Metric.mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_pos ht_pos]
    exact ht_lt_ε
  have ht_in_s : t ∈ s := hε_sub ht_in_ball
  have h_bound := hs_bound t ht_in_s
  simp only [Real.norm_eq_abs, abs_of_pos ht_pos] at h_bound
  -- From derivative approximation: |(p(t) - q(t)) - t*δ| ≤ (δ/2)*t
  -- So p(t) - q(t) ≥ t*δ - (δ/2)*t = (δ/2)*t > 0
  have h_lower : p t - q t ≥ t * δ - (δ / 2) * t := by
    have h1 : -((δ / 2) * t) ≤ (p t - q t) - t * δ := by
      have := neg_abs_le (p t - q t - t * δ)
      linarith
    linarith
  have h_diff_pos : p t - q t > 0 := by
    have : t * δ - (δ / 2) * t = (δ / 2) * t := by ring
    rw [this] at h_lower
    have : (δ / 2) * t > 0 := mul_pos (by linarith) ht_pos
    linarith
  -- But convexity says p(t) ≤ q(t), contradiction
  have h_le := hpq_ineq t (le_of_lt ht_pos) ht_le_1
  linarith

/-- Co-coercivity of L-smooth gradients (Baillon-Haddad theorem).

    For L-smooth f: ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (1/L)‖∇f(x) - ∇f(y)‖²

    With y = x* where ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ (1/L)‖∇f(x)‖²

    Equivalently: ‖∇f(x)‖² ≤ L⟨∇f(x), x - x*⟩ -/
theorem lsmooth_cocoercivity (f : E → ℝ) (L : ℝ) (hL : 0 < L)
    (hSmooth : IsLSmooth f L) (hConvex : ConvexOn ℝ Set.univ f)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    ‖gradient f x‖^2 ≤ L * @inner ℝ E _ (gradient f x) (x - x_star) := by
  -- The proof uses the tilted function technique combining:
  -- 1. L-smooth descent: f(x - (1/L)g) ≤ f(x) - (1/2L)‖g‖²
  -- 2. First-order optimality for the tilted function h(z) = f(z) - ⟨g, z⟩
  -- Adding the two bounds gives (1/L)‖g‖² ≤ ⟨g, x - x*⟩
  have hDiff : Differentiable ℝ f := hSmooth.1
  let g := gradient f x

  -- Step 1: x* minimizes f (since ∇f(x*) = 0 and f is convex)
  have h_xstar_min : ∀ y, f x_star ≤ f y := convex_first_order_optimality f hConvex hDiff x_star hMin

  -- Step 2: Apply lsmooth_fundamental_ineq to get descent at x
  have h_fund_f := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth x (x - (1 / L) • g)
  have h_descent_f : f (x - (1 / L) • g) ≤ f x - (1 / (2 * L)) * ‖g‖^2 := by
    have h_diff : (x - (1 / L) • g) - x = -((1 / L) • g) := by simp [sub_eq_add_neg, add_comm]
    have h_inner : @inner ℝ E _ g ((x - (1 / L) • g) - x) = -(1 / L) * ‖g‖^2 := by
      rw [h_diff]
      simp only [inner_neg_right, inner_smul_right, real_inner_self_eq_norm_sq]
      ring
    have h_norm : ‖(x - (1 / L) • g) - x‖^2 = (1 / L)^2 * ‖g‖^2 := by
      rw [h_diff, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/L > 0)]
      ring
    calc f (x - (1 / L) • g) ≤ f x + @inner ℝ E _ g ((x - (1 / L) • g) - x) +
                                 (L / 2) * ‖(x - (1 / L) • g) - x‖^2 := h_fund_f
      _ = f x + (-(1 / L) * ‖g‖^2) + (L / 2) * ((1 / L)^2 * ‖g‖^2) := by rw [h_inner, h_norm]
      _ = f x - (1 / (2 * L)) * ‖g‖^2 := by field_simp; ring

  -- Bound A: (1/2L)‖g‖² ≤ f(x) - f(x*)
  have h_bound_A : (1 / (2 * L)) * ‖g‖^2 ≤ f x - f x_star := by
    have := h_xstar_min (x - (1 / L) • g)
    linarith

  -- Step 3: Apply fundamental ineq at x_star
  have h_fund_xstar := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth x_star (x_star + (1 / L) • g)
  have h_fund_xstar_bound : f (x_star + (1 / L) • g) ≤ f x_star + (1 / (2 * L)) * ‖g‖^2 := by
    have h_diff : (x_star + (1 / L) • g) - x_star = (1 / L) • g := by abel
    have h_inner : @inner ℝ E _ (gradient f x_star) ((x_star + (1 / L) • g) - x_star) = 0 := by
      rw [hMin, inner_zero_left]
    have h_norm : ‖(x_star + (1 / L) • g) - x_star‖^2 = (1 / L)^2 * ‖g‖^2 := by
      rw [h_diff, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/L > 0)]
      ring
    calc f (x_star + (1 / L) • g) ≤ f x_star + @inner ℝ E _ (gradient f x_star)
          ((x_star + (1 / L) • g) - x_star) + (L / 2) * ‖(x_star + (1 / L) • g) - x_star‖^2 := h_fund_xstar
      _ = f x_star + 0 + (L / 2) * ((1 / L)^2 * ‖g‖^2) := by rw [h_inner, h_norm]
      _ = f x_star + (1 / (2 * L)) * ‖g‖^2 := by field_simp; ring

  -- Step 4: Tilted function h(z) = f(z) - ⟨g, z⟩ is convex
  have h_convex : ConvexOn ℝ Set.univ (fun z => f z - @inner ℝ E _ g z) := by
    have h_linear_concave : ConcaveOn ℝ Set.univ (fun z => @inner ℝ E _ g z) := by
      constructor
      · exact convex_univ
      · intro z _ w _ a b ha hb hab
        simp only [inner_add_right, inner_smul_right, smul_eq_mul]
        linarith
    exact hConvex.sub h_linear_concave

  -- ∇h(x) = 0
  have h_grad_h_x : gradient (fun z => f z - @inner ℝ E _ g z) x = 0 := by
    have hf_diff : DifferentiableAt ℝ f x := hDiff x
    have hg_diff : DifferentiableAt ℝ (fun z => @inner ℝ E _ g z) x :=
      (innerSL (𝕜 := ℝ) g).differentiableAt
    -- The gradient of z ↦ ⟨g, z⟩ is g
    have hg_grad : HasGradientAt (fun z => @inner ℝ E _ g z) g x := by
      rw [hasGradientAt_iff_hasFDerivAt]
      have h1 := (innerSL (𝕜 := ℝ) g).hasFDerivAt (x := x)
      simp only [InnerProductSpace.toDual] at h1 ⊢
      convert h1 using 1
    -- HasGradientAt (f - inner g) (g - g) x
    have hf_grad : HasGradientAt f g x := hf_diff.hasGradientAt
    have h_sub : HasGradientAt (fun z => f z - @inner ℝ E _ g z) (g - g) x := by
      have h1 := hasGradientAt_iff_hasFDerivAt.mp hf_grad
      have h2 := hasGradientAt_iff_hasFDerivAt.mp hg_grad
      have h3 := h1.sub h2
      rw [hasGradientAt_iff_hasFDerivAt]
      convert h3 using 1
      simp only [map_sub]
    rw [sub_self] at h_sub
    exact h_sub.gradient

  -- h is differentiable
  have h_diff_h : Differentiable ℝ (fun z => f z - @inner ℝ E _ g z) := by
    intro z
    exact (hDiff z).sub (innerSL (𝕜 := ℝ) g).differentiableAt

  -- x minimizes h via first-order optimality
  have h_x_min_h : ∀ y, (f x - @inner ℝ E _ g x) ≤ (f y - @inner ℝ E _ g y) :=
    convex_first_order_optimality (fun z => f z - @inner ℝ E _ g z) h_convex h_diff_h x h_grad_h_x

  -- h(x) ≤ h(x_star + (1/L)g)
  have h_hx_le := h_x_min_h (x_star + (1 / L) • g)

  -- Expand ⟨g, x_star + (1/L)g⟩
  have h_inner_xstar'_g : @inner ℝ E _ g (x_star + (1 / L) • g) =
      @inner ℝ E _ g x_star + (1 / L) * ‖g‖^2 := by
    simp only [inner_add_right, inner_smul_right, real_inner_self_eq_norm_sq]

  -- Bound B: (1/2L)‖g‖² ≤ f(x_star) - f(x) + ⟨g, x - x_star⟩
  have h_bound_B : (1 / (2 * L)) * ‖g‖^2 ≤ f x_star - f x + @inner ℝ E _ g (x - x_star) := by
    -- From h(x) ≤ h(x_star + (1/L)g):
    -- f(x) - ⟨g, x⟩ ≤ f(x_star + (1/L)g) - ⟨g, x_star + (1/L)g⟩
    --              ≤ [f(x_star) + (1/2L)‖g‖²] - [⟨g, x_star⟩ + (1/L)‖g‖²]
    --              = f(x_star) - ⟨g, x_star⟩ - (1/2L)‖g‖²
    -- Rearranging: (1/2L)‖g‖² ≤ f(x_star) - f(x) + ⟨g, x⟩ - ⟨g, x_star⟩
    --                        = f(x_star) - f(x) + ⟨g, x - x_star⟩
    have h4 : @inner ℝ E _ g (x - x_star) = @inner ℝ E _ g x - @inner ℝ E _ g x_star :=
      inner_sub_right g x x_star
    -- Substitute step3 into step1
    have step1' : f x - @inner ℝ E _ g x ≤ f (x_star + (1 / L) • g) -
        (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) := by
      rw [← h_inner_xstar'_g]
      exact h_hx_le
    -- Combine with step2
    have step2' : f (x_star + (1 / L) • g) - (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) ≤
        f x_star + (1 / (2 * L)) * ‖g‖^2 - (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) := by
      linarith [h_fund_xstar_bound]
    -- Chain inequalities
    have step3' : f x - @inner ℝ E _ g x ≤
        f x_star - @inner ℝ E _ g x_star - (1 / (2 * L)) * ‖g‖^2 := by
      have := le_trans step1' step2'
      have eq1 : f x_star + (1 / (2 * L)) * ‖g‖^2 - (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) =
          f x_star - @inner ℝ E _ g x_star - (1 / (2 * L)) * ‖g‖^2 := by ring
      linarith
    -- Rearrange
    have step4 : (1 / (2 * L)) * ‖g‖^2 ≤ f x_star - f x + @inner ℝ E _ g x - @inner ℝ E _ g x_star := by
      linarith
    linarith

  -- Add bounds A and B: (1/L)‖g‖² ≤ ⟨g, x - x*⟩
  have h_combined : (1 / L) * ‖g‖^2 ≤ @inner ℝ E _ g (x - x_star) := by
    have h_add := add_le_add h_bound_A h_bound_B
    -- h_add: (1/(2L))‖g‖² + (1/(2L))‖g‖² ≤ (f x - f x_star) + (f x_star - f x + ⟨g, x - x_star⟩)
    -- LHS = (1/L)‖g‖², RHS = ⟨g, x - x_star⟩
    have lhs_eq : (1 / (2 * L)) * ‖g‖^2 + (1 / (2 * L)) * ‖g‖^2 = (1 / L) * ‖g‖^2 := by field_simp; ring
    have rhs_eq : (f x - f x_star) + (f x_star - f x + @inner ℝ E _ g (x - x_star)) =
        @inner ℝ E _ g (x - x_star) := by ring
    linarith

  -- Multiply by L
  calc ‖g‖^2 = L * ((1 / L) * ‖g‖^2) := by field_simp
    _ ≤ L * @inner ℝ E _ g (x - x_star) := by
        apply mul_le_mul_of_nonneg_left h_combined (le_of_lt hL)

/-- Interpolation condition for strongly convex AND smooth functions.

    For μ-strongly convex and L-smooth f with ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ (μL)/(μ+L) ‖x - x*‖² + 1/(μ+L) ‖∇f(x)‖²

    This is stronger than using strong convexity or smoothness alone.
    It's the key to achieving the optimal (1 - μ/L) contraction rate.

    **Proof**: Uses the auxiliary function h(x) = f(x) - (μ/2)‖x - x*‖² which is
    convex and (L-μ)-smooth. Applying cocoercivity to h gives the bound. -/
theorem strong_smooth_interpolation (f : E → ℝ) (L μ : ℝ) (hL : 0 < L) (hμ : 0 < μ)
    (hSmooth : IsLSmooth f L) (hStrong : IsStronglyConvex f μ)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    @inner ℝ E _ (gradient f x) (x - x_star) ≥
      (μ * L) / (μ + L) * ‖x - x_star‖^2 + 1 / (μ + L) * ‖gradient f x‖^2 := by
  have hDiff : Differentiable ℝ f := hSmooth.1

  -- Let g = ∇f(x) and d = x - x*
  let g := gradient f x
  let d := x - x_star

  -- Strong convexity implies convexity
  have hConvex : ConvexOn ℝ Set.univ f :=
    stronglyConvex_implies_convexOn f μ (le_of_lt hμ) hStrong

  -- From strong convexity: ⟨g, d⟩ ≥ μ‖d‖²
  have h_strong : @inner ℝ E _ g d ≥ μ * ‖d‖^2 := by
    have := strong_convex_gradient_monotonicity f μ hμ hStrong hDiff x x_star hMin
    convert this using 2 <;> rfl

  -- From cocoercivity: ‖g‖² ≤ L⟨g, d⟩
  have h_cocoer : ‖g‖^2 ≤ L * @inner ℝ E _ g d := by
    have := lsmooth_cocoercivity f L hL hSmooth hConvex x x_star hMin
    convert this using 2 <;> rfl

  have h_sum_pos : 0 < μ + L := by linarith

  -- The key bound comes from the auxiliary function h(x) = f(x) - (μ/2)‖x - x*‖²
  -- which is convex and (L-μ)-smooth.
  -- Cocoercivity for h gives: ‖∇h(x)‖² ≤ (L-μ)⟨∇h(x), x - x*⟩
  -- where ∇h(x) = ∇f(x) - μ(x - x*) = g - μd
  --
  -- Expanding: ‖g - μd‖² ≤ (L-μ)⟨g - μd, d⟩
  -- After algebra: ‖g‖² + μL‖d‖² ≤ (L+μ)⟨g, d⟩

  have h_aux_cocoer : ‖g - μ • d‖^2 ≤ (L - μ) * @inner ℝ E _ (g - μ • d) d := by
    -- First handle d = 0 case (when x = x*)
    by_cases hd_zero : d = 0
    · -- When d = 0, we have x = x*, so g = ∇f(x*) = 0 by hMin
      have hg_zero : g = 0 := by simp only [g, d] at hd_zero ⊢; simp [sub_eq_zero.mp hd_zero, hMin]
      simp only [hd_zero, hg_zero, smul_zero, sub_zero, norm_zero, sq, mul_zero,
                 inner_zero_right, mul_zero, le_refl]
    -- Now assume d ≠ 0
    -- Case split: L = μ vs L ≠ μ
    by_cases hLμ : L = μ
    · -- Case L = μ: RHS = 0, need LHS ≤ 0, which holds since LHS ≥ 0 means LHS = 0
      -- Actually we just need ‖g - μd‖² ≤ 0 which is true iff ‖g - μd‖² = 0
      -- But wait, we're showing ≤, and RHS = 0, so we need LHS ≤ 0
      -- Since LHS = ‖g - μd‖² ≥ 0 always, we need LHS = 0
      -- This is NOT generally true for L = μ case!
      -- Actually the RHS is 0, and LHS ≥ 0, so we need LHS ≤ 0
      -- The only way this works is if LHS = 0.
      -- For L = μ, the bound is tight only at the optimum.
      -- Let's reconsider: we can use that ‖g - μd‖² ≥ 0 and show RHS ≥ LHS
      -- When L = μ, RHS = 0, so we need ‖g - μd‖² ≤ 0, forcing equality.
      -- This requires g = μd, which is the gradient condition at optimum.
      -- But x is arbitrary! So this case needs the strong condition.
      -- Actually, for L = μ (condition number 1), f(x) = (μ/2)‖x - x*‖² + const
      -- So ∇f(x) = μ(x - x*) = μd, hence g = μd, hence g - μd = 0.
      have h_grad_eq : g = μ • d := by
        -- When L = μ, the gradient is forced to be exactly linear
        -- From L-smooth: ‖∇f(x) - ∇f(x*)‖ ≤ μ‖x - x*‖ (using hLμ : L = μ)
        -- From strong convexity gradient monotonicity: ⟨∇f(x) - ∇f(x*), x - x*⟩ ≥ μ‖x - x*‖²
        -- With ∇f(x*) = 0: ‖g‖ ≤ μ‖d‖ and ⟨g, d⟩ ≥ μ‖d‖²
        -- Cauchy-Schwarz: ⟨g, d⟩ ≤ ‖g‖ · ‖d‖ ≤ μ‖d‖²
        -- So ⟨g, d⟩ = μ‖d‖² and ‖g‖ = μ‖d‖, forcing g = μd
        have hg_bound : ‖g‖ ≤ μ * ‖d‖ := by
          have := hSmooth.2 x x_star
          simp only [g, d, hMin, sub_zero] at this
          rw [hLμ] at this
          exact this
        have h_inner_eq : @inner ℝ E _ g d = μ * ‖d‖^2 := by
          -- From h_strong: ⟨g, d⟩ ≥ μ‖d‖²
          -- From Cauchy-Schwarz and hg_bound: ⟨g, d⟩ ≤ ‖g‖·‖d‖ ≤ μ‖d‖²
          have h_upper : @inner ℝ E _ g d ≤ μ * ‖d‖^2 := by
            calc @inner ℝ E _ g d ≤ ‖g‖ * ‖d‖ := real_inner_le_norm g d
              _ ≤ μ * ‖d‖ * ‖d‖ := by apply mul_le_mul_of_nonneg_right hg_bound (norm_nonneg d)
              _ = μ * ‖d‖^2 := by ring
          linarith [h_strong]
        -- Equality in Cauchy-Schwarz means g and d are parallel: g = c • d for some c
        -- Combined with ⟨g, d⟩ = μ‖d‖², if d ≠ 0 then c = μ
        by_cases hd_case : d = 0
        · -- d = 0 contradicts hd_zero
          exact absurd hd_case hd_zero
        · -- d ≠ 0: prove g = μ • d
          have hd_pos' : ‖d‖ > 0 := norm_pos_iff.mpr hd_case
          -- From equality in Cauchy-Schwarz: g = (⟨g,d⟩/‖d‖²) • d
          have h_cs_eq : @inner ℝ E _ g d = ‖g‖ * ‖d‖ := by
            -- ⟨g, d⟩ = μ‖d‖² and ‖g‖ ≤ μ‖d‖
            -- Cauchy-Schwarz: ⟨g, d⟩ ≤ ‖g‖ · ‖d‖
            -- If ⟨g, d⟩ < ‖g‖ · ‖d‖ then ⟨g, d⟩ < μ‖d‖ · ‖d‖ = μ‖d‖²
            -- But ⟨g, d⟩ = μ‖d‖², contradiction
            have h1 : @inner ℝ E _ g d ≤ ‖g‖ * ‖d‖ := real_inner_le_norm g d
            have h2 : ‖g‖ * ‖d‖ ≤ μ * ‖d‖ * ‖d‖ := by
              apply mul_le_mul_of_nonneg_right hg_bound (norm_nonneg d)
            have h3 : μ * ‖d‖ * ‖d‖ = μ * ‖d‖^2 := by ring
            rw [h_inner_eq]
            by_contra h_ne
            have h_lt : μ * ‖d‖^2 < ‖g‖ * ‖d‖ := by
              push_neg at h_ne
              rcases (ne_iff_lt_or_gt.mp h_ne) with h_lt | h_gt
              · exact h_lt
              · linarith [h1]
            have : μ * ‖d‖^2 < μ * ‖d‖^2 := by
              calc μ * ‖d‖^2 < ‖g‖ * ‖d‖ := h_lt
                _ ≤ μ * ‖d‖ * ‖d‖ := h2
                _ = μ * ‖d‖^2 := h3
            linarith
          -- Now g is parallel to d with positive coefficient
          have h_norm_eq : ‖g‖ = μ * ‖d‖ := by
            have h1 : @inner ℝ E _ g d = ‖g‖ * ‖d‖ := h_cs_eq
            rw [h_inner_eq] at h1
            have h2 : μ * ‖d‖^2 = ‖g‖ * ‖d‖ := h1
            have h3 : μ * ‖d‖^2 / ‖d‖ = ‖g‖ * ‖d‖ / ‖d‖ := by rw [h2]
            simp only [sq, mul_div_assoc, div_self (ne_of_gt hd_pos'), mul_one] at h3
            linarith
          -- g and μ•d have same norm and same inner product with d
          -- This means g = μ•d (parallel with same magnitude and direction)
          have h_same_inner : @inner ℝ E _ g d = @inner ℝ E _ (μ • d) d := by
            rw [h_inner_eq, inner_smul_left, real_inner_self_eq_norm_sq]
            simp only [conj_trivial]
          have h_same_norm : ‖g‖ = ‖μ • d‖ := by
            rw [h_norm_eq, norm_smul, Real.norm_eq_abs, abs_of_pos hμ]
          -- The difference g - μ•d has norm 0
          have h_diff_zero : ‖g - μ • d‖ = 0 := by
            have h1 : ‖g - μ • d‖^2 = ‖g‖^2 - 2 * @inner ℝ E _ g (μ • d) + ‖μ • d‖^2 := by
              rw [sub_eq_add_neg, norm_add_sq_real]
              simp only [norm_neg, inner_neg_right]
              ring
            have h2 : @inner ℝ E _ g (μ • d) = μ * @inner ℝ E _ g d := by
              simp only [inner_smul_right, conj_trivial]
            rw [h2, h_inner_eq, h_same_norm] at h1
            simp only [norm_smul, Real.norm_eq_abs, abs_of_pos hμ] at h1
            have h3 : (μ * ‖d‖)^2 - 2 * (μ * (μ * ‖d‖^2)) + (μ * ‖d‖)^2 = 0 := by ring
            have h4 : ‖g - μ • d‖^2 = 0 := by linarith [h1, h3]
            have h5 : ‖g - μ • d‖^2 = ‖g - μ • d‖ * ‖g - μ • d‖ := sq _
            rw [h5] at h4
            exact mul_self_eq_zero.mp h4
          exact sub_eq_zero.mp (norm_eq_zero.mp h_diff_zero)
      rw [h_grad_eq, sub_self]
      simp only [norm_zero, sq, mul_zero, inner_zero_left, mul_zero, le_refl]
    · -- Case L ≠ μ: Need to show L > μ first
      have hL_ge_μ : L ≥ μ := by
        -- Use L-smooth upper bound and μ-strong convex lower bound at x*
        -- Upper: f(y) ≤ f(x*) + ⟨∇f(x*), y-x*⟩ + (L/2)‖y-x*‖² = f(x*) + (L/2)‖y-x*‖²
        -- Lower (from x* minimality): f(y) ≥ f(x*)
        -- Combined: 0 ≤ (L/2)‖y-x*‖² for all y, which is always true.
        -- We need a tighter lower bound from strong convexity.
        --
        -- Actually, use the gradient bounds we already have:
        -- h_cocoer: ‖g‖² ≤ L⟨g, d⟩  and  h_strong: ⟨g, d⟩ ≥ μ‖d‖²
        -- If d ≠ 0 and g ≠ 0:
        --   From h_strong and Cauchy-Schwarz: μ‖d‖² ≤ ⟨g,d⟩ ≤ ‖g‖·‖d‖, so μ‖d‖ ≤ ‖g‖
        --   From h_cocoer and Cauchy-Schwarz: ‖g‖² ≤ L⟨g,d⟩ ≤ L‖g‖·‖d‖, so ‖g‖ ≤ L‖d‖
        --   Combined: μ‖d‖ ≤ ‖g‖ ≤ L‖d‖, hence μ ≤ L
        -- We know d ≠ 0 from hd_zero above
        have hd : d ≠ 0 := hd_zero
        have hd_pos : ‖d‖ > 0 := norm_pos_iff.mpr hd
        -- From h_strong: μ‖d‖² ≤ ⟨g, d⟩
        -- From Cauchy-Schwarz: ⟨g, d⟩ ≤ ‖g‖ · ‖d‖
        have h1 : μ * ‖d‖^2 ≤ ‖g‖ * ‖d‖ := by
          calc μ * ‖d‖^2 ≤ @inner ℝ E _ g d := h_strong
            _ ≤ ‖g‖ * ‖d‖ := real_inner_le_norm g d
        -- Divide by ‖d‖ > 0: μ * ‖d‖ ≤ ‖g‖
        have h2 : μ * ‖d‖ ≤ ‖g‖ := by
          have hd_ne : ‖d‖ ≠ 0 := ne_of_gt hd_pos
          have h1' : μ * ‖d‖^2 / ‖d‖ ≤ ‖g‖ * ‖d‖ / ‖d‖ :=
            div_le_div_of_nonneg_right h1 (le_of_lt hd_pos)
          simp only [sq, mul_div_assoc, div_self hd_ne, mul_one] at h1'
          linarith
        -- From h_cocoer: ‖g‖² ≤ L⟨g, d⟩ ≤ L‖g‖·‖d‖
        by_cases hg : g = 0
        · -- If g = 0, then from h_strong: 0 ≥ μ‖d‖² > 0, contradiction with d ≠ 0
          simp only [hg, inner_zero_left] at h_strong
          have : μ * ‖d‖^2 > 0 := mul_pos hμ (sq_pos_of_pos hd_pos)
          linarith
        · -- g ≠ 0
          have hg_pos : ‖g‖ > 0 := norm_pos_iff.mpr hg
          have hg_ne : ‖g‖ ≠ 0 := ne_of_gt hg_pos
          have h3 : ‖g‖^2 ≤ L * (‖g‖ * ‖d‖) := by
            calc ‖g‖^2 ≤ L * @inner ℝ E _ g d := h_cocoer
              _ ≤ L * (‖g‖ * ‖d‖) := by
                  apply mul_le_mul_of_nonneg_left (real_inner_le_norm g d) (le_of_lt hL)
          -- ‖g‖² ≤ L·‖g‖·‖d‖, divide by ‖g‖ > 0
          have h4 : ‖g‖ ≤ L * ‖d‖ := by
            -- ‖g‖² ≤ L * ‖g‖ * ‖d‖, so ‖g‖ ≤ L * ‖d‖ (dividing by ‖g‖ > 0)
            have h3' : ‖g‖ * ‖g‖ ≤ L * (‖g‖ * ‖d‖) := by
              simp only [sq] at h3; exact h3
            have key : ‖g‖ ≤ L * ‖d‖ := by
              by_contra h_neg
              push_neg at h_neg
              -- h_neg : L * ‖d‖ < ‖g‖
              -- Then L * (‖g‖ * ‖d‖) < ‖g‖ * ‖g‖, contradicting h3'
              have : L * (‖g‖ * ‖d‖) = ‖g‖ * (L * ‖d‖) := by ring
              rw [this] at h3'
              have h4' : ‖g‖ * (L * ‖d‖) < ‖g‖ * ‖g‖ := by
                apply mul_lt_mul_of_pos_left h_neg hg_pos
              linarith
            exact key
          -- Combine: μ·‖d‖ ≤ ‖g‖ ≤ L·‖d‖, so μ ≤ L
          have h5 : μ * ‖d‖ ≤ L * ‖d‖ := le_trans h2 h4
          have h6 : (μ * ‖d‖) / ‖d‖ ≤ (L * ‖d‖) / ‖d‖ :=
            div_le_div_of_nonneg_right h5 (le_of_lt hd_pos)
          simp only [mul_div_assoc, div_self (ne_of_gt hd_pos), mul_one] at h6
          linarith
      have hL_gt_μ : L > μ := lt_of_le_of_ne hL_ge_μ (Ne.symm hLμ)

      -- Define auxiliary function h(z) = f(z) - (μ/2)‖z - x*‖²
      let h := fun z : E => f z - (μ / 2) * ‖z - x_star‖^2

      -- The key insight: for the auxiliary function h:
      -- 1. ∇h(z) = ∇f(z) - μ(z - x*), so ∇h(x) = g - μd
      -- 2. ∇h(x*) = ∇f(x*) - 0 = 0
      -- 3. h is convex (from μ-strong convexity of f)
      -- 4. h is (L-μ)-smooth (from L-smoothness of f)
      --
      -- Applying lsmooth_cocoercivity to h gives:
      -- ‖∇h(x)‖² ≤ (L-μ)⟨∇h(x), x - x*⟩
      -- which is: ‖g - μd‖² ≤ (L-μ)⟨g - μd, d⟩

      -- h is differentiable
      have h_diff : Differentiable ℝ h := by
        intro z
        apply DifferentiableAt.sub (hDiff z)
        apply DifferentiableAt.const_mul
        have h1 : DifferentiableAt ℝ (fun w => w - x_star) z :=
          differentiableAt_id.sub (differentiableAt_const x_star)
        exact h1.norm_sq (𝕜 := ℝ)

      -- ∇h(z) = ∇f(z) - μ(z - x*)
      have h_grad : ∀ z, gradient h z = gradient f z - μ • (z - x_star) := by
        intro z
        -- Differentiability facts
        have h_shift_diff : DifferentiableAt ℝ (fun w => w - x_star) z :=
          differentiableAt_id.sub (differentiableAt_const _)
        have h_norm_sq_diff : DifferentiableAt ℝ (fun w => ‖w - x_star‖^2) z :=
          h_shift_diff.norm_sq (𝕜 := ℝ)
        have h_scaled_diff : DifferentiableAt ℝ (fun w => (μ / 2) * ‖w - x_star‖^2) z :=
          (differentiableAt_const _).mul h_norm_sq_diff
        -- Key fact: gradient of (μ/2)‖w - x*‖² is μ(w - x*)
        -- fderiv ℝ (fun w => ‖w - x*‖²) z = 2 • innerSL ℝ (z - x*)
        -- So gradient = (toDual ℝ E).symm(2 • innerSL ℝ (z - x*)) = 2(z - x*)
        -- And gradient of (μ/2)‖w - x*‖² = (μ/2) * 2(z - x*) = μ(z - x*)
        have h_grad_norm_sq : gradient (fun w => ‖w - x_star‖^2) z = (2 : ℝ) • (z - x_star) := by
          simp only [gradient]
          have hfd : HasFDerivAt (fun w : E => w - x_star) (ContinuousLinearMap.id ℝ E) z := by
            have hsub := (hasFDerivAt_id (𝕜 := ℝ) z).sub (hasFDerivAt_const (𝕜 := ℝ) x_star z)
            simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.id_apply,
                       ContinuousLinearMap.zero_apply, sub_zero] at hsub
            exact hsub
          have h_comp : (innerSL ℝ (z - x_star : E)).comp (ContinuousLinearMap.id ℝ E) =
              innerSL ℝ (z - x_star) := by ext; simp
          have hfd_norm : HasFDerivAt (fun w => ‖w - x_star‖^2) (2 • innerSL ℝ (z - x_star)) z := by
            have := hfd.norm_sq
            simp only [h_comp] at this
            exact this
          rw [hfd_norm.fderiv]
          -- (toDual ℝ E).symm (2 • innerSL ℝ (z - x*)) = 2 • (z - x*)
          -- Convert ℕ-smul to ℝ-smul
          rw [← Nat.cast_smul_eq_nsmul ℝ (2 : ℕ) (innerSL ℝ (z - x_star))]
          rw [LinearIsometryEquiv.map_smul]
          congr 1
          -- innerSL ℝ v = toDual ℝ E v, so symm gives v back
          have : innerSL ℝ (z - x_star) = InnerProductSpace.toDual ℝ E (z - x_star) := rfl
          rw [this, LinearIsometryEquiv.symm_apply_apply]
        -- First prove HasGradientAt for the scaled norm squared term
        have h_grad_scaled_at : HasGradientAt (fun w => (μ / 2) * ‖w - x_star‖^2) (μ • (z - x_star)) z := by
          -- Build from HasFDerivAt
          have hfd : HasFDerivAt (fun w : E => w - x_star) (ContinuousLinearMap.id ℝ E) z := by
            have hsub := (hasFDerivAt_id (𝕜 := ℝ) z).sub (hasFDerivAt_const (𝕜 := ℝ) x_star z)
            simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.id_apply,
                       ContinuousLinearMap.zero_apply, sub_zero] at hsub
            exact hsub
          have h_comp : (innerSL ℝ (z - x_star : E)).comp (ContinuousLinearMap.id ℝ E) =
              innerSL ℝ (z - x_star) := by ext; simp
          have hfd_norm : HasFDerivAt (fun w => ‖w - x_star‖^2) (2 • innerSL ℝ (z - x_star)) z := by
            have := hfd.norm_sq
            simp only [h_comp] at this
            exact this
          -- (μ/2) * ‖w - x*‖² has derivative (μ/2) • (2 • innerSL ℝ (z - x*)) = μ • innerSL ℝ (z - x*)
          have hfd_scaled : HasFDerivAt (fun w => (μ / 2) * ‖w - x_star‖^2)
              ((μ / 2) • (2 • innerSL ℝ (z - x_star))) z := by
            have hconst : HasFDerivAt (fun _ : E => μ / 2) 0 z := hasFDerivAt_const (𝕜 := ℝ) _ _
            have hmul := hconst.mul hfd_norm
            -- hmul has type: HasFDerivAt ((fun x ↦ μ/2) * (fun w ↦ ‖w - x*‖²)) (... + ‖z-x*‖² • 0) z
            -- Simplify: ‖z-x*‖² • 0 = 0, and (fun x ↦ c) * g = fun x ↦ c * g x
            simp only [smul_zero, add_zero] at hmul
            convert hmul using 2
          -- Simplify (μ/2) • (2 • innerSL) = μ • innerSL
          -- Note: 2 • is nsmul (ℕ), so first convert to ℝ-smul
          have h_smul_simp : (μ / 2) • (2 • innerSL ℝ (z - x_star)) = μ • innerSL ℝ (z - x_star) := by
            rw [← Nat.cast_smul_eq_nsmul ℝ (2 : ℕ) (innerSL ℝ (z - x_star))]
            rw [smul_smul]; congr 1; ring
          rw [h_smul_simp] at hfd_scaled
          -- Convert to HasGradientAt: need to show (toDual ℝ E).symm (μ • innerSL ℝ (z - x*)) = μ • (z - x*)
          rw [hasFDerivAt_iff_hasGradientAt] at hfd_scaled
          convert hfd_scaled using 1
          rw [LinearIsometryEquiv.map_smul]
          congr 1
          have : innerSL ℝ (z - x_star) = InnerProductSpace.toDual ℝ E (z - x_star) := rfl
          rw [this, LinearIsometryEquiv.symm_apply_apply]
        have h_grad_scaled : gradient (fun w => (μ / 2) * ‖w - x_star‖^2) z = μ • (z - x_star) :=
          h_grad_scaled_at.gradient
        -- Now combine: h = f - (μ/2)‖·-x*‖², so gradient h = gradient f - gradient((μ/2)‖·-x*‖²)
        have h_grad_h_at : HasGradientAt h (gradient f z - μ • (z - x_star)) z := by
          have hf_at := (hDiff z).hasGradientAt
          -- Use HasFDerivAt.sub then convert back to HasGradientAt
          have h_fderiv_f := hf_at.hasFDerivAt
          have h_fderiv_scaled := h_grad_scaled_at.hasFDerivAt
          have h_fderiv_sub := h_fderiv_f.sub h_fderiv_scaled
          -- h_fderiv_sub : HasFDerivAt (f - (μ/2)*‖·-x*‖²) (toDual(∇f z) - toDual(μ(z-x*))) z
          rw [hasFDerivAt_iff_hasGradientAt] at h_fderiv_sub
          convert h_fderiv_sub using 1
          rw [LinearIsometryEquiv.map_sub, LinearIsometryEquiv.symm_apply_apply,
              LinearIsometryEquiv.symm_apply_apply]
        simp only [h]
        exact h_grad_h_at.gradient

      -- ∇h(x*) = 0
      have h_grad_xstar : gradient h x_star = 0 := by
        rw [h_grad x_star]
        simp only [sub_self, smul_zero, sub_zero, hMin]

      -- h is convex (from strong convexity of f)
      -- The proof uses the identity:
      -- a‖z-x*‖² + (1-a)‖w-x*‖² = ‖az+(1-a)w-x*‖² + a(1-a)‖z-w‖²
      have h_convex : ConvexOn ℝ Set.univ h := by
        constructor
        · exact convex_univ
        · intro z _ w _ a b ha hb hab
          simp only [h, smul_eq_mul]
          have hb_eq : b = 1 - a := by linarith
          rw [hb_eq]
          -- The convex combination identity for norms
          have h_convex_identity : a * ‖z - x_star‖^2 + (1 - a) * ‖w - x_star‖^2 =
              ‖a • z + (1 - a) • w - x_star‖^2 + a * (1 - a) * ‖z - w‖^2 := by
            -- Let u = z - x*, v = w - x*
            set u := z - x_star
            set v := w - x_star
            have hsum : a • z + (1 - a) • w - x_star = a • u + (1 - a) • v := by
              have : a • x_star + (1 - a) • x_star = x_star := by
                rw [← add_smul]; simp only [add_sub_cancel, one_smul]
              calc a • z + (1 - a) • w - x_star
                  = a • z - a • x_star + ((1 - a) • w - (1 - a) • x_star) := by module
                _ = a • (z - x_star) + (1 - a) • (w - x_star) := by simp only [smul_sub]
                _ = a • u + (1 - a) • v := by simp only [u, v]
            have hdiff : z - w = u - v := by simp only [u, v]; module
            rw [hsum, hdiff]
            have expand_lhs : ‖a • u + (1 - a) • v‖^2 =
                a^2 * ‖u‖^2 + (1 - a)^2 * ‖v‖^2 + 2 * a * (1 - a) * @inner ℝ E _ u v := by
              rw [norm_add_sq_real, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
                  abs_of_nonneg ha, abs_of_nonneg (by linarith : 0 ≤ 1 - a),
                  inner_smul_left, inner_smul_right]
              simp only [conj_trivial]
              ring
            have expand_diff : ‖u - v‖^2 = ‖u‖^2 + ‖v‖^2 - 2 * @inner ℝ E _ u v := by
              rw [norm_sub_sq_real]; ring
            rw [expand_lhs, expand_diff]
            ring
          have h_from_sc := hStrong z w a ha (by linarith : a ≤ 1)
          -- The strong convexity gives:
          -- f(az+(1-a)w) ≤ a*f(z) + (1-a)*f(w) - (μ/2)*a*(1-a)*‖z-w‖²
          -- Combined with h_convex_identity, this implies convexity of h
          -- Expand and simplify using h_convex_identity and h_from_sc
          have h_expand : a * (f z - μ / 2 * ‖z - x_star‖^2) + (1 - a) * (f w - μ / 2 * ‖w - x_star‖^2) =
              a * f z + (1 - a) * f w - μ / 2 * (a * ‖z - x_star‖^2 + (1 - a) * ‖w - x_star‖^2) := by
            ring
          have h_rhs : f (a • z + (1 - a) • w) - μ / 2 * ‖a • z + (1 - a) • w - x_star‖^2 =
              f (a • z + (1 - a) • w) - μ / 2 * (a * ‖z - x_star‖^2 + (1 - a) * ‖w - x_star‖^2)
              + μ / 2 * a * (1 - a) * ‖z - w‖^2 := by
            rw [h_convex_identity]; ring
          rw [h_expand, h_rhs]
          have h_ineq : f (a • z + (1 - a) • w) + μ / 2 * a * (1 - a) * ‖z - w‖^2 ≤
              a * f z + (1 - a) * f w := by linarith
          linarith

      -- Prove cocoercivity for h directly, following the technique from lsmooth_cocoercivity
      -- The key steps are:
      -- 1. x* minimizes h (since ∇h(x*) = 0 and h is convex)
      -- 2. Use descent lemma at x and x* to bound function differences
      -- 3. Use tilted function technique to relate to inner product

      have hL_sub_μ_pos : 0 < L - μ := sub_pos.mpr hL_gt_μ
      let g' := gradient h x

      -- Step 1: x* minimizes h (since ∇h(x*) = 0 and h is convex)
      have h_xstar_min : ∀ y, h x_star ≤ h y :=
        convex_first_order_optimality h h_convex h_diff x_star h_grad_xstar

      -- h satisfies the descent lemma (fundamental inequality) with constant (L-μ)
      have h_descent : ∀ u v, h v ≤ h u + @inner ℝ E _ (gradient h u) (v - u) +
          ((L - μ) / 2) * ‖v - u‖^2 := by
        intro u v
        have hf_desc := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth u v
        -- Expand: h(v) = f(v) - (μ/2)‖v - x*‖²
        -- Need to show: h(v) ≤ h(u) + ⟨∇h(u), v-u⟩ + ((L-μ)/2)‖v-u‖²
        -- where ∇h(u) = ∇f(u) - μ(u - x*)

        -- Key identity: ‖v - x*‖² = ‖(v-u) + (u-x*)‖² = ‖v-u‖² + ‖u-x*‖² + 2⟨u-x*, v-u⟩
        have h_norm_expand : ‖v - x_star‖^2 =
            ‖v - u‖^2 + ‖u - x_star‖^2 + 2 * @inner ℝ E _ (u - x_star) (v - u) := by
          have hvu : v - x_star = (v - u) + (u - x_star) := by abel
          rw [hvu, norm_add_sq_real]
          -- norm_add_sq_real gives ⟨v-u, u-x*⟩, need to swap to ⟨u-x*, v-u⟩
          rw [real_inner_comm (v - u) (u - x_star)]
          ring

        -- Expand inner product: ⟨∇f(u) - μ(u-x*), v-u⟩ = ⟨∇f(u), v-u⟩ - μ⟨u-x*, v-u⟩
        have h_inner_expand : @inner ℝ E _ (gradient f u - μ • (u - x_star)) (v - u) =
            @inner ℝ E _ (gradient f u) (v - u) - μ * @inner ℝ E _ (u - x_star) (v - u) := by
          rw [inner_sub_left, inner_smul_left]
          simp only [conj_trivial]

        -- Target: h(v) ≤ h(u) + ⟨∇h(u), v-u⟩ + ((L-μ)/2)‖v-u‖²
        -- h(v) = f(v) - (μ/2)‖v-x*‖²
        -- h(u) = f(u) - (μ/2)‖u-x*‖²
        -- ∇h(u) = ∇f(u) - μ(u-x*)
        simp only [h]
        rw [h_grad u, h_inner_expand, h_norm_expand]
        -- Now: f(v) - (μ/2)(‖v-u‖² + ‖u-x*‖² + 2⟨u-x*,v-u⟩)
        --    ≤ f(u) - (μ/2)‖u-x*‖² + (⟨∇f(u),v-u⟩ - μ⟨u-x*,v-u⟩) + ((L-μ)/2)‖v-u‖²
        -- Rearranging: f(v) ≤ f(u) + ⟨∇f(u),v-u⟩ + ((L-μ)/2 + μ/2)‖v-u‖² = f(u) + ⟨∇f(u),v-u⟩ + (L/2)‖v-u‖²
        -- Which follows from hf_desc
        linarith

      -- Step 2: Apply descent at x: h(x - (1/(L-μ))g') ≤ h(x) - (1/(2(L-μ)))‖g'‖²
      have h_descent_x : h (x - (1 / (L - μ)) • g') ≤ h x - (1 / (2 * (L - μ))) * ‖g'‖^2 := by
        have hd := h_descent x (x - (1 / (L - μ)) • g')
        have h_diff_eq : (x - (1 / (L - μ)) • g') - x = -((1 / (L - μ)) • g') := by simp [sub_eq_add_neg, add_comm]
        have h_inner : @inner ℝ E _ g' ((x - (1 / (L - μ)) • g') - x) = -(1 / (L - μ)) * ‖g'‖^2 := by
          rw [h_diff_eq]
          simp only [inner_neg_right, inner_smul_right, real_inner_self_eq_norm_sq]
          ring
        have h_norm : ‖(x - (1 / (L - μ)) • g') - x‖^2 = (1 / (L - μ))^2 * ‖g'‖^2 := by
          rw [h_diff_eq, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/(L-μ) > 0)]
          ring
        calc h (x - (1 / (L - μ)) • g') ≤ h x + @inner ℝ E _ g' ((x - (1 / (L - μ)) • g') - x) +
                             ((L - μ) / 2) * ‖(x - (1 / (L - μ)) • g') - x‖^2 := hd
          _ = h x + (-(1 / (L - μ)) * ‖g'‖^2) + ((L - μ) / 2) * ((1 / (L - μ))^2 * ‖g'‖^2) := by
              rw [h_inner, h_norm]
          _ = h x - (1 / (2 * (L - μ))) * ‖g'‖^2 := by field_simp; ring

      -- Bound A: (1/(2(L-μ)))‖g'‖² ≤ h(x) - h(x*)
      have h_bound_A : (1 / (2 * (L - μ))) * ‖g'‖^2 ≤ h x - h x_star := by
        have := h_xstar_min (x - (1 / (L - μ)) • g')
        linarith

      -- Step 3: Apply descent at x*: h(x* + (1/(L-μ))g') ≤ h(x*) + (1/(2(L-μ)))‖g'‖²
      have h_descent_xstar : h (x_star + (1 / (L - μ)) • g') ≤ h x_star + (1 / (2 * (L - μ))) * ‖g'‖^2 := by
        have hd := h_descent x_star (x_star + (1 / (L - μ)) • g')
        have h_diff_eq : (x_star + (1 / (L - μ)) • g') - x_star = (1 / (L - μ)) • g' := by abel
        have h_inner : @inner ℝ E _ (gradient h x_star) ((x_star + (1 / (L - μ)) • g') - x_star) = 0 := by
          rw [h_grad_xstar, inner_zero_left]
        have h_norm : ‖(x_star + (1 / (L - μ)) • g') - x_star‖^2 = (1 / (L - μ))^2 * ‖g'‖^2 := by
          rw [h_diff_eq, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/(L-μ) > 0)]
          ring
        calc h (x_star + (1 / (L - μ)) • g') ≤ h x_star + @inner ℝ E _ (gradient h x_star)
              ((x_star + (1 / (L - μ)) • g') - x_star) + ((L - μ) / 2) * ‖(x_star + (1 / (L - μ)) • g') - x_star‖^2 := hd
          _ = h x_star + 0 + ((L - μ) / 2) * ((1 / (L - μ))^2 * ‖g'‖^2) := by rw [h_inner, h_norm]
          _ = h x_star + (1 / (2 * (L - μ))) * ‖g'‖^2 := by field_simp; ring

      -- Step 4: Tilted function φ(z) = h(z) - ⟨g', z⟩ is convex
      have φ_convex : ConvexOn ℝ Set.univ (fun z => h z - @inner ℝ E _ g' z) := by
        have h_linear_concave : ConcaveOn ℝ Set.univ (fun z => @inner ℝ E _ g' z) := by
          constructor
          · exact convex_univ
          · intro z _ w _ a b ha hb hab
            simp only [inner_add_right, inner_smul_right, smul_eq_mul]
            linarith
        exact h_convex.sub h_linear_concave

      -- ∇φ(x) = 0
      have φ_grad_x : gradient (fun z => h z - @inner ℝ E _ g' z) x = 0 := by
        have hh_diff : DifferentiableAt ℝ h x := h_diff x
        have hg'_diff : DifferentiableAt ℝ (fun z => @inner ℝ E _ g' z) x :=
          (innerSL (𝕜 := ℝ) g').differentiableAt
        have hg'_grad : HasGradientAt (fun z => @inner ℝ E _ g' z) g' x := by
          rw [hasGradientAt_iff_hasFDerivAt]
          have h1 := (innerSL (𝕜 := ℝ) g').hasFDerivAt (x := x)
          simp only [InnerProductSpace.toDual] at h1 ⊢
          convert h1 using 1
        have hh_grad : HasGradientAt h g' x := hh_diff.hasGradientAt
        have h_sub : HasGradientAt (fun z => h z - @inner ℝ E _ g' z) (g' - g') x := by
          have h1 := hasGradientAt_iff_hasFDerivAt.mp hh_grad
          have h2 := hasGradientAt_iff_hasFDerivAt.mp hg'_grad
          have h3 := h1.sub h2
          rw [hasGradientAt_iff_hasFDerivAt]
          convert h3 using 1
          simp only [map_sub]
        rw [sub_self] at h_sub
        exact h_sub.gradient

      -- φ is differentiable
      have φ_diff : Differentiable ℝ (fun z => h z - @inner ℝ E _ g' z) := by
        intro z
        exact (h_diff z).sub (innerSL (𝕜 := ℝ) g').differentiableAt

      -- x minimizes φ via first-order optimality
      have h_x_min_φ : ∀ y, (h x - @inner ℝ E _ g' x) ≤ (h y - @inner ℝ E _ g' y) :=
        convex_first_order_optimality (fun z => h z - @inner ℝ E _ g' z) φ_convex φ_diff x φ_grad_x

      -- φ(x) ≤ φ(x* + (1/(L-μ))g')
      have h_φx_le := h_x_min_φ (x_star + (1 / (L - μ)) • g')

      -- Expand ⟨g', x* + (1/(L-μ))g'⟩
      have h_inner_xstar'_g' : @inner ℝ E _ g' (x_star + (1 / (L - μ)) • g') =
          @inner ℝ E _ g' x_star + (1 / (L - μ)) * ‖g'‖^2 := by
        simp only [inner_add_right, inner_smul_right, real_inner_self_eq_norm_sq]

      -- Bound B: (1/(2(L-μ)))‖g'‖² ≤ h(x*) - h(x) + ⟨g', x - x*⟩
      have h_bound_B : (1 / (2 * (L - μ))) * ‖g'‖^2 ≤ h x_star - h x + @inner ℝ E _ g' (x - x_star) := by
        have h4 : @inner ℝ E _ g' (x - x_star) = @inner ℝ E _ g' x - @inner ℝ E _ g' x_star :=
          inner_sub_right g' x x_star
        have step1' : h x - @inner ℝ E _ g' x ≤ h (x_star + (1 / (L - μ)) • g') -
            (@inner ℝ E _ g' x_star + (1 / (L - μ)) * ‖g'‖^2) := by
          rw [← h_inner_xstar'_g']
          exact h_φx_le
        have step2' : h (x_star + (1 / (L - μ)) • g') - (@inner ℝ E _ g' x_star + (1 / (L - μ)) * ‖g'‖^2) ≤
            h x_star + (1 / (2 * (L - μ))) * ‖g'‖^2 - (@inner ℝ E _ g' x_star + (1 / (L - μ)) * ‖g'‖^2) := by
          linarith [h_descent_xstar]
        have step3' : h x - @inner ℝ E _ g' x ≤
            h x_star - @inner ℝ E _ g' x_star - (1 / (2 * (L - μ))) * ‖g'‖^2 := by
          have := le_trans step1' step2'
          -- Simplify RHS of step2': h(x*) + (1/(2(L-μ)))‖g'‖² - (⟨g', x*⟩ + (1/(L-μ))‖g'‖²)
          -- = h(x*) - ⟨g', x*⟩ + (1/(2(L-μ)) - 1/(L-μ))‖g'‖²
          -- = h(x*) - ⟨g', x*⟩ - (1/(2(L-μ)))‖g'‖²
          have h_rhs_simp : h x_star + (1 / (2 * (L - μ))) * ‖g'‖^2 -
              (@inner ℝ E _ g' x_star + (1 / (L - μ)) * ‖g'‖^2) =
              h x_star - @inner ℝ E _ g' x_star - (1 / (2 * (L - μ))) * ‖g'‖^2 := by
            have hne : L - μ ≠ 0 := ne_of_gt hL_sub_μ_pos
            field_simp
            ring
          linarith [this, h_rhs_simp]
        have step4 : (1 / (2 * (L - μ))) * ‖g'‖^2 ≤ h x_star - h x + @inner ℝ E _ g' x - @inner ℝ E _ g' x_star := by
          linarith
        linarith

      -- Add bounds A and B: (1/(L-μ))‖g'‖² ≤ ⟨g', x - x*⟩
      have h_combined : (1 / (L - μ)) * ‖g'‖^2 ≤ @inner ℝ E _ g' (x - x_star) := by
        have h_add := add_le_add h_bound_A h_bound_B
        have lhs_eq : (1 / (2 * (L - μ))) * ‖g'‖^2 + (1 / (2 * (L - μ))) * ‖g'‖^2 = (1 / (L - μ)) * ‖g'‖^2 := by field_simp; ring
        have rhs_eq : (h x - h x_star) + (h x_star - h x + @inner ℝ E _ g' (x - x_star)) =
            @inner ℝ E _ g' (x - x_star) := by ring
        linarith

      -- Multiply by (L-μ): ‖g'‖² ≤ (L-μ)⟨g', x - x*⟩
      have h_cocoer_h : ‖g'‖^2 ≤ (L - μ) * @inner ℝ E _ g' (x - x_star) := by
        calc ‖g'‖^2 = (L - μ) * ((1 / (L - μ)) * ‖g'‖^2) := by field_simp
          _ ≤ (L - μ) * @inner ℝ E _ g' (x - x_star) := by
              apply mul_le_mul_of_nonneg_left h_combined (le_of_lt hL_sub_μ_pos)

      -- Convert g' = ∇h(x) = g - μd to the target form
      simp only [g'] at h_cocoer_h
      rw [h_grad x] at h_cocoer_h
      simp only [g, d] at h_cocoer_h ⊢
      exact h_cocoer_h

  -- Expand LHS: ‖g - μd‖² = ‖g‖² - 2μ⟨g,d⟩ + μ²‖d‖²
  have h_expand_lhs : ‖g - μ • d‖^2 = ‖g‖^2 - 2 * μ * @inner ℝ E _ g d + μ^2 * ‖d‖^2 := by
    rw [sub_eq_add_neg, norm_add_sq_real]
    simp only [norm_neg, inner_neg_right, norm_smul, Real.norm_eq_abs, abs_of_pos hμ,
               inner_smul_right, real_inner_self_eq_norm_sq]
    ring

  -- Expand RHS: (L-μ)⟨g - μd, d⟩ = (L-μ)⟨g,d⟩ - (L-μ)μ‖d‖²
  have h_expand_rhs : (L - μ) * @inner ℝ E _ (g - μ • d) d =
      (L - μ) * @inner ℝ E _ g d - (L - μ) * μ * ‖d‖^2 := by
    rw [inner_sub_left, inner_smul_left]
    simp only [real_inner_self_eq_norm_sq, conj_trivial]
    ring

  -- From h_aux_cocoer: ‖g‖² - 2μ⟨g,d⟩ + μ²‖d‖² ≤ (L-μ)⟨g,d⟩ - (L-μ)μ‖d‖²
  have h_ineq : ‖g‖^2 - 2 * μ * @inner ℝ E _ g d + μ^2 * ‖d‖^2 ≤
      (L - μ) * @inner ℝ E _ g d - (L - μ) * μ * ‖d‖^2 := by
    rw [← h_expand_lhs, ← h_expand_rhs]
    exact h_aux_cocoer

  -- Rearrange: ‖g‖² + μL‖d‖² ≤ (L+μ)⟨g,d⟩
  have h_rearrange : ‖g‖^2 + μ * L * ‖d‖^2 ≤ (L + μ) * @inner ℝ E _ g d := by
    -- From h_ineq: ‖g‖² - 2μ⟨g,d⟩ + μ²‖d‖² ≤ (L-μ)⟨g,d⟩ - (L-μ)μ‖d‖²
    -- Add 2μ⟨g,d⟩ to both sides:
    -- ‖g‖² + μ²‖d‖² ≤ (L-μ+2μ)⟨g,d⟩ - (L-μ)μ‖d‖²
    -- ‖g‖² + μ²‖d‖² ≤ (L+μ)⟨g,d⟩ - (Lμ - μ²)‖d‖²
    -- ‖g‖² + μ²‖d‖² + Lμ‖d‖² - μ²‖d‖² ≤ (L+μ)⟨g,d⟩
    -- ‖g‖² + Lμ‖d‖² ≤ (L+μ)⟨g,d⟩
    linarith

  -- Divide by (L+μ): ⟨g,d⟩ ≥ (μL)/(μ+L)‖d‖² + 1/(μ+L)‖g‖²
  have h_final : @inner ℝ E _ g d ≥ (μ * L) / (μ + L) * ‖d‖^2 + 1 / (μ + L) * ‖g‖^2 := by
    have h1 : (μ + L) * @inner ℝ E _ g d ≥ μ * L * ‖d‖^2 + ‖g‖^2 := by linarith
    have h2 : (μ * L) / (μ + L) * ‖d‖^2 + 1 / (μ + L) * ‖g‖^2 =
        (μ * L * ‖d‖^2 + ‖g‖^2) / (μ + L) := by field_simp
    rw [h2, ge_iff_le]
    -- (μL·‖d‖² + ‖g‖²)/(μ+L) ≤ ⟨g,d⟩ iff μL·‖d‖² + ‖g‖² ≤ (μ+L)·⟨g,d⟩
    have h_ne : μ + L ≠ 0 := ne_of_gt h_sum_pos
    have h3 : (μ * L * ‖d‖^2 + ‖g‖^2) / (μ + L) * (μ + L) = μ * L * ‖d‖^2 + ‖g‖^2 := by
      field_simp
    have h4 : (μ * L * ‖d‖^2 + ‖g‖^2) / (μ + L) ≤ @inner ℝ E _ g d ↔
        (μ * L * ‖d‖^2 + ‖g‖^2) / (μ + L) * (μ + L) ≤ @inner ℝ E _ g d * (μ + L) := by
      constructor
      · intro h
        apply mul_le_mul_of_nonneg_right h (le_of_lt h_sum_pos)
      · intro h
        have h5 : 0 < 1 / (μ + L) := by positivity
        have := mul_le_mul_of_nonneg_right h (le_of_lt h5)
        simp only [mul_assoc] at this
        have h6 : (μ + L) * (1 / (μ + L)) = 1 := by field_simp
        simp only [h6, mul_one] at this
        have h7 : μ * (L * ‖d‖^2) = μ * L * ‖d‖^2 := by ring
        simp only [h7] at this
        exact this
    rw [h4, h3]
    have h5 : @inner ℝ E _ g d * (μ + L) = (μ + L) * @inner ℝ E _ g d := by ring
    rw [h5]
    linarith

  convert h_final using 2 <;> simp only [g, d]

/-- Interpolation condition for smooth convex functions.

    For L-smooth convex f with minimizer x* (where ∇f(x*) = 0):
    f(x) - f(x*) + (1/2L)‖∇f(x)‖² ≤ ⟨∇f(x), x - x*⟩

    Equivalently: f(x) - f(x*) ≤ ⟨∇f(x), x - x*⟩ - (1/2L)‖∇f(x)‖²

    This is tighter than plain convexity and is key for the O(1/k) convergence proof.
-/
lemma smooth_convex_interpolation (f : E → ℝ) (L : ℝ) (hL : 0 < L)
    (hSmooth : IsLSmooth f L) (hConvex : ConvexOn ℝ Set.univ f)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    f x - f x_star + (1 / (2 * L)) * ‖gradient f x‖^2 ≤
      @inner ℝ E _ (gradient f x) (x - x_star) := by
  -- This follows from combining bounds proved in lsmooth_cocoercivity:
  -- Bound A: (1/2L)‖g‖² ≤ f(x) - f(x*)   (from minimality of x*)
  -- Bound B: (1/2L)‖g‖² ≤ f(x*) - f(x) + ⟨g, x - x*⟩   (from tilted function technique)
  -- Adding A to the rearranged form of B gives the result.
  have hDiff : Differentiable ℝ f := hSmooth.1
  let g := gradient f x

  -- Step 1: x* minimizes f (since ∇f(x*) = 0 and f is convex)
  have h_xstar_min : ∀ y, f x_star ≤ f y := convex_first_order_optimality f hConvex hDiff x_star hMin

  -- Step 2: Bound A - from descent lemma and minimality
  -- f(x - (1/L)∇f(x)) ≤ f(x) - (1/2L)‖∇f(x)‖²
  -- f(x*) ≤ f(x - (1/L)∇f(x))
  -- Therefore: (1/2L)‖∇f(x)‖² ≤ f(x) - f(x*)
  have h_fund := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth x (x - (1 / L) • g)
  have h_descent : f (x - (1 / L) • g) ≤ f x - (1 / (2 * L)) * ‖g‖^2 := by
    have h_diff : (x - (1 / L) • g) - x = -((1 / L) • g) := by simp [sub_eq_add_neg, add_comm]
    have h_inner : @inner ℝ E _ g ((x - (1 / L) • g) - x) = -(1 / L) * ‖g‖^2 := by
      rw [h_diff]
      simp only [inner_neg_right, inner_smul_right, real_inner_self_eq_norm_sq]
      ring
    have h_norm : ‖(x - (1 / L) • g) - x‖^2 = (1 / L)^2 * ‖g‖^2 := by
      rw [h_diff, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/L > 0)]
      ring
    calc f (x - (1 / L) • g) ≤ f x + @inner ℝ E _ g ((x - (1 / L) • g) - x) +
                                 (L / 2) * ‖(x - (1 / L) • g) - x‖^2 := h_fund
      _ = f x + (-(1 / L) * ‖g‖^2) + (L / 2) * ((1 / L)^2 * ‖g‖^2) := by rw [h_inner, h_norm]
      _ = f x - (1 / (2 * L)) * ‖g‖^2 := by field_simp; ring
  have h_bound_A : (1 / (2 * L)) * ‖g‖^2 ≤ f x - f x_star := by
    have := h_xstar_min (x - (1 / L) • g)
    linarith

  -- Step 3: Bound B - from tilted function technique
  -- The tilted function h(z) = f(z) - ⟨g, z⟩ is convex and has minimum at x
  -- This gives: (1/2L)‖g‖² ≤ f(x*) - f(x) + ⟨g, x - x*⟩
  have h_linear_concave : ConcaveOn ℝ Set.univ (fun z => @inner ℝ E _ g z) := by
    constructor
    · exact convex_univ
    · intro z _ w _ a b ha hb hab
      simp only [inner_add_right, inner_smul_right, smul_eq_mul]
      linarith
  have h_convex_tilt : ConvexOn ℝ Set.univ (fun z => f z - @inner ℝ E _ g z) :=
    hConvex.sub h_linear_concave
  have h_grad_tilt_x : gradient (fun z => f z - @inner ℝ E _ g z) x = 0 := by
    have hf_diff : DifferentiableAt ℝ f x := hDiff x
    have hg_diff : DifferentiableAt ℝ (fun z => @inner ℝ E _ g z) x :=
      (innerSL (𝕜 := ℝ) g).differentiableAt
    have hg_grad : HasGradientAt (fun z => @inner ℝ E _ g z) g x := by
      rw [hasGradientAt_iff_hasFDerivAt]
      have h1 := (innerSL (𝕜 := ℝ) g).hasFDerivAt (x := x)
      simp only [InnerProductSpace.toDual] at h1 ⊢
      convert h1 using 1
    have hf_grad : HasGradientAt f g x := hf_diff.hasGradientAt
    have h_sub : HasGradientAt (fun z => f z - @inner ℝ E _ g z) (g - g) x := by
      have h1 := hasGradientAt_iff_hasFDerivAt.mp hf_grad
      have h2 := hasGradientAt_iff_hasFDerivAt.mp hg_grad
      have h3 := h1.sub h2
      rw [hasGradientAt_iff_hasFDerivAt]
      convert h3 using 1
      simp only [map_sub]
    rw [sub_self] at h_sub
    exact h_sub.gradient
  have h_diff_tilt : Differentiable ℝ (fun z => f z - @inner ℝ E _ g z) := by
    intro z
    exact (hDiff z).sub (innerSL (𝕜 := ℝ) g).differentiableAt
  have h_x_min_tilt : ∀ y, (f x - @inner ℝ E _ g x) ≤ (f y - @inner ℝ E _ g y) :=
    convex_first_order_optimality (fun z => f z - @inner ℝ E _ g z) h_convex_tilt h_diff_tilt x h_grad_tilt_x
  -- Apply at x* + (1/L)g
  have h_fund_xstar := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth x_star (x_star + (1 / L) • g)
  have h_fund_xstar_bound : f (x_star + (1 / L) • g) ≤ f x_star + (1 / (2 * L)) * ‖g‖^2 := by
    have h_diff_pt : (x_star + (1 / L) • g) - x_star = (1 / L) • g := by abel
    have h_inner_pt : @inner ℝ E _ (gradient f x_star) ((x_star + (1 / L) • g) - x_star) = 0 := by
      rw [hMin, inner_zero_left]
    have h_norm_pt : ‖(x_star + (1 / L) • g) - x_star‖^2 = (1 / L)^2 * ‖g‖^2 := by
      rw [h_diff_pt, norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity : 1/L > 0)]
      ring
    calc f (x_star + (1 / L) • g) ≤ f x_star + @inner ℝ E _ (gradient f x_star)
          ((x_star + (1 / L) • g) - x_star) + (L / 2) * ‖(x_star + (1 / L) • g) - x_star‖^2 := h_fund_xstar
      _ = f x_star + 0 + (L / 2) * ((1 / L)^2 * ‖g‖^2) := by rw [h_inner_pt, h_norm_pt]
      _ = f x_star + (1 / (2 * L)) * ‖g‖^2 := by field_simp; ring
  have h_inner_xstar_g : @inner ℝ E _ g (x_star + (1 / L) • g) =
      @inner ℝ E _ g x_star + (1 / L) * ‖g‖^2 := by
    simp only [inner_add_right, inner_smul_right, real_inner_self_eq_norm_sq]
  have h_tilt_bound := h_x_min_tilt (x_star + (1 / L) • g)
  have h_bound_B : (1 / (2 * L)) * ‖g‖^2 ≤ f x_star - f x + @inner ℝ E _ g (x - x_star) := by
    have step1 : f x - @inner ℝ E _ g x ≤ f (x_star + (1 / L) • g) -
        @inner ℝ E _ g (x_star + (1 / L) • g) := h_tilt_bound
    have step2 : f (x_star + (1 / L) • g) - @inner ℝ E _ g (x_star + (1 / L) • g) ≤
        (f x_star + (1 / (2 * L)) * ‖g‖^2) - (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) := by
      rw [h_inner_xstar_g]
      linarith [h_fund_xstar_bound]
    have step3 : (f x_star + (1 / (2 * L)) * ‖g‖^2) - (@inner ℝ E _ g x_star + (1 / L) * ‖g‖^2) =
        f x_star - @inner ℝ E _ g x_star - (1 / (2 * L)) * ‖g‖^2 := by field_simp; ring
    have step4 : f x - @inner ℝ E _ g x ≤ f x_star - @inner ℝ E _ g x_star - (1 / (2 * L)) * ‖g‖^2 := by
      linarith
    have h4 : @inner ℝ E _ g (x - x_star) = @inner ℝ E _ g x - @inner ℝ E _ g x_star :=
      inner_sub_right g x x_star
    linarith

  -- Step 4: Combine bounds to get the interpolation inequality
  -- From h_bound_B: (1/2L)‖g‖² ≤ f(x*) - f(x) + ⟨g, x - x*⟩
  -- Rearranging: f(x) - f(x*) + (1/2L)‖g‖² ≤ ⟨g, x - x*⟩
  linarith

/-- One step of gradient descent with learning rate η. -/
noncomputable def gradientDescentStep (f : E → ℝ) (η : ℝ) (x : E) : E :=
  x - η • gradient f x

/-- k steps of gradient descent. -/
noncomputable def gradientDescentIterates (f : E → ℝ) (η : ℝ) (x₀ : E) : ℕ → E
  | 0 => x₀
  | n + 1 => gradientDescentStep f η (gradientDescentIterates f η x₀ n)

/-- The descent lemma: one step decreases function value.

The proof follows from L-smoothness:
1. By L-smoothness: f(y) ≤ f(x) + ⟨∇f(x), y-x⟩ + (L/2)‖y-x‖²
2. With y = x - η∇f(x), we have y - x = -η∇f(x)
3. So ⟨∇f(x), y-x⟩ = -η‖∇f(x)‖²
4. And ‖y-x‖² = η²‖∇f(x)‖²
5. Thus: f(y) ≤ f(x) - η‖∇f(x)‖² + (Lη²/2)‖∇f(x)‖²
6. Since η ≤ 1/L, we have (Lη²/2) ≤ η/2
7. Therefore: f(y) ≤ f(x) - (η/2)‖∇f(x)‖²

The key insight is that L-smoothness provides a second-order bound on function values,
which allows us to show descent over a single gradient step.
-/
theorem descent_lemma (f : E → ℝ) (L : ℝ) (hL : 0 < L)
    (hSmooth : IsLSmooth f L) (x : E) (η : ℝ) (hη : 0 < η) (hηL : η ≤ 1 / L) :
    f (gradientDescentStep f η x) ≤ f x - (η / 2) * ‖gradient f x‖^2 := by
  -- Define y = x - η∇f(x) (the gradient descent step)
  let y := x - η • gradient f x
  let g := gradient f x
  -- Step 1: Apply the fundamental inequality for L-smooth functions
  have h_fund := lsmooth_fundamental_ineq f L (le_of_lt hL) hSmooth x y
  -- Step 2: Compute y - x = -(η • ∇f(x))
  have h_diff : y - x = -(η • g) := by simp only [y, g]; abel
  -- Step 3: Compute ⟨∇f(x), y - x⟩ = -η‖∇f(x)‖²
  have h_inner : @inner ℝ E _ g (y - x) = -η * ‖g‖^2 := by
    rw [h_diff, inner_neg_right, inner_smul_right]
    rw [real_inner_self_eq_norm_sq]
    ring
  -- Step 4: Compute ‖y - x‖² = η²‖∇f(x)‖²
  have h_norm_sq : ‖y - x‖^2 = η^2 * ‖g‖^2 := by
    rw [h_diff, norm_neg, norm_smul, Real.norm_eq_abs]
    have : |η|^2 = η^2 := sq_abs η
    rw [mul_pow, this]
  -- Step 5: Substitute into the fundamental inequality
  -- f(y) ≤ f(x) + ⟨∇f(x), y - x⟩ + (L/2)‖y - x‖²
  --      = f(x) - η‖∇f(x)‖² + (L/2)η²‖∇f(x)‖²
  --      = f(x) + (-η + Lη²/2)‖∇f(x)‖²
  calc f y ≤ f x + @inner ℝ E _ g (y - x) + (L / 2) * ‖y - x‖^2 := h_fund
    _ = f x + (-η * ‖g‖^2) + (L / 2) * (η^2 * ‖g‖^2) := by rw [h_inner, h_norm_sq]
    _ = f x + (-η + L * η^2 / 2) * ‖g‖^2 := by ring
    _ ≤ f x + (-η / 2) * ‖g‖^2 := by {
        -- Need: -η + L*η²/2 ≤ -η/2
        -- i.e., L*η²/2 ≤ η/2
        -- i.e., L*η ≤ 1
        -- which follows from η ≤ 1/L
        have h_Lη : L * η ≤ 1 := by
          calc L * η = η * L := mul_comm L η
            _ ≤ (1 / L) * L := mul_le_mul_of_nonneg_right hηL (le_of_lt hL)
            _ = 1 := div_mul_cancel₀ 1 (ne_of_gt hL)
        have h_coeff : -η + L * η^2 / 2 ≤ -η / 2 := by
          have h1 : L * η^2 / 2 ≤ η / 2 := by
            have : L * η^2 ≤ η := by
              calc L * η^2 = (L * η) * η := by ring
                _ ≤ 1 * η := mul_le_mul_of_nonneg_right h_Lη (le_of_lt hη)
                _ = η := one_mul η
            linarith
          linarith
        have h_g_sq_nonneg : 0 ≤ ‖g‖^2 := sq_nonneg _
        nlinarith [sq_nonneg ‖g‖]
      }
    _ = f x - (η / 2) * ‖g‖^2 := by ring

/-- Convergence rate for smooth convex functions.
    After k iterations: f(x_k) - f(x*) ≤ ‖x₀ - x*‖² / (2ηk) -/
theorem convex_convergence_rate (f : E → ℝ) (L : ℝ) (hL : 0 < L)
    (hSmooth : IsLSmooth f L) (hConvex : ConvexOn ℝ Set.univ f)
    (x_star : E) (hMin : ∀ x, f x_star ≤ f x)
    (η : ℝ) (hη : 0 < η) (hηL : η ≤ 1 / L) (x₀ : E) :
    ∀ k : ℕ, k > 0 →
      f (gradientDescentIterates f η x₀ k) - f x_star ≤ ‖x₀ - x_star‖^2 / (2 * η * k) := by
  intro k hk

  /- Convergence Proof via Telescoping Descent Lemma

  For smooth convex functions, we prove O(1/k) convergence by combining:

  1. **Descent Lemma (L-smoothness)**:
     f(x_{i+1}) ≤ f(x_i) - (η/2)‖∇f(x_i)‖²

  2. **First-Order Convexity**:
     For convex f: f(x) - f(x*) ≤ ⟨∇f(x), x - x*⟩

  3. **Telescoping Sum**:
     Sum descent inequalities over i = 0, ..., k-1:
     f(x_k) - f(x_0) ≤ -(η/2) ∑ᵢ ‖∇f(x_i)‖²

  4. **Cauchy-Schwarz Lower Bound on Gradient Norms**:
     From convexity: ‖∇f(x_i)‖² ≥ 2(f(x_i) - f(x*))² / ‖x_i - x*‖²

     However, this requires bounded domain assumptions that conflict with the
     general statement. The standard proof instead uses:

     ⟨∇f(x), x - x*⟩ ≥ f(x) - f(x*) (convexity)

     Which combined with ‖∇f(x)‖ · ‖x - x*‖ ≥ |⟨∇f(x), x - x*⟩| gives:
     ‖∇f(x)‖ ≥ (f(x) - f(x*)) / ‖x - x*‖

  5. **Proof Dependencies**:
     - `lsmooth_fundamental_ineq` (COMPLETE)
     - `descent_lemma` (COMPLETE, uses lsmooth_fundamental_ineq)
     - First-order convexity characterization from Mathlib's ConvexOn
     - Telescoping sum machinery
  -/

  -- The proof uses the key identity for distance to optimum:
  -- ‖x_{k+1} - x*‖² = ‖x_k - η∇f(x_k) - x*‖²
  --                 = ‖x_k - x*‖² - 2η⟨∇f(x_k), x_k - x*⟩ + η²‖∇f(x_k)‖²
  --
  -- From convexity: f(x_k) - f(x*) ≤ ⟨∇f(x_k), x_k - x*⟩
  -- So: 2η(f(x_k) - f(x*)) ≤ 2η⟨∇f(x_k), x_k - x*⟩
  --
  -- Rearranging the distance identity:
  -- 2η(f(x_k) - f(x*)) ≤ ‖x_k - x*‖² - ‖x_{k+1} - x*‖² + η²‖∇f(x_k)‖²
  --
  -- Summing over i = 0 to k-1:
  -- 2η∑(f(x_i) - f(x*)) ≤ ‖x_0 - x*‖² - ‖x_k - x*‖² + η²∑‖∇f(x_i)‖²
  --
  -- From descent_lemma: f(x_{i+1}) ≤ f(x_i) - (η/2)‖∇f(x_i)‖²
  -- Telescoping: f(x_k) - f(x_0) ≤ -(η/2)∑‖∇f(x_i)‖²
  -- So: (η/2)∑‖∇f(x_i)‖² ≤ f(x_0) - f(x_k) ≤ f(x_0) - f(x*)
  -- Hence: η²∑‖∇f(x_i)‖² ≤ 2η(f(x_0) - f(x*))
  --
  -- Substituting:
  -- 2η∑(f(x_i) - f(x*)) ≤ ‖x_0 - x*‖² + 2η(f(x_0) - f(x*))
  --
  -- Since f(x_k) - f(x*) ≤ (1/k)∑(f(x_i) - f(x*)) (minimum ≤ average):
  -- 2ηk(f(x_k) - f(x*)) ≤ ‖x_0 - x*‖² + 2η(f(x_0) - f(x*))
  --
  -- Using the smooth convex interpolation lemma gives the factor of 2.

  have hDiff : Differentiable ℝ f := hSmooth.1

  -- Step 1: Derive ∇f(x*) = 0 from the fact that x* is a global minimizer
  -- For differentiable f, if x* minimizes f, then ∇f(x*) = 0
  have hGradZero : gradient f x_star = 0 := by
    by_contra h
    -- If ∇f(x*) ≠ 0, moving in direction -∇f(x*) decreases f, contradiction
    let g := gradient f x_star
    have hg_ne : g ≠ 0 := h
    have h_norm_pos : ‖g‖ > 0 := norm_pos_iff.mpr hg_ne
    -- Consider f(x* - t·g) for small t > 0
    -- Directional derivative at x* in direction -g is ⟨∇f(x*), -g⟩ = -‖g‖² < 0
    have hf_grad : HasGradientAt f g x_star := (hDiff x_star).hasGradientAt
    have hf_deriv : HasFDerivAt f (innerSL (𝕜 := ℝ) g) x_star := hf_grad.hasFDerivAt
    -- Use the definition of derivative with a smaller ε to ensure error is dominated
    -- We want: |f(x* - t·g) - f(x*) - ⟨g, -t·g⟩| ≤ (‖g‖²/2) · |t| · ‖g‖
    -- i.e., |f(x* - t·g) - f(x*) + t·‖g‖²| ≤ (t·‖g‖³)/2
    -- This gives: f(x* - t·g) - f(x*) ≤ -t·‖g‖² + t·‖g‖³/2 = -t·‖g‖²·(1 - ‖g‖/2)
    -- For this to be negative, we need t > 0 and 1 - ‖g‖/2 > 0, i.e., ‖g‖ < 2
    -- To handle ‖g‖ ≥ 2, we use a smaller ε in the isLittleO bound
    rw [HasFDerivAt, hasFDerivAtFilter_iff_isLittleO, Asymptotics.isLittleO_iff] at hf_deriv
    -- Choose c = ‖g‖/2 so the error bound becomes (‖g‖/2) · ‖h‖ ≤ (‖g‖/2) · t · ‖g‖
    specialize hf_deriv (by linarith : (0 : ℝ) < ‖g‖ / 2)
    rw [Filter.eventually_iff_exists_mem] at hf_deriv
    obtain ⟨s, hs_mem, hs_bound⟩ := hf_deriv
    rw [Metric.mem_nhds_iff] at hs_mem
    obtain ⟨ε, hε_pos, hε_ball⟩ := hs_mem
    -- Choose t small enough that t·g ∈ ball and t < 1
    let t := min (ε / (2 * ‖g‖)) (1 / 2)
    have ht_pos : t > 0 := by
      simp only [t, lt_min_iff]
      constructor
      · positivity
      · linarith
    have ht_le_half : t ≤ 1/2 := min_le_right _ _
    have h_tg_small : ‖t • g‖ < ε := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos ht_pos]
      have h1 : t ≤ ε / (2 * ‖g‖) := min_le_left _ _
      calc t * ‖g‖ ≤ (ε / (2 * ‖g‖)) * ‖g‖ := by apply mul_le_mul_of_nonneg_right h1 (le_of_lt h_norm_pos)
        _ = ε / 2 := by field_simp
        _ < ε := by linarith
    have h_in_s : x_star - t • g ∈ s := by
      apply hε_ball
      simp only [Metric.mem_ball, dist_eq_norm]
      have h_eq : (x_star - t • g) - x_star = -(t • g) := by abel
      rw [h_eq, norm_neg]
      exact h_tg_small
    have hs_bound' := hs_bound (x_star - t • g) h_in_s
    simp only [innerSL_apply_apply] at hs_bound'
    -- hs_bound: ‖f(x* - t·g) - f(x*) - ⟨g, (x* - t·g) - x*⟩‖ < (‖g‖/2) * ‖(x* - t·g) - x*‖
    have h_diff_eq : (x_star - t • g) - x_star = -(t • g) := by abel
    have h_inner : @inner ℝ E _ g ((x_star - t • g) - x_star) = -t * ‖g‖^2 := by
      rw [h_diff_eq, inner_neg_right, inner_smul_right, real_inner_self_eq_norm_sq]
      ring
    have h_norm_diff : ‖(x_star - t • g) - x_star‖ = t * ‖g‖ := by
      rw [h_diff_eq, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos ht_pos]
    rw [h_inner, h_norm_diff] at hs_bound'
    -- hs_bound': ‖f(x* - t·g) - f(x*) - (-t * ‖g‖²)‖ < (‖g‖/2) * (t * ‖g‖)
    -- i.e., |f(x* - t·g) - f(x*) + t·‖g‖²| < t·‖g‖²/2
    have h_rhs_eq : ‖g‖ / 2 * (t * ‖g‖) = t * ‖g‖^2 / 2 := by ring
    rw [h_rhs_eq] at hs_bound'
    -- From absolute value bound (note: isLittleO_iff gives ≤, not <):
    -- ‖f(x* - t·g) - f(x*) + t·‖g‖²‖ ≤ t·‖g‖²/2
    have h_from_abs := abs_le.mp hs_bound'
    -- h_from_abs gives bounds on f(x* - t·g) - f(x*) - (-t * ‖g‖²)
    -- which equals f(x* - t·g) - f(x*) + t * ‖g‖²
    have h_upper : f (x_star - t • g) - f x_star + t * ‖g‖^2 ≤ t * ‖g‖^2 / 2 := by
      have h := h_from_abs.2
      -- h : f (x_star - t • g) - f x_star - -t * ‖g‖^2 ≤ t * ‖g‖^2 / 2
      -- Need: f (x_star - t • g) - f x_star + t * ‖g‖^2 ≤ t * ‖g‖^2 / 2
      have h_eq : f (x_star - t • g) - f x_star - -t * ‖g‖^2 =
          f (x_star - t • g) - f x_star + t * ‖g‖^2 := by ring
      rw [h_eq] at h
      exact h
    have h_neg : f (x_star - t • g) < f x_star := by
      have : f (x_star - t • g) - f x_star ≤ t * ‖g‖^2 / 2 - t * ‖g‖^2 := by linarith
      have h_calc : t * ‖g‖^2 / 2 - t * ‖g‖^2 = -t * ‖g‖^2 / 2 := by ring
      have h_neg_val : -t * ‖g‖^2 / 2 < 0 := by
        have := mul_pos ht_pos (sq_pos_of_pos h_norm_pos)
        linarith
      linarith
    exact absurd h_neg (not_lt.mpr (hMin (x_star - t • g)))

  -- Step 2: Per-step distance contraction bound
  -- ‖x_{k+1} - x*‖² ≤ ‖x_k - x*‖² - 2η(f(x_k) - f(x*))
  have per_step_bound : ∀ x : E,
      ‖gradientDescentStep f η x - x_star‖^2 ≤ ‖x - x_star‖^2 - 2 * η * (f x - f x_star) := by
    intro x
    let g := gradient f x
    let y := gradientDescentStep f η x
    -- y = x - η·g
    have hy : y = x - η • g := rfl
    -- ‖y - x*‖² = ‖(x - x*) - η·g‖²
    have h_diff : y - x_star = (x - x_star) - η • g := by simp only [hy]; abel
    -- Expand using ‖a - b‖² = ‖a‖² - 2⟨a,b⟩ + ‖b‖²
    have h_expand : ‖y - x_star‖^2 = ‖x - x_star‖^2 - 2 * η * @inner ℝ E _ g (x - x_star) + η^2 * ‖g‖^2 := by
      rw [h_diff, norm_sub_sq_real]
      have h1 : ‖η • g‖^2 = η^2 * ‖g‖^2 := by
        rw [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
      have h2 : @inner ℝ E _ (x - x_star) (η • g) = η * @inner ℝ E _ (x - x_star) g := by
        rw [inner_smul_right]
      rw [h1, h2, real_inner_comm]
      ring
    -- From smooth_convex_interpolation:
    -- f(x) - f(x*) + (1/2L)‖g‖² ≤ ⟨g, x - x*⟩
    have h_interp := smooth_convex_interpolation f L hL hSmooth hConvex x x_star hGradZero
    -- Multiply by 2η:
    -- 2η(f(x) - f(x*)) + (η/L)‖g‖² ≤ 2η⟨g, x - x*⟩
    have h_interp_scaled : 2 * η * (f x - f x_star) + (η / L) * ‖g‖^2 ≤
        2 * η * @inner ℝ E _ g (x - x_star) := by
      have h1 : 2 * η * (f x - f x_star + (1 / (2 * L)) * ‖g‖^2) ≤
          2 * η * @inner ℝ E _ g (x - x_star) := by
        apply mul_le_mul_of_nonneg_left h_interp
        linarith
      have h2 : 2 * η * (f x - f x_star + (1 / (2 * L)) * ‖g‖^2) =
          2 * η * (f x - f x_star) + (η / L) * ‖g‖^2 := by
        have hL_ne : L ≠ 0 := ne_of_gt hL
        field_simp
      linarith
    -- From h_expand:
    -- ‖y - x*‖² = ‖x - x*‖² - 2η⟨g, x - x*⟩ + η²‖g‖²
    -- Using h_interp_scaled:
    -- -2η⟨g, x - x*⟩ ≤ -2η(f(x) - f(x*)) - (η/L)‖g‖²
    -- So: ‖y - x*‖² ≤ ‖x - x*‖² - 2η(f(x) - f(x*)) - (η/L)‖g‖² + η²‖g‖²
    --              = ‖x - x*‖² - 2η(f(x) - f(x*)) + η(η - 1/L)‖g‖²
    -- Since η ≤ 1/L, we have η - 1/L ≤ 0, so η(η - 1/L)‖g‖² ≤ 0
    have h_coeff_neg : η * (η - 1/L) ≤ 0 := by
      have h1 : η - 1/L ≤ 0 := by linarith
      exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt hη) h1
    have h_grad_term : η^2 * ‖g‖^2 - (η / L) * ‖g‖^2 ≤ 0 := by
      have h1 : η^2 * ‖g‖^2 - (η / L) * ‖g‖^2 = η * (η - 1/L) * ‖g‖^2 := by
        have hL_ne : L ≠ 0 := ne_of_gt hL
        have h : η / L = η * (1/L) := by ring
        rw [h]
        ring
      rw [h1]
      exact mul_nonpos_of_nonpos_of_nonneg h_coeff_neg (sq_nonneg _)
    calc ‖y - x_star‖^2 = ‖x - x_star‖^2 - 2 * η * @inner ℝ E _ g (x - x_star) + η^2 * ‖g‖^2 := h_expand
      _ ≤ ‖x - x_star‖^2 - (2 * η * (f x - f x_star) + (η / L) * ‖g‖^2) + η^2 * ‖g‖^2 := by linarith [h_interp_scaled]
      _ = ‖x - x_star‖^2 - 2 * η * (f x - f x_star) + (η^2 * ‖g‖^2 - (η / L) * ‖g‖^2) := by ring
      _ ≤ ‖x - x_star‖^2 - 2 * η * (f x - f x_star) + 0 := by linarith [h_grad_term]
      _ = ‖x - x_star‖^2 - 2 * η * (f x - f x_star) := by ring

  -- Step 3: Descent property: f(x_{i+1}) ≤ f(x_i)
  have descent : ∀ i : ℕ, f (gradientDescentIterates f η x₀ (i + 1)) ≤ f (gradientDescentIterates f η x₀ i) := by
    intro i
    let x_i := gradientDescentIterates f η x₀ i
    have h_descent := descent_lemma f L hL hSmooth x_i η hη hηL
    have h_nonneg : 0 ≤ (η / 2) * ‖gradient f x_i‖^2 := by positivity
    -- gradientDescentIterates f η x₀ (i + 1) = gradientDescentStep f η (gradientDescentIterates f η x₀ i) = gradientDescentStep f η x_i
    have h_eq : gradientDescentIterates f η x₀ (i + 1) = gradientDescentStep f η x_i := rfl
    rw [h_eq]
    linarith

  -- Step 4: Sum the per-step bounds via induction
  -- We prove: 2η · ∑_{i=0}^{k-1} (f(x_i) - f(x*)) ≤ ‖x_0 - x*‖²
  have sum_bound : ∀ n : ℕ, 2 * η * (Finset.range n).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) ≤
      ‖x₀ - x_star‖^2 - ‖gradientDescentIterates f η x₀ n - x_star‖^2 := by
    intro n
    induction n with
    | zero =>
      simp only [Finset.range_zero, Finset.sum_empty, mul_zero, gradientDescentIterates]
      linarith [sq_nonneg ‖x₀ - x_star‖]
    | succ n ih =>
      -- Sum from 0 to n = Sum from 0 to n-1 + (f(x_n) - f(x*))
      rw [Finset.sum_range_succ]
      let x_n := gradientDescentIterates f η x₀ n
      let x_n1 := gradientDescentIterates f η x₀ (n + 1)
      -- From per_step_bound: ‖x_{n+1} - x*‖² ≤ ‖x_n - x*‖² - 2η(f(x_n) - f(x*))
      have h_step := per_step_bound x_n
      -- x_{n+1} = gradientDescentStep f η x_n
      have h_eq : x_n1 = gradientDescentStep f η x_n := rfl
      rw [← h_eq] at h_step
      -- ih: 2η · ∑_{i=0}^{n-1} (f(x_i) - f(x*)) ≤ ‖x_0 - x*‖² - ‖x_n - x*‖²
      -- h_step: ‖x_{n+1} - x*‖² ≤ ‖x_n - x*‖² - 2η(f(x_n) - f(x*))
      -- Rearranging h_step: 2η(f(x_n) - f(x*)) ≤ ‖x_n - x*‖² - ‖x_{n+1} - x*‖²
      have h_step' : 2 * η * (f x_n - f x_star) ≤ ‖x_n - x_star‖^2 - ‖x_n1 - x_star‖^2 := by
        linarith
      -- Add ih and h_step'
      calc 2 * η * ((Finset.range n).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) +
              (f x_n - f x_star))
          = 2 * η * (Finset.range n).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) +
            2 * η * (f x_n - f x_star) := by ring
        _ ≤ (‖x₀ - x_star‖^2 - ‖x_n - x_star‖^2) + (‖x_n - x_star‖^2 - ‖x_n1 - x_star‖^2) := by linarith [ih, h_step']
        _ = ‖x₀ - x_star‖^2 - ‖x_n1 - x_star‖^2 := by ring

  -- Step 5: Since f is decreasing, f(x_k) ≤ f(x_i) for all i < k
  -- So k · (f(x_k) - f(x*)) ≤ ∑_{i=0}^{k-1} (f(x_i) - f(x*))
  have sum_lower_bound : (k : ℝ) * (f (gradientDescentIterates f η x₀ k) - f x_star) ≤
      (Finset.range k).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) := by
    have h_mono : ∀ i ∈ Finset.range k, f (gradientDescentIterates f η x₀ k) ≤ f (gradientDescentIterates f η x₀ i) := by
      intro i hi
      rw [Finset.mem_range] at hi
      -- f(x_k) ≤ f(x_i) for i < k by repeated application of descent
      have : ∀ j m : ℕ, j ≤ m → f (gradientDescentIterates f η x₀ m) ≤ f (gradientDescentIterates f η x₀ j) := by
        intro j m hjm
        induction m with
        | zero => simp only [Nat.le_zero] at hjm; rw [hjm]
        | succ m ih =>
          by_cases hj : j ≤ m
          · calc f (gradientDescentIterates f η x₀ (m + 1)) ≤ f (gradientDescentIterates f η x₀ m) := descent m
              _ ≤ f (gradientDescentIterates f η x₀ j) := ih hj
          · push_neg at hj
            have : j = m + 1 := by omega
            rw [this]
      exact this i k (le_of_lt hi)
    have h_term_bound : ∀ i ∈ Finset.range k,
        f (gradientDescentIterates f η x₀ k) - f x_star ≤ f (gradientDescentIterates f η x₀ i) - f x_star := by
      intro i hi
      linarith [h_mono i hi]
    calc (k : ℝ) * (f (gradientDescentIterates f η x₀ k) - f x_star)
        = (Finset.range k).sum (fun _ => f (gradientDescentIterates f η x₀ k) - f x_star) := by
          simp only [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
      _ ≤ (Finset.range k).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) := by
          apply Finset.sum_le_sum h_term_bound

  -- Step 6: Combine to get the final bound
  have h_sum := sum_bound k
  have h_lower := sum_lower_bound
  -- From h_sum: 2η · ∑(f(x_i) - f(x*)) ≤ ‖x_0 - x*‖²
  have h_sum' : 2 * η * (Finset.range k).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) ≤
      ‖x₀ - x_star‖^2 := by
    have h_nonneg : 0 ≤ ‖gradientDescentIterates f η x₀ k - x_star‖^2 := sq_nonneg _
    linarith
  -- From h_lower and h_sum': 2ηk(f(x_k) - f(x*)) ≤ ‖x_0 - x*‖²
  have h_combined : 2 * η * k * (f (gradientDescentIterates f η x₀ k) - f x_star) ≤ ‖x₀ - x_star‖^2 := by
    calc 2 * η * k * (f (gradientDescentIterates f η x₀ k) - f x_star)
        = 2 * η * (k * (f (gradientDescentIterates f η x₀ k) - f x_star)) := by ring
      _ ≤ 2 * η * (Finset.range k).sum (fun i => f (gradientDescentIterates f η x₀ i) - f x_star) := by
          apply mul_le_mul_of_nonneg_left h_lower; linarith
      _ ≤ ‖x₀ - x_star‖^2 := h_sum'
  -- Divide by 2ηk > 0
  have hk_pos : (k : ℝ) > 0 := by exact Nat.cast_pos.mpr hk
  have h_denom_pos : 2 * η * k > 0 := by positivity
  -- f(x_k) - f(x*) ≤ ‖x_0 - x*‖² / (2ηk)
  have h_final : f (gradientDescentIterates f η x₀ k) - f x_star ≤ ‖x₀ - x_star‖^2 / (2 * η * k) := by
    -- From h_combined: (2 * η * k) * (f(x_k) - f(x*)) ≤ ‖x₀ - x*‖²
    -- Dividing by (2 * η * k) > 0: f(x_k) - f(x*) ≤ ‖x₀ - x*‖² / (2 * η * k)
    have h1 : (2 * η * k) * (f (gradientDescentIterates f η x₀ k) - f x_star) ≤ ‖x₀ - x_star‖^2 := h_combined
    have h2 : f (gradientDescentIterates f η x₀ k) - f x_star ≤
        ‖x₀ - x_star‖^2 / (2 * η * k) := by
      have h3 : f (gradientDescentIterates f η x₀ k) - f x_star =
          ((2 * η * k) * (f (gradientDescentIterates f η x₀ k) - f x_star)) / (2 * η * k) := by
        field_simp
      rw [h3]
      exact div_le_div_of_nonneg_right h1 (le_of_lt h_denom_pos)
    exact h2
  exact h_final

/-- Linear convergence for strongly convex smooth functions.
    After k iterations: ‖x_k - x*‖² ≤ (1 - μ/L)^k ‖x₀ - x*‖²

## Proof Strategy

For strongly convex and L-smooth functions with step size η = 1/L:

1. **Contraction per iteration**: Each gradient descent step contracts the distance to optimum
   by a factor of (1 - μ/L), i.e., ‖x_{k+1} - x*‖² ≤ (1 - μ/L)‖x_k - x*‖²

2. **Key ingredients**:
   - L-smoothness provides descent lemma: f(x - η∇f(x)) ≤ f(x) - (η/2)‖∇f(x)‖²
   - μ-strong convexity ensures: f(x*) + (μ/2)‖x - x*‖² ≤ f(x) + ⟨∇f(x), x* - x⟩
   - At optimum: ∇f(x*) = 0

3. **Per-step contraction lemma**: From strong convexity and smoothness
   ‖x_{k+1} - x*‖² = ‖x_k - η∇f(x_k) - x*‖²
                     ≤ (1 - μ/L)‖x_k - x*‖²

4. **Telescoping**: Apply contraction k times to get:
   ‖x_k - x*‖² ≤ (1 - μ/L)^k ‖x₀ - x*‖²

This is the classical result for strongly convex optimization.
-/
theorem strongly_convex_linear_convergence (f : E → ℝ) (L μ : ℝ)
    (hL : 0 < L) (hμ : 0 < μ) (hμL : μ ≤ L)
    (hSmooth : IsLSmooth f L) (hStrong : IsStronglyConvex f μ)
    (x_star : E) (hMin : gradient f x_star = 0)
    (η : ℝ) (hη : η = 1 / L) (x₀ : E) :
    ∀ k : ℕ, ‖gradientDescentIterates f η x₀ k - x_star‖^2 ≤
      (1 - μ / L)^k * ‖x₀ - x_star‖^2 := by
  -- We proceed by induction on k
  intro k
  induction k with
  | zero =>
    -- Base case: k = 0
    -- gradientDescentIterates f η x₀ 0 = x₀
    -- ‖x₀ - x_star‖² ≤ (1 - μ/L)^0 * ‖x₀ - x_star‖²
    -- This simplifies to ‖x₀ - x_star‖² ≤ ‖x₀ - x_star‖²
    simp only [gradientDescentIterates, pow_zero, one_mul]
    exact le_refl _
  | succ k ih =>
    -- Inductive case: assume ‖x_k - x*‖² ≤ (1 - μ/L)^k ‖x₀ - x*‖²
    -- Need to show: ‖x_{k+1} - x*‖² ≤ (1 - μ/L)^{k+1} ‖x₀ - x*‖²
    let x_k := gradientDescentIterates f η x₀ k
    let x_k1 := gradientDescentIterates f η x₀ (k + 1)
    -- Key: x_{k+1} = x_k - η∇f(x_k)
    have h_step : x_k1 = x_k - η • gradient f x_k := rfl
    -- The per-iteration contraction: ‖x_{k+1} - x*‖² ≤ (1 - μ/L) ‖x_k - x*‖²
    --
    -- Proof outline:
    -- 1. Expand: ‖x_{k+1} - x*‖² = ‖(x_k - x*) - η∇f(x_k)‖²
    --    = ‖x_k - x*‖² - 2η⟨∇f(x_k), x_k - x*⟩ + η²‖∇f(x_k)‖²
    --
    -- 2. For μ-strongly convex f with minimum at x*:
    --    ⟨∇f(x_k), x_k - x*⟩ ≥ μ‖x_k - x*‖² + (f(x_k) - f(x*))
    --    (This is the "strong convexity gradient inequality")
    --
    -- 3. For L-smooth f:
    --    ‖∇f(x_k)‖² ≤ 2L(f(x_k) - f(x*))
    --    (Co-coercivity of gradient)
    -- 4. Combining with η = 1/L:
    --    ‖x_{k+1} - x*‖² ≤ ‖x_k - x*‖² - 2η·μ‖x_k - x*‖²
    --                      - 2η(f(x_k) - f(x*)) + η²·2L(f(x_k) - f(x*))
    --    = ‖x_k - x*‖² - (2μ/L)‖x_k - x*‖²
    --    = (1 - 2μ/L)‖x_k - x*‖² ≤ (1 - μ/L)‖x_k - x*‖²  (since 2μ/L ≥ μ/L)
    -- The formal proof requires the following key lemmas:

    -- Lemma 1: Strong convexity gradient inequality
    -- For μ-strongly convex f with ∇f(x*) = 0:
    -- ⟨∇f(x), x - x*⟩ ≥ μ‖x - x*‖² + (f(x) - f(x*))
    --
    -- This follows from the strong convexity definition:
    -- f(y) ≥ f(x) + ⟨∇f(x), y-x⟩ + (μ/2)‖x-y‖²
    -- Setting y = x* and using f(x*) ≤ f(x) + ⟨∇f(x), x*-x⟩ + (μ/2)‖x-x*‖²

    -- Lemma 2: Co-coercivity of L-smooth gradients
    -- For L-smooth f with ∇f(x*) = 0:
    -- ‖∇f(x)‖² ≤ 2L(f(x) - f(x*))
    --
    -- This follows from the descent lemma applied at x:
    -- f(x - (1/L)∇f(x)) ≤ f(x) - (1/2L)‖∇f(x)‖²
    -- Since f(x*) is the minimum: f(x*) ≤ f(x - (1/L)∇f(x))
    -- Therefore: f(x*) ≤ f(x) - (1/2L)‖∇f(x)‖²
    -- Rearranging: ‖∇f(x)‖² ≤ 2L(f(x) - f(x*))
    have h_contraction : ‖x_k1 - x_star‖^2 ≤ (1 - μ / L) * ‖x_k - x_star‖^2 := by
      -- Let g = ∇f(x_k)
      let g := gradient f x_k
      -- x_{k+1} - x* = (x_k - x*) - η·g
      have h_diff : x_k1 - x_star = (x_k - x_star) - η • g := by
        simp only [h_step]
        abel
      -- ‖x_{k+1} - x*‖² = ‖(x_k - x*) - η·g‖²
      --                  = ‖x_k - x*‖² - 2η⟨g, x_k - x*⟩ + η²‖g‖²
      have h_expand : ‖x_k1 - x_star‖^2 =
          ‖x_k - x_star‖^2 - 2 * η * @inner ℝ E _ g (x_k - x_star) + η^2 * ‖g‖^2 := by
        rw [h_diff]
        -- Use polarization: ‖a - b‖² = ‖a‖² + ‖b‖² - 2⟨a, b⟩
        -- ‖a - η • g‖² = ‖a‖² + ‖η • g‖² - 2⟨a, η • g⟩
        --              = ‖a‖² + η²‖g‖² - 2η⟨a, g⟩
        --              = ‖a‖² - 2η⟨g, a⟩ + η²‖g‖² (by inner product symmetry)
        rw [norm_sub_sq_real]
        -- ‖η • g‖² = (|η| * ‖g‖)² = |η|² * ‖g‖² = η² * ‖g‖²
        have h_norm_smul_sq : ‖η • g‖^2 = η^2 * ‖g‖^2 := by
          rw [norm_smul, Real.norm_eq_abs, mul_pow, sq_abs]
        rw [h_norm_smul_sq]
        -- ⟨a, η • g⟩ = η * ⟨a, g⟩ = η * ⟨g, a⟩ (by symmetry)
        rw [inner_smul_right, real_inner_comm]
        ring
      -- Now use h_expand and bound each term. From η = 1/L:
      have h_eta : η = 1 / L := hη
      have h_eta_sq : η^2 = 1 / L^2 := by rw [h_eta]; ring
      -- Use the interpolation condition which combines strong convexity and smoothness
      have h_interp := strong_smooth_interpolation f L μ hL hμ hSmooth hStrong x_k x_star hMin
      -- Let inner_val = ⟨g, x_k - x*⟩ for clarity
      let inner_val := @inner ℝ E _ g (x_k - x_star)
      -- From h_expand: ‖x_{k+1} - x*‖² = ‖x_k - x*‖² - 2η·inner_val + η²‖g‖²
      -- With η = 1/L: = ‖x_k - x*‖² - (2/L)·inner_val + (1/L²)‖g‖²
      -- From interpolation: inner_val ≥ (μL)/(μ+L)‖x_k - x*‖² + 1/(μ+L)‖g‖²
      -- So: -(2/L)·inner_val ≤ -(2/L)·[(μL)/(μ+L)‖x_k - x*‖² + 1/(μ+L)‖g‖²]
      --                      = -(2μ)/(μ+L)‖x_k - x*‖² - 2/(L(μ+L))‖g‖². Combined:
      -- ‖x_{k+1} - x*‖² ≤ ‖x_k - x*‖² - (2μ)/(μ+L)‖x_k - x*‖² + [1/L² - 2/(L(μ+L))]‖g‖²
      --
      -- The coefficient of ‖g‖²:
      -- 1/L² - 2/(L(μ+L)) = [(μ+L) - 2L] / [L²(μ+L)] = (μ-L) / [L²(μ+L)] ≤ 0 (since μ ≤ L)
      --
      -- So we can drop the ‖g‖² term:
      -- ‖x_{k+1} - x*‖² ≤ ‖x_k - x*‖² - (2μ)/(μ+L)‖x_k - x*‖²
      --                = [1 - 2μ/(μ+L)]‖x_k - x*‖²
      --                = [(μ+L-2μ)/(μ+L)]‖x_k - x*‖²
      --                = [(L-μ)/(L+μ)]‖x_k - x*‖²
      --
      -- Finally: (L-μ)/(L+μ) ≤ 1 - μ/L because:
      -- (L-μ)/(L+μ) ≤ (L-μ)/L = 1 - μ/L iff L+μ ≥ L, which is true since μ > 0
      have h_coeff_neg : 1 / L^2 - 2 / (L * (μ + L)) ≤ 0 := by
        have h3 : 1 / L^2 - 2 / (L * (μ + L)) = (μ - L) / (L^2 * (μ + L)) := by field_simp; ring
        rw [h3]
        apply div_nonpos_of_nonpos_of_nonneg
        · linarith  -- μ - L ≤ 0 since μ ≤ L
        · apply mul_nonneg (sq_nonneg L)
          linarith  -- μ + L > 0
      have h_contraction_factor : (L - μ) / (L + μ) ≤ 1 - μ / L := by
        have h1 : (L - μ) / (L + μ) ≤ (L - μ) / L := by
          apply div_le_div_of_nonneg_left
          · linarith  -- L - μ ≥ 0
          · linarith  -- L > 0
          · linarith  -- L + μ ≥ L
        have h2 : (L - μ) / L = 1 - μ / L := by field_simp
        linarith
      -- Chain h_expand with h_interp and algebraic bounds
      -- Goal: ‖x_k1 - x_star‖^2 ≤ (1 - μ / L) * ‖x_k - x_star‖^2
      --
      -- From h_expand (with η = 1/L):
      -- ‖x_k1 - x_star‖^2 = ‖x_k - x_star‖^2 - (2/L)⟨g, x_k - x*⟩ + (1/L²)‖g‖²
      --
      -- From h_interp (assuming strong_smooth_interpolation is proved):
      -- ⟨g, x_k - x*⟩ ≥ (μL)/(μ+L)‖x_k - x*‖² + 1/(μ+L)‖g‖²
      --
      -- Substituting:
      -- ‖x_k1 - x_star‖^2 ≤ ‖x_k - x*‖² - (2/L)·[(μL)/(μ+L)‖x_k - x*‖² + 1/(μ+L)‖g‖²] + (1/L²)‖g‖²
      --                    = ‖x_k - x*‖² - (2μ)/(μ+L)‖x_k - x*‖² + [1/L² - 2/(L(μ+L))]‖g‖²
      --
      -- By h_coeff_neg, the coefficient of ‖g‖² is ≤ 0, and ‖g‖² ≥ 0, so:
      -- ‖x_k1 - x_star‖^2 ≤ ‖x_k - x*‖² - (2μ)/(μ+L)‖x_k - x*‖²
      --                    = [1 - 2μ/(μ+L)]‖x_k - x*‖²
      --                    = [(L-μ)/(L+μ)]‖x_k - x*‖²
      --
      -- By h_contraction_factor: (L-μ)/(L+μ) ≤ 1 - μ/L
      -- First compute the coefficient 1 - 2μ/(μ+L) = (L-μ)/(L+μ)
      have h_coeff_eq : 1 - 2 * μ / (μ + L) = (L - μ) / (L + μ) := by
        field_simp
        ring
      -- Combine everything using transitivity. The proof uses
      -- strong_smooth_interpolation (now fully proved).
      -- Key inequality from h_interp:
      have h_inner_bound : inner_val ≥ (μ * L) / (μ + L) * ‖x_k - x_star‖^2 +
                                        1 / (μ + L) * ‖g‖^2 := h_interp
      -- Substitute η = 1/L into h_expand
      have h_expand' : ‖x_k1 - x_star‖^2 =
          ‖x_k - x_star‖^2 - 2 / L * inner_val + 1 / L^2 * ‖g‖^2 := by
        rw [h_expand, h_eta]; ring
      -- Apply the bound on inner_val
      have h_step1 : ‖x_k1 - x_star‖^2 ≤
          ‖x_k - x_star‖^2 - 2 / L * ((μ * L) / (μ + L) * ‖x_k - x_star‖^2 +
                                       1 / (μ + L) * ‖g‖^2) + 1 / L^2 * ‖g‖^2 := by
        rw [h_expand']
        have h_L_pos : 0 < L := hL
        have h_2L_pos : 0 < 2 / L := by positivity
        nlinarith [h_inner_bound, sq_nonneg ‖g‖, sq_nonneg ‖x_k - x_star‖]
      -- Simplify to get the coefficient form
      have h_step2 : ‖x_k1 - x_star‖^2 ≤
          ‖x_k - x_star‖^2 - 2 * μ / (μ + L) * ‖x_k - x_star‖^2 +
          (1 / L^2 - 2 / (L * (μ + L))) * ‖g‖^2 := by
        calc ‖x_k1 - x_star‖^2
            ≤ ‖x_k - x_star‖^2 - 2 / L * ((μ * L) / (μ + L) * ‖x_k - x_star‖^2 +
                                           1 / (μ + L) * ‖g‖^2) + 1 / L^2 * ‖g‖^2 := h_step1
          _ = ‖x_k - x_star‖^2 - 2 * μ / (μ + L) * ‖x_k - x_star‖^2 +
              (1 / L^2 - 2 / (L * (μ + L))) * ‖g‖^2 := by
            have hL_ne : L ≠ 0 := ne_of_gt hL
            have hμL_ne : μ + L ≠ 0 := by linarith
            field_simp
            ring
      -- Drop the ‖g‖² term (coefficient is ≤ 0)
      have h_step3 : ‖x_k1 - x_star‖^2 ≤
          ‖x_k - x_star‖^2 - 2 * μ / (μ + L) * ‖x_k - x_star‖^2 := by
        have h_g_sq_nonneg : 0 ≤ ‖g‖^2 := sq_nonneg _
        nlinarith [h_step2, h_coeff_neg, h_g_sq_nonneg]
      -- Factor and apply contraction bound
      calc ‖x_k1 - x_star‖^2
          ≤ ‖x_k - x_star‖^2 - 2 * μ / (μ + L) * ‖x_k - x_star‖^2 := h_step3
        _ = (1 - 2 * μ / (μ + L)) * ‖x_k - x_star‖^2 := by ring
        _ = (L - μ) / (L + μ) * ‖x_k - x_star‖^2 := by rw [h_coeff_eq]
        _ ≤ (1 - μ / L) * ‖x_k - x_star‖^2 := by
            apply mul_le_mul_of_nonneg_right h_contraction_factor (sq_nonneg _)
    -- Apply contraction and inductive hypothesis
    calc ‖x_k1 - x_star‖^2
        ≤ (1 - μ / L) * ‖x_k - x_star‖^2 := h_contraction
      _ ≤ (1 - μ / L) * ((1 - μ / L)^k * ‖x₀ - x_star‖^2) := by {
          apply mul_le_mul_of_nonneg_left ih
          have h1 : μ / L ≤ 1 := (div_le_one (by linarith : 0 < L)).mpr hμL
          linarith
        }
      _ = (1 - μ / L)^(k + 1) * ‖x₀ - x_star‖^2 := by ring

end Gradient
