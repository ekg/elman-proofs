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

  -- The proof uses the first-order characterization of strong convexity.
  -- The key steps are documented in the comments above.
  -- Formalizing the derivative limit argument requires careful handling of
  -- the strong convexity bound as t → 0⁺, which involves Mathlib's asymptotic analysis.
  -- The key bound is: ⟨∇f(x), x - x*⟩ ≥ f(x) - f(x*) + (μ/2)‖x - x*‖² ≥ (μ/2)‖x - x*‖²
  -- where the second inequality follows from x* being the global minimum
  -- (since ∇f(x*) = 0 for strongly convex f implies x* is the unique minimizer).
  sorry

/-- Interpolation condition for strongly convex AND smooth functions.

    For μ-strongly convex and L-smooth f with ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ (μL)/(μ+L) ‖x - x*‖² + 1/(μ+L) ‖∇f(x)‖²

    This is stronger than using strong convexity or smoothness alone.
    It's the key to achieving the optimal (1 - μ/L) contraction rate.
-/
theorem strong_smooth_interpolation (f : E → ℝ) (L μ : ℝ) (hL : 0 < L) (hμ : 0 < μ)
    (hSmooth : IsLSmooth f L) (hStrong : IsStronglyConvex f μ)
    (x x_star : E) (hMin : gradient f x_star = 0) :
    @inner ℝ E _ (gradient f x) (x - x_star) ≥
      (μ * L) / (μ + L) * ‖x - x_star‖^2 + 1 / (μ + L) * ‖gradient f x‖^2 := by
  -- This is the interpolation condition for functions that are BOTH strongly convex
  -- AND smooth. It provides a tighter bound than either alone.
  --
  -- **Available ingredients**:
  -- 1. Strong convexity (gradient monotonicity): ⟨∇f(x) - ∇f(y), x - y⟩ ≥ μ‖x - y‖²
  -- 2. Co-coercivity (from L-smoothness): ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (1/L)‖∇f(x) - ∇f(y)‖²
  --
  -- **The interpolation condition**:
  -- ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (μL)/(μ+L)‖x - y‖² + 1/(μ+L)‖∇f(x) - ∇f(y)‖²
  --
  -- **Proof strategy**:
  -- The key is to use BOTH conditions simultaneously in an optimal way.
  --
  -- Consider the auxiliary function: h(x) = f(x) - (μ/2)‖x‖²
  -- Since f is μ-strongly convex, h is convex.
  -- Since f is L-smooth, h is (L-μ)-smooth.
  -- Apply co-coercivity to h at the optimum.
  --
  -- Alternatively, use the proximal operator characterization:
  -- For the proximal of f at x with parameter 1/L:
  -- prox_{f/L}(x) = argmin_z [f(z) + (L/2)‖z - x‖²]
  --
  -- **Simplified proof when y = x* (∇f(x*) = 0)**:
  -- Let g = ∇f(x). We need:
  -- ⟨g, x - x*⟩ ≥ (μL)/(μ+L)‖x - x*‖² + 1/(μ+L)‖g‖²
  --
  -- From strong convexity at x*: ⟨g, x - x*⟩ ≥ μ‖x - x*‖² (using ∇f(x*) = 0)
  -- From co-coercivity: ⟨g, x - x*⟩ ≥ (1/L)‖g‖² (using ∇f(x*) = 0)
  --
  -- The weighted combination uses both:
  -- (μ+L)⟨g, x - x*⟩ = L⟨g, x - x*⟩ + μ⟨g, x - x*⟩
  --                   ≥ L·μ‖x - x*‖² + μ·(1/L)‖g‖²
  --                   = μL‖x - x*‖² + (μ/L)‖g‖²
  --
  -- This gives: ⟨g, x - x*⟩ ≥ (μL)/(μ+L)‖x - x*‖² + μ/(L(μ+L))‖g‖²
  --
  -- The coefficient μ/(L(μ+L)) is weaker than 1/(μ+L) when μ < L (typical case).
  -- The sharper bound requires the full interpolation argument using:
  -- - The Fenchel conjugate f* which is (1/μ)-smooth and (1/L)-strongly convex
  -- - Or the "operator splitting" viewpoint
  --
  -- For our purposes in the convergence theorem, the weaker bound suffices
  -- since we only need ⟨g, x - x*⟩ ≥ c₁‖x - x*‖² + c₂‖g‖² for some c₁, c₂ > 0.

  sorry

/-- Co-coercivity of L-smooth gradients (Baillon-Haddad theorem).

    For L-smooth f: ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (1/L)‖∇f(x) - ∇f(y)‖²

    With y = x* where ∇f(x*) = 0:
    ⟨∇f(x), x - x*⟩ ≥ (1/L)‖∇f(x)‖²

    Equivalently: ‖∇f(x)‖² ≤ L⟨∇f(x), x - x*⟩

    ## Proof Outline

    **Method 1: Via descent lemma**

    From the descent lemma with step size 1/L:
    f(x - (1/L)∇f(x)) ≤ f(x) - (1/(2L))‖∇f(x)‖²

    Since x* minimizes f:
    f(x*) ≤ f(x - (1/L)∇f(x)) ≤ f(x) - (1/(2L))‖∇f(x)‖²

    Also from L-smoothness at x*:
    f(x) ≤ f(x*) + ⟨∇f(x*), x - x*⟩ + (L/2)‖x - x*‖²
         = f(x*) + (L/2)‖x - x*‖²  (since ∇f(x*) = 0)

    Combining and using strong convexity-type arguments gives the result.

    **Method 2: Direct from Baillon-Haddad**

    The general Baillon-Haddad theorem states that for L-smooth f:
    ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (1/L)‖∇f(x) - ∇f(y)‖²

    Setting y = x* with ∇f(x*) = 0 gives the result.
-/
theorem lsmooth_cocoercivity (f : E → ℝ) (L : ℝ) (hL : 0 < L)
    (hSmooth : IsLSmooth f L) (x x_star : E) (hMin : gradient f x_star = 0) :
    ‖gradient f x‖^2 ≤ L * @inner ℝ E _ (gradient f x) (x - x_star) := by
  -- Setting y = x* in the general Baillon-Haddad theorem:
  -- ⟨∇f(x) - ∇f(x*), x - x*⟩ ≥ (1/L)‖∇f(x) - ∇f(x*)‖²
  -- Since ∇f(x*) = 0:
  -- ⟨∇f(x), x - x*⟩ ≥ (1/L)‖∇f(x)‖²
  -- Multiplying by L:
  -- L⟨∇f(x), x - x*⟩ ≥ ‖∇f(x)‖²

  -- The Baillon-Haddad theorem itself requires proving:
  -- ⟨∇f(x) - ∇f(y), x - y⟩ ≥ (1/L)‖∇f(x) - ∇f(y)‖²
  -- This is a deep result about L-smooth functions.

  -- **Proof of Baillon-Haddad (general form)**
  --
  -- Define h(z) = f(z) - (1/2L)‖∇f(z)‖²
  --
  -- Claim: h is convex if f is L-smooth.
  --
  -- Proof: We need to show h(αx + (1-α)y) ≤ αh(x) + (1-α)h(y) for α ∈ [0,1].
  --
  -- This follows from the descent lemma applied to f and its conjugate properties.
  -- Specifically, for L-smooth f:
  -- f(y) ≤ f(x) + ⟨∇f(x), y-x⟩ + (L/2)‖y-x‖²
  --
  -- Applying this at x = z - (1/L)∇f(z) shows that the gradient step
  -- decreases f by at least (1/2L)‖∇f(z)‖².
  --
  -- The convexity of h then gives:
  -- ⟨∇h(x) - ∇h(y), x - y⟩ ≥ 0
  -- which expands to the Baillon-Haddad inequality.
  --
  -- **Alternative proof via descent lemma**
  --
  -- From the descent lemma: f(x - (1/L)∇f(x)) ≤ f(x) - (1/2L)‖∇f(x)‖²
  -- Since x* is the minimum: f(x*) ≤ f(x - (1/L)∇f(x))
  --
  -- Also from L-smoothness at x*:
  -- f(x) ≤ f(x*) + ⟨∇f(x*), x - x*⟩ + (L/2)‖x - x*‖²
  --      = f(x*) + (L/2)‖x - x*‖²  (since ∇f(x*) = 0)
  --
  -- Combining these with the standard Cauchy-Schwarz argument gives the result.

  sorry

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

/-- One step of gradient descent with learning rate η. -/
noncomputable def gradientDescentStep (f : E → ℝ) (η : ℝ) (x : E) : E :=
  x - η • gradient f x

/-- k steps of gradient descent. -/
noncomputable def gradientDescentIterates (f : E → ℝ) (η : ℝ) (x₀ : E) : ℕ → E
  | 0 => x₀
  | n + 1 => gradientDescentStep f η (gradientDescentIterates f η x₀ n)

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
     - `lsmooth_fundamental_ineq` (sorry) - needed for descent_lemma
     - `descent_lemma` (complete, uses lsmooth_fundamental_ineq)
     - First-order convexity characterization from Mathlib's ConvexOn
     - Telescoping sum machinery

     Once `lsmooth_fundamental_ineq` is proved, the descent_lemma becomes available
     and this theorem can be completed using standard convex optimization arguments.
  -/

  -- The proof structure:
  -- 1. Apply descent_lemma k times to get:
  --    f(x_k) ≤ f(x₀) - (η/2) ∑ᵢ ‖∇f(xᵢ)‖²
  --
  -- 2. Use first-order convexity: f(x) - f(x*) ≤ ⟨∇f(x), x - x*⟩
  --    This follows from ConvexOn hypothesis
  --
  -- 3. Apply Cauchy-Schwarz and average:
  --    (1/k) ∑ᵢ (f(xᵢ) - f(x*)) ≤ (1/k) ∑ᵢ ⟨∇f(xᵢ), xᵢ - x*⟩
  --
  -- 4. Use the gradient descent recurrence to show:
  --    ∑ᵢ ⟨∇f(xᵢ), xᵢ - x*⟩ ≤ ‖x₀ - x*‖² / (2η)

  sorry

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
      -- Combine everything using transitivity. The proof depends on
      -- strong_smooth_interpolation which currently has a sorry.
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

end Gradient
