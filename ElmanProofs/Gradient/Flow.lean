/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Convex.Basic
import Mathlib.Analysis.Calculus.MeanValue
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

  -- The formal proof for x ≠ y requires Mathlib's integration machinery.
  -- Key steps outlined above.

  sorry

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

  5. **Key Challenge**:
     To complete the proof rigorously, we would need to:
     - Establish descent_lemma (currently sorry in this file)
     - Formalize the first-order convexity condition in Lean
     - Handle the case analysis on whether ‖x_i - x*‖ > 0
     - Carefully track all inequality directions in the telescoping sum
     - Prove that the sum ∑ᵢ (f(x_i) - f(x*)) telescopes properly

  Given that descent_lemma is marked sorry, completing this proof requires
  first establishing the foundation. The mathematical argument is sound but
  requires careful formalization of convex analysis results in Lean 4.
  -/

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
    have h_step : x_k1 = x_k - η • gradient f x_k := by
      simp only [gradientDescentIterates, gradientDescentStep]

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
    --
    -- 4. Combining with η = 1/L:
    --    ‖x_{k+1} - x*‖² ≤ ‖x_k - x*‖² - 2η·μ‖x_k - x*‖² - 2η(f(x_k) - f(x*)) + η²·2L(f(x_k) - f(x*))
    --    = ‖x_k - x*‖² - (2μ/L)‖x_k - x*‖² - (2/L)(f(x_k) - f(x*)) + (2/L)(f(x_k) - f(x*))
    --    = (1 - 2μ/L)‖x_k - x*‖²
    --    ≤ (1 - μ/L)‖x_k - x*‖²  (since 2μ/L ≥ μ/L)

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
        -- ‖a - η·g‖² = ‖a‖² - 2η⟨g,a⟩ + η²‖g‖² where a = x_k - x_star
        -- This is the polarization identity for inner product spaces
        sorry

      -- The rest requires Lemmas 1 and 2 above to bound the RHS by (1 - μ/L)‖x_k - x*‖²
      sorry

    -- Apply contraction and inductive hypothesis
    calc ‖x_k1 - x_star‖^2
        ≤ (1 - μ / L) * ‖x_k - x_star‖^2 := h_contraction
      _ ≤ (1 - μ / L) * ((1 - μ / L)^k * ‖x₀ - x_star‖^2) := by {
          apply mul_le_mul_of_nonneg_left ih
          have h1 : μ / L ≤ 1 := div_le_one_of_le hμL (le_of_lt hL)
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
  have h_diff : y - x = -(η • g) := by
    show (x - η • g) - x = -(η • g)
    abel

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
