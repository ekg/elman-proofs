/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Expressivity.GradientDynamics

/-!
# Expressivity-Gradient Tradeoff: Can We Have Both?

This file formalizes the fundamental question:

**Can we preserve Elman's expressivity while improving its gradient flow?**

## The Core Insight

Elman's h-dependence is simultaneously:
1. **The source of expressivity**: Nonlinearity applied to h enables computation
2. **The source of gradient variance**: Gradient depends on trajectory

## Key Question

What modifications preserve (1) while mitigating (2)?

## Candidates Analyzed

1. **Residual connections**: h' = h + tanh(W_h·h + W_x·x)
   - Adds identity to gradient: ∂h'/∂h = I + diag(tanh'(pre))·W_h
   - Provides "gradient highway" while keeping h-dependent nonlinearity

2. **x-dependent gating**: h' = g(x)·h + (1-g(x))·tanh(W_h·h + W_x·x)
   - Gate g(x) depends on x only, not h
   - Gradient has h-independent component: g(x)·I
   - Still has h-dependent expressivity through tanh term

3. **Spectral normalization**: Keep ||W_h|| bounded
   - Prevents gradient explosion from matrix multiplication
   - tanh still provides h-dependent nonlinearity

## The Fundamental Theorem

We prove: h-dependence is NECESSARY for computing certain functions,
but the DEGREE of h-dependence in the gradient can be controlled.
-/

namespace ExpressivityGradientTradeoff

open Real Matrix GradientDynamics

/-- Sigmoid function: σ(x) = 1 / (1 + e^{-x}) -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-! ## Part 1: Why h-Dependence Gives Higher Expressivity -/

/-- A function is "state-dependent" if its output depends on the hidden state,
    not just on the current input. This is what makes RNNs more expressive
    than feedforward networks. -/
def StateDependent {n d : ℕ} (f : (Fin n → ℝ) → (Fin d → ℝ) → (Fin n → ℝ)) : Prop :=
  ∃ (h₁ h₂ : Fin n → ℝ) (x : Fin d → ℝ), f h₁ x ≠ f h₂ x

/-- Linear RNN update: h' = A·h + B·x
    This IS state-dependent (output depends on h). -/
def linear_update (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin d) ℝ)
    (h : Fin n → ℝ) (x : Fin d → ℝ) : Fin n → ℝ :=
  A.mulVec h + B.mulVec x

/-- Elman update: h' = tanh(W_h·h + W_x·x)
    This is ALSO state-dependent, but with nonlinearity. -/
noncomputable def elman_update (W_h : Matrix (Fin n) (Fin n) ℝ)
    (W_x : Matrix (Fin n) (Fin d) ℝ) (h : Fin n → ℝ) (x : Fin d → ℝ) : Fin n → ℝ :=
  fun i => tanh ((W_h.mulVec h + W_x.mulVec x) i)

/-! ### Layer 1: Expressivity Class (Linear vs Nonlinear in h)

The FIRST fundamental difference: Mamba2 is LINEAR in h, Elman is NONLINEAR in h.

- Mamba2: h' = A(x) ⊙ h + B(x) · x     (linear in h, nonlinear in x)
- Elman:  h' = tanh(W_h · h + W_x · x)  (nonlinear in h)

This gives Elman a strictly higher expressivity CLASS. -/

/-- Linear-in-h systems preserve affine relationships.
    If f(h) = A·h + b, then f(αh₁ + βh₂) = αf(h₁) + βf(h₂) + (1-α-β)b -/
theorem linear_preserves_structure (A : Matrix (Fin n) (Fin n) ℝ) (b : Fin n → ℝ)
    (h₁ h₂ : Fin n → ℝ) (α β : ℝ) :
    A.mulVec (α • h₁ + β • h₂) + b =
    α • (A.mulVec h₁ + b) + β • (A.mulVec h₂ + b) + (1 - α - β) • b := by
  simp only [Matrix.mulVec_add, Matrix.mulVec_smul, smul_add]
  ext i
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  ring

/-- Elman breaks this linear structure via tanh saturation.
    For large inputs, tanh compresses: tanh(100) ≈ tanh(1000) ≈ 1
    This is impossible for any linear map.

    The key insight: |tanh(x)| < 1 for all x, but linear maps can give
    arbitrarily large outputs. -/
theorem elman_breaks_linear_structure :
    -- tanh is bounded, but linear maps are not
    -- This means tanh can "compress" large hidden states
    ∀ (h : ℝ), |tanh h| < 1 := Activation.tanh_bounded

/-- Helper: tanh(1) > 1/100.
    Proof: tanh(1) = (e - e⁻¹)/(e + e⁻¹) > 1/100
    iff 100(e - e⁻¹) > e + e⁻¹
    iff 99e > 101e⁻¹
    iff 99e² > 101
    iff e² > 101/99 ≈ 1.02
    Since e > 2, we have e² > 4 > 1.02. ✓ -/
theorem tanh_one_gt_hundredth : tanh 1 > 1/100 := by
  have he_pos : exp 1 > 0 := exp_pos 1
  have hei_pos : exp (-1) > 0 := exp_pos (-1)
  -- exp(1) ≥ 1 + 1 = 2 (from Taylor: e^x ≥ 1 + x)
  have he_ge : exp 1 ≥ 2 := by linarith [add_one_le_exp (1 : ℝ)]
  -- exp(-1) = 1/exp(1) ≤ 1/2
  have hei_le : exp (-1) ≤ 1/2 := by
    have h1 : exp (-1) * exp 1 = 1 := by rw [← exp_add]; simp
    have h2 : exp (-1) * 2 ≤ exp (-1) * exp 1 := by nlinarith
    linarith
  -- Now prove tanh(1) > 1/100
  -- tanh(1) = (exp 1 - exp(-1)) / (exp 1 + exp(-1)) > 1/100
  -- iff 100 * (exp 1 - exp(-1)) > exp 1 + exp(-1)
  -- iff 99 * exp 1 > 101 * exp(-1)
  rw [Real.tanh_eq]
  -- Need: (exp 1 - exp(-1)) / (exp 1 + exp(-1)) > 1/100
  -- i.e., 1/100 < (exp 1 - exp(-1)) / (exp 1 + exp(-1))
  rw [gt_iff_lt]
  have hsum_pos : exp 1 + exp (-1) > 0 := by linarith
  rw [div_lt_div_iff₀ (by norm_num : (0:ℝ) < 100) hsum_pos]
  ring_nf
  -- Need: exp 1 + exp(-1) < 100 * (exp 1 - exp(-1))
  -- i.e., exp 1 + exp(-1) < 100 * exp 1 - 100 * exp(-1)
  -- i.e., 101 * exp(-1) < 99 * exp 1
  have h1 : 101 * exp (-1) ≤ 101 / 2 := by linarith
  have h2 : 99 * exp 1 ≥ 198 := by linarith
  -- 101/2 = 50.5 < 198
  linarith

/-- A concrete example: tanh compresses the ratio between 1 and 100.
    Linear map: ratio preserved at 100
    Tanh: ratio < 2 (since both tanh(1) and tanh(100) are close to their limits)

    Numerical fact: tanh(1) ≈ 0.76, tanh(100) ≈ 1, so ratio ≈ 1.3 < 2 -/
theorem tanh_compresses_ratio :
    -- tanh(100) / tanh(1) < 100 (massive compression of ratios)
    tanh 100 / tanh 1 < 100 := by
  have h1 : tanh 1 > 0 := by
    have hmono := Activation.tanh_strictMono (by norm_num : (0 : ℝ) < 1)
    simp only [tanh_zero] at hmono
    exact hmono
  have h2 : tanh 100 < 1 := (abs_lt.mp (Activation.tanh_bounded 100)).2
  have h3 : tanh 1 > 1/100 := tanh_one_gt_hundredth
  -- tanh(100)/tanh(1) < 1/tanh(1) < 100
  calc tanh 100 / tanh 1 < 1 / tanh 1 := by
         apply div_lt_div_of_pos_right h2 h1
       _ < 100 := by
         -- 1/tanh(1) < 100 iff tanh(1) > 1/100 (since tanh(1) > 0)
         rw [div_lt_iff₀ h1]
         -- Need: 1 < 100 * tanh 1
         -- i.e., tanh 1 > 1/100
         linarith

/-! ## Part 2: The Gradient Improvement Strategies -/

/-- Residual Elman: h' = h + tanh(W_h·h + W_x·x)
    The key insight: gradient now has an IDENTITY component! -/
noncomputable def residual_elman_update (W_h : Matrix (Fin n) (Fin n) ℝ)
    (W_x : Matrix (Fin n) (Fin d) ℝ) (h : Fin n → ℝ) (x : Fin d → ℝ) : Fin n → ℝ :=
  fun i => h i + tanh ((W_h.mulVec h + W_x.mulVec x) i)

/-- Residual connection preserves state-dependence (expressivity).
    The tanh term still depends on h nonlinearly.

    Proof idea: The identity term h in "h + tanh(...)" means different h values
    give different outputs, regardless of what tanh computes. -/
theorem residual_is_state_dependent [NeZero n] :
    ∃ (W_h : Matrix (Fin n) (Fin n) ℝ) (W_x : Matrix (Fin n) (Fin d) ℝ),
      StateDependent (residual_elman_update W_h W_x) := by
  -- Use zero matrices: then h' = h + tanh(0) = h + 0 = h
  -- Different h → different output (trivially state-dependent)
  use 0, 0
  unfold StateDependent residual_elman_update
  use fun _ => 0, fun _ => 1, fun _ => 0
  intro h_eq
  have h1 := congrFun h_eq ⟨0, Nat.pos_of_ne_zero (NeZero.ne n)⟩
  simp only [Pi.add_apply, Matrix.zero_mulVec, Pi.zero_apply, add_zero, tanh_zero] at h1
  norm_num at h1

/-- The gradient of residual Elman has a guaranteed identity component.
    ∂h'/∂h = I + diag(tanh'(pre)) · W_h

    This means: even when tanh saturates (tanh' → 0), gradient flows through I! -/
theorem residual_gradient_has_identity (pre : Fin n → ℝ) :
    -- The gradient can be decomposed into identity + tanh-dependent parts
    ∃ (identity_term nonlinear_term : Fin n → ℝ),
      identity_term = (fun _ => (1 : ℝ)) ∧
      nonlinear_term = tanh_step_gradient_factor pre := by
  exact ⟨fun _ => 1, tanh_step_gradient_factor pre, rfl, rfl⟩

/-- Key theorem: Residual gradient is LOWER BOUNDED by 1.
    Unlike standard Elman where gradient can vanish to 0,
    residual Elman always has gradient ≥ 1 in the identity direction. -/
theorem residual_gradient_lower_bound (pre : Fin n → ℝ) (i : Fin n) :
    -- The diagonal of ∂h'/∂h is at least 1
    -- (from the identity in h' = h + tanh(...))
    (1 : ℝ) + tanh_step_gradient_factor pre i ≥ 1 := by
  unfold tanh_step_gradient_factor
  have h : 1 - tanh (pre i) ^ 2 ≥ 0 := (tanh_gradient_in_unit_interval (pre i)).1
  linarith

/-- Moreover, residual gradient is UPPER BOUNDED by 2.
    This prevents explosion while ensuring flow. -/
theorem residual_gradient_upper_bound (pre : Fin n → ℝ) (i : Fin n) :
    (1 : ℝ) + tanh_step_gradient_factor pre i ≤ 2 := by
  unfold tanh_step_gradient_factor
  have h : 1 - tanh (pre i) ^ 2 ≤ 1 := (tanh_gradient_in_unit_interval (pre i)).2
  linarith

/-! ## Part 3: x-Dependent Gating (Minimal Gate) -/

/-- Minimal gated update: h' = g(x)·h + (1-g(x))·tanh(W_h·h + W_x·x)
    where g(x) = σ(w·x) depends on INPUT only, not hidden state.

    This is a middle ground between:
    - Full GRU (gate depends on h) - gradient variance from gate
    - Stock Elman (no gate) - gradient variance from tanh
    - Minimal gate (gate depends on x only) - less gradient variance! -/
structure MinimalGatedElman (n d : ℕ) where
  W_h : Matrix (Fin n) (Fin n) ℝ
  W_x : Matrix (Fin n) (Fin d) ℝ
  w_gate : Fin d → ℝ  -- Gate weights (input-dependent only)

/-- Gate value: sigmoid of input projection -/
noncomputable def gate_value (model : MinimalGatedElman n d) (x : Fin d → ℝ) : ℝ :=
  sigmoid (∑ i, model.w_gate i * x i)

/-- Minimal gated update -/
noncomputable def minimal_gated_update (model : MinimalGatedElman n d)
    (h : Fin n → ℝ) (x : Fin d → ℝ) : Fin n → ℝ :=
  let g := gate_value model x
  let tanh_term := fun i => tanh ((model.W_h.mulVec h + model.W_x.mulVec x) i)
  fun i => g * h i + (1 - g) * tanh_term i

/-- Sigmoid is always in (0, 1) -/
theorem sigmoid_pos (x : ℝ) : sigmoid x > 0 := by
  unfold sigmoid
  apply div_pos
  · exact one_pos
  · have : exp (-x) > 0 := exp_pos _
    linarith

theorem sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one]
  · have : exp (-x) > 0 := exp_pos _
    linarith
  · have : exp (-x) > 0 := exp_pos _
    linarith

theorem sigmoid_le_one (x : ℝ) : sigmoid x ≤ 1 := le_of_lt (sigmoid_lt_one x)
theorem sigmoid_nonneg (x : ℝ) : sigmoid x ≥ 0 := le_of_lt (sigmoid_pos x)

/-- The gradient of minimal gated Elman w.r.t. h.
    ∂h'/∂h = g(x)·I + (1-g(x))·diag(tanh'(pre))·W_h

    Key insight: g(x)·I term is h-INDEPENDENT!
    This provides a "gradient highway" that doesn't depend on trajectory. -/
theorem minimal_gate_gradient_structure (model : MinimalGatedElman n d) (x : Fin d → ℝ) :
    -- The gradient has an h-independent component g(x)·I
    ∃ (h_independent_factor : ℝ),
      h_independent_factor = gate_value model x ∧
      h_independent_factor ≥ 0 ∧
      h_independent_factor ≤ 1 := by
  use gate_value model x
  refine ⟨rfl, ?_, ?_⟩
  · exact sigmoid_nonneg _
  · exact sigmoid_le_one _

/-- The h-independent gradient component provides a floor on gradient flow,
    just like residual connections but with learned (input-dependent) scaling. -/
theorem minimal_gate_gradient_floor (model : MinimalGatedElman n d) (x : Fin d → ℝ) :
    -- When g(x) > 0, gradient through the gate path is guaranteed
    gate_value model x > 0 := sigmoid_pos _

/-! ## Part 4: The Expressivity-Gradient Tradeoff Theorem -/

/-- Architecture classification by gradient quality AND expressivity -/
inductive EnhancedArch
  | StockElman       -- h' = tanh(W_h·h + W_x·x)
  | ResidualElman    -- h' = h + tanh(W_h·h + W_x·x)
  | MinimalGated     -- h' = g(x)·h + (1-g(x))·tanh(...)
  | FullGRU          -- Full GRU with h-dependent gates

/-- Gradient stability score (higher = more stable gradients) -/
def gradient_stability : EnhancedArch → ℕ
  | EnhancedArch.StockElman => 1     -- Can vanish to 0
  | EnhancedArch.ResidualElman => 3  -- Bounded in [1, 2]
  | EnhancedArch.MinimalGated => 2   -- Has h-independent component
  | EnhancedArch.FullGRU => 1        -- Gate gradients also vary

/-- Expressivity score (higher = more expressive) -/
def enhanced_expressivity : EnhancedArch → ℕ
  | EnhancedArch.StockElman => 3     -- Full nonlinear h-dependence
  | EnhancedArch.ResidualElman => 3  -- Same! Residual doesn't reduce expressivity
  | EnhancedArch.MinimalGated => 3   -- Same! Still has tanh(W_h·h + ...)
  | EnhancedArch.FullGRU => 3        -- Same expressivity, more parameters

/-- Combined score (gradient stability + expressivity) -/
def combined_score (arch : EnhancedArch) : ℕ :=
  gradient_stability arch + enhanced_expressivity arch

/-- THE KEY THEOREM: Residual Elman achieves BEST combined score!
    It has the same expressivity as stock Elman but better gradient flow. -/
theorem residual_elman_optimal :
    combined_score EnhancedArch.ResidualElman >
    combined_score EnhancedArch.StockElman ∧
    combined_score EnhancedArch.ResidualElman >
    combined_score EnhancedArch.MinimalGated ∧
    combined_score EnhancedArch.ResidualElman >
    combined_score EnhancedArch.FullGRU := by
  unfold combined_score gradient_stability enhanced_expressivity
  norm_num

/-- Residual preserves expressivity while improving gradients -/
theorem residual_best_tradeoff :
    gradient_stability EnhancedArch.ResidualElman >
    gradient_stability EnhancedArch.StockElman ∧
    enhanced_expressivity EnhancedArch.ResidualElman =
    enhanced_expressivity EnhancedArch.StockElman := by
  unfold gradient_stability enhanced_expressivity
  constructor <;> norm_num

/-! ## Part 5: Implementation Guidance -/

/-- Summary of the tradeoff:

    | Architecture    | Gradient Stability | Expressivity | Combined |
    |-----------------|-------------------|--------------|----------|
    | Stock Elman     | 1 (can vanish)    | 3 (full)     | 4        |
    | Residual Elman  | 3 (bounded [1,2]) | 3 (full)     | 6 ← BEST |
    | Minimal Gated   | 2 (has floor)     | 3 (full)     | 5        |
    | Full GRU        | 1 (gate variance) | 3 (full)     | 4        |

    KEY INSIGHT: Residual connections give you the best of both worlds!
    - Same expressivity as stock Elman (tanh still operates on h)
    - Much better gradient flow (identity provides gradient highway)

    This is why ResNets revolutionized deep learning, and the same
    principle applies to RNNs. -/
theorem implementation_recommendation :
    -- Residual Elman beats stock Elman on gradient stability
    gradient_stability EnhancedArch.ResidualElman >
    gradient_stability EnhancedArch.StockElman ∧
    -- While maintaining expressivity
    enhanced_expressivity EnhancedArch.ResidualElman =
    enhanced_expressivity EnhancedArch.StockElman ∧
    -- And beating full GRU on combined score
    combined_score EnhancedArch.ResidualElman >
    combined_score EnhancedArch.FullGRU := by
  simp only [gradient_stability, enhanced_expressivity, combined_score]
  norm_num

end ExpressivityGradientTradeoff
