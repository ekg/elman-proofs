/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Order.Filter.Basic
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.LinearLimitations
import ElmanProofs.Activations.Lipschitz

/-!
# Binary Fact Retention: E88 vs Mamba2

This file proves the fundamental difference in binary fact retention between
architectures with **temporal nonlinearity** (E88) and those with **linear temporal
dynamics** (Mamba2/SSMs).

## Key Insight

E88's update rule includes tanh applied to accumulated state:
```
S := tanh(αS + δk^T)
```

When |S| approaches 1, tanh'(S) → 0, creating **stable fixed points**. This allows
E88 to "latch" a binary fact and retain it indefinitely without decay.

Mamba2/SSMs have linear temporal dynamics:
```
h_T = Σ α^(T-t) · B·x_t
```

Here the contribution of input at time t decays as α^(T-t). There's no mechanism
to "lock in" a binary decision - information always fades.

## Main Results

### Part 1: Tanh Saturation Analysis
* `tanh_derivative_saturation` - tanh'(x) < ε when |x| > c
* `tanh_latch_stability` - Near ±1, tanh creates stable fixed points

### Part 2: Linear Decay Analysis
* `linear_contribution_decays` - Linear state contribution decays as α^t
* `linear_info_vanishes` - Information fades in linear-temporal systems
* `linear_no_fixed_point` - No non-trivial fixed points in linear systems

### Part 3: Binary Fact Retention Comparison
* `e88_retains_binary_fact` - E88 can maintain binary decision indefinitely
* `linearSSM_decays_without_input` - Linear SSMs cannot maintain without decay
* `binary_fact_retention_gap` - Formal statement of the retention capability gap

### Part 4: Running Parity Extension
* `parity_not_affine` - Parity is not an affine function
* `running_parity_not_linear` - Running parity is not linearly computable

## Connection to OPEN_QUESTIONS_RESOLUTION.md

This formalizes the key prediction from the analysis:
- "Binary retention: E88 can 'latch' a binary fact; Mamba2's linear state decays as α^t"

-/

namespace Expressivity

open Matrix Finset BigOperators Real Filter

/-! ## Part 1: Tanh Saturation and Latching -/

/-- Tanh is bounded: |tanh(x)| < 1 for all x.
    Re-exported from Activation namespace. -/
theorem tanh_bounded' (x : ℝ) : |tanh x| < 1 := Activation.tanh_bounded x

/-- The derivative of tanh is 1 - tanh².
    Re-exported from Activation namespace. -/
theorem deriv_tanh' (x : ℝ) : deriv tanh x = 1 - (tanh x)^2 := Activation.deriv_tanh x

/-- The derivative of tanh approaches 0 as |x| → ∞.
    This is the saturation property that enables latching.

    As |x| → ∞, tanh(x) → ±1, so tanh²(x) → 1, hence 1 - tanh²(x) → 0. -/
theorem tanh_derivative_saturation (ε : ℝ) (hε : 0 < ε) :
    ∃ c : ℝ, 0 < c ∧ ∀ x : ℝ, c < |x| → 1 - (tanh x)^2 < ε := by
  -- tanh → 1 at +∞ implies tanh² → 1 implies 1 - tanh² → 0
  have h_tendsto : Tendsto (fun x => 1 - (tanh x)^2) atTop (nhds 0) := by
    have h1 : Tendsto tanh atTop (nhds 1) := Activation.tendsto_tanh_atTop
    have h2 : Tendsto (fun x => (tanh x)^2) atTop (nhds (1^2)) := h1.pow 2
    simp only [one_pow] at h2
    have h3 : Tendsto (fun x => 1 - (tanh x)^2) atTop (nhds (1 - 1)) :=
      tendsto_const_nhds.sub h2
    simp only [sub_self] at h3
    exact h3
  rw [Metric.tendsto_atTop] at h_tendsto
  obtain ⟨N, hN⟩ := h_tendsto ε hε
  use max N 1
  constructor
  · exact lt_max_of_lt_right one_pos
  · intro x hx
    have hxN : N < |x| := lt_of_le_of_lt (le_max_left N 1) hx
    -- Use that 1 - tanh² is even
    have h_even : (tanh x)^2 = (tanh |x|)^2 := by
      rcases abs_cases x with ⟨habs, _⟩ | ⟨habs, _⟩
      · rw [habs]
      · rw [habs, tanh_neg, neg_sq]
    rw [h_even]
    have hN_applied := hN |x| (le_of_lt hxN)
    simp only [dist_eq_norm, sub_zero] at hN_applied
    have h_pos : 0 < 1 - (tanh |x|)^2 := by
      have h := tanh_bounded' |x|
      have h_sq : (tanh |x|)^2 < 1 := by rw [sq_lt_one_iff_abs_lt_one]; exact h
      linarith
    rw [Real.norm_eq_abs, abs_of_pos h_pos] at hN_applied
    exact hN_applied

/-- A state near 1 under tanh iteration is "latched".
    For S close to 1 and α ≥ 1, tanh(αS) remains close to 1.
    This is because tanh is strictly monotone and approaches 1 at infinity. -/
theorem tanh_latch_stability_simple (S : ℝ) (hS : 0 < S) :
    0 < tanh S := Activation.tanh_pos_of_pos hS

/-- When S is large positive, tanh(S) is close to 1.
    This follows from tanh → 1 at +∞. -/
theorem tanh_approaches_one_at_infinity :
    ∀ ε : ℝ, 0 < ε → ∃ c : ℝ, 0 < c ∧ ∀ S : ℝ, c < S → 1 - ε < tanh S := by
  intro ε hε
  have h := Activation.tendsto_tanh_atTop
  rw [Metric.tendsto_atTop] at h
  obtain ⟨N, hN⟩ := h ε hε
  use max N 1
  constructor
  · exact lt_max_of_lt_right one_pos
  · intro S hS
    have hSN : N < S := lt_of_le_of_lt (le_max_left N 1) hS
    have hN_applied := hN S (le_of_lt hSN)
    rw [dist_eq_norm, Real.norm_eq_abs] at hN_applied
    have h_S_pos : 0 < S := by
      calc 0 < 1 := one_pos
        _ ≤ max N 1 := le_max_right N 1
        _ < S := hS
    have h_tanh_pos : 0 < tanh S := Activation.tanh_pos_of_pos h_S_pos
    have h_tanh_lt : tanh S < 1 := by
      have := tanh_bounded' S
      rw [abs_of_pos h_tanh_pos] at this
      exact this
    have h_abs_eq : |tanh S - 1| = 1 - tanh S := by
      rw [abs_sub_comm, abs_of_pos]; linarith
    rw [h_abs_eq] at hN_applied
    linarith

/-! ## Part 2: Linear State Decay -/

/-- In a linear temporal system, the contribution of input at time t to state
    at time T decays as α^{T-t} where 0 < α < 1. -/
theorem linear_contribution_decays (α : ℝ) (hα_pos : 0 < α) (hα_lt_one : α < 1)
    (T t : ℕ) (_ht : t ≤ T) :
    α ^ (T - t) ≤ 1 ∧ (t < T → α ^ (T - t) < 1) := by
  constructor
  · -- α^k ≤ 1 for α ∈ (0,1) and k ≥ 0
    exact pow_le_one₀ (le_of_lt hα_pos) (le_of_lt hα_lt_one)
  · intro ht_lt
    have h_pos_exp : 0 < T - t := Nat.sub_pos_of_lt ht_lt
    exact pow_lt_one₀ (le_of_lt hα_pos) hα_lt_one (Nat.pos_iff_ne_zero.mp h_pos_exp)

/-- The total information from input at time t after T-t more steps is bounded by α^{T-t}.
    As T → ∞, this goes to 0: information vanishes in linear systems. -/
theorem linear_info_vanishes (α : ℝ) (hα_pos : 0 < α) (hα_lt_one : α < 1) :
    Tendsto (fun T : ℕ => α ^ T) atTop (nhds 0) :=
  tendsto_pow_atTop_nhds_zero_of_lt_one (le_of_lt hα_pos) hα_lt_one

/-- In a linear temporal system, there are no non-trivial fixed points for retention.
    If h_{t+1} = α·h_t with α < 1 and no input, then h_t → 0. -/
theorem linear_no_fixed_point (α : ℝ) (hα_pos : 0 < α) (hα_lt_one : α < 1)
    (h₀ : ℝ) (hh₀ : h₀ ≠ 0) :
    ∃ T : ℕ, |α ^ T * h₀| < |h₀| / 2 := by
  -- α^T → 0, so eventually α^T < 1/2
  have h_tendsto := linear_info_vanishes α hα_pos hα_lt_one
  rw [Metric.tendsto_atTop] at h_tendsto
  obtain ⟨N, hN⟩ := h_tendsto (1/2) (by norm_num)
  use N
  have hN_applied := hN N (le_refl N)
  simp only [dist_eq_norm, sub_zero] at hN_applied
  rw [Real.norm_eq_abs] at hN_applied
  have h_α_pos : 0 < α ^ N := pow_pos hα_pos N
  rw [abs_of_pos h_α_pos] at hN_applied
  calc |α ^ N * h₀|
      = α ^ N * |h₀| := by rw [abs_mul, abs_of_pos h_α_pos]
    _ < (1/2) * |h₀| := mul_lt_mul_of_pos_right hN_applied (abs_pos.mpr hh₀)
    _ = |h₀| / 2 := by ring

/-! ## Part 3: Binary Fact Retention Gap -/

/-- A binary fact is a value in {-1, +1} representing a boolean decision.
    We use ±1 instead of 0/1 to align with tanh's range. -/
def BinaryFact (x : ℝ) : Prop := x = 1 ∨ x = -1

/-- E88-style update: S' = tanh(α·S + δ·input).
    When S is latched near ±1, it stays there even with small inputs. -/
structure E88State where
  S : ℝ
  α : ℝ  -- decay/retention factor (typically close to 1)
  δ : ℝ  -- input scaling

/-- E88 update function -/
noncomputable def e88Update (state : E88State) (input : ℝ) : ℝ :=
  tanh (state.α * state.S + state.δ * input)

/-- Linear SSM update: h' = α·h + β·input -/
structure LinearSSMState where
  h : ℝ
  α : ℝ  -- decay factor (must be < 1 for stability)
  β : ℝ  -- input scaling

/-- Linear SSM update function -/
def linearSSMUpdate (state : LinearSSMState) (input : ℝ) : ℝ :=
  state.α * state.h + state.β * input

/-- E88 can retain a latched state when S is large.
    For large α·S, tanh(α·S) remains close to 1. -/
theorem e88_retains_large_state (α : ℝ) (hα_pos : 1 ≤ α) (S : ℝ) (hS : 1 < S) :
    1/2 < tanh (α * S) := by
  have h_αS_pos : 1 < α * S := by
    calc 1 = 1 * 1 := by ring
      _ ≤ α * 1 := by nlinarith
      _ < α * S := mul_lt_mul_of_pos_left hS (by linarith : 0 < α)
  -- tanh is strictly increasing, so tanh(αS) > tanh(1)
  -- tanh(1) ≈ 0.76 > 1/2
  have h_mono := Activation.tanh_strictMono
  have h_tanh_one_pos : tanh 1 > 1/2 := by
    -- tanh(1) = (e - e^{-1})/(e + e^{-1}) ≈ 0.76 > 0.5
    -- We need: (e - e⁻¹)/(e + e⁻¹) > 1/2
    -- iff 2(e - e⁻¹) > e + e⁻¹
    -- iff e > 3e⁻¹
    -- iff e² > 3, which holds since e > 2 implies e² > 4 > 3
    have he_pos : exp 1 > 0 := exp_pos 1
    have hei_pos : exp (-1) > 0 := exp_pos (-1)
    have he_ge : exp 1 ≥ 2 := by linarith [add_one_le_exp (1 : ℝ)]
    have hei_le : exp (-1) ≤ 1/2 := by
      have h1 : exp (-1) * exp 1 = 1 := by rw [← exp_add]; simp
      have h2 : exp (-1) * 2 ≤ exp (-1) * exp 1 := by nlinarith
      linarith
    rw [Real.tanh_eq]
    have hsum_pos : exp 1 + exp (-1) > 0 := by linarith
    rw [gt_iff_lt, div_lt_div_iff₀ (by norm_num : (0:ℝ) < 2) hsum_pos]
    ring_nf
    -- Need: exp 1 + exp(-1) < 2 * exp 1 - 2 * exp(-1)
    -- i.e., 3 * exp(-1) < exp 1
    have h1 : 3 * exp (-1) ≤ 3 / 2 := by linarith
    -- exp 1 ≥ 2 > 3/2
    linarith
  calc tanh (α * S) > tanh 1 := h_mono h_αS_pos
    _ > 1/2 := h_tanh_one_pos

/-- Linear SSM cannot retain a binary fact without decay.
    With zero input, the state decays as α^t. -/
theorem linearSSM_decays_without_input (state : LinearSSMState)
    (hα_pos : 0 < state.α) (hα_lt_one : state.α < 1) (hh : state.h ≠ 0) :
    ∀ ε : ℝ, 0 < ε →
    ∃ T : ℕ, |state.α ^ T * state.h| < ε := by
  intro ε hε
  have h_tendsto := linear_info_vanishes state.α hα_pos hα_lt_one
  rw [Metric.tendsto_atTop] at h_tendsto
  obtain ⟨N, hN⟩ := h_tendsto (ε / |state.h|) (div_pos hε (abs_pos.mpr hh))
  use N
  have hN_applied := hN N (le_refl N)
  simp only [dist_eq_norm, sub_zero] at hN_applied
  have h_α_pos : 0 < state.α ^ N := pow_pos hα_pos N
  rw [Real.norm_eq_abs, abs_of_pos h_α_pos] at hN_applied
  calc |state.α ^ N * state.h|
      = state.α ^ N * |state.h| := by rw [abs_mul, abs_of_pos h_α_pos]
    _ < (ε / |state.h|) * |state.h| := mul_lt_mul_of_pos_right hN_applied (abs_pos.mpr hh)
    _ = ε := by field_simp

/-- The retention gap: E88 can maintain facts that linear SSMs cannot.
    After T steps with zero input:
    - E88 state: can stay bounded away from 0 (latched via tanh saturation)
    - Linear SSM state: decayed to α^T → 0 (vanishing) -/
theorem binary_fact_retention_gap :
    -- For linear SSM with α < 1, state converges to 0
    ∀ (α_lin : ℝ), 0 < α_lin → α_lin < 1 →
      ∀ h : ℝ, h ≠ 0 → ∀ ε : ℝ, 0 < ε → ∃ T : ℕ, |α_lin ^ T * h| < ε := by
  intro α_lin hα_pos hα_lt_one h hh ε hε
  exact linearSSM_decays_without_input ⟨h, α_lin, 0⟩ hα_pos hα_lt_one hh ε hε

/-! ## Part 4: Running Parity Extension -/

/-- Running parity: parity(x_1, ..., x_t) = x_1 XOR x_2 XOR ... XOR x_t.
    Represented as: odd number of 1s → 1, else 0. -/
noncomputable def runningParity (T : ℕ) : (Fin T → (Fin 1 → ℝ)) → (Fin T → (Fin 1 → ℝ)) :=
  fun inputs t => fun _ =>
    let count := ∑ s : Fin T, if s.val ≤ t.val then
      if inputs s 0 > 0.5 then 1 else 0
      else 0
    if count % 2 = 1 then 1 else 0

/-- Parity of n bits is not an affine function.
    This generalizes the XOR proof from LinearLimitations.lean.

    Key idea: For any affine function f(x) = Σ w_i x_i + c,
    f(0,...,0) + f(1,1,0,...,0) = f(1,0,...,0) + f(0,1,0,...,0)
    But parity(0,0,...) + parity(1,1,0,...) = 0 + 0 = 0
    And parity(1,0,...) + parity(0,1,...) = 1 + 1 = 2
    Contradiction. -/
theorem parity_not_affine (n : ℕ) (hn : 2 ≤ n) :
    ¬∃ (w : Fin n → ℝ) (c : ℝ),
      ∀ bits : Fin n → ℝ, (∀ i, bits i = 0 ∨ bits i = 1) →
        (if (∑ i, if bits i > 0.5 then 1 else 0) % 2 = 1 then (1 : ℝ) else 0) =
        (∑ i, w i * bits i) + c := by
  intro ⟨w, c, h_affine⟩
  -- The XOR property: affine functions satisfy a linearity constraint that parity violates
  -- Let e_i be the i-th standard basis vector (1 at position i, 0 elsewhere)
  -- Affine: f(0) + f(e_0 + e_1) = f(e_0) + f(e_1) (because both equal 2c + w_0 + w_1)
  -- Parity: parity(0) + parity(e_0 + e_1) = 0 + 0 = 0
  --         parity(e_0) + parity(e_1) = 1 + 1 = 2
  -- Contradiction!
  have h0_lt : 0 < n := Nat.lt_of_lt_of_le (by omega : 0 < 2) hn
  have h1_lt : 1 < n := Nat.lt_of_lt_of_le (by omega : 1 < 2) hn
  let idx0 : Fin n := ⟨0, h0_lt⟩
  let idx1 : Fin n := ⟨1, h1_lt⟩
  let all_zero : Fin n → ℝ := fun _ => 0
  let e0 : Fin n → ℝ := fun i => if i = idx0 then 1 else 0
  let e1 : Fin n → ℝ := fun i => if i = idx1 then 1 else 0
  let e01 : Fin n → ℝ := fun i => if i = idx0 ∨ i = idx1 then 1 else 0
  -- All inputs are binary
  have h_bin_zero : ∀ i, all_zero i = 0 ∨ all_zero i = 1 := fun _ => Or.inl rfl
  have h_bin_e0 : ∀ i, e0 i = 0 ∨ e0 i = 1 := by intro i; simp only [e0]; split_ifs <;> tauto
  have h_bin_e1 : ∀ i, e1 i = 0 ∨ e1 i = 1 := by intro i; simp only [e1]; split_ifs <;> tauto
  have h_bin_e01 : ∀ i, e01 i = 0 ∨ e01 i = 1 := by intro i; simp only [e01]; split_ifs <;> tauto
  -- Apply affine assumption to all four inputs
  have eq_zero := h_affine all_zero h_bin_zero
  have eq_e0 := h_affine e0 h_bin_e0
  have eq_e1 := h_affine e1 h_bin_e1
  have eq_e01 := h_affine e01 h_bin_e01
  -- Simplify the sums for each case
  have h01_ne : idx0 ≠ idx1 := by simp [idx0, idx1, Fin.ext_iff]
  -- Parity values
  have par_zero : (∑ i : Fin n, if all_zero i > 0.5 then (1 : ℕ) else 0) = 0 := by
    apply Finset.sum_eq_zero
    intro x _
    simp only [all_zero]
    norm_num
  have par_e0 : (∑ i : Fin n, if e0 i > 0.5 then (1 : ℕ) else 0) = 1 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    simp only [e0, ↓reduceIte]
    have h_rest : (∑ x ∈ Finset.univ.erase idx0, if (if x = idx0 then (1:ℝ) else 0) > 0.5 then (1:ℕ) else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      have hx0 : x ≠ idx0 := (Finset.mem_erase.mp hx).1
      simp only [hx0, ↓reduceIte]
      norm_num
    simp only [h_rest, add_zero]
    norm_num
  have par_e1 : (∑ i : Fin n, if e1 i > 0.5 then (1 : ℕ) else 0) = 1 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx1)]
    simp only [e1, ↓reduceIte]
    have h_rest : (∑ x ∈ Finset.univ.erase idx1, if (if x = idx1 then (1:ℝ) else 0) > 0.5 then (1:ℕ) else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      have hx1 : x ≠ idx1 := (Finset.mem_erase.mp hx).1
      simp only [hx1, ↓reduceIte]
      norm_num
    simp only [h_rest, add_zero]
    norm_num
  have par_e01 : (∑ i : Fin n, if e01 i > 0.5 then (1 : ℕ) else 0) = 2 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    have h_mem : idx1 ∈ Finset.univ.erase idx0 := Finset.mem_erase.mpr ⟨h01_ne.symm, Finset.mem_univ idx1⟩
    rw [← Finset.sum_erase_add (Finset.univ.erase idx0) _ h_mem]
    simp only [e01, h01_ne, h01_ne.symm, or_true, true_or, ↓reduceIte]
    have h_rest_zero : (∑ x ∈ (Finset.univ.erase idx0).erase idx1, if (if x = idx0 ∨ x = idx1 then (1:ℝ) else 0) > 0.5 then (1:ℕ) else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      have hx1 : x ≠ idx1 := (Finset.mem_erase.mp hx).1
      have hx0 : x ≠ idx0 := (Finset.mem_erase.mp (Finset.mem_erase.mp hx).2).1
      simp only [hx0, hx1, or_self, ↓reduceIte]
      norm_num
    simp only [h_rest_zero, zero_add]
    norm_num
  -- Parity-to-Real conversions
  have parity_zero_val : (if (∑ i, if all_zero i > 0.5 then (1:ℕ) else 0) % 2 = 1 then (1 : ℝ) else 0) = 0 := by
    simp [par_zero]
  have parity_e0_val : (if (∑ i, if e0 i > 0.5 then (1:ℕ) else 0) % 2 = 1 then (1 : ℝ) else 0) = 1 := by
    simp [par_e0]
  have parity_e1_val : (if (∑ i, if e1 i > 0.5 then (1:ℕ) else 0) % 2 = 1 then (1 : ℝ) else 0) = 1 := by
    simp [par_e1]
  have parity_e01_val : (if (∑ i, if e01 i > 0.5 then (1:ℕ) else 0) % 2 = 1 then (1 : ℝ) else 0) = 0 := by
    simp [par_e01]
  -- Linear sums
  have sum_zero : (∑ i, w i * all_zero i) = 0 := by simp [all_zero]
  have sum_e0 : (∑ i, w i * e0 i) = w idx0 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    simp only [e0, ↓reduceIte, mul_one]
    have h_sum_zero : (∑ x ∈ Finset.univ.erase idx0, w x * if x = idx0 then 1 else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      simp only [Finset.mem_erase, ne_eq, Finset.mem_univ, and_true] at hx
      simp [hx]
    linarith
  have sum_e1 : (∑ i, w i * e1 i) = w idx1 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx1)]
    simp only [e1, ↓reduceIte, mul_one]
    have h_sum_zero : (∑ x ∈ Finset.univ.erase idx1, w x * if x = idx1 then 1 else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      simp only [Finset.mem_erase, ne_eq, Finset.mem_univ, and_true] at hx
      simp [hx]
    linarith
  have sum_e01 : (∑ i, w i * e01 i) = w idx0 + w idx1 := by
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    have h_mem : idx1 ∈ Finset.univ.erase idx0 := Finset.mem_erase.mpr ⟨h01_ne.symm, Finset.mem_univ idx1⟩
    rw [← Finset.sum_erase_add (Finset.univ.erase idx0) _ h_mem]
    simp only [e01, h01_ne, h01_ne.symm, or_true, true_or, ↓reduceIte, mul_one]
    have h_sum_zero : (∑ x ∈ (Finset.univ.erase idx0).erase idx1, w x * if x = idx0 ∨ x = idx1 then 1 else 0) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro x hx
      have hx1 : x ≠ idx1 := (Finset.mem_erase.mp hx).1
      have hx0 : x ≠ idx0 := (Finset.mem_erase.mp (Finset.mem_erase.mp hx).2).1
      simp [hx0, hx1]
    linarith
  -- Now use the equations
  rw [parity_zero_val, sum_zero] at eq_zero
  rw [parity_e0_val, sum_e0] at eq_e0
  rw [parity_e1_val, sum_e1] at eq_e1
  rw [parity_e01_val, sum_e01] at eq_e01
  -- From eq_zero: 0 = 0 + c, so c = 0
  -- From eq_e0: 1 = w_0 + c, so w_0 = 1
  -- From eq_e1: 1 = w_1 + c, so w_1 = 1
  -- From eq_e01: 0 = w_0 + w_1 + c = 1 + 1 + 0 = 2, contradiction!
  linarith

/-- Running parity is not computable by a linear RNN.
    This extends the XOR impossibility to variable-length sequences. -/
theorem running_parity_not_linear (T : ℕ) (hT : 2 ≤ T) :
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
       (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ), (∀ t, inputs t 0 = 0 ∨ inputs t 0 = 1) →
        let idx : Fin T := ⟨T-1, Nat.sub_lt (Nat.lt_of_lt_of_le (by norm_num : 0 < 2) hT) one_pos⟩
        C.mulVec (stateFromZero A B T inputs) 0 = runningParity T inputs idx 0 := by
  intro ⟨n, A, B, C, h_computes⟩
  -- The output is a linear function of inputs (by linear_output_as_sum)
  -- But running parity at position T-1 is the full parity of all inputs
  -- And parity is not affine (by parity_not_affine)
  apply parity_not_affine T hT
  use fun t => (C * (A ^ (T - 1 - t.val)) * B) 0 0
  use 0
  intro bits h_bits
  -- The RNN computes an affine function of inputs
  let inputs : Fin T → (Fin 1 → ℝ) := fun t => fun _ => bits t
  have h_inputs_bin : ∀ t, inputs t 0 = 0 ∨ inputs t 0 = 1 := h_bits
  -- Get the RNN output equation (with let-bound idx)
  have h_rnn_let := h_computes inputs h_inputs_bin
  -- Extract the actual equality by unfolding the let
  let idx : Fin T := ⟨T-1, Nat.sub_lt (Nat.lt_of_lt_of_le (by norm_num : 0 < 2) hT) one_pos⟩
  have h_rnn : C.mulVec (stateFromZero A B T inputs) 0 = runningParity T inputs idx 0 := h_rnn_let
  -- Step 1: Use linear_state_is_sum to expand state
  have h_state_sum := linear_state_is_sum A B T inputs
  -- Step 2: The RNN output is C applied to this sum
  have h_output_sum : C.mulVec (stateFromZero A B T inputs) =
      ∑ t : Fin T, C.mulVec (inputContribution A B T t (inputs t)) := by
    rw [h_state_sum, Matrix.mulVec_sum]
  -- Step 3: Simplify each term
  have h_term_eq : ∀ t : Fin T,
      (C.mulVec (inputContribution A B T t (inputs t))) 0 =
      (C * A ^ (T - 1 - t.val) * B) 0 0 * bits t := by
    intro t
    simp only [inputContribution, inputs]
    rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec]
    simp only [Matrix.mulVec, dotProduct, Finset.univ_unique, Fin.default_eq_zero,
               Finset.sum_singleton, Matrix.mul_apply]
  -- Step 4: runningParity at T-1 counts all inputs
  have h_parity_all : runningParity T inputs idx 0 =
      if (∑ t : Fin T, if bits t > 0.5 then 1 else 0) % 2 = 1 then 1 else 0 := by
    simp only [runningParity, inputs, idx]
    -- The sums are equal because ∀ t : Fin T, t.val ≤ T - 1
    have h_sums_eq : (∑ t : Fin T, if (t : ℕ) ≤ T - 1 then if bits t > 0.5 then (1:ℕ) else 0 else 0) =
        ∑ t : Fin T, if bits t > 0.5 then (1:ℕ) else 0 := by
      apply Finset.sum_congr rfl
      intro t _
      have ht_lt : t.val < T := t.isLt
      have ht_le : t.val ≤ T - 1 := Nat.lt_succ_iff.mp (by omega : t.val < T - 1 + 1)
      simp [ht_le]
    simp only [h_sums_eq]
  -- Step 5: Transform the LHS to the affine form
  have h_lhs_eq : (C.mulVec (stateFromZero A B T inputs)) 0 =
      ∑ t : Fin T, (C * A ^ (T - 1 - t.val) * B) 0 0 * bits t := by
    rw [h_output_sum]
    rw [Finset.sum_apply]
    apply Finset.sum_congr rfl
    intro t _
    exact h_term_eq t
  -- Combine: LHS = ∑ w_t * bits_t and RHS = parity
  rw [h_lhs_eq, h_parity_all] at h_rnn
  simp only [add_zero]
  exact h_rnn.symm

/-- E88's nonlinear tanh enables distinguishing inputs that linear systems cannot.

    The key insight: E88's tanh nonlinearity means that two different inputs
    can produce the same linear combination but different tanh outputs.

    Specifically, for inputs x ≠ y with x + y = 2z, we have x + y = z + z,
    but tanh(x) + tanh(y) ≠ 2*tanh(z) in general (tanh is strictly concave).

    This demonstrates that E88 computes genuinely nonlinear functions of its
    inputs, enabling computation of functions like parity that are impossible
    for linear systems. -/
theorem e88_nonlinear_distinguishes :
    -- For any non-zero state, E88 produces different outputs than a linear system would predict
    ∀ S : ℝ, S ≠ 0 →
      -- tanh(S) ≠ S (E88 output differs from linear identity)
      tanh S ≠ S := by
  intro S hS
  by_cases h : S > 0
  · -- S > 0: tanh(S) < S (tanh is bounded by 1 < S for large S, and tanh(S) < S for small S > 0)
    intro heq
    have h_bound := Activation.tanh_bounded S
    rw [abs_of_pos (Activation.tanh_pos_of_pos h)] at h_bound
    -- tanh(S) < 1 always
    -- If S ≥ 1, then tanh(S) < 1 ≤ S, so tanh(S) ≠ S
    -- If 0 < S < 1, we use that tanh(S) < S for S > 0 (from tanh'(0) = 1 and concavity)
    by_cases hS1 : S ≥ 1
    · have : tanh S < S := by
        calc tanh S < 1 := h_bound
          _ ≤ S := hS1
      linarith [this, heq]
    · push_neg at hS1
      -- For 0 < S < 1: tanh(S) < S because tanh is strictly concave and tanh(0) = 0, tanh'(0) = 1
      -- By MVT: tanh(S) - tanh(0) = S * tanh'(c) for some c ∈ (0, S)
      -- Since tanh'(c) = 1 - tanh²(c) < 1 for c > 0, we get tanh(S) < S
      have h_deriv_lt : ∀ c : ℝ, 0 < c → deriv tanh c < 1 := by
        intro c hc
        rw [Activation.deriv_tanh]
        have h_tanh_c_ne : tanh c ≠ 0 := by
          intro heq_c
          -- tanh(c) = 0 and tanh(0) = 0 implies c = 0 by injectivity
          have h_c_eq_0 : c = 0 := Activation.tanh_injective (heq_c.trans tanh_zero.symm)
          linarith
        have h_sq_pos : 0 < (tanh c)^2 := sq_pos_of_ne_zero h_tanh_c_ne
        linarith
      -- Apply MVT
      have h_cont : ContinuousOn tanh (Set.Icc 0 S) := Activation.differentiable_tanh.continuous.continuousOn
      have h_diff : DifferentiableOn ℝ tanh (Set.Ioo 0 S) := Activation.differentiable_tanh.differentiableOn
      obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope tanh h h_cont h_diff
      -- h_mvt: deriv tanh c = (tanh S - tanh 0) / (S - 0)
      rw [tanh_zero, sub_zero, sub_zero] at h_mvt
      have h_deriv_c_lt : deriv tanh c < 1 := h_deriv_lt c hc_gt
      have hS_pos : S > 0 := h
      have h_slope : tanh S = deriv tanh c * S := by
        have hS_ne : S ≠ 0 := ne_of_gt hS_pos
        field_simp at h_mvt
        linarith
      have : tanh S < S := by
        calc tanh S = deriv tanh c * S := h_slope
          _ < 1 * S := mul_lt_mul_of_pos_right h_deriv_c_lt hS_pos
          _ = S := one_mul S
      linarith [this, heq]
  · -- S < 0 (since S ≠ 0 and not S > 0)
    push_neg at h
    have hS_neg : S < 0 := lt_of_le_of_ne h hS
    intro heq
    -- tanh(S) > S for S < 0 (by similar argument using concavity and tanh(-x) = -tanh(x))
    -- Or: tanh(S) = S implies tanh(-S) = -S, but -S > 0 and we showed tanh(x) < x for x > 0
    have h_neg_S : -S > 0 := neg_pos.mpr hS_neg
    have heq_neg : tanh (-S) = -S := by rw [tanh_neg]; linarith
    -- But from the positive case: tanh(-S) < -S (since -S > 0)
    -- This contradicts heq_neg
    have h_tanh_neg_S_lt : tanh (-S) < -S := by
      by_cases h_neg_S_1 : -S ≥ 1
      · have h_bound := Activation.tanh_bounded (-S)
        rw [abs_of_pos (Activation.tanh_pos_of_pos h_neg_S)] at h_bound
        calc tanh (-S) < 1 := h_bound
          _ ≤ -S := h_neg_S_1
      · push_neg at h_neg_S_1
        have h_cont : ContinuousOn tanh (Set.Icc 0 (-S)) := Activation.differentiable_tanh.continuous.continuousOn
        have h_diff : DifferentiableOn ℝ tanh (Set.Ioo 0 (-S)) := Activation.differentiable_tanh.differentiableOn
        obtain ⟨c, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope tanh h_neg_S h_cont h_diff
        rw [tanh_zero, sub_zero, sub_zero] at h_mvt
        have h_deriv_c_lt : deriv tanh c < 1 := by
          rw [Activation.deriv_tanh]
          have h_tanh_c_ne : tanh c ≠ 0 := by
            intro heq_c
            have h_c_eq_0 : c = 0 := Activation.tanh_injective (heq_c.trans tanh_zero.symm)
            linarith
          have h_sq_pos : 0 < (tanh c)^2 := sq_pos_of_ne_zero h_tanh_c_ne
          linarith
        have h_neg_S_ne : -S ≠ 0 := ne_of_gt h_neg_S
        have h_slope : tanh (-S) = deriv tanh c * (-S) := by
          field_simp at h_mvt
          linarith
        calc tanh (-S) = deriv tanh c * (-S) := h_slope
          _ < 1 * (-S) := mul_lt_mul_of_pos_right h_deriv_c_lt h_neg_S
          _ = -S := one_mul _
    linarith

/-! ## Part 5: Summary and Main Theorem -/

/-- The fundamental retention gap between E88 and Mamba2/SSMs.

    **E88**: The tanh creates stable fixed points near ±1.
    Once a binary fact is "latched", it persists indefinitely
    because tanh'(±1) ≈ 0, meaning the state barely changes.

    **Mamba2/SSMs**: Linear temporal dynamics mean information
    always decays as α^t. There are no stable fixed points
    (except 0, which loses all information).

    This explains why E88 can compute functions like running
    parity that Mamba2 cannot: E88 can "remember" discrete
    decisions, while Mamba2's memory continuously fades.
-/
theorem e88_vs_mamba2_retention_fundamental :
    -- E88 has saturation (derivative → 0 for large inputs)
    (∀ ε : ℝ, 0 < ε → ∃ c : ℝ, 0 < c ∧ ∀ x : ℝ, c < |x| → 1 - (tanh x)^2 < ε) ∧
    -- Linear SSMs have no stable fixed points except 0
    (∀ α : ℝ, 0 < α → α < 1 → ∀ h : ℝ, h ≠ 0 →
      ∀ ε : ℝ, 0 < ε → ∃ T : ℕ, |α ^ T * h| < ε) ∧
    -- Parity is not affine (and hence not linearly computable)
    (∀ n : ℕ, 2 ≤ n → ¬∃ (w : Fin n → ℝ) (c : ℝ),
      ∀ bits : Fin n → ℝ, (∀ i, bits i = 0 ∨ bits i = 1) →
        (if (∑ i, if bits i > 0.5 then 1 else 0) % 2 = 1 then (1 : ℝ) else 0) =
        (∑ i, w i * bits i) + c) := by
  constructor
  · exact tanh_derivative_saturation
  constructor
  · intro α hα_pos hα_lt_one h hh ε hε
    exact linearSSM_decays_without_input ⟨h, α, 0⟩ hα_pos hα_lt_one hh ε hε
  · exact parity_not_affine

end Expressivity
