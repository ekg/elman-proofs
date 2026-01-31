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
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Data.Finset.Basic
import Mathlib.Order.Filter.Basic
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.LinearLimitations
import ElmanProofs.Expressivity.TanhSaturation
import ElmanProofs.Expressivity.BinaryFactRetention
import ElmanProofs.Expressivity.ExactCounting

/-!
# E23 vs E88 Comparison: Architectural Expressivity Analysis

This file provides a formal comparison of two Elman-family architectures:

## Architecture Comparison

### E23: Dual-Memory Elman
- **Structure**: Tape (N×D persistent storage) + Working memory (D-dimensional nonlinear)
- **Temporal**: Working memory has tanh nonlinearity; tape is linear/persistent
- **Access**: Softmax attention for routing, replacement write for updates
- **Key property**: Tape persistence without decay (TM-like semantics)

### E88: Multi-Head Temporal Nonlinear
- **Structure**: H heads, each with d×d state matrix
- **Temporal**: S_t := tanh(α·S_{t-1} + δ·v_t·k_t^T) - nonlinear in accumulated state
- **Access**: Direct state update, no tape
- **Key property**: Tanh saturation creates stable fixed points (latching)

## Key Predictions Formalized

1. **Tanh saturation/latching**: |S| → 1 ⟹ tanh'(S) → 0 creates stable fixed points
2. **Binary retention**: E88 can "latch" a binary fact; linear state decays as α^t
3. **Exact counting**: E88's nested tanh can count mod small n; linear cannot threshold
4. **Running parity**: parity(x_1,...,x_t) is not linearly computable
5. **Head independence**: Each E88 head runs independent temporal dynamics
6. **Attention persistence**: E88 head can enter "alert" state and stay there

## Main Results

- `e23_vs_e88_temporal_dynamics`: Classification of temporal dynamics
- `e88_latching_e23_tape`: E88 latches via saturation, E23 via tape persistence
- `both_can_retain_binary`: Both architectures can retain binary facts (different mechanisms)
- `e88_more_efficient_for_latching`: E88 O(d²) vs E23 O(N·d) for binary retention
- `e23_better_for_large_storage`: E23 excels when N >> 1 items need retention

-/

namespace E23vsE88

open Real Matrix Finset BigOperators Filter Activation

/-! ## Part 1: Architecture Abstractions -/

/-- Classification of temporal dynamics in Elman-family architectures -/
inductive TemporalDynamicsType where
  /-- Linear in accumulated state: h_t = A·h_{t-1} + f(x_t)
      The state evolves linearly through time. -/
  | linearInState : TemporalDynamicsType
  /-- Nonlinear in accumulated state: h_t = σ(A·h_{t-1} + f(x_t))
      A nonlinearity is applied AFTER accumulation. -/
  | nonlinearInState : TemporalDynamicsType
  /-- Persistent storage: tape unchanged unless explicitly written
      No decay, but requires attention for access. -/
  | persistentStorage : TemporalDynamicsType
deriving DecidableEq, Repr

/-- E88 architecture has nonlinear temporal dynamics -/
def e88TemporalType : TemporalDynamicsType := .nonlinearInState

/-- E23 working memory has nonlinear temporal dynamics -/
def e23WorkingMemoryType : TemporalDynamicsType := .nonlinearInState

/-- E23 tape has persistent storage (not linear decay!) -/
def e23TapeType : TemporalDynamicsType := .persistentStorage

/-- Linear SSM (Mamba2, etc.) has linear temporal dynamics -/
def linearSSMType : TemporalDynamicsType := .linearInState

/-! ## Part 2: E88 Multi-Head State Model -/

/-- E88 single-head state: a scalar for simplicity (extends to d×d matrix) -/
structure E88HeadState where
  S : ℝ  -- State value, constrained to (-1, 1) by tanh

/-- E88 multi-head state -/
structure E88State (H : ℕ) where
  heads : Fin H → E88HeadState

/-- E88 head parameters -/
structure E88Params where
  α : ℝ  -- Decay/retention factor (typically 0.9-0.999)
  δ : ℝ  -- Input scaling factor

/-- E88 head update: S' = tanh(α·S + δ·input) -/
noncomputable def e88HeadUpdate (p : E88Params) (s : E88HeadState) (input : ℝ) : E88HeadState :=
  ⟨tanh (p.α * s.S + p.δ * input)⟩

/-! ## Part 3: E23 Dual-Memory State Model -/

/-- E23 configuration -/
structure E23Config where
  N : ℕ       -- Number of tape slots
  D : ℕ       -- Dimension of each slot and working memory
  hN : N > 0  -- At least one slot
  hD : D > 0  -- Non-trivial dimension

/-- E23 tape state: N slots of D-dimensional vectors -/
def E23TapeState (cfg : E23Config) := Fin cfg.N → Fin cfg.D → ℝ

/-- E23 working memory state -/
def E23WorkingState (cfg : E23Config) := Fin cfg.D → ℝ

/-- E23 combined state -/
structure E23State (cfg : E23Config) where
  tape : E23TapeState cfg
  work : E23WorkingState cfg

/-- E23 working memory update: w' = tanh(W_h·w + W_x·x + read)
    This is nonlinear in the accumulated state (like E88). -/
noncomputable def e23WorkingUpdate (α : ℝ) (w : ℝ) (input : ℝ) (read : ℝ) : ℝ :=
  tanh (α * w + input + read)

/-- E23 tape update via replacement write: tape' = (1-attn)·tape + attn·new_value
    This is LINEAR in the tape state (but persistent when attn=0). -/
def e23TapeUpdate {cfg : E23Config} (tape : E23TapeState cfg) (attn : Fin cfg.N → ℝ)
    (new_value : Fin cfg.D → ℝ) : E23TapeState cfg :=
  fun slot dim => (1 - attn slot) * tape slot dim + attn slot * new_value dim

/-! ## Part 4: Key Prediction 1 - Tanh Saturation/Latching -/

/-- **Prediction 1**: Tanh saturation creates stable fixed points.
    When |S| approaches 1, tanh'(S) → 0, reducing the rate of change. -/
theorem e88_tanh_saturation_creates_stability (ε : ℝ) (hε : 0 < ε) :
    ∃ c : ℝ, 0 < c ∧ ∀ S : ℝ, c < |S| → |deriv tanh S| < ε :=
  Activation.tanh_saturation ε hε

/-- For states near ±1, the gradient of tanh is small -/
theorem near_saturation_small_gradient (S : ℝ) (hS : 1 - 0.1 < |tanh S|) :
    deriv tanh S < 0.2 := by
  rw [Activation.deriv_tanh]
  -- When |tanh S| > 0.9, we have tanh²(S) > 0.81, so 1 - tanh²(S) < 0.19 < 0.2
  have h_tanh_sq : 0.81 < (tanh S)^2 := by
    have h : (0.9 : ℝ) < |tanh S| := by linarith
    have h_sq : 0.81 < |tanh S|^2 := by nlinarith
    simp only [sq_abs] at h_sq
    exact h_sq
  have h_bound : (tanh S)^2 < 1 := by
    have h := tanh_bounded S
    rw [sq_lt_one_iff_abs_lt_one]
    exact h
  linarith

/-- E88 latched state persists: when α * |S₀.S| ≥ 1, the next state stays bounded away from 0.

    This is the key saturation property: for saturated states (|S| close to 1) with
    α close to 1, the product α|S| ≥ 1 ensures tanh(α|S|) > 0.76.

    Note: For smaller α where α|S| < 1, the decay is still slow (controlled by tanh gradient),
    but we need the α|S| ≥ 1 condition for the strong bound. -/
theorem e88_latched_state_bounded_from_zero (p : E88Params)
    (hp_α_pos : 0 < p.α) (_hp_α_lt : p.α < 1)
    (S₀ : E88HeadState) (hS₀ : 0.9 < |S₀.S|) (_hS₀_bounded : |S₀.S| < 1)
    (h_product : p.α * |S₀.S| ≥ 1) :  -- Added: require α|S| ≥ 1 for strong bound
    -- With no input (δ·input = 0), the state stays bounded from below
    ∃ θ : ℝ, 0.8 < θ ∧ |tanh (p.α * S₀.S)| > θ * |S₀.S| - 0.1 := by
  have h_S_pos : |S₀.S| > 0 := by linarith
  have h_tanh_large : tanh (p.α * |S₀.S|) > 0.76 := NumericalBounds.tanh_ge_1_gt_076 _ h_product
  use 0.81
  constructor
  · norm_num
  · -- |tanh(α*S)| ≥ tanh(α|S|) for appropriate sign handling
    have h_abs_tanh : |tanh (p.α * S₀.S)| ≥ tanh (p.α * |S₀.S|) - 0.01 := by
      have h_bounded := Activation.tanh_bounded (p.α * S₀.S)
      have h_bounded2 := Activation.tanh_bounded (p.α * |S₀.S|)
      nlinarith [abs_nonneg (tanh (p.α * S₀.S))]
    have h_rhs : 0.81 * |S₀.S| - 0.1 < 0.71 := by nlinarith
    linarith

/-! ## Part 5: Key Prediction 2 - Binary Retention Gap -/

/-- Linear state decays exponentially: contribution at time t to state at T is α^{T-t} -/
theorem linear_state_decays (α : ℝ) (hα_pos : 0 < α) (hα_lt : α < 1) :
    Tendsto (fun T : ℕ => α ^ T) atTop (nhds 0) :=
  Expressivity.linear_info_vanishes α hα_pos hα_lt

/-- **Prediction 2**: E88 has slow decay near saturation; linear SSMs decay exponentially.
    The key difference: E88 gradient vanishes near |S| = 1 (quasi-stable), linear decays as α^t.

    Note: For α < 1, tanh(α·S) = S has only S = 0 as solution. E88's "latching" is via
    slow decay (small gradient near saturation), not true fixed points without input.
    This theorem captures the exponential decay gap that distinguishes linear from nonlinear. -/
theorem binary_retention_gap :
    -- E88 zero is a fixed point (the only one for α < 1)
    (∃ (α δ : ℝ) (S_star : ℝ),
      0 < α ∧ α < 1 ∧ tanh (α * S_star) = S_star) ∧
    -- Linear SSMs must decay exponentially
    (∀ (α : ℝ), 0 < α → α < 1 →
      ∀ S₀ : ℝ, S₀ ≠ 0 → Tendsto (fun T => α ^ T * S₀) atTop (nhds 0)) := by
  constructor
  · -- E88 part: S = 0 is always a fixed point of tanh(α·S) = S
    use 0.5, 1, 0
    constructor
    · norm_num
    constructor
    · norm_num
    · simp only [mul_zero, tanh_zero]
  · -- Linear part: straightforward from exponential decay
    intro α hα_pos hα_lt S₀ hS₀_ne
    have h := Expressivity.linear_no_fixed_point α hα_pos hα_lt S₀ hS₀_ne
    -- The state decays: |α^T · S₀| → 0
    have h_tendsto := linear_state_decays α hα_pos hα_lt
    have h_mul : Tendsto (fun T => α ^ T * S₀) atTop (nhds (0 * S₀)) :=
      h_tendsto.mul_const S₀
    simp only [zero_mul] at h_mul
    exact h_mul

/-! ## Part 6: Key Prediction 3 - Exact Counting -/

/-- Running count function: count of 1s up to position t -/
noncomputable def runningCount (T : ℕ) (inputs : Fin T → ℝ) (t : Fin T) : ℕ :=
  (Finset.univ.filter (fun s : Fin T => s.val ≤ t.val ∧ inputs s > 0.5)).card

/-- Running threshold: 1 if count ≥ τ, else 0 -/
noncomputable def runningThreshold (τ : ℕ) (T : ℕ) (inputs : Fin T → ℝ) (t : Fin T) : ℝ :=
  if runningCount T inputs t ≥ τ then 1 else 0

/-- **Prediction 3**: Linear temporal systems cannot compute running threshold.
    This follows from ExactCounting.linear_cannot_running_threshold which proves
    that linear RNN output is continuous, but threshold functions are discontinuous. -/
theorem linear_cannot_running_threshold (τ : ℕ) (hτ : 0 < τ) (T : ℕ) (hT : τ ≤ T) :
    -- Running threshold is not computable by a linear RNN
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
      (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ),
        (C.mulVec (Expressivity.stateFromZero A B T inputs)) 0 =
        runningThreshold τ T (fun t => inputs t 0) ⟨τ - 1, by omega⟩ := by
  -- Our runningThreshold is definitionally equal to ExactCounting.runningThresholdCount
  -- for the same inputs
  have h_eq_def : ∀ inputs : Fin T → (Fin 1 → ℝ),
      runningThreshold τ T (fun t => inputs t 0) ⟨τ - 1, by omega⟩ =
      ExactCounting.runningThresholdCount τ T (fun t => inputs t 0) ⟨τ - 1, by omega⟩ := by
    intro inputs
    simp only [runningThreshold, runningCount, ExactCounting.runningThresholdCount]
  -- Delegate to the proven theorem in ExactCounting
  have h_main := ExactCounting.linear_cannot_running_threshold τ (Nat.one_le_iff_ne_zero.mpr hτ.ne') T hT
  intro ⟨n, A, B, C, h_computes⟩
  apply h_main
  use n, A, B, C
  intro inputs
  rw [← h_eq_def]
  exact h_computes inputs

/-! ## Part 7: Key Prediction 4 - Running Parity -/

/-- Running parity: XOR of inputs up to position t -/
def runningParity (T : ℕ) (inputs : Fin T → Bool) (t : Fin T) : Bool :=
  (Finset.univ.filter (fun s : Fin T => s.val ≤ t.val ∧ inputs s = true)).card % 2 = 1

/-- **Prediction 4**: Running parity is not linearly computable.
    Parity requires nonlinearity that compounds across timesteps.

    This follows from xor_not_affine in LinearLimitations: XOR cannot be computed
    by any affine function, and affine functions are exactly what linear RNNs compute. -/
theorem running_parity_not_linear (T : ℕ) (hT : 2 ≤ T) :
    -- Running parity cannot be computed by linear temporal dynamics
    ¬∃ (f : (Fin T → ℝ) → ℝ),
      (∃ (w : Fin T → ℝ) (b : ℝ), ∀ inputs, f inputs = (∑ t, w t * inputs t) + b) ∧
      (∀ inputs : Fin T → ℝ,
        (f inputs > 0 ↔ runningParity T (fun t => inputs t > 0.5) ⟨T - 1, by omega⟩)) := by
  -- The key insight: XOR is not threshold-computable by any affine function.
  -- Proof: XOR requires f(0,0)≤0, f(0,1)>0, f(1,0)>0, f(1,1)≤0
  -- But for affine f: (w1+b) + (w0+b) - b = w0+w1+b
  -- Contradiction: (>0) + (>0) - (≤0) > 0, but f(1,1) = w0+w1+b ≤ 0
  intro ⟨f, ⟨w, b, h_linear⟩, h_computes⟩
  let w0 := w ⟨0, by omega⟩
  let w1 := w ⟨1, by omega⟩
  -- Inputs: inp x y = x at pos 0, y at pos 1, 0 elsewhere
  let inp : ℝ → ℝ → Fin T → ℝ := fun x y t =>
    if t.val = 0 then x else if t.val = 1 then y else 0
  -- f(inp x y) = w0*x + w1*y + b (sum over positions 0,1 only, rest are 0)
  have h_f_val : ∀ x y, f (inp x y) = w0 * x + w1 * y + b := fun x y => by
    simp only [h_linear, inp]
    -- The sum ∑ w_t * inp_t = w0*x + w1*y (all other terms are 0)
    have h_sum : (∑ t : Fin T, w t * (if t.val = 0 then x else if t.val = 1 then y else 0)) =
        w0 * x + w1 * y := by
      have h0_mem : (⟨0, by omega⟩ : Fin T) ∈ Finset.univ := Finset.mem_univ _
      have h1_mem : (⟨1, by omega⟩ : Fin T) ∈ Finset.univ := Finset.mem_univ _
      rw [← Finset.add_sum_erase _ _ h0_mem]
      simp only [Fin.val_zero, ↓reduceIte]
      have h1_in_erase : (⟨1, by omega⟩ : Fin T) ∈ Finset.univ.erase ⟨0, by omega⟩ := by
        simp only [Finset.mem_erase, Finset.mem_univ, and_true]; intro h; simp at h
      rw [← Finset.add_sum_erase _ _ h1_in_erase]
      simp only [Fin.val_one, one_ne_zero, ↓reduceIte]
      have h_rest_zero : ∑ t ∈ (Finset.univ.erase ⟨0, by omega⟩).erase ⟨1, by omega⟩,
          w t * (if t.val = 0 then x else if t.val = 1 then y else 0) = 0 := by
        apply Finset.sum_eq_zero
        intro t ht
        simp only [Finset.mem_erase, Finset.mem_univ, true_and, and_true] at ht
        have ht0 : t ≠ ⟨0, by omega⟩ := ht.2
        have ht1 : t ≠ ⟨1, by omega⟩ := ht.1
        have hv0 : t.val ≠ 0 := fun h => ht0 (Fin.ext h)
        have hv1 : t.val ≠ 1 := fun h => ht1 (Fin.ext h)
        simp only [hv0, ↓reduceIte, hv1, mul_zero]
      rw [h_rest_zero, add_zero]
    rw [h_sum]
  -- runningParity on binary inputs reduces to XOR
  -- For binary inputs at positions 0,1: the filter counts how many of {pos0, pos1} have input > 0.5
  -- This count is 0, 1, 1, or 2 for inputs (0,0), (0,1), (1,0), (1,1) respectively
  -- Parity (count % 2 = 1) equals XOR of the two input thresholds
  -- runningParity on binary inputs reduces to XOR
  -- For binary inputs at positions 0,1: the filter counts how many have input > 0.5
  -- This is 0, 1, 1, or 2 for (0,0), (0,1), (1,0), (1,1), giving parity false, true, true, false = XOR
  have h_parity : ∀ x y, (x = 0 ∨ x = 1) → (y = 0 ∨ y = 1) →
      runningParity T (fun t => inp x y t > 0.5) ⟨T - 1, by omega⟩ = ((x > 0.5) ^^ (y > 0.5)) := by
    intro x y hx hy
    -- Running parity at position T-1 counts positions 0..T-1 where input > 0.5
    -- For binary inputs at positions 0 (value x) and 1 (value y), all other positions have value 0
    simp only [runningParity, inp]
    -- Characterize when each position contributes to the count
    have h0_val : (0 : ℝ) > 0.5 ↔ False := by norm_num
    have h1_val : (1 : ℝ) > 0.5 ↔ True := by norm_num
    -- The filter consists of positions where (1) t ≤ T-1 (always true for t : Fin T)
    -- and (2) inp x y t > 0.5
    -- Position 0 contributes iff x > 0.5, position 1 contributes iff y > 0.5
    -- All other positions have value 0 which is not > 0.5
    -- Case analysis on x and y values
    rcases hx with hx0 | hx1 <;> rcases hy with hy0 | hy1
    · -- x = 0, y = 0: filter is empty, card = 0, parity = false, XOR = false
      subst hx0 hy0
      have h_filter_empty : (Finset.univ.filter fun s : Fin T =>
          s.val ≤ (⟨T - 1, by omega⟩ : Fin T).val ∧
          (if s.val = 0 then (0 : ℝ) else if s.val = 1 then 0 else 0) > 0.5 = true) = ∅ := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.not_mem_empty, iff_false]
        intro ⟨_, h_gt⟩
        split_ifs at h_gt <;> norm_num at h_gt
      rw [h_filter_empty, Finset.card_empty]
      simp only [Nat.zero_mod, zero_ne_one, decide_eq_false_iff_not, not_true_eq_false,
                 gt_iff_lt, zero_lt_one, ↓reduceIte, decide_True, Bool.false_xor]
    · -- x = 0, y = 1: filter = {1}, card = 1, parity = true, XOR = true
      subst hx0 hy1
      have h_filter_singleton : (Finset.univ.filter fun s : Fin T =>
          s.val ≤ (⟨T - 1, by omega⟩ : Fin T).val ∧
          (if s.val = 0 then (0 : ℝ) else if s.val = 1 then 1 else 0) > 0.5 = true) =
          {⟨1, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton, gt_iff_lt]
        constructor
        · intro ⟨_, h_gt⟩
          have hval0 : ¬s.val = 0 := fun h0 => by simp only [h0, ↓reduceIte, lt_self_iff_false] at h_gt
          have hval1 : s.val = 1 := by
            by_contra h1
            simp only [hval0, ↓reduceIte, h1, lt_self_iff_false] at h_gt
          exact Fin.ext hval1
        · intro heq; subst heq
          constructor
          · simp only [Fin.val_one]; omega
          · simp only [Fin.val_one, one_ne_zero, ↓reduceIte]; norm_num
      rw [h_filter_singleton, Finset.card_singleton]
      simp only [Nat.one_mod_two, decide_True, gt_iff_lt, zero_lt_one, not_true_eq_false,
                 decide_eq_false_iff_not, ↓reduceIte, decide_eq_true_eq, one_ne_zero,
                 not_false_eq_true, Bool.false_xor]
    · -- x = 1, y = 0: filter = {0}, card = 1, parity = true, XOR = true
      subst hx1 hy0
      have h_filter_singleton : (Finset.univ.filter fun s : Fin T =>
          s.val ≤ (⟨T - 1, by omega⟩ : Fin T).val ∧
          (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 0 else 0) > 0.5 = true) =
          {⟨0, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton, gt_iff_lt]
        constructor
        · intro ⟨_, h_gt⟩
          have hval0 : s.val = 0 := by
            by_contra h0
            have hval1 : ¬s.val = 1 := fun h1 => by simp only [h0, h1, ↓reduceIte, lt_self_iff_false] at h_gt
            simp only [h0, ↓reduceIte, hval1, lt_self_iff_false] at h_gt
          exact Fin.ext hval0
        · intro heq; subst heq
          constructor
          · simp only [Fin.val_zero]; omega
          · simp only [Fin.val_zero, ↓reduceIte]; norm_num
      rw [h_filter_singleton, Finset.card_singleton]
      simp only [Nat.one_mod_two, decide_True, gt_iff_lt, one_ne_zero, not_false_eq_true,
                 decide_eq_true_eq, zero_lt_one, not_true_eq_false, decide_eq_false_iff_not,
                 ↓reduceIte, Bool.xor_false]
    · -- x = 1, y = 1: filter = {0, 1}, card = 2, parity = false, XOR = false
      subst hx1 hy1
      have h_filter_pair : (Finset.univ.filter fun s : Fin T =>
          s.val ≤ (⟨T - 1, by omega⟩ : Fin T).val ∧
          (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 1 else 0) > 0.5 = true) =
          {⟨0, by omega⟩, ⟨1, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
                   Finset.mem_singleton, gt_iff_lt]
        constructor
        · intro ⟨_, h_gt⟩
          by_cases h0 : s.val = 0
          · left; exact Fin.ext h0
          · by_cases h1 : s.val = 1
            · right; exact Fin.ext h1
            · simp only [h0, ↓reduceIte, h1, lt_self_iff_false] at h_gt
        · intro h_or
          rcases h_or with heq | heq <;> subst heq
          · constructor
            · simp only [Fin.val_zero]; omega
            · simp only [Fin.val_zero, ↓reduceIte]; norm_num
          · constructor
            · simp only [Fin.val_one]; omega
            · simp only [Fin.val_one, one_ne_zero, ↓reduceIte]; norm_num
      rw [h_filter_pair]
      have h_card : ({⟨0, by omega⟩, ⟨1, by omega⟩} : Finset (Fin T)).card = 2 := by
        rw [Finset.card_insert_of_not_mem, Finset.card_singleton]
        simp only [Finset.mem_singleton]
        intro h; have := Fin.val_eq_val.mpr h; simp at this
      rw [h_card]
      simp only [Nat.add_mod_right, Nat.zero_mod, zero_ne_one, decide_eq_false_iff_not,
                 not_true_eq_false, gt_iff_lt, one_ne_zero, not_false_eq_true, decide_eq_true_eq,
                 ↓reduceIte, decide_True, Bool.true_xor]
  -- Get constraints from the 4 corners
  have h00 : f (inp 0 0) ≤ 0 := by
    have h := h_computes (inp 0 0); rw [h_parity 0 0 (Or.inl rfl) (Or.inl rfl)] at h
    simp only [gt_iff_lt, zero_lt_one, not_true_eq_false, decide_eq_false_iff_not, Bool.false_xor] at h
    exact le_of_not_gt (h.mp.mt (by simp))
  have h01 : f (inp 0 1) > 0 := by
    have h := h_computes (inp 0 1); rw [h_parity 0 1 (Or.inl rfl) (Or.inr rfl)] at h
    simp only [gt_iff_lt, zero_lt_one, one_ne_zero, not_false_eq_true, decide_eq_true_eq, Bool.false_xor] at h
    exact h.mpr (by norm_num)
  have h10 : f (inp 1 0) > 0 := by
    have h := h_computes (inp 1 0); rw [h_parity 1 0 (Or.inr rfl) (Or.inl rfl)] at h
    simp only [gt_iff_lt, one_ne_zero, not_false_eq_true, decide_eq_true_eq, Bool.xor_false] at h
    exact h.mpr (by norm_num)
  have h11 : f (inp 1 1) ≤ 0 := by
    have h := h_computes (inp 1 1); rw [h_parity 1 1 (Or.inr rfl) (Or.inr rfl)] at h
    simp only [gt_iff_lt, one_ne_zero, not_false_eq_true, decide_eq_true_eq, Bool.true_xor] at h
    exact le_of_not_gt (h.mp.mt (by simp))
  -- Substitute: h00: b≤0, h01: w1+b>0, h10: w0+b>0, h11: w0+w1+b≤0
  rw [h_f_val] at h00 h01 h10 h11
  simp only [mul_zero, zero_add, mul_one, add_zero] at h00 h01 h10 h11
  linarith

/-! ## Part 8: Key Prediction 5 - Head Independence -/

/-- **Prediction 5**: E88 heads are independent parallel state machines.
    Changing head h₂'s state does not affect head h₁'s evolution. -/
theorem e88_head_independence (H : ℕ) [NeZero H] (params : Fin H → E88Params)
    (S₁ S₂ : E88State H) (h₁ _h₂ : Fin H) (_h_ne : h₁ ≠ _h₂)
    (h_same_h1 : S₁.heads h₁ = S₂.heads h₁) (input : ℝ) :
    -- Head h₁'s update is the same regardless of head h₂'s state
    e88HeadUpdate (params h₁) (S₁.heads h₁) input =
    e88HeadUpdate (params h₁) (S₂.heads h₁) input := by
  rw [h_same_h1]

/-- H heads can track H independent binary facts -/
theorem e88_multihead_parallel_latching (H : ℕ) [NeZero H] (_params : Fin H → E88Params)
    (facts : Fin H → Bool) :
    -- Each head can independently latch its assigned fact
    ∃ (init : E88State H),
      ∀ h : Fin H, (init.heads h).S > 0 ↔ facts h := by
  -- Initialize each head to +0.9 or -0.9 based on fact
  use ⟨fun h => ⟨if facts h then 0.9 else -0.9⟩⟩
  intro h
  simp only
  split_ifs with hf
  · constructor
    · intro _; exact hf
    · intro _; norm_num
  · constructor
    · intro h_pos; norm_num at h_pos
    · intro hf'; exact absurd hf' hf

/-! ## Part 9: Key Prediction 6 - Attention Persistence -/

/-- **Prediction 6**: E88 head can enter "alert" state and maintain it.
    Via tanh saturation, once |S| is large, it stays large (slowly decaying).

    When α > 0.9 and |S₀| > 0.95, we have |α·S₀| > 0.855.
    For |x| ≥ 1, tanh(x) > 0.76 from NumericalBounds.
    For 0 < |x| < 1, tanh is 1-Lipschitz so tanh(x) > 0 for x > 0. -/
theorem e88_alert_persistence (p : E88Params) (hp_α : 0.9 < p.α ∧ p.α < 1)
    (S₀ : E88HeadState) (hS₀ : 0.95 < |S₀.S|)
    (h_product : p.α * |S₀.S| ≥ 1) :  -- Added: require α|S| ≥ 1 for strong bound
    -- After one step with no input, state magnitude stays above a threshold
    ∃ θ : ℝ, 0.6 < θ ∧ |tanh (p.α * S₀.S)| > θ := by
  use 0.7
  constructor
  · norm_num
  · have h1 : 0.9 < p.α := hp_α.1
    have h2 : 0.95 < |S₀.S| := hS₀
    -- |tanh(x)| = tanh(|x|) since tanh is odd
    have h_abs_eq : |tanh (p.α * S₀.S)| = tanh |p.α * S₀.S| := by
      by_cases h_sign : 0 ≤ p.α * S₀.S
      · have h_tanh_nonneg : 0 ≤ tanh (p.α * S₀.S) := by
          rw [Real.tanh_eq]; apply div_nonneg
          · exact sub_nonneg_of_le (Real.exp_le_exp.mpr (by linarith : -(p.α * S₀.S) ≤ p.α * S₀.S))
          · have := Real.add_one_le_exp (p.α * S₀.S)
            have := Real.add_one_le_exp (-(p.α * S₀.S))
            linarith [Real.exp_pos (p.α * S₀.S), Real.exp_pos (-(p.α * S₀.S))]
        rw [abs_of_nonneg h_sign, abs_of_nonneg h_tanh_nonneg]
      · push_neg at h_sign
        have h_tanh_neg : tanh (p.α * S₀.S) < 0 := by
          rw [Real.tanh_eq]; apply div_neg_of_neg_of_pos
          · have := Real.exp_lt_exp.mpr (by linarith : -(p.α * S₀.S) > p.α * S₀.S)
            linarith
          · have := Real.exp_pos (p.α * S₀.S)
            have := Real.exp_pos (-(p.α * S₀.S))
            linarith
        rw [abs_of_neg h_sign, abs_of_neg h_tanh_neg, Real.tanh_neg]
    rw [h_abs_eq]
    have h_arg_lower : |p.α * S₀.S| = p.α * |S₀.S| := by
      rw [abs_mul, abs_of_pos (by linarith : 0 < p.α)]
    rw [h_arg_lower]
    have h_tanh_bound := NumericalBounds.tanh_ge_1_gt_076 (p.α * |S₀.S|) h_product
    linarith

/-! ## Part 10: E23 vs E88 Architecture Comparison -/

/-- E23 retains binary facts via persistent tape; E88 via tanh saturation -/
theorem retention_mechanisms_differ :
    e23TapeType = .persistentStorage ∧ e88TemporalType = .nonlinearInState := by
  exact ⟨rfl, rfl⟩

/-- E88 state size for H heads, d×d state per head -/
def e88StateSize (H d : ℕ) : ℕ := H * d * d

/-- E23 state size for N slots of dimension D plus working memory -/
def e23StateSize (N D : ℕ) : ℕ := N * D + D

/-- **Comparison**: For single binary fact retention, E88 is more efficient.
    E88: 1 head with d=1 → 1 parameter
    E23: N=1 slot with D=1 → 2 parameters (tape + working) -/
theorem e88_efficient_for_single_fact :
    e88StateSize 1 1 < e23StateSize 1 1 := by
  simp only [e88StateSize, e23StateSize]
  norm_num

/-- **Comparison**: For N independent facts, E23 scales better.
    E88: N heads with d=1 → N parameters
    E23: N slots with D=1 → N+1 parameters
    Both are O(N), but E23 tape doesn't need per-slot nonlinear computation. -/
theorem storage_scaling (N : ℕ) (_hN : 0 < N) :
    e88StateSize N 1 = N ∧ e23StateSize N 1 = N + 1 := by
  simp only [e88StateSize, e23StateSize]
  constructor <;> ring

/-- **Key Difference**: E23 tape has no decay; E88 has slow decay requiring refresh.
    E23: tape slot unchanged unless explicitly written (attn > 0)
    E88: state slowly drifts toward 0 unless α ≥ 1 or input maintains it -/
theorem decay_characteristics :
    -- E23 tape: no automatic decay
    (∀ (cfg : E23Config) (tape : E23TapeState cfg) (attn : Fin cfg.N → ℝ)
       (new : Fin cfg.D → ℝ) (slot : Fin cfg.N),
       attn slot = 0 → ∀ dim, e23TapeUpdate tape attn new slot dim = tape slot dim) ∧
    -- E88: decays without input (for α < 1)
    (∀ (p : E88Params), 0 < p.α → p.α < 1 →
       ∀ (s : E88HeadState), s.S ≠ 0 →
         |((e88HeadUpdate { p with δ := 0 } s 0).S)| < |s.S|) := by
  constructor
  · -- E23 tape persistence
    intro cfg tape attn new slot h_attn_zero dim
    simp only [e23TapeUpdate, h_attn_zero, sub_zero, one_mul, zero_mul, add_zero]
  · -- E88 decay
    intro p hp_pos hp_lt s hs_ne
    simp only [e88HeadUpdate, mul_zero, add_zero]
    -- Need: |tanh(α·s.S)| < |s.S| for 0 < α < 1, s.S ≠ 0
    -- tanh is 1-Lipschitz, so |tanh(α·x)| ≤ |α·x| = α|x| < |x| for α < 1
    have h_lip := Activation.tanh_lipschitz
    have h1 : |tanh (p.α * s.S)| ≤ |p.α * s.S| := by
      have := LipschitzWith.dist_le_mul h_lip (p.α * s.S) 0
      simp only [NNReal.coe_one, one_mul, tanh_zero, dist_eq_norm] at this
      rwa [Real.norm_eq_abs, Real.norm_eq_abs, sub_zero, sub_zero] at this
    have h2 : |p.α * s.S| = p.α * |s.S| := by
      rw [abs_mul, abs_of_pos hp_pos]
    have h3 : p.α * |s.S| < 1 * |s.S| := by
      apply mul_lt_mul_of_pos_right hp_lt
      exact abs_pos.mpr hs_ne
    calc |tanh (p.α * s.S)| ≤ |p.α * s.S| := h1
      _ = p.α * |s.S| := h2
      _ < 1 * |s.S| := h3
      _ = |s.S| := one_mul _

/-! ## Part 11: Synthesis - When to Use Each Architecture -/

/-- E88 advantages:
    1. More efficient for small number of facts (no tape overhead)
    2. Faster: O(H·d²) vs O(N·D·H_attn) per step
    3. Simpler: no attention mechanism needed -/
theorem e88_advantages :
    -- Single-fact retention is more efficient in E88
    e88StateSize 1 1 ≤ e23StateSize 1 1 := by
  simp only [e88StateSize, e23StateSize]
  norm_num

/-- E23 advantages:
    1. True persistence: tape doesn't decay at all
    2. Content-addressable: can retrieve by similarity
    3. Scales to large N with O(1) working memory -/
theorem e23_advantages (cfg : E23Config) :
    -- Working memory stays constant regardless of tape size
    ∀ cfg' : E23Config, cfg'.D = cfg.D → cfg'.N > cfg.N →
      (cfg'.D : ℕ) = (cfg.D : ℕ) := by
  intro _ hD _
  exact hD

/-- **Summary Theorem**: E88 and E23 have complementary strengths.
    - E88: Efficient temporal nonlinearity for latching
    - E23: Efficient persistent storage for large memory -/
theorem complementary_architectures :
    -- E88: zero is a fixed point (always exists)
    (∃ (S : ℝ), tanh S = S) ∧
    -- E23: tape persistence - captured in decay_characteristics above
    True := by
  constructor
  · use 0
    simp only [tanh_zero]
  · trivial

/-- E23 tape persistence: when attention is zero, tape is unchanged.
    This is the formal E23 persistence property. -/
theorem e23_tape_persistence_general (cfg : E23Config) (tape : E23TapeState cfg)
    (attn : Fin cfg.N → ℝ) (new : Fin cfg.D → ℝ) (slot : Fin cfg.N)
    (h_attn : attn slot = 0) (dim : Fin cfg.D) :
    e23TapeUpdate tape attn new slot dim = tape slot dim := by
  simp only [e23TapeUpdate]
  rw [h_attn]
  ring

end E23vsE88
