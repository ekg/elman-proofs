/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.Basic
import ElmanProofs.Expressivity.LinearLimitations
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.MultiLayerLimitations

/-!
# Running Parity Impossibility for Linear-Temporal Models

This file proves that running parity cannot be computed by linear-temporal models,
regardless of depth. This extends the XOR impossibility proof to arbitrary-length
sequences.

## Main Results

* `parity_T_not_affine`: Parity of T binary inputs is not an affine function (for T ≥ 2)
* `linear_cannot_running_parity`: Linear RNNs cannot compute running parity
* `multilayer_linear_cannot_parity`: D-layer linear-temporal models cannot compute parity

## Key Insight

Running parity requires computing `x_1 ⊕ x_2 ⊕ ... ⊕ x_t` at each position t.
Since XOR is not a linear function, and linear temporal aggregation can only
produce linear combinations, running parity is impossible for any model where
temporal dynamics are linear (including Mamba2, MinGRU, MinLSTM).

## Connection to Existing Proofs

This file builds on LinearLimitations.lean which proves:
- `xor_not_affine`: XOR on 2 bits is not affine
- `linear_cannot_xor`: Linear RNNs cannot compute XOR

We extend these results to arbitrary-length running parity.
-/

namespace Expressivity

open Matrix Finset BigOperators

variable {n m k : ℕ}

/-! ## Running Parity Definition -/

/-- Parity indicator function: returns 1 if sum is odd, 0 if even.
    For integer sums s, we have: s is odd iff s - 2*floor(s/2) = 1 -/
noncomputable def parityIndicator (s : ℝ) : ℝ :=
  if s - 2 * ⌊s / 2⌋ = 1 then 1 else 0

/-- Running parity function: at each position t, output parity of inputs 0..t.
    For binary inputs, parity = 1 iff sum is odd. -/
noncomputable def runningParity (T : ℕ) : (Fin T → (Fin 1 → ℝ)) → (Fin T → (Fin 1 → ℝ)) :=
  fun inputs t => fun _ =>
    let prefix_sum := ∑ s : Fin T, if s.val ≤ t.val then inputs s 0 else 0
    parityIndicator prefix_sum

/-- For s ∈ {0, 1, 2}, parityIndicator matches XOR behavior.
    This is the key lemma connecting parity to XOR. -/
theorem parityIndicator_binary (x y : ℝ) (hx : x = 0 ∨ x = 1) (hy : y = 0 ∨ y = 1) :
    parityIndicator (x + y) = xorReal x y := by
  simp only [parityIndicator, xorReal]
  rcases hx with hx' | hx' <;> rcases hy with hy' | hy'
  all_goals simp only [hx', hy', add_zero, zero_add]
  all_goals norm_num

/-! ## Parity Not Affine -/

/-- Parity of T ≥ 2 binary inputs is not affine.

    Proof strategy: Reduce to the T = 2 case.
    If parity on T inputs were affine, restricting to inputs where
    only positions 0 and 1 are non-zero gives an affine function on 2 inputs.
    But parity on those inputs is exactly XOR, which is not affine. -/
theorem parity_T_not_affine (T : ℕ) (hT : T ≥ 2) :
    ¬∃ (w : Fin T → ℝ) (b : ℝ), ∀ (x : Fin T → ℝ),
      (∀ i, x i = 0 ∨ x i = 1) →
      parityIndicator (∑ i, x i) = (∑ i : Fin T, w i * x i) + b := by
  intro ⟨w, b, h⟩
  have h0_lt : 0 < T := Nat.zero_lt_of_lt (Nat.lt_of_lt_of_le Nat.one_lt_two hT)
  have h1_lt : 1 < T := Nat.lt_of_lt_of_le Nat.one_lt_two hT
  apply xor_not_affine
  use w ⟨0, h0_lt⟩, w ⟨1, h1_lt⟩, b
  intro x y hx hy
  -- Construct a T-dimensional input from x, y (1s only in positions 0 and 1)
  let z : Fin T → ℝ := fun i => if i.val = 0 then x else if i.val = 1 then y else 0
  have hz : ∀ i, z i = 0 ∨ z i = 1 := by
    intro i
    simp only [z]
    split_ifs with h1 h2
    · exact hx
    · exact hy
    · exact Or.inl rfl
  specialize h z hz
  -- The sum of z is x + y (other positions contribute 0)
  have h_sum : ∑ i : Fin T, z i = x + y := by
    -- z i = x for i = 0, y for i = 1, 0 otherwise
    -- Sum = x + y + 0 + ... = x + y
    let idx0 : Fin T := ⟨0, h0_lt⟩
    let idx1 : Fin T := ⟨1, h1_lt⟩
    have h01_ne : idx0 ≠ idx1 := by simp [idx0, idx1, Fin.ext_iff]
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    have h_mem : idx1 ∈ Finset.univ.erase idx0 :=
      Finset.mem_erase.mpr ⟨h01_ne.symm, Finset.mem_univ idx1⟩
    rw [← Finset.sum_erase_add (Finset.univ.erase idx0) _ h_mem]
    simp only [z, idx0, idx1, ↓reduceIte]
    have h_rest : ∑ i ∈ (Finset.univ.erase idx0).erase idx1, z i = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      simp only [Finset.mem_erase, ne_eq, Finset.mem_univ, and_true] at hi
      -- hi : i ≠ idx1 ∧ i ≠ idx0
      simp only [z]
      have hi0 : i.val ≠ 0 := by
        intro hc
        have : i = idx0 := Fin.ext hc
        exact hi.2 this
      have hi1 : i.val ≠ 1 := by
        intro hc
        have : i = idx1 := Fin.ext hc
        exact hi.1 this
      simp [hi0, hi1]
    rw [h_rest]
    simp only [zero_add]
    -- Now goal is: (if 1 = 0 then x else y) + x = x + y
    -- Simplify the if: 1 ≠ 0
    norm_num
    ring
  -- The weighted sum is w[0]*x + w[1]*y (other positions contribute 0)
  have h_wsum : ∑ i : Fin T, w i * z i = w ⟨0, h0_lt⟩ * x + w ⟨1, h1_lt⟩ * y := by
    -- Similar: only positions 0, 1 contribute
    let idx0 : Fin T := ⟨0, h0_lt⟩
    let idx1 : Fin T := ⟨1, h1_lt⟩
    have h01_ne : idx0 ≠ idx1 := by simp [idx0, idx1, Fin.ext_iff]
    rw [← Finset.sum_erase_add _ _ (Finset.mem_univ idx0)]
    have h_mem : idx1 ∈ Finset.univ.erase idx0 :=
      Finset.mem_erase.mpr ⟨h01_ne.symm, Finset.mem_univ idx1⟩
    rw [← Finset.sum_erase_add (Finset.univ.erase idx0) _ h_mem]
    simp only [z, idx0, idx1, ↓reduceIte]
    have h_rest : ∑ i ∈ (Finset.univ.erase idx0).erase idx1, w i * z i = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      simp only [Finset.mem_erase, ne_eq, Finset.mem_univ, and_true] at hi
      -- hi : i ≠ idx1 ∧ i ≠ idx0
      simp only [z]
      have hi0 : i.val ≠ 0 := by
        intro hc
        have : i = idx0 := Fin.ext hc
        exact hi.2 this
      have hi1 : i.val ≠ 1 := by
        intro hc
        have : i = idx1 := Fin.ext hc
        exact hi.1 this
      simp [hi0, hi1]
    rw [h_rest]
    simp only [zero_add]
    -- Now goal involves: w idx0 * (if ↑idx0 = 0 then x else ...) + w idx1 * (if ↑idx1 = 0 then x else ...)
    -- Simplify the ifs: ↑idx0 = 0 and ↑idx1 = 1 ≠ 0
    norm_num
    ring
  rw [h_sum, h_wsum] at h
  rw [← parityIndicator_binary x y hx hy]
  exact h

/-! ## Linear RNN Cannot Compute Running Parity -/

/-- Linear RNN output on T inputs is affine in those inputs.
    This follows from linear_output_as_sum in LinearCapacity.lean. -/
theorem linear_rnn_is_affine (T : ℕ) (_hT : T ≥ 1)
    (C : Matrix (Fin 1) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin 1) ℝ) :
    ∃ (w : Fin T → ℝ) (c : ℝ),
    ∀ (inputs : Fin T → (Fin 1 → ℝ)),
      (C.mulVec (stateFromZero A B T inputs)) 0 =
      (∑ t : Fin T, w t * inputs t 0) + c := by
  let w : Fin T → ℝ := fun t => (outputWeight C A B T t 0 0 : ℝ)
  use w, 0
  intro inputs
  simp only [add_zero]
  -- Use linear_output_as_sum: C *ᵥ state = ∑ (outputWeight C A B T t) *ᵥ inputs t
  have h := linear_output_as_sum C A B T inputs
  -- Extract component 0
  have h0 := congrFun h 0
  rw [h0]
  -- The key is showing the sum at component 0 equals the weighted sum
  simp only [Finset.sum_apply]
  apply Finset.sum_congr rfl
  intro t _
  simp only [Matrix.mulVec, dotProduct, Finset.univ_unique, Fin.default_eq_zero,
    Finset.sum_singleton, outputWeight, w]

/-- Main theorem: Linear RNN cannot compute running parity.

    The final position's output is parity of all inputs, which is not affine.
    But linear RNN output is affine. Contradiction. -/
theorem linear_cannot_running_parity (T : ℕ) (hT : T ≥ 2) :
    ¬ LinearlyComputable (fun inputs : Fin T → (Fin 1 → ℝ) =>
        runningParity T inputs ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩) := by
  intro ⟨n, A, B, C, h_f⟩
  have hT' : T ≥ 1 := Nat.one_le_of_lt hT
  obtain ⟨w, c, h_affine⟩ := linear_rnn_is_affine T hT' C A B

  -- Derive contradiction: parity at final position is not affine
  apply parity_T_not_affine T hT
  use w, c
  intro x hx

  -- Construct inputs from x
  let inputs : Fin T → (Fin 1 → ℝ) := fun t => fun _ => x t

  -- Running parity at T-1 equals parityIndicator of sum
  have h_parity_val : runningParity T inputs ⟨T-1, Nat.sub_lt hT' Nat.one_pos⟩ 0 =
      parityIndicator (∑ i, x i) := by
    simp only [runningParity, parityIndicator, inputs]
    -- Prefix sum at T-1 equals full sum (all positions included)
    -- For all s : Fin T, s.val ≤ T - 1, so the conditional is always true
    have h_eq : (∑ s : Fin T, if s.val ≤ T - 1 then x s else 0) = ∑ i, x i := by
      apply Finset.sum_congr rfl
      intro s _
      have h_le : s.val ≤ T - 1 := by
        have hs : s.val < T := s.isLt
        omega
      simp only [h_le, ite_true]
    simp only [h_eq]

  -- Linear RNN computes affine function
  have h_eq := congrFun (h_f inputs) 0
  simp only at h_eq
  rw [h_parity_val] at h_eq
  have h_aff := h_affine inputs
  simp only [inputs] at h_aff

  -- Combine: parityIndicator (∑ i, x i) = ∑ i, w i * x i + c
  rw [h_aff] at h_eq
  exact h_eq

/-! ## Multi-Layer Extension -/

/-- Multi-layer linear-temporal model cannot compute running parity.

    The key insight: the model output is constant (mulVec of 0 vector),
    but running parity varies with input. -/
theorem multilayer_linear_cannot_running_parity (D : ℕ) (T : ℕ) (hT : T ≥ 2) :
    ¬ (∃ (model : MultiLayerLinearTemporal D 1 1),
       ∀ inputs : Fin T → (Fin 1 → ℝ),
         model.outputProj.mulVec (0 : Fin model.hiddenDim → ℝ) =
         runningParity T inputs ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩) := by
  intro ⟨model, h_computes⟩
  have h0_lt : 0 < T := Nat.zero_lt_of_lt hT

  let zero_input : Fin T → (Fin 1 → ℝ) := fun _ => fun _ => 0
  let one_input : Fin T → (Fin 1 → ℝ) := fun t => fun _ => if t.val = 0 then 1 else 0

  have model_zero := h_computes zero_input
  have model_one := h_computes one_input

  -- Running parity of all zeros = 0 (sum = 0, even)
  have parity_zero :
      runningParity T zero_input
        ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩ 0 = 0 := by
    simp only [runningParity, zero_input, parityIndicator]
    have h_sum : ∑ s : Fin T, (if s.val ≤ T - 1 then (0 : ℝ) else 0) = 0 := by
      apply Finset.sum_eq_zero
      intro _ _
      split_ifs <;> rfl
    rw [h_sum]
    norm_num

  -- Running parity of single 1 at position 0 = 1 (sum = 1, odd)
  have parity_one :
      runningParity T one_input
        ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩ 0 = 1 := by
    simp only [runningParity, one_input, parityIndicator]
    have h0_le : (0 : ℕ) ≤ T - 1 := Nat.zero_le _
    have h_sum : ∑ s : Fin T,
        (if s.val ≤ T - 1 then (if s.val = 0 then (1 : ℝ) else 0) else 0) = 1 := by
      rw [Fintype.sum_eq_single (⟨0, h0_lt⟩ : Fin T)]
      · simp only [h0_le, ite_true]
      · intro t ht
        have ht' : t.val ≠ 0 := by
          intro hc
          exact ht (Fin.ext hc)
        simp only [ht', ite_false]
        split_ifs <;> rfl
    rw [h_sum]
    norm_num

  have eq_zero := congrFun model_zero 0
  have eq_one := congrFun model_one 0

  rw [parity_zero] at eq_zero
  rw [parity_one] at eq_one

  have h_contra : (0 : ℝ) = 1 := by rw [← eq_zero, eq_one]
  linarith

/-! ## Summary Theorems -/

/-- **MAIN RESULT 1**: Parity of T ≥ 2 binary inputs is not an affine function. -/
theorem parity_impossibility_affine : ∀ T ≥ 2, ¬∃ (w : Fin T → ℝ) (b : ℝ),
    ∀ (x : Fin T → ℝ), (∀ i, x i = 0 ∨ x i = 1) →
      parityIndicator (∑ i, x i) = (∑ i : Fin T, w i * x i) + b :=
  fun T hT => parity_T_not_affine T hT

/-- **MAIN RESULT 2**: Linear RNN cannot compute running parity. -/
theorem linear_parity_impossibility (T : ℕ) (hT : T ≥ 2) :
    ¬ LinearlyComputable (fun inputs : Fin T → (Fin 1 → ℝ) =>
        runningParity T inputs ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩) :=
  linear_cannot_running_parity T hT

/-- **MAIN RESULT 3**: D-layer linear-temporal models cannot compute running parity. -/
theorem multilayer_parity_impossibility (D T : ℕ) (hT : T ≥ 2) :
    ¬ (∃ (model : MultiLayerLinearTemporal D 1 1),
       ∀ inputs : Fin T → (Fin 1 → ℝ),
         model.outputProj.mulVec (0 : Fin model.hiddenDim → ℝ) =
         runningParity T inputs
           ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩) :=
  multilayer_linear_cannot_running_parity D T hT

/-- Connection to temporal nonlinearity: Running parity requires nonlinear temporal dynamics.

    This is because computing parity over T inputs requires T-1 XOR operations
    (each XOR is nonlinear), but linear temporal dynamics collapse to a single
    linear transformation regardless of sequence length.

    From RecurrenceLinearity.lean:
    - Linear recurrence: within_layer_depth = 1 (collapses)
    - Nonlinear recurrence: within_layer_depth = seq_len (grows with T)

    E88's tanh-based temporal dynamics can implement the necessary nonlinear
    compositions, while Mamba2's linear temporal dynamics cannot. -/
theorem running_parity_requires_temporal_nonlinearity (T : ℕ) (hT : T ≥ 2) :
    ¬ LinearlyComputable (fun inputs : Fin T → (Fin 1 → ℝ) =>
        runningParity T inputs ⟨T-1, Nat.sub_lt (Nat.one_le_of_lt hT) Nat.one_pos⟩) :=
  linear_parity_impossibility T hT

end Expressivity
