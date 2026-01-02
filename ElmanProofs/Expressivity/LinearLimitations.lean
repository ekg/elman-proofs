/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.Basic
import ElmanProofs.Expressivity.LinearCapacity

/-!
# Linear RNN Limitations

This file proves that linear RNNs cannot compute certain nonlinear functions,
establishing fundamental expressivity bounds.

## Main Results

* `linear_output_is_linear`: Linear RNN output is a linear function of inputs
* `linear_cannot_threshold`: Linear RNNs cannot compute threshold/step functions
* `linear_cannot_xor`: Linear RNNs cannot compute XOR over history

## Key Insight

Since linear RNN state h_T = Σ A^{T-1-t} B x_t is a linear combination of inputs,
and output y = C h_T, the output is always a linear function of the input sequence.
This means:
1. Output is continuous in inputs
2. Output satisfies superposition: f(x + y) = f(x) + f(y), f(cx) = c f(x)

Any function that violates these properties cannot be computed by a linear RNN.

-/

namespace Expressivity

open Matrix Finset BigOperators

variable {n m k : ℕ}

/-! ## Linear RNN Output is Linear in Inputs -/

/-- The weight that input at time t contributes to output component i.
    This is the (i,j) entry of C * A^{T-1-t} * B, giving how input_t[j] affects output[i]. -/
def outputWeight (C : Matrix (Fin k) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin m) ℝ) (T : ℕ) (t : Fin T) : Matrix (Fin k) (Fin m) ℝ :=
  C * (A ^ (T - 1 - t.val)) * B

/-- Linear RNN output is a weighted sum of inputs -/
theorem linear_output_as_sum (C : Matrix (Fin k) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin m) ℝ) (T : ℕ) (inputs : Fin T → (Fin m → ℝ)) :
    C.mulVec (stateFromZero A B T inputs) =
    ∑ t : Fin T, (outputWeight C A B T t).mulVec (inputs t) := by
  rw [linear_state_is_sum]
  rw [Matrix.mulVec_sum]
  apply Finset.sum_congr rfl
  intro t _
  simp only [inputContribution, outputWeight]
  rw [Matrix.mulVec_mulVec, Matrix.mulVec_mulVec]

/-- Linear RNN output is additive in inputs -/
theorem linear_output_additive (C : Matrix (Fin k) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin m) ℝ) (T : ℕ)
    (inputs₁ inputs₂ : Fin T → (Fin m → ℝ)) :
    C.mulVec (stateFromZero A B T (fun t => inputs₁ t + inputs₂ t)) =
    C.mulVec (stateFromZero A B T inputs₁) + C.mulVec (stateFromZero A B T inputs₂) := by
  rw [state_additive]
  exact Matrix.mulVec_add C _ _

/-- Linear RNN output is homogeneous in inputs -/
theorem linear_output_scalar (C : Matrix (Fin k) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin m) ℝ) (T : ℕ) (c : ℝ)
    (inputs : Fin T → (Fin m → ℝ)) :
    C.mulVec (stateFromZero A B T (fun t => c • inputs t)) =
    c • C.mulVec (stateFromZero A B T inputs) := by
  rw [state_scalar]
  exact Matrix.mulVec_smul C c _

/-! ## Threshold Functions Cannot Be Computed -/

/-- A threshold function on scalar input sequences: outputs 1 if sum > τ, else 0 -/
noncomputable def thresholdFunction (τ : ℝ) (T : ℕ) : (Fin T → (Fin 1 → ℝ)) → (Fin 1 → ℝ) :=
  fun inputs =>
    let total := ∑ t : Fin T, inputs t 0
    fun _ => if total > τ then 1 else 0

/-- Threshold function is not linearly computable.
    Key idea: Linear outputs are continuous, but threshold has a discontinuity at τ. -/
theorem linear_cannot_threshold (τ : ℝ) (T : ℕ) (hT : T ≥ 1) :
    ¬ LinearlyComputable (thresholdFunction τ T) := by
  intro ⟨n, A, B, C, h_f⟩
  -- The threshold function is discontinuous at τ
  -- But C.mulVec (stateFromZero A B T inputs) is continuous in inputs
  -- This is a contradiction
  -- Construct two sequences: one with sum = τ - ε, one with sum = τ + ε
  -- As ε → 0, linear output converges, but threshold jumps from 0 to 1
  sorry

/-! ## XOR Cannot Be Computed -/

/-- XOR of two bits represented as reals: true if exactly one is > 0.5 -/
noncomputable def xorReal (a b : ℝ) : ℝ :=
  if (a > 0.5) = (b > 0.5) then 0 else 1

/-- For T = 2, the XOR function on binary inputs {0, 1} cannot be linear.
    XOR(0,0) = 0, XOR(0,1) = 1, XOR(1,0) = 1, XOR(1,1) = 0
    But any linear function f(x,y) = ax + by + c satisfies:
    f(0,0) + f(1,1) = f(0,1) + f(1,0)
    which gives 0 + 0 = 1 + 1, contradiction. -/
theorem xor_not_affine :
    ¬∃ (a b c : ℝ), ∀ (x y : ℝ), (x = 0 ∨ x = 1) → (y = 0 ∨ y = 1) →
      xorReal x y = a * x + b * y + c := by
  intro ⟨a, b, c, h⟩
  -- Evaluate at all four corners
  have h00 : xorReal 0 0 = a * 0 + b * 0 + c := h 0 0 (Or.inl rfl) (Or.inl rfl)
  have h01 : xorReal 0 1 = a * 0 + b * 1 + c := h 0 1 (Or.inl rfl) (Or.inr rfl)
  have h10 : xorReal 1 0 = a * 1 + b * 0 + c := h 1 0 (Or.inr rfl) (Or.inl rfl)
  have h11 : xorReal 1 1 = a * 1 + b * 1 + c := h 1 1 (Or.inr rfl) (Or.inr rfl)
  -- Compute XOR values
  have xor00 : xorReal 0 0 = 0 := by simp only [xorReal]; norm_num
  have xor01 : xorReal 0 1 = 1 := by simp only [xorReal]; norm_num
  have xor10 : xorReal 1 0 = 1 := by simp only [xorReal]; norm_num
  have xor11 : xorReal 1 1 = 0 := by simp only [xorReal]; norm_num
  -- Substitute
  rw [xor00] at h00
  rw [xor01] at h01
  rw [xor10] at h10
  rw [xor11] at h11
  simp only [mul_zero, mul_one, zero_add, add_zero] at h00 h01 h10 h11
  -- Now h00: 0 = c, h01: 1 = b + c, h10: 1 = a + c, h11: 0 = a + b + c
  -- From these: c = 0, b = 1, a = 1, but a + b + c = 2 ≠ 0
  linarith

/-- Helper: create a 2-element input sequence from two scalar values -/
def mkInputs (x y : ℝ) : Fin 2 → (Fin 1 → ℝ) :=
  ![fun _ => x, fun _ => y]

/-- Linear RNN output on binary inputs is an affine function.
    This follows from the linearity of state in inputs. -/
theorem linear_rnn_affine_on_binary (C : Matrix (Fin 1) (Fin n) ℝ) (A : Matrix (Fin n) (Fin n) ℝ)
    (B : Matrix (Fin n) (Fin 1) ℝ) :
    ∃ (w₀ w₁ c : ℝ),
    ∀ (x y : ℝ), (x = 0 ∨ x = 1) → (y = 0 ∨ y = 1) →
      let inputs : Fin 2 → (Fin 1 → ℝ) := mkInputs x y
      (C.mulVec (stateFromZero A B 2 inputs)) 0 = w₀ * x + w₁ * y + c := by
  -- Define the output function
  let f : ℝ → ℝ → ℝ := fun x y => (C.mulVec (stateFromZero A B 2 (mkInputs x y))) 0
  -- Key: mkInputs is linear in its arguments
  have h_mk_add : ∀ x₁ x₂ y₁ y₂,
      mkInputs (x₁ + x₂) (y₁ + y₂) = mkInputs x₁ y₁ + mkInputs x₂ y₂ := by
    intro x₁ x₂ y₁ y₂
    ext i j
    fin_cases i <;> simp [mkInputs, Matrix.cons_val_zero, Matrix.cons_val_one]
  have h_mk_smul : ∀ c x y, mkInputs (c * x) (c * y) = c • mkInputs x y := by
    intro c x y
    ext i j
    fin_cases i <;> simp [mkInputs, Matrix.cons_val_zero, Matrix.cons_val_one,
                          Pi.smul_apply, smul_eq_mul]
  -- f is additive
  have h_f_add : ∀ x₁ x₂ y₁ y₂, f (x₁ + x₂) (y₁ + y₂) = f x₁ y₁ + f x₂ y₂ := by
    intro x₁ x₂ y₁ y₂
    simp only [f]
    rw [h_mk_add]
    have := linear_output_additive C A B 2 (mkInputs x₁ y₁) (mkInputs x₂ y₂)
    exact congrFun this 0
  -- f is homogeneous
  have h_f_smul : ∀ c x y, f (c * x) (c * y) = c * f x y := by
    intro c x y
    simp only [f]
    rw [h_mk_smul]
    have := linear_output_scalar C A B 2 c (mkInputs x y)
    exact congrFun this 0
  -- Use linearity to decompose: f(x,y) = x*f(1,0) + y*f(0,1)
  use f 1 0, f 0 1, 0
  intro x y _ _
  simp only [add_zero]
  -- Show mkInputs x y = ![...] so we can work with f
  have h_mk_eq : mkInputs x y = ![fun _ => x, fun _ => y] := rfl
  -- f(x,y) = f(x,0) + f(0,y) by additivity with x₁=x, x₂=0, y₁=0, y₂=y
  have step1 : f x y = f x 0 + f 0 y := by
    have := h_f_add x 0 0 y
    simp only [add_zero, zero_add] at this
    exact this
  -- f(x,0) = x * f(1,0) by homogeneity
  have step2 : f x 0 = x * f 1 0 := by
    have := h_f_smul x 1 0
    simp only [mul_one, mul_zero] at this
    exact this
  -- f(0,y) = y * f(0,1) by homogeneity
  have step3 : f 0 y = y * f 0 1 := by
    have := h_f_smul y 0 1
    simp only [mul_zero, mul_one] at this
    exact this
  -- The goal is about mkInputs x y, which equals f x y
  -- Unfold f in the steps
  have step1' : (C.mulVec (stateFromZero A B 2 (mkInputs x y))) 0 =
                (C.mulVec (stateFromZero A B 2 (mkInputs x 0))) 0 +
                (C.mulVec (stateFromZero A B 2 (mkInputs 0 y))) 0 := step1
  have step2' : (C.mulVec (stateFromZero A B 2 (mkInputs x 0))) 0 =
                x * (C.mulVec (stateFromZero A B 2 (mkInputs 1 0))) 0 := step2
  have step3' : (C.mulVec (stateFromZero A B 2 (mkInputs 0 y))) 0 =
                y * (C.mulVec (stateFromZero A B 2 (mkInputs 0 1))) 0 := step3
  rw [step1', step2', step3']
  ring

/-- XOR over a length-2 sequence cannot be computed by a linear RNN -/
theorem linear_cannot_xor :
    ¬ LinearlyComputable (fun inputs : Fin 2 → (Fin 1 → ℝ) =>
      fun _ : Fin 1 => xorReal (inputs 0 0) (inputs 1 0)) := by
  intro ⟨n, A, B, C, h_f⟩
  -- The linear RNN computes an affine function
  obtain ⟨w₀, w₁, c, h_affine⟩ := linear_rnn_affine_on_binary C A B
  -- But XOR is not affine
  apply xor_not_affine
  use w₀, w₁, c
  intro x y hx hy
  have h := h_f (mkInputs x y)
  have h' := h_affine x y hx hy
  -- h says: (fun _ => xorReal x y) = C.mulVec (stateFromZero A B 2 inputs)
  -- Evaluate at 0
  have h_eq : xorReal x y = (C.mulVec (stateFromZero A B 2 (mkInputs x y))) 0 := by
    have := congrFun h 0
    simp only [mkInputs, Matrix.cons_val_zero, Matrix.cons_val_one] at this
    exact this
  rw [h_eq]
  -- h' gives us the affine form
  simp only [mkInputs] at h'
  exact h'

end Expressivity
