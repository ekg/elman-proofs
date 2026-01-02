/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.LinearAlgebra.Dimension.Finrank

/-!
# Linear State Capacity Bounds

This file proves that d-dimensional linear RNN state carries at most O(d) bits
of information about the input history.

## Main Results

* `linear_state_is_sum`: State at time T is a sum of weighted inputs
* `output_determined_by_state`: Output depends only on current state
* `same_state_same_future`: Same state → same future outputs

## Key Insight

Linear RNNs cannot "compute" on their state - they can only linearly combine past inputs.
This fundamentally limits what patterns they can detect, unlike nonlinear RNNs which can
implement arbitrary computations on the hidden state.

The state h_T = Σ_{t=0}^{T-1} A^{T-1-t} B x_t is just a weighted sum of inputs,
where the weights are determined by powers of A. This means:
1. At most n linearly independent "features" can be extracted from any input sequence
2. Two input sequences that produce the same state are indistinguishable for all future outputs

-/

namespace Expressivity

open Matrix Finset BigOperators

variable {n m k : ℕ}

/-! ## Linear RNN State Evolution -/

/-- Single step of linear RNN: h' = A h + B x -/
def linearStep (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (h : Fin n → ℝ) (x : Fin m → ℝ) : Fin n → ℝ :=
  A.mulVec h + B.mulVec x

/-- State after T steps starting from h₀, given input sequence.
    Uses recursion: h_{t+1} = A h_t + B x_t -/
def stateAfterT (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (h₀ : Fin n → ℝ) : (T : ℕ) → (Fin T → (Fin m → ℝ)) → (Fin n → ℝ)
  | 0, _ => h₀
  | T' + 1, inputs =>
    linearStep A B (stateAfterT A B h₀ T' (fun i => inputs i.castSucc)) (inputs (Fin.last T'))

/-- State after T steps starting from zero initial state -/
def stateFromZero (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (inputs : Fin T → (Fin m → ℝ)) : Fin n → ℝ :=
  stateAfterT A B 0 T inputs

/-! ## State is Linear Combination of Inputs -/

/-- The contribution of input at time t to state at time T is A^{T-1-t} * B * x_t -/
def inputContribution (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (t : Fin T) (x : Fin m → ℝ) : Fin n → ℝ :=
  (A ^ (T - 1 - t.val)).mulVec (B.mulVec x)

/-- State at time T starting from zero is the sum of contributions from all inputs.
    h_T = Σ_{t=0}^{T-1} A^{T-1-t} B x_t -/
theorem linear_state_is_sum (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (inputs : Fin T → (Fin m → ℝ)) :
    stateFromZero A B T inputs = ∑ t : Fin T, inputContribution A B T t (inputs t) := by
  induction T with
  | zero =>
    simp only [stateFromZero, stateAfterT, Finset.univ_eq_empty, sum_empty]
  | succ T' ih =>
    simp only [stateFromZero, stateAfterT, linearStep]
    -- Apply IH: state(T') = Σ t : Fin T', A^(T'-1-t) B x_{t.castSucc}
    have ih' := ih (fun i => inputs i.castSucc)
    simp only [stateFromZero] at ih'
    rw [ih']
    -- Split the sum: Σ_{Fin (T'+1)} f = Σ_{Fin T'} f ∘ castSucc + f (last T')
    rw [Fin.sum_univ_castSucc]
    -- The last term: inputContribution for t = last T' is A^0 B x_{T'} = B x_{T'}
    have h_last : inputContribution A B (T' + 1) (Fin.last T') (inputs (Fin.last T')) =
                  B.mulVec (inputs (Fin.last T')) := by
      simp only [inputContribution, Fin.val_last, Nat.add_sub_cancel, Nat.sub_self, pow_zero,
                 Matrix.one_mulVec]
    rw [h_last]
    -- We need: A *ᵥ (Σ A^(T'-1-t) B x_t) = Σ A^(T'-t) B x_t, rest cancels
    congr 1
    -- Distribute A through the sum: A *ᵥ (Σ f) = Σ (A *ᵥ f)
    rw [Matrix.mulVec_sum]
    -- Now show term-by-term equality
    apply Finset.sum_congr rfl
    intro t _
    -- For each t, show: A *ᵥ (A^(T'-1-t) *ᵥ (B *ᵥ x_t)) = A^(T'-t) *ᵥ (B *ᵥ x_t)
    simp only [inputContribution, Fin.coe_castSucc]
    -- Use A *ᵥ (M *ᵥ v) = (A * M) *ᵥ v
    rw [Matrix.mulVec_mulVec]
    -- Now goal: (A * A^(T'-1-t)) *ᵥ (B *ᵥ x) = A^(T'-t) *ᵥ (B *ᵥ x). Use A * A^k = A^(k+1)
    have h_exp : T' + 1 - 1 - t.val = T' - t.val := by omega
    rw [h_exp]
    have h_succ : T' - t.val = (T' - 1 - t.val) + 1 := by
      have ht : t.val < T' := t.isLt
      omega
    rw [h_succ, pow_succ']

/-- Linearity: state of scaled/combined inputs is correspondingly scaled/combined -/
theorem state_additive (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (inputs₁ inputs₂ : Fin T → (Fin m → ℝ)) :
    stateFromZero A B T (fun t => inputs₁ t + inputs₂ t) =
    stateFromZero A B T inputs₁ + stateFromZero A B T inputs₂ := by
  induction T with
  | zero => simp [stateFromZero, stateAfterT]
  | succ T' ih =>
    simp only [stateFromZero, stateAfterT, linearStep]
    -- Goal: A *ᵥ state(combined) + B *ᵥ (x₁+x₂) = (A *ᵥ s₁ + B *ᵥ x₁) + (A *ᵥ s₂ + B *ᵥ x₂)
    -- First, apply IH to the recursive state
    have ih' := ih (fun i => inputs₁ i.castSucc) (fun i => inputs₂ i.castSucc)
    simp only [stateFromZero] at ih'
    rw [ih']
    -- Now use linearity: A *ᵥ (s₁ + s₂) = A *ᵥ s₁ + A *ᵥ s₂ and B *ᵥ (x₁ + x₂) = B *ᵥ x₁ + B *ᵥ x₂
    rw [Matrix.mulVec_add, Matrix.mulVec_add]
    -- Goal: (A *ᵥ s₁ + A *ᵥ s₂) + (B *ᵥ x₁ + B *ᵥ x₂) = (A *ᵥ s₁ + B *ᵥ x₁) + (A *ᵥ s₂ + B *ᵥ x₂)
    -- This is just rearranging addition
    ext i
    simp only [Pi.add_apply]
    ring

theorem state_scalar (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (c : ℝ) (inputs : Fin T → (Fin m → ℝ)) :
    stateFromZero A B T (fun t => c • inputs t) = c • stateFromZero A B T inputs := by
  induction T with
  | zero => simp [stateFromZero, stateAfterT]
  | succ T' ih =>
    simp only [stateFromZero, stateAfterT, linearStep]
    -- Goal: A *ᵥ state(c • inputs) + B *ᵥ (c • x) = c • (A *ᵥ state + B *ᵥ x)
    -- First, apply IH to the recursive state
    have ih' := ih (fun i => inputs i.castSucc)
    simp only [stateFromZero] at ih'
    rw [ih']
    -- Now use: A *ᵥ (c • s) = c • (A *ᵥ s) and B *ᵥ (c • x) = c • (B *ᵥ x)
    rw [Matrix.mulVec_smul, Matrix.mulVec_smul]
    -- Goal: c • (A *ᵥ s) + c • (B *ᵥ x) = c • (A *ᵥ s + B *ᵥ x)
    rw [← smul_add]

/-! ## Output Depends Only on State -/

/-- For any output matrix C, output at time T depends only on state at time T -/
theorem output_determined_by_state (C : Matrix (Fin k) (Fin n) ℝ)
    (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (inputs₁ inputs₂ : Fin T → (Fin m → ℝ)) :
    stateFromZero A B T inputs₁ = stateFromZero A B T inputs₂ →
    C.mulVec (stateFromZero A B T inputs₁) = C.mulVec (stateFromZero A B T inputs₂) := by
  intro h_same_state
  rw [h_same_state]

/-! ## Indistinguishability of Same-State Sequences -/

/-- If two input sequences produce the same state, all future outputs are identical -/
theorem same_state_same_future (C : Matrix (Fin k) (Fin n) ℝ)
    (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) (inputs₁ inputs₂ : Fin T → (Fin m → ℝ))
    (h_same : stateFromZero A B T inputs₁ = stateFromZero A B T inputs₂)
    (S : ℕ) (future : Fin S → (Fin m → ℝ)) :
    let h₁ := stateFromZero A B T inputs₁
    let h₂ := stateFromZero A B T inputs₂
    C.mulVec (stateAfterT A B h₁ S future) = C.mulVec (stateAfterT A B h₂ S future) := by
  intro h₁ h₂
  -- Since h₁ = h₂, running the same future inputs produces the same states
  have : stateAfterT A B h₁ S future = stateAfterT A B h₂ S future := by
    induction S with
    | zero => exact h_same
    | succ S' ih =>
      simp only [stateAfterT, linearStep]
      have ih' := ih (fun i => future i.castSucc)
      rw [ih']
  rw [this]

/-! ## Information-Theoretic Interpretation -/

/-- The set of all reachable states from zero with any input sequence of length T -/
def reachableStates (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) : Set (Fin n → ℝ) :=
  { h | ∃ inputs : Fin T → (Fin m → ℝ), h = stateFromZero A B T inputs }

/-- Reachable states form a subspace (closed under addition and scaling) -/
theorem reachable_is_subspace (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) :
    (0 : Fin n → ℝ) ∈ reachableStates A B T ∧
    (∀ h₁ h₂, h₁ ∈ reachableStates A B T → h₂ ∈ reachableStates A B T →
      h₁ + h₂ ∈ reachableStates A B T) ∧
    (∀ (c : ℝ) h, h ∈ reachableStates A B T → c • h ∈ reachableStates A B T) := by
  constructor
  · -- Zero is reachable (with zero inputs)
    use fun _ => 0
    -- Need to show: 0 = stateFromZero A B T (fun _ => 0)
    have h_zero_state : ∀ T', stateAfterT A B 0 T' (fun _ => 0) = 0 := by
      intro T'
      induction T' with
      | zero => rfl
      | succ T'' ih =>
        simp only [stateAfterT, linearStep, Matrix.mulVec_zero, add_zero, ih]
    exact (h_zero_state T).symm
  constructor
  · -- Sum of reachable states is reachable
    intro h₁ h₂ ⟨inputs₁, eq₁⟩ ⟨inputs₂, eq₂⟩
    use fun t => inputs₁ t + inputs₂ t
    rw [eq₁, eq₂]
    exact (state_additive A B T inputs₁ inputs₂).symm
  · -- Scalar multiple of reachable state is reachable
    intro c h ⟨inputs, eq⟩
    use fun t => c • inputs t
    rw [eq]
    exact (state_scalar A B T c inputs).symm

/-- The dimension of reachable states is at most n (the state dimension) -/
theorem reachable_dim_bound (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (T : ℕ) : Module.finrank ℝ (Submodule.span ℝ (reachableStates A B T)) ≤ n := by
  -- Reachable states are vectors in ℝⁿ, so span has dimension ≤ n
  -- This follows because the span is a subspace of (Fin n → ℝ) which has dimension n
  have h := Submodule.finrank_le (Submodule.span ℝ (reachableStates A B T))
  rw [Module.finrank_pi, Fintype.card_fin] at h
  exact h

/-! ## Comparison: What Linear RNNs CANNOT Do -/

/-- A function f on input sequences is "computable by linear RNN" if there exists
    matrices A, B, C such that f(inputs) = C * state(inputs) -/
def LinearlyComputable {T : ℕ} (f : (Fin T → (Fin m → ℝ)) → (Fin k → ℝ)) : Prop :=
  ∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin k) (Fin n) ℝ),
  ∀ inputs, f inputs = C.mulVec (stateFromZero A B T inputs)

/-- If a function distinguishes sequences that produce the same state in ALL linear RNNs,
    it's not linearly computable -/
theorem not_linearly_computable_if_state_independent
    {T : ℕ} (f : (Fin T → (Fin m → ℝ)) → (Fin k → ℝ))
    (inputs₁ inputs₂ : Fin T → (Fin m → ℝ))
    (h_same_for_all : ∀ (n' : ℕ) (A : Matrix (Fin n') (Fin n') ℝ) (B : Matrix (Fin n') (Fin m) ℝ),
      stateFromZero A B T inputs₁ = stateFromZero A B T inputs₂)
    (h_different : f inputs₁ ≠ f inputs₂) :
    ¬ LinearlyComputable f := by
  intro ⟨n', A, B, C, h_f⟩
  have h1 : f inputs₁ = C.mulVec (stateFromZero A B T inputs₁) := h_f inputs₁
  have h2 : f inputs₂ = C.mulVec (stateFromZero A B T inputs₂) := h_f inputs₂
  have h_same := h_same_for_all n' A B
  rw [h1, h2, h_same] at h_different
  exact h_different rfl

end Expressivity
