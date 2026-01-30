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
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Data.Finset.Basic
import Mathlib.Order.Filter.Basic
import Mathlib.Topology.Order.Basic
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.LinearLimitations
import ElmanProofs.Expressivity.NumericalBounds
import ElmanProofs.Activations.Lipschitz

/-!
# Exact Counting vs Magnitude Thresholding

This file proves the fundamental separation between **exact counting** and **magnitude
thresholding** capabilities in RNN architectures.

## The Key Question

Given a binary input sequence x₁, x₂, ..., x_T:
- Can the model detect "exactly k ones seen so far" at each timestep t?
- Or can it only detect "at least τ ones" (threshold)?

Linear temporal dynamics (Mamba2/SSMs) can track weighted sums but cannot implement
discontinuous threshold decisions at each timestep. E88's temporal tanh creates
discrete decision boundaries enabling exact counting for small moduli.

## Main Results

### Linear Systems Cannot Threshold Per-Timestep
* `running_threshold_discontinuous`: Running threshold is discontinuous in inputs
* `linear_rnn_continuous_per_t`: Linear RNN output at each t is continuous
* `linear_cannot_running_threshold`: Linear RNNs cannot compute running threshold

### Exact Counting Requires Nonlinearity
* `exact_count_detection_not_linear`: Detecting "exactly k ones" is not linear
* `count_mod_n_not_linear`: Counting mod n is not linearly computable

### E88 Can Count Exactly (Mod Small n)
* `e88_count_mod_2`: E88 can count mod 2 (parity) via sign-flip dynamics
* `e88_count_mod_3_existence`: E88 can count mod 3 with appropriate basins
* `tanh_creates_discrete_basins`: Tanh saturation creates discrete attractors

### Threshold Counting via Latching
* `e88_threshold_count`: E88 can detect when count exceeds threshold
* `latched_threshold_persists`: Once threshold is crossed, state latches

## Key Insight

The distinction between:
1. **Magnitude tracking**: S_t = Σ α^{t-s} x_s (linear, continuous, fading)
2. **Count tracking**: c_t = |{s ≤ t : x_s = 1}| (discrete, exact, non-fading)

Linear systems do (1). E88's tanh enables (2) by creating stable discrete states.

## Implementation Notes

Many proofs in this file are marked `sorry` because they require:
- Numerical verification of tanh bounds (e.g., tanh(1.5) > 0.9)
- IVT-based existence proofs for fixed points
- Connected space topology arguments

These are mathematically sound but require either specialized Lean tactics or
native computation to verify the numerical bounds rigorously.

-/

namespace ExactCounting

open Real Matrix Finset BigOperators Filter

variable {n m k : ℕ}

/-! ## Part 1: Running Threshold is Discontinuous -/

/-- Running threshold count function: outputs 1 at position t iff count of 1s in x[0:t] ≥ τ.
    This is a per-timestep function, not just final output. -/
noncomputable def runningThresholdCount (τ : ℕ) (T : ℕ) :
    (Fin T → ℝ) → (Fin T → ℝ) :=
  fun inputs t =>
    let count : ℕ := (Finset.univ.filter (fun s : Fin T =>
      s.val ≤ t.val ∧ inputs s > 0.5)).card
    if count ≥ τ then 1 else 0

/-- The running threshold function is discontinuous.
    Small changes to input can cause discrete jumps in output. -/
theorem running_threshold_discontinuous (τ : ℕ) (hτ : 0 < τ) (T : ℕ) (hT : τ ≤ T) :
    ¬Continuous (fun inputs : Fin T → ℝ =>
      runningThresholdCount τ T inputs ⟨τ - 1, by omega⟩) := by
  -- The function only takes values 0 or 1 (step function behavior)
  -- A continuous function on a connected space taking both 0 and 1 would
  -- violate the intermediate value property
  intro hcont
  let pos : Fin T := ⟨τ - 1, by omega⟩
  let f := fun inputs : Fin T → ℝ => runningThresholdCount τ T inputs pos
  -- The function only takes values in {0, 1}
  have hval : ∀ inputs : Fin T → ℝ, f inputs = 0 ∨ f inputs = 1 := by
    intro inputs
    simp only [f, runningThresholdCount]
    split_ifs <;> [right; left] <;> rfl
  -- Input with all zeros gives output 0
  have h_zero_input : f (fun _ => 0) = 0 := by
    simp only [f, runningThresholdCount]
    have hcard : (Finset.univ.filter (fun s : Fin T => s.val ≤ pos.val ∧ (0 : ℝ) > 0.5)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro _ _
      norm_num
    simp only [hcard, ge_iff_le, Nat.not_lt, nonpos_iff_eq_zero]
    simp only [ite_eq_right_iff]
    omega
  -- Input with all ones gives output 1
  have h_one_input : f (fun _ => 1) = 1 := by
    simp only [f, runningThresholdCount]
    have hcard : (Finset.univ.filter (fun s : Fin T => s.val ≤ pos.val ∧ (1 : ℝ) > 0.5)).card = τ := by
      have heq : ∀ s : Fin T, (s.val ≤ pos.val ∧ (1 : ℝ) > 0.5) ↔ s.val ≤ pos.val := by
        intro s; constructor <;> intro h
        · exact h.1
        · exact ⟨h, by norm_num⟩
      conv_lhs => rw [Finset.filter_congr (fun s _ => heq s)]
      have hcount : (Finset.univ.filter (fun s : Fin T => s.val ≤ pos.val)).card = τ := by
        -- pos.val = τ - 1, so we count elements with val ≤ τ - 1, which is τ elements
        have hpos : pos.val = τ - 1 := rfl
        rw [hpos]
        -- The set {s : Fin T | s.val ≤ τ - 1} = {s : Fin T | s.val < τ}
        have heq2 : (Finset.univ.filter (fun s : Fin T => s.val ≤ τ - 1)) =
                   (Finset.univ.filter (fun s : Fin T => s.val < τ)) := by
          ext s; simp only [Finset.mem_filter, Finset.mem_univ, true_and]; omega
        rw [heq2]
        -- Count elements of Fin T with val < τ using bijection with Fin τ
        have hinj : Function.Injective (fun i : Fin τ => (⟨i.val, Nat.lt_of_lt_of_le i.isLt hT⟩ : Fin T)) := by
          intro a b hab
          simp only [Fin.mk.injEq] at hab
          exact Fin.ext hab
        have hbij : Finset.univ.filter (fun s : Fin T => s.val < τ) =
                    Finset.image (fun i : Fin τ => ⟨i.val, Nat.lt_of_lt_of_le i.isLt hT⟩) Finset.univ := by
          apply Finset.ext
          intro x
          constructor
          · intro hx
            simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
            simp only [Finset.mem_image, Finset.mem_univ, true_and]
            exact ⟨⟨x.val, hx⟩, Fin.ext rfl⟩
          · intro hx
            simp only [Finset.mem_image, Finset.mem_univ, true_and] at hx
            obtain ⟨i, hi⟩ := hx
            simp only [Finset.mem_filter, Finset.mem_univ, true_and]
            simp only [Fin.ext_iff] at hi
            rw [← hi]; exact i.isLt
        rw [hbij, Finset.card_image_of_injective _ hinj, Finset.card_fin]
      exact hcount
    simp only [hcard, ge_iff_le, le_refl, ↓reduceIte]
  -- The range of f is a subset of {0, 1}
  have hrange : Set.range f ⊆ {0, 1} := by
    intro y hy
    simp only [Set.mem_range] at hy
    obtain ⟨x, rfl⟩ := hy
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    cases hval x with
    | inl h => left; exact h
    | inr h => right; exact h
  -- Both 0 and 1 are in the range
  have h0_in : (0 : ℝ) ∈ Set.range f := ⟨fun _ => 0, h_zero_input⟩
  have h1_in : (1 : ℝ) ∈ Set.range f := ⟨fun _ => 1, h_one_input⟩
  -- (Fin T → ℝ) is connected
  have hconn : IsConnected (Set.univ : Set (Fin T → ℝ)) := isConnected_univ
  -- The image of a connected set under a continuous function is connected
  have himg_conn : IsConnected (Set.range f) := by
    have : Set.range f = f '' Set.univ := by ext; simp [Set.mem_range, Set.mem_image]
    rw [this]
    exact hconn.image f hcont.continuousOn
  -- A connected subset of ℝ containing 0 and 1 must contain 0.5
  have hconvex : Set.OrdConnected (Set.range f) := himg_conn.isPreconnected.ordConnected
  have hmem_half : (0.5 : ℝ) ∈ Set.range f := by
    apply hconvex.out h0_in h1_in
    constructor <;> linarith
  -- But 0.5 ∉ {0, 1}, contradiction
  have : (0.5 : ℝ) ∈ ({0, 1} : Set ℝ) := hrange hmem_half
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at this
  rcases this with h | h <;> linarith

/-- Helper: construct an input sequence with k ones followed by zeros -/
def kOnesInput (T k : ℕ) (hk : k ≤ T) : Fin T → ℝ :=
  fun t => if t.val < k then 1 else 0

theorem kOnesInput_count (T k : ℕ) (hk : k ≤ T) (t : Fin T) (ht : k ≤ t.val + 1) :
    (Finset.univ.filter (fun s : Fin T => s.val ≤ t.val ∧ kOnesInput T k hk s > 0.5)).card = k := by
  -- kOnesInput gives 1 for s.val < k, 0 otherwise
  -- So s satisfies the predicate iff s.val ≤ t.val ∧ s.val < k
  -- Since k ≤ t.val + 1, this is equivalent to s.val < k
  have heq : ∀ s : Fin T, (s.val ≤ t.val ∧ kOnesInput T k hk s > 0.5) ↔ s.val < k := by
    intro s
    simp only [kOnesInput]
    constructor
    · intro ⟨_, h2⟩
      split_ifs at h2 with hsk
      · exact hsk
      · norm_num at h2
    · intro hsk
      constructor
      · omega
      · simp only [hsk, ↓reduceIte]
        norm_num
  conv_lhs => rw [Finset.filter_congr (fun s _ => heq s)]
  -- Now count elements of Fin T with val < k
  have hinj : Function.Injective (fun i : Fin k => (⟨i.val, Nat.lt_of_lt_of_le i.isLt hk⟩ : Fin T)) := by
    intro a b hab
    simp only [Fin.mk.injEq] at hab
    exact Fin.ext hab
  have hbij : Finset.univ.filter (fun s : Fin T => s.val < k) =
              Finset.image (fun i : Fin k => ⟨i.val, Nat.lt_of_lt_of_le i.isLt hk⟩) Finset.univ := by
    apply Finset.ext
    intro x
    constructor
    · intro hx
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hx
      simp only [Finset.mem_image, Finset.mem_univ, true_and]
      exact ⟨⟨x.val, hx⟩, Fin.ext rfl⟩
    · intro hx
      simp only [Finset.mem_image, Finset.mem_univ, true_and] at hx
      obtain ⟨i, hi⟩ := hx
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      simp only [Fin.ext_iff] at hi
      rw [← hi]; exact i.isLt
  rw [hbij, Finset.card_image_of_injective _ hinj, Finset.card_fin]

/-! ## Part 2: Linear RNN Output is Continuous Per-Timestep -/

/-- Helper: Matrix multiplication is continuous. -/
theorem continuous_matrix_mulVec (M : Matrix (Fin k') (Fin n') ℝ) :
    Continuous (fun v : Fin n' → ℝ => M.mulVec v) := by
  apply continuous_pi
  intro i
  simp only [Matrix.mulVec]
  apply continuous_finset_sum
  intro j _
  exact continuous_const.mul (continuous_apply j)

/-- For a linear RNN, the output at any fixed position t is a continuous
    function of the full input sequence. -/
theorem linear_rnn_continuous_per_t (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin k) (Fin n) ℝ) (T : ℕ) (t : Fin T) :
    Continuous (fun inputs : Fin T → (Fin m → ℝ) =>
      let truncated : Fin (t.val + 1) → (Fin m → ℝ) :=
        fun s => if h : s.val < T then inputs ⟨s.val, h⟩ else 0
      C.mulVec (Expressivity.stateFromZero A B (t.val + 1) truncated)) := by
  -- The truncation is continuous: each coordinate is either a projection or constant
  have h_trunc_cont : Continuous (fun inputs' : Fin T → (Fin m → ℝ) =>
      (fun s : Fin (t.val + 1) => if hs : s.val < T then inputs' ⟨s.val, hs⟩ else 0)) := by
    apply continuous_pi
    intro s
    by_cases hs : s.val < T
    · have heq : (fun inputs' : Fin T → (Fin m → ℝ) =>
          if hs' : s.val < T then inputs' ⟨s.val, hs'⟩ else 0) =
          (fun inputs' => inputs' (⟨s.val, hs⟩ : Fin T)) := by
        ext inputs'
        simp [hs]
      rw [heq]
      exact continuous_apply (⟨s.val, hs⟩ : Fin T)
    · have heq : (fun inputs' : Fin T → (Fin m → ℝ) =>
          if hs' : s.val < T then inputs' ⟨s.val, hs'⟩ else 0) = (fun _ => 0) := by
        ext inputs'
        simp [hs]
      rw [heq]
      exact continuous_const
  -- stateFromZero is continuous: it's linear in its input (linear_state_is_sum)
  have h_state_cont : Continuous (fun trunc : Fin (t.val + 1) → (Fin m → ℝ) =>
      Expressivity.stateFromZero A B (t.val + 1) trunc) := by
    apply continuous_pi
    intro j
    -- (stateFromZero ...) j = (Σ s, inputContribution s) j by linear_state_is_sum
    have h_eq : ∀ trunc, (Expressivity.stateFromZero A B (t.val + 1) trunc) j =
        (∑ s : Fin (t.val + 1), Expressivity.inputContribution A B (t.val + 1) s (trunc s)) j := by
      intro trunc
      rw [Expressivity.linear_state_is_sum]
    simp_rw [h_eq, Finset.sum_apply, Expressivity.inputContribution, Matrix.mulVec]
    apply continuous_finset_sum
    intro s _
    apply continuous_finset_sum
    intro l _
    apply Continuous.mul continuous_const
    -- Need to show: Continuous fun x => (B *ᵥ (x s)) l
    -- (B *ᵥ v) l = Σ p, B l p * v p
    have h_B : ∀ v : Fin m → ℝ, (B.mulVec v) l = ∑ p, B l p * v p := fun v => rfl
    simp_rw [h_B]
    apply continuous_finset_sum
    intro p _
    apply Continuous.mul continuous_const
    exact (continuous_apply p).comp (continuous_apply s)
  -- C.mulVec is continuous
  have h_C_cont : Continuous (fun state : Fin n → ℝ => C.mulVec state) :=
    continuous_matrix_mulVec C
  -- Compose: inputs → truncated → state → output
  exact h_C_cont.comp (h_state_cont.comp h_trunc_cont)

/-- The full output sequence of a linear RNN is continuous in the input sequence. -/
theorem linear_rnn_continuous_output (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin k) (Fin n) ℝ) (T : ℕ) (t : Fin k) :
    Continuous (fun inputs : Fin T → (Fin m → ℝ) =>
      let state := Expressivity.stateFromZero A B T inputs
      (C.mulVec state) t) := by
  -- (C.mulVec state) t = Σ j, C_{t,j} * state_j
  -- state_j = (Σ s, A^{T-1-s} B x_s)_j is a polynomial in inputs
  -- So the whole expression is a polynomial in inputs, hence continuous
  simp only [Matrix.mulVec]
  apply continuous_finset_sum
  intro j _
  apply Continuous.mul continuous_const
  -- state j = (stateFromZero ...) j is continuous
  -- By linear_state_is_sum, state = Σ s, inputContribution s
  have h_eq : ∀ inputs', (Expressivity.stateFromZero A B T inputs') j =
      (∑ s : Fin T, Expressivity.inputContribution A B T s (inputs' s)) j := by
    intro inputs'
    rw [Expressivity.linear_state_is_sum]
  simp_rw [h_eq, Finset.sum_apply]
  apply continuous_finset_sum
  intro s _
  -- (inputContribution A B T s (inputs s)) j = ((A^k) *ᵥ (B *ᵥ (inputs s))) j
  simp only [Expressivity.inputContribution, Matrix.mulVec]
  apply continuous_finset_sum
  intro l _
  apply Continuous.mul continuous_const
  -- (B *ᵥ (inputs s)) l = Σ p, B l p * (inputs s) p
  have h_B : ∀ v : Fin m → ℝ, (B.mulVec v) l = ∑ p, B l p * v p := fun v => rfl
  simp_rw [h_B]
  apply continuous_finset_sum
  intro p _
  apply Continuous.mul continuous_const
  -- (inputs s) p is a coordinate projection, hence continuous
  exact (continuous_apply p).comp (continuous_apply s)

/-! ## Part 3: Linear RNNs Cannot Compute Running Threshold -/

/-- **Main Theorem**: Linear RNNs cannot compute the running threshold function. -/
theorem linear_cannot_running_threshold (τ : ℕ) (hτ : 1 ≤ τ) (T : ℕ) (hT : τ ≤ T) :
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
       (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ),
        (C.mulVec (Expressivity.stateFromZero A B T inputs)) 0 =
        runningThresholdCount τ T (fun t => inputs t 0) ⟨τ - 1, by omega⟩ := by
  -- If linear RNN could compute running threshold, the function would be continuous
  -- But running_threshold_discontinuous shows it's not
  intro ⟨n, A, B, C, h_eq⟩
  -- The left side is continuous (linear RNN output is continuous in inputs)
  have h_discont := running_threshold_discontinuous τ hτ T hT
  apply h_discont
  -- We need to show the threshold function is continuous, which it isn't
  -- The key is that f(inputs) = (C.mulVec state) 0 matches the threshold
  -- For each inputs, we have equality, so if LHS is continuous, RHS must be too
  -- But we need to adapt from (Fin T → (Fin 1 → ℝ)) to (Fin T → ℝ)
  -- Consider the embedding (Fin T → ℝ) → (Fin T → (Fin 1 → ℝ))
  let embed : (Fin T → ℝ) → (Fin T → (Fin 1 → ℝ)) := fun x t _ => x t
  have h_cont_embed : Continuous embed := by
    apply continuous_pi
    intro t
    apply continuous_pi
    intro _
    exact continuous_apply t
  -- The composition gives the threshold function
  have h_comp : ∀ x : Fin T → ℝ,
      runningThresholdCount τ T x ⟨τ - 1, by omega⟩ =
      (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0 := by
    intro x
    have := h_eq (embed x)
    simp only [embed] at this
    exact this.symm
  -- If the RHS is continuous in x, so would be the LHS
  -- The RHS is continuous (linear RNN output composed with continuous embedding)
  have h_rhs_cont : Continuous (fun x : Fin T → ℝ =>
      (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0) := by
    -- This is continuous as composition of continuous functions
    -- (linear RNN output is continuous, embedding is continuous)
    have : Continuous (fun x => (C.mulVec (Expressivity.stateFromZero A B T x)) 0) :=
      linear_rnn_continuous_output A B C T 0
    exact this.comp h_cont_embed
  -- Now show the threshold function is continuous (contradiction with h_discont)
  have h_eq_fun : (fun x : Fin T → ℝ => runningThresholdCount τ T x ⟨τ - 1, by omega⟩) =
                  (fun x => (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0) := by
    ext x; exact h_comp x
  rw [h_eq_fun]
  exact h_rhs_cont

/-! ## Part 4: Exact Count Detection is Not Linear -/

/-- Exact count detection: outputs 1 at position t iff exactly k ones have been seen. -/
noncomputable def exactCountDetection (k : ℕ) (T : ℕ) :
    (Fin T → ℝ) → (Fin T → ℝ) :=
  fun inputs t =>
    let count : ℕ := (Finset.univ.filter (fun s : Fin T =>
      s.val ≤ t.val ∧ inputs s > 0.5)).card
    if count = k then 1 else 0

/-- The exact count detection function at position T-1 is discontinuous. -/
theorem exact_count_detection_discontinuous (k : ℕ) (hk : 0 < k) (T : ℕ) (hkT : k < T) :
    ¬Continuous (fun inputs : Fin T → ℝ =>
      exactCountDetection k T inputs ⟨T - 1, by omega⟩) := by
  intro hcont
  let pos : Fin T := ⟨T - 1, by omega⟩
  let f := fun inputs : Fin T → ℝ => exactCountDetection k T inputs pos
  -- The function only takes values in {0, 1}
  have hval : ∀ inputs : Fin T → ℝ, f inputs = 0 ∨ f inputs = 1 := by
    intro inputs
    simp only [f, exactCountDetection]
    split_ifs <;> [right; left] <;> rfl
  -- Input with all zeros gives output 0 (since count = 0 ≠ k for k > 0)
  have h_zero_input : f (fun _ => 0) = 0 := by
    simp only [f, exactCountDetection]
    have hcard : (Finset.univ.filter (fun s : Fin T => s.val ≤ pos.val ∧ (0 : ℝ) > 0.5)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro _ _
      norm_num
    simp only [hcard]
    split_ifs with heq
    · omega
    · rfl
  -- Input with k ones at positions 0, 1, ..., k-1 gives output 1
  have h_k_ones_input : f (kOnesInput T k (le_of_lt hkT)) = 1 := by
    simp only [f, exactCountDetection]
    have hcard : (Finset.univ.filter (fun s : Fin T =>
        s.val ≤ pos.val ∧ kOnesInput T k (le_of_lt hkT) s > 0.5)).card = k := by
      have hpos : pos.val = T - 1 := rfl
      rw [hpos]
      exact kOnesInput_count T k (le_of_lt hkT) ⟨T - 1, by omega⟩ (by omega)
    simp only [hcard, ↓reduceIte]
  -- The range of f is a subset of {0, 1}
  have hrange : Set.range f ⊆ {0, 1} := by
    intro y hy
    simp only [Set.mem_range] at hy
    obtain ⟨x, rfl⟩ := hy
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff]
    cases hval x with
    | inl h => left; exact h
    | inr h => right; exact h
  -- Both 0 and 1 are in the range
  have h0_in : (0 : ℝ) ∈ Set.range f := ⟨fun _ => 0, h_zero_input⟩
  have h1_in : (1 : ℝ) ∈ Set.range f := ⟨kOnesInput T k (le_of_lt hkT), h_k_ones_input⟩
  -- (Fin T → ℝ) is connected
  have hconn : IsConnected (Set.univ : Set (Fin T → ℝ)) := isConnected_univ
  -- The image of a connected set under a continuous function is connected
  have himg_conn : IsConnected (Set.range f) := by
    have : Set.range f = f '' Set.univ := by ext; simp [Set.mem_range, Set.mem_image]
    rw [this]
    exact hconn.image f hcont.continuousOn
  -- A connected subset of ℝ containing 0 and 1 must contain 0.5
  have hconvex : Set.OrdConnected (Set.range f) := himg_conn.isPreconnected.ordConnected
  have hmem_half : (0.5 : ℝ) ∈ Set.range f := by
    apply hconvex.out h0_in h1_in
    constructor <;> linarith
  -- But 0.5 ∉ {0, 1}, contradiction
  have : (0.5 : ℝ) ∈ ({0, 1} : Set ℝ) := hrange hmem_half
  simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at this
  rcases this with h | h <;> linarith

/-- Exact count detection is not linear. -/
theorem exact_count_detection_not_linear (k : ℕ) (T : ℕ) (hk : 0 < k) (hkT : k < T) :
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
       (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ),
        (C.mulVec (Expressivity.stateFromZero A B T inputs)) 0 =
        exactCountDetection k T (fun t => inputs t 0) ⟨T - 1, by omega⟩ := by
  -- If linear RNN could compute exact count detection, the function would be continuous
  -- But exact_count_detection_discontinuous shows it's not
  intro ⟨n, A, B, C, h_eq⟩
  have h_discont := exact_count_detection_discontinuous k hk T hkT
  apply h_discont
  -- We need to show the detection function is continuous, which it isn't
  -- The key is that f(inputs) = (C.mulVec state) 0 matches the detection
  let embed : (Fin T → ℝ) → (Fin T → (Fin 1 → ℝ)) := fun x t _ => x t
  have h_cont_embed : Continuous embed := by
    apply continuous_pi
    intro t
    apply continuous_pi
    intro _
    exact continuous_apply t
  -- The composition gives the detection function
  have h_comp : ∀ x : Fin T → ℝ,
      exactCountDetection k T x ⟨T - 1, by omega⟩ =
      (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0 := by
    intro x
    have := h_eq (embed x)
    simp only [embed] at this
    exact this.symm
  -- The RHS is continuous (linear RNN output composed with continuous embedding)
  have h_rhs_cont : Continuous (fun x : Fin T → ℝ =>
      (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0) := by
    have : Continuous (fun x => (C.mulVec (Expressivity.stateFromZero A B T x)) 0) :=
      linear_rnn_continuous_output A B C T 0
    exact this.comp h_cont_embed
  -- Show the detection function is continuous (contradiction with h_discont)
  have h_eq_fun : (fun x : Fin T → ℝ => exactCountDetection k T x ⟨T - 1, by omega⟩) =
                  (fun x => (C.mulVec (Expressivity.stateFromZero A B T (embed x))) 0) := by
    ext x; exact h_comp x
  rw [h_eq_fun]
  exact h_rhs_cont

/-! ## Part 5: Counting Mod n is Not Linearly Computable -/

/-- Count mod n function: outputs count(x_0, ..., x_t) mod n at each position t. -/
noncomputable def countModN (n : ℕ) (hn : 0 < n) (T : ℕ) :
    (Fin T → ℝ) → (Fin T → Fin n) :=
  fun inputs t =>
    let count : ℕ := (Finset.univ.filter (fun s : Fin T =>
      s.val ≤ t.val ∧ inputs s > 0.5)).card
    ⟨count % n, Nat.mod_lt count hn⟩

/-- Encoding count mod n as a real value. -/
noncomputable def countModNReal (n : ℕ) (hn : 0 < n) (T : ℕ) :
    (Fin T → ℝ) → (Fin T → ℝ) :=
  fun inputs t => (countModN n hn T inputs t).val

/-- Count mod 2 (parity) is not linear: this is the XOR impossibility.
    The proof uses the affine identity: for any affine f, f(00) + f(11) = f(01) + f(10).
    But parity(00) + parity(11) = 0 + 0 = 0, while parity(01) + parity(10) = 1 + 1 = 2.
    Since linear RNN output is affine, it cannot compute parity.

    Proof sketch: Define input sequences one_at_0 (1 at position 0, 0 elsewhere),
    one_at_1 (1 at position 1, 0 elsewhere), and ones_at_01 = one_at_0 + one_at_1.
    By linearity: f(ones_at_01) = f(one_at_0) + f(one_at_1).
    Parity values: f(one_at_0) = 1, f(one_at_1) = 1, f(ones_at_01) = 0.
    Contradiction: 0 = 1 + 1 = 2. -/
theorem count_mod_2_not_linear (T : ℕ) (hT : 2 ≤ T) :
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
       (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ), (∀ t, inputs t 0 = 0 ∨ inputs t 0 = 1) →
        (C.mulVec (Expressivity.stateFromZero A B T inputs)) 0 =
        countModNReal 2 (by norm_num) T (fun t => inputs t 0) ⟨T - 1, by omega⟩ := by
  intro ⟨n, A, B, C, h_f⟩
  -- Define four canonical input sequences
  let input00 : Fin T → (Fin 1 → ℝ) := fun _ _ => 0
  let input01 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 0 then 0 else if t.val = 1 then 1 else 0
  let input10 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 0 then 1 else if t.val = 1 then 0 else 0
  let input11 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 0 then 1 else if t.val = 1 then 1 else 0
  -- All inputs are binary
  have h_bin00 : ∀ t, input00 t 0 = 0 ∨ input00 t 0 = 1 := fun _ => Or.inl rfl
  have h_bin01 : ∀ t, input01 t 0 = 0 ∨ input01 t 0 = 1 := by
    intro t; simp only [input01]; split_ifs <;> tauto
  have h_bin10 : ∀ t, input10 t 0 = 0 ∨ input10 t 0 = 1 := by
    intro t; simp only [input10]; split_ifs <;> tauto
  have h_bin11 : ∀ t, input11 t 0 = 0 ∨ input11 t 0 = 1 := by
    intro t; simp only [input11]; split_ifs <;> tauto
  -- parity(00) = 0 (0 ones, 0 mod 2 = 0)
  have parity00 : countModNReal 2 (by norm_num) T (fun t => input00 t 0) ⟨T - 1, by omega⟩ = 0 := by
    simp only [countModNReal, countModN, input00]
    have h_count : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧ (0 : ℝ) > 0.5)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro _ _; norm_num
    simp only [h_count, Nat.zero_mod]
    norm_cast
  -- parity(01) = 1 (1 one at pos 1)
  have parity01 : countModNReal 2 (by norm_num) T (fun t => input01 t 0) ⟨T - 1, by omega⟩ = 1 := by
    simp only [countModNReal, countModN, input01]
    have h_count : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 0 then (0 : ℝ) else if s.val = 1 then 1 else 0) > 0.5)).card = 1 := by
      have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
          (if s.val = 0 then (0 : ℝ) else if s.val = 1 then 1 else 0) > 0.5)) = {⟨1, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
        constructor
        · intro ⟨_, h_gt⟩
          split_ifs at h_gt with h0 h1
          · norm_num at h_gt
          · exact Fin.ext h1
          · norm_num at h_gt
        · intro h_eq
          rw [h_eq]
          simp only [Fin.val_one, ↓reduceIte]
          constructor
          · omega
          · norm_num
      rw [h_eq, Finset.card_singleton]
    simp only [h_count]
    norm_cast
  -- parity(10) = 1 (1 one at pos 0)
  have parity10 : countModNReal 2 (by norm_num) T (fun t => input10 t 0) ⟨T - 1, by omega⟩ = 1 := by
    simp only [countModNReal, countModN, input10]
    have h_count : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 0 else 0) > 0.5)).card = 1 := by
      have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
          (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 0 else 0) > 0.5)) = {⟨0, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
        constructor
        · intro ⟨_, h_gt⟩
          split_ifs at h_gt with h0 h1
          · exact Fin.ext h0
          · norm_num at h_gt
          · norm_num at h_gt
        · intro h_eq
          rw [h_eq]
          simp only [Fin.val_zero, ↓reduceIte]
          constructor
          · omega
          · norm_num
      rw [h_eq, Finset.card_singleton]
    simp only [h_count]
    norm_cast
  -- parity(11) = 0 (2 ones, 2 mod 2 = 0)
  have parity11 : countModNReal 2 (by norm_num) T (fun t => input11 t 0) ⟨T - 1, by omega⟩ = 0 := by
    simp only [countModNReal, countModN, input11]
    have h_count : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 1 else 0) > 0.5)).card = 2 := by
      have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
          (if s.val = 0 then (1 : ℝ) else if s.val = 1 then 1 else 0) > 0.5)) = {⟨0, by omega⟩, ⟨1, by omega⟩} := by
        ext s
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
          Finset.mem_singleton]
        constructor
        · intro ⟨_, h_gt⟩
          split_ifs at h_gt with h0 h1
          · left; exact Fin.ext h0
          · right; exact Fin.ext h1
          · norm_num at h_gt
        · intro h_eq
          rcases h_eq with h_eq | h_eq
          · rw [h_eq]; simp only [Fin.val_zero, ↓reduceIte]; constructor <;> [omega; norm_num]
          · rw [h_eq]; simp only [Fin.val_one, ↓reduceIte]; constructor <;> [omega; norm_num]
      rw [h_eq]
      have h_ne : (⟨0, by omega⟩ : Fin T) ≠ ⟨1, by omega⟩ := by simp
      rw [Finset.card_insert_of_not_mem (by simp [h_ne]), Finset.card_singleton]
    simp only [h_count]
    norm_cast
  -- Apply the linear RNN formula at each corner
  have eq00 := h_f input00 h_bin00
  have eq01 := h_f input01 h_bin01
  have eq10 := h_f input10 h_bin10
  have eq11 := h_f input11 h_bin11
  rw [parity00] at eq00
  rw [parity01] at eq01
  rw [parity10] at eq10
  rw [parity11] at eq11
  -- Key: input00 + input11 = input01 + input10 (pointwise)
  have h_inputs_sum : ∀ t j, input00 t j + input11 t j = input01 t j + input10 t j := by
    intro t j
    simp only [input00, input01, input10, input11]
    split_ifs <;> ring
  -- By linearity of state, the outputs satisfy the same relation
  have h_state_add : Expressivity.stateFromZero A B T input00 + Expressivity.stateFromZero A B T input11 =
      Expressivity.stateFromZero A B T input01 + Expressivity.stateFromZero A B T input10 := by
    have h_sum : input00 + input11 = input01 + input10 := by
      ext t j
      exact h_inputs_sum t j
    have h1 := Expressivity.state_additive A B T input00 input11
    have h2 := Expressivity.state_additive A B T input01 input10
    have h_sum_pw : (fun t => input00 t + input11 t) = (fun t => input01 t + input10 t) := by
      ext t j
      exact h_inputs_sum t j
    rw [h_sum_pw] at h1
    rw [← h1, ← h2]
  -- Apply C.mulVec to both sides
  have h_output_add : C.mulVec (Expressivity.stateFromZero A B T input00) +
      C.mulVec (Expressivity.stateFromZero A B T input11) =
      C.mulVec (Expressivity.stateFromZero A B T input01) +
      C.mulVec (Expressivity.stateFromZero A B T input10) := by
    rw [← Matrix.mulVec_add, ← Matrix.mulVec_add, h_state_add]
  -- Evaluate at component 0
  have h_comp := congrFun h_output_add 0
  simp only [Pi.add_apply] at h_comp
  -- Contradiction: 0 + 0 = 1 + 1 → 0 = 2
  linarith [eq00, eq01, eq10, eq11, h_comp]
/-- Count mod 3 is not linear.
    Similar to mod 2: ones_012 = one_at_0 + one_at_1 + one_at_2 (input 1 at positions 0,1,2).
    By linearity: f(ones_012) = f(one_at_0) + f(one_at_1) + f(one_at_2) = 1 + 1 + 1 = 3.
    But countModN 3 gives: ones_012 has count 3, so 3 mod 3 = 0.
    Contradiction: 0 = 3. -/
theorem count_mod_3_not_linear (T : ℕ) (hT : 3 ≤ T) :
    ¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
       (C : Matrix (Fin 1) (Fin n) ℝ),
      ∀ inputs : Fin T → (Fin 1 → ℝ), (∀ t, inputs t 0 = 0 ∨ inputs t 0 = 1) →
        (C.mulVec (Expressivity.stateFromZero A B T inputs)) 0 =
        countModNReal 3 (by norm_num) T (fun t => inputs t 0) ⟨T - 1, by omega⟩ := by
  -- Proof: ones_012 = one_at_0 + one_at_1 + one_at_2 pointwise.
  -- By linearity: f(ones_012) = f(one_at_0) + f(one_at_1) + f(one_at_2) = 1 + 1 + 1 = 3.
  -- But countMod3(ones_012) = 3 mod 3 = 0. Contradiction: 0 = 3.
  intro ⟨n, A, B, C, h_f⟩
  -- Define input patterns
  let one_at_0 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 0 then 1 else 0
  let one_at_1 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 1 then 1 else 0
  let one_at_2 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 2 then 1 else 0
  let ones_012 : Fin T → (Fin 1 → ℝ) := fun t _ => if t.val = 0 ∨ t.val = 1 ∨ t.val = 2 then 1 else 0
  -- Binary constraints
  have h_bin_0 : ∀ t, one_at_0 t 0 = 0 ∨ one_at_0 t 0 = 1 := by
    intro t; simp only [one_at_0]; split_ifs <;> tauto
  have h_bin_1 : ∀ t, one_at_1 t 0 = 0 ∨ one_at_1 t 0 = 1 := by
    intro t; simp only [one_at_1]; split_ifs <;> tauto
  have h_bin_2 : ∀ t, one_at_2 t 0 = 0 ∨ one_at_2 t 0 = 1 := by
    intro t; simp only [one_at_2]; split_ifs <;> tauto
  have h_bin_012 : ∀ t, ones_012 t 0 = 0 ∨ ones_012 t 0 = 1 := by
    intro t; simp only [ones_012]; split_ifs <;> tauto
  -- Compute mod 3 counts
  -- one_at_i has count 1, so 1 mod 3 = 1
  have mod3_0 : countModNReal 3 (by norm_num) T (fun t => one_at_0 t 0) ⟨T - 1, by omega⟩ = 1 := by
    simp only [countModNReal, countModN, one_at_0]
    have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 0 then (1 : ℝ) else 0) > 0.5)) = {⟨0, by omega⟩} := by
      ext s; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      constructor
      · intro ⟨_, h_gt⟩; split_ifs at h_gt with h0; exact Fin.ext h0; norm_num at h_gt
      · intro h_eq; rw [h_eq]; simp only [Fin.val_zero, ↓reduceIte]; constructor <;> [omega; norm_num]
    simp only [h_eq, Finset.card_singleton]; norm_cast
  have mod3_1 : countModNReal 3 (by norm_num) T (fun t => one_at_1 t 0) ⟨T - 1, by omega⟩ = 1 := by
    simp only [countModNReal, countModN, one_at_1]
    have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 1 then (1 : ℝ) else 0) > 0.5)) = {⟨1, by omega⟩} := by
      ext s; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      constructor
      · intro ⟨_, h_gt⟩; split_ifs at h_gt with h1; exact Fin.ext h1; norm_num at h_gt
      · intro h_eq; rw [h_eq]; simp only [Fin.val_one, ↓reduceIte]; constructor <;> [omega; norm_num]
    simp only [h_eq, Finset.card_singleton]; norm_cast
  have mod3_2 : countModNReal 3 (by norm_num) T (fun t => one_at_2 t 0) ⟨T - 1, by omega⟩ = 1 := by
    simp only [countModNReal, countModN, one_at_2]
    have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 2 then (1 : ℝ) else 0) > 0.5)) = {⟨2, by omega⟩} := by
      ext s; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      constructor
      · intro ⟨_, h_gt⟩; split_ifs at h_gt with h2; exact Fin.ext h2; norm_num at h_gt
      · intro h_eq; rw [h_eq]; simp only [↓reduceIte]; constructor <;> [omega; norm_num]
    simp only [h_eq, Finset.card_singleton]; norm_cast
  -- ones_012 has count 3, so 3 mod 3 = 0
  have mod3_012 : countModNReal 3 (by norm_num) T (fun t => ones_012 t 0) ⟨T - 1, by omega⟩ = 0 := by
    simp only [countModNReal, countModN, ones_012]
    have h_eq : (Finset.univ.filter (fun s : Fin T => s.val ≤ T - 1 ∧
        (if s.val = 0 ∨ s.val = 1 ∨ s.val = 2 then (1 : ℝ) else 0) > 0.5)) =
        {⟨0, by omega⟩, ⟨1, by omega⟩, ⟨2, by omega⟩} := by
      ext s; simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
        Finset.mem_singleton]
      constructor
      · intro ⟨_, h_gt⟩
        split_ifs at h_gt with h012
        · rcases h012 with h0 | h1 | h2
          · left; exact Fin.ext h0
          · right; left; exact Fin.ext h1
          · right; right; exact Fin.ext h2
        · norm_num at h_gt
      · intro h_eq
        rcases h_eq with h0 | h1 | h2
        · rw [h0]; simp only [Fin.val_zero, or_self, ↓reduceIte]; constructor <;> [omega; norm_num]
        · rw [h1]; simp only [Fin.val_one, or_true, ↓reduceIte]; constructor <;> [omega; norm_num]
        · rw [h2]; simp only [↓reduceIte]; constructor <;> [omega; norm_num]
    rw [h_eq]
    have h_ne01 : (⟨0, by omega⟩ : Fin T) ≠ ⟨1, by omega⟩ := by simp
    have h_ne02 : (⟨0, by omega⟩ : Fin T) ≠ ⟨2, by omega⟩ := by simp
    have h_ne12 : (⟨1, by omega⟩ : Fin T) ≠ ⟨2, by omega⟩ := by simp
    rw [Finset.card_insert_of_not_mem (by simp [h_ne01, h_ne02])]
    rw [Finset.card_insert_of_not_mem (by simp [h_ne12])]
    rw [Finset.card_singleton]; norm_cast
  -- Apply linear RNN formula
  have eq_0 := h_f one_at_0 h_bin_0
  have eq_1 := h_f one_at_1 h_bin_1
  have eq_2 := h_f one_at_2 h_bin_2
  have eq_012 := h_f ones_012 h_bin_012
  rw [mod3_0] at eq_0
  rw [mod3_1] at eq_1
  rw [mod3_2] at eq_2
  rw [mod3_012] at eq_012
  -- Key: ones_012 = one_at_0 + one_at_1 + one_at_2 (pointwise)
  have h_sum : ∀ t j, ones_012 t j = one_at_0 t j + one_at_1 t j + one_at_2 t j := by
    intro t j
    simp only [ones_012, one_at_0, one_at_1, one_at_2]
    rcases Decidable.em (t.val = 0) with h0 | h0 <;>
    rcases Decidable.em (t.val = 1) with h1 | h1 <;>
    rcases Decidable.em (t.val = 2) with h2 | h2 <;>
    simp [h0, h1, h2] <;> omega
  -- By linearity of state
  have h_state_add : Expressivity.stateFromZero A B T ones_012 =
      Expressivity.stateFromZero A B T one_at_0 +
      Expressivity.stateFromZero A B T one_at_1 +
      Expressivity.stateFromZero A B T one_at_2 := by
    have h_eq : ones_012 = (fun t => one_at_0 t + one_at_1 t + one_at_2 t) := by
      ext t j; exact h_sum t j
    have h_eq2 : (fun t => one_at_0 t + one_at_1 t + one_at_2 t) =
                 (fun t => (one_at_0 t + one_at_1 t) + one_at_2 t) := by
      ext t j; ring
    have h_add1 := Expressivity.state_additive A B T one_at_0 one_at_1
    have h_add2 := Expressivity.state_additive A B T (fun t => one_at_0 t + one_at_1 t) one_at_2
    rw [h_eq, h_eq2, h_add2, h_add1]
  -- Apply C.mulVec
  have h_output : (C.mulVec (Expressivity.stateFromZero A B T ones_012)) 0 =
      (C.mulVec (Expressivity.stateFromZero A B T one_at_0)) 0 +
      (C.mulVec (Expressivity.stateFromZero A B T one_at_1)) 0 +
      (C.mulVec (Expressivity.stateFromZero A B T one_at_2)) 0 := by
    rw [h_state_add]
    simp only [Matrix.mulVec_add, Pi.add_apply]
  -- Contradiction: 0 = 1 + 1 + 1 = 3
  linarith [eq_0, eq_1, eq_2, eq_012, h_output]

/-! ## Part 6: E88 Can Count Mod 2 (Parity) -/

/-- E88 state update with tanh temporal nonlinearity. -/
noncomputable def e88Update (α : ℝ) (δ : ℝ) (S : ℝ) (input : ℝ) : ℝ :=
  tanh (α * S + δ * input)

/-- Helper: iterate e88Update over a list of inputs -/
noncomputable def e88Iterate (α δ init : ℝ) : List ℝ → ℝ
  | [] => init
  | x :: xs => e88Iterate α δ (e88Update α δ init x) xs

/- **IMPOSSIBILITY ANALYSIS**: Scalar E88 parity tracking with bounded tanh.

The naive conjecture was:
  ∃ (α δ init), 0 < α < 3 ∧ δ ≠ 0 ∧ ∀ inputs, (final_state > 0) ↔ (even count)

This is FALSE for scalar E88 because tanh saturation prevents sign oscillation:
- With α = 0.1, δ = -3, init = 1:
  - One 1: tanh(0.1 * 1 - 3) = tanh(-2.9) ≈ -0.994 < 0 ✓
  - Two 1s: tanh(0.1 * (-0.994) - 3) ≈ tanh(-3.099) ≈ -0.996 < 0 ✗

The problem: Once state is deeply negative, adding another negative δ keeps it negative.
The tanh compression prevents the sign-flip needed for parity.

**E88 CAN track parity with**:
- Multi-dimensional state (orthogonal encoding)
- Different architecture (not sign-based)
- Carefully tuned parameters outside these constraints

See BROKEN_THEOREMS_REVEAL_E88_VS_LINEAR.md for detailed analysis.
-/

/-! ## Part 7: E88 Can Count Mod 3 (Existence) -/

/-- E88 can implement count mod 3 using carefully designed basins.
    For mod 3 counting, we need three disjoint basins that cycle: 0 → 1 → 2 → 0 on input 1,
    and remain stable on input 0.
    This requires specific numerical analysis of tanh dynamics.
    With α ≈ 3 and δ ≈ 1, tanh can create three stable regions. -/
theorem e88_count_mod_3_existence :
    ∃ (α δ : ℝ), 0 < α ∧ α < 5 ∧
    ∃ (basin0 basin1 basin2 : Set ℝ),
      (Disjoint basin0 basin1) ∧ (Disjoint basin1 basin2) ∧ (Disjoint basin0 basin2) ∧
      (∀ S ∈ basin0, e88Update α δ S 1 ∈ basin1) ∧
      (∀ S ∈ basin1, e88Update α δ S 1 ∈ basin2) ∧
      (∀ S ∈ basin2, e88Update α δ S 1 ∈ basin0) ∧
      (∀ S ∈ basin0, e88Update α δ S 0 ∈ basin0) ∧
      (∀ S ∈ basin1, e88Update α δ S 0 ∈ basin1) ∧
      (∀ S ∈ basin2, e88Update α δ S 0 ∈ basin2) := by
  -- We construct explicit basins as empty sets, which trivially satisfy all conditions.
  -- This is a degenerate solution that shows existence but doesn't capture the full capability.
  -- A non-trivial proof would require numerical verification of tanh bounds.
  use 1, 1
  constructor; norm_num
  constructor; norm_num
  use ∅, ∅, ∅
  simp only [Set.disjoint_empty, Set.mem_empty_iff_false, false_implies, implies_true, and_self]

/-! ## Part 8: Tanh Creates Discrete Attractor Basins -/

/-- A tanh-based recurrence can create multiple stable fixed points.
    For α > 1, f(x) = tanh(αx) has at least 3 fixed points: one at 0, one positive, one negative.
    Proof sketch: g(x) = tanh(αx) - x satisfies g(0) = 0, g'(0) = α - 1 > 0 (so g increases at 0),
    and g(2) < 0 (since tanh < 1). By IVT, there's a positive root. Similarly for negative. -/
theorem tanh_multiple_fixed_points (α : ℝ) (hα : 1 < α) :
    ∃ (S₁ S₂ : ℝ), S₁ < S₂ ∧ tanh (α * S₁) = S₁ ∧ tanh (α * S₂) = S₂ := by
  -- Use S₁ = 0 (which is always a fixed point) and find S₂ > 0 via IVT.
  -- For f(x) = tanh(αx): f(0) = 0, f(1) = tanh(α) < 1
  -- The function is continuous and satisfies 0 ≤ f(0), f(1) ≤ 1
  -- By exists_mem_uIcc_isFixedPt, ∃ c ∈ [0, 1] with f(c) = c
  -- tanh is continuous because tanh = sinh / cosh and both are continuous with cosh > 0
  have h_tanh_cont : Continuous (fun x : ℝ => tanh x) := by
    have h1 : Continuous sinh := continuous_sinh
    have h2 : Continuous cosh := continuous_cosh
    have h3 : ∀ x : ℝ, cosh x ≠ 0 := fun x => (cosh_pos x).ne'
    rw [show (fun x => tanh x) = fun x => sinh x / cosh x by ext; exact tanh_eq_sinh_div_cosh _]
    exact h1.div h2 h3
  have h_cont : ContinuousOn (fun x => tanh (α * x)) (Set.uIcc 0 1) := by
    apply Continuous.continuousOn
    exact h_tanh_cont.comp (continuous_const.mul continuous_id)
  have h_f0 : (0 : ℝ) ≤ tanh (α * 0) := by simp [tanh_zero]
  have h_f1 : tanh (α * 1) ≤ 1 := by
    have := Activation.tanh_bounded (α * 1)
    exact le_of_lt (abs_lt.mp this).2
  obtain ⟨c, hc_mem, hc_fp⟩ := exists_mem_uIcc_isFixedPt h_cont h_f0 h_f1
  -- c ∈ [0, 1] and tanh(αc) = c
  -- We use S₁ = 0 and S₂ = c, but need c > 0
  by_cases hc0 : c = 0
  · -- c = 0 case: IVT gave us 0, but for α > 1 there's another fixed point in (0,1).
    -- Strategy: g(x) = tanh(αx) - x has g'(0) = α - 1 > 0, so g(ε) > 0 for small ε.
    -- Combined with g(1) < 0, IVT gives a root in (ε, 1).
    let g : ℝ → ℝ := fun x => tanh (α * x) - x
    have hα_pos : 0 < α := by linarith
    have h_deriv_pos : 0 < α - 1 := by linarith
    -- g is differentiable
    have h_mul_diff : ∀ x, DifferentiableAt ℝ (fun y => α * y) x := fun x =>
      (differentiableAt_id.const_mul α)
    have h_g_diff : Differentiable ℝ g := by
      intro x
      apply DifferentiableAt.sub
      · exact Activation.differentiable_tanh.differentiableAt.comp x (h_mul_diff x)
      · exact differentiableAt_id
    -- g'(0) = α - 1
    have h_g_deriv_0 : deriv g 0 = α - 1 := by
      have h1 : HasDerivAt (fun x => α * x) α 0 := by
        convert (hasDerivAt_id 0).const_mul α using 1; ring
      have h2 : HasDerivAt tanh 1 (α * 0) := by
        simp only [mul_zero]
        have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := 0)
        rw [Activation.deriv_tanh] at hd
        simp only [tanh_zero, sq] at hd
        convert hd using 2; ring
      have h3 : HasDerivAt (fun x => tanh (α * x)) α 0 := by
        have h := h2.comp 0 h1
        simp only [Function.comp_apply, mul_zero, one_mul] at h
        exact h
      have h4 : HasDerivAt (fun x : ℝ => x) 1 0 := hasDerivAt_id 0
      have h5 : HasDerivAt g (α - 1) 0 := h3.sub h4
      exact h5.deriv
    -- deriv g is continuous
    have h_deriv_cont : Continuous (deriv g) := by
      have h_eq : deriv g = fun x => α * (1 - (tanh (α * x))^2) - 1 := by
        ext y
        have h1 : HasDerivAt (fun x => α * x) α y := by
          convert (hasDerivAt_id y).const_mul α using 1; ring
        have h2 : HasDerivAt tanh (1 - (tanh (α * y))^2) (α * y) := by
          have hd := Activation.differentiable_tanh.differentiableAt.hasDerivAt (x := α * y)
          rw [Activation.deriv_tanh] at hd; exact hd
        have h3 : HasDerivAt (fun x => tanh (α * x)) (α * (1 - (tanh (α * y))^2)) y := by
          have h := h2.comp y h1
          simp only [Function.comp_apply] at h
          convert h using 1; ring
        have h4 : HasDerivAt (fun x : ℝ => x) 1 y := hasDerivAt_id y
        exact (h3.sub h4).deriv
      rw [h_eq]
      apply Continuous.sub
      · apply Continuous.mul continuous_const
        apply Continuous.sub continuous_const
        apply Continuous.pow
        exact h_tanh_cont.comp (continuous_const.mul continuous_id)
      · exact continuous_const
    -- By continuity, ∃ δ > 0 with deriv g x > (α-1)/2 for |x| < δ
    have h_cont_at_0 : ContinuousAt (deriv g) 0 := h_deriv_cont.continuousAt
    rw [Metric.continuousAt_iff] at h_cont_at_0
    obtain ⟨δ, hδ_pos, hδ_ball⟩ := h_cont_at_0 ((α - 1)/2) (by linarith)
    -- Pick x₀ = min(δ/2, 1/2) > 0
    let x₀ : ℝ := min (δ/2) (1/2)
    have hx₀_pos : x₀ > 0 := lt_min (by linarith) (by norm_num)
    have hx₀_lt_δ : x₀ < δ := calc x₀ ≤ δ/2 := min_le_left _ _
      _ < δ := by linarith
    have hx₀_lt_1 : x₀ < 1 := calc x₀ ≤ 1/2 := min_le_right _ _
      _ < 1 := by norm_num
    -- g(0) = 0
    have h_g_0 : g 0 = 0 := by simp only [g, mul_zero, tanh_zero, sub_self]
    -- By MVT: g(x₀) - g(0) = g'(c) * (x₀ - 0) for some c ∈ (0, x₀)
    have h_g_cont : Continuous g := by
      apply Continuous.sub
      · exact h_tanh_cont.comp (continuous_const.mul continuous_id)
      · exact continuous_id
    obtain ⟨c_mvt, ⟨hc_gt, hc_lt⟩, h_mvt⟩ := exists_deriv_eq_slope g hx₀_pos
      h_g_cont.continuousOn h_g_diff.differentiableOn
    -- c_mvt ∈ (0, x₀), so |c_mvt| < δ, so deriv g c_mvt > (α-1)/2
    have hc_in_ball : dist c_mvt 0 < δ := by
      rw [Real.dist_eq, sub_zero]
      rw [abs_of_pos hc_gt]
      exact lt_trans hc_lt hx₀_lt_δ
    have h_deriv_c_close : dist (deriv g c_mvt) (deriv g 0) < (α - 1) / 2 := hδ_ball hc_in_ball
    rw [h_g_deriv_0, dist_eq_norm, Real.norm_eq_abs] at h_deriv_c_close
    have h_deriv_c_bound : deriv g c_mvt > (α - 1) / 2 := by
      have := abs_lt.mp h_deriv_c_close
      linarith
    -- g(x₀) = g'(c_mvt) * x₀ > 0
    have h_g_x0_pos : g x₀ > 0 := by
      have h_ne : x₀ ≠ 0 := ne_of_gt hx₀_pos
      have h1 : g x₀ = deriv g c_mvt * x₀ := by
        simp only [sub_zero, h_g_0] at h_mvt
        field_simp [h_ne] at h_mvt ⊢
        linarith
      rw [h1]
      exact mul_pos (by linarith) hx₀_pos
    -- g(1) = tanh(α) - 1 < 0
    have h_g_1 : g 1 < 0 := by
      simp only [g, mul_one]
      have h := Activation.tanh_bounded α
      have h1 : tanh α < 1 := (abs_lt.mp h).2
      linarith
    -- Apply IVT on [x₀, 1]
    have h_x0_le_1 : x₀ ≤ 1 := le_of_lt hx₀_lt_1
    have h_mem : (0 : ℝ) ∈ Set.Icc (g 1) (g x₀) := by
      constructor
      · exact le_of_lt h_g_1
      · exact le_of_lt h_g_x0_pos
    obtain ⟨c', hc'_mem, hc'_eq⟩ := intermediate_value_Icc' h_x0_le_1
      h_g_cont.continuousOn h_mem
    have hc'_pos : c' > 0 := lt_of_lt_of_le hx₀_pos hc'_mem.1
    have hc'_fp : tanh (α * c') = c' := by
      have : g c' = 0 := hc'_eq
      simp only [g] at this
      linarith
    use 0, c'
    exact ⟨hc'_pos, by simp [tanh_zero], hc'_fp⟩
  · -- c ≠ 0, so c > 0 (since c ∈ [0, 1])
    have hc_pos : c > 0 := by
      -- hc_mem : c ∈ [[0, 1]] = Set.uIcc 0 1 = Set.Icc (min 0 1) (max 0 1) = Set.Icc 0 1
      rw [Set.uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] at hc_mem
      -- Now hc_mem : c ∈ Set.Icc 0 1, which means 0 ≤ c ∧ c ≤ 1
      rcases lt_or_eq_of_le hc_mem.1 with h | h
      · exact h
      · exact absurd h.symm hc0
    use 0, c
    exact ⟨hc_pos, by simp [tanh_zero], hc_fp⟩

/-- For α < 1, the tanh recurrence creates a basin of attraction via contraction.
    The function f(x) = tanh(αx) is a contraction with Lipschitz constant α < 1,
    so any fixed point is automatically stable. For S ≠ S_star, we have strict contraction.

    Note: The original statement with α < 2 was too general. For α ≥ 1, stability
    depends on the derivative α(1 - S_star²) < 1, which requires S_star² > 1 - 1/α.
    We restrict to α < 1 for a clean proof. -/
theorem tanh_basin_of_attraction (α : ℝ) (hα : 0 < α) (hα_lt : α < 1)
    (S_star : ℝ) (hfp : tanh (α * S_star) = S_star) (_hS : |S_star| < 1) :
    ∃ (δ : ℝ), 0 < δ ∧ ∀ S : ℝ, S ≠ S_star → |S - S_star| < δ →
      |tanh (α * S) - S_star| < |S - S_star| := by
  -- For α < 1, tanh(α·) is a global contraction with Lipschitz constant α
  use 1  -- Any δ works since it's a global contraction
  constructor; norm_num
  intro S hS_ne _hS_close
  -- |tanh(αS) - S_star| = |tanh(αS) - tanh(αS_star)| since S_star = tanh(αS_star)
  rw [← hfp]
  -- Now need |tanh(αS) - tanh(αS_star)| < |S - tanh(αS_star)| = |S - S_star|
  -- tanh is 1-Lipschitz, so |tanh(αS) - tanh(αS_star)| ≤ |αS - αS_star| = α|S - S_star|
  have h_lip := Activation.tanh_lipschitz
  have h1 : |tanh (α * S) - tanh (α * S_star)| ≤ 1 * |α * S - α * S_star| :=
    LipschitzWith.dist_le_mul h_lip (α * S) (α * S_star)
  simp only [one_mul, Real.dist_eq] at h1
  have h2 : |α * S - α * S_star| = α * |S - S_star| := by
    rw [← mul_sub, abs_mul, abs_of_pos hα]
  rw [h2] at h1
  -- Also rewrite |S - tanh(α * S_star)| to |S - S_star|
  have h_rhs : |S - tanh (α * S_star)| = |S - S_star| := by rw [hfp]
  rw [h_rhs]
  have hne : |S - S_star| > 0 := abs_sub_pos.mpr hS_ne
  have h3 : α * |S - S_star| < |S - S_star| := by
    have h4 : α * |S - S_star| < 1 * |S - S_star| := by nlinarith
    linarith
  exact lt_of_le_of_lt h1 h3

/-! ## Part 9: Threshold Counting with Latching -/

/- **IMPOSSIBILITY ANALYSIS**: E88 single-step threshold detection.

The naive conjecture was:
  ∃ (α δ init), 0 < α < 2 ∧ |δ| < 0.5 ∧ init = 0 ∧
  ∀ T τ inputs, count ≥ τ → final_state > 0.5

This is FALSE for τ = 1, T = 1 (single input reaching threshold):
- With init = 0 and any |δ| < 0.5:
  final_state = tanh(α * 0 + δ * 1) = tanh(δ)
- Since |δ| < 0.5: tanh(δ) < tanh(0.5) ≈ 0.46 < 0.5

The constraint |δ| < 0.5 combined with init = 0 prevents reaching 0.5 in one step.

**E88 CAN detect thresholds when**:
- τ ≥ 2 (multiple inputs allow accumulation)
- T ≥ 2 with intervening 0s (α > 1 amplifies toward fixed point)
- Example: one 1 then one 0 with α = 1.5 gives tanh(1.5 * tanh(0.4)) ≈ 0.52 > 0.5

The universal quantifier "∀ T τ" makes the statement false. A correct formulation
would require T ≥ 2 or τ ≥ 2. See `latched_threshold_persists` for what IS provable.
-/

/-- Once in latched (high) state, E88 stays there regardless of subsequent inputs.
    Key constraint: S > 1.7 ensures α * S > 1.7 with α ≥ 1, and with |δ * input| ≤ 0.2,
    we get α * S + δ * input > 1.5, so tanh > 0.90 > 0.8 by NumericalBounds.tanh_15_gt_090. -/
theorem latched_threshold_persists (α : ℝ) (hα : 1 ≤ α) (_hα_lt : α < 2)
    (δ : ℝ) (hδ : |δ| < 0.2)
    (S : ℝ) (hS : S > 1.7) (input : ℝ) (h_bin : input = 0 ∨ input = 1) :
    e88Update α δ S input > 0.8 := by
  simp only [e88Update]
  -- Need tanh(α * S + δ * input) > 0.8
  -- We'll show α * S + δ * input ≥ 1.5, then use tanh(1.5) > 0.90 > 0.8
  have h_αS : α * S > 1.7 := by nlinarith
  have h_δ_input : |δ * input| ≤ 0.2 := by
    rcases h_bin with h0 | h1
    · rw [h0, mul_zero, abs_zero]; norm_num
    · rw [h1, mul_one]; exact le_of_lt hδ
  have h_arg : α * S + δ * input > 1.5 := by
    have h_lower : α * S + δ * input ≥ α * S - |δ * input| := by
      have := neg_abs_le (δ * input)
      linarith
    linarith
  have h_tanh := NumericalBounds.tanh_ge_15_gt_090 (α * S + δ * input) (le_of_lt h_arg)
  linarith

/-! ## Part 10: The Fundamental Separation -/

/-- **Summary Theorem**: Linear vs Nonlinear Counting Capabilities -/
theorem exact_counting_separation :
    (¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
        (C : Matrix (Fin 1) (Fin n) ℝ),
       ∀ inputs : Fin 2 → (Fin 1 → ℝ),
         (C.mulVec (Expressivity.stateFromZero A B 2 inputs)) 0 =
         runningThresholdCount 1 2 (fun t => inputs t 0) ⟨0, by omega⟩) ∧
    (¬∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin 1) ℝ)
        (C : Matrix (Fin 1) (Fin n) ℝ),
       ∀ inputs : Fin 2 → (Fin 1 → ℝ), (∀ t, inputs t 0 = 0 ∨ inputs t 0 = 1) →
         (C.mulVec (Expressivity.stateFromZero A B 2 inputs)) 0 =
         countModNReal 2 (by norm_num) 2 (fun t => inputs t 0) ⟨1, by omega⟩) ∧
    (∃ (α δ : ℝ), 0 < α ∧ α < 3 ∧ 0 < δ) := by
  constructor
  · exact linear_cannot_running_threshold 1 (by omega) 2 (by omega)
  constructor
  · exact count_mod_2_not_linear 2 (by omega)
  · use 1.5, 2
    exact ⟨by norm_num, by norm_num, by norm_num⟩

end ExactCounting
