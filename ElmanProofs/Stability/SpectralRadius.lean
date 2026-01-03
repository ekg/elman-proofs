/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.LinearAlgebra.Eigenspace.Basic
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.Normed.Algebra.Spectrum
import Mathlib.Topology.Algebra.InfiniteSum.Basic
import Mathlib.Analysis.Matrix.Normed

/-!
# Spectral Radius and RNN Stability

This file develops the theory connecting spectral radius to RNN stability.

## Main Definitions

* `spectralRadius`: The largest absolute value of eigenvalues (via Gelfand's formula)
* `isSpectrallyStable`: Spectral radius < 1

## Main Theorems

* `spectral_radius_le_norm`: ρ(A) ≤ ‖A‖ for any matrix norm
* `power_iteration_converges`: If ρ(A) < 1, then A^n → 0

## Application to RNNs

For the recurrence h_{t+1} = R_h · h_t + input:
- If ρ(R_h) < 1, perturbations decay exponentially
- The system has a unique stable fixed point
- Memory of initial conditions fades geometrically

## Implementation Notes

The spectral radius is defined using Gelfand's formula: ρ(A) = lim_{k→∞} ‖A^k‖^(1/k).
This avoids dealing with complex eigenvalues directly while being mathematically equivalent
to the maximum absolute eigenvalue definition.

-/

namespace SpectralRadius

open Matrix
open scoped Matrix

variable {n : ℕ} [NeZero n]

/-- Spectral radius via Gelfand's formula: ρ(A) = inf_k ‖A^k‖^(1/k).
    This equals lim_{k→∞} ‖A^k‖^(1/k) = sup{|λ| : λ is an eigenvalue of A}.

    For practical computation, we use the infimum characterization.
    The actual limit exists and equals this infimum by subadditivity. -/
noncomputable def spectralRadius (A : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  -- Use Frobenius norm as a simple matrix norm
  let frobNorm := fun M : Matrix (Fin n) (Fin n) ℝ =>
    Real.sqrt (∑ i, ∑ j, (M i j)^2)
  ⨅ k : ℕ, (frobNorm (A^(k+1)))^(1 / (k+1 : ℝ))

/-- A matrix is spectrally stable if its spectral radius is less than 1. -/
def IsSpectrallyStable (A : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  spectralRadius A < 1

/-- Frobenius norm of a matrix. -/
noncomputable def frobNorm (M : Matrix (Fin n) (Fin n) ℝ) : ℝ :=
  Real.sqrt (∑ i, ∑ j, (M i j)^2)

/-- Spectral radius is bounded by the Frobenius norm.

    ## Proof Sketch

    The spectral radius is defined as ⨅ k, ‖A^(k+1)‖^{1/(k+1)}.
    Taking k = 0 gives ‖A^1‖^{1/1} = ‖A‖ = frobNorm A.
    Since the infimum is ≤ any element in the set, ρ(A) ≤ frobNorm A.

    The proof uses ciInf_le with h_bdd showing 0 is a lower bound.
-/
theorem spectral_radius_le_frobNorm (A : Matrix (Fin n) (Fin n) ℝ) :
    spectralRadius A ≤ frobNorm A := by
  unfold spectralRadius frobNorm
  let f : ℕ → ℝ := fun k => (Real.sqrt (∑ i, ∑ j, (A ^ (k + 1)) i j ^ 2)) ^ ((1 : ℝ) / (↑k + 1))
  show ⨅ k, f k ≤ Real.sqrt (∑ i, ∑ j, A i j ^ 2)
  have h_bdd : BddBelow (Set.range f) := by
    use 0
    intro x hx
    simp only [Set.mem_range] at hx
    obtain ⟨k, rfl⟩ := hx
    simp only [f]
    apply Real.rpow_nonneg
    exact Real.sqrt_nonneg _
  have h := ciInf_le h_bdd 0
  have h_f0 : f 0 = Real.sqrt (∑ i, ∑ j, A i j ^ 2) := by
    simp only [f, zero_add, pow_one]
    have h1 : (1 : ℝ) / (↑(0 : ℕ) + 1) = 1 := by norm_num
    rw [h1, Real.rpow_one]
  rw [h_f0] at h
  exact h

/-- Powers of spectrally stable matrices converge to zero.

    ## Proof Outline

    Given ρ(A) < 1 (spectral stability):

    1. **Choose r with ρ(A) < r < 1**: By definition of spectral radius as infimum,
       for any r > ρ(A), there exists K such that ‖A^K‖^{1/K} < r.

    2. **Bound on powers**: This means ‖A^K‖ < r^K, and by submultiplicativity:
       ‖A^{mK}‖ ≤ ‖A^K‖^m < r^{mK}

    3. **General bound**: For any k ≥ K, write k = mK + j where 0 ≤ j < K.
       Then ‖A^k‖ = ‖A^{mK} · A^j‖ ≤ ‖A^K‖^m · ‖A^j‖ ≤ C · r^{mK}
       where C = max{‖A^j‖ : j < K}

    4. **Convergence**: Since r < 1, r^k → 0 as k → ∞.
       Thus ‖A^k‖ → 0, which implies each entry |A^k_{ij}| → 0.

    5. **Entry bound**: |(A^k)_{ij}| ≤ ‖A^k‖ for any matrix norm,
       so entry-wise convergence follows from norm convergence.
-/
theorem powers_tendsto_zero (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : IsSpectrallyStable A) :
    ∀ ε > 0, ∃ N, ∀ k ≥ N, ∀ i j, |(A^k) i j| < ε := by
  intro ε hε

  -- From spectral stability: ρ(A) < 1
  unfold IsSpectrallyStable spectralRadius at hA

  -- Step 1: Extract witnessing K such that ‖A^(K+1)‖_F^(1/(K+1)) < 1
  -- Use exists_lt_of_ciInf_lt: if iInf < a, then ∃ i, f(i) < a
  obtain ⟨K, hK⟩ := exists_lt_of_ciInf_lt hA

  -- hK : (frobNorm (A^(K+1)))^(1/(K+1)) < 1
  let r := (frobNorm (A^(K+1)))^(1 / (K+1 : ℝ))
  have h_r_lt_one : r < 1 := hK
  have h_r_nonneg : 0 ≤ r := by
    simp only [r]
    apply Real.rpow_nonneg
    apply Real.sqrt_nonneg

  -- Step 2: From r = ‖A^(K+1)‖^(1/(K+1)) < 1, get ‖A^(K+1)‖ ≤ r^(K+1) < 1
  -- The bound ‖A^(K+1)‖ ≤ 1 is key for geometric decay

  have h_frob_bound : frobNorm (A^(K+1)) ≤ 1 := by
    -- r = frobNorm(A^(K+1))^(1/(K+1)) < 1
    -- so frobNorm(A^(K+1)) = r^(K+1) < 1
    have h_pos : (0 : ℝ) < K + 1 := by positivity
    have h_frob_nonneg : 0 ≤ frobNorm (A^(K+1)) := Real.sqrt_nonneg _

    -- Key: x^(1/p) < 1 with x ≥ 0, p > 0 implies x < 1
    -- Since (frobNorm)^(1/(K+1)) = r < 1 and frobNorm ≥ 0, we have frobNorm < 1

    -- Proof: If x ≥ 1 and p > 0, then x^(1/p) ≥ 1 (since x^(1/p) ≥ 1^(1/p) = 1)
    -- Contrapositive: x^(1/p) < 1 implies x < 1
    by_contra h_not_le
    push_neg at h_not_le
    -- h_not_le : 1 < frobNorm (A^(K+1))
    have h_ge_one : 1 ≤ frobNorm (A^(K+1)) := le_of_lt h_not_le
    have h_rpow_ge : (1:ℝ)^(1/(K+1:ℝ)) ≤ (frobNorm (A^(K+1)))^(1/(K+1:ℝ)) := by
      apply Real.rpow_le_rpow
      · norm_num
      · exact h_ge_one
      · apply div_nonneg; norm_num; linarith
    simp only [Real.one_rpow] at h_rpow_ge
    -- Now h_rpow_ge : 1 ≤ r, but h_r_lt_one : r < 1. Contradiction.
    linarith

  -- Step 3: Entry-wise bound from Frobenius norm
  -- For any matrix M, |M i j| ≤ ‖M‖_F
  have h_entry_bound : ∀ M : Matrix (Fin n) (Fin n) ℝ, ∀ i j, |M i j| ≤ frobNorm M := by
    intro M i j
    simp only [frobNorm]
    -- |M i j| ≤ √(∑ i', ∑ j', (M i' j')²)
    -- Since (M i j)² ≤ ∑ i', ∑ j', (M i' j')², we have |M i j| ≤ √(sum)
    have h_sq_le : (M i j)^2 ≤ ∑ i', ∑ j', (M i' j')^2 := by
      have h_row : (M i j)^2 ≤ ∑ j' : Fin n, (M i j')^2 := by
        have h_nonneg : ∀ k ∈ Finset.univ, 0 ≤ (M i k)^2 := fun k _ => sq_nonneg _
        exact Finset.single_le_sum h_nonneg (Finset.mem_univ j)
      have h_col : ∑ j' : Fin n, (M i j')^2 ≤ ∑ i' : Fin n, ∑ j' : Fin n, (M i' j')^2 := by
        have h_nonneg : ∀ k ∈ Finset.univ, 0 ≤ ∑ j' : Fin n, (M k j')^2 := by
          intro k _; apply Finset.sum_nonneg; intro l _; exact sq_nonneg _
        exact Finset.single_le_sum h_nonneg (Finset.mem_univ i)
      exact le_trans h_row h_col
    have h_abs_sq : |M i j|^2 = (M i j)^2 := sq_abs (M i j)
    rw [← h_abs_sq] at h_sq_le
    have h_sum_nonneg : 0 ≤ ∑ i', ∑ j', (M i' j')^2 := by
      apply Finset.sum_nonneg; intro i' _
      apply Finset.sum_nonneg; intro j' _
      exact sq_nonneg _
    -- |M i j| ≤ √(sum) iff |M i j|² ≤ sum (for nonneg values)
    -- Use: a ≤ √b ↔ a² ≤ b (for a ≥ 0)
    have h := Real.sqrt_le_sqrt h_sq_le
    rw [Real.sqrt_sq (abs_nonneg _)] at h
    exact h

  -- Step 4: Submultiplicativity of Frobenius norm via Cauchy-Schwarz
  have h_submul : ∀ (M N : Matrix (Fin n) (Fin n) ℝ), frobNorm (M * N) ≤ frobNorm M * frobNorm N := by
    intro M N
    simp only [frobNorm]
    -- Goal: √(∑ i, ∑ j, (M * N)_{i,j}²) ≤ √(∑ i, ∑ k, M_{i,k}²) * √(∑ k, ∑ j, N_{k,j}²)
    -- By Cauchy-Schwarz: (∑_k M_{i,k} * N_{k,j})² ≤ (∑_k M_{i,k}²) * (∑_k N_{k,j}²)

    -- Submultiplicativity via Mathlib's frobenius_norm_mul
    -- Our frobNorm M = ‖M‖ using Frobenius norm instance
    letI := Matrix.frobeniusSeminormedAddCommGroup (α := ℝ) (m := Fin n) (n := Fin n)
    have h_frobNorm_eq : ∀ P : Matrix (Fin n) (Fin n) ℝ, frobNorm P = ‖P‖ := by
      intro P
      simp only [frobNorm]
      rw [Matrix.frobenius_norm_def]
      -- Need: √(∑ i, ∑ j, P i j ^ 2) = (∑ i, ∑ j, ‖P i j‖ ^ 2) ^ (1 / 2)
      have h_sum_eq : ∑ i : Fin n, ∑ j : Fin n, (P i j)^2 =
                      ∑ i : Fin n, ∑ j : Fin n, ‖P i j‖^(2 : ℝ) := by
        apply Finset.sum_congr rfl; intro i _
        apply Finset.sum_congr rfl; intro j _
        rw [Real.norm_eq_abs]
        -- P i j ^ 2 = |P i j| ^ 2 because |x|^2 = x^2
        rw [(sq_abs (P i j)).symm]
        -- Now need: |P i j| ^ 2 = |P i j| ^ (2 : ℝ)
        rw [← Real.rpow_natCast |P i j| 2]
        simp only [Nat.cast_ofNat]
      rw [h_sum_eq]
      -- Now: √x = x^(1/2) for x ≥ 0
      have h_sum_nonneg : 0 ≤ ∑ i : Fin n, ∑ j : Fin n, ‖P i j‖^(2 : ℝ) := by
        apply Finset.sum_nonneg; intro i _
        apply Finset.sum_nonneg; intro j _
        apply Real.rpow_nonneg (norm_nonneg _)
      rw [Real.sqrt_eq_rpow, one_div]
    -- Now use: frobNorm (M * N) ≤ frobNorm M * frobNorm N
    simp only [frobNorm] at h_frobNorm_eq ⊢
    have h1 := h_frobNorm_eq (M * N)
    have h2 := h_frobNorm_eq M
    have h3 := h_frobNorm_eq N
    rw [h1, h2, h3]
    exact Matrix.frobenius_norm_mul M N

  -- Step 5: Bound on powers of A^(K+1) for m ≥ 1
  have h_pow_bound : ∀ m, 1 ≤ m → frobNorm ((A^(K+1))^m) ≤ (frobNorm (A^(K+1)))^m := by
    intro m hm
    induction m with
    | zero => omega
    | succ m ih =>
      match m with
      | 0 =>
        -- m+1 = 1, so goal is frobNorm((A^(K+1))^1) ≤ frobNorm(A^(K+1))^1
        simp only [zero_add, pow_one, le_refl]
      | m' + 1 =>
        have h_m_ge : 1 ≤ m' + 1 := Nat.one_le_iff_ne_zero.mpr (Nat.succ_ne_zero m')
        calc frobNorm ((A ^ (K + 1)) ^ (m' + 1 + 1))
            = frobNorm ((A ^ (K + 1)) ^ (m' + 1) * (A ^ (K + 1))) := by rw [pow_succ]
          _ ≤ frobNorm ((A ^ (K + 1)) ^ (m' + 1)) * frobNorm (A ^ (K + 1)) := h_submul _ _
          _ ≤ (frobNorm (A ^ (K + 1))) ^ (m' + 1) * frobNorm (A ^ (K + 1)) := by
              apply mul_le_mul_of_nonneg_right (ih h_m_ge) (Real.sqrt_nonneg _)
          _ = (frobNorm (A ^ (K + 1))) ^ (m' + 1 + 1) := by ring

  -- Step 6: Let C = max of frobNorm(A^j) for j = 0, ..., K
  let C := Finset.sup' (Finset.range (K + 1)) ⟨0, Finset.mem_range.mpr (Nat.zero_lt_succ K)⟩
    (fun j => frobNorm (A ^ j))
  have hC_nonneg : 0 ≤ C := by
    have h_mem : 0 ∈ Finset.range (K + 1) := Finset.mem_range.mpr (Nat.zero_lt_succ K)
    have h_val_nonneg : 0 ≤ frobNorm (A ^ 0) := by simp only [pow_zero, frobNorm]; exact Real.sqrt_nonneg _
    exact le_trans h_val_nonneg (Finset.le_sup' (fun j => frobNorm (A ^ j)) h_mem)

  -- Step 7: For any k, bound frobNorm(A^k) ≤ frobNorm(A^(K+1))^(k/(K+1)) * C
  have h_decomp_bound : ∀ k, frobNorm (A^k) ≤ (frobNorm (A^(K+1)))^(k / (K+1)) * C := by
    intro k
    set m := k / (K + 1) with hm_def
    set j := k % (K + 1) with hj_def
    have h_k_eq : k = (K + 1) * m + j := (Nat.div_add_mod k (K + 1)).symm
    have hj_lt : j < K + 1 := Nat.mod_lt k (Nat.zero_lt_succ K)
    have h_pow_eq : A^k = (A^(K+1))^m * A^j := by rw [h_k_eq, pow_add, pow_mul]
    rw [h_pow_eq]
    have hj_mem : j ∈ Finset.range (K + 1) := Finset.mem_range.mpr hj_lt
    have hj_le_C : frobNorm (A ^ j) ≤ C := Finset.le_sup' (fun i => frobNorm (A ^ i)) hj_mem
    match m with
    | 0 =>
      simp only [pow_zero, Matrix.one_mul, one_mul]
      exact hj_le_C
    | m' + 1 =>
      have hm_pos : 1 ≤ m' + 1 := Nat.one_le_iff_ne_zero.mpr (Nat.succ_ne_zero m')
      calc frobNorm ((A ^ (K + 1)) ^ (m' + 1) * A ^ j)
          ≤ frobNorm ((A ^ (K + 1)) ^ (m' + 1)) * frobNorm (A ^ j) := h_submul _ _
        _ ≤ (frobNorm (A ^ (K + 1))) ^ (m' + 1) * frobNorm (A ^ j) := by
            apply mul_le_mul_of_nonneg_right (h_pow_bound (m' + 1) hm_pos) (Real.sqrt_nonneg _)
        _ ≤ (frobNorm (A ^ (K + 1))) ^ (m' + 1) * C := by
            apply mul_le_mul_of_nonneg_left hj_le_C (pow_nonneg (Real.sqrt_nonneg _) _)

  -- Step 8: Case split on frobNorm(A^(K+1)) = 0
  by_cases h_frob_zero : frobNorm (A^(K+1)) = 0
  · -- frobNorm(A^(K+1)) = 0 means A^(K+1) = 0
    have h_zero : A^(K+1) = 0 := by
      ext i j
      have := h_entry_bound (A^(K+1)) i j
      rw [h_frob_zero] at this
      exact abs_eq_zero.mp (le_antisymm this (abs_nonneg _))
    use K + 1
    intro k hk i j
    have h_decomp : k = (K + 1) + (k - (K + 1)) := (Nat.add_sub_cancel' hk).symm
    rw [h_decomp, pow_add, h_zero, Matrix.zero_mul]
    simp only [Matrix.zero_apply, abs_zero]
    exact hε

  · -- frobNorm(A^(K+1)) > 0 but < 1
    have h_frob_pos : 0 < frobNorm (A^(K+1)) := (Real.sqrt_nonneg _).lt_of_ne' h_frob_zero
    have h_frob_lt_one : frobNorm (A^(K+1)) < 1 := by
      rcases h_frob_bound.eq_or_lt with h_eq | h_lt
      · exfalso
        have h_r_eq_one : r = 1 := by simp only [r]; rw [h_eq, Real.one_rpow]
        linarith
      · exact h_lt

    -- C > 0 (since it includes frobNorm(1) = √n > 0)
    have hC_pos : 0 < C := by
      have h_mem : 0 ∈ Finset.range (K + 1) := Finset.mem_range.mpr (Nat.zero_lt_succ K)
      have h_frob_one_pos : 0 < frobNorm (A ^ 0) := by
        simp only [pow_zero, frobNorm]
        apply Real.sqrt_pos.mpr
        have h_sum : ∑ i : Fin n, ∑ j : Fin n, ((1 : Matrix (Fin n) (Fin n) ℝ) i j)^2 = n := by
          simp only [Matrix.one_apply]
          -- Sum is ∑ i, ∑ j, (if i = j then 1 else 0)^2 = ∑ i, 1 = n
          have h_inner : ∀ i : Fin n, ∑ j : Fin n, (if i = j then (1 : ℝ) else 0)^2 = 1 := by
            intro i
            rw [Fintype.sum_eq_single i]
            · simp
            · intro j hij; simp [Ne.symm hij]
          simp only [h_inner, Finset.sum_const, Finset.card_fin, nsmul_eq_mul, mul_one]
        rw [h_sum]
        exact Nat.cast_pos.mpr (NeZero.pos n)
      exact lt_of_lt_of_le h_frob_one_pos (Finset.le_sup' (fun j => frobNorm (A ^ j)) h_mem)

    -- frobNorm(A^(K+1))^m → 0 as m → ∞
    have h_tendsto : Filter.Tendsto (fun m => (frobNorm (A^(K+1)))^m) Filter.atTop (nhds 0) :=
      tendsto_pow_atTop_nhds_zero_of_lt_one (Real.sqrt_nonneg _) h_frob_lt_one
    rw [Metric.tendsto_atTop] at h_tendsto
    obtain ⟨M, hM⟩ := h_tendsto (ε / C) (div_pos hε hC_pos)

    use (K + 1) * M
    intro k hk i j

    have h_div_ge : M ≤ k / (K + 1) := by
      rw [Nat.le_div_iff_mul_le (Nat.zero_lt_succ K)]
      rw [mul_comm]
      exact hk

    have h_pow_le : (frobNorm (A^(K+1)))^(k / (K + 1)) ≤ (frobNorm (A^(K+1)))^M :=
      pow_le_pow_of_le_one (Real.sqrt_nonneg _) (le_of_lt h_frob_lt_one) h_div_ge

    have h_pow_bound' : (frobNorm (A^(K+1)))^M < ε / C := by
      have h_dist := hM M le_rfl
      simp only [Real.dist_eq, sub_zero] at h_dist
      have h_nonneg : 0 ≤ (frobNorm (A^(K+1)))^M := pow_nonneg (Real.sqrt_nonneg _) M
      rwa [abs_of_nonneg h_nonneg] at h_dist

    calc |(A^k) i j|
        ≤ frobNorm (A^k) := h_entry_bound (A^k) i j
      _ ≤ (frobNorm (A^(K+1)))^(k / (K + 1)) * C := h_decomp_bound k
      _ ≤ (frobNorm (A^(K+1)))^M * C := mul_le_mul_of_nonneg_right h_pow_le (le_of_lt hC_pos)
      _ < (ε / C) * C := mul_lt_mul_of_pos_right h_pow_bound' hC_pos
      _ = ε := div_mul_cancel₀ ε (ne_of_gt hC_pos)

/-- Spectral radius of diagonal matrix is max of absolute diagonal entries.

    ## Proof Outline

    For a diagonal matrix D = diag(d₀, d₁, ..., d_{n-1}):

    1. **Powers of diagonal matrices**: D^k = diag(d₀^k, d₁^k, ..., d_{n-1}^k)

    2. **Frobenius norm of diagonal**: ‖D^k‖_F = √(∑ᵢ |dᵢ^k|²) = √(∑ᵢ |dᵢ|^{2k})

    3. **k-th root**: ‖D^k‖_F^{1/k} = (∑ᵢ |dᵢ|^{2k})^{1/2k}

    4. **Limit as k → ∞**: This is the ℓ^{2k} → ℓ^∞ limit for the sequence (|d₀|, |d₁|, ...)
       which converges to max{|dᵢ|}

    The formal proof requires:
    - `Matrix.diagonal_pow` for powers of diagonal matrices
    - Analysis of the limit (∑ᵢ xᵢ^p)^{1/p} → max{xᵢ} as p → ∞
-/
theorem diagonal_spectral_radius (d : Fin n → ℝ) :
    spectralRadius (Matrix.diagonal d) = Finset.sup' Finset.univ ⟨0, Finset.mem_univ 0⟩
      (fun i => |d i|) := by
  unfold spectralRadius
  simp only

  -- Let m = max{|dᵢ|}
  let m := Finset.sup' Finset.univ ⟨0, Finset.mem_univ 0⟩ (fun i => |d i|)

  -- Define the Frobenius norm locally
  let frobNorm := fun M : Matrix (Fin n) (Fin n) ℝ => Real.sqrt (∑ i, ∑ j, (M i j)^2)

  -- Key step 1: (diagonal d)^(k+1) = diagonal (d^(k+1))
  have h_diag_pow : ∀ k : ℕ, (Matrix.diagonal d)^(k+1) = Matrix.diagonal (d^(k+1)) :=
    fun k => Matrix.diagonal_pow d (k+1)

  -- Key step 2: For diagonal matrices, Frobenius norm simplifies to √(∑ᵢ (dᵢ)²)
  -- because off-diagonal entries are 0
  have h_frob_diag : ∀ e : Fin n → ℝ,
      frobNorm (Matrix.diagonal e) = Real.sqrt (∑ i, (e i)^2) := by
    intro e
    simp only [frobNorm]
    congr 1
    apply Finset.sum_congr rfl
    intro i _
    -- ∑ j, (diagonal e i j)^2 = (e i)^2 because only diagonal entry is nonzero
    have h_sum : ∑ j : Fin n, (Matrix.diagonal e i j)^2 = (e i)^2 := by
      rw [Fintype.sum_eq_single i]
      · simp only [Matrix.diagonal_apply_eq]
      · intro j hij
        have h_off : Matrix.diagonal e i j = 0 := Matrix.diagonal_apply_ne' e hij
        simp only [h_off, sq, mul_zero]
    exact h_sum

  -- Key step 3: Term simplification
  -- (frobNorm ((diagonal d)^(k+1)))^(1/(k+1))
  -- = (frobNorm (diagonal (d^(k+1))))^(1/(k+1))
  -- = (√(∑ᵢ (dᵢ^(k+1))²))^(1/(k+1))
  -- = (√(∑ᵢ |dᵢ|^(2(k+1))))^(1/(k+1))
  -- = (∑ᵢ |dᵢ|^(2(k+1)))^(1/(2(k+1)))

  have h_term : ∀ k : ℕ,
      (frobNorm ((Matrix.diagonal d)^(k+1)))^(1 / (k+1 : ℝ)) =
      (∑ i, |d i|^(2*(k+1)))^(1 / (2*(k+1) : ℝ)) := by
    intro k
    rw [h_diag_pow k, h_frob_diag]
    -- √(∑ᵢ (d^(k+1) i)²)^(1/(k+1)) = (∑ᵢ |dᵢ|^(2(k+1)))^(1/(2(k+1)))

    -- First, show (d i ^ (k+1))^2 = |d i|^(2*(k+1))
    have h_pow_sq : ∀ i, ((d^(k+1)) i)^2 = |d i|^(2*(k+1)) := by
      intro i
      simp only [Pi.pow_apply]
      -- (d i)^(k+1)^2 = (d i)^(2*(k+1))
      have h1 : (d i ^ (k + 1)) ^ 2 = d i ^ (2 * (k + 1)) := by ring
      rw [h1]
      -- (d i)^(2*(k+1)) = |d i|^(2*(k+1)) when exponent is even
      have h_even : Even (2 * (k + 1)) := ⟨k + 1, by ring⟩
      exact (Even.pow_abs h_even (d i)).symm

    -- Rewrite the sum using h_pow_sq
    have h_sum_eq : (∑ i, ((d^(k+1)) i)^2) = ∑ i, |d i|^(2*(k+1)) := by
      apply Finset.sum_congr rfl
      intro i _
      exact h_pow_sq i

    rw [h_sum_eq]

    -- (√x)^(1/(k+1)) = x^(1/(2(k+1))) when x ≥ 0
    have h_sum_nonneg : 0 ≤ ∑ i : Fin n, |d i| ^ (2 * (k + 1)) := by
      apply Finset.sum_nonneg
      intro i _
      apply pow_nonneg
      exact abs_nonneg _
    rw [Real.sqrt_eq_rpow]
    rw [← Real.rpow_mul h_sum_nonneg]
    congr 1
    -- Need: 1 / 2 * (1 / (k + 1)) = 1 / (2 * (k + 1))
    have h_pos : (0 : ℝ) < k + 1 := by positivity
    field_simp [ne_of_gt h_pos]

  -- Key step 4: Prove the infimum equals m
  -- Lower bound: each term ≥ m (since the p-norm is ≥ max)
  -- Upper bound: as k → ∞, the p-norm approaches m

  apply le_antisymm
  · -- Upper bound: iInf ≤ m
    -- Strategy: Show term(k) ≤ n^(1/(2(k+1))) * m, use that this → m as k → ∞
    -- Then iInf ≤ lim term(k) = m

    -- Get the argmax for m
    obtain ⟨i_max, _, hi_max⟩ := Finset.exists_mem_eq_sup' ⟨(0 : Fin n), Finset.mem_univ 0⟩
      (fun i => |d i|)

    have h_m_eq : m = |d i_max| := hi_max
    have h_m_bound : ∀ j, |d j| ≤ m := fun j =>
      Finset.le_sup' (fun i => |d i|) (Finset.mem_univ j)

    have h_m_nonneg : 0 ≤ m := by rw [h_m_eq]; exact abs_nonneg _

    -- Key bound: term(k) ≤ n^(1/(2(k+1))) * m
    have h_term_bound : ∀ k : ℕ,
        (frobNorm ((Matrix.diagonal d)^(k+1)))^(1 / (k+1 : ℝ)) ≤
        (n : ℝ)^(1 / (2*(k+1) : ℝ)) * m := by
      intro k
      rw [h_term k]

      let p : ℝ := 2 * (↑k + 1)
      have h_p_pos : 0 < p := by simp only [p]; positivity

      -- Upper bound on sum: ∑ᵢ |dᵢ|^p ≤ n * m^p
      have h_sum_bound : ∑ i, |d i| ^ (2 * (k + 1)) ≤ (n : ℝ) * m ^ (2 * (k + 1)) := by
        calc ∑ i : Fin n, |d i| ^ (2 * (k + 1))
            ≤ ∑ _ : Fin n, m ^ (2 * (k + 1)) := by
                apply Finset.sum_le_sum
                intro j _
                exact pow_le_pow_left₀ (abs_nonneg _) (h_m_bound j) _
          _ = (Finset.univ : Finset (Fin n)).card • m ^ (2 * (k + 1)) := by
                rw [Finset.sum_const]
          _ = n * m ^ (2 * (k + 1)) := by
                rw [Finset.card_fin, nsmul_eq_mul]

      -- (∑ᵢ |dᵢ|^p)^(1/p) ≤ (n * m^p)^(1/p) = n^(1/p) * m
      have h_exp_nonneg : 0 ≤ 1 / p := div_nonneg (by norm_num) (le_of_lt h_p_pos)
      have h_sum_nonneg : 0 ≤ ∑ i, |d i| ^ (2 * (k + 1)) := by
        apply Finset.sum_nonneg; intro i _; apply pow_nonneg; exact abs_nonneg _
      have h_nm_nonneg : 0 ≤ (n : ℝ) * m ^ (2 * (k + 1)) := by
        apply mul_nonneg; exact Nat.cast_nonneg n; apply pow_nonneg h_m_nonneg

      have h_p_eq : p = ↑(2 * (k + 1)) := by
        simp only [p, Nat.cast_mul, Nat.cast_ofNat, Nat.cast_add, Nat.cast_one]

      -- All coercions handled explicitly
      have h_one_div_p : (1 : ℝ) / p = 1 / (2 * (↑k + 1)) := by simp only [p]

      calc (∑ i, |d i|^(2*(k+1)))^((1:ℝ)/(2*(↑k+1)))
          = (∑ i, |d i|^(2*(k+1)))^(1/p) := by rw [← h_one_div_p]
        _ ≤ ((n : ℝ) * m ^ (2 * (k + 1)))^(1/p) := by
            apply Real.rpow_le_rpow h_sum_nonneg h_sum_bound h_exp_nonneg
        _ = (n : ℝ)^(1/p) * (m ^ (2 * (k + 1)))^(1/p) := by
            rw [Real.mul_rpow (Nat.cast_nonneg n) (pow_nonneg h_m_nonneg _)]
        _ = (n : ℝ)^(1/p) * m^(↑(2 * (k + 1)) * (1/p)) := by
            congr 1
            rw [← Real.rpow_natCast m (2*(k+1)), ← Real.rpow_mul h_m_nonneg]
        _ = (n : ℝ)^(1/p) * m^(1:ℝ) := by
            congr 1
            rw [h_p_eq]
            field_simp [ne_of_gt h_p_pos]
        _ = (n : ℝ)^(1/p) * m := by rw [Real.rpow_one]
        _ = (n : ℝ)^((1:ℝ)/(2*(↑k+1))) * m := by rw [← h_one_div_p]

    -- For iInf ≤ m: use ciInf_le to bound by any term
    -- Since term(0) is in the sequence, iInf ≤ term(0) ≤ n^(1/2) * m
    -- But we need iInf ≤ m

    -- The sequence term(k) → m as k → ∞ (since n^(1/p) → 1)
    -- For the infimum, we use: if term(k) ≥ m for all k and term(k) → m,
    -- then iInf = m

    -- Since we've already shown m ≤ iInf (in the lower bound), and iInf ≤ term(k) for all k,
    -- to complete the proof we need to show the sequence gets arbitrarily close to m.

    -- The p-norm convergence to ∞-norm as p → ∞ is a standard result but
    -- requires topological limit arguments. For now, we use the direct construction:

    -- Use ciInf_le_of_le: iInf ≤ g(i) for any i in the index set
    -- At k = 0: term(0) = (∑ᵢ |dᵢ|²)^(1/2) ≤ √n * m
    -- As k → ∞: term(k) → m

    -- Case split on m: if m = 0, all |d i| = 0, so the sum is 0 and iInf = 0 = m
    -- If m > 0, we need to use convergence of p-norm to max-norm

    by_cases hm : m = 0
    · -- Case m = 0: all |d i| = 0, so iInf = 0 = m
      -- If m = 0, then all |d i| = 0 (since m = max |d i|)
      have h_all_zero : ∀ i, d i = 0 := by
        intro i
        have h_le_m : |d i| ≤ m := h_m_bound i
        rw [hm] at h_le_m
        exact abs_eq_zero.mp (le_antisymm h_le_m (abs_nonneg _))

      -- Therefore term(k) = (∑ᵢ 0^p)^(1/p) = 0 for all k
      have h_term_zero : ∀ k : ℕ, (frobNorm ((Matrix.diagonal d)^(k+1)))^(1 / (k+1 : ℝ)) = 0 := by
        intro k
        rw [h_term k]
        have h_sum_zero : ∑ i : Fin n, |d i| ^ (2 * (k + 1)) = 0 := by
          apply Finset.sum_eq_zero
          intro i _
          rw [h_all_zero i, abs_zero, zero_pow]
          omega
        rw [h_sum_zero]
        have h_exp_ne_zero : 1 / (2 * (↑k + 1) : ℝ) ≠ 0 := by
          apply div_ne_zero
          · norm_num
          · positivity
        exact Real.zero_rpow h_exp_ne_zero

      -- iInf of constant 0 is 0, and m = 0, so iInf ≤ m
      have h_fn_eq : (fun k : ℕ => (frobNorm ((Matrix.diagonal d)^(k+1)))^(1 / (k+1 : ℝ))) =
                     (fun _ : ℕ => (0 : ℝ)) := by
        ext k; exact h_term_zero k
      -- Rewrite the iInf using h_fn_eq
      conv_lhs => rw [h_fn_eq]
      simp only [ciInf_const]
      exact h_m_nonneg

    · -- Case m > 0: we need iInf ≤ m
      -- Since m ≤ term(k) for all k (shown in lower bound) and term(k) → m,
      -- by squeeze theorem iInf = m

      have h_m_pos : 0 < m := by
        apply h_m_nonneg.lt_of_ne'
        intro h_eq
        -- h_eq : m = 0, need to show False using hm : ¬(m = 0)
        exact hm h_eq

      -- The key is showing that for any ε > 0, ∃ K such that term(K) < m + ε
      -- This uses that n^(1/p) → 1 as p → ∞

      -- For concreteness, use the explicit bound:
      -- term(k) = (∑ᵢ |dᵢ|^{2(k+1)})^{1/(2(k+1))}
      --         ≤ (n · m^{2(k+1)})^{1/(2(k+1))}
      --         = n^{1/(2(k+1))} · m

      -- As k → ∞, n^{1/(2(k+1))} → 1, so term(k) → m

      -- To prove iInf ≤ m without using limits explicitly, we use:
      -- m ≤ iInf (from lower bound, proven below)
      -- iInf ≤ term(k) for all k (by definition of iInf)
      -- Combined: m ≤ iInf ≤ term(k) for all k

      -- Since the lower bound gives m ≤ iInf, we have equality if we can show
      -- that the iInf is exactly achieved by the limit of term(k).

      -- The mathematical argument is: n^(1/p) → 1 as p → ∞, so term(k) → m.
      -- Since term(k) ≥ m always, and term(k) approaches m from above,
      -- inf{term(k)} = m.

      -- For a formal proof without topological machinery, we observe:
      -- The sequence term(k) is bounded: m ≤ term(k) ≤ n^(1/2) * m (at k=0)
      -- The infimum exists and lies in [m, n^(1/2) * m]
      -- By the squeeze argument (term(k) → m), the infimum must be m

      -- Alternative: prove directly using Real.iInf properties
      -- We'll show that for any a > m, iInf < a (so iInf ≤ m)

      -- Use: iInf ≤ m ↔ ∀ ε > 0, iInf < m + ε
      --                ↔ ∀ ε > 0, ∃ k, term(k) < m + ε

      -- For large enough k, n^{1/(2(k+1))} is arbitrarily close to 1
      -- so term(k) ≤ n^{1/(2(k+1))} * m is arbitrarily close to m

      -- The formal proof of n^{1/p} → 1 requires showing that
      -- log(n)/p → 0, hence n^{1/p} = exp(log(n)/p) → exp(0) = 1

      -- For the Lean formalization, we document the mathematical correctness
      -- and defer the detailed limit proof to future work if needed.

      -- Since we prove m ≤ iInf in the lower bound, and the terms approach m,
      -- the equality follows. We use antisymmetry with the lower bound.

      -- Key: show that n^(1/(2(k+1))) → 1 as k → ∞
      -- Then term(k) ≤ n^(1/(2(k+1))) * m → m
      -- Since iInf ≤ term(k) for all k and term(k) → m from above, iInf ≤ m

      -- Step 1: The exponent 1/(2*(k+1)) → 0 as k → ∞
      have h_exp_tendsto : Filter.Tendsto (fun k : ℕ => 1 / (2 * (↑k + 1) : ℝ)) Filter.atTop (nhds 0) := by
        have h1 : Filter.Tendsto (fun k : ℕ => (2 * (↑k + 1) : ℝ)) Filter.atTop Filter.atTop := by
          have h_nat : Filter.Tendsto (fun k : ℕ => (k : ℝ)) Filter.atTop Filter.atTop :=
            tendsto_natCast_atTop_atTop
          have h_add : Filter.Tendsto (fun k : ℕ => (↑k + 1 : ℝ)) Filter.atTop Filter.atTop :=
            h_nat.atTop_add tendsto_const_nhds
          exact Filter.Tendsto.const_mul_atTop (by norm_num : (0 : ℝ) < 2) h_add
        have h2 : Filter.Tendsto (fun k : ℕ => (2 * (↑k + 1) : ℝ)⁻¹) Filter.atTop (nhds 0) :=
          tendsto_inv_atTop_zero.comp h1
        simp only [one_div] at h2 ⊢
        exact h2

      -- Step 2: n^y → n^0 = 1 as y → 0, by continuity of a^· at 0 when a > 0
      have h_n_pos : (0 : ℝ) < n := Nat.cast_pos.mpr (NeZero.pos n)
      have h_rpow_tendsto : Filter.Tendsto (fun k : ℕ => (n : ℝ)^(1 / (2 * (↑k + 1) : ℝ)))
          Filter.atTop (nhds 1) := by
        -- Continuity of a^· at 0 (for a > 0)
        have h_cont : ContinuousAt (fun y : ℝ => (n : ℝ) ^ y) 0 :=
          Real.continuousAt_const_rpow (ne_of_gt h_n_pos) (b := 0)
        have h_at_zero : (n : ℝ) ^ (0 : ℝ) = 1 := Real.rpow_zero n
        -- Compose with h_exp_tendsto: ℕ → ℝ with limit 0
        have h_comp := h_cont.tendsto.comp h_exp_tendsto
        rwa [h_at_zero] at h_comp

      -- Step 3: term(k) ≤ n^(1/(2(k+1))) * m → m as k → ∞
      have h_term_tendsto_m : Filter.Tendsto
          (fun k : ℕ => (n : ℝ)^(1 / (2 * (↑k + 1) : ℝ)) * m) Filter.atTop (nhds m) := by
        have h_eq : (1 : ℝ) * m = m := one_mul m
        have h1 := h_rpow_tendsto.mul_const m
        simp only [one_mul] at h1
        exact h1

      -- Step 4: For any ε > 0, eventually term(k) < m + ε
      have h_eventually : ∀ ε > 0, ∃ K : ℕ,
          (frobNorm ((Matrix.diagonal d)^(K+1)))^(1 / (K+1 : ℝ)) < m + ε := by
        intro ε hε
        rw [Metric.tendsto_atTop] at h_term_tendsto_m
        obtain ⟨N, hN⟩ := h_term_tendsto_m ε hε
        use N
        have h_bound := h_term_bound N
        have h_dist := hN N le_rfl
        simp only [Real.dist_eq] at h_dist
        have h_close : (n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) * m < m + ε := by
          have h_ge : m ≤ (n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) * m := by
            have h_one_le : 1 ≤ (n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) := by
              apply Real.one_le_rpow
              · exact Nat.one_le_cast.mpr (NeZero.one_le)
              · apply div_nonneg; norm_num; positivity
            calc m = 1 * m := (one_mul m).symm
                 _ ≤ (n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) * m := by
                   apply mul_le_mul_of_nonneg_right h_one_le (le_of_lt h_m_pos)
          have h_abs_eq : |(n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) * m - m| =
                          (n : ℝ)^(1 / (2 * (↑N + 1) : ℝ)) * m - m := by
            rw [abs_of_nonneg (sub_nonneg.mpr h_ge)]
          rw [h_abs_eq] at h_dist
          linarith
        exact lt_of_le_of_lt h_bound h_close

      -- Step 5: iInf ≤ m follows from: ∀ ε > 0, iInf < m + ε
      rw [le_iff_forall_pos_lt_add]
      intro ε hε
      obtain ⟨K, hK⟩ := h_eventually ε hε
      have h_bdd : BddBelow (Set.range (fun k : ℕ =>
          (frobNorm ((Matrix.diagonal d)^(k+1)))^(1 / (k+1 : ℝ)))) := by
        use 0; intro x hx; obtain ⟨k, rfl⟩ := hx
        apply Real.rpow_nonneg (Real.sqrt_nonneg _)
      exact lt_of_le_of_lt (ciInf_le h_bdd K) hK

  · -- Lower bound: m ≤ iInf
    -- For any k, term(k) ≥ m because the p-norm is always ≥ max element
    apply le_ciInf
    intro k
    rw [h_term k]
    -- Need: m ≤ (∑ᵢ |dᵢ|^(2(k+1)))^(1/(2(k+1)))
    -- Since m = max{|dᵢ|}, we have m^(2(k+1)) ≤ ∑ᵢ |dᵢ|^(2(k+1))
    -- Taking (2(k+1))-th root preserves the inequality

    -- Get the argmax
    obtain ⟨i_max, _, hi_max⟩ := Finset.exists_mem_eq_sup' ⟨(0 : Fin n), Finset.mem_univ 0⟩
      (fun i => |d i|)

    -- m = |d i_max| and |d j| ≤ m for all j
    have h_m_eq : m = |d i_max| := hi_max
    have h_m_bound : ∀ j, |d j| ≤ m := by
      intro j
      show |d j| ≤ Finset.sup' Finset.univ ⟨0, Finset.mem_univ 0⟩ (fun i => |d i|)
      exact Finset.le_sup' (fun i => |d i|) (Finset.mem_univ j)

    -- m^(2(k+1)) ≤ ∑ᵢ |dᵢ|^(2(k+1)) because sum includes m^(2(k+1)) term
    have h_pow_le_sum : m^(2*(k+1)) ≤ ∑ i, |d i|^(2*(k+1)) := by
      rw [h_m_eq]
      have h_nonneg : ∀ j ∈ Finset.univ, (0 : ℝ) ≤ |d j| ^ (2 * (k + 1)) := by
        intro j _; apply pow_nonneg; exact abs_nonneg _
      exact Finset.single_le_sum h_nonneg (Finset.mem_univ i_max)

    -- Take (2(k+1))-th root
    have h_exp_pos : (0 : ℝ) < 2 * (k + 1) := by positivity

    -- m ≥ 0 since it's a sup of absolute values
    have h_m_nonneg : 0 ≤ m := by
      rw [h_m_eq]
      exact abs_nonneg _

    -- Goal: m ≤ (∑ i, |d i|^(2*(k+1)))^(1/(2*(↑k+1)))
    -- Strategy: m^p ≤ sum, take p-th root, get m ≤ sum^(1/p)

    -- Exponent in ℝ: let p = 2 * (k + 1) as a real number
    let p : ℝ := 2 * (↑k + 1)

    have h_p_eq_nat : p = ↑(2 * (k + 1)) := by
      simp only [p, Nat.cast_mul, Nat.cast_ofNat, Nat.cast_add, Nat.cast_one]

    have h_p_pos : 0 < p := by simp only [p]; positivity

    have h_exp_nonneg : (0 : ℝ) ≤ 1 / p := div_nonneg (by norm_num) (le_of_lt h_p_pos)

    -- m^(2*(k+1)) ≤ sum, and since m ≥ 0, taking p-th root gives m ≤ sum^(1/p)
    have h_sum_nonneg : 0 ≤ ∑ i, |d i| ^ (2 * (k + 1)) := by
      apply Finset.sum_nonneg; intro i _; apply pow_nonneg; exact abs_nonneg _

    -- Rewrite h_pow_le_sum in terms of rpow
    have h_pow_le_sum' : m ^ p ≤ ∑ i, |d i| ^ (2 * (k + 1)) := by
      rw [h_p_eq_nat, Real.rpow_natCast]
      exact h_pow_le_sum

    calc m = m ^ (1 : ℝ) := (Real.rpow_one m).symm
       _ = m ^ (p * (1 / p)) := by
           congr 1
           field_simp [ne_of_gt h_p_pos]
       _ = (m ^ p) ^ (1 / p) := by rw [Real.rpow_mul h_m_nonneg]
       _ ≤ (∑ i, |d i|^(2*(k+1)))^(1/p) := by
           apply Real.rpow_le_rpow
           · exact Real.rpow_nonneg h_m_nonneg p
           · exact h_pow_le_sum'
           · exact h_exp_nonneg

/-- Scaling a matrix scales its spectral radius.

    ## Proof Structure

    **Step 1**: (c • A)^k = c^k • A^k (scalar-matrix power commutation)

    This follows from `smul_pow` in Mathlib: for a scalar c and matrix A,
    (c • A)^k = c^k • A^k because scalar multiplication commutes with
    matrix multiplication.

    **Step 2**: Frobenius norm is absolutely homogeneous

    ‖c • M‖_F = √(∑ᵢⱼ |c · Mᵢⱼ|²) = √(|c|² · ∑ᵢⱼ |Mᵢⱼ|²) = |c| · ‖M‖_F

    **Step 3**: Compute the k-th root

    ‖(c • A)^(k+1)‖^{1/(k+1)} = ‖c^(k+1) • A^(k+1)‖^{1/(k+1)}
                               = (|c|^(k+1) · ‖A^(k+1)‖)^{1/(k+1)}
                               = |c| · ‖A^(k+1)‖^{1/(k+1)}

    **Step 4**: Taking infimum

    ρ(c • A) = ⨅ k, ‖(c • A)^(k+1)‖^{1/(k+1)}
             = ⨅ k, |c| · ‖A^(k+1)‖^{1/(k+1)}
             = |c| · ⨅ k, ‖A^(k+1)‖^{1/(k+1)}
             = |c| · ρ(A)

    The last step uses: ⨅ k, (c · f(k)) = c · ⨅ k, f(k) for c ≥ 0.
-/
theorem spectral_radius_smul (c : ℝ) (A : Matrix (Fin n) (Fin n) ℝ) :
    spectralRadius (c • A) = |c| * spectralRadius A := by
  unfold spectralRadius
  simp only

  -- Define the Frobenius norm function for convenience
  let frobNorm := fun M : Matrix (Fin n) (Fin n) ℝ => Real.sqrt (∑ i, ∑ j, (M i j)^2)

  -- Key lemma 1: (c • A)^(k+1) = c^(k+1) • A^(k+1)
  have h_smul_pow : ∀ k : ℕ, (c • A)^(k+1) = c^(k+1) • A^(k+1) := by
    intro k
    induction k with
    | zero => simp only [Nat.zero_add, pow_one]
    | succ k ih =>
      -- (c • A)^(k+2) = (c • A)^(k+1) * (c • A) by pow_succ
      --              = (c^(k+1) • A^(k+1)) * (c • A) by ih
      --              = c^(k+1) • (A^(k+1) * (c • A)) by Matrix.smul_mul
      --              = c^(k+1) • (c • (A^(k+1) * A)) by Matrix.mul_smul
      --              = (c^(k+1) * c) • (A^(k+1) * A) by smul_smul
      --              = c^(k+2) • A^(k+2) by pow_succ
      have step1 : (c • A)^(k+1+1) = (c • A)^(k+1) * (c • A) := pow_succ (c • A) (k+1)
      have step2 : (c • A)^(k+1) * (c • A) = (c^(k+1) • A^(k+1)) * (c • A) := by rw [ih]
      have step3 : (c^(k+1) • A^(k+1)) * (c • A) = c^(k+1) • (A^(k+1) * (c • A)) :=
        Matrix.smul_mul _ _ _
      have step4 : A^(k+1) * (c • A) = c • (A^(k+1) * A) := Matrix.mul_smul _ _ _
      have step5 : c^(k+1) • (c • (A^(k+1) * A)) = (c^(k+1) * c) • (A^(k+1) * A) := smul_smul _ _ _
      have step6 : A^(k+1) * A = A^(k+1+1) := (pow_succ A (k+1)).symm
      have step7 : c^(k+1) * c = c^(k+1+1) := (pow_succ c (k+1)).symm
      calc (c • A)^(k+1+1)
          = (c • A)^(k+1) * (c • A) := step1
        _ = (c^(k+1) • A^(k+1)) * (c • A) := step2
        _ = c^(k+1) • (A^(k+1) * (c • A)) := step3
        _ = c^(k+1) • (c • (A^(k+1) * A)) := by rw [step4]
        _ = (c^(k+1) * c) • (A^(k+1) * A) := step5
        _ = (c^(k+1) * c) • A^(k+1+1) := by rw [step6]
        _ = c^(k+1+1) • A^(k+1+1) := by rw [step7]

  -- Key lemma 2: Frobenius norm of scalar multiple
  have h_frob_smul : ∀ (s : ℝ) (M : Matrix (Fin n) (Fin n) ℝ),
      frobNorm (s • M) = |s| * frobNorm M := by
    intro s M
    simp only [frobNorm]
    -- ∑ i, ∑ j, (s * M i j)² = s² * ∑ i, ∑ j, (M i j)²
    have h_sum_eq : ∑ i, ∑ j, ((s • M) i j)^2 = s^2 * ∑ i, ∑ j, (M i j)^2 := by
      simp only [Matrix.smul_apply, smul_eq_mul]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro i _
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro j _
      ring
    rw [h_sum_eq]
    -- √(s² * x) = |s| * √x
    rw [Real.sqrt_mul (sq_nonneg s)]
    rw [Real.sqrt_sq_eq_abs]

  -- Now combine: the infimum of |c| * f(k) = |c| * infimum of f(k)
  -- This requires |c| ≥ 0 and properties of iInf

  -- First, show that frobNorm((c • A)^(k+1))^(1/(k+1)) = |c| * frobNorm(A^(k+1))^(1/(k+1))
  have h_term_eq : ∀ k : ℕ,
      (frobNorm ((c • A)^(k+1)))^(1 / (k+1 : ℝ)) =
      |c| * (frobNorm (A^(k+1)))^(1 / (k+1 : ℝ)) := by
    intro k
    rw [h_smul_pow k]
    rw [h_frob_smul (c^(k+1)) (A^(k+1))]
    -- |c^(k+1)| = |c|^(k+1)
    rw [abs_pow]
    -- (|c|^(k+1) * frobNorm(A^(k+1)))^(1/(k+1)) = |c| * frobNorm(A^(k+1))^(1/(k+1))
    rw [Real.mul_rpow (pow_nonneg (abs_nonneg c) _) (Real.sqrt_nonneg _)]
    -- |c|^(k+1)^(1/(k+1)) = |c|
    have h_pow_rpow : (|c|^(k+1 : ℕ))^(1 / (k+1 : ℝ)) = |c| := by
      rw [← Real.rpow_natCast |c| (k+1)]
      rw [← Real.rpow_mul (abs_nonneg c)]
      simp only [Nat.cast_add, Nat.cast_one]
      have h_pos : (0 : ℝ) < k + 1 := by positivity
      have h_cancel : (↑k + 1 : ℝ) * (1 / (↑k + 1)) = 1 := by
        field_simp
      rw [h_cancel]
      exact Real.rpow_one |c|
    rw [h_pow_rpow]

  -- Now use that iInf over (|c| * f k) = |c| * iInf over (f k)
  -- when |c| ≥ 0 and the infimum is over a nonempty set

  -- The formal proof requires:
  -- `Real.iInf_mul_of_nonneg` or similar lemma from Mathlib
  -- which states: ⨅ k, (a * f k) = a * ⨅ k, (f k) when a ≥ 0

  -- Rewrite using h_term_eq to factor out |c|
  have h_fn_eq : (fun k : ℕ => (frobNorm ((c • A) ^ (k + 1))) ^ ((1 : ℝ) / (↑k + 1))) =
                 (fun k : ℕ => |c| * (frobNorm (A ^ (k + 1))) ^ ((1 : ℝ) / (↑k + 1))) := by
    ext k
    exact h_term_eq k
  conv_lhs => rw [h_fn_eq]

  -- Now need: ⨅ k, |c| * g(k) = |c| * ⨅ k, g(k)
  -- Case split on whether c = 0
  by_cases hc : c = 0
  · -- If c = 0, then |c| = 0 and both sides are 0
    simp only [hc, abs_zero, zero_mul]
    -- Need to show ⨅ k, (0 : ℝ) = 0, which requires the set to be nonempty
    simp only [ciInf_const]
  · -- If c ≠ 0, then |c| > 0
    have hc_pos : 0 < |c| := abs_pos.mpr hc

    -- For a > 0: ⨅ k, a * f(k) = a * ⨅ k, f(k)
    -- This uses that multiplication by positive constant preserves infimum

    -- The key is that x ↦ |c| * x is an order isomorphism on [0, ∞)
    -- so it preserves infimums

    let g := fun k : ℕ => (frobNorm (A ^ (k + 1))) ^ (1 / (↑k + 1 : ℝ))

    -- Each g(k) ≥ 0 since it's a power of a sqrt
    have hg_nonneg : ∀ k, 0 ≤ g k := by
      intro k
      apply Real.rpow_nonneg
      apply Real.sqrt_nonneg

    -- Use the property: for a > 0, ⨅ x, a * f(x) = a * ⨅ x, f(x)
    -- This is `Real.iInf_mul_left` or similar in Mathlib
    -- Proof: The infimum is characterized by being the greatest lower bound
    -- a * (⨅ f) is a lower bound of {a * f(k)} since a > 0 preserves order
    -- And if b is any lower bound, then b/a ≤ f(k) for all k, so b/a ≤ ⨅ f, hence b ≤ a * ⨅ f

    -- For the formal proof, we need to show:
    -- 1. |c| * ⨅ g is a lower bound of {|c| * g(k)}
    -- 2. Any lower bound b satisfies b ≤ |c| * ⨅ g

    -- Use Real.mul_iInf_of_nonneg: |c| * ⨅ k, g k = ⨅ k, |c| * g k
    -- This requires showing the infimum is bounded below
    symm
    rw [Real.mul_iInf_of_nonneg (le_of_lt hc_pos)]

/-- Spectral normalization: scale matrix to have spectral radius ≤ target. -/
noncomputable def spectralNormalize (A : Matrix (Fin n) (Fin n) ℝ) (target : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  if spectralRadius A = 0 then A
  else (target / spectralRadius A) • A

theorem spectralNormalize_radius (A : Matrix (Fin n) (Fin n) ℝ) (target : ℝ)
    (ht : 0 < target) (hA : spectralRadius A ≠ 0) :
    spectralRadius (spectralNormalize A target) = target := by
  -- Unfold the definition of spectralNormalize
  unfold spectralNormalize
  simp only [hA, ↓reduceIte]
  -- spectralNormalize A target = (target / ρ(A)) • A
  -- By spectral_radius_smul: ρ((target / ρ(A)) • A) = |target / ρ(A)| * ρ(A)
  -- Since target > 0 and ρ(A) > 0 (nonzero), target / ρ(A) > 0
  -- So |target / ρ(A)| = target / ρ(A)
  -- Therefore: ρ(spectralNormalize A target) = (target / ρ(A)) * ρ(A) = target

  -- Once spectral_radius_smul is proved, this follows:
  have h_smul := spectral_radius_smul (target / spectralRadius A) A
  rw [h_smul]
  -- Need: spectralRadius A > 0 (since it's ≠ 0 and ≥ 0 by definition as infimum of nonneg terms)
  have h_rho_nonneg : 0 ≤ spectralRadius A := by
    unfold spectralRadius
    apply Real.iInf_nonneg
    intro k
    apply Real.rpow_nonneg
    apply Real.sqrt_nonneg
  have h_rho_pos : 0 < spectralRadius A := by
    rcases h_rho_nonneg.eq_or_lt with h | h
    · exact absurd h hA.symm
    · exact h
  have h_div_pos : 0 < target / spectralRadius A := div_pos ht h_rho_pos
  rw [abs_of_pos h_div_pos]
  field_simp

end SpectralRadius
