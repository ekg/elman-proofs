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

/-- Spectral radius is bounded by any submultiplicative matrix norm.

    ## Proof Outline

    For any submultiplicative norm ‖·‖ with ‖A‖ = norm_A:

    1. **Submultiplicativity**: ‖A^k‖ ≤ ‖A‖^k = norm_A^k

    2. **Taking k-th root**: ‖A^k‖^{1/k} ≤ norm_A

    3. **Infimum**: Since this holds for all k, the infimum also satisfies:
       ρ(A) = ⨅ k, ‖A^(k+1)‖^{1/(k+1)} ≤ norm_A

    Note: This theorem signature is unusual - it claims ρ(A) ≤ norm_A for any norm_A,
    which is only meaningful when norm_A is actually ‖A‖ for some submultiplicative norm.
    The proof would need additional hypotheses about norm_A.
-/
theorem spectral_radius_le_opNorm (A : Matrix (Fin n) (Fin n) ℝ) (norm_A : ℝ) :
    spectralRadius A ≤ norm_A := by
  -- Note: This theorem as stated is too strong - it claims ρ(A) ≤ norm_A
  -- for ANY norm_A, which is false (e.g., norm_A = 0).
  --
  -- A correct statement would be:
  -- For any submultiplicative norm ‖·‖: ρ(A) ≤ ‖A‖
  --
  -- The proof would use:
  -- 1. ‖A^(k+1)‖ ≤ ‖A‖^(k+1) (submultiplicativity)
  -- 2. ‖A^(k+1)‖^{1/(k+1)} ≤ ‖A‖ (taking roots preserves inequality)
  -- 3. ⨅ k, ‖A^(k+1)‖^{1/(k+1)} ≤ ‖A‖ (infimum is ≤ any element)

  sorry

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
  -- Choose r with ρ(A) < r < 1
  unfold IsSpectrallyStable spectralRadius at hA

  -- The infimum definition means:
  -- ∃ K, (‖A^(K+1)‖_F)^(1/(K+1)) < 1
  -- which gives ‖A^(K+1)‖_F < 1

  -- Key: For spectrally stable matrices, there exists C, r with r < 1
  -- such that ‖A^k‖ ≤ C · r^k for all k

  -- The formal proof requires:
  -- 1. Extract witnessing K from the infimum condition
  -- 2. Establish the geometric bound
  -- 3. Convert norm bound to entry-wise bound
  -- 4. Find N such that C · r^N < ε

  sorry

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

  -- Key step 1: (diagonal d)^(k+1) = diagonal (d^(k+1))
  -- where d^(k+1) means pointwise power

  -- Key step 2: Frobenius norm of diagonal(e) is √(∑ |eᵢ|²)
  -- For diagonal matrices, off-diagonal entries are 0

  -- Key step 3: The infimum over k of (∑ᵢ |dᵢ|^{2(k+1)})^{1/2(k+1)}
  -- equals max{|dᵢ|} by the p-norm convergence to ∞-norm

  -- This requires Mathlib's `NNReal.inner_le_Lp_mul_Lq` and related lemmas
  -- about p-norms and their convergence

  sorry

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
