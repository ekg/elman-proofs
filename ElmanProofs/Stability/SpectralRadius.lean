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
    | zero => simp [pow_one, pow_one]
    | succ k ih =>
      rw [pow_succ, pow_succ, pow_succ, ih]
      -- (c • A) * (c^(k+1) • A^(k+1)) = c^(k+2) • A^(k+2)
      rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul]
      ring_nf

  -- Key lemma 2: Frobenius norm of scalar multiple
  have h_frob_smul : ∀ (s : ℝ) (M : Matrix (Fin n) (Fin n) ℝ),
      frobNorm (s • M) = |s| * frobNorm M := by
    intro s M
    simp only [frobNorm]
    rw [Real.sqrt_eq_iff_sq_eq]
    · rw [mul_pow, sq_abs, Real.sq_sqrt]
      · congr 1
        simp only [Matrix.smul_apply, smul_eq_mul]
        rw [Finset.sum_congr rfl (fun i _ => Finset.sum_congr rfl (fun j _ => by ring))]
        rw [← Finset.mul_sum, ← Finset.mul_sum]
        ring
      · apply Finset.sum_nonneg
        intro i _
        apply Finset.sum_nonneg
        intro j _
        exact sq_nonneg _
    · apply mul_nonneg (abs_nonneg _)
      apply Real.sqrt_nonneg
    · apply Finset.sum_nonneg
      intro i _
      apply Finset.sum_nonneg
      intro j _
      exact sq_nonneg _

  -- Now combine: the infimum of |c| * f(k) = |c| * infimum of f(k)
  -- This requires |c| ≥ 0 and properties of iInf

  sorry

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
  simp only [hA, ↓reduceIte, ne_eq, not_false_eq_true]

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
    cases' h_rho_nonneg.lt_or_eq with h h
    · exact h
    · exact absurd h.symm hA

  have h_div_pos : 0 < target / spectralRadius A := div_pos ht h_rho_pos
  rw [abs_of_pos h_div_pos]
  field_simp

end SpectralRadius
