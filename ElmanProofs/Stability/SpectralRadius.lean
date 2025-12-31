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

/-- Spectral radius is bounded by operator norm. -/
theorem spectral_radius_le_opNorm (A : Matrix (Fin n) (Fin n) ℝ) (norm_A : ℝ) :
    spectralRadius A ≤ norm_A := by
  sorry

/-- Powers of spectrally stable matrices converge to zero. -/
theorem powers_tendsto_zero (A : Matrix (Fin n) (Fin n) ℝ)
    (hA : IsSpectrallyStable A) :
    ∀ ε > 0, ∃ N, ∀ k ≥ N, ∀ i j, |(A^k) i j| < ε := by
  sorry

/-- Spectral radius of diagonal matrix is max of absolute diagonal entries. -/
theorem diagonal_spectral_radius (d : Fin n → ℝ) :
    spectralRadius (Matrix.diagonal d) = Finset.sup' Finset.univ ⟨0, Finset.mem_univ 0⟩
      (fun i => |d i|) := by
  sorry

/-- Scaling a matrix scales its spectral radius. -/
theorem spectral_radius_smul (c : ℝ) (A : Matrix (Fin n) (Fin n) ℝ) :
    spectralRadius (c • A) = |c| * spectralRadius A := by
  sorry

/-- Spectral normalization: scale matrix to have spectral radius ≤ target. -/
noncomputable def spectralNormalize (A : Matrix (Fin n) (Fin n) ℝ) (target : ℝ) :
    Matrix (Fin n) (Fin n) ℝ :=
  if spectralRadius A = 0 then A
  else (target / spectralRadius A) • A

theorem spectralNormalize_radius (A : Matrix (Fin n) (Fin n) ℝ) (target : ℝ)
    (ht : 0 < target) (hA : spectralRadius A ≠ 0) :
    spectralRadius (spectralNormalize A target) = target := by
  sorry

end SpectralRadius
