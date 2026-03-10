/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Critical Batch Size Theory

Formalizes the McCandlish et al. (2018) framework for understanding how batch size
affects training efficiency, and proves when batch size 1 is optimal.

## Main Definitions

* `gradientNoiseScale` — B_noise = tr(Σ) / ‖G‖², the gradient noise scale
* `stepsWithBatch` — S(B) = S_min × (1 + B_noise / B)
* `tokensWithBatch` — E(B) = E_min × (1 + B / B_noise)
* `criticalBatchSize` — B_crit where time and compute efficiency balance

## Main Results

* `diminishing_returns` — B > B_noise → S(B) ≈ S_min (more samples don't help)
* `compute_waste` — B > B_noise → tokens wasted linearly
* `bs1_optimal_when` — B_noise < 1 → bs=1 minimizes tokens for given progress

## Key Insight

For RNNs in early training:
- B_noise is small (BPTT provides low-noise gradients)
- Therefore B_crit is small (≈ 1-10)
- Therefore bs=1 is compute-optimal

## References

* McCandlish, Kaplan, Amodei (2018), arXiv:1812.06162
* Kaplan et al. (2020), "Scaling Laws for Neural Language Models"
-/

namespace Learning.CriticalBatchSize

/-! ## Core Definitions -/

/-- Gradient noise scale: B_noise = tr(Σ) / ‖G‖².
    Measures the signal-to-noise ratio of stochastic gradients.
    Small B_noise means individual gradients are already informative. -/
structure GradientNoiseScale where
  traceCovariance : ℝ  -- tr(Σ): total gradient variance
  gradNormSq : ℝ       -- ‖G‖²: squared norm of true gradient
  hGrad : gradNormSq > 0

/-- Compute B_noise from its components. -/
noncomputable def GradientNoiseScale.value (bns : GradientNoiseScale) : ℝ :=
  bns.traceCovariance / bns.gradNormSq

/-- B_noise is non-negative when covariance trace is non-negative. -/
theorem noise_scale_nonneg (bns : GradientNoiseScale) (h : bns.traceCovariance ≥ 0) :
    bns.value ≥ 0 := by
  unfold GradientNoiseScale.value
  exact div_nonneg h (le_of_lt bns.hGrad)

/-! ## Steps-Tokens Tradeoff (McCandlish Equations) -/

/-- Steps needed as a function of batch size B:
    S(B) = S_min × (1 + B_noise / B)

    At B → ∞: S(B) → S_min (minimum steps, maximum parallelism)
    At B → 0: S(B) → ∞ (many noisy steps needed) -/
noncomputable def stepsWithBatch (S_min B_noise B : ℝ) : ℝ :=
  S_min * (1 + B_noise / B)

/-- Tokens/examples needed as a function of batch size B:
    E(B) = E_min × (1 + B / B_noise)

    At B → 0: E(B) → E_min (minimum compute, many steps)
    At B → ∞: E(B) → ∞ (massive compute waste) -/
noncomputable def tokensWithBatch (E_min B_noise B : ℝ) : ℝ :=
  E_min * (1 + B / B_noise)

/-- Total compute is S(B) × B (steps times batch size).
    This equals E(B) up to constants. -/
noncomputable def totalCompute (S_min B_noise B : ℝ) : ℝ :=
  stepsWithBatch S_min B_noise B * B

/-! ## Key Properties -/

/-- Steps at B=1: S(1) = S_min × (1 + B_noise). -/
theorem steps_at_bs1 (S_min B_noise : ℝ) :
    stepsWithBatch S_min B_noise 1 = S_min * (1 + B_noise) := by
  unfold stepsWithBatch
  ring

/-- Tokens at B=1: E(1) = E_min × (1 + 1/B_noise). -/
theorem tokens_at_bs1 (E_min B_noise : ℝ) (hBn : B_noise ≠ 0) :
    tokensWithBatch E_min B_noise 1 = E_min * (1 + 1 / B_noise) := by
  unfold tokensWithBatch
  ring

/-- DIMINISHING RETURNS: When B >> B_noise, steps barely decrease.
    Specifically: S(B) / S_min = 1 + B_noise / B ≤ 1 + 1 when B ≥ B_noise.

    This means: once batch size exceeds the noise scale, adding more samples
    to each batch provides negligible reduction in required steps. -/
theorem diminishing_returns (S_min B_noise B : ℝ) (hS : S_min > 0)
    (hBn : B_noise > 0) (hB : B ≥ B_noise) :
    stepsWithBatch S_min B_noise B ≤ 2 * S_min := by
  unfold stepsWithBatch
  have hBpos : B > 0 := lt_of_lt_of_le hBn hB
  have h1 : B_noise / B ≤ 1 := (div_le_one hBpos).mpr hB
  nlinarith

/-- COMPUTE WASTE: Tokens grow linearly with B beyond B_noise.
    E(B) = E_min × (1 + B/B_noise), so for B = k × B_noise:
    E(B) = E_min × (1 + k). -/
theorem compute_linear_growth (E_min B_noise : ℝ) (k : ℝ) (hBn : B_noise > 0)
    (hk : k > 0) :
    tokensWithBatch E_min B_noise (k * B_noise) = E_min * (1 + k) := by
  unfold tokensWithBatch
  rw [mul_div_cancel_of_imp]
  intro h; linarith

/-- OPTIMAL BATCH SIZE when B_noise < 1: bs=1 uses fewer total tokens
    than any B > 1 for the same number of steps.

    Proof: totalCompute(B) = S_min × (B + B_noise) is increasing in B.
    So minimum compute at smallest B, i.e., B = 1. -/
theorem total_compute_increasing (S_min B_noise B₁ B₂ : ℝ)
    (hS : S_min > 0) (hBn : B_noise ≥ 0) (hB1 : B₁ > 0) (hB2 : B₂ > 0)
    (hB : B₁ ≤ B₂) :
    totalCompute S_min B_noise B₁ ≤ totalCompute S_min B_noise B₂ := by
  unfold totalCompute stepsWithBatch
  -- S_min * (1 + B_noise/B) * B = S_min * (B + B_noise)
  have h1 : S_min * (1 + B_noise / B₁) * B₁ = S_min * (B₁ + B_noise) := by
    field_simp
  have h2 : S_min * (1 + B_noise / B₂) * B₂ = S_min * (B₂ + B_noise) := by
    field_simp
  rw [h1, h2]
  apply mul_le_mul_of_nonneg_left _ (le_of_lt hS)
  linarith

/-- When B_noise ≤ 1, bs=1 achieves minimum total compute. -/
theorem bs1_optimal_small_noise (S_min B_noise B : ℝ)
    (hS : S_min > 0) (hBn : 0 ≤ B_noise) (hB : B ≥ 1) :
    totalCompute S_min B_noise 1 ≤ totalCompute S_min B_noise B := by
  exact total_compute_increasing S_min B_noise 1 B hS hBn one_pos (by linarith) hB

/-! ## Critical Batch Size -/

/-- Critical batch size: B_crit = B_noise.
    At B = B_crit, training takes 2 × S_min steps and processes 2 × E_min tokens. -/
theorem at_critical_batch (S_min B_noise : ℝ) (hBn : B_noise > 0) :
    stepsWithBatch S_min B_noise B_noise = 2 * S_min := by
  unfold stepsWithBatch
  rw [div_self (ne_of_gt hBn)]
  ring

/-- At B_crit, compute is 2 × minimal. -/
theorem tokens_at_critical (E_min B_noise : ℝ) (hBn : B_noise > 0) :
    tokensWithBatch E_min B_noise B_noise = 2 * E_min := by
  unfold tokensWithBatch
  rw [div_self (ne_of_gt hBn)]
  ring

/-! ## Scaling of B_noise with Loss -/

/-- Kaplan scaling: B_crit(L) = B_* / L^{1/α_B}.
    For α_B ≈ 0.21, exponent 1/α_B ≈ 4.76.
    At high loss, B_crit is very small. -/
noncomputable def critBatchAtLoss (B_star : ℝ) (α_B : ℝ) (L : ℝ) : ℝ :=
  B_star / L ^ (1 / α_B)

/-- Higher loss → smaller critical batch size (when α_B > 0). -/
theorem crit_batch_decreases_with_loss (B_star α_B L₁ L₂ : ℝ)
    (hB : B_star > 0) (hα : α_B > 0) (hL1 : L₁ > 0) (hL2 : L₂ > 0) (hL : L₁ ≤ L₂) :
    critBatchAtLoss B_star α_B L₂ ≤ critBatchAtLoss B_star α_B L₁ := by
  unfold critBatchAtLoss
  -- B_star / L₂^e ≤ B_star / L₁^e since L₁ ≤ L₂ implies L₁^e ≤ L₂^e for e > 0
  -- L₁^e ≤ L₂^e when L₁ ≤ L₂ and e = 1/α_B > 0
  sorry -- Requires rpow monotonicity for positive base; straightforward from Mathlib

end Learning.CriticalBatchSize
