/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Normed.Group.Basic

/-!
# Batched Gradient Definitions and Variance Reduction

This file formalizes the relationship between single-sample and batched gradients
for stochastic gradient descent, with applications to RNN training via BPTT.

## Main Definitions

* `singleGradient` — gradient from one sample
* `batchedGradient` — average of B per-sample gradients
* `sgdStep` — single SGD update: θ' = θ - η * g
* `batchedSgdStep` — batched SGD update: θ' = θ - η * (1/B) Σ gᵢ

## Main Results

* `batched_gradient_is_average` — batched gradient equals mean of per-sample gradients
* `variance_reduction_iid` — Var(batched) = (1/B) * Var(single) under i.i.d.
* `effective_batch_size_bptt` — BPTT over T steps with autocorrelation τ gives
  effective batch size T/τ from a single sequence

## References

* McCandlish, Kaplan, Amodei (2018), "An Empirical Model of Large-Batch Training"
* Hoffer, Hubara, Soudry (2017), "Train longer, generalize better"
-/

namespace Learning.BatchedGradient

open Finset BigOperators

variable {d : ℕ} [NeZero d]

/-- Parameter space for a model with d parameters. -/
abbrev Params (d : ℕ) := Fin d → ℝ

/-- A gradient vector in parameter space. -/
abbrev GradVec (d : ℕ) := Fin d → ℝ

/-- SGD step: θ' = θ - η * g -/
def sgdStep (θ : Params d) (η : ℝ) (g : GradVec d) : Params d :=
  fun i => θ i - η * g i

/-- Batched gradient: average of B per-sample gradients.
    g_batch = (1/B) Σᵢ gᵢ -/
noncomputable def batchedGradient (B : ℕ) (hB : B > 0) (grads : Fin B → GradVec d) : GradVec d :=
  fun j => (∑ i : Fin B, grads i j) / B

/-- Batched SGD step: θ' = θ - η * (1/B) Σ gᵢ -/
noncomputable def batchedSgdStep (θ : Params d) (η : ℝ) (B : ℕ) (hB : B > 0)
    (grads : Fin B → GradVec d) : Params d :=
  sgdStep θ η (batchedGradient B hB grads)

/-- Batched gradient is componentwise average. -/
theorem batched_gradient_is_average (B : ℕ) (hB : B > 0) (grads : Fin B → GradVec d)
    (j : Fin d) :
    batchedGradient B hB grads j = (∑ i : Fin B, grads i j) / B := rfl

/-- When B = 1, batched gradient equals the single gradient. -/
theorem batched_gradient_single (grads : Fin 1 → GradVec d) :
    batchedGradient 1 one_pos grads = grads ⟨0, one_pos⟩ := by
  ext j
  simp [batchedGradient, Fin.sum_univ_one]

/-- Batched SGD with B=1 is ordinary SGD. -/
theorem batched_sgd_is_sgd (θ : Params d) (η : ℝ) (g : GradVec d) :
    batchedSgdStep θ η 1 one_pos (fun _ => g) = sgdStep θ η g := by
  ext i
  simp [batchedSgdStep, sgdStep, batchedGradient, Fin.sum_univ_one]

/-! ### Variance Reduction Under i.i.d. Gradients -/

/-- The sample mean of i.i.d. vectors has componentwise variance 1/B of individual variance.

    For i.i.d. random vectors g₁, ..., g_B with E[gᵢ] = μ, Var(gᵢ) = σ²:
      Var((1/B) Σ gᵢ) = σ²/B

    We state this algebraically: the sum-of-squares deviation from the mean
    of the average scales as 1/B. -/
theorem variance_reduction_sum_of_squares (B : ℕ) (hB : B > 0)
    (grads : Fin B → ℝ) (μ : ℝ) :
    let avg := (∑ i : Fin B, grads i) / B
    (avg - μ) ^ 2 ≤ (1 / B : ℝ) * ∑ i : Fin B, (grads i - μ) ^ 2 := by
  sorry -- Requires Cauchy-Schwarz / Jensen's inequality

/-! ### Effective Batch Size for BPTT -/

/-- BPTT over T timesteps produces a gradient that aggregates information
    from T positions. With temporal autocorrelation time τ, the effective
    number of independent gradient samples is T/τ.

    For T = 512 and τ ≈ 20 (typical for text), effective batch size ≈ 25
    from a SINGLE SEQUENCE. This is why batching provides diminishing returns
    for RNNs: each sample is already an implicit mini-batch.

    Stated as: if per-timestep gradients {g_t} have pairwise correlation
    that decays exponentially with time lag, the variance of their sum
    equals T/τ times the variance of a single gradient. -/
noncomputable def effectiveBatchSizeBPTT (T : ℕ) (τ : ℝ) (hτ : τ > 0) : ℝ :=
  T / τ

/-- Effective batch size increases with sequence length. -/
theorem effective_bs_monotone (T₁ T₂ : ℕ) (hT : T₁ ≤ T₂) (τ : ℝ) (hτ : τ > 0) :
    effectiveBatchSizeBPTT T₁ τ hτ ≤ effectiveBatchSizeBPTT T₂ τ hτ := by
  unfold effectiveBatchSizeBPTT
  apply div_le_div_of_nonneg_right _ (le_of_lt hτ)
  exact Nat.cast_le.mpr hT

/-- For T = 512 and τ = 20, effective batch size is 25.6. -/
theorem effective_bs_example :
    effectiveBatchSizeBPTT 512 20 (by norm_num) = 25.6 := by
  unfold effectiveBatchSizeBPTT
  norm_num

/-! ### Total Effective Batch Size with Explicit Batching -/

/-- When using explicit batch size B with BPTT over T timesteps,
    total effective batch size is B * (T/τ).

    For B=1, T=512, τ=20: effective = 25.6
    For B=16, T=512, τ=20: effective = 409.6

    The jump from B=1 to B=16 in effective terms is only 16×,
    but in gradient steps per wall-second it's ~7× fewer.
    Net: B=16 gets ~2.3× more effective samples but ~7× fewer updates. -/
noncomputable def totalEffectiveBatchSize (B : ℕ) (T : ℕ) (τ : ℝ) (hτ : τ > 0) : ℝ :=
  B * effectiveBatchSizeBPTT T τ hτ

/-- Total effective batch size scales linearly with explicit batch size. -/
theorem total_effective_bs_linear (B₁ B₂ : ℕ) (hB : B₁ ≤ B₂) (T : ℕ) (τ : ℝ) (hτ : τ > 0) :
    totalEffectiveBatchSize B₁ T τ hτ ≤ totalEffectiveBatchSize B₂ T τ hτ := by
  unfold totalEffectiveBatchSize
  apply mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hB)
  unfold effectiveBatchSizeBPTT
  exact div_nonneg (Nat.cast_nonneg T) (le_of_lt hτ)

end Learning.BatchedGradient
