/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Topology.MetricSpace.Contracting

/-!
# Contraction Mappings for RNN Analysis

This file connects Mathlib's contraction mapping theory to RNN dynamics.
The key insight is that RNNs with spectral radius < 1 are contractions.

## Main Results

* `contraction_unique_fixed_point`: Contractions have unique fixed points (Banach)
* `contraction_comp`: Composition of contractions is a contraction
* `lipschitz_comp_contraction`: Lipschitz ∘ contraction is contraction

## Application to RNNs

For an RNN with recurrence h_{t+1} = σ(W_x x_t + R_h h_t + b):
- If ‖R_h‖ < 1 and σ is 1-Lipschitz, the hidden state map is a contraction
- This guarantees convergence to a unique attractor for fixed input

-/

namespace Contraction

open Metric Function

variable {X : Type*} [MetricSpace X]

/-- A map is a contraction if it reduces distances by a factor K ∈ [0, 1). -/
structure IsContraction (f : X → X) (K : ℝ) : Prop where
  /-- K is non-negative -/
  K_nonneg : 0 ≤ K
  /-- K is less than 1 -/
  K_lt_one : K < 1
  /-- f is K-Lipschitz -/
  lipschitz : ∀ x y, dist (f x) (f y) ≤ K * dist x y

/-- Contractions are Lipschitz with constant < 1. -/
theorem contraction_lipschitz {f : X → X} {K : ℝ} (hf : IsContraction f K) :
    LipschitzWith ⟨K, hf.K_nonneg⟩ f := by
  have h := LipschitzWith.of_dist_le' (fun x y => hf.lipschitz x y)
  convert h using 1
  ext
  simp [Real.toNNReal_of_nonneg hf.K_nonneg]

/-- Composition of contractions is a contraction. -/
theorem contraction_comp {f g : X → X} {Kf Kg : ℝ}
    (hf : IsContraction f Kf) (hg : IsContraction g Kg) :
    IsContraction (f ∘ g) (Kf * Kg) := by
  constructor
  · exact mul_nonneg hf.K_nonneg hg.K_nonneg
  · rcases hf.K_nonneg.lt_or_eq with hpos | hzero
    · have h1 : Kf * Kg < Kf * 1 := mul_lt_mul_of_pos_left hg.K_lt_one hpos
      linarith [hf.K_lt_one]
    · subst hzero; simp
  · intro x y
    calc dist ((f ∘ g) x) ((f ∘ g) y)
        = dist (f (g x)) (f (g y)) := rfl
      _ ≤ Kf * dist (g x) (g y) := hf.lipschitz (g x) (g y)
      _ ≤ Kf * (Kg * dist x y) := mul_le_mul_of_nonneg_left (hg.lipschitz x y) hf.K_nonneg
      _ = (Kf * Kg) * dist x y := by ring

/-- Lipschitz composition preserves contraction. -/
theorem lipschitz_comp_contraction {f : X → X} {σ : X → X} {K L : ℝ}
    (hf : IsContraction f K) (hσ_nonneg : 0 ≤ L)
    (hσ : ∀ x y, dist (σ x) (σ y) ≤ L * dist x y)
    (hKL : K * L < 1) :
    IsContraction (σ ∘ f) (K * L) := by
  constructor
  · exact mul_nonneg hf.K_nonneg hσ_nonneg
  · exact hKL
  · intro x y
    calc dist ((σ ∘ f) x) ((σ ∘ f) y)
        = dist (σ (f x)) (σ (f y)) := rfl
      _ ≤ L * dist (f x) (f y) := hσ (f x) (f y)
      _ ≤ L * (K * dist x y) := mul_le_mul_of_nonneg_left (hf.lipschitz x y) hσ_nonneg
      _ = (K * L) * dist x y := by ring

/-- Convert our IsContraction to Mathlib's ContractingWith. -/
theorem isContraction_to_contractingWith {f : X → X} {K : ℝ} (hf : IsContraction f K) :
    ContractingWith ⟨K, hf.K_nonneg⟩ f := by
  constructor
  · exact mod_cast hf.K_lt_one
  · exact contraction_lipschitz hf

/-- For complete spaces, contractions have unique fixed points.
    This is the Banach fixed-point theorem. -/
theorem contraction_unique_fixed_point [CompleteSpace X] [Nonempty X] {f : X → X} {K : ℝ}
    (hf : IsContraction f K) :
    ∃! x, f x = x := by
  have hc := isContraction_to_contractingWith hf
  obtain ⟨x₀⟩ : Nonempty X := inferInstance
  have hfin : edist x₀ (f x₀) ≠ ⊤ := edist_ne_top x₀ (f x₀)
  -- Use Mathlib's ContractingWith.exists_fixedPoint
  obtain ⟨x_star, hfixed, hconv, hlim⟩ := hc.exists_fixedPoint x₀ hfin
  refine ⟨x_star, hfixed, ?_⟩
  intro y hy
  -- Use contractivity: dist(x_star, y) = dist(f x_star, f y) ≤ K * dist(x_star, y)
  -- So (1 - K) * dist(x_star, y) ≤ 0, but 1 - K > 0, so dist = 0
  have hdist : dist x_star y ≤ K * dist x_star y := by
    calc dist x_star y = dist (f x_star) (f y) := by rw [hfixed, hy]
      _ ≤ K * dist x_star y := hf.lipschitz x_star y
  have hK : 0 < 1 - K := by linarith [hf.K_lt_one]
  have h : (1 - K) * dist x_star y ≤ 0 := by linarith
  have hdist_nonneg : 0 ≤ dist x_star y := dist_nonneg
  have hdist_zero : dist x_star y = 0 := by nlinarith
  rw [dist_comm] at hdist_zero
  exact dist_eq_zero.mp hdist_zero

/-- Convergence rate: distance to fixed point decreases geometrically. -/
theorem contraction_convergence_rate [CompleteSpace X] [Nonempty X] {f : X → X} {K : ℝ}
    (hf : IsContraction f K) (x₀ : X) :
    let x_star := (contraction_unique_fixed_point hf).choose
    ∀ n : ℕ, dist (f^[n] x₀) x_star ≤ K^n * dist x₀ x_star := by
  intro x_star n
  have hfix : f x_star = x_star := (contraction_unique_fixed_point hf).choose_spec.1
  induction n with
  | zero => simp
  | succ n ih =>
    have key : f^[n + 1] x₀ = f (f^[n] x₀) := iterate_succ_apply' f n x₀
    calc dist (f^[n + 1] x₀) x_star
        = dist (f (f^[n] x₀)) x_star := by rw [key]
      _ = dist (f (f^[n] x₀)) (f x_star) := by rw [hfix]
      _ ≤ K * dist (f^[n] x₀) x_star := hf.lipschitz _ _
      _ ≤ K * (K^n * dist x₀ x_star) := mul_le_mul_of_nonneg_left ih hf.K_nonneg
      _ = K^(n + 1) * dist x₀ x_star := by ring

end Contraction
