/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Analysis.Normed.Field.Basic
import ElmanProofs.Dynamics.Contraction

/-!
# RNN Recurrence Dynamics

This file formalizes the dynamics of recurrent neural networks:
  h_{t+1} = σ(W_x · x_t + R_h · h_t + b)

## Main Definitions

* `RNNCell`: A recurrent cell with input weight, recurrence weight, and activation
* `RNNCell.step`: The single-step evolution of hidden state
* `RNNCell.isContraction`: Conditions under which the RNN is a contraction

## Main Theorems

* `RNNCell.spectral_contraction`: ‖R_h‖ < 1 and Lipschitz σ implies contraction
* `RNNCell.unique_attractor`: Contraction RNNs have unique fixed point for fixed input

-/

namespace RNN

open scoped Matrix
open Contraction

variable {n m : ℕ} -- n = hidden dim, m = input dim

/-- An RNN cell with hidden dimension n and input dimension m. -/
structure Cell (n m : ℕ) where
  /-- Input-to-hidden weight matrix -/
  W_x : Matrix (Fin n) (Fin m) ℝ
  /-- Hidden-to-hidden recurrence matrix -/
  R_h : Matrix (Fin n) (Fin n) ℝ
  /-- Bias vector -/
  b : Fin n → ℝ
  /-- Activation function -/
  act : (Fin n → ℝ) → (Fin n → ℝ)
  /-- Lipschitz constant of activation -/
  act_lipschitz : ℝ

/-- The hidden state evolution for a fixed input x. -/
def Cell.stepWithInput (cell : Cell n m) (x : Fin m → ℝ) (h : Fin n → ℝ) : Fin n → ℝ :=
  cell.act (fun i => (cell.W_x *ᵥ x) i + (cell.R_h *ᵥ h) i + cell.b i)

/-- The hidden state evolution ignoring input (for attractor analysis). -/
def Cell.autonomousStep (cell : Cell n m) (h : Fin n → ℝ) : Fin n → ℝ :=
  cell.act (fun i => (cell.R_h *ᵥ h) i + cell.b i)

/-- An RNN cell is contracting if ‖R_h‖ · L_σ < 1. -/
def Cell.isContracting (cell : Cell n m) (R_h_norm : ℝ) : Prop :=
  R_h_norm * cell.act_lipschitz < 1

/-- Main theorem: Contracting RNNs have unique attractors.
    If ‖R_h‖ · L_σ < 1, then for any fixed input, the hidden state
    converges to a unique fixed point. -/
theorem Cell.unique_attractor (cell : Cell n m) (x : Fin m → ℝ)
    (R_h_norm : ℝ)
    (hContr : cell.isContracting R_h_norm)
    (hσ_pos : 0 ≤ cell.act_lipschitz)
    (hR_pos : 0 ≤ R_h_norm)
    -- Key hypotheses: the activation is Lipschitz and R_h has bounded operator norm
    (h_act_lipschitz : ∀ u v : Fin n → ℝ,
      dist (cell.act u) (cell.act v) ≤ cell.act_lipschitz * dist u v)
    (h_R_h_norm : ∀ h₁ h₂ : Fin n → ℝ,
      dist (cell.R_h *ᵥ h₁) (cell.R_h *ᵥ h₂) ≤ R_h_norm * dist h₁ h₂) :
    ∃! h_star : Fin n → ℝ, cell.stepWithInput x h_star = h_star := by
  -- Define the step function for fixed input x
  let f := fun h => cell.stepWithInput x h
  -- We'll prove that f is a contraction with constant R_h_norm * act_lipschitz
  -- Then apply the unique fixed point theorem.
  -- First, we need to show f is a contraction
  -- This requires showing: dist(f(h₁), f(h₂)) ≤ K * dist(h₁, h₂)
  -- where K = R_h_norm * act_lipschitz < 1.
  -- For now, assume the activation function is L-Lipschitz:
  -- ∀ u v, dist(σ(u), σ(v)) ≤ L_σ * dist(u, v)
  -- where L_σ = cell.act_lipschitz.
  -- Key insight: The RNN step function should be a contraction
  -- because h only appears linearly in R_h·h, and σ applies Lipschitz compression.
  -- We invoke the contraction fixed point theorem:
  have h_contraction : Contraction.IsContraction f (R_h_norm * cell.act_lipschitz) := by
    -- This is the core step: proving the step function is a contraction
    -- Given: ‖R_h‖ * L_σ < 1
    -- We need: ∀ h₁ h₂, dist(σ(R_h·h₁ + ...), σ(R_h·h₂ + ...)) ≤ K * dist(h₁, h₂)
    constructor
    · -- K_nonneg: 0 ≤ R_h_norm * act_lipschitz
      exact mul_nonneg hR_pos hσ_pos
    · -- K_lt_one: R_h_norm * act_lipschitz < 1
      exact hContr
    · -- lipschitz property: dist(f(h₁), f(h₂)) ≤ K * dist(h₁, h₂)
      intro h₁ h₂
      -- Define the pre-activation vectors
      let u₁ := fun i => (cell.W_x *ᵥ x) i + (cell.R_h *ᵥ h₁) i + cell.b i
      let u₂ := fun i => (cell.W_x *ᵥ x) i + (cell.R_h *ᵥ h₂) i + cell.b i
      -- f(h₁) = σ(u₁) and f(h₂) = σ(u₂)
      have hf₁ : f h₁ = cell.act u₁ := rfl
      have hf₂ : f h₂ = cell.act u₂ := rfl
      -- Key insight: the W_x·x and b terms cancel out in the distance
      -- u₁ - u₂ = (W_x·x + R_h·h₁ + b) - (W_x·x + R_h·h₂ + b) = R_h·(h₁ - h₂)
      -- So dist(u₁, u₂) depends only on R_h and the difference of h₁, h₂.
      -- First, note that dist(u₁, u₂) is the distance between two vectors
      -- that differ only in their R_h component.
      have h_u_dist : dist u₁ u₂ ≤ R_h_norm * dist h₁ h₂ := by
        -- The vectors u₁ and u₂ differ only in the R_h component:
        -- u₁ i = (W_x *ᵥ x) i + (R_h *ᵥ h₁) i + b i
        -- u₂ i = (W_x *ᵥ x) i + (R_h *ᵥ h₂) i + b i
        -- So u₁ - u₂ = R_h·h₁ - R_h·h₂ = R_h·(h₁ - h₂).
        -- Define v₁ = R_h *ᵥ h₁ and v₂ = R_h *ᵥ h₂
        let v₁ := cell.R_h *ᵥ h₁
        let v₂ := cell.R_h *ᵥ h₂
        -- The key insight: u₁ and u₂ differ only in their R_h components
        -- so dist(u₁, u₂) equals dist(R_h·h₁, R_h·h₂) up to components that cancel.
        -- By h_R_h_norm, we have dist(R_h·h₁, R_h·h₂) ≤ R_h_norm * dist(h₁, h₂).
        -- Showing that u₁ - u₂ = v₁ - v₂:
        -- For each component i:
        -- (u₁ - u₂) i = (W_x *ᵥ x) i + (R_h *ᵥ h₁) i + b i - ((W_x *ᵥ x) i + (R_h *ᵥ h₂) i + b i)
        --             = (R_h *ᵥ h₁) i - (R_h *ᵥ h₂) i
        --             = (v₁ - v₂) i
        -- Therefore dist(u₁, u₂) = dist(v₁, v₂).
        have h_u_eq_v : u₁ - u₂ = v₁ - v₂ := by
          ext i
          change u₁ i - u₂ i = v₁ i - v₂ i
          simp only [show u₁ i = (cell.W_x *ᵥ x) i + (cell.R_h *ᵥ h₁) i + cell.b i from rfl]
          simp only [show u₂ i = (cell.W_x *ᵥ x) i + (cell.R_h *ᵥ h₂) i + cell.b i from rfl]
          ring
        -- Now use dist_comm to convert u₁ - u₂ = v₁ - v₂ to dist(u₁, u₂) = dist(v₁, v₂)
        have h_u_v_dist : dist u₁ u₂ = dist v₁ v₂ := by
          rw [dist_eq_norm u₁ u₂, dist_eq_norm v₁ v₂]
          congr 1
        -- Apply h_R_h_norm to get the bound
        rw [h_u_v_dist]
        exact h_R_h_norm h₁ h₂
      -- Apply Lipschitz property of cell.act
      have h_act : dist (cell.act u₁) (cell.act u₂) ≤ cell.act_lipschitz * dist u₁ u₂ :=
        h_act_lipschitz u₁ u₂
      -- Combine the bounds
      calc dist (f h₁) (f h₂)
          = dist (cell.act u₁) (cell.act u₂) := by rfl
        _ ≤ cell.act_lipschitz * dist u₁ u₂ := h_act_lipschitz u₁ u₂
        _ ≤ cell.act_lipschitz * (R_h_norm * dist h₁ h₂) :=
            mul_le_mul_of_nonneg_left h_u_dist hσ_pos
        _ = (R_h_norm * cell.act_lipschitz) * dist h₁ h₂ := by ring
  -- Now use the contraction unique fixed point theorem
  -- The theorem requires [CompleteSpace (Fin n → ℝ)] and [Nonempty (Fin n → ℝ)]
  haveI : Nonempty (Fin n → ℝ) := inferInstance
  have h_unique := Contraction.contraction_unique_fixed_point h_contraction
  -- h_unique : ∃! h, f h = h
  -- which is ∃ h, f h = h ∧ ∀ y, f y = y → y = h
  exact h_unique

end RNN
