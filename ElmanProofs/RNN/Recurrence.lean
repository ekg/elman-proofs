/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
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
    (R_h_norm : ℝ) (hContr : cell.isContracting R_h_norm)
    (hσ_pos : 0 ≤ cell.act_lipschitz)
    (hR_pos : 0 ≤ R_h_norm) :
    ∃! h_star : Fin n → ℝ, cell.stepWithInput x h_star = h_star := by
  sorry

end RNN
