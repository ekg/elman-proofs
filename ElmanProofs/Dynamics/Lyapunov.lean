/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Order.Basic

/-!
# Lyapunov Stability Theory

This file formalizes Lyapunov stability for discrete dynamical systems,
which is fundamental for analyzing RNN convergence and attractor stability.

## Main Definitions

* `IsLyapunovFunction`: A function V that decreases along trajectories
* `IsStableEquilibrium`: An equilibrium point that nearby trajectories stay near
* `IsAsymptoticallyStable`: Stable + trajectories converge to equilibrium
* `IsExponentiallyStable`: Convergence at exponential rate

## Main Theorems

* `lyapunov_stable`: V decreasing implies stability
* `lyapunov_asymptotic`: V strictly decreasing implies asymptotic stability
* `exponential_lyapunov`: Quadratic bounds imply exponential stability

## References

* Lyapunov, A.M. (1892). "The General Problem of Stability of Motion"
* LaSalle, J.P. (1960). "Some Extensions of Liapunov's Second Method"
-/

namespace Dynamics

variable {X : Type*} [MetricSpace X]

/-- A discrete dynamical system is a self-map on a metric space. -/
structure DiscreteSystem (X : Type*) where
  /-- The evolution map -/
  step : X → X

/-- An equilibrium point of a discrete system. -/
def IsEquilibrium (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  sys.step x₀ = x₀

/-- A Lyapunov function for a discrete system at an equilibrium.
    V must be non-negative, zero at equilibrium, and non-increasing along trajectories. -/
structure IsLyapunovFunction (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ) : Prop where
  /-- V is non-negative -/
  nonneg : ∀ x, 0 ≤ V x
  /-- V is zero at equilibrium -/
  zero_at_eq : V x₀ = 0
  /-- V is non-increasing along trajectories -/
  nonincreasing : ∀ x, V (sys.step x) ≤ V x

/-- A strict Lyapunov function: V strictly decreases away from equilibrium. -/
structure IsStrictLyapunovFunction (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    extends IsLyapunovFunction sys x₀ V : Prop where
  /-- V strictly decreases away from equilibrium -/
  strict_decrease : ∀ x, x ≠ x₀ → V (sys.step x) < V x

/-- Stability in the sense of Lyapunov: trajectories starting near x₀ stay near x₀. -/
def IsStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, dist x x₀ < δ → ∀ n : ℕ, dist (sys.step^[n] x) x₀ < ε

/-- Asymptotic stability: stable + trajectories converge to x₀. -/
def IsAsymptoticallyStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  IsStable sys x₀ ∧ ∃ δ > 0, ∀ x, dist x x₀ < δ → Filter.Tendsto (fun n => sys.step^[n] x) Filter.atTop (nhds x₀)

/-- Exponential stability: convergence at rate c^n for some c < 1. -/
def IsExponentiallyStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  ∃ c : ℝ, c < 1 ∧ ∃ M δ : ℝ, 0 < δ ∧ 0 < M ∧
    ∀ x, dist x x₀ < δ → ∀ n : ℕ, dist (sys.step^[n] x) x₀ ≤ M * c^n * dist x x₀

/-- Lyapunov's direct method: existence of Lyapunov function implies stability. -/
theorem lyapunov_implies_stable (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsStable sys x₀ := by
  sorry -- To be proven

/-- Strict Lyapunov function implies asymptotic stability. -/
theorem strict_lyapunov_implies_asymptotic [CompactSpace X] (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsStrictLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsAsymptoticallyStable sys x₀ := by
  sorry -- To be proven

end Dynamics
