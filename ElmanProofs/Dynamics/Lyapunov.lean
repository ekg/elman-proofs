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
    V must be non-negative, zero at equilibrium, and non-increasing. -/
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

/-- Stability in the sense of Lyapunov: trajectories stay near x₀. -/
def IsStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, dist x x₀ < δ → ∀ n : ℕ, dist (sys.step^[n] x) x₀ < ε

/-- Asymptotic stability: stable + trajectories converge to x₀. -/
def IsAsymptoticallyStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  IsStable sys x₀ ∧
  ∃ δ > 0, ∀ x, dist x x₀ < δ →
    Filter.Tendsto (fun n => sys.step^[n] x) Filter.atTop (nhds x₀)

/-- Exponential stability: convergence at rate c^n for some c < 1. -/
def IsExponentiallyStable (sys : DiscreteSystem X) (x₀ : X) : Prop :=
  ∃ c : ℝ, c < 1 ∧ ∃ M δ : ℝ, 0 < δ ∧ 0 < M ∧
    ∀ x, dist x x₀ < δ → ∀ n : ℕ, dist (sys.step^[n] x) x₀ ≤ M * c^n * dist x x₀

/-- Lyapunov function values decrease along trajectories. -/
theorem lyapunov_iterate_nonincreasing (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsLyapunovFunction sys x₀ V) (x : X) (n : ℕ) :
    V (sys.step^[n] x) ≤ V x := by
  induction n with
  | zero => simp
  | succ n ih =>
    have key : sys.step^[n + 1] x = sys.step (sys.step^[n] x) :=
      Function.iterate_succ_apply' sys.step n x
    calc V (sys.step^[n + 1] x)
        = V (sys.step (sys.step^[n] x)) := by rw [key]
      _ ≤ V (sys.step^[n] x) := hV.nonincreasing _
      _ ≤ V x := ih

/-- Lyapunov's direct method: existence of Lyapunov function implies stability.

    ## Proof Strategy

    Key insight: if V(x) < inf{V(y) : dist y x₀ = ε}, then x stays in B_ε(x₀).

    Given ε > 0, we need to find δ > 0 such that:
    ∀ x, dist x x₀ < δ → ∀ n, dist (sys.step^[n] x) x₀ < ε

    **Step 1: Find the minimum of V on the boundary sphere.**

    Let S_ε = {y : dist y x₀ = ε} be the sphere of radius ε.
    Since V is continuous and positive on S_ε, and S_ε is compact in finite dimensions,
    V attains a minimum m = inf{V(y) : y ∈ S_ε} > 0.

    **Step 2: Choose δ using continuity of V.**

    Since V is continuous at x₀ and V(x₀) = 0, there exists δ ∈ (0, ε) such that
    dist x x₀ < δ → V(x) < m.

    **Step 3: Prove stability.**

    For any x with dist x x₀ < δ and any n:
    - V(sys.step^[n] x) ≤ V(x) < m   (by Lyapunov property)
    - If dist (sys.step^[n] x) x₀ ≥ ε, then V(sys.step^[n] x) ≥ m (by def of m)
    - Contradiction! So dist (sys.step^[n] x) x₀ < ε.

    **Note:** The formal proof requires either:
    - CompactSpace or ProperSpace for X to ensure V attains minimum on S_ε
    - Or a "radially unbounded" / "proper" hypothesis on V
-/
theorem lyapunov_implies_stable (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsStable sys x₀ := by
  intro ε hε

  /- Proof Strategy:

     **Step 1**: Define m = inf{V(y) : dist y x₀ = ε}

     The sphere S_ε = {y : dist y x₀ = ε} is non-empty (take any point at distance ε).
     Since V is continuous and positive on S_ε (all points ≠ x₀), we have m > 0.

     **Technical issue**: Without CompactSpace, we can't guarantee V attains
     its infimum on S_ε. In a proper metric space (locally compact + complete),
     closed bounded sets are compact, so this would work.

     For now, we assume m := sInf (V '' {y | dist y x₀ = ε}) > 0.
     This is justified when:
     - X is finite dimensional
     - X is a proper metric space
     - V is "coercive" (V(x) → ∞ as ‖x‖ → ∞)

     **Step 2**: Find δ > 0 using continuity of V at x₀

     Since V is continuous at x₀ with V(x₀) = 0, and m > 0:
     ∃ δ > 0, dist x x₀ < δ → V(x) < m

     We also need δ < ε to ensure the ball B_δ(x₀) ⊆ B_ε(x₀).

     **Step 3**: Prove stability by contradiction

     Suppose some iterate sys.step^[n] x leaves B_ε(x₀).
     Then dist (sys.step^[n] x) x₀ ≥ ε.
     By continuity of the path (discrete), there exists m ≤ n with
     dist (sys.step^[m] x) x₀ = ε (first exit time).
     Thus V(sys.step^[m] x) ≥ m (by definition of m).
     But V(sys.step^[m] x) ≤ V(x) < m (by Lyapunov decreasing + choice of δ).
     Contradiction.
  -/

  -- The formal proof requires:
  -- 1. Existence of infimum m > 0 on the sphere (needs compactness or properness)
  -- 2. Continuity argument to find δ
  -- 3. Contradiction via Lyapunov decrease

  -- Without additional topological hypotheses, we cannot complete the proof.
  -- See `strict_lyapunov_implies_asymptotic` for the CompactSpace version.

  sorry

/-- Strict Lyapunov function implies asymptotic stability.

    ## Proof via LaSalle's Invariance Principle

    In a compact space with a strict Lyapunov function:

    **Part 1: Stability** (from `lyapunov_implies_stable`)

    This follows from the non-strict Lyapunov theorem.

    **Part 2: Convergence via LaSalle's Principle**

    Key concepts:
    - **ω-limit set**: ω(x) = {y : ∃ subsequence sys.step^[n_k] x → y}
    - **LaSalle's principle**: ω(x) ⊆ largest invariant subset of {z : V(step z) = V(z)}

    For strict Lyapunov functions:
    1. The set {z : V(step z) = V(z)} = {x₀} (strict decrease away from x₀)
    2. {x₀} is invariant (it's a fixed point)
    3. Therefore ω(x) ⊆ {x₀}

    In compact X:
    4. ω(x) is non-empty for any x (by Bolzano-Weierstrass)
    5. So ω(x) = {x₀}
    6. The full sequence converges: sys.step^[n] x → x₀

    **Technical Details**

    The proof uses:
    - `IsCompact.exists_tendsto_of_frequently_mem` for subsequential limits
    - Continuity of V to show ω-limit points satisfy V(step y) = V(y)
    - `tendsto_nhds_unique` for uniqueness of limits
-/
theorem strict_lyapunov_implies_asymptotic [CompactSpace X]
    (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsStrictLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsAsymptoticallyStable sys x₀ := by
  constructor

  -- Part 1: Stability from the non-strict Lyapunov theorem
  · exact lyapunov_implies_stable sys x₀ V hV.toIsLyapunovFunction hV_cont hV_pos

  -- Part 2: Asymptotic convergence
  /- LaSalle's Invariance Principle proof:

     **Step 1**: V along trajectory is monotonically decreasing and bounded

     For any x, the sequence V(sys.step^[n] x) is:
     - Monotone decreasing: V(step y) ≤ V(y) by Lyapunov property
     - Bounded below by 0: V(y) ≥ 0 for all y

     Therefore V(sys.step^[n] x) → L for some L ≥ 0.

     **Step 2**: ω-limit set characterization

     In CompactSpace X, every sequence has a convergent subsequence.
     The ω-limit set ω(x) = {y : ∃ subsequence sys.step^[n_k] x → y} is non-empty.

     For any y ∈ ω(x):
     - V(y) = L (by continuity of V and subsequential convergence)
     - V(step y) = L (by continuity of V ∘ step and the fact that
       step(sys.step^[n_k] x) → step(y) by continuity of step)

     **Step 3**: Strict decrease forces ω(x) = {x₀}

     By strict Lyapunov: if y ≠ x₀, then V(step y) < V(y).
     But we showed V(step y) = V(y) = L for y ∈ ω(x).
     Therefore y = x₀ for all y ∈ ω(x), i.e., ω(x) = {x₀}.

     **Step 4**: Unique ω-limit implies full convergence

     If all subsequential limits equal x₀, then sys.step^[n] x → x₀.
     (In compact metric spaces, this is standard: if every convergent
     subsequence has the same limit, the full sequence converges.)

     **Lean formalization requires**:
     - `IsCompact.exists_seq_tendsto_of_frequently` for subsequential limits
     - Continuity of composition for step ∘ iterate
     - `tendsto_of_subseq_tendsto` or similar for convergence from subsequences
  -/

  use 1  -- δ = 1 works since we have global convergence in CompactSpace
  constructor
  · linarith
  intro x _
  -- The sequence sys.step^[n] x has all subsequential limits equal to x₀
  -- by the LaSalle argument above. In a compact metric space, this implies
  -- the full sequence converges to x₀.

  -- The formal proof uses:
  -- 1. Monotone convergence of V(sys.step^[n] x) to some L ≥ 0
  -- 2. Compactness to extract convergent subsequence to some y
  -- 3. Continuity to show V(y) = V(step y) = L
  -- 4. Strict Lyapunov to conclude y = x₀, hence L = 0
  -- 5. Unique subsequential limit implies full convergence

  sorry

/-- Contraction implies exponential stability. -/
theorem contraction_exponentially_stable (sys : DiscreteSystem X) (x₀ : X)
    (hfixed : sys.step x₀ = x₀) (K : ℝ) (hK_lt : K < 1) (hK_nn : 0 ≤ K)
    (hcontr : ∀ x y, dist (sys.step x) (sys.step y) ≤ K * dist x y) :
    IsExponentiallyStable sys x₀ := by
  use K, hK_lt, 1, 1
  constructor
  · linarith
  constructor
  · linarith
  intro x _ n
  simp only [one_mul]
  induction n with
  | zero => simp
  | succ n ih =>
    have key : sys.step^[n + 1] x = sys.step (sys.step^[n] x) :=
      Function.iterate_succ_apply' sys.step n x
    calc dist (sys.step^[n + 1] x) x₀
        = dist (sys.step (sys.step^[n] x)) x₀ := by rw [key]
      _ = dist (sys.step (sys.step^[n] x)) (sys.step x₀) := by rw [hfixed]
      _ ≤ K * dist (sys.step^[n] x) x₀ := hcontr _ _
      _ ≤ K * (K^n * dist x x₀) := mul_le_mul_of_nonneg_left ih hK_nn
      _ = K^(n + 1) * dist x x₀ := by ring

end Dynamics
