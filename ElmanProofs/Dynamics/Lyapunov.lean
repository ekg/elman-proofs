/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Topology.Order.Basic
import Mathlib.Topology.Order.MonotoneConvergence
import Mathlib.Topology.Compactness.Compact
import Mathlib.Topology.MetricSpace.Bounded
import Mathlib.Topology.Sequences

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

    Key insight: if V(x) < inf{V(y) : dist y x₀ ≥ ε}, then x stays in B_ε(x₀).

    Given ε > 0, we need to find δ > 0 such that:
    ∀ x, dist x x₀ < δ → ∀ n, dist (sys.step^[n] x) x₀ < ε

    **Step 1: Find the minimum of V on the complement of the open ball.**

    Let S = {y : dist y x₀ ≥ ε} = (ball x₀ ε)ᶜ.
    S is closed (complement of open ball), hence compact in CompactSpace.
    V '' S is compact in ℝ (continuous image of compact).
    If S is nonempty, V attains its minimum m = sInf(V '' S) > 0 on S.

    **Step 2: Choose δ using continuity of V.**

    Since V is continuous at x₀ and V(x₀) = 0, there exists δ ∈ (0, ε) such that
    dist x x₀ < δ → V(x) < m.

    **Step 3: Prove stability by contradiction.**

    For any x with dist x x₀ < δ and any n:
    - V(sys.step^[n] x) ≤ V(x) < m   (by Lyapunov property)
    - If dist (sys.step^[n] x) x₀ ≥ ε, then sys.step^[n] x ∈ S
    - So V(sys.step^[n] x) ≥ m (by definition of m as infimum)
    - Contradiction! So dist (sys.step^[n] x) x₀ < ε.
-/
theorem lyapunov_implies_stable [CompactSpace X]
    (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsStable sys x₀ := by
  intro ε hε
  -- Define S = complement of open ball = {x : dist x x₀ ≥ ε}
  set S := (Metric.ball x₀ ε)ᶜ with hS_def

  -- Case 1: S is empty (all points are within ε of x₀)
  by_cases hS_empty : S = ∅
  · -- Any δ > 0 works since all points are already within ε
    use ε, hε
    intro x _ n
    have h_all_in_ball : ∀ y, dist y x₀ < ε := by
      intro y
      by_contra h_not
      have h_in_compl : y ∈ (Metric.ball x₀ ε)ᶜ := by
        simp only [Set.mem_compl_iff, Metric.mem_ball]
        exact fun h => h_not h
      rw [← hS_def, hS_empty] at h_in_compl
      exact Set.notMem_empty y h_in_compl
    exact h_all_in_ball (sys.step^[n] x)

  -- Case 2: S is nonempty
  · have hS_nonempty : S.Nonempty := Set.nonempty_iff_ne_empty.mpr hS_empty

    -- S is closed (complement of open ball)
    have hS_closed : IsClosed S := Metric.isOpen_ball.isClosed_compl

    -- S is compact in CompactSpace
    have hS_compact : IsCompact S := hS_closed.isCompact

    -- V '' S is compact in ℝ
    have hVS_compact : IsCompact (V '' S) := hS_compact.image hV_cont

    -- V '' S is nonempty
    have hVS_nonempty : (V '' S).Nonempty := Set.Nonempty.image V hS_nonempty

    -- sInf (V '' S) is in V '' S
    have h_sInf_mem : sInf (V '' S) ∈ V '' S := hVS_compact.sInf_mem hVS_nonempty

    -- Extract the witness: ∃ y ∈ S, V(y) = sInf (V '' S)
    obtain ⟨y, hy_in_S, hy_eq⟩ := h_sInf_mem

    -- m := sInf (V '' S) > 0 since y ≠ x₀ (y is at distance ≥ ε from x₀)
    set m := sInf (V '' S) with hm_def
    have hy_ne_x0 : y ≠ x₀ := by
      intro h_eq
      simp only [hS_def, Set.mem_compl_iff, Metric.mem_ball] at hy_in_S
      rw [h_eq, dist_self] at hy_in_S
      exact hy_in_S hε
    have hm_pos : 0 < m := by
      rw [← hy_eq]
      exact hV_pos y hy_ne_x0

    -- By continuity of V at x₀, find δ such that V(x) < m when dist x x₀ < δ
    have h_cont_at := hV_cont.continuousAt (x := x₀)
    rw [Metric.continuousAt_iff] at h_cont_at
    obtain ⟨δ', hδ'_pos, hδ'⟩ := h_cont_at m hm_pos

    -- Use δ = min(δ', ε) to ensure δ < ε
    use min δ' ε, lt_min hδ'_pos hε
    intro x hx n

    -- V(x) < m since dist x x₀ < δ' ≤ δ
    have hVx_lt_m : V x < m := by
      have hx_lt_δ' : dist x x₀ < δ' := lt_of_lt_of_le hx (min_le_left δ' ε)
      have := hδ' hx_lt_δ'
      rw [hV.zero_at_eq, Real.dist_eq, sub_zero, abs_of_nonneg (hV.nonneg x)] at this
      exact this

    -- V(step^n(x)) ≤ V(x) < m
    have hV_iter_lt_m : V (sys.step^[n] x) < m :=
      lt_of_le_of_lt (lyapunov_iterate_nonincreasing sys x₀ V hV x n) hVx_lt_m

    -- If step^n(x) ∈ S, then V(step^n(x)) ≥ m, contradiction
    by_contra h_not_lt
    push_neg at h_not_lt
    have h_in_S : sys.step^[n] x ∈ S := by
      simp only [hS_def, Set.mem_compl_iff, Metric.mem_ball]
      intro h_lt
      exact not_lt.mpr h_not_lt h_lt
    have hV_ge_m : m ≤ V (sys.step^[n] x) := by
      apply csInf_le
      · exact hVS_compact.bddBelow
      · exact Set.mem_image_of_mem V h_in_S
    exact not_lt.mpr hV_ge_m hV_iter_lt_m

/-- Key lemma: all subsequential limits of the trajectory equal the equilibrium.

    The proof uses LaSalle's invariance principle:
    - V along the trajectory is antitone and bounded below, so V(traj(φ n)) → L
    - By continuity of V, V(y) = L where y is the subsequential limit
    - By continuity of step, V(step y) = L as well
    - By strict Lyapunov, V(step y) = V(y) implies y = x₀
-/
lemma subseq_limit_eq_equilibrium [CompactSpace X]
    (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsStrictLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hstep_cont : Continuous sys.step)
    (x : X) (y : X) (φ : ℕ → ℕ) (hφ_mono : StrictMono φ)
    (hφ_tendsto : Filter.Tendsto (fun n => sys.step^[φ n] x) Filter.atTop (nhds y)) :
    y = x₀ := by
  /- The key insight is that for any subsequential limit y:
     - V(y) = lim V(traj(φ n)) = L (some limit of the antitone sequence)
     - V(step y) = lim V(traj(φ n + 1)) = L (same limit, since antitone bounded below)
     - By strict Lyapunov: V(step y) = V(y) implies y = x₀ -/
  have hV_eq : V y = V (sys.step y) := by
    -- This requires showing that V(traj(φ n)) and V(traj(φ n + 1)) have the same limit
    -- which follows from both being subsequences of an antitone bounded sequence

    -- Define the trajectory
    let traj : ℕ → X := fun n => sys.step^[n] x

    -- V along trajectory is antitone (non-increasing)
    have h_antitone : Antitone (V ∘ traj) := by
      intro m n hmn
      simp only [Function.comp_apply]
      -- V(traj n) = V(step^[n-m](traj m)) ≤ V(traj m) by Lyapunov property
      have h_iter : traj n = sys.step^[n - m] (traj m) := by
        simp only [traj]
        rw [← Function.iterate_add_apply]
        congr 1
        omega
      rw [h_iter]
      exact lyapunov_iterate_nonincreasing sys x₀ V hV.toIsLyapunovFunction (traj m) (n - m)

    -- V along trajectory is bounded below by 0
    have h_bdd_below : BddBelow (Set.range (V ∘ traj)) := by
      use 0
      intro z hz
      simp only [Set.mem_range, Function.comp_apply] at hz
      obtain ⟨n, rfl⟩ := hz
      exact hV.nonneg _

    -- The antitone bounded sequence converges to its infimum
    have h_tendsto_inf : Filter.Tendsto (V ∘ traj) Filter.atTop (nhds (⨅ n, V (traj n))) :=
      tendsto_atTop_ciInf h_antitone h_bdd_below

    -- Set L = lim V(traj(n))
    let L := ⨅ n, V (traj n)

    -- Key: V(traj(φ n)) → L (subsequence of convergent sequence)
    have h_subseq_tendsto : Filter.Tendsto (fun n => V (traj (φ n))) Filter.atTop (nhds L) :=
      h_tendsto_inf.comp (hφ_mono.tendsto_atTop)

    -- By continuity of V at y: V(traj(φ n)) → V(y)
    have h_V_y : Filter.Tendsto (fun n => V (traj (φ n))) Filter.atTop (nhds (V y)) :=
      hV_cont.continuousAt.tendsto.comp hφ_tendsto

    -- Therefore V(y) = L
    have hVy_eq_L : V y = L := tendsto_nhds_unique h_V_y h_subseq_tendsto

    -- Now show V(step(y)) = L
    -- First: traj(φ n + 1) = step(traj(φ n))
    have h_shift : ∀ n, traj (φ n + 1) = sys.step (traj (φ n)) := fun n => by
      simp only [traj, Function.iterate_succ_apply']

    -- step(traj(φ n)) → step(y) by continuity of step
    have h_step_tendsto : Filter.Tendsto (fun n => sys.step (traj (φ n))) Filter.atTop (nhds (sys.step y)) :=
      hstep_cont.continuousAt.tendsto.comp hφ_tendsto

    -- V(step(traj(φ n))) → V(step(y)) by continuity of V
    have h_V_step_y : Filter.Tendsto (fun n => V (sys.step (traj (φ n)))) Filter.atTop (nhds (V (sys.step y))) :=
      hV_cont.continuousAt.tendsto.comp h_step_tendsto

    -- V(traj(φ n + 1)) = V(step(traj(φ n)))
    have h_V_shift : ∀ n, V (traj (φ n + 1)) = V (sys.step (traj (φ n))) := fun n => by rw [h_shift]

    -- V(traj(φ n + 1)) → L (subsequence)
    -- Need: φ n + 1 is also strictly increasing and tends to infinity
    have h_shift_mono : StrictMono (fun n => φ n + 1) := fun a b hab =>
      Nat.add_lt_add_right (hφ_mono hab) 1

    have h_subseq_shift_tendsto : Filter.Tendsto (fun n => V (traj (φ n + 1))) Filter.atTop (nhds L) :=
      h_tendsto_inf.comp (h_shift_mono.tendsto_atTop)

    -- Using h_V_shift, rewrite h_subseq_shift_tendsto
    have h_V_step_to_L : Filter.Tendsto (fun n => V (sys.step (traj (φ n)))) Filter.atTop (nhds L) := by
      simp only [← h_V_shift]
      exact h_subseq_shift_tendsto

    -- By uniqueness of limits: V(step(y)) = L
    have hV_step_eq_L : V (sys.step y) = L := tendsto_nhds_unique h_V_step_y h_V_step_to_L

    -- Conclude: V(y) = L = V(step(y))
    rw [hVy_eq_L, hV_step_eq_L]
  by_contra h_ne
  have : V (sys.step y) < V y := hV.strict_decrease y h_ne
  rw [hV_eq] at this
  exact lt_irrefl _ this

/-- Strict Lyapunov function implies asymptotic stability.

    ## Proof via LaSalle's Invariance Principle

    **Part 1: Stability** follows from `lyapunov_implies_stable`.

    **Part 2: Convergence** uses the unique cluster point argument:
    1. All subsequential limits equal x₀ (by `subseq_limit_eq_equilibrium`)
    2. In compact metric space, unique cluster point implies convergence
    3. Contradiction: if not convergent, extract subsequence staying outside ε-ball
       but any such subsequence has a further convergent subsequence to x₀
-/
theorem strict_lyapunov_implies_asymptotic [CompactSpace X]
    (sys : DiscreteSystem X) (x₀ : X) (V : X → ℝ)
    (hV : IsStrictLyapunovFunction sys x₀ V) (hV_cont : Continuous V)
    (hstep_cont : Continuous sys.step)
    (hV_pos : ∀ x, x ≠ x₀ → 0 < V x) : IsAsymptoticallyStable sys x₀ := by
  constructor
  · exact lyapunov_implies_stable sys x₀ V hV.toIsLyapunovFunction hV_cont hV_pos
  use 1, by linarith
  intro x _
  let traj : ℕ → X := fun n => sys.step^[n] x
  -- All subsequential limits equal x₀
  have h_unique_limit : ∀ (y : X) (φ : ℕ → ℕ), StrictMono φ →
      Filter.Tendsto (traj ∘ φ) Filter.atTop (nhds y) → y = x₀ := by
    intro y φ hφ_mono hφ_tendsto
    exact subseq_limit_eq_equilibrium sys x₀ V hV hV_cont hstep_cont x y φ hφ_mono hφ_tendsto
  -- Prove convergence by contradiction
  rw [Metric.tendsto_atTop]
  by_contra h_not
  push_neg at h_not
  obtain ⟨ε, hε, h_inf⟩ := h_not
  -- Construct strictly increasing sequence staying outside ε-ball
  have h_build : ∃ ψ : ℕ → ℕ, StrictMono ψ ∧ ∀ k, ε ≤ dist (traj (ψ k)) x₀ := by
    -- For each N, there exists n ≥ N with dist ≥ ε
    have h_exists : ∀ N, ∃ n, N ≤ n ∧ ε ≤ dist (traj n) x₀ := by
      intro N
      obtain ⟨n, hn_ge, hn_dist⟩ := h_inf N
      exact ⟨n, hn_ge, hn_dist⟩
    -- Use Classical.choose to define f
    let f : ℕ → ℕ := fun N => Classical.choose (h_exists N)
    have hf : ∀ N, N ≤ f N ∧ ε ≤ dist (traj (f N)) x₀ := fun N => Classical.choose_spec (h_exists N)
    -- Define ψ iteratively: ψ 0 = f 0, ψ (k+1) = f (ψ k + 1)
    let ψ : ℕ → ℕ := fun k => k.rec (f 0) (fun _ prev => f (prev + 1))
    use ψ
    constructor
    · -- StrictMono ψ: show ψ a < ψ b for a < b
      intro a b hab
      induction b with
      | zero => exact absurd hab (Nat.not_lt_zero a)
      | succ b ih =>
        have h_ψ_succ : ψ (b + 1) = f (ψ b + 1) := rfl
        rcases Nat.lt_succ_iff_lt_or_eq.mp hab with h_lt | h_eq
        · -- a < b: use induction hypothesis
          have h_ind := ih h_lt
          have h_le : ψ b + 1 ≤ f (ψ b + 1) := (hf (ψ b + 1)).1
          rw [h_ψ_succ]
          omega
        · -- a = b: need ψ b < f (ψ b + 1)
          rw [h_eq, h_ψ_succ]
          have h_le : ψ b + 1 ≤ f (ψ b + 1) := (hf (ψ b + 1)).1
          omega
    · -- ∀ k, ε ≤ dist (traj (ψ k)) x₀
      intro k
      induction k with
      | zero => exact (hf 0).2
      | succ k _ => exact (hf (ψ k + 1)).2
  obtain ⟨ψ, hψ_mono, hψ_dist⟩ := h_build
  -- Extract convergent subsequence by compactness
  have := isCompact_univ.tendsto_subseq (fun n => Set.mem_univ ((traj ∘ ψ) n))
  obtain ⟨z, _, θ, hθ_mono, hθ_tendsto⟩ := this
  -- z = x₀ by unique limit property
  have hz_eq : z = x₀ := h_unique_limit z (ψ ∘ θ) (hψ_mono.comp hθ_mono) hθ_tendsto
  -- But dist(z, x₀) ≥ ε, contradiction
  have h_dist_ge : ∀ k, ε ≤ dist ((traj ∘ ψ ∘ θ) k) x₀ := fun k => hψ_dist (θ k)
  have h_dist_tendsto : Filter.Tendsto (fun k => dist ((traj ∘ ψ ∘ θ) k) x₀)
      Filter.atTop (nhds (dist z x₀)) := by
    have h_cont : Continuous (fun y => dist y x₀) := Continuous.dist continuous_id continuous_const
    exact h_cont.continuousAt.tendsto.comp hθ_tendsto
  have h_dist_z : ε ≤ dist z x₀ := by
    apply ge_of_tendsto h_dist_tendsto
    filter_upwards with k
    exact h_dist_ge k
  rw [hz_eq, dist_self] at h_dist_z
  exact not_lt.mpr h_dist_z hε

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
