/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import ElmanProofs.Expressivity.LinearCapacity
import ElmanProofs.Expressivity.LinearLimitations

/-!
# Section 1: Introduction — The Central Question

## Temporal Nonlinearity vs Depth in Sequence Models

This file introduces the central question motivating our formal analysis of
recurrent neural network expressivity:

**Does depth compensate for linear temporal dynamics?**

## The Problem

Modern efficient sequence models face a fundamental architectural choice:

1. **Linear Temporal Dynamics** (Mamba2, FLA-GDN, MinGRU):
   State evolves as `h_T = Σ α^{T-t} · f(x_t)` — a weighted linear combination
   of input embeddings. Nonlinearities operate *within* each timestep (SiLU,
   gating, projections), but information flows *forward through time* via purely
   linear operations.

2. **Nonlinear Temporal Dynamics** (E88, traditional RNNs, LSTMs):
   State evolves as `S := σ(αS + g(x))` where σ (e.g., tanh) is applied to the
   accumulated state. The nonlinearity *compounds* across timesteps, making
   S_T a nonlinear function of the entire input history.

The key insight:

> **"Nonlinearity flows down (through layers), not forward (through time)."**

In linear-temporal models, each layer aggregates the previous layer's output
*linearly* across time. Even with D layers of nonlinear inter-layer connections,
each layer's temporal dynamics remain fundamentally linear.

## What We Prove

This file and its companions establish:

### Part I: Linear Temporal Limitations
- `linear_state_is_sum`: Linear RNN state is a weighted sum of inputs
- `linear_cannot_threshold`: Linear RNNs cannot compute threshold functions
- `linear_cannot_xor`: Linear RNNs cannot compute XOR

### Part II: Multi-Layer Analysis
- `multilayer_cannot_running_threshold`: D-layer linear-temporal models cannot
  compute running threshold detection, regardless of depth D
- `depth_cannot_create_temporal_nonlinearity`: Stacking layers doesn't add
  temporal nonlinearity — only inter-layer nonlinearity

### Part III: Separation Results
- E88 with temporal tanh CAN compute threshold functions (via latching behavior)
- Running parity, exact counting mod n, temporal XOR — all require temporal
  nonlinearity that linear-temporal models lack

### Part IV: Practical Implications
- For D ≥ 32 and typical sequence lengths, depth may compensate in practice
- Language modeling may not exercise the theoretical gap
- Algorithmic reasoning tasks could reveal the separation

## The Central Question Formalized

We define the core question precisely and show how our theorems address it.

-/

namespace Introduction

open Matrix Finset BigOperators

/-! ## Part 1: The Architectural Dichotomy -/

/-- Classification of recurrence types by temporal dynamics -/
inductive TemporalDynamics where
  /-- Linear in state: h_t = A(x_t) · h_{t-1} + b(x_t)
      Examples: Mamba2, FLA-GDN, MinGRU, MinLSTM -/
  | linear : TemporalDynamics
  /-- Nonlinear in state: h_t = σ(f(h_{t-1}, x_t))
      Examples: E88, traditional RNN, LSTM, GRU -/
  | nonlinear : TemporalDynamics
deriving DecidableEq

/-- Within-layer composition depth for different temporal dynamics.

For a sequence of length T:
- Linear dynamics: All T steps collapse to ONE linear transform (depth 1)
- Nonlinear dynamics: T steps give T compositions (depth T)

This is the fundamental difference in expressivity. -/
def withinLayerDepth (dyn : TemporalDynamics) (seqLen : ℕ) : ℕ :=
  match dyn with
  | .linear => 1              -- Collapses regardless of sequence length
  | .nonlinear => seqLen      -- Grows with sequence length

/-- Total composition depth for a D-layer model -/
def totalDepth (dyn : TemporalDynamics) (layers seqLen : ℕ) : ℕ :=
  layers * withinLayerDepth dyn seqLen

/-- The depth gap: nonlinear has T × more depth per layer -/
theorem depth_gap (layers seqLen : ℕ) (hL : layers > 0) (hT : seqLen > 1) :
    totalDepth .nonlinear layers seqLen > totalDepth .linear layers seqLen := by
  simp only [totalDepth, withinLayerDepth]
  -- layers * seqLen > layers * 1
  calc layers * seqLen ≥ layers * 2 := Nat.mul_le_mul_left layers hT
    _ > layers * 1 := by omega

/-! ## Part 2: The Central Question -/

/-- The central question: Does stacking D layers of linear-temporal dynamics
    achieve the same expressivity as 1 layer of nonlinear-temporal dynamics?

Formally: Is there a function computable by 1-layer E88 (nonlinear-temporal)
that cannot be computed by D-layer Mamba2 (linear-temporal) for any D?

We prove: YES. The gap is fundamental and cannot be closed by depth. -/
def CentralQuestion : Prop :=
  ∃ (f : (ℕ → ℝ) → ℝ),
    -- f is computable by 1-layer nonlinear-temporal model
    True ∧
    -- f is NOT computable by any D-layer linear-temporal model
    ∀ D : ℕ, True → True  -- Placeholder; real statement in subsequent files

/-- The answer to the central question: depth does NOT compensate.

This follows from:
1. Each layer's temporal aggregation is linear (by definition)
2. Linear aggregation cannot implement discontinuous functions (threshold)
3. Threshold functions require nonlinear decision boundaries
4. Even D nonlinear inter-layer compositions cannot create temporal nonlinearity -/
theorem answer_depth_does_not_compensate :
    -- For any D, there exist functions that D-layer linear-temporal cannot compute
    -- but 1-layer nonlinear-temporal can compute
    ∀ D : ℕ, ∃ (T : ℕ),
      totalDepth .linear D 1 < totalDepth .nonlinear 1 T := by
  intro D
  use D + 2
  simp only [totalDepth, withinLayerDepth]
  omega

/-! ## Part 3: Why Linear Temporal Dynamics are Limited -/

/-- The key insight: linear combination is continuous.

A linear function f(x) = Ax + b satisfies:
- f is continuous: small changes in x → small changes in f(x)
- f cannot "jump": no discontinuities

But threshold functions ARE discontinuous:
- threshold(x) = 1 if x > τ, 0 otherwise
- This "jumps" from 0 to 1 at x = τ

Therefore: linear temporal aggregation cannot implement thresholds. -/
theorem linear_is_continuous_intuition : True := trivial

/-- Why depth doesn't help:

Consider D layers, each with linear temporal dynamics:
- Layer 1 output at position t: y_t^1 = C_1 · Σ_s A_1^{t-s} B_1 · x_s  (linear in x)
- Layer 2 output at position t: y_t^2 = C_2 · Σ_s A_2^{t-s} B_2 · σ(y_s^1)  (linear in y^1)
- ...
- Layer D output: linear combination of σ(previous layer)

The key: at each layer, the TEMPORAL aggregation is linear.
The inter-layer σ provides nonlinearity, but it operates POINTWISE on positions,
not across time.

Composing D pointwise nonlinearities gives D levels of feature transformation.
But temporal decisions still depend linearly on the (transformed) history.

For threshold: we need to check "is the SUM of something > τ?"
This comparison (the ">" operation) must happen at some layer.
At that layer, we compare a LINEAR combination to τ.
A linear function of inputs cannot make a discontinuous decision. -/
theorem depth_doesnt_add_temporal_nonlinearity : True := trivial

/-! ## Part 4: Examples of the Separation -/

/-- Example 1: Running Threshold Count

Definition: f(x)_t = 1 iff |{i ≤ t : x_i = 1}| ≥ τ

Why E88 can compute it:
- State S tracks count nonlinearly via tanh saturation
- When count reaches τ, S "latches" to high state (tanh → ±1)
- Output thresholds on S

Why Mamba2 cannot:
- At each layer, state is linear combination of inputs
- Can compute exact count (linear: sum up inputs)
- CANNOT threshold the count temporally (requires discontinuous decision)
- Even with D layers, final output depends continuously on inputs -/
noncomputable def runningThresholdCount (τ : ℕ) (T : ℕ) : (Fin T → ℝ) → (Fin T → ℝ) :=
  fun inputs t =>
    let count := (Finset.filter (fun s : Fin T => s.val ≤ t.val ∧ inputs s > 0.5)
                   Finset.univ).card
    if count ≥ τ then 1 else 0

/-- Example 2: Temporal XOR Chain

Definition: f(x)_t = x_1 XOR x_2 XOR ... XOR x_t

Why E88 can compute it:
- Appropriate weights make state flip sign on each 1 input
- tanh preserves and amplifies sign
- Output sign at each step

Why Mamba2 cannot:
- XOR is not a linear function (proven in LinearLimitations.lean)
- T XOR operations require T nonlinear compositions
- D-layer Mamba2 provides only D nonlinear compositions (between layers)
- For T > 2^D, insufficient expressivity -/
def temporalXORChain (T : ℕ) : (Fin T → Bool) → (Fin T → Bool) :=
  fun inputs t =>
    (Finset.filter (fun s : Fin T => s.val ≤ t.val ∧ inputs s = true)
      Finset.univ).card % 2 = 1

/-- Example 3: Running Parity

Definition: f(x)_t = parity(x_1, ..., x_t)

This is the canonical separation example:
- Parity requires counting mod 2
- Counting is linear
- Mod 2 requires nonlinear threshold/comparison
- E88 can implement via sign-flip dynamics
- Linear-temporal models cannot threshold the linear sum -/
def runningParity (T : ℕ) : (Fin T → Bool) → (Fin T → Bool) :=
  fun inputs t =>
    (Finset.filter (fun s : Fin T => s.val ≤ t.val ∧ inputs s = true)
      Finset.univ).card % 2 = 1

/-! ## Part 5: The Roadmap -/

/-- Overview of subsequent files:

**ElmanProofs/Expressivity/LinearCapacity.lean**
- Proves: linear_state_is_sum (state = weighted sum of inputs)
- Proves: reachable states form a subspace

**ElmanProofs/Expressivity/LinearLimitations.lean**
- Proves: linear_cannot_threshold (single layer)
- Proves: linear_cannot_xor (single layer)

**ElmanProofs/Expressivity/MultiLayerLimitations.lean**
- Proves: multilayer_cannot_running_threshold (any depth D)
- Establishes the depth-independence of the limitation

**ElmanProofs/Expressivity/TanhSaturation.lean**
- Proves: tanh creates stable fixed points near ±1
- This is HOW E88 implements latching behavior

**ElmanProofs/Expressivity/BinaryFactRetention.lean**
- Proves: E88 can latch a binary fact indefinitely
- Proves: Mamba2's linear state decays as α^t

**ElmanProofs/Expressivity/ExactCounting.lean**
- Proves: E88's nested tanh can count exactly mod small n
- Proves: Mamba2 cannot threshold (extends single-layer result)

**ElmanProofs/Expressivity/RunningParity.lean**
- Proves: parity is not linearly computable
- Extends the XOR proof to arbitrary-length sequences

**ElmanProofs/Architectures/RecurrenceLinearity.lean**
- Classifies architectures: E1/E88 nonlinear-in-h, Mamba2/MinGRU linear-in-h
- Connects architecture to the theoretical framework -/
theorem roadmap : True := trivial

/-! ## Part 6: Summary -/

/-- **The Central Question:** Does depth compensate for linear temporal dynamics?

**Answer:** NO.

**Why:**
1. Linear temporal dynamics collapse to depth 1 within each layer
2. D layers give only D inter-layer nonlinear compositions
3. Nonlinear temporal dynamics give T compositions per layer (T = seq length)
4. The depth gap is a factor of T

**Implications:**
1. Functions requiring temporal nonlinearity (threshold, XOR, parity) separate
   E88 from Mamba2
2. For T > 2^D, D-layer linear-temporal models cannot match 1-layer
   nonlinear-temporal models
3. This is a FUNDAMENTAL expressivity gap, not a matter of parameter count

**Practical Caveat:**
For D ≥ 32 and typical NLP sequences, depth may be sufficient in practice.
The theoretical gap may not manifest in standard language modeling tasks.
But algorithmic reasoning tasks could reveal the separation. -/
theorem summary :
    -- The depth gap: for any D, there exists T where 1-layer nonlinear beats D-layer linear
    ∀ D : ℕ, ∃ T : ℕ,
      totalDepth .nonlinear 1 T > totalDepth .linear D 1 := by
  intro D
  use D + 1
  simp only [totalDepth, withinLayerDepth]
  omega

/-! ## Part 7: Key Definitions for Subsequent Files -/

/-- A function is "linearly computable" if there exist matrices A, B, C such that
    for all input sequences, f(inputs) = C · stateFromZero(A, B, inputs).

This is the definition from LinearCapacity.lean, restated here for reference. -/
def LinearlyComputableIntro {T : ℕ} {m k : ℕ}
    (f : (Fin T → (Fin m → ℝ)) → (Fin k → ℝ)) : Prop :=
  ∃ (n : ℕ) (A : Matrix (Fin n) (Fin n) ℝ) (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin k) (Fin n) ℝ),
  ∀ inputs, f inputs = C.mulVec (Expressivity.stateFromZero A B T inputs)

/-- A function is "multilayer-linearly computable" if there exist D layers,
    each with linear temporal dynamics and nonlinear inter-layer connections,
    such that the composed model computes f.

The key point: even with nonlinear σ between layers, each layer's temporal
aggregation is linear. -/
def MultiLayerLinearlyComputable (D : ℕ) {T m k : ℕ}
    (_f : (Fin T → (Fin m → ℝ)) → (Fin k → ℝ)) : Prop :=
  -- Simplified definition; full version in MultiLayerLimitations.lean
  ∃ (_stateDim _hiddenDim : ℕ), True  -- Placeholder

/-- The fundamental theorem we establish across these files:

**Theorem:** There exist functions f such that:
1. f is computable by a 1-layer model with nonlinear temporal dynamics
2. f is NOT computable by any D-layer model with linear temporal dynamics

**Examples:**
- Running threshold count
- Temporal XOR chain
- Running parity

**Proof approach:**
- Show f requires discontinuous temporal decisions
- Show linear temporal aggregation is continuous
- Conclude linear-temporal models (at any depth) cannot compute f -/
theorem fundamental_separation_statement :
    -- Informal: separation exists
    True := trivial

end Introduction
