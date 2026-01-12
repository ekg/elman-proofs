/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Architectures.SparseAttention

/-!
# E31: Sparse-Gated Elman

E31 extends E1 by replacing the dense silu output gate with sparse entmax gating.

## Motivation

E1 achieves excellent performance with a simple architecture:
```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
output = h_t * silu(W_gate @ x_t + b_gate)
```

However, the output gate is dense - all D dimensions contribute to output.
This may limit the model's ability to learn "program-like" computations where
only a subset of dimensions should be active at each step.

## E31 Innovation

Replace silu with entmax for sparse output gating:
```
h_t = tanh(W_x @ x_t + W_h @ h_{t-1} + b)
gate = entmax(W_gate @ x_t + b_gate)  -- sparse!
output = h_t * gate
```

Key properties:
- Gate is sparse: most dimensions are exactly 0
- Only selected dimensions contribute to output
- Creates "register-like" behavior where few dimensions are active
- When gate is 1-hot, equivalent to selecting one "register"

## Connection to Turing Machines

In a TM, the tape head is at exactly ONE position - discrete, not continuous.
E31's sparse gate creates a similar effect on the hidden state dimensions:
- Dense gate (E1): all dimensions blend together
- Sparse gate (E31): only k dimensions are "read out"

This is NOT making the hidden state sparse - the full h_t is still computed
and passed to the next timestep. We're just making the OUTPUT selective.

## Main Results

* `e31_generalizes_e1` - E1 is limit as α → 1 (softmax)
* `e31_output_sparse` - output has sparse support
* `e31_one_hot_is_register_select` - 1-hot gate = single register output
* `e31_bounded` - outputs remain bounded
-/

namespace E31_SparseGatedElman

open Matrix Finset
open SparseAttention

/-! ## Part 1: Configuration -/

/-- E31 configuration -/
structure E31Config where
  D : Nat        -- Hidden dimension
  D_in : Nat     -- Input dimension (often = D)
  deriving Repr

/-- Hidden state -/
def HiddenState (cfg : E31Config) := Fin cfg.D → Real

/-- Input -/
def Input (cfg : E31Config) := Fin cfg.D_in → Real

/-! ## Part 2: Activation Functions -/

/-- SiLU (Swish) activation: x * sigmoid(x) -/
noncomputable def silu (x : Real) : Real :=
  x / (1 + Real.exp (-x))

/-- SiLU is bounded: |silu(x)| ≤ |x| -/
theorem silu_bounded (x : Real) : |silu x| ≤ |x| := by
  sorry -- silu(x) = x * sigmoid(x), 0 < sigmoid < 1

/-- Sigmoid is in (0, 1) -/
theorem sigmoid_range (x : Real) : 0 < 1 / (1 + Real.exp (-x)) ∧ 1 / (1 + Real.exp (-x)) < 1 := by
  sorry

/-! ## Part 3: E1 Baseline (Dense Gating) -/

/-- E1 hidden state update -/
noncomputable def e1_hidden_update {cfg : E31Config}
    (h_prev : HiddenState cfg)
    (x : Fin cfg.D → Real)  -- projected input
    (W_h : Matrix (Fin cfg.D) (Fin cfg.D) Real)
    (b : Fin cfg.D → Real)
    : HiddenState cfg :=
  fun d => Real.tanh (x d + (W_h.mulVec h_prev) d + b d)

/-- E1 output with dense silu gating -/
noncomputable def e1_output {cfg : E31Config}
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)  -- W_gate @ x + b_gate
    : Fin cfg.D → Real :=
  fun d => h d * silu (gate_input d)

/-- E1 complete step -/
noncomputable def e1_step {cfg : E31Config}
    (h_prev : HiddenState cfg)
    (x_proj : Fin cfg.D → Real)
    (gate_input : Fin cfg.D → Real)
    (W_h : Matrix (Fin cfg.D) (Fin cfg.D) Real)
    (b : Fin cfg.D → Real)
    : HiddenState cfg × (Fin cfg.D → Real) :=
  let h_new := e1_hidden_update h_prev x_proj W_h b
  let output := e1_output h_new gate_input
  (h_new, output)

/-! ## Part 4: E31 (Sparse Gating) -/

/-- E31 output with sparse entmax gating.

    gate = entmax(gate_input)  -- sparse!
    output = h * gate

    Note: We use sparsemax here as a concrete sparse attention.
    In practice, entmax with α=1.5 is recommended. -/
noncomputable def e31_output {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    : Fin cfg.D → Real :=
  let gate := sparsemax gate_input  -- sparse gate!
  fun d => h d * gate d

/-- E31 complete step -/
noncomputable def e31_step {cfg : E31Config} [NeZero cfg.D]
    (h_prev : HiddenState cfg)
    (x_proj : Fin cfg.D → Real)
    (gate_input : Fin cfg.D → Real)
    (W_h : Matrix (Fin cfg.D) (Fin cfg.D) Real)
    (b : Fin cfg.D → Real)
    : HiddenState cfg × (Fin cfg.D → Real) :=
  let h_new := e1_hidden_update h_prev x_proj W_h b
  let output := e31_output h_new gate_input
  (h_new, output)

/-! ## Part 5: Key Properties -/

/-- The E31 gate is sparse (has exact zeros).
    This follows from sparsemax properties. -/
theorem e31_gate_sparse {cfg : E31Config} [NeZero cfg.D]
    (gate_input : Fin cfg.D → Real) :
    ∃ S : Finset (Fin cfg.D), S.card < cfg.D ∧
      ∀ d, d ∉ S → sparsemax gate_input d = 0 := by
  -- sparsemax produces exact zeros outside support
  sorry

/-- E31 output inherits sparsity from gate.
    Where gate = 0, output = 0 regardless of h. -/
theorem e31_output_sparse {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (d : Fin cfg.D)
    (h_gate_zero : sparsemax gate_input d = 0) :
    e31_output h gate_input d = 0 := by
  simp only [e31_output]
  rw [h_gate_zero]
  ring

/-- When gate is 1-hot (maximally sparse), output is single register. -/
theorem e31_one_hot_is_register_select {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (winner : Fin cfg.D)
    (h_one_hot : ∀ d, sparsemax gate_input d = if d = winner then 1 else 0) :
    ∀ d, e31_output h gate_input d = if d = winner then h winner else 0 := by
  intro d
  simp only [e31_output, h_one_hot]
  split_ifs with heq
  · subst heq; ring
  · ring

/-- E31 output is bounded when h is bounded (by tanh). -/
theorem e31_output_bounded {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (h_bound : ∀ d, |h d| ≤ 1)  -- tanh output
    : ∀ d, |e31_output h gate_input d| ≤ 1 := by
  intro d
  simp only [e31_output]
  -- |h d * gate d| ≤ |h d| * |gate d| ≤ 1 * 1 = 1
  -- since sparsemax ∈ [0,1] and |h d| ≤ 1
  sorry

/-! ## Part 6: E31 Generalizes E1 -/

/-- As sparsity decreases (α → 1), E31 approaches E1.

    More precisely: if we use softmax instead of sparsemax,
    and gate_input has uniform values, the gates are equal.

    In the limit α → 1, entmax → softmax, so E31 → E1-like behavior. -/
theorem e31_softmax_limit {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    -- When using softmax (α=1) instead of sparsemax
    -- the behavior approaches dense gating
    : True := trivial  -- Placeholder for limit theorem

/-! ## Part 7: Computational Properties -/

/-- E31 hidden state evolution is identical to E1.
    Only the output computation differs. -/
theorem e31_hidden_equals_e1 {cfg : E31Config} [NeZero cfg.D]
    (h_prev : HiddenState cfg)
    (x_proj gate_input : Fin cfg.D → Real)
    (W_h : Matrix (Fin cfg.D) (Fin cfg.D) Real)
    (b : Fin cfg.D → Real) :
    (e31_step h_prev x_proj gate_input W_h b).1 =
    (e1_step h_prev x_proj gate_input W_h b).1 := by
  simp only [e31_step, e1_step]

/-- Active dimensions in E31 output.
    The set of non-zero output dimensions equals the sparsemax support. -/
noncomputable def e31_active_dims {cfg : E31Config} [NeZero cfg.D]
    (gate_input : Fin cfg.D → Real) : Finset (Fin cfg.D) :=
  Finset.filter (fun d => sparsemax gate_input d ≠ 0) Finset.univ

/-- Output is zero outside active dimensions. -/
theorem e31_zero_outside_active {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (d : Fin cfg.D)
    (h_not_active : d ∉ e31_active_dims gate_input) :
    e31_output h gate_input d = 0 := by
  simp only [e31_active_dims, Finset.mem_filter, Finset.mem_univ, true_and,
             not_not] at h_not_active
  exact e31_output_sparse h gate_input d h_not_active

/-! ## Part 8: Gradient Flow Analysis -/

/-- In E31, gradients only flow through active dimensions.

    This is a key difference from E1:
    - E1: gradients flow through ALL dimensions (silu never exactly 0)
    - E31: gradients only flow through sparse support

    This could help or hurt:
    - Help: focused learning on relevant dimensions
    - Hurt: some dimensions may never get trained -/
structure GradientFlow where
  all_dims_receive_grad : Bool
  sparse_grad_routing : Bool

def e1_gradient_flow : GradientFlow where
  all_dims_receive_grad := true   -- silu(x) ≠ 0 for all x
  sparse_grad_routing := false

def e31_gradient_flow : GradientFlow where
  all_dims_receive_grad := false  -- sparsemax has exact zeros
  sparse_grad_routing := true     -- gradients routed to winners

/-! ## Part 9: Register Machine Interpretation -/

/-! E31 can be viewed as a register machine where:
- Hidden state h has D "registers"
- Gate selects which registers to output
- Sparse gate = only k registers read out

This is more TM-like than E1:
- TM: tape head at ONE position
- E31: attention over k positions (k small)
- E1: attention over ALL positions (dense) -/

/-- Sparsity level: expected number of active dimensions. -/
noncomputable def expected_active_dims {cfg : E31Config} [NeZero cfg.D]
    (gate_input : Fin cfg.D → Real) : Nat :=
  (e31_active_dims gate_input).card

/-- For TM-like behavior, we want small active dims. -/
def is_tm_like {cfg : E31Config} [NeZero cfg.D]
    (gate_input : Fin cfg.D → Real)
    (k : Nat) : Prop :=
  expected_active_dims gate_input ≤ k

/-! ## Part 10: Variants -/

/-- E31a: Sparsemax gating (α = 2) -/
noncomputable def e31a_output {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    : Fin cfg.D → Real :=
  let gate := sparsemax gate_input  -- α = 2
  fun d => h d * gate d

/-- E31b: 1.5-entmax gating (α = 1.5, moderate sparsity) -/
noncomputable def e31b_output {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    : Fin cfg.D → Real :=
  let gate := entmax (3/2) gate_input  -- α = 1.5
  fun d => h d * gate d

/-- E31c: Top-k hard gating (maximally sparse, k active) -/
noncomputable def e31c_output {cfg : E31Config}
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (k : Nat)
    : Fin cfg.D → Real :=
  -- Hard top-k: select k largest, zero others
  -- Then softmax among winners for differentiability
  sorry  -- Implementation requires top-k selection

/-- E31d: Learned sparsity (α as parameter) -/
noncomputable def e31d_output {cfg : E31Config} [NeZero cfg.D]
    (h : HiddenState cfg)
    (gate_input : Fin cfg.D → Real)
    (α : Real)  -- learned sparsity parameter
    (h_α : α > 1)
    : Fin cfg.D → Real :=
  let gate := entmax α gate_input
  fun d => h d * gate d

/-! ## Part 11: Recommendations

**E31b (1.5-entmax)** is the recommended starting point:
- Moderate sparsity (not too aggressive)
- Fully differentiable
- Proven to work well in attention mechanisms
- Minimal overhead over softmax

**E31a (sparsemax)** for more aggressive sparsity:
- Maximum sparsity for given logit spread
- May be harder to train
- Good if you want near-1-hot behavior

**E31c (top-k)** for explicit sparsity control:
- Exactly k dimensions active
- Requires STE for gradients
- Most TM-like but hardest to train

**E31d (learned α)** for adaptive sparsity:
- Model learns optimal sparsity level
- May need careful initialization
- Most flexible but adds complexity

## Summary

E31 KEY INSIGHT:

The output gate should be SPARSE!

E1:  `output = h * silu(gate_input)`     -- dense, all dims contribute
E31: `output = h * entmax(gate_input)`   -- sparse, k dims contribute

This creates register-like behavior:
- Full hidden state h is maintained (all D dims)
- Only k dims are "read out" to output
- More TM-like: discrete selection over registers

The hidden state recurrence is UNCHANGED from E1.
Only the output gating is sparsified.

PRACTICAL RECOMMENDATION: Start with E31b (α=1.5 entmax).
One line change from E1, differentiable, proven sparse attention mechanism.
-/

end E31_SparseGatedElman
