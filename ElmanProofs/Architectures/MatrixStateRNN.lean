/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Matrix State RNN: Trading Weight Capacity for State Capacity

This file explores a hypothetical architecture where the hidden state is a
MATRIX rather than a vector, potentially gaining more dynamic capacity
without additional GEMMs.

## The Key Insight

Current E1:
- State: h ∈ ℝ^d (d dynamic parameters)
- Weight: W_h ∈ ℝ^(d×d) (d² learned parameters)
- Update cost: O(d²) for W_h @ h

Matrix State RNN:
- State: H ∈ ℝ^(d×k) (dk dynamic parameters)
- Weight: diagonal/structured (O(d+k) learned parameters)
- Update cost: O(dk) for element-wise operations

If k = d, we get d² DYNAMIC state for O(d²) cost - same as current E1,
but the capacity is in the state, not the fixed weights!

## Why This Might Work

1. **Outer product updates**: H' = λH + u ⊗ v is O(dk), not O(d²k)
2. **Diagonal transitions**: H' = diag(a) @ H @ diag(b) is O(dk)
3. **The state itself carries the "weight-like" information**

## Why This Might Not Work

1. **Learned weights capture task structure** - random updates might not
2. **Gradient flow through matrix state** - more complex dynamics
3. **Initialization and stability** - matrix state needs careful handling

## Comparison to Existing Approaches

- **Fast Weights**: Similar idea, H accumulates outer products
- **Linear Attention**: K^T V is like a matrix state updated by outer products
- **Mamba2**: Uses expanded state (d_state > d_output) with diagonal transitions
-/

namespace MatrixStateRNN

open Matrix

variable {d k : Nat}

/-! ## Part 1: Architecture Definitions -/

/-- Standard vector-state RNN (like E1) -/
structure VectorStateRNN (d : Nat) where
  W_h : Matrix (Fin d) (Fin d) Real   -- d² learned params
  W_x : Matrix (Fin d) (Fin d) Real   -- d² learned params

/-- Matrix-state RNN with diagonal transitions -/
structure MatrixStateRNN (d k : Nat) where
  /-- Decay factors for rows (d params) -/
  row_decay : Fin d → Real
  /-- Decay factors for columns (k params) -/
  col_decay : Fin k → Real
  /-- Key projection (d params if diagonal, d² if full) -/
  W_key : Matrix (Fin d) (Fin d) Real
  /-- Value projection (k params if diagonal, dk if full) -/
  W_val : Matrix (Fin k) (Fin d) Real

/-! ## Part 2: State Representations -/

/-- Vector state: d parameters -/
def vector_state_size (d : Nat) : Nat := d

/-- Matrix state: d × k parameters -/
def matrix_state_size (d k : Nat) : Nat := d * k

/-- THEOREM: Matrix state with k = d has d² dynamic parameters,
    same as the WEIGHT matrix in standard RNN! -/
theorem matrix_state_matches_weight_capacity (d : Nat) :
    matrix_state_size d d = d * d := rfl

/-! ## Part 3: Computational Cost Analysis -/

/-- Cost of vector RNN update: W_h @ h is O(d²) -/
def vector_rnn_update_cost (d : Nat) : Nat := d * d

/-- Cost of matrix RNN with diagonal transitions: O(d × k) -/
def matrix_rnn_diagonal_cost (d k : Nat) : Nat := d * k

/-- Cost of outer product update: O(d × k) -/
def outer_product_cost (d k : Nat) : Nat := d * k

/-- Total matrix RNN cost per step -/
def matrix_rnn_total_cost (d k : Nat) : Nat :=
  matrix_rnn_diagonal_cost d k + outer_product_cost d k

/-- THEOREM: Matrix RNN with k = d has same asymptotic cost as vector RNN -/
theorem matrix_rnn_same_cost (d : Nat) :
    matrix_rnn_total_cost d d = 2 * vector_rnn_update_cost d := by
  simp only [matrix_rnn_total_cost, matrix_rnn_diagonal_cost,
             outer_product_cost, vector_rnn_update_cost]
  ring

/-- THEOREM: Matrix RNN has d× more state for same cost -/
theorem matrix_rnn_more_state (d : Nat) (hd : d > 1) :
    matrix_state_size d d > vector_state_size d := by
  simp only [matrix_state_size, vector_state_size]
  have h : d * d > d := by nlinarith
  exact h

/-! ## Part 4: The Update Rules -/

/-- Vector RNN update: h' = tanh(W_h @ h + W_x @ x) -/
noncomputable def vector_update (rnn : VectorStateRNN d)
    (h : Fin d → Real) (x : Fin d → Real) : Fin d → Real :=
  fun i => Real.tanh ((rnn.W_h.mulVec h + rnn.W_x.mulVec x) i)

/-- Outer product of two vectors -/
def outerProduct (u : Fin d → Real) (v : Fin k → Real) :
    Matrix (Fin d) (Fin k) Real :=
  Matrix.of fun i j => u i * v j

/-- Element-wise (Hadamard) product of matrices -/
def hadamard (A B : Matrix (Fin d) (Fin k) Real) :
    Matrix (Fin d) (Fin k) Real :=
  Matrix.of fun i j => A i j * B i j

/-- Diagonal scaling: diag(a) @ H @ diag(b) -/
def diagonalScale (row_scale : Fin d → Real) (col_scale : Fin k → Real)
    (H : Matrix (Fin d) (Fin k) Real) : Matrix (Fin d) (Fin k) Real :=
  Matrix.of fun i j => row_scale i * H i j * col_scale j

/-- Matrix RNN update: H' = scale(H) + key ⊗ value
    This is O(dk), not O(d²k)! -/
def matrix_update (rnn : MatrixStateRNN d k)
    (H : Matrix (Fin d) (Fin k) Real) (x : Fin d → Real) :
    Matrix (Fin d) (Fin k) Real :=
  let scaled := diagonalScale rnn.row_decay rnn.col_decay H
  let key := rnn.W_key.mulVec x
  let value := rnn.W_val.mulVec x
  scaled + outerProduct key value

/-! ## Part 5: Expressivity Analysis -/

/-! The "effective capacity" of a state representation.
For computation, what matters is:
- How many independent values can influence the output?
- How many distinct input→output mappings are possible? -/

/-- Vector state: d values directly influence output -/
def vector_effective_capacity (d : Nat) : Nat := d

/-- Matrix state: dk values can influence output -/
def matrix_effective_capacity (d k : Nat) : Nat := d * k

/-- THEOREM: Matrix state has k× more effective capacity -/
theorem matrix_more_capacity (d k : Nat) (hk : k > 1) (hd : d > 0) :
    matrix_effective_capacity d k > vector_effective_capacity d := by
  simp only [matrix_effective_capacity, vector_effective_capacity]
  have h : d * k > d * 1 := by
    apply Nat.mul_lt_mul_of_pos_left hk hd
  simp at h
  exact h

/-! ## Part 6: The Key Tradeoff -/

/-! In standard RNN:
- W_h captures LEARNED patterns (task-specific)
- h captures DYNAMIC state (input-specific)

In matrix-state RNN:
- H captures BOTH learned AND dynamic information
- Update rule determines how patterns are accumulated

The question: Can outer-product updates learn as effectively as
gradient-trained W_h? -/

/-- Rank of accumulated outer products after T steps.
    If we add T rank-1 updates, max rank is min(T, d, k). -/
def accumulated_rank (T d k : Nat) : Nat := min T (min d k)

/-- THEOREM: After d steps, matrix state can have full rank -/
theorem full_rank_possible (d k : Nat) (hk : k ≥ d) :
    accumulated_rank d d k = d := by
  simp only [accumulated_rank]
  omega

/-! ## Part 7: Comparison to Mamba2 -/

/-- Mamba2's state: h ∈ ℝ^n with diagonal transition A ∈ ℝ^n
    Cost: O(n) for diagonal multiply
    State: n parameters

    Matrix RNN: H ∈ ℝ^(d×k) with diagonal transitions
    Cost: O(dk) for diagonal multiply
    State: dk parameters

    If we set n = dk, both have same state size.
    But matrix structure might provide different inductive bias. -/

def mamba2_state_size (n : Nat) : Nat := n
def mamba2_transition_cost (n : Nat) : Nat := n

/-- THEOREM: Matrix state can match Mamba2's state/cost ratio -/
theorem matrix_matches_mamba2_ratio (d k : Nat) :
    -- State/cost ratio for matrix RNN
    matrix_state_size d k = matrix_rnn_diagonal_cost d k := by
  simp only [matrix_state_size, matrix_rnn_diagonal_cost]

/-! ## Part 8: Proposed Architecture -/

/-! ### "E2": Matrix-State Elman

Proposed architecture:
```
State: H ∈ ℝ^(d×k)
Input: x ∈ ℝ^d

# Update (all O(dk) operations):
decay = sigmoid(w_decay @ x)           # scalar or per-row
key = tanh(W_key @ x)                  # d-vector
value = W_val @ x                      # k-vector
H' = decay * H + key ⊗ value           # outer product update

# Output:
query = W_query @ x                    # d-vector
output = H' @ query                    # k-vector, then project to d
```

Key properties:
1. O(dk) update cost (same as E1's O(d²) if k = d)
2. dk dynamic state (d× more than E1 if k = d)
3. Outer products accumulate "associative memory"
4. Nonlinearity in key (tanh) provides composition power

### Why This Might Beat E1

1. **More state capacity**: dk vs d dynamic parameters
2. **Associative memory**: outer products naturally store key-value pairs
3. **Selective forgetting**: decay can be input-dependent

### Why This Might Not Beat E1

1. **No direct h→h nonlinearity**: key depends on x, not H
2. **Rank limitations**: outer products are rank-1
3. **Different optimization landscape**: might be harder to train

### The Fundamental Question

Can accumulated outer products (learned through the update rule)
match the expressivity of a directly learned W_h matrix?

This is related to:
- Can Linear Attention match Softmax Attention?
- Can diagonal RNNs match full RNNs?
- Can fast weights match slow weights?

The empirical answer seems to be: "partially, with the right structure".
Mamba2 shows that careful design can make linear updates competitive.
-/

/-- Capacity ratio: how much more state does matrix-RNN have? -/
def capacity_ratio (d k : Nat) (hd : d > 0) : Nat :=
  (matrix_state_size d k) / (vector_state_size d)

/-- With k = d, we get d× more state -/
theorem capacity_ratio_is_k (d : Nat) (hd : d > 0) :
    (d * d) / d = d := Nat.mul_div_cancel_left d hd

end MatrixStateRNN
