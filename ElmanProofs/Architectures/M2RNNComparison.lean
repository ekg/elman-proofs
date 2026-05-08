/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# M2RNN vs E88: Matrix-State Recurrence Scaffold

This file records the formal comparison framework for the Dao et al. M2RNN
paper and E88.

The goal is not to prove the hard expressivity theorems here. The goal is to
name the update maps and structural axes precisely enough that later proof work
can target them.

Main distinction:

* M2RNN candidate: `tanh(H W + k v^T)`, followed by an external forget carry.
* E88 delta form: `tanh((lambda I - k k^T) H + k v^T)`.

Both are nonlinear matrix-state recurrences. E88 differs by using a delta-rule
write, which expands to an input-dependent projection-like left transition.
-/

namespace M2RNNComparison

open Matrix

abbrev Vec (n : Nat) := Fin n → Real
abbrev MatState (K V : Nat) := Matrix (Fin K) (Fin V) Real

variable {K V : Nat}

/-! ## Shared Matrix-State Operations -/

/-- Elementwise map over a matrix. -/
def matrixMap (f : Real → Real) (H : MatState K V) : MatState K V :=
  Matrix.of fun i j => f (H i j)

/-- Elementwise `tanh` over a matrix. -/
noncomputable def matrixTanh (H : MatState K V) : MatState K V :=
  matrixMap Real.tanh H

/-- Outer product `k v^T`, with shape `K x V`. -/
def outerKV (k : Vec K) (v : Vec V) : MatState K V :=
  Matrix.of fun i j => k i * v j

/-- Query readout `H^T q`, with shape `V`. -/
def queryReadout (H : MatState K V) (q : Vec K) : Vec V :=
  H.transpose.mulVec q

/-! ## M2RNN Core -/

/-- M2RNN candidate state:

`Z_t = tanh(H_{t-1} W + k_t v_t^T)`.
-/
noncomputable def m2rnnCandidate
    (W : Matrix (Fin V) (Fin V) Real)
    (H : MatState K V) (k : Vec K) (v : Vec V) : MatState K V :=
  matrixTanh (H * W + outerKV k v)

/-- M2RNN state update:

`H_t = f_t H_{t-1} + (1 - f_t) Z_t`.

Here `f` is scalar per head. Vector-gated variants can be represented later by
replacing scalar multiplication with row/column gating.
-/
noncomputable def m2rnnUpdate
    (W : Matrix (Fin V) (Fin V) Real) (f : Real)
    (H : MatState K V) (k : Vec K) (v : Vec V) : MatState K V :=
  f • H + (1 - f) • m2rnnCandidate W H k v

/-- M2RNN readout with residual value path:

`y_t = H_t^T q_t + w_r * v_t`.
-/
def m2rnnReadout (H : MatState K V) (q : Vec K) (v wr : Vec V) : Vec V :=
  queryReadout H q + fun j => wr j * v j

/-! ## E88 Delta Matrix-State Core -/

/-- Delta-rule retrieval error:

`delta_t = v_t - H_{t-1}^T k_t`.
-/
def e88Delta (H : MatState K V) (k : Vec K) (v : Vec V) : Vec V :=
  v - queryReadout H k

/-- E88 delta update in direct write form:

`H_t = tanh(lambda H_{t-1} + k_t delta_t^T)`.
-/
noncomputable def e88DeltaUpdateDirect
    (lambda : Real) (H : MatState K V) (k : Vec K) (v : Vec V) : MatState K V :=
  matrixTanh (lambda • H + outerKV k (e88Delta H k v))

/-- E88's expanded delta transition:

`A_t(k) = lambda I - k k^T`.

With normalized `k`, this is projection/Householder-adjacent: it modifies the
key direction and leaves orthogonal directions controlled by `lambda`.
-/
def e88DeltaTransition (lambda : Real) (k : Vec K) :
    Matrix (Fin K) (Fin K) Real :=
  lambda • (1 : Matrix (Fin K) (Fin K) Real) - outerKV k k

/-- E88 delta update in expanded transition form:

`H_t = tanh((lambda I - k k^T) H_{t-1} + k_t v_t^T)`.

This is the form that makes the comparison to M2RNN sharp:

* M2RNN: fixed learned right transition `H W`.
* E88: input-dependent left transition `A_t(k) H`.
-/
noncomputable def e88DeltaUpdateExpanded
    (lambda : Real) (H : MatState K V) (k : Vec K) (v : Vec V) : MatState K V :=
  matrixTanh (e88DeltaTransition lambda k * H + outerKV k v)

/-! ## Structural Classification -/

/-- Coarse recurrence class. -/
inductive RecurrenceClass where
  | affineMatrixState
  | nonlinearVectorState
  | nonlinearMatrixState
  | nonlinearDeltaMatrixState
  deriving Repr, DecidableEq

/-- Which side the previous state is multiplied on before the nonlinear map. -/
inductive TransitionSide where
  | none
  | left
  | right
  | bilateral
  | elementwise
  deriving Repr, DecidableEq

/-- Structural feature signature for comparing matrix-state RNNs. -/
structure MatrixRecurrenceFeatures where
  recurrenceClass : RecurrenceClass
  transitionSide : TransitionSide
  learnedFixedTransition : Bool
  inputDependentTransition : Bool
  rawOuterWrite : Bool
  deltaWrite : Bool
  externalForgetCarry : Bool
  residualValuePath : Bool
  queryReadout : Bool
  deriving Repr, DecidableEq

/-- M2RNN's feature signature. -/
def m2rnnFeatures : MatrixRecurrenceFeatures where
  recurrenceClass := RecurrenceClass.nonlinearMatrixState
  transitionSide := TransitionSide.right
  learnedFixedTransition := true
  inputDependentTransition := false
  rawOuterWrite := true
  deltaWrite := false
  externalForgetCarry := true
  residualValuePath := true
  queryReadout := true

/-- E88's delta-rule feature signature. -/
def e88Features : MatrixRecurrenceFeatures where
  recurrenceClass := RecurrenceClass.nonlinearDeltaMatrixState
  transitionSide := TransitionSide.left
  learnedFixedTransition := false
  inputDependentTransition := true
  rawOuterWrite := false
  deltaWrite := true
  externalForgetCarry := false
  residualValuePath := false
  queryReadout := true

/-- M2RNN and E88 are both matrix-state nonlinear recurrent architectures. -/
def isNonlinearMatrixFamily (features : MatrixRecurrenceFeatures) : Bool :=
  features.recurrenceClass = RecurrenceClass.nonlinearMatrixState ||
  features.recurrenceClass = RecurrenceClass.nonlinearDeltaMatrixState

theorem m2rnn_is_nonlinear_matrix_family :
    isNonlinearMatrixFamily m2rnnFeatures = true := by
  rfl

theorem e88_is_nonlinear_matrix_family :
    isNonlinearMatrixFamily e88Features = true := by
  rfl

/-- Despite belonging to the same high-level family, M2RNN and E88 differ
structurally. -/
theorem m2rnn_features_not_e88_features :
    m2rnnFeatures ≠ e88Features := by
  decide

/-- The core axis: M2RNN uses a learned fixed right transition. -/
theorem m2rnn_has_learned_right_transition :
    m2rnnFeatures.learnedFixedTransition = true ∧
    m2rnnFeatures.transitionSide = TransitionSide.right := by
  constructor <;> rfl

/-- The core axis: E88 uses an input-dependent left transition induced by the
delta correction. -/
theorem e88_has_input_dependent_left_transition :
    e88Features.inputDependentTransition = true ∧
    e88Features.transitionSide = TransitionSide.left := by
  constructor <;> rfl

/-- The write-rule axis: M2RNN writes raw key/value associations; E88 writes
retrieval errors. -/
theorem write_rule_separates_m2rnn_and_e88 :
    m2rnnFeatures.rawOuterWrite = true ∧
    e88Features.deltaWrite = true := by
  constructor <;> rfl

/-! ## Reduction Hooks

The shared ancestor of both models is a raw-write nonlinear matrix RNN:

`H_t = tanh(H_{t-1} + k_t v_t^T)`.

M2RNN reaches this by setting `W = I` and `f = 0`.
E88 reaches this only after disabling delta correction and setting
`lambda = 1`; the real E88 delta rule instead expands to
`(lambda I - k k^T) H + k v^T`.
-/

/-- Raw-write nonlinear matrix recurrence used as a common ancestor. -/
noncomputable def rawWriteNonlinearUpdate
    (H : MatState K V) (k : Vec K) (v : Vec V) : MatState K V :=
  matrixTanh (H + outerKV k v)

/-- M2RNN with identity transition and no forget carry has the raw-write core. -/
theorem m2rnn_identity_no_forget_is_raw_write
    (H : MatState K V) (k : Vec K) (v : Vec V) :
    m2rnnUpdate (1 : Matrix (Fin V) (Fin V) Real) 0 H k v =
      rawWriteNonlinearUpdate H k v := by
  simp [m2rnnUpdate, m2rnnCandidate, rawWriteNonlinearUpdate]

end M2RNNComparison
