/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Nat.Basic
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Architectures.M2RNNComparison

/-!
# Resource-Bounded Formalism for Recurrent Architectures

This module records the comparison frame we need for E88/NDM, M2RNN, Gated
DeltaNet, and Mamba2.

The point is not to argue over the broad phrase "matrix state". Matrix state is
common. The useful questions are:

* Does temporal nonlinearity occur inside the recurrent state path?
* Is the stack pure recurrent or hybridized with attention/linear layers?
* What memory semantics does the update implement?
* How many independent recurrent programs are exposed to the machine?
* Are two architectures the same one-step update family, or merely in the same
  broad nonlinear temporal class?

The definitions here are deliberately lightweight. They are intended as a
stable vocabulary for later theorems about one-step transition separations,
resource-bounded expressivity, and empirical protocol obligations.
-/

namespace RecurrentResourceFormalism

/-! ## Structural Axes -/

/-- Shape of persistent recurrent state. -/
inductive StateGeometry where
  | vector
  | matrix
  | externalTape
  | attentionKV
  deriving Repr, DecidableEq

/-- Where, if anywhere, the temporal nonlinearity is applied. -/
inductive TemporalNonlinearity where
  | none
  | candidateOnly
  | fullState
  | gatedFullState
  deriving Repr, DecidableEq

/-- Side on which the previous state is transformed before the update. -/
inductive TransitionSide where
  | none
  | left
  | right
  | bilateral
  | elementwise
  deriving Repr, DecidableEq

/-- Whether the state transition is fixed, input-dependent, or state-dependent. -/
inductive TransitionControl where
  | fixedLearned
  | inputDependent
  | stateDependent
  | inputAndStateDependent
  deriving Repr, DecidableEq

/-- How new information enters memory. -/
inductive WriteRule where
  | none
  | rawOuterProduct
  | selectiveEraseWrite
  | deltaCorrecting
  | attentionCopy
  deriving Repr, DecidableEq

/-- Where linear carry/highway paths sit relative to the nonlinear candidate. -/
inductive CarryPlacement where
  | none
  | insideNonlinearity
  | outsideNonlinearity
  | residualReadout
  deriving Repr, DecidableEq

/-- Operational memory semantics, distinct from raw tensor shape. -/
inductive MemorySemantics where
  | fadingAccumulator
  | rawAssociativeTable
  | errorCorrectingAssociativeTable
  | latchAttractor
  | copyBuffer
  deriving Repr, DecidableEq

/-- Hardware-level execution model for the recurrent part. -/
inductive ImplementationMode where
  | parallelScan
  | sequentialSingleProgram
  | sequentialManyProgram
  | hybrid
  | quadraticAttention
  deriving Repr, DecidableEq

/-- Boolean version of "has recurrent temporal nonlinearity". -/
def hasTemporalNonlinearity : TemporalNonlinearity → Bool
  | .none => false
  | .candidateOnly => true
  | .fullState => true
  | .gatedFullState => true

/-- Boolean version of "uses delta/error-correcting memory writes". -/
def hasDeltaCorrectingWrite : WriteRule → Bool
  | .deltaCorrecting => true
  | _ => false

/-- Boolean version of the broader GDN/E88 delta-family write axis. GDN uses a
linear selective erase/write delta rule; E88/NDM uses a nonlinear
delta-correcting version. -/
def hasDeltaStyleWrite : WriteRule → Bool
  | .selectiveEraseWrite => true
  | .deltaCorrecting => true
  | _ => false

/-- Boolean version of "uses matrix-valued persistent state". -/
def hasMatrixState : StateGeometry → Bool
  | .matrix => true
  | _ => false

/-- Architecture feature signature for resource-bounded comparisons. -/
structure ArchitectureSignature where
  name : String
  stateGeometry : StateGeometry
  temporalNonlinearity : TemporalNonlinearity
  transitionSide : TransitionSide
  transitionControl : TransitionControl
  writeRule : WriteRule
  carryPlacement : CarryPlacement
  memorySemantics : MemorySemantics
  implementationMode : ImplementationMode
  totalLayers : Nat
  nonlinearTemporalLayers : Nat
  headsPerLayer : Nat
  stateScalarsPerHead : Nat
  pureRecurrentStack : Bool
  scanCompatible : Bool
  deriving Repr, DecidableEq

/-! ## Resource Measures -/

/-- Number of independent recurrent programs available per token and batch item. -/
def programsPerBatchToken (a : ArchitectureSignature) (batch : Nat) : Nat :=
  a.totalLayers * a.headsPerLayer * batch

/-- Total recurrent state scalars per layer, ignoring batch. -/
def stateScalarsPerLayer (a : ArchitectureSignature) : Nat :=
  a.headsPerLayer * a.stateScalarsPerHead

/-- Total recurrent state scalars in the stack, ignoring batch. -/
def stateScalarsPerStack (a : ArchitectureSignature) : Nat :=
  a.totalLayers * stateScalarsPerLayer a

/-- Sequential recurrent steps exposed by a full training chunk. -/
def sequentialProgramSteps (a : ArchitectureSignature) (batch seqLen : Nat) : Nat :=
  programsPerBatchToken a batch * seqLen

/-- All recurrent layers are temporally nonlinear. -/
def allLayersTemporallyNonlinear (a : ArchitectureSignature) : Bool :=
  a.nonlinearTemporalLayers == a.totalLayers

/-- Pure recurrent stack with temporal nonlinearity in every layer. -/
def isPureNonlinearRecurrentStack (a : ArchitectureSignature) : Bool :=
  a.pureRecurrentStack &&
  allLayersTemporallyNonlinear a &&
  hasTemporalNonlinearity a.temporalNonlinearity

/-- The resource-bounded comparison mode: not broad computability, but capability
under fixed wallclock/memory/precision/training protocol. -/
inductive ComparisonMode where
  | broadComputabilityClass
  | oneStepTransitionFamily
  | resourceBoundedCapability
  | trainabilityAndStability
  | hardwareUtilization
  deriving Repr, DecidableEq

/-! ## Canonical Signatures -/

/-- E88/NDM: pure nonlinear delta memory.

The production 1.27B run currently instantiates this with 12 layers, 370 heads
per layer, and 32×32 state per head.
-/
def e88NDM (layers heads nState : Nat) : ArchitectureSignature where
  name := "E88/NDM"
  stateGeometry := .matrix
  temporalNonlinearity := .fullState
  transitionSide := .left
  transitionControl := .inputDependent
  writeRule := .deltaCorrecting
  carryPlacement := .insideNonlinearity
  memorySemantics := .errorCorrectingAssociativeTable
  implementationMode := .sequentialManyProgram
  totalLayers := layers
  nonlinearTemporalLayers := layers
  headsPerLayer := heads
  stateScalarsPerHead := nState * nState
  pureRecurrentStack := true
  scanCompatible := false

/-- Homogeneous/pure M2RNN: nonlinear matrix-state recurrence, but raw writes and
a fixed learned right transition. -/
def m2rnnPure (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "M2RNN pure"
  stateGeometry := .matrix
  temporalNonlinearity := .candidateOnly
  transitionSide := .right
  transitionControl := .fixedLearned
  writeRule := .rawOuterProduct
  carryPlacement := .outsideNonlinearity
  memorySemantics := .rawAssociativeTable
  implementationMode := .sequentialManyProgram
  totalLayers := layers
  nonlinearTemporalLayers := layers
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := false

/-- Hybrid M2RNN/attention or M2RNN/GDN stack. It may contain nonlinear recurrent
layers, but the stack is not pure recurrent. -/
def m2rnnHybrid
    (layers nonlinearLayers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "M2RNN hybrid"
  stateGeometry := .matrix
  temporalNonlinearity := .candidateOnly
  transitionSide := .right
  transitionControl := .fixedLearned
  writeRule := .rawOuterProduct
  carryPlacement := .outsideNonlinearity
  memorySemantics := .rawAssociativeTable
  implementationMode := .hybrid
  totalLayers := layers
  nonlinearTemporalLayers := nonlinearLayers
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := false
  scanCompatible := false

/-- Gated DeltaNet / FLA-GDN: matrix state and delta-style associative memory,
but linear/affine in the recurrent state for fixed inputs. -/
def gatedDeltaNet (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "Gated DeltaNet"
  stateGeometry := .matrix
  temporalNonlinearity := .none
  transitionSide := .left
  transitionControl := .inputDependent
  writeRule := .selectiveEraseWrite
  carryPlacement := .none
  memorySemantics := .rawAssociativeTable
  implementationMode := .parallelScan
  totalLayers := layers
  nonlinearTemporalLayers := 0
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := true

/-- Mamba2-like selective SSM: vector/diagonal state, input-dependent but linear
in the temporal state. -/
def mamba2SSM (layers heads stateScalarsPerHead : Nat) : ArchitectureSignature where
  name := "Mamba2"
  stateGeometry := .vector
  temporalNonlinearity := .none
  transitionSide := .elementwise
  transitionControl := .inputDependent
  writeRule := .none
  carryPlacement := .none
  memorySemantics := .fadingAccumulator
  implementationMode := .parallelScan
  totalLayers := layers
  nonlinearTemporalLayers := 0
  headsPerLayer := heads
  stateScalarsPerHead := stateScalarsPerHead
  pureRecurrentStack := true
  scanCompatible := true

/-- Concrete current E88/NDM production geometry. -/
def e88NDM_1p27B : ArchitectureSignature :=
  e88NDM 12 370 32

/-! ## Basic Theorems -/

theorem e88NDM_1p27B_is_pure_nonlinear_recurrent_stack :
    isPureNonlinearRecurrentStack e88NDM_1p27B = true := by
  rfl

theorem e88NDM_1p27B_has_delta_memory :
    hasDeltaCorrectingWrite e88NDM_1p27B.writeRule = true := by
  rfl

theorem e88NDM_1p27B_has_matrix_state :
    hasMatrixState e88NDM_1p27B.stateGeometry = true := by
  rfl

theorem e88NDM_1p27B_programs_per_batch_token (batch : Nat) :
    programsPerBatchToken e88NDM_1p27B batch = 12 * 370 * batch := by
  rfl

theorem e88NDM_1p27B_programs_per_batch_token_bs5 :
    programsPerBatchToken e88NDM_1p27B 5 = 22200 := by
  rfl

theorem e88NDM_1p27B_state_scalars_per_layer :
    stateScalarsPerLayer e88NDM_1p27B = 370 * (32 * 32) := by
  rfl

theorem pure_m2rnn_is_nonlinear_matrix_recurrent
    (layers heads state : Nat) :
    hasMatrixState (m2rnnPure layers heads state).stateGeometry = true ∧
    hasTemporalNonlinearity (m2rnnPure layers heads state).temporalNonlinearity = true := by
  constructor <;> rfl

theorem pure_m2rnn_is_not_delta_correcting
    (layers heads state : Nat) :
    hasDeltaCorrectingWrite (m2rnnPure layers heads state).writeRule = false := by
  rfl

theorem hybrid_m2rnn_is_not_pure_recurrent_stack
    (layers nonlinearLayers heads state : Nat) :
    (m2rnnHybrid layers nonlinearLayers heads state).pureRecurrentStack = false := by
  rfl

theorem gated_delta_net_has_matrix_state_but_no_temporal_nonlinearity
    (layers heads state : Nat) :
    hasMatrixState (gatedDeltaNet layers heads state).stateGeometry = true ∧
    hasTemporalNonlinearity (gatedDeltaNet layers heads state).temporalNonlinearity = false := by
  constructor <;> rfl

theorem e88_and_gdn_share_delta_style_write
    (layers heads state : Nat) :
    hasDeltaStyleWrite e88NDM_1p27B.writeRule = true ∧
    hasDeltaStyleWrite (gatedDeltaNet layers heads state).writeRule = true := by
  constructor <;> rfl

theorem e88_and_gdn_split_on_temporal_nonlinearity
    (layers heads state : Nat) :
    hasTemporalNonlinearity e88NDM_1p27B.temporalNonlinearity = true ∧
    hasTemporalNonlinearity (gatedDeltaNet layers heads state).temporalNonlinearity = false := by
  constructor <;> rfl

theorem mamba2_has_no_temporal_nonlinearity
    (layers heads state : Nat) :
    hasTemporalNonlinearity (mamba2SSM layers heads state).temporalNonlinearity = false := by
  rfl

/-- E88/NDM and homogeneous M2RNN share the broad nonlinear matrix-state family. -/
theorem e88_and_m2rnn_share_broad_nonlinear_matrix_family
    (m2Layers m2Heads m2State : Nat) :
    hasMatrixState e88NDM_1p27B.stateGeometry = true ∧
    hasTemporalNonlinearity e88NDM_1p27B.temporalNonlinearity = true ∧
    hasMatrixState (m2rnnPure m2Layers m2Heads m2State).stateGeometry = true ∧
    hasTemporalNonlinearity (m2rnnPure m2Layers m2Heads m2State).temporalNonlinearity = true := by
  constructor
  · rfl
  constructor
  · rfl
  constructor <;> rfl

/-- But E88/NDM and M2RNN are not the same one-step transition family. -/
theorem e88_and_m2rnn_differ_as_one_step_transition_families
    (m2Layers m2Heads m2State : Nat) :
    e88NDM_1p27B.transitionSide ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionSide ∧
    e88NDM_1p27B.writeRule ≠
      (m2rnnPure m2Layers m2Heads m2State).writeRule ∧
    e88NDM_1p27B.transitionControl ≠
      (m2rnnPure m2Layers m2Heads m2State).transitionControl := by
  constructor
  · intro h
    cases h
  constructor
  · intro h
    cases h
  · intro h
    cases h

/-- The M2RNNComparison scaffold and this resource formalism agree on the delta
write axis for E88. -/
theorem agrees_with_m2rnn_comparison_on_e88_delta_axis :
    M2RNNComparison.e88Features.deltaWrite = true ∧
    hasDeltaCorrectingWrite e88NDM_1p27B.writeRule = true := by
  constructor <;> rfl

/-- The M2RNNComparison scaffold and this resource formalism agree that M2RNN is
not a delta-write architecture. -/
theorem agrees_with_m2rnn_comparison_on_m2rnn_raw_write_axis :
    M2RNNComparison.m2rnnFeatures.deltaWrite = false ∧
    hasDeltaCorrectingWrite (m2rnnPure 1 1 1).writeRule = false := by
  constructor <;> rfl

/-! ## Concrete One-Step Transition Separation -/

abbrev TwoVec := M2RNNComparison.Vec 2
abbrev TwoMat := Matrix (Fin 2) (Fin 2) Real

/-- First basis key in a two-dimensional key space. -/
def key0 : TwoVec :=
  fun i => if i = 0 then 1 else 0

/-- Second basis key in a two-dimensional key space. -/
def key1 : TwoVec :=
  fun i => if i = 1 then 1 else 0

/-- E88's expanded delta transition is genuinely key-dependent: two different
keys induce two different left-transition matrices.

This is the concrete core of the one-step transition-family separation. M2RNN's
learned right transition `W` is fixed across keys inside a layer; E88's delta
rule induces `lambda I - k k^T`, which changes with the current key. -/
theorem e88_two_keys_induce_distinct_left_transitions :
    M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 ≠
      M2RNNComparison.e88DeltaTransition (K := 2) 1 key1 := by
  intro h
  have h00 := congrArg (fun M : TwoMat => M 0 0) h
  simp [M2RNNComparison.e88DeltaTransition, M2RNNComparison.outerKV, key0, key1] at h00

/-- No fixed two-dimensional transition matrix can equal E88's key-dependent
transition for both basis keys. -/
theorem no_fixed_transition_matches_e88_two_basis_keys
    (A : TwoMat) :
    ¬ (A = M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 ∧
       A = M2RNNComparison.e88DeltaTransition (K := 2) 1 key1) := by
  intro h
  exact e88_two_keys_induce_distinct_left_transitions (h.1.symm.trans h.2)

/-- M2RNN preactivation in the 2×2 witness setting:
`H W + k vᵀ`. -/
def m2rnnPreactivation2
    (W H : TwoMat) (k v : TwoVec) : TwoMat :=
  H * W + M2RNNComparison.outerKV k v

/-- E88/NDM expanded delta preactivation in the 2×2 witness setting:
`(I - k kᵀ) H + k vᵀ`. -/
def e88DeltaPreactivation2
    (H : TwoMat) (k v : TwoVec) : TwoMat :=
  M2RNNComparison.e88DeltaTransition (K := 2) 1 k * H +
    M2RNNComparison.outerKV k v

/-- No fixed M2RNN right-transition matrix can make the M2RNN preactivation
family match E88's key-dependent delta preactivation for two basis keys.

This is the one-step transition-family version of the separation. M2RNN's
candidate path has a fixed right transition `W`; E88/NDM's expanded delta rule
has an input-dependent left transition `I - k kᵀ`. If the two families matched
for all states and values at both `key0` and `key1`, then testing on `H = I`
and `v = 0` would force the same fixed `W` to equal two distinct E88
transitions. -/
theorem no_fixed_m2rnn_preactivation_matches_e88_two_basis_keys
    (W : TwoMat) :
    ¬ ((∀ H v,
          m2rnnPreactivation2 W H key0 v =
            e88DeltaPreactivation2 H key0 v) ∧
       (∀ H v,
          m2rnnPreactivation2 W H key1 v =
            e88DeltaPreactivation2 H key1 v)) := by
  intro h
  have hw0 :
      W = M2RNNComparison.e88DeltaTransition (K := 2) 1 key0 := by
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.outerKV] using h.1 (1 : TwoMat) (0 : TwoVec)
  have hw1 :
      W = M2RNNComparison.e88DeltaTransition (K := 2) 1 key1 := by
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.outerKV] using h.2 (1 : TwoMat) (0 : TwoVec)
  exact no_fixed_transition_matches_e88_two_basis_keys W ⟨hw0, hw1⟩

/-- The same fixed-transition separation lifts through the `tanh` candidate
path. Since `tanh` is injective over reals, uniformly matching the M2RNN
candidate `tanh(H W + k vᵀ)` to the E88 candidate
`tanh((I - k kᵀ) H + k vᵀ)` would imply equality of the preactivations, which
the previous theorem rules out for two basis keys.

This still isolates the candidate path. The external M2RNN forget interpolation
is a further resource axis handled by later theorems. -/
theorem no_fixed_m2rnn_candidate_matches_e88_two_basis_keys
    (W : TwoMat) :
    ¬ ((∀ H v,
          M2RNNComparison.m2rnnCandidate W H key0 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key0 v) ∧
       (∀ H v,
          M2RNNComparison.m2rnnCandidate W H key1 v =
            M2RNNComparison.e88DeltaUpdateExpanded 1 H key1 v)) := by
  intro h
  apply no_fixed_m2rnn_preactivation_matches_e88_two_basis_keys W
  constructor
  · intro H v
    ext i j
    have hij := congrArg (fun M : TwoMat => M i j) (h.1 H v)
    have hpre := Activation.tanh_injective hij
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.m2rnnCandidate, M2RNNComparison.e88DeltaUpdateExpanded,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap] using hpre
  · intro H v
    ext i j
    have hij := congrArg (fun M : TwoMat => M i j) (h.2 H v)
    have hpre := Activation.tanh_injective hij
    simpa [m2rnnPreactivation2, e88DeltaPreactivation2,
      M2RNNComparison.m2rnnCandidate, M2RNNComparison.e88DeltaUpdateExpanded,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap] using hpre

/-! ## Interpretation Hooks

These are not capability theorems yet. They are hooks for the theorems we want:

* broad nonlinear temporal class: E88/NDM and M2RNN are likely not separated
  by classical computability class alone;
* one-step transition family: E88/NDM and M2RNN are structurally separated;
* resource-bounded capability: compare at fixed params, wallclock, memory,
  precision, optimizer, tokens, and context length;
* hardware utilization: E88/NDM exposes many independent small recurrent
  programs, making pure nonlinear recurrence trainable at scale.
-/

end RecurrentResourceFormalism
