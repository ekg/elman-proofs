// Section 12: The Formal Verification System
// ElmanProofs: Machine-Checked Mathematical Certainty

= The Formal Verification System

This section describes the Lean 4 formalization underlying all results in this document. Unlike typical mathematical arguments in machine learning papers---which may contain subtle errors, unstated assumptions, or incomplete reasoning---the proofs here have been *machine-checked*. Every theorem is verified by the Lean proof assistant, providing the gold standard of mathematical certainty.

== Why Formal Verification Matters

Mathematical proofs in ML papers often contain gaps:

- *Implicit assumptions*: "Assume the model is well-behaved" without specifying what this means
- *Hand-waving*: "The rest follows by standard arguments" when the "standard arguments" are subtle
- *Notational ambiguity*: Overloaded symbols that mean different things in different contexts
- *Unverified bounds*: "For sufficiently large $n$..." without specifying how large

Formal verification eliminates these issues. When we say "linear RNNs cannot compute XOR," we mean:

1. There is a precise mathematical definition of "linear RNN"
2. There is a precise mathematical definition of "XOR function"
3. The Lean type checker has verified that no linear RNN matches the XOR function
4. Every logical step in the proof has been mechanically checked

This is not a sketch-level argument. It is mathematical certainty.

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *The Gold Standard*: Formal proofs in Lean 4 with Mathlib provide the same level of rigor as proofs in pure mathematics. The computer verifies every logical step. Errors are impossible---either the proof type-checks, or it doesn't.
]

== Repository Structure

The proofs are organized in the *ElmanProofs* repository: `github.com/ekg/elman-proofs`

#figure(
  table(
    columns: 2,
    stroke: 0.5pt,
    align: (left, left),
    [*Directory*], [*Contents*],
    [`ElmanProofs/Expressivity/`], [Core expressivity theorems: linear limitations, separation results, tanh dynamics],
    [`ElmanProofs/Architectures/`], [Formalized architectures: E1, E88, Mamba2, E23, FLA-GDN],
    [`ElmanProofs/Activations/`], [Activation function properties: Lipschitz bounds, saturation],
    [`ElmanProofs/Dynamics/`], [Dynamical systems: contraction, attractors, gradient flow],
    [`ElmanProofs/Information/`], [Computational complexity: linear vs nonlinear, composition depth],
    [`ElmanProofs/Memory/`], [Memory capacity: attractors, retrieval guarantees],
    [`ElmanProofs/Gradient/`], [Gradient analysis: flow, stability, convergence],
  ),
  caption: [Repository structure of ElmanProofs],
)

The proofs build on *Mathlib*, the standard mathematics library for Lean 4. Mathlib provides the foundational mathematics: real analysis, linear algebra, topology, and measure theory.

== Key Proof Files

The central results come from these files:

=== Core Impossibility Results

#block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 10pt,
  radius: 4pt,
)[
  *LinearCapacity.lean* (254 lines)

  Proves that linear RNN state is a weighted sum of inputs:
  $ h_T = sum_(t=0)^(T-1) A^(T-1-t) B x_t $

  Key theorems:
  - `linear_state_is_sum`: State at time $T$ is exactly the weighted sum
  - `state_additive`: State is additive in inputs
  - `state_scalar`: State is homogeneous in scalar multiplication
  - `reachable_dim_bound`: Reachable states have dimension at most $n$
]

#block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 10pt,
  radius: 4pt,
)[
  *LinearLimitations.lean* (339 lines)

  Proves what linear RNNs *cannot* compute:
  - `linear_cannot_threshold`: Threshold functions are impossible (line 107)
  - `xor_not_affine`: XOR is not an affine function (line 218)
  - `linear_cannot_xor`: Linear RNNs cannot compute XOR (line 315)

  The proofs use the algebraic structure of linear functions: additivity and homogeneity imply continuity, but threshold and XOR are discontinuous or non-affine.
]

#block(
  fill: rgb("#fff7f0"),
  stroke: rgb("#cc6633"),
  inset: 10pt,
  radius: 4pt,
)[
  *MultiLayerLimitations.lean* (448 lines)

  Extends impossibility to multi-layer architectures:
  - `layer_output_as_weighted_sum`: Each layer's output is a weighted sum of its inputs
  - `multilayer_cannot_running_threshold`: $D$-layer linear-temporal models cannot compute running threshold (line 231)
  - `multilayer_cannot_threshold`: Original threshold is also impossible
  - `e88_separates_from_linear_temporal`: E88 computes functions that linear-temporal cannot

  The key insight: stacking layers adds *depth* but not *temporal nonlinearity*. Each layer still aggregates linearly across time.
]

=== Running Parity and XOR Extensions

#block(
  fill: rgb("#f7fff0"),
  stroke: rgb("#66cc33"),
  inset: 10pt,
  radius: 4pt,
)[
  *RunningParity.lean* (200+ lines)

  Extends XOR impossibility to arbitrary-length sequences:
  - `parity_T_not_affine`: Parity of $T >= 2$ bits is not affine (line 80)
  - `linear_cannot_running_parity`: Linear RNNs cannot compute running parity (line 200)

  The proof reduces parity to XOR: if parity on $T$ inputs were affine, restricting to positions 0 and 1 would give an affine function computing XOR---but XOR is not affine.
]

=== Tanh Saturation and Binary Retention

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 10pt,
  radius: 4pt,
)[
  *TanhSaturation.lean* (800+ lines)

  Proves the saturation dynamics that enable E88's expressivity:
  - `tanh_saturates_to_one`: $tanh(x) arrow 1$ as $x arrow infinity$
  - `tanh_derivative_vanishes`: $tanh'(x) arrow 0$ as $|x| arrow infinity$
  - `tanhRecurrence_is_contraction`: Tanh recurrence is contractive for $|alpha| < 1$
  - `tanhRecurrence_unique_fixedpoint`: Unique fixed point exists
  - `near_saturation_low_gradient`: Near saturation, gradient is small (latching)

  This formalizes why tanh creates stable memory: once a state approaches $plus.minus 1$, the low gradient keeps it there.
]

#block(
  fill: rgb("#f0f7ff"),
  stroke: rgb("#3366cc"),
  inset: 10pt,
  radius: 4pt,
)[
  *BinaryFactRetention.lean* (200+ lines)

  Proves the gap between latching (E88) and decay (linear):
  - `linear_contribution_decays`: Linear state contribution decays as $alpha^t$
  - `linear_info_vanishes`: Information fades in linear-temporal systems
  - `linear_no_fixed_point`: No non-trivial fixed points in linear decay
  - `tanh_approaches_one_at_infinity`: Tanh enables stable non-zero fixed points

  This is the core separation: E88 can *latch* a binary fact; Mamba2's linear state *decays*.
]

=== Architecture Classification

#block(
  fill: rgb("#f7f0ff"),
  stroke: rgb("#9933cc"),
  inset: 10pt,
  radius: 4pt,
)[
  *RecurrenceLinearity.lean* (390 lines)

  Classifies architectures by recurrence type:
  - `minGRU_is_linear_in_h`: MinGRU is linear in $h$: $h_t = (1-z_t) dot h_(t-1) + z_t dot tilde(h)_t$
  - `e1_is_nonlinear_in_h`: E1 is nonlinear in $h$: $h_t = tanh(W dot h_(t-1) + ...)$
  - `mamba2_is_linear_in_h`: Mamba2 SSM is linear in $h$: $h_t = A(x) dot h_(t-1) + B(x) dot x_t$
  - `within_layer_depth`: Linear = 1 composition, nonlinear = $T$ compositions (line 215)
  - `e1_more_depth_than_minGRU`: E1 has more composition depth than MinGRU

  This explains the hierarchy: *"Nonlinearity flows down (through layers), not forward (through time)."*
]

== Proof Verification Status

The core expressivity proofs are *fully verified*---no axiom holes, no `sorry` statements, no unproven assumptions.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    align: (left, center, left),
    [*File*], [*Status*], [*Key Theorems*],
    [LinearCapacity.lean], [$checkmark$ Complete], [linear_state_is_sum, state_additive],
    [LinearLimitations.lean], [$checkmark$ Complete], [linear_cannot_threshold, linear_cannot_xor],
    [MultiLayerLimitations.lean], [$checkmark$ Complete], [multilayer_cannot_running_threshold],
    [RunningParity.lean], [$checkmark$ Complete], [parity_T_not_affine, linear_cannot_running_parity],
    [TanhSaturation.lean], [$checkmark$ Complete], [tanhRecurrence_unique_fixedpoint],
    [BinaryFactRetention.lean], [$checkmark$ Complete], [linear_contribution_decays],
    [RecurrenceLinearity.lean], [$checkmark$ Complete], [mamba2_is_linear_in_h, within_layer_depth],
  ),
  caption: [Verification status of core expressivity proofs. All central results are machine-checked.],
)

Some peripheral files contain proofs-in-progress (marked with `sorry` in Lean), particularly:
- Numerical bounds for transcendental functions (requires interval arithmetic)
- Some spectral/eigenvalue analysis
- Certain fixed point constructions

These do not affect the core separation results, which are fully verified.

== How to Read the Lean Code

For readers unfamiliar with Lean 4, here is a brief guide to understanding the proof files.

=== Basic Syntax

```lean
-- A theorem statement
theorem linear_cannot_threshold (τ : ℝ) (T : ℕ) (hT : T ≥ 1) :
    ¬ LinearlyComputable (thresholdFunction τ T) := by
  -- Proof tactics here
```

- `theorem name (args) : statement := by` declares a theorem
- `(hT : T ≥ 1)` is a hypothesis named `hT` saying $T >= 1$
- `¬` means "not" (negation)
- `by` introduces a tactic proof

=== Common Patterns

```lean
-- Definition of a function
def thresholdFunction (τ : ℝ) (T : ℕ) : (Fin T → (Fin 1 → ℝ)) → (Fin 1 → ℝ) :=
  fun inputs =>
    let total := ∑ t : Fin T, inputs t 0
    fun _ => if total > τ then 1 else 0

-- Proof by contradiction
intro ⟨n, A, B, C, h_f⟩     -- Assume the negated statement holds
-- ... derive contradiction
```

=== Type Annotations

```lean
Matrix (Fin n) (Fin n) ℝ    -- n×n real matrix
Fin T → (Fin m → ℝ)         -- Sequence of T inputs, each of dimension m
```

=== Reading Strategy

1. *Start with theorem statements*: The `theorem` and `lemma` declarations tell you what is being proven
2. *Check the types*: Type annotations in definitions reveal the mathematical structure
3. *Skip the tactics*: The proof details (after `by`) are less important than the statements
4. *Look for `sorry`*: Any occurrence means the proof is incomplete

== Building and Verifying the Proofs

To verify the proofs yourself:

```bash
# Clone the repository
git clone https://github.com/ekg/elman-proofs
cd elman-proofs

# Build with Lake (Lean's package manager)
lake build

# If successful, all proofs have been verified
```

The build process checks every proof. If it completes without error, the mathematical content is verified.

Requirements:
- Lean 4 (v4.x)
- Mathlib (automatically fetched by Lake)

== What Formal Verification Guarantees

Formal verification provides specific guarantees:

#block(
  fill: rgb("#f0fff7"),
  stroke: rgb("#33cc66"),
  inset: 12pt,
  radius: 4pt,
)[
  *What It Guarantees*:
  1. *Logical validity*: Every proof step follows from the axioms and previous steps
  2. *Type correctness*: All mathematical objects are used consistently with their types
  3. *No hidden assumptions*: All hypotheses are explicitly stated
  4. *No gaps*: The proof is complete---no "exercise for the reader"
]

#block(
  fill: rgb("#fff0f0"),
  stroke: rgb("#cc3333"),
  inset: 12pt,
  radius: 4pt,
)[
  *What It Does NOT Guarantee*:
  1. *Relevance*: The theorem might not be what you actually care about
  2. *Applicability*: The mathematical model might not match the real-world system
  3. *Optimality*: A better result might exist
  4. *Efficiency*: The proof says nothing about computational cost
]

The gap between mathematical truth and practical relevance is bridged by careful modeling. Our definitions of "linear RNN" and "threshold function" are precise and match the architectures we care about.

== The Broader Context

Formal verification of ML theory is rare but growing. Notable examples:

- *Verified cryptography*: libsodium, HACL\*
- *Verified compilers*: CompCert (C compiler)
- *Verified operating systems*: seL4 microkernel

We contribute to this tradition by formally verifying the expressivity theory of sequence models. The goal: *move ML from empirical claims to mathematical certainty*.

#block(
  fill: rgb("#f7f7ff"),
  stroke: rgb("#6666cc"),
  inset: 12pt,
  radius: 4pt,
)[
  *Summary*: The ElmanProofs repository provides machine-checked proofs of the expressivity results in this document. Linear-temporal models *provably* cannot compute running parity, threshold, or XOR. E88's temporal nonlinearity *provably* overcomes these limitations. These are not conjectures---they are theorems verified by computer.

  Source: `github.com/ekg/elman-proofs`
]

