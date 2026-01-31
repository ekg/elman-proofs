// Expressivity Analysis: Temporal Nonlinearity vs Depth
// A Formal Analysis of E88, Mamba2, FLA, and GDN
//
// Compiled from ElmanProofs Lean 4 formalizations

#set document(
  title: "Expressivity Analysis: Temporal Nonlinearity vs Depth",
  author: "ElmanProofs Contributors",
)

// Sans-serif font throughout
#set text(font: "DejaVu Sans", size: 10.5pt)
#set par(justify: true, leading: 0.65em)
#set page(margin: (x: 1in, y: 1in), numbering: "1")

// Prevent page breaks before headings
#show heading.where(level: 1): it => {
  v(1.5em)
  text(size: 16pt, weight: "bold", it.body)
  v(0.75em)
}

#show heading.where(level: 2): it => {
  v(1em)
  text(size: 13pt, weight: "bold", it.body)
  v(0.5em)
}

#show heading.where(level: 3): it => {
  v(0.75em)
  text(size: 11pt, weight: "bold", it.body)
  v(0.25em)
}

// Code blocks
#show raw.where(block: true): it => block(
  fill: rgb("#f5f5f5"),
  stroke: rgb("#dddddd"),
  inset: 8pt,
  radius: 3pt,
  width: 100%,
  it
)

// Title page
#align(center)[
  #v(2in)
  #text(size: 24pt, weight: "bold")[Expressivity Analysis]
  #v(0.5em)
  #text(size: 18pt)[Temporal Nonlinearity vs Depth]
  #v(1em)
  #text(size: 14pt, style: "italic")[Where Should Nonlinearity Live?]
  #v(2em)
  #text(size: 11pt)[ElmanProofs Contributors]
  #v(0.5em)
  #text(size: 10pt)[January 2026]
  #v(2em)
  #line(length: 60%, stroke: 0.5pt)
  #v(1em)
  #text(size: 9pt)[
    All theorems formalized in Lean 4 with Mathlib. \
    Source: `ElmanProofs/Expressivity/`
  ]
]

#pagebreak()

// Table of contents
#outline(
  title: "Contents",
  indent: 1.5em,
)

#pagebreak()

// Abstract
#align(center)[
  #text(size: 14pt, weight: "bold")[Abstract]
]
#v(1em)

Every sequence model must answer a fundamental question: where should nonlinearity live? The answer determines computational limits that no amount of scaling can overcome.

This document develops the theory of _recurrence linearity_ and its consequences. We prove that models with linear temporal dynamics---Mamba2, Fast Linear Attention, Gated Delta Networks---are mathematically constrained in ways that models with nonlinear temporal dynamics are not. A $D$-layer linear-temporal model has composition depth $D$, regardless of sequence length. An E88-style model with nonlinear recurrence has composition depth $D times T$, where $T$ is the sequence length. The gap is multiplicative.

The consequences are concrete. Functions like running parity and threshold counting are provably impossible for linear-temporal models at any depth. E88 computes them with a single layer. The separation is not empirical---it is mathematical, verified in Lean 4 with no gaps in the proofs.

We also confront the tension between theory and practice: despite E88's provably greater expressivity, Mamba2 achieves better perplexity on language modeling benchmarks. This apparent contradiction resolves once we distinguish what an architecture _can compute_ from what it _learns efficiently_. The theoretical hierarchy concerns ultimate limits; training dynamics determine what happens within those limits.

The document traces a journey from the fundamental question---where should nonlinearity live?---through the mathematical machinery of linear recurrence, the impossibility results, E88's escape via tanh saturation, the circuit complexity perspective, output feedback and emergent tape, and finally to the practical implications. The reader emerges with a complete understanding of the expressivity hierarchy among modern sequence models.

#v(2em)

// Include sections
#include "01-introduction.typ"
#include "02-foundations.typ"
#include "03-linear-limitation.typ"
#include "04-e88.typ"
#include "05-e23-vs-e88.typ"
#include "06-separation.typ"
#include "07-implications.typ"
#include "08-tc0-bounds.typ"
#include "09-output-feedback.typ"
#include "10-multi-pass-rnn.typ"
#include "11-experimental-results.typ"
#include "12-formal-system.typ"
// 13-theory-practice-gap.typ - merged into 11
#include "14-composition-depth-text.typ"
// 15-uncanny-valley-reasoning.typ - merged into 14
#include "16-depth-examples.typ"

// Conclusion
#pagebreak()

= Conclusion

We began with a question: where should nonlinearity live? The answer, we have seen, determines fundamental computational limits.

Transformers place nonlinearity between layers. State-space models like Mamba2 do the same, but process time linearly within each layer. E88 places nonlinearity in time itself, with $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ compounding across timesteps.

These choices create a strict hierarchy:
$ "Linear SSM" subset.neq "TC"^0 "(Transformer)" subset.neq "E88" subset.neq "E23 (UTM)" $

Linear-temporal models fall below TC⁰---they cannot compute even parity. Transformers sit at TC⁰, with constant depth. E88 exceeds TC⁰, its depth growing with sequence length. E23 achieves full Turing completeness through explicit tape.

The separation is witnessed by concrete functions. Running parity: impossible for any linear-temporal model, achievable by single-layer E88. Threshold: impossible for linear (continuous functions cannot equal discontinuous ones), achievable by E88 via saturation. The proofs are complete, verified in Lean 4, with no gaps.

Yet theory is not practice. Mamba2 outperforms E88 on language modeling despite being provably less expressive. The resolution: expressivity determines what _can_ be computed with unlimited resources; benchmarks measure what is learned in fixed time on specific tasks. The theory tells us about limits; training dynamics tell us what happens within those limits.

The practical implications follow from the theory. For tasks whose composition depth is bounded by $D = 32$, linear-temporal models suffice---and train faster. For algorithmic reasoning, state tracking, and any task requiring temporal decisions, the linear-temporal limitation bites. Depth adds nonlinearity in the wrong dimension; only temporal nonlinearity provides depth through time.

Chain-of-thought, the emergent tape, and output feedback all work because they provide working memory---not magical reasoning ability. When a model writes output and reads it back, it creates external storage that overcomes fixed state limitations. This is computation, not cognition.

The reader now understands the hierarchy of sequence models. Linear-temporal architectures are fast and sufficient for most natural language. Nonlinear-temporal architectures are slower but strictly more powerful. The choice depends on the task. For pattern aggregation, linear suffices. For sequential reasoning, nonlinearity is mathematically required.

The question of where nonlinearity should live has an answer: it depends on what you need to compute. And now we know, with mathematical certainty, what each choice can and cannot do.

#pagebreak()

= References

The formal proofs are available in the ElmanProofs repository (`github.com/ekg/elman-proofs`):

- `LinearCapacity.lean` — Linear RNN state as weighted sum of inputs
- `LinearLimitations.lean` — Core impossibility results: threshold, XOR, parity
- `MultiLayerLimitations.lean` — Multi-layer extension of impossibility results
- `TanhSaturation.lean` — Saturation dynamics, bifurcation, latching
- `ExactCounting.lean` — Threshold and counting impossibility/possibility
- `RunningParity.lean` — Parity impossibility and E88 construction
- `E23_DualMemory.lean` — E23 tape-based memory formalization
- `MatrixStateRNN.lean` — E88 matrix state formalization
- `MultiHeadTemporalIndependence.lean` — Head independence theorem
- `E23vsE88Comparison.lean` — Direct comparison of E23 and E88 capabilities
- `AttentionPersistence.lean` — Bifurcation and fixed point analysis
- `OutputFeedback.lean` — Emergent tape memory and CoT equivalence
- `TC0Bounds.lean` — TC⁰ circuit complexity bounds for Transformers
- `TC0VsUnboundedRNN.lean` — Hierarchy: Linear SSM < TC⁰ < E88
- `ComputationalClasses.lean` — Chomsky hierarchy for RNNs
- `MultiPass.lean` — Multi-pass RNN computational class with tape modification (~2000 lines)
- `E88MultiPass.lean` — E88 multi-pass depth hierarchy and random access theorems
- `RecurrenceLinearity.lean` — Architecture classification by recurrence type

#v(2em)

#align(center)[
  #text(size: 9pt, style: "italic")[
    Document generated from ElmanProofs Lean 4 formalizations. \
    All core expressivity theorems mechanically verified. \
    To verify: clone the repository and run `lake build`.
  ]
]
