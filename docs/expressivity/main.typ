// Expressivity Analysis: Temporal Nonlinearity vs Depth
// A Formal Analysis of E88, Mamba2, and Modern RNN Architectures
//
// Compiled from ElmanProofs Lean 4 formalizations

#set document(
  title: "Expressivity Analysis: Temporal Nonlinearity vs Depth",
  author: "Erik Garrison",
)

// Sans-serif font throughout
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set page(margin: (x: 1in, y: 1in), numbering: "1")

// Prevent page breaks before headings
#show heading.where(level: 1): it => {
  v(1.5em)
  align(left)[#text(size: 16pt, weight: "bold", it.body)]
  v(0.75em)
}

#show heading.where(level: 2): it => {
  v(1em)
  align(left)[#text(size: 13pt, weight: "bold", it.body)]
  v(0.5em)
}

#show heading.where(level: 3): it => {
  v(0.75em)
  align(left)[#text(size: 11pt, weight: "bold", it.body)]
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
  #text(size: 24pt, weight: "bold")[Proof-Guided Architecture Exploration]
  #v(0.5em)
  #text(size: 18pt)[Temporal Nonlinearity vs Depth]
  #v(1em)
  #text(size: 14pt, style: "italic")[Using Formal Proofs to Design Sequence Models]
  #v(2em)
  #text(size: 11pt)[Erik Garrison]
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

We demonstrate a methodology for architecture design: use formal impossibility proofs to explore the design space systematically. Formalizing linear-temporal recurrence reveals what such models cannot compute (running parity, threshold functions), which in turn reveals what architectural features are necessary (temporal nonlinearity). We establish a strict computational hierarchy with each separation witnessed by constructive examples. The method guides architecture exploration by converting empirical questions ("which architecture is better?") into mathematical constraints ("which functions are computable?"). All theorems mechanically verified in Lean 4.

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

// Appendix: Formal Proofs
#include "appendix-proofs.typ"

// Conclusion
#pagebreak()

= Conclusion

We began with a question: where should nonlinearity live? The answer, we have seen, determines fundamental computational limits.

Transformers place nonlinearity between layers. State-space models like Mamba2 do the same, but process time linearly within each layer. E88 places nonlinearity in time itself, with $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ compounding across timesteps.

These choices create the strict computational hierarchy established in the introduction: Linear-temporal models fall below TC⁰---they cannot compute even parity. Transformers sit at TC⁰, with constant depth. E88 exceeds TC⁰, its depth growing with sequence length. E23 achieves full Turing completeness through explicit tape.

The separation is witnessed by concrete functions. Running parity: impossible for any linear-temporal model, achievable by single-layer E88. Threshold: impossible for linear (continuous functions cannot equal discontinuous ones), achievable by E88 via saturation. The proofs are complete, verified in Lean 4, with no gaps.

Theory and practice partially align. At short context (512 tokens), Mamba2 outperforms E88 on language modeling despite being provably less expressive---training dynamics dominate. At 32K context, CMA-ES architecture search inverts the ranking: E88 achieves 1.100 vs Mamba2's 1.188, leading by 0.088 nats. The compositional depth advantage manifests where sequence length makes it matter. Expressivity determines what _can_ be computed; context length determines whether that computation is required.

The practical implications follow from the theory. For tasks whose composition depth is bounded by $D = 32$, linear-temporal models suffice---and train faster. For algorithmic reasoning, state tracking, and any task requiring temporal decisions, the linear-temporal limitation bites. Depth adds nonlinearity in the wrong dimension; only temporal nonlinearity provides depth through time.

Chain-of-thought, the emergent tape, and output feedback all work because they provide working memory---not magical reasoning ability. When a model writes output and reads it back, it creates external storage that overcomes fixed state limitations. This is computation, not cognition.

The reader now understands the hierarchy of sequence models. Linear-temporal architectures are fast and sufficient for most natural language. Nonlinear-temporal architectures are slower but strictly more powerful. The choice depends on the task. For pattern aggregation, linear suffices. For sequential reasoning, nonlinearity is mathematically required.

The question of where nonlinearity should live has an answer: it depends on what you need to compute. And now we know, with mathematical certainty, what each choice can and cannot do.

#pagebreak()

= References

== Formal Proofs

The Lean 4 formalizations are available in the ElmanProofs repository:

#block(inset: (left: 1em))[
  `LinearCapacity.lean` — Linear RNN state as weighted sum of inputs \
  `LinearLimitations.lean` — Core impossibility results: threshold, XOR, parity \
  `MultiLayerLimitations.lean` — Multi-layer extension of impossibility results \
  `TanhSaturation.lean` — Saturation dynamics, bifurcation, latching \
  `ExactCounting.lean` — Threshold and counting impossibility/possibility \
  `RunningParity.lean` — Parity impossibility and E88 construction \
  `E23_DualMemory.lean` — E23 tape-based memory formalization \
  `MatrixStateRNN.lean` — E88 matrix state formalization \
  `MultiHeadTemporalIndependence.lean` — Head independence theorem \
  `E23vsE88Comparison.lean` — Direct comparison of E23 and E88 capabilities \
  `AttentionPersistence.lean` — Bifurcation and fixed point analysis \
  `OutputFeedback.lean` — Emergent tape memory and CoT equivalence \
  `TC0Bounds.lean` — TC⁰ circuit complexity bounds for Transformers \
  `TC0VsUnboundedRNN.lean` — Hierarchy: Linear SSM < TC⁰ < E88 \
  `ComputationalClasses.lean` — Chomsky hierarchy for RNNs \
  `MultiPass.lean` — Multi-pass RNN computational class with tape modification \
  `E88MultiPass.lean` — E88 multi-pass depth hierarchy and random access theorems \
  `RecurrenceLinearity.lean` — Architecture classification by recurrence type
]

#v(1em)

To verify: clone the repository and run `lake build`.

== Bibliography

#bibliography("references.yml", style: "ieee")

#v(2em)

#align(center)[
  #text(size: 9pt, style: "italic")[
    Document generated from ElmanProofs Lean 4 formalizations. \
    All core expressivity theorems mechanically verified.
  ]
]
