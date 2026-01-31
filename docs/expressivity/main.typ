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
  #text(size: 14pt, style: "italic")[A Formal Analysis of E88, Mamba2, FLA, and GDN]
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

This document presents a formal analysis of expressivity differences between sequence model architectures, focusing on where nonlinearity enters the computation. We prove that models with _linear temporal dynamics_ (Mamba2, Fast Linear Attention, Gated Delta Networks) have fundamentally limited expressivity compared to models with _nonlinear temporal dynamics_ (E88).

The key results:
- Linear-temporal models have composition depth $D$ (layer count), regardless of sequence length
- E88-style models have composition depth $D times T$ (layers times timesteps)
- Functions like running parity and threshold counting are _provably impossible_ for linear-temporal models
- E88's tanh saturation creates stable fixed points enabling binary memory

All proofs are mechanically verified in Lean 4, providing mathematical certainty about these architectural limitations.

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

// Final page
#pagebreak()

= References

The formal proofs are available in the ElmanProofs repository (`github.com/ekg/elman-proofs`):

- `LinearCapacity.lean` — Linear RNN state capacity
- `LinearLimitations.lean` — Core impossibility results
- `MultiLayerLimitations.lean` — Depth vs temporal nonlinearity
- `TanhSaturation.lean` — Saturation dynamics
- `BinaryFactRetention.lean` — E88 vs linear memory
- `ExactCounting.lean` — Threshold and counting
- `RunningParity.lean` — Parity impossibility
- `E23_DualMemory.lean` — E23 formalization
- `E88_MultiHead.lean` — E88 formalization
- `OutputFeedback.lean` — Emergent tape memory and CoT equivalence
- `TC0Bounds.lean` — TC0 circuit complexity bounds
- `TC0VsUnboundedRNN.lean` — Hierarchy: Linear SSM < TC0 < E88
- `ComputationalClasses.lean` — Chomsky hierarchy for RNNs
- `MultiPass.lean` — Multi-pass RNN computational class
- `RecurrenceLinearity.lean` — Architecture classification by recurrence type

#v(2em)

#align(center)[
  #text(size: 9pt, style: "italic")[
    Document generated from ElmanProofs Lean 4 formalizations. \
    All core expressivity theorems mechanically verified. \
    See Section 12 for verification details.
  ]
]
