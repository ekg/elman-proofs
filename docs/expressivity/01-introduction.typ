// Section 1: Introduction

#import "traditional-math-style.typ": *

= Introduction

Every sequence model faces a choice: where should nonlinearity live? This architectural decision determines which functions a model can compute and why chain-of-thought reasoning works.

Consider the three dominant approaches. Transformers @vaswani2017attention apply nonlinearity between layers: the sequence flows through attention, then through a feedforward network, then through attention again. Time is handled all at once through the attention mechanism; depth accumulates vertically. State-space models like Mamba2 @dao2024mamba2 make a different choice: nonlinearity still flows between layers, but now time flows _linearly_ within each layer. The state $h_t$ is a linear function of $h_(t-1)$. This enables parallel computation through associative scans, but it constrains what can be computed. The third path---the one we will examine most closely---places nonlinearity in time itself, following the classical Elman architecture @elman1990finding. In E88, the state $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ involves a nonlinear function of the previous state. Every timestep adds a layer of composition depth.

These three choices lead to three different computational classes. Our central result makes this precise.

#theorem("Composition Depth Gap")[
  For a $D$-layer model processing sequences of length $T$:
  #h(1em) Linear temporal dynamics yields composition depth $D$.
  #h(1em) Nonlinear temporal dynamics yields composition depth $D times T$.
]

Tasks like running parity---computing $x_1 xor x_2 xor dots xor x_t$ at each position---require $T$ sequential decisions. A $D$-layer model with linear temporal dynamics provides only $D$ levels of composition. When $T > D$, running parity is impossible.

We begin with linear recurrent systems and prove what they cannot compute, then show how E88's tanh saturation escapes these limitations through stable fixed points. The complete hierarchy:

$ "Linear SSM" subset.neq "TC"^0 "(Transformer)" subset.neq "E88" subset.neq "E23 (UTM)" $

Linear state-space models fall below TC⁰---they cannot compute parity. Transformers sit at TC⁰: constant depth, unbounded fan-in. E88 exceeds TC⁰ through depth that grows with sequence length. E23 achieves Turing completeness with explicit tape memory.

Despite E88's greater computational power, Mamba2 often achieves better perplexity on language modeling. This gap reveals the difference between what an architecture _can_ compute and what it _learns efficiently_.

When a model writes output and reads it back, the output stream becomes an _emergent tape_. With output feedback, even limited architectures achieve bounded Turing machine power:
$ "E88+Feedback" equiv "Transformer+CoT" equiv "DTIME"(T) $
Chain-of-thought works by providing working memory.

All theorems are mechanically verified in Lean 4 @moura2021lean4 with Mathlib @mathlib2020. Clone the repository, run `lake build`, and verify that every proof compiles.
