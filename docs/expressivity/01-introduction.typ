// Section 1: Introduction

#import "traditional-math-style.typ": *

= Introduction

Every sequence model faces a choice: where should nonlinearity live?

This is not a hyperparameter. It is a fundamental architectural decision with mathematical consequences. The answer determines which functions a model can compute, which it cannot, and why chain-of-thought reasoning works when it does.

Consider the three dominant approaches. Transformers @vaswani2017attention apply nonlinearity between layers: the sequence flows through attention, then through a feedforward network, then through attention again. Time is handled all at once through the attention mechanism; depth accumulates vertically. State-space models like Mamba2 @dao2024mamba2 make a different choice: nonlinearity still flows between layers, but now time flows _linearly_ within each layer. The state $h_t$ is a linear function of $h_(t-1)$. This enables parallel computation through associative scans, but it constrains what can be computed. The third path---the one we will examine most closely---places nonlinearity in time itself, following the classical Elman architecture @elman1990finding. In E88, the state $S_t = tanh(alpha S_(t-1) + delta v_t k_t^top)$ involves a nonlinear function of the previous state. Every timestep adds a layer of composition depth.

These three choices lead to three different computational classes. Our central result makes this precise.

#theorem("Composition Depth Gap")[
  For a $D$-layer model processing sequences of length $T$:
  #h(1em) Linear temporal dynamics yields composition depth $D$.
  #h(1em) Nonlinear temporal dynamics yields composition depth $D times T$.
]

The implications unfold from here. Tasks like running parity---computing $x_1 xor x_2 xor dots xor x_t$ at each position---require $T$ sequential decisions. A $D$-layer model with linear temporal dynamics provides only $D$ levels of composition. When $T > D$, running parity is not merely difficult; it is impossible by theorem.

This document develops the theory and its consequences. We begin with the mathematical machinery of linear recurrent systems and prove what they cannot compute. We then show how E88's tanh saturation escapes these limitations, creating stable fixed points that act as permanent memory. The complete hierarchy emerges:

$ "Linear SSM" subset.neq "TC"^0 "(Transformer)" subset.neq "E88" subset.neq "E23 (UTM)" $

Linear state-space models fall _below_ TC⁰ in circuit complexity---they cannot even compute parity. Transformers sit exactly at TC⁰: constant depth, unbounded fan-in. E88 exceeds TC⁰ because its effective depth grows with sequence length. And E23, with its explicit tape memory, achieves full Turing completeness.

We will also confront the gap between theory and practice. Despite E88's provably greater computational power, Mamba2 often achieves better perplexity on language modeling benchmarks. This is not a contradiction; it reveals the difference between what an architecture _can_ compute and what it _learns efficiently_. The theoretical hierarchy tells us about ultimate limits; training dynamics determine what happens within those limits.

The story has a final twist. When a model writes output and reads it back, the output stream becomes an _emergent tape_. Chain-of-thought, scratchpad computation, autoregressive self-conditioning---these are all instances of the same phenomenon. With output feedback, even limited architectures can achieve bounded Turing machine power:
$ "E88+Feedback" equiv "Transformer+CoT" equiv "DTIME"(T) $
Chain-of-thought works because it provides working memory, not because it enables magical reasoning.

All theorems in this document are mechanically verified in Lean 4 @moura2021lean4 with Mathlib @mathlib2020. This is mathematical certainty, not argument by plausibility. We invite verification: clone the repository, run `lake build`, and confirm that every proof compiles without gaps.

The journey begins with the question of where nonlinearity should live. By the end, we will understand the hierarchy of sequence models and know, with mathematical precision, what each can and cannot do.
