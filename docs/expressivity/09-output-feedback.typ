// Section 9: Output Feedback and Emergent Tape

#import "traditional-math-style.typ": *

= Output Feedback and Emergent Tape

The hierarchy we have established assumes fixed internal state. But in practice, models often operate differently: they write output, then read it back as input. This simple change has profound consequences.

== The Emergent Tape

When a model's output becomes part of its input, the output stream functions as external memory. The model has created an _emergent tape_.

Consider a Transformer with chain-of-thought prompting. The model generates tokens, which are prepended to the context, which the model reads on the next step. The generated text is not just output---it is working memory. Similarly, an RNN in autoregressive mode reads its own previous outputs. The "scratchpad" techniques in language models exploit exactly this mechanism.

#theorem("Emergent Tape")[
  A model with output feedback and $T$ computation steps has effective memory capacity $O(T)$ bits, achieving $"DTIME"(T)$ computational power regardless of fixed state dimension.#leanfile("OutputFeedback.lean:282")
]

The proof is constructive: simulate a $T$-step Turing machine by encoding the tape in the output stream. The model reads the tape, makes a transition, writes the updated tape. $T$ steps suffice for $T$-bounded computation.

This theorem explains why chain-of-thought works. It is not that writing out reasoning steps clarifies the model's "thinking." It is that the written steps provide memory that the model's fixed state cannot. The improvement is computational, not psychological.

== Access Patterns

The emergent tape equalizes computational class but not efficiency. The difference is in how the tape is accessed.

An RNN with feedback has sequential access. To read position $p$ in the output history requires processing $p$ timesteps. Each random access costs $O(T)$.

A Transformer with chain-of-thought has random access. Attention can look at any position in the context with $O(1)$ cost. The same computation that requires $k$ sequential passes for an RNN can be done in one pass by a Transformer.

Both achieve $"DTIME"(T)$---bounded Turing machine computation. But for algorithms that require multiple random accesses, the Transformer is polynomially more efficient.

== The Extended Hierarchy

With output feedback, we can state the complete memory hierarchy.

#theorem("Strict Hierarchy with Feedback")[
  $ "Fixed Mamba2" subset.neq "Fixed E88" subset.neq "E88+Feedback" equiv "Transformer+CoT" subset.neq "E23" $#leanfile("OutputFeedback.lean:498")
]

Each separation has a witness.

_Mamba2 < E88_: Running parity. We have proved this at length.

_E88 < E88+Feedback_: Palindrome detection. Recognizing whether the input is a palindrome requires comparing position $i$ with position $T-i$. Fixed state cannot store the entire input. With feedback, the model can write the input and read it backward.

_CoT < E23_: The halting problem. $"DTIME"(T)$ is bounded computation; E23 with unbounded tape achieves all recursive functions. There exist problems solvable by E23 that no bounded-tape model can solve.

== Why Chain-of-Thought Works

The emergent tape principle demystifies chain-of-thought reasoning.

#keyinsight[
  Chain-of-thought works because it provides working memory, not because it enables magical reasoning ability. The scratchpad is computationally necessary for multi-step algorithms. A model computing $A$ then $B$ then $C$, where each depends on the previous, needs somewhere to store intermediate results.
]

Without CoT, a Transformer has fixed internal state. Complex multi-step computations overflow this state.

With CoT, the output provides unbounded working memory. The model can execute algorithms correctly.

This is computational necessity, not interpretability.

#centerrule

Models with emergent tape memory achieve bounded Turing machine power. The differences between architectures collapse: E88+Feedback equals Transformer+CoT equals $"DTIME"(T)$.
