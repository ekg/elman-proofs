// Section 2: Mathematical Foundations

#import "traditional-math-style.typ": *

= Mathematical Foundations

The central concept is _recurrence linearity_: whether the new state depends linearly or nonlinearly on the previous state. Siegelmann and Sontag @siegelmann1995computational established that RNNs with rational weights are Turing complete.

== Linear Recurrent Systems

#definition("Linear RNN")[
  A linear RNN has dynamics $h_t = A h_(t-1) + B x_t$ and output $y_t = C h_t$, where $A in RR^(n times n)$, $B in RR^(n times m)$, $C in RR^(k times n)$.
]

Unrolling the recurrence gives a closed form.

#theorem("State as Weighted Sum")[
  For a linear RNN starting from $h_0 = 0$: $h_T = sum_(t=0)^(T-1) A^(T-1-t) B x_t$.
]#leanref("LinearCapacity.lean:72", "linear_state_is_sum")

#proof[
  Expand the recurrence: $h_1 = B x_0$, $h_2 = A B x_0 + B x_1$, and so on. The general term is a weighted sum of past inputs with weights given by powers of $A$.
]

The state $h_T$ is a linear combination of inputs, so the output $y_T = C h_T$ is linear in the input sequence: $y(alpha x + beta z) = alpha y(x) + beta y(z)$.

== The Impossibility Results

The threshold function outputs 1 if the sum of inputs exceeds threshold $tau$, else 0. This is discontinuous; linear functions are continuous.

#theorem("Threshold Impossibility")[
  No linear RNN computes $"thresh"_tau$: output 1 if $sum_t x_t > tau$, else 0.
]#leanref("LinearLimitations.lean:107", "linear_cannot_threshold")

#proof[
  A linear output has the form $y(x) = g dot x$ for some fixed vector $g$. At inputs above threshold, $y = 1$ regardless of the exact sum. But if $y = 1$ for all $x$ with $sum x_i > tau$, and the output is linear, then $g = 0$, contradicting $y = 1$.
]

XOR fails linearity algebraically.

#theorem("XOR Is Not Affine")[
  No affine function equals XOR on ${0,1}^2$.
]#leanref("LinearLimitations.lean:218", "xor_not_affine")

#proof[
  Suppose $f(x,y) = a x + b y + c$ agrees with XOR. Then $f(0,0) = 0$ gives $c = 0$. $f(0,1) = 1$ gives $b = 1$. $f(1,0) = 1$ gives $a = 1$. But then $f(1,1) = a + b + c = 2 eq.not 0 = "XOR"(1,1)$.
]

== Depth Does Not Help

#definition("Multi-Layer Linear-Temporal Model")[
  A $D$-layer model with linear temporal dynamics within each layer and nonlinear activations (ReLU, GeLU) between layers.
]

#theorem("Depth Cannot Compensate")[
  For any $D >= 1$, a $D$-layer linear-temporal model cannot compute threshold, parity, or XOR.
]#leanref("MultiLayerLimitations.lean:231", "multilayer_cannot_running_threshold")

Each layer aggregates features linearly across time. Stacking adds vertical depth (nonlinearity between layers) but not horizontal depth (nonlinearity through time).

== The Composition Depth Gap

#theorem("Composition Depth Gap")[
  For a $D$-layer model processing sequences of length $T$:
  #h(1em) Linear temporal dynamics: total composition depth $D$.
  #h(1em) Nonlinear temporal dynamics: total composition depth $D times T$.
]#leanref("RecurrenceLinearity.lean:229", "e1_more_depth_than_minGRU")

Composition depth is the longest chain of nonlinear operations from input to output. Linear-temporal models have only interlayer activations ($D$ nonlinearities). Nonlinear-temporal models add nonlinearity within each timestep ($D times T$ total).

== Architecture Classification

#proposition("Classification")[
  _Linear in $h$_: MinGRU, MinLSTM, Mamba2 SSM. All have the form $h_t = A(x_t) h_(t-1) + b(x_t)$---linear in $h$, even if $A$ and $b$ depend on the input.

  _Nonlinear in $h$_: E1, E88, standard RNN, LSTM, GRU. All have the form $h_t = sigma(W h_(t-1) + V x_t)$---the previous state passes through a nonlinearity.
]#leanref("RecurrenceLinearity.lean:148,171", "e1_is_nonlinear_in_h, mamba2_is_linear_in_h")

Input-dependent gating does not change the classification. In Mamba2, $A(x_t)$ and $B(x_t)$ depend on input, but the recurrence $h_t = A(x_t) h_(t-1) + B(x_t) x_t$ is still linear in $h$.

=== Mathematical Formulations by Architecture

We now formalize each architecture with precise mathematical definitions extracted from our Lean formalizations.

#definition("E88: Gated Elman Network")[
  The E88 update rule with hidden state $h in RR^d$:
  $ h_t = tanh(W_h h_(t-1) + W_x x_t + b) circle.filled.tiny sigma(W_g h_(t-1) + V_g x_t + b_g) $
  where:
  - $W_h in RR^(d times d)$ is the recurrence matrix
  - $W_x in RR^(d times m)$ is the input projection
  - $W_g, V_g$ are gate matrices
  - $sigma(z) = 1/(1 + e^(-z))$ is the sigmoid function
  - $circle.filled.tiny$ denotes element-wise multiplication

  *Key property*: Nonlinear in $h_(t-1)$ due to $tanh(W_h h_(t-1) + dots.h)$.
]#leanref("E1_GatedElman.lean:84", "e1_update")

For matrix-state variants, we define the E88 matrix update:

#definition("E88 Matrix State")[
  The E88 matrix state update with $S in RR^(d times d)$:
  $ S_t = tanh(alpha S_(t-1) + v_t k_t^top) $
  where $alpha in (0,2)$ is retention coefficient, and $v_t, k_t in RR^d$ provide rank-1 updates.

  *Key property*: Nonlinear in $S_(t-1)$ through nested tanh application. Each timestep adds one composition level.
]

#definition("Mamba2: Selective State Space Model")[
  The Mamba2 SSM update with state $h in RR^n$ and output $y in RR^d$:
  $ h_t &= A(x_t) h_(t-1) + B(x_t) x_t \
     y_t &= C(x_t) h_t + D x_t $
  where:
  - $A(x_t) in RR^(n times n)$ is the input-dependent transition matrix
  - $B(x_t) in RR^(n times d)$ is the input-dependent input projection
  - $C(x_t) in RR^(d times n)$ is the input-dependent output projection
  - $D in RR^(d times d)$ is the feedthrough matrix

  *Key property*: Linear in $h_(t-1)$ despite selectivity. The Jacobian $partial h_t / partial h_(t-1) = A(x_t)$ has no state dependence.
]#leanref("Mamba2_SSM.lean:88,171", "ssm_step, mamba2_is_linear_in_h")

#definition("Gated Delta Network")[
  The GDN matrix state update with $S in RR^(d times d)$:
  $ S_t = alpha_t S_(t-1) (I - beta_t k_t k_t^top) + beta_t v_t k_t^top $
  Expanding: $ S_t = alpha_t S_(t-1) - alpha_t beta_t S_(t-1) k_t k_t^top + beta_t v_t k_t^top $
  where:
  - $alpha_t in (0,1)$ is the decay gate (uniform memory erasure)
  - $beta_t in (0,1)$ is the write gate (update strength)
  - $k_t in RR^d$ is the key (what to update)
  - $v_t in RR^d$ is the value (new content)

  *Vector analog* (E62): For vector state $h in RR^d$:
  $ h_t = (1 - k_t) circle.filled.tiny h_(t-1) + k_t circle.filled.tiny v_t $

  *Key property*: Linear in $S_(t-1)$ (or $h_(t-1)$). Update is affine: $S_t = A(x_t) S_(t-1) + b(x_t)$.
]#leanref("GatedDeltaRule.lean:222,113", "matrixDeltaUpdate, selectiveWriteUpdate")

#definition("Transformer Attention")[
  The multi-head attention mechanism:
  $ "Attention"(Q, K, V) = "softmax"((Q K^top)/sqrt(d_k)) V $
  where $Q, K, V$ are linear projections of the input.

  *Key property*: No temporal recurrence within a layer. Each position attends to all positions in parallel. Composition depth equals number of layers $D$, independent of sequence length $T$.
]

#definition("Linear Attention")[
  The associative linearization of attention:
  $ "Output"_t = (q_t dot.op S_t) / Z_t quad "where" quad S_t = sum_(i=1)^t k_i times.circle v_i $

  Recurrent form:
  $ S_t &= S_(t-1) + k_t times.circle v_t \
     Z_t &= Z_(t-1) + k_t \
     y_t &= (q_t dot.op S_t) / (q_t dot.op Z_t) $

  *Key property*: Linear in $S_(t-1)$. The state is a weighted sum of outer products.
]

#definition("MinGRU")[
  Minimal gated recurrent unit with hidden state $h in RR^d$:
  $ z_t &= sigma(W_z x_t) \
     tilde(h)_t &= W_h x_t \
     h_t &= (1 - z_t) circle.filled.tiny h_(t-1) + z_t circle.filled.tiny tilde(h)_t $

  Expanding: $h_t = "diag"(1 - z_t) h_(t-1) + "diag"(z_t) tilde(h)_t$

  *Key property*: Linear in $h_(t-1)$. The coefficient matrix is $A(x_t) = "diag"(1 - z_t)$, which depends only on input $x_t$, not on previous state.
]#leanref("RecurrenceLinearity.lean:91,109", "minGRU_coeff, minGRU_is_linear_in_h")

#definition("MinLSTM")[
  Minimal long short-term memory with forget gate $f_t$ and input gate $i_t$:
  $ f_t &= sigma(W_f x_t) \
     i_t &= sigma(W_i x_t) \
     tilde(c)_t &= W_c x_t \
     c_t &= f_t circle.filled.tiny c_(t-1) + i_t circle.filled.tiny tilde(c)_t $

  *Key property*: Linear in $c_(t-1)$ like MinGRU. Cell state update is $c_t = "diag"(f_t) c_(t-1) + "diag"(i_t) tilde(c)_t$.
]

=== Architectural Parameters and Complexity

We extract computational costs from the Lean formalizations:

#proposition("Computational Complexity")[
  Per-token computation costs for dimension $d$ hidden state, input dimension $m$:

  *E88 (E1 variant)*: $
    "FLOPS"_"E1" = 4d^2 + 2d m + 3d
  $
  Cost breakdown: 4 matrix-vector products ($W_h, W_x, W_g, V_g$) plus element-wise operations (tanh, sigmoid, multiply).

  *Mamba2*: $
    "FLOPS"_"Mamba2" = 6d^2 + 4d + n^2
  $
  Additional costs: selectivity computation (~$2d^2$), convolution (~$4d$), state-space dimension $n approx 16$.

  *Throughput ratio*: E88 achieves ~$2 times$ throughput of Mamba2 at equal dimension due to $6d^2 / 4d^2 approx 1.5 times$ FLOPS ratio.
]#leanref("E1_GatedElman.lean:222,154", "e1_flops_per_token, mamba2_flops_per_token")

=== Jacobian Structure and Gradient Flow

The linearity classification has immediate implications for gradient flow:

#theorem("Jacobian Bounds")[
  For E88 with gates $g in (0,1)$ and tanh derivatives $tanh' in [0,1]$:
  $ norm(partial h_t / partial h_(t-1)) <= norm(W_h) $

  For Mamba2:
  $ partial h_t / partial h_(t-1) = A(x_t) $
  where $A(x_t)$ is input-dependent but state-independent.

  For GDN with gate $k in (0,1)$:
  $ partial h_t / partial h_(t-1) = "diag"(1 - k) $
  diagonal matrix with entries in $(0,1)$.
]#leanref("E1_GatedElman.lean:130,148", "sigmoid_bounded, tanh_deriv_bounded")

#theorem("Gradient Composition Through Time")[
  For a $T$-step sequence through one layer:

  *Linear recurrence*: $
    (partial h_T) / (partial h_0) = product_(t=1)^T A(x_t) = "single matrix product"
  $
  Effective composition depth: $1$ (matrices compose into single transformation).

  *Nonlinear recurrence*: $
    (partial h_T) / (partial h_0) = product_(t=1)^T J_t quad "where" quad J_t = "diag"(tanh'(z_t)) W_h "diag"(g_t)
  $
  Effective composition depth: $T$ (cannot collapse further).
]#leanref("RecurrenceLinearity.lean:189,201", "linear_composition_depth, nonlinear_composition_depth")

This explains the fundamental expressivity gap: linear recurrences collapse temporal dependencies into a single transformation, while nonlinear recurrences preserve $T$ levels of composition within each layer.

#centerrule

Linear recurrent systems compute linear functions of their input history. Depth adds nonlinearity between layers, not through time.
