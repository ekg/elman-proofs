# Longhorn Connection Report

## Paper: "Longhorn: State Space Models are Amortized Online Learners"

**Authors**: Bo Liu, Rui Wang, Lemeng Wu, Yihao Feng, Peter Stone, Qiang Liu
**Venue**: NeurIPS 2024
**ArXiv**: [2407.14207](https://arxiv.org/abs/2407.14207)

---

## 1. The "RNNs as Online Learners" Framework

### Core Idea

Longhorn proposes that the recurrent state update of any SSM can be viewed as solving an online convex optimization problem at each timestep. The state `s_t` is updated by minimizing a composite loss:

```
s_t = argmin_s { D_phi(s, s_{t-1}) + beta_t * l_t(s) }
```

where:
- `D_phi(s, s_{t-1})` is a Bregman divergence measuring how far the new state deviates from the old state (the **stability** term --- "don't forget too much")
- `l_t(s)` is a task-specific loss on the current input (the **plasticity** term --- "learn the new data")
- `beta_t` controls the tradeoff: high `beta_t` means aggressive learning, low `beta_t` means conservative retention

The claim is that every existing SSM implicitly corresponds to choosing a specific loss function `l_t`, a specific stability regularizer `D_phi`, and then applying either a gradient step (explicit) or a closed-form solution (implicit) to the composite objective.

### What "Amortized" Means

The framework treats the SSM layer as a "meta-module" that compresses history into state `s_t`. This compression is then "amortized" --- reused by downstream layers for all future predictions. The recurrent update optimizes this compression implicitly at each step, rather than requiring a separate optimization loop. In other words, the SSM's forward pass IS the optimization procedure; no inner loop is needed because the update rule is the closed-form solution to the online learning problem.

This is closely related to the TTT (Test-Time Training) framework of Sun et al. (2024), which makes the hidden state a full machine learning model that is trained on each test sequence. Longhorn can be seen as a special case where the "model" being trained is a linear associative memory, and the "training step" is an implicit closed-form update.

---

## 2. Loss Functions and Their Corresponding Update Rules

This is the central technical contribution: a unifying table showing what loss each SSM optimizes.

### Table of Correspondences

| Model | Online Learning Objective L_t(s) | Resulting Update Rule |
|-------|----------------------------------|----------------------|
| **Linear Attention** | `\|\|s - s_{t-1}\|\|^2 - 2<s^T k_t, x_t>` | `s_t = s_{t-1} + k_t x_t^T` |
| **RetNet** | `gamma\|\|s - s_{t-1}\|\|^2 + (1-gamma)\|\|s\|\|^2 - 2<s^T k_t, x_t>` | `s_t = gamma * s_{t-1} + k_t x_t^T` |
| **GLA** | `(s-s_{t-1})^T diag(alpha_t)(s-s_{t-1}) + s^T diag(1-alpha_t) s - 2<s^T k_t, x_t>` | `s_t = diag(alpha_t) s_{t-1} + k_t x_t^T` |
| **Griffin** | Weighted norm with input gate `(1-alpha_t)` | `s_t = alpha_t * s_{t-1} + sqrt(1-alpha_t) * i_t * x_t` |
| **DeltaNet** | `\|\|s - s_{t-1}\|\|^2 + beta_t \|\|s^T k_t - x_t\|\|^2` (one gradient step) | `s_t = (I - beta_t k_t k_t^T) s_{t-1} + beta_t v_t k_t^T` |
| **Longhorn** | `\|\|s - s_{t-1}\|\|^2 + beta_t \|\|s^T k_t - x_t\|\|^2` (closed-form solution) | `s_t = (1 - Delta_t * k_t^{circ 2}) * s_{t-1} + Delta_t * k_t * x_t` |
| **Mamba** | Not an explicit online learning objective; uses a learned `A` matrix with input-dependent `Delta` discretization | `s_t = exp(A * Delta(x_t)) * s_{t-1} + Delta(x_t) * B(x_t) * x_t` |

where for Longhorn: `Delta_t = beta_t / (1 + beta_t * k_t^T k_t)`

### Key Insight: The Loss Determines Expressivity

**Linear Attention** minimizes a linear loss in `s`. This means it never "forgets" --- the state grows unboundedly by accumulating `k_t x_t^T` outer products. The stability term is just `\|\|s - s_{t-1}\|\|^2`, which penalizes change but doesn't decay old state.

**RetNet / GLA** add an explicit regularization `\|\|s\|\|^2` that shrinks the state toward zero. This creates the forgetting/decay mechanism. The gamma or alpha_t coefficients control how aggressively old state is forgotten.

**DeltaNet and Longhorn** minimize a regression loss `\|\|s^T k_t - x_t\|\|^2` --- they want the state to act as a linear associative memory that, given key `k_t`, retrieves value `x_t`. The crucial difference from Linear Attention is that this is a quadratic loss in `s`, not linear. This creates a **coupling between the state and the key**, producing the `(I - beta k k^T)` forget gate that selectively erases information along the `k_t` direction before writing new information.

**Mamba** is notable because it does NOT naturally fit this framework. Mamba's `A` matrix is a learned parameter with input-dependent discretization, not derived from any particular online learning objective. The paper essentially shows that Mamba's design is ad-hoc relative to this principled framework.

### Explicit vs Implicit Online Learning

DeltaNet and Longhorn optimize the SAME loss function but differ in how they solve it:

- **DeltaNet (explicit)**: Takes one gradient descent step on the loss. The transition matrix `A(x_t) = I - beta_t k_t k_t^T` can have eigenvalues outside [-1, 1], requiring key normalization for stability. Cannot extrapolate beyond training context length.

- **Longhorn (implicit)**: Solves the loss exactly via closed-form solution (setting gradient to zero and using the Sherman-Morrison formula). The effective step size `Delta_t = beta_t / (1 + beta_t k_t^T k_t)` is automatically bounded in [0, 1], guaranteeing stability. Can extrapolate to 16x longer contexts.

This is the optimization theory distinction: implicit methods (backward Euler) are unconditionally stable while explicit methods (forward Euler) require step size constraints.

---

## 3. Nonlinearity, State-Tracking, and the Limits of the Online Learning Framework

### What the Framework Says About Nonlinearity

**The framework is fundamentally limited to linear state updates.** The online convex optimization setup assumes the loss `L_t(s)` is convex in `s`, which means the optimal `s_t` is obtained by a linear operation on `s_{t-1}`. Every model in the table above has a state update of the form:

```
s_t = A(x_t) s_{t-1} + b(x_t)
```

This is **linear in s_{t-1}**, even though `A` and `b` can depend nonlinearly on the input `x_t`. The online learning framework cannot produce a tanh, sigmoid, or any other nonlinear function of the previous state.

This is not an accident --- it is a structural feature. Convex optimization over the state means the landscape is "bowl-shaped", with a unique minimum. Nonlinear state dynamics (like E88's tanh) create multiple attractors, which correspond to **non-convex** landscapes.

### What This Means for State-Tracking

The paper does not address state-tracking problems (parity, permutation composition, finite automaton simulation). But a concurrent/subsequent line of work by Grazzi et al. ("Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues", ICLR 2025 Oral) provides the missing piece:

**Linear RNNs with eigenvalues restricted to [0, 1] cannot solve parity.** This is exactly the constraint that Longhorn, DeltaNet, Linear Attention, RetNet, GLA, and Mamba all satisfy by default. Their state transition matrices have non-negative eigenvalues.

**Extending eigenvalues to [-1, 1] enables parity and some state-tracking.** This is achieved in DeltaNet by simply multiplying `beta` by 2, changing `A = I - beta k k^T` to potentially have negative eigenvalues. This allows the model to "flip" state components, which is exactly what parity requires.

**But even with [-1, 1] eigenvalues, linear RNNs cannot solve all state-tracking problems.** Counting modulo 3 requires non-triangular transition matrices. Full permutation group tracking (S_5) requires nonlinear RNNs or input-dependent transition matrices with specific structure.

The hierarchy is:
```
[0,1] eigenvalues < [-1,1] eigenvalues < nonlinear state updates
```

This maps cleanly onto:
```
Mamba/Longhorn < DeltaNet (extended) < E88/E1H
```

### The Online Learning Framework Cannot Derive Nonlinear State Updates

If we try to use a non-convex loss `l_t(s)`, the argmin may not exist, may not be unique, and may not have a closed-form. The entire parallel-scan / associative-scan machinery breaks down because the scan assumes the state update is a linear transformation that can be composed via matrix multiplication.

This is a fundamental tradeoff: **the online learning framework gives you principled design and efficient parallelism, but it restricts you to the linear-state regime. E88's tanh lies outside this framework.**

---

## 4. Connection to E88/E1H Architectures

### Can We Derive E88's Update Rule from an Online Learning Objective?

**No, not within the convex online learning framework.** Here is why.

E88's update is:
```
S_t = tanh(alpha * S_{t-1} + delta * v_t k_t^T)
```

E1H's update is:
```
h_t = tanh(W_h h_{t-1} + W_x x_t + b) * sigmoid(W_g h_{t-1} + V_g x_t + b_g)
```

Both apply a nonlinear function (tanh) to a linear combination of the previous state and current input. To get this from the online learning framework, we would need a loss function whose minimizer has the form `s_t = tanh(alpha s_{t-1} + ...)`. But the tanh arises from a non-convex constraint or a non-convex loss, which breaks the framework's assumptions.

However, we can understand E88's update rule by analogy to the framework:

#### The "Implicit Nonlinear Regression" Interpretation

Consider the loss:
```
L_t(s) = ||s - s_{t-1}||^2 + beta_t ||s^T k_t - x_t||^2 + R(s)
```

where `R(s)` is a regularizer that constrains `s` to have entries in `(-1, 1)`. One such regularizer is the barrier function `R(s) = -sum log(1 - s_ij^2)` (log-barrier for the box constraint `|s_ij| < 1`).

The KKT conditions for this constrained optimization would produce an update where the state is "pushed" toward the box boundary but never crosses it --- analogous to what tanh does. The tanh function can be seen as implementing a soft version of this box constraint. In this interpretation:

- **Longhorn's closed-form solution** minimizes `||s^T k - x||^2` subject to `s ≈ s_{t-1}` (unconstrained, or softly constrained by the quadratic penalty)
- **E88's tanh** minimizes something similar, but with the additional constraint `s_ij in (-1, 1)`, enforced by the tanh saturation

This is speculative but geometrically precise: tanh maps R -> (-1, 1), exactly implementing bounded state.

#### The "Hebbian + Normalization" Interpretation

Another way to see E88: the inner linear update `alpha S_{t-1} + v_t k_t^T` is Hebbian learning (accumulate outer products, with decay), and the tanh is a form of element-wise normalization that prevents runaway growth. In the online learning framework:

- **Linear Attention**: Hebbian learning without normalization (state grows unboundedly)
- **RetNet**: Hebbian learning with exponential decay normalization
- **DeltaNet/Longhorn**: Hebbian learning with selective erasure (delta rule)
- **E88**: Hebbian learning with element-wise bounded normalization (tanh saturation)

The tanh is thus a different answer to the same question every SSM faces: "how do we prevent the state from blowing up?" Linear SSMs answer with decay factors and selective erasure. E88 answers with bounded nonlinear compression.

#### What E88 Gains from Nonlinearity

In the Longhorn framework, the state `s_t` is always a linear function of `s_{t-1}`. This means:
- The state space is a vector space (no distinct regions, no attractors)
- Information stored at time `s` decays continuously
- The model cannot maintain discrete state (binary flags, counters)

E88's tanh creates:
- **Multiple attractors**: The bifurcation at `alpha = 1` creates stable fixed points at `+/- S*`, enabling binary storage
- **Saturation-based memory**: Once `|S_ij|` exceeds ~0.9, the derivative `tanh'` is near zero, making that entry insensitive to new inputs --- a form of content-addressable write protection
- **Non-commutative dynamics**: `tanh(A + B) != tanh(A) + tanh(B)`, so the order of inputs matters in a way that linear updates cannot capture

These properties are exactly what state-tracking problems require. Parity needs sign-flipping (covered by the bifurcation). Threshold detection needs a saturating accumulator (covered by tanh's bounded output). Finite state machine simulation needs distinct stable states (covered by the multiple attractors).

---

## 5. Why Nonlinear State Updates Are More Expressive

### The Composition Depth Argument (from our work)

Our paper formalizes this as **composition depth**:
- **Linear-temporal models** (Mamba2, Longhorn, DeltaNet, etc.): The state transition matrices compose into a single matrix product. Total composition depth = D (number of layers).
- **Nonlinear-temporal models** (E88, E1H): Each timestep adds one level of nonlinear composition. Total composition depth = D * T (layers times sequence length).

For a 25-layer model processing T=32768 tokens:
- Mamba2/Longhorn: composition depth 25
- E88: composition depth 819,200

This is not about parameter count or state size --- it is about the length of the longest chain of nonlinear operations from input to output.

### The Longhorn Framework Explains WHY Linear SSMs Have This Limitation

The online learning framework makes the limitation transparent. Every model in the table solves:
```
s_t = argmin_s { stability(s, s_{t-1}) + plasticity(s, x_t) }
```

When both terms are convex and quadratic, the solution is always:
```
s_t = A s_{t-1} + b
```

The matrices compose: `s_T = A_T * A_{T-1} * ... * A_1 * s_0 + (accumulated b's)`. This is a single linear transformation of the initial state, regardless of sequence length. All temporal structure collapses into one matrix product.

E88 breaks this because tanh is not the solution to any convex optimization problem. The "loss function" that would produce `s_t = tanh(alpha s_{t-1} + v k^T)` is non-convex, and the update cannot be collapsed into a matrix product.

### Longhorn's Contribution to Understanding the Hierarchy

The Longhorn framework clarifies the hierarchy we prove:

```
Linear SSM (Mamba2) ⊊ E1H ⊊ E88
```

In Longhorn's terms:
1. **Linear SSMs** solve convex online learning problems. Their state is a linear function of history. They cannot maintain discrete state, compute parity, or detect thresholds.

2. **E1H** (vector-state Elman RNN with tanh) adds nonlinear temporal composition. The tanh creates multiple attractors, enabling binary latching, parity, and threshold detection. But the vector state `h in R^d` limits the total information that can be stored and retrieved.

3. **E88** (matrix-state Elman RNN with tanh) has the same nonlinear temporal composition as E1H, plus `d^2` state capacity (vs d for E1H) and content-addressable retrieval via `S q`. The matrix state enables more information to be stored simultaneously and retrieved independently.

The Longhorn framework lives entirely within level (1). It provides the most principled design within the linear-state regime, but it cannot reach levels (2) or (3).

### The Eigenvalue Extension Connection

The Grazzi et al. (ICLR 2025) result on negative eigenvalues provides a partial bridge:
- Standard DeltaNet/Longhorn: eigenvalues in [0, 1] --- cannot solve parity
- Extended DeltaNet: eigenvalues in [-1, 1] --- CAN solve parity (but still linear in state)
- E88: nonlinear state updates --- can solve parity AND has multiple stable attractors

The eigenvalue extension gives linear RNNs some state-tracking ability (parity, permutation composition of transpositions), but E88's nonlinear dynamics provide qualitatively more: stable fixed points, bounded state, and saturation-based memory that the linear models cannot achieve even with negative eigenvalues.

---

## 6. Summary of Key Takeaways

### For Our Paper

1. **The Longhorn framework validates our hierarchy from a different angle.** Our separation results (linear SSM cannot compute parity/threshold) align with the fact that all models derivable from convex online learning objectives are linear in state. The framework makes the limitation principled rather than architectural.

2. **The online learning perspective explains WHY adding tanh helps.** In the Longhorn framework, the state is an associative memory that linearly interpolates between remembering and learning. E88's tanh replaces linear interpolation with bounded nonlinear compression, creating qualitatively different dynamics (attractors, latching, non-commutative composition).

3. **The framework provides the right language for the theory-practice gap.** At 512 tokens, E88's tanh contributes nothing because the online learning objective (predict next byte) doesn't require state-tracking. At 32K tokens, the compositional depth advantage manifests. Longhorn's framework explains why: at short context, the linear associative memory (which linear SSMs solve optimally!) is sufficient. At long context, the accumulated compositional depth of nonlinear updates provides an advantage that no linear online learning update can match.

4. **Longhorn's length extrapolation result is relevant.** Longhorn extrapolates to 16x longer contexts (2K -> 32K). Our E88 also shows improvement at 32K relative to Mamba2 (ranking inversion). The mechanisms are different: Longhorn extrapolates via stable implicit updates, E88 via compositional depth. But both point to long-context as the regime where architectural choices matter most.

### What We Cannot Claim

1. We cannot derive E88's update from the online learning framework. The tanh takes E88 outside the convex optimization regime. We can describe E88 by analogy (bounded Hebbian learning, box-constrained regression) but not derive it from a loss function within the framework.

2. The Longhorn framework does not directly address the question "when does natural language require state-tracking?" Our ablation at 512 tokens shows tanh contributes nothing; at 32K the ranking inverts but we attribute this primarily to architectural advantages (more heads, matrix state) rather than specifically to the tanh nonlinearity.

3. Longhorn's framework does not explain the batch size effect. The bs=1 advantage we observe (0.27 nats, 31% improvement) is about optimization dynamics, not architecture design, and lies outside the online-learning-on-state framework.

### Possible Future Directions

1. **Non-convex online learning framework.** Can we extend Longhorn's framework to non-convex losses that produce tanh-like updates? This would provide a principled derivation of E88 from optimization principles. The challenge is that non-convex problems don't have unique solutions or efficient parallel implementations.

2. **Hybrid architectures.** Longhorn provides optimal linear state updates; E88 provides nonlinear state updates. A hybrid that uses Longhorn-style implicit updates for most of the state and E88-style tanh for a few "state-tracking" channels could combine the efficiency of linear updates with the expressivity of nonlinear ones.

3. **Loss-function-guided architecture search.** Instead of searching over architectures directly (as we do with CMA-ES), search over online learning objectives. Each objective implies an update rule; evaluate the update rule on downstream tasks. This could systematically explore the space between Longhorn (convex, quadratic regression) and E88 (non-convex, bounded).

---

## Appendix: Mathematical Details

### Longhorn's Closed-Form Solution (Theorem 3.1)

Starting from the loss:
```
L_t(s) = ||s - s_{t-1}||^2 + beta_t ||s^T k_t - x_t||^2
```

Setting the gradient to zero:
```
2(s - s_{t-1}) + 2 beta_t (s^T k_t - x_t) k_t = 0
s + beta_t (s^T k_t) k_t = s_{t-1} + beta_t x_t k_t
(I + beta_t k_t k_t^T) s = s_{t-1} + beta_t x_t k_t
```

Using the Sherman-Morrison formula to invert `(I + beta_t k_t k_t^T)`:
```
s_t = (I - Delta_t k_t k_t^T) s_{t-1} + Delta_t k_t x_t^T
where Delta_t = beta_t / (1 + beta_t k_t^T k_t)
```

The practical diagonal approximation replaces the rank-1 outer product `k_t k_t^T` with element-wise square `k_t^{circ 2}`:
```
S_t = (1 - Delta_t (x) k_t^{circ 2}) * S_{t-1} + (Delta_t * x_t) (x) k_t
```

where `(x)` denotes outer product and `*` denotes Hadamard product.

### DeltaNet's Update (for comparison)

Same loss, but one gradient step instead of closed form:
```
s_t = s_{t-1} - beta_t * grad L_t(s_{t-1})
    = s_{t-1} - beta_t (s_{t-1}^T k_t - x_t) k_t
    = (I - beta_t k_t k_t^T) s_{t-1} + beta_t v_t k_t^T
```

This looks similar to Longhorn, but `beta_t` is unbounded (no denominator `1 + beta_t k_t^T k_t`), so eigenvalues of `I - beta_t k_t k_t^T` can exceed 1 in magnitude, causing instability.

### E88's Update (for comparison)

```
S_t = tanh(alpha S_{t-1} + delta v_t k_t^T)
```

The inner term `alpha S_{t-1} + delta v_t k_t^T` is structurally similar to what Longhorn computes: a decayed previous state plus a rank-1 outer product. The tanh wrapping is the critical difference that places E88 outside the online learning framework.

If we were to "linearize" E88 (remove the tanh), we would get:
```
S_t = alpha S_{t-1} + delta v_t k_t^T
```

This is exactly RetNet's update rule (Table row 2 above), with `gamma = alpha` and the outer product `v_t k_t^T` playing the role of `k_t x_t^T`. Our ablation at 512 tokens showing "tanh -> linear: ~0.00 nats change" is consistent with this: at short context, E88 without tanh IS RetNet, and RetNet is a reasonable model.

The tanh starts to matter at longer context because RetNet's state decays exponentially (information from step 0 is attenuated by `alpha^T`), while E88's tanh saturation creates stable fixed points that resist decay.

---

## References

- Longhorn: [arxiv.org/abs/2407.14207](https://arxiv.org/abs/2407.14207)
- DeltaNet: [arxiv.org/abs/2406.06484](https://arxiv.org/abs/2406.06484)
- DeltaNet blog (Songlin Yang): [sustcsonglin.github.io/blog/2024/deltanet-1/](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- Unlocking State-Tracking (Grazzi et al., ICLR 2025): [arxiv.org/abs/2411.12537](https://arxiv.org/abs/2411.12537)
- DeltaProduct: [arxiv.org/abs/2502.10297](https://arxiv.org/abs/2502.10297)
- TTT (Test-Time Training): [arxiv.org/abs/2407.04620](https://arxiv.org/abs/2407.04620)
- Gated Delta Networks (ICLR 2025): [jankautz.com/publications/GatedDeltaNet_ICLR25.pdf](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)
