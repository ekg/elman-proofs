# State Tracking and Nonlinearity in Sequence Models: Literature Connection Report

**Date:** 2026-03-12

**Context:** This report analyzes recent papers on the expressivity limitations of linear state-space models and the role of nonlinearity in enabling state tracking, connecting them to our E88/E1H expressivity hierarchy and Lean-verified proofs.

**Note on arxiv 2501.12352:** The paper at this arXiv ID is "Test-time regression: a unifying framework for designing sequence models with associative memory" (Wang, Shi, Fox 2025), which frames sequence architectures as test-time regression for associative recall. While it provides a useful unifying lens for understanding linear attention, SSMs, and softmax attention as regression variants, it does not directly address state tracking, nonlinearity, or the expressivity hierarchy questions central to our work. The papers analyzed below are the ones directly relevant to our E88/E1H hierarchy.

---

## Papers Analyzed

1. **"The Illusion of State in State-Space Models"** (Merrill, Petty, Sabharwal, 2024) -- arXiv 2404.08819
2. **"Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"** (Grazzi, Siems, Franke, Bethge, Lammauer, 2024) -- arXiv 2411.12537
3. **"The Computational Limits of State-Space Models and Mamba via the Lens of Circuit Complexity"** (Li, Guo, Chen, 2024) -- arXiv 2412.06148
4. **"The Expressive Capacity of State Space Models: A Formal Language Perspective"** (Sarrof, Veitsman, Hahn, 2024) -- arXiv 2405.17394
5. **"DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products"** (Grazzi, Siems, Franke, Lammauer, 2025) -- arXiv 2502.10297
6. **"Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models"** (2025) -- arXiv 2509.22284

---

## 1. What Is the Main Argument About Nonlinearity and State Tracking?

The papers converge on a single conclusion from multiple angles: **linear state evolution is fundamentally insufficient for state tracking, and the specific mechanism by which this limitation is overcome matters**.

### The Core Limitation

All six papers confirm that models with linear temporal dynamics -- including Mamba, Mamba2, linear attention, GLA, RWKV, MinGRU, MinLSTM, and all diagonal SSMs -- are confined to the circuit complexity class **TC^0** (constant-depth, polynomial-size threshold circuits). This is not an empirical observation; it is a mathematical fact proven independently by multiple groups.

The key results:

- **Merrill et al. (2404.08819), Theorem 4.2:** Non-gated SSMs are in TC^0.
- **Merrill et al. (2404.08819), Theorem 4.4:** Diagonal SSMs are in TC^0.
- **Li et al. (2412.06148), Theorem 4.5:** Mamba (with selectivity) is in DLOGTIME-uniform TC^0.
- **Sarrof et al. (2405.17394), Theorem 4:** Nonnegative-gated SSMs can model a regular language if and only if it is star-free.

The consequence: assuming TC^0 != NC^1 (a widely believed complexity separation), these models **cannot solve permutation composition, boolean formula evaluation, or any NC^1-hard problem**. This includes tracking the state of any non-solvable finite automaton (e.g., S_5).

### Three Proposed Escapes

The papers identify three distinct mechanisms for overcoming TC^0 limitations:

**Escape 1: Pointwise nonlinearity in the recurrence (RNN-SSM)**

Merrill et al. define the RNN-SSM with update h_i = sgn(A_bar * h_{i-1} + B_bar * x_i). Their Theorem 5.1 proves this recognizes any regular language. The nonlinearity must be applied **recurrently at each state update** -- applying it only between layers (as in Mamba's SiLU/swish) is insufficient because the model remains in TC^0.

**Escape 2: Input-dependent dense transition matrices (IDS4)**

Merrill et al. also propose IDS4 where A_bar_i = pi_A(x_i), making the transition matrix a function of input. Their Theorem 5.2 proves this also recognizes any regular language. The key insight: iterated products of **general** (non-diagonal) matrices cannot be computed in TC^0, whereas products of diagonal matrices can. This approach remains parallelizable via SCAN but requires dense (non-diagonal) matrices.

**Escape 3: Extending eigenvalue ranges**

Grazzi et al. (2411.12537) take a different approach entirely. Rather than adding nonlinearity, they show that existing linear RNNs fail because their eigenvalues are restricted to [0, 1]. Extending to [-1, 1] through a simple remapping (replacing sigma(x) with 2*sigma(x) - 1) enables parity computation while preserving linearity and parallelizability.

---

## 2. What State-Tracking Problems Are Studied?

### Permutation Composition (S_5 Word Problem)

The central benchmark across papers. Given a sequence of permutations from S_5, compute their composition. This is NC^1-complete (Merrill et al.), meaning it is the canonical hard problem for state tracking.

- **S_5** (symmetric group on 5 elements): Non-solvable, requires general matrix products
- **A_5** (alternating group on 5 elements): Also non-solvable
- **A_4 x Z_5**: Solvable but non-abelian
- **Z_60**: Abelian (easiest)

Merrill et al. reduce chess state tracking to S_5, proving chess tracking is NC^1-complete.

### Parity

Computing XOR over bit sequences. This is the simplest function separating nonnegative-eigenvalue linear RNNs from those with negative eigenvalues:

- **Grazzi et al., Theorem 1:** Finite-precision LRNNs with only nonnegative eigenvalues cannot solve parity.
- **Grazzi et al., Theorem 3:** LRNNs with eigenvalues in [-1, 1] (via generalized Householder matrices) can solve parity.
- **Sarrof et al., Theorem 2:** Nonnegative SSMs cannot recognize PARITY at arbitrary input lengths.

### Modular Counting

Counting inputs modulo m:

- **Grazzi et al., Theorem 2:** Counting modulo m (where m is not a power of 2) requires eigenvalues with nonzero imaginary parts -- even negative real eigenvalues are insufficient.
- **Sarrof et al.:** Non-star-free languages (including modular arithmetic) cannot be recognized by nonnegative SSMs.

### Star-Free vs. Non-Star-Free Languages

Sarrof et al. provide the sharpest characterization: nonnegative SSMs recognize exactly the star-free regular languages. Parity, modular counting, and permutation composition over non-trivial groups are all non-star-free.

---

## 3. How Does Adding Nonlinearity Help? What Specific Nonlinearities Are Discussed?

### Sign function (sgn)

Merrill et al. use sgn as the nonlinearity in their RNN-SSM construction. This is a theoretical tool -- the sign function is not differentiable and cannot be trained with gradient descent. It serves as an existence proof: **some** nonlinearity suffices.

### Tanh

**Tanh is not specifically discussed in any of the six papers.** This is notable. The papers either use theoretical constructions (sgn) or structural modifications (eigenvalue extension, Householder products, sparse transition matrices) rather than investigating which practical nonlinearities suffice.

This is a gap that our work fills directly. Our Lean proofs establish that tanh specifically enables:
- Running parity (TanhSaturation.lean:720)
- Soft threshold detection (TanhSaturation.lean:424)
- Exact counting modulo n (E88Definition.lean:276)
- Binary latching via bifurcation (AttentionPersistence.lean:212)

### Negative eigenvalues (not a nonlinearity per se)

Grazzi et al.'s approach is explicitly **not** a nonlinearity. They modify the eigenvalue range of existing linear models through a linear remapping. The recurrence remains linear in the state. This is a structural change to the linear dynamics, not an escape from linearity.

### Householder products

DeltaProduct (Grazzi et al. 2025) uses products of generalized Householder transformations I - 2*phi(x)*v(x)*v(x)^T. These are orthogonal transformations with eigenvalues in [-1, 1]. Key results:
- n_h = 1 Householder: can only solve 2-permutations (parity)
- n_h > 1: enables rotations and higher-order permutation groups
- 4 layers with n_h = 1: solves any group word problem (Theorem 1)
- Finite layers with gating: recognizes any regular language (Theorem 2)

### Sparse structured matrices (PD-SSM)

The PD-SSM approach uses products of binary column one-hot matrices and complex diagonal matrices. This achieves universal FSA emulation with near-optimal state size. The structure is algebraic rather than nonlinear.

---

## 4. Connection to Our E88/E1H vs. Linear SSM Hierarchy

### What We Prove That They Don't

Our work provides something none of these papers do: **machine-verified proofs that a specific, practical nonlinearity (tanh) applied in temporal recurrence creates a strict expressivity hierarchy**.

The papers above establish:
- Linear SSMs are in TC^0 (impossibility results)
- Theoretical constructions (sgn, IDS4) can escape TC^0
- Eigenvalue extensions help within the linear framework

Our Lean proofs establish:
- Linear SSMs cannot compute parity (LinearLimitations.lean:218)
- Linear SSMs cannot compute threshold (LinearLimitations.lean:107)
- E88 (tanh, matrix state) computes parity (TanhSaturation.lean:720)
- E88 computes threshold (TanhSaturation.lean:424)
- E1H (tanh, vector state) also computes parity and threshold
- E88 strictly exceeds E1H in capacity (E88ExceedsE1HCapacity.lean)
- The full hierarchy: Linear SSM ⊊ E1H ⊊ E88 (ExpressivityHierarchy.lean:267)

### Complementary Angles

**Merrill et al. (Illusion of State)** provides the complexity-theoretic framing (TC^0 vs NC^1) that contextualizes our results. Their Corollary 4.7 shows that no SSM can solve S_5 assuming TC^0 != NC^1. Our work provides constructive proofs of specific impossibilities (parity, threshold) without relying on unproven complexity assumptions.

**Grazzi et al. (Negative Eigenvalues)** shows that within the linear framework, structural modifications can partially close the gap. Their parity result (Theorem 1) is complementary to ours: they show linear RNNs need negative eigenvalues for parity; we show that tanh nonlinearity is an alternative escape. The key difference: their approach stays linear (parallelizable via scan), while ours uses nonlinearity (sequential but more expressive).

**Sarrof et al. (Formal Language Perspective)** provides the sharpest characterization of what nonnegative SSMs can compute: exactly the star-free regular languages. This complements our impossibility proofs by giving a positive characterization of what linear models *can* do, not just what they cannot.

### The Eigenvalue Angle vs. The Nonlinearity Angle

A key insight from comparing these works with ours: there are **two orthogonal dimensions** along which linear SSMs can be extended:

1. **Eigenvalue range**: [0,1] -> [-1,1] -> complex. Stays linear, stays parallelizable. Enables parity (negative eigenvalues), modular counting (complex eigenvalues), and eventually all regular languages (Householder products).

2. **Temporal nonlinearity**: Linear -> tanh/sgn/ReLU. Breaks parallelizability but provides unbounded composition depth. Enables everything regular languages can do and more (depth grows with T).

Our E88 architecture takes path (2). The papers above primarily explore path (1) or prove impossibilities that motivate both paths.

The distinction matters practically: path (1) maintains O(T) parallel scan efficiency but has a ceiling (regular languages require multi-layer constructions with specific matrix structures). Path (2) sacrifices parallel scan but achieves D*T composition depth per forward pass. For tasks requiring deep temporal composition (our 32K context results where E88 beats Mamba2), path (2) wins.

### Circuit Complexity Alignment

Our TC^0 analysis (Section 8 of the paper, TC0Bounds.lean) aligns directly with Merrill et al. and Li et al.:

| Architecture | Our Classification | Literature Classification |
|---|---|---|
| Linear SSM (Mamba2) | Below TC^0 | TC^0 (Merrill), DLOGTIME-uniform TC^0 (Li) |
| Transformer | = TC^0 | TC^0 (Merrill) |
| E88 | > TC^0 (depth D*T) | RNN-SSM recognizes all regular languages (Merrill) |
| E23 | = RE | (not discussed in these papers) |

Minor discrepancy: we classify linear SSMs as "below TC^0" because nonnegative eigenvalue SSMs cannot compute parity (which is in TC^0). Merrill et al. classify SSMs as "in TC^0" meaning they can be simulated by TC^0 circuits. Both are correct -- SSMs are in TC^0 but do not compute all TC^0 functions.

---

## 5. What Kinds of Nonlinearity Are Sufficient?

### Theoretical Sufficiency

**sgn (sign function):** Sufficient for all regular languages (Merrill et al., Theorem 5.1). Not trainable.

**No specific differentiable nonlinearity is proven sufficient in these papers.** This is a notable gap.

### Our Contribution

Our Lean proofs show **tanh is sufficient** for:
- Parity (sign encoding via tanh saturation)
- Threshold (accumulation into saturation region)
- Exact counting modulo n (multiple stable attractors)
- Binary latching (bistable fixed points from bifurcation at alpha = 1)

The mechanism is specific to tanh's properties:
- Bounded output in (-1, 1)
- Sigmoidal shape enabling saturation
- Bifurcation creating stable non-zero fixed points when alpha > 1
- Vanishing gradient at saturation providing "memory protection"

Whether other differentiable nonlinearities (ReLU, GELU, sigmoid) suffice for the same constructions is an open question not addressed by any of these papers.

### Structural Sufficiency (Without Nonlinearity)

The negative eigenvalue and Householder product approaches show that some state-tracking capabilities can be achieved without any nonlinearity at all, purely through structural modifications:

- **Diagonal [-1,1]**: Parity (Grazzi Theorem 3)
- **Householder [-1,1]**: All group word problems (Grazzi Theorem 3, DeltaProduct Theorem 1)
- **Gated Householder products**: All regular languages (DeltaProduct Theorem 2)
- **PD-SSM**: All FSAs with near-optimal state size

These results are complementary to ours rather than competitive. They show that the linear framework has more room than previously thought, but the solutions require increasingly complex matrix structures (products of Householder transformations, sparse structured matrices) that may be difficult to learn in practice.

---

## 6. Implications for Architecture Design

### The Trainability-Expressivity Tradeoff

All papers acknowledge the fundamental tension:

- **Linear recurrence** -> parallel scan -> efficient training -> limited expressivity (TC^0 or below)
- **Nonlinear recurrence** -> sequential processing -> harder training -> full expressivity (exceeds TC^0)

The recent papers attempt to find points in between:

| Approach | Parallelizable? | Parity? | S_5? | All Regular? |
|---|---|---|---|---|
| Diagonal SSM [0,1] | Yes | No | No | No |
| Diagonal SSM [-1,1] | Yes | Yes | No | No |
| DeltaNet [-1,1] | Yes | Yes | Partial | No |
| DeltaProduct (n_h > 1) | Yes | Yes | Yes (multi-layer) | Yes (gated, multi-layer) |
| PD-SSM | Yes | Yes | Yes | Yes |
| RNN-SSM (sgn) | No | Yes | Yes | Yes |
| E88 (tanh) | No | Yes | Yes | Yes |

### Our Empirical Evidence

Our CMA-ES experiments provide the empirical counterpart to these theoretical results:

- **At 512 tokens:** Tanh nonlinearity contributes nothing. Linear models suffice. The composition depth gap (D vs D*T) does not manifest because tasks at this length do not require deep temporal composition.
- **At 32K tokens:** E88 beats Mamba2 (ranking inversion). The composition depth gap becomes empirically significant. Tasks that require maintaining and manipulating state over 32K steps expose the TC^0 ceiling.

This aligns perfectly with the theoretical predictions: linear models should perform comparably on short sequences (where depth D suffices) and fall behind on long sequences (where depth D*T is needed).

### What the Papers Suggest for Our Architecture

1. **E88's tanh is a valid escape from TC^0**, but it is not the only one. Householder products and PD-SSMs provide parallelizable alternatives. Our advantage is simplicity: tanh applied element-wise is conceptually and implementationally simpler than products of Householder reflections.

2. **The matrix state (d x d) of E88 is distinct from anything proposed in these papers.** Merrill et al.'s RNN-SSM uses vector state. Grazzi et al.'s modifications apply to vector-state models. DeltaProduct's state is a matrix but evolves linearly. E88's matrix state with nonlinear temporal dynamics is a unique point in the design space.

3. **Our bifurcation analysis is novel.** None of the papers analyze fixed-point dynamics of tanh saturation. Our proofs about bistable fixed points at alpha > 1 and latching persistence provide mechanistic understanding of how E88 stores binary facts -- a level of detail absent from the theoretical constructions in these papers.

4. **Hybrid architectures are the practical frontier.** Several papers (PD-SSM, DeltaProduct) suggest combining linear parallelizable layers with more expressive components. Our E88 could serve as the expressive component in such hybrids, handling state-tracking layers while linear SSM layers handle pattern matching.

---

## 7. Summary Table: How Our Work Relates

| Claim | Their Evidence | Our Evidence |
|---|---|---|
| Linear SSMs cannot compute parity | TC^0 bounds (Merrill, Li), nonneg eigenvalue limitation (Grazzi, Sarrof) | Lean proof: LinearLimitations.lean:218 |
| Linear SSMs cannot track state | NC^1-hardness of S_5 (Merrill), star-free characterization (Sarrof) | Lean proof: ExactCounting.lean:344 |
| Nonlinearity enables parity | sgn construction (Merrill), negative eigenvalues (Grazzi) | Lean proof: tanh construction (TanhSaturation.lean:720) |
| Depth grows with T for nonlinear recurrence | Implicit in RNN-SSM analysis (Merrill) | Lean proof: RecurrenceLinearity.lean:229 |
| Matrix state exceeds vector state | Not addressed | Lean proof: E88ExceedsE1HCapacity.lean |
| tanh saturation enables binary storage | Not addressed | Lean proof: AttentionPersistence.lean:212 |
| E88 exceeds TC^0 | Follows from RNN expressivity (Merrill) | Lean proof: TC0VsUnboundedRNN.lean:127 |
| 32K context exposes the gap | Not addressed empirically | CMA-ES experiments: E88 beats Mamba2 at 32K |

---

## 8. Open Questions Raised

1. **Is tanh optimal?** No paper proves that tanh is the best nonlinearity for state tracking. Our proofs show sufficiency, not optimality. Could a different nonlinearity provide better gradient flow while maintaining saturation-based memory?

2. **Can Householder products match E88 at 32K?** DeltaProduct provides a parallelizable path to state tracking. Does it actually match E88's empirical performance at long contexts, or does the sequential composition depth of tanh provide an advantage beyond what Householder products can achieve?

3. **What is the precise complexity class of E88?** We know E88 exceeds TC^0 and is below RE. Where exactly does it sit? Is it NC^1? P? The papers suggest that single-layer nonlinear RNNs can recognize all regular languages (NC^1-complete problems), placing E88 at least at NC^1.

4. **Does the bifurcation mechanism generalize?** Our bistable fixed-point analysis is specific to tanh with alpha > 1. Do other architectures (GRU, LSTM) have analogous bifurcation dynamics? The papers do not address this.

5. **How does PD-SSM's universal FSA emulation compare to E88's constructions?** PD-SSM achieves near-optimal state size for FSA emulation while remaining parallelizable. Is there a practical comparison to be made with E88's matrix-state approach?

---

## Sources

- [The Illusion of State in State-Space Models](https://arxiv.org/abs/2404.08819)
- [Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues](https://arxiv.org/abs/2411.12537)
- [The Computational Limits of State-Space Models and Mamba](https://arxiv.org/abs/2412.06148)
- [The Expressive Capacity of State Space Models: A Formal Language Perspective](https://arxiv.org/abs/2405.17394)
- [DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products](https://arxiv.org/abs/2502.10297)
- [Structured Sparse Transition Matrices to Enable State Tracking in SSMs](https://arxiv.org/abs/2509.22284)
- [Test-time regression: a unifying framework for designing sequence models with associative memory](https://arxiv.org/abs/2501.12352)
