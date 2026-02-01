# Academic Paper Structure Research: ML Theory Papers

Research on how top machine learning theory papers structure their content, particularly for RNN/Transformer expressivity and computational complexity results.

**Date**: 2026-01-31
**Purpose**: Guide the presentation of Elman RNN expressivity proofs in academic format

---

## Executive Summary

Top ML theory papers at NeurIPS, ICML, and ICLR follow a consistent pattern:

1. **Main text** (8-10 pages): Intuition → Formal statements → Proof sketches
2. **Appendix**: Complete rigorous proofs with technical lemmas
3. **Balance**: ~60% prose/intuition, ~40% formal math in main body
4. **Theorem presentation**: Motivate → State formally → Sketch intuition → Defer full proof

---

## Key Findings

### 1. Proof Placement Strategy

**Official Guidelines** (from NeurIPS/ICML):

> "Proofs can appear in either the main paper or the supplemental material, but if they appear in the supplemental material, authors are encouraged to provide a short proof sketch to provide intuition." (NeurIPS Paper Checklist)

> "Any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material." (NeurIPS Guidelines)

> "Reviewers are not required to read the appendix." (Both NeurIPS and ICML)

**Common Practice**:
- Main body: Theorem statements + proof sketches (1-2 paragraphs of intuition)
- Appendix: Complete formal proofs with all technical lemmas
- Pattern: "All proofs can be found in the Appendix" after main theorems

### 2. Intuition vs Rigor Balance

**Early Sections (Introduction, Background)**:
- Heavy prose with gradual formalization
- Descriptive language alongside equations
- Visual diagrams and color-coded explanations
- 70-80% prose, 20-30% math

**Middle Sections (Main Results)**:
- Theorem-centric with contextual explanation
- Formal definitions inline with minimal preamble
- "Remark" and "Example" blocks for intuition
- 40-50% prose, 50-60% math

**Late Sections (Technical Results)**:
- Dense notation and formal statements
- Brief contextual notes only
- Assumes substantial technical background
- 20-30% prose, 70-80% math

### 3. Theorem Statement Style

**Best Practice** (from ICML):

> "Theorem statements should be self-contained, or the reader will find it easy to identify in the paper the conditions under which the statements hold. A superbly written paper will also give intuitive arguments for why the statements hold."

**Typical Structure**:
1. Informal motivation paragraph
2. Formal theorem statement with explicit assumptions
3. Brief intuition/proof sketch (2-5 sentences)
4. Reference to appendix for full proof

**Example Pattern**:
```
We now characterize which functions can be computed by linear RNNs.

**Theorem 4.1** (Linear RNN Expressivity). Let f: Σ* → ℝ be computable
by a linear RNN with hidden dimension d. Then f can be expressed as a
weighted sum of at most d features, where each feature is a regular
language over Σ.

**Proof sketch.** The key insight is that linear state updates form a
semigroup under composition. Each hidden dimension evolves independently
as a weighted finite automaton. Full proof in Appendix A.2.
```

---

## Exemplar Papers Analysis

### Paper 1: "Theoretical Foundations of Deep Selective State-Space Models"
**Venue**: NeurIPS 2024
**Authors**: Muca Cirone et al.
**Topic**: SSM/Mamba expressivity using rough path theory

**Structure**:
1. Introduction - Motivation and context (2 pages)
2. State-space Models - Review of S4, Mamba, GLA (2 pages)
3. SSMs as Linear CDEs - Theoretical framework (2 pages)
4. Expressivity of Linear CDEs - Core results (2 pages)
5. Path-to-Path Learning - Additional theory (1 page)
6. Empirical Validation - Experiments (1 page)
7. Conclusions (0.5 pages)

**Key Insights**:
- All major proofs deferred to appendix
- Section 4 contains 6 theorems with minimal prose
- Heavy mathematical machinery in appendix (rough paths, signatures)
- Main text focuses on intuition and implications

**Notable Quote**: "The diagonal weight choice considerably restricts the family of learnable functionals, as it corresponds to running N independent 1-dimensional systems."

**Proof Strategy**: State theorem → Brief remark → "All proofs can be found in the Appendix"

---

### Paper 2: "Saturated Transformers are Constant-Depth Threshold Circuits"
**Venue**: TACL 2022 / NeurIPS
**Authors**: William Merrill, Ashish Sabharwal, Noah A. Smith
**Topic**: Circuit complexity characterization of transformers

**Structure** (inferred from abstract):
- Connects transformer expressivity to TC⁰ complexity class
- Uses circuit complexity as formal framework
- Builds on prior work showing hard-attention transformers limited to AC⁰

**Key Contribution**:
> "Saturated transformers with floating-point values can be simulated by constant-depth threshold circuits, giving the class TC⁰ as an upper bound on the formal languages they recognize."

**Theoretical Approach**:
- Formal language recognition as the task
- Circuit simulation as the proof technique
- Complexity classes (AC⁰, TC⁰) as the organizing framework

---

### Paper 3: "On the Turing Completeness of Modern Neural Network Architectures"
**Venue**: ICLR 2019 → JMLR 2021
**Authors**: Pérez, Marinković, Barceló
**Topic**: Turing completeness of Transformers and Neural GPU

**Structure**:
- Proves Transformers are Turing complete with positional encodings
- Shows completeness with "bounded architectures" (fixed neurons/params)
- Uses hard attention and piecewise linear activations in proofs

**Key Result**:
> "Neither the Transformer nor the Neural GPU requires access to external additional memory to become Turing complete."

**Proof Technique**:
- Constructive: Shows how to simulate Turing machine
- Complexity analysis: Bounded vs unbounded precision

---

### Paper 4: "RNNs Are Not Transformers (Yet)"
**Venue**: ICLR 2025
**Authors**: Multiple authors
**Topic**: CoT benefits for RNNs vs Transformers

**Structure**:
- Main body: Theorem statements + empirical validation
- Appendix E.5: Lower bound proofs using communication complexity
- Technical framework: Information-theoretic arguments

**Key Results**:
> "RNNs with O(log n) bit memory cannot solve tasks like Index, AR, c-gram retrieval, and Counting, while Transformers can solve them perfectly with constant size."

**Proof Organization**:
- Communication complexity framework in main text
- Information-theoretic lower bounds in appendix
- Separation results stated as theorems with proof sketches

---

### Paper 5: "Separations in the Representational Capabilities of Transformers and Recurrent Architectures"
**Venue**: NeurIPS 2024
**Authors**: Multiple authors
**Topic**: Task-specific architectural separations

**Key Results**:
- Index lookup: 1-layer Transformer (log width) vs RNN (linear state)
- Dyck languages: Constant-size RNNs vs 1-layer Transformers (linear size)
- Decision tasks: 2-layer Transformers (log size) vs linear size for 1-layer/RNNs

**Proof Techniques**:
- Upper bounds: "Existence of N nearly orthogonal vectors in O(log N) dimensional space"
- Lower bounds: "Reductions from communication complexity problems"

**Empirical Validation**: Experiments validate theoretical findings on practical sequence lengths

---

## Common Structural Patterns

### Section Flow (Typical 8-10 Page Paper)

1. **Introduction** (1.5-2 pages)
   - Motivation and context
   - Overview of main contributions
   - Related work summary
   - Paper organization

2. **Background/Preliminaries** (1-1.5 pages)
   - Notation and definitions
   - Review of relevant architectures
   - Problem formalization

3. **Main Results** (3-4 pages)
   - Core theorems with proof sketches
   - Intuitive explanations
   - Examples and remarks

4. **Additional Results/Extensions** (1-2 pages)
   - Corollaries and variations
   - Computational considerations
   - Connections to related work

5. **Empirical Validation** (1-1.5 pages)
   - Experiments validating theory
   - Synthetic tasks demonstrating separations
   - Practical implications

6. **Conclusions** (0.5-1 page)
   - Summary of contributions
   - Open questions
   - Future directions

7. **Appendix** (No page limit)
   - Complete proofs
   - Technical lemmas
   - Additional experiments
   - Extended related work

### Mathematical Notation Conventions

**Gradual Introduction**:
- Define notation when first used
- Collect key notation in table (often in appendix)
- Use consistent symbols throughout

**Common Conventions**:
- Sequences: x₁, x₂, ..., xₜ or x_{1:t}
- Hidden states: h, s, or z
- Parameters: W, U, V for weight matrices
- Activation functions: σ, tanh, ReLU
- Time indices: t, i, j
- Dimensions: d, n, m

**Complexity Classes**:
- AC⁰: Constant-depth polynomial-size circuits with AND/OR gates
- TC⁰: AC⁰ + threshold gates
- NC¹: Log-depth polynomial-size circuits
- Regular languages: Recognizable by finite automata

---

## Recommendations for Elman Expressivity Paper

Based on this research, here's how to structure our Elman RNN expressivity results:

### 1. Main Text Organization

**Section 1: Introduction**
- Motivate: Why study Elman vs modern architectures?
- Preview: Linear limitations → Nonlinear capabilities
- Contributions: "We formally prove that..."

**Section 2: Preliminaries**
- Define Elman RNN precisely
- Define Mamba2 SSM for comparison
- Introduce notion of "threshold computation" and "parity"

**Section 3: Linear RNN Limitations**
- Theorem 3.1: Linear RNNs cannot compute XOR
- Theorem 3.2: Linear RNNs cannot compute threshold
- Proof sketches emphasizing key insight: linear = weighted sum

**Section 4: Nonlinear Elman Capabilities**
- Theorem 4.1: Tanh saturation enables binary latching
- Theorem 4.2: Exact modular counting via nested tanh
- Theorem 4.3: Running parity computation
- Proof sketches emphasizing: saturation → stable states

**Section 5: Architectural Comparison**
- Compare Elman, LSTM, GRU, Mamba2
- Characterize which architectures have "latching" capability
- Explain temporal vs depth expressivity trade-off

**Section 6: Empirical Validation**
- Synthetic tasks: XOR, parity, counting
- Show Elman achieves perfect accuracy; Mamba2 fails
- Validate theoretical predictions

**Section 7: Discussion and Conclusions**
- Implications for architecture design
- Open questions about multi-layer networks
- Future work

### 2. Proof Sketch Style

For each theorem, follow this template:

```markdown
**Theorem X.Y** (Descriptive Name). [Formal statement with explicit assumptions]

**Proof sketch.** [2-3 sentences explaining the key insight and proof technique. Example:]
"The key observation is that linear recurrences preserve the property that
h_t is a weighted sum of x₁,...,xₜ. Since XOR is not a linear functional
on binary inputs, it cannot be expressed this way. We formalize this via
the linear independence of XOR truth table rows. □"

**Remark X.Z.** [Optional: Intuition, examples, or practical implications]

*Full proof in Appendix X.*
```

### 3. Appendix Organization

**Appendix A: Complete Proofs of Section 3**
- A.1: Proof of Theorem 3.1 (Linear RNNs cannot compute XOR)
- A.2: Proof of Theorem 3.2 (Threshold impossibility)
- A.3: Supporting lemmas

**Appendix B: Complete Proofs of Section 4**
- B.1: Proof of Theorem 4.1 (Tanh saturation)
- B.2: Proof of Theorem 4.2 (Modular counting)
- B.3: Proof of Theorem 4.3 (Running parity)
- B.4: Technical lemmas on tanh dynamics

**Appendix C: Additional Results**
- C.1: Multi-layer extensions
- C.2: Attention mechanism analysis
- C.3: Head independence proofs

**Appendix D: Experimental Details**
- D.1: Task definitions
- D.2: Training procedures
- D.3: Hyperparameters
- D.4: Additional results

### 4. Mathematical Rigor Guidelines

**Main Text**:
- State theorems with complete formal conditions
- Give clear proof sketches (don't just say "see appendix")
- Use examples to build intuition
- Balance: ~50% prose, ~50% math in results sections

**Appendix**:
- Complete formal proofs using Lean-style structure
- State and prove all intermediate lemmas
- Cross-reference main text theorems
- Can be ~80% math, ~20% connecting prose

### 5. Writing Style

**Do**:
- Use active voice: "We prove that..." not "It is proven that..."
- Give intuition before formalism
- Use examples to illustrate theorems
- State assumptions explicitly
- Define all notation before use

**Don't**:
- Assume reader knows background beyond standard ML
- Use "clearly" or "obviously" (if it's clear, it doesn't need saying)
- Present theorems without motivation
- Hide assumptions in prose
- Use notation inconsistently

---

## References and Sources

### Primary Papers Analyzed

1. **Theoretical Foundations of Deep Selective State-Space Models**
   Nicola Muca Cirone et al., NeurIPS 2024
   [ArXiv: 2402.19047](https://arxiv.org/abs/2402.19047) | [NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/file/e6231c5f46598cfd09ff1970524e0436-Paper-Conference.pdf)

2. **Saturated Transformers are Constant-Depth Threshold Circuits**
   William Merrill, Ashish Sabharwal, Noah A. Smith, TACL 2022
   [ArXiv: 2106.16213](https://arxiv.org/abs/2106.16213) | [ACL Anthology](https://aclanthology.org/2022.tacl-1.49/)

3. **On the Turing Completeness of Modern Neural Network Architectures**
   Jorge Pérez, Javier Marinković, Pablo Barceló, ICLR 2019 → JMLR 2021
   [ArXiv: 1901.03429](https://arxiv.org/abs/1901.03429) | [OpenReview](https://openreview.net/forum?id=HyGBdo0qFm) | [JMLR](https://jmlr.org/papers/volume22/20-302/20-302.pdf)

4. **RNNs Are Not Transformers (Yet): The Key Bottleneck on In-Context Retrieval**
   ICLR 2025
   [ArXiv: 2402.18510](https://arxiv.org/html/2402.18510v2) | [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/79dc391a2c1067e9ac2b764e31a60377-Paper-Conference.pdf)

5. **Separations in the Representational Capabilities of Transformers and Recurrent Architectures**
   NeurIPS 2024
   [NeurIPS Virtual](https://neurips.cc/virtual/2024/poster/96535)

6. **Repeat After Me: Transformers are Better than State Space Models at Copying**
   ArXiv 2024
   [ArXiv: 2402.01032](https://arxiv.org/html/2402.01032v1) | [Kempner Institute](https://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/)

7. **Unlocking State-Tracking in Linear RNNs**
   ICLR 2025
   [ICLR Proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/5a0ce3abb720b740419e193c87afd080-Paper-Conference.pdf)

8. **A Formal Hierarchy of RNN Architectures**
   William Merrill, Gail Weiss et al., ACL 2020
   [ACL Anthology](https://aclanthology.org/2020.acl-main.43/) | [Semantic Scholar](https://www.semanticscholar.org/paper/A-Formal-Hierarchy-of-RNN-Architectures-Merrill-Weiss/45e5d7637a585a87d967a4a357d17c5d89aecea2)

9. **Rational Recurrences**
   Hao Peng, Roy Schwartz et al., EMNLP 2018
   [ArXiv: 1808.09357](https://arxiv.org/abs/1808.09357) | [Semantic Scholar](https://www.semanticscholar.org/paper/Rational-Recurrences-Peng-Schwartz/a56ebc39b8c527774be705cccdcb5f66c7302e0c)

### Conference Guidelines

10. **NeurIPS Paper Checklist Guidelines**
    [NeurIPS Guidelines](https://neurips.cc/public/guides/PaperChecklist)

11. **ICML Paper Best Practices**
    [ICML 2022 Best Practices](https://icml.cc/Conferences/2022/BestPractices) | [ICML 2024 Guidelines](https://icml.cc/Conferences/2024/PaperGuidelines)

12. **ICML Submission and Formatting Instructions**
    [ICML 2025 Example Paper](https://media.icml.cc/Conferences/ICML2025/Styles/example_paper.pdf)

### Additional Resources

13. **Formal Language Theory and Neural Networks**
    [Rycolab ESSLLI Course](https://rycolab.io/classes/esslli-23/)

14. **Computational Expressivity of Neural Language Models**
    Alexandra Butoi, Robin Chan, ACL 2024
    [ACL Anthology](https://aclanthology.org/2024.acl-tutorials.3v2.pdf)

15. **From S4 to Mamba: A Comprehensive Survey on Structured State Space Models**
    ArXiv 2025
    [ArXiv: 2503.18970](https://arxiv.org/abs/2503.18970)

---

## Key Takeaways for Elman Paper

1. **Proof strategy**: Complete proofs in appendix, intuitive sketches in main text
2. **Theorem style**: Motivate → State formally → Sketch → Reference appendix
3. **Balance**: ~60% prose/intuition, ~40% formal math in main body
4. **Structure**: 7 sections, ~8-10 pages main text + unlimited appendix
5. **Empirical validation**: Essential to demonstrate practical relevance of theory
6. **Comparison focus**: Position Elman vs Mamba2/Transformers with formal separations
7. **Circuit complexity**: Consider framing in terms of complexity classes (like TC⁰ approach)
8. **Formal languages**: Use parity, XOR, threshold as canonical test problems
9. **Writing style**: Active voice, intuition before formalism, examples throughout
10. **Appendix**: Organize by section correspondence, provide complete Lean-style proofs

---

*Generated: 2026-01-31*
*Research conducted for: Elman RNN Expressivity Paper*
*Next steps: Draft paper outline based on these findings*
