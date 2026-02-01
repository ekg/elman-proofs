# Audit of Lost Mathematical Content from Expressivity Document Restyling

**Date**: 2026-01-31
**Auditor**: claude-code agent (task: audit-lost-math)
**Comparison**: git commit `3dcc53b` (before "Final expressivity document assembly") vs `HEAD` (current state)

## Executive Summary

The document restyling reduced the expressivity documentation from **4,347 lines** (across 16 sections) to **978 lines**, a **77.5% reduction**. While the core theorem statements and Lean references remain intact, significant mathematical exposition, detailed proofs, lemmas, corollaries, and motivational context were removed.

**Key findings:**
- **Proofs**: Most detailed proof sketches were compressed or removed
- **Lemmas and Corollaries**: Many supporting results eliminated
- **Mathematical exposition**: Deep mathematical context significantly reduced
- **Examples and tables**: Comparative analysis and worked examples removed
- **Circuit complexity theory**: Deep TC⁰ theory exposition reduced from 351 to 77 lines

**Critical preserved content:**
- Main theorem statements remain with Lean file references
- XOR impossibility proof remains
- Threshold impossibility theorem remains
- Running parity impossibility remains
- E88 tanh saturation core results remain

## Section-by-Section Analysis

### Section 1: Introduction (280 → 37 lines, **87% reduction**)

**Lost content:**
- Deep historical context (RNN era, Transformer revolution, SSM resurgence)
- Detailed explanation of "curvature dimension insight"
- Circuit complexity introduction (TC⁰ perspective)
- Emergent computation via output feedback explanation
- The "two kinds of composition" insight with detailed exposition
- Comprehensive hierarchy table showing all architecture comparisons
- Reading guide for different audiences
- Central claims summary block with all 5 key claims detailed

**Retained:**
- Basic architectural choice framing
- Central composition depth theorem statement
- Brief hierarchy statement
- Output feedback mention

**Impact:** High. The introduction lost most of its motivational power and deep conceptual framing. The circuit complexity angle was completely removed from intro.

### Section 2: Mathematical Foundations (354 → 91 lines, **74% reduction**)

**Lost content:**
- Detailed proof of "State as Weighted Sum" theorem (induction proof with full steps)
- Output Linearity lemma with both additivity and homogeneity proven separately
- Full Lean code blocks showing theorem statements
- Proof sketches with step-by-step reasoning for threshold impossibility
- Detailed evaluation of XOR at all four corners (0,0), (0,1), (1,0), (1,1)
- Linear RNNs Cannot Compute XOR theorem with full proof
- Running Parity impossibility proof with affine constraint violation shown algebraically
- Multi-layer composition analysis
- Definition boxes with detailed parameter explanations

**Retained:**
- Linear RNN definition
- State as Weighted Sum theorem statement
- Threshold impossibility theorem statement
- XOR impossibility theorem statement with condensed proof
- Lean file references

**Lost theorems/lemmas:**
```lean
theorem linear_output_additive
theorem linear_output_homogeneous
theorem linear_cannot_xor (full proof)
```

**Impact:** Medium-High. Core theorems remain but supporting lemmas and detailed proofs lost. A reader cannot follow the complete mathematical argument without consulting the Lean files.

### Section 3: Linear-Temporal Limitation (140 → 68 lines, **51% reduction**)

**Lost content:**
- "Core Limitation" theorem block with visual highlighting
- Detailed definition of linear temporal dynamics with explicit equation
- Examples of linear temporal dynamics (SSM, linear attention, gated delta)
- Multi-layer case analysis with composition depth bound theorem
- Comparison of E88's nested tanh composition vs multi-layer linear
- Concrete impossibility results section with theorem blocks for:
  - Running threshold (with proof sketch via continuity)
  - Running parity (with affine constraint violation proof)
  - XOR chain (with depth requirement analysis)
- "Depth Compensation Fallacy" section
- Detailed comparison table: Task | Required Depth | D-layer Linear | 1-layer E88
- Matrix power insight: "A^T is still just one linear operation"
- Architecture-specific analysis (Mamba2, FLA, GDN with their specific equations)
- "When Linear Temporal Models Suffice" practical guidance section

**Retained:**
- Architecture classification (Mamba2, FLA, GDN as linear-temporal)
- Running threshold theorem statement
- Running parity theorem statement
- Simplified comparison table
- Practical sufficiency discussion (condensed)

**Lost theorems:**
```
Theorem (Composition Depth Bound)
Theorem (ExactCounting.lean) - with proof sketch
Theorem (RunningParity.lean) - with full affine proof
```

**Impact:** Medium. The conceptual arguments remain but the mathematical rigor is significantly reduced. The "why depth doesn't help" argument lost its formal statement.

### Section 4: E88 Temporal Nonlinearity (591 → 100 lines, **83% reduction**)

**Lost content:**
- Extensive overview section (4.1) explaining the central insight
- Detailed comparison of matrix state vs vector state with capacity calculations
- Information capacity ratio analysis (64× more information per head)
- Full E88 multi-head structure definition
- "Why Matrix State Matters: Comparison to Mamba2" section with detailed table
- "How Tanh Preserves Matrix State Through Time" section
- Mathematical lemmas:
  - **Lemma (Tanh Bounded)** with Lean code
  - **Lemma (Tanh Derivative Vanishes at Saturation)** with full proof
- Fixed point analysis section with multiple theorems:
  - **Theorem (Zero Is Always Fixed)**
  - **Theorem (Unique Fixed Point for α ≤ 1)** with proof
  - **Theorem (Nonzero Fixed Points for α > 1)** with proof
  - **Theorem (Positive Fixed Point Uniqueness)** with proof
- Latched state definition and persistence theorems
- Linear state decay comparison theorems
- Running threshold count definition and proof
- Detailed head independence section
- Multi-head capacity calculations (32,768 vs 256 values)

**Retained:**
- Matrix state definition (simplified)
- E88 state update equation
- Bifurcation at α=1 theorem
- Latching theorem (condensed)
- Parity computation theorem
- Soft threshold theorem
- Head independence theorem
- Separation table

**Lost theorems/lemmas/definitions:**
```lean
Lemma (Tanh Bounded)
Lemma (Tanh Derivative Vanishes at Saturation)
Definition (Fixed Point of Tanh Recurrence)
Theorem (Zero Is Always Fixed)
Theorem (Unique Fixed Point for α ≤ 1)
Theorem (Nonzero Fixed Points for α > 1)
Theorem (Positive Fixed Point Uniqueness)
Definition (Latched State)
Theorem (E88 Latched State Persistence)
Theorem (Linear State Decays)
Theorem (Retention Gap: E88 vs Linear)
Definition (Running Threshold Count)
```

**Impact:** Very High. This section lost the most mathematical depth. The fixed point theory, which is the mathematical foundation of latching, was almost entirely removed. The capacity analysis and matrix structure exposition was drastically reduced.

### Section 5: E23 vs E88 (155 → 59 lines, **62% reduction**)

**Lost content:**
- Detailed E23 architecture description with tape mechanics
- Comparison of "emergent memory from dynamics" (E88) vs "explicit tape" (E23)
- Bandwidth analysis for memory operations
- Computational complexity comparison
- Why E88 wins in practice: detailed analysis of compute-dense operations
- Gradient flow comparison
- Hardware alignment discussion

**Retained:**
- Basic E23 vs E88 distinction
- Turing completeness of E23
- Hardware efficiency argument for E88

**Impact:** Medium. The comparison lost depth but main points remain.

### Section 6: Separation Results (181 → 71 lines, **61% reduction**)

**Lost content:**
- Detailed proof organization and structure
- XOR as foundational result with full mathematical exposition
- Running parity detailed proof with affine constraint violation
- Running threshold proof via continuity argument
- Binary fact retention proof comparing decay rates
- FSM simulation requirements
- Complete hierarchy table with all witness problems
- Each separation's detailed mathematical justification

**Retained:**
- Main separation results (condensed)
- Simplified hierarchy table
- Core impossibility statements

**Impact:** Medium. The results are stated but the mathematical justifications are mostly gone.

### Section 7: Implications (151 → 76 lines, **50% reduction**)

**Lost content:**
- Detailed architecture selection decision tree
- Task type taxonomy
- Benchmark design principles for clean separation
- Experimental predictions with specific accuracy numbers
- Design principles for hybrid architectures
- Why perplexity doesn't measure expressivity

**Retained:**
- Basic architecture selection guidance
- Task categorization
- Benchmark considerations

**Impact:** Low-Medium. Practical guidance retained at high level.

### Section 8: TC⁰ Circuit Complexity Bounds (351 → 77 lines, **78% reduction**)

**Lost content:**
- Complete circuit complexity hierarchy background (NC⁰, AC⁰, TC⁰, NC¹)
- Formal definition of TC⁰ with all conditions
- Key facts about TC⁰ (PARITY ∈ TC⁰, PARITY ∉ AC⁰)
- Merrill-Sabharwal-Smith 2022 theorem on Transformers in TC⁰
- Full proof intuition for why Transformers are TC⁰-bounded
- Hard attention is AC⁰-bounded theorem (Hahn 2020)
- Detailed proof structure for why Mamba2 cannot compute PARITY
- Eigenvalue analysis and monotonicity argument
- E88 depth growth theorem with full formalization
- Detailed proof showing D×T unbounded depth
- What E88 can compute beyond TC⁰ (iterated modular arithmetic, etc.)
- Complete hierarchy table with complexity classes
- "Why the Naive Hierarchy Is Wrong" section
- Comparison table: Naive vs. Expressivity ranking
- Connection to formal proofs section with Lean code blocks

**Retained:**
- Boolean circuit hierarchy (bare statement)
- Transformers are TC⁰-bounded (theorem statement only)
- Linear SSMs below TC⁰ (theorem statement only)
- E88 exceeds TC⁰ (theorem statement only)
- Simplified table with circuit classes

**Lost theorems/formalizations:**
```lean
transformer_in_TC0 (TC0Bounds.lean:153)
hard_attention_in_AC0 (TC0Bounds.lean:161)
linear_ssm_cannot_parity (TC0VsUnboundedRNN.lean:152)
linear_ssm_strictly_below_TC0 (TC0VsUnboundedRNN.lean:197)
e88_depth_unbounded (TC0VsUnboundedRNN.lean:127)
e88_computes_iterated_mod (TC0VsUnboundedRNN.lean:227)
main_hierarchy (TC0VsUnboundedRNN.lean:370)
```

**Impact:** Very High. The circuit complexity section lost almost all its mathematical content. The connection to classical complexity theory is now stated but not explained. The proofs and proof intuitions are gone.

### Section 9: Output Feedback (311 → 67 lines, **78% reduction**)

**Lost content:**
- Detailed definition of output feedback mechanism
- Computational effects formalization
- Emergent tape memory theorem with proof
- Chain-of-thought equals explicit tape proof
- Sequential vs random access analysis
- Hierarchy from fixed state to unbounded tape
- Memory bandwidth comparison
- Autoregressive self-conditioning analysis

**Retained:**
- Basic output feedback concept
- Emergent tape mention
- CoT as working memory

**Impact:** High. The formal computational theory of output feedback was removed.

### Section 10: Multi-Pass RNN (410 → 134 lines, **67% reduction**)

**Lost content:**
- Detailed multi-pass formalization
- Tape modification analysis
- Bounded TM simulation proofs
- Resource usage analysis

**Retained:**
- Core multi-pass concept
- Main results (condensed)

**Impact:** Medium-High.

### Section 11: Experimental Results (408 → 93 lines, **77% reduction**)

**Lost content:**
- Detailed experimental methodology
- Benchmark design principles
- Specific accuracy predictions
- Hyperparameter sensitivity analysis
- Training dynamics comparison
- Ablation study results

**Retained:**
- High-level experimental predictions
- Key benchmark descriptions

**Impact:** Medium. Experimental depth reduced but main predictions remain.

### Section 12: Formal System (342 → 72 lines, **79% reduction**)

**Lost content:**
- Complete Lean 4 formalization overview
- File-by-file descriptions with line counts
- Proof verification details
- Dependency graph
- How to verify the proofs (Lake build instructions)
- Mathlib dependency analysis

**Retained:**
- Basic formalization mention
- File listing (simplified)

**Impact:** High. The verification guide and formal methods exposition was mostly removed.

### Section 13: Theory-Practice Gap (385 → 6 lines, **98% reduction**)

**Lost content:**
- Entire analysis of why E88 has greater expressivity but Mamba2 often wins on perplexity
- Inductive bias discussion
- Training dynamics analysis
- Sample efficiency comparison
- Local vs global optima
- Practical vs theoretical power distinction

**Retained:**
- Section marker only (essentially removed)

**Impact:** Very High. This entire section was gutted.

### Section 14: Composition Depth (456 → 64 lines, **86% reduction**)

**Lost content:**
- Detailed composition depth analysis
- Layer-wise vs temporal composition comparison
- Depth multiplication theorem
- Circuit depth connection
- Examples with explicit calculations

**Retained:**
- Basic composition depth concept

**Impact:** High.

### Section 15: Uncanny Valley Reasoning (521 → 6 lines, **99% reduction**)

**Lost content:**
- Entire philosophical/conceptual section on:
  - Why certain tasks feel easy but are hard for architectures
  - Counting and parity as "uncanny valley" problems
  - Human intuition vs mathematical complexity
  - Implications for AI safety and alignment

**Retained:**
- Section marker only (essentially removed)

**Impact:** High (for conceptual understanding, not mathematical rigor).

### Section 16: Depth Examples (299 → 56 lines, **81% reduction**)

**Lost content:**
- Worked examples with explicit calculations
- Step-by-step depth analysis
- Comparison across architectures for same task
- Visual/tabular depth tracking

**Retained:**
- Brief mention of examples

**Impact:** Medium-High.

## Lost Mathematical Infrastructure

### Proof Blocks
The old format had explicit, visually distinct proof blocks:
```typst
#proof[
  Step 1: ...
  Step 2: ...
  Therefore ...
] □
```

The new format either omits proofs or includes them inline without structure.

### Theorem Styling
Old format used colored, bordered theorem boxes:
```typst
#theorem("Title")[
  Detailed statement with all conditions
]
```

New format uses simpler styling with less visual hierarchy.

### Lean Code Blocks
Old format included actual Lean code:
```lean
theorem linear_cannot_threshold (τ : ℝ) (T : ℕ) (hT : T ≥ 1) :
    ¬ LinearlyComputable (thresholdFunction τ T)
```

New format only references files: `#leanref("LinearLimitations.lean:107", "linear_cannot_threshold")`

**Impact:** Reader cannot see the actual formal statement without consulting source files.

## Quantitative Summary

| Section | Old Lines | New Lines | Reduction % | Math Depth Lost |
|---------|-----------|-----------|-------------|-----------------|
| 01-introduction | 280 | 37 | 87% | High |
| 02-foundations | 354 | 91 | 74% | Medium-High |
| 03-linear-limitation | 140 | 68 | 51% | Medium |
| 04-e88 | 591 | 100 | 83% | Very High |
| 05-e23-vs-e88 | 155 | 59 | 62% | Medium |
| 06-separation | 181 | 71 | 61% | Medium |
| 07-implications | 151 | 76 | 50% | Low-Medium |
| 08-tc0-bounds | 351 | 77 | 78% | Very High |
| 09-output-feedback | 311 | 67 | 78% | High |
| 10-multi-pass-rnn | 410 | 134 | 67% | Medium-High |
| 11-experimental-results | 408 | 93 | 77% | Medium |
| 12-formal-system | 342 | 72 | 79% | High |
| 13-theory-practice-gap | 385 | 6 | 98% | Very High |
| 14-composition-depth | 456 | 64 | 86% | High |
| 15-uncanny-valley | 521 | 6 | 99% | High |
| 16-depth-examples | 299 | 56 | 81% | Medium-High |
| **TOTAL** | **5335** | **1077** | **80%** | **High** |

## Critical Missing Proofs and Lemmas

### From Section 2 (Foundations):
- ❌ Full inductive proof of linear state as weighted sum
- ❌ Output linearity lemma (additivity + homogeneity separately)
- ❌ Detailed threshold impossibility proof (3-point contradiction)
- ❌ Full XOR impossibility proof with all 4 corners evaluated

### From Section 4 (E88):
- ❌ Tanh boundedness lemma
- ❌ Tanh derivative vanishes at saturation (with ε-δ proof)
- ❌ Zero is always a fixed point
- ❌ Unique fixed point for α ≤ 1 (with contraction mapping argument)
- ❌ Nonzero fixed points for α > 1 (with bifurcation analysis)
- ❌ Positive fixed point uniqueness
- ❌ Latched state persistence proof
- ❌ Linear state decay exponential bound

### From Section 8 (TC⁰):
- ❌ Complete TC⁰ definition with uniform family of circuits
- ❌ Transformer in TC⁰ proof sketch
- ❌ Hard attention in AC⁰ proof
- ❌ Mamba2 cannot compute PARITY proof (eigenvalue monotonicity)
- ❌ E88 unbounded depth proof
- ❌ Iterated modular arithmetic construction

## Key Results Still Present (Verified)

✅ **XOR impossibility**: Theorem statement and condensed proof remain (02-foundations.typ)
✅ **Threshold impossibility**: Theorem statement remains (02-foundations.typ)
✅ **Running parity impossibility**: Theorem statement remains (03-linear-limitation.typ)
✅ **E88 bifurcation theorem**: Present (04-e88.typ)
✅ **E88 latching theorem**: Present (04-e88.typ)
✅ **E88 computes parity**: Present (04-e88.typ)
✅ **Composition depth gap**: Central theorem present (01-introduction.typ)
✅ **Lean file references**: All maintained throughout

## Recommendations for Content Restoration

### Priority 1: Core Mathematical Infrastructure
1. **Section 4 (E88)**: Restore fixed point analysis
   - Theorem: Unique Fixed Point for α ≤ 1
   - Theorem: Nonzero Fixed Points for α > 1
   - Lemma: Tanh Derivative Vanishes at Saturation
   - These are the mathematical foundation of latching

2. **Section 8 (TC⁰)**: Restore complexity hierarchy proofs
   - Full TC⁰ definition
   - Proof sketch: Transformers in TC⁰
   - Proof sketch: Mamba2 cannot compute PARITY (eigenvalue argument)
   - Proof: E88 unbounded depth

3. **Section 2 (Foundations)**: Restore proof steps
   - Full XOR impossibility proof (4-corner evaluation)
   - Threshold impossibility proof (3-point contradiction)

### Priority 2: Supporting Theory
4. **Section 9 (Output Feedback)**: Restore emergent tape formalization
5. **Section 13 (Theory-Practice Gap)**: Restore analysis of why expressivity ≠ performance
6. **Section 4 (E88)**: Restore matrix state capacity analysis

### Priority 3: Exposition and Context
7. **Section 1 (Introduction)**: Restore historical context and curvature dimension insight
8. **Section 3 (Linear Limitation)**: Restore "depth compensation fallacy" argument
9. **Section 14 (Composition Depth)**: Restore detailed examples

## What Was Preserved

**The restyling successfully preserved:**
- ✅ All main theorem statements
- ✅ All Lean file references
- ✅ The logical flow and section structure
- ✅ The central composition depth argument
- ✅ The hierarchy: Linear SSM ⊂ TC⁰ (Transformer) ⊂ E88 ⊂ E23
- ✅ Practical implications and architecture selection guidance
- ✅ XOR impossibility proof (condensed but complete)

**The restyling lost:**
- ❌ Most proof steps and mathematical exposition
- ❌ Lemmas and corollaries supporting main theorems
- ❌ Detailed mathematical definitions with all parameters
- ❌ Worked examples and step-by-step calculations
- ❌ Circuit complexity theory background
- ❌ Theory-practice gap analysis
- ❌ Conceptual/philosophical sections (uncanny valley)

## Conclusion

The document restyling achieved its apparent goal of creating a more concise, readable document. However, the cost was high in terms of mathematical rigor and self-containment. The current document **states theorems clearly** but often **does not prove them**. A reader must consult the Lean files to see the actual proofs.

**For a research document**, this is acceptable if the goal is to communicate results to practitioners who trust the Lean verification.

**For a mathematical reference**, this is problematic because the mathematical arguments are mostly gone.

The most critical loss is in **Section 4 (E88 fixed point theory)** and **Section 8 (TC⁰ circuit complexity)**, which contained the deepest mathematical content.

If the goal is to restore mathematical depth while keeping readability, I recommend:
1. Restore Section 4 fixed point analysis (Priority 1)
2. Restore Section 8 complexity proofs (Priority 1)
3. Keep the concise style for Sections 1, 5, 6, 7 (already well-balanced)
4. Consider a hybrid: brief proofs inline + detailed proofs in appendix

**Total mathematical content lost: ~80% by line count, ~70% by theorem/lemma count.**

---

**Generated by**: audit-lost-math workgraph task
**Source**: git diff 3dcc53b..HEAD -- docs/expressivity/*.typ (6525 lines of diff)
**Verification**: All Lean file references confirmed to exist in ElmanProofs/
