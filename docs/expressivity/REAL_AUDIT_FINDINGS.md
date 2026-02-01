# Real Audit of Expressivity Document
**Date**: 2026-02-01
**Task**: audit-and-fix
**Purpose**: Verify whether the "restoration" actually restored the mathematical content

## Executive Summary

The claim was that commit `0b952ab` "restored math content and add bibliography". Let me verify what was actually restored versus what the audit said was missing.

### What Was Actually Restored (Commit 0b952ab)

**Section 4 (E88)**:
- ✅ Lemma (Tanh Boundedness) - WITH proof
- ✅ Lemma (Tanh Derivative Vanishes at Saturation) - WITH proof
- ✅ Definition (Fixed Point of Tanh Recurrence)
- ✅ Theorem (Zero Is Always a Fixed Point) - WITH proof
- ✅ Theorem (Unique Fixed Point for α ≤ 1) - WITH proof sketch
- ✅ Theorem (Nonzero Fixed Points for α > 1) - WITH proof sketch
- ✅ Definition (Latched State)
- ✅ Theorem (E88 Latched State Persistence) - WITH proof sketch
- ✅ Theorem (Linear State Exponential Decay) - WITH proof
- ✅ Theorem (E88 Latches; Linear Decays) - stated

**Section 8 (TC⁰)**:
- ✅ Full TC⁰ definition with all 4 conditions
- ✅ Theorem (PARITY Separates AC⁰ from TC⁰) - WITH proof sketch
- ✅ Theorem (Transformers Are TC⁰-Bounded) - WITH detailed proof sketch
- ✅ Theorem (Linear SSMs Are Below TC⁰) - WITH Perron-Frobenius proof sketch
- ✅ Theorem (E88 Exceeds TC⁰) - WITH full proof (not just sketch!)

**Section 11 (Experimental)**:
- ✅ Verified CMA-ES table completeness

**Bibliography**:
- ✅ Added references.yml with 669 lines of IEEE citations
- ✅ Integrated into main.typ

### What Is Still Missing (According to Audit)

Let me now check section by section what the audit said was missing but is NOT in the current document.

## Section-by-Section Comparison

### Section 1: Introduction
**Current line count**: Need to check
**Audit said missing (87% reduction)**:
- Deep historical context (RNN era, Transformer revolution, SSM resurgence)
- Detailed explanation of "curvature dimension insight"
- Circuit complexity introduction in intro
- Emergent computation via output feedback explanation
- The "two kinds of composition" insight with detailed exposition
- Comprehensive hierarchy table showing all architecture comparisons
- Reading guide for different audiences
- Central claims summary block with all 5 key claims detailed

**Actually restored**: None of these

### Section 2: Foundations
**Audit said missing (74% reduction)**:
- ❌ Full inductive proof of linear state as weighted sum (only proof sketch remains)
- ❌ Output Linearity lemma (additivity + homogeneity separately)
- ❌ Full Lean code blocks showing theorem statements
- ❌ Detailed evaluation of XOR at all four corners (only condensed proof)
- ❌ Running Parity impossibility proof with affine constraint violation shown algebraically
- ❌ Multi-layer composition analysis (detailed)

**Current state**:
- Has theorem statements
- Has brief proofs
- Missing: detailed step-by-step proofs, lemmas broken out separately

### Section 3: Linear Limitation
**Audit said missing (51% reduction)**:
- ❌ "Core Limitation" theorem block with visual highlighting
- ❌ Detailed definition of linear temporal dynamics with explicit equation
- ❌ Examples of linear temporal dynamics (SSM, linear attention, gated delta)
- ❌ Multi-layer case analysis with composition depth bound theorem
- ❌ Comparison of E88's nested tanh composition vs multi-layer linear
- ❌ Concrete impossibility results section with theorem blocks
- ❌ Running threshold proof sketch via continuity
- ❌ Running parity affine constraint violation proof
- ❌ XOR chain depth requirement analysis
- ❌ "Depth Compensation Fallacy" section
- ❌ Detailed comparison table
- ❌ Matrix power insight: "A^T is still just one linear operation"
- ❌ Architecture-specific analysis (Mamba2, FLA, GDN with equations)
- ❌ "When Linear Temporal Models Suffice" practical guidance section

**Current state**: Very condensed (63 lines). Most mathematical exposition gone.

### Section 4: E88
**Audit said missing (83% reduction, "Very High" impact)**:

**What audit claimed was missing**:
- ❌ Extensive overview section (4.1) explaining the central insight
- ❌ Detailed comparison of matrix state vs vector state with capacity calculations
- ❌ Information capacity ratio analysis (64× more information per head)
- ❌ Full E88 multi-head structure definition
- ❌ "Why Matrix State Matters: Comparison to Mamba2" section with detailed table
- ❌ "How Tanh Preserves Matrix State Through Time" section
- Many lemmas/theorems listed as lost in audit

**What was ACTUALLY restored in 0b952ab**:
- ✅ Lemma (Tanh Bounded) WITH proof
- ✅ Lemma (Tanh Derivative Vanishes) WITH proof
- ✅ All fixed point theorems WITH proofs/sketches
- ✅ Latched state definition and theorem
- ✅ Linear decay theorem

**What is STILL missing**:
- ❌ Detailed overview/motivation (4.1)
- ❌ Matrix vs vector capacity calculations (table comparing E88 vs Mamba2)
- ❌ Full multi-head structure formalization
- ❌ "How Tanh Preserves Matrix State" exposition
- ❌ Definition (Running Threshold Count) - this was in the audit as missing
- ❌ Many supporting lemmas for the main theorems

**Current state**: 153 lines (up from ~100 before restoration). This is a PARTIAL restoration. Key fixed point theory is back, but contextual exposition is still missing.

### Section 8: TC⁰ Bounds
**Audit said missing (78% reduction, "Very High" impact)**:

**What audit claimed was missing**:
- Complete circuit complexity hierarchy background (NC⁰, AC⁰, TC⁰, NC¹)
- Formal definition of TC⁰ with all conditions
- Key facts about TC⁰
- Merrill-Sabharwal-Smith 2022 theorem on Transformers in TC⁰
- Full proof intuition for why Transformers are TC⁰-bounded
- Hard attention is AC⁰-bounded theorem (Hahn 2020)
- Detailed proof structure for why Mamba2 cannot compute PARITY
- Eigenvalue analysis and monotonicity argument
- E88 depth growth theorem with full formalization
- What E88 can compute beyond TC⁰ (iterated modular arithmetic, etc.)
- Complete hierarchy table with complexity classes
- "Why the Naive Hierarchy Is Wrong" section

**What was ACTUALLY restored**:
- ✅ Full TC⁰ definition (4 conditions)
- ✅ Circuit hierarchy exposition (NC⁰ ⊂ AC⁰ ⊂ TC⁰ ⊂ NC¹ ⊂ P)
- ✅ PARITY separation theorem WITH proof sketch
- ✅ Transformers in TC⁰ WITH detailed proof sketch
- ✅ Linear SSMs below TC⁰ WITH Perron-Frobenius proof sketch
- ✅ E88 exceeds TC⁰ WITH full proof
- ✅ "The Inverted Ranking" section

**What is STILL missing**:
- ❌ Hard attention in AC⁰ theorem (Hahn 2020)
- ❌ Iterated modular arithmetic examples
- ❌ "Why the Naive Hierarchy Is Wrong" detailed exposition
- ❌ Some Lean code references that were in original

**Current state**: 108 lines (up from ~77). This is a SUBSTANTIAL restoration. Most of the key mathematical content is back.

### Remaining Sections Quick Check

**Section 9 (Output Feedback)**: 67 lines
- Audit said 78% reduction, High impact
- Missing: Detailed formalization, emergent tape theorem with proof, chain-of-thought equals explicit tape proof

**Section 10 (Multi-Pass RNN)**: 134 lines
- Audit said 67% reduction, Medium-High impact
- Missing: Detailed multi-pass formalization, tape modification analysis, bounded TM simulation proofs

**Section 11 (Experimental Results)**: 93 lines
- Audit said 77% reduction, Medium impact
- Restoration verified CMA-ES table completeness

**Section 12 (Formal System)**: 72 lines
- Audit said 79% reduction, High impact
- Missing: Complete Lean 4 overview, file-by-file descriptions, proof verification details, dependency graph, Lake build instructions

**Section 13 (Theory-Practice Gap)**: 6 lines
- Audit said 98% reduction, Very High impact
- Essentially removed entirely

**Section 14 (Composition Depth)**: 64 lines
- Audit said 86% reduction, High impact
- Missing: Detailed composition depth analysis, depth multiplication theorem, examples with calculations

**Section 15 (Uncanny Valley)**: 6 lines
- Audit said 99% reduction, High impact
- Essentially removed entirely

**Section 16 (Depth Examples)**: 56 lines
- Audit said 81% reduction, Medium-High impact
- Missing: Worked examples with calculations, step-by-step depth analysis

## Critical Findings

### 1. The "Restoration" Was Partial

Commit 0b952ab successfully restored:
- ✅ **Section 4 fixed point theory** (most critical mathematical infrastructure)
- ✅ **Section 8 circuit complexity proofs** (most critical complexity theory)
- ✅ Bibliography

But did NOT restore:
- ❌ Sections 1, 3 remain heavily compressed
- ❌ Section 2 missing detailed proofs
- ❌ Sections 9, 10 missing proof depth
- ❌ Sections 13, 15 essentially gutted
- ❌ Most supporting lemmas and corollaries throughout

### 2. Document Is Now a Hybrid

**What it has**:
- Main theorem statements (all preserved)
- Key impossibility results with proofs (XOR, threshold, parity)
- Fixed point theory (now restored in Section 4)
- Circuit complexity hierarchy (now restored in Section 8)
- Lean file references throughout

**What it lacks**:
- Detailed mathematical exposition and motivation
- Step-by-step proof development
- Supporting lemmas broken out separately
- Worked examples with calculations
- Conceptual/philosophical sections
- Practical guidance sections in depth

### 3. The Document Is NOT "A Rigorous Math Paper"

**Current status**: This is a **technical summary with selected proofs**, not a complete mathematical exposition.

A rigorous math paper would have:
- All lemmas stated and proven separately
- Theorems built up from lemmas
- Detailed proof steps, not just sketches
- Worked examples
- Motivation for each section

Current document has:
- Main theorems stated clearly
- Some proofs (especially Sections 4 and 8 after restoration)
- Many proof sketches rather than full proofs
- Minimal worked examples
- Compressed motivation

### 4. Specific Mathematical Gaps

**Still missing proofs/lemmas**:

From Section 2:
- Output Linearity lemma (additivity + homogeneity)
- Full inductive proof of state as weighted sum
- Four-corner XOR evaluation (only condensed version present)

From Section 3:
- Composition Depth Bound theorem (formal statement)
- ExactCounting.lean theorem with proof sketch
- RunningParity.lean with full affine proof
- "Depth doesn't help" formal proof

From Section 4 (even after restoration):
- Definition (Running Threshold Count) - missing
- Detailed capacity calculations (E88 vs Mamba2 comparison table with calculations)
- Multi-head formalization details

From Section 8 (even after restoration):
- Hard attention in AC⁰ theorem
- Iterated modular arithmetic construction
- Detailed examples of what E88 computes beyond TC⁰

From Section 9:
- Emergent tape memory theorem WITH proof
- Chain-of-thought equals explicit tape proof
- Computational effects formalization

### 5. Formatting Issues

**Lost infrastructure** (still not restored):
- Detailed theorem blocks with visual hierarchy (old format had colored boxes)
- Proof blocks with structured steps
- Lean code blocks inline (replaced with file references)
- Definition boxes with all parameters explained

**Current format**:
- Uses traditional-math-style.typ import
- Has theorem/lemma/definition/proof macros
- But much simpler than original colored boxes
- Lean code is referenced, not shown

## Verdict

### Question 1: Was everything actually restored?

**NO.** The restoration was partial:
- Section 4 fixed point theory: ✅ Mostly restored
- Section 8 circuit complexity: ✅ Substantially restored
- Everything else: ❌ Still heavily compressed

### Question 2: Do sections have proper math content, not just prose summaries?

**MIXED**:
- Sections 2, 4, 8: ✅ Have real math content with proofs
- Sections 3, 6: ⚠️ Have theorem statements but missing proof depth
- Sections 9, 10, 14, 16: ⚠️ Have some math but missing detailed proofs
- Sections 13, 15: ❌ Essentially prose summaries only

### Question 3: Do theorems have actual proofs shown, not just 'proof sketch'?

**MIXED**:
- Full proofs: XOR impossibility (Section 2), Zero is fixed point (Section 4), Linear state decay (Section 4), E88 exceeds TC⁰ (Section 8), PARITY in TC⁰ (Section 8)
- Proof sketches: Many theorems still only have sketches, especially in Sections 3, 6, 9, 10
- No proof at all: Some theorems just stated with Lean references

### Question 4: Remaining formatting issues?

**YES**:
- Mathematical exposition is compressed throughout (except Sections 4, 8)
- Supporting lemmas not broken out separately
- Worked examples mostly removed
- Conceptual motivation sections (13, 15) gutted
- Practical guidance compressed

### Question 5: Is this a rigorous math paper?

**NO**. It is a **technical summary with selected detailed proofs**.

A reader can:
- ✅ Understand the main results
- ✅ See proofs of key theorems (Sections 4, 8)
- ✅ Follow the logical flow
- ✅ Find the Lean files for complete proofs

A reader cannot:
- ❌ See all supporting lemmas proven separately
- ❌ Follow every proof step without consulting Lean
- ❌ Work through detailed examples
- ❌ Understand the full motivation and context

## Recommendations

### Priority 1: Restore Missing Proofs (High Mathematical Impact)

**Section 2 (Foundations)**:
1. Restore full inductive proof of "State as Weighted Sum" (not just sketch)
2. Add Output Linearity lemma with additivity and homogeneity proven separately
3. Restore four-corner XOR evaluation (currently only condensed)

**Section 3 (Linear Limitation)**:
1. Restore Composition Depth Bound theorem with formal statement and proof
2. Restore running threshold impossibility proof (continuity argument)
3. Restore running parity impossibility proof (full affine constraint violation)
4. Restore "Depth Compensation Fallacy" formal argument

**Section 9 (Output Feedback)**:
1. Restore Emergent Tape Memory theorem with proof
2. Restore Chain-of-thought equals explicit tape proof

### Priority 2: Restore Mathematical Infrastructure

**Section 4 (E88)** - even though much was restored:
1. Restore matrix vs vector capacity calculation table
2. Restore "Why Matrix State Matters" detailed comparison
3. Add Definition (Running Threshold Count) - was in audit as missing

**Section 8 (TC⁰)** - even though much was restored:
1. Add Hard attention in AC⁰ theorem (Hahn 2020)
2. Add iterated modular arithmetic construction example

### Priority 3: Restore Supporting Context

**Sections 13, 15** (Theory-Practice Gap, Uncanny Valley):
- These are essentially removed (6 lines each from 385 and 521)
- Decision: Either restore them properly or remove section markers entirely
- Having section markers with no content is worse than removing them

**Section 1 (Introduction)**:
- Restore central claims summary
- Restore "two kinds of composition" insight
- Restore reading guide

### Priority 4: Worked Examples

**Section 16 (Depth Examples)**:
- Restore worked examples with explicit calculations
- Show step-by-step depth analysis for concrete sequences

**Throughout**:
- Add worked examples for key theorems
- Show calculations for capacity comparisons
- Include numerical examples for impossibility results

## What's Actually Wrong

The most critical issue is: **The document claims to be a formal analysis but doesn't show enough mathematics to stand alone.**

The structure is:
1. State theorem
2. Brief proof sketch (sometimes)
3. Reference Lean file

This is fine for a **technical summary** or **research announcement**.

But the task description says: "The document should be a rigorous math paper, not a summary."

By that standard, the document **fails** because:
- Most proofs are sketches, not complete
- Supporting lemmas are not broken out
- Mathematical development is compressed
- Reader must consult Lean files to see complete arguments

## Conclusion

**The previous "fixes" did NOT actually fix things properly.**

What happened:
1. Original document (3dcc53b): 5335 lines, very detailed, all proofs
2. Restyling (04827f8): Reduced to 1077 lines (80% reduction)
3. Restoration (0b952ab): Added back ~100 lines to critical sections (4, 8)
4. Current: ~1200 lines, partial restoration

**Current state**:
- Critical mathematical infrastructure in Sections 4 and 8: ✅ Restored
- Everything else: ❌ Still heavily compressed
- Overall document: ⚠️ Not a complete rigorous math paper

**To make this a rigorous math paper**, need to:
1. Restore all proof steps (not just sketches)
2. Break out supporting lemmas
3. Add worked examples
4. Restore mathematical development in Sections 1-3, 9-10
5. Either restore or remove Sections 13, 15

**Estimated additional work**: ~2000-3000 lines to restore to "rigorous math paper" standard, or decision to accept current "technical summary with selected proofs" format.
