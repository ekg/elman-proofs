# Expressivity Document Audit Summary

**Date**: 2026-02-01
**Auditor**: claude-code agent
**Task**: audit-and-fix - Real audit of document state

## TL;DR

**The previous "fixes" did NOT properly restore the document to rigorous math paper standards.**

- **Original** (commit 3dcc53b): 5,335 lines, comprehensive proofs
- **After restyling** (commit 04827f8): 1,077 lines (80% reduction)
- **After "restoration"** (commit 0b952ab): ~1,200 lines
- **Current status**: Technical summary with selected detailed proofs

## What Was Actually Restored

✅ **Section 4 (E88)**: Fixed point theory restored (lemmas, theorems, proofs)
✅ **Section 8 (TC⁰)**: Circuit complexity proofs restored (full TC⁰ definition, proof sketches)
✅ **Bibliography**: 669 lines of IEEE citations added

## What Is Still Missing

### High-Priority Mathematical Content

**Section 2 (Foundations)**:
- ❌ Full inductive proof of "State as Weighted Sum" (has sketch only)
- ❌ Output Linearity lemma broken out separately
- ❌ Four-corner XOR evaluation detailed (has condensed version)

**Section 3 (Linear Limitation)**:
- ❌ "Depth Compensation Fallacy" formal section
- ❌ Composition Depth Bound theorem with proof
- ❌ Running threshold continuity proof
- ❌ Running parity affine constraint violation proof
- ❌ Architecture-specific analysis (Mamba2, FLA, GDN equations)

**Section 4 (E88)** - even after restoration:
- ❌ "Why Matrix State Matters: Comparison to Mamba2" detailed table
- ❌ Information capacity ratio analysis (64× calculation)
- ❌ "How Tanh Preserves Matrix State Through Time" section
- ❌ Definition (Running Threshold Count)

**Section 9 (Output Feedback)**:
- ❌ Emergent Tape Memory theorem with proof
- ❌ Chain-of-thought equals explicit tape proof

### Sections Essentially Gutted

- **Section 13** (Theory-Practice Gap): 385 lines → 6 lines (98% reduction)
- **Section 15** (Uncanny Valley Reasoning): 521 lines → 6 lines (99% reduction)

These have section markers but no real content. Should either restore or remove.

## Current Document Assessment

### What It Is
- ✅ Clear statement of main theorems
- ✅ Lean file references throughout
- ✅ Selected detailed proofs (Sections 4, 8)
- ✅ Logical flow preserved
- ✅ Key impossibility results present

### What It Is Not
- ❌ Complete proofs for all theorems (many are sketches)
- ❌ Supporting lemmas broken out systematically
- ❌ Worked examples with calculations
- ❌ Comprehensive mathematical exposition
- ❌ Self-contained (reader must consult Lean files)

## Rigorous Math Paper Test

**Question**: Is this a rigorous math paper, not a summary?

**Answer**: **NO**. It is a **technical summary with selected detailed proofs**.

### Why It Fails the Test

1. **Proofs are often sketches**: Many theorems have proof sketches, not step-by-step proofs
2. **Lemmas not broken out**: Supporting results often inline or missing
3. **Examples missing**: Few worked examples with calculations
4. **Exposition compressed**: Motivation and context reduced to bare minimum
5. **Reader dependency**: Must consult Lean files to see complete arguments

### Current Line Counts by Section

```
Section 1 (Introduction):          33 lines (was 280, 87% reduction)
Section 2 (Foundations):          105 lines (was 354, 70% reduction)
Section 3 (Linear Limitation):     62 lines (was 140, 56% reduction)
Section 4 (E88):                  152 lines (was 591, 74% reduction) ⬆ partially restored
Section 5 (E23 vs E88):            59 lines (was 155, 62% reduction)
Section 6 (Separation):            64 lines (was 181, 65% reduction)
Section 7 (Implications):          73 lines (was 151, 52% reduction)
Section 8 (TC⁰):                  107 lines (was 351, 70% reduction) ⬆ substantially restored
Section 9 (Output Feedback):       65 lines (was 311, 79% reduction)
Section 10 (Multi-Pass RNN):      133 lines (was 410, 68% reduction)
Section 11 (Experimental):        212 lines (was 408, 48% reduction)
Section 12 (Formal System):        72 lines (was 342, 79% reduction)
Section 13 (Theory-Practice):       6 lines (was 385, 98% reduction) ⚠ gutted
Section 14 (Composition Depth):    64 lines (was 456, 86% reduction)
Section 15 (Uncanny Valley):        6 lines (was 521, 99% reduction) ⚠ gutted
Section 16 (Depth Examples):       56 lines (was 299, 81% reduction)
```

**Total**: ~1,200 lines (was 5,335, **77% reduction** overall)

## Specific Examples of Missing Content

### Example 1: Section 3 Missing "Depth Compensation Fallacy"

**Original had**:
- Formal section with detailed argument
- Table comparing required depth vs available depth
- Mathematical insight: "A^T is still just one linear operation"
- Architecture-specific examples

**Current has**:
- Brief mention that depth doesn't help
- No formal argument
- No table

### Example 2: Section 4 Missing Matrix State Exposition

**Original had**:
- Detailed comparison table (E88 vs Mamba2)
- Capacity calculations: 4,096 vs 64-256 values
- "Why Matrix State Matters" section
- "How Tanh Preserves Matrix State" section
- Multi-head structure formalization

**Current has**:
- Brief statement of matrix vs vector
- Capacity mentioned but not detailed
- Fixed point theory (restored ✅)
- No exposition sections

### Example 3: Proof Detail Comparison

**Original "State as Weighted Sum" proof**:
```
#proof[
  By induction on T. Base case: h_0 = 0 is the empty sum.
  Inductive step:
  h_{T+1} = A h_T + B x_T
          = A (sum_{t=0}^{T-1} A^{T-1-t} B x_t) + B x_T
          = sum_{t=0}^T A^{T-t} B x_t.
]

#lemma("Output Linearity")[
  The output y_T = C h_T is a linear function of the input sequence...
  This sum is additive: y(x + x') = y(x) + y(x'), and homogeneous...
]
```

**Current proof**:
```
#proof[
  Expand the recurrence: h_1 = B x_0, h_2 = A B x_0 + B x_1, and so on.
  The general term is a weighted sum of past inputs with weights given
  by powers of A.
]
```

Less detailed, no Output Linearity lemma broken out.

## Recommendations

### Option 1: Restore to Full Rigorous Paper (~2000-3000 more lines)

**Priority 1 (Core Math)**:
1. Section 2: Full proofs, Output Linearity lemma
2. Section 3: Depth Compensation Fallacy, formal proofs
3. Section 4: Matrix state exposition, capacity tables
4. Section 9: Emergent tape proofs

**Priority 2 (Context)**:
1. Sections 13, 15: Either restore fully or remove section markers
2. Section 1: Restore motivation and central claims
3. Section 16: Worked examples

**Estimated work**: 2-3 weeks of focused restoration

### Option 2: Accept Current Format as "Technical Summary"

**Actions**:
1. Change document title/abstract to reflect it's a summary
2. Remove empty section markers (13, 15)
3. Add note: "For complete proofs, see Lean formalizations"
4. Accept that it's not a complete math paper

**Estimated work**: 1-2 days of cleanup

### Option 3: Hybrid - Detailed Proofs in Appendix

**Actions**:
1. Keep current concise main text
2. Add appendix with full proofs
3. Reference appendix from main text
4. Sections 13, 15 to appendix or remove

**Estimated work**: 1-2 weeks

## Bottom Line

The document is in a **hybrid state**:
- Better than a pure summary (has some detailed proofs)
- Worse than a rigorous math paper (many gaps)
- Acceptable for researchers who trust Lean verification
- Insufficient for readers who want self-contained mathematics

**The claim that "previous fixes restored the content" is FALSE.**

Only Sections 4 and 8 were substantially restored. The rest remains heavily compressed.

## Decision Required

**What is the document's purpose?**

1. **Research announcement**: Current state is fine ✅
2. **Technical summary for practitioners**: Current state is fine ✅
3. **Rigorous mathematical reference**: Needs significant restoration ❌
4. **Self-contained exposition**: Needs significant restoration ❌

The task description said "rigorous math paper, not a summary" - by that standard, **the document currently fails**.
