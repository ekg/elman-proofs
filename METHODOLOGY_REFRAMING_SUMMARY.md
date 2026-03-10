# Methodology Reframing Summary

## Changes Made

Reframed the introduction (and supporting elements) to emphasize this work as a **methodology contribution** rather than just a collection of expressivity results.

### Key Changes

#### 1. Introduction Structure (01-introduction.typ)

**Before:** Presented as "we found these results about different architectures"
**After:** Presented as "here's a method for exploring architecture space using formal proofs"

New structure:
- **§1.1 The Method**: Formal proofs as architecture design tool
- **§1.2 The Investigation**: Three architecture classes as examples
- **§1.3 The First Result**: Impossibility proofs for linear-temporal models
- **§1.4 The Design Implication**: What the proofs tell us about necessary features
- **§1.5 The Hierarchy**: Systematic classification from proof method
- **§1.6 The Empirical Puzzle**: Theory vs practice gap as methodological insight
- **§1.7 The Extension**: Output feedback as emergent memory
- **§1.8 Summary of Methodology**: 5-step design process

#### 2. Abstract (main.typ)

**Before:**
> We prove that placing nonlinearity in time versus between layers determines fundamental computational limits...

**After:**
> We demonstrate a methodology for architecture design: use formal impossibility proofs to explore the design space systematically...

Emphasizes the **process** rather than just the **findings**.

#### 3. Title and Subtitle (main.typ)

**Before:**
- Title: "Expressivity Analysis"
- Subtitle: "Where Should Nonlinearity Live?"

**After:**
- Title: "Proof-Guided Architecture Exploration"
- Subtitle: "Using Formal Proofs to Design Sequence Models"

Much clearer that this is about a design methodology.

## The 5-Step Methodology

The reframed introduction now explicitly presents the method:

1. **Formalize architecture classes precisely** (linear vs nonlinear temporal dynamics)
2. **Prove impossibility results** (what cannot be computed)
3. **Extract design constraints** (what architectural features are necessary)
4. **Construct witness functions** showing separations are sharp
5. **Verify all proofs mechanically** in Lean 4

## Why This Matters

### Before Reframing
- Readers might think: "This is just another expressivity paper showing model X is more powerful than model Y"
- Contribution appears incremental
- Method is implicit

### After Reframing
- Readers understand: "This demonstrates a systematic way to explore architecture design space"
- Contribution is the **methodology** of using formal proofs to guide architecture decisions
- Method is explicit and reusable

## Key Messaging

The reframing emphasizes:

1. **Proofs as design tools**: Impossibility results reveal necessary architectural features
2. **Systematic exploration**: Not empirical trial-and-error, but mathematical constraints
3. **Theory-practice dialogue**: The gap between what's computable and what's learnable is itself informative
4. **Reusable method**: The 5-step process can be applied to other architecture questions

## Preserved Content

All technical results remain intact:
- Linear SSM ⊊ TC⁰ ⊊ E88 ⊊ E23 hierarchy
- Impossibility of running parity for linear-temporal models
- Composition depth gap theorem
- Emergent tape memory results

The proofs haven't changed—only how we frame their purpose and contribution.

## Output Files

- `docs/expressivity/01-introduction.typ` - Reframed introduction
- `docs/expressivity/main.typ` - Updated abstract and title
- `docs/expressivity/main-final-methodology.pdf` - Compiled document with new framing

## Next Steps

Consider adding:
1. A "Related Work" section explicitly contrasting empirical vs proof-guided architecture design
2. A "Future Directions" section showing how the method applies to other architecture questions
3. More explicit discussion of how practitioners can use this methodology

The methodology is now the contribution, not just the specific findings about E88 vs Mamba2.
