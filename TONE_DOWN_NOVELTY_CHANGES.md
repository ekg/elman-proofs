# Toning Down "Novel Architecture" Claims - Summary

## Objective
Remove overclaims that E88 is a "novel architecture". The novelty is in the **formal analysis and proofs**, not in the architecture itself, which is a variant of classical Elman (1990) networks.

## Changes Made

### Section 02-foundations.typ

#### Change 1: Section Heading (line 155)
**Before:**
```
=== E88: Our Novel Architecture
```

**After:**
```
=== E88: The Classical Elman Architecture with Matrix State
```

#### Change 2: Introduction Paragraph (lines 159)
**Before:**
```
We introduce *E88*, a nonlinear recurrent architecture that breaks this limitation.
```

**After:**
```
E88 is a variant of the classical Elman (1990) architecture with matrix-valued state
and tanh nonlinearity in the recurrence. While the architecture itself is not new,
this work provides the first formal analysis of its expressivity advantages over
linear-temporal models.
```

#### Change 3: Definition Box (line 171)
**Before:**
```
*Key innovation*: The recurrence S_t = tanh(α S_{t-1} + ...) is *nonlinear in S_{t-1}*
```

**After:**
```
*Key property*: The recurrence S_t = tanh(α S_{t-1} + ...) is *nonlinear in S_{t-1}*
```

#### Change 4: Contribution Statement (line 200)
**Before:**
```
E88 is the key contribution of this work. It demonstrates that nonlinearity in the
temporal recurrence---not just deep stacking of layers---is essential for expressing
functions that require maintaining discrete state over time.
```

**After:**
```
The key contribution of this work is the formal analysis demonstrating that nonlinearity
in the temporal recurrence---not just deep stacking of layers---is essential for expressing
functions that require maintaining discrete state over time. While the E88 architecture
builds on classical Elman networks, the theorems about its saturation dynamics and
expressivity separation from linear-temporal models are novel.
```

## What Remains Correctly Attributed

1. **Introduction (01-introduction.typ)** correctly states E88 "follows the classical Elman architecture"
2. **TC⁰ Bounds (08-tc0-bounds.typ)** correctly calls E88 the "old" architecture descended from Elman networks
3. **Abstract** focuses on the proofs and analysis, not claiming architectural novelty

## Key Message
The document now clearly states:
- **Not novel**: The E88 architecture (matrix state + tanh recurrence is from Elman 1990)
- **Novel**: The formal Lean 4 proofs about saturation dynamics, expressivity separation, and the hierarchy Linear SSM ⊊ TC⁰ ⊊ E88 ⊊ E23

## Verification
Document compiles successfully with `typst compile main.typ` (only deprecation warnings about `times.circle` symbol).
