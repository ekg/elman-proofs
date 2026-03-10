# Before/After Comparison: Methodology Reframing

## Introduction Opening

### BEFORE
> Every sequence model faces a choice: where should nonlinearity live? This architectural decision determines which functions a model can compute and why chain-of-thought reasoning works.

**Tone:** Question-driven, result-focused

### AFTER
> How should we design sequence models? The standard approach is empirical: propose an architecture, train it on benchmarks, report metrics. This paper demonstrates a different methodology: use formal proofs to explore the architecture space systematically.

**Tone:** Method-focused, contrasts with standard practice

---

## First Major Section

### BEFORE
Section title: (none - dives straight into architecture descriptions)
> Consider the three dominant approaches. Transformers...

**Structure:** Descriptive catalog of architectures

### AFTER
Section title: **"The Method: Formal Proofs as Architecture Design Tool"**
> The central principle is simple. Prove what a class of architectures _cannot_ compute. These impossibility results reveal what architectural features are _necessary_ for specific computational tasks.

**Structure:** Explicit methodology statement, then examples

---

## Results Presentation

### BEFORE
> Our central result makes this precise.
>
> [Theorem about composition depth]

**Framing:** "Here's what we found"

### AFTER
> The impossibility proofs reveal the architectural constraint:
>
> [Theorem about composition depth]
>
> This is not a benchmark result. It is a mathematical fact about compositional structure.

**Framing:** "Here's what the proof method reveals" + explicit contrast with empirical results

---

## Hierarchy Presentation

### BEFORE
> We begin with linear recurrent systems and prove what they cannot compute, then show how E88's tanh saturation escapes these limitations through stable fixed points. The complete hierarchy:
>
> Linear SSM ⊊ TC⁰ ⊊ E88 ⊊ E23

**Framing:** Sequential narrative of what we proved

### AFTER
> The proof-guided exploration establishes a strict hierarchy:
>
> Linear SSM ⊊ TC⁰ ⊊ E88 ⊊ E23
>
> Each separation is witnessed by a concrete computable function...

**Framing:** This hierarchy is a product of the systematic method

---

## Theory-Practice Gap

### BEFORE
> Despite E88's greater computational power, Mamba2 often achieves better perplexity on language modeling. This gap reveals the difference between what an architecture _can_ compute and what it _learns efficiently_.

**Tone:** This is a curious observation

### AFTER
> Here the methodology encounters a puzzle. Despite E88's provably greater computational capacity, Mamba2 achieves better perplexity...
>
> This gap is not a failure of the method. It reveals the difference between what an architecture _can_ compute with unlimited resources and what it _learns efficiently_ with gradient descent on finite data.

**Tone:** This is a methodological insight - understanding the limits of the approach is part of the method

---

## Closing

### BEFORE
> All theorems are mechanically verified in Lean 4...

**Message:** "Trust our results"

### AFTER
> This paper demonstrates a design method:
>
> 1. Formalize architecture classes precisely
> 2. Prove impossibility results
> 3. Extract design constraints
> 4. Construct witness functions
> 5. Verify all proofs mechanically in Lean 4
>
> The result is not just a collection of theorems, but a systematic approach to architecture exploration.

**Message:** "Use this method yourself"

---

## Abstract

### BEFORE
> We prove that placing nonlinearity in time versus between layers determines fundamental computational limits. Linear-temporal models (Mamba2, SSMs) cannot compute running parity at any depth...

**Focus:** What we proved (results)

### AFTER
> We demonstrate a methodology for architecture design: use formal impossibility proofs to explore the design space systematically. Formalizing linear-temporal recurrence reveals what such models cannot compute (running parity, threshold functions), which in turn reveals what architectural features are necessary...

**Focus:** How we did it (method)

---

## Title

### BEFORE
**"Expressivity Analysis: Temporal Nonlinearity vs Depth"**
Subtitle: "Where Should Nonlinearity Live?"

**Message:** This is an analysis comparing different approaches

### AFTER
**"Proof-Guided Architecture Exploration: Temporal Nonlinearity vs Depth"**
Subtitle: "Using Formal Proofs to Design Sequence Models"

**Message:** This is a method for systematic exploration

---

## Key Differences

| Aspect | Before | After |
|--------|--------|-------|
| **Contribution** | Specific results about E88 vs Mamba2 | A reusable methodology |
| **Novelty** | New theorems | New design approach |
| **Audience** | Theoreticians interested in results | Practitioners who want to apply the method |
| **Replicability** | Clone and verify proofs | Use 5-step process for other architectures |
| **Tone** | "We found X" | "Here's how to systematically find X" |

## Why It Matters

A paper framed as "we proved E88 > Mamba2 in expressivity" is incremental.

A paper framed as "here's a methodology for exploring architecture space using proofs" is foundational.

Same technical content, fundamentally different contribution.
