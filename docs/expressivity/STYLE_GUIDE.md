# Expressivity Document Style Guide

## Narrative Philosophy

This document should read like a PAPER, not a fact sheet. Every section should:

1. **Build on what came before** - Reference previous results, show how we're progressing
2. **Tell a story** - We're answering "where should nonlinearity live?" and discovering the answer
3. **Feel like discovery** - Theorems are revelations in the narrative, not bullet points
4. **Be lyrical** - Good mathematical writing has rhythm and elegance

## Anti-patterns to Avoid

- "Here are 5 facts about X" → Instead, weave facts into narrative
- Repeated theorem statements across sections → State once, reference later
- Lean code blocks in main text → Footnotes only
- Colored boxes everywhere → Simple, traditional formatting
- Disconnected sections → Each section should say "now we turn to..."

## Narrative Arc

1. **Introduction**: The fundamental question. Three possible answers. Why it matters.
2. **Foundations**: The mathematical tools we need (building blocks)
3. **The Limitation**: What linear-temporal models CANNOT do (rising tension)
4. **The Solution**: How E88's matrix state solves this (the turn)
5. **The Hierarchy**: The complete picture emerges (resolution)
6. **Implications**: What this means for practice (denouement)

## Typst Style

- Sans-serif font (keep current)
- No rounded colored boxes - use simple rules or indentation
- Italic theorem statements
- Proofs: "_Proof._ [body] □"
- Lean references as footnotes: `#footnote[Formalized in LinearLimitations.lean:315]`
- Inline math in prose: "the state $S_t$ evolves as..."
- Display math for key equations only

## When Restyling Each Section

Ask yourself:
1. How does this section connect to the previous one?
2. What is the ONE key insight this section delivers?
3. Am I dumping facts or telling a story?
4. Would a mathematician enjoy reading this prose?
