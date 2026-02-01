# Typst Academic Paper Formatting: Research Findings

## Executive Summary

This document summarizes best practices for formatting academic papers in Typst, with a focus on mathematical papers. Key findings:

1. **Theorem environments**: Use `ctheorems` or `great-theorems` packages for numbered theorem/definition/remark blocks
2. **Highlighting key results**: Use `note-me` or `gentle-clues` packages for admonitions/callouts (NOT block quotes)
3. **Block quotes**: Reserved for actual quotations using `#quote(block: true)[]`
4. **Emphasis**: Use `_italic_` for emphasis, `*bold*` for strong emphasis, `` `code` `` for inline code
5. **Professional templates**: `clean-math-paper` provides a solid foundation for mathematical papers

---

## 1. Theorem, Definition, and Remark Environments

### Recommended Package: `ctheorems`

The most widely used package for mathematical environments is **ctheorems** (version 1.1.3+).

#### Basic Setup

```typst
#import "@preview/ctheorems:1.1.3": *
#show: thmrules

// Create theorem-like environments
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#eeffee"))
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#efe6ff"))
#let definition = thmbox("definition", "Definition", fill: rgb("#e8f4f8"))

// Create plain environments (no box)
#let corollary = thmplain("corollary", "Corollary", base: "theorem")
#let remark = thmplain("remark", "Remark")
#let example = thmplain("example", "Example")

// Create proof environment
#let proof = thmproof("proof", "Proof")
```

#### Usage Examples

```typst
#theorem("Fundamental Theorem")[
  Linear RNNs cannot compute XOR.
] <thm:linear-xor>

#proof[
  The proof follows from linearity...
  #qedhere  // Places QED symbol at end of proof
]

#definition[
  A *linear RNN* is one where the state update is linear in the hidden state.
]

#remark[
  This result extends to any threshold function.
] <rem:threshold>

// Reference theorems
As shown in @thm:linear-xor, linear models are limited...
```

#### Key Features

- **Automatic numbering**: Theorems are numbered automatically
- **Shared counters**: Use `base:` parameter to share numbering (e.g., corollary inherits theorem counter)
- **Styling options**:
  - `fill:` - background color for boxed environments
  - `inset:` - internal padding
  - `titlefmt:` - title formatting (e.g., `strong`)
  - `numbering: none` - for unnumbered environments
- **Cross-referencing**: Use labels `<thm:name>` and references `@thm:name`
- **QED symbols**: `#qedhere` in proofs places end-of-proof marker

#### Alternative: `great-theorems`

The `great-theorems` package offers similar functionality with a different API:

```typst
#import "@preview/great-theorems:0.1.0": *

#let theorem = mathblock(
  blocktitle: "Theorem",
  counter: "theorem"
)

#theorem[Content here]
```

**Template Recommendation**: The `clean-math-paper` template uses `great-theorems` by default.

---

## 2. Highlighting Key Results (NOT Quotations)

### Critical Distinction

**DO NOT use block quotes for highlighting your own key results.** Block quotes are for quoting external sources. For highlighting important findings, use admonition/callout boxes.

### Recommended Package: `note-me`

For GitHub-style admonitions (best for technical papers):

```typst
#import "@preview/note-me:0.6.0": *

// Predefined admonition types
#note[
  Linear state is computed as a weighted sum of inputs:
  $ h_t = sum_(i=1)^t alpha^(t-i) x_i $
]

#important[
  This proves that linear RNNs cannot compute XOR, regardless of depth.
]

#tip[
  For practical implications, consider using nonlinear activations.
]

#warning[
  This limitation applies to all linear-in-h architectures, including Mamba2.
]

#caution[
  The proof requires careful handling of the linearity assumption.
]
```

#### Customization

```typst
// Custom admonition with specific styling
#admonition(
  icon: "⚡",
  color: rgb("#ff6b6b"),
  title: "Key Result",
  background-color: rgb("#fff0f0")
)[
  Tanh saturation creates stable latching behavior in E88.
]
```

#### Preventing Page Breaks

Wrap admonitions in `#box()` to keep them together:

```typst
#box(width: 1fr)[
  #important[
    This result must not be split across pages.
  ]
]
```

### Alternative: `gentle-clues`

For a softer, more colorful style (inspired by mdbook-admonish):

```typst
#import "@preview/gentle-clues:1.3.0": *

#info[This is informational content...]
#tip(title: "Best Practice")[Use this approach for better results]
```

**Features**:
- Automatic language detection from `context text.lang`
- Customizable colors and titles per clue
- Custom clue type definitions
- Task counters (for TODO-style clues)

---

## 3. Block Quotes (For Actual Quotations)

Use the built-in `quote` function **only** for quoting external sources:

```typst
// Block quote with attribution
#quote(block: true, attribution: [Einstein, 1905])[
  The most incomprehensible thing about the world is that it is comprehensible.
]

// Block quote without quotes
#quote(block: true, quotes: false)[
  This is a long quotation from another paper...
]

// Inline quote (default)
#quote[Short quoted phrase]
```

### Customization

```typst
// Center-aligned quotes with more padding
#set quote(block: true)
#show quote: set align(center)
#show quote: set pad(x: 5em)

#quote(attribution: [@einstein1905])[
  Quoted text here...
]
```

### Key Parameters

- `block: true` - Display as block quote (default: false for inline)
- `quotes: auto|true|false` - Control quotation marks (auto: only for inline)
- `attribution: content|label` - Source attribution (can reference bibliography)

**Default styling**: Block quotes have 1em left/right padding.

---

## 4. Text Emphasis and Inline Formatting

### Standard Conventions

```typst
// Emphasis (italic)
This is _emphasized text_.

// Strong emphasis (bold)
This is *very important*.

// Inline code/monospace
The function `compute_state()` returns the hidden state.

// Combining
This is *_bold and italic_*.
```

### Function-based Formatting

For more control or partial-word styling:

```typst
// When markdown symbols don't work
#emph[emphasized content]
#strong[bold content]

// Highlighting
#highlight[highlighted background]
#highlight(fill: yellow)[custom color highlighting]
```

### Best Practices for Math Papers

1. **Definitions**: Use italics for newly defined terms at first occurrence
2. **Variables**: Math mode automatically italicizes variables
3. **Code/functions**: Use backticks for function names and code
4. **Emphasis**: Use sparingly for genuinely important points
5. **Avoid**: Don't use bold for theorem statements (use the theorem environment instead)

---

## 5. Professional Math Paper Templates

### `clean-math-paper`

A comprehensive template for mathematical papers with built-in theorem support.

#### Installation

```bash
typst init @preview/clean-math-paper:0.2.5
```

#### Features

- Pre-configured theorem/definition/lemma/proof environments
- Equation numbering (section.equation format)
- Only labeled equations are numbered
- Cross-reference support
- Customizable heading colors
- Multi-language support

#### Example Structure

```typst
#import "@preview/clean-math-paper:0.2.5": paper, theorem, proof, definition

#show: paper.with(
  title: "Linear Limitations of RNNs",
  authors: (
    (name: "Your Name", affiliation: "Institution", email: "email@example.com"),
  ),
  abstract: [
    We prove that linear RNNs cannot compute XOR...
  ],
)

= Introduction

Linear recurrent neural networks...

#definition[
  A *linear RNN* satisfies $h_t = W h_(t-1) + U x_t$.
]

#theorem(title: "XOR Impossibility")[
  No linear RNN can compute XOR on arbitrary sequences.
]<thm:main>

#proof[
  By linearity of the state update...
]

= Conclusions

As proven in @thm:main, linear architectures are fundamentally limited.
```

### Other Notable Templates

1. **`arkheion`** - ArXiv-style preprint template
2. **`smooth-tmlr`** - TMLR journal format
3. **`clean-iclr`** - ICLR conference with arXiv preprint option
4. **`bloated-neurips`** - NeurIPS 2023/2024/2025 with arXiv preprint

All available at: https://typst.app/universe/

---

## 6. Equations and Mathematical Notation

### Numbering and Labeling

```typst
// Only labeled equations are numbered
$
  h_t = sum_(k=1)^t alpha^(t-k) x_k
$<eq:linear-state>

// Reference equations
From @eq:linear-state, we see that the state is a weighted sum.

// Inline math
The hidden state $h_t in RR^d$ evolves according to...
```

### Multi-line Equations

```typst
$
  h_t &= W h_(t-1) + U x_t \
      &= W(W h_(t-2) + U x_(t-1)) + U x_t \
      &= W^2 h_(t-2) + W U x_(t-1) + U x_t
$<eq:expansion>
```

---

## 7. Best Practices Summary

### For Mathematical Papers

1. **Structure**:
   - Use a professional template (`clean-math-paper` or similar)
   - Organize with numbered sections (`= Introduction`, `= Methods`, etc.)
   - Include abstract and bibliography

2. **Theorems and Definitions**:
   - Use `ctheorems` or `great-theorems` package
   - Choose boxed (`thmbox`) for important results, plain (`thmplain`) for remarks
   - Always label important theorems for cross-referencing
   - Use consistent naming scheme (e.g., `<thm:name>`, `<def:name>`)

3. **Highlighting Key Results**:
   - **For your own findings**: Use `#important[]` or `#note[]` admonitions
   - **For external quotes**: Use `#quote(block: true)[]`
   - **Never** use block quotes to highlight your own results

4. **Emphasis**:
   - `_italic_` for emphasis and defined terms
   - `*bold*` sparingly for critical points
   - `` `code` `` for function names and code snippets
   - Math mode for all mathematical symbols and variables

5. **Equations**:
   - Label important equations for cross-referencing
   - Use alignment (`&=`) for multi-line derivations
   - Remember: only labeled equations get numbers

6. **Cross-references**:
   - Theorems: `@thm:name`
   - Equations: `@eq:name` (prefix with "eq:")
   - Sections: `@sec:name`
   - Bibliography: `@citation-key`

---

## 8. Example: Formatting a Key Result

### ❌ WRONG: Using block quote

```typst
#quote(block: true)[
  *Key Result*: Linear RNNs cannot compute XOR.
]
```

**Problem**: This looks like you're quoting yourself, which is confusing.

### ✅ CORRECT: Using admonition

```typst
#important[
  *Key Result*: Linear RNNs cannot compute XOR.
]
```

### ✅ BEST: Using theorem environment

```typst
#theorem(title: "Linear XOR Impossibility")[
  No linear RNN architecture can compute XOR on sequences of arbitrary length.
]<thm:xor-impossibility>

#proof[
  Let $h_t$ denote the hidden state after processing inputs $x_1, ..., x_t$.
  By linearity, we can write:
  $
    h_t = sum_(i=1)^t alpha_i x_i
  $<eq:linear-expansion>

  For XOR computation, we need the output to depend on parity, which is not
  a linear function of the state. Contradiction. #qedhere
]

#remark[
  This result extends to all linear-in-$h$ architectures, including Mamba2.
]
```

---

## 9. References and Sources

### Typst Documentation
- [Tutorial – Typst Documentation](https://typst.app/docs/tutorial/)
- [Writing in Typst – Typst Documentation](https://typst.app/docs/tutorial/writing-in-typst/)
- [Advanced Styling – Typst Documentation](https://typst.app/docs/tutorial/advanced-styling/)
- [Quote Function – Typst Documentation](https://typst.app/docs/reference/model/quote/)
- [Emphasis Function – Typst Documentation](https://typst.app/docs/reference/model/emph/)
- [Math – Typst Documentation](https://typst.app/docs/reference/math/)

### Packages
- [ctheorems – Typst Universe](https://typst.app/universe/package/ctheorems/)
- [great-theorems – Typst Universe](https://typst.app/universe/package/great-theorems/)
- [note-me – Typst Universe](https://typst.app/universe/package/note-me/)
- [gentle-clues – Typst Universe](https://typst.app/universe/package/gentle-clues/)
- [theorion – Typst Universe](https://typst.app/universe/package/theorion/)
- [lemmify – Typst Universe](https://typst.app/universe/package/lemmify/)

### Templates
- [clean-math-paper – Typst Universe](https://typst.app/universe/package/clean-math-paper/)
- [GitHub - JoshuaLampert/clean-math-paper](https://github.com/JoshuaLampert/clean-math-paper)
- [arkheion – Typst Universe](https://typst.app/universe/package/arkheion)
- [starter-journal-article – Typst Universe](https://typst.app/universe/package/starter-journal-article/)
- [charged-ieee – Typst Universe](https://typst.app/universe/package/charged-ieee/)

### Template Collections
- [GitHub - daskol/typst-templates](https://github.com/daskol/typst-templates)
- [Typst Universe Search](https://typst.app/universe/search/)

### Examples and Tutorials
- [Markup language - Typst Examples Book](https://sitandr.github.io/typst-examples-book/book/basics/tutorial/markup.html)
- [02: Text Formatting - Typst](https://www.codingetc.com/Typst_tutorial/02_text_formatting.html)
- [Asciidoctor-like Admonitions for Typst - Typst Forum](https://forum.typst.app/t/asciidoctor-like-admonitions-for-typst/2564)

### GitHub Discussions
- [Block quote syntax · typst/typst · Discussion #2361](https://github.com/typst/typst/discussions/2361)
- [How to implement the same functionality as \\newtheorem in LaTeX · Issue #337](https://github.com/typst/typst/issues/337)

---

## 10. Implementation Recommendations for Expressivity Document

Based on this research, here are specific recommendations for the expressivity document:

### 1. Use `ctheorems` for Formal Results

```typst
#import "@preview/ctheorems:1.1.3": *
#show: thmrules

#let theorem = thmbox("theorem", "Theorem", fill: rgb("#e8f4f8"))
#let lemma = thmbox("lemma", "Lemma", fill: rgb("#e8f4f8"))
#let definition = thmbox("definition", "Definition", fill: rgb("#fff4e6"))
#let corollary = thmplain("corollary", "Corollary", base: "theorem")
#let remark = thmplain("remark", "Remark")
#let proof = thmproof("proof", "Proof")
```

### 2. Use `note-me` for Highlighting Key Insights

```typst
#import "@preview/note-me:0.6.0": *

// For major findings
#important[
  E88's tanh saturation enables binary fact latching that Mamba2's linear
  state cannot achieve.
]

// For technical details
#note[
  The proof relies on showing that linear combinations cannot represent
  threshold functions.
]

// For implications
#tip[
  This suggests that nonlinearity in the recurrence is essential for
  certain temporal computations.
]
```

### 3. Structure for Mathematical Content

```typst
= Linear Limitations

#definition[
  An RNN is *linear-in-h* if its state update is linear in the hidden state:
  $ h_t = f(h_(t-1), x_t) "where" f "is linear in its first argument" $
]

#theorem(title: "XOR Impossibility")[
  No linear-in-h RNN can compute XOR on arbitrary-length sequences.
]<thm:xor>

#proof[
  By linearity, $h_t = sum_(i=1)^t alpha_i x_i$ for some coefficients.
  The XOR function is not computable from such a weighted sum. #qedhere
]

#remark[
  This applies to both classical linear RNNs and modern architectures like Mamba2.
]

#important[
  *Key Insight*: The limitation is architectural, not a matter of training.
  Linear-in-h architectures are fundamentally incapable of threshold computation.
]
```

### 4. Avoid Block Quotes for Own Results

**Don't do this**:
```typst
#quote(block: true)[
  Linear RNNs cannot compute XOR.
]
```

**Instead, use theorem environment or admonition**:
```typst
#theorem[Linear RNNs cannot compute XOR.]
// OR
#important[Linear RNNs cannot compute XOR.]
```

---

## Conclusion

Typst provides excellent support for academic mathematical writing through:
1. **Dedicated theorem packages** (`ctheorems`, `great-theorems`) for formal mathematical environments
2. **Admonition packages** (`note-me`, `gentle-clues`) for highlighting key findings
3. **Built-in quote function** for actual quotations
4. **Professional templates** (`clean-math-paper`, `arkheion`) for consistent formatting
5. **Simple markup** for emphasis and inline formatting

The key principle: **use the right tool for the right purpose** - theorems for formal results, admonitions for highlighting insights, quotes for external sources, and emphasis for textual importance.
