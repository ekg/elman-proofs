// Traditional Math Style for Typst
// Based on LaTeX theorem environments (amsthm style)
//
// USAGE:
//   #import "traditional-math-style.typ": *
//   #show: setup-traditional-style
//
// AVAILABLE ENVIRONMENTS:
//   #theorem("Name")[body]      - Italic statement, bold "Theorem N (Name)."
//   #lemma("Name")[body]        - Italic statement, bold "Lemma N (Name)."
//   #corollary("Name")[body]    - Italic statement, bold "Corollary N (Name)."
//   #proposition("Name")[body]  - Italic statement, bold "Proposition N (Name)."
//   #definition("Name")[body]   - Upright statement, bold "Definition N (Name)."
//   #remark(none)[body]         - Italic "Remark.", upright body
//   #example(none)[body]        - Italic "Example.", upright body
//   #proof[body]                - Italic "Proof." with filled QED square
//   #proof-sketch[body]         - Italic "Proof sketch." with hollow QED square
//
// LEAN REFERENCES (as footnotes, not inline code):
//   #leanref("File.lean:42", "theorem name")  - Full footnote with signature
//   #leanfile("File.lean:42")                 - Short footnote with just location
//
// VISUAL ELEMENTS:
//   #lightrule, #mediumrule, #heavyrule  - Horizontal rules
//   #centerrule                          - Short centered decorative rule
//   #keyinsight[body]                    - Emphasized quote between rules
//   #codeblock("code", lang: "lean4")    - Minimal code display
//   #simpletable(...)                    - Table with light styling
//   #numeq($equation$)                   - Numbered equation
//
// DESIGN PRINCIPLES:
// - No colored boxes or rounded corners (traditional math style)
// - Italic theorem statements (as in LaTeX/amsthm)
// - 'Proof.' in italic with QED symbol
// - Simple horizontal rules for visual separation
// - Lean references as footnotes, not inline code blocks
// - Sans-serif font maintained for screen readability

// ============================================================================
// COUNTERS
// ============================================================================

#let theorem-counter = counter("theorem")

// Reset theorem counter at each section (heading level 1)
// Call this in show rules if needed

// ============================================================================
// THEOREM-LIKE ENVIRONMENTS
// ============================================================================

// Theorem: italic body, bold header
#let theorem(title, body) = {
  theorem-counter.step()
  block(
    above: 1.2em,
    below: 1.2em,
    width: 100%,
  )[
    #text(weight: "bold")[Theorem]
    #context text(weight: "bold")[#theorem-counter.display()]
    #if title != none [ (#text(weight: "bold")[#title])]#text(weight: "bold")[.]
    #h(0.3em)
    #emph[#body]
  ]
}

// Lemma: italic body, bold header
#let lemma(title, body) = {
  theorem-counter.step()
  block(
    above: 1.2em,
    below: 1.2em,
    width: 100%,
  )[
    #text(weight: "bold")[Lemma]
    #context text(weight: "bold")[#theorem-counter.display()]
    #if title != none [ (#text(weight: "bold")[#title])]#text(weight: "bold")[.]
    #h(0.3em)
    #emph[#body]
  ]
}

// Corollary: italic body, bold header
#let corollary(title, body) = {
  theorem-counter.step()
  block(
    above: 1.2em,
    below: 1.2em,
    width: 100%,
  )[
    #text(weight: "bold")[Corollary]
    #context text(weight: "bold")[#theorem-counter.display()]
    #if title != none [ (#text(weight: "bold")[#title])]#text(weight: "bold")[.]
    #h(0.3em)
    #emph[#body]
  ]
}

// Proposition: italic body, bold header
#let proposition(title, body) = {
  theorem-counter.step()
  block(
    above: 1.2em,
    below: 1.2em,
    width: 100%,
  )[
    #text(weight: "bold")[Proposition]
    #context text(weight: "bold")[#theorem-counter.display()]
    #if title != none [ (#text(weight: "bold")[#title])]#text(weight: "bold")[.]
    #h(0.3em)
    #emph[#body]
  ]
}

// Definition: upright body (definitions are not italicized in traditional style)
#let definition(title, body) = {
  theorem-counter.step()
  block(
    above: 1.2em,
    below: 1.2em,
    width: 100%,
  )[
    #text(weight: "bold")[Definition]
    #context text(weight: "bold")[#theorem-counter.display()]
    #if title != none [ (#text(weight: "bold")[#title])]#text(weight: "bold")[.]
    #h(0.3em)
    #body
  ]
}

// Remark: upright body
#let remark(title, body) = {
  block(
    above: 1em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Remark]
    #if title != none [ (#text(style: "italic")[#title])]#text(style: "italic")[.]
    #h(0.3em)
    #body
  ]
}

// Example: upright body
#let example(title, body) = {
  block(
    above: 1em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Example]
    #if title != none [ (#text(style: "italic")[#title])]#text(style: "italic")[.]
    #h(0.3em)
    #body
  ]
}

// Observation: upright body (similar to remark but for empirical observations)
#let observation(title, body) = {
  block(
    above: 1em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Observation]
    #if title != none [ (#text(style: "italic")[#title])]#text(style: "italic")[.]
    #h(0.3em)
    #body
  ]
}

// Finding: for key empirical or theoretical findings (between rules)
#let finding(body) = {
  block(
    above: 1em,
    below: 1em,
    inset: (left: 1em, right: 1em, top: 0.5em, bottom: 0.5em),
    width: 100%,
  )[
    #line(length: 100%, stroke: 0.3pt)
    #v(0.3em)
    #body
    #v(0.3em)
    #line(length: 100%, stroke: 0.3pt)
  ]
}

// Failure mode: for describing architectural or algorithmic failure modes
#let failure(title, body) = {
  block(
    above: 1em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Failure mode]
    #if title != none [ (#text(style: "italic")[#title])]#text(style: "italic")[.]
    #h(0.3em)
    #body
  ]
}

// ============================================================================
// PROOF ENVIRONMENT
// ============================================================================

// Proof with italic 'Proof.' and QED symbol (filled square)
#let proof(body) = {
  block(
    above: 0.8em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Proof.]
    #h(0.3em)
    #body
    #h(1fr)
    $square.filled$
  ]
}

// Proof sketch variant
#let proof-sketch(body) = {
  block(
    above: 0.8em,
    below: 1em,
    width: 100%,
  )[
    #text(style: "italic")[Proof sketch.]
    #h(0.3em)
    #body
    #h(1fr)
    $square.stroked$
  ]
}

// ============================================================================
// LEAN REFERENCE (AS FOOTNOTE)
// ============================================================================

// Create a footnote reference to Lean formalization
// Usage: #leanref("LinearCapacity.lean:72", "theorem linear_state_is_sum")
#let leanref(location, signature) = {
  footnote[Lean formalization: #raw(location, lang: none). See #raw(signature, lang: "lean4").]
}

// Short form for just file reference
#let leanfile(location) = {
  footnote[Lean formalization: #raw(location, lang: none).]
}

// ============================================================================
// HORIZONTAL RULES FOR SECTION BREAKS
// ============================================================================

// Light rule for minor breaks
#let lightrule = line(length: 100%, stroke: 0.3pt + rgb("#888888"))

// Medium rule for subsection breaks
#let mediumrule = line(length: 100%, stroke: 0.5pt + rgb("#444444"))

// Heavy rule for major section breaks
#let heavyrule = line(length: 100%, stroke: 1pt + rgb("#000000"))

// Centered short rule (decorative)
#let centerrule = align(center)[#line(length: 30%, stroke: 0.5pt)]

// ============================================================================
// DISPLAY MATH HELPERS
// ============================================================================

// Numbered equation
#let equation-counter = counter("equation")

#let numeq(body) = {
  equation-counter.step()
  grid(
    columns: (1fr, auto),
    align: (center, right),
    body,
    context [(#equation-counter.display())],
  )
}

// ============================================================================
// QUOTE ENVIRONMENT (for key insights)
// ============================================================================

#let keyinsight(body) = {
  block(
    above: 1em,
    below: 1em,
    inset: (left: 2em, right: 2em, top: 0.5em, bottom: 0.5em),
    width: 100%,
  )[
    #line(length: 100%, stroke: 0.5pt)
    #v(0.3em)
    #emph[#body]
    #v(0.3em)
    #line(length: 100%, stroke: 0.5pt)
  ]
}

// ============================================================================
// CODE DISPLAY (for rare cases where code must be shown)
// ============================================================================

// Simple code block without heavy styling
#let codeblock(code, lang: "lean4") = {
  block(
    above: 0.8em,
    below: 0.8em,
    inset: (left: 1em, top: 0.5em, bottom: 0.5em),
    width: 100%,
    stroke: (left: 1pt + rgb("#cccccc")),
  )[
    #raw(code, lang: lang, block: true)
  ]
}

// ============================================================================
// FIGURE/TABLE STYLING
// ============================================================================

// Simple table style without heavy borders
#let simpletable(..args) = {
  table(
    stroke: 0.5pt,
    align: left,
    ..args
  )
}

// ============================================================================
// DOCUMENT SETUP FUNCTION
// ============================================================================

// Call this at the start of document or in main.typ
#let setup-traditional-style(doc) = {
  // Reset theorem counter at each level-1 heading
  show heading.where(level: 1): it => {
    theorem-counter.update(0)
    equation-counter.update(0)
    it
  }
  doc
}
