// Traditional Math Style for Typst
// Based on LaTeX theorem environments (amsthm style)
// Using the ctheorems package for proper numbering and inline behavior
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
// IMPORT CTHEOREMS PACKAGE
// ============================================================================

#import "@preview/ctheorems:1.1.3": *

// ============================================================================
// THEOREM-LIKE ENVIRONMENTS (using ctheorems)
// ============================================================================

// Base theorem environment with traditional styling
// - No fill/stroke (plain traditional style)
// - Italic body for theorems
// - Bold heading
#let theorem-base = thmbox(
  "theorem",
  "Theorem",
  fill: none,
  stroke: none,
  inset: 0pt,
  radius: 0pt,
  padding: (top: 1.2em, bottom: 1.2em),
  namefmt: name => text(weight: "bold")[#name],
  titlefmt: title => text(weight: "bold")[(#title)],
  bodyfmt: body => emph[#body]
)

// Lemma: shares counter with theorem
#let lemma-base = thmbox(
  "lemma",
  "Lemma",
  base: "theorem",
  fill: none,
  stroke: none,
  inset: 0pt,
  radius: 0pt,
  padding: (top: 1.2em, bottom: 1.2em),
  namefmt: name => text(weight: "bold")[#name],
  titlefmt: title => text(weight: "bold")[(#title)],
  bodyfmt: body => emph[#body]
)

// Corollary: shares counter with theorem
#let corollary-base = thmbox(
  "corollary",
  "Corollary",
  base: "theorem",
  fill: none,
  stroke: none,
  inset: 0pt,
  radius: 0pt,
  padding: (top: 1.2em, bottom: 1.2em),
  namefmt: name => text(weight: "bold")[#name],
  titlefmt: title => text(weight: "bold")[(#title)],
  bodyfmt: body => emph[#body]
)

// Proposition: shares counter with theorem
#let proposition-base = thmbox(
  "proposition",
  "Proposition",
  base: "theorem",
  fill: none,
  stroke: none,
  inset: 0pt,
  radius: 0pt,
  padding: (top: 1.2em, bottom: 1.2em),
  namefmt: name => text(weight: "bold")[#name],
  titlefmt: title => text(weight: "bold")[(#title)],
  bodyfmt: body => emph[#body]
)

// Definition: upright body (not italicized in traditional style)
#let definition-base = thmbox(
  "definition",
  "Definition",
  base: "theorem",
  fill: none,
  stroke: none,
  inset: 0pt,
  radius: 0pt,
  padding: (top: 1.2em, bottom: 1.2em),
  namefmt: name => text(weight: "bold")[#name],
  titlefmt: title => text(weight: "bold")[(#title)],
  bodyfmt: body => body  // upright, not italic
)

// Wrapper functions to match our API: theorem(title)[body]
#let theorem(title, body) = {
  if title == none {
    theorem-base[#body]
  } else {
    theorem-base(title: title)[#body]
  }
}

#let lemma(title, body) = {
  if title == none {
    lemma-base[#body]
  } else {
    lemma-base(title: title)[#body]
  }
}

#let corollary(title, body) = {
  if title == none {
    corollary-base[#body]
  } else {
    corollary-base(title: title)[#body]
  }
}

#let proposition(title, body) = {
  if title == none {
    proposition-base[#body]
  } else {
    proposition-base(title: title)[#body]
  }
}

#let definition(title, body) = {
  if title == none {
    definition-base[#body]
  } else {
    definition-base(title: title)[#body]
  }
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
// PROOF ENVIRONMENT (using ctheorems)
// ============================================================================

// Proof with italic 'Proof.' and QED symbol (filled square)
// Using thmproof from ctheorems ensures proper inline behavior
#let proof = thmproof(
  "proof",
  "Proof",
  base: none,
  titlefmt: title => text(style: "italic")[#title.],
  bodyfmt: body => [#h(0.3em)#body#h(1fr)$square.filled$]
)

// Proof sketch variant with hollow square
#let proof-sketch = thmproof(
  "proof-sketch",
  "Proof sketch",
  base: none,
  titlefmt: title => text(style: "italic")[#title.],
  bodyfmt: body => [#h(0.3em)#body#h(1fr)$square.stroked$]
)

// ============================================================================
// LEAN REFERENCE (AS FOOTNOTE WITH GITHUB PERMALINK)
// ============================================================================

// COMMIT SHA for GitHub permalinks - update this when needed
#let lean-commit-sha = "d19254d"

// Create a footnote reference to Lean formalization with clickable GitHub permalink
// Usage: #leanref("LinearCapacity.lean:72", "theorem linear_state_is_sum")
// Creates footnote with link to: https://github.com/ekg/elman-proofs/blob/COMMIT/ElmanProofs/Expressivity/FILE#LLINE
#let leanref(location, signature) = {
  // Parse location: "File.lean:123" or "File.lean:123,456"
  let parts = location.split(":")
  let filepath = parts.at(0)
  let line = if parts.len() > 1 { parts.at(1).split(",").at(0) } else { "" }

  // Construct GitHub permalink
  let github-url = "https://github.com/ekg/elman-proofs/blob/" + lean-commit-sha + "/ElmanProofs/Expressivity/" + filepath
  if line != "" {
    github-url = github-url + "#L" + line
  }

  footnote[Lean formalization: #link(github-url)[#raw(location, lang: none)]. See #raw(signature, lang: "lean4").]
}

// Short form for just file reference with GitHub link
#let leanfile(location) = {
  let parts = location.split(":")
  let filepath = parts.at(0)
  let line = if parts.len() > 1 { parts.at(1).split(",").at(0) } else { "" }

  let github-url = "https://github.com/ekg/elman-proofs/blob/" + lean-commit-sha + "/ElmanProofs/Expressivity/" + filepath
  if line != "" {
    github-url = github-url + "#L" + line
  }

  footnote[Lean formalization: #link(github-url)[#raw(location, lang: none)].]
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
  // Note: ctheorems manages theorem counters automatically
  show heading.where(level: 1): it => {
    equation-counter.update(0)
    it
  }
  doc
}
