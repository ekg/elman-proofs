# Research: Section and Subsection Heading Styles in Typst

## Problem Statement
The current document (docs/expressivity/main.typ) has subheadings that appear centered and look awkward. Need to determine proper heading styles for an academic mathematics paper.

## Current Heading Style in main.typ

From `main.typ` lines 17-33:

```typst
#show heading.where(level: 1): it => {
  v(1.5em)
  text(size: 16pt, weight: "bold", it.body)
  v(0.75em)
}

#show heading.where(level: 2): it => {
  v(1em)
  text(size: 13pt, weight: "bold", it.body)
  v(0.5em)
}

#show heading.where(level: 3): it => {
  v(0.75em)
  text(size: 11pt, weight: "bold", it.body)
  v(0.25em)
}
```

**Current properties:**
- **Level 1 (=)**: 16pt bold, 1.5em space above, 0.75em below
- **Level 2 (==)**: 13pt bold, 1em space above, 0.5em below
- **Level 3 (===)**: 11pt bold, 0.75em space above, 0.25em below
- **No explicit alignment** - defaults to document alignment

**Body text**: 11pt New Computer Modern, justified

## Academic Math Paper Standards

### LaTeX Article Class Defaults

From [LaTeX.org forums](https://latex.org/forum/viewtopic.php?t=15171) and [LaTeX.org section heading styles](https://latex.org/forum/viewtopic.php?t=7179):

- Section headings are **left-aligned by default** in LaTeX article class
- Section headings are **bold** by default
- No additional customization needed for standard left-aligned bold headings
- Centered headings require explicit modification (not the default)

### Academic Standards Across Disciplines

From [Bridges Math Organization](https://www.bridgesmathart.org/for-authors-and-participants/formatting/) and [Wordvice formatting guide](https://blog.wordvice.com/formatting-research-paper-headings-and-subheadings/):

- **Mathematics papers**: Bold for Title, Section Headings, and SubSection Headings
- **General academic**: Section headings use bold 14pt font, subheadings use 12pt bold
- **Default alignment**: Left-aligned text except for titles and references
- **AMA Style**: Section headings are bold and left-aligned; subsection headings are italicized and left-aligned
- **APA Style (7th ed.)**: Level 1 headings centered and bold; Level 2+ headings are flush left (left-aligned) and bold

### Consensus

**For mathematics papers, the standard is:**
1. **Left-aligned** (not centered)
2. **Bold** (already implemented)
3. Font size hierarchy: larger for higher-level headings
4. Adequate whitespace above/below

## Problem Diagnosis

The current Typst code **does not explicitly set alignment**, which means headings inherit the document's default alignment. However, since the body text uses `#set par(justify: true)`, this should **not** cause centering.

**Possible causes of perceived centering:**
1. Short headings may appear centered on justified text
2. PDF viewer rendering
3. Specific section files using `#align(center)[]` blocks

Let me check a section file to see if explicit centering is used:

From `01-introduction.typ` line 5:
```typst
= Introduction
```

No explicit centering found in the section itself. Headings should be left-aligned.

## Recommended Heading Style for Math Papers

Based on LaTeX article class defaults and academic standards:

```typst
#show heading.where(level: 1): it => {
  v(1.5em)
  align(left)[
    text(size: 16pt, weight: "bold", it.body)
  ]
  v(0.75em)
}

#show heading.where(level: 2): it => {
  v(1em)
  align(left)[
    text(size: 13pt, weight: "bold", it.body)
  ]
  v(0.5em)
}

#show heading.where(level: 3): it => {
  v(0.75em)
  align(left)[
    text(size: 11pt, weight: "bold", it.body)
  ]
  v(0.25em)
}
```

### Rationale

1. **Explicit left alignment**: Makes intention clear and prevents any ambiguity
2. **Bold weight**: Standard for all academic math papers
3. **Size hierarchy**: 16pt → 13pt → 11pt provides clear visual distinction
4. **Spacing**: Current spacing is reasonable (more above than below each heading)
5. **Consistency with LaTeX defaults**: Matches what readers expect from math papers

### Alternative: LaTeX Article Class Closer Match

For even closer alignment with LaTeX article defaults:

```typst
// Level 1: \section in LaTeX article (~17pt, \Large)
#show heading.where(level: 1): it => {
  v(3.5ex, weak: true)  // LaTeX default spacing
  align(left)[
    text(size: 14.4pt, weight: "bold", it.body)  // \Large size
  ]
  v(2.3ex, weak: true)
}

// Level 2: \subsection in LaTeX article (~12pt, \large)
#show heading.where(level: 2): it => {
  v(3.25ex, weak: true)
  align(left)[
    text(size: 12pt, weight: "bold", it.body)  // \large size
  ]
  v(1.5ex, weak: true)
}

// Level 3: \subsubsection in LaTeX article (normalsize)
#show heading.where(level: 3): it => {
  v(3.25ex, weak: true)
  align(left)[
    text(size: 11pt, weight: "bold", it.body)  // normalsize
  ]
  v(1.5ex, weak: true)
}
```

## Additional Considerations

### Numbering
LaTeX article class numbers sections by default. The current Typst document does not show section numbers in the headings. Consider adding:

```typst
#set heading(numbering: "1.1")
```

### Page Break Avoidance
Consider adding to heading rules:
```typst
pagebreak(weak: true)  // Avoid orphan headings at bottom of page
```

## Recommendation Summary

**Immediate fix**: Add explicit `align(left)[]` to all three heading levels.

**Optimal style for math paper**:
- Left-aligned, bold, unnumbered (current choice) or numbered (LaTeX default)
- Font sizes: 16pt (h1), 13pt (h2), 11pt (h3) - slightly larger than LaTeX but acceptable
- OR: 14.4pt (h1), 12pt (h2), 11pt (h3) - closer to LaTeX \Large, \large, normalsize
- Current spacing is adequate

**Why left-aligned matters**: Math papers follow LaTeX conventions where section headings are structural markers that guide the reader through logical progression. Left alignment makes them stand out as organizational elements rather than decorative titles.

## Sources

- [Justify Headings to left Side - LaTeX.org](https://latex.org/forum/viewtopic.php?t=15171)
- [Change section heading styles - LaTeX.org](https://latex.org/forum/viewtopic.php?t=7179)
- [Formatting Research Paper Headings and Subheadings - Wordvice](https://blog.wordvice.com/formatting-research-paper-headings-and-subheadings/)
- [Paper Formatting Guidelines – The Bridges Organization](https://www.bridgesmathart.org/for-authors-and-participants/formatting/)
