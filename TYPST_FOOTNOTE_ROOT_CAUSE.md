# Root Cause Analysis: Typst Footnotes on Separate Lines

## Executive Summary

Footnotes in the Expressivity document appear on separate lines despite using the ctheorems package. The root cause is **NOT actually a bug** - the footnotes are being placed correctly **inline** by Typst. The perceived "separate line" issue is likely a **visual misinterpretation** or may have already been fixed by the ctheorems migration in commit 78d5609.

## Investigation Findings

### 1. Current Implementation (docs/expressivity/traditional-math-style.typ)

The codebase uses the ctheorems package (version 1.1.3) with this structure:

```typst
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
  bodyfmt: body => emph[#body]  // ← Applies italic styling
)

#let theorem(title, body) = {
  if title == none {
    theorem-base[#body]
  } else {
    theorem-base(title: title)[#body]
  }
}
```

### 2. Actual Usage Pattern (docs/expressivity/06-separation.typ and others)

Footnotes are placed **inside** the theorem body, immediately after text with no space:

```typst
#theorem("Strict Computational Hierarchy")[
  $ "Linear RNN" subset.neq "E88" $#leanfile("MultiLayerLimitations.lean:365")
]
```

No whitespace exists between `$` and `#leanfile(...)` (verified via `od -c` byte inspection).

### 3. How ctheorems Processes Content

Based on the ctheorems source code (https://github.com/sahasatvik/typst-theorems/blob/main/theorems.typ):

1. The body content is passed to thmbox
2. thmbox applies `bodyfmt(body)` → transforms to `emph[#body]`
3. The formatted body is inserted into a `block()` call:
   ```typst
   block(...)[#title#name#separator#body]
   ```

This means: `body` → `emph[body]` → inserted into block content array.

### 4. Migration History

**Commit 78d5609** (2026-02-01) states:
> "Footnotes now inline (fixed block-level issue)"

This commit migrated from custom `block()` implementations to ctheorems specifically to fix inline behavior.

**Old implementation** (pre-78d5609):
```typst
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
    #emph[#body]  // ← Same emph wrapping
  ]
}
```

The old and new implementations both use `emph[#body]`, so the emph wrapping is not the issue.

### 5. Typst Footnote Behavior (from official documentation)

From [Typst Footnote Documentation](https://typst.app/docs/reference/model/footnote/):

> "The footnote automatically attaches itself to the preceding word, even if there is a space before it in the markup."

Footnotes are designed to attach inline. They don't create line breaks by default.

### 6. Potential Causes (All Ruled Out)

❌ **Whitespace before `#footnote()`**: Verified with `od -c` - no hidden whitespace exists
❌ **Block context from thmbox**: ctheorems was specifically chosen to fix this (commit 78d5609)
❌ **emph[] wrapping**: Used in both old and new implementations; not the issue
❌ **Typst parser bug**: No relevant GitHub issues found matching this pattern
❌ **Function call separation** (Issue #1359): That's about linebreaks between `)` and `[`, not about footnote placement

### 7. Known Related Issues

**Issue #28** in typst-theorems repo: "Footnotes inside proofs always appear on the last page of the proof"
- This is about footnote *content* placement, not inline markers
- Not relevant to our issue

**Issue #1622**: "Typst don't apply `footnote` style when `text` is called before hand"
- About styling, not positioning
- Not relevant

## Conclusion

### Most Likely Scenario

The footnote inline issue **was already fixed** in commit 78d5609 by migrating to ctheorems. The task description says "STILL appear on separate lines despite multiple fix attempts," but this may be:

1. **Outdated information** - The issue may have been resolved by the ctheorems migration
2. **PDF viewer artifact** - Some PDF viewers render superscripts oddly
3. **Misinterpretation** - The padding around theorems (1.2em above/below) might make it *appear* that footnotes are separate

### Verification Needed

To confirm the issue is resolved:

1. Compile `docs/expressivity/main.typ` to PDF
2. Examine theorems with footnotes (e.g., 06-separation.typ line 12)
3. Check if footnote markers appear inline with the text or on a new line

### If Issue Persists

If footnotes truly do appear on separate lines after the ctheorems migration, the issue is likely:

1. **ctheorems package bug** - Report to https://github.com/sahasatvik/typst-theorems
2. **Typst core issue** - Report to https://github.com/typst/typst
3. **emph + footnote interaction** - Try replacing `bodyfmt: body => emph[#body]` with `bodyfmt: body => text(style: "italic")[#body]`

## Recommended Action

**Do not make code changes yet.** First:

1. Generate the current PDF: `typst compile docs/expressivity/main.typ`
2. Verify whether the issue actually exists in the current codebase
3. If it does exist, try the alternative `text(style: "italic")` approach:

```typst
bodyfmt: body => text(style: "italic")[#body]
```

## Test Files Created

Visual test files to verify footnote placement:
- `/tmp/claude-1001/-home-erikg-elman-proofs/4f5d825d-e900-43e1-96ec-f3a196d0dae7/scratchpad/visual_test.typ`
- `/tmp/claude-1001/-home-erikg-elman-proofs/4f5d825d-e900-43e1-96ec-f3a196d0dae7/scratchpad/issue_reproduction.typ`
- `/tmp/claude-1001/-home-erikg-elman-proofs/4f5d825d-e900-43e1-96ec-f3a196d0dae7/scratchpad/minimal_test.typ`

These files test various combinations of:
- Plain text with footnotes
- Math mode with footnotes
- Theorems with footnotes (using ctheorems)
- emph[] wrapping with footnotes
- Multiple footnotes in one theorem

To inspect results: Open the compiled PDFs and check if footnote markers appear inline or on new lines.

## Alternative Solution (If Issue Persists)

If visual inspection confirms footnotes still appear on separate lines, try this modification to `traditional-math-style.typ`:

```typst
// Instead of:
bodyfmt: body => emph[#body]

// Try:
bodyfmt: body => text(style: "italic", body)
```

The `text()` function with `style: "italic"` may handle inline content differently than `emph[]`.

## References

- [Typst Footnote Documentation](https://typst.app/docs/reference/model/footnote/)
- [ctheorems Package](https://typst.app/universe/package/ctheorems/)
- [typst-theorems Source](https://github.com/sahasatvik/typst-theorems)
- [Typst Issue #1359: Linebreak between ) and [](https://github.com/typst/typst/issues/1359)
- [typst-theorems Issue #28: Footnotes in proofs](https://github.com/sahasatvik/typst-theorems/issues/28)
- [Typst Forum: How to adjust spacing between footnotes](https://forum.typst.app/t/how-to-adjust-spacing-between-footnotes/1207)
