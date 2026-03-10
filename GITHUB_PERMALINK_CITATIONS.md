# GitHub Permalink Citations in Typst

## Research Summary

This document presents a research analysis on using GitHub permalinks as citations instead of footnotes in the Expressivity Analysis paper. The goal is to make Lean theorem references clickable links to the actual source code.

---

## 1. GitHub Permalink Format

### URL Structure

GitHub permalinks to specific lines use the format:
```
https://github.com/<org>/<repo>/blob/<commit-sha>/<file-path>#L<line-number>
```

**Example:**
```
https://github.com/erggarrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/LinearLimitations.lean#L84
```

### Key Features

- **Commit SHA**: Use full or short (7+ char) commit hash to create permanent reference
- **Line Numbers**: Single line `#L42` or range `#L42-L50`
- **Keyboard Shortcut**: Press `y` while viewing a file on GitHub to convert branch URL to permalink
- **Line Selection**: Click line numbers to add to URL; shift-click for ranges

### Creating Permalinks

1. Navigate to file on GitHub
2. Press `y` key to get permalink with current commit
3. Click line number(s) to add `#L` fragment
4. Copy URL from browser

---

## 2. Typst Bibliography & URL Support

### Native Capabilities

Typst has built-in bibliography support via the Hayagriva format:

```typst
#bibliography("references.yml", style: "ieee")
#cite(<key>)
```

### The Blinky Package

For clickable URLs in bibliography entries, use the `blinky` package:

```typst
#import "@preview/blinky:0.1.0": *

#show bibliography: link-bib-urls
#bibliography("refs.bib", style: "ieee")
```

**Features:**
- Automatically makes titles clickable if BibTeX entry has `url` or `doi` field
- Works with any CSL citation style
- Requires special `==` markers in CSL style (handled automatically)

---

## 3. Hayagriva URL Field Format

Hayagriva (Typst's native bibliography format) supports URLs with optional access dates:

### YAML Format

```yaml
lean-linear-capacity:
  type: misc
  title: Linear State Capacity Theorem
  author: Garrison, Erik
  url:
    value: https://github.com/erggarrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/LinearCapacity.lean#L72
    date: 2026-01-30
  parent:
    - type: repository
      title: ElmanProofs
      url: https://github.com/erggarrison/elman-proofs
```

### Compact Format

```yaml
url: { value: https://github.com/.../file.lean#L42, date: 2026-01-30 }
```

---

## 4. Academic Best Practices for Citing Code

### Citation Style Recommendations

**For Software/Code Citations:**

1. **APA Format** (most common for software):
   ```
   Author. (Year). Repository Name [Computer software]. GitHub. URL
   ```

2. **MLA Format**:
   ```
   Author. Title of Repository, Version, Date, GitHub, URL
   ```

3. **IEEE Format** (technical papers):
   ```
   [#] Author, "Title," Version, Publisher, Date. [Online]. Available: URL
   ```

### GitHub CITATION.cff

GitHub repositories can include a `CITATION.cff` file to specify preferred citation:

```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Garrison
    given-names: Erik
title: "ElmanProofs"
version: 1.0.0
date-released: 2026-01-30
url: "https://github.com/erggarrison/elman-proofs"
```

### Best Practice Recommendations

1. **Use DOIs when possible**: Archive on Zenodo to get permanent DOI
2. **Cite specific versions**: Use commit SHAs or release tags, not branch names
3. **Include access dates**: URLs can change; document when you accessed
4. **Cite at appropriate granularity**:
   - Repository-level for citing the whole project
   - File-level for specific algorithms/theorems
   - Line-level for exact theorem statements

---

## 5. Proposed Implementation for ElmanProofs

### Option A: Inline Links (Simple)

Replace current footnote system with inline clickable citations:

```typst
// Define link helper
#let lean(location, url) = link(url)[#text(fill: blue)[#location]]

// Usage in document
#theorem("Running Threshold")[
  No $D$-layer linear-temporal model computes running threshold.
  Linear-temporal outputs are continuous; threshold is discontinuous.
] #lean("ExactCounting.lean:344", "https://github.com/erggarrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/ExactCounting.lean#L344")
```

**Pros:**
- Direct, simple implementation
- Immediately clickable
- No bibliography management needed

**Cons:**
- Long URLs clutter source code
- Hard to maintain if commit changes
- Not traditional academic citation format

---

### Option B: Bibliography with Blinky (Academic)

Create bibliography entries for each major theorem file:

**references.yml:**
```yaml
linear-capacity-l72:
  type: misc
  title: "Linear State Capacity Theorem"
  author: Garrison, Erik
  date: 2026-01-30
  url: https://github.com/erggarrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/LinearCapacity.lean#L72
  parent:
    - type: repository
      title: "ElmanProofs: Formal Expressivity Analysis"
      url: https://github.com/erggarrison/elman-proofs

exact-counting-l344:
  type: misc
  title: "Running Threshold Impossibility"
  author: Garrison, Erik
  date: 2026-01-30
  url: https://github.com/erggarrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/ExactCounting.lean#L344
  parent:
    - type: repository
      title: "ElmanProofs: Formal Expressivity Analysis"
      url: https://github.com/ergrison/elman-proofs
```

**In document:**
```typst
#import "@preview/blinky:0.1.0": *

#theorem("Running Threshold")[
  No $D$-layer linear-temporal model computes running threshold.
  Linear-temporal outputs are continuous; threshold is discontinuous. @exact-counting-l344
]

#show bibliography: link-bib-urls
#bibliography("references.yml", style: "ieee")
```

**Pros:**
- Traditional academic format
- Centralized URL management
- Proper citation in bibliography section
- Clickable titles via Blinky

**Cons:**
- More setup required
- Bibliography can get large with many theorem citations
- Citations appear as [1], [2], etc., not as readable file:line

---

### Option C: Hybrid Approach (RECOMMENDED)

Use custom citation macro that creates readable inline references with clickable links:

```typst
// Store base URL in one place
#let repo-base = "https://github.com/ergrison/elman-proofs/blob/d19254d/ElmanProofs/Expressivity/"

// Custom citation that shows file:line and links to GitHub
#let lean-cite(file, line, label: none) = {
  let url = repo-base + file + "#L" + str(line)
  let display = if label != none {
    label
  } else {
    file + ":" + str(line)
  }
  link(url)[#text(fill: rgb("#0066cc"), size: 0.9em)[#display]]
}

// Usage
#theorem("Running Threshold")[
  No $D$-layer linear-temporal model computes running threshold.
  Linear-temporal outputs are continuous; threshold is discontinuous.
] #lean-cite("ExactCounting.lean", 344)
```

**Rendered output:**
```
Theorem 1 (Running Threshold). No D-layer linear-temporal model computes
running threshold. Linear-temporal outputs are continuous; threshold is
discontinuous. ExactCounting.lean:344
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
                                 (blue, clickable link)
```

**Pros:**
- Readable: Shows actual file and line number
- Clickable: Direct link to source
- Maintainable: Change commit SHA in one place
- Clean: Inline, not footnote
- Professional: Looks like code citation

**Cons:**
- Custom macro (but simple)
- Not traditional bibliography citation

---

## 6. Implementation Example

Here's a complete working example:

**traditional-math-style.typ additions:**

```typst
// GitHub permalink citation system
#let repo-base = "https://github.com/ergrison/elman-proofs/blob/"
#let commit-sha = "d19254d"  // Update this when documenting new version
#let code-base = "/ElmanProofs/Expressivity/"

// Create clickable citation to Lean source code
#let lean-cite(file, line, label: none) = {
  let url = repo-base + commit-sha + code-base + file + "#L" + str(line)
  let display = if label != none {
    label
  } else {
    file + ":" + str(line)
  }
  // Slightly smaller, blue text, inline link
  h(0.2em)
  link(url)[#text(fill: rgb("#0066cc"), size: 0.9em)[#display]]
}

// Alternative: with line range
#let lean-cite-range(file, start, end, label: none) = {
  let url = repo-base + commit-sha + code-base + file + "#L" + str(start) + "-L" + str(end)
  let display = if label != none {
    label
  } else {
    file + ":" + str(start) + "-" + str(end)
  }
  h(0.2em)
  link(url)[#text(fill: rgb("#0066cc"), size: 0.9em)[#display]]
}
```

**Usage in document:**

```typst
#import "traditional-math-style.typ": *

#theorem("Linear State Representation")[
  The state $h_t$ of any linear-temporal RNN equals
  $h_t = sum_(i=1)^t alpha^(t-i) x_i$ for some weights $alpha^(t-i)$.
] #lean-cite("LinearCapacity.lean", 72)

#theorem("XOR Impossibility")[
  No linear function computes XOR: $f(0,0) + f(1,1) != f(0,1) + f(1,0)$.
] #lean-cite("LinearLimitations.lean", 156)

#theorem("Running Parity")[
  No linear-temporal model computes $y_t = x_1 xor ... xor x_t$.
] #lean-cite-range("RunningParity.lean", 200, 215, label: "Parity.impossible")
```

---

## 7. Comparison with Current System

### Current (Footnotes):

```typst
#let leanref(location, signature) = footnote[
  Lean formalization: #raw(location, lang: none).
  See #raw(signature, lang: "lean4").
]

#theorem("Test")[Statement.] #leanref("File.lean:42", "theorem name")
```

**Issues:**
- Footnotes appear at page bottom
- Not clickable
- Requires two page locations (statement + footnote)
- Harder to follow when reading

### Proposed (Inline Links):

```typst
#theorem("Test")[Statement.] #lean-cite("File.lean", 42)
```

**Benefits:**
- Inline, immediately visible
- Clickable to view source
- One location, easier to scan
- Modern web-style citation
- Still preserves file:line reference

---

## 8. Recommendations

### For the ElmanProofs paper:

1. **Adopt Option C (Hybrid)**: Custom inline clickable citations
2. **Keep commit SHA in one place**: Easy to update when documenting new version
3. **Use for all theorem citations**: Consistent style throughout
4. **Remove footnote system**: Cleaner, more direct
5. **Update as needed**: When committing new proofs, update SHA in one place

### Migration path:

1. Add `lean-cite` and `lean-cite-range` to `traditional-math-style.typ`
2. Replace all `#leanref()` and `#leanfile()` calls with `#lean-cite()`
3. Test PDF rendering to ensure links work
4. Remove old footnote definitions

### Example replacement:

**Before:**
```typst
]#leanfile("ExactCounting.lean:344")
```

**After:**
```typst
] #lean-cite("ExactCounting.lean", 344)
```

---

## 9. Technical Considerations

### PDF Compatibility

- Typst generates PDFs with clickable hyperlinks
- Links work in all modern PDF viewers
- Printed versions show blue text (still useful as reference)

### Maintenance

- When proofs change location, update line numbers
- When creating new version, update commit SHA once
- Consider using release tags instead of commit SHAs

### Accessibility

- Screen readers will read the link text (file:line)
- High contrast between blue link and black text
- Clear visual indication that text is clickable

---

## Conclusion

**Replacing footnotes with GitHub permalink citations is both feasible and recommended.**

The hybrid approach (Option C) provides the best balance:
- ✅ Clickable links to actual source code
- ✅ Readable inline citations (file:line format)
- ✅ Easy maintenance (one place to update commit SHA)
- ✅ Professional academic appearance
- ✅ Modern web-style citations
- ✅ Direct connection between theorem and proof

This approach is superior to footnotes for code citations and aligns with modern practices for citing software and formal proofs.

---

## Sources

- [Creating a permanent link to a code snippet - GitHub Docs](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-a-permanent-link-to-a-code-snippet)
- [Getting permanent links to files - GitHub Docs](https://docs.github.com/en/repositories/working-with-files/using-files/getting-permanent-links-to-files)
- [Bibliography Function – Typst Documentation](https://typst.app/docs/reference/model/bibliography/)
- [blinky – Typst Universe](https://typst.app/universe/package/blinky/)
- [GitHub - alexanderkoller/typst-blinky](https://github.com/alexanderkoller/typst-blinky)
- [hayagriva/docs/file-format.md - GitHub](https://github.com/typst/hayagriva/blob/main/docs/file-format.md)
- [How To Cite A Github Repository - Academia Insider](https://academiainsider.com/how-to-cite-a-github-repository-cite-github-repositories-with-right-citation/)
- [About CITATION files - GitHub Docs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [Citing Source Code in MLA Style | MLA Style Center](https://style.mla.org/citing-source-code/)
