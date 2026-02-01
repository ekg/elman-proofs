# Lean 4 to Typst Conversion Tools - Research Report

**Date:** 2026-01-31
**Purpose:** Identify tools and workflows to convert Lean 4 proofs to Typst for inclusion in academic papers

## Executive Summary

**No direct Lean 4 → Typst conversion tools exist** as of January 2026. The Lean ecosystem primarily targets HTML and LaTeX output. To include Lean proofs in Typst documents, manual extraction or multi-step conversion workflows are required.

## Key Findings

### 1. Alectryon + LeanInk (Most Mature Literate Tool)

**What it is:**
- [Alectryon](https://github.com/cpitclaudel/alectryon): A literate programming tool originally for Coq, extended to Lean 4 via [LeanInk](https://github.com/leanprover/LeanInk)
- Produces richly annotated proof documents with tactic states and type information

**Lean 4 Support:** ✅ Yes (via LeanInk integration)

**Output Formats:**
- ✅ HTML (with interactive proof states)
- ✅ LaTeX (for embedding in papers)
- ✅ reStructuredText (.rst)
- ✅ JSON (fragment format for custom backends)
- ❌ Typst (not supported)

**Workflow for Typst:**
1. Use Alectryon to generate LaTeX output from .lean files
2. Manually convert LaTeX to Typst (Typst has some LaTeX compatibility via [mitex](https://typst.app/universe/package/mitex/))
3. OR: Use Alectryon's JSON output and write a custom Typst backend

**Limitations:**
- Alectryon support for Lean 4 is still in active development
- Requires using a [fork of Alectryon](https://github.com/Kha/alectryon) for full Lean 4 support
- No native Typst backend exists

**Sources:**
- [Alectryon GitHub](https://github.com/cpitclaudel/alectryon)
- [LeanInk GitHub](https://github.com/leanprover/LeanInk)
- [Lean Zulip: Literate programming discussion](https://leanprover-community.github.io/archive/stream/270676-lean4/topic/Literate.20programming.20in.20Lean4.20.2F.20Lean4.20.2B.20Org.20mode.html)

---

### 2. doc-gen4 (Official Lean Documentation Generator)

**What it is:**
- [Official documentation generator](https://github.com/leanprover/doc-gen4) for Lean 4 projects
- Used to generate API documentation for Mathlib and other Lean libraries

**Output Formats:**
- ✅ HTML only (requires HTTP server due to Same Origin Policy)
- ❌ No PDF, Markdown, LaTeX, or Typst output

**Workflow for Typst:**
- Not suitable for academic papers
- Designed for browsable API documentation, not literate proofs

**Sources:**
- [doc-gen4 GitHub](https://github.com/leanprover/doc-gen4)
- [Lean Zulip: doc-gen4 discussion](https://leanprover-community.github.io/archive/stream/270676-lean4/topic/doc-gen4.html)

---

### 3. Verso (New Documentation Authoring Tool)

**What it is:**
- [Modern Lean documentation tool](https://github.com/leanprover/verso) announced in 2024-2025
- Markdown-like syntax with full Lean metaprogramming integration
- Used for "Functional Programming in Lean" and the new Lean Reference Manual

**Output Formats:**
- ✅ HTML (with rich annotations, tactic proof states)
- ✅ JSON (cross-reference database: xref.json)
- ❌ No PDF or Typst output mentioned

**Workflow for Typst:**
- Currently HTML-focused
- Extensible architecture could theoretically support Typst backend (would require implementation)
- Best for book-length documentation, not paper proofs

**Sources:**
- [Verso GitHub](https://github.com/leanprover/verso)
- [Functional Programming in Lean](https://lean-lang.org/functional_programming_in_lean/) (built with Verso)
- [Lean Reference Manual](https://lean-lang.org/doc/reference/latest/) (built with Verso)

---

### 4. LaTeX Integration (Traditional Approach)

**What it is:**
- Embed Lean code snippets in LaTeX documents using syntax highlighting packages
- Manual approach used by most academic papers with Lean proofs

**Tools:**
- `listings` package with [lstlean.md configuration](https://github.com/leanprover-community/lean/blob/master/extras/latex/lstlean.md)
- `minted` package with Pygments lexer for better Unicode support
- Lean community maintains "Best practices for highlighting Lean code in LaTeX documents"

**Workflow for Typst:**
1. Write Lean proofs in .lean files
2. Extract theorem statements and proof sketches manually
3. Use Typst's `raw` blocks with Lean syntax highlighting
4. Manually maintain synchronization between .lean files and paper

**Limitations:**
- No automation of proof state display
- No verification that paper text matches actual Lean code
- Labor-intensive for large proofs

**Sources:**
- [Lean Zulip: Lean + LaTeX discussion](https://leanprover-community.github.io/archive/stream/113488-general/topic/Lean.20.2B.20LaTeX.3F.html)
- [Lean Zulip: latex topic](https://leanprover-community.github.io/archive/stream/113488-general/topic/latex.html)

---

## How Other Lean Projects Include Proofs in Papers

### Common Patterns:

1. **Theorem Statements Only**
   - Extract formal theorem statements from Lean
   - Rewrite in mathematical notation for paper
   - Cite Lean source code repository for full proofs
   - Example: "See `ElmanProofs/Expressivity/LinearLimitations.lean:42` for formalization"

2. **Proof Sketches**
   - Present high-level proof structure in paper
   - Reference Lean files for mechanically checked details
   - Use paper for intuition, Lean for rigor

3. **Literate Lean Books** (not papers)
   - Tools like Verso, Alectryon, mdbook used for longer-form documentation
   - Full integration of code and prose
   - Not suitable for conference/journal page limits

4. **Hybrid Approach**
   - Key lemmas shown in paper with natural language proofs
   - Full Lean development in appendix or supplementary materials
   - GitHub repository referenced for all source code

### Example Resources:
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/) - literate book format
- [100 Theorems in Lean](https://leanprover-community.github.io/100.html) - theorem statements with links
- [Mathlib Style Guide](https://leanprover-community.github.io/contribute/style.html) - formatting conventions

---

## Practical Recommendations for ElmanProofs → Typst

Given the current tool landscape, here are recommended approaches ranked by effort:

### Option 1: Manual Extraction (Lowest Effort)
**Best for:** Short, focused theorems you want in the paper body

**Steps:**
1. Read your Lean files: `ElmanProofs/Expressivity/LinearLimitations.lean`, etc.
2. Extract key theorem statements
3. Manually translate to Typst math notation:
   ```typst
   #theorem[
     Linear RNNs cannot compute XOR.
   ]

   #proof[
     See `ElmanProofs/Expressivity/LinearLimitations.lean:87` for
     mechanically verified proof.
   ]
   ```
4. Include Lean code in appendix or supplementary materials

**Pros:** Simple, full control over presentation
**Cons:** No automation, easy to desync from Lean source

---

### Option 2: Alectryon → LaTeX → Typst (Medium Effort)
**Best for:** Detailed proof presentations with tactic states

**Steps:**
1. Install Alectryon with Lean 4 support (may need fork)
2. Generate LaTeX: `alectryon --backend latex file.lean`
3. Convert LaTeX fragments to Typst manually or via tools
4. Embed in Typst document

**Pros:** Preserves proof states, semi-automated
**Cons:** Multi-step conversion, formatting may break

---

### Option 3: JSON Export + Custom Typst Renderer (High Effort)
**Best for:** Repeated use across multiple papers/projects

**Steps:**
1. Use Alectryon to generate JSON: `alectryon --backend json file.lean`
2. Write Python/Rust script to convert JSON to Typst syntax
3. Integrate into build pipeline

**Pros:** Fully automated, customizable formatting
**Cons:** Significant upfront development, maintenance burden

---

### Option 4: Code Inclusion with References (Recommended for ElmanProofs)
**Best for:** Academic papers with page limits

**Steps:**
1. Present theorem statements and proof ideas in natural language in Typst
2. Include short Lean snippets in code blocks for key definitions:
   ```typst
   ```lean
   theorem xor_not_linear : ∀ (f : ℝ → ℝ → ℝ), IsLinear f → ¬ComputesXOR f := by
     sorry
   ```
   ```
3. Add inline references: "See `LinearLimitations.lean:87`"
4. Provide full Lean sources in:
   - GitHub repository
   - arXiv supplementary materials
   - Appendix (if space allows)

**Pros:** Clean paper presentation, verifiable proofs available
**Cons:** Readers must consult external sources for full proofs

---

## Specific to Your ElmanProofs Project

### Your Lean Files:
- `ElmanProofs/Expressivity/LinearLimitations.lean` - XOR, threshold impossibility
- `ElmanProofs/Expressivity/LinearCapacity.lean` - linear state characterization
- `ElmanProofs/Expressivity/MultiLayerLimitations.lean` - multi-layer extensions
- `ElmanProofs/Architectures/RecurrenceLinearity.lean` - architecture taxonomy
- `ElmanProofs/Architectures/Mamba2_SSM.lean` - Mamba2 formalization

### Recommended Approach:
1. **In Typst paper (`docs/expressivity/11-experimental-results.typ`):**
   - State theorems in mathematical notation
   - Provide proof intuitions/sketches
   - Reference Lean files by path and line number

2. **In appendix or supplementary materials:**
   - Include full Lean source code with syntax highlighting
   - Use Typst's `#raw(lang: "lean", read("../../ElmanProofs/..."))` if feasible
   - OR: Provide as separate `.lean` files in submission

3. **For long-term reuse:**
   - Consider writing a simple Python script to:
     - Parse Lean files for theorem statements
     - Extract docstrings
     - Generate Typst theorem blocks automatically
   - Keep script in `docs/scripts/lean_to_typst.py`

---

## Summary Table

| Tool | Lean 4 Support | Typst Output | Use Case |
|------|----------------|--------------|----------|
| **Alectryon + LeanInk** | ✅ (via fork) | ❌ (LaTeX/HTML only) | Literate proofs with tactic states |
| **doc-gen4** | ✅ | ❌ (HTML only) | API documentation |
| **Verso** | ✅ | ❌ (HTML only) | Book-length documentation |
| **LaTeX packages** | ✅ | 🟨 (via manual conversion) | Syntax highlighting in papers |
| **Manual extraction** | ✅ | ✅ | Custom formatting, full control |

---

## Future Possibilities

### Potential Developments:
1. **Typst backend for Alectryon** - Could be proposed to maintainers or implemented as community extension
2. **Verso Typst output** - Extensible architecture could support this with development effort
3. **Lean LSP → Typst integration** - Could extract proof states via Language Server Protocol
4. **Academic Lean template for Typst** - Community could develop best practices package

### Where to Ask:
- [Lean Zulip](https://leanprover.zulipchat.com/) - Active community, search "literate" or "LaTeX"
- [Lean Community GitHub discussions](https://github.com/leanprover-community)
- Alectryon issue tracker for feature requests

---

## Sources

- [Alectryon GitHub Repository](https://github.com/cpitclaudel/alectryon)
- [LeanInk GitHub Repository](https://github.com/leanprover/LeanInk)
- [doc-gen4 GitHub Repository](https://github.com/leanprover/doc-gen4)
- [Verso Documentation Tool](https://github.com/leanprover/verso)
- [Functional Programming in Lean](https://lean-lang.org/functional_programming_in_lean/)
- [Mathematics in Lean](https://leanprover-community.github.io/mathematics_in_lean/)
- [Lean Zulip: Literate Programming](https://leanprover-community.github.io/archive/stream/270676-lean4/topic/Literate.20programming.20in.20Lean4.20.2F.20Lean4.20.2B.20Org.20mode.html)
- [Lean Zulip: LaTeX Integration](https://leanprover-community.github.io/archive/stream/113488-general/topic/Lean.20.2B.20LaTeX.3F.html)
- [Typst mitex Package](https://typst.app/universe/package/mitex/)
- [Lean Together 2026](https://leanprover-community.github.io/lt2026/)
- [Mathlib Style Guidelines](https://leanprover-community.github.io/contribute/style.html)

---

**Conclusion:** For the ElmanProofs project, manual extraction of theorem statements with references to Lean source files is the most practical approach for a Typst academic paper. Future automation could be built using Alectryon's JSON backend or a custom Lean AST parser.
