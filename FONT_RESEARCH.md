# Font Research for Math Papers in Typst

## Current Font
**DejaVu Sans** (10.5pt) - A sans-serif font currently used in `docs/expressivity/main.typ:12`

**Problem**: Sans-serif fonts are not suitable for complex mathematical typesetting in academic papers. They lack the professional appearance and readability expected in formal mathematical publications.

## Available Fonts in Typst
Based on system check (`typst fonts`), these fonts are available:
- ✓ New Computer Modern (serif + math)
- ✓ New Computer Modern Math (math companion)
- ✓ Libertinus Serif (serif + math support via Libertinus Math)
- ✓ DejaVu Serif (serif variant)
- ✓ DejaVu Math TeX Gyre (math support)
- ✗ STIX Two (not installed, but highly recommended if available)

## Research Findings

### Industry Standard Math Fonts

1. **Computer Modern** (LaTeX default)
   - The gold standard for mathematical typesetting
   - Instantly recognizable in academic papers
   - Creates "authoritative" look for formal mathematics
   - Full math symbol support (designed by Knuth for TeX)

2. **STIX Two Math**
   - Designed specifically for scientific publishing
   - 4,605 glyphs (most comprehensive coverage)
   - Used by major publishers (AMS, IEEE, Elsevier)
   - Modern, professional appearance
   - **Limitation**: Not in default Typst installation (requires manual install)

3. **Libertinus Math**
   - Open source continuation of Linux Libertine
   - 3,648 glyphs (good coverage)
   - Elegant, humanist serif design
   - Excellent for readability
   - **Available in Typst by default**

4. **Cambria Math**
   - Microsoft's math font
   - Extensive symbol support
   - Optimal at 11-13pt
   - **Not available in Typst**

### Font Characteristics Comparison

| Font | Style | Math Glyphs | Professional Look | Readability | Availability |
|------|-------|-------------|-------------------|-------------|--------------|
| New Computer Modern | Classic serif | ★★★★★ | ★★★★★ | ★★★★ | ✓ Built-in |
| Libertinus Serif | Humanist serif | ★★★★ | ★★★★ | ★★★★★ | ✓ Built-in |
| DejaVu Serif | Modern serif | ★★★ | ★★★ | ★★★★ | ✓ Built-in |
| STIX Two | Scientific serif | ★★★★★ | ★★★★★ | ★★★★ | ✗ Manual install |
| DejaVu Sans (current) | Sans-serif | ★★ | ★ | ★★★ | ✓ Built-in |

## Top Recommendations

### Option 1: New Computer Modern (RECOMMENDED)
**Best for**: Formal mathematical papers, theorem-heavy documents

```typst
#set text(font: "New Computer Modern", size: 11pt)
```

**Advantages**:
- Gold standard in mathematical typesetting
- Instant recognition and authority
- Designed specifically for mathematics
- Full Greek letter and operator support
- Perfect for papers with heavy proofs/theorems
- Default in academic LaTeX papers

**Why choose this**: Your document contains formal theorems, Lean formalizations, and heavy mathematical notation. Computer Modern signals "serious mathematics" to readers.

### Option 2: Libertinus Serif
**Best for**: More elegant, humanist appearance while maintaining mathematical rigor

```typst
#set text(font: "Libertinus Serif", size: 11pt)
```

**Advantages**:
- More modern, elegant appearance than Computer Modern
- Excellent readability for extended text
- Full math support via Libertinus Math
- Open source, actively maintained
- Slightly warmer, more approachable than Computer Modern

**Why choose this**: If you want professionalism without the "classic LaTeX" look, or if you have significant prose sections alongside math.

### Option 3: STIX Two (if available)
**Best for**: Maximum glyph coverage, scientific publishing standards

```typst
#set text(font: "STIX Two", size: 11pt)
```

**Advantages**:
- Industry standard for scientific publishers
- Most comprehensive symbol coverage (4,605 glyphs)
- Modern, professional appearance
- Designed for complex scientific notation

**Limitation**: Requires manual installation. To install:
```bash
# Download STIX Two fonts from https://github.com/stipub/stixfonts
# Install system-wide or use --font-path flag with typst
```

## Sample Comparison

I compiled a comparison document showing all available fonts with your actual mathematical content:
- Location: `/tmp/claude-1001/-home-erikg-elman-proofs/.../scratchpad/font-comparison.pdf`
- Size: 117KB (4 pages)

The comparison includes:
- Greek letters: α, β, γ, δ, ε, θ, λ, μ, π, σ, φ, ψ, ω
- Operators: ⊊, ×, ⊤, ∫, ∑, ∏, ∇, ∂, ∞
- Complex expressions: $S_t = tanh(αS_{t-1} + δv_tk_t^⊤)$
- Hierarchy notation: $\text{Linear SSM} ⊊ \text{TC}^0 ⊊ \text{E88}$

## Implementation Recommendation

### Primary Recommendation: New Computer Modern

**Change in `docs/expressivity/main.typ`:**

```typst
// From:
#set text(font: "DejaVu Sans", size: 10.5pt)

// To:
#set text(font: "New Computer Modern", size: 11pt)
```

**Rationale**:
1. Your document is formal mathematical analysis with theorem-heavy content
2. Contains extensive Lean formalizations and proofs
3. Needs to signal academic rigor and mathematical authority
4. Computer Modern is the expected font for formal mathematical papers
5. Already available (no installation needed)
6. 11pt is the standard size for academic papers (10.5pt is slightly too small)

### Alternative: Libertinus Serif for Better Readability

If you prefer a more modern appearance with better readability for longer text sections:

```typst
#set text(font: "Libertinus Serif", size: 11pt)
```

Both fonts support all your mathematical notation fully, including:
- Greek letters (α, β, δ, etc.)
- Operators (∑, ∏, ⊊, ×, ⊤, etc.)
- Subscripts and superscripts
- Complex mathematical expressions

## Sources

- [Fonts for Text and Mathematics](https://empslocal.ex.ac.uk/people/staff/gv219/aofd/fonts/)
- [Is there a perfect maths font? - Chalkdust](https://chalkdustmagazine.com/blog/is-there-a-perfect-maths-font/)
- [Mathematical fonts - Overleaf](https://www.overleaf.com/learn/latex/Mathematical_fonts)
- [Math – Typst Documentation](https://typst.app/docs/reference/math/)
- [Text Function – Typst Documentation](https://typst.app/docs/reference/text/text/)
- [Coverage of LaTeX math symbols in dedicated math fonts | Plurimath](https://www.plurimath.org/blog/2023-08-14-font-coverage-latex-symbols/)
- [Manual: math fonts](https://type.today/en/journal/mathfonts)

## Conclusion

**Switch from DejaVu Sans to New Computer Modern at 11pt**. This provides the professional mathematical appearance expected in formal academic papers while maintaining full support for all mathematical notation in your document.
