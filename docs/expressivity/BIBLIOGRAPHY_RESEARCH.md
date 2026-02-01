# Typst Bibliography and Citation Research

Research findings on how to handle references and citations in Typst for academic CS/ML papers.

## Overview

Typst uses **Hayagriva** as its bibliography management system, which is a Rust-based tool developed specifically for Typst. Hayagriva supports both:
- **BibTeX/BibLaTeX** `.bib` files (for compatibility with existing bibliographies)
- **Hayagriva YAML** `.yml` or `.yaml` files (native format, easier to write and read)

## Quick Start

In your `.typ` file:
```typst
// At the end of your document
#bibliography("references.yml")

// Or use BibTeX
#bibliography("references.bib")

// Citation in text
According to @elman1990finding, recurrent networks...

// Multiple citations
Recent work @hochreiter1997long @vaswani2017attention shows...

// With page numbers or supplements
As shown in @elman1990finding[p. 23], ...
```

## Citation Styles for CS/ML Papers

### Built-in Styles

Typst includes built-in support for common academic citation styles:

```typst
#bibliography("references.yml", style: "ieee")
#bibliography("references.yml", style: "apa")
#bibliography("references.yml", style: "chicago-author-date")
```

### Custom CSL Styles

For ACM or other specific CS conference styles, use CSL (Citation Style Language) files:

```typst
#bibliography("references.yml", style: "acm-sig-proceedings.csl")
```

**Common CS/ML citation styles:**
- **IEEE**: Numeric citations `[1]`, common in computer engineering and ML
- **ACM**: Numeric citations in square brackets, standard for ACM conferences (ICML, NeurIPS when they use ACM format)
- **APA**: Author-year format `(Smith, 2020)`, sometimes used in cognitive science ML papers
- **Nature**: Superscript numbers, used in some interdisciplinary ML venues

Download CSL files from: https://github.com/citation-style-language/styles

## Hayagriva YAML Format

### Basic Structure

A Hayagriva file is a YAML document containing a mapping of bibliography entries:

```yaml
entry-key:
  type: EntryType
  title: Title of Work
  author: ["LastName, FirstName", "LastName, FirstName"]
  date: YYYY-MM-DD
  # Additional fields...
```

### Supported Entry Types

- `Article`: Journal articles, conference papers
- `Book`: Books, monographs
- `Chapter`: Book chapters
- `Thesis`: PhD dissertations, master's theses
- `Web`: Websites, blog posts, online documentation
- `Repository`: Software repositories (GitHub, GitLab)
- `Conference`: Conference papers (with parent for proceedings)
- `Manuscript`: Preprints, unpublished works
- `Report`: Technical reports
- `Video`, `Audio`: Multimedia content

### Essential Fields

**Core fields (most entry types):**
- `type`: Required entry type
- `title`: Work title (formattable text)
- `author`: Person or list of persons in "LastName, FirstName" format
- `date`: ISO date format (YYYY-MM-DD or YYYY-MM)

**Common optional fields:**
- `parent`: Parent publication (journal, conference proceedings)
- `publisher`: Publishing organization
- `url`: Web address (can include access date)
- `doi`: Digital Object Identifier
- `serial-number`: ISBN, ISSN, arXiv ID, PMID, etc.
- `volume`, `issue`: For journals
- `page-range`: "start-end" or single page
- `abstract`: Work summary
- `location`: Publication location or conference venue

## Example Bibliography File

Here's a complete example `references.yml` for a CS/ML paper with various citation types:

```yaml
# Classic foundational paper
elman1990finding:
  type: Article
  title: Finding structure in time
  author: Elman, Jeffrey L.
  date: 1990
  parent:
    type: Periodical
    title: Cognitive Science
    volume: 14
    issue: 2
  page-range: 179-211
  doi: 10.1207/s15516709cog1402_1

# Modern deep learning paper
vaswani2017attention:
  type: Conference
  title: "Attention is all you need"
  author:
    - Vaswani, Ashish
    - Shazeer, Noam
    - Parmar, Niki
    - Uszkoreit, Jakob
    - Jones, Llion
    - Gomez, Aidan N.
    - Kaiser, Lukasz
    - Polosukhin, Illia
  date: 2017
  parent:
    type: Proceedings
    title: Advances in Neural Information Processing Systems
    volume: 30
  page-range: 5998-6008
  url: https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

# LSTM paper
hochreiter1997long:
  type: Article
  title: Long short-term memory
  author:
    - Hochreiter, Sepp
    - Schmidhuber, Jürgen
  date: 1997
  parent:
    type: Periodical
    title: Neural Computation
    volume: 9
    issue: 8
  page-range: 1735-1780
  doi: 10.1162/neco.1997.9.8.1735

# arXiv preprint
gu2023mamba:
  type: Manuscript
  title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
  author:
    - Gu, Albert
    - Dao, Tri
  date: 2023-12-01
  serial-number:
    arxiv: "2312.00752"
  url: https://arxiv.org/abs/2312.00752

# Book
goodfellow2016deep:
  type: Book
  title: Deep Learning
  author:
    - Goodfellow, Ian
    - Bengio, Yoshua
    - Courville, Aaron
  publisher: MIT Press
  date: 2016
  url: https://www.deeplearningbook.org/
  isbn: 978-0262035613

# GitHub repository (software citation)
lean4:
  type: Repository
  title: "Lean 4: Theorem Prover and Programming Language"
  author:
    - de Moura, Leonardo
    - Ullrich, Sebastian
  date: 2021
  publisher: GitHub
  url: https://github.com/leanprover/lean4
  version: 4.0.0

# Lean theorem prover system description
moura2015lean:
  type: Conference
  title: "The Lean Theorem Prover (System Description)"
  author:
    - de Moura, Leonardo
    - Kong, Soonho
    - Avigad, Jeremy
    - van Doorn, Floris
    - von Raumer, Jakob
  date: 2015
  parent:
    type: Proceedings
    title: "Automated Deduction - CADE-25"
    editor: Felty, Amy P.
    publisher: Springer
    volume: 9195
  page-range: 378-388
  doi: 10.1007/978-3-319-21401-6_26
  serial-number:
    isbn: 978-3-319-21401-6

# Lean 4 system description
moura2021lean4:
  type: Conference
  title: "The Lean 4 Theorem Prover and Programming Language"
  author:
    - de Moura, Leonardo
    - Ullrich, Sebastian
  date: 2021
  parent:
    type: Proceedings
    title: "Automated Deduction - CADE-28"
    publisher: Springer
    volume: 12699
  page-range: 625-635
  doi: 10.1007/978-3-030-79876-5_37

# Mathlib library
mathlib2020:
  type: Conference
  title: "The Lean Mathematical Library"
  author:
    - The mathlib Community
  date: 2020
  parent:
    type: Proceedings
    title: "Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs"
  page-range: 367-381
  doi: 10.1145/3372885.3373824

# Technical report
siegelmann1995computational:
  type: Report
  title: On the computational power of neural nets
  author:
    - Siegelmann, Hava T.
    - Sontag, Eduardo D.
  date: 1995
  parent:
    type: Organization
    title: Journal of Computer and System Sciences
    volume: 50
    issue: 1
  page-range: 132-150
  doi: 10.1006/jcss.1995.1013

# This repository (self-citation)
elmanproofs2025:
  type: Repository
  title: "ElmanProofs: Formal verification of Elman network expressivity"
  author: Gonzalez, Erik
  date: 2025
  publisher: GitHub
  url: https://github.com/yourusername/elman-proofs
  license: Apache-2.0
```

## Special Cases

### Citing Lean Proofs

For referencing your own Lean formalizations:

```yaml
linear-limitations-proof:
  type: Manuscript
  title: "Formal proof: Linear RNNs cannot compute XOR"
  author: Your Name
  date: 2025
  parent:
    type: Repository
    title: ElmanProofs
    publisher: GitHub
  url: https://github.com/yourusername/elman-proofs/blob/main/ElmanProofs/Expressivity/LinearLimitations.lean
```

### Citing GitHub Repositories

Use `Repository` type with these fields:
- `type: Repository`
- `publisher: GitHub` (or GitLab, Bitbucket, etc.)
- `url`: Full repository URL
- `version`: Release version or commit hash (optional)
- `date`: Latest commit date or release date

### Citing arXiv Papers

Use `serial-number` with `arxiv` key:

```yaml
example-arxiv:
  type: Manuscript
  title: Paper Title
  author: Author Name
  date: 2024-01-15
  serial-number:
    arxiv: "2401.12345"
  url: https://arxiv.org/abs/2401.12345
```

### Multiple Identifiers

```yaml
example-multi-id:
  type: Article
  title: Example Paper
  author: Smith, John
  date: 2023
  serial-number:
    doi: 10.1234/example
    arxiv: "2301.12345"
    pmid: "12345678"
```

## Converting from BibTeX

Hayagriva can read `.bib` files directly, but to convert to YAML:

```bash
# Install Hayagriva CLI
cargo install hayagriva-cli

# Convert BibTeX to YAML
hayagriva convert references.bib references.yml
```

## Best Practices for CS/ML Papers

1. **Use DOIs when available**: Makes papers easier to find
2. **Include arXiv IDs**: Many ML papers appear on arXiv first
3. **Cite specific versions**: For software and preprints, specify version/date
4. **Use URLs for online resources**: Include access dates for web content
5. **Follow venue conventions**: Check if conference/journal has preferred citation style
6. **Cite software properly**: Use CITATION.cff if repository provides one

## Integration with Typst Document

Example `main.typ` structure:

```typst
#import "template.typ": *

#show: project.with(
  title: "Expressivity Separation: Elman vs. Linear RNNs",
  authors: (
    "Your Name",
  ),
)

= Introduction

Recurrent neural networks @elman1990finding have long been...

= Background

Modern architectures @vaswani2017attention have largely replaced...

= Formal Verification

We formalize these results in Lean 4 @moura2021lean4, building
on the Mathlib library @mathlib2020.

// At the end
#bibliography("references.yml", style: "ieee")
```

## Resources

### Official Documentation
- [Typst Bibliography Function](https://typst.app/docs/reference/model/bibliography/)
- [Typst Cite Function](https://typst.app/docs/reference/model/cite/)
- [Hayagriva GitHub](https://github.com/typst/hayagriva)
- [Hayagriva File Format Spec](https://github.com/typst/hayagriva/blob/main/docs/file-format.md)

### CSL Styles
- [Citation Style Language Repository](https://github.com/citation-style-language/styles)
- [Zotero Style Repository](https://www.zotero.org/styles) (search for styles)

### Citation Guidelines
- [GitHub CITATION.cff Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- [Lean Community Citations](https://leanprover-community.github.io/papers.html)

## Summary

- **Format**: Use `.yml` (native, recommended) or `.bib` (compatibility)
- **Citation styles**: Built-in IEEE/APA/Chicago or custom CSL files
- **Software/repos**: Use `Repository` type with GitHub as publisher
- **arXiv**: Use `serial-number.arxiv` field
- **Lean proofs**: Cite as `Repository` or `Manuscript` with GitHub URL
- **Integration**: `#bibliography("file.yml")` and `@key` for citations

For this project, I recommend:
1. Create `docs/expressivity/references.yml` using the example above
2. Use IEEE style: `#bibliography("references.yml", style: "ieee")`
3. Add entries for: Elman (1990), key RNN papers, Lean 4, Mathlib, and your own repository
