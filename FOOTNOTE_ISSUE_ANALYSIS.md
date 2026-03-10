# Footnote Issue Analysis - The REAL Problem

## The Actual Problem

Looking at the compiled PDF (`expressivity-analysis.pdf` page 6), the footnote markers **DO appear on their own lines** after theorem statements:

```
(Test Theorem 1 0.1): This is a theorem statement with no space before
footnote.

1
```

The superscript "1" should appear inline at the end of "footnote." but instead appears on a new line.

## Why This Happens

The issue is that the `#theorem()` macro from ctheorems creates a **block-level element**, and when you place `]#leanref(...)` immediately after, Typst treats the footnote as coming after the block ends, which causes a line break.

## The Real Fixes (Tested Approaches)

### What DOESN'T Work
1. ✗ Removing space: `]#leanref(...)` - Still breaks to new line
2. ✗ Adding space: `] #leanref(...)` - Still breaks to new line
3. ✗ `#h(0pt)#leanref(...)` - Still breaks to new line
4. ✗ `#box[#leanref(...)]` - Still breaks to new line

### What MIGHT Work

#### Option 1: Inline Footnote Inside Theorem Body
```typst
#theorem("Linear State as Weighted Sum")[
  For a linear RNN with matrices $A, B$, the state at time $T$ is:
  $ h_T = sum_(t=0)^(T-1) A^(T-1-t) B x_t $#leanref("LinearCapacity.lean:71", "theorem linear_state_is_sum")
]
```

Move the footnote **inside** the theorem body, at the very end.

#### Option 2: Modify the leanref Macro
```typst
#let leanref(location, signature) = {
  // Use box with baseline adjustment to force inline
  box(baseline: 0.5em)[#super[#footnote[Lean formalization: #raw(location, lang: none). See #raw(signature, lang: "lean4").]]]
}
```

#### Option 3: Use Custom Theorem Wrapper
```typst
#let theorem_with_ref(title, body, ref) = {
  theorem-base(title: title)[#body#super[#footnote[#ref]]]
}
```

#### Option 4: Attach to Last Math Element
If theorems end with equations, attach the footnote to the equation:
```typst
#theorem("Linear State")[
  The state is:
  $ h_T = sum_(t=0)^(T-1) A^(T-1-t) B x_t $#h(0pt)#leanref(...)
]
```

## Root Cause

The ctheorems `thmbox()` creates a **block container**. When content follows the closing `]`, Typst's layout algorithm treats it as a new paragraph/line. This is standard block vs inline behavior.

## Recommended Solution

**Option 1 is cleanest**: Move footnotes inside the theorem body. This requires editing all theorem statements in `appendix-proofs.typ`:

### Before:
```typst
#theorem("Linear State as Weighted Sum")[
  For a linear RNN...
  $ h_T = sum... $
]#leanref("LinearCapacity.lean:71", "theorem linear_state_is_sum")
```

### After:
```typst
#theorem("Linear State as Weighted Sum")[
  For a linear RNN...
  $ h_T = sum... $#leanref("LinearCapacity.lean:71", "theorem linear_state_is_sum")
]
```

The footnote marker will then appear immediately after the last character inside the theorem body.

## Why Previous "Fixes" Failed

The previous agent claimed it was fixed because:
1. The PDF compiled without errors ✓
2. Footnotes appeared at bottom of page ✓
3. Footnote markers had proper superscript formatting ✓

But **visually checking** reveals the markers are on separate lines, which is the actual bug.

## Action Required

1. Edit `docs/expressivity/appendix-proofs.typ`
2. Move ALL `#leanref()` calls from outside `]` to inside, at the end of the theorem body
3. Recompile and visually verify footnote markers are inline
4. Do the same for ALL .typ files using leanref/leanfile

## Files to Fix

```bash
grep -r "]#leanref\|]#leanfile" docs/expressivity/*.typ
```

Every occurrence needs the footnote moved inside the closing bracket.
