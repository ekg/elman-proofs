# Appendix Deduplication Summary

## Task: Cut Appendix Duplication

The appendix previously repeated many proofs that were already fully proven in the main text sections (§2-§4). This created unnecessary duplication and made the document longer than needed.

## Changes Made

### Removed from Appendix (Already in Main Text)

The following proofs were **removed** from the appendix because they are already fully proven in the main text:

1. **Linear RNN Limitations** (§2: 02-foundations.typ)
   - ✓ Linear State as Weighted Sum (lines 17-23)
   - ✓ Threshold Impossibility (lines 31-37)
   - ✓ XOR is Not Affine (lines 41-47)

2. **Tanh Saturation and Fixed Points** (§4: 04-e88.typ)
   - ✓ Tanh Boundedness (lines 26-32)
   - ✓ Tanh Derivative Vanishes at Saturation (lines 34-40)
   - ✓ Fixed Points (lines 44-70)
   - ✓ Latching and Binary Memory (lines 76-104)

### Kept in Appendix (NOT in Main Text)

The following content was **retained** in the appendix because it provides detailed proofs not fully covered in the main text:

1. **Running Parity** (only referenced in §3, full proof in appendix)
   - Parity is Not Affine theorem + proof
   - Linear RNNs Cannot Compute Running Parity theorem + proof
   - Multi-layer extension

2. **TC⁰ Circuit Complexity Bounds** (stated in §8, detailed proofs in appendix)
   - Linear SSM < TC⁰ proof
   - TC⁰ < E88 (unbounded T) proof
   - Hierarchy summary

3. **Advanced E88 Capabilities** (NOT covered in main text)
   - Exact Counting Modulo n theorem + proof
   - Multi-Head Independence theorem + proof
   - Alert State and Attentional Persistence theorems + proofs
   - E88 Exceeds TC⁰ for Unbounded Sequences detailed proof

## Result

The appendix now serves its proper purpose: providing detailed formal proofs for theorems that are **stated but not fully proven** in the main text. It no longer duplicates content that is already complete in the main sections.

### Appendix Structure (After Deduplication)

```
= Appendix: Formal Proofs

Introduction explaining the appendix contains proofs NOT in main text

== Running Parity
   - Detailed proofs for parity impossibility

== TC⁰ Circuit Complexity Bounds
   - Detailed complexity class proofs

== Advanced E88 Capabilities
   - Exact counting modulo n
   - Multi-head independence
   - Alert state and persistence
   - E88 exceeds TC⁰

== Conclusion
   - Summary of key results
```

## Files Modified

- `docs/expressivity/appendix-proofs.typ` - Removed duplicate proofs, kept unique content
- `docs/expressivity/main.pdf` - Regenerated with de-duplicated appendix

## Verification

✓ PDF compiles successfully
✓ Appendix contains only non-duplicate content
✓ All unique formal proofs retained
✓ Main text proofs remain intact in their original sections
