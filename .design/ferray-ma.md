# Feature: ferray-ma — Masked arrays for missing and invalid data

## Summary
Implements `numpy.ma`: masked arrays that represent missing or invalid data inline with array data. `MaskedArray` pairs a data array with a boolean mask array. All operations (arithmetic, reductions, comparisons, ufuncs) respect the mask, skipping masked elements. Provides masking constructors for NaN, Inf, equality, and comparison conditions, plus mask manipulation utilities.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, Element, Dimension, FerrayError), `ferray-ufunc` (elementwise ops respect masks), `ferray-stats` (masked reductions)
- **Downstream**: ferray (re-export)
- **Phase**: 3 — Completeness

## Requirements

### MaskedArray Type (Section 18)
- REQ-1: `MaskedArray<T, D>` wraps a `NdArray<T, D>` (data) and a `NdArray<bool, D>` (mask, where `true` = masked/invalid)
- REQ-2: Constructor: `MaskedArray::new(data, mask)` — shapes must match
- REQ-3: `ma.data()` returns the underlying data array, `ma.mask()` returns the mask

### Masked Reductions
- REQ-4: `ma.mean()`, `ma.sum()`, `ma.min()`, `ma.max()`, `ma.var()`, `ma.std()`, `ma.count()` — all skip masked elements
- REQ-5: `ma.filled(fill_value)` — return a regular array with masked positions replaced by fill_value
- REQ-6: `ma.compressed()` — return a 1D array of unmasked elements only

### Masking Constructors
- REQ-7: `ma::masked_where(&condition, &a)` — mask where condition is true
- REQ-8: `ma::masked_invalid(&a)` — mask NaN and Inf values
- REQ-9: `ma::masked_equal(&a, value)`, `ma::masked_greater(&a, value)`, `ma::masked_less(&a, value)`, `ma::masked_not_equal(&a, value)`, `ma::masked_greater_equal(&a, value)`, `ma::masked_less_equal(&a, value)`, `ma::masked_inside(&a, v1, v2)`, `ma::masked_outside(&a, v1, v2)`

### Masked Arithmetic
- REQ-10: Binary operations between MaskedArrays produce a MaskedArray where the output mask is the union of both input masks
- REQ-11: Operations between a MaskedArray and a regular array treat the regular array as fully unmasked

### Masked Ufunc Support
- REQ-12: All ferray-ufunc elementwise operations must accept `MaskedArray` inputs and produce `MaskedArray` outputs, propagating masks correctly. Masked elements are skipped (not computed).

### Masked Sorting
- REQ-13: `ma.sort(axis)` — sort unmasked elements, masked elements move to end
- REQ-14: `ma.argsort(axis)` — return indices that sort unmasked elements

### Mask Manipulation
- REQ-15: `ma.harden_mask()` — prevent mask from being unset by assignment. `ma.soften_mask()` — allow mask to be unset.
- REQ-16: `ma::getmask(&ma)` — return mask array (or `nomask` sentinel if no mask). `ma::getdata(&ma)` — return underlying data.
- REQ-17: `ma::is_masked(&ma)` — return true if any element is masked. `ma::count_masked(&ma, axis)` — count masked elements.

## Acceptance Criteria
- [ ] AC-1: `MaskedArray::new([1,2,3,4,5], [false,false,true,false,false]).mean()` returns 3.0 (skips element 3)
- [ ] AC-2: `filled(0.0)` replaces masked elements with 0.0, leaves others unchanged
- [ ] AC-3: `compressed()` returns only unmasked elements as a 1D array
- [ ] AC-4: `masked_invalid(&[1.0, NaN, 3.0, Inf])` masks indices 1 and 3
- [ ] AC-5: `ma1 + ma2` produces correct mask union and correct values for unmasked positions
- [ ] AC-6: `cargo test -p ferray-ma` passes. `cargo clippy` clean.
- [ ] AC-7: `sin(masked_array)` returns a MaskedArray with same mask, correct values for unmasked elements
- [ ] AC-8: `ma.sort(axis=0)` places masked elements at end. `harden_mask()` prevents subsequent assignments from clearing mask bits.
- [ ] AC-9: `is_masked(&ma)` returns true when any element is masked, false for fully unmasked arrays

## Architecture

### Crate Layout
```
ferray-ma/
  Cargo.toml
  src/
    lib.rs
    masked_array.rs           # MaskedArray<T, D> type
    reductions.rs             # Masked mean, sum, min, max, var, std, count
    constructors.rs           # masked_where, masked_invalid, masked_equal, etc.
    arithmetic.rs             # Masked binary ops with mask union
    ufunc_support.rs          # Trait impls for ufunc integration
    sorting.rs                # Masked sort, argsort
    mask_ops.rs               # harden_mask, soften_mask, getmask, getdata, is_masked, count_masked
    filled.rs                 # filled, compressed
```

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Masked array I/O (serialize mask alongside data — post-1.0)
- Sparse mask representation (always dense bool array)
