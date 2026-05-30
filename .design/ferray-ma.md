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
- REQ-4: `ma.mean()`, `ma.sum()`, `ma.min()`, `ma.max()`, `ma.var()`, `ma.std()`, `ma.count()` — all skip masked elements. **All-masked semantics** match `numpy.ma` (`numpy/ma/core.py:5249` `elif newmask: result = masked`): an all-masked `sum()` is NOT the additive identity `0.0` but the `masked` singleton; ferray has no Python `masked` and surfaces this as `NaN` (consistent with `mean`/`var`/`std`).
- REQ-5: `ma.filled(fill_value)` — return a regular array with masked positions replaced by fill_value. The **default** `fill_value` (used by `filled_default` / arithmetic / ufunc result-masking) follows numpy's per-dtype `default_filler` (`numpy/ma/core.py:163`): `1e20` for float, `999999` for integer, `true` for bool — NOT `T::zero()`.
- REQ-6: `ma.compressed()` — return a 1D array of unmasked elements only

### REQ status (this iteration — numpy.ma fill-value + all-masked semantics)
- REQ-4 SHIPPED — all-masked `sum` returns NaN (not 0.0) via `sum` in `reductions.rs`; consumer: pinned divergence `divergence_all_masked_sum_is_masked_not_zero` + `ma.sum` is called by the `MaskAware`/ufunc reduction surface. Cite `numpy/ma/core.py:5249`.
- REQ-5 SHIPPED — per-dtype default fill via `default_fill_value` in `masked_array.rs`, wired into `MaskedArray::new` / `from_data`; non-test production consumers: `filled_default` in `filled.rs`, masked-position filling in `masked_unary`/`masked_binary` (`ufunc_support.rs`) and `ma_apply_unary` (`interop.rs`). Cite `numpy/ma/core.py:163`.

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

### Specialized Masked Algorithms (numpy/ma/core.py, numpy/ma/extras.py)
- REQ-18: `ma_where(condition, x, y)` — masked select (`numpy.ma.where`). `data = where(filled(cond,False), x, y)`; result masked where the chosen source is masked OR the condition is masked. SHIPPED — `ma_where` in `algorithms.rs`; consumer `ferray-python/src/ma.rs::where_`.
- REQ-19: `ma_choose(indices, choices)` — masked choose (`numpy.ma.choose`). Masked index filled with 0; result masked where the chosen source OR the index is masked. SHIPPED — `ma_choose` in `algorithms.rs`; consumer `ferray-python/src/ma.rs::choose`.
- REQ-20: `ma_diff(a, n, axis)` — masked adjacent difference (`numpy.ma.diff`). `out[i]=a[i+1]-a[i]` masked iff either operand is masked; recurses for `n>1`. SHIPPED — `ma_diff` in `algorithms.rs`; consumer `ferray-python/src/ma.rs::diff`.
- REQ-21: `ma_ediff1d(ary, to_begin, to_end)` — flattened first difference (`numpy.ma.ediff1d`); `to_begin`/`to_end` always unmasked. SHIPPED — `ma_ediff1d` in `algorithms.rs`; consumer `ferray-python/src/ma.rs::ediff1d`.
- REQ-22: `ma_nonzero(a)` — indices of non-zero UNMASKED elements (`MaskedArray.nonzero` = `filled(self,0).nonzero()`). SHIPPED — `ma_nonzero` in `algorithms.rs`; consumer `ferray-python/src/ma.rs::nonzero`.
- REQ-23: `ma_vander` matches `numpy.ma.vander` — masked inputs fill the data with 0 and the result carries NO mask (nomask). SHIPPED — `ma_vander` in `extras.rs`; consumer `ferray-python/src/ma.rs::vander`.
- REQ-24: `ma_isin`/`ma_in1d` match `numpy.ma.isin`/`in1d` — membership over the underlying data, result carries NO mask (nomask). SHIPPED — `ma_isin`/`ma_in1d` in `extras.rs`; consumers `ferray-python/src/ma.rs::isin`/`in1d`.

### REQ status (this iteration — numpy.ma set ops + row/col suppression + masked cov/corrcoef, refs #835)
- REQ-25: masked set ops `numpy.ma.intersect1d`/`union1d`/`setdiff1d`/`setxor1d` (`numpy/ma/extras.py:1317`/`:1463`/`:1485`/`:1350`) operate on the unique unmasked values, with a single trailing **masked** slot whose survival follows numpy's `unique`-based composition (masked values considered equal). SHIPPED — `ma_intersect1d`/`ma_union1d`/`ma_setdiff1d`/`ma_setxor1d` in `extras.rs` (built on `ma_unique_masked`); non-test production consumers `ferray-python/src/ma.rs::intersect1d`/`union1d`/`setdiff1d`/`setxor1d`.
- REQ-26: `numpy.ma.unique` masked-slot contract — sorted unique unmasked values plus ONE trailing masked element iff any input was masked (`numpy/ma/extras.py:1267`). SHIPPED — `ma_unique_masked` in `extras.rs`; consumer: the set ops above (`ma_union1d` etc.) compose over it.
- REQ-27: `numpy.ma.compress_rowcols`/`compress_rows`/`compress_cols` (`numpy/ma/extras.py:920`/`:953`/`:991`) suppress whole rows/cols containing any masked value of a 2-D masked array, returning a plain (unmasked) array. SHIPPED — `ma_compress_rowcols`/`ma_compress_rows`/`ma_compress_cols` in `extras.rs`; non-test production consumers `ferray-python/src/ma.rs::compress_rowcols`/`compress_rows`/`compress_cols`.
- REQ-28: `numpy.ma.mask_rowcols` (`numpy/ma/extras.py:830`) masks whole rows/cols of a 2-D masked array that contain a masked value. SHIPPED — `ma_mask_rowcols` in `extras.rs`; non-test production consumer `ferray-python/src/ma.rs::mask_rowcols`.
- REQ-29: masked covariance/correlation `numpy.ma.cov`/`corrcoef` for the `y=None` case (`numpy/ma/extras.py:1547`/`:1672`): each variable centered by its own masked mean, pairwise normalised by the count of jointly-unmasked observations (`fact = notmask·notmaskᵀ − ddof`), entry masked where `fact ≤ 0`; corrcoef divides by the outer product of the per-variable stddevs. SHIPPED — `ma_cov`/`ma_corrcoef` in `extras.rs`; non-test production consumers `ferray-python/src/ma.rs::cov`/`corrcoef`. The two-variable-set `y` argument remains a ferray-ma library gap (binding raises a clear `ValueError`).

### REQ status (this iteration — numpy.ma mask-structure run-length + 2-D dot + multi-axis argsort, refs #835)
- REQ-14 (argsort axis) SHIPPED — multi-axis `argsort_axis` in `extras.rs` fills masked positions to the per-lane maximum (`endwith=True`) so they trail, then per-lane stable argsort (`numpy.ma.argsort`, `numpy/ma/core.py`); non-test production consumer `ferray-python/src/ma.rs::argsort` (dispatches `axis=None` to the flat `argsort`, an explicit axis to `argsort_axis`).
- REQ-30: `numpy.ma.clump_masked`/`clump_unmasked` (`numpy/ma/extras.py:2189`/`:2235`) — half-open `(start,stop)` slices of contiguous masked / unmasked runs of a 1-D masked array (numpy's `_ezclump`, `numpy/ma/extras.py:2105`). SHIPPED — `clump_masked`/`clump_unmasked` in `extras.rs`; non-test production consumers `ferray-python/src/ma.rs::clump_masked`/`clump_unmasked` (convert pairs to Python `slice(start,stop)`).
- REQ-31: `numpy.ma.flatnotmasked_contiguous`/`flatnotmasked_edges` (`numpy/ma/extras.py:1818`/`:1762`) — unmasked contiguous slices, and `[first,last]` unmasked edges (`None` if all masked). SHIPPED — `flatnotmasked_contiguous`/`flatnotmasked_edges` in `extras.rs`; non-test production consumers `ferray-python/src/ma.rs::flatnotmasked_contiguous`/`flatnotmasked_edges`.
- REQ-32: `numpy.ma.notmasked_contiguous`/`notmasked_edges` (`numpy/ma/extras.py:1936`/`:1878`) — `axis=None` delegates to the flat forms; 2-D `notmasked_contiguous_axis` returns one slice-list per lane along the orthogonal axis. SHIPPED — `notmasked_contiguous_axis`/`notmasked_edges` in `extras.rs`; non-test production consumers `ferray-python/src/ma.rs::notmasked_contiguous`/`notmasked_edges`. The multi-axis coordinate-tuple form of `notmasked_edges` remains a ferray-ma gap (binding raises a clear `ValueError`).
- REQ-33: 2-D `numpy.ma.dot(a, b)`, non-strict default (`numpy/ma/core.py:8214`) — data `dot(filled(a,0), filled(b,0))`, result `out[i,j]` masked iff every contributing `k` has `a[i,k]` OR `b[k,j]` masked (`m = ~dot(~mask_a, ~mask_b)`). SHIPPED — `ma_dot_2d` on `MaskedArray<T, Ix2>` in `extras.rs`; non-test production consumer `ferray-python/src/ma.rs::dot` (1-D returns the scalar inner product; 2-D returns the masked matrix product).

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
