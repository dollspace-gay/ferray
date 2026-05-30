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

### Stateful mask: harden/soften/put/putmask (#835)

The final four `numpy.ma` callables ferray lacks are the only ones that **mutate
the masked array in place** and/or depend on its mutable hard-mask flag. They
cannot be delegated like the structured-mask batch: `harden_mask`/`soften_mask`
flip per-array state, and `put`/`putmask` perform in-place index/mask assignment
whose behavior is *gated by* that state. This section adds them as REQ-34..REQ-37.

**numpy semantics (quoted upstream):**

- `harden_mask(self)` (`numpy/ma/core.py:3620`): body is `self._hardmask = True; return self`. `soften_mask(self)` (`:3638`): `self._hardmask = False; return self`. Module-level `numpy.ma.harden_mask(a)`/`soften_mask(a)` are not separate functions in 2.4.x — they are exposed via `_frommethod` wrappers that call the method and return the array; the observable contract is "flip the flag, return the (same, mutated) array".
- The hard-mask **effect on assignment** lives in `__setmask__` (`numpy/ma/core.py:3531`: `if self._hardmask: current_mask |= mask`) and in `__setitem__` (`:3450` `elif not self._hardmask:` sets data+mask; `:3461`-`3472` the hard branch only writes data where `~mindx`, never clearing the mask). i.e. when hard, masked positions are *protected*: assignment cannot unmask them and does not overwrite their data.
- `MaskedArray.put(self, indices, values, mode='raise')` (`numpy/ma/core.py:4837`): `self._data.flat[n] = values[n]` (values repeat if shorter); a masked `values` updates the mask, else the written positions are unmasked. The hard-mask gate (`:4899`): `if self._hardmask and self._mask is not nomask:` drops every `(index, value)` whose target position is currently masked **before** the data write — hard-masked slots are neither overwritten nor unmasked.
- `numpy.ma.put(a, indices, values, mode='raise')` (`numpy/ma/core.py:7496`) is `return a.put(indices, values, mode=mode)` — pure delegation to the method.
- `numpy.ma.putmask(a, mask, values)` (`numpy/ma/core.py:7537`): `np.copyto(a._data, valdata, where=mask)` for the data; the mask update branches on hard-mask (`:7581` `elif a._hardmask:` unions `valmask` into the existing mask only where `mask` is True, never clearing) vs soft (`:7586` copies `valmask` where `mask`). `values` is cycled to the array shape (numpy.putmask semantics). A masked-array `values` does NOT turn an `ndarray` `a` into a MaskedArray (binding note only).

**Central data-model decision.** The ferray-ma library data model needs **NO new field**: `MaskedArray` already carries `pub(crate) hard_mask: bool` (`masked_array.rs`, the struct def) plus `harden_mask`/`soften_mask` (`mask_ops.rs`) that flip it, `is_hard_mask` to read it, and the hard-mask *assignment effect* is already implemented in `set_mask_flat` (`masked_array.rs`: `if self.hard_mask && !value { return Ok(()) }`) and `set_mask` (the hard-mask union branch). The only **net-new library work** is two flat in-place setters that respect that existing flag: a `put` (flat-indexed data+mask assignment) and a `putmask` (boolean-mask-gated data+mask assignment). Both build directly on the existing in-place mutation surface — `data_mut() -> Option<&mut [T]>`, `set_mask_flat`, and `ensure_materialized_mut`/`set_mask` — so this is roughly **80-140 LOC of library code** in a new `put.rs` (or appended to `mask_ops.rs`) plus unit tests, NOT a new dtype model and NOT a new field. The f64-only `PyMaskedArray` model is untouched at the data-model level.

**Binding-shim work (the larger share).** `PyMaskedArray` is `#[pyclass(... from_py_object)] #[derive(Clone)]` with every method taking `&self` (`ferray-python/src/ma.rs`, the `#[pymethods] impl PyMaskedArray` block). harden/soften/put/putmask MUTATE, so they need `&mut self` methods on the pyclass (PyO3 supports `&mut self` receivers on a non-frozen pyclass; `PyMaskedArray` is not declared `frozen`, so this is available). Module-level `ma.harden_mask(a)`/`soften_mask(a)`/`put(a, ...)`/`putmask(a, ...)` are thin `#[pyfunction]`s that call the method on the passed `PyMaskedArray` and return it (matching numpy returning the mutated array for harden/soften; `put`/`putmask` return `None` like numpy). These four `#[pyfunction]`s plus the four `#[pymethods]` register in `register_ma_module` in `ferray-python/src/lib.rs`. This is roughly **150-220 LOC of binding code** (the in-place index/value/mask coercion across the PyO3 boundary, mode='raise'/'wrap'/'clip' handling for `put`, and `values` cycling for `putmask` are the fiddly parts).

**Library REQ vs binding-shim REQ split.** REQ-34/35 (harden/soften) are **already SHIPPED at the library level** (the flag + its assignment effect) — their only gap is the binding exposure. REQ-36/37 (put/putmask) are **net-new at both levels**: the library lacks the two in-place setters and the binding lacks the four bindings.

**File manifest a future acto-builder needs (≤8 files):**
- `ferray-ma/src/put.rs` (new) — `MaskedArray::put(&mut self, indices, values, mode)` and `MaskedArray::putmask(&mut self, mask, values)`, both gating on `self.hard_mask`; built on `data_mut`/`set_mask_flat`/`ensure_materialized_mut`. Plus `#[cfg(test)]` unit tests (R-CHAR-3: expected values from a live `numpy.ma` call).
- `ferray-ma/src/lib.rs` — `mod put;` and any re-export.
- `ferray-python/src/ma.rs` — four `&mut self` `#[pymethods]` (`harden_mask`/`soften_mask`/`put`/`putmask`) + four module-level `#[pyfunction]`s.
- `ferray-python/src/lib.rs` — register the four `#[pyfunction]`s in `register_ma_module`.
- `ferray-python/tests/test_expansion_ma_batch5.py` (new) — pytest comparing `ferray.ma` vs `numpy.ma` for the hard/soft assignment-suppression behavior and put/putmask data+mask outcomes (the subtle hard-mask interaction is the test focus).

**Existing in-place mutation path to build on (quoted).** ferray-ma DOES already have a real in-place mutation surface — this is not net-new plumbing. `MaskedArray::data_mut(&mut self) -> Option<&mut [T]>` (`masked_array.rs`) exposes the contiguous data slice for direct write, and `MaskedArray::set_mask_flat(&mut self, flat_idx, value)` (`masked_array.rs`) already implements the exact hard-mask suppression `put` needs: `if self.hard_mask && !value { return Ok(()) }`. `put`/`putmask` compose these; the architectural risk is low.

### REQ status (this iteration — stateful mask harden/soften/put/putmask, #835)

| REQ | Status | Evidence |
|---|---|---|
| REQ-34 (harden_mask) | SHIPPED | Library: `harden_mask` in `mask_ops.rs` flips `hard_mask`, its effect enforced by `set_mask_flat`/`set_mask` in `masked_array.rs` (mirrors `numpy/ma/core.py:3635`). Binding: `PyMaskedArray::harden_mask` (`&mut self` method) + the `hardmask` getter in `ferray-python/src/ma.rs`, plus the module-level `harden_mask` `#[pyfunction]` (returns the same object). Non-test consumer: `register_ma_module` in `ferray-python/src/lib.rs` registers `ma::harden_mask`; pytest `test_harden_mask_returns_same_object` confirms the in-place flag flip sticks on the caller. |
| REQ-35 (soften_mask) | SHIPPED | Library: `soften_mask` in `mask_ops.rs` (mirrors `numpy/ma/core.py:3653`). Binding: `PyMaskedArray::soften_mask` (`&mut self`) + module-level `soften_mask` `#[pyfunction]` in `ferray-python/src/ma.rs`. Non-test consumer: registered in `register_ma_module` (`ferray-python/src/lib.rs`); once soft, `set_mask_flat` permits clearing (pytest `test_soften_then_put_clears_mask_like_numpy`). |
| REQ-36 (put) | SHIPPED | Library: `MaskedArray::put` in `put.rs` composes `data_mut` + `set_mask_flat` honouring the hard-mask drop (`numpy/ma/core.py:4899`), values-repeat, masked-`values` propagation, and `PutMode` raise/wrap/clip (`numpy/ma/core.py:4837`). Binding: `PyMaskedArray::put` (`&mut self`) + module-level `put` `#[pyfunction]` in `ferray-python/src/ma.rs` (out-of-bounds raise → `IndexError` via `put_err_to_pyerr`). Non-test consumer: registered in `register_ma_module` (`ferray-python/src/lib.rs`). |
| REQ-37 (putmask) | SHIPPED | Library: `MaskedArray::putmask` in `put.rs` mirrors `numpy/ma/core.py:7537` — data `copyto(where=mask)` always (even hard-masked positions), mask branch on `hard_mask` (`:7581`), `values` broadcast (scalar or same-length) like numpy's `copyto`. Binding: `PyMaskedArray::putmask` (`&mut self`) + module-level `putmask` `#[pyfunction]` in `ferray-python/src/ma.rs`. Non-test consumer: registered in `register_ma_module` (`ferray-python/src/lib.rs`). |

REQ-15 (the original library-only `harden_mask`/`soften_mask` requirement) remains SHIPPED at the library level via `mask_ops.rs`; REQ-34/35 above extend it to the *binding-exposed* in-place contract, now SHIPPED. With REQ-34..37 SHIPPED, `ferray.ma` exposes all 219 numpy.ma callables (the sole excluded name is `np.ma.test`, numpy's own pytest entry point).

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
