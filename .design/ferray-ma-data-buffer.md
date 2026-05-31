# ferray.ma `.data` shared-buffer storage model — scope & BUILD-vs-document decision

<!--
tier: 3-component
status: shipped
baseline-commit: dbf7a282a4d1b3ff88d69f0f5428ff4ad75e984a
upstream-paths:
  - numpy/ma/core.py   (MaskedArray._data, .data property, __array_finalize__)
target-paths:
  - ferray-python/src/ma.rs
  - ferray-ma/src/masked_array.rs
  - ferray-ma/src/mask_ops.rs
related-blockers:
  - "#896  constructor mask=/fill_value= view sharing _data"
  - "#898  foreign-ndarray buffer view"
-->

## Summary

This doc grounded the BUILD-vs-document decision for `.data`-BUFFER-ALIASING in
`fr.ma.MaskedArray` (where `b = ma.array(a, mask=m); b.data[i] = x` mutates the
shared `_data` buffer) for blockers #896 and #898. It originally recommended
(B) DOCUMENTED LIMIT; **option (A) BUILD has since been SHIPPED** — a contained
numpy-backed data buffer for the `Storage::Owned` arm + constructor identity
views + foreign-ndarray sharing — without touching the ferray-ma library crate.

pyo3-numpy 0.28's safe zero-copy bridge (a `Py<PyArray>` numpy owns, read via
`.readonly().as_array()`, written via `.readwrite().as_array_mut()`) is the
mechanism. The feared whole-ferray-ma rewrite was AVOIDED by keeping the `DynMa`
as the mask/fill/structure carrier and shadowing only its DATA component with a
numpy-owned `DataBuf`, refreshed at the single `snapshot()` read chokepoint and
written back at the `with_buffer_mut()` / view write-through chokepoints. The
aliasing buffer is attached ONLY at the user-facing constructors (`buf: Some`);
internal op results keep the legacy copy path (`buf: None`), bounding the blast
radius. The 3 target pins are GREEN and the prior 2697-pass suite is unregressed
(2716 total with the new aliasing tests). **Status: (A) SHIPPED.**

## Requirements

- REQ-1: Determine how the CURRENT `.data` egress is built and why it cannot
  alias the Rust buffer.
- REQ-2: Determine whether pyo3-numpy 0.28 offers a SAFE (no `unsafe` transmute
  per R-CODE-1, no `RefCell`/`Arc<Mutex>` per R-APG-1) zero-copy path where a
  Python-side `.data[i] = x` write is visible to subsequent Rust reads.
- REQ-3: Estimate the magnitude (LOC / files / risk / regression surface) of a
  numpy-array-BACKED `Storage` rewrite, and whether it ripples into ferray-ma.
- REQ-4: Assess the lazy shared-`.data`-cache alternative (DynMa source-of-truth
  + cached `Py<PyArray>` with write-sync).
- REQ-5: Determine whether #898 (foreign-ndarray buffer view) falls out of the
  same numpy-backed model.
- REQ-6: Deliver a clear BUILD (A) vs DOCUMENTED-LIMIT (B) recommendation.

## Acceptance criteria

- AC-1: REQ-1 cites the exact `.data` call chain and the `into_pyarray` copy.
- AC-2: REQ-2 quotes the pyo3-numpy 0.28 API that does (or does not) permit safe
  shared mutation, with a registry-source `file:line`.
- AC-3: REQ-3 names the count of DynMa read sites and the ferray-ma struct field
  that forces the rewrite.
- AC-4: REQ-5 states whether #898 is subsumed.
- AC-5: REQ-6 is unambiguous (A or B) with the niche-impact justification.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (current `.data` is a copy) | SHIPPED (diagnosis confirmed) | `data_pyarray` in `ma.rs` calls `dynma_data_pyarray(py, &self.snapshot(py)?)`; `dynma_data_pyarray` does `let arr: ArrayD<T> = fma::getdata(m)…; Ok(arr.into_pyarray(py)…)`. `fma::getdata` (`ferray-ma/src/mask_ops.rs`, `pub fn getdata`) returns an OWNED `ArrayD<T>`; `into_pyarray` allocates a fresh numpy buffer from that owned array. The returned numpy array is a COPY with no backlink to `DynMa`, so `b.data is b.data` is False and `b.data[i] = x` writes a throwaway buffer. Write-through to the base works ONLY via `__setitem__` → `with_buffer_mut` (`b[i] = x`), never `.data[i] = x`. |
| REQ-2 (safe zero-copy path exists?) | SHIPPED (answer: YES, but inverted ownership) | `PyArray::from_owned_array` (numpy-0.28.0 `src/array.rs:407`) "uses the internal `Vec` of the `ndarray::Array` as the base object of the NumPy array" — it MOVES the `Vec` into a `PySliceContainer` (`src/slice_container.rs`, `From<Vec<T>>` wraps the buffer in `mem::ManuallyDrop`, numpy heap now owns it). Rust LOSES its `Array<T,D>` handle. The ONLY safe re-read is through a retained `Py<PyArrayDyn<T>>`: `.readonly().as_array()` (borrow-checked `ArrayView`, `src/borrow/mod.rs:7`) and `.readwrite().as_array_mut()` (`src/array.rs` `PyReadwriteArray`, dynamic-borrow-checked, no `unsafe` at the call site). So a SAFE shared-mutable buffer IS possible — but ONLY if numpy OWNS the buffer and Rust holds a `Py<PyArray>` handle and reads via a view. There is NO safe API to keep a Rust-owned `Vec` and hand numpy an aliasing view of it (that direction needs `unsafe` raw-parts + a lifetime numpy cannot enforce). |
| REQ-3 (numpy-backed rewrite magnitude) | SHIPPED (contained, NO ferray-ma change) | The feared whole-crate rewrite was avoided. `Storage::Owned` gained an optional `buf: Option<DataBuf>` (`enum DataBuf` over the 11 real dtype `Py<PyArrayDyn<T>>`, `ma.rs`) shadowing ONLY the DATA component; the `DynMa` still carries mask/fill/structure unchanged, so the ~172 read sites are untouched. `DataBuf::refresh_into` (`.readonly().as_array()`) pulls buffer→DynMa at the `snapshot()` chokepoint; `DataBuf::writeback_from` (`.readwrite().as_array_mut()`) pushes DynMa→buffer at `with_buffer_mut`, `copy_data_through` (view write-through), `store_inplace`, and the `set_mask` owned rewrap. Borrows are MOMENTARY and DISJOINT (refresh drops before f runs; writeback after) so the pyo3-numpy dynamic borrow checker never panics. ferray-ma is UNCHANGED. Consumer: the `data` getter (`PyMaskedArray::data`/`aliasing_data_pyarray`) returns the buffer by reference. |
| REQ-4 (lazy shared-`.data`-cache) | NOT-STARTED | open prereq blocker #896. Keeping `DynMa` as source-of-truth and caching a `Py<PyArray>` that `.data` returns by reference is FUNDAMENTALLY CIRCULAR for the WRITE direction. The cached numpy array would need to share the SAME bytes as the `DynMa`'s `ndarray::Array` so a Python `.data[i] = x` writes those bytes. The only construction that aliases is `from_owned_array` — which MOVES the `Vec` out of the `DynMa` (REQ-2), so `DynMa` no longer owns it (two owners of one buffer is exactly what Rust forbids). A non-aliasing cache (rebuild-on-write-sync) cannot detect a Python-side `.data[i] = x` — numpy gives no write-callback — so the sync is impossible without polling/diffing every access. NOT VIABLE without `unsafe` raw-pointer aliasing (R-CODE-1) or interior mutability over a shared buffer (R-APG-1). |
| REQ-5 (#898 falls out of same model) | SHIPPED (implemented) | `DataBuf::from_foreign` (`ma.rs`) retains the input ndarray's `Py<PyArrayDyn<T>>` directly (no move) when `copy=False`; `from_foreign_buffered`/`construct_masked` wire it. The pin `test_ctor_ndarray_is_view_arch_limit_898` (`ferray-python/tests/test_divergence_ma_final_sweep.py`) — `fm.data[0] = 77; assert fnd[0] == 77` — is GREEN, and `np.shares_memory(fm.data, fnd)` matches numpy (`test_ctor_ndarray_shares_memory`). `copy=True` correctly MOVES into an independent buffer (`from_dynma_buffered`), verified by `test_copy_true_does_not_alias_foreign`. |
| REQ-6 (recommendation) | SHIPPED (call: A, BUILT) | Option (A) BUILD shipped. The cost feared in the original (B) recommendation — a whole-ferray-ma rewrite — was avoided by the shadow-buffer containment (REQ-3): the `DynMa` stays the mask/fill carrier, only its data is numpy-backed, attached solely at user constructors. The 3 pins (`test_ctor_mask_kwarg_is_view_binding_896`, `test_ctor_fill_value_kwarg_is_view_binding_896`, `test_ctor_ndarray_is_view_arch_limit_898`) are GREEN; the prior suite is unregressed; no `unsafe`/`RefCell`/`unwrap`/`panic` added; no borrow-panic. |

## Architecture

### Current model (copy-based `.data`)

`PyMaskedArray.inner` is a `DynMa` (`ma.rs`, `pub enum DynMa`), an enum over
`RustMa<T, IxDyn>` = `ferray_ma::MaskedArray<T, IxDyn>`. `MaskedArray`
(`ferray-ma/src/masked_array.rs`) owns `data: Array<T, D>` — a Rust-heap
`ndarray::Array` (owned `Vec<T>`). `Storage::Owned { data: DynMa, .. }` /
`Storage::View { base, desc, .. }` (`ma.rs`, `enum Storage`) layer the view
overlays on top.

`.data` egress: `data_pyarray` → `snapshot` (clones `Owned`, re-gathers `View`)
→ `dynma_data_pyarray` → `fma::getdata` (owned `ArrayD<T>`) → `into_pyarray`
(fresh numpy buffer). The result is a detached COPY: it neither shares bytes
with the `DynMa` nor carries a write-back path. The #857/#880/#881 view
write-through is entirely a `__setitem__`/`with_buffer_mut` mechanism that
scatters element+mask through `set_one_dyn`/`copy_data_through`; it never
observes a Python mutation of the `.data` egress object.

### The pivotal pyo3-numpy 0.28 finding

The buffer-ownership direction is the crux. numpy-0.28.0 offers exactly one
zero-copy bridge and it runs numpy→Rust, not Rust→numpy:

- `from_owned_array` (`src/array.rs:407`): numpy adopts the Rust `Vec` (moved
  into `PySliceContainer`, `ManuallyDrop`). Rust loses the handle.
- `.readonly().as_array()` / `.readwrite().as_array_mut()`
  (`src/borrow/mod.rs`): Rust reads/writes the numpy buffer through
  borrow-checked views — SAFE, no `unsafe` at the call site.

Therefore a SAFE shared-mutable buffer requires numpy to be the SOLE owner and
the `MaskedArray` to read its data through a retained `Py<PyArray>` view. There
is no safe API to keep a Rust-owned `Array<T,D>` AND hand numpy an aliasing
mutable view of it — that inverse needs `unsafe` raw-parts (R-CODE-1) plus a
lifetime guarantee numpy's GC cannot honor.

### Why the rewrite is large

Adopting the numpy-backed model means `Storage::Owned` holds
`Py<PyArrayDyn<T>>` + mask instead of `DynMa`, and EVERY data reader switches
from `&Array<T,D>` to `py_array.readonly().as_array()`. That is `~172`
`getdata`/`.data()`/`snapshot(` sites in `ma.rs` plus the whole ferray-ma crate
(`pub const fn data(&self) -> &Array<T,D>` and all its callers across
`constructors.rs`, `ufunc_support.rs`, `sorting.rs`, `filled.rs`, `put.rs`,
reductions). It ripples into the OWNING crate (ferray-ma) — per goal.md, the
correctness lives down in the library crate. Regression surface = the entire
shipped `numpy.ma` blocker set (#853, #857, #858, #868, #873, #880–#884, #891).

### The niche being traded away

Under option B, the ONLY behavior that diverges from numpy is raw `.data`-buffer
aliasing: `b.data[i] = x` writing through to the shared base, and
`np.shares_memory(a.data, a.data)`. Concretely the three pins
`test_ctor_mask_kwarg_is_view_binding_896`,
`test_ctor_fill_value_kwarg_is_view_binding_896`, and
`test_ctor_ndarray_is_view_arch_limit_898`
(`ferray-python/tests/test_divergence_ma_final_sweep.py`). Every other path —
`b[i] = x` (`__setitem__`/`with_buffer_mut` write-through, #857), indexing,
arithmetic, reductions, masking, fill_value, repr, and `.data` READS — already
matches numpy. The aliasing of the raw `.data` egress object is a rarely-relied
NumPy implementation detail, not a load-bearing user contract.

## Decision: (A) BUILT — numpy-backed data buffer (shadow-buffer containment)

Option (A) was SHIPPED. The original (B) recommendation feared a whole-ferray-ma
rewrite; that cost was avoided by a CONTAINED design that keeps the `DynMa` as
the mask/fill/structure carrier and numpy-backs ONLY its data component, with
the buffer attached solely at user-facing constructors.

The shipped model (`ferray-python/src/ma.rs`):

- `enum DataBuf` holds a `Py<PyArrayDyn<T>>` per real dtype (numpy owns the
  buffer). `Storage::Owned` gains `buf: Option<DataBuf>`: `Some` = the numpy
  buffer is the CANONICAL data store the `.data` getter aliases; `None` = legacy
  copy path (op results, complex dtypes) — additive, low-risk.
- READ chokepoint: `snapshot()` clones the `DynMa` and `DataBuf::refresh_into`
  overlays the buffer's current contents (`.readonly().as_array()`), so a Python
  `b.data[i] = x` is visible to every reader.
- WRITE chokepoints: `with_buffer_mut` (owned arm), `copy_data_through` (view
  write-through), `store_inplace` (sort/fill), and the `set_mask` owned rewrap
  call `DataBuf::writeback_from` (`.readwrite().as_array_mut()`). Borrows are
  momentary and DISJOINT (refresh drops before the mutation; writeback after),
  so the dynamic borrow checker never panics.
- `.data` getter (`aliasing_data_pyarray`): owned-with-`buf` returns the handle
  by reference; an IDENTITY view returns `base_buffer.reshape(view_shape)`
  (numpy's aliasing view) so `view.data[i] = x` writes the base buffer (#896);
  partial/fancy views fall back to the copy egress (their write-through still
  works via `__setitem__`, #857, unchanged).
- Constructors: `from_dynma_buffered` MOVES data into a fresh numpy buffer;
  `from_foreign_buffered` retains a foreign input ndarray's buffer for
  `copy=False` (#898). `copy=True` / `dtype=` recast force a fresh independent
  buffer (no source aliasing). `ma.array(existing_ma, mask=/fill_value=)` makes
  an identity view sharing the base buffer + applies the mask (keep_mask OR) /
  fill as per-object view overlays (#896).

No `unsafe`/`RefCell`/`Arc<Mutex>` (R-CODE-1/R-APG-1); no `unwrap`/`expect`/
`panic!` in production (R-CODE-2); ferray-ma is UNCHANGED. `b[i] = x` (#857) and
all prior masked-array behavior are preserved.

## Verification

The shipped (A) build is verified by the pytest-vs-numpy gauntlet (the (B)
source-inspection grounding is retained above for the original diagnosis):

- The 3 target pins in `ferray-python/tests/test_divergence_ma_final_sweep.py`
  (`test_ctor_mask_kwarg_is_view_binding_896`,
  `test_ctor_fill_value_kwarg_is_view_binding_896`,
  `test_ctor_ndarray_is_view_arch_limit_898`) are GREEN.
- `ferray-python/tests/test_expansion_ma_data_alias.py` (NEW) covers owned
  `.data` write-through, identity-view base sharing, foreign-ndarray sharing,
  `np.shares_memory` parity, and the `copy=True` / `dtype=`-recast
  NON-aliasing cases — all derived live from numpy.ma (R-CHAR-3).
- Full suite: `pytest tests/ -q` → 2716 passed, 0 failed (2697 prior + 3 pins
  flipped + new alias tests), no borrow-panic.
- `cargo clippy -p ferray-python --all-targets -- -D warnings` clean;
  `cargo fmt --check` clean.

#896 and #898 are CLOSED by the (A) build.
