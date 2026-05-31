# ferray.ma `.data` shared-buffer storage model — scope & BUILD-vs-document decision

<!--
tier: 3-component
status: draft
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

This is a SCOPING doc (not an implementation contract): it determines whether
`fr.ma.MaskedArray` can support numpy's `.data`-BUFFER-ALIASING semantics —
where `b = ma.array(a, mask=m); b.data[i] = x` mutates `a`'s shared `_data`
buffer — and at what cost/risk, grounding a BUILD-vs-document decision for
blockers #896 and #898.

The investigation is conclusive. pyo3-numpy 0.28 DOES offer a safe zero-copy
Rust↔numpy shared-mutable-buffer path (a `Py<PyArray>` whose buffer numpy owns,
read by Rust through `.readonly().as_array()`). But adopting it as the
`MaskedArray` storage model is a multi-session rewrite of the entire ferray-ma
data plane disproportionate to the single niche `.data[i] = x` semantic — every
common op already works through `__setitem__`. **Recommendation: (B) DOCUMENTED
LIMIT.**

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
| REQ-3 (numpy-backed rewrite magnitude) | NOT-STARTED | open prereq blocker #896. `ferray_ma::MaskedArray<T,D>` (`ferray-ma/src/masked_array.rs`, `pub struct MaskedArray`) hard-owns `data: Array<T, D>` (owned `ndarray` Vec). The entire ferray-ma data plane (`constructors.rs`, `ufunc_support.rs`, `sorting.rs`, `filled.rs`, `manipulation.rs`, reductions, `put.rs`) is built on `&Array<T,D>` access. Replacing it with a numpy-backed buffer means changing `pub const fn data(&self) -> &Array<T, D>` and every consumer to read via a numpy view — `~172` `getdata`/`.data()`/`snapshot(` call-sites in `ma.rs` alone, plus the whole ferray-ma crate. RIPPLES into ferray-ma (the owning crate, per goal.md). Risk: HIGH — touches #853 (DynMa dtype enum), #857/#880/#881/#882/#883 (view overlays — every overlay reads `snapshot`), #858/#873 (reductions over native variant), #884 (repr via `as_numpy_ma`). Multi-session, regression surface = the entire `numpy.ma` Python surface. |
| REQ-4 (lazy shared-`.data`-cache) | NOT-STARTED | open prereq blocker #896. Keeping `DynMa` as source-of-truth and caching a `Py<PyArray>` that `.data` returns by reference is FUNDAMENTALLY CIRCULAR for the WRITE direction. The cached numpy array would need to share the SAME bytes as the `DynMa`'s `ndarray::Array` so a Python `.data[i] = x` writes those bytes. The only construction that aliases is `from_owned_array` — which MOVES the `Vec` out of the `DynMa` (REQ-2), so `DynMa` no longer owns it (two owners of one buffer is exactly what Rust forbids). A non-aliasing cache (rebuild-on-write-sync) cannot detect a Python-side `.data[i] = x` — numpy gives no write-callback — so the sync is impossible without polling/diffing every access. NOT VIABLE without `unsafe` raw-pointer aliasing (R-CODE-1) or interior mutability over a shared buffer (R-APG-1). |
| REQ-5 (#898 falls out of same model) | SHIPPED (diagnosis: YES, subsumed) | With a numpy-backed `Storage::Owned` holding `Py<PyArrayDyn<T>>`, `fr.ma.array(np_ndarray, copy=False)` becomes trivial: retain the input `Py<PyArray>` directly as the buffer (no `from_owned_array` move needed — the foreign array IS the buffer). The pin `test_ctor_ndarray_is_view_arch_limit_898` (`ferray-python/tests/test_divergence_ma_final_sweep.py`) — `fm.data[0] = 77; assert fnd[0] == 77` — is satisfied by the SAME model as #896. So #898 is NOT a separate build; it is a corollary of the #896 numpy-backed model. Both stand or fall together. |
| REQ-6 (recommendation) | SHIPPED (call: B, documented limit) | See "Recommendation" below. The numpy-backed model (option A) is technically SAFE-feasible (REQ-2) but its cost (REQ-3: whole-crate rewrite, HIGH regression risk across ~10 prior shipped blockers) is disproportionate to the niche `.data[i] = x` raw-buffer-aliasing semantic — which is the ONLY thing that differs. All common operations (`__setitem__`, indexing, arithmetic, reductions, masking, repr, `.data` READS) already match numpy. Recommend (B). |

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

## Recommendation

**(B) DOCUMENTED LIMIT.** `.data`-buffer-aliasing is intractable in ferray's
current architecture without either (1) the full numpy-backed-storage rewrite —
a multi-session reconstruction of the ferray-ma data plane (~172 ma.rs sites +
the whole library crate) with HIGH regression risk across ~10 already-shipped
masked-array blockers — or (2) `unsafe` raw-pointer aliasing (R-CODE-1) /
interior mutability over a shared buffer (R-APG-1), both forbidden.

The cost is disproportionate to the single niche semantic it buys: raw
`.data[i] = x` write-through. ALL common masked-array operations already match
numpy; only mutation of the raw `.data` egress object differs. `b[i] = x` (the
idiomatic in-place write) already propagates correctly via #857.

#898 (foreign-ndarray view) is the SAME limit — it is subsumed by the same
numpy-backed model, so it is documented under the same decision, not a separate
build.

**If option A is ever revisited**, the scope is: convert `Storage::Owned` to
hold `Py<PyArrayDyn<T>>` (per-dtype via the existing `match_ma!`/`DynMa`
machinery) with numpy as the buffer owner; rewrite `fma::getdata`/`.data()`
consumers to read via `.readonly().as_array()`; keep the mask as a separate
buffer. Biggest risk: the dynamic borrow checker (`src/borrow/mod.rs`) PANICS on
overlapping borrows — a reduction holding a read view while a write view is open
would crash, so every op must scope its borrows disjointly (a pervasive
invariant across the whole crate). That risk, plus the ferray-ma ripple, is why
B is the call now.

## Verification

This is a scoping doc; its claims are verified by source inspection, not a test
gauntlet. The grounding commands:

- `grep -n "fn data_pyarray\|fn dynma_data_pyarray\|into_pyarray" ferray-python/src/ma.rs`
  — confirms the copy-based `.data` chain (REQ-1).
- `grep -c "fma::getdata\|\.data()\|snapshot(" ferray-python/src/ma.rs` → 172
  — the DynMa read-site count driving the rewrite estimate (REQ-3).
- `grep -n "pub struct MaskedArray\|data: Array<T, D>" ferray-ma/src/masked_array.rs`
  — confirms the owned-`Vec` storage that forces the ripple (REQ-3).
- `sed -n '407,419p' ~/.cargo/registry/.../numpy-0.28.0/src/array.rs` and
  `src/slice_container.rs` `From<Vec<T>>` — confirms `from_owned_array` MOVES the
  Vec (numpy owns it; Rust loses the handle), the pivotal REQ-2 finding.
- `~/.cargo/registry/.../numpy-0.28.0/src/borrow/mod.rs:1-7` — confirms the safe
  borrow-checked `as_array`/`as_array_mut` views (REQ-2).
- The three failing pins in
  `ferray-python/tests/test_divergence_ma_final_sweep.py`
  (`test_ctor_mask_kwarg_is_view_binding_896`,
  `test_ctor_fill_value_kwarg_is_view_binding_896`,
  `test_ctor_ndarray_is_view_arch_limit_898`) define the exact behavior under
  decision (B).

These pins remain RED under decision (B) and document the limit; #896 and #898
stay OPEN (not closed) as the recorded architectural limit.
