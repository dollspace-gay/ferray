# Feature: MaskedArray slice-view writeback semantics (#857)

<!--
tier: 3-component
status: draft
baseline-commit: b505ca7d
upstream-paths:
  - numpy/ma/core.py
-->

## Summary

`numpy.ma.MaskedArray.__getitem__` returns a **VIEW** for basic indexing
(int, slice, ellipsis, newaxis, multi-dimensional int/slice tuples) that shares
the parent's `_data` AND `_mask` buffers, so a write through the sub-array
propagates back to the parent (`b = a[1:3]; b[0] = 99` sets `a[1] == 99`). For
advanced/fancy indexing (integer-array, boolean-mask `a[a>1]`, list-of-indices
`a[[0,2]]`) numpy returns a **COPY** — writes do NOT propagate.

`ferray.ma.MaskedArray.__getitem__` (the `PyMaskedArray` PyO3 class in
`ferray-python/src/ma.rs`) returns a **COPY for every index kind**: it gathers
flat C-order positions and builds a fresh `RustMa` via `ArrayD::from_vec`. This
is **already CORRECT for the fancy/bool/list cases** (numpy copies too) and
**diverges only for basic-indexing slices/int-rows**, where numpy writeback is
expected but ferray loses it.

This doc scopes whether numpy view-writeback for `fr.ma.MaskedArray` is
TRACTABLE in this PyO3 binding + the ferray-ma library, and renders a clear
**build-vs-defer** decision grounded in what `ndarray 0.17` + `PyO3 0.28`
actually allow. It is a SCOPING doc — it proposes no `.rs` edits.

## Requirements

- REQ-1 (index-kind taxonomy): Document precisely which index kinds numpy
  returns as a VIEW (basic: int, slice, ellipsis, newaxis, int/slice tuples)
  versus a COPY (advanced: int-array, bool-array, list-of-indices), and record
  that ferray ALREADY matches numpy for every copy case — the only gap is
  basic-index view-writeback.
- REQ-2 (storage-model finding): Document the ferray-ma storage model
  (`MaskedArray<T,D>` owns its `data` buffer; the type is not parameterized over
  ownership) and the consequence: a `MaskedArray` cannot today be a writeback
  view into another's buffer.
- REQ-3 (PyO3 option analysis): Evaluate options A (Arc-shared buffer),
  B (base-object reference), C (`ArrayViewMut` in the pyclass), D (defer) against
  the concrete capabilities of `ndarray 0.17` (owned `Array` vs `ArcArray` CoW
  vs lifetime-bound `ArrayViewMut`) and `PyO3 0.28` (`'static` pyclass bound,
  `Py<T>` keepalive, no borrowed-lifetime erasure without `unsafe`).
- REQ-4 (decision + prerequisite): Render a build-vs-defer decision. If defer,
  name the EXACT library prerequisite (an ownership-parameterized or shared
  writeback storage model in ferray-ma) and quantify its magnitude (LOC, crates,
  sessions) so the deferral is a tracked work item, not an excuse.

## Acceptance criteria

- AC-1 (REQ-1): the doc lists each numpy index kind with VIEW/COPY classification
  traceable to `numpy/ma/core.py:3287` (`dout = self.data[indx]`) +
  `core.py:3358` (`dout = dout.view(type(self))`), and asserts ferray's current
  `Gather`-copy behavior is byte-identical to numpy for the COPY kinds (verified
  by the green `test_getitem_fancy_realmask_source_keeps_real_mask_array` /
  `test_getitem_diagonal_gather_realmask_mask_is_array` pins).
- AC-2 (REQ-2): the doc quotes the `MaskedArray<T,D>` struct
  (`data: Array<T, D>`) and the `Array<T,D>` wrapper
  (`inner: ndarray::Array<T, D::NdarrayDim>`, an OWNED repr) as evidence that
  data is owned, not shareable, and not generic over ownership.
- AC-3 (REQ-3): each of A/B/C/D carries a concrete viability verdict citing the
  `ndarray 0.17` / `PyO3 0.28` mechanism that enables or blocks it.
- AC-4 (REQ-4): the decision is binary (BUILD via a named option, or DEFER with a
  named prerequisite + magnitude). The prerequisite is filed as a tracked blocker
  referenced by `#`-number.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (index-kind taxonomy) | SHIPPED | numpy classifies in `__getitem__` (`numpy/ma/core.py:3287` `dout = self.data[indx]`; `:3358` `dout = dout.view(type(self))`): `self.data[indx]` uses `ndarray.__getitem__`, which returns a VIEW for basic indexing and a COPY for advanced. ferray's `fn __getitem__` in `ma.rs` resolves every index via `resolve_index` (`ma.rs`) into a single `ResolvedIndex::Gather { positions, shape }` and rebuilds a fresh `RustMa` with `ArrayD::<T>::from_vec(...)` — a COPY for ALL kinds. The COPY cases MATCH numpy: pins `test_getitem_fancy_realmask_source_keeps_real_mask_array` and `test_getitem_diagonal_gather_realmask_mask_is_array` (advanced indexing) are GREEN. Consumer: `PyMaskedArray::__getitem__` is the registered `#[pymethods]` dunder consumed by every `fr.ma` subscript. |
| REQ-2 (storage-model finding) | SHIPPED | `pub struct MaskedArray<T: Element, D: Dimension>` in `masked_array.rs` stores `data: Array<T, D>` — an OWNED array. `pub struct Array<T,D>` in `array/owned.rs` wraps `inner: ndarray::Array<T, D::NdarrayDim>` (ndarray's `OwnedRepr`, owns its `Vec`). The type is NOT generic over ownership; there is no `MaskedArrayView`. Consumer: `DynMa` variants in `ma.rs` each wrap `RustMa<T, IxDyn>` (= `MaskedArray<T,IxDyn>`), and `PyMaskedArray { inner: DynMa }` holds it by value. |
| REQ-3 (PyO3 option analysis) | SHIPPED | A/B/C/D each analyzed below against `ndarray 0.17` (`Array` owned, `ArcArray` is `Arc<Vec<T>>` COPY-ON-WRITE in `array/arc.rs`, `ArrayViewMut<'a,...>` lifetime-bound in `array/view_mut.rs`) and `PyO3 0.28` (`#[pyclass]` requires `'static`; `Py<T>` for keepalive; no safe lifetime erasure). Verdict: A/B both require a ferray-ma library rearchitecture; C is unsound; D is the recommendation. |
| REQ-4 (decision + prerequisite) | NOT-STARTED | DECISION: DEFER. The fix is NOT a binding tweak — it needs a writeback-capable shared/view storage model in the ferray-ma LIBRARY (owned `Array` cannot share a buffer; `ArcArray` is copy-on-write, the OPPOSITE of view-writeback). Open prereq blocker #879 (filed this session) for the ferray-ma shared-buffer `MaskedArrayView` prerequisite. Magnitude quantified in "Decision" below: library + binding rearchitecture, multi-session. |

## Architecture

### REQ-1 — index-kind taxonomy (VIEW vs COPY)

numpy's `MaskedArray.__getitem__` (`numpy/ma/core.py:3277`):

```python
dout = self.data[indx]          # core.py:3287 — ndarray.__getitem__
...
dout = dout.view(type(self))    # core.py:3358 — keep it a view of the same buffer
dout._update_from(self)
...
dout._mask = reshape(mout, dout.shape)   # core.py:3398 — mask is _mask[indx], same buffer
```

`self.data[indx]` is `numpy.ndarray.__getitem__`. numpy's indexing contract
(the R-DEV-1 view-vs-copy semantics ferray must MATCH):

| Index kind | Example | numpy result | ferray today | Match? |
|---|---|---|---|---|
| int (row of N-D) | `a[0]` | VIEW | COPY | NO — gap |
| slice | `a[1:3]`, `a[::2]` | VIEW | COPY | NO — gap |
| ellipsis | `a[...]` | VIEW | COPY | NO — gap |
| newaxis | `a[None]` | VIEW | COPY | NO — gap |
| int/slice tuple | `a[1, :]`, `a[:, 0]` | VIEW | COPY | NO — gap |
| integer array | `a[[0, 2]]` | **COPY** | COPY | YES — correct |
| boolean mask | `a[a > 1]` | **COPY** | COPY | YES — correct |
| diagonal gather | `a[[0,1],[0,1]]` | **COPY** | COPY | YES — correct |

The advanced/fancy/bool COPY cases are ALREADY CORRECT — `ferray` returns a copy
exactly as numpy does, and the GREEN pins
`test_getitem_fancy_realmask_source_keeps_real_mask_array`,
`test_getitem_diagonal_gather_realmask_mask_is_array`, and
`test_getitem_empty_index_realmask_source_keeps_empty_mask_array` confirm the
result mask/shape match. **The gap is ONLY the basic-indexing VIEW kinds.**

Crucially, ferray's `resolve_index` (in `ma.rs`) erases the distinction: it
subscripts a `numpy.arange(size).reshape(shape)[indx]` grid and collapses the
result into flat C-order `positions`. Basic and advanced indexing produce the
SAME `Gather { positions, shape }` — so the binding has, by construction,
discarded the information ("was this a basic index?") needed to decide
view-vs-copy. Restoring view-writeback requires re-introducing that distinction
AND a buffer the child can write through.

### REQ-2 — ferray-ma storage model (OWNED, not shareable)

`masked_array.rs`:

```rust
pub struct MaskedArray<T: Element, D: Dimension> {
    /// The underlying data array.
    data: Array<T, D>,
    mask: Arc<OnceLock<Array<bool, D>>>,
    real_mask: bool,
    pub(crate) hard_mask: bool,
    pub(crate) fill_value: T,
}
```

The `data` field is an OWNED `Array<T, D>`. From `array/owned.rs`:

```rust
pub struct Array<T: Element, D: Dimension> {
    pub(crate) inner: ndarray::Array<T, D::NdarrayDim>,
    ...
}
```

`ndarray::Array<T, D>` is `ArrayBase<OwnedRepr<T>, D>` — it OWNS its `Vec<T>`.
Consequences:

1. A `MaskedArray` cannot be a writeback view into another `MaskedArray`'s data
   buffer: the type is not parameterized over ownership (no
   `MaskedArray<Repr, T, D>`), and `data: Array<T,D>` is hardcoded to the owned
   repr.
2. `Clone for MaskedArray` (`masked_array.rs`) DEEP-CLONES the data
   (`data: self.data.clone()`) and only structurally shares the MASK via `Arc`.
   The data was never shareable; the comment even notes *"the data array is
   still deep-cloned because ferray-core's Array doesn't have Arc-based
   structural sharing."*
3. The mask's `Arc<OnceLock<...>>` is COPY-ON-WRITE (`make_mask_unique` /
   `ensure_materialized_mut` in `masked_array.rs` deep-copy on mutation when
   `Arc::strong_count > 1`). CoW is the OPPOSITE of view-writeback: a child's
   write would clone the mask away from the parent, not propagate into it.

So the storage model offers no path to writeback today, for data OR mask.

### REQ-3 — PyO3 0.28 + ndarray 0.17 option analysis

The view must satisfy two hard constraints simultaneously: (a) the child shares
the parent's data+mask buffer so writes propagate, and (b) the parent stays
alive while the child holds the view. Evaluating each option:

**Option A — Arc-shared buffer in the library.**
Change `MaskedArray` (or a new repr-generic `MaskedArray`) to store data behind
a shared, offset/stride-addressed buffer so a slice shares the `Arc` plus a
window descriptor, and `__setitem__` writes through the shared buffer.

- ndarray 0.17 reality: the existing `ArcArray<T,D>` (`array/arc.rs`) is
  `data: Arc<Vec<T>>` + `offset`/`strides`, but it is documented COPY-ON-WRITE:
  *"When a mutation is requested and the reference count is greater than 1, the
  buffer is cloned first"*. That gives shared READS but NOT shared writeback —
  exactly wrong for view semantics. A true writeback needs `Arc<RefCell<Vec<T>>>`
  / `Arc<Mutex<Vec<T>>>` (interior mutability) or raw-pointer aliasing, which
  the anti-pattern-gate (R-APG-1) explicitly bans (`Arc<Mutex<T>>` /
  `Rc<RefCell<T>>` escape hatches), and which would ripple to EVERY `MaskedArray`
  method + every ferray-ma consumer.
- Cost: rearchitect the whole ferray-ma storage model AND its data accessors
  (`data()`, `data_mut()`, every arithmetic/reduction/manipulation method) to go
  through the shared repr. Ripples into `ferray-python/src/ma.rs` (the `DynMa`
  match arms, `__getitem__`/`__setitem__`, and all ~hundreds of method bindings
  that call `.data()`).
- Risk: HIGH (cross-crate rearchitecture; CoW-vs-writeback semantics conflict
  with the existing #512 mask-CoW design; banned interior-mutability pattern).
- Verdict: NOT a binding fix. Tractable only as a library prerequisite (REQ-4).

**Option B — base-object reference in the binding.**
`PyMaskedArray` holds `Option<Py<PyMaskedArray>>` base + a view descriptor
(offset/positions); `__getitem__` on a basic index returns a child referencing
the base; get/set index into the base's buffer through the descriptor.

- PyO3 0.28 reality: `Py<PyMaskedArray>` keeps the parent alive (constraint (b)
  is satisfiable). But to WRITE through the base, `__setitem__` on the child must
  obtain `&mut` into the parent's `DynMa` while it lives behind a `Py<>` /
  `Bound<>`. PyO3 0.28 exposes that via `bound.borrow_mut()` →
  `PyRefMut<'py, PyMaskedArray>`, which is a RUNTIME-checked borrow
  (the pyclass uses interior `RefCell`-like borrow flags). That is mechanically
  possible, but: (i) every child `__getitem__`/`__setitem__`/`data()` /
  reduction / arithmetic method must now branch on "am I a view?" and redirect
  through the base's buffer + descriptor — touching essentially the entire
  `PyMaskedArray` method surface; (ii) the `#[derive(Clone)]` on `PyMaskedArray`
  and the `from_py_object` round-trip break view identity (a cloned view detaches
  from its base); (iii) a view-of-a-view chains base references and the
  descriptor composition (offset-of-offset) must be tracked. The
  `resolve_index` grid currently returns flat C-order positions, so a slice's
  child→parent mapping is expressible as a `Vec<usize>` of parent positions, but
  reads/writes through it re-touch every accessor.
- Cost: binding-only in principle, but the "am I a view?" branch infects the
  WHOLE `PyMaskedArray` surface (every `#[pymethods]` entry must read/write
  through the base when `base.is_some()`). Realistically a near-rewrite of the
  `ma.rs` method dispatch.
- Risk: HIGH (borrow-flag panics across the boundary if a view and its base are
  borrowed simultaneously — e.g. `a[b[0]]`; clone/identity semantics; view-chain
  descriptor composition). The runtime borrow check turns aliasing into a
  `PyBorrowError` rather than UB, which avoids `unsafe`, but the surface-wide
  redirect is the same magnitude as Option A.
- Verdict: NOT a small fix; comparable magnitude to A, concentrated in
  `ferray-python` instead of `ferray-ma`.

**Option C — store an `ArrayViewMut` in the pyclass.**
`ArrayViewMut<'a, T, D>` (`array/view_mut.rs`) wraps
`ndarray::ArrayViewMut<'a, ...>` — it is LIFETIME-BOUND (`'a` ties it to the
borrowed parent).

- PyO3 0.28 reality: `#[pyclass]` requires the struct be `'static` (no borrowed
  lifetime parameters; a pyclass is heap-owned by the interpreter for an
  unbounded duration). Erasing `'a` into `'static` requires `unsafe`
  `std::mem::transmute` of the lifetime PLUS manual parent-keepalive — exactly
  the `unsafe` outside-leaf-primitives that R-CODE-1 forbids, and a classic
  use-after-free footgun if the parent is mutated/dropped (ndarray reallocates,
  invalidating the erased pointer).
- Verdict: IMPOSSIBLE safely. Rejected.

**Option D — DEFER with a tracked library prerequisite.**
Document that ferray returns a COPY for basic-index slices; data+mask READS
through the sub-array are ALREADY CORRECT; only in-place writeback through a
child slice differs. File the prerequisite as a blocker.

- Real-world impact is NARROW. `a[1:3] = value` (setitem on the PARENT) ALREADY
  WORKS — it dispatches to `PyMaskedArray::__setitem__` (`ma.rs`), which resolves
  the slice positions and writes them in place on `self`; it never creates a
  child. The ONLY broken idiom is `child = a[1:3]; child[k] = value` writeback,
  the less-common two-step alias-then-write pattern.
- Verdict: RECOMMENDED. See Decision.

### REQ-4 — Decision

**DECISION: DEFER, with a tracked ferray-ma library prerequisite.**

View-writeback for `fr.ma.MaskedArray` is NOT tractable as a binding change at
acceptable risk. The fix demands a writeback-capable shared/view storage model
in the ferray-ma LIBRARY, because:

- the owned `Array<T,D>` data buffer cannot be shared (REQ-2);
- the existing `ArcArray` is copy-on-write, the OPPOSITE of writeback (REQ-3.A);
- a true writeback buffer needs interior mutability (`Arc<RefCell/Mutex>`) banned
  by R-APG-1, or raw-pointer aliasing / lifetime-`transmute` banned by R-CODE-1;
- both A and B ripple across the ENTIRE `MaskedArray` / `PyMaskedArray` method
  surface (every accessor, arithmetic op, reduction, manipulation, plus the
  `DynMa` dispatch and the `#[derive(Clone)]` / `from_py_object` round-trip).

**Magnitude (honest):** a real fix is a multi-file, multi-session rearchitecture:

- ferray-ma library: parameterize `MaskedArray` over data ownership (a
  `MaskedArrayView` / repr-generic data field) OR add an `Arc<RefCell<...>>`
  shared-buffer repr — `masked_array.rs` struct + Clone + every data accessor +
  the #512 mask-CoW interaction; downstream every `ferray-ma` op that reads
  `.data()` / writes `.data_mut()`. Order ~500–1000+ LOC touched across the
  crate, plus an R-APG-1 override discussion (the banned-pattern gate).
- ferray-python binding: `PyMaskedArray.__getitem__` must distinguish BASIC from
  ADVANCED indexing (today `resolve_index` erases it — `ma.rs`), return a view
  child for basic, and route the view child's reads/writes/methods through the
  base buffer. Touches the `DynMa` dispatch and the whole `#[pymethods]` surface
  in `ma.rs`.
- Crates: 2 (ferray-ma + ferray-python). Sessions: multiple (a library builder
  pass + a binding builder pass + critic re-audits).

This magnitude is a LEGITIMATE justification for deferral with a tracked
prerequisite — NOT a convenience excuse. The prerequisite is a real, scoped work
item: **a shared/view-capable writeback storage model in ferray-ma** (filed as
blocker #879 this session). The dependent REQ-4 stays NOT-STARTED until that
library prerequisite lands; only then does the binding-side basic-index view
become buildable.

Until then, the divergence pin
`test_getitem_slice_is_view_writeback_KNOWN_copy_divergence` documents the
narrow, builder-disclosed gap (basic-slice writeback only). It MUST NOT be
xfail/skip-escaped (R-DEFER-3); it remains a tracked-RED pin under #857 referencing
the prerequisite blocker.

## Verification

This is a scoping doc; its "verification" is that the COPY cases already match
numpy (the GREEN advanced-index pins) and that the divergence is precisely the
basic-index VIEW writeback (the one RED pin). Commands:

- `cd ferray-python && maturin develop`
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/test_divergence_ma_subscript_audit.py -q`
  - GREEN (copy cases already correct, REQ-1):
    `test_getitem_fancy_realmask_source_keeps_real_mask_array`,
    `test_getitem_diagonal_gather_realmask_mask_is_array`,
    `test_getitem_empty_index_realmask_source_keeps_empty_mask_array`,
    `test_getitem_slice_realmask_source_keeps_real_mask_array`.
  - RED (the deferred basic-index VIEW writeback gap, REQ-4):
    `test_getitem_slice_is_view_writeback_KNOWN_copy_divergence` — stays RED
    until prerequisite blocker #879 lands; it is the contract for the future
    builder.
- numpy oracle (live, R-CHAR-3) establishing the taxonomy in REQ-1:
  - `python3 -c "import numpy as np; a=np.ma.array([1.,2,3,4]); b=a[1:3]; b[0]=99; print(a[1])"` → `99.0` (basic slice is a VIEW).
  - `python3 -c "import numpy as np; a=np.ma.array([1.,2,3,4]); b=a[[1,2]]; b[0]=99; print(a[1])"` → `2.0` (fancy index is a COPY — ferray already matches).
