# Feature: PyMaskedArray numpy-faithfulness — subscripting / dtype-preservation / mask-roundtrip (#850)

<!--
tier: 3-component
status: draft
baseline-commit: 702bb258
upstream-paths:
  - numpy/ma/core.py
-->

## Summary
`ferray.ma.MaskedArray` is exposed to CPython by the `PyMaskedArray` PyO3 class in
`ferray-python/src/ma.rs`. The class is **pinned to f64** at the marshalling
boundary and supports no subscripting. This doc scopes three confirmed live
divergences from numpy 2.4.4 (`ferray-python/.venv`) and proposes an incremental,
binary-classified plan under epic #850. It does NOT re-document the library-level
masked-array contract (that lives in `.design/ferray-ma.md`) nor the stateful
hardmask/`put`/`putmask` slice already shipped there.

This is a **PyO3-binding (marshalling) doc**, not a library doc. The pivotal
finding below establishes that the owning library `ferray-ma` is already generic
over `T: Element`, so all three divergences are binding-side marshalling work, not
library generification.

## Pivotal finding — `ferray-ma::MaskedArray` is GENERIC over `T: Element`, NOT f64-only

The library type is fully generic. From `pub struct MaskedArray in masked_array.rs`:

```rust
pub struct MaskedArray<T: Element, D: Dimension> {
    data: Array<T, D>,
    mask: Arc<OnceLock<Array<bool, D>>>,
    real_mask: bool,
    pub(crate) hard_mask: bool,
    pub(crate) fill_value: T,
}
```

and `impl<T: Element, D: Dimension> MaskedArray<T, D>` with `pub fn new(data:
Array<T, D>, mask: Array<bool, D>)`, `pub fn from_data(data: Array<T, D>)`. The
per-dtype default fill value is ALREADY dispatched across the full numeric set in
`fn default_fill_value in masked_array.rs`: it matches `DType::F32`, `F64`, `I8..I64`,
`U8..U64`, `Bool` and constructs numpy's `default_filler` value for each
(`1e20` float / `999999` int / `true` bool, mirroring `numpy/ma/core.py:163`).

**Conclusion**: the f64-only restriction lives ENTIRELY in the binding. Every
`PyMaskedArray` constructor/extractor in `ma.rs` hard-casts input through
`fn coerce_dtype in conv.rs` (`numpy.asarray(obj, "float64")`):

- `py_new` → `coerce_dtype(py, data, "float64")` then `PyReadonlyArrayDyn<f64>`.
- `fn extract_data in ma.rs` → `coerce_dtype(py, obj, "float64")`.
- `fn coerce_to_ma_impl in ma.rs` → `extract_data` (always f64).
- the `#[pyclass]` field is the concrete monomorphization `inner: RustMa<f64, IxDyn>`
  and `fn dtype` returns the literal `"float64"`.

Because the library is generic, **dtype-preservation is a binding-marshalling
problem** (dispatch over the input dtype at the PyO3 boundary), NOT a library
generification. ferray-python already owns the exact pattern needed:
`enum DynArray` in `ferray_core` plus `fn pyany_to_dynarray`/`fn dynarray_to_pyarray`
in `conv.rs`, and the `match_dtype_all!` macro (`conv.rs`) that monomorphizes a body
across all 11 element types `{f64,f32,i64,i32,i16,i8,u64,u32,u16,u8,bool}` from a
runtime dtype string. PyO3 classes cannot be generic, so the established
ferray-python idiom is an **enum-over-dtype field** (the `DynArray` shape) inside the
pyclass, dispatched at each method.

## Requirements

- REQ-1 (subscripting): `PyMaskedArray` supports `__getitem__` and `__setitem__`.
  `a[i]` returns the value (or the `numpy.ma.masked` singleton when that element is
  masked); slice / fancy / bool index returns a sub-`MaskedArray`. `a[i] = v` assigns
  data; `a[i] = ferray.ma.masked` masks the element; assignment respects the
  hardmask (a hard mask never clears a bit). Mirrors `numpy/ma/core.py` `__getitem__`
  (`core.py:3319`) / `__setitem__` (`core.py:3409`).
- REQ-2 (dtype preservation): a `PyMaskedArray` built from int/bool/float/complex
  input preserves that dtype. `ferray.ma.array([1,2,3]).dtype == "int64"` and
  `.filled().dtype == int64`, matching numpy. No silent int→f64 cast at the
  boundary (R-CODE-4).
- REQ-3 (mask-preserving construction): constructing a `PyMaskedArray` FROM an
  existing masked array (`numpy.ma.MaskedArray` or a `ferray.ma.MaskedArray`) with
  `mask=None` carries over the source mask (and data dtype). `ferray.ma.array(
  np.ma.array([1,2,3], mask=[1,0,0])).mask` equals `array([True,False,False])`,
  matching numpy's `_data`/`_mask` copy in `MaskedArray.__new__` (`core.py:2820`).

## Acceptance criteria
- AC-1 (REQ-1): pytest `divergence_ma_subscript.py` — `a[0]`, `a[0]=5`,
  `a[1:]`, `a[[0,2]]`, `a[[True,False,True]]`, masked-element returns
  `np.ma.masked`, hardmask-respecting `__setitem__` — all green vs numpy oracle.
- AC-2 (REQ-2): `fr.ma.array([1,2,3]).dtype == np.ma.array([1,2,3]).dtype` and
  `.filled().dtype` parity for int64/float32/bool/complex128 inputs.
- AC-3 (REQ-3): `fr.ma.array(np.ma.array(d, mask=m)).mask` array-equal to
  `np.ma.array(d, mask=m).mask` across int/float/bool data.

## REQ status table

| REQ | Status | Evidence |
|---|---|---|
| REQ-1 (subscripting) | NOT-STARTED | No `__getitem__`/`__setitem__` on `PyMaskedArray` (`grep '__getitem__\|__setitem__' ma.rs` → empty). Live: `fr.ma.array([1,2,3])[0]` raises `TypeError: not subscriptable`; `a[0]=5` raises `TypeError: does not support item assignment`. The library primitives exist (`pub fn data`, `pub fn data_mut`, `pub fn mask`, `pub fn set_mask_flat`, `pub fn put` in `masked_array.rs`/`put.rs`); only the binding `#[pymethods]` are absent. Open prereq blocker #852. |
| REQ-2 (dtype preservation) | SHIPPED | `PyMaskedArray { inner: DynMa }` where `enum DynMa` (`ma.rs`) holds one `RustMa<T, IxDyn>` per real dtype (the conv.rs `DynArray` shape), dispatched by the `match_ma!` macro (`ma.rs`). Construction infers the input dtype via `fn build_dynma`/`fn dynma_from_input` (`numpy.asarray` + `dtype_name`, NO `"float64"` cast); `fn dtype` returns `DynMa::dtype_name` (the native name). Non-test production consumers: the `#[pymethods] fn data`/`fn filled`/`fn dtype`/`fn __getitem__`/`fn __setitem__`/`fn put`/`fn fill_value` registrations on `PyMaskedArray` and the `#[pyfunction] fn array`/`fn masked_array`/`fn filled`/`fn put`/`fn putmask`/`fn count`/`fn set_fill_value` free functions in `ma.rs` (all dispatch the active variant). Live now: `fr.ma.array([1,2,3]).dtype == "int64"`, `.filled().dtype == int64`; `fr.ma.array([True,False],mask=[1,0]).dtype == "bool"`; `fr.ma.array([1,2,3],dtype=np.int32).dtype == "int32"`. Verified by `tests/test_expansion_ma_dtype.py` (109 cases vs numpy.ma oracle). COMPLEX/structured remain on the delegation/f64 paths (follow-up). Closed #853. |
| REQ-3 (mask round-trip) | NOT-STARTED | `fn py_new` with `mask=None` routes the input (incl. a `numpy.ma.MaskedArray`) through `coerce_dtype(..,"float64")` = `numpy.asarray(...)`, which DROPS the `_mask`, then `RustMa::from_data` → nomask sentinel. Live: `fr.ma.array(np.ma.array([1,2,3],mask=[1,0,0])).mask` returns `False` (scalar nomask), numpy returns `array([True,False,False])`. Note `fn extract_values_with_mask in ma.rs` ALREADY reads a numpy.ma mask via `numpy.ma.getmaskarray` for the `put`/`putmask` value path — the same read can feed the constructor. Open prereq blocker #851. |

REQ-2 (dtype preservation) is now SHIPPED (#853, the `DynMa` enum refactor of
`ma.rs`); REQ-1 (subscripting) and REQ-3 (mask round-trip) were shipped earlier
(#852 / #851) and re-verified against the dtype-preserving boundary.

## Architecture

### Current boundary shape (the f64 pin)
- `#[pyclass] PyMaskedArray { inner: RustMa<f64, IxDyn> }` (`ma.rs`). Concrete f64
  monomorphization — the only dtype the class can hold.
- Construction: `fn py_new` → `fn coerce_dtype` (`conv.rs`, `numpy.asarray(obj,
  "float64")`) → `PyReadonlyArrayDyn<f64>` → `as_ferray` → `RustMa::from_data` or
  `RustMa::new`. The `"float64"` literal is the cast site; `mask=None` is where a
  source array's mask is dropped (it never reaches the f64 `asarray` output).
- Egress: `fn data`/`fn filled`/`fn compressed` push `ArrayD<f64>` out via
  `into_pyarray`; `fn dtype` returns the constant `"float64"`.

### Target shape (generic, dtype-dispatched at the boundary)
The library being generic, the faithful binding holds an enum-over-dtype, exactly
like `ferray_core::DynArray` (`conv.rs` `pyany_to_dynarray`/`dynarray_to_pyarray`):

```text
enum DynMa {                         // new, in ma.rs (mirrors DynArray)
    F64(RustMa<f64, IxDyn>), F32(..), I64(..), I32(.. ) , ..., Bool(..),
}
#[pyclass] PyMaskedArray { inner: DynMa }
```

Each `#[pymethods]` body dispatches on the variant (the `match_dtype_all!` idiom,
`conv.rs`). Construction dispatches on `dtype_name(asarray(data))` instead of
hard-casting to `"float64"`; egress returns the variant's true dtype. This is the
SAME enum-dispatch pattern already used for file-IO (`DynArray`) and per-dtype
ufunc/creation dispatch (`match_dtype_all!`).

The upstream contract for each REQ:
- subscript get/set — `numpy/ma/core.py:3319` (`__getitem__`), `:3409`
  (`__setitem__`), masked-element → `masked` singleton, hardmask-respecting set.
- dtype preservation — `numpy/ma/core.py:2820` (`__new__` keeps `_data.dtype`).
- mask round-trip — `numpy/ma/core.py:2820`-`2900` (`_data`/`_mask` copied from a
  masked input), `getmaskarray` (`core.py:1521`).

### Existing binding assets to reuse
- `fn coerce_dtype`, `fn dtype_name` (`conv.rs`) — dtype probing.
- `enum DynArray` + `fn pyany_to_dynarray` + `fn dynarray_to_pyarray` (`conv.rs`) —
  the canonical enum-over-dtype marshalling template.
- `macro_rules! match_dtype_all` (`conv.rs`) — monomorphize a body across 11 dtypes.
- `fn extract_values_with_mask` (`ma.rs`) — already reads a `numpy.ma` `_mask` via
  `numpy.ma.getmaskarray` (REQ-3 reuses this read for the constructor path).
- `pub fn data_mut`, `pub fn set_mask_flat`, `pub fn put` (`masked_array.rs`,
  `put.rs`) — library element/mask write primitives for `__setitem__`.

## Incremental plan (ordered value/cost; each its own future blocker under #850)

The three items are independent and can be built/merged separately. Build order is
smallest-and-safest first.

### (a) Mask-preserving construction from a masked input — SMALLEST, HIGHEST-VALUE FIRST
- Scope: in `fn py_new` (and the `fn array`/`fn masked_array` free fns), when
  `mask=None` AND `data` is a `numpy.ma.MaskedArray` or `PyMaskedArray`, read its
  mask (numpy side: `numpy.ma.getmaskarray`, already wired in
  `fn extract_values_with_mask`) instead of dropping it, and construct with
  `RustMa::new(data, mask)`.
- Magnitude: ~15-30 LOC, binding-only, NO library change. If shipped before (c),
  the data still round-trips as f64 (dtype-preservation is a separate item), but the
  MASK is preserved — closes REQ-3 on its own.
- File manifest: `ferray-python/src/ma.rs` (constructor branch);
  `ferray-python/tests/divergence_ma_subscript.py` or a new
  `tests/divergence_ma_construct.py` (mask round-trip pins).
- Blocker: #851.

### (b) Subscripting `__getitem__`/`__setitem__` on the existing f64 array — MEDIUM
- Scope: add `#[pymethods] fn __getitem__`/`fn __setitem__` to `PyMaskedArray`.
  Parse the Python index (int / slice / bool array / fancy int array) — reuse
  numpy index normalization at the boundary (e.g. `numpy.asarray(idx)` + the
  library's flat-index primitives), return scalar / `masked` singleton
  (`fn ma_masked_singleton`, `conv.rs`) / sub-`MaskedArray`. `__setitem__` writes via
  `pub fn data_mut` + `pub fn set_mask_flat`; assigning `ferray.ma.masked` masks the
  position; honor the hardmask via the existing `set_mask_flat` hard-mask gate.
- Magnitude: ~120-200 LOC binding + index parsing helper. Builds on the f64 array
  as-is; does NOT require (c). May want a small library helper for slice→sub-array
  if `Array` slicing isn't already exposed — confirm during build (the manifest
  caps at the binding; flag if a library accessor is missing → sub-blocker).
- File manifest: `ferray-python/src/ma.rs` (`__getitem__`/`__setitem__` + an index
  parser); `ferray-python/tests/divergence_ma_subscript.py`. Possible
  `ferray-ma/src/masked_array.rs` IF a slice/sub-view accessor is missing (separate
  acto-builder, library-owned).
- Blocker: #852.

### (c) Dtype preservation across the boundary — LARGEST (binding marshalling, NOT library generification)
- Scope: replace `inner: RustMa<f64, IxDyn>` with an `enum DynMa` over the 11
  element types (the `DynArray` shape) and dispatch every `#[pymethods]` body on the
  variant via the `match_dtype_all!` idiom. Construction dispatches on the probed
  input dtype (`dtype_name`) rather than `coerce_dtype(.., "float64")`. `fn dtype`
  returns the variant's true name; `fn filled`/`fn data`/`fn compressed` egress the
  variant's dtype.
- Magnitude: this is the largest of the three because it touches EVERY method on the
  class (~40 `#[pymethods]` + free functions in `ma.rs`, which is ~4000 lines). It is
  NOT a multi-session library generification — the library is already generic — but
  it IS a substantial single-crate binding refactor: every reduction/ufunc/creation
  method must thread the enum. Rough estimate **400-700 LOC** of `ma.rs` churn,
  one focused acto-builder dispatch (possibly two if the elementwise-ufunc surface is
  split from the core class). Honest magnitude: meaningfully larger than (a)/(b), but
  bounded to `ferray-python/src/ma.rs` (+ maybe one `DynMa` helper module) — a single
  binding crate, not a workspace-wide change.
- Caveat — complex/structured dtypes: numpy.ma supports complex128 masked arrays;
  ferray-ma's masked surface and the `match_dtype_all!` set do NOT include complex
  (`default_fill_value` falls back to `T::zero()` for complex, and the interop
  `NpElement` set excludes complex). So (c) preserves the 11 real/bool dtypes; a
  complex masked array still needs a follow-up (out of scope here, file separately if
  REQ-2's complex clause is needed). The int/bool/float divergence — the live one —
  is fully covered.
- File manifest: `ferray-python/src/ma.rs` (the `DynMa` enum + all method dispatch);
  optionally a new `ferray-python/src/ma_dyn.rs` helper for the enum + marshalling;
  `ferray-python/tests/divergence_ma_dtype.py`.
- Blocker: #853.

### Sequencing
(a) first — smallest, highest leverage, zero library risk, independently closes
REQ-3. Then (b) — independent of dtype work, delivers the most-visibly-missing API
(subscripting). Then (c) — the dtype refactor — last, because it churns every method
and is best done once (a)/(b) have pinned the method surface with passing tests.
After (c) lands, (a) and (b) must be re-verified against the enum (their tests
should already cover the dtype-preserved paths).

## Verification

Per the ferray-python pytest oracle model (goal.md §verification (B)):
- `cd ferray-python && maturin develop` before each pytest run.
- `cd ferray-python && PYTHONPATH=python python3 -m pytest tests/ -q` — the
  per-item divergence files (`divergence_ma_construct.py` /
  `divergence_ma_subscript.py` / `divergence_ma_dtype.py`) green vs the numpy 2.4.4
  oracle; no previously-green `tests/test_ma.py` regressed.
- `cargo clippy -p ferray-python --all-targets -- -D warnings`; `cargo fmt --check`.

Each REQ flips to SHIPPED only when its item's divergence pytest is green AND a
non-test consumer exists (the `#[pymethods]`/free-function registration IS the
public API surface for the binding, R-DEFER-1/S5 — boundary methods are the
consumers).

Current state (baseline 702bb258, numpy 2.4.4 oracle):
- REQ-1: `a[0]` → `TypeError: not subscriptable` (FAIL).
- REQ-2: `fr.ma.array([1,2,3]).filled().dtype` = `float64`, numpy `int64` (FAIL).
- REQ-3: `fr.ma.array(np.ma.array([1,2,3],mask=[1,0,0])).mask` = `False`,
  numpy `[True False False]` (FAIL).
