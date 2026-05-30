# Feature: ferray-core — N-dimensional array type and foundational primitives

## Summary
The foundational crate for the ferray workspace. Provides `NdArray<T, D>` — a generic N-dimensional array type with full ownership model (owned, view, mutable view, arc, cow), broadcasting, basic and advanced indexing, array creation routines, shape manipulation, type promotion, and the `FerrayError` error hierarchy. Every other ferray crate depends on this. Internally backed by `ndarray` for storage, but `ndarray` is not part of the public API.

## Dependencies
- **Upstream**: None (this is the root of the dependency graph)
- **Downstream**: ferray-ufunc, ferray-stats, ferray-io, ferray-linalg, ferray-fft, ferray-random, ferray-polynomial, ferray-strings, ferray-ma, ferray-stride-tricks, ferray-numpy-interop, ferray (re-export)
- **External crates**: `ndarray` 0.17 (internal storage), `num-complex` (Complex types), `half` (f16, feature-gated), `thiserror` 2.0 (error derive), `serde` 1.0 (feature-gated serialization)

## Phase
Phase 1 — Core Array and Ufuncs (BLOCKING: must complete before any other crate)

## Agent Splitting Guidance
This crate is too large for a single agent. The coordinator SHOULD split it into sub-agents:
- **Agent 1a: core-types** (opus) — NdArray<T,D>, ownership model (REQ-1 through REQ-5), Dimension trait, MemoryLayout, DType system (REQ-6 through REQ-8), error handling (REQ-27, REQ-28), buffer interop (REQ-29, REQ-30), introspection (REQ-35, REQ-36), iterator/Display (REQ-37, REQ-38, REQ-39)
- **Agent 1b: core-indexing** (opus) — Broadcasting (REQ-9 through REQ-11), basic + advanced indexing (REQ-12 through REQ-15), extended indexing (REQ-15a)
- **Agent 1c: core-creation-manipulation** (sonnet) — Array creation (REQ-16 through REQ-19), shape manipulation (REQ-20 through REQ-22), extended manipulation (REQ-22a), constants/finfo (REQ-33, REQ-34)
- **Agent 1d: core-macros** (sonnet) — FerrayRecord proc macro (REQ-8), s![] macro, promoted_type! macro (REQ-23 through REQ-26)

Agent 1a must complete before 1b/1c/1d can start. 1b, 1c, 1d can run in parallel.

## Requirements

### Core Array Type
- REQ-1: Define `NdArray<T, D: Dimension>` with fields for data representation, shape, strides, and `MemoryLayout` enum (C, Fortran, Custom)
- REQ-2: Implement type aliases: `Array1<T>`, `Array2<T>`, `Array3<T>`, `ArrayD<T>` (dynamic rank), and float-specialized aliases (`F32Array1`, `F64Array2`, etc.)
- REQ-3: Implement the full ownership model: `Array<T, D>` (owned), `ArrayView<'a, T, D>` (immutable borrow), `ArrayViewMut<'a, T, D>` (mutable borrow), `ArcArray<T, D>` (ref-counted CoW), `CowArray<'a, T, D>` (owned-or-borrowed)
- REQ-4: `ArcArray` must implement copy-on-write semantics: mutation clones the buffer when `Arc` refcount > 1; views derived from `ArcArray` observe the data at creation time regardless of subsequent mutations to the source
- REQ-5: Implement `From<ndarray::Array>` and `Into<ndarray::Array>` conversion traits. Zero-copy where memory layouts are compatible. `ndarray` must NOT appear in any public type signature.

### Dtype System
- REQ-6: Define the `Element` trait bounding valid array element types. Implement for: `f16` (feature-gated), `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `bool`
- REQ-7: Implement `DType` runtime enum mirroring all supported element types, with `size_of()`, `alignment()`, `is_float()`, `is_integer()`, `is_complex()` introspection methods
- REQ-8: Implement structured dtype support via `#[derive(FerrayRecord)]` proc macro. Arrays of structs must support zero-copy strided views of individual fields.

### Broadcasting
- REQ-9: Implement NumPy's full broadcasting rules: (1) prepend 1s to shape of lower-dim array, (2) stretch size-1 dimensions, (3) error on size mismatch where neither is 1
- REQ-10: Broadcasting must not materialize the broadcast array — use virtual expansion via strides where possible
- REQ-11: Provide `broadcast_to()`, `broadcast_arrays()`, `broadcast_shapes()` functions

### Indexing
- REQ-12: Basic indexing (integer + slice) via `s![]` macro returning views (zero-copy)
- REQ-13: Advanced (fancy) indexing via integer arrays (`index_select`) and boolean arrays (`boolean_index`), always returning copies
- REQ-14: `insert_axis` / `remove_axis` for dimension manipulation (np.newaxis equivalent)
- REQ-15: `boolean_index_assign` for masked assignment (`a[mask] = value`)
- REQ-15a: Extended indexing functions: `take(&a, indices, axis)`, `take_along_axis(&a, indices, axis)`, `put(&a, indices, values)`, `put_along_axis(&a, indices, values, axis)`, `choose(indices, choices)`, `compress(condition, &a, axis)`, `select(condlist, choicelist, default)`, `indices(dimensions)`, `ix_(&sequences)`, `diag_indices(n)`, `diag_indices_from(&a)`, `tril_indices(n, k, m)`, `triu_indices(n, k, m)`, `tril_indices_from(&a, k)`, `triu_indices_from(&a, k)`, `ravel_multi_index(multi_index, dims)`, `unravel_index(indices, shape)`, `flatnonzero(&a)`, `fill_diagonal(&mut a, val)`, `ndindex(shape)` (iterator), `ndenumerate(&a)` (iterator)

### Array Creation
- REQ-16: Top-level creation functions: `array()`, `asarray()`, `frombuffer()`, `fromiter()`, `zeros()`, `ones()`, `full()`, `zeros_like()`, `ones_like()`, `full_like()`
- REQ-17: `empty()` must return `Array<MaybeUninit<T>, D>` requiring explicit `assume_init()` — never silently return garbage data
- REQ-18: Range functions: `arange()`, `linspace()`, `logspace()`, `geomspace()`, `meshgrid()`, `mgrid()`, `ogrid()`
- REQ-19: Identity/diagonal: `eye()`, `identity()`, `diag()`, `diagflat()`, `tri()`, `tril()`, `triu()`

### Shape Manipulation
- REQ-20: Methods: `reshape()`, `ravel()`, `flatten()`, `squeeze()`, `expand_dims()`, `broadcast_to()`
- REQ-21: Join/split: `concatenate()`, `stack()`, `vstack()`, `hstack()`, `dstack()`, `block()`, `split()`, `array_split()`, `vsplit()`, `hsplit()`, `dsplit()`
- REQ-22: Transpose/reorder: `transpose()`, `swapaxes()`, `moveaxis()`, `rollaxis()`, `flip()`, `fliplr()`, `flipud()`, `rot90()`, `roll()`
- REQ-22a: Extended manipulation functions: `pad(&a, pad_width, mode)` with modes Constant, Edge, Reflect, Symmetric, Wrap; `tile(&a, reps)`, `repeat(&a, repeats, axis)`, `delete(&a, indices, axis)`, `insert(&a, index, values, axis)`, `append(&a, values, axis)`, `resize(&a, new_shape)`, `trim_zeros(&a, trim)`

### Type Casting and Promotion
- REQ-23: Implement NumPy's type promotion rules (smallest type that represents both inputs without precision loss). Provide `promoted_type!()` compile-time macro and `result_type()` runtime function.
- REQ-24: Mixed-type binary operations must NOT compile implicitly — require explicit `.astype()` or `add_promoted()` / `mul_promoted()` etc.
- REQ-25: Cast functions: `astype()`, `view()` (reinterpret), `can_cast()`, `common_type()`, `min_scalar_type()`, `promote_types()`
- REQ-26: Type inspection predicates: `issubdtype()`, `isrealobj()`, `iscomplexobj()`

### Error Handling
- REQ-27: Define `FerrayError` as a `#[non_exhaustive]` enum with variants: `ShapeMismatch`, `BroadcastFailure`, `AxisOutOfBounds`, `IndexOutOfBounds`, `SingularMatrix`, `ConvergenceFailure`, `InvalidDtype`, `NumericalInstability`, `IoError`, `InvalidValue`
- REQ-28: All public functions return `Result<T, FerrayError>`. Zero panics in library code.

### Memory Layout and Interop
- REQ-29: Implement `AsRawBuffer` trait exposing `as_ptr()`, `shape()`, `strides()`, `dtype()`, `is_c_contiguous()`, `is_f_contiguous()` for zero-copy interop
- REQ-30: Implement `DynArray` enum (runtime-typed array) covering all supported dtypes, for use when element type is not known at compile time

### Constants
- REQ-33: Provide mathematical constants in `ferray_core::constants`: `PI`, `E`, `INF`, `NEG_INF`, `NAN`, `EULER_GAMMA`, `PZERO` (+0.0), `NZERO` (-0.0). Also provide `NEWAXIS` as an alias for `None` in indexing contexts (expand_dims sentinel).

### Type Introspection (finfo / iinfo)
- REQ-34: Implement `finfo<T: Float>()` returning `FloatInfo { eps, min, max, smallest_normal, smallest_subnormal, bits, nmant, nexp, maxexp, minexp }` and `iinfo<T: Integer>()` returning `IntInfo { min, max, bits }` — matching `np.finfo` and `np.iinfo`

### Array Introspection Properties
- REQ-35: All `NdArray` variants must expose: `.shape()` → &[usize], `.ndim()` → usize, `.size()` → usize (total elements), `.itemsize()` → usize (bytes per element), `.nbytes()` → usize (total bytes), `.dtype()` → DType, `.is_empty()` → bool
- REQ-36: Additional properties: `.T` or `.t()` → transposed view (zero-copy), `.flat()` → flat iterator over all elements, `.copy()` → deep clone, `.to_vec()` → convert to nested `Vec`, `.to_bytes()` → raw byte slice, `.flags()` → struct with `c_contiguous`, `f_contiguous`, `owndata`, `writeable` booleans

### Iterator and Closure Operations
- REQ-37: Implement iteration: `.iter()` / `.iter_mut()` (element iteration), `.into_iter()` (consuming), `.indexed_iter()` (with multi-dimensional indices), `.lanes(axis)` (iterate over lanes along axis), `.axis_iter(axis)` / `.axis_iter_mut(axis)` (iterate over sub-arrays along axis)
- REQ-38: Implement closure-based operations: `.mapv(|x| ...)` (map closure elementwise, return new array), `.mapv_inplace(|x| ...)` (in-place map), `.zip_mut_with(&other, |a, b| ...)` (zip two arrays mutably), `.fold_axis(axis, init, |acc, x| ...)` (fold along axis)

### Display and Formatting
- REQ-39: Implement `Display` and `Debug` for all `NdArray` variants. Display format must match NumPy's array printing: aligned columns, truncated output for large arrays (>1000 elements shows first/last 3 rows with `...`), configurable precision via `set_print_options(precision, threshold, linewidth, edgeitems)`. Default: precision=8, threshold=1000, linewidth=75, edgeitems=3.

### Reductions (argmin / argmax / integer-mean — NumPy parity)
These extend the instance-method reductions in `array/reductions.rs`
(`sum`/`prod`/`min`/`max`/`mean`/`var`/`std`/`any`/`all` + the `ReduceAcc`
accumulator-promotion trait shipped in #780). The functions below are present in
NumPy but absent or under-constrained in ferray-core today.

- REQ-40: Implement `argmax` — return the index (NumPy `intp`, i.e. `i64` on a
  64-bit platform) of the maximum element. Whole-array form returns the flat
  index; the axis form returns an `Array<i64, IxDyn>` with the reduced axis
  removed. Semantics, exact (`numpy/_core/fromnumeric.py:1221-1314`):
  (a) **first occurrence** on ties — "In case of multiple occurrences of the
  maximum values, the indices corresponding to the first occurrence are
  returned" (`fromnumeric.py:1261-1262`);
  (b) **NaN-propagating, NaN-first** — when any NaN is present, the returned
  index is that of the *first* NaN, not of the numeric maximum (live oracle
  numpy 2.4.5: `np.argmax([1.0, nan, 3.0, nan]) == 1`);
  (c) **empty → error** — an empty reduction raises `ValueError`
  ("attempt to get argmax of an empty sequence", numpy 2.4.5); ferray returns
  `Err(FerrayError)` rather than panicking;
  (d) **result dtype is `intp`/`int64`**, independent of the input element dtype
  (`np.argmax(np.arange(5)).dtype == int64`).
- REQ-41: Implement `argmin` — the minimum-index mirror of REQ-40, identical
  contract with min substituted for max (`numpy/_core/fromnumeric.py:1321-...`):
  first-occurrence on ties (`fromnumeric.py:1361-1362`), NaN-first propagation
  (`np.argmin([1.0, nan, 3.0, nan]) == 1`), empty → `ValueError`
  ("attempt to get argmin of an empty sequence"), result dtype `intp`/`int64`.
- REQ-42: Integer/bool `mean` (and `mean_axis`) must promote to `f64`. NumPy
  casts a bool/integer input to `float64` before averaging: "Cast bool, unsigned
  int, and int to float64 by default ... `dtype = mu.dtype('f8')`"
  (`numpy/_core/_methods.py:124-127`), so `np.mean(np.array([1,2,3], np.int32))`
  is `float64 2.0` and `np.mean([True,False,True])` is `float64 0.6667` (live
  oracle numpy 2.4.5). Today ferray's `mean`/`mean_axis` are bounded `T: Element
  + Float`, so `Array::<i32, _>::mean()` does not compile and an integer mean is
  unreachable. The contract: the mean of an integer or bool array yields `f64`.

## Acceptance Criteria
- [ ] AC-1: `NdArray<f64, Ix2>` can be created, indexed, sliced, reshaped, and transposed. All operations return correct results verified against NumPy fixtures.
- [ ] AC-2: All five ownership variants (Array, ArrayView, ArrayViewMut, ArcArray, CowArray) compile and enforce their borrow semantics — mutable aliasing is a compile error.
- [ ] AC-3: `ArcArray` CoW semantics verified: mutating a shared ArcArray triggers a clone; views see stale data after source mutation.
- [ ] AC-4: Broadcasting: `(4,3) + (3,)`, `(4,1) * (4,3)`, `(2,1,4) + (3,4)` all produce correct shapes and values matching NumPy output.
- [ ] AC-5: Advanced indexing: `index_select` and `boolean_index` return copies; basic indexing returns views. Verified by checking pointer identity.
- [ ] AC-6: `ferray::empty((3,4))` returns `Array2<MaybeUninit<f64>>` — calling methods that read elements without `assume_init()` is a compile error.
- [ ] AC-7: `From<ndarray::Array2<f64>>` round-trips losslessly. `ndarray` does not appear in any public type signature (verified by `cargo doc` inspection).
- [ ] AC-8: `#[derive(FerrayRecord)]` generates a working structured array with zero-copy field views.
- [ ] AC-9: All array creation/manipulation functions compile and produce results matching NumPy for representative inputs.
- [ ] AC-10: `cargo test -p ferray-core` passes with zero failures. `cargo clippy -p ferray-core -- -D warnings` produces zero warnings.
- [ ] AC-11: Type promotion: `f32 + f64 -> f64`, `i32 + f32 -> f64`, `Complex<f32> + f64 -> Complex<f64>` all resolve correctly via `promoted_type!()`.
- [ ] AC-12: Mixed-type `Array1<f32> + Array1<f64>` fails to compile without explicit promotion.
- [ ] AC-13: `take(&a, &[0, 2], Axis(1))` returns correct columns. `ravel_multi_index` and `unravel_index` round-trip correctly. `pad` with all 5 modes produces NumPy-matching output.
- [ ] AC-14: `finfo::<f64>().eps` == `f64::EPSILON`. `iinfo::<i32>().min` == `i32::MIN`.
- [ ] AC-15: `.ndim()`, `.size()`, `.nbytes()`, `.itemsize()` return correct values for arrays of various shapes and dtypes.
- [ ] AC-16: `.mapv(|x| x * 2.0)` produces correct element-doubled array. `.iter()` visits all elements in logical order. `.axis_iter(Axis(0))` yields correct sub-arrays.
- [ ] AC-17: `println!("{}", array_2d)` produces NumPy-style formatted output. Arrays larger than threshold show truncated output with `...`.
- [ ] AC-18: `ferray_core::constants::PI` == `std::f64::consts::PI`. `ferray_core::constants::INF` == `f64::INFINITY`.

- [ ] AC-19: `Array::<i64, _>::from_vec([3,1,4,1,5,9,2]).argmax()` returns `5`
  and `.argmin()` returns `1` (first occurrence on the tied `1`s), matching
  `np.argmax`/`np.argmin`. NaN case: `argmax`/`argmin` of `[1.0, NaN, 3.0, NaN]`
  both return index `1` (NaN-first). Empty array → `Err(FerrayError)`, never a
  panic. Result element type is `i64`.
- [ ] AC-20: `Array::<i32, _>::from_vec([1,2,3]).mean()` returns `f64` `2.0`;
  `Array::<bool, _>` mean returns `f64` matching `np.mean` (e.g. `0.6666...`).
  Verified against the numpy oracle; the int/bool `mean` path compiles.

## Architecture

### Crate Layout
```
ferray-core/
  Cargo.toml
  src/
    lib.rs                    # Public API re-exports
    array/
      mod.rs                  # NdArray<T, D> definition
      owned.rs                # OwnedRepr, Array<T, D>
      view.rs                 # ViewRepr, ArrayView<'a, T, D>
      view_mut.rs             # MutViewRepr, ArrayViewMut<'a, T, D>
      arc.rs                  # ArcRepr, ArcArray<T, D> with CoW
      cow.rs                  # CowRepr, CowArray<'a, T, D>
      aliases.rs              # Array1, Array2, F64Array2, etc.
      display.rs              # impl Display/Debug for NdArray, set_print_options
      iter.rs                 # iter, iter_mut, indexed_iter, lanes, axis_iter
      methods.rs              # mapv, mapv_inplace, zip_mut_with, fold_axis
      introspect.rs           # ndim, size, itemsize, nbytes, flags, copy, to_vec, to_bytes
    dimension/
      mod.rs                  # Dimension trait, Ix1..Ix6, IxDyn
      broadcast.rs            # Broadcasting logic
    dtype/
      mod.rs                  # Element trait, DType enum
      promotion.rs            # Type promotion rules, promoted_type! macro
      casting.rs              # astype, can_cast, promote_types
      finfo.rs                # finfo<T>, iinfo<T> type introspection
    indexing/
      mod.rs                  # Index trait, s![] macro
      basic.rs                # Integer + slice indexing → views
      advanced.rs             # Fancy + boolean indexing → copies
      extended.rs             # take, put, choose, compress, select, indices, ix_, diag_indices, etc.
    creation/
      mod.rs                  # zeros, ones, full, empty, arange, linspace, eye, etc.
    manipulation/
      mod.rs                  # reshape, concat, stack, transpose, flip, etc.
      extended.rs             # pad, tile, repeat, delete, insert, append, resize, trim_zeros
    constants.rs              # PI, E, INF, NAN, NEWAXIS, EULER_GAMMA, etc.
    error.rs                  # FerrayError enum
    buffer.rs                 # AsRawBuffer trait
    dynarray.rs               # DynArray runtime-typed enum
    record.rs                 # FerrayRecord derive macro support
    layout.rs                 # MemoryLayout enum
    prelude.rs                # use ferray_core::prelude::*
  ferray-core-macros/         # Proc macro crate for #[derive(FerrayRecord)] and s![]
    Cargo.toml
    src/lib.rs
```

### Key Design Decisions
- `ndarray` is a private dependency. `NdArray` wraps `ndarray::ArrayBase` internally but exposes its own API. This insulates users from ndarray version churn.
- The `Dimension` trait hierarchy mirrors ndarray's (`Ix1`..`Ix6`, `IxDyn`) but is re-defined in ferray-core's namespace.
- Broadcasting is implemented as a standalone function that returns virtual shape/stride pairs, not new allocations.
- `FerrayRecord` proc macro lives in a separate `ferray-core-macros` crate (Rust proc macro constraint).
- Do NOT use `std::simd` — use `pulp` for all SIMD. `std::simd` is unstable.

## Open Questions

*None — all design decisions are resolved in the source design document.*

## Out of Scope
- SIMD-accelerated inner loops (handled by ferray-ufunc)
- Mathematical functions beyond basic arithmetic operator overloads (handled by ferray-ufunc)
- Statistics functions (handled by ferray-stats)
- File I/O (handled by ferray-io)
- Linear algebra (handled by ferray-linalg)
- Random number generation (handled by ferray-random)


## REQ status (reductions extension — argmin / argmax / integer-mean)

Two states only (goal.md R-DEFER-2). ferray-core cites use symbol anchors
(R-CITE-2b); numpy upstream cites use `file:line` against the read-only tree at
`/home/doll/numpy-ref` (oracle: numpy 2.4.5).

| REQ | Status | Evidence |
|---|---|---|
| REQ-40 (argmax) | SHIPPED | `pub fn argmax` (flattened, `Option<i64>`) and `pub fn argmax_axis` (`Array<i64, IxDyn>`) on `Array` in `reductions.rs`, sharing the `arg_reduce` helper (first-occurrence on ties via strict comparison; NaN-first via `partial_cmp(&x).is_none()`). NumPy contract: `numpy/_core/fromnumeric.py:1222` (`argmax`), first-occurrence ties `:1261-1262`, NaN-first / `intp` result; empty flattened → `None` (the `Option` analog `min`/`max` use, mapped to `ValueError` at the ferray-python boundary). Consumers (non-test): the `ArrayView::argmax` mirror in `reductions.rs` and the public-API boundary method itself (like `sum`/`min`); pinned by `argmax_argmin_basic`/`_nan_first`/`_axis_2d` in `ferray-core/tests/divergence_argreduce.rs` (oracle: numpy 2.4.5). |
| REQ-41 (argmin) | SHIPPED | `pub fn argmin`/`argmin_axis` on `Array` (and the `ArrayView::argmin` mirror) in `reductions.rs` — the min mirror of REQ-40 through the shared `arg_reduce(.., take_min=true)`. NumPy contract: `numpy/_core/fromnumeric.py:1322` (`argmin`), first-occurrence ties `:1361-1362`, NaN-first, `intp` result, empty→`None`. Pinned by `argmin_first_occurrence_on_ties`/`argmax_argmin_empty_returns_none` in `divergence_argreduce.rs`. |
| REQ-42 (integer/bool mean → f64) | SHIPPED | The `MeanAcc` trait in `reductions.rs` maps bool / signed-int / unsigned-int element types to an `f64` accumulator-and-result while `f32`/`f64`/complex keep their dtype (`Mean == Self`); `Array::mean`/`mean_axis` and the `ArrayView::mean` mirror are now bounded by `MeanAcc` instead of `Float`, so `Array::<i32, _>::mean()` returns `f64`. NumPy casts bool/unsigned/signed int to `float64` before averaging: `numpy/_core/_methods.py:124-127` ("Cast bool, unsigned int, and int to float64 by default ... `dtype = mu.dtype('f8')`"). Consumers (non-test): `Array::var`/`std` call `self.mean()?` (bounded `MeanAcc<Mean = T>`) and the `ArrayView` mirror, all in `reductions.rs`. Pinned by `mean_i32_returns_f64`/`mean_bool_returns_f64`/`mean_f32_stays_f32`/`mean_axis_int_returns_f64` in `divergence_argreduce.rs`. |

### Verification (gates the builder for REQ-40..42)
- `cargo test -p ferray-core` — existing reduction unit tests in `reductions.rs`
  stay green; the builder adds first-occurrence / NaN-first / empty-error tests
  for argmin/argmax and an int/bool-mean→f64 test.
- numpy oracle (R-CHAR-3): expected values from live numpy 2.4.5 —
  `np.argmax([1.0, nan, 3.0, nan]) == 1`, `np.argmin([3,1,4,1,5,1]) == 1`,
  `np.argmax([])` raises `ValueError`, `np.mean(np.int32([1,2,3])) == 2.0`
  (dtype `float64`), `np.mean([True,False,True])` ≈ `0.6667` (dtype `float64`).
- `cargo clippy -p ferray-core --all-targets -- -D warnings` and
  `cargo fmt --check`.

All three REQs are SHIPPED: the impls landed in `reductions.rs` and the oracle
checks above pass via `ferray-core/tests/divergence_argreduce.rs` (14 pins,
green under both the default and `FERRAY_FORCE_SCALAR=1` builds), with the
corresponding rows in the `reductions.rs` `//!` REQ-status table updated.

