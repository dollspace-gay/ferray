---
title: ferrum-core — N-dimensional array type and foundational primitives
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

The foundational crate for the ferrum workspace. Provides `NdArray<T, D>` — a generic N-dimensional array type with full ownership model (owned, view, mutable view, arc, cow), broadcasting, basic and advanced indexing, array creation routines, shape manipulation, type promotion, and the `FerrumError` error hierarchy. Every other ferrum crate depends on this. Internally backed by `ndarray` for storage, but `ndarray` is not part of the public API.

### Requirements

- REQ-1: Define `NdArray<T, D: Dimension>` with fields for data representation, shape, strides, and `MemoryLayout` enum (C, Fortran, Custom)
- REQ-2: Implement type aliases: `Array1<T>`, `Array2<T>`, `Array3<T>`, `ArrayD<T>` (dynamic rank), and float-specialized aliases (`F32Array1`, `F64Array2`, etc.)
- REQ-3: Implement the full ownership model: `Array<T, D>` (owned), `ArrayView<'a, T, D>` (immutable borrow), `ArrayViewMut<'a, T, D>` (mutable borrow), `ArcArray<T, D>` (ref-counted CoW), `CowArray<'a, T, D>` (owned-or-borrowed)
- REQ-4: `ArcArray` must implement copy-on-write semantics: mutation clones the buffer when `Arc` refcount > 1; views derived from `ArcArray` observe the data at creation time regardless of subsequent mutations to the source
- REQ-5: Implement `From<ndarray::Array>` and `Into<ndarray::Array>` conversion traits. Zero-copy where memory layouts are compatible. `ndarray` must NOT appear in any public type signature. ### Dtype System
- REQ-6: Define the `Element` trait bounding valid array element types. Implement for: `f16` (feature-gated), `f32`, `f64`, `Complex<f32>`, `Complex<f64>`, `i8`, `i16`, `i32`, `i64`, `i128`, `u8`, `u16`, `u32`, `u64`, `u128`, `bool`
- REQ-7: Implement `DType` runtime enum mirroring all supported element types, with `size_of()`, `alignment()`, `is_float()`, `is_integer()`, `is_complex()` introspection methods
- REQ-8: Implement structured dtype support via `#[derive(FerrumRecord)]` proc macro. Arrays of structs must support zero-copy strided views of individual fields. ### Broadcasting
- REQ-9: Implement NumPy's full broadcasting rules: (1) prepend 1s to shape of lower-dim array, (2) stretch size-1 dimensions, (3) error on size mismatch where neither is 1
- REQ-10: Broadcasting must not materialize the broadcast array — use virtual expansion via strides where possible
- REQ-11: Provide `broadcast_to()`, `broadcast_arrays()`, `broadcast_shapes()` functions ### Indexing
- REQ-12: Basic indexing (integer + slice) via `s![]` macro returning views (zero-copy)
- REQ-13: Advanced (fancy) indexing via integer arrays (`index_select`) and boolean arrays (`boolean_index`), always returning copies
- REQ-14: `insert_axis` / `remove_axis` for dimension manipulation (np.newaxis equivalent)
- REQ-15: `boolean_index_assign` for masked assignment (`a[mask] = value`)
- REQ-15a: Extended indexing functions: `take(&a, indices, axis)`, `take_along_axis(&a, indices, axis)`, `put(&a, indices, values)`, `put_along_axis(&a, indices, values, axis)`, `choose(indices, choices)`, `compress(condition, &a, axis)`, `select(condlist, choicelist, default)`, `indices(dimensions)`, `ix_(&sequences)`, `diag_indices(n)`, `diag_indices_from(&a)`, `tril_indices(n, k, m)`, `triu_indices(n, k, m)`, `tril_indices_from(&a, k)`, `triu_indices_from(&a, k)`, `ravel_multi_index(multi_index, dims)`, `unravel_index(indices, shape)`, `flatnonzero(&a)`, `fill_diagonal(&mut a, val)`, `ndindex(shape)` (iterator), `ndenumerate(&a)` (iterator) ### Array Creation
- REQ-16: Top-level creation functions: `array()`, `asarray()`, `frombuffer()`, `fromiter()`, `zeros()`, `ones()`, `full()`, `zeros_like()`, `ones_like()`, `full_like()`
- REQ-17: `empty()` must return `Array<MaybeUninit<T>, D>` requiring explicit `assume_init()` — never silently return garbage data
- REQ-18: Range functions: `arange()`, `linspace()`, `logspace()`, `geomspace()`, `meshgrid()`, `mgrid()`, `ogrid()`
- REQ-19: Identity/diagonal: `eye()`, `identity()`, `diag()`, `diagflat()`, `tri()`, `tril()`, `triu()` ### Shape Manipulation
- REQ-20: Methods: `reshape()`, `ravel()`, `flatten()`, `squeeze()`, `expand_dims()`, `broadcast_to()`
- REQ-21: Join/split: `concatenate()`, `stack()`, `vstack()`, `hstack()`, `dstack()`, `block()`, `split()`, `array_split()`, `vsplit()`, `hsplit()`, `dsplit()`
- REQ-22: Transpose/reorder: `transpose()`, `swapaxes()`, `moveaxis()`, `rollaxis()`, `flip()`, `fliplr()`, `flipud()`, `rot90()`, `roll()`
- REQ-22a: Extended manipulation functions: `pad(&a, pad_width, mode)` with modes Constant, Edge, Reflect, Symmetric, Wrap; `tile(&a, reps)`, `repeat(&a, repeats, axis)`, `delete(&a, indices, axis)`, `insert(&a, index, values, axis)`, `append(&a, values, axis)`, `resize(&a, new_shape)`, `trim_zeros(&a, trim)` ### Type Casting and Promotion
- REQ-23: Implement NumPy's type promotion rules (smallest type that represents both inputs without precision loss). Provide `promoted_type!()` compile-time macro and `result_type()` runtime function.
- REQ-24: Mixed-type binary operations must NOT compile implicitly — require explicit `.astype()` or `add_promoted()` / `mul_promoted()` etc.
- REQ-25: Cast functions: `astype()`, `view()` (reinterpret), `can_cast()`, `common_type()`, `min_scalar_type()`, `promote_types()`
- REQ-26: Type inspection predicates: `issubdtype()`, `isrealobj()`, `iscomplexobj()` ### Error Handling
- REQ-27: Define `FerrumError` as a `#[non_exhaustive]` enum with variants: `ShapeMismatch`, `BroadcastFailure`, `AxisOutOfBounds`, `IndexOutOfBounds`, `SingularMatrix`, `ConvergenceFailure`, `InvalidDtype`, `NumericalInstability`, `IoError`, `InvalidValue`
- REQ-28: All public functions return `Result<T, FerrumError>`. Zero panics in library code. ### Memory Layout and Interop
- REQ-29: Implement `AsRawBuffer` trait exposing `as_ptr()`, `shape()`, `strides()`, `dtype()`, `is_c_contiguous()`, `is_f_contiguous()` for zero-copy interop
- REQ-30: Implement `DynArray` enum (runtime-typed array) covering all supported dtypes, for use when element type is not known at compile time ### Constants
- REQ-33: Provide mathematical constants in `ferrum_core::constants`: `PI`, `E`, `INF`, `NEG_INF`, `NAN`, `EULER_GAMMA`, `PZERO` (+0.0), `NZERO` (-0.0). Also provide `NEWAXIS` as an alias for `None` in indexing contexts (expand_dims sentinel). ### Type Introspection (finfo / iinfo)
- REQ-34: Implement `finfo<T: Float>()` returning `FloatInfo { eps, min, max, smallest_normal, smallest_subnormal, bits, nmant, nexp, maxexp, minexp }` and `iinfo<T: Integer>()` returning `IntInfo { min, max, bits }` — matching `np.finfo` and `np.iinfo` ### Array Introspection Properties
- REQ-35: All `NdArray` variants must expose: `.shape()` → &[usize], `.ndim()` → usize, `.size()` → usize (total elements), `.itemsize()` → usize (bytes per element), `.nbytes()` → usize (total bytes), `.dtype()` → DType, `.is_empty()` → bool
- REQ-36: Additional properties: `.T` or `.t()` → transposed view (zero-copy), `.flat()` → flat iterator over all elements, `.copy()` → deep clone, `.to_vec()` → convert to nested `Vec`, `.to_bytes()` → raw byte slice, `.flags()` → struct with `c_contiguous`, `f_contiguous`, `owndata`, `writeable` booleans ### Iterator and Closure Operations
- REQ-37: Implement iteration: `.iter()` / `.iter_mut()` (element iteration), `.into_iter()` (consuming), `.indexed_iter()` (with multi-dimensional indices), `.lanes(axis)` (iterate over lanes along axis), `.axis_iter(axis)` / `.axis_iter_mut(axis)` (iterate over sub-arrays along axis)
- REQ-38: Implement closure-based operations: `.mapv(|x| ...)` (map closure elementwise, return new array), `.mapv_inplace(|x| ...)` (in-place map), `.zip_mut_with(&other, |a, b| ...)` (zip two arrays mutably), `.fold_axis(axis, init, |acc, x| ...)` (fold along axis) ### Display and Formatting
- REQ-39: Implement `Display` and `Debug` for all `NdArray` variants. Display format must match NumPy's array printing: aligned columns, truncated output for large arrays (>1000 elements shows first/last 3 rows with `...`), configurable precision via `set_print_options(precision, threshold, linewidth, edgeitems)`. Default: precision=8, threshold=1000, linewidth=75, edgeitems=3.

### Acceptance Criteria

- [ ] AC-1: `NdArray<f64, Ix2>` can be created, indexed, sliced, reshaped, and transposed. All operations return correct results verified against NumPy fixtures.
- [ ] AC-2: All five ownership variants (Array, ArrayView, ArrayViewMut, ArcArray, CowArray) compile and enforce their borrow semantics — mutable aliasing is a compile error.
- [ ] AC-3: `ArcArray` CoW semantics verified: mutating a shared ArcArray triggers a clone; views see stale data after source mutation.
- [ ] AC-4: Broadcasting: `(4,3) + (3,)`, `(4,1) * (4,3)`, `(2,1,4) + (3,4)` all produce correct shapes and values matching NumPy output.
- [ ] AC-5: Advanced indexing: `index_select` and `boolean_index` return copies; basic indexing returns views. Verified by checking pointer identity.
- [ ] AC-6: `ferrum::empty((3,4))` returns `Array2<MaybeUninit<f64>>` — calling methods that read elements without `assume_init()` is a compile error.
- [ ] AC-7: `From<ndarray::Array2<f64>>` round-trips losslessly. `ndarray` does not appear in any public type signature (verified by `cargo doc` inspection).
- [ ] AC-8: `#[derive(FerrumRecord)]` generates a working structured array with zero-copy field views.
- [ ] AC-9: All array creation/manipulation functions compile and produce results matching NumPy for representative inputs.
- [ ] AC-10: `cargo test -p ferrum-core` passes with zero failures. `cargo clippy -p ferrum-core -- -D warnings` produces zero warnings.
- [ ] AC-11: Type promotion: `f32 + f64 -> f64`, `i32 + f32 -> f64`, `Complex<f32> + f64 -> Complex<f64>` all resolve correctly via `promoted_type!()`.
- [ ] AC-12: Mixed-type `Array1<f32> + Array1<f64>` fails to compile without explicit promotion.
- [ ] AC-13: `take(&a, &[0, 2], Axis(1))` returns correct columns. `ravel_multi_index` and `unravel_index` round-trip correctly. `pad` with all 5 modes produces NumPy-matching output.
- [ ] AC-14: `finfo::<f64>().eps` == `f64::EPSILON`. `iinfo::<i32>().min` == `i32::MIN`.
- [ ] AC-15: `.ndim()`, `.size()`, `.nbytes()`, `.itemsize()` return correct values for arrays of various shapes and dtypes.
- [ ] AC-16: `.mapv(|x| x * 2.0)` produces correct element-doubled array. `.iter()` visits all elements in logical order. `.axis_iter(Axis(0))` yields correct sub-arrays.
- [ ] AC-17: `println!("{}", array_2d)` produces NumPy-style formatted output. Arrays larger than threshold show truncated output with `...`.
- [ ] AC-18: `ferrum_core::constants::PI` == `std::f64::consts::PI`. `ferrum_core::constants::INF` == `f64::INFINITY`.

### Architecture

### Crate Layout
```
ferrum-core/
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
    error.rs                  # FerrumError enum
    buffer.rs                 # AsRawBuffer trait
    dynarray.rs               # DynArray runtime-typed enum
    record.rs                 # FerrumRecord derive macro support
    layout.rs                 # MemoryLayout enum
    prelude.rs                # use ferrum_core::prelude::*
  ferrum-core-macros/         # Proc macro crate for #[derive(FerrumRecord)] and s![]
    Cargo.toml
    src/lib.rs
```

### Key Design Decisions
- `ndarray` is a private dependency. `NdArray` wraps `ndarray::ArrayBase` internally but exposes its own API. This insulates users from ndarray version churn.
- The `Dimension` trait hierarchy mirrors ndarray's (`Ix1`..`Ix6`, `IxDyn`) but is re-defined in ferrum-core's namespace.
- Broadcasting is implemented as a standalone function that returns virtual shape/stride pairs, not new allocations.
- `FerrumRecord` proc macro lives in a separate `ferrum-core-macros` crate (Rust proc macro constraint).
- Do NOT use `std::simd` — use `pulp` for all SIMD. `std::simd` is unstable.

### Out of Scope

- SIMD-accelerated inner loops (handled by ferrum-ufunc)
- Mathematical functions beyond basic arithmetic operator overloads (handled by ferrum-ufunc)
- Statistics functions (handled by ferrum-stats)
- File I/O (handled by ferrum-io)
- Linear algebra (handled by ferrum-linalg)
- Random number generation (handled by ferrum-random)

### dependencies

- **Upstream**: None (this is the root of the dependency graph)
- **Downstream**: ferrum-ufunc, ferrum-stats, ferrum-io, ferrum-linalg, ferrum-fft, ferrum-random, ferrum-polynomial, ferrum-strings, ferrum-ma, ferrum-stride-tricks, ferrum-numpy-interop, ferrum (re-export)
- **External crates**: `ndarray` 0.17 (internal storage), `num-complex` (Complex types), `half` (f16, feature-gated), `thiserror` 2.0 (error derive), `serde` 1.0 (feature-gated serialization)

### phase

Phase 1 — Core Array and Ufuncs (BLOCKING: must complete before any other crate)

### agent splitting guidance

This crate is too large for a single agent. The coordinator SHOULD split it into sub-agents:
- **Agent 1a: core-types** (opus) — NdArray<T,D>, ownership model (REQ-1 through REQ-5), Dimension trait, MemoryLayout, DType system (REQ-6 through REQ-8), error handling (REQ-27, REQ-28), buffer interop (REQ-29, REQ-30), introspection (REQ-35, REQ-36), iterator/Display (REQ-37, REQ-38, REQ-39)
- **Agent 1b: core-indexing** (opus) — Broadcasting (REQ-9 through REQ-11), basic + advanced indexing (REQ-12 through REQ-15), extended indexing (REQ-15a)
- **Agent 1c: core-creation-manipulation** (sonnet) — Array creation (REQ-16 through REQ-19), shape manipulation (REQ-20 through REQ-22), extended manipulation (REQ-22a), constants/finfo (REQ-33, REQ-34)
- **Agent 1d: core-macros** (sonnet) — FerrumRecord proc macro (REQ-8), s![] macro, promoted_type! macro (REQ-23 through REQ-26)

Agent 1a must complete before 1b/1c/1d can start. 1b, 1c, 1d can run in parallel.

