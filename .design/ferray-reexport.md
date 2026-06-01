# Feature: ferray — Main re-export crate providing `use ferray::prelude::*`

## Summary
The top-level `ferray` crate that re-exports all subcrates into a unified namespace. Users depend only on `ferray` — they never need to know about `ferray-core`, `ferray-ufunc`, etc. The prelude provides `use ferray::prelude::*` covering 95% of use cases, matching NumPy's `import numpy as np` experience. Feature flags control which submodules are compiled.

## Dependencies
- **Upstream**: All ferray subcrates (ferray-core, ferray-ufunc, ferray-stats, ferray-io, ferray-linalg, ferray-fft, ferray-random, ferray-polynomial, ferray-strings, ferray-ma, ferray-stride-tricks, ferray-window)
- **Downstream**: End users, ferray-numpy-interop
- **Phase**: Assembled incrementally — Phase 1 integration adds core+ufunc+stats+io, Phase 2 adds linalg+fft+random+polynomial, Phase 3 adds strings+ma+stride-tricks

## Requirements

### Namespace Structure
- REQ-1: Top-level namespace `ferray::` contains all array creation functions (zeros, ones, array, arange, linspace, etc.) and free-function equivalents of array methods
- REQ-2: Submodule namespaces: `ferray::linalg::`, `ferray::fft::`, `ferray::random::`, `ferray::polynomial::`, `ferray::strings::`, `ferray::ma::`, `ferray::io::`, `ferray::lib::stride_tricks::`, `ferray::window::`
- REQ-3: `ferray::prelude::*` exports: `NdArray`, `Array1`-`Array3`, `ArrayD`, `ArrayView`, `ArrayViewMut`, `ArcArray`, `CowArray`, `Axis`, `s![]` macro, `Element` trait, `FerrayError`, all array creation functions, all common math functions
- REQ-3a: `ferray::` re-exports all constants from `ferray_core::constants`: `ferray::PI`, `ferray::E`, `ferray::INF`, `ferray::NAN`, `ferray::EULER_GAMMA`, `ferray::NEWAXIS`, `ferray::PZERO`, `ferray::NZERO`, `ferray::NEG_INF`

### Feature Flags (Section 24)
- REQ-4: Implement feature flags: `full`, `blas`, `rayon` (default), `simd` (default), `complex` (default), `f16`, `strings` (default), `ma` (default), `io` (default), `window` (default), `serde`, `arrow`, `polars`, `numpy`, `no_std`, `nightly-simd`
- REQ-5: `default = ["rayon", "simd", "complex", "strings", "ma", "io"]`

### Configuration
- REQ-6: `ferray::set_num_threads(n)` — configure the ferray-owned Rayon thread pool (initialized once via `OnceLock`, subsequent calls are no-ops or return an error)
- REQ-7: `ferray::with_num_threads(n, || { ... })` — scoped execution on a cached thread pool. Do NOT create a new ThreadPool per call — use a pool cache (e.g., `DashMap<usize, Arc<ThreadPool>>`) to avoid the extreme cost of repeated thread creation.
- REQ-8: Expose parallel threshold constants: `ferray::config::PARALLEL_THRESHOLD_ELEMENTWISE`, etc.

### Workspace Cargo.toml
- REQ-9: Define the Cargo workspace with all subcrates and proper inter-crate dependency versions (path dependencies)

## Acceptance Criteria
- [ ] AC-1: `use ferray::prelude::*` compiles and provides access to array creation, math functions, and core types without additional imports
- [ ] AC-2: `ferray::linalg::matmul(&a, &b)` works without directly depending on `ferray-linalg`
- [ ] AC-3: Disabling the `strings` feature flag removes `ferray::strings` module entirely (compile error on use)
- [ ] AC-4: `cargo build -p ferray --no-default-features` compiles (core-only, no rayon/simd/strings/ma/io)
- [ ] AC-5: `cargo doc -p ferray` produces unified documentation with all submodules visible
- [ ] AC-6: `cargo test --workspace` passes after each phase integration
- [ ] AC-7: `ferray::PI` == `std::f64::consts::PI`. `ferray::INF` == `f64::INFINITY`. Constants are accessible from the top-level namespace.
- [ ] AC-8: `ferray::with_num_threads(2, || ...)` called 100 times in a loop does not create 100 thread pools (verified by pool cache hit)

## Architecture

### Crate Layout
```
ferray/
  Cargo.toml                  # Feature flags, re-export dependencies
  src/
    lib.rs                    # pub use ferray_core::*; pub mod linalg; etc.
    prelude.rs                # Curated re-exports for use ferray::prelude::*
    config.rs                 # set_num_threads, with_num_threads, threshold constants, pool cache
    constants.rs              # Re-export ferray_core::constants as ferray::PI, ferray::INF, etc.
```

### Workspace Cargo.toml (root)
```toml
[workspace]
resolver = "2"
members = [
    "ferray",
    "ferray-core",
    "ferray-core-macros",
    "ferray-ufunc",
    "ferray-stats",
    "ferray-io",
    "ferray-linalg",
    "ferray-fft",
    "ferray-random",
    "ferray-polynomial",
    "ferray-window",
    "ferray-strings",
    "ferray-ma",
    "ferray-stride-tricks",
    "ferray-numpy-interop",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.88"
license = "MIT OR Apache-2.0"
```

## ferray-python top-level alias surface (expansion batch 1)

NumPy exposes a set of top-level *alias* and *introspection* functions at
its package root. The `ferray-python` shim mirrors them so
`import ferray as np` keeps drop-in parity. Two implementation patterns,
matching how numpy itself defines them:

- **Pure 1:1 aliases** — numpy defines these as bare Python assignments
  (`numpy/_core/__init__.py` `acos = numeric.arccos`). ferray mirrors them
  as re-exports of the already-bound `#[pyfunction]` in
  `python/ferray/__init__.py`. Covers `acos/acosh/asin/asinh/atan/atanh/
  atan2`, `bitwise_invert/bitwise_left_shift/bitwise_right_shift`,
  `concat`, `permute_dims`, `pow`, `amax`, `amin`, `cumulative_sum`,
  `cumulative_prod`.
- **Boundary introspection** — `#[pyfunction]`s in `aliases.rs` operating
  on the numpy `ndarray`/`dtype` objects the bindings already produce.
  Covers `ndim/shape/size/isscalar/isfortran` (array introspection) and
  `astype/can_cast/promote_types/result_type/min_scalar_type/issubdtype/
  isdtype/common_type` (dtype introspection), plus the `divmod` ufunc
  (composed from the bound `floor_divide`/`mod` ufuncs to keep the
  integer-dtype tuple contract).

### REQ status (ferray-python alias surface)

- REQ-PY-ALIAS-1 SHIPPED — array introspection `ndim`/`shape`/`size`/
  `isscalar`/`isfortran` as `#[pyfunction]` in `aliases.rs` (`pub fn ndim`,
  `pub fn shape`, `pub fn size`, `pub fn isscalar`, `pub fn isfortran`);
  registered + consumed at the extension surface in `lib.rs` (`_ferray`
  fn, `m.add_function(wrap_pyfunction!(aliases::ndim, m)?)`) and re-exported
  in `python/ferray/__init__.py`. Verified `tests/test_expansion_aliases.py`.
- REQ-PY-ALIAS-2 SHIPPED — dtype introspection `astype`/`can_cast`/
  `promote_types`/`result_type`/`min_scalar_type`/`issubdtype`/`isdtype`/
  `common_type` as `#[pyfunction]` in `aliases.rs`; registered + consumed
  in `lib.rs` (`_ferray` fn) and re-exported in `__init__.py`. Verified
  `tests/test_expansion_aliases.py`.
- REQ-PY-ALIAS-3 SHIPPED — `divmod` ufunc tuple in `aliases.rs`
  (`pub fn divmod`, composing `ufunc::floor_divide`/`ufunc::mod_`);
  registered in `lib.rs`, re-exported in `__init__.py`. Verified
  `tests/test_expansion_aliases.py::test_divmod_int`.
- REQ-PY-ALIAS-4 SHIPPED — pure trig/bitwise/manip/reduction aliases as
  `__init__.py` re-exports (`acos = arccos`, … `cumulative_prod = cumprod`),
  consumed by `__all__`. Verified `tests/test_expansion_aliases.py`.
- REQ-PY-ALIAS-5 NOT-STARTED — `around` (numpy `round(a, decimals)`): the
  bound ferray `round` is unary and lacks the `decimals` argument, so it
  cannot alias numpy's `around(a, decimals=0)` without divergence on
  non-zero decimals. Needs `decimals` support in the ferray-ufunc `round`
  binding (follow-up builder).
- REQ-PY-ALIAS-6 NOT-STARTED — `all`/`any`/`nancumsum`/`nancumprod`/
  `nanpercentile`/`nanquantile`: the ferray-core `all`/`any` reductions and
  ferray-stats nan-cumulative fns exist but are not bound through
  ferray-python (no `stats.rs` binding); `all`/`any` also lack an `axis`
  kwarg. Needs new `stats.rs` bindings (out of this manifest — follow-up
  builder).

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Publishing to crates.io (manual human step)
- CI/CD pipeline configuration
