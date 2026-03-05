---
title: ferrum — Main re-export crate providing `use ferrum::prelude::*`
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

The top-level `ferrum` crate that re-exports all subcrates into a unified namespace. Users depend only on `ferrum` — they never need to know about `ferrum-core`, `ferrum-ufunc`, etc. The prelude provides `use ferrum::prelude::*` covering 95% of use cases, matching NumPy's `import numpy as np` experience. Feature flags control which submodules are compiled.

### Requirements

- REQ-1: Top-level namespace `ferrum::` contains all array creation functions (zeros, ones, array, arange, linspace, etc.) and free-function equivalents of array methods
- REQ-2: Submodule namespaces: `ferrum::linalg::`, `ferrum::fft::`, `ferrum::random::`, `ferrum::polynomial::`, `ferrum::strings::`, `ferrum::ma::`, `ferrum::io::`, `ferrum::lib::stride_tricks::`, `ferrum::window::`
- REQ-3: `ferrum::prelude::*` exports: `NdArray`, `Array1`-`Array3`, `ArrayD`, `ArrayView`, `ArrayViewMut`, `ArcArray`, `CowArray`, `Axis`, `s![]` macro, `Element` trait, `FerrumError`, all array creation functions, all common math functions
- REQ-3a: `ferrum::` re-exports all constants from `ferrum_core::constants`: `ferrum::PI`, `ferrum::E`, `ferrum::INF`, `ferrum::NAN`, `ferrum::EULER_GAMMA`, `ferrum::NEWAXIS`, `ferrum::PZERO`, `ferrum::NZERO`, `ferrum::NEG_INF` ### Feature Flags (Section 24)
- REQ-4: Implement feature flags: `full`, `blas`, `rayon` (default), `simd` (default), `complex` (default), `f16`, `strings` (default), `ma` (default), `io` (default), `window` (default), `serde`, `arrow`, `polars`, `numpy`, `no_std`, `nightly-simd`
- REQ-5: `default = ["rayon", "simd", "complex", "strings", "ma", "io"]` ### Configuration
- REQ-6: `ferrum::set_num_threads(n)` — configure the ferrum-owned Rayon thread pool (initialized once via `OnceLock`, subsequent calls are no-ops or return an error)
- REQ-7: `ferrum::with_num_threads(n, || { ... })` — scoped execution on a cached thread pool. Do NOT create a new ThreadPool per call — use a pool cache (e.g., `DashMap<usize, Arc<ThreadPool>>`) to avoid the extreme cost of repeated thread creation.
- REQ-8: Expose parallel threshold constants: `ferrum::config::PARALLEL_THRESHOLD_ELEMENTWISE`, etc. ### Workspace Cargo.toml
- REQ-9: Define the Cargo workspace with all subcrates and proper inter-crate dependency versions (path dependencies)

### Acceptance Criteria

- [ ] AC-1: `use ferrum::prelude::*` compiles and provides access to array creation, math functions, and core types without additional imports
- [ ] AC-2: `ferrum::linalg::matmul(&a, &b)` works without directly depending on `ferrum-linalg`
- [ ] AC-3: Disabling the `strings` feature flag removes `ferrum::strings` module entirely (compile error on use)
- [ ] AC-4: `cargo build -p ferrum --no-default-features` compiles (core-only, no rayon/simd/strings/ma/io)
- [ ] AC-5: `cargo doc -p ferrum` produces unified documentation with all submodules visible
- [ ] AC-6: `cargo test --workspace` passes after each phase integration
- [ ] AC-7: `ferrum::PI` == `std::f64::consts::PI`. `ferrum::INF` == `f64::INFINITY`. Constants are accessible from the top-level namespace.
- [ ] AC-8: `ferrum::with_num_threads(2, || ...)` called 100 times in a loop does not create 100 thread pools (verified by pool cache hit)

### Architecture

### Crate Layout
```
ferrum/
  Cargo.toml                  # Feature flags, re-export dependencies
  src/
    lib.rs                    # pub use ferrum_core::*; pub mod linalg; etc.
    prelude.rs                # Curated re-exports for use ferrum::prelude::*
    config.rs                 # set_num_threads, with_num_threads, threshold constants, pool cache
    constants.rs              # Re-export ferrum_core::constants as ferrum::PI, ferrum::INF, etc.
```

### Workspace Cargo.toml (root)
```toml
[workspace]
resolver = "2"
members = [
    "ferrum",
    "ferrum-core",
    "ferrum-core-macros",
    "ferrum-ufunc",
    "ferrum-stats",
    "ferrum-io",
    "ferrum-linalg",
    "ferrum-fft",
    "ferrum-random",
    "ferrum-polynomial",
    "ferrum-window",
    "ferrum-strings",
    "ferrum-ma",
    "ferrum-stride-tricks",
    "ferrum-numpy-interop",
]

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.85"
license = "MIT OR Apache-2.0"
```

### Out of Scope

- Publishing to crates.io (manual human step)
- CI/CD pipeline configuration

### dependencies

- **Upstream**: All ferrum subcrates (ferrum-core, ferrum-ufunc, ferrum-stats, ferrum-io, ferrum-linalg, ferrum-fft, ferrum-random, ferrum-polynomial, ferrum-strings, ferrum-ma, ferrum-stride-tricks, ferrum-window)
- **Downstream**: End users, ferrum-numpy-interop
- **Phase**: Assembled incrementally — Phase 1 integration adds core+ufunc+stats+io, Phase 2 adds linalg+fft+random+polynomial, Phase 3 adds strings+ma+stride-tricks

