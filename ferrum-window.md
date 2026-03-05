---
title: ferrum-window — Window functions for signal processing and spectral analysis
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

Implements NumPy's window functions: `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`. These are used in signal processing, spectral analysis, and FFT windowing. Also includes `vectorize` and other functional programming utilities (`piecewise`, `apply_along_axis`, `apply_over_axes`) that are missing from other ferrum crates.

### Requirements

- REQ-1: `window::bartlett(m)` — Bartlett (triangular) window of length M, returning `Array1<f64>`
- REQ-2: `window::blackman(m)` — Blackman window
- REQ-3: `window::hamming(m)` — Hamming window
- REQ-4: `window::hanning(m)` — Hann window (NumPy calls it `hanning`)
- REQ-5: `window::kaiser(m, beta)` — Kaiser window with shape parameter beta (uses `i0` Bessel function from ferrum-ufunc)
- REQ-6: All window functions return `Array1<f64>` of length M. For M <= 1, return `[1.0]` or `[]` matching NumPy behavior. ### Functional Programming Utilities
- REQ-7: `vectorize(f)` — wrap a scalar function `Fn(T) -> U` to operate elementwise on arrays. Returns a closure that accepts `&NdArray<T, D>` and returns `NdArray<U, D>`. This is NumPy's `np.vectorize`.
- REQ-8: `piecewise(&x, condlist, funclist, default)` — evaluate a piecewise-defined function. Each condition is a boolean array, each function maps the corresponding elements.
- REQ-9: `apply_along_axis(func, axis, &a)` — apply a function along one axis of an array. The function receives 1D slices and returns 1D results.
- REQ-10: `apply_over_axes(func, &a, axes)` — apply a function repeatedly over multiple axes

### Acceptance Criteria

- [ ] AC-1: `bartlett(5)` matches `np.bartlett(5)` to within 4 ULPs
- [ ] AC-2: `kaiser(5, 14.0)` matches `np.kaiser(5, 14.0)` to within 4 ULPs
- [ ] AC-3: All 5 window functions return correct length and match NumPy fixtures
- [ ] AC-4: `vectorize(|x: f64| x.powi(2))(&array)` produces element-squared array identical to `square(&array)`
- [ ] AC-5: `apply_along_axis(|col| col.sum(), Axis(0), &matrix)` produces column sums matching `matrix.sum(axis=0)`
- [ ] AC-6: `cargo test -p ferrum-window` passes. `cargo clippy` clean.

### Architecture

### Crate Layout
```
ferrum-window/
  Cargo.toml
  src/
    lib.rs
    windows/
      mod.rs                  # bartlett, blackman, hamming, hanning, kaiser
    functional.rs             # vectorize, piecewise, apply_along_axis, apply_over_axes
```

### Design Notes
- Window functions are pure math — each is a closed-form formula evaluated over `linspace(0, 1, m)` or similar.
- `kaiser` requires the modified Bessel function `i0`, which is provided by ferrum-ufunc.
- `vectorize` in Rust is simpler than NumPy's version since Rust closures are already typed. It's essentially `.mapv()` wrapped as a reusable callable.

### Out of Scope

- Advanced window functions (Tukey, flat-top, etc. — post-1.0)
- scipy.signal window functions

### dependencies

- **Upstream**: `ferrum-core` (NdArray, Array1, Element, FerrumError), `ferrum-ufunc` (math functions for window computation)
- **Downstream**: ferrum (re-export as `ferrum::window::`)
- **Phase**: 2 — Submodules (ships alongside fft, random, polynomial)

