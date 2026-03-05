---
title: ferrum-numpy-interop — PyO3 zero-copy interop with NumPy and Arrow/Polars
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-05
updated: 2026-03-05
---


## Design Specification

### Summary

A companion crate (feature-flagged, not part of the main ferrum crate) providing zero-copy conversion between `ferrum::Array` and NumPy arrays via PyO3, and between ferrum arrays and Apache Arrow `ArrayData` / Polars `Series`. Enables ferrum to be used as the compute backend for Python extensions and data pipeline integrations.

### Requirements

- REQ-1: `np_array.as_ferrum::<T, D>()` — zero-copy conversion from `PyReadonlyArray` to `ferrum::ArrayView`. Fails if dtype or dimension mismatch.
- REQ-2: `ferrum_array.into_pyarray(py)` — zero-copy conversion from owned `ferrum::Array` to `PyArray`. Data ownership transfers to Python.
- REQ-3: Support all NumPy numeric dtypes: f32, f64, i32, i64, u8, u32, bool, Complex64, Complex128 ### Arrow Interop (Section 21.3)
- REQ-4: `ferrum_array.to_arrow()` — convert 1D ferrum array to `arrow::array::PrimitiveArray`. Zero-copy for C-contiguous arrays.
- REQ-5: `arrow_array.into_ferrum::<T>()` — convert Arrow array to ferrum Array1. Zero-copy where possible. ### Polars Interop
- REQ-6: `ferrum_array.to_polars_series(name)` — convert to `polars::Series`
- REQ-7: `series.into_ferrum::<T>()` — convert Polars Series to ferrum Array1 ### Safety
- REQ-8: All conversions validate dtype and layout compatibility before returning. No silent reinterpretation of memory.

### Acceptance Criteria

- [ ] AC-1: A Python extension using PyO3 can accept a NumPy array, process it via ferrum, and return a NumPy array — all zero-copy
- [ ] AC-2: Arrow round-trip: `ferrum -> arrow -> ferrum` produces bit-identical data
- [ ] AC-3: dtype mismatch returns a clear error, not UB or silent corruption
- [ ] AC-4: `cargo test -p ferrum-numpy-interop` passes with Python available. `cargo clippy` clean.

### Architecture

### Crate Layout
```
ferrum-numpy-interop/
  Cargo.toml                  # Feature flags: numpy (default), arrow, polars
  src/
    lib.rs
    numpy_conv.rs             # PyO3 NumPy <-> ferrum conversions
    arrow_conv.rs             # Arrow <-> ferrum conversions
    polars_conv.rs            # Polars <-> ferrum conversions
    dtype_map.rs              # Mapping between ferrum DType, NumPy dtype, Arrow DataType
```

### Out of Scope

- Python-side ferrum API (ferrum is a Rust library; Python bindings are for interop only)
- GPU array interop (cupy, jax — post-1.0)

### dependencies

- **Upstream**: `ferrum-core` (NdArray, AsRawBuffer, DType, FerrumError)
- **Downstream**: None (optional companion crate)
- **External crates**: `pyo3` 0.23 (Python binding), `numpy` (PyO3 NumPy bindings), `arrow` 54 (Arrow interop, feature-gated), `polars` (Polars interop, feature-gated)
- **Phase**: 3 — Completeness

