# Feature: ferray-numpy-interop — PyO3 zero-copy interop with NumPy and Arrow/Polars

## Summary
A companion crate (feature-flagged, not part of the main ferray crate) providing zero-copy conversion between `ferray::Array` and NumPy arrays via PyO3, and between ferray arrays and Apache Arrow `ArrayData` / Polars `Series`. Enables ferray to be used as the compute backend for Python extensions and data pipeline integrations.

## Dependencies
- **Upstream**: `ferray-core` (NdArray, AsRawBuffer, DType, FerrayError)
- **Downstream**: None (optional companion crate)
- **External crates**: `pyo3` 0.23 (Python binding), `numpy` (PyO3 NumPy bindings), `arrow` 54 (Arrow interop, feature-gated), `polars` (Polars interop, feature-gated)
- **Phase**: 3 — Completeness

## Requirements

### NumPy Interop (Section 21.2)
- REQ-1: `np_array.as_ferray::<T, D>()` — zero-copy conversion from `PyReadonlyArray` to `ferray::ArrayView`. Fails if dtype or dimension mismatch.
- REQ-2: `ferray_array.into_pyarray(py)` — zero-copy conversion from owned `ferray::Array` to `PyArray`. Data ownership transfers to Python.
- REQ-3: Support all NumPy numeric dtypes: f32, f64, i32, i64, u8, u32, bool, Complex64, Complex128

### Arrow Interop (Section 21.3)
- REQ-4: `ferray_array.to_arrow()` — convert 1D ferray array to `arrow::array::PrimitiveArray`. Zero-copy for C-contiguous arrays.
- REQ-5: `arrow_array.into_ferray::<T>()` — convert Arrow array to ferray Array1. Zero-copy where possible.

### Polars Interop
- REQ-6: `ferray_array.to_polars_series(name)` — convert to `polars::Series`
- REQ-7: `series.into_ferray::<T>()` — convert Polars Series to ferray Array1

### Safety
- REQ-8: All conversions validate dtype and layout compatibility before returning. No silent reinterpretation of memory.

## Acceptance Criteria
- [ ] AC-1: A Python extension using PyO3 can accept a NumPy array, process it via ferray, and return a NumPy array — all zero-copy
- [ ] AC-2: Arrow round-trip: `ferray -> arrow -> ferray` produces bit-identical data
- [ ] AC-3: dtype mismatch returns a clear error, not UB or silent corruption
- [ ] AC-4: `cargo test -p ferray-numpy-interop` passes with Python available. `cargo clippy` clean.

## Architecture

### Crate Layout
```
ferray-numpy-interop/
  Cargo.toml                  # Feature flags: numpy (default), arrow, polars
  src/
    lib.rs
    numpy_conv.rs             # PyO3 NumPy <-> ferray conversions
    arrow_conv.rs             # Arrow <-> ferray conversions
    polars_conv.rs            # Polars <-> ferray conversions
    dtype_map.rs              # Mapping between ferray DType, NumPy dtype, Arrow DataType
```

## Open Questions

*None — all design decisions resolved.*

## Out of Scope
- Python-side ferray API (ferray is a Rust library; Python bindings are for interop only)
- GPU array interop (cupy, jax — post-1.0)
