# ferray-numpy-interop

The interop layer that bridges ferray arrays to the Python/NumPy ecosystem (PyO3 + the `numpy` crate) and to Apache Arrow / Polars, with a shared dtype mapping layer.

Part of the [ferray](../README.md) workspace.

## Overview

`ferray-numpy-interop` provides owning conversions between ferray's `NdArray` /
`Array1` / `Array2` / `ArrayD` and three external array ecosystems. Every
backend is gated behind a Cargo feature and **all are off by default** — the
default build pulls in nothing but `ferray-core`.

- **Dtype mapping** (`dtype_map`): bidirectional translation between ferray's
  `DType` and the external type systems. `dtype_to_arrow` / `arrow_to_dtype`
  (feature `arrow`) and `dtype_to_polars` / `polars_to_dtype` (feature
  `polars`) are the single source of truth for type correspondence. Each
  returns `FerrayError::InvalidDtype` for types with no equivalent (e.g.
  Arrow/Polars have no native complex, `I128`/`U128`, or bfloat16).
- **NumPy bridge** (`numpy_conv`, feature `python`): the `AsFerray` trait
  (`PyReadonlyArray{1,2,Dyn}` → ferray) and the `IntoNumPy` trait
  (ferray → `PyArray`), built on PyO3 and the `numpy` crate.
- **Arrow bridge** (`arrow_conv` + `extras`, feature `arrow`): the
  `FromArrow` / `ToArrow` (and `…Bool`) traits plus helpers
  `array2_to_arrow_columns` / `array2_from_arrow_columns`,
  `arrayd_to_arrow_flat` / `arrayd_from_arrow_flat`,
  `record_batch_from_columns` / `record_batch_to_columns`, and
  null/validity-mask variants (`array1_to_arrow_with_mask`,
  `arrow_to_dynarray_with_mask`, `dynarray_to_arrow{,_with_mask}`).
- **Polars bridge** (`polars_conv` + `extras`, feature `polars`): the
  `FromPolars` / `ToPolars` (and `…Bool`) traits plus
  `array2_to_polars_dataframe` / `array2_from_polars_dataframe` and
  `dataframe_from_columns`.

Conversions copy the data buffer (one `memcpy` per conversion, not per
element); the crate is explicit about this rather than claiming zero-copy.
See the `//!` module docs in `src/lib.rs` for the per-path rationale.

## Feature flags

| Feature | Enables | Notes |
|---------|---------|-------|
| *(default)* | nothing | `default = []`; only `ferray-core` is pulled in. |
| `python` | `dep:pyo3`, `dep:numpy` | `AsFerray` / `IntoNumPy` NumPy bridge via PyO3. |
| `arrow` | `dep:arrow` | `FromArrow` / `ToArrow`, `dtype_to_arrow` / `arrow_to_dtype`, RecordBatch round-trip, validity masks. |
| `polars` | `dep:polars` | `FromPolars` / `ToPolars`, `dtype_to_polars` / `polars_to_dtype`, DataFrame round-trip. |
| `f16` | `dep:half`, `ferray-core/f16` | Maps `DType::F16` (IEEE binary16) in the dtype layer; round-trips to Arrow `Float16`. |
| `bf16` | `dep:half`, `ferray-core/bf16` | Enables `DType::BF16` mapping; Arrow has no native bf16, so `dtype_to_arrow` returns `InvalidDtype` with a hint. |
| `complex` | `dep:num-complex` | `Complex<f32>` / `Complex<f64>` dtype mapping. Arrow has no complex type; with `arrow` + `complex` the crate exposes `complex32_to_arrow` / `complex64_to_arrow` and `arrow_to_complex32` / `arrow_to_complex64` (FixedSizeList `[real, imag]` workaround). |

## Example

The dtype mapping layer is the most stable surface. Guard the Arrow path
behind `--features arrow`:

```rust
// cargo build -p ferray-numpy-interop --features arrow
use ferray_numpy_interop::{dtype_to_arrow, arrow_to_dtype};
use ferray_core::DType;

let arrow_dt = dtype_to_arrow(DType::F64)?;          // arrow::datatypes::DataType::Float64
let back = arrow_to_dtype(&arrow_dt)?;               // DType::F64
assert_eq!(back, DType::F64);

// Types with no Arrow equivalent surface a clear error:
assert!(dtype_to_arrow(DType::Complex64).is_err());
# Ok::<(), ferray_core::FerrayError>(())
```

The NumPy bridge (`--features python`) hangs off the `AsFerray` trait:

```rust
// cargo build -p ferray-numpy-interop --features python
use ferray_numpy_interop::AsFerray;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

fn process(arr: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
    let a = arr.as_ferray()?;   // ferray Array1<f64>
    let _ = a;
    Ok(())
}
```

## MSRV & edition

- Rust edition 2024, MSRV 1.88.
- Licensed under MIT OR Apache-2.0.

## Testing note

`cargo test --workspace` links cleanly because the `pyo3/extension-module`
feature is gated behind ferray-python's own feature, not enabled here. To run
this crate's PyO3-backed tests, pass the feature explicitly:

```bash
cargo test -p ferray-numpy-interop --features python
cargo test -p ferray-numpy-interop --features arrow,polars,f16,bf16,complex
```
