# ferray-numpy-interop

Zero-copy conversions between ferray arrays and Python/Arrow/Polars for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **NumPy** (feature `python`): `AsFerray` and `IntoNumPy` traits via PyO3
- **Apache Arrow** (feature `arrow`): `FromArrow`/`ToArrow` traits + `RecordBatch` round-trip + null/missing handling
- **Polars** (feature `polars`): `FromPolars`/`ToPolars` DataFrame round-trip
- **Dtype coverage**: f32/f64, i8-i128, u8-u128, bool, **f16**, **bf16**, **complex<f32>**, **complex<f64>** — all with dtype validation on conversion
- **`DynArray` integration**: convert `Array<T, IxDyn>` to/from PyArrayDyn / Arrow dynamic-typed columns
- Zero-copy when arrays are C-contiguous, safe copy otherwise

All backends are feature-gated and disabled by default.

## Usage

```toml
[dependencies.ferray-numpy-interop]
version = "0.1"
features = ["python"]  # or "arrow", "polars"
```

```rust
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

fn process(py: Python<'_>, arr: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
    let ferray_arr = arr.as_ferray()?;
    // ... process ...
    Ok(())
}
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `numpy` feature.

## License

MIT OR Apache-2.0
