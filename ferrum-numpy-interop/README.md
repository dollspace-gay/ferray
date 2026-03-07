# ferrum-numpy-interop

Zero-copy conversions between ferrum arrays and Python/Arrow/Polars for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **NumPy** (feature `python`): `AsFerrum` and `IntoNumPy` traits via PyO3
- **Apache Arrow** (feature `arrow`): `FromArrow`, `ToArrow` conversion traits
- **Polars** (feature `polars`): `FromPolars`, `ToPolars` conversion traits
- Zero-copy when arrays are C-contiguous, safe copy otherwise
- Dtype validation on all conversions

All backends are feature-gated and disabled by default.

## Usage

```toml
[dependencies.ferrum-numpy-interop]
version = "0.1"
features = ["python"]  # or "arrow", "polars"
```

```rust
use ferrum_numpy_interop::{AsFerrum, IntoNumPy};
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

fn process(py: Python<'_>, arr: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
    let ferrum_arr = arr.as_ferrum()?;
    // ... process ...
    Ok(())
}
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `numpy` feature.

## License

MIT OR Apache-2.0
