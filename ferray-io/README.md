# ferray-io

NumPy-compatible file I/O for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **`.npy`** read/write — single array serialization (all dtypes, C/Fortran order)
- **`.npz`** read/write — compressed archives of named arrays
- **Memory mapping** — `mmap_npy` for zero-copy access to large files
- **Text I/O** — `loadtxt`, `savetxt`, `genfromtxt`, `fromregex` with delimiter/comment/skip support
- **Format helpers** (`lib.format`): `descr_to_dtype`, `header_data_from_array_1_0`, `read_array`, `write_array` for low-level header / dtype manipulation
- **`DataSource`** — URL / compressed-file abstraction (handles `.gz`, `.bz2`, HTTP-fetched inputs transparently)
- Structured dtype support for compound types

## Usage

```rust
use ferray_io::{save_npy, load_npy};
use ferray_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 1.0, 1000)?;
save_npy("data.npy", &a)?;
let b: Array1<f64> = load_npy("data.npy")?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `io` feature (enabled by default).

## License

MIT OR Apache-2.0
