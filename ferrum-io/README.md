# ferrum-io

NumPy-compatible file I/O for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **`.npy`** read/write — single array serialization (all dtypes, C/Fortran order)
- **`.npz`** read/write — compressed archives of named arrays
- **Memory mapping** — `mmap_npy` for zero-copy access to large files
- **Text I/O** — `loadtxt`, `savetxt`, `genfromtxt` with delimiter/comment/skip support
- Structured dtype support for compound types

## Usage

```rust
use ferrum_io::{save_npy, load_npy};
use ferrum_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 1.0, 1000)?;
save_npy("data.npy", &a)?;
let b: Array1<f64> = load_npy("data.npy")?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `io` feature (enabled by default).

## License

MIT OR Apache-2.0
