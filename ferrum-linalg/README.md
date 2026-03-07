# ferrum-linalg

Linear algebra operations for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **Matrix products**: `dot`, `matmul`, `multi_dot`, `tensordot`, `kron`, `einsum`
- **Decompositions**: Cholesky, QR, SVD, LU, eigendecomposition (`eig`, `eigh`, `eigvals`)
- **Solvers**: `solve`, `lstsq`, `inv`, `pinv`, `matrix_power`, `tensorsolve`
- **Norms**: `norm`, `cond`, `det`, `slogdet`, `matrix_rank`, `trace`
- **Batched operations**: 3D+ stacked matrix operations with Rayon parallelism
- Powered by [faer](https://crates.io/crates/faer) — pure Rust, no BLAS dependency

## Usage

```rust
use ferrum_linalg::{matmul, det, solve};
use ferrum_core::prelude::*;

let a = Array2::<f64>::eye(3)?;
let d = det(&a)?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `linalg` feature.

## License

MIT OR Apache-2.0
