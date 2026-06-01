# ferray-linalg

`numpy.linalg` for the [ferray](../README.md) workspace — products, decompositions, solvers, norms, and eigenproblems for real and complex matrices.

Part of the [ferray](../README.md) workspace — backed by [faer](https://faer-rs.github.io/) with hand-tuned kernels.

## Overview

All functions return `Result<T, FerrayError>` and never panic. Most operate on `Array<T, Ix2>` (or `IxDyn`) where `T: LinalgFloat` (sealed to `f32` / `f64`); complex matrices are handled by the dedicated `complex::*` entry points. Stacked (3D+) inputs are dispatched per-batch with Rayon via the `*_batched` variants.

- **Products** — `dot`, `matmul`, `matvec`, `vecmat`, `vecdot`, `inner`, `outer`, `vdot`, `cross`, `kron`, `multi_dot`, `tensordot` (with `TensordotAxes`), and `einsum`.
- **Norms & measures** — `norm` / `norm_axis` / `vector_norm` / `matrix_norm` over every `NormOrder` mode (`Fro`, `Nuc`, `Inf`, `NegInf`, `L1`, `L2`, `NegL1`, `NegL2`, `P(p)`), plus `cond`, `det` / `slogdet`, `trace` / `trace_offset`, `diagonal`, `matrix_transpose`, and `matrix_rank`.
- **Solvers & inversion** — `solve` (and `solve_dyn`), `lstsq`, `inv`, `pinv` (with `rcond`), `matrix_power`, `tensorsolve`, and `tensorinv`. TRSM primitives live in the `trsm` module.
- **Decompositions** — `cholesky` / `cholesky_upper` (with `UPLO`), `qr` (with `QrMode::Reduced` / `Complete`), `svd` / `svdvals` / `svd_hermitian`, and `lu`.
- **Eigenproblems** — `eig`, `eigvals`, and the Hermitian/symmetric `eigh` / `eigvalsh` / `eigvalsh_uplo`.
- **Complex** — `matmul_complex`, `solve_complex`, `solve_complex_vec`, `inv_complex`, `det_complex` for `Array<Complex<T>, Ix2>`.

Batched siblings: `solve_batched`, `inv_batched`, `pinv_batched`, `matrix_power_batched`, `cholesky_batched`, `qr_batched`, `svd_batched`, `eigh_batched`, `eigvalsh_batched`, `det_batched`, `slogdet_batched`, `matrix_rank_batched`.

## NumPy correspondence

| NumPy | ferray-linalg |
|-------|---------------|
| `np.dot` | `dot` |
| `np.matmul` / `@` | `matmul` |
| `np.tensordot` | `tensordot` (`TensordotAxes`) |
| `np.linalg.solve` | `solve` |
| `np.linalg.inv` | `inv` |
| `np.linalg.lstsq` | `lstsq` |
| `np.linalg.pinv` | `pinv` |
| `np.linalg.qr` | `qr` (`QrMode`) |
| `np.linalg.svd` / `svdvals` | `svd` / `svdvals` |
| `np.linalg.eig` / `eigvals` | `eig` / `eigvals` |
| `np.linalg.eigh` / `eigvalsh` | `eigh` / `eigvalsh` |
| `np.linalg.cholesky` | `cholesky` (`UPLO`) |
| `np.linalg.det` / `slogdet` | `det` / `slogdet` |
| `np.linalg.norm` | `norm` (`NormOrder`) |
| `np.linalg.cond` | `cond` |
| `np.linalg.matrix_power` | `matrix_power` |
| `np.linalg.matrix_rank` | `matrix_rank` |
| `np.linalg.tensorsolve` / `tensorinv` | `tensorsolve` / `tensorinv` |

## Feature flags

| Feature | Effect |
|---------|--------|
| *(default)* | Hand-tuned + faer kernels only; no extra runtime deps, workspace MSRV 1.88. |
| `openblas` | Routes f32/f64/c32/c64 GEMM through system `cblas_*gemm` for a perf-floor guarantee. Expects a system OpenBLAS (`libopenblas-dev` or equivalent); pass `--features openblas-src/static` for vendored static linking. |
| `avx512` | AVX-512F DGEMM kernel (OpenBLAS Skylake-X port), runtime-detected. Bumps effective MSRV to 1.89. |
| `avxvnni` | AVX-VNNI i8 GEMM via `vpdpbusd`, runtime-detected. Bumps effective MSRV to 1.87. |
| `f16` | IEEE binary16 inputs → f32-accumulator GEMM (`gemm_f16_f32`). Pulls in `half`. |
| `bf16` | bfloat16 inputs → f32-accumulator GEMM (`gemm_bf16_f32`). Pulls in `half`. |

All accelerated GEMM features fall back to the portable AVX2 path when the CPU lacks the instruction set.

## Example

```rust
use ferray_core::{Array2, Array1};
use ferray_linalg::{solve, svd};

// Solve A x = b for a square system.
let a: Array2<f64> = Array2::from_shape_vec((2, 2), vec![3.0, 2.0, 1.0, 2.0])?;
let b: Array1<f64> = Array1::from_vec(vec![5.0, 5.0]);
let x = solve(&a, &b.into_dyn())?; // x ≈ [0, 2.5]

// Thin SVD: A = U · diag(s) · Vᵀ.
let (u, s, vt) = svd(&a, /* full_matrices = */ false)?;
println!("singular values: {:?}", s);
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Edition 2024, MSRV 1.88 (default features; `avxvnni` raises it to 1.87 and `avx512` to 1.89).
- Licensed under MIT OR Apache-2.0.

Re-exported through the umbrella [`ferray`](../README.md) crate as `ferray::linalg`.
