# ferray-linalg

Linear algebra operations for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **Matrix products**: `dot`, `matmul`, `multi_dot`, `tensordot`, `kron`, `einsum`, `vecdot`, `matvec`, `vecmat` (NumPy 2.0 gufunc-style)
- **Decompositions**: Cholesky, QR, SVD, LU, eigendecomposition (`eig`, `eigh`, `eigvals`)
- **Solvers**: `solve`, `lstsq`, `inv`, `pinv`, `matrix_power`, `tensorsolve`, plus native TRSM primitives (`trsm_left_lower_f64`, `trsm_left_upper_f64`, f32 siblings)
- **Norms**: `norm`, `cond`, `det`, `slogdet`, `matrix_rank`, `trace`
- **Batched operations**: 3D+ stacked matrix operations with Rayon parallelism
- **Hand-tuned GEMM kernels** (`crate::gemm` module):
  - `gemm_f64` / `gemm_f32` — DGEMM / SGEMM, AVX2+FMA Haswell port (4×8 / 4×16 tiles), AVX-512 Skylake-X port behind `avx512` feature
  - `gemm_c64` / `gemm_c32` — ZGEMM / CGEMM Haswell + Skylake-X ports
  - `gemm_i16`, `gemm_i8`, `gemm_i8_signed` — quantized integer GEMM with AVX-VNNI fast path
  - `quantized_matmul` — full ONNX `QLinearMatMul` / oneDNN `s8s8s32` formula with zero-points + scales
  - `gemm_bf16_f32` / `gemm_f16_f32` — mixed-precision (behind `bf16` / `f16` features)
  - Optional system-OpenBLAS backend behind `openblas` cargo feature

## Performance

Hand-tuned single-process peak GFLOPS on a Raptor Lake 14700K with the dedicated 8-thread P-core pool:

| Kernel | Peak GFLOPS | At N |
|---|---:|---:|
| DGEMM (f64) | 260 | 1024 |
| SGEMM (f32) | 987 | 1536 |
| ZGEMM (c64) | 456 | 1024 |
| CGEMM (c32) | 852 | 1024 |
| i8 quantized | 1488 GOPS | 1024 (AVX-VNNI) |

Beats NumPy MKL at matmul 10×10 (4.9×), matmul 50×50 (1.05×), and most complex/quantized sizes; competitive with MKL elsewhere. See the [BENCHMARKS.md](../BENCHMARKS.md) report for full numbers.

## Usage

```rust
use ferray_linalg::{matmul, det, solve};
use ferray_core::prelude::*;

let a = Array2::<f64>::eye(3)?;
let d = det(&a)?;
```

For low-level GEMM access:

```rust
use ferray_linalg::gemm::{gemm_f64, gemm_i8};

// 1024x1024 DGEMM via the hand-tuned kernel
let mut c = vec![0.0_f64; 1024 * 1024];
gemm_f64(1024, 1024, 1024, 1.0, &a, &b, 0.0, &mut c);

// Quantized i8 inference (u8 activations × i8 weights → i32)
unsafe { gemm_i8(m, n, k, a_u8.as_ptr(), b_i8.as_ptr(), c_i32.as_mut_ptr(), false); }
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate.

## License

MIT OR Apache-2.0
