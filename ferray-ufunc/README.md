# ferray-ufunc

SIMD-accelerated universal functions for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **120+ elementwise ops**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil`, `round`, `clip`, `nan_to_num`, etc.
- **Binary operations**: `add`, `subtract`, `multiply`, `divide`, `power`, `floor_divide`, `mod_`, `gcd`, `lcm`, with broadcasting
- **20 complex transcendentals**: `sin_complex`, `cos_complex`, `tan_complex`, `sinh_complex`, `cosh_complex`, `tanh_complex`, plus `asin/acos/atan/asinh/acosh/atanh`, `exp_complex`, `ln_complex`, `log2_complex`, `log10_complex`, `sqrt_complex`, `power_complex`, plus precision-preserving `expm1_complex` and `log1p_complex` (cancellation-avoiding paths near z=0)
- **f32 SIMD parity**: `abs`, `neg`, `square`, `reciprocal`, `sqrt` all have first-class f32 SIMD kernels (8 lanes per AVX2 ymm) alongside the existing f64 kernels
- **CORE-MATH** for arctan/asin/sinh/cosh family (correctly rounded, ≤0.5 ULP)
- **libm** for sin/cos/tan (matches NumPy's path; ~2.4× faster per element than core-math, accuracy band ~1-2 ULP). Correctly-rounded variants reachable via `cr_math::CrMath::cr_sin`/`cr_cos`/`cr_tan`.
- **`exp_fast()`** — Even/Odd Remez decomposition, ~30% faster than CORE-MATH at ≤1 ULP accuracy
- Portable SIMD via `pulp` (SSE2/AVX2/AVX-512/NEON) on stable Rust
- Scalar fallback with `FERRAY_FORCE_SCALAR=1` environment variable
- Rayon parallelism for arrays > 100K elements on compute-bound ops

## Performance

vs NumPy 2.4.4 on Raptor Lake 14700K:

| Operation | Size | Speedup vs NumPy |
|-----------|-----:|-----------------:|
| sin / cos / tan | 1M | **4.2-4.8× faster** |
| arctan | 1M | **4.3× faster** |
| tanh | 1M | **2.8× faster** |
| exp / log | 1M | **1.7-1.8× faster** |
| sin / cos / tan | 100K | **1.2-1.5× faster** |
| sin / cos / tan | 1K | **1.0-1.2×** (parity-or-better) |

## Usage

```rust
use ferray_ufunc::{sin, exp, exp_fast, sin_complex};
use ferray_core::prelude::*;
use num_complex::Complex64;

let a = Array1::<f64>::linspace(0.0, 6.28, 1000)?;
let b = sin(&a)?;

// Default: libm-equivalent (matches NumPy, ~1-2 ULP)
let c = exp(&b)?;

// Fast mode (≤1 ULP, ~30% faster)
let c_fast = exp_fast(&b)?;

// Complex transcendentals
let z = Array1::<Complex64>::from_vec(/* ... */)?;
let sin_z = sin_complex(&z)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate.

## License

MIT OR Apache-2.0
