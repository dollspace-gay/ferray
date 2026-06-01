# ferray-ufunc

NumPy-equivalent element-wise universal functions (ufuncs) and their reductions for the ferray N-dimensional array library.

Part of the [ferray](../README.md) workspace — powered by `pulp` runtime SIMD dispatch (SSE2/AVX2/AVX-512/NEON).

## Overview

ufuncs are generic free functions that preserve the input array's element type
and rank: `sin` on an `Array2<f64>` returns an `Array2<f64>`. The crate ships
100+ such functions, organized into families under `src/ops/`:

- **Arithmetic** — `add`, `subtract`, `multiply`, `divide`, `true_divide`,
  `floor_divide`, `power`, `remainder`, `mod_`, `fmod`, `divmod`, `negative`,
  `absolute`, `sign`, `reciprocal`, `square`, `sqrt`, `cbrt`, `gcd`, `lcm`,
  `cross`, plus cumulative ops (`cumsum`, `cumprod`, `diff`, `ediff1d`,
  `gradient`, `trapezoid`) and `add_reduce` / `nan_*_reduce` reductions.
- **Trigonometric** (`ops::trig`) — `sin`, `cos`, `tan`, `arcsin`, `arccos`,
  `arctan`, `arctan2`, `hypot`, `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`,
  `arctanh`, `degrees`/`radians`, `deg2rad`/`rad2deg`, `unwrap`, plus fast
  paths `sin_fast` / `cos_fast`.
- **Exp / log** (`ops::explog`) — `exp`, `exp2`, `expm1`, `exp_fast`, `log`,
  `log2`, `log10`, `log1p`, `logaddexp`, `logaddexp2`.
- **Rounding** (`ops::rounding`) — `floor`, `ceil`, `trunc`, `rint`, `round`,
  `around`, `fix` (with integer/bool-preserving `*_int` variants per REQ-24).
- **Float intrinsics** (`ops::floatintrinsic`) — `signbit`, `copysign`,
  `nextafter`, `spacing`, `frexp`, `ldexp`, `modf`, `isnan`, `isinf`,
  `isfinite`, `isposinf`/`isneginf`, `clip`, `nan_to_num`, `maximum`/`minimum`,
  `fmax`/`fmin`, `float_power`.
- **Complex** (`ops::complex`) — `real`, `imag`, `conj`/`conjugate`, `angle`,
  `abs`, predicates (`isreal`, `iscomplex`, `isscalar`), plus the `*_complex`
  transcendentals (`sin_complex`, `exp_complex`, `ln_complex`, `sqrt_complex`,
  `power_complex`, …).
- **Bitwise** (`ops::bitwise`) — `bitwise_and`/`or`/`xor`, `invert`/
  `bitwise_not`, `left_shift`, `right_shift`, `bitwise_count`.
- **Comparison** (`ops::comparison`) — `equal`, `not_equal`, `less`,
  `less_equal`, `greater`, `greater_equal`, `isclose`, `allclose`,
  `array_equal`, `array_equiv` (with `*_broadcast` variants).
- **Logical** (`ops::logical`) — `logical_and`/`or`/`xor`/`not`, `all`, `any`.
- **Datetime / timedelta** (`ops::datetime`) — `add_datetime_timedelta`,
  `sub_datetime`, `sub_datetime_timedelta`, `add_timedelta`, `sub_timedelta`,
  `isnat_datetime`, `isnat_timedelta`.
- **Special / misc** — `sinc`, `i0`, `convolve` (+ `interp`).

Dtype promotion (`promoted::*`) follows NumPy: integer/bool inputs to float
ufuncs promote to the smallest safe-cast float (`add_promoted`, `sin_promote`,
`hypot_promote`, …). First-class ufunc objects (`Ufunc`, `add_ufunc`) expose
`.reduce` / `.accumulate` / `.outer` / `.at`. Operator overloads (`+`, `-`,
`*`, `/`, `%`, `&`, `|`, `^`, `!`, `<<`, `>>`) are provided via
`operator_overloads`.

## NumPy correspondence

| ferray-ufunc | NumPy |
|---|---|
| `add`, `multiply`, `power`, `floor_divide` | `np.add`, `np.multiply`, `np.power`, `np.floor_divide` |
| `sin`, `arctan2`, `exp`, `log`, `tanh` | `np.sin`, `np.arctan2`, `np.exp`, `np.log`, `np.tanh` |
| `floor`, `rint`, `round`, `trunc`, `fix` | `np.floor`, `np.rint`, `np.round`, `np.trunc`, `np.fix` |
| `signbit`, `copysign`, `nextafter`, `spacing`, `frexp`, `ldexp` | `np.signbit`, `np.copysign`, `np.nextafter`, `np.spacing`, `np.frexp`, `np.ldexp` |
| `bitwise_and`, `left_shift`, `invert` | `np.bitwise_and`, `np.left_shift`, `np.invert` |
| `equal`, `greater`, `isclose`, `allclose` | `np.equal`, `np.greater`, `np.isclose`, `np.allclose` |
| `real`, `conj`, `angle`, `sin_complex` | `np.real`, `np.conj`, `np.angle`, `np.sin` (complex) |
| `add_datetime_timedelta`, `isnat_datetime` | `np.add` (`datetime64`/`timedelta64`), `np.isnat` |

## SIMD

Contiguous inner loops dispatch to SIMD kernels at runtime via `pulp`, which
selects the best available instruction set (SSE2 / AVX2 / AVX-512 / NEON) for
the host CPU. Setting `FERRAY_FORCE_SCALAR=1` forces the scalar fallback path,
used to cross-verify SIMD results against scalar reference output in tests.
Large compute-bound arrays are additionally parallelized with `rayon`.

## Feature flags

| Feature | Default | Description |
|---|---|---|
| `f16` | off | IEEE binary16 (`half`) ufunc variants (`add_f16`, `sin_f16`, …); pulls in `ferray-core/f16`. |
| `fft-convolve` | off | Enables an O(n log n) FFT-based fast path (`fftconvolve`) for large f32/f64 inputs via `ferray-fft`. The default `convolve` stays direct, matching NumPy's contract. |

## Example

```rust
use ferray_ufunc::{add, sin, exp, copysign};
use ferray_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 6.28, 1000)?;

// Element-wise transcendentals (preserve shape and dtype)
let s = sin(&a)?;
let e = exp(&s)?;

// Binary ufuncs with broadcasting
let sum = add(&s, &e)?;

// Float intrinsic: copy the sign of `s` onto `e`
let signed = copysign(&e, &s)?;
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

Edition 2024, MSRV 1.88. Licensed under MIT OR Apache-2.0.
