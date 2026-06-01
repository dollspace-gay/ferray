# ferray-window

NumPy-equivalent window functions (plus a few functional-programming utilities) for the ferray workspace.

Part of the [ferray](../README.md) workspace — a Rust-native, drop-in replacement for NumPy.

## Overview

This crate provides window functions for signal processing and spectral
analysis. All window functions take a length `m: usize` and return an
`Array1<f64>` (the f32 siblings return `Array1<f32>`), matching NumPy's
output to high precision.

The five NumPy window functions:

- `bartlett(m)` — triangular window
- `blackman(m)` — Blackman window
- `hamming(m)` — Hamming window
- `hanning(m)` — Hann window
- `kaiser(m, beta)` — Kaiser window parameterised by `beta`

Each also has an f32 sibling: `bartlett_f32`, `blackman_f32`, `hamming_f32`,
`hanning_f32`, `kaiser_f32`.

SciPy-style extras (beyond NumPy's five): `boxcar`, `triang`, `cosine`,
`exponential`, `gaussian`, `general_cosine`, `general_hamming`, `nuttall`,
`parzen`, `bohman`, `flattop`, `lanczos`, `tukey`, `taylor`, `chebwin`, and
`dpss` (Slepian taper).

Functional-programming utilities are also re-exported: `vectorize`,
`piecewise`, `apply_along_axis`, `apply_over_axes`, and `sum_axis_keepdims`.

## NumPy correspondence

| ferray                | NumPy            |
|-----------------------|------------------|
| `bartlett(m)`         | `np.bartlett(M)` |
| `blackman(m)`         | `np.blackman(M)` |
| `hamming(m)`          | `np.hamming(M)`  |
| `hanning(m)`          | `np.hanning(M)`  |
| `kaiser(m, beta)`     | `np.kaiser(M, beta)` |

## Feature flags

None. The crate builds with default features only and adds no extra runtime
dependencies.

## Example

```rust
use ferray_window::{hanning, kaiser};

// Hann window of length 64.
let w = hanning(64)?;

// Kaiser window of length 64 with beta = 14.0.
let k = kaiser(64, 14.0)?;

assert_eq!(w.len(), 64);
assert_eq!(k.len(), 64);
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Edition: 2024
- MSRV: 1.88
- License: MIT OR Apache-2.0
