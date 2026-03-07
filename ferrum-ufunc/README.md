# ferrum-ufunc

SIMD-accelerated universal functions for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- 40+ elementwise operations: `sin`, `cos`, `exp`, `log`, `sqrt`, `abs`, `floor`, `ceil`, etc.
- Binary operations: `add`, `sub`, `mul`, `div`, `pow` with broadcasting
- CORE-MATH correctly-rounded transcendentals (< 0.5 ULP from mathematical truth)
- Portable SIMD via `pulp` (SSE2/AVX2/AVX-512/NEON) on stable Rust
- Scalar fallback with `FERRUM_FORCE_SCALAR=1` environment variable
- SIMD paths for f32, f64, i32, i64 on all contiguous inner loops

## Performance

Uses [CORE-MATH](https://core-math.gitlabpages.inria.fr/) for every transcendental — the only correctly-rounded math library in production. Every result is the closest representable floating-point value to the mathematical truth.

## Usage

```rust
use ferrum_ufunc::{sin, exp, add};
use ferrum_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 6.28, 1000)?;
let b = sin(&a)?;
let c = exp(&b)?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate.

## License

MIT OR Apache-2.0
