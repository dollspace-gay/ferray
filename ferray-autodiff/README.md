# ferray-autodiff

Forward-mode automatic differentiation for ferray, built on dual numbers.

Part of the [ferray](../README.md) workspace — a ferray-native extension beyond the NumPy surface (NumPy has no autodiff). This crate has no NumPy counterpart; it adds differentiation primitives that ferray itself does not expose.

## Overview

The core type is `DualNumber<T>`, a dual number `a + b*eps` (where `eps^2 = 0`)
that threads a derivative through every arithmetic operation. The real part
`real` carries the value; the dual part `dual` carries the derivative. Build
them with `DualNumber::new`, `DualNumber::constant` (dual = 0), or
`DualNumber::variable` (dual = 1, the seed you differentiate against).

Only **forward mode** is implemented. There is no reverse-mode / tape-based
backpropagation in this crate.

Differentiable functions (methods on `DualNumber` and free-function mirrors):
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`,
`exp`, `ln`, `log2`, `log10`, `sqrt`, `abs`. `DualNumber<T: Float>` also
implements the standard `num_traits` traits (`Zero`, `One`, `Num`, `Float`,
…), so generic numeric code differentiates transparently.

Scalar / slice API (`api` module, re-exported at the crate root):

- `derivative(f, x)` — derivative of a univariate `f` at `x`.
- `gradient(f, &point)` — gradient of a scalar-valued `f: R^n -> R` as a `Vec<T>`.
- `jacobian(f, &point)` — Jacobian of a vector-valued `f: R^n -> R^m`, returned
  as `(flat_row_major: Vec<T>, m: usize)`.

Array-aware API over `ferray_core::Array` (`array_ops` module, re-exported):

- `derivative_elementwise(f, &arr)` and `value_and_derivative_elementwise(f, &arr)`
  — elementwise `f'` (and value) over an array of any dimension.
- `gradient_vector(f, &arr)` and `value_and_gradient(f, &arr)` — gradient of a
  scalar-valued function of a 1-D array.
- `jacobian_array(f, &arr)` — `m x n` Jacobian as an `Array<T, Ix2>`.

Array-aware functions return `FerrayResult` and require C-contiguous inputs.

## Feature flags

This crate defines no Cargo features. It depends only on `ferray-core` and
`num-traits`.

## Example

```rust
use ferray_autodiff::{derivative, gradient, DualNumber};

// d/dx sin(x) at x = 0  ->  cos(0) = 1
let d = derivative(|x| x.sin(), 0.0_f64);
assert!((d - 1.0).abs() < 1e-10);

// Drive a DualNumber directly: f(x) = x^2 at x = 3, f'(x) = 2x = 6
let y = {
    let x = DualNumber::variable(3.0_f64);
    x * x
};
assert_eq!(y.real, 9.0);
assert_eq!(y.dual, 6.0);

// Gradient of f(x, y) = x^2 + y^2  ->  (2x, 2y)
let g = gradient(|v| v[0] * v[0] + v[1] * v[1], &[3.0_f64, 4.0]);
assert_eq!(g, vec![6.0, 8.0]);
```

## MSRV & edition

- Rust edition 2024, MSRV 1.88.
- License: MIT OR Apache-2.0.
