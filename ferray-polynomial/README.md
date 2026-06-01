# ferray-polynomial

A Rust-native implementation of `numpy.polynomial` — basis classes, evaluation, calculus, fitting, and companion-matrix root-finding.

Part of the [ferray](../README.md) workspace.

## Overview

`ferray-polynomial` provides polynomial operations in six bases, each exposed as
a distinct type and re-exported at the crate root:

- `Polynomial` — the power (monomial) basis
- `Chebyshev` — Chebyshev polynomials of the first kind
- `Legendre`
- `Laguerre`
- `Hermite` — physicist's Hermite polynomials
- `HermiteE` — probabilist's Hermite polynomials

Every basis class implements the common `Poly` trait, which carries the full
operation surface: `eval` / `eval_many`, `deriv`, `integ`, `roots`, `degree`,
`coeffs`, `trim`, `truncate`, the arithmetic methods `add` / `sub` / `mul` /
`pow` / `divmod`, and the least-squares constructors `fit` / `fit_weighted`.
`roots()` is computed from companion-matrix eigenvalues via `ferray-linalg`
(faer-backed Schur), returning every root — including complex conjugate pairs —
as `Complex<f64>`.

Each class also carries a NumPy-style `domain` / `window` pair, so coefficients
can be fit and evaluated on a non-canonical interval via an affine map (see
`with_domain` / `with_window`). Basis conversion pivots through the power basis:
the `ToPowerBasis`, `FromPowerBasis`, and `ConvertBasis` traits let any basis be
converted to any other (`cheb2poly`, `poly2cheb`, …) through the canonical
power-basis representation.

`f32` siblings are available for callers working in single precision:
`PolynomialF32`, `ChebyshevF32`, `LegendreF32`, `LaguerreF32`, `HermiteF32`, and
`HermiteEF32`. `ComplexPolynomial` covers complex power-basis coefficients.

The free-function surface mirrors `numpy.polynomial`'s module-level API,
including `polyfromroots` (the basis-class form of `np.poly`), per-basis
`*fromroots` / `*line` / `*mulx` / `*gauss` constructors, multivariate
`polyval2d` / `polyval3d` / `polygrid2d` / `polygrid3d`, the Vandermonde builders
`polyvander2d` / `polyvander3d`, and the explicit `*2poly` / `poly2*` basis
converters. `np.roots` is backed by `roots::find_roots_from_power_coeffs`.

## NumPy correspondence

| NumPy | ferray-polynomial |
|-------|-------------------|
| `numpy.polynomial.Polynomial` | `Polynomial` |
| `numpy.polynomial.Chebyshev` | `Chebyshev` |
| `numpy.polynomial.Legendre` | `Legendre` |
| `numpy.polynomial.Laguerre` | `Laguerre` |
| `numpy.polynomial.Hermite` | `Hermite` |
| `numpy.polynomial.HermiteE` | `HermiteE` |
| `np.poly` | `polyfromroots` |
| `np.roots` | `roots::find_roots_from_power_coeffs` |

## Feature flags

This crate has no Cargo feature flags; all functionality is available by
default. Root-finding depends on `ferray-linalg` for the companion-matrix
eigenvalue solve.

## Example

```rust
use ferray_polynomial::{Chebyshev, Poly};

// 1 + 2*T_1(x) + T_2(x) in the Chebyshev basis on the default [-1, 1] window.
let c = Chebyshev::new(&[1.0, 2.0, 1.0]);

let y = c.eval(0.5)?;          // evaluate at a single point
let ys = c.eval_many(&[-1.0, 0.0, 1.0])?;

let d = c.deriv(1)?;           // first derivative (still a Chebyshev)
let roots = c.roots()?;        // companion-matrix eigenvalues, Vec<Complex<f64>>

// Least-squares fit of degree 3 to sample data.
let fit = Chebyshev::fit(&[0.0, 1.0, 2.0, 3.0], &[1.0, 2.0, 0.0, 5.0], 3)?;
# Ok::<(), ferray_core::FerrayError>(())
```

## MSRV & edition

- Rust edition 2024, MSRV 1.88
- License: MIT OR Apache-2.0
