# ferray-polynomial

Polynomial operations for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **6 basis classes**: Power, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
- **Poly trait** lifted to `T: Element + Float` (also accepts complex coefficients)
- **Unified interface** for evaluation, fitting, roots, arithmetic, conversion
- **Fitting**: least-squares polynomial fitting via `ferray-linalg`, `polyfit_weighted` for non-uniform weights
- **Root-finding**: companion matrix eigenvalue method, `polyfromroots` and per-basis variants
- **Arithmetic**: add, subtract, multiply, divide, power, `mulx` (multiply-by-x) on polynomial objects
- **Multivariate evaluation**: `val2d`/`val3d`, `grid2d`/`grid3d`, `vander2d`/`vander3d` for all 6 bases
- **Gauss quadrature**: `chebgauss`, `leggauss`, `hermgauss`, `lagguass`, `hermegauss`
- **Linear-polynomial constructors**: `polyline`/`chebline`/`legline`/`lagline`/`hermline`/`hermeline`
- **Chebyshev extras**: `chebpts1`, `chebpts2`, `chebweight`, `chebinterpolate`
- **Explicit basis-conversion**: `cheb2poly`, `poly2cheb`, etc.
- **`polyvalfromroots`** for evaluating from a root vector

## Usage

```rust
use ferray_polynomial::Power;

let p = Power::new(vec![1.0, 2.0, 3.0])?; // 1 + 2x + 3x^2
let val = p.eval(2.0)?; // 1 + 4 + 12 = 17
let roots = p.roots()?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate with the `polynomial` feature.

## License

MIT OR Apache-2.0
