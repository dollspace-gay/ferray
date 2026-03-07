# ferrum-polynomial

Polynomial operations for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **6 basis classes**: Power, Chebyshev, Legendre, Laguerre, Hermite, HermiteE
- **Poly trait**: unified interface for evaluation, fitting, roots, arithmetic, conversion
- **Fitting**: least-squares polynomial fitting via `ferrum-linalg`
- **Root-finding**: companion matrix eigenvalue method
- **Arithmetic**: add, subtract, multiply, divide, power on polynomial objects

## Usage

```rust
use ferrum_polynomial::Power;

let p = Power::new(vec![1.0, 2.0, 3.0])?; // 1 + 2x + 3x^2
let val = p.eval(2.0)?; // 1 + 4 + 12 = 17
let roots = p.roots()?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `polynomial` feature.

## License

MIT OR Apache-2.0
