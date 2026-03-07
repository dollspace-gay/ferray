# ferrum-autodiff

Forward-mode automatic differentiation for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- `DualNumber<T>` — tracks value and derivative through computation
- Operator overloading for all arithmetic operations
- Differentiable transcendentals: `sin`, `cos`, `exp`, `log`, `sqrt`, `pow`, etc.
- Chain rule applied automatically via dual number arithmetic

## Usage

```rust
use ferrum_autodiff::DualNumber;

// f(x) = sin(x), f'(x) = cos(x)
let x = DualNumber::new(0.0_f64, 1.0); // x = 0, dx/dx = 1
let y = x.sin();
assert!((y.value - 0.0).abs() < 1e-10);     // sin(0) = 0
assert!((y.derivative - 1.0).abs() < 1e-10); // cos(0) = 1
```

## License

MIT OR Apache-2.0
