# ferrum-window

Window functions and functional programming utilities for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **Window functions**: `hann`, `hamming`, `blackman`, `kaiser`, `bartlett`, `gaussian`, `tukey`, etc.
- **Functional**: `vectorize`, `piecewise`, `apply_along_axis`, `apply_over_axes`

## Usage

```rust
use ferrum_window::{hann, hamming};

let w = hann(256, true)?;
let h = hamming(256, true)?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate with the `window` feature (enabled by default).

## License

MIT OR Apache-2.0
