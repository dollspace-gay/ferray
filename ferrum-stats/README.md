# ferrum-stats

Statistical functions, reductions, sorting, histograms, and set operations for the [ferrum](https://crates.io/crates/ferrum) scientific computing library.

## What's in this crate

- **Reductions**: `sum`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax` with SIMD pairwise summation
- **NaN-aware**: `nansum`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`
- **Sorting**: `sort`, `argsort`, `partition`, `argpartition`
- **Histograms**: `histogram`, `histogram2d`, `histogramdd`, `bincount`
- **Set operations**: `unique`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`
- **Correlation**: `corrcoef`, `cov`, `correlate`
- Axis-aware reductions with Rayon parallelism for large arrays

## Usage

```rust
use ferrum_stats::{mean, std, var};
use ferrum_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 1.0, 1000)?;
let m = mean(&a, None)?;
let s = std(&a, None, None)?;
```

This crate is re-exported through the main [`ferrum`](https://crates.io/crates/ferrum) crate.

## License

MIT OR Apache-2.0
