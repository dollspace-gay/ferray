# ferray-stats

Statistical functions, reductions, sorting, histograms, and set operations for the [ferray](https://crates.io/crates/ferray) scientific computing library.

## What's in this crate

- **Reductions**: `sum`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax` with SIMD pairwise summation
- **NaN-aware**: `nansum`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`
- **Sorting**: `sort`, `argsort`, `partition`, `argpartition`, `sort_complex`
- **Histograms**: `histogram`, `histogram2d`, `histogramdd`, `histogram_bin_edges`, `bincount`, `digitize`, `searchsorted`
- **Set operations**: `unique`, `unique_values`, `unique_counts`, `unique_inverse`, `unique_all`, `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`
- **Correlation**: `corrcoef`, `cov`, `correlate`
- **Other**: `ptp` (peak-to-peak), `average` (weighted mean)
- Axis-aware reductions with Rayon parallelism for large arrays

## Performance

ferray's pairwise-sum implementation uses a 4096-element base chunk with 4-way SIMD accumulators, cutting horizontal-reduction overhead 16× vs the previous 256-element base. Result on a Raptor Lake 14700K vs NumPy 2.4.4:

| Operation | Size | Speedup vs NumPy |
|-----------|-----:|-----------------:|
| var | 1K | **28.8× faster** |
| std | 1K | **17.8× faster** |
| mean | 1K | **13.4× faster** |
| var | 100K | **2.6× faster** |
| std | 100K | **2.7× faster** |
| var | 1M | **3.4× faster** |
| std | 1M | **3.1× faster** |
| sum | 1K | **1.9× faster** |
| sum | 100K | **1.5× faster** |
| sum | 1M | **1.1× faster** (DRAM-bandwidth ceiling) |

## Usage

```rust
use ferray_stats::{mean, std_, var};
use ferray_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 1.0, 1000)?;
let m = mean(&a, None)?;
let s = std_(&a, None, None)?;
let v = var(&a, None, None)?;
```

This crate is re-exported through the main [`ferray`](https://crates.io/crates/ferray) crate.

## License

MIT OR Apache-2.0
