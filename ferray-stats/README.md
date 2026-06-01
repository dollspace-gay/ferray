# ferray-stats

NumPy-equivalent statistics, reductions, sorting, searching, and histograms for the ferray scientific computing library.

Part of the [ferray](../README.md) workspace — a Rust-native, drop-in NumPy replacement. All functions operate on ferray-core's `NdArray<T, D>`, return `FerrayResult<T>`, and never panic.

## Overview

- **Reductions** — `sum`, `prod`, `min`, `max`, `argmin`, `argmax`, `ptp` (peak-to-peak), `cumsum`, `cumprod`, plus pre-allocated `_into` and reusable `_with` variants. Summation uses pairwise SIMD accumulation for accuracy and speed. NaN-propagating by default.
- **Descriptive statistics** — `mean`, `var`, `std_`, `average` (weighted), `median`, with `ddof` control and axis support. scipy-parity helpers: `skew`, `kurtosis`, `zscore`, `mode`, `iqr`, `sem`, `gmean`, `hmean`.
- **Quantiles** — `quantile`, `percentile`, and `median`, with `quantile_with_method` / `percentile_with_method` exposing all nine Hyndman–Fan interpolation methods plus the selection styles (`Linear`, `Lower`, `Higher`, `Nearest`, `Midpoint`) via the `QuantileMethod` enum.
- **NaN-aware variants** — `nansum`, `nanprod`, `nanmin`, `nanmax`, `nanargmin`, `nanargmax`, `nanmean`, `nanvar`, `nanstd`, `nancumsum`, `nancumprod`, `nanmedian`, `nanquantile`, `nanpercentile`.
- **Sorting** — `sort`, `argsort`, `partition`, `argpartition`, `lexsort`, `sort_complex`, and `searchsorted` / `searchsorted_with_sorter` (with `SortKind` and `Side` enums).
- **Searching** — `nonzero`, `where_` / `where_condition` / `where_broadcast`, `count_nonzero`, `unique` (and `unique_values`, `unique_counts`, `unique_inverse`, `unique_all`, `unique_axis`), plus set ops (`union1d`, `intersect1d`, `setdiff1d`, `setxor1d`, `in1d`, `isin`).
- **Histograms** — `histogram`, `histogram2d`, `histogramdd`, `histogram_bin_edges`, `bincount`, `digitize` (with the `Bins` strategy enum).
- **Correlation & hypothesis tests** — `corrcoef`, `cov`, `correlate`, plus `pearsonr`, `spearmanr`, `ttest_1samp`, `ttest_ind`, `ks_2samp`, `chi2_contingency`.

Large arrays dispatch to a Rayon-parallel path above an element-count threshold (`parallel` module).

## NumPy correspondence

| NumPy | ferray-stats |
|-------|--------------|
| `np.sum` / `np.prod` / `np.ptp` | `sum`, `prod`, `ptp` |
| `np.mean` / `np.std` / `np.var` | `mean`, `std_`, `var` |
| `np.median` / `np.average` | `median`, `average` |
| `np.quantile` / `np.percentile` | `quantile`, `percentile` (+ `*_with_method`) |
| `np.argmin` / `np.argmax` | `argmin`, `argmax` |
| `np.cumsum` / `np.cumprod` | `cumsum`, `cumprod` |
| `np.sort` / `np.argsort` | `sort`, `argsort` |
| `np.partition` / `np.argpartition` | `partition`, `argpartition` |
| `np.lexsort` / `np.searchsorted` | `lexsort`, `searchsorted` |
| `np.nonzero` / `np.flatnonzero` | `nonzero` |
| `np.where` / `np.count_nonzero` | `where_`, `count_nonzero` |
| `np.unique` | `unique` (+ `unique_counts`, `unique_inverse`, …) |
| `np.histogram` / `np.bincount` / `np.digitize` | `histogram`, `bincount`, `digitize` |
| `np.nan*` (e.g. `np.nanmean`) | `nanmean`, `nanstd`, `nanquantile`, … |

## Feature flags

This crate defines no Cargo features. It always builds against `ferray-core`, `ferray-ufunc`, `rayon`, `num-traits`, `num-complex`, and `pulp` on the workspace MSRV with no optional runtime dependencies.

## Example

```rust
use ferray_stats::{mean, std_, quantile, QuantileMethod, quantile_with_method};
use ferray_core::prelude::*;

let a = Array1::<f64>::linspace(0.0, 1.0, 1000)?;

// Descriptive statistics (axis = None reduces the whole array).
let m = mean(&a, None)?;
let s = std_(&a, None, 0)?;          // ddof = 0

// Median via the 0.5 quantile (NumPy-default linear interpolation).
let med = quantile(&a, 0.5, None)?;

// Explicit interpolation method.
let p90 = quantile_with_method(&a, 0.9, None, QuantileMethod::Hazen)?;
# Ok::<(), ferray_core::FerrayError>(())
```

This crate is re-exported through the umbrella [`ferray`](../README.md) crate.

## MSRV & edition

- Edition 2024, MSRV 1.88
- License: MIT OR Apache-2.0
