// ferrum-stats: Statistical functions, reductions, sorting, searching,
// set operations, histograms, correlations, and covariance.
//
// This crate provides NumPy-equivalent statistical functions as free functions
// operating on ferrum-core's `Array<T, D>` type. All public functions return
// `FerrumResult<T>` and never panic.
//
// # Modules
// - `reductions`: sum, prod, min, max, argmin, argmax, mean, var, std, median,
//   percentile, quantile, cumsum, cumprod, and NaN-aware variants.
// - `correlation`: correlate, corrcoef, cov.
// - `histogram`: histogram, histogram2d, histogramdd, bincount, digitize.
// - `sorting`: sort, argsort, searchsorted.
// - `searching`: unique, nonzero, where_, count_nonzero.
// - `set_ops`: union1d, intersect1d, setdiff1d, setxor1d, in1d, isin.
// - `parallel`: Rayon threshold dispatch for large array operations.

pub mod correlation;
pub mod histogram;
pub mod parallel;
pub mod reductions;
pub mod searching;
pub mod set_ops;
pub mod sorting;

// ---------------------------------------------------------------------------
// Flat re-exports for ergonomic use
// ---------------------------------------------------------------------------

// Reductions
pub use reductions::{
    sum, prod, min, max, argmin, argmax, mean, var, std_,
    cumsum, cumprod,
};

// Quantile-based
pub use reductions::quantile::{
    median, percentile, quantile,
    nanmedian, nanpercentile,
};

// NaN-aware reductions
pub use reductions::nan_aware::{
    nansum, nanprod, nanmin, nanmax, nanmean, nanvar, nanstd,
    nancumsum, nancumprod,
};

// Correlation and covariance
pub use correlation::{correlate, corrcoef, cov, CorrelateMode};

// Histogram
pub use histogram::{histogram, histogram2d, histogramdd, bincount, digitize, Bins};

// Sorting
pub use sorting::{sort, argsort, searchsorted, SortKind, Side};

// Searching
pub use searching::{unique, nonzero, where_, count_nonzero, UniqueResult};

// Set operations
pub use set_ops::{union1d, intersect1d, setdiff1d, setxor1d, in1d, isin};
