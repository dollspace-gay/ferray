// ferray-stats: Statistical functions, reductions, sorting, searching,
// set operations, histograms, correlations, and covariance.
//
// This crate provides NumPy-equivalent statistical functions as free functions
// operating on ferray-core's `Array<T, D>` type. All public functions return
// `FerrayResult<T>` and never panic.
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
    argmax, argmin, cumprod, cumsum, max, max_into, max_with, mean, mean_as_f64, mean_into,
    mean_where, min, min_into, min_with, prod, prod_into, prod_with, std_, std_into, sum,
    sum_as_f64, sum_into, sum_with, var, var_into,
};

// Quantile-based
pub use reductions::quantile::{
    QuantileMethod, median, nanmedian, nanpercentile, nanquantile, percentile,
    percentile_with_method, quantile, quantile_with_method,
};

// NaN-aware reductions
pub use reductions::nan_aware::{
    nancumprod, nancumsum, nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar,
};

// Correlation and covariance
pub use correlation::{CorrelateMode, corrcoef, correlate, cov};

// Histogram
pub use histogram::{
    Bins, bincount, bincount_u64, bincount_weighted, digitize, histogram, histogram2d,
    histogramdd,
};

// Sorting
pub use sorting::{
    Side, SortKind, argpartition, argsort, lexsort, partition, searchsorted,
    searchsorted_with_sorter, sort,
};

// Searching
pub use searching::{UniqueResult, count_nonzero, nonzero, unique, where_, where_condition};

// Set operations
pub use set_ops::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d};
