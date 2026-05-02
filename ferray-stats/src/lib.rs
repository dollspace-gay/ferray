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

// Statistical kernels divide running sums by sample counts (`n as f64`),
// truncate `f64` results to histogram bins, and rely on exact float
// equality for argmin/argmax tie-breaking and unique/set ops on bit
// patterns. All of these are part of the spec, not bugs.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod correlation;
pub mod descriptive;
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
    argmax, argmin, average, cumprod, cumsum, max, max_into, max_with, mean, mean_as_f64,
    mean_into, mean_where, min, min_into, min_with, prod, prod_into, prod_with, ptp, std_,
    std_into, sum, sum_as_f64, sum_into, sum_with, var, var_into,
};

// Quantile-based
pub use reductions::quantile::{
    QuantileMethod, median, nanmedian, nanpercentile, nanquantile, percentile,
    percentile_with_method, quantile, quantile_with_method,
};

// NaN-aware reductions
pub use reductions::nan_aware::{
    nanargmax, nanargmin, nancumprod, nancumsum, nanmax, nanmean, nanmin, nanprod, nanstd, nansum,
    nanvar,
};

// Correlation and covariance
pub use correlation::{CorrelateMode, corrcoef, correlate, cov};

// Histogram
pub use histogram::{
    Bins, bincount, bincount_u64, bincount_weighted, digitize, histogram, histogram_bin_edges,
    histogram2d, histogramdd,
};

// Sorting
pub use sorting::{
    Side, SortKind, argpartition, argsort, lexsort, partition, searchsorted,
    searchsorted_with_sorter, sort, sort_complex,
};

// Searching
pub use searching::{
    UniqueResult, count_nonzero, nonzero, unique, unique_all, unique_axis, unique_counts,
    unique_inverse, unique_values, where_, where_broadcast, where_condition,
};

// Set operations
pub use set_ops::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d};

// Descriptive statistics — scipy.stats parity (#470)
pub use descriptive::{ModeResult, gmean, hmean, iqr, kurtosis, mode, sem, skew, zscore};
