//! # ferray — a NumPy-equivalent library for Rust
//!
//! `ferray` is the top-level umbrella crate that re-exports every
//! ferray sub-crate under a unified namespace, analogous to `import
//! numpy as np` in Python. Most users should start here rather than
//! depending on the individual `ferray-core`, `ferray-ufunc`,
//! `ferray-stats`, etc. crates directly.
//!
//! ```ignore
//! use ferray::prelude::*;
//!
//! let a = zeros::<f64, Ix2>(Ix2::new([2, 3])).unwrap();
//! let b = sin(&a).unwrap();
//! ```
//!
//! ## Sub-modules
//!
//! Optional sub-crates are feature-gated and exposed as named
//! modules (`ferray::fft`, `ferray::linalg`, `ferray::random`,
//! `ferray::io`, `ferray::polynomial`, `ferray::window`,
//! `ferray::strings`, `ferray::ma`, `ferray::stride_tricks`,
//! `ferray::autodiff`, `ferray::numpy_interop`).
//!
//! ## Features
//!
//! | flag              | pulls in                                   |
//! |-------------------|--------------------------------------------|
//! | `full`            | every sub-crate below + f16 + serde        |
//! | `io`              | `ferray-io` (.npy/.npz/text/memmap)        |
//! | `linalg`          | `ferray-linalg` (decomps, solvers, norms)  |
//! | `fft`             | `ferray-fft` (rustfft + realfft)           |
//! | `random`          | `ferray-random` (PCG64 / Xoshiro256)       |
//! | `polynomial`      | `ferray-polynomial` (power + 5 bases)      |
//! | `window`          | `ferray-window` (Kaiser, Bartlett, …)      |
//! | `strings`         | `ferray-strings` (vectorized string ops)   |
//! | `ma`              | `ferray-ma` (masked arrays)                |
//! | `stride-tricks`   | `ferray-stride-tricks` (`as_strided`, …)     |
//! | `autodiff`        | `ferray-autodiff` (forward-mode dual)      |
//! | `numpy`           | `ferray-numpy-interop` (`PyO3` bridge)       |
//! | `f16` / `bf16`    | half-precision element types               |
//! | `serde`           | Array serde impls via `ferray-core/serde`  |
//!
//! The umbrella crate itself contains only re-exports and a thin
//! thread-pool config layer, so it forbids unsafe code.

#![forbid(unsafe_code)]
// Umbrella crate: re-exports only. Workspace convention is to document
// FerrayError variants on the type, not on every returning function.
#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod config;
pub mod prelude;

// ---------------------------------------------------------------------------
// REQ-1: Top-level namespace — core types and creation functions
// ---------------------------------------------------------------------------

// Core types
pub use ferray_core::ArcArray;
pub use ferray_core::Array;
pub use ferray_core::ArrayFlags;
pub use ferray_core::ArrayView;
pub use ferray_core::ArrayViewMut;
pub use ferray_core::AsRawBuffer;
pub use ferray_core::CowArray;
pub use ferray_core::DynArray;
pub use ferray_core::FerrayRecord;
pub use ferray_core::FieldDescriptor;

// Dimension types
pub use ferray_core::dimension::{self, Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferray_core::dtype::casting;
pub use ferray_core::dtype::finfo;
pub use ferray_core::dtype::promotion;
pub use ferray_core::dtype::{DateTime64, NAT, TimeUnit, Timedelta64};
pub use ferray_core::{DType, Element, SliceInfoElem};

// Error handling
pub use ferray_core::{FerrayError, FerrayResult};

// Memory layout
pub use ferray_core::MemoryLayout;

// Macros
pub use ferray_core::{promoted_type, s};

// num_complex re-export so downstream users don't need a direct
// num-complex dependency just to name Complex<T> (#333).
pub use num_complex::Complex;

// Type aliases
pub use ferray_core::aliases;

// Display configuration
pub use ferray_core::array::display::{get_print_options, set_print_options};

// ---------------------------------------------------------------------------
// REQ-3a: Constants (ferray::PI, ferray::INF, etc.)
// ---------------------------------------------------------------------------

pub use ferray_core::constants::{E, EULER_GAMMA, INF, NAN, NEG_INF, NEWAXIS, NZERO, PI, PZERO};

// ---------------------------------------------------------------------------
// REQ-1: Array creation functions at top level
// ---------------------------------------------------------------------------

pub use ferray_core::creation::{
    arange, array, asanyarray, asarray, asarray_chkfinite, ascontiguousarray, asfortranarray, copy,
    diag, diagflat, empty, empty_like, eye, frombuffer, frombuffer_view, fromfile, fromfunction,
    fromiter, fromstring, full, full_like, geomspace, identity, linspace, logspace, meshgrid,
    mgrid, ogrid, ones, ones_like, require, tri, tril, triu, vander, zeros, zeros_like,
};

// ---------------------------------------------------------------------------
// REQ-1: Manipulation functions at top level
// ---------------------------------------------------------------------------

pub use ferray_core::manipulation::{
    array_split, block, broadcast_to, c_, concatenate, dsplit, dstack, expand_dims, flatten, flip,
    fliplr, flipud, hsplit, hstack, moveaxis, r_, ravel, reshape, roll, rollaxis, rot90, split,
    squeeze, stack, swapaxes, transpose, vsplit, vstack,
};

pub use ferray_core::manipulation::extended::{
    append, atleast_1d, atleast_2d, atleast_3d, delete, insert, pad, repeat, resize, tile,
    trim_zeros,
};

// Broadcasted elementwise assignment (np.copyto equivalent) (#352, #353)
pub use ferray_core::ops::{copyto, copyto_where};

// Indexing
pub use ferray_core::indexing::extended::{
    MaskKind, argwhere, choose, compress, diag_indices, diag_indices_from, extract, flatnonzero,
    indices, ix_, mask_indices, ndenumerate, ndindex, nonzero, place, putmask, ravel_multi_index,
    select, take, take_along_axis, tril_indices, tril_indices_from, triu_indices,
    triu_indices_from, unravel_index,
};

// ---------------------------------------------------------------------------
// REQ-1: Ufunc functions at top level
// ---------------------------------------------------------------------------

// Trigonometric
pub use ferray_ufunc::{
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, cos, cosh, deg2rad, degrees, hypot,
    rad2deg, radians, sin, sinh, tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ferray_ufunc::{
    exp, exp_fast, exp2, expm1, log, log1p, log2, log10, logaddexp, logaddexp2,
};

// Rounding
pub use ferray_ufunc::{around, ceil, fix, floor, rint, round, trunc};

// Arithmetic
pub use ferray_ufunc::{
    absolute, add, add_accumulate, add_broadcast, add_reduce, add_reduce_all, add_reduce_axes,
    add_reduce_keepdims, cbrt, cross, cumprod, cumsum, cumulative_prod, cumulative_sum, diff,
    divide, divide_broadcast, divmod, ediff1d, fabs, floor_divide, fmod, gcd, gradient, heaviside,
    lcm, mod_, multiply, multiply_broadcast, multiply_outer, nan_add_reduce, nan_add_reduce_all,
    nan_add_reduce_axes, nan_max_reduce, nan_max_reduce_all, nan_max_reduce_axes, nan_min_reduce,
    nan_min_reduce_all, nan_min_reduce_axes, nan_multiply_reduce, nan_multiply_reduce_all,
    nan_multiply_reduce_axes, nancumprod, nancumsum, negative, positive, power, reciprocal,
    remainder, sign, sqrt, square, subtract, subtract_broadcast, trapezoid, true_divide,
};

// Float intrinsics
pub use ferray_ufunc::{
    clip, copysign, float_power, fmax, fmin, frexp, isfinite, isinf, isnan, isneginf, isposinf,
    ldexp, maximum, minimum, modf, nan_to_num, nextafter, signbit, spacing,
};

// Complex
pub use ferray_ufunc::{
    abs, angle, conj, conjugate, imag, iscomplex, iscomplex_real, iscomplexobj, isreal,
    isreal_real, isrealobj, isscalar, real,
};

// Datetime / timedelta
pub use ferray_ufunc::{
    add_datetime_timedelta, add_datetime_timedelta_promoted, add_timedelta, add_timedelta_promoted,
    isnat_datetime, isnat_timedelta, sub_datetime, sub_datetime_promoted, sub_datetime_timedelta,
    sub_timedelta,
};

// Bitwise
pub use ferray_ufunc::{
    BitwiseCount, BitwiseOps, ShiftOps, bitwise_and, bitwise_count, bitwise_not, bitwise_or,
    bitwise_xor, invert, left_shift, right_shift,
};

// Logical / comparison / i0 / sinc / convolve / interp traits used
// by downstream generics (#220).
pub use ferray_ufunc::Logical;

// Comparison
pub use ferray_ufunc::{
    allclose, array_equal, array_equiv, equal, equal_broadcast, greater, greater_broadcast,
    greater_equal, greater_equal_broadcast, isclose, isclose_broadcast, less, less_broadcast,
    less_equal, less_equal_broadcast, not_equal, not_equal_broadcast,
};

// Logical
pub use ferray_ufunc::{
    all, all_axis, any, any_axis, logical_and, logical_not, logical_or, logical_xor,
};

// Special
pub use ferray_ufunc::{i0, sinc};

// Convolution and interpolation
pub use ferray_ufunc::{ConvolveMode, convolve, interp, interp_one};

// Operator-style functions
pub use ferray_ufunc::{
    array_add, array_bitand, array_bitnot, array_bitor, array_bitxor, array_div, array_mul,
    array_neg, array_rem, array_shl, array_shr, array_sub,
};

// ---------------------------------------------------------------------------
// REQ-1: Stats functions at top level
// ---------------------------------------------------------------------------

// Reductions — `cumsum`/`cumprod` already come from `ferray_ufunc`
// above, so the stats versions are aliased in under different names
// via full paths at use sites rather than re-exported here (#335).
pub use ferray_stats::{
    argmax, argmin, average, max, max_into, max_with, mean, mean_into, mean_where, min, min_into,
    min_with, prod, prod_into, prod_with, ptp, std_, std_into, sum, sum_into, sum_with, var,
    var_into,
};

// Quantile-based
pub use ferray_stats::{
    QuantileMethod, median, percentile, percentile_with_method, quantile, quantile_with_method,
};

// NaN-aware reductions — same rationale as above, `nancumsum` /
// `nancumprod` come from ferray-ufunc so we only re-export the
// stats-specific reducers (#335).
pub use ferray_stats::{
    nanargmax, nanargmin, nanmax, nanmean, nanmedian, nanmin, nanpercentile, nanprod, nanstd,
    nansum, nanvar,
};

// Correlation and covariance
pub use ferray_stats::{CorrelateMode, corrcoef, correlate, cov};

// Histogram
pub use ferray_stats::{
    Bins, bincount, bincount_u64, bincount_weighted, digitize, histogram, histogram_bin_edges,
    histogram2d, histogramdd,
};

// Sorting and searching
pub use ferray_stats::{
    Side, SortKind, argsort, lexsort, searchsorted, searchsorted_with_sorter, sort, sort_complex,
};
// `nonzero` now lives in ferray-core (#373) and is re-exported above via
// `ferray_core::indexing::extended`; don't re-export the stats version too.
pub use ferray_stats::{
    UniqueResult, count_nonzero, unique, unique_all, unique_counts, unique_inverse, unique_values,
    where_,
};

// Set operations
pub use ferray_stats::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d};

// ---------------------------------------------------------------------------
// REQ-2: Submodule namespaces
// ---------------------------------------------------------------------------

/// I/O operations: .npy, .npz, text files, memory mapping.
#[cfg(feature = "io")]
pub use ferray_io as io;

/// Linear algebra operations.
#[cfg(feature = "linalg")]
pub use ferray_linalg as linalg;

// Top-level re-exports of the matmul/vecdot/matvec/vecmat ufunc-style
// products, matching numpy where these live at top level
// (`numpy.matmul`, `numpy.vecdot`, `numpy.matvec`, `numpy.vecmat`).
#[cfg(feature = "linalg")]
pub use ferray_linalg::{matmul, matvec, vecdot, vecmat};

/// FFT operations.
#[cfg(feature = "fft")]
pub use ferray_fft as fft;

/// Random number generation and distributions.
#[cfg(feature = "random")]
pub use ferray_random as random;

/// Polynomial operations.
#[cfg(feature = "polynomial")]
pub use ferray_polynomial as polynomial;

/// Window functions.
#[cfg(feature = "window")]
pub use ferray_window as window;

/// String operations on character arrays.
#[cfg(feature = "strings")]
pub use ferray_strings as strings;

/// Masked arrays.
#[cfg(feature = "ma")]
pub use ferray_ma as ma;

/// Stride manipulation utilities.
#[cfg(feature = "stride-tricks")]
pub use ferray_stride_tricks as stride_tricks;

// Top-level re-exports of broadcast helpers, matching numpy where these
// live at `numpy.broadcast_arrays` / `numpy.broadcast_shapes`.
#[cfg(feature = "stride-tricks")]
pub use ferray_stride_tricks::{broadcast_arrays, broadcast_shapes};

/// Forward-mode automatic differentiation via `DualNumber<T>`.
#[cfg(feature = "autodiff")]
pub use ferray_autodiff as autodiff;

/// Python/NumPy interop via `PyO3`.
#[cfg(feature = "numpy")]
pub use ferray_numpy_interop as numpy_interop;

// ---------------------------------------------------------------------------
// REQ-6, REQ-7: Thread pool configuration re-exports
// ---------------------------------------------------------------------------

pub use config::{set_num_threads, with_num_threads};

// ---------------------------------------------------------------------------
// REQ-8: Parallel threshold constants
// ---------------------------------------------------------------------------

/// Parallel threshold configuration constants.
pub mod threshold {
    pub use crate::config::{
        PARALLEL_THRESHOLD_COMPUTE, PARALLEL_THRESHOLD_ELEMENTWISE, PARALLEL_THRESHOLD_REDUCTION,
        PARALLEL_THRESHOLD_SORT,
    };
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Re-export sanity tests assert exact equality with std constants by design.
mod tests {
    use super::*;

    #[test]
    fn constants_accessible_ac7() {
        assert_eq!(PI, std::f64::consts::PI);
        assert_eq!(E, std::f64::consts::E);
        assert_eq!(INF, f64::INFINITY);
        assert_eq!(NEG_INF, f64::NEG_INFINITY);
        assert!(NAN.is_nan());
    }

    #[test]
    fn prelude_compiles_ac1() {
        use crate::prelude::*;
        // Verify core types are accessible
        let arr = zeros::<f64, Ix2>(Ix2::new([2, 3])).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        let _: &[usize] = arr.shape();
    }

    #[test]
    fn creation_at_toplevel() {
        let a = zeros::<f64, Ix1>(Ix1::new([5])).unwrap();
        assert_eq!(a.size(), 5);
        let b = ones::<f64, Ix1>(Ix1::new([3])).unwrap();
        assert_eq!(b.size(), 3);
        let c = arange(0i32, 10, 2).unwrap();
        assert_eq!(c.size(), 5);
    }

    #[test]
    fn ufuncs_at_toplevel() {
        let a = ones::<f64, Ix1>(Ix1::new([4])).unwrap();
        let s = sin(&a).unwrap();
        assert_eq!(s.shape(), &[4]);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn threshold_constants_accessible() {
        assert!(threshold::PARALLEL_THRESHOLD_ELEMENTWISE > 0);
        assert!(threshold::PARALLEL_THRESHOLD_COMPUTE > 0);
    }
}
