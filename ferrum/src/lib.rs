// ferrum: Main re-export crate providing `use ferrum::prelude::*`
//
// This is the top-level crate that users depend on. It re-exports all
// subcrates into a unified namespace, matching NumPy's `import numpy as np`.

pub mod config;
pub mod prelude;

// ---------------------------------------------------------------------------
// REQ-1: Top-level namespace — core types and creation functions
// ---------------------------------------------------------------------------

// Core types
pub use ferrum_core::ArcArray;
pub use ferrum_core::Array;
pub use ferrum_core::ArrayFlags;
pub use ferrum_core::ArrayView;
pub use ferrum_core::ArrayViewMut;
pub use ferrum_core::AsRawBuffer;
pub use ferrum_core::CowArray;
pub use ferrum_core::DynArray;
pub use ferrum_core::FerrumRecord;
pub use ferrum_core::FieldDescriptor;

// Dimension types
pub use ferrum_core::dimension::{self, Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferrum_core::dtype::casting;
pub use ferrum_core::dtype::finfo;
pub use ferrum_core::dtype::promotion;
pub use ferrum_core::{DType, Element, SliceInfoElem};

// Error handling
pub use ferrum_core::{FerrumError, FerrumResult};

// Memory layout
pub use ferrum_core::MemoryLayout;

// Macros
pub use ferrum_core::{FerrumRecord as DeriveFerrumRecord, promoted_type, s};

// Type aliases
pub use ferrum_core::aliases;

// Display configuration
pub use ferrum_core::array::display::{get_print_options, set_print_options};

// ---------------------------------------------------------------------------
// REQ-3a: Constants (ferrum::PI, ferrum::INF, etc.)
// ---------------------------------------------------------------------------

pub use ferrum_core::constants::{E, EULER_GAMMA, INF, NAN, NEG_INF, NEWAXIS, NZERO, PI, PZERO};

// ---------------------------------------------------------------------------
// REQ-1: Array creation functions at top level
// ---------------------------------------------------------------------------

pub use ferrum_core::creation::{
    arange, array, asarray, diag, diagflat, empty, eye, frombuffer, fromiter, full, full_like,
    geomspace, identity, linspace, logspace, meshgrid, mgrid, ogrid, ones, ones_like, tri, tril,
    triu, zeros, zeros_like,
};

// ---------------------------------------------------------------------------
// REQ-1: Manipulation functions at top level
// ---------------------------------------------------------------------------

pub use ferrum_core::manipulation::{
    array_split, block, broadcast_to, concatenate, dsplit, dstack, expand_dims, flatten, flip,
    fliplr, flipud, hsplit, hstack, moveaxis, ravel, reshape, roll, rollaxis, rot90, split,
    squeeze, stack, swapaxes, transpose, vsplit, vstack,
};

pub use ferrum_core::manipulation::extended::{
    append, delete, insert, pad, repeat, resize, tile, trim_zeros,
};

// ---------------------------------------------------------------------------
// REQ-1: Ufunc functions at top level
// ---------------------------------------------------------------------------

// Trigonometric
pub use ferrum_ufunc::{
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, cos, cosh, deg2rad, degrees, hypot,
    rad2deg, radians, sin, sinh, tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ferrum_ufunc::{exp, exp2, exp_fast, expm1, log, log1p, log2, log10, logaddexp, logaddexp2};

// Rounding
pub use ferrum_ufunc::{around, ceil, fix, floor, rint, round, trunc};

// Arithmetic
pub use ferrum_ufunc::{
    absolute, add, add_accumulate, add_broadcast, add_reduce, cbrt, cross, cumprod, cumsum, diff,
    divide, divide_broadcast, divmod, ediff1d, fabs, floor_divide, fmod, gcd, gradient, heaviside,
    lcm, mod_, multiply, multiply_broadcast, multiply_outer, nancumprod, nancumsum, negative,
    positive, power, reciprocal, remainder, sign, sqrt, square, subtract, subtract_broadcast,
    trapezoid, true_divide,
};

// Float intrinsics
pub use ferrum_ufunc::{
    clip, copysign, float_power, fmax, fmin, frexp, isfinite, isinf, isnan, isneginf, isposinf,
    ldexp, maximum, minimum, nan_to_num, nextafter, signbit, spacing,
};

// Complex
pub use ferrum_ufunc::{abs, angle, conj, conjugate, imag, real};

// Bitwise
pub use ferrum_ufunc::{
    bitwise_and, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift, right_shift,
};

// Comparison
pub use ferrum_ufunc::{
    allclose, array_equal, array_equiv, equal, greater, greater_equal, isclose, less, less_equal,
    not_equal,
};

// Logical
pub use ferrum_ufunc::{all, any, logical_and, logical_not, logical_or, logical_xor};

// Special
pub use ferrum_ufunc::{i0, sinc};

// Convolution and interpolation
pub use ferrum_ufunc::{ConvolveMode, convolve, interp, interp_one};

// Operator-style functions
pub use ferrum_ufunc::{
    array_add, array_bitand, array_bitnot, array_bitor, array_bitxor, array_div, array_mul,
    array_neg, array_rem, array_shl, array_shr, array_sub,
};

// ---------------------------------------------------------------------------
// REQ-1: Stats functions at top level
// ---------------------------------------------------------------------------

// Reductions
pub use ferrum_stats::{
    argmax, argmin, cumprod as stats_cumprod, cumsum as stats_cumsum, max, mean, min, prod, std_,
    sum, var,
};

// Quantile-based
pub use ferrum_stats::{median, percentile, quantile};

// NaN-aware reductions
pub use ferrum_stats::{
    nancumprod as stats_nancumprod, nancumsum as stats_nancumsum, nanmax, nanmean, nanmedian,
    nanmin, nanpercentile, nanprod, nanstd, nansum, nanvar,
};

// Correlation and covariance
pub use ferrum_stats::{CorrelateMode, corrcoef, correlate, cov};

// Histogram
pub use ferrum_stats::{Bins, bincount, digitize, histogram, histogram2d, histogramdd};

// Sorting and searching
pub use ferrum_stats::{Side, SortKind, argsort, searchsorted, sort};
pub use ferrum_stats::{UniqueResult, count_nonzero, nonzero, unique, where_};

// Set operations
pub use ferrum_stats::{in1d, intersect1d, isin, setdiff1d, setxor1d, union1d};

// ---------------------------------------------------------------------------
// REQ-2: Submodule namespaces
// ---------------------------------------------------------------------------

/// I/O operations: .npy, .npz, text files, memory mapping.
#[cfg(feature = "io")]
pub use ferrum_io as io;

/// Linear algebra operations.
#[cfg(feature = "linalg")]
pub use ferrum_linalg as linalg;

/// FFT operations.
#[cfg(feature = "fft")]
pub use ferrum_fft as fft;

/// Random number generation and distributions.
#[cfg(feature = "random")]
pub use ferrum_random as random;

/// Polynomial operations.
#[cfg(feature = "polynomial")]
pub use ferrum_polynomial as polynomial;

/// Window functions.
#[cfg(feature = "window")]
pub use ferrum_window as window;

/// String operations on character arrays.
#[cfg(feature = "strings")]
pub use ferrum_strings as strings;

/// Masked arrays.
#[cfg(feature = "ma")]
pub use ferrum_ma as ma;

/// Stride manipulation utilities.
#[cfg(feature = "stride-tricks")]
pub use ferrum_stride_tricks as stride_tricks;

/// Python/NumPy interop via PyO3.
#[cfg(feature = "numpy")]
pub use ferrum_numpy_interop as numpy_interop;

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
    fn threshold_constants_accessible() {
        assert!(threshold::PARALLEL_THRESHOLD_ELEMENTWISE > 0);
        assert!(threshold::PARALLEL_THRESHOLD_COMPUTE > 0);
    }
}
