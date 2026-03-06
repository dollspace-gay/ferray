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
pub use ferrum_core::Array;
pub use ferrum_core::ArrayView;
pub use ferrum_core::ArrayViewMut;
pub use ferrum_core::ArcArray;
pub use ferrum_core::CowArray;
pub use ferrum_core::ArrayFlags;
pub use ferrum_core::DynArray;
pub use ferrum_core::FieldDescriptor;
pub use ferrum_core::FerrumRecord;
pub use ferrum_core::AsRawBuffer;

// Dimension types
pub use ferrum_core::dimension::{self, Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};

// Dtype system
pub use ferrum_core::{DType, Element, SliceInfoElem};
pub use ferrum_core::dtype::casting;
pub use ferrum_core::dtype::promotion;
pub use ferrum_core::dtype::finfo;

// Error handling
pub use ferrum_core::{FerrumError, FerrumResult};

// Memory layout
pub use ferrum_core::MemoryLayout;

// Macros
pub use ferrum_core::{s, promoted_type, FerrumRecord as DeriveFerrumRecord};

// Type aliases
pub use ferrum_core::aliases;

// Display configuration
pub use ferrum_core::array::display::{set_print_options, get_print_options};

// ---------------------------------------------------------------------------
// REQ-3a: Constants (ferrum::PI, ferrum::INF, etc.)
// ---------------------------------------------------------------------------

pub use ferrum_core::constants::{PI, E, INF, NEG_INF, NAN, EULER_GAMMA, PZERO, NZERO, NEWAXIS};

// ---------------------------------------------------------------------------
// REQ-1: Array creation functions at top level
// ---------------------------------------------------------------------------

pub use ferrum_core::creation::{
    array, asarray, zeros, ones, full,
    zeros_like, ones_like, full_like,
    empty, frombuffer, fromiter,
    arange, linspace, logspace, geomspace,
    eye, identity, diag, diagflat, tri, tril, triu,
    meshgrid, mgrid, ogrid,
};

// ---------------------------------------------------------------------------
// REQ-1: Manipulation functions at top level
// ---------------------------------------------------------------------------

pub use ferrum_core::manipulation::{
    reshape, ravel, flatten, transpose, swapaxes, moveaxis,
    expand_dims, squeeze, concatenate, stack, split,
    vstack, hstack, dstack, block, array_split, vsplit, hsplit, dsplit,
    flip, flipud, fliplr, rot90, roll, rollaxis,
    broadcast_to,
};

pub use ferrum_core::manipulation::extended::{
    pad, tile, repeat, resize,
    append, insert, delete, trim_zeros,
};

// ---------------------------------------------------------------------------
// REQ-1: Ufunc functions at top level
// ---------------------------------------------------------------------------

// Trigonometric
pub use ferrum_ufunc::{
    sin, cos, tan, arcsin, arccos, arctan, arctan2,
    sinh, cosh, tanh, arcsinh, arccosh, arctanh,
    deg2rad, rad2deg, degrees, radians, hypot, unwrap,
};

// Exponential and logarithmic
pub use ferrum_ufunc::{
    exp, exp2, expm1, log, log2, log10, log1p,
    logaddexp, logaddexp2,
};

// Rounding
pub use ferrum_ufunc::{round, floor, ceil, trunc, fix, rint, around};

// Arithmetic
pub use ferrum_ufunc::{
    add, subtract, multiply, divide, power, sqrt, square,
    absolute, negative, positive, sign, remainder, mod_, fmod,
    floor_divide, true_divide, divmod, reciprocal,
    cumsum, cumprod, diff, gradient, ediff1d,
    cross, trapezoid, heaviside,
    gcd, lcm, cbrt, fabs,
    nancumsum, nancumprod,
    add_reduce, add_accumulate, add_broadcast,
    subtract_broadcast, multiply_broadcast, multiply_outer,
    divide_broadcast,
};

// Float intrinsics
pub use ferrum_ufunc::{
    isnan, isinf, isfinite, isposinf, isneginf, signbit,
    clip, nan_to_num, nextafter, spacing, copysign,
    maximum, minimum, fmax, fmin,
    float_power, frexp, ldexp,
};

// Complex
pub use ferrum_ufunc::{abs, real, imag, conj, conjugate, angle};

// Bitwise
pub use ferrum_ufunc::{
    bitwise_and, bitwise_or, bitwise_xor, bitwise_not, invert,
    left_shift, right_shift,
};

// Comparison
pub use ferrum_ufunc::{
    equal, not_equal, less, less_equal, greater, greater_equal,
    allclose, isclose, array_equal, array_equiv,
};

// Logical
pub use ferrum_ufunc::{
    logical_and, logical_or, logical_xor, logical_not,
    all, any,
};

// Special
pub use ferrum_ufunc::{sinc, i0};

// Convolution and interpolation
pub use ferrum_ufunc::{convolve, ConvolveMode, interp, interp_one};

// Operator-style functions
pub use ferrum_ufunc::{
    array_add, array_sub, array_mul, array_div, array_rem, array_neg,
    array_bitand, array_bitor, array_bitxor, array_bitnot,
    array_shl, array_shr,
};

// ---------------------------------------------------------------------------
// REQ-1: Stats functions at top level
// ---------------------------------------------------------------------------

// Reductions
pub use ferrum_stats::{
    sum, prod, min, max, argmin, argmax, mean, var, std_,
    cumsum as stats_cumsum, cumprod as stats_cumprod,
};

// Quantile-based
pub use ferrum_stats::{median, percentile, quantile};

// NaN-aware reductions
pub use ferrum_stats::{
    nansum, nanprod, nanmin, nanmax, nanmean, nanvar, nanstd,
    nanmedian, nanpercentile,
    nancumsum as stats_nancumsum, nancumprod as stats_nancumprod,
};

// Correlation and covariance
pub use ferrum_stats::{correlate, corrcoef, cov, CorrelateMode};

// Histogram
pub use ferrum_stats::{histogram, histogram2d, histogramdd, bincount, digitize, Bins};

// Sorting and searching
pub use ferrum_stats::{sort, argsort, searchsorted, SortKind, Side};
pub use ferrum_stats::{unique, nonzero, where_, count_nonzero, UniqueResult};

// Set operations
pub use ferrum_stats::{union1d, intersect1d, setdiff1d, setxor1d, in1d, isin};

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
        PARALLEL_THRESHOLD_ELEMENTWISE,
        PARALLEL_THRESHOLD_COMPUTE,
        PARALLEL_THRESHOLD_REDUCTION,
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
