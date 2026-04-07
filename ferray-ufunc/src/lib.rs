// ferray-ufunc: Universal functions and SIMD-accelerated elementwise operations
//
// This crate provides 100+ NumPy-equivalent ufunc functions for the ferray
// N-dimensional array library. All functions are generic over element type
// and dimensionality, preserving the input array's rank.
//
// Features:
// - Trigonometric, exponential, logarithmic, rounding, arithmetic functions
// - Cumulative operations (cumsum, cumprod, diff, gradient)
// - Convolution and interpolation
// - Float intrinsics (isnan, clip, nan_to_num, etc.)
// - Complex number operations
// - Bitwise and logical operations
// - Comparison functions (allclose, isclose, etc.)
// - Special functions (sinc, i0)
// - SIMD dispatch via pulp (REQ-17)
// - Rayon parallelism for large arrays (REQ-21)
// - Operator overloads (+, -, *, /, %, &, |, ^, !, <<, >>)

pub mod cr_math;
pub mod dispatch;
pub mod fast_exp;
pub mod helpers;
pub mod kernels;
pub mod operator_overloads;
pub mod ops;
pub mod parallel;

// ---------------------------------------------------------------------------
// Public re-exports — flat namespace for ergonomic use
// ---------------------------------------------------------------------------

// Trigonometric
pub use ops::trig::{
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, cos, cosh, deg2rad, degrees, hypot,
    rad2deg, radians, sin, sinh, tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ops::explog::{exp, exp_fast, exp2, expm1, log, log1p, log2, log10, logaddexp, logaddexp2};

// Rounding
pub use ops::rounding::{around, ceil, fix, floor, rint, round, trunc};

// Arithmetic
pub use ops::arithmetic::{
    absolute, add, add_accumulate, add_broadcast, add_reduce, cbrt, cross, cumprod, cumsum, diff,
    divide, divide_broadcast, divmod, ediff1d, fabs, floor_divide, fmod, gcd, gcd_int, gradient,
    heaviside, lcm, lcm_int, mod_, multiply, multiply_broadcast, multiply_outer, nancumprod,
    nancumsum, negative, positive, power, reciprocal, remainder, sign, sqrt, square, subtract,
    subtract_broadcast, trapezoid, true_divide,
};

// Float intrinsics
pub use ops::floatintrinsic::{
    clip, clip_ord, copysign, float_power, fmax, fmin, frexp, isfinite, isinf, isnan, isneginf,
    isposinf, ldexp, maximum, minimum, nan_to_num, nextafter, signbit, spacing,
};

// Complex
pub use ops::complex::{abs, angle, conj, conjugate, imag, real};

// Bitwise
pub use ops::bitwise::{
    BitwiseOps, ShiftOps, bitwise_and, bitwise_not, bitwise_or, bitwise_xor, invert, left_shift,
    right_shift,
};

// Comparison
pub use ops::comparison::{
    allclose, array_equal, array_equiv, equal, greater, greater_equal, isclose, less, less_equal,
    not_equal,
};

// Logical
pub use ops::logical::{Logical, all, any, logical_and, logical_not, logical_or, logical_xor};

// Special
pub use ops::special::{i0, sinc};

// Convolution
pub use ops::convolution::{ConvolveMode, convolve};

// Interpolation
pub use ops::interpolation::{interp, interp_one};

// f16 variants (feature-gated)
#[cfg(feature = "f16")]
pub use ops::arithmetic::{
    absolute_f16, add_f16, cbrt_f16, divide_f16, floor_divide_f16, multiply_f16, negative_f16,
    power_f16, reciprocal_f16, remainder_f16, sign_f16, sqrt_f16, square_f16, subtract_f16,
};
#[cfg(feature = "f16")]
pub use ops::explog::{exp_f16, exp2_f16, expm1_f16, log_f16, log1p_f16, log2_f16, log10_f16};
#[cfg(feature = "f16")]
pub use ops::floatintrinsic::{
    clip_f16, isfinite_f16, isinf_f16, isnan_f16, maximum_f16, minimum_f16, nan_to_num_f16,
};
#[cfg(feature = "f16")]
pub use ops::rounding::{ceil_f16, floor_f16, round_f16, trunc_f16};
#[cfg(feature = "f16")]
pub use ops::special::sinc_f16;
#[cfg(feature = "f16")]
pub use ops::trig::{
    arccos_f16, arccosh_f16, arcsin_f16, arcsinh_f16, arctan_f16, arctan2_f16, arctanh_f16,
    cos_f16, cosh_f16, degrees_f16, hypot_f16, radians_f16, sin_f16, sinh_f16, tan_f16, tanh_f16,
};

// Operator-style convenience functions
pub use operator_overloads::{
    array_add, array_bitand, array_bitnot, array_bitor, array_bitxor, array_div, array_mul,
    array_neg, array_rem, array_shl, array_shr, array_sub,
};
