// ferrum-ufunc: Universal functions and SIMD-accelerated elementwise operations
//
// This crate provides 100+ NumPy-equivalent ufunc functions for the ferrum
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

pub mod dispatch;
pub mod helpers;
pub mod kernels;
pub mod ops;
pub mod operator_overloads;
pub mod parallel;

// ---------------------------------------------------------------------------
// Public re-exports — flat namespace for ergonomic use
// ---------------------------------------------------------------------------

// Trigonometric
pub use ops::trig::{
    arccos, arcsin, arctan, arctan2, arccosh, arcsinh, arctanh,
    cos, cosh, deg2rad, degrees, hypot, rad2deg, radians, sin, sinh,
    tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ops::explog::{
    exp, exp2, expm1, log, log10, log1p, log2, logaddexp, logaddexp2,
};

// Rounding
pub use ops::rounding::{around, ceil, fix, floor, rint, round, trunc};

// Arithmetic
pub use ops::arithmetic::{
    absolute, add, add_accumulate, add_broadcast, add_reduce,
    cbrt, cross, cumprod, cumsum, diff, divide, divide_broadcast,
    divmod, ediff1d, fabs, floor_divide, fmod, gcd, gradient, heaviside,
    lcm, mod_, multiply, multiply_broadcast, multiply_outer, nancumprod,
    nancumsum, negative, positive, power, reciprocal, remainder, sign,
    sqrt, square, subtract, subtract_broadcast, trapezoid, true_divide,
};

// Float intrinsics
pub use ops::floatintrinsic::{
    clip, copysign, float_power, fmax, fmin, frexp, isfinite, isinf,
    isnan, isneginf, isposinf, ldexp, maximum, minimum, nan_to_num,
    nextafter, signbit, spacing,
};

// Complex
pub use ops::complex::{abs, angle, conj, conjugate, imag, real};

// Bitwise
pub use ops::bitwise::{
    bitwise_and, bitwise_not, bitwise_or, bitwise_xor, invert,
    left_shift, right_shift, BitwiseOps, ShiftOps,
};

// Comparison
pub use ops::comparison::{
    allclose, array_equal, array_equiv, equal, greater, greater_equal,
    isclose, less, less_equal, not_equal,
};

// Logical
pub use ops::logical::{
    all, any, logical_and, logical_not, logical_or, logical_xor, Logical,
};

// Special
pub use ops::special::{i0, sinc};

// Convolution
pub use ops::convolution::{convolve, ConvolveMode};

// Interpolation
pub use ops::interpolation::{interp, interp_one};

// Operator-style convenience functions
pub use operator_overloads::{
    array_add, array_sub, array_mul, array_div, array_rem, array_neg,
    array_bitand, array_bitor, array_bitxor, array_bitnot,
    array_shl, array_shr,
};
