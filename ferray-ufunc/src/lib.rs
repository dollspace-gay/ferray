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

// Ufunc kernels do bit-level float manipulation (`fast_exp`, `fast_trig`,
// `frexp`, `nextafter`, `spacing`), evaluate libm/CORE-MATH-style range
// reductions that necessarily mix integer and float types, and surface
// rounding intrinsics (`floor`, `ceil`, `trunc`, `rint`, `fix`) whose
// contract is a lossy `f64 -> i64` cast. These casts and float
// comparisons are part of the function contracts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Numerical kernels use i, j, k, n, m as canonical loop / shape names.
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms,
    // `mul_add` is an accuracy/throughput tradeoff that the project
    // chooses on a per-kernel basis; clippy can't tell which is right.
    clippy::suboptimal_flops,
    // Iterative solvers genuinely need to compare floats in their loop
    // condition (Newton's method, fixed-point iteration).
    clippy::while_float
)]

pub mod cr_math;
pub mod dispatch;
pub mod errstate;
pub mod fast_exp;
pub mod fast_trig;
pub mod helpers;
pub mod kernels;
pub mod operator_overloads;
pub mod ops;
pub mod parallel;
pub mod promoted;
#[cfg(test)]
mod test_util;
pub mod ufunc_methods;
pub mod ufunc_object;

// ---------------------------------------------------------------------------
// Public re-exports — flat namespace for ergonomic use
// ---------------------------------------------------------------------------

// Trigonometric
pub use ops::trig::{
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, cos, cos_into, cosh, deg2rad,
    degrees, hypot, rad2deg, radians, sin, sin_into, sinh, tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ops::explog::{
    exp, exp_fast, exp_into, exp2, expm1, log, log_into, log1p, log2, log10, logaddexp, logaddexp2,
};

// Rounding
pub use ops::rounding::{around, ceil, fix, floor, rint, round, trunc};

// Arithmetic
pub use ops::arithmetic::{
    absolute, absolute_into, add, add_accumulate, add_broadcast, add_into, add_reduce,
    add_reduce_all, add_reduce_axes, add_reduce_keepdims, cbrt, cross, cumprod, cumsum,
    cumulative_prod, cumulative_sum, diff, divide, divide_broadcast, divide_into, divmod, ediff1d,
    fabs, floor_divide, fmod, gcd, gcd_int, gradient, heaviside, lcm, lcm_int, mod_, multiply,
    multiply_broadcast, multiply_into, multiply_outer, nan_add_reduce, nan_add_reduce_all,
    nan_add_reduce_axes, nan_max_reduce, nan_max_reduce_all, nan_max_reduce_axes, nan_min_reduce,
    nan_min_reduce_all, nan_min_reduce_axes, nan_multiply_reduce, nan_multiply_reduce_all,
    nan_multiply_reduce_axes, nancumprod, nancumsum, negative, negative_into, positive, power,
    reciprocal, remainder, sign, sqrt, sqrt_into, square, square_into, subtract,
    subtract_broadcast, subtract_into, trapezoid, true_divide,
};

// Float intrinsics
pub use ops::floatintrinsic::{
    clip, clip_ord, copysign, float_power, fmax, fmin, frexp, isfinite, isinf, isnan, isneginf,
    isposinf, ldexp, maximum, minimum, modf, nan_to_num, nextafter, signbit, spacing,
};

// Complex
pub use ops::complex::{
    abs, angle, conj, conjugate, imag, iscomplex, iscomplex_real, iscomplexobj, isreal,
    isreal_real, isrealobj, isscalar, real,
};

// Complex transcendentals (sin/cos/tan, sinh/cosh/tanh, asin/acos/atan,
// asinh/acosh/atanh, exp, expm1, ln (natural log), log2, log10, log1p,
// sqrt, power) — NumPy parity for `np.sin(complex_array)` and friends.
// Suffixed with _complex to avoid name clash with the real-valued ufuncs
// re-exported above.
pub use ops::complex::{
    acos_complex, acosh_complex, asin_complex, asinh_complex, atan_complex, atanh_complex,
    cos_complex, cosh_complex, exp_complex, expm1_complex, ln_complex, log1p_complex, log2_complex,
    log10_complex, power_complex, sin_complex, sinh_complex, sqrt_complex, tan_complex,
    tanh_complex,
};

// Datetime / timedelta
pub use ops::datetime::{
    add_datetime_timedelta, add_datetime_timedelta_promoted, add_timedelta, add_timedelta_promoted,
    isnat_datetime, isnat_timedelta, sub_datetime, sub_datetime_promoted, sub_datetime_timedelta,
    sub_timedelta,
};

// Bitwise
pub use ops::bitwise::{
    BitwiseCount, BitwiseOps, ShiftOps, bitwise_and, bitwise_count, bitwise_not, bitwise_or,
    bitwise_xor, invert, left_shift, right_shift,
};

// Comparison
pub use ops::comparison::{
    allclose, array_equal, array_equiv, equal, equal_broadcast, greater, greater_broadcast,
    greater_equal, greater_equal_broadcast, isclose, isclose_broadcast, less, less_broadcast,
    less_equal, less_equal_broadcast, not_equal, not_equal_broadcast,
};

// Logical
pub use ops::logical::{
    Logical, all, all_axis, any, any_axis, logical_and, logical_not, logical_or, logical_xor,
};

// Special
pub use ops::special::{i0, sinc};

// Convolution
pub use ops::convolution::{ConvolveMode, convolve};
#[cfg(feature = "fft-convolve")]
pub use ops::convolution::fftconvolve;

// Interpolation
pub use ops::interpolation::{interp, interp_one};

// Generic ufunc methods (reduce / accumulate / outer / at) — work with
// any `Fn(T, T) -> T` op, equivalent to NumPy's `np.<ufunc>.reduce` etc.
pub use ufunc_methods::{
    accumulate_axis, at, outer as ufunc_outer, reduce_all, reduce_axes, reduce_axis,
    reduce_axis_keepdims,
};

// Mixed-type (promoted) arithmetic — `np.add(i32_arr, f64_arr)` style.
// Uses ferray-core's `Promoted` trait to resolve the output type at
// compile time and `PromoteTo` to cast both operands.
pub use promoted::{add_promoted, divide_promoted, multiply_promoted, subtract_promoted};

// First-class ufunc objects: `add_ufunc().reduce(arr, 0)` etc. The
// [`Ufunc`] struct wraps any binary op + identity and exposes the
// NumPy ufunc methods (.call / .reduce / .accumulate / .outer / .at).
pub use ufunc_object::{Ufunc, add_ufunc, divide_ufunc, multiply_ufunc, subtract_ufunc};

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
