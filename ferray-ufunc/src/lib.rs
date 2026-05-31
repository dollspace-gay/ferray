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
//
// ## REQ status — crate-root re-export + ufunc-registration surface
//
// This module is the crate root: the public re-export surface that registers
// every routed ufunc family. Per-REQ detail lives in each op module's own
// `## REQ status` table; this root tracks the REQ-1 free-function convention
// and the registration surface itself.
//
// SHIPPED:
//   - REQ-1 (ufuncs are generic free functions preserving input
//     dimensionality, NOT a `Ufunc`/`IxDyn` trait): enforced workspace-wide —
//     every `pub use` block below re-exports `fn op<T, D>(&Array<T, D>) ->
//     FerrayResult<Array<T, D>>`-shaped free functions, so `sin` on an
//     `Array2<f64>` returns `Array2<f64>`. The crate root IS the registration
//     surface: `pub use ops::trig::{…}` (REQ-5), `pub use ops::explog::{…}`
//     (REQ-6), `pub use ops::rounding::{…}` (REQ-7/REQ-24),
//     `pub use ops::arithmetic::{…}` (REQ-8), `pub use ops::floatintrinsic::{…}`
//     (REQ-10), `pub use ops::complex::{…}` (REQ-11), `pub use ops::datetime::
//     {…}` (#942), `pub use ops::bitwise::{…}` (REQ-12), `pub use
//     ops::comparison::{…}` (REQ-14/REQ-15), `pub use ops::logical::{…}`
//     (REQ-16), `pub use operator_overloads::{…}` (REQ-9/REQ-13), plus the
//     `pub use promoted::{…}` REQ-23/REQ-25 int/bool-promotion families and the
//     `pub use ufunc_object::{Ufunc, …}` first-class ufunc objects (REQ-3
//     reduce/accumulate/outer). REQ-2 (`UnaryOp`/`BinaryOp` kernel-dispatch
//     traits) is intentionally crate-internal, NOT re-exported here.
//   - Crate-level lint configuration: the `#![allow(...)]` block below is a
//     crate-root allow scoped to the numerical-kernel cast/float-cmp lints
//     that the ufunc contracts deliberately incur (documented inline).
//   - Non-test production consumer: these re-exports ARE the public API that
//     ferray-python's binding crate (`ferray-python/src/ufunc.rs`) and external
//     callers import (`use ferray_ufunc::{sin, add, isnan, …}`). The crate root
//     has no callee of its own — it is the registration boundary.
//
// NOT-STARTED: none — REQ-1's free-function convention is shipped for every
// routed family; per-REQ shipped/not-started detail is in the op modules.

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
    arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, cos, cos_fast, cos_into, cosh,
    deg2rad, degrees, hypot, rad2deg, radians, sin, sin_fast, sin_into, sinh, tan, tanh, unwrap,
};

// Exponential and logarithmic
pub use ops::explog::{
    exp, exp_fast, exp_into, exp2, expm1, log, log_into, log1p, log2, log10, logaddexp, logaddexp2,
};

// Rounding
pub use ops::rounding::{around, ceil, fix, floor, rint, round, trunc};

// Rounding — integer/bool input identity (REQ-24): floor/ceil/trunc/round/
// fix/around register `TD(bints)` first in NumPy, so integer input is
// returned unchanged in the input dtype (NOT promoted to float). `rint` has
// no `TD(bints)` and instead promotes to float — that path is `rint_promote`
// in the unary-promote family below.
pub use ops::rounding::{around_int, ceil_int, fix_int, floor_int, round_int, trunc_int};

// Rounding — bool-input round/around promote to float16 (REQ-24 bool
// exception): `round`/`around` dispatch to `ndarray.round` (no `TD(bints)`,
// like `rint`), so `np.round(bool)→float16`, unlike floor/ceil/trunc/fix
// which keep bool. Feature-gated on `f16` (the promoted output type).
#[cfg(feature = "f16")]
pub use ops::rounding::{around_bool, round_bool};

// Arithmetic
pub use ops::arithmetic::{
    TrueDivide, WrappingArith, absolute, absolute_int, absolute_into, add, add_accumulate,
    add_broadcast, add_into, add_reduce, add_reduce_all, add_reduce_axes, add_reduce_keepdims,
    cbrt, cross, cumprod, cumsum, cumulative_prod, cumulative_sum, diff, divide, divide_broadcast,
    divide_into, divmod, ediff1d, fabs, floor_divide, floor_divide_int, fmod, gcd, gcd_int,
    gradient, heaviside, lcm, lcm_int, mod_, mod_int, multiply, multiply_broadcast, multiply_into,
    multiply_outer, nan_add_reduce, nan_add_reduce_all, nan_add_reduce_axes, nan_max_reduce,
    nan_max_reduce_all, nan_max_reduce_axes, nan_min_reduce, nan_min_reduce_all,
    nan_min_reduce_axes, nan_multiply_reduce, nan_multiply_reduce_all, nan_multiply_reduce_axes,
    nancumprod, nancumsum, negative, negative_int, negative_into, positive, power, power_int,
    reciprocal, remainder, remainder_int, sign, sign_int, sqrt, sqrt_into, square, square_into,
    subtract, subtract_broadcast, subtract_into, trapezoid, true_divide,
};

// Float intrinsics
pub use ops::floatintrinsic::{
    clip, clip_ord, copysign, float_power, fmax, fmin, frexp, isfinite, isinf, isnan, isneginf,
    isposinf, ldexp, maximum, maximum_ord, minimum, minimum_ord, modf, nan_to_num, nextafter,
    signbit, spacing,
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
#[cfg(feature = "fft-convolve")]
pub use ops::convolution::fftconvolve;
pub use ops::convolution::{ConvolveMode, convolve};

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

// Unary float-ufunc integer/bool input promotion (REQ-23): the entire
// transcendental / trig / sqrt / rint family accepts integer and bool input
// and promotes to the smallest safe-cast float (bool/int8/uint8 -> float16,
// int16/uint16 -> float32, int32/int64/uint32/uint64 -> float64), matching
// NumPy's ufunc dtype resolution. The output element type is
// `<T as PromoteFloat>::Out`.
pub use promoted::{
    PromoteFloat, arccos_promote, arccosh_promote, arcsin_promote, arcsinh_promote, arctan_promote,
    arctanh_promote, cbrt_promote, cos_promote, cosh_promote, exp_promote, exp2_promote,
    expm1_promote, fabs_promote, log_promote, log1p_promote, log2_promote, log10_promote,
    rint_promote, sin_promote, sinh_promote, sqrt_promote, tan_promote, tanh_promote,
};

// Binary float-ufunc integer/bool input promotion (REQ-25): hypot/arctan2/
// logaddexp/logaddexp2/copysign/nextafter accept an integer/bool input pair
// and promote to the smallest safe-cast float (same kind rule as REQ-23,
// reusing the `PromoteFloat` trait above), matching NumPy's ufunc dtype
// resolution. The output element type is `<T as PromoteFloat>::Out`.
pub use promoted::{
    arctan2_promote, copysign_promote, hypot_promote, logaddexp_promote, logaddexp2_promote,
    nextafter_promote,
};

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
