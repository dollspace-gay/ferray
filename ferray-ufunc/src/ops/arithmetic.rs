// ferray-ufunc: Arithmetic functions
//
// add, subtract, multiply, divide, true_divide, floor_divide, power,
// remainder, mod_, fmod, divmod, absolute, fabs, sign, negative, positive,
// reciprocal, sqrt, cbrt, square, heaviside, gcd, lcm
//
// Cumulative: cumsum, cumprod, nancumsum, nancumprod
// Differences: diff, ediff1d, gradient
// Products: cross
// Integration: trapezoid
//
// Reduction: add_reduce, add_accumulate, multiply_outer

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::helpers::{
    binary_broadcast_op, binary_elementwise_op, binary_elementwise_op_into, try_simd_f32_binary,
    try_simd_f64_binary, unary_float_op, unary_float_op_compute, unary_float_op_into,
};
use crate::kernels::simd_f32::{add_f32, div_f32, mul_f32, sub_f32};
use crate::kernels::simd_f64::{add_f64, div_f64, mul_f64, sub_f64};

// ---------------------------------------------------------------------------
// Basic arithmetic (binary, same-shape)
// ---------------------------------------------------------------------------

/// Elementwise addition with `NumPy` broadcasting.
///
/// Same-shape f64 / f32 inputs go through the explicit SIMD slice
/// kernel (`add_f64` / `add_f32` via `pulp::Arch` runtime dispatch).
/// Other dtypes and broadcasting paths fall through to the generic
/// auto-vectorised loop in `binary_elementwise_op`. (#88)
pub fn add<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    if let Some(r) = try_simd_f64_binary(a, b, add_f64) {
        return r;
    }
    if let Some(r) = try_simd_f32_binary(a, b, add_f32) {
        return r;
    }
    binary_elementwise_op(a, b, |x, y| x + y)
}

/// In-place elementwise addition, equivalent to `NumPy`'s
/// `np.add(a, b, out=out)`. Writes `a + b` directly into `out` without
/// allocating. All three arrays must be contiguous (C-order) and have the
/// same shape; broadcasting is not supported on this fast path — use
/// [`add`] if you need it.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if shapes differ.
/// - `FerrayError::InvalidValue` if any array is non-contiguous.
pub fn add_into<T, D>(a: &Array<T, D>, b: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    binary_elementwise_op_into(a, b, out, "add", |x, y| x + y)
}

/// Elementwise subtraction with `NumPy` broadcasting. SIMD-dispatched
/// for same-shape f64/f32 inputs (#88).
pub fn subtract<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
    D: Dimension,
{
    if let Some(r) = try_simd_f64_binary(a, b, sub_f64) {
        return r;
    }
    if let Some(r) = try_simd_f32_binary(a, b, sub_f32) {
        return r;
    }
    binary_elementwise_op(a, b, |x, y| x - y)
}

/// In-place subtraction — the `_into` counterpart of [`subtract`].
pub fn subtract_into<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    out: &mut Array<T, D>,
) -> FerrayResult<()>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
    D: Dimension,
{
    binary_elementwise_op_into(a, b, out, "subtract", |x, y| x - y)
}

/// Elementwise multiplication with `NumPy` broadcasting. SIMD-dispatched
/// for same-shape f64/f32 inputs (#88).
pub fn multiply<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    if let Some(r) = try_simd_f64_binary(a, b, mul_f64) {
        return r;
    }
    if let Some(r) = try_simd_f32_binary(a, b, mul_f32) {
        return r;
    }
    binary_elementwise_op(a, b, |x, y| x * y)
}

/// In-place multiplication — the `_into` counterpart of [`multiply`].
pub fn multiply_into<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    out: &mut Array<T, D>,
) -> FerrayResult<()>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    binary_elementwise_op_into(a, b, out, "multiply", |x, y| x * y)
}

/// Elementwise division with `NumPy` broadcasting. SIMD-dispatched
/// for same-shape f64/f32 inputs (#88).
pub fn divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Div<Output = T> + Copy,
    D: Dimension,
{
    if let Some(r) = try_simd_f64_binary(a, b, div_f64) {
        return r;
    }
    if let Some(r) = try_simd_f32_binary(a, b, div_f32) {
        return r;
    }
    binary_elementwise_op(a, b, |x, y| x / y)
}

/// In-place division — the `_into` counterpart of [`divide`].
pub fn divide_into<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    out: &mut Array<T, D>,
) -> FerrayResult<()>
where
    T: Element + std::ops::Div<Output = T> + Copy,
    D: Dimension,
{
    binary_elementwise_op_into(a, b, out, "divide", |x, y| x / y)
}

/// Alias for [`divide`] — true division (float).
pub fn true_divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| x / y)
}

/// Floor division: floor(a / b).
pub fn floor_divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| (x / y).floor())
}

/// Elementwise power: a^b.
pub fn power<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, num_traits::Float::powf)
}

/// Elementwise remainder (Python-style modulo).
pub fn remainder<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let z = <T as Element>::zero();
    binary_elementwise_op(a, b, |x, y| {
        let r = x % y;
        // Python/NumPy mod: result has same sign as divisor
        if (r < z && y > z) || (r > z && y < z) {
            r + y
        } else {
            r
        }
    })
}

/// Alias for [`remainder`].
pub fn mod_<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    remainder(a, b)
}

/// C-style fmod (remainder has same sign as dividend).
pub fn fmod<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| x % y)
}

/// Return `(floor_divide, remainder)` as a tuple of arrays, with broadcasting.
///
/// Computes both results in a single pass over the (broadcast) data,
/// avoiding the redundant division that would occur from calling
/// `floor_divide` and `remainder` separately.
pub fn divmod<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<(Array<T, D>, Array<T, D>)>
where
    T: Element + Float,
    D: Dimension,
{
    use ferray_core::dimension::broadcast::{broadcast_shapes, broadcast_to};

    let z = <T as Element>::zero();

    // Inline the divmod kernel so we can route both fast and broadcast
    // paths through a single closure body.
    let kernel = |x: T, y: T| -> (T, T) {
        let q = (x / y).floor();
        let mut r = x - q * y;
        if (r < z && y > z) || (r > z && y < z) {
            r = r + y;
        }
        (q, r)
    };

    // Fast path: identical shapes.
    if a.shape() == b.shape() {
        let mut quot_data = Vec::with_capacity(a.size());
        let mut rem_data = Vec::with_capacity(a.size());
        for (&x, &y) in a.iter().zip(b.iter()) {
            let (q, r) = kernel(x, y);
            quot_data.push(q);
            rem_data.push(r);
        }
        let quot = Array::from_vec(a.dim().clone(), quot_data)?;
        let rem = Array::from_vec(a.dim().clone(), rem_data)?;
        return Ok((quot, rem));
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "divmod: shapes {:?} and {:?} are not broadcast-compatible",
            a.shape(),
            b.shape()
        ))
    })?;
    let a_view = broadcast_to(a, &target_shape)?;
    let b_view = broadcast_to(b, &target_shape)?;
    let n: usize = target_shape.iter().product();
    let mut quot_data = Vec::with_capacity(n);
    let mut rem_data = Vec::with_capacity(n);
    for (&x, &y) in a_view.iter().zip(b_view.iter()) {
        let (q, r) = kernel(x, y);
        quot_data.push(q);
        rem_data.push(r);
    }
    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "divmod: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;
    let quot = Array::from_vec(result_dim.clone(), quot_data)?;
    let rem = Array::from_vec(result_dim, rem_data)?;
    Ok((quot, rem))
}

// ---------------------------------------------------------------------------
// Unary arithmetic
// ---------------------------------------------------------------------------

/// Elementwise absolute value.
///
/// Uses hardware SIMD for contiguous f64 arrays.
pub fn absolute<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if let Some(r) = crate::helpers::try_simd_f64_unary(input, crate::dispatch::simd_abs_f64) {
        return r;
    }
    if let Some(r) = crate::helpers::try_simd_f32_unary(input, crate::dispatch::simd_abs_f32) {
        return r;
    }
    unary_float_op(input, T::abs)
}

/// In-place elementwise absolute value — `_into` counterpart of [`absolute`].
pub fn absolute_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into(input, out, "absolute", T::abs)
}

/// Alias for [`absolute`] — float abs.
pub fn fabs<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    absolute(input)
}

/// Elementwise sign: -1 for negative, 0 for zero, +1 for positive.
pub fn sign<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, |x| {
        if x.is_nan() {
            <T as Float>::nan()
        } else if x > <T as Element>::zero() {
            <T as Element>::one()
        } else if x < <T as Element>::zero() {
            -<T as Element>::one()
        } else {
            <T as Element>::zero()
        }
    })
}

/// Elementwise negation.
///
/// Uses hardware SIMD for contiguous f64 arrays.
pub fn negative<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if let Some(r) = crate::helpers::try_simd_f64_unary(input, crate::dispatch::simd_neg_f64) {
        return r;
    }
    if let Some(r) = crate::helpers::try_simd_f32_unary(input, crate::dispatch::simd_neg_f32) {
        return r;
    }
    unary_float_op(input, |x| -x)
}

/// In-place elementwise negation — `_into` counterpart of [`negative`].
pub fn negative_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into(input, out, "negative", |x| -x)
}

/// Elementwise positive (identity for numeric types).
pub fn positive<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, |x| x)
}

/// Elementwise reciprocal: 1/x.
///
/// Uses hardware SIMD for contiguous f64 arrays.
pub fn reciprocal<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if let Some(r) = crate::helpers::try_simd_f64_unary(input, crate::dispatch::simd_reciprocal_f64)
    {
        return r;
    }
    if let Some(r) = crate::helpers::try_simd_f32_unary(input, crate::dispatch::simd_reciprocal_f32)
    {
        return r;
    }
    unary_float_op(input, T::recip)
}

/// Elementwise square root.
///
/// Uses hardware SIMD (`vsqrtpd`) for contiguous f64 arrays.
pub fn sqrt<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if let Some(r) = crate::helpers::try_simd_f64_unary(input, crate::dispatch::simd_sqrt_f64) {
        return r;
    }
    unary_float_op(input, T::sqrt)
}

/// In-place elementwise square root — the `_into` counterpart of [`sqrt`].
pub fn sqrt_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into(input, out, "sqrt", T::sqrt)
}

/// Elementwise cube root.
pub fn cbrt<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + crate::cr_math::CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_cbrt)
}

/// Elementwise square: x^2.
///
/// Uses hardware SIMD for contiguous f64 arrays.
pub fn square<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if let Some(r) = crate::helpers::try_simd_f64_unary(input, crate::dispatch::simd_square_f64) {
        return r;
    }
    if let Some(r) = crate::helpers::try_simd_f32_unary(input, crate::dispatch::simd_square_f32) {
        return r;
    }
    unary_float_op(input, |x| x * x)
}

/// In-place elementwise square — the `_into` counterpart of [`square`].
pub fn square_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into(input, out, "square", |x| x * x)
}

/// Heaviside step function.
///
/// `heaviside(x, h0)` returns 0 for x < 0, h0 for x == 0, and 1 for x > 0.
pub fn heaviside<T, D>(x: &Array<T, D>, h0: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(x, h0, |xi, h0i| {
        if xi.is_nan() {
            xi
        } else if xi < <T as Element>::zero() {
            <T as Element>::zero()
        } else if xi == <T as Element>::zero() {
            h0i
        } else {
            <T as Element>::one()
        }
    })
}

/// Integer GCD (works on float representations of integers).
pub fn gcd<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, |mut x, mut y| {
        if x.is_nan() || y.is_nan() {
            return T::nan();
        }
        x = x.abs();
        y = y.abs();
        while y != <T as Element>::zero() {
            let t = y;
            y = x % y;
            x = t;
        }
        x
    })
}

/// Integer LCM (works on float representations of integers).
pub fn lcm<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            return T::nan();
        }
        let ax = x.abs();
        let ay = y.abs();
        if ax == <T as Element>::zero() || ay == <T as Element>::zero() {
            return <T as Element>::zero();
        }
        // lcm = |a*b| / gcd(a,b)
        let mut gx = ax;
        let mut gy = ay;
        while gy != <T as Element>::zero() {
            let t = gy;
            gy = gx % gy;
            gx = t;
        }
        ax / gx * ay
    })
}

/// Integer GCD using the Euclidean algorithm.
///
/// Works on actual integer element types (i8, i16, i32, i64, u8, u16, u32, u64, etc.).
/// For float-typed arrays, use [`gcd`] instead.
pub fn gcd_int<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy + PartialEq + std::ops::Rem<Output = T> + num_traits::Signed,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| {
        let mut ax = x.abs();
        let mut ay = y.abs();
        while ay != <T as Element>::zero() {
            let t = ay;
            ay = ax % ay;
            ax = t;
        }
        ax
    })
}

/// Integer LCM using the Euclidean GCD algorithm.
///
/// Works on actual integer element types. For float-typed arrays, use [`lcm`] instead.
pub fn lcm_int<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element
        + Copy
        + PartialEq
        + std::ops::Rem<Output = T>
        + std::ops::Div<Output = T>
        + std::ops::Mul<Output = T>
        + num_traits::Signed,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| {
        let ax = x.abs();
        let ay = y.abs();
        if ax == <T as Element>::zero() || ay == <T as Element>::zero() {
            return <T as Element>::zero();
        }
        let mut gx = ax;
        let mut gy = ay;
        while gy != <T as Element>::zero() {
            let t = gy;
            gy = gx % gy;
            gx = t;
        }
        ax / gx * ay
    })
}

// ---------------------------------------------------------------------------
// Broadcasting binary arithmetic
// ---------------------------------------------------------------------------

/// Elementwise addition with broadcasting.
pub fn add_broadcast<T, D1, D2>(a: &Array<T, D1>, b: &Array<T, D2>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D1: Dimension,
    D2: Dimension,
{
    binary_broadcast_op(a, b, |x, y| x + y)
}

/// Elementwise subtraction with broadcasting.
pub fn subtract_broadcast<T, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
    D1: Dimension,
    D2: Dimension,
{
    binary_broadcast_op(a, b, |x, y| x - y)
}

/// Elementwise multiplication with broadcasting.
pub fn multiply_broadcast<T, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D1: Dimension,
    D2: Dimension,
{
    binary_broadcast_op(a, b, |x, y| x * y)
}

/// Elementwise division with broadcasting.
pub fn divide_broadcast<T, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Div<Output = T> + Copy,
    D1: Dimension,
    D2: Dimension,
{
    binary_broadcast_op(a, b, |x, y| x / y)
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

/// Reduce by addition along an axis (column sums, row sums, etc.).
///
/// Equivalent to `np.add.reduce(arr, axis=...)`. Delegates to the generic
/// [`crate::ufunc_methods::reduce_axis`] with the `+` kernel and `0` seed.
///
/// AC-2: `add_reduce` computes correct column sums.
pub fn add_reduce<T, D>(input: &Array<T, D>, axis: usize) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis(input, axis, <T as Element>::zero(), |acc, x| acc + x)
}

/// Reduce by addition along an axis with an optional `keepdims` flag.
///
/// Equivalent to `np.add.reduce(arr, axis=..., keepdims=...)` /
/// `np.sum(arr, axis=..., keepdims=...)`. When `keepdims = true` the
/// reduced axis is preserved as a size-1 dimension so the result is
/// broadcastable back against the original input — the classic pattern
/// for row/column centering (`arr - arr.sum(axis=1, keepdims=True)`).
///
/// With `keepdims = false` this behaves exactly like [`add_reduce`].
/// Added for #394.
pub fn add_reduce_keepdims<T, D>(
    input: &Array<T, D>,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis_keepdims(
        input,
        axis,
        <T as Element>::zero(),
        keepdims,
        |acc, x| acc + x,
    )
}

/// Reduce by addition over multiple axes simultaneously.
///
/// Equivalent to `np.add.reduce(arr, axis=axes, keepdims=keepdims)` /
/// `np.sum(arr, axis=axes, keepdims=keepdims)` where `axes` is a tuple
/// of axes to collapse. Reduces every listed axis in a single pass over
/// the input — never materializes intermediates the way chained
/// `add_reduce` calls would, and the order of `axes` is irrelevant.
/// Added for #395.
pub fn add_reduce_axes<T, D>(
    input: &Array<T, D>,
    axes: &[usize],
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axes(input, axes, <T as Element>::zero(), keepdims, |acc, x| {
        acc + x
    })
}

/// Reduce by addition over the entire array (the `axis=None` form).
///
/// Equivalent to `np.add.reduce(arr, axis=None)` / `np.sum(arr)`.
/// Returns a single scalar — use [`add_reduce_axes`] when you want a
/// wrapped array result that supports `keepdims`.
///
/// Added for #395.
pub fn add_reduce_all<T, D>(input: &Array<T, D>) -> T
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    crate::ufunc_methods::reduce_all(input, <T as Element>::zero(), |acc, x| acc + x)
}

// ---------------------------------------------------------------------------
// NaN-aware reductions (#388)
//
// Parallel to add_reduce / multiply_reduce / max_reduce / min_reduce but
// with NaN-skipping kernels. ferray-stats already exposes high-level
// nansum / nanmean / etc. wrappers; these are the lower-level ufunc
// primitives that match the cumulative nancumsum/nancumprod pattern in
// the same module — they live here so the full reduction family
// (whole-array + axis + axes + keepdims) is available without depending
// on ferray-stats.
//
// All four functions require `T: Element + Float` so the kernel can call
// `.is_nan()`. NaNs are dropped via per-element preprocessing into the
// reduction identity (0 for sum, 1 for product, +inf for min, -inf for
// max). Whole-array forms return a scalar; axis-aware forms delegate to
// the generic reduce_axes / reduce_axis_keepdims helpers.
// ---------------------------------------------------------------------------

/// Reduce by NaN-skipping addition along an axis with optional keepdims.
///
/// Equivalent to `np.nansum(arr, axis=axis, keepdims=keepdims)`. NaN
/// elements are treated as zero and contribute nothing to the sum.
pub fn nan_add_reduce<T, D>(
    input: &Array<T, D>,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis_keepdims(
        input,
        axis,
        <T as Element>::zero(),
        keepdims,
        |acc, x| acc + nan_to_zero(x),
    )
}

/// Reduce by NaN-skipping addition over multiple axes simultaneously.
pub fn nan_add_reduce_axes<T, D>(
    input: &Array<T, D>,
    axes: &[usize],
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axes(input, axes, <T as Element>::zero(), keepdims, |acc, x| {
        acc + nan_to_zero(x)
    })
}

/// Reduce by NaN-skipping addition over the entire array.
///
/// Equivalent to `np.nansum(arr)` / `np.nansum(arr, axis=None)`. Returns
/// a scalar. NaN elements contribute nothing to the sum; an array of all
/// NaNs sums to zero.
pub fn nan_add_reduce_all<T, D>(input: &Array<T, D>) -> T
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_all(input, <T as Element>::zero(), |acc, x| acc + nan_to_zero(x))
}

/// Reduce by NaN-skipping multiplication along an axis with optional keepdims.
///
/// Equivalent to `np.nanprod(arr, axis=axis, keepdims=keepdims)`. NaN
/// elements are treated as one and contribute nothing to the product.
pub fn nan_multiply_reduce<T, D>(
    input: &Array<T, D>,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis_keepdims(
        input,
        axis,
        <T as Element>::one(),
        keepdims,
        |acc, x| acc * nan_to_one(x),
    )
}

/// Reduce by NaN-skipping multiplication over multiple axes.
pub fn nan_multiply_reduce_axes<T, D>(
    input: &Array<T, D>,
    axes: &[usize],
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axes(input, axes, <T as Element>::one(), keepdims, |acc, x| {
        acc * nan_to_one(x)
    })
}

/// Reduce by NaN-skipping multiplication over the entire array.
pub fn nan_multiply_reduce_all<T, D>(input: &Array<T, D>) -> T
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_all(input, <T as Element>::one(), |acc, x| acc * nan_to_one(x))
}

/// Reduce by NaN-skipping maximum along an axis with optional keepdims.
///
/// Equivalent to `np.nanmax(arr, axis=axis, keepdims=keepdims)`. NaN
/// elements are skipped (treated as `-inf`).
pub fn nan_max_reduce<T, D>(
    input: &Array<T, D>,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis_keepdims(
        input,
        axis,
        <T as Float>::neg_infinity(),
        keepdims,
        |acc, x| {
            if x.is_nan() {
                acc
            } else if x > acc {
                x
            } else {
                acc
            }
        },
    )
}

/// Reduce by NaN-skipping maximum over multiple axes.
pub fn nan_max_reduce_axes<T, D>(
    input: &Array<T, D>,
    axes: &[usize],
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axes(
        input,
        axes,
        <T as Float>::neg_infinity(),
        keepdims,
        |acc, x| {
            if x.is_nan() {
                acc
            } else if x > acc {
                x
            } else {
                acc
            }
        },
    )
}

/// Reduce by NaN-skipping maximum over the entire array.
///
/// Equivalent to `np.nanmax(arr)`. Returns `-inf` for an all-NaN input
/// rather than raising — callers that need the all-NaN error semantics
/// should use ferray-stats' `nanmax` (which checks the result and errors
/// out instead of returning the seed).
pub fn nan_max_reduce_all<T, D>(input: &Array<T, D>) -> T
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_all(input, <T as Float>::neg_infinity(), |acc, x| {
        if x.is_nan() {
            acc
        } else if x > acc {
            x
        } else {
            acc
        }
    })
}

/// Reduce by NaN-skipping minimum along an axis with optional keepdims.
///
/// Equivalent to `np.nanmin(arr, axis=axis, keepdims=keepdims)`. NaN
/// elements are skipped (treated as `+inf`).
pub fn nan_min_reduce<T, D>(
    input: &Array<T, D>,
    axis: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axis_keepdims(
        input,
        axis,
        <T as Float>::infinity(),
        keepdims,
        |acc, x| {
            if x.is_nan() {
                acc
            } else if x < acc {
                x
            } else {
                acc
            }
        },
    )
}

/// Reduce by NaN-skipping minimum over multiple axes.
pub fn nan_min_reduce_axes<T, D>(
    input: &Array<T, D>,
    axes: &[usize],
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_axes(input, axes, <T as Float>::infinity(), keepdims, |acc, x| {
        if x.is_nan() {
            acc
        } else if x < acc {
            x
        } else {
            acc
        }
    })
}

/// Reduce by NaN-skipping minimum over the entire array.
pub fn nan_min_reduce_all<T, D>(input: &Array<T, D>) -> T
where
    T: Element + Float,
    D: Dimension,
{
    crate::ufunc_methods::reduce_all(input, <T as Float>::infinity(), |acc, x| {
        if x.is_nan() {
            acc
        } else if x < acc {
            x
        } else {
            acc
        }
    })
}

#[inline]
fn nan_to_zero<T: Float + Element>(x: T) -> T {
    if x.is_nan() {
        <T as Element>::zero()
    } else {
        x
    }
}

#[inline]
fn nan_to_one<T: Float + Element>(x: T) -> T {
    if x.is_nan() { <T as Element>::one() } else { x }
}

/// Running (cumulative) addition along an axis.
///
/// AC-2: `add_accumulate` produces running sums.
pub fn add_accumulate<T, D>(input: &Array<T, D>, axis: usize) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    cumsum(input, Some(axis))
}

/// Outer product: `multiply_outer(a, b)[i, j] = a[i] * b[j]`.
///
/// Equivalent to `np.multiply.outer(a, b)`. Delegates to the generic
/// [`crate::ufunc_methods::outer`] with the `*` kernel.
///
/// AC-3: `multiply_outer` produces correct outer product.
pub fn multiply_outer<T>(a: &Array<T, Ix1>, b: &Array<T, Ix1>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
{
    crate::ufunc_methods::outer(a, b, |x, y| x * y)
}

// ---------------------------------------------------------------------------
// Cumulative operations
// ---------------------------------------------------------------------------

/// Shared cumulative kernel: build the result buffer by applying
/// `preprocess` to every input element, then walk it in place with
/// `accumulate` along `axis` (or flat if `None`). Factored out so
/// `cumsum`, `cumprod`, `nancumsum` and `nancumprod` all share a single
/// pass — previously `nancumsum`/`nancumprod` materialized a cleaned
/// copy and then called `cumsum`/`cumprod`, which materialized a
/// second buffer (#156).
fn cumulative_with_preprocess<T, D, Pre, Acc>(
    input: &Array<T, D>,
    axis: Option<usize>,
    preprocess: Pre,
    accumulate: Acc,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    Pre: Fn(T) -> T,
    Acc: Fn(T, T) -> T,
{
    if let Some(ax) = axis {
        if ax >= input.ndim() {
            return Err(FerrayError::axis_out_of_bounds(ax, input.ndim()));
        }
        let shape = input.shape().to_vec();
        let mut result: Vec<T> = input.iter().map(|&x| preprocess(x)).collect();
        let mut stride = 1usize;
        for d in shape.iter().skip(ax + 1) {
            stride *= d;
        }
        let axis_len = shape[ax];
        let outer_size: usize = shape[..ax].iter().product();
        let inner_size = stride;

        for outer in 0..outer_size {
            for inner in 0..inner_size {
                let base = outer * axis_len * inner_size + inner;
                for k in 1..axis_len {
                    let prev = base + (k - 1) * inner_size;
                    let curr = base + k * inner_size;
                    result[curr] = accumulate(result[prev], result[curr]);
                }
            }
        }
        Array::from_vec(input.dim().clone(), result)
    } else {
        let mut data: Vec<T> = input.iter().map(|&x| preprocess(x)).collect();
        for i in 1..data.len() {
            data[i] = accumulate(data[i - 1], data[i]);
        }
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Cumulative sum along an axis (or flattened if axis is None).
///
/// When `axis=None`, data is flattened and accumulated, but the result retains
/// the original shape (unlike `NumPy` which returns a 1-D array). This is due to
/// the generic return type `Array<T, D>`.
///
/// AC-11: `cumsum([1,2,3,4]) == [1,3,6,10]`.
pub fn cumsum<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    cumulative_with_preprocess(input, axis, |x| x, |a, b| a + b)
}

/// Cumulative product along an axis (or flattened if axis is None).
///
/// When `axis=None`, data is flattened and accumulated, but the result retains
/// the original shape (unlike `NumPy` which returns a 1-D array). This is due to
/// the generic return type `Array<T, D>`.
pub fn cumprod<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    cumulative_with_preprocess(input, axis, |x| x, |a, b| a * b)
}

/// Cumulative sum (Array API standard name).
///
/// Alias of [`cumsum`] matching the Python Array API specification's
/// `cumulative_sum` name (added to `numpy` in 2.0).
pub fn cumulative_sum<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    cumsum(input, axis)
}

/// Cumulative product (Array API standard name).
///
/// Alias of [`cumprod`] matching the Python Array API specification's
/// `cumulative_prod` name (added to `numpy` in 2.0).
pub fn cumulative_prod<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    cumprod(input, axis)
}

/// Cumulative sum ignoring NaNs.
pub fn nancumsum<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    cumulative_with_preprocess(
        input,
        axis,
        |x| {
            if x.is_nan() {
                <T as Element>::zero()
            } else {
                x
            }
        },
        |a, b| a + b,
    )
}

/// Cumulative product ignoring NaNs.
pub fn nancumprod<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    cumulative_with_preprocess(
        input,
        axis,
        |x| {
            if x.is_nan() { <T as Element>::one() } else { x }
        },
        |a, b| a * b,
    )
}

// ---------------------------------------------------------------------------
// Differences
// ---------------------------------------------------------------------------

/// Compute the n-th discrete difference along the given axis.
///
/// AC-11: `diff([1,3,6,10], 1) == [2,3,4]`.
pub fn diff<T>(input: &Array<T, Ix1>, n: usize) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
{
    let mut data: Vec<T> = input.iter().copied().collect();
    for _ in 0..n {
        if data.len() <= 1 {
            data.clear();
            break;
        }
        let mut new_data = Vec::with_capacity(data.len() - 1);
        for i in 1..data.len() {
            new_data.push(data[i] - data[i - 1]);
        }
        data = new_data;
    }
    Array::from_vec(Ix1::new([data.len()]), data)
}

/// Differences between consecutive elements of an array, with optional
/// prepend/append values.
pub fn ediff1d<T>(
    input: &Array<T, Ix1>,
    to_end: Option<&[T]>,
    to_begin: Option<&[T]>,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
{
    let data: Vec<T> = input.iter().copied().collect();
    let mut result = Vec::new();

    if let Some(begin) = to_begin {
        result.extend_from_slice(begin);
    }

    for i in 1..data.len() {
        result.push(data[i] - data[i - 1]);
    }

    if let Some(end) = to_end {
        result.extend_from_slice(end);
    }

    Array::from_vec(Ix1::new([result.len()]), result)
}

/// Compute the gradient of a 1-D array using central differences.
///
/// Edge values use forward/backward differences.
pub fn gradient<T>(input: &Array<T, Ix1>, spacing: Option<T>) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + Float,
{
    let data: Vec<T> = input.iter().copied().collect();
    let n = data.len();
    if n == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    let h = spacing.unwrap_or_else(|| <T as Element>::one());
    let two = <T as Element>::one() + <T as Element>::one();
    let mut result = Vec::with_capacity(n);

    if n == 1 {
        result.push(<T as Element>::zero());
    } else {
        // Forward difference for first element
        result.push((data[1] - data[0]) / h);
        // Central differences for interior
        for i in 1..n - 1 {
            result.push((data[i + 1] - data[i - 1]) / (two * h));
        }
        // Backward difference for last element
        result.push((data[n - 1] - data[n - 2]) / h);
    }

    Array::from_vec(Ix1::new([n]), result)
}

// ---------------------------------------------------------------------------
// Cross product
// ---------------------------------------------------------------------------

/// Cross product of two 3-element 1-D arrays.
pub fn cross<T>(a: &Array<T, Ix1>, b: &Array<T, Ix1>) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + std::ops::Mul<Output = T> + std::ops::Sub<Output = T> + Copy,
{
    if a.size() != 3 || b.size() != 3 {
        return Err(FerrayError::invalid_value(
            "cross product requires 3-element vectors",
        ));
    }
    let ad: Vec<T> = a.iter().copied().collect();
    let bd: Vec<T> = b.iter().copied().collect();
    let result = vec![
        ad[1] * bd[2] - ad[2] * bd[1],
        ad[2] * bd[0] - ad[0] * bd[2],
        ad[0] * bd[1] - ad[1] * bd[0],
    ];
    Array::from_vec(Ix1::new([3]), result)
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

/// Integrate using the trapezoidal rule.
///
/// If `dx` is provided, it is the spacing between sample points.
/// If `x` is provided, it gives the sample point coordinates.
pub fn trapezoid<T>(y: &Array<T, Ix1>, x: Option<&Array<T, Ix1>>, dx: Option<T>) -> FerrayResult<T>
where
    T: Element + Float,
{
    let ydata: Vec<T> = y.iter().copied().collect();
    let n = ydata.len();
    if n < 2 {
        return Ok(<T as Element>::zero());
    }

    let two = <T as Element>::one() + <T as Element>::one();
    let mut total = <T as Element>::zero();

    if let Some(xarr) = x {
        let xdata: Vec<T> = xarr.iter().copied().collect();
        if xdata.len() != n {
            return Err(FerrayError::shape_mismatch(
                "x and y must have the same length for trapezoid",
            ));
        }
        for i in 1..n {
            total = total + (ydata[i] + ydata[i - 1]) / two * (xdata[i] - xdata[i - 1]);
        }
    } else {
        let h = dx.unwrap_or_else(|| <T as Element>::one());
        for i in 1..n {
            total = total + (ydata[i] + ydata[i - 1]) / two * h;
        }
    }

    Ok(total)
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted) — generated via the shared macros (#142).
// ---------------------------------------------------------------------------

use crate::helpers::{binary_f16_fn, unary_f16_fn};

unary_f16_fn!(
    /// Elementwise absolute value for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    absolute_f16,
    f32::abs
);
unary_f16_fn!(
    /// Elementwise negation for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    negative_f16,
    |x: f32| -x
);
unary_f16_fn!(
    /// Elementwise square root for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    sqrt_f16,
    f32::sqrt
);
unary_f16_fn!(
    /// Elementwise cube root for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    cbrt_f16,
    f32::cbrt
);
unary_f16_fn!(
    /// Elementwise square for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    square_f16,
    |x: f32| x * x
);
unary_f16_fn!(
    /// Elementwise reciprocal for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    reciprocal_f16,
    f32::recip
);
unary_f16_fn!(
    /// Elementwise sign for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    sign_f16,
    |x: f32| {
        if x.is_nan() {
            f32::NAN
        } else if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
);
binary_f16_fn!(
    /// Elementwise addition for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    add_f16,
    |x: f32, y: f32| x + y
);
binary_f16_fn!(
    /// Elementwise subtraction for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    subtract_f16,
    |x: f32, y: f32| x - y
);
binary_f16_fn!(
    /// Elementwise multiplication for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    multiply_f16,
    |x: f32, y: f32| x * y
);
binary_f16_fn!(
    /// Elementwise division for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    divide_f16,
    |x: f32, y: f32| x / y
);
binary_f16_fn!(
    /// Elementwise power for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    power_f16,
    f32::powf
);
binary_f16_fn!(
    /// Floor division for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    floor_divide_f16,
    |x: f32, y: f32| (x / y).floor()
);
binary_f16_fn!(
    /// Elementwise remainder for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    remainder_f16,
    |x: f32, y: f32| {
        let r = x % y;
        if (r < 0.0 && y > 0.0) || (r > 0.0 && y < 0.0) {
            r + y
        } else {
            r
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    use crate::test_util::arr1;

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_f32(data: Vec<f32>) -> Array<f32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    // ---- f32 coverage (#721) -------------------------------------------
    //
    // Existing arithmetic tests run against f64 only; the f32 SIMD
    // dispatch (try_simd_f32_binary / try_simd_f32_unary) is exercised
    // here. Arrays are sized at 32 elements so the SIMD path engages
    // (the threshold is >= a small multiple of the lane width).

    #[test]
    fn add_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| i as f32).collect());
        let b = arr1_f32((0..n).map(|i| i as f32 * 2.0).collect());
        let r = add(&a, &b).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32 * 3.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn sub_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| i as f32 * 5.0).collect());
        let b = arr1_f32((0..n).map(|i| i as f32 * 2.0).collect());
        let r = subtract(&a, &b).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32 * 3.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn mul_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| i as f32).collect());
        let b = arr1_f32(vec![3.0_f32; n]);
        let r = multiply(&a, &b).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32 * 3.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn div_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| (i as f32 + 1.0) * 4.0).collect());
        let b = arr1_f32(vec![2.0_f32; n]);
        let r = divide(&a, &b).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32 + 1.0) * 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn abs_f32_simd_path() {
        let n = 32;
        let a = arr1_f32(
            (0..n)
                .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
                .collect(),
        );
        let r = absolute(&a).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32)).abs() < 1e-6);
        }
    }

    #[test]
    fn neg_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| i as f32).collect());
        let r = negative(&a).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (-(i as f32))).abs() < 1e-6);
        }
    }

    #[test]
    fn reciprocal_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((1..=n).map(|i| i as f32).collect());
        let r = reciprocal(&a).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            let want = 1.0_f32 / ((i + 1) as f32);
            assert!((v - want).abs() < 1e-6);
        }
    }

    #[test]
    fn square_f32_simd_path() {
        let n = 32;
        let a = arr1_f32((0..n).map(|i| i as f32).collect());
        let r = square(&a).unwrap();
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            let want = (i as f32) * (i as f32);
            assert!((v - want).abs() < 1e-4);
        }
    }

    #[test]
    fn add_f32_below_simd_threshold_scalar_path() {
        // Tiny array — SIMD dispatch typically falls back to scalar.
        let a = arr1_f32(vec![1.5, 2.5, 3.5]);
        let b = arr1_f32(vec![0.5, 0.5, 0.5]);
        let r = add(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn add_f32_force_scalar_env_var() {
        // FERRAY_FORCE_SCALAR=1 should bypass SIMD; result must still
        // be correct.
        // SAFETY: This test is single-threaded by default per cargo
        // test runner; we set then unset the env var around the call.
        unsafe {
            std::env::set_var("FERRAY_FORCE_SCALAR", "1");
        }
        let a = arr1_f32((0..32).map(|i| i as f32).collect());
        let b = arr1_f32(vec![1.0_f32; 32]);
        let r = add(&a, &b).unwrap();
        unsafe {
            std::env::remove_var("FERRAY_FORCE_SCALAR");
        }
        for (i, &v) in r.as_slice().unwrap().iter().enumerate() {
            assert!((v - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![4.0, 5.0, 6.0]);
        let r = add(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtract() {
        let a = arr1(vec![5.0, 7.0, 9.0]);
        let b = arr1(vec![1.0, 2.0, 3.0]);
        let r = subtract(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_multiply() {
        let a = arr1(vec![2.0, 3.0, 4.0]);
        let b = arr1(vec![5.0, 6.0, 7.0]);
        let r = multiply(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_divide() {
        let a = arr1(vec![10.0, 20.0, 30.0]);
        let b = arr1(vec![2.0, 4.0, 5.0]);
        let r = divide(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn test_floor_divide() {
        let a = arr1(vec![7.0, -7.0]);
        let b = arr1(vec![2.0, 2.0]);
        let r = floor_divide(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[3.0, -4.0]);
    }

    #[test]
    fn test_power() {
        let a = arr1(vec![2.0, 3.0]);
        let b = arr1(vec![3.0, 2.0]);
        let r = power(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[8.0, 9.0]);
    }

    #[test]
    fn test_remainder() {
        let a = arr1(vec![7.0, -7.0]);
        let b = arr1(vec![3.0, 3.0]);
        let r = remainder(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_fmod() {
        let a = arr1(vec![7.0, -7.0]);
        let b = arr1(vec![3.0, 3.0]);
        let r = fmod(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_absolute() {
        let a = arr1(vec![-1.0, 2.0, -3.0]);
        let r = absolute(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sign() {
        let a = arr1(vec![-5.0, 0.0, 3.0]);
        let r = sign(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[-1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_negative() {
        let a = arr1(vec![1.0, -2.0, 3.0]);
        let r = negative(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_sqrt() {
        let a = arr1(vec![1.0, 4.0, 9.0, 16.0]);
        let r = sqrt(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_cbrt() {
        let a = arr1(vec![8.0, 27.0]);
        let r = cbrt(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 2.0).abs() < 1e-12);
        assert!((s[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_square() {
        let a = arr1(vec![2.0, 3.0, 4.0]);
        let r = square(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_reciprocal() {
        let a = arr1(vec![2.0, 4.0, 5.0]);
        let r = reciprocal(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0.5, 0.25, 0.2]);
    }

    #[test]
    fn test_heaviside() {
        let x = arr1(vec![-1.0, 0.0, 1.0]);
        let h0 = arr1(vec![0.5, 0.5, 0.5]);
        let r = heaviside(&x, &h0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0.0, 0.5, 1.0]);
    }

    #[test]
    fn test_gcd() {
        let a = arr1(vec![12.0, 15.0]);
        let b = arr1(vec![8.0, 25.0]);
        let r = gcd(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn test_lcm() {
        let a = arr1(vec![4.0, 6.0]);
        let b = arr1(vec![6.0, 8.0]);
        let r = lcm(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[12.0, 24.0]);
    }

    #[test]
    fn test_gcd_int() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![12, 15, 0]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![8, 25, 7]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4, 5, 7]);
    }

    #[test]
    fn test_lcm_int() {
        let a = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![4, 6, 0]).unwrap();
        let b = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![6, 8, 5]).unwrap();
        let r = lcm_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[12, 24, 0]);
    }

    #[test]
    fn test_gcd_int_negative() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![-12, 15]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![8, -25]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4, 5]);
    }

    #[test]
    fn test_cumsum_ac11() {
        // AC-11: cumsum([1,2,3,4]) == [1,3,6,10]
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let r = cumsum(&a, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_i32() {
        let a = arr1_i32(vec![1, 2, 3, 4]);
        let r = cumsum(&a, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1, 3, 6, 10]);
    }

    #[test]
    fn test_cumprod() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let r = cumprod(&a, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cumulative_sum_alias() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let r = cumulative_sum(&a, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumulative_prod_alias() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let r = cumulative_prod(&a, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_diff_ac11() {
        // AC-11: diff([1,3,6,10], 1) == [2,3,4]
        let a = arr1(vec![1.0, 3.0, 6.0, 10.0]);
        let r = diff(&a, 1).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_diff_n2() {
        let a = arr1(vec![1.0, 3.0, 6.0, 10.0]);
        let r = diff(&a, 2).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 1.0]);
    }

    #[test]
    fn test_ediff1d() {
        let a = arr1(vec![1.0, 2.0, 4.0, 7.0]);
        let r = ediff1d(&a, None, None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gradient() {
        let a = arr1(vec![1.0, 2.0, 4.0, 7.0, 11.0]);
        let r = gradient(&a, None).unwrap();
        let s = r.as_slice().unwrap();
        // forward: 2-1=1, central: (4-1)/2=1.5, (7-2)/2=2.5, (11-4)/2=3.5, backward: 11-7=4
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 1.5).abs() < 1e-12);
        assert!((s[2] - 2.5).abs() < 1e-12);
        assert!((s[3] - 3.5).abs() < 1e-12);
        assert!((s[4] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_cross() {
        let a = arr1(vec![1.0, 0.0, 0.0]);
        let b = arr1(vec![0.0, 1.0, 0.0]);
        let r = cross(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_trapezoid() {
        // Integrate y=x from 0 to 4: area = 8
        let y = arr1(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let r = trapezoid(&y, None, Some(1.0)).unwrap();
        assert!((r - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_trapezoid_with_x() {
        let y = arr1(vec![0.0, 1.0, 4.0]);
        let x = arr1(vec![0.0, 1.0, 2.0]);
        let r = trapezoid(&y, Some(&x), None).unwrap();
        // (0+1)/2*1 + (1+4)/2*1 = 0.5 + 2.5 = 3.0
        assert!((r - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_add_reduce_ac2() {
        // AC-2: add_reduce computes correct column sums
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let r = add_reduce(&a, 0).unwrap();
        assert_eq!(r.shape(), &[3]);
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_accumulate_ac2() {
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        let r = add_accumulate(&a, 0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn add_reduce_keepdims_true_preserves_row_axis() {
        // (2,3) + axis=1 + keepdims=true → (2,1) with row sums.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let r = add_reduce_keepdims(&a, 1, true).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn add_reduce_keepdims_true_preserves_col_axis() {
        // (2,3) + axis=0 + keepdims=true → (1,3) with column sums.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let r = add_reduce_keepdims(&a, 0, true).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn add_reduce_keepdims_false_matches_legacy_add_reduce() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let legacy = add_reduce(&a, 1).unwrap();
        let new_false = add_reduce_keepdims(&a, 1, false).unwrap();
        assert_eq!(legacy.shape(), new_false.shape());
        assert_eq!(legacy.as_slice().unwrap(), new_false.as_slice().unwrap());
    }

    #[test]
    fn add_reduce_axes_two_axes_3d() {
        // (2, 3, 4) reducing axes (0, 2) → length-3 result.
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = add_reduce_axes(&a, &[0, 2], false).unwrap();
        assert_eq!(r.shape(), &[3]);
        // For each j in 0..3: sum_{i,k} (i*12 + j*4 + k)
        let expected: Vec<f64> = (0..3)
            .map(|j| {
                let mut s = 0.0;
                for i in 0..2 {
                    for k in 0..4 {
                        s += f64::from(i * 12 + j * 4 + k);
                    }
                }
                s
            })
            .collect();
        assert_eq!(r.as_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn add_reduce_axes_keepdims_preserves_rank() {
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = add_reduce_axes(&a, &[0, 2], true).unwrap();
        assert_eq!(r.shape(), &[1, 3, 1]);
    }

    #[test]
    fn add_reduce_axes_all_axes_collapses_to_scalar_array() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let r = add_reduce_axes(&a, &[0, 1], false).unwrap();
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.as_slice().unwrap(), &[21.0]);
    }

    #[test]
    fn add_reduce_all_returns_scalar_sum() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let s = add_reduce_all(&a);
        assert!((s - 21.0).abs() < 1e-12);
    }

    #[test]
    fn add_reduce_all_integer_input_works() {
        // Multi-axis reductions must work for integer Element types too.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let s = add_reduce_all(&a);
        assert_eq!(s, 21);
    }

    // ---- nan-aware reductions (#388) ----

    #[test]
    fn nan_add_reduce_all_skips_nans() {
        let a = arr1(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0]);
        let s = nan_add_reduce_all(&a);
        assert!((s - 9.0).abs() < 1e-12);
    }

    #[test]
    fn nan_add_reduce_all_nans_only_returns_zero() {
        let a = arr1(vec![f64::NAN, f64::NAN]);
        let s = nan_add_reduce_all(&a);
        assert!((s - 0.0).abs() < 1e-12);
    }

    #[test]
    fn nan_add_reduce_axis_skips_nans_per_row() {
        // (2, 3) with row-1 having a NaN; reduce axis=1 → row sums.
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0])
                .unwrap();
        let r = nan_add_reduce(&a, 1, false).unwrap();
        assert_eq!(r.shape(), &[2]);
        let s = r.as_slice().unwrap();
        assert!((s[0] - 6.0).abs() < 1e-12);
        assert!((s[1] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn nan_add_reduce_axes_multi_axis_skips_nans() {
        use ferray_core::dimension::Ix3;
        // (2, 2, 2) with one NaN; reduce axes (0, 2).
        let data = vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0, 7.0, 8.0];
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 2, 2]), data).unwrap();
        let r = nan_add_reduce_axes(&a, &[0, 2], false).unwrap();
        assert_eq!(r.shape(), &[2]);
        // For j=0: sum(1, 2, NaN→0, 6) = 9.0
        // For j=1: sum(3, 4, 7, 8) = 22.0
        let s = r.as_slice().unwrap();
        assert!((s[0] - 9.0).abs() < 1e-12);
        assert!((s[1] - 22.0).abs() < 1e-12);
    }

    #[test]
    fn nan_multiply_reduce_all_skips_nans() {
        let a = arr1(vec![2.0, f64::NAN, 3.0, f64::NAN, 4.0]);
        let p = nan_multiply_reduce_all(&a);
        assert!((p - 24.0).abs() < 1e-12);
    }

    #[test]
    fn nan_multiply_reduce_all_nans_only_returns_one() {
        let a = arr1(vec![f64::NAN, f64::NAN]);
        let p = nan_multiply_reduce_all(&a);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nan_multiply_reduce_axis_per_row() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![2.0, 3.0, 4.0, 5.0, f64::NAN, 6.0])
                .unwrap();
        let r = nan_multiply_reduce(&a, 1, false).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 24.0).abs() < 1e-12); // 2*3*4
        assert!((s[1] - 30.0).abs() < 1e-12); // 5*1*6
    }

    #[test]
    fn nan_max_reduce_all_skips_nans() {
        let a = arr1(vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0, 2.0]);
        let m = nan_max_reduce_all(&a);
        assert!((m - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nan_max_reduce_all_nans_only_returns_neg_infinity() {
        let a = arr1(vec![f64::NAN, f64::NAN]);
        let m = nan_max_reduce_all(&a);
        assert!(m.is_infinite() && m.is_sign_negative());
    }

    #[test]
    fn nan_max_reduce_axis_per_row_with_nans() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0, 4.0],
        )
        .unwrap();
        let r = nan_max_reduce(&a, 1, false).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 3.0).abs() < 1e-12);
        assert!((s[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn nan_min_reduce_all_skips_nans() {
        let a = arr1(vec![5.0, f64::NAN, 3.0, f64::NAN, 1.0, 4.0]);
        let m = nan_min_reduce_all(&a);
        assert!((m - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nan_min_reduce_all_nans_only_returns_infinity() {
        let a = arr1(vec![f64::NAN, f64::NAN]);
        let m = nan_min_reduce_all(&a);
        assert!(m.is_infinite() && m.is_sign_positive());
    }

    #[test]
    fn nan_min_reduce_axis_per_row_with_nans() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![5.0, f64::NAN, 3.0, f64::NAN, 5.0, 4.0],
        )
        .unwrap();
        let r = nan_min_reduce(&a, 1, false).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 3.0).abs() < 1e-12);
        assert!((s[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn nan_reductions_with_no_nans_match_regular_reductions() {
        // When the input has no NaNs the nan-aware versions must give
        // the exact same result as the regular reductions.
        let a = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((nan_add_reduce_all(&a) - 15.0).abs() < 1e-12);
        assert!((nan_multiply_reduce_all(&a) - 120.0).abs() < 1e-12);
        assert!((nan_max_reduce_all(&a) - 5.0).abs() < 1e-12);
        assert!((nan_min_reduce_all(&a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn nan_add_reduce_keepdims_preserves_axis() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0])
                .unwrap();
        let r = nan_add_reduce(&a, 1, true).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
    }

    #[test]
    fn test_multiply_outer_ac3() {
        // AC-3: multiply_outer produces correct outer product
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![4.0, 5.0]);
        let r = multiply_outer(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_nancumsum() {
        let a = arr1(vec![1.0, f64::NAN, 3.0, 4.0]);
        let r = nancumsum(&a, None).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 1.0); // NaN treated as 0
        assert_eq!(s[2], 4.0);
        assert_eq!(s[3], 8.0);
    }

    #[test]
    fn test_nancumprod() {
        let a = arr1(vec![1.0, f64::NAN, 3.0, 4.0]);
        let r = nancumprod(&a, None).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 1.0); // NaN treated as 1
        assert_eq!(s[2], 3.0);
        assert_eq!(s[3], 12.0);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let r = add_broadcast(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
    }

    #[test]
    fn test_divmod() {
        let a = arr1(vec![7.0, -7.0]);
        let b = arr1(vec![3.0, 3.0]);
        let (q, r) = divmod(&a, &b).unwrap();
        assert_eq!(q.as_slice().unwrap(), &[2.0, -3.0]);
        let rs = r.as_slice().unwrap();
        assert!((rs[0] - 1.0).abs() < 1e-12);
        assert!((rs[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_positive() {
        let a = arr1(vec![-1.0, 2.0]);
        let r = positive(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[-1.0, 2.0]);
    }

    #[test]
    fn test_true_divide() {
        let a = arr1(vec![10.0, 20.0]);
        let b = arr1(vec![3.0, 7.0]);
        let r = true_divide(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 10.0 / 3.0).abs() < 1e-12);
        assert!((s[1] - 20.0 / 7.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Broadcasting tests for arithmetic ops (issue #379)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_broadcasts_within_same_rank() {
        // (3, 1) + (1, 4) -> (3, 4) — both Ix2
        let col = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let row =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let r = add(&col, &row).unwrap();
        assert_eq!(r.shape(), &[3, 4]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![
                11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0,
            ]
        );
    }

    #[test]
    fn test_subtract_broadcasts() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = subtract(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
        );
    }

    #[test]
    fn test_multiply_broadcasts() {
        let col = Array::<i32, Ix2>::from_vec(Ix2::new([3, 1]), vec![1, 2, 3]).unwrap();
        let row = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![10, 20, 30]).unwrap();
        let r = multiply(&col, &row).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![10, 20, 30, 20, 40, 60, 30, 60, 90]
        );
    }

    #[test]
    fn test_divide_broadcasts() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![10.0, 5.0, 2.0]).unwrap();
        let r = divide(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 4.0, 15.0, 4.0, 10.0, 30.0]
        );
    }

    #[test]
    fn test_power_broadcasts() {
        let bases = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![2.0, 3.0, 4.0]).unwrap();
        let exps = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = power(&bases, &exps).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![2.0, 4.0, 8.0, 3.0, 9.0, 27.0, 4.0, 16.0, 64.0]
        );
    }

    #[test]
    fn test_remainder_broadcasts() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
                .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![3.0, 4.0, 5.0]).unwrap();
        let r = remainder(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![1.0, 0.0, 4.0, 1.0, 3.0, 2.0]
        );
    }

    #[test]
    fn test_divmod_broadcasts() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![7.0, 13.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 3.0, 4.0]).unwrap();
        // Need both inputs to have the same D for the typed divmod entry point.
        // Use the cross-rank broadcast helper instead — but divmod is typed,
        // so route via an explicit Ix2 reshape of b.
        let b2 = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![2.0, 3.0, 4.0]).unwrap();
        let (q, r) = divmod(&a, &b2).unwrap();
        assert_eq!(q.shape(), &[2, 3]);
        assert_eq!(r.shape(), &[2, 3]);
        // a broadcasts to [[7,7,7],[13,13,13]], b broadcasts to [[2,3,4],[2,3,4]]
        // divmod(7,2)=(3,1), divmod(7,3)=(2,1), divmod(7,4)=(1,3)
        // divmod(13,2)=(6,1), divmod(13,3)=(4,1), divmod(13,4)=(3,1)
        let q_vec: Vec<f64> = q.iter().copied().collect();
        let r_vec: Vec<f64> = r.iter().copied().collect();
        assert_eq!(q_vec, vec![3.0, 2.0, 1.0, 6.0, 4.0, 3.0]);
        assert_eq!(r_vec, vec![1.0, 1.0, 3.0, 1.0, 1.0, 1.0]);
        let _ = b; // silence unused
    }

    #[test]
    fn test_gcd_int_broadcasts() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 1]), vec![12, 18, 24]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 2]), vec![8, 9]).unwrap();
        let r = gcd_int(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        // gcd(12,8)=4, gcd(12,9)=3, gcd(18,8)=2, gcd(18,9)=9, gcd(24,8)=8, gcd(24,9)=3
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![4, 3, 2, 9, 8, 3]
        );
    }

    #[test]
    fn test_lcm_int_broadcasts() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![4, 6]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 2]), vec![6, 8]).unwrap();
        let r = lcm_int(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // lcm(4,6)=12, lcm(4,8)=8, lcm(6,6)=6, lcm(6,8)=24
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![12, 8, 6, 24]);
    }

    #[test]
    fn test_add_incompatible_shapes_errors() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(add(&a, &b).is_err());
    }

    #[cfg(feature = "f16")]
    mod f16_tests {
        use super::*;

        fn arr1_f16(data: &[f32]) -> Array<half::f16, Ix1> {
            let n = data.len();
            let vals: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            Array::from_vec(Ix1::new([n]), vals).unwrap()
        }

        #[test]
        fn test_add_f16() {
            let a = arr1_f16(&[1.0, 2.0, 3.0]);
            let b = arr1_f16(&[4.0, 5.0, 6.0]);
            let r = add_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 5.0).abs() < 0.01);
            assert!((s[1].to_f32() - 7.0).abs() < 0.01);
            assert!((s[2].to_f32() - 9.0).abs() < 0.01);
        }

        #[test]
        fn test_multiply_f16() {
            let a = arr1_f16(&[2.0, 3.0]);
            let b = arr1_f16(&[4.0, 5.0]);
            let r = multiply_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 8.0).abs() < 0.01);
            assert!((s[1].to_f32() - 15.0).abs() < 0.1);
        }

        #[test]
        fn test_sqrt_f16() {
            let a = arr1_f16(&[1.0, 4.0, 9.0, 16.0]);
            let r = sqrt_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() - 2.0).abs() < 0.01);
            assert!((s[2].to_f32() - 3.0).abs() < 0.01);
            assert!((s[3].to_f32() - 4.0).abs() < 0.01);
        }

        #[test]
        fn test_absolute_f16() {
            let a = arr1_f16(&[-1.0, 2.0, -3.0]);
            let r = absolute_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() - 2.0).abs() < 0.01);
            assert!((s[2].to_f32() - 3.0).abs() < 0.01);
        }

        #[test]
        fn test_power_f16() {
            let a = arr1_f16(&[2.0, 3.0]);
            let b = arr1_f16(&[3.0, 2.0]);
            let r = power_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 8.0).abs() < 0.1);
            assert!((s[1].to_f32() - 9.0).abs() < 0.1);
        }

        #[test]
        fn test_divide_f16() {
            let a = arr1_f16(&[10.0, 20.0]);
            let b = arr1_f16(&[2.0, 4.0]);
            let r = divide_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 5.0).abs() < 0.01);
            assert!((s[1].to_f32() - 5.0).abs() < 0.01);
        }
    }

    // -----------------------------------------------------------------------
    // In-place (_into) variants (issue #378)
    // -----------------------------------------------------------------------

    mod into_tests {
        use super::*;
        use ferray_core::Array;
        use ferray_core::dimension::Ix1;

        fn arr(data: &[f64]) -> Array<f64, Ix1> {
            Array::<f64, Ix1>::from_vec(Ix1::new([data.len()]), data.to_vec()).unwrap()
        }

        #[test]
        fn add_into_writes_result() {
            let a = arr(&[1.0, 2.0, 3.0]);
            let b = arr(&[10.0, 20.0, 30.0]);
            let mut out = arr(&[0.0, 0.0, 0.0]);
            add_into(&a, &b, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[11.0, 22.0, 33.0]);
        }

        #[test]
        fn subtract_into_writes_result() {
            let a = arr(&[10.0, 20.0, 30.0]);
            let b = arr(&[1.0, 2.0, 3.0]);
            let mut out = arr(&[0.0, 0.0, 0.0]);
            subtract_into(&a, &b, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[9.0, 18.0, 27.0]);
        }

        #[test]
        fn multiply_into_writes_result() {
            let a = arr(&[1.0, 2.0, 3.0]);
            let b = arr(&[4.0, 5.0, 6.0]);
            let mut out = arr(&[0.0; 3]);
            multiply_into(&a, &b, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[4.0, 10.0, 18.0]);
        }

        #[test]
        fn divide_into_writes_result() {
            let a = arr(&[10.0, 20.0, 30.0]);
            let b = arr(&[2.0, 4.0, 6.0]);
            let mut out = arr(&[0.0; 3]);
            divide_into(&a, &b, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[5.0, 5.0, 5.0]);
        }

        #[test]
        fn add_into_shape_mismatch_errors() {
            let a = arr(&[1.0, 2.0, 3.0]);
            let b = arr(&[1.0, 2.0]);
            let mut out = arr(&[0.0, 0.0, 0.0]);
            assert!(add_into(&a, &b, &mut out).is_err());
        }

        #[test]
        fn add_into_out_shape_mismatch_errors() {
            let a = arr(&[1.0, 2.0, 3.0]);
            let b = arr(&[4.0, 5.0, 6.0]);
            let mut out = arr(&[0.0, 0.0]); // wrong size
            assert!(add_into(&a, &b, &mut out).is_err());
        }

        #[test]
        fn sqrt_into_writes_result() {
            let a = arr(&[1.0, 4.0, 9.0, 16.0]);
            let mut out = arr(&[0.0; 4]);
            sqrt_into(&a, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        }

        #[test]
        fn square_into_writes_result() {
            let a = arr(&[1.0, -2.0, 3.0, -4.0]);
            let mut out = arr(&[0.0; 4]);
            square_into(&a, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
        }

        #[test]
        fn absolute_into_writes_result() {
            let a = arr(&[-1.0, 2.0, -3.0]);
            let mut out = arr(&[0.0; 3]);
            absolute_into(&a, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        }

        #[test]
        fn negative_into_writes_result() {
            let a = arr(&[1.0, -2.0, 3.0]);
            let mut out = arr(&[0.0; 3]);
            negative_into(&a, &mut out).unwrap();
            assert_eq!(out.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
        }

        #[test]
        fn into_variants_are_chainable_no_alloc() {
            // A realistic pattern: apply a pipeline in-place over and over
            // without touching the allocator after initial setup.
            let mut state = arr(&[1.0, 2.0, 3.0, 4.0]);
            let ones = arr(&[1.0; 4]);
            let mut scratch = arr(&[0.0; 4]);
            for _ in 0..100 {
                add_into(&state, &ones, &mut scratch).unwrap();
                std::mem::swap(&mut state, &mut scratch);
            }
            // After 100 increments of 1: [101, 102, 103, 104]
            assert_eq!(state.as_slice().unwrap(), &[101.0, 102.0, 103.0, 104.0]);
        }

        #[test]
        fn exp_into_matches_exp() {
            use crate::ops::explog::{exp, exp_into};
            let a = arr(&[0.0, 1.0, 2.0]);
            let expected = exp(&a).unwrap();
            let mut out = arr(&[0.0; 3]);
            exp_into(&a, &mut out).unwrap();
            for (&x, &y) in expected
                .as_slice()
                .unwrap()
                .iter()
                .zip(out.as_slice().unwrap().iter())
            {
                assert!((x - y).abs() < 1e-14);
            }
        }

        #[test]
        fn sin_into_matches_sin() {
            use crate::ops::trig::{sin, sin_into};
            let a = arr(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
            let expected = sin(&a).unwrap();
            let mut out = arr(&[0.0; 3]);
            sin_into(&a, &mut out).unwrap();
            for (&x, &y) in expected
                .as_slice()
                .unwrap()
                .iter()
                .zip(out.as_slice().unwrap().iter())
            {
                assert!((x - y).abs() < 1e-14);
            }
        }

        #[test]
        fn cos_into_matches_cos() {
            use crate::ops::trig::{cos, cos_into};
            let a = arr(&[0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
            let expected = cos(&a).unwrap();
            let mut out = arr(&[0.0; 3]);
            cos_into(&a, &mut out).unwrap();
            for (&x, &y) in expected
                .as_slice()
                .unwrap()
                .iter()
                .zip(out.as_slice().unwrap().iter())
            {
                assert!((x - y).abs() < 1e-14);
            }
        }

        #[test]
        fn log_into_matches_log() {
            use crate::ops::explog::{log, log_into};
            let a = arr(&[1.0, std::f64::consts::E, 10.0]);
            let expected = log(&a).unwrap();
            let mut out = arr(&[0.0; 3]);
            log_into(&a, &mut out).unwrap();
            for (&x, &y) in expected
                .as_slice()
                .unwrap()
                .iter()
                .zip(out.as_slice().unwrap().iter())
            {
                assert!((x - y).abs() < 1e-14);
            }
        }
    }
}
