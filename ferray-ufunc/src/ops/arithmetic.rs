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

use crate::helpers::{binary_broadcast_op, binary_float_op, unary_float_op};

// ---------------------------------------------------------------------------
// Basic arithmetic (binary, same-shape)
// ---------------------------------------------------------------------------

/// Elementwise addition.
pub fn add<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "add: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise subtraction.
pub fn subtract<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "subtract: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise multiplication.
pub fn multiply<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "multiply: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise division.
pub fn divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Div<Output = T> + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "divide: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Alias for [`divide`] — true division (float).
pub fn true_divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x / y)
}

/// Floor division: floor(a / b).
pub fn floor_divide<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| (x / y).floor())
}

/// Elementwise power: a^b.
pub fn power<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x.powf(y))
}

/// Elementwise remainder (Python-style modulo).
pub fn remainder<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let z = <T as Element>::zero();
    binary_float_op(a, b, |x, y| {
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
    binary_float_op(a, b, |x, y| x % y)
}

/// Return (floor_divide, remainder) as a tuple of arrays.
pub fn divmod<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<(Array<T, D>, Array<T, D>)>
where
    T: Element + Float,
    D: Dimension,
{
    Ok((floor_divide(a, b)?, remainder(a, b)?))
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
    unary_float_op(input, T::abs)
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
    unary_float_op(input, |x| -x)
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

/// Elementwise cube root.
pub fn cbrt<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + crate::cr_math::CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_cbrt)
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
    unary_float_op(input, |x| x * x)
}

/// Heaviside step function.
///
/// `heaviside(x, h0)` returns 0 for x < 0, h0 for x == 0, and 1 for x > 0.
pub fn heaviside<T, D>(x: &Array<T, D>, h0: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(x, h0, |xi, h0i| {
        if xi < <T as Element>::zero() {
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
    binary_float_op(a, b, |mut x, mut y| {
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
    binary_float_op(a, b, |x, y| {
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
/// AC-2: `add_reduce` computes correct column sums.
pub fn add_reduce<T, D>(input: &Array<T, D>, axis: usize) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = input.shape().to_vec();
    let axis_len = shape[axis];

    // Compute output shape: remove the axis
    let mut out_shape: Vec<usize> = Vec::with_capacity(ndim - 1);
    for (i, &s) in shape.iter().enumerate() {
        if i != axis {
            out_shape.push(s);
        }
    }
    let out_size: usize = out_shape.iter().product();

    // Compute strides for input traversal
    let mut stride = 1usize;
    for d in shape.iter().skip(axis + 1) {
        stride *= d;
    }
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size = stride;

    let data: Vec<T> = input.iter().copied().collect();
    let mut result = vec![<T as Element>::zero(); out_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let mut acc = <T as Element>::zero();
            for k in 0..axis_len {
                let idx = outer * axis_len * inner_size + k * inner_size + inner;
                acc = acc + data[idx];
            }
            result[outer * inner_size + inner] = acc;
        }
    }

    if out_shape.is_empty() {
        out_shape.push(1);
    }
    Array::from_vec(IxDyn::from(&out_shape[..]), result)
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

/// Outer product: multiply_outer(a, b)[i, j] = a[i] * b[j].
///
/// AC-3: multiply_outer produces correct outer product.
pub fn multiply_outer<T>(a: &Array<T, Ix1>, b: &Array<T, Ix1>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
{
    use ferray_core::dimension::Ix2;
    let m = a.size();
    let n = b.size();
    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();
    let mut data = Vec::with_capacity(m * n);
    for &ai in &a_data {
        for &bj in &b_data {
            data.push(ai * bj);
        }
    }
    let arr = Array::from_vec(Ix2::new([m, n]), data)?;
    // Convert to IxDyn
    let dyn_data: Vec<T> = arr.iter().copied().collect();
    Array::from_vec(IxDyn::from(&[m, n][..]), dyn_data)
}

// ---------------------------------------------------------------------------
// Cumulative operations
// ---------------------------------------------------------------------------

/// Cumulative sum along an axis (or flattened if axis is None).
///
/// AC-11: `cumsum([1,2,3,4]) == [1,3,6,10]`.
pub fn cumsum<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    if let Some(ax) = axis {
        if ax >= input.ndim() {
            return Err(FerrayError::axis_out_of_bounds(ax, input.ndim()));
        }
        // Work along the given axis
        let shape = input.shape().to_vec();
        let data: Vec<T> = input.iter().copied().collect();
        let mut result = data.clone();
        // Compute strides manually
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
                    result[curr] = result[prev] + result[curr];
                }
            }
        }
        Array::from_vec(input.dim().clone(), result)
    } else {
        // Flatten and cumsum
        let mut data: Vec<T> = input.iter().copied().collect();
        for i in 1..data.len() {
            data[i] = data[i - 1] + data[i];
        }
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Cumulative product along an axis (or flattened if axis is None).
pub fn cumprod<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    if let Some(ax) = axis {
        if ax >= input.ndim() {
            return Err(FerrayError::axis_out_of_bounds(ax, input.ndim()));
        }
        let shape = input.shape().to_vec();
        let data: Vec<T> = input.iter().copied().collect();
        let mut result = data.clone();
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
                    result[curr] = result[prev] * result[curr];
                }
            }
        }
        Array::from_vec(input.dim().clone(), result)
    } else {
        let mut data: Vec<T> = input.iter().copied().collect();
        for i in 1..data.len() {
            data[i] = data[i - 1] * data[i];
        }
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Cumulative sum ignoring NaNs.
pub fn nancumsum<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    // Replace NaN with zero, then cumsum
    let cleaned: Vec<T> = input
        .iter()
        .map(|&x| {
            if x.is_nan() {
                <T as Element>::zero()
            } else {
                x
            }
        })
        .collect();
    let arr = Array::from_vec(input.dim().clone(), cleaned)?;
    cumsum(&arr, axis)
}

/// Cumulative product ignoring NaNs.
pub fn nancumprod<T, D>(input: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let cleaned: Vec<T> = input
        .iter()
        .map(|&x| if x.is_nan() { <T as Element>::one() } else { x })
        .collect();
    let arr = Array::from_vec(input.dim().clone(), cleaned)?;
    cumprod(&arr, axis)
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
// f16 variants (f32-promoted)
// ---------------------------------------------------------------------------

/// Elementwise absolute value for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn absolute_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::abs)
}

/// Elementwise negation for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn negative_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, |x| -x)
}

/// Elementwise square root for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn sqrt_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::sqrt)
}

/// Elementwise cube root for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn cbrt_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::cbrt)
}

/// Elementwise square for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn square_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, |x| x * x)
}

/// Elementwise reciprocal for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn reciprocal_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::recip)
}

/// Elementwise sign for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn sign_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, |x| {
        if x.is_nan() {
            f32::NAN
        } else if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    })
}

/// Elementwise addition for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn add_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| x + y)
}

/// Elementwise subtraction for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn subtract_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| x - y)
}

/// Elementwise multiplication for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn multiply_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| x * y)
}

/// Elementwise division for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn divide_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| x / y)
}

/// Elementwise power for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn power_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, f32::powf)
}

/// Floor division for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn floor_divide_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| (x / y).floor())
}

/// Elementwise remainder for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn remainder_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| {
        let r = x % y;
        if (r < 0.0 && y > 0.0) || (r > 0.0 && y < 0.0) {
            r + y
        } else {
            r
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
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
}
