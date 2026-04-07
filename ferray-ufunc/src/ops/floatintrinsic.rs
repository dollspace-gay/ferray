// ferray-ufunc: Floating-point intrinsic functions
//
// isnan, isinf, isfinite, isneginf, isposinf, nan_to_num, nextafter,
// spacing, ldexp, frexp, signbit, copysign, float_power, fmax, fmin,
// maximum, minimum, clip

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::helpers::{binary_float_op, binary_mixed_op, unary_float_op, unary_map_op};

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// Elementwise test for NaN.
pub fn isnan<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_nan())
}

/// Elementwise test for infinity (positive or negative).
pub fn isinf<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_infinite())
}

/// Elementwise test for finiteness.
pub fn isfinite<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_finite())
}

/// Elementwise test for negative infinity.
pub fn isneginf<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_infinite() && x < <T as Element>::zero())
}

/// Elementwise test for positive infinity.
pub fn isposinf<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_infinite() && x > <T as Element>::zero())
}

/// Elementwise sign bit test.
pub fn signbit<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_map_op(input, |x| x.is_sign_negative())
}

// ---------------------------------------------------------------------------
// Replacement
// ---------------------------------------------------------------------------

/// Replace NaN with zero, and infinity with large finite numbers.
///
/// `nan` replaces NaN (default 0), `posinf` replaces +inf, `neginf` replaces -inf.
pub fn nan_to_num<T, D>(
    input: &Array<T, D>,
    nan: Option<T>,
    posinf: Option<T>,
    neginf: Option<T>,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let nan_val = nan.unwrap_or_else(|| <T as Element>::zero());
    let posinf_val = posinf.unwrap_or_else(T::max_value);
    let neginf_val = neginf.unwrap_or_else(T::min_value);
    unary_float_op(input, |x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() {
            if x > <T as Element>::zero() {
                posinf_val
            } else {
                neginf_val
            }
        } else {
            x
        }
    })
}

/// Clip (limit) values to [a_min, a_max] for float types.
pub fn clip<T, D>(input: &Array<T, D>, a_min: T, a_max: T) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if a_min > a_max {
        return Err(FerrayError::invalid_value("clip: a_min must be <= a_max"));
    }
    unary_float_op(input, |x| {
        if x < a_min {
            a_min
        } else if x > a_max {
            a_max
        } else {
            x
        }
    })
}

/// Clip (limit) values to [a_min, a_max] for any ordered type, including integers.
///
/// This is a more general version of [`clip`] that works on integer arrays
/// (i8, i16, i32, i64, u8, u16, u32, u64) as well as floats.
///
/// Equivalent to `numpy.clip`.
pub fn clip_ord<T, D>(input: &Array<T, D>, a_min: T, a_max: T) -> FerrayResult<Array<T, D>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a_min > a_max {
        return Err(FerrayError::invalid_value("clip_ord: a_min must be <= a_max"));
    }
    let data: Vec<T> = input
        .iter()
        .map(|&x| {
            if x < a_min {
                a_min
            } else if x > a_max {
                a_max
            } else {
                x
            }
        })
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

// ---------------------------------------------------------------------------
// Float manipulation
// ---------------------------------------------------------------------------

/// IEEE 754 nextafter for f64: return the next representable value after `a` towards `b`.
#[inline]
fn nextafter_f64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        // Return smallest subnormal with the sign of (b - a)
        return if b > 0.0 {
            f64::from_bits(1)
        } else {
            f64::from_bits(1u64 | (1u64 << 63))
        };
    }
    let bits = a.to_bits();
    let new_bits = if (a < b) == (a > 0.0) {
        // Moving away from zero: increment magnitude
        bits + 1
    } else {
        // Moving toward zero: decrement magnitude
        bits - 1
    };
    f64::from_bits(new_bits)
}

/// IEEE 754 nextafter for f32: return the next representable value after `a` towards `b`.
#[inline]
fn nextafter_f32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        return f32::NAN;
    }
    if a == b {
        return b;
    }
    if a == 0.0 {
        return if b > 0.0 {
            f32::from_bits(1)
        } else {
            f32::from_bits(1u32 | (1u32 << 31))
        };
    }
    let bits = a.to_bits();
    let new_bits = if (a < b) == (a > 0.0) {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(new_bits)
}

/// Return the next floating-point value after x1 towards x2, using IEEE 754 bit manipulation.
pub fn nextafter<T, D>(x1: &Array<T, D>, x2: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T is f64, verified by TypeId check above.
        let a_f64 = unsafe { &*(x1 as *const Array<T, D> as *const Array<f64, D>) };
        let b_f64 = unsafe { &*(x2 as *const Array<T, D> as *const Array<f64, D>) };
        let result = binary_float_op(a_f64, b_f64, nextafter_f64)?;
        Ok(unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(result)) })
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: T is f32, verified by TypeId check above.
        let a_f32 = unsafe { &*(x1 as *const Array<T, D> as *const Array<f32, D>) };
        let b_f32 = unsafe { &*(x2 as *const Array<T, D> as *const Array<f32, D>) };
        let result = binary_float_op(a_f32, b_f32, nextafter_f32)?;
        Ok(unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(result)) })
    } else {
        // Fallback for other float types: use f64 round-trip
        binary_float_op(x1, x2, |a, b| {
            let a64 = a.to_f64().unwrap();
            let b64 = b.to_f64().unwrap();
            T::from(nextafter_f64(a64, b64)).unwrap()
        })
    }
}

/// IEEE 754 spacing (ULP) for f64.
#[inline]
fn spacing_f64(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return f64::from_bits(1); // smallest positive subnormal
    }
    nextafter_f64(ax, f64::INFINITY) - ax
}

/// IEEE 754 spacing (ULP) for f32.
#[inline]
fn spacing_f32(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() {
        return f32::NAN;
    }
    let ax = x.abs();
    if ax == 0.0 {
        return f32::from_bits(1); // smallest positive subnormal
    }
    nextafter_f32(ax, f32::INFINITY) - ax
}

/// Return the spacing of values: the ULP (unit in the last place),
/// computed via IEEE 754 bit manipulation.
pub fn spacing<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        let f64_input = unsafe { &*(input as *const Array<T, D> as *const Array<f64, D>) };
        let result = unary_float_op(f64_input, spacing_f64)?;
        Ok(unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(result)) })
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        let f32_input = unsafe { &*(input as *const Array<T, D> as *const Array<f32, D>) };
        let result = unary_float_op(f32_input, spacing_f32)?;
        Ok(unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(result)) })
    } else {
        // Fallback for other float types: use f64 round-trip
        unary_float_op(input, |x| {
            let x64 = x.to_f64().unwrap();
            T::from(spacing_f64(x64)).unwrap()
        })
    }
}

/// Multiply `x` by `2^n` (ldexp), with NumPy broadcasting.
///
/// `n` is provided as an integer array; broadcast-compatible with `x`.
pub fn ldexp<T, D>(x: &Array<T, D>, n: &Array<i32, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let two = <T as Element>::one() + <T as Element>::one();
    binary_mixed_op(x, n, move |xi, ni| xi * two.powi(ni))
}

/// Decompose f64 into mantissa and exponent via IEEE 754 bit extraction.
///
/// Returns (mantissa, exponent) where mantissa is in [0.5, 1.0) and x = mantissa * 2^exponent.
#[inline]
fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & (1u64 << 63);
    let mut exp = ((bits >> 52) & 0x7FF) as i32;
    if exp == 0 {
        // Subnormal: multiply by 2^64 to normalize, then adjust exponent
        let nx = x * (1u64 << 63) as f64 * 2.0;
        let nbits = nx.to_bits();
        exp = ((nbits >> 52) & 0x7FF) as i32 - 64;
        let frac = f64::from_bits((nbits & !(0x7FFu64 << 52)) | (0x3FEu64 << 52));
        return (if sign != 0 { -frac.abs() } else { frac }, exp - 0x3FE);
    }
    // Set exponent to -1 (biased: 0x3FE) to place mantissa in [0.5, 1.0)
    let frac = f64::from_bits((bits & !(0x7FFu64 << 52)) | (0x3FEu64 << 52));
    // exponent = (biased_exp - 1023) + 1 = biased_exp - 0x3FE
    (frac, exp - 0x3FE)
}

/// Decompose f32 into mantissa and exponent via IEEE 754 bit extraction.
///
/// Returns (mantissa, exponent) where mantissa is in [0.5, 1.0) and x = mantissa * 2^exponent.
#[inline]
fn frexp_f32(x: f32) -> (f32, i32) {
    if x == 0.0 || x.is_nan() || x.is_infinite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let sign = bits & (1u32 << 31);
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    if exp == 0 {
        // Subnormal: multiply by 2^32 to normalize, then adjust exponent
        let nx = x * (1u32 << 31) as f32 * 2.0;
        let nbits = nx.to_bits();
        exp = ((nbits >> 23) & 0xFF) as i32 - 32;
        let frac = f32::from_bits((nbits & !(0xFFu32 << 23)) | (0x7Eu32 << 23));
        return (if sign != 0 { -frac.abs() } else { frac }, exp - 0x7E);
    }
    let frac = f32::from_bits((bits & !(0xFFu32 << 23)) | (0x7Eu32 << 23));
    (frac, exp - 0x7E)
}

/// Decompose into mantissa and exponent: x = m * 2^e.
///
/// Returns (mantissa_array, exponent_array) where mantissa is in [0.5, 1.0).
/// This matches C's `frexp`: for x=4.0, returns (0.5, 3) since 0.5 * 2^3 = 4.
///
/// Uses IEEE 754 bit extraction for f64 and f32 (O(1) per element), with
/// f64 round-trip fallback for other float types.
pub fn frexp<T, D>(input: &Array<T, D>) -> FerrayResult<(Array<T, D>, Array<i32, D>)>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T is f64, verified by TypeId check above.
        let f64_input = unsafe { &*(input as *const Array<T, D> as *const Array<f64, D>) };
        let mut mantissas = Vec::with_capacity(f64_input.size());
        let mut exponents = Vec::with_capacity(f64_input.size());
        for &x in f64_input.iter() {
            let (m, e) = frexp_f64(x);
            mantissas.push(m);
            exponents.push(e);
        }
        let m_arr: Array<f64, D> = Array::from_vec(f64_input.dim().clone(), mantissas)?;
        let e_arr = Array::from_vec(f64_input.dim().clone(), exponents)?;
        let m_result = unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(m_arr)) };
        Ok((m_result, e_arr))
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: T is f32, verified by TypeId check above.
        let f32_input = unsafe { &*(input as *const Array<T, D> as *const Array<f32, D>) };
        let mut mantissas = Vec::with_capacity(f32_input.size());
        let mut exponents = Vec::with_capacity(f32_input.size());
        for &x in f32_input.iter() {
            let (m, e) = frexp_f32(x);
            mantissas.push(m);
            exponents.push(e);
        }
        let m_arr: Array<f32, D> = Array::from_vec(f32_input.dim().clone(), mantissas)?;
        let e_arr = Array::from_vec(f32_input.dim().clone(), exponents)?;
        let m_result = unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(m_arr)) };
        Ok((m_result, e_arr))
    } else {
        // Fallback for other float types: use f64 round-trip
        let mut mantissas = Vec::with_capacity(input.size());
        let mut exponents = Vec::with_capacity(input.size());
        for &x in input.iter() {
            let x64 = x.to_f64().unwrap();
            let (m64, e) = frexp_f64(x64);
            mantissas.push(T::from(m64).unwrap());
            exponents.push(e);
        }
        Ok((
            Array::from_vec(input.dim().clone(), mantissas)?,
            Array::from_vec(input.dim().clone(), exponents)?,
        ))
    }
}

/// Elementwise copysign: magnitude of x1, sign of x2.
pub fn copysign<T, D>(x1: &Array<T, D>, x2: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(x1, x2, |a, b| {
        let mag = a.abs();
        if b.is_sign_negative() { -mag } else { mag }
    })
}

/// Float power: x1^x2, always returning float.
pub fn float_power<T, D>(x1: &Array<T, D>, x2: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(x1, x2, T::powf)
}

/// Elementwise maximum, propagating NaN.
pub fn maximum<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            <T as Float>::nan()
        } else if x >= y {
            x
        } else {
            y
        }
    })
}

/// Elementwise minimum, propagating NaN.
pub fn minimum<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            <T as Float>::nan()
        } else if x <= y {
            x
        } else {
            y
        }
    })
}

/// Elementwise maximum, ignoring NaN.
pub fn fmax<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| {
        if x.is_nan() {
            y
        } else if y.is_nan() || x >= y {
            x
        } else {
            y
        }
    })
}

/// Elementwise minimum, ignoring NaN.
pub fn fmin<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| {
        if x.is_nan() {
            y
        } else if y.is_nan() || x <= y {
            x
        } else {
            y
        }
    })
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted)
// ---------------------------------------------------------------------------

/// Elementwise test for NaN for f16 arrays.
#[cfg(feature = "f16")]
pub fn isnan_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<bool, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_to_bool_op(input, f32::is_nan)
}

/// Elementwise test for infinity for f16 arrays.
#[cfg(feature = "f16")]
pub fn isinf_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<bool, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_to_bool_op(input, f32::is_infinite)
}

/// Elementwise test for finiteness for f16 arrays.
#[cfg(feature = "f16")]
pub fn isfinite_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<bool, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_to_bool_op(input, f32::is_finite)
}

/// Clip (limit) values to [a_min, a_max] for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn clip_f16<D>(
    input: &Array<half::f16, D>,
    a_min: half::f16,
    a_max: half::f16,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    let min_f32 = a_min.to_f32();
    let max_f32 = a_max.to_f32();
    if min_f32 > max_f32 {
        return Err(FerrayError::invalid_value("clip: a_min must be <= a_max"));
    }
    crate::helpers::unary_f16_op(input, |x| {
        if x < min_f32 {
            min_f32
        } else if x > max_f32 {
            max_f32
        } else {
            x
        }
    })
}

/// Replace NaN with zero, and infinity with large finite numbers, for f16 arrays.
#[cfg(feature = "f16")]
pub fn nan_to_num_f16<D>(
    input: &Array<half::f16, D>,
    nan: Option<half::f16>,
    posinf: Option<half::f16>,
    neginf: Option<half::f16>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    let nan_val = nan.unwrap_or(half::f16::ZERO).to_f32();
    let posinf_val = posinf.unwrap_or(half::f16::MAX).to_f32();
    let neginf_val = neginf.unwrap_or(half::f16::MIN).to_f32();
    crate::helpers::unary_f16_op(input, |x| {
        if x.is_nan() {
            nan_val
        } else if x.is_infinite() {
            if x > 0.0 { posinf_val } else { neginf_val }
        } else {
            x
        }
    })
}

/// Elementwise maximum for f16 arrays via f32 promotion, propagating NaN.
#[cfg(feature = "f16")]
pub fn maximum_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            f32::NAN
        } else if x >= y {
            x
        } else {
            y
        }
    })
}

/// Elementwise minimum for f16 arrays via f32 promotion, propagating NaN.
#[cfg(feature = "f16")]
pub fn minimum_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            f32::NAN
        } else if x <= y {
            x
        } else {
            y
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_isnan() {
        let a = arr1(vec![1.0, f64::NAN, 3.0]);
        let r = isnan(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_isinf() {
        let a = arr1(vec![1.0, f64::INFINITY, f64::NEG_INFINITY]);
        let r = isinf(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, true]);
    }

    #[test]
    fn test_isfinite() {
        let a = arr1(vec![1.0, f64::INFINITY, f64::NAN]);
        let r = isfinite(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false]);
    }

    #[test]
    fn test_isneginf() {
        let a = arr1(vec![f64::NEG_INFINITY, f64::INFINITY, 0.0]);
        let r = isneginf(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false]);
    }

    #[test]
    fn test_isposinf() {
        let a = arr1(vec![f64::NEG_INFINITY, f64::INFINITY, 0.0]);
        let r = isposinf(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_nan_to_num() {
        let a = arr1(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
        let r = nan_to_num(&a, None, None, None).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert!(s[1].is_finite() && s[1] > 0.0);
        assert!(s[2].is_finite() && s[2] < 0.0);
        assert_eq!(s[3], 1.0);
    }

    #[test]
    fn test_clip() {
        let a = arr1(vec![-1.0, 0.5, 1.5, 3.0]);
        let r = clip(&a, 0.0, 2.0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0.0, 0.5, 1.5, 2.0]);
    }

    #[test]
    fn test_clip_ord_integer() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![-10, 0, 5, 100, 255]).unwrap();
        let r = clip_ord(&a, 0, 200).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0, 0, 5, 100, 200]);
    }

    #[test]
    fn test_clip_ord_u8() {
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([4]), vec![0, 50, 200, 255]).unwrap();
        let r = clip_ord(&a, 10, 128).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10, 50, 128, 128]);
    }

    #[test]
    fn test_signbit() {
        let a = arr1(vec![-1.0, 0.0, 1.0]);
        let r = signbit(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false]);
    }

    #[test]
    fn test_copysign() {
        let a = arr1(vec![1.0, -2.0, 3.0]);
        let b = arr1(vec![-1.0, 1.0, -1.0]);
        let r = copysign(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_maximum() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![4.0, 2.0, 3.0]);
        let r = maximum(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4.0, 5.0, 3.0]);
    }

    #[test]
    fn test_minimum() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![4.0, 2.0, 3.0]);
        let r = minimum(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_maximum_nan_propagation() {
        let a = arr1(vec![1.0, f64::NAN]);
        let b = arr1(vec![2.0, 3.0]);
        let r = maximum(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert!(s[1].is_nan());
    }

    #[test]
    fn test_fmax_ignores_nan() {
        let a = arr1(vec![1.0, f64::NAN]);
        let b = arr1(vec![2.0, 3.0]);
        let r = fmax(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], 3.0);
    }

    #[test]
    fn test_fmin_ignores_nan() {
        let a = arr1(vec![1.0, f64::NAN]);
        let b = arr1(vec![2.0, 3.0]);
        let r = fmin(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 3.0);
    }

    #[test]
    fn test_float_power() {
        let a = arr1(vec![2.0, 3.0]);
        let b = arr1(vec![3.0, 2.0]);
        let r = float_power(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[8.0, 9.0]);
    }

    #[test]
    fn test_spacing() {
        let a = arr1(vec![1.0]);
        let r = spacing(&a).unwrap();
        let s = r.as_slice().unwrap();
        // spacing(1.0) == f64::EPSILON (2^-52)
        assert_eq!(s[0], f64::EPSILON);
    }

    #[test]
    fn test_spacing_zero() {
        let a = arr1(vec![0.0]);
        let r = spacing(&a).unwrap();
        let s = r.as_slice().unwrap();
        // spacing(0.0) == smallest positive subnormal = 5e-324
        assert_eq!(s[0], f64::from_bits(1));
    }

    #[test]
    fn test_spacing_nan_inf() {
        let a = arr1(vec![f64::NAN, f64::INFINITY, f64::NEG_INFINITY]);
        let r = spacing(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!(s[0].is_nan());
        assert!(s[1].is_nan());
        assert!(s[2].is_nan());
    }

    #[test]
    fn test_ldexp() {
        let x = arr1(vec![1.0, 2.0]);
        let n = arr1_i32(vec![2, 3]);
        let r = ldexp(&x, &n).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 4.0).abs() < 1e-12);
        assert!((s[1] - 16.0).abs() < 1e-12);
    }

    #[test]
    fn test_ldexp_broadcasts() {
        use ferray_core::dimension::Ix2;
        let x = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 3.0]).unwrap();
        let n = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![1, 2, 3]).unwrap();
        let r = ldexp(&x, &n).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        // 1*2^{1,2,3} = {2,4,8}, 3*2^{1,2,3} = {6,12,24}
        let v: Vec<f64> = r.iter().copied().collect();
        assert_eq!(v, vec![2.0, 4.0, 8.0, 6.0, 12.0, 24.0]);
    }

    #[test]
    fn test_frexp() {
        let a = arr1(vec![4.0]);
        let (m, e) = frexp(&a).unwrap();
        let ms = m.as_slice().unwrap();
        let es = e.as_slice().unwrap();
        // 4 = 0.5 * 2^3
        assert!((ms[0] - 0.5).abs() < 1e-12);
        assert_eq!(es[0], 3);
    }

    #[test]
    fn test_nextafter_zero_toward_positive() {
        let a = arr1(vec![0.0]);
        let b = arr1(vec![1.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // nextafter(0, 1) == smallest positive subnormal = 5e-324
        assert_eq!(s[0], f64::from_bits(1));
    }

    #[test]
    fn test_nextafter_zero_toward_negative() {
        let a = arr1(vec![0.0]);
        let b = arr1(vec![-1.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // nextafter(0, -1) == smallest negative subnormal = -5e-324
        assert_eq!(s[0], f64::from_bits(1u64 | (1u64 << 63)));
    }

    #[test]
    fn test_nextafter_equal() {
        let a = arr1(vec![1.0]);
        let b = arr1(vec![1.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
    }

    #[test]
    fn test_nextafter_nan() {
        let a = arr1(vec![f64::NAN, 1.0]);
        let b = arr1(vec![1.0, f64::NAN]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!(s[0].is_nan());
        assert!(s[1].is_nan());
    }

    #[test]
    fn test_nextafter_one_toward_two() {
        let a = arr1(vec![1.0]);
        let b = arr1(vec![2.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // nextafter(1.0, 2.0) == 1.0 + epsilon
        assert_eq!(s[0], 1.0 + f64::EPSILON);
    }

    #[test]
    fn test_nextafter_negative() {
        let a = arr1(vec![-1.0]);
        let b = arr1(vec![-2.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // nextafter(-1.0, -2.0) == -1.0 - epsilon
        assert_eq!(s[0], -1.0 - f64::EPSILON);
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
        fn test_isnan_f16() {
            let a = arr1_f16(&[1.0, f32::NAN, 3.0]);
            let r = isnan_f16(&a).unwrap();
            assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
        }

        #[test]
        fn test_isinf_f16() {
            let a = arr1_f16(&[1.0, f32::INFINITY, f32::NEG_INFINITY]);
            let r = isinf_f16(&a).unwrap();
            assert_eq!(r.as_slice().unwrap(), &[false, true, true]);
        }

        #[test]
        fn test_isfinite_f16() {
            let a = arr1_f16(&[1.0, f32::INFINITY, f32::NAN]);
            let r = isfinite_f16(&a).unwrap();
            assert_eq!(r.as_slice().unwrap(), &[true, false, false]);
        }

        #[test]
        fn test_clip_f16() {
            let a = arr1_f16(&[-1.0, 0.5, 1.5, 3.0]);
            let r = clip_f16(&a, half::f16::from_f32(0.0), half::f16::from_f32(2.0)).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 0.0).abs() < 0.01);
            assert!((s[1].to_f32() - 0.5).abs() < 0.01);
            assert!((s[2].to_f32() - 1.5).abs() < 0.01);
            assert!((s[3].to_f32() - 2.0).abs() < 0.01);
        }

        #[test]
        fn test_nan_to_num_f16() {
            let a = arr1_f16(&[f32::NAN, 1.0]);
            let r = nan_to_num_f16(&a, None, None, None).unwrap();
            let s = r.as_slice().unwrap();
            assert_eq!(s[0].to_f32(), 0.0);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_maximum_f16() {
            let a = arr1_f16(&[1.0, 5.0, 3.0]);
            let b = arr1_f16(&[4.0, 2.0, 3.0]);
            let r = maximum_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 4.0).abs() < 0.01);
            assert!((s[1].to_f32() - 5.0).abs() < 0.01);
            assert!((s[2].to_f32() - 3.0).abs() < 0.01);
        }

        #[test]
        fn test_minimum_f16() {
            let a = arr1_f16(&[1.0, 5.0, 3.0]);
            let b = arr1_f16(&[4.0, 2.0, 3.0]);
            let r = minimum_f16(&a, &b).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() - 2.0).abs() < 0.01);
            assert!((s[2].to_f32() - 3.0).abs() < 0.01);
        }
    }
}
