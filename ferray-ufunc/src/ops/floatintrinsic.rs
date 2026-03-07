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

use crate::helpers::{binary_float_op, unary_float_op, unary_map_op};

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

/// Clip (limit) values to [a_min, a_max].
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

// ---------------------------------------------------------------------------
// Float manipulation
// ---------------------------------------------------------------------------

/// Return the next floating-point value after x1 towards x2.
pub fn nextafter<T, D>(x1: &Array<T, D>, x2: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(x1, x2, |a, b| {
        if a == b {
            b
        } else if a < b {
            // Move toward positive infinity
            let tiny = T::min_positive_value();
            let eps = a.abs() * T::epsilon();
            a + if eps > tiny { eps } else { tiny }
        } else {
            let tiny = T::min_positive_value();
            let eps = a.abs() * T::epsilon();
            a - if eps > tiny { eps } else { tiny }
        }
    })
}

/// Return the spacing of values: the ULP (unit in the last place).
pub fn spacing<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, |x| {
        if x.is_nan() || x.is_infinite() {
            <T as Float>::nan()
        } else {
            let tiny = T::min_positive_value();
            let eps = x.abs() * T::epsilon();
            if eps > tiny { eps } else { tiny }
        }
    })
}

/// Multiply x by 2^n (ldexp).
///
/// `n` is provided as an integer array of same shape.
pub fn ldexp<T, D>(x: &Array<T, D>, n: &Array<i32, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    if x.shape() != n.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "ldexp: shapes {:?} and {:?} do not match",
            x.shape(),
            n.shape()
        )));
    }
    let two = <T as Element>::one() + <T as Element>::one();
    let data: Vec<T> = x
        .iter()
        .zip(n.iter())
        .map(|(&xi, &ni)| xi * two.powi(ni))
        .collect();
    Array::from_vec(x.dim().clone(), data)
}

/// Decompose into mantissa and exponent: x = m * 2^e.
///
/// Returns (mantissa_array, exponent_array) where mantissa is in [0.5, 1.0).
/// This matches C's `frexp`: for x=4.0, returns (0.5, 3) since 0.5 * 2^3 = 4.
pub fn frexp<T, D>(input: &Array<T, D>) -> FerrayResult<(Array<T, D>, Array<i32, D>)>
where
    T: Element + Float,
    D: Dimension,
{
    let mut mantissas = Vec::with_capacity(input.size());
    let mut exponents = Vec::with_capacity(input.size());

    let zero = <T as Element>::zero();
    let half = T::from(0.5).unwrap();
    let two = T::from(2.0).unwrap();

    for &x in input.iter() {
        if x == zero || x.is_nan() || x.is_infinite() {
            mantissas.push(x);
            exponents.push(0);
            continue;
        }
        let mut abs_x = x.abs();
        let mut exp: i32 = 0;

        // Normalize to [0.5, 1.0)
        while abs_x >= T::from(1.0).unwrap() {
            abs_x = abs_x / two;
            exp += 1;
        }
        while abs_x < half {
            abs_x = abs_x * two;
            exp -= 1;
        }

        if x.is_sign_negative() {
            mantissas.push(-abs_x);
        } else {
            mantissas.push(abs_x);
        }
        exponents.push(exp);
    }

    Ok((
        Array::from_vec(input.dim().clone(), mantissas)?,
        Array::from_vec(input.dim().clone(), exponents)?,
    ))
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
        assert!(s[0] > 0.0 && s[0] < 1e-10);
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
    fn test_nextafter() {
        let a = arr1(vec![0.0]);
        let b = arr1(vec![1.0]);
        let r = nextafter(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!(s[0] > 0.0);
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
