// ferray-ufunc: Trigonometric functions
//
// sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot,
// sinh, cosh, tanh, arcsinh, arccosh, arctanh,
// degrees, radians, deg2rad, rad2deg, unwrap

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::cr_math::CrMath;
use crate::helpers::{binary_float_op, unary_float_op};

// ---------------------------------------------------------------------------
// Unary trig
// ---------------------------------------------------------------------------

/// Elementwise sine.
pub fn sin<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_sin)
}

/// Elementwise cosine.
pub fn cos<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_cos)
}

/// Elementwise tangent.
pub fn tan<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_tan)
}

/// Elementwise arc sine.
pub fn arcsin<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_asin)
}

/// Elementwise arc cosine.
pub fn arccos<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_acos)
}

/// Elementwise arc tangent.
pub fn arctan<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_atan)
}

/// Elementwise two-argument arc tangent (atan2).
pub fn arctan2<T, D>(y: &Array<T, D>, x: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    binary_float_op(y, x, T::cr_atan2)
}

/// Elementwise hypotenuse: sqrt(a^2 + b^2).
pub fn hypot<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    binary_float_op(a, b, T::cr_hypot)
}

// ---------------------------------------------------------------------------
// Hyperbolic
// ---------------------------------------------------------------------------

/// Elementwise hyperbolic sine.
pub fn sinh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_sinh)
}

/// Elementwise hyperbolic cosine.
pub fn cosh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_cosh)
}

/// Elementwise hyperbolic tangent.
pub fn tanh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_tanh)
}

/// Elementwise inverse hyperbolic sine.
pub fn arcsinh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_asinh)
}

/// Elementwise inverse hyperbolic cosine.
pub fn arccosh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_acosh)
}

/// Elementwise inverse hyperbolic tangent.
pub fn arctanh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, T::cr_atanh)
}

// ---------------------------------------------------------------------------
// Degree/radian conversion
// ---------------------------------------------------------------------------

/// Convert radians to degrees.
pub fn degrees<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::to_degrees)
}

/// Convert degrees to radians.
pub fn radians<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::to_radians)
}

/// Alias for [`radians`].
pub fn deg2rad<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    radians(input)
}

/// Alias for [`degrees`].
pub fn rad2deg<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    degrees(input)
}

/// Unwrap by changing deltas between values to their 2*pi complement.
///
/// Works on 1-D arrays. `discont` defaults to pi if `None`.
pub fn unwrap<T, D>(input: &Array<T, D>, discont: Option<T>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(<T as Element>::zero);
    let two_pi = pi + pi;
    let discont = discont.unwrap_or(pi);

    let data: Vec<T> = input.iter().copied().collect();
    if data.is_empty() {
        return Array::from_vec(input.dim().clone(), data);
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    let mut cumulative = <T as Element>::zero();

    for i in 1..data.len() {
        let mut diff = data[i] - data[i - 1];
        if diff > discont || diff < -discont {
            diff = diff - two_pi * ((diff + pi) / two_pi).floor();
        }
        cumulative = cumulative + diff - (data[i] - data[i - 1]);
        result.push(data[i] + cumulative);
    }

    Array::from_vec(input.dim().clone(), result)
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted)
// ---------------------------------------------------------------------------

/// Elementwise sine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn sin_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::sin)
}

/// Elementwise cosine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn cos_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::cos)
}

/// Elementwise tangent for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn tan_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::tan)
}

/// Elementwise arc sine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arcsin_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::asin)
}

/// Elementwise arc cosine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arccos_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::acos)
}

/// Elementwise arc tangent for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arctan_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::atan)
}

/// Elementwise two-argument arc tangent for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arctan2_f16<D>(
    y: &Array<half::f16, D>,
    x: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(y, x, f32::atan2)
}

/// Elementwise hypotenuse for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn hypot_f16<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::binary_f16_op(a, b, f32::hypot)
}

/// Elementwise hyperbolic sine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn sinh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::sinh)
}

/// Elementwise hyperbolic cosine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn cosh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::cosh)
}

/// Elementwise hyperbolic tangent for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn tanh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::tanh)
}

/// Elementwise inverse hyperbolic sine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arcsinh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::asinh)
}

/// Elementwise inverse hyperbolic cosine for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arccosh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::acosh)
}

/// Elementwise inverse hyperbolic tangent for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn arctanh_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::atanh)
}

/// Convert radians to degrees for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn degrees_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::to_degrees)
}

/// Convert degrees to radians for f16 arrays via f32 promotion.
#[cfg(feature = "f16")]
pub fn radians_f16<D>(input: &Array<half::f16, D>) -> FerrayResult<Array<half::f16, D>>
where
    D: Dimension,
{
    crate::helpers::unary_f16_op(input, f32::to_radians)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_sin() {
        let a = arr1(vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
        let r = sin(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
        assert!((s[2]).abs() < 1e-12);
    }

    #[test]
    fn test_cos() {
        let a = arr1(vec![0.0, std::f64::consts::PI]);
        let r = cos(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tan() {
        let a = arr1(vec![0.0, std::f64::consts::FRAC_PI_4]);
        let r = tan(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_arcsin() {
        let a = arr1(vec![0.0, 1.0]);
        let r = arcsin(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_arctan2() {
        let y = arr1(vec![0.0, 1.0]);
        let x = arr1(vec![1.0, 0.0]);
        let r = arctan2(&y, &x).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_hypot() {
        let a = arr1(vec![3.0, 5.0]);
        let b = arr1(vec![4.0, 12.0]);
        let r = hypot(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 5.0).abs() < 1e-12);
        assert!((s[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn test_sinh_cosh_tanh() {
        let a = arr1(vec![0.0]);
        assert!((sinh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
        assert!((cosh(&a).unwrap().as_slice().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((tanh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
    }

    #[test]
    fn test_degrees_radians() {
        let a = arr1(vec![std::f64::consts::PI]);
        let deg = degrees(&a).unwrap();
        assert!((deg.as_slice().unwrap()[0] - 180.0).abs() < 1e-10);

        let back = radians(&deg).unwrap();
        assert!((back.as_slice().unwrap()[0] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_deg2rad_rad2deg() {
        let a = arr1(vec![180.0]);
        let r = deg2rad(&a).unwrap();
        assert!((r.as_slice().unwrap()[0] - std::f64::consts::PI).abs() < 1e-12);

        let d = rad2deg(&arr1(vec![std::f64::consts::PI])).unwrap();
        assert!((d.as_slice().unwrap()[0] - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_unwrap_basic() {
        let a = arr1(vec![0.0, 0.5, 1.0, -0.5, -1.0]);
        let r = unwrap(&a, None).unwrap();
        // No discontinuity larger than pi, so should be unchanged
        let s = r.as_slice().unwrap();
        for (i, &v) in s.iter().enumerate() {
            assert!((v - a.as_slice().unwrap()[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_arcsinh_arccosh_arctanh() {
        let a = arr1(vec![0.0]);
        assert!((arcsinh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);

        let b = arr1(vec![1.0]);
        assert!((arccosh(&b).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
        assert!((arctanh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
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
        fn test_sin_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]);
            let r = sin_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
            assert!(s[2].to_f32().abs() < 0.01);
        }

        #[test]
        fn test_cos_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::PI]);
            let r = cos_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() + 1.0).abs() < 0.01);
        }

        #[test]
        fn test_tan_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::FRAC_PI_4]);
            let r = tan_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_arctan2_f16() {
            let y = arr1_f16(&[0.0, 1.0]);
            let x = arr1_f16(&[1.0, 0.0]);
            let r = arctan2_f16(&y, &x).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - std::f32::consts::FRAC_PI_2).abs() < 0.01);
        }

        #[test]
        fn test_degrees_radians_f16() {
            let a = arr1_f16(&[std::f32::consts::PI]);
            let deg = degrees_f16(&a).unwrap();
            assert!((deg.as_slice().unwrap()[0].to_f32() - 180.0).abs() < 0.5);

            let back = radians_f16(&deg).unwrap();
            assert!((back.as_slice().unwrap()[0].to_f32() - std::f32::consts::PI).abs() < 0.01);
        }
    }
}
