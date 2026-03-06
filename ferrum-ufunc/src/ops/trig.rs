// ferrum-ufunc: Trigonometric functions
//
// sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot,
// sinh, cosh, tanh, arcsinh, arccosh, arctanh,
// degrees, radians, deg2rad, rad2deg, unwrap

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::helpers::{unary_float_op, binary_float_op};

// ---------------------------------------------------------------------------
// Unary trig
// ---------------------------------------------------------------------------

/// Elementwise sine.
pub fn sin<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::sin)
}

/// Elementwise cosine.
pub fn cos<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::cos)
}

/// Elementwise tangent.
pub fn tan<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::tan)
}

/// Elementwise arc sine.
pub fn arcsin<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::asin)
}

/// Elementwise arc cosine.
pub fn arccos<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::acos)
}

/// Elementwise arc tangent.
pub fn arctan<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::atan)
}

/// Elementwise two-argument arc tangent (atan2).
pub fn arctan2<T, D>(y: &Array<T, D>, x: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(y, x, T::atan2)
}

/// Elementwise hypotenuse: sqrt(a^2 + b^2).
pub fn hypot<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, T::hypot)
}

// ---------------------------------------------------------------------------
// Hyperbolic
// ---------------------------------------------------------------------------

/// Elementwise hyperbolic sine.
pub fn sinh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::sinh)
}

/// Elementwise hyperbolic cosine.
pub fn cosh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::cosh)
}

/// Elementwise hyperbolic tangent.
pub fn tanh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::tanh)
}

/// Elementwise inverse hyperbolic sine.
pub fn arcsinh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::asinh)
}

/// Elementwise inverse hyperbolic cosine.
pub fn arccosh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::acosh)
}

/// Elementwise inverse hyperbolic tangent.
pub fn arctanh<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::atanh)
}

// ---------------------------------------------------------------------------
// Degree/radian conversion
// ---------------------------------------------------------------------------

/// Convert radians to degrees.
pub fn degrees<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::to_degrees)
}

/// Convert degrees to radians.
pub fn radians<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::to_radians)
}

/// Alias for [`radians`].
pub fn deg2rad<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    radians(input)
}

/// Alias for [`degrees`].
pub fn rad2deg<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    degrees(input)
}

/// Unwrap by changing deltas between values to their 2*pi complement.
///
/// Works on 1-D arrays. `discont` defaults to pi if `None`.
pub fn unwrap<T, D>(input: &Array<T, D>, discont: Option<T>) -> FerrumResult<Array<T, D>>
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
        if diff > discont {
            while diff > pi {
                diff = diff - two_pi;
            }
        } else if diff < -discont {
            while diff < -pi {
                diff = diff + two_pi;
            }
        }
        cumulative = cumulative + diff - (data[i] - data[i - 1]);
        result.push(data[i] + cumulative);
    }

    Array::from_vec(input.dim().clone(), result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::Ix1;

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
}
