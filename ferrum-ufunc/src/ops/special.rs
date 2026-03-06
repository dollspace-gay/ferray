// ferrum-ufunc: Special functions
//
// sinc, i0 (modified Bessel function of the first kind, order 0)

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::helpers::unary_float_op;

/// Normalized sinc function: sin(pi*x) / (pi*x).
///
/// AC-13: `sinc(0.0) == 1.0`.
pub fn sinc<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| <T as Element>::one());
    unary_float_op(input, |x| {
        if x == <T as Element>::zero() {
            <T as Element>::one()
        } else {
            let px = pi * x;
            px.sin() / px
        }
    })
}

/// Modified Bessel function of the first kind, order 0.
///
/// Uses a polynomial approximation (Abramowitz and Stegun).
/// AC-13: `i0(0.0) == 1.0`.
pub fn i0<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, bessel_i0)
}

/// Scalar modified Bessel function I_0(x) using polynomial approximation.
///
/// Uses the Abramowitz and Stegun approximation for |x| <= 3.75 and
/// an asymptotic expansion for |x| > 3.75.
fn bessel_i0<T: Float>(x: T) -> T {
    let ax = x.abs();
    let three_point_seven_five = T::from(3.75).unwrap();

    if ax <= three_point_seven_five {
        let t = (ax / three_point_seven_five).powi(2);
        // Coefficients from A&S 9.8.1
        let c0 = T::from(1.0).unwrap();
        let c1 = T::from(3.5156229).unwrap();
        let c2 = T::from(3.0899424).unwrap();
        let c3 = T::from(1.2067492).unwrap();
        let c4 = T::from(0.2659732).unwrap();
        let c5 = T::from(0.0360768).unwrap();
        let c6 = T::from(0.0045813).unwrap();
        c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6)))))
    } else {
        // Asymptotic expansion from A&S 9.8.2
        let t = three_point_seven_five / ax;
        let c0 = T::from(0.39894228).unwrap();
        let c1 = T::from(0.01328592).unwrap();
        let c2 = T::from(0.00225319).unwrap();
        let c3 = T::from(-0.00157565).unwrap();
        let c4 = T::from(0.00916281).unwrap();
        let c5 = T::from(-0.02057706).unwrap();
        let c6 = T::from(0.02635537).unwrap();
        let c7 = T::from(-0.01647633).unwrap();
        let c8 = T::from(0.00392377).unwrap();
        let poly = c0
            + t * (c1
                + t * (c2
                    + t * (c3
                        + t * (c4
                            + t * (c5
                                + t * (c6 + t * (c7 + t * c8)))))));
        poly * ax.exp() / ax.sqrt()
    }
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
    fn test_sinc_zero_ac13() {
        // AC-13: sinc(0.0) == 1.0
        let a = arr1(vec![0.0]);
        let r = sinc(&a).unwrap();
        assert_eq!(r.as_slice().unwrap()[0], 1.0);
    }

    #[test]
    fn test_sinc_nonzero() {
        let a = arr1(vec![1.0, -1.0, 0.5]);
        let r = sinc(&a).unwrap();
        let s = r.as_slice().unwrap();
        // sinc(1) = sin(pi) / pi ~ 0
        assert!(s[0].abs() < 1e-12);
        // sinc(-1) = sin(-pi) / (-pi) ~ 0
        assert!(s[1].abs() < 1e-12);
        // sinc(0.5) = sin(pi/2) / (pi/2) = 1 / (pi/2) = 2/pi
        assert!((s[2] - 2.0 / std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_i0_zero_ac13() {
        // AC-13: i0(0.0) == 1.0
        let a = arr1(vec![0.0]);
        let r = i0(&a).unwrap();
        assert!((r.as_slice().unwrap()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_i0_known_values() {
        let a = arr1(vec![1.0, 2.0]);
        let r = i0(&a).unwrap();
        let s = r.as_slice().unwrap();
        // I0(1) ~ 1.2660658
        assert!((s[0] - 1.2660658).abs() < 1e-4);
        // I0(2) ~ 2.2795853
        assert!((s[1] - 2.2795853).abs() < 1e-4);
    }

    #[test]
    fn test_sinc_f32() {
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([2]), vec![0.0f32, 1.0]).unwrap();
        let r = sinc(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-6);
        assert!(s[1].abs() < 1e-5);
    }
}
