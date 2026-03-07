// ferray-ufunc: Complex number functions
//
// real, imag, conj, conjugate, angle, abs (returns real magnitude)

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_complex::Complex;
use num_traits::Float;

/// Extract the real part of a complex array.
///
/// Works with `Complex<f32>` and `Complex<f64>` arrays.
pub fn real<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<T> = input.iter().map(|c| c.re).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Extract the imaginary part of a complex array.
pub fn imag<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<T> = input.iter().map(|c| c.im).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Compute the complex conjugate.
pub fn conj<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<Complex<T>> = input.iter().map(|c| c.conj()).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Alias for [`conj`].
pub fn conjugate<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    conj(input)
}

/// Compute the angle (argument/phase) of complex numbers.
///
/// Returns values in radians, in the range [-pi, pi].
pub fn angle<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<T> = input.iter().map(|c| c.im.atan2(c.re)).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Compute the absolute value (magnitude) of complex numbers.
///
/// Returns a real array.
pub fn abs<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<T> = input
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im).sqrt())
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;
    use num_complex::Complex64;

    fn arr1_c64(data: Vec<Complex64>) -> Array<Complex64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_real() {
        let a = arr1_c64(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        let r = real(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0]);
    }

    #[test]
    fn test_imag() {
        let a = arr1_c64(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)]);
        let r = imag(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2.0, 4.0]);
    }

    #[test]
    fn test_conj() {
        let a = arr1_c64(vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)]);
        let r = conj(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], Complex64::new(1.0, -2.0));
        assert_eq!(s[1], Complex64::new(3.0, 4.0));
    }

    #[test]
    fn test_conjugate_alias() {
        let a = arr1_c64(vec![Complex64::new(1.0, 2.0)]);
        let r = conjugate(&a).unwrap();
        assert_eq!(r.as_slice().unwrap()[0], Complex64::new(1.0, -2.0));
    }

    #[test]
    fn test_angle() {
        let a = arr1_c64(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(-1.0, 0.0),
        ]);
        let r = angle(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 0.0).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((s[2] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_abs() {
        let a = arr1_c64(vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 1.0)]);
        let r = abs(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 5.0).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
    }
}
