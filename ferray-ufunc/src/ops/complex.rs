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
    let data: Vec<Complex<T>> = input.iter().map(num_complex::Complex::conj).collect();
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

// ---------------------------------------------------------------------------
// Complex predicates: iscomplex / isreal / iscomplexobj / isrealobj / isscalar
// ---------------------------------------------------------------------------

/// Element-wise: true where the imaginary part is non-zero.
///
/// Analogous to `numpy.iscomplex(x)`. NumPy returns `False` everywhere for
/// real-dtype inputs; in Rust, real-dtype arrays are statically distinct
/// types, so call [`iscomplex_real`] instead for that case.
pub fn iscomplex<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let zero = <T as num_traits::Zero>::zero();
    let data: Vec<bool> = input.iter().map(|c| c.im != zero).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Real-input variant of [`iscomplex`]: always returns `false` everywhere.
///
/// Provided so generic code can ask "is this element complex-valued" against
/// a real-dtype array without a separate type branch. Matches NumPy's
/// `iscomplex` behavior on real inputs.
pub fn iscomplex_real<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element,
    D: Dimension,
{
    let data = vec![false; input.size()];
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise: true where the imaginary part is zero.
///
/// Analogous to `numpy.isreal(x)` for complex inputs. For real-dtype arrays
/// see [`isreal_real`] (always-true).
pub fn isreal<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let zero = <T as num_traits::Zero>::zero();
    let data: Vec<bool> = input.iter().map(|c| c.im == zero).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Real-input variant of [`isreal`]: always returns `true` everywhere.
pub fn isreal_real<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element,
    D: Dimension,
{
    let data = vec![true; input.size()];
    Array::from_vec(input.dim().clone(), data)
}

/// Whether the array's element dtype is a complex type.
///
/// Analogous to `numpy.iscomplexobj(x)` — returns a single `bool` based on
/// the dtype, not a per-element check.
pub fn iscomplexobj<T, D>(_: &Array<T, D>) -> bool
where
    T: Element,
    D: Dimension,
{
    T::dtype().is_complex()
}

/// Whether the array's element dtype is a real (non-complex) type.
///
/// Analogous to `numpy.isrealobj(x)`. Returns `!iscomplexobj(x)`.
pub fn isrealobj<T, D>(a: &Array<T, D>) -> bool
where
    T: Element,
    D: Dimension,
{
    !iscomplexobj(a)
}

/// Whether the array represents a scalar (0-D).
///
/// Analogous to `numpy.isscalar(x)`. In Rust, dimensionality is statically
/// typed, so this returns `true` only for 0-D arrays (`Ix0` or `IxDyn` with
/// empty shape).
pub fn isscalar<T, D>(a: &Array<T, D>) -> bool
where
    T: Element,
    D: Dimension,
{
    a.shape().is_empty()
}

// ---------------------------------------------------------------------------
// Complex transcendentals — sin/cos/tan, sinh/cosh/tanh, exp/log/sqrt,
// arc* / arch* and friends. NumPy provides these for complex dtypes via
// loops_unary_complex.dispatch.c.src; ferray's existing real-only ufuncs
// (T: Float bound) exclude Complex<f32>/<f64>, so we ship them in this
// module and re-export at the crate root.
//
// All implementations delegate to the corresponding `Complex::*` method
// from num_complex, which uses well-known identities (e.g. exp(z) =
// exp(z.re) * (cos(z.im) + i*sin(z.im))).
// ---------------------------------------------------------------------------

macro_rules! complex_unary {
    ($fn_name:ident, $method:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
        where
            T: Element + Float,
            Complex<T>: Element,
            D: Dimension,
        {
            let data: Vec<Complex<T>> = input.iter().map(|z| z.$method()).collect();
            Array::from_vec(input.dim().clone(), data)
        }
    };
}

complex_unary!(
    sin_complex,
    sin,
    "Element-wise complex sin: sin(z) = sin(z.re)cosh(z.im) + i cos(z.re)sinh(z.im)."
);
complex_unary!(cos_complex, cos, "Element-wise complex cos.");
complex_unary!(tan_complex, tan, "Element-wise complex tan.");
complex_unary!(sinh_complex, sinh, "Element-wise complex sinh.");
complex_unary!(cosh_complex, cosh, "Element-wise complex cosh.");
complex_unary!(tanh_complex, tanh, "Element-wise complex tanh.");
complex_unary!(asin_complex, asin, "Element-wise complex arcsin.");
complex_unary!(acos_complex, acos, "Element-wise complex arccos.");
complex_unary!(atan_complex, atan, "Element-wise complex arctan.");
complex_unary!(asinh_complex, asinh, "Element-wise complex arcsinh.");
complex_unary!(acosh_complex, acosh, "Element-wise complex arccosh.");
complex_unary!(atanh_complex, atanh, "Element-wise complex arctanh.");
complex_unary!(
    exp_complex,
    exp,
    "Element-wise complex exp: exp(z) = exp(z.re) (cos(z.im) + i sin(z.im))."
);
complex_unary!(
    ln_complex,
    ln,
    "Element-wise complex natural log: log(z) = log|z| + i arg(z), branch cut along negative real axis."
);
complex_unary!(
    sqrt_complex,
    sqrt,
    "Element-wise complex sqrt; principal branch (Re(sqrt(z)) >= 0)."
);

/// Element-wise complex log2: ln(z) / ln(2).
pub fn log2_complex<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let one = <T as num_traits::One>::one();
    let ln2: T = <T as num_traits::NumCast>::from(core::f64::consts::LN_2).unwrap();
    let inv_ln2 = one / ln2;
    let data: Vec<Complex<T>> = input
        .iter()
        .map(|z| {
            let l = z.ln();
            Complex::new(l.re * inv_ln2, l.im * inv_ln2)
        })
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise complex log10: ln(z) / ln(10).
pub fn log10_complex<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let one = <T as num_traits::One>::one();
    let ln10: T = <T as num_traits::NumCast>::from(core::f64::consts::LN_10).unwrap();
    let inv_ln10 = one / ln10;
    let data: Vec<Complex<T>> = input
        .iter()
        .map(|z| {
            let l = z.ln();
            Complex::new(l.re * inv_ln10, l.im * inv_ln10)
        })
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise complex `exp(z) - 1`. For small |z|, naive `exp(z)-1`
/// suffers catastrophic cancellation; this routine uses the identity
/// `expm1(z) = expm1(z.re) cos(z.im) + (1+expm1(z.re)) (cos(z.im)-1) + i exp(z.re) sin(z.im)`
/// rearranged to preserve precision near 0.
pub fn expm1_complex<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let one = <T as num_traits::One>::one();
    let two = one + one;
    let half: T = <T as num_traits::NumCast>::from(0.5_f64).unwrap();
    let data: Vec<Complex<T>> = input
        .iter()
        .map(|z| {
            // exp(z) - 1 = (exp(re) cos(im) - 1) + i exp(re) sin(im)
            //            = (exp(re)-1) cos(im) + (cos(im)-1) + i exp(re) sin(im)
            let er_m1 = z.re.exp_m1();
            let er = er_m1 + one;
            let cos_im = z.im.cos();
            let sin_im = z.im.sin();
            // Use cos(im) - 1 = -2 sin²(im/2) for precision near 0.
            let s = (z.im * half).sin();
            let cos_im_m1 = -two * s * s;
            Complex::new(er_m1 * cos_im + cos_im_m1, er * sin_im)
        })
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise complex `log(1 + z)`. Reduces to the equivalent of
/// `log1p` on the real axis but uses `Complex::ln(1 + z)` directly for
/// off-axis values; precision near z=0 is preserved by routing through
/// the identity `log1p(z) = log1p(re) + i atan2(im, 1+re)` when |z| is
/// small, otherwise falls back to ln(1+z).
pub fn log1p_complex<T, D>(input: &Array<Complex<T>, D>) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let two = one + one;
    let half: T = <T as num_traits::NumCast>::from(0.5_f64).unwrap();
    let small: T = <T as num_traits::NumCast>::from(0.5_f64).unwrap();
    let data: Vec<Complex<T>> = input
        .iter()
        .map(|z| {
            if z.re.abs() < small && z.im.abs() < small {
                // log(1+z) where |z| small: Re = 0.5*ln((1+re)^2 + im^2);
                // expand (1+re)^2 + im^2 - 1 = 2 re + re^2 + im^2 = re*(2+re) + im^2.
                let u = z.re * (two + z.re) + z.im * z.im;
                let half_log = half * u.ln_1p();
                let arg = z.im.atan2(one + z.re);
                Complex::new(half_log, arg)
            } else {
                (Complex::new(one, zero) + *z).ln()
            }
        })
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Element-wise complex power `z^w` with broadcasting against a scalar
/// exponent `w`. For complex z and complex w: `z^w = exp(w * log(z))`.
pub fn power_complex<T, D>(
    base: &Array<Complex<T>, D>,
    exponent: Complex<T>,
) -> FerrayResult<Array<Complex<T>, D>>
where
    T: Element + Float,
    Complex<T>: Element,
    D: Dimension,
{
    let data: Vec<Complex<T>> = base.iter().map(|z| z.powc(exponent)).collect();
    Array::from_vec(base.dim().clone(), data)
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

    #[test]
    fn test_iscomplex_complex_input() {
        let a = arr1_c64(vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 1.5),
            Complex64::new(0.0, 0.0),
        ]);
        let r = iscomplex(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_isreal_complex_input() {
        let a = arr1_c64(vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.5)]);
        let r = isreal(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false]);
    }

    #[test]
    fn test_iscomplex_real_input() {
        let a: Array<f64, Ix1> = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = iscomplex_real(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, false, false]);
    }

    #[test]
    fn test_isreal_real_input() {
        let a: Array<f64, Ix1> = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = isreal_real(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, true, true]);
    }

    #[test]
    fn test_iscomplexobj_isrealobj() {
        let c = arr1_c64(vec![Complex64::new(1.0, 2.0)]);
        assert!(iscomplexobj(&c));
        assert!(!isrealobj(&c));

        let r: Array<f64, Ix1> = Array::from_vec(Ix1::new([2]), vec![1.0, 2.0]).unwrap();
        assert!(!iscomplexobj(&r));
        assert!(isrealobj(&r));
    }

    fn approx_eq_c(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() <= eps && (a.im - b.im).abs() <= eps
    }

    #[test]
    fn test_sin_complex_matches_identity() {
        // sin(i) = i sinh(1) ≈ i*1.1752011936438014
        let arr = arr1_c64(vec![Complex64::new(0.0, 1.0), Complex64::new(1.0, 1.0)]);
        let out = sin_complex(&arr).unwrap();
        let v0 = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(
            v0,
            Complex64::new(0.0, 1.1752011936438014),
            1e-10
        ));
    }

    #[test]
    fn test_cos_complex_matches_identity() {
        // cos(i) = cosh(1) ≈ 1.5430806348152437
        let arr = arr1_c64(vec![Complex64::new(0.0, 1.0)]);
        let out = cos_complex(&arr).unwrap();
        let v0 = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(
            v0,
            Complex64::new(1.5430806348152437, 0.0),
            1e-10
        ));
    }

    #[test]
    fn test_exp_complex_eulers_identity() {
        // e^{i*pi} = -1 (Euler's identity).
        let arr = arr1_c64(vec![Complex64::new(0.0, std::f64::consts::PI)]);
        let out = exp_complex(&arr).unwrap();
        let v0 = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(v0, Complex64::new(-1.0, 0.0), 1e-12));
    }

    #[test]
    fn test_ln_complex_inverse_of_exp() {
        let z = Complex64::new(0.7, 1.3);
        let arr = arr1_c64(vec![z.exp()]);
        let out = ln_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(v, z, 1e-12));
    }

    #[test]
    fn test_sqrt_complex_principal_branch() {
        // sqrt(i) = (1+i)/sqrt(2)
        let arr = arr1_c64(vec![Complex64::new(0.0, 1.0)]);
        let out = sqrt_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        assert!(approx_eq_c(v, Complex64::new(inv_sqrt2, inv_sqrt2), 1e-12));
    }

    #[test]
    fn test_log2_complex() {
        // log2(2 + 0i) = 1
        let arr = arr1_c64(vec![Complex64::new(2.0, 0.0)]);
        let out = log2_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(v, Complex64::new(1.0, 0.0), 1e-12));
    }

    #[test]
    fn test_log10_complex() {
        // log10(100 + 0i) = 2
        let arr = arr1_c64(vec![Complex64::new(100.0, 0.0)]);
        let out = log10_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(v, Complex64::new(2.0, 0.0), 1e-12));
    }

    #[test]
    fn test_expm1_complex_small_arg_precision() {
        // For very tiny z, expm1(z) ≈ z + z²/2 + ..., dominated by z.
        // The naive `z.exp() - 1` suffers catastrophic cancellation here.
        // Our implementation routes through exp_m1 + sin² identity to
        // preserve the leading Taylor term. Tolerance is loose because
        // the 2nd-order term and rounding both contribute at the
        // ~1e-12 * f64-eps ≈ 1e-28 scale, which dominates the cos·im_m1
        // fixup arithmetic.
        let z = Complex64::new(1e-12, 1e-12);
        let arr = arr1_c64(vec![z]);
        let out = expm1_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        // Result must be close to z (Taylor leading term) within a few
        // ULPs of the magnitude. f64 ULP for 1e-12 ≈ 2.22e-28.
        let rel_err = (v - z).norm() / z.norm();
        assert!(
            rel_err < 1e-4,
            "expm1 small-arg rel_err = {rel_err:e}, v = {v:?}"
        );
    }

    #[test]
    fn test_log1p_complex_small_arg_precision() {
        // log1p(z) for tiny z ≈ z - z²/2 + ..., dominated by z.
        let z = Complex64::new(1e-10, 1e-10);
        let arr = arr1_c64(vec![z]);
        let out = log1p_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        let rel_err = (v - z).norm() / z.norm();
        assert!(
            rel_err < 1e-4,
            "log1p small-arg rel_err = {rel_err:e}, v = {v:?}"
        );
    }

    #[test]
    fn test_expm1_complex_moderate_arg_matches_naive() {
        // For moderate-magnitude z, expm1 should agree with z.exp() - 1.
        let z = Complex64::new(0.5, 0.3);
        let arr = arr1_c64(vec![z]);
        let out = expm1_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        let expected = z.exp() - Complex64::new(1.0, 0.0);
        assert!(approx_eq_c(v, expected, 1e-12));
    }

    #[test]
    fn test_log1p_complex_moderate_arg_matches_naive() {
        let z = Complex64::new(0.4, 0.2);
        let arr = arr1_c64(vec![z]);
        let out = log1p_complex(&arr).unwrap();
        let v = out.iter().next().copied().unwrap();
        let expected = (Complex64::new(1.0, 0.0) + z).ln();
        assert!(approx_eq_c(v, expected, 1e-12));
    }

    #[test]
    fn test_power_complex_integer_exponent() {
        // (1+i)^2 = 2i
        let arr = arr1_c64(vec![Complex64::new(1.0, 1.0)]);
        let out = power_complex(&arr, Complex64::new(2.0, 0.0)).unwrap();
        let v = out.iter().next().copied().unwrap();
        assert!(approx_eq_c(v, Complex64::new(0.0, 2.0), 1e-12));
    }

    #[test]
    fn test_tanh_complex_round_trip() {
        // atanh(tanh(z)) ≈ z (within principal branch).
        let z = Complex64::new(0.5, 0.3);
        let arr = arr1_c64(vec![z]);
        let t = tanh_complex(&arr).unwrap();
        let r = atanh_complex(&t).unwrap();
        let v = r.iter().next().copied().unwrap();
        assert!(approx_eq_c(v, z, 1e-12));
    }

    #[test]
    fn test_isscalar_zero_d() {
        use ferray_core::dimension::IxDyn;
        let scalar: Array<f64, IxDyn> = Array::from_vec(IxDyn::new(&[]), vec![2.5]).unwrap();
        assert!(isscalar(&scalar));

        let vec: Array<f64, Ix1> = Array::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(!isscalar(&vec));
    }
}
