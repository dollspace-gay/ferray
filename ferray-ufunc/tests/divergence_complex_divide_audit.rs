//! Divergence pin: ferray-ufunc `divide` (TrueDivide) for `Complex<f64>`
//! OVERFLOWS to a non-finite quotient where NumPy's complex division stays
//! FINITE.
//!
//! NumPy `np.divide` over complex128 computes `(2+0j) / (6.675e-308+0j)` as a
//! FINITE `2.99625468164794e+307 + 0j` (live numpy 2.4.4:
//! `np.divide(np.array([2+0j]), np.array([6.675e-308+0j]))[0]`
//! `== (2.99625468164794e+307+0j)`, `np.isfinite(...) is True`).
//!
//! ferray-ufunc's `TrueDivide for Complex<f64>` forwards to `num_complex`'s
//! `Div`, whose Smith-algorithm division of this operand pair OVERFLOWS to
//! `inf + NaN*i` (non-finite). This is a real numerical-contract divergence
//! (R-DEV-1): num_complex scales by `1/|b|` components, hitting `2 / 6.675e-308`
//! which overflows before the final divide. It surfaces in numpy.ma as a
//! SPURIOUS mask, since `_DomainedBinaryOperation` masks `~isfinite(result)`
//! (`numpy/ma/core.py:1213`).
//!
//! Expected value is the live numpy quotient (R-CHAR-3), NOT copied from ferray.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_ufunc::ops::arithmetic::divide;
use num_complex::Complex64;

fn c64(data: Vec<Complex64>) -> Array<Complex64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).expect("build complex array")
}

/// Divergence: `divide` for Complex<f64> overflows a quotient numpy keeps finite.
/// numpy: `(2+0j)/(6.675e-308+0j) = 2.99625468164794e+307+0j` (finite).
/// ferray (num_complex): `inf + NaN*i` (non-finite).
#[test]
fn complex_divide_finite_quotient_does_not_overflow() {
    let a = c64(vec![Complex64::new(2.0, 0.0)]);
    let b = c64(vec![Complex64::new(6.675e-308, 0.0)]);
    let q = *divide(&a, &b)
        .expect("divide ok")
        .iter()
        .next()
        .expect("non-empty");

    // numpy keeps this finite.
    assert!(
        q.is_finite(),
        "ferray complex divide overflowed to {q:?}; numpy yields a finite \
         2.99625468164794e+307+0j"
    );
    // And it equals numpy's quotient.
    let expected = Complex64::new(2.996_254_681_647_94e307, 0.0);
    assert!(
        (q.re - expected.re).abs() <= expected.re.abs() * 1e-12 && q.im == 0.0,
        "ferray {q:?} != numpy {expected:?}"
    );
}
