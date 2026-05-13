// ferray-ufunc: Special functions
//
// sinc, i0 (modified Bessel function of the first kind, order 0)

// Bessel/erf polynomial coefficients are 21-digit scientific constants
// taken verbatim from published reference tables (Cephes); underscore
// separators would diverge from the canonical form, and the trailing
// digits beyond f64 precision are preserved so the listing matches the
// Cephes source line-for-line (the compiler rounds correctly on parse).
#![allow(clippy::unreadable_literal, clippy::excessive_precision)]

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::cr_math::CrMath;
use crate::helpers::unary_float_op;

/// Normalized sinc function: sin(pi*x) / (pi*x).
///
/// AC-13: `sinc(0.0) == 1.0`.
pub fn sinc<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(|| <T as Element>::one());
    unary_float_op(input, |x| {
        if x == <T as Element>::zero() {
            <T as Element>::one()
        } else {
            let px = pi * x;
            px.cr_sin() / px
        }
    })
}

/// Modified Bessel function of the first kind, order 0.
///
/// Uses a Cephes-style Chebyshev polynomial evaluation (Stephen L. Moshier,
/// public-domain Cephes Math Library) to deliver ~1e-15 relative precision
/// at f64 — matching numpy/scipy's reference implementation.
/// AC-13: `i0(0.0) == 1.0`.
pub fn i0<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op(input, bessel_i0_scalar)
}

/// Cephes Chebyshev series for `I0(x) * exp(-x)` on `[0, 8]`.
///
/// 30 coefficients reproduced verbatim from the public-domain Cephes Math
/// Library (Stephen L. Moshier, `cephes/double/i0.c`). The series is
/// evaluated via Clenshaw's recurrence by [`chbevl`].
const CEPHES_I0_A: [f64; 30] = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1,
];

/// Cephes Chebyshev series for `exp(-x) * sqrt(x) * I0(x)` on `[8, ∞)`.
///
/// 25 coefficients reproduced verbatim from the public-domain Cephes Math
/// Library (Stephen L. Moshier, `cephes/double/i0.c`).
const CEPHES_I0_B: [f64; 25] = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1,
];

/// Scalar modified Bessel function `I_0(x)` to Cephes precision (~1e-15 f64).
///
/// Uses Chebyshev polynomial expansions from Cephes (Moshier): coefficient
/// set `A` for `|x| ≤ 8` and set `B` for `|x| > 8`. Exposed so downstream
/// crates (notably ferray-window's Kaiser implementation) can depend on
/// the canonical definition instead of maintaining their own copy (see
/// #530). Closes the conformance gap in #750.
pub fn bessel_i0_scalar<T: Float + CrMath>(x: T) -> T {
    let ax = x.abs();
    let eight = T::from(8.0).unwrap();
    let two = T::from(2.0).unwrap();
    let thirty_two = T::from(32.0).unwrap();

    if ax <= eight {
        // Argument transform: t = |x|/2 - 2  ∈ [-2, 2]
        let t = ax / two - two;
        ax.cr_exp() * chbevl_cephes(t, &CEPHES_I0_A)
    } else {
        // Argument transform: t = 32/|x| - 2  ∈ (-2, 2]
        let t = thirty_two / ax - two;
        ax.cr_exp() * chbevl_cephes(t, &CEPHES_I0_B) / ax.sqrt()
    }
}

/// Clenshaw recurrence evaluation matching Cephes `chbevl` byte-for-byte.
///
/// Computes `0.5 * (b₀ − b₂)` after the canonical do-while sweep, where
///   b₀ = x·b₁ − b₂ + cᵢ
/// updates run for `n − 1` iterations starting from `b₀ = c₀, b₁ = 0`.
#[inline]
fn chbevl_cephes<T: Float + CrMath>(x: T, coeffs: &[f64]) -> T {
    debug_assert!(!coeffs.is_empty(), "chbevl: empty coefficient array");
    let mut b0 = T::from(coeffs[0]).unwrap();
    let mut b1 = T::zero();
    let mut b2 = T::zero();
    for &c in &coeffs[1..] {
        b2 = b1;
        b1 = b0;
        b0 = x * b1 - b2 + T::from(c).unwrap();
    }
    let half = T::from(0.5).unwrap();
    half * (b0 - b2)
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted) — generated via the shared unary_f16_fn!
// macro (#142).
// ---------------------------------------------------------------------------

use crate::helpers::unary_f16_fn;

unary_f16_fn!(
    /// Normalized sinc function for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    sinc_f16,
    |x: f32| {
        if x == 0.0 {
            1.0
        } else {
            let px = std::f32::consts::PI * x;
            core_math::sinf(px) / px
        }
    }
);

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    use crate::test_util::arr1;

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
        // Cephes Chebyshev expansion is bit-near-exact at the origin.
        let a = arr1(vec![0.0]);
        let r = i0(&a).unwrap();
        assert!((r.as_slice().unwrap()[0] - 1.0).abs() < 1e-13);
    }

    #[test]
    fn test_i0_known_values() {
        // Reference values from scipy.special.i0 (Cephes); tightened to
        // 1e-13 after replacing A&S 9.8.1/9.8.2 with the Cephes Chebyshev
        // expansion (issue #750).
        let a = arr1(vec![1.0, 2.0]);
        let r = i0(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.266_065_877_752_008_2).abs() < 1e-13);
        assert!((s[1] - 2.279_585_302_336_067).abs() < 1e-13);
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
