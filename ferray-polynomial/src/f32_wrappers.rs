//! f32 sibling types for the non-power polynomial bases (#725–#729).
//!
//! Each `XF32` newtype wraps the existing f64 `X` polynomial,
//! converting on the public f32 boundary. This gives ergonomic
//! `Array<f32, _>` interop without forcing every callsite to
//! upcast manually, while letting the heavy work (basis conversion,
//! Vandermonde fitting, eigenvalue root-finding) stay on the f64
//! infrastructure.
//!
//! For full f32 native computation the alternative path is the
//! Poly<T> trait genericization tracked in #731. The wrappers here
//! cost one f32→f64 promotion per coefficient on construction and
//! one f64→f32 demotion on observation.

use ferray_core::error::FerrayError;

use crate::chebyshev::Chebyshev;
use crate::hermite::Hermite;
use crate::hermite_e::HermiteE;
use crate::laguerre::Laguerre;
use crate::legendre::Legendre;
use crate::traits::Poly;

macro_rules! impl_f32_wrapper {
    ($wrapper:ident, $f64ty:ty) => {
        #[doc = concat!(
                    "f32 sibling of [`",
                    stringify!($f64ty),
                    "`]. Wraps the f64 type and promotes/demotes on the \
             public boundary. See module-level docs for the \
             tradeoffs vs. the planned generic path (#731)."
                )]
        #[derive(Debug, Clone)]
        pub struct $wrapper {
            inner: $f64ty,
        }

        impl $wrapper {
            /// Construct from f32 coefficients in this basis.
            #[must_use]
            pub fn new(coeffs: &[f32]) -> Self {
                let promoted: Vec<f64> = coeffs.iter().map(|&c| f64::from(c)).collect();
                Self {
                    inner: <$f64ty>::new(&promoted),
                }
            }

            /// Borrow the underlying f64 polynomial. Useful for
            /// reaching the full surface (fit, roots, divmod, ...)
            /// without converting twice.
            #[must_use]
            pub fn as_f64(&self) -> &$f64ty {
                &self.inner
            }

            /// Convert to the f64 sibling type, copying coefficients.
            #[must_use]
            pub fn to_f64(&self) -> $f64ty {
                self.inner.clone()
            }

            /// Construct from an existing f64 polynomial (no precision loss).
            #[must_use]
            pub fn from_f64(p: $f64ty) -> Self {
                Self { inner: p }
            }

            /// f32 coefficient view. Allocates on each call.
            #[must_use]
            pub fn coeffs_f32(&self) -> Vec<f32> {
                self.inner.coeffs().iter().map(|&c| c as f32).collect()
            }

            /// Polynomial degree.
            #[must_use]
            pub fn degree(&self) -> usize {
                self.inner.degree()
            }

            /// Evaluate at a single point.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 evaluation.
            pub fn eval(&self, x: f32) -> Result<f32, FerrayError> {
                Ok(self.inner.eval(f64::from(x))? as f32)
            }

            /// Differentiate `m` times.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 derivative.
            pub fn deriv(&self, m: usize) -> Result<Self, FerrayError> {
                Ok(Self {
                    inner: self.inner.deriv(m)?,
                })
            }

            /// Integrate `m` times with f32 integration constants.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 integration.
            pub fn integ(&self, m: usize, k: &[f32]) -> Result<Self, FerrayError> {
                let k_f64: Vec<f64> = k.iter().map(|&c| f64::from(c)).collect();
                Ok(Self {
                    inner: self.inner.integ(m, &k_f64)?,
                })
            }

            /// Pointwise sum.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 add.
            pub fn add(&self, other: &Self) -> Result<Self, FerrayError> {
                Ok(Self {
                    inner: self.inner.add(&other.inner)?,
                })
            }

            /// Pointwise difference.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 sub.
            pub fn sub(&self, other: &Self) -> Result<Self, FerrayError> {
                Ok(Self {
                    inner: self.inner.sub(&other.inner)?,
                })
            }

            /// Polynomial multiplication.
            ///
            /// # Errors
            /// Propagates errors from the underlying f64 mul.
            pub fn mul(&self, other: &Self) -> Result<Self, FerrayError> {
                Ok(Self {
                    inner: self.inner.mul(&other.inner)?,
                })
            }
        }
    };
}

impl_f32_wrapper!(ChebyshevF32, Chebyshev);
impl_f32_wrapper!(HermiteF32, Hermite);
impl_f32_wrapper!(HermiteEF32, HermiteE);
impl_f32_wrapper!(LegendreF32, Legendre);
impl_f32_wrapper!(LaguerreF32, Laguerre);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chebyshev_f32_eval_t1() {
        // T_1(x) = x. Evaluating at 0.5 should give 0.5.
        let p = ChebyshevF32::new(&[0.0, 1.0]);
        assert!((p.eval(0.5).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn chebyshev_f32_deriv_constant_zero() {
        let p = ChebyshevF32::new(&[5.0]);
        let d = p.deriv(1).unwrap();
        assert_eq!(d.coeffs_f32(), &[0.0]);
    }

    #[test]
    fn chebyshev_f32_add_then_eval() {
        let p = ChebyshevF32::new(&[1.0, 2.0]);
        let q = ChebyshevF32::new(&[3.0, 4.0]);
        let r = p.add(&q).unwrap();
        // (1+3) + (2+4)*T_1(x) = 4 + 6x. Eval at 0.5 → 4 + 3 = 7.
        assert!((r.eval(0.5).unwrap() - 7.0).abs() < 1e-6);
    }

    #[test]
    fn hermite_f32_h1_evaluates_as_2x() {
        // H_1(x) = 2x. coeffs [0, 1] in physicist Hermite basis.
        let p = HermiteF32::new(&[0.0, 1.0]);
        assert!((p.eval(0.5).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn hermite_e_f32_he2_evaluates_correctly() {
        // He_2(x) = x^2 - 1. coeffs [0, 0, 1].
        // At x=2: 4 - 1 = 3.
        let p = HermiteEF32::new(&[0.0, 0.0, 1.0]);
        assert!((p.eval(2.0).unwrap() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn legendre_f32_p1_evaluates_as_x() {
        let p = LegendreF32::new(&[0.0, 1.0]);
        assert!((p.eval(0.7).unwrap() - 0.7).abs() < 1e-6);
    }

    #[test]
    fn laguerre_f32_l0_is_one() {
        let p = LaguerreF32::new(&[1.0]);
        for &x in &[0.0_f32, 0.5, 1.5, 5.0] {
            assert!((p.eval(x).unwrap() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn f32_wrapper_to_from_f64_roundtrip() {
        let p32 = ChebyshevF32::new(&[1.0, 2.0, 3.0]);
        let p64 = p32.to_f64();
        let back = ChebyshevF32::from_f64(p64);
        assert_eq!(p32.coeffs_f32(), back.coeffs_f32());
    }

    #[test]
    fn f32_wrapper_deriv_integ_inverse_within_tolerance() {
        // For Chebyshev: deriv(integ(p)) should reproduce p modulo
        // the integration constant.
        let p = ChebyshevF32::new(&[1.0, 2.0, 3.0]);
        let int_p = p.integ(1, &[0.0]).unwrap();
        let der = int_p.deriv(1).unwrap();
        let p_c = p.coeffs_f32();
        let d_c = der.coeffs_f32();
        assert_eq!(p_c.len(), d_c.len());
        for (a, b) in p_c.iter().zip(d_c.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}
