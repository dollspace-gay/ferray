//! Complex-coefficient power-basis polynomial (#730).
//!
//! `ComplexPolynomial` mirrors [`crate::power::Polynomial`] for
//! complex coefficients. Mostly used for signal-processing
//! transfer functions, z-transform manipulation, and complex root
//! finding — the standard real Polynomial covers the common
//! statistics / fitting cases.
//!
//! Coefficients are `Vec<Complex<f64>>` in ascending order. Domain
//! / window are real `[f64; 2]`; complex inputs are accepted via
//! [`ComplexPolynomial::eval_complex`].
//!
//! Roots are tracked separately — finding roots of a
//! complex-coefficient polynomial requires a complex companion-
//! matrix eigensolver, which ferray-linalg does not yet expose.
//! Real-input fitting also stays out of scope here.

use ferray_core::error::{FerrayError, FerrayResult};
use num_complex::Complex;

/// Complex-coefficient power-basis polynomial.
#[derive(Debug, Clone)]
pub struct ComplexPolynomial {
    coeffs: Vec<Complex<f64>>,
}

impl ComplexPolynomial {
    /// Build from coefficients in ascending order. An empty slice
    /// becomes the zero polynomial `[0 + 0i]`.
    #[must_use]
    pub fn new(coeffs: &[Complex<f64>]) -> Self {
        let coeffs = if coeffs.is_empty() {
            vec![Complex::new(0.0, 0.0)]
        } else {
            coeffs.to_vec()
        };
        Self { coeffs }
    }

    /// Coefficient slice.
    #[must_use]
    pub fn coeffs(&self) -> &[Complex<f64>] {
        &self.coeffs
    }

    /// Polynomial degree (length - 1, with trailing exact zeros stripped).
    #[must_use]
    pub fn degree(&self) -> usize {
        let mut deg = self.coeffs.len().saturating_sub(1);
        let zero = Complex::new(0.0, 0.0);
        while deg > 0 && self.coeffs[deg] == zero {
            deg -= 1;
        }
        deg
    }

    /// Evaluate at a real point via complex Horner.
    #[must_use]
    pub fn eval(&self, x: f64) -> Complex<f64> {
        self.eval_complex(Complex::new(x, 0.0))
    }

    /// Evaluate at a complex point via Horner.
    #[must_use]
    pub fn eval_complex(&self, x: Complex<f64>) -> Complex<f64> {
        let mut acc = Complex::new(0.0, 0.0);
        for &c in self.coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    /// Differentiate `m` times. Returns the zero polynomial if all
    /// coefficients fall away.
    #[must_use]
    pub fn deriv(&self, m: usize) -> Self {
        if m == 0 {
            return self.clone();
        }
        let mut coeffs = self.coeffs.clone();
        for _ in 0..m {
            if coeffs.len() <= 1 {
                coeffs = vec![Complex::new(0.0, 0.0)];
                break;
            }
            let mut next = Vec::with_capacity(coeffs.len() - 1);
            for (i, &c) in coeffs.iter().enumerate().skip(1) {
                next.push(c * Complex::new(i as f64, 0.0));
            }
            coeffs = next;
        }
        if coeffs.is_empty() {
            coeffs = vec![Complex::new(0.0, 0.0)];
        }
        Self { coeffs }
    }

    /// Integrate `m` times with complex integration constants.
    #[must_use]
    pub fn integ(&self, m: usize, k: &[Complex<f64>]) -> Self {
        if m == 0 {
            return self.clone();
        }
        let mut coeffs = self.coeffs.clone();
        for step in 0..m {
            let constant = k.get(step).copied().unwrap_or(Complex::new(0.0, 0.0));
            let mut next = Vec::with_capacity(coeffs.len() + 1);
            next.push(constant);
            for (i, &c) in coeffs.iter().enumerate() {
                next.push(c / Complex::new((i + 1) as f64, 0.0));
            }
            coeffs = next;
        }
        Self { coeffs }
    }

    /// Pointwise sum of two polynomials.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![Complex::new(0.0, 0.0); n];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] += c;
        }
        Self { coeffs: out }
    }

    /// Pointwise difference.
    #[must_use]
    pub fn sub(&self, other: &Self) -> Self {
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![Complex::new(0.0, 0.0); n];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] -= c;
        }
        Self { coeffs: out }
    }

    /// Convolution-style polynomial multiplication.
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let na = self.coeffs.len();
        let nb = other.coeffs.len();
        let mut out = vec![Complex::new(0.0, 0.0); na + nb - 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] += a * b;
            }
        }
        Self { coeffs: out }
    }

    /// Construct a complex polynomial whose roots are exactly
    /// `roots` (mirroring [`crate::power::Polynomial::from_roots`]).
    #[must_use]
    pub fn from_roots(roots: &[Complex<f64>]) -> Self {
        let mut coeffs = vec![Complex::new(1.0, 0.0)];
        for &r in roots {
            let n = coeffs.len();
            let mut next = vec![Complex::new(0.0, 0.0); n + 1];
            for (i, &c) in coeffs.iter().enumerate() {
                next[i] -= r * c;
                next[i + 1] += c;
            }
            coeffs = next;
        }
        Self { coeffs }
    }

    /// Trim trailing coefficients with magnitude `<= tol`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `tol < 0.0`.
    pub fn trim(&self, tol: f64) -> FerrayResult<Self> {
        if tol < 0.0 {
            return Err(FerrayError::invalid_value(
                "ComplexPolynomial::trim: tolerance must be non-negative",
            ));
        }
        let mut last = self.coeffs.len();
        while last > 1 && self.coeffs[last - 1].norm() <= tol {
            last -= 1;
        }
        Ok(Self {
            coeffs: self.coeffs[..last].to_vec(),
        })
    }

    /// Truncate to the given number of leading terms.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size == 0`.
    pub fn truncate(&self, size: usize) -> FerrayResult<Self> {
        if size == 0 {
            return Err(FerrayError::invalid_value(
                "ComplexPolynomial::truncate: size must be at least 1",
            ));
        }
        let len = size.min(self.coeffs.len());
        Ok(Self {
            coeffs: self.coeffs[..len].to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(re: f64, im: f64) -> Complex<f64> {
        Complex::new(re, im)
    }

    #[test]
    fn eval_constant_complex() {
        let p = ComplexPolynomial::new(&[c(2.0, 3.0)]);
        let v = p.eval(0.5);
        assert_eq!(v, c(2.0, 3.0));
    }

    #[test]
    fn eval_real_input_imaginary_coeffs() {
        // p(x) = i + (1+i)x at x=1 → i + (1+i) = 1 + 2i
        let p = ComplexPolynomial::new(&[c(0.0, 1.0), c(1.0, 1.0)]);
        let v = p.eval(1.0);
        assert!((v.re - 1.0).abs() < 1e-12);
        assert!((v.im - 2.0).abs() < 1e-12);
    }

    #[test]
    fn eval_complex_at_imaginary_unit() {
        // p(x) = 1 + x^2. Eval at x = i: 1 + i^2 = 1 - 1 = 0.
        let p = ComplexPolynomial::new(&[c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]);
        let v = p.eval_complex(c(0.0, 1.0));
        assert!(v.re.abs() < 1e-12);
        assert!(v.im.abs() < 1e-12);
    }

    #[test]
    fn deriv_of_constant_is_zero() {
        let p = ComplexPolynomial::new(&[c(5.0, 5.0)]);
        let d = p.deriv(1);
        assert_eq!(d.coeffs(), &[c(0.0, 0.0)]);
    }

    #[test]
    fn deriv_of_linear() {
        // p(x) = (1+i) + (2+3i)x → p'(x) = 2 + 3i
        let p = ComplexPolynomial::new(&[c(1.0, 1.0), c(2.0, 3.0)]);
        let d = p.deriv(1);
        assert_eq!(d.coeffs(), &[c(2.0, 3.0)]);
    }

    #[test]
    fn integ_then_deriv_recovers_self() {
        let p = ComplexPolynomial::new(&[c(1.0, 2.0), c(3.0, -1.0), c(0.0, 4.0)]);
        let q = p.integ(1, &[c(0.0, 0.0)]).deriv(1);
        for (a, b) in p.coeffs().iter().zip(q.coeffs().iter()) {
            assert!((a - b).norm() < 1e-12);
        }
    }

    #[test]
    fn add_sub_pointwise() {
        let a = ComplexPolynomial::new(&[c(1.0, 0.0), c(2.0, 0.0)]);
        let b = ComplexPolynomial::new(&[c(0.0, 1.0), c(0.0, -1.0)]);
        let s = a.add(&b);
        assert_eq!(s.coeffs(), &[c(1.0, 1.0), c(2.0, -1.0)]);
        let d = a.sub(&b);
        assert_eq!(d.coeffs(), &[c(1.0, -1.0), c(2.0, 1.0)]);
    }

    #[test]
    fn mul_known_value() {
        // (1 + i*x) * (1 - i*x) = 1 + x^2 (since i * -i = 1)
        let a = ComplexPolynomial::new(&[c(1.0, 0.0), c(0.0, 1.0)]);
        let b = ComplexPolynomial::new(&[c(1.0, 0.0), c(0.0, -1.0)]);
        let p = a.mul(&b);
        assert_eq!(
            p.coeffs(),
            &[c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]
        );
    }

    #[test]
    fn from_roots_reproduces_zeros() {
        let roots = vec![c(1.0, 0.0), c(0.0, 1.0), c(-1.0, 0.0)];
        let p = ComplexPolynomial::from_roots(&roots);
        for &r in &roots {
            let v = p.eval_complex(r);
            assert!(v.norm() < 1e-10, "eval at root {r}: {v}");
        }
    }

    #[test]
    fn trim_strips_trailing_small_coeffs() {
        let p = ComplexPolynomial::new(&[c(1.0, 0.0), c(2.0, 0.0), c(1e-15, 0.0)]);
        let trimmed = p.trim(1e-12).unwrap();
        assert_eq!(trimmed.coeffs().len(), 2);
    }

    #[test]
    fn truncate_keeps_leading() {
        let p = ComplexPolynomial::new(&[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)]);
        let t = p.truncate(2).unwrap();
        assert_eq!(t.coeffs(), &[c(1.0, 0.0), c(2.0, 0.0)]);
    }
}
