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
//!
//! ## REQ status
//!
//! Two states only (SHIPPED / NOT-STARTED), per goal.md.
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | REQ-POLY-COMPLEX (`numpy.poly` complex roots, #836) | SHIPPED | Impl: `poly_from_roots` returns [`PolyCoeffs::Real`] when the roots are conjugate-closed (`roots_are_conjugate_closed`, mirroring `numpy/lib/_polynomial_impl.py:156-160` `a.real.copy()`), else [`PolyCoeffs::Complex`]. Consumer: `ferray-python/src/polynomial.rs::poly` accepts a complex root sequence and marshals through `poly_from_roots`, returning a `float64` array when imag cancels else `complex128`. |

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
    /// Coefficients are produced in ascending (lowest-first) order.
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

/// Coefficients of the monic polynomial whose roots are `roots`, in ascending
/// (lowest-first) order. Either real (when the imaginary parts cancel) or
/// complex, matching `numpy.poly`'s dtype contract.
#[derive(Debug, Clone, PartialEq)]
pub enum PolyCoeffs {
    /// Every imaginary part cancelled — numpy returns a real (`float64`) array.
    Real(Vec<f64>),
    /// At least one root is not part of a conjugate pair — `complex128`.
    Complex(Vec<Complex<f64>>),
}

/// Build the coefficients of the monic polynomial whose zeros are `roots`,
/// mirroring `numpy.poly(seq_of_zeros)`
/// (`numpy/lib/_polynomial_impl.py:poly`).
///
/// numpy convolves `[1, -zero]` for each root, then — if the accumulated
/// coefficients are complex — checks whether the roots are conjugate-closed
/// (`NX.all(NX.sort(roots) == NX.sort(roots.conjugate()))`,
/// `_polynomial_impl.py:156-160`); if so it returns `a.real.copy()`. This
/// function reproduces that decision, returning [`PolyCoeffs::Real`] for the
/// conjugate-closed case and [`PolyCoeffs::Complex`] otherwise. Coefficients
/// are ascending (lowest-first); callers wanting numpy's highest-first layout
/// reverse the result.
///
/// An empty `roots` slice yields the constant polynomial `[1.0]`
/// (`_polynomial_impl.py:148-149`).
#[must_use]
pub fn poly_from_roots(roots: &[Complex<f64>]) -> PolyCoeffs {
    let complex_coeffs = ComplexPolynomial::from_roots(roots).coeffs().to_vec();

    if roots_are_conjugate_closed(roots) {
        // Imaginary parts cancel: numpy takes `a.real.copy()`.
        PolyCoeffs::Real(complex_coeffs.iter().map(|c| c.re).collect())
    } else {
        PolyCoeffs::Complex(complex_coeffs)
    }
}

/// `true` when sorting `roots` and sorting their conjugates yields the same
/// sequence — numpy's `NX.all(NX.sort(roots) == NX.sort(roots.conjugate()))`
/// test for "all complex roots are complex conjugates"
/// (`numpy/lib/_polynomial_impl.py:159`).
fn roots_are_conjugate_closed(roots: &[Complex<f64>]) -> bool {
    // numpy sorts complex arrays lexicographically by (real, imag) and then
    // compares the two sorted arrays with `==`, under which `-0.0 == +0.0`.
    // `total_cmp` distinguishes signed zeros, which would mis-order a root
    // like `-0.0 - 1j` (numpy's `astype(complex)` produces a `-0.0` real part
    // for `-1j`). Normalize `-0.0` to `+0.0` (via `+ 0.0`) before ordering so
    // the sort agrees with numpy's `==` semantics, while keeping `total_cmp`'s
    // deterministic NaN handling (no panic).
    let norm = |v: f64| v + 0.0;
    let cmp = |x: &Complex<f64>, y: &Complex<f64>| {
        norm(x.re)
            .total_cmp(&norm(y.re))
            .then(norm(x.im).total_cmp(&norm(y.im)))
    };

    let mut sorted: Vec<Complex<f64>> = roots.to_vec();
    sorted.sort_by(cmp);
    let mut sorted_conj: Vec<Complex<f64>> = roots.iter().map(Complex::conj).collect();
    sorted_conj.sort_by(cmp);

    // `Complex<f64>` `PartialEq` uses `f64` equality, so `-0.0 == +0.0` here,
    // matching numpy's element-wise `==`.
    sorted == sorted_conj
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
        assert_eq!(p.coeffs(), &[c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)]);
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

    // ---- poly_from_roots (#836) ------------------------------------------
    // Expected coefficients from the live numpy 2.4.5 oracle:
    //   np.poly([1+2j, 1-2j])  -> array([ 1., -2.,  5.])   (highest-first)
    //   np.poly([1j, -1j])     -> array([1., 0., 1.])
    //   np.poly([1, 2, 3])     -> array([ 1., -6., 11., -6.])
    // poly_from_roots returns LOWEST-first, so reverse to compare.

    #[test]
    fn poly_from_roots_conjugate_pair_is_real() {
        // np.poly([1+2j, 1-2j]) == [1, -2, 5] (highest-first) => lowest [5,-2,1]
        let roots = vec![c(1.0, 2.0), c(1.0, -2.0)];
        let coeffs = poly_from_roots(&roots);
        assert_eq!(coeffs, PolyCoeffs::Real(vec![5.0, -2.0, 1.0]));
    }

    #[test]
    fn poly_from_roots_imaginary_conjugates_real() {
        // np.poly([1j, -1j]) == [1, 0, 1] => lowest-first [1, 0, 1]
        let roots = vec![c(0.0, 1.0), c(0.0, -1.0)];
        let coeffs = poly_from_roots(&roots);
        assert_eq!(coeffs, PolyCoeffs::Real(vec![1.0, 0.0, 1.0]));
    }

    #[test]
    fn poly_from_roots_real_roots_stay_real() {
        // np.poly([1,2,3]) == [1, -6, 11, -6] => lowest-first [-6, 11, -6, 1]
        let roots = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];
        let coeffs = poly_from_roots(&roots);
        assert_eq!(coeffs, PolyCoeffs::Real(vec![-6.0, 11.0, -6.0, 1.0]));
    }

    #[test]
    fn poly_from_roots_unpaired_complex_stays_complex() {
        // np.poly([1+2j]) == [1, -(1+2j)] => lowest-first [-(1+2j), 1]
        let roots = vec![c(1.0, 2.0)];
        let coeffs = poly_from_roots(&roots);
        assert_eq!(
            coeffs,
            PolyCoeffs::Complex(vec![c(-1.0, -2.0), c(1.0, 0.0)])
        );
    }

    #[test]
    fn poly_from_roots_empty_is_constant_one() {
        // np.poly([]) == 1.0
        let coeffs = poly_from_roots(&[]);
        assert_eq!(coeffs, PolyCoeffs::Real(vec![1.0]));
    }

    #[test]
    fn poly_from_roots_signed_zero_real_part_still_conjugate_closed() {
        // numpy's `np.asarray([1j,-1j]).astype(complex)` yields a `-0.0` real
        // part for `-1j`; the pair is still conjugate-closed and numpy returns
        // a REAL [1, 0, 1]. total_cmp must not let the signed zero break it.
        let roots = vec![c(0.0, 1.0), c(-0.0, -1.0)];
        let coeffs = poly_from_roots(&roots);
        assert_eq!(coeffs, PolyCoeffs::Real(vec![1.0, 0.0, 1.0]));
    }
}
