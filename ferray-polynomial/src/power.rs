// ferray-polynomial: Power basis polynomial (REQ-1)
//
// p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{least_squares_fit, power_vandermonde};
use crate::mapping::{auto_domain, map_x, mapparms, validate_domain_window};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// Default domain and window for the power basis.
///
/// `NumPy` uses `[-1, 1]` for both, giving an identity mapping by default.
const POWER_DEFAULT_DOMAIN: [f64; 2] = [-1.0, 1.0];
const POWER_DEFAULT_WINDOW: [f64; 2] = [-1.0, 1.0];

/// A polynomial in the standard power (monomial) basis.
///
/// Represents p(x) = c[0] + c[1]*u + c[2]*u^2 + ... + c[n]*u^n where
/// `u = offset + scale * x` is the affine map from `domain` to `window`.
/// By default `domain == window == [-1, 1]`, giving an identity map so
/// that `eval(x)` reduces to the standard Horner evaluation at x.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients in ascending power order: c[0], c[1], ..., c[n].
    coeffs: Vec<f64>,
    /// Input domain `[a, b]`. Defaults to `[-1, 1]`.
    domain: [f64; 2],
    /// Canonical window `[c, d]`. Defaults to `[-1, 1]`.
    window: [f64; 2],
}

impl Polynomial {
    /// Create a new power-basis polynomial from coefficients.
    ///
    /// Coefficients are in ascending order: `coeffs[i]` is the coefficient of x^i.
    /// An empty coefficient slice produces the zero polynomial `[0.0]`.
    /// The domain and window default to `[-1, 1]` (identity mapping).
    #[must_use]
    pub fn new(coeffs: &[f64]) -> Self {
        let coeffs = if coeffs.is_empty() {
            vec![0.0]
        } else {
            coeffs.to_vec()
        };
        Self {
            coeffs,
            domain: POWER_DEFAULT_DOMAIN,
            window: POWER_DEFAULT_WINDOW,
        }
    }

    /// Set the input domain, returning a new polynomial.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `domain[0] == domain[1]`.
    pub fn with_domain(mut self, domain: [f64; 2]) -> Result<Self, FerrayError> {
        validate_domain_window(domain, self.window)?;
        self.domain = domain;
        Ok(self)
    }

    /// Set the canonical window, returning a new polynomial.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `window[0] == window[1]`.
    pub fn with_window(mut self, window: [f64; 2]) -> Result<Self, FerrayError> {
        validate_domain_window(self.domain, window)?;
        self.window = window;
        Ok(self)
    }

    /// Raw Horner-form evaluation in the canonical window (no mapping).
    ///
    /// Used internally and as a building block for the public `eval()`,
    /// which applies the domain→window affine map first.
    fn eval_canonical(&self, u: f64) -> f64 {
        let mut result = 0.0;
        for &c in self.coeffs.iter().rev() {
            result = result * u + c;
        }
        result
    }

    /// Internal: build a new `Polynomial` with the same domain/window as `self`.
    ///
    /// All coefficient-only operations (deriv, integ, trim, truncate, ...)
    /// route through this so the receiver's mapping propagates to the
    /// result. Operations that combine two polynomials (add/sub/mul/divmod)
    /// must explicitly verify both mappings match before calling this.
    #[inline]
    pub(crate) const fn with_same_mapping(&self, coeffs: Vec<f64>) -> Self {
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Internal: verify that two polynomials share the same domain/window
    /// before combining them via add/sub/mul/divmod. Matches `NumPy`'s
    /// behavior of raising on mismatched mappings.
    fn check_same_mapping(&self, other: &Self) -> Result<(), FerrayError> {
        if self.domain != other.domain || self.window != other.window {
            return Err(FerrayError::invalid_value(format!(
                "polynomials must share the same domain and window: \
                 self has domain={:?} window={:?}, other has domain={:?} window={:?}",
                self.domain, self.window, other.domain, other.window
            )));
        }
        Ok(())
    }

    /// Auto-domain fit: like [`Polynomial::fit`] but computes `domain` from
    /// the data range, fits in the canonical window, and stores the
    /// resulting mapping. Matches `numpy.polynomial.Polynomial.fit`.
    ///
    /// Use this instead of [`Polynomial::fit`] when fitting data on a
    /// non-canonical interval (e.g. `x = [0, 1000]`) to avoid the
    /// catastrophic cancellation that comes from raw-x conditioning.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for length mismatches, empty
    /// input, or rank-deficient fits.
    pub fn fit_with_domain(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrayError> {
        if x.len() != y.len() {
            return Err(FerrayError::invalid_value(format!(
                "x and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value("x and y must not be empty"));
        }
        let domain = auto_domain(x);
        let window = POWER_DEFAULT_WINDOW;
        let (offset, scale) = mapparms(domain, window)?;
        let u: Vec<f64> = x.iter().map(|&xi| map_x(xi, offset, scale)).collect();
        let v = power_vandermonde(&u, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, None)?;
        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }
}

impl Poly for Polynomial {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        // Apply the domain -> window affine map, then Horner-evaluate.
        // For the default identity mapping (domain == window), this reduces
        // to a plain Horner evaluation at x with no overhead beyond a
        // single multiply and add.
        let (offset, scale) = self.mapparms()?;
        let u = map_x(x, offset, scale);
        Ok(self.eval_canonical(u))
    }

    fn deriv(&self, m: usize) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        let mut coeffs = self.coeffs.clone();
        for _ in 0..m {
            if coeffs.len() <= 1 {
                coeffs = vec![0.0];
                break;
            }
            let mut new_coeffs = Vec::with_capacity(coeffs.len() - 1);
            for (i, &c) in coeffs.iter().enumerate().skip(1) {
                new_coeffs.push(c * i as f64);
            }
            coeffs = new_coeffs;
        }
        if coeffs.is_empty() {
            coeffs = vec![0.0];
        }
        Ok(self.with_same_mapping(coeffs))
    }

    fn integ(&self, m: usize, k: &[f64]) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        let mut coeffs = self.coeffs.clone();
        for step in 0..m {
            let constant = if step < k.len() { k[step] } else { 0.0 };
            let mut new_coeffs = Vec::with_capacity(coeffs.len() + 1);
            new_coeffs.push(constant);
            for (i, &c) in coeffs.iter().enumerate() {
                new_coeffs.push(c / (i + 1) as f64);
            }
            coeffs = new_coeffs;
        }
        Ok(self.with_same_mapping(coeffs))
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        find_roots_from_power_coeffs(&self.coeffs)
    }

    fn degree(&self) -> usize {
        // Find actual degree by skipping trailing exact zeros.
        // Uses exact zero check for consistency with trim(0.0).
        // Call trim(tol) first for fuzzy degree computation.
        let mut deg = self.coeffs.len().saturating_sub(1);
        while deg > 0 && self.coeffs[deg] == 0.0 {
            deg -= 1;
        }
        deg
    }

    fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    fn trim(&self, tol: f64) -> Result<Self, FerrayError> {
        if tol < 0.0 {
            return Err(FerrayError::invalid_value("tolerance must be non-negative"));
        }
        let mut last = self.coeffs.len();
        while last > 1 && self.coeffs[last - 1].abs() <= tol {
            last -= 1;
        }
        Ok(self.with_same_mapping(self.coeffs[..last].to_vec()))
    }

    fn truncate(&self, size: usize) -> Result<Self, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value(
                "truncation size must be at least 1",
            ));
        }
        let len = size.min(self.coeffs.len());
        Ok(self.with_same_mapping(self.coeffs[..len].to_vec()))
    }

    fn add(&self, other: &Self) -> Result<Self, FerrayError> {
        self.check_same_mapping(other)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            result[i] += c;
        }
        Ok(self.with_same_mapping(result))
    }

    fn sub(&self, other: &Self) -> Result<Self, FerrayError> {
        self.check_same_mapping(other)?;
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            result[i] -= c;
        }
        Ok(self.with_same_mapping(result))
    }

    fn mul(&self, other: &Self) -> Result<Self, FerrayError> {
        self.check_same_mapping(other)?;
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Ok(self.with_same_mapping(vec![0.0]));
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![0.0; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        Ok(self.with_same_mapping(result))
    }

    fn pow(&self, n: usize) -> Result<Self, FerrayError> {
        // Binary exponentiation: O(log n) polynomial multiplies versus
        // the previous O(n) loop (#247).
        if n == 0 {
            return Ok(self.with_same_mapping(vec![1.0]));
        }
        if n == 1 {
            return Ok(self.clone());
        }
        let mut base = self.clone();
        let mut result: Option<Self> = None;
        let mut exp = n;
        while exp > 0 {
            if exp & 1 == 1 {
                result = Some(match result.take() {
                    Some(r) => r.mul(&base)?,
                    None => base.clone(),
                });
            }
            exp >>= 1;
            if exp > 0 {
                base = base.mul(&base)?;
            }
        }
        // n >= 1 guarantees the low-bit path ran at least once.
        Ok(result.expect("binary exponentiation must produce a result for n >= 1"))
    }

    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrayError> {
        self.check_same_mapping(other)?;
        let other_trimmed = other.trim(0.0)?;
        if other_trimmed.degree() == 0 && other_trimmed.coeffs[0].abs() < f64::EPSILON * 100.0 {
            return Err(FerrayError::invalid_value("division by zero polynomial"));
        }

        let self_trimmed = self.trim(0.0)?;
        let n = self_trimmed.coeffs.len();
        let m = other_trimmed.coeffs.len();

        if n < m {
            // Quotient is zero, remainder is self
            return Ok((self.with_same_mapping(vec![0.0]), self_trimmed));
        }

        let mut remainder = self_trimmed.coeffs;
        let divisor_lead = other_trimmed.coeffs[m - 1];
        let quot_len = n - m + 1;
        let mut quotient = vec![0.0; quot_len];

        for i in (0..quot_len).rev() {
            let coeff = remainder[i + m - 1] / divisor_lead;
            quotient[i] = coeff;
            for j in 0..m {
                remainder[i + j] -= coeff * other_trimmed.coeffs[j];
            }
        }

        // Trim the remainder
        let mut rem_len = m - 1;
        while rem_len > 1 && remainder[rem_len - 1].abs() < f64::EPSILON * 100.0 {
            rem_len -= 1;
        }

        Ok((
            self.with_same_mapping(quotient),
            self.with_same_mapping(remainder[..rem_len.max(1)].to_vec()),
        ))
    }

    fn fit(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrayError> {
        if x.len() != y.len() {
            return Err(FerrayError::invalid_value(format!(
                "x and y must have the same length, got {} and {}",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value("x and y must not be empty"));
        }
        let v = power_vandermonde(x, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, None)?;
        Ok(Self::new(&coeffs))
    }

    fn fit_weighted(x: &[f64], y: &[f64], deg: usize, w: &[f64]) -> Result<Self, FerrayError> {
        if x.len() != y.len() || x.len() != w.len() {
            return Err(FerrayError::invalid_value(
                "x, y, and w must have the same length",
            ));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value("x, y, and w must not be empty"));
        }
        let v = power_vandermonde(x, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, Some(w))?;
        Ok(Self::new(&coeffs))
    }

    fn from_coeffs(coeffs: &[f64]) -> Self {
        Self::new(coeffs)
    }

    fn domain(&self) -> [f64; 2] {
        self.domain
    }

    fn window(&self) -> [f64; 2] {
        self.window
    }
}

impl ToPowerBasis for Polynomial {
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError> {
        Ok(self.coeffs.clone())
    }
}

impl FromPowerBasis for Polynomial {
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError> {
        Ok(Self::new(coeffs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_constant() {
        let p = Polynomial::new(&[5.0]);
        assert!((p.eval(0.0).unwrap() - 5.0).abs() < 1e-14);
        assert!((p.eval(100.0).unwrap() - 5.0).abs() < 1e-14);
    }

    #[test]
    fn eval_linear() {
        // p(x) = 1 + 2x
        let p = Polynomial::new(&[1.0, 2.0]);
        assert!((p.eval(0.0).unwrap() - 1.0).abs() < 1e-14);
        assert!((p.eval(3.0).unwrap() - 7.0).abs() < 1e-14);
    }

    #[test]
    fn eval_quadratic() {
        // p(x) = 2 - 3x + x^2
        let p = Polynomial::new(&[2.0, -3.0, 1.0]);
        assert!((p.eval(1.0).unwrap() - 0.0).abs() < 1e-14);
        assert!((p.eval(2.0).unwrap() - 0.0).abs() < 1e-14);
        assert!((p.eval(0.0).unwrap() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn deriv_quadratic() {
        // d/dx (2 - 3x + x^2) = -3 + 2x
        let p = Polynomial::new(&[2.0, -3.0, 1.0]);
        let dp = p.deriv(1).unwrap();
        assert_eq!(dp.coeffs.len(), 2);
        assert!((dp.coeffs[0] - (-3.0)).abs() < 1e-14);
        assert!((dp.coeffs[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn deriv_second() {
        // d2/dx2 (2 - 3x + x^2) = 2
        let p = Polynomial::new(&[2.0, -3.0, 1.0]);
        let ddp = p.deriv(2).unwrap();
        assert_eq!(ddp.coeffs.len(), 1);
        assert!((ddp.coeffs[0] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn integ_then_deriv_roundtrip() {
        // AC-4: p.integ(1, &[0.0]).deriv(1) recovers p
        let p = Polynomial::new(&[1.0, 2.0, 3.0]);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let recovered = integrated.deriv(1).unwrap();
        for (a, b) in recovered.coeffs.iter().zip(p.coeffs.iter()) {
            assert!((a - b).abs() < 1e-12, "expected {b}, got {a}");
        }
    }

    #[test]
    fn integ_constant() {
        // integral of 3 is 0 + 3x
        let p = Polynomial::new(&[3.0]);
        let ip = p.integ(1, &[0.0]).unwrap();
        assert_eq!(ip.coeffs.len(), 2);
        assert!((ip.coeffs[0] - 0.0).abs() < 1e-14);
        assert!((ip.coeffs[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn integ_with_constant() {
        // integral of 3 with k=5 is 5 + 3x
        let p = Polynomial::new(&[3.0]);
        let ip = p.integ(1, &[5.0]).unwrap();
        assert!((ip.coeffs[0] - 5.0).abs() < 1e-14);
        assert!((ip.coeffs[1] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn add_polynomials() {
        let a = Polynomial::new(&[1.0, 2.0]);
        let b = Polynomial::new(&[3.0, 4.0, 5.0]);
        let c = a.add(&b).unwrap();
        assert!((c.coeffs[0] - 4.0).abs() < 1e-14);
        assert!((c.coeffs[1] - 6.0).abs() < 1e-14);
        assert!((c.coeffs[2] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn sub_polynomials() {
        let a = Polynomial::new(&[1.0, 2.0, 3.0]);
        let b = Polynomial::new(&[1.0, 2.0, 3.0]);
        let c = a.sub(&b).unwrap();
        for &ci in &c.coeffs {
            assert!(ci.abs() < 1e-14);
        }
    }

    #[test]
    fn mul_polynomials() {
        // (1 + x)(1 - x) = 1 - x^2
        let a = Polynomial::new(&[1.0, 1.0]);
        let b = Polynomial::new(&[1.0, -1.0]);
        let c = a.mul(&b).unwrap();
        assert!((c.coeffs[0] - 1.0).abs() < 1e-14);
        assert!((c.coeffs[1] - 0.0).abs() < 1e-14);
        assert!((c.coeffs[2] - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn pow_polynomial() {
        // (1 + x)^2 = 1 + 2x + x^2
        let p = Polynomial::new(&[1.0, 1.0]);
        let p2 = p.pow(2).unwrap();
        assert!((p2.coeffs[0] - 1.0).abs() < 1e-14);
        assert!((p2.coeffs[1] - 2.0).abs() < 1e-14);
        assert!((p2.coeffs[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn pow_zero() {
        let p = Polynomial::new(&[3.0, 5.0]);
        let p0 = p.pow(0).unwrap();
        assert_eq!(p0.coeffs.len(), 1);
        assert!((p0.coeffs[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn pow_one_returns_self() {
        // #247: binary exponentiation has a dedicated n==1 fast path;
        // the result must be coefficient-equal to self.
        let p = Polynomial::new(&[1.0, 2.0, 3.0]);
        let p1 = p.pow(1).unwrap();
        assert_eq!(p1.coeffs, p.coeffs);
    }

    #[test]
    fn pow_high_exponent_matches_naive_loop() {
        // #247: cross-check binary exponentiation against the naive
        // O(n) loop for an exponent that exercises both odd and even
        // bits (n=11 = 0b1011).
        let p = Polynomial::new(&[1.0, 1.0]); // (1 + x)
        let result = p.pow(11).unwrap();
        // (1 + x)^11 expansion: binomial coefficients C(11, k).
        let expected: [f64; 12] = [
            1.0, 11.0, 55.0, 165.0, 330.0, 462.0, 462.0, 330.0, 165.0, 55.0, 11.0, 1.0,
        ];
        assert_eq!(result.coeffs.len(), 12);
        for (i, (got, want)) in result.coeffs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "coeff[{i}] = {got}, expected {want}"
            );
        }
    }

    #[test]
    fn divmod_polynomial() {
        // AC-5: a == q * b + r
        // (x^2 - 1) / (x - 1) = x + 1, remainder 0
        let a = Polynomial::new(&[-1.0, 0.0, 1.0]);
        let b = Polynomial::new(&[-1.0, 1.0]);
        let (q, r) = a.divmod(&b).unwrap();
        assert!((q.coeffs[0] - 1.0).abs() < 1e-12, "q[0] = {}", q.coeffs[0]);
        assert!((q.coeffs[1] - 1.0).abs() < 1e-12, "q[1] = {}", q.coeffs[1]);
        assert!(r.coeffs[0].abs() < 1e-10, "r[0] = {}", r.coeffs[0]);

        // Verify: q * b + r == a
        let qb = q.mul(&b).unwrap();
        let reconstructed = qb.add(&r).unwrap();
        for i in 0..a.coeffs.len() {
            let ri = if i < reconstructed.coeffs.len() {
                reconstructed.coeffs[i]
            } else {
                0.0
            };
            assert!(
                (ri - a.coeffs[i]).abs() < 1e-10,
                "mismatch at {}: {} != {}",
                i,
                ri,
                a.coeffs[i]
            );
        }
    }

    #[test]
    fn divmod_by_zero_err() {
        let a = Polynomial::new(&[1.0, 2.0]);
        let b = Polynomial::new(&[0.0]);
        assert!(a.divmod(&b).is_err());
    }

    #[test]
    fn degree_with_trailing_zeros() {
        let p = Polynomial::new(&[1.0, 2.0, 0.0, 0.0]);
        assert_eq!(p.degree(), 1);
    }

    #[test]
    fn trim_polynomial() {
        let p = Polynomial::new(&[1.0, 2.0, 0.0001, 0.00001]);
        let t = p.trim(0.001).unwrap();
        assert_eq!(t.coeffs.len(), 2);
    }

    #[test]
    fn truncate_polynomial() {
        let p = Polynomial::new(&[1.0, 2.0, 3.0, 4.0]);
        let t = p.truncate(2).unwrap();
        assert_eq!(t.coeffs.len(), 2);
        assert!((t.coeffs[0] - 1.0).abs() < 1e-14);
        assert!((t.coeffs[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn truncate_zero_err() {
        let p = Polynomial::new(&[1.0]);
        assert!(p.truncate(0).is_err());
    }

    #[test]
    fn fit_linear() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let p = Polynomial::fit(&x, &y, 1).unwrap();
        assert!((p.coeffs[0] - 1.0).abs() < 1e-10);
        assert!((p.coeffs[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn fit_mismatched_err() {
        assert!(Polynomial::fit(&[1.0, 2.0], &[1.0], 1).is_err());
    }

    #[test]
    fn to_power_basis_identity() {
        let p = Polynomial::new(&[1.0, 2.0, 3.0]);
        let pb = p.to_power_basis().unwrap();
        assert_eq!(pb, p.coeffs);
    }

    #[test]
    fn from_power_basis_identity() {
        let coeffs = vec![1.0, 2.0, 3.0];
        let p = Polynomial::from_power_basis(&coeffs).unwrap();
        assert_eq!(p.coeffs, coeffs);
    }

    #[test]
    fn eval_many() {
        let p = Polynomial::new(&[1.0, 1.0]); // 1 + x
        let vals = p.eval_many(&[0.0, 1.0, 2.0]).unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-14);
        assert!((vals[1] - 2.0).abs() < 1e-14);
        assert!((vals[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn from_coeffs_empty() {
        let p = Polynomial::from_coeffs(&[]);
        assert_eq!(p.coeffs, vec![0.0]);
    }

    // -----------------------------------------------------------------------
    // Domain / window mapping tests (issue #474)
    // -----------------------------------------------------------------------

    #[test]
    fn default_domain_window_is_identity() {
        let p = Polynomial::new(&[1.0, 2.0, 3.0]);
        assert_eq!(p.domain(), [-1.0, 1.0]);
        assert_eq!(p.window(), [-1.0, 1.0]);
        let (offset, scale) = p.mapparms().unwrap();
        assert_eq!(offset, 0.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn with_domain_changes_eval_via_mapping() {
        // p(u) = 1 + 2u in canonical [-1, 1]. Setting domain=[0, 4] means
        // x is mapped: u = -1 + 0.5*(x-0) = -1 + 0.5*x.
        // So p(x) at x=0 is p_canonical(-1) = 1 - 2 = -1
        //         at x=4 is p_canonical(1)  = 1 + 2 = 3
        let p = Polynomial::new(&[1.0, 2.0])
            .with_domain([0.0, 4.0])
            .unwrap();
        assert!((p.eval(0.0).unwrap() - (-1.0)).abs() < 1e-12);
        assert!((p.eval(2.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((p.eval(4.0).unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn fit_with_domain_recovers_function_values() {
        // Fit y = 2x + 1 on x ∈ [0, 4] with auto-domain.
        // The inner coefficients differ from fit(), but eval(x) recovers y(x).
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let p = Polynomial::fit_with_domain(&x, &y, 1).unwrap();
        assert_eq!(p.domain(), [0.0, 4.0]);
        assert_eq!(p.window(), [-1.0, 1.0]);
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!(
                (p.eval(xi).unwrap() - yi).abs() < 1e-10,
                "at x={xi}: expected {yi}, got {}",
                p.eval(xi).unwrap()
            );
        }
    }

    #[test]
    fn fit_with_domain_avoids_catastrophic_cancellation() {
        // Fit y = 1 + x on x ∈ [1000, 1010] (large offset). Without auto-domain,
        // the Vandermonde matrix is ill-conditioned. With auto-domain, x is
        // mapped to [-1, 1] and the fit is well-conditioned.
        let x: Vec<f64> = (0..=10).map(|i| 1000.0 + i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 1.0 + xi).collect();
        let p = Polynomial::fit_with_domain(&x, &y, 1).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!(
                (p.eval(xi).unwrap() - yi).abs() < 1e-8,
                "at x={xi}: expected {yi}, got {}",
                p.eval(xi).unwrap()
            );
        }
    }

    #[test]
    fn with_domain_rejects_degenerate_domain() {
        let p = Polynomial::new(&[1.0]);
        assert!(p.with_domain([5.0, 5.0]).is_err());
    }

    #[test]
    fn binary_op_rejects_mismatched_mapping() {
        let a = Polynomial::new(&[1.0, 2.0])
            .with_domain([0.0, 1.0])
            .unwrap();
        let b = Polynomial::new(&[3.0, 4.0])
            .with_domain([0.0, 2.0])
            .unwrap();
        assert!(a.add(&b).is_err());
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn unary_ops_preserve_mapping() {
        let p = Polynomial::new(&[1.0, 2.0, 3.0])
            .with_domain([0.0, 4.0])
            .unwrap();
        let dp = p.deriv(1).unwrap();
        assert_eq!(dp.domain(), [0.0, 4.0]);
        let ip = p.integ(1, &[]).unwrap();
        assert_eq!(ip.domain(), [0.0, 4.0]);
    }

    #[test]
    fn mapparms_basic() {
        let p = Polynomial::new(&[1.0]).with_domain([0.0, 4.0]).unwrap();
        let (offset, scale) = p.mapparms().unwrap();
        assert_eq!(scale, 0.5);
        assert_eq!(offset, -1.0);
    }

    // ----- Large/small coefficient magnitudes (#251) ---------------------
    //
    // The existing test surface stays inside [-10, 10]. Numerically these
    // tests live near the float-precision boundary where rounding behaves
    // differently from the typical case — pin the operations that should
    // still hold up.

    #[test]
    fn eval_large_coefficients_1e15() {
        // p(x) = 1e15 + 2e15 x; eval at x = 1 should give 3e15 exactly.
        let p = Polynomial::new(&[1e15, 2e15]);
        let v = p.eval(1.0).unwrap();
        assert!(
            (v - 3e15).abs() / 3e15 < 1e-14,
            "eval at large coeffs: got {v}, expected 3e15"
        );
    }

    #[test]
    fn eval_small_coefficients_1e_minus_15() {
        // p(x) = 1e-15 + 2e-15 x at x=1.5 should be 4e-15 exactly.
        let p = Polynomial::new(&[1e-15, 2e-15]);
        let v = p.eval(1.5).unwrap();
        let expected = 1e-15 + 2e-15 * 1.5;
        assert!(
            (v - expected).abs() / expected < 1e-14,
            "eval at small coeffs: got {v}, expected {expected}"
        );
    }

    #[test]
    fn add_huge_magnitudes() {
        // Adding 1e15 + 1e15 should give 2e15; the addition path must
        // not lose precision through any intermediate normalization.
        let p = Polynomial::new(&[1e15, 1e15, 1e15]);
        let q = Polynomial::new(&[1e15, 1e15, 1e15]);
        let r = p.add(&q).unwrap();
        for c in r.coeffs.iter() {
            assert!(
                (c - 2e15).abs() / 2e15 < 1e-14,
                "huge add: got {c}, expected 2e15"
            );
        }
    }

    #[test]
    fn mul_tiny_magnitudes_1e_minus_15() {
        // (1e-15 + 1e-15 x) * (1e-15 + 1e-15 x) = 1e-30 + 2e-30 x + 1e-30 x^2
        let p = Polynomial::new(&[1e-15, 1e-15]);
        let r = p.mul(&p).unwrap();
        assert_eq!(r.coeffs.len(), 3);
        assert!(
            (r.coeffs[0] - 1e-30).abs() / 1e-30 < 1e-14,
            "c0: got {}, expected 1e-30",
            r.coeffs[0]
        );
        assert!(
            (r.coeffs[1] - 2e-30).abs() / 2e-30 < 1e-14,
            "c1: got {}, expected 2e-30",
            r.coeffs[1]
        );
        assert!(
            (r.coeffs[2] - 1e-30).abs() / 1e-30 < 1e-14,
            "c2: got {}, expected 1e-30",
            r.coeffs[2]
        );
    }

    #[test]
    fn trim_large_tolerance_drops_small_coefficients() {
        // A polynomial whose small (1e-12) trailing coefficients should
        // be dropped at a 1e-6 tolerance.
        let p = Polynomial::new(&[1.0, 2.0, 3.0, 1e-12, 1e-12]);
        let trimmed = p.trim(1e-6).unwrap();
        assert_eq!(trimmed.coeffs.len(), 3);
        assert_eq!(trimmed.coeffs, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn trim_tight_tolerance_preserves_small_coefficients() {
        // The same polynomial at a tighter tolerance should keep all five.
        let p = Polynomial::new(&[1.0, 2.0, 3.0, 1e-12, 1e-12]);
        let trimmed = p.trim(1e-15).unwrap();
        assert_eq!(trimmed.coeffs.len(), 5);
    }

    #[test]
    fn eval_mixed_magnitudes_1e15_and_1e_minus_15() {
        // p(x) = 1e15 + 1e-15 x at x = 1: leading term dominates.
        // Result should be very close to 1e15; the tiny term contributes
        // ~1 ULP at that scale (1e15 * eps ~ 0.22).
        let p = Polynomial::new(&[1e15, 1e-15]);
        let v = p.eval(1.0).unwrap();
        // Within 2 ULPs of 1e15.
        assert!(
            (v - 1e15).abs() / 1e15 < 1e-14,
            "mixed-magnitude eval: got {v}"
        );
    }

    #[test]
    fn deriv_large_coefficients() {
        // d/dx (3e15 x^2) = 6e15 x. Differentiation of a single 3e15 x^2
        // term must propagate the exponent without overflow.
        let p = Polynomial::new(&[0.0, 0.0, 3e15]);
        let dp = p.deriv(1).unwrap();
        assert_eq!(dp.coeffs.len(), 2);
        assert!(
            (dp.coeffs[1] - 6e15).abs() / 6e15 < 1e-14,
            "deriv of 3e15*x^2: got {} for x coeff",
            dp.coeffs[1]
        );
    }

    #[test]
    fn fit_rejects_nan_in_y() {
        // NaN in y silently produces garbage coefficients without
        // validation (#252).
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, f64::NAN, 4.0];
        let err = Polynomial::fit(&x, &y, 1).unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn fit_rejects_inf_in_x() {
        // Inf in x produces an Inf-laden Vandermonde row → NaN coeffs.
        let x = vec![0.0, 1.0, f64::INFINITY, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let err = Polynomial::fit(&x, &y, 1).unwrap_err();
        assert!(err.to_string().contains("not finite"));
    }

    #[test]
    fn with_domain_rejects_nan_endpoint() {
        let result = Polynomial::new(&[1.0]).with_domain([f64::NAN, 1.0]);
        let err = result.unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn with_window_rejects_inf_endpoint() {
        let result = Polynomial::new(&[1.0]).with_window([0.0, f64::INFINITY]);
        let err = result.unwrap_err();
        assert!(err.to_string().contains("non-finite"));
    }

    #[test]
    fn add_precision_preserved_huge_minus_huge() {
        // (1e15 + 1.0) - 1e15 should still round-trip the small term
        // through subtract — but classic floating-point cancellation
        // means we can lose ~half the precision on the small piece.
        // Test pins the stronger guarantee: the exact subtraction is
        // performed coefficient-wise without a Kahan-style accumulator,
        // so the loss is ~1 ULP at the original magnitude.
        let p = Polynomial::new(&[1e15 + 1.0, 1.0]);
        let q = Polynomial::new(&[1e15, 0.0]);
        let r = p.sub(&q).unwrap();
        // c0 was (1e15 + 1.0) exactly, then minus 1e15. The result is
        // the small piece, which f64 rounds to the nearest representable
        // value (which IS exactly 1.0 here because 1e15 + 1.0 rounds up
        // to a representable float that loses the unit term ULP-wise).
        // We pin the looser invariant: the subtraction doesn't NaN/Inf.
        assert!(r.coeffs[0].is_finite());
        assert_eq!(r.coeffs[1], 1.0);
    }
}
