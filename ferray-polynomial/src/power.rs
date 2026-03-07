// ferray-polynomial: Power basis polynomial (REQ-1)
//
// p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{least_squares_fit, power_vandermonde};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// A polynomial in the standard power (monomial) basis.
///
/// Represents p(x) = c[0] + c[1]*x + c[2]*x^2 + ... + c[n]*x^n
/// where `coeffs[i]` is the coefficient of x^i.
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial {
    /// Coefficients in ascending power order: c[0], c[1], ..., c[n].
    coeffs: Vec<f64>,
}

impl Polynomial {
    /// Create a new power-basis polynomial from coefficients.
    ///
    /// Coefficients are in ascending order: `coeffs[i]` is the coefficient of x^i.
    /// An empty coefficient slice produces the zero polynomial `[0.0]`.
    pub fn new(coeffs: &[f64]) -> Self {
        if coeffs.is_empty() {
            Self { coeffs: vec![0.0] }
        } else {
            Self {
                coeffs: coeffs.to_vec(),
            }
        }
    }
}

impl Poly for Polynomial {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        // Horner's method
        let mut result = 0.0;
        for &c in self.coeffs.iter().rev() {
            result = result * x + c;
        }
        Ok(result)
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
        Ok(Self { coeffs })
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
        Ok(Self { coeffs })
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        find_roots_from_power_coeffs(&self.coeffs)
    }

    fn degree(&self) -> usize {
        // Trim trailing zeros to find actual degree
        let mut deg = self.coeffs.len().saturating_sub(1);
        while deg > 0 && self.coeffs[deg].abs() < f64::EPSILON * 100.0 {
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
        Ok(Self {
            coeffs: self.coeffs[..last].to_vec(),
        })
    }

    fn truncate(&self, size: usize) -> Result<Self, FerrayError> {
        if size == 0 {
            return Err(FerrayError::invalid_value(
                "truncation size must be at least 1",
            ));
        }
        let len = size.min(self.coeffs.len());
        Ok(Self {
            coeffs: self.coeffs[..len].to_vec(),
        })
    }

    fn add(&self, other: &Self) -> Result<Self, FerrayError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            result[i] += c;
        }
        Ok(Self { coeffs: result })
    }

    fn sub(&self, other: &Self) -> Result<Self, FerrayError> {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut result = vec![0.0; len];
        for (i, &c) in self.coeffs.iter().enumerate() {
            result[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            result[i] -= c;
        }
        Ok(Self { coeffs: result })
    }

    fn mul(&self, other: &Self) -> Result<Self, FerrayError> {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Ok(Self { coeffs: vec![0.0] });
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut result = vec![0.0; len];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                result[i + j] += a * b;
            }
        }
        Ok(Self { coeffs: result })
    }

    fn pow(&self, n: usize) -> Result<Self, FerrayError> {
        if n == 0 {
            return Ok(Self { coeffs: vec![1.0] });
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = result.mul(self)?;
        }
        Ok(result)
    }

    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrayError> {
        let other_trimmed = other.trim(0.0)?;
        if other_trimmed.degree() == 0 && other_trimmed.coeffs[0].abs() < f64::EPSILON * 100.0 {
            return Err(FerrayError::invalid_value("division by zero polynomial"));
        }

        let self_trimmed = self.trim(0.0)?;
        let n = self_trimmed.coeffs.len();
        let m = other_trimmed.coeffs.len();

        if n < m {
            // Quotient is zero, remainder is self
            return Ok((Self { coeffs: vec![0.0] }, self_trimmed));
        }

        let mut remainder = self_trimmed.coeffs.clone();
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
            Self { coeffs: quotient },
            Self {
                coeffs: remainder[..rem_len.max(1)].to_vec(),
            },
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
        Ok(Self { coeffs })
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
        Ok(Self { coeffs })
    }

    fn from_coeffs(coeffs: &[f64]) -> Self {
        Self::new(coeffs)
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
            assert!((a - b).abs() < 1e-12, "expected {}, got {}", b, a);
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
}
