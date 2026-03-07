// ferray-polynomial: Laguerre basis polynomial (REQ-4)
//
// Laguerre polynomials L_n(x).
// Recurrence: L_0(x) = 1, L_1(x) = 1-x,
//   (n+1)*L_{n+1}(x) = (2n+1-x)*L_n(x) - n*L_{n-1}(x)

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{laguerre_vandermonde, least_squares_fit};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// A polynomial in the Laguerre basis.
///
/// Represents p(x) = c[0]*L_0(x) + c[1]*L_1(x) + ... + c[n]*L_n(x)
/// where L_k are the Laguerre polynomials.
#[derive(Debug, Clone, PartialEq)]
pub struct Laguerre {
    /// Coefficients in the Laguerre basis.
    coeffs: Vec<f64>,
}

impl Laguerre {
    /// Create a new Laguerre polynomial from coefficients.
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

/// Evaluate Laguerre series at x using Clenshaw's algorithm.
fn clenshaw_laguerre(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }
    if n == 2 {
        return coeffs[0] + coeffs[1] * (1.0 - x);
    }

    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;

    for k in (1..n).rev() {
        let kf = k as f64;
        let alpha = (2.0 * kf + 1.0 - x) / (kf + 1.0);
        let beta = -(kf + 1.0) / (kf + 2.0);
        let b_k = coeffs[k] + alpha * b_k1 + beta * b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    coeffs[0] + (1.0 - x) * b_k1 - b_k2 / 2.0
}

/// Convert Laguerre coefficients to power basis coefficients.
fn laguerre_to_power(lag_coeffs: &[f64]) -> Vec<f64> {
    let n = lag_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    let mut power = vec![0.0; n];

    let mut l_prev = vec![0.0; n];
    let mut l_curr = vec![0.0; n];
    l_prev[0] = 1.0; // L_0 = 1

    power[0] += lag_coeffs[0];

    if n == 1 {
        return power;
    }

    l_curr[0] = 1.0;
    l_curr[1] = -1.0; // L_1 = 1 - x
    for i in 0..n {
        power[i] += lag_coeffs[1] * l_curr[i];
    }

    for (k, &ck) in lag_coeffs.iter().enumerate().take(n).skip(2) {
        let kf = k as f64;
        let mut l_next = vec![0.0; n];
        // L_k = ((2k-1-x)*L_{k-1} - (k-1)*L_{k-2}) / k
        for i in 0..n {
            l_next[i] += (2.0 * kf - 1.0) * l_curr[i] / kf;
        }
        for i in 0..(n - 1) {
            l_next[i + 1] -= l_curr[i] / kf;
        }
        for i in 0..n {
            l_next[i] -= (kf - 1.0) * l_prev[i] / kf;
        }
        for i in 0..n {
            power[i] += ck * l_next[i];
        }
        l_prev = l_curr;
        l_curr = l_next;
    }

    power
}

/// Convert power basis coefficients to Laguerre coefficients.
fn power_to_laguerre(power_coeffs: &[f64]) -> Vec<f64> {
    let n = power_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    // We use the relationship between x^k and Laguerre polynomials:
    // x^k = k! * sum_{j=0}^{k} (-1)^j * C(k,j) * L_j(x) / j!
    // which simplifies to: x^k = k! * sum_{j=0}^{k} (-1)^j * C(k,j) / j! * L_j

    // More efficiently, build x^k in the Laguerre basis incrementally.
    // x * L_j = -(j+1)*L_{j+1} + (2j+1)*L_j - j*L_{j-1}
    let mut lag = vec![0.0; n];
    let mut x_pow = vec![0.0; n]; // x^k in Laguerre basis

    x_pow[0] = 1.0; // x^0 = L_0
    lag[0] += power_coeffs[0];

    if n == 1 {
        return lag;
    }

    // x^1: x = -L_1 + L_0 (since L_1 = 1-x, so x = 1 - L_1 = L_0 - L_1)
    let mut x_pow_new = vec![0.0; n];
    // x * L_0: use the formula x * L_j with j=0
    // x * L_0 = -(0+1)*L_1 + (2*0+1)*L_0 - 0*L_{-1}
    // = -L_1 + L_0
    x_pow_new[0] = 1.0; // L_0 coefficient
    x_pow_new[1] = -1.0; // L_1 coefficient
    x_pow = x_pow_new;

    for (i, &c) in x_pow.iter().enumerate() {
        lag[i] += power_coeffs[1] * c;
    }

    for &pc in &power_coeffs[2..n] {
        // x^k = x * x^{k-1}
        let mut x_pow_next = vec![0.0; n];
        for j in 0..n {
            if x_pow[j].abs() < f64::EPSILON * 1e-100 {
                continue;
            }
            let jf = j as f64;
            // x * L_j = -(j+1)*L_{j+1} + (2j+1)*L_j - j*L_{j-1}
            if j + 1 < n {
                x_pow_next[j + 1] += x_pow[j] * (-(jf + 1.0));
            }
            x_pow_next[j] += x_pow[j] * (2.0 * jf + 1.0);
            if j >= 1 {
                x_pow_next[j - 1] += x_pow[j] * (-jf);
            }
        }

        for (i, &c) in x_pow_next.iter().enumerate() {
            lag[i] += pc * c;
        }

        x_pow = x_pow_next;
    }

    lag
}

impl Poly for Laguerre {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        Ok(clenshaw_laguerre(&self.coeffs, x))
    }

    fn deriv(&self, m: usize) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        let mut power = laguerre_to_power(&self.coeffs);
        for _ in 0..m {
            if power.len() <= 1 {
                power = vec![0.0];
                break;
            }
            let mut new_power = Vec::with_capacity(power.len() - 1);
            for (i, &c) in power.iter().enumerate().skip(1) {
                new_power.push(c * i as f64);
            }
            power = new_power;
        }
        if power.is_empty() {
            power = vec![0.0];
        }
        Ok(Self {
            coeffs: power_to_laguerre(&power),
        })
    }

    fn integ(&self, m: usize, k: &[f64]) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        let mut power = laguerre_to_power(&self.coeffs);
        for step in 0..m {
            let constant = if step < k.len() { k[step] } else { 0.0 };
            let mut new_power = Vec::with_capacity(power.len() + 1);
            new_power.push(constant);
            for (i, &c) in power.iter().enumerate() {
                new_power.push(c / (i + 1) as f64);
            }
            power = new_power;
        }
        Ok(Self {
            coeffs: power_to_laguerre(&power),
        })
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        let power = laguerre_to_power(&self.coeffs);
        find_roots_from_power_coeffs(&power)
    }

    fn degree(&self) -> usize {
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
        let a_power = laguerre_to_power(&self.coeffs);
        let b_power = laguerre_to_power(&other.coeffs);

        let n = a_power.len() + b_power.len() - 1;
        let mut product = vec![0.0; n];
        for (i, &ai) in a_power.iter().enumerate() {
            for (j, &bj) in b_power.iter().enumerate() {
                product[i + j] += ai * bj;
            }
        }

        Ok(Self {
            coeffs: power_to_laguerre(&product),
        })
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
        let a_power = laguerre_to_power(&self.coeffs);
        let b_power = laguerre_to_power(&other.coeffs);

        let a_poly = crate::power::Polynomial::new(&a_power);
        let b_poly = crate::power::Polynomial::new(&b_power);
        let (q_poly, r_poly) = a_poly.divmod(&b_poly)?;

        Ok((
            Self {
                coeffs: power_to_laguerre(q_poly.coeffs()),
            },
            Self {
                coeffs: power_to_laguerre(r_poly.coeffs()),
            },
        ))
    }

    fn fit(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrayError> {
        if x.len() != y.len() {
            return Err(FerrayError::invalid_value(
                "x and y must have the same length",
            ));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value("x and y must not be empty"));
        }
        let v = laguerre_vandermonde(x, deg);
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
        let v = laguerre_vandermonde(x, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, Some(w))?;
        Ok(Self { coeffs })
    }

    fn from_coeffs(coeffs: &[f64]) -> Self {
        Self::new(coeffs)
    }
}

impl ToPowerBasis for Laguerre {
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError> {
        Ok(laguerre_to_power(&self.coeffs))
    }
}

impl FromPowerBasis for Laguerre {
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError> {
        Ok(Self {
            coeffs: power_to_laguerre(coeffs),
        })
    }
}

impl From<crate::power::Polynomial> for Laguerre {
    fn from(p: crate::power::Polynomial) -> Self {
        Self {
            coeffs: power_to_laguerre(p.coeffs()),
        }
    }
}

impl From<Laguerre> for crate::power::Polynomial {
    fn from(l: Laguerre) -> Self {
        crate::power::Polynomial::new(&laguerre_to_power(&l.coeffs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_l0() {
        let p = Laguerre::new(&[1.0]);
        assert!((p.eval(0.5).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn eval_l1() {
        let p = Laguerre::new(&[0.0, 1.0]); // L_1 = 1 - x
        assert!((p.eval(0.5).unwrap() - 0.5).abs() < 1e-14);
    }

    #[test]
    fn eval_l2() {
        // L_2(x) = (x^2 - 4x + 2)/2
        let p = Laguerre::new(&[0.0, 0.0, 1.0]);
        let x = 0.5;
        let expected = (x * x - 4.0 * x + 2.0) / 2.0;
        assert!(
            (p.eval(x).unwrap() - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            p.eval(x).unwrap()
        );
    }

    #[test]
    fn laguerre_roundtrip() {
        let original = vec![1.0, 2.0, 3.0];
        let power = laguerre_to_power(&original);
        let recovered = power_to_laguerre(&power);

        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-10,
                "index {}: {} != {}",
                i,
                orig,
                rec
            );
        }
    }

    #[test]
    fn integ_then_deriv() {
        let p = Laguerre::new(&[1.0, 2.0, 3.0]);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let recovered = integrated.deriv(1).unwrap();

        // Compare by evaluating at several points
        for x in [0.0, 0.5, 1.0, 2.0] {
            let expected = p.eval(x).unwrap();
            let got = recovered.eval(x).unwrap();
            assert!(
                (expected - got).abs() < 1e-8,
                "at x={}: expected {}, got {}",
                x,
                expected,
                got
            );
        }
    }
}
