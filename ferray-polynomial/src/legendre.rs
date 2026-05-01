// ferray-polynomial: Legendre basis polynomial (REQ-3)
//
// Legendre polynomials P_n(x).
// Recurrence: P_0(x) = 1, P_1(x) = x,
//   (n+1)*P_{n+1}(x) = (2n+1)*x*P_n(x) - n*P_{n-1}(x)

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{least_squares_fit, legendre_vandermonde};
use crate::mapping::{auto_domain, map_x, mapparms, validate_domain_window};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// Default domain and window for the Legendre basis. `NumPy` uses `[-1, 1]`.
const LEGENDRE_DEFAULT_DOMAIN: [f64; 2] = [-1.0, 1.0];
const LEGENDRE_DEFAULT_WINDOW: [f64; 2] = [-1.0, 1.0];

/// A polynomial in the Legendre basis.
///
/// Represents p(x) = c[0]*`P_0(u)` + c[1]*`P_1(u)` + ... + c[n]*`P_n(u)` where
/// `u = offset + scale * x` is the affine map from `domain` to `window`.
/// By default the mapping is identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Legendre {
    /// Coefficients in the Legendre basis.
    coeffs: Vec<f64>,
    /// Input domain `[a, b]`. Defaults to `[-1, 1]`.
    domain: [f64; 2],
    /// Canonical window `[c, d]`. Defaults to `[-1, 1]`.
    window: [f64; 2],
}

impl Legendre {
    /// Create a new Legendre polynomial from coefficients. Defaults to
    /// identity mapping (`domain = window = [-1, 1]`).
    #[must_use]
    pub fn new(coeffs: &[f64]) -> Self {
        let coeffs = if coeffs.is_empty() {
            vec![0.0]
        } else {
            coeffs.to_vec()
        };
        Self {
            coeffs,
            domain: LEGENDRE_DEFAULT_DOMAIN,
            window: LEGENDRE_DEFAULT_WINDOW,
        }
    }

    /// Set the input domain, returning a new polynomial.
    pub fn with_domain(mut self, domain: [f64; 2]) -> Result<Self, FerrayError> {
        validate_domain_window(domain, self.window)?;
        self.domain = domain;
        Ok(self)
    }

    /// Set the canonical window, returning a new polynomial.
    pub fn with_window(mut self, window: [f64; 2]) -> Result<Self, FerrayError> {
        validate_domain_window(self.domain, window)?;
        self.window = window;
        Ok(self)
    }

    /// Auto-domain fit: like [`Legendre::fit`] but computes `domain` from
    /// the data range and fits in the canonical window.
    pub fn fit_with_domain(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrayError> {
        if x.len() != y.len() {
            return Err(FerrayError::invalid_value(
                "x and y must have the same length",
            ));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value("x and y must not be empty"));
        }
        let domain = auto_domain(x);
        let window = LEGENDRE_DEFAULT_WINDOW;
        let (offset, scale) = mapparms(domain, window)?;
        let u: Vec<f64> = x.iter().map(|&xi| map_x(xi, offset, scale)).collect();
        let v = legendre_vandermonde(&u, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, None)?;
        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }

    /// Internal: build a new Legendre with the same mapping as self.
    #[inline]
    const fn with_same_mapping(&self, coeffs: Vec<f64>) -> Self {
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Internal: verify two Legendre polynomials share the same mapping.
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
}

/// Evaluate Legendre series at x using Clenshaw's algorithm.
fn clenshaw_legendre(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }
    if n == 2 {
        return coeffs[1].mul_add(x, coeffs[0]);
    }

    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;

    for k in (1..n).rev() {
        let kf = k as f64;
        let alpha = 2.0f64.mul_add(kf, 1.0) / (kf + 1.0) * x;
        let beta = -(kf + 1.0) / (kf + 2.0);
        let b_k = coeffs[k] + alpha * b_k1 + beta * b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    x.mul_add(b_k1, coeffs[0]) - b_k2 / 2.0
}

/// Convert Legendre coefficients to power basis coefficients.
fn legendre_to_power(leg_coeffs: &[f64]) -> Vec<f64> {
    let n = leg_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    let mut power = vec![0.0; n];

    // Build P_k as power-basis polynomials
    let mut p_prev = vec![0.0; n];
    let mut p_curr = vec![0.0; n];
    p_prev[0] = 1.0; // P_0 = 1

    // Add c[0] * P_0
    power[0] += leg_coeffs[0];

    if n == 1 {
        return power;
    }

    p_curr[1] = 1.0; // P_1 = x
    for i in 0..n {
        power[i] += leg_coeffs[1] * p_curr[i];
    }

    for (k, &ck) in leg_coeffs.iter().enumerate().take(n).skip(2) {
        let kf = k as f64;
        let mut p_next = vec![0.0; n];
        // P_k = ((2k-1)*x*P_{k-1} - (k-1)*P_{k-2}) / k
        for i in 0..(n - 1) {
            p_next[i + 1] += 2.0f64.mul_add(kf, -1.0) * p_curr[i] / kf;
        }
        for i in 0..n {
            p_next[i] -= (kf - 1.0) * p_prev[i] / kf;
        }
        for i in 0..n {
            power[i] += ck * p_next[i];
        }
        p_prev = p_curr;
        p_curr = p_next;
    }

    power
}

/// Convert power basis coefficients to Legendre coefficients.
fn power_to_legendre(power_coeffs: &[f64]) -> Vec<f64> {
    let n = power_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    // Build x^k in terms of Legendre polynomials.
    // x^0 = P_0
    // x^1 = P_1
    // x * P_j = ((j+1)*P_{j+1} + j*P_{j-1}) / (2j+1)
    let mut leg = vec![0.0; n];
    let mut x_pow = vec![0.0; n]; // x^k in Legendre basis

    x_pow[0] = 1.0;
    leg[0] += power_coeffs[0];

    if n == 1 {
        return leg;
    }

    x_pow = vec![0.0; n];
    x_pow[1] = 1.0;
    for (i, &c) in x_pow.iter().enumerate() {
        leg[i] += power_coeffs[1] * c;
    }

    for &pk in &power_coeffs[2..n] {
        // x^k = x * x^{k-1}
        let mut x_pow_next = vec![0.0; n];
        for j in 0..n {
            if x_pow[j].abs() < f64::EPSILON * 1e-100 {
                continue;
            }
            let jf = j as f64;
            if j == 0 {
                // x * P_0 = P_1
                if 1 < n {
                    x_pow_next[1] += x_pow[j];
                }
            } else {
                // x * P_j = ((j+1)*P_{j+1} + j*P_{j-1}) / (2j+1)
                if j + 1 < n {
                    x_pow_next[j + 1] += x_pow[j] * (jf + 1.0) / 2.0f64.mul_add(jf, 1.0);
                }
                x_pow_next[j - 1] += x_pow[j] * jf / 2.0f64.mul_add(jf, 1.0);
            }
        }

        for (i, &c) in x_pow_next.iter().enumerate() {
            leg[i] += pk * c;
        }

        x_pow = x_pow_next;
    }

    leg
}

/// Multiply two Legendre series via power basis conversion.
fn mul_legendre(a: &[f64], b: &[f64]) -> Vec<f64> {
    let a_power = legendre_to_power(a);
    let b_power = legendre_to_power(b);

    let n = a_power.len() + b_power.len() - 1;
    let mut product = vec![0.0; n];
    for (i, &ai) in a_power.iter().enumerate() {
        for (j, &bj) in b_power.iter().enumerate() {
            product[i + j] += ai * bj;
        }
    }

    power_to_legendre(&product)
}

impl Poly for Legendre {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        let (offset, scale) = self.mapparms()?;
        let u = map_x(x, offset, scale);
        Ok(clenshaw_legendre(&self.coeffs, u))
    }

    fn deriv(&self, m: usize) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        // Direct Legendre derivative recurrence (#131): the identity
        //   P'_n(x) = (2n-1)·P_{n-1}(x) + (2n-5)·P_{n-3}(x) + ...
        // gives a closed form for the derivative coefficients without
        // roundtripping through the power basis. For c[k] coefficients
        // of P_k:
        //   d[k] = (2k+1) · Σ c[j] over odd j-k with j > k
        // Apply m times.
        let mut coeffs = self.coeffs.clone();
        for _ in 0..m {
            if coeffs.len() <= 1 {
                coeffs = vec![0.0];
                break;
            }
            let n = coeffs.len();
            let mut new_coeffs = vec![0.0; n - 1];
            // Walk from highest order down so each d[k] accumulates
            // from already-completed higher terms.
            for k in (0..n - 1).rev() {
                let mut acc = 0.0;
                let mut j = k + 1;
                while j < n {
                    acc += coeffs[j];
                    j += 2;
                }
                new_coeffs[k] = (2.0 * k as f64 + 1.0) * acc;
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
        // Direct Legendre integral recurrence (#131): the identity
        //   ∫ P_n(x) dx = (P_{n+1}(x) - P_{n-1}(x)) / (2n+1)
        // gives a sparse two-term update in coefficient space:
        //   for each c[j], add c[j]/(2j+1) to int_coef[j+1] and
        //                  subtract c[j]/(2j+1) from int_coef[j-1]
        //                  (j == 0 only adds to int_coef[1]).
        // The integration constant `constant` lands in int_coef[0].
        let mut coeffs = self.coeffs.clone();
        for step in 0..m {
            let constant = if step < k.len() { k[step] } else { 0.0 };
            let n = coeffs.len();
            let mut new_coeffs = vec![0.0; n + 1];
            new_coeffs[0] = constant;
            for j in 0..n {
                let denom = 2.0 * j as f64 + 1.0;
                let term = coeffs[j] / denom;
                new_coeffs[j + 1] += term;
                if j >= 1 {
                    new_coeffs[j - 1] -= term;
                }
            }
            coeffs = new_coeffs;
        }
        Ok(self.with_same_mapping(coeffs))
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        let power = legendre_to_power(&self.coeffs);
        find_roots_from_power_coeffs(&power)
    }

    fn degree(&self) -> usize {
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
        Ok(self.with_same_mapping(mul_legendre(&self.coeffs, &other.coeffs)))
    }

    fn pow(&self, n: usize) -> Result<Self, FerrayError> {
        if n == 0 {
            return Ok(self.with_same_mapping(vec![1.0]));
        }
        let mut result = self.clone();
        for _ in 1..n {
            result = result.mul(self)?;
        }
        Ok(result)
    }

    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrayError> {
        self.check_same_mapping(other)?;
        let a_power = legendre_to_power(&self.coeffs);
        let b_power = legendre_to_power(&other.coeffs);

        let a_poly = crate::power::Polynomial::new(&a_power);
        let b_poly = crate::power::Polynomial::new(&b_power);
        let (q_poly, r_poly) = a_poly.divmod(&b_poly)?;

        Ok((
            self.with_same_mapping(power_to_legendre(q_poly.coeffs())),
            self.with_same_mapping(power_to_legendre(r_poly.coeffs())),
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
        let v = legendre_vandermonde(x, deg);
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
        let v = legendre_vandermonde(x, deg);
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

impl ToPowerBasis for Legendre {
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError> {
        Ok(legendre_to_power(&self.coeffs))
    }
}

impl FromPowerBasis for Legendre {
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError> {
        Ok(Self::new(&power_to_legendre(coeffs)))
    }
}

impl From<crate::power::Polynomial> for Legendre {
    fn from(p: crate::power::Polynomial) -> Self {
        Self::new(&power_to_legendre(p.coeffs()))
    }
}

impl From<Legendre> for crate::power::Polynomial {
    fn from(l: Legendre) -> Self {
        Self::new(&legendre_to_power(&l.coeffs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_p0() {
        let p = Legendre::new(&[1.0]);
        assert!((p.eval(0.5).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn eval_p1() {
        let p = Legendre::new(&[0.0, 1.0]);
        assert!((p.eval(0.5).unwrap() - 0.5).abs() < 1e-14);
    }

    #[test]
    fn eval_p2() {
        // P_2(x) = (3x^2 - 1)/2
        let p = Legendre::new(&[0.0, 0.0, 1.0]);
        let x = 0.5;
        let expected = (3.0 * x * x - 1.0) / 2.0;
        assert!((p.eval(x).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn legendre_roundtrip() {
        let original = vec![1.0, 2.0, 3.0];
        let power = legendre_to_power(&original);
        let recovered = power_to_legendre(&power);

        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!((orig - rec).abs() < 1e-12, "index {i}: {orig} != {rec}");
        }
    }

    #[test]
    fn fit_legendre() {
        let x: Vec<f64> = (0..20).map(|i| -1.0 + 2.0 * i as f64 / 19.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();

        let leg = Legendre::fit(&x, &y, 5).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let eval = leg.eval(xi).unwrap();
            assert!(
                (eval - yi).abs() < 1e-8,
                "at x={xi}: expected {yi}, got {eval}"
            );
        }
    }

    #[test]
    fn integ_then_deriv() {
        let p = Legendre::new(&[1.0, 2.0, 3.0]);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let recovered = integrated.deriv(1).unwrap();

        for i in 0..p.coeffs.len() {
            let expected = p.coeffs[i];
            let got = if i < recovered.coeffs.len() {
                recovered.coeffs[i]
            } else {
                0.0
            };
            assert!(
                (expected - got).abs() < 1e-10,
                "index {i}: expected {expected}, got {got}"
            );
        }
    }
}
