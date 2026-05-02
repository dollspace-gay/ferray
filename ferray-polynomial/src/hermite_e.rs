// ferray-polynomial: HermiteE (probabilist's) basis polynomial (REQ-6)
//
// Probabilist's Hermite polynomials He_n(x).
// Recurrence: He_0(x) = 1, He_1(x) = x,
//   He_{n+1}(x) = x*He_n(x) - n*He_{n-1}(x)

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{hermite_e_vandermonde, least_squares_fit};
use crate::mapping::{auto_domain, map_x, mapparms, validate_domain_window};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// Default domain and window for the `HermiteE` basis. `NumPy` uses `[-1, 1]`.
const HERMITE_E_DEFAULT_DOMAIN: [f64; 2] = [-1.0, 1.0];
const HERMITE_E_DEFAULT_WINDOW: [f64; 2] = [-1.0, 1.0];

/// A polynomial in the probabilist's Hermite basis.
///
/// Represents p(x) = c[0]*`He_0(u)` + c[1]*`He_1(u)` + ... + c[n]*`He_n(u)`
/// where `u = offset + scale * x` is the affine map from `domain` to `window`.
/// By default the mapping is identity.
#[derive(Debug, Clone, PartialEq)]
pub struct HermiteE {
    /// Coefficients in the probabilist's Hermite basis.
    coeffs: Vec<f64>,
    /// Input domain `[a, b]`. Defaults to `[-1, 1]`.
    domain: [f64; 2],
    /// Canonical window `[c, d]`. Defaults to `[-1, 1]`.
    window: [f64; 2],
}

impl HermiteE {
    /// Create a new `HermiteE` polynomial from coefficients. Defaults to
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
            domain: HERMITE_E_DEFAULT_DOMAIN,
            window: HERMITE_E_DEFAULT_WINDOW,
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

    /// Auto-domain fit: like [`HermiteE::fit`] but computes `domain` from
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
        let window = HERMITE_E_DEFAULT_WINDOW;
        let (offset, scale) = mapparms(domain, window)?;
        let u: Vec<f64> = x.iter().map(|&xi| map_x(xi, offset, scale)).collect();
        let v = hermite_e_vandermonde(&u, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, None)?;
        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }

    /// Internal: build a new `HermiteE` with the same mapping as self.
    #[inline]
    pub(crate) const fn with_same_mapping(&self, coeffs: Vec<f64>) -> Self {
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Internal: verify two `HermiteE` polynomials share the same mapping.
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

/// Evaluate probabilist's Hermite series at x using Clenshaw's algorithm.
fn clenshaw_hermite_e(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }

    // HermiteE recurrence: He_n = x * He_{n-1} - (n-1) * He_{n-2}
    // Standard Clenshaw form P_n = α_n P_{n-1} + β_n P_{n-2} gives
    //   α_n(x) = x,  β_n = -(n-1)
    // Backward iteration: b_k = c_k + α_{k+1}(x) b_{k+1} + β_{k+2} b_{k+2}
    //                         = c_k + x * b_{k+1} - (k + 1) * b_{k+2}
    // (the previous code used `-k * b_{k+2}`, which silently produced wrong
    // values once `n >= 4`).
    let mut b_k1 = 0.0;
    let mut b_k2 = 0.0;

    for k in (1..n).rev() {
        let b_k = ((k + 1) as f64).mul_add(-b_k2, x.mul_add(b_k1, coeffs[k]));
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    x.mul_add(b_k1, coeffs[0]) - b_k2
}

/// Convert `HermiteE` coefficients to power basis coefficients.
fn hermite_e_to_power(he_coeffs: &[f64]) -> Vec<f64> {
    let n = he_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    let size = n;
    let mut power = vec![0.0; size];

    let mut he_prev = vec![0.0; size];
    let mut he_curr = vec![0.0; size];
    he_prev[0] = 1.0; // He_0 = 1

    power[0] += he_coeffs[0];

    if n == 1 {
        return power;
    }

    he_curr[1] = 1.0; // He_1 = x
    for i in 0..size {
        power[i] += he_coeffs[1] * he_curr[i];
    }

    for (k, &hec) in he_coeffs.iter().enumerate().take(n).skip(2) {
        let kf = k as f64;
        let mut he_next = vec![0.0; size];
        // He_k = x*He_{k-1} - (k-1)*He_{k-2}
        for i in 0..(size - 1) {
            he_next[i + 1] += he_curr[i];
        }
        for i in 0..size {
            he_next[i] -= (kf - 1.0) * he_prev[i];
        }
        for i in 0..size {
            power[i] += hec * he_next[i];
        }
        he_prev = he_curr;
        he_curr = he_next;
    }

    power
}

/// Convert power basis coefficients to `HermiteE` coefficients.
fn power_to_hermite_e(power_coeffs: &[f64]) -> Vec<f64> {
    let n = power_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    // x * He_j = He_{j+1} + j*He_{j-1}
    let mut he = vec![0.0; n];
    let mut x_pow = vec![0.0; n]; // x^k in HermiteE basis

    x_pow[0] = 1.0; // x^0 = He_0
    he[0] += power_coeffs[0];

    if n == 1 {
        return he;
    }

    // x^1 = He_1
    x_pow = vec![0.0; n];
    x_pow[1] = 1.0;
    for (i, &c) in x_pow.iter().enumerate() {
        he[i] += power_coeffs[1] * c;
    }

    for &pc in &power_coeffs[2..n] {
        // x^k = x * x^{k-1}
        let mut x_pow_next = vec![0.0; n];
        for j in 0..n {
            if x_pow[j].abs() < f64::EPSILON * 1e-100 {
                continue;
            }
            let jf = j as f64;
            // x * He_j = He_{j+1} + j*He_{j-1}
            if j + 1 < n {
                x_pow_next[j + 1] += x_pow[j];
            }
            if j >= 1 {
                x_pow_next[j - 1] += x_pow[j] * jf;
            }
        }

        for (i, &c) in x_pow_next.iter().enumerate() {
            he[i] += pc * c;
        }

        x_pow = x_pow_next;
    }

    he
}

impl Poly for HermiteE {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        let (offset, scale) = self.mapparms()?;
        let u = map_x(x, offset, scale);
        Ok(clenshaw_hermite_e(&self.coeffs, u))
    }

    fn deriv(&self, m: usize) -> Result<Self, FerrayError> {
        if m == 0 {
            return Ok(self.clone());
        }
        // Direct recurrence (#720): d/dx He_n(x) = n * He_{n-1}(x).
        // For p = Σ c_k He_k, p' = Σ_{k≥1} k c_k He_{k-1}, so
        // new[j] = (j+1) * c[j+1].
        let mut coeffs = self.coeffs.clone();
        for _ in 0..m {
            if coeffs.len() <= 1 {
                coeffs = vec![0.0];
                break;
            }
            let mut new_coeffs = Vec::with_capacity(coeffs.len() - 1);
            for (j, &c) in coeffs.iter().enumerate().skip(1) {
                new_coeffs.push(j as f64 * c);
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
        // Direct recurrence (#720): ∫He_n(x) dx = He_{n+1}(x) / (n+1).
        // new[k+1] = c[k] / (k+1), with the integration constant on
        // index 0.
        let mut coeffs = self.coeffs.clone();
        for step in 0..m {
            let constant = if step < k.len() { k[step] } else { 0.0 };
            let mut new_coeffs = vec![0.0_f64; coeffs.len() + 1];
            new_coeffs[0] = constant;
            for (j, &c) in coeffs.iter().enumerate() {
                new_coeffs[j + 1] = c / (j + 1) as f64;
            }
            coeffs = new_coeffs;
        }
        Ok(self.with_same_mapping(coeffs))
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        let power = hermite_e_to_power(&self.coeffs);
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
        let a_power = hermite_e_to_power(&self.coeffs);
        let b_power = hermite_e_to_power(&other.coeffs);

        let n = a_power.len() + b_power.len() - 1;
        let mut product = vec![0.0; n];
        for (i, &ai) in a_power.iter().enumerate() {
            for (j, &bj) in b_power.iter().enumerate() {
                product[i + j] += ai * bj;
            }
        }

        Ok(self.with_same_mapping(power_to_hermite_e(&product)))
    }

    fn pow(&self, n: usize) -> Result<Self, FerrayError> {
        // Binary exponentiation (#247) — O(log n) polynomial multiplies.
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
        Ok(result.expect("binary exponentiation must produce a result for n >= 1"))
    }

    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrayError> {
        self.check_same_mapping(other)?;
        let a_power = hermite_e_to_power(&self.coeffs);
        let b_power = hermite_e_to_power(&other.coeffs);

        let a_poly = crate::power::Polynomial::new(&a_power);
        let b_poly = crate::power::Polynomial::new(&b_power);
        let (q_poly, r_poly) = a_poly.divmod(&b_poly)?;

        Ok((
            self.with_same_mapping(power_to_hermite_e(q_poly.coeffs())),
            self.with_same_mapping(power_to_hermite_e(r_poly.coeffs())),
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
        let v = hermite_e_vandermonde(x, deg);
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
        let v = hermite_e_vandermonde(x, deg);
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

    fn with_mapping(self, domain: [f64; 2], window: [f64; 2]) -> Result<Self, FerrayError> {
        self.with_domain(domain)?.with_window(window)
    }
}

impl ToPowerBasis for HermiteE {
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError> {
        Ok(hermite_e_to_power(&self.coeffs))
    }
}

impl FromPowerBasis for HermiteE {
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError> {
        Ok(Self::new(&power_to_hermite_e(coeffs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eval_he0() {
        let p = HermiteE::new(&[1.0]);
        assert!((p.eval(0.5).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn eval_he1() {
        // He_1(x) = x
        let p = HermiteE::new(&[0.0, 1.0]);
        assert!((p.eval(0.5).unwrap() - 0.5).abs() < 1e-14);
    }

    #[test]
    fn eval_he2() {
        // He_2(x) = x^2 - 1
        let p = HermiteE::new(&[0.0, 0.0, 1.0]);
        let x = 0.5;
        let expected = x * x - 1.0;
        assert!((p.eval(x).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn hermite_e_roundtrip() {
        let original = vec![1.0, 2.0, 3.0];
        let power = hermite_e_to_power(&original);
        let recovered = power_to_hermite_e(&power);

        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!((orig - rec).abs() < 1e-10, "index {i}: {orig} != {rec}");
        }
    }

    #[test]
    fn integ_then_deriv() {
        let p = HermiteE::new(&[1.0, 2.0, 3.0]);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let recovered = integrated.deriv(1).unwrap();

        for x in [0.0, 0.5, 1.0, 2.0] {
            let expected = p.eval(x).unwrap();
            let got = recovered.eval(x).unwrap();
            assert!(
                (expected - got).abs() < 1e-6,
                "at x={x}: expected {expected}, got {got}"
            );
        }
    }
}
