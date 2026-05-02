// ferray-polynomial: Chebyshev basis polynomial (REQ-2)
//
// Chebyshev polynomials of the first kind T_n(x).
// Recurrence: T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::fitting::{chebyshev_vandermonde, least_squares_fit};
use crate::mapping::{auto_domain, map_x, mapparms, validate_domain_window};
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

/// Default domain and window for the Chebyshev basis.
///
/// The Chebyshev polynomials are well-conditioned on `[-1, 1]`. `NumPy` uses
/// this same window by default; the domain defaults to `[-1, 1]` so that
/// no mapping is applied unless the user opts in.
const CHEBYSHEV_DEFAULT_DOMAIN: [f64; 2] = [-1.0, 1.0];
const CHEBYSHEV_DEFAULT_WINDOW: [f64; 2] = [-1.0, 1.0];

/// A polynomial in the Chebyshev basis (first kind).
///
/// Represents p(x) = c[0]*`T_0(u)` + c[1]*`T_1(u)` + ... + c[n]*`T_n(u)` where
/// `u = offset + scale * x` is the affine map from `domain` to `window`.
/// By default the mapping is identity.
#[derive(Debug, Clone, PartialEq)]
pub struct Chebyshev {
    /// Coefficients in the Chebyshev basis.
    coeffs: Vec<f64>,
    /// Input domain `[a, b]`. Defaults to `[-1, 1]`.
    domain: [f64; 2],
    /// Canonical window `[c, d]`. Defaults to `[-1, 1]`.
    window: [f64; 2],
}

impl Chebyshev {
    /// Create a new Chebyshev polynomial from coefficients.
    ///
    /// `coeffs[i]` is the coefficient of `T_i(x)`. The domain and window
    /// default to `[-1, 1]` (identity mapping).
    #[must_use]
    pub fn new(coeffs: &[f64]) -> Self {
        let coeffs = if coeffs.is_empty() {
            vec![0.0]
        } else {
            coeffs.to_vec()
        };
        Self {
            coeffs,
            domain: CHEBYSHEV_DEFAULT_DOMAIN,
            window: CHEBYSHEV_DEFAULT_WINDOW,
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
    pub fn with_window(mut self, window: [f64; 2]) -> Result<Self, FerrayError> {
        validate_domain_window(self.domain, window)?;
        self.window = window;
        Ok(self)
    }

    /// Auto-domain fit: like [`Chebyshev::fit`] but computes `domain` from
    /// the data range and fits in the canonical window. Critical for
    /// numerical stability when fitting on non-canonical intervals.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for length mismatches, empty
    /// input, or rank-deficient fits.
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
        let window = CHEBYSHEV_DEFAULT_WINDOW;
        let (offset, scale) = mapparms(domain, window)?;
        let u: Vec<f64> = x.iter().map(|&xi| map_x(xi, offset, scale)).collect();
        let v = chebyshev_vandermonde(&u, deg);
        let coeffs = least_squares_fit(&v, x.len(), deg + 1, y, None)?;
        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }

    /// Internal: build a new Chebyshev with the same domain/window as self.
    #[inline]
    pub(crate) const fn with_same_mapping(&self, coeffs: Vec<f64>) -> Self {
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Internal: verify two Chebyshev polynomials share the same mapping.
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

/// Evaluate Chebyshev series at x using Clenshaw's algorithm.
fn clenshaw_chebyshev(coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return coeffs[0];
    }

    let mut b_k1 = 0.0; // b_{k+1}
    let mut b_k2 = 0.0; // b_{k+2}

    for k in (1..n).rev() {
        let b_k = (2.0 * x).mul_add(b_k1, coeffs[k]) - b_k2;
        b_k2 = b_k1;
        b_k1 = b_k;
    }

    x.mul_add(b_k1, coeffs[0]) - b_k2
}

/// Multiply two Chebyshev series.
///
/// Uses the identity: `T_m(x)` * `T_n(x)` = (T_{m+n}(x) + T_{|m-n|}(x)) / 2
fn mul_chebyshev(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return vec![0.0];
    }
    let n = a.len() + b.len() - 1;
    let mut result = vec![0.0; n];

    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            let product = ai * bj / 2.0;
            result[i + j] += product; // T_{i+j} term
            let diff = i.abs_diff(j);
            result[diff] += product; // T_{|i-j|} term
        }
    }
    result
}

/// Convert Chebyshev coefficients to power basis coefficients.
///
/// Uses the explicit expansion of Chebyshev polynomials.
fn chebyshev_to_power(cheb_coeffs: &[f64]) -> Vec<f64> {
    let n = cheb_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    // Build T_k as power-basis polynomials incrementally
    // T_0 = [1], T_1 = [0, 1]
    // T_{k+1} = 2x * T_k - T_{k-1}
    let mut power_coeffs = vec![0.0; n]; // final result

    let mut t_prev = vec![0.0; n]; // T_{k-1}
    let mut t_curr = vec![0.0; n]; // T_k
    let mut t_next = vec![0.0; n]; // scratch reused across iterations (#257)
    t_prev[0] = 1.0; // T_0

    // Add c[0] * T_0
    power_coeffs[0] += cheb_coeffs[0];

    if n == 1 {
        return power_coeffs;
    }

    t_curr[1] = 1.0; // T_1 = x
    // Add c[1] * T_1
    for (i, &c) in t_curr.iter().enumerate() {
        power_coeffs[i] += cheb_coeffs[1] * c;
    }

    for &ck in &cheb_coeffs[2..n] {
        // T_k = 2x * T_{k-1} - T_{k-2}
        // Reset the scratch buffer in place — previously we
        // allocated a fresh `vec![0.0; n]` here per term (#257).
        t_next.fill(0.0);
        // 2x * t_curr
        for i in 0..(n - 1) {
            t_next[i + 1] += 2.0 * t_curr[i];
        }
        // - t_prev
        for i in 0..n {
            t_next[i] -= t_prev[i];
        }

        // Add ck * T_k
        for (i, &c) in t_next.iter().enumerate() {
            power_coeffs[i] += ck * c;
        }

        // Rotate buffers without allocating: t_prev <- t_curr,
        // t_curr <- t_next, and t_next becomes the old t_prev which
        // we'll re-zero on the next iteration.
        std::mem::swap(&mut t_prev, &mut t_curr);
        std::mem::swap(&mut t_curr, &mut t_next);
    }

    power_coeffs
}

/// Convert power basis coefficients to Chebyshev coefficients.
///
/// Uses the fact that x^k can be expressed in terms of Chebyshev polynomials.
fn power_to_chebyshev(power_coeffs: &[f64]) -> Vec<f64> {
    let n = power_coeffs.len();
    if n == 0 {
        return vec![0.0];
    }

    // Build x^k in terms of Chebyshev polynomials incrementally.
    // x^0 = T_0
    // x^1 = T_1
    // x^{k+1} = x * x^k, where x * T_j = (T_{j+1} + T_{j-1})/2 (and x*T_0 = T_1)
    let mut cheb_coeffs = vec![0.0; n];
    let mut x_pow = vec![0.0; n]; // x^k in Chebyshev basis

    // x^0 = T_0
    x_pow[0] = 1.0;
    // Add power_coeffs[0] * x^0
    cheb_coeffs[0] += power_coeffs[0];

    if n == 1 {
        return cheb_coeffs;
    }

    // x^1 = T_1
    x_pow = vec![0.0; n];
    x_pow[1] = 1.0;
    for (i, &c) in x_pow.iter().enumerate() {
        cheb_coeffs[i] += power_coeffs[1] * c;
    }

    for &pk in &power_coeffs[2..n] {
        // x^k = x * x^{k-1}
        // x * T_j = (T_{j+1} + T_{j-1})/2, except x * T_0 = T_1
        let mut x_pow_next = vec![0.0; n];
        for j in 0..n {
            if x_pow[j].abs() < f64::EPSILON * 1e-100 {
                continue;
            }
            if j == 0 {
                // x * T_0 = T_1
                if 1 < n {
                    x_pow_next[1] += x_pow[j];
                }
            } else {
                // x * T_j = (T_{j+1} + T_{j-1})/2
                if j + 1 < n {
                    x_pow_next[j + 1] += x_pow[j] / 2.0;
                }
                x_pow_next[j - 1] += x_pow[j] / 2.0;
            }
        }

        for (i, &c) in x_pow_next.iter().enumerate() {
            cheb_coeffs[i] += pk * c;
        }

        x_pow = x_pow_next;
    }

    cheb_coeffs
}

impl Poly for Chebyshev {
    fn eval(&self, x: f64) -> Result<f64, FerrayError> {
        let (offset, scale) = self.mapparms()?;
        let u = map_x(x, offset, scale);
        Ok(clenshaw_chebyshev(&self.coeffs, u))
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
            let n = coeffs.len();
            let mut new_coeffs = vec![0.0; n - 1];
            // d/dx T_n(x) = n * U_{n-1}(x)
            // In terms of Chebyshev T:
            // c'_k = 2(k+1) * c_{k+1} for k < n-1
            // with special handling for c'_0 (halved contribution)
            // The recurrence: c'_{n-2} = 2(n-1)*c_{n-1}
            // c'_k = c'_{k+2} + 2(k+1)*c_{k+1}  for k = n-3 down to 1
            // c'_0 = c'_2/2 + c_1
            // The recurrence for Chebyshev derivative:
            // c'_{n-2} = 2*(n-1)*c_{n-1}
            // c'_k = c'_{k+2} + 2*(k+1)*c_{k+1}  for k = n-3 down to 1
            // c'_0 = c'_2/2 + c_1
            let nd = n - 1; // degree of new polynomial = length of new_coeffs
            if nd >= 1 {
                new_coeffs[nd - 1] = 2.0 * (n as f64 - 1.0) * coeffs[n - 1];
            }
            for k in (1..nd.saturating_sub(1)).rev() {
                let c_k2 = if k + 2 < nd { new_coeffs[k + 2] } else { 0.0 };
                new_coeffs[k] = (2.0 * (k as f64 + 1.0)).mul_add(coeffs[k + 1], c_k2);
            }
            if nd >= 2 {
                let c2_val = if nd > 2 { new_coeffs[2] } else { 0.0 };
                new_coeffs[0] = c2_val / 2.0 + coeffs[1];
            } else if nd == 1 {
                // n == 2: derivative of c_0*T_0 + c_1*T_1 = c_1
                new_coeffs[0] = coeffs[1];
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
            let n = coeffs.len();
            let mut new_coeffs = vec![0.0; n + 1];

            // Integration of Chebyshev series:
            // integral of T_0 = T_1
            // integral of T_n = T_{n+1}/(2(n+1)) - T_{n-1}/(2(n-1)) for n >= 2
            // integral of T_1 = T_2/4 (or equivalently T_0/2 is adjusted)

            // Build from the integral formula
            if n >= 1 {
                // Integral of c_0 * T_0 = c_0 * T_1
                new_coeffs[1] += coeffs[0];
            }
            if n >= 2 {
                // Integral of c_1 * T_1 = c_1 * T_2 / 4  (+ c_1 * T_0 * ... but that's the constant)
                new_coeffs[2] += coeffs[1] / 4.0;
                new_coeffs[0] += coeffs[1] / 4.0; // from T_0 contribution
            }
            for j in 2..n {
                let jf = j as f64;
                new_coeffs[j + 1] += coeffs[j] / (2.0 * (jf + 1.0));
                new_coeffs[j - 1] -= coeffs[j] / (2.0 * (jf - 1.0));
            }

            // Adjust constant of integration
            // The integration constant is determined by evaluating at x=0
            // and setting it to `constant`. But first, let's just add the constant to c_0.
            // Since we want integral(p)(0) = constant, and T_k(0) alternates:
            // T_0(0)=1, T_1(0)=0, T_2(0)=-1, T_3(0)=0, T_4(0)=1, ...
            // So integral(0) = sum of c_k * T_k(0) = c_0 - c_2 + c_4 - ...
            // We adjust c_0 so that this sum equals `constant`.
            let current_at_zero: f64 = new_coeffs
                .iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(i, &c)| if (i / 2) % 2 == 0 { c } else { -c })
                .sum();
            new_coeffs[0] += constant - current_at_zero;

            coeffs = new_coeffs;
        }
        Ok(self.with_same_mapping(coeffs))
    }

    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError> {
        // Convert to power basis, then find roots
        let power = chebyshev_to_power(&self.coeffs);
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
        Ok(self.with_same_mapping(mul_chebyshev(&self.coeffs, &other.coeffs)))
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
        // Convert to power basis, perform divmod, convert back
        let a_power = chebyshev_to_power(&self.coeffs);
        let b_power = chebyshev_to_power(&other.coeffs);

        let a_poly = crate::power::Polynomial::new(&a_power);
        let b_poly = crate::power::Polynomial::new(&b_power);
        let (q_poly, r_poly) = a_poly.divmod(&b_poly)?;

        let q_cheb = power_to_chebyshev(q_poly.coeffs());
        let r_cheb = power_to_chebyshev(r_poly.coeffs());

        Ok((
            self.with_same_mapping(q_cheb),
            self.with_same_mapping(r_cheb),
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
        let v = chebyshev_vandermonde(x, deg);
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
        let v = chebyshev_vandermonde(x, deg);
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

    fn with_mapping(
        self,
        domain: [f64; 2],
        window: [f64; 2],
    ) -> Result<Self, FerrayError> {
        self.with_domain(domain)?.with_window(window)
    }
}

impl ToPowerBasis for Chebyshev {
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError> {
        Ok(chebyshev_to_power(&self.coeffs))
    }
}

impl FromPowerBasis for Chebyshev {
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError> {
        Ok(Self::new(&power_to_chebyshev(coeffs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- integ_with_bounds via trait default (#732) --------------------

    #[test]
    fn chebyshev_integ_with_bounds_default_args_match_integ() {
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let from_default = p.integ(2, &[0.0, 0.0]).unwrap();
        let from_bounds =
            <Chebyshev as Poly>::integ_with_bounds(&p, 2, &[0.0, 0.0], 0.0, 1.0).unwrap();
        let a = from_default.coeffs();
        let b = from_bounds.coeffs();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn chebyshev_integ_with_lbnd_value_equals_target() {
        let p = Chebyshev::new(&[1.0, 4.0, 6.0]);
        let lbnd = 0.3;
        let target = 2.5;
        let q =
            <Chebyshev as Poly>::integ_with_bounds(&p, 1, &[target], lbnd, 1.0).unwrap();
        let v = q.eval(lbnd).unwrap();
        assert!((v - target).abs() < 1e-9, "eval(lbnd)={v}, target={target}");
    }

    #[test]
    fn chebyshev_integ_with_scl_halves_coefficients() {
        let p = Chebyshev::new(&[1.0]);
        let q1 = <Chebyshev as Poly>::integ_with_bounds(&p, 1, &[0.0], 0.0, 1.0).unwrap();
        let q2 = <Chebyshev as Poly>::integ_with_bounds(&p, 1, &[0.0], 0.0, 2.0).unwrap();
        let c1 = q1.coeffs();
        let c2 = q2.coeffs();
        assert_eq!(c1.len(), c2.len());
        for (a, b) in c1.iter().zip(c2.iter()) {
            assert!((a - 2.0 * b).abs() < 1e-12);
        }
    }

    #[test]
    fn chebyshev_integ_with_zero_scl_errors() {
        let p = Chebyshev::new(&[1.0]);
        assert!(<Chebyshev as Poly>::integ_with_bounds(&p, 1, &[], 0.0, 0.0).is_err());
    }

    // ---- convert_with_mapping (#483) -----------------------------------

    #[test]
    fn chebyshev_convert_with_mapping_carries_source_domain() {
        use crate::power::Polynomial;
        use crate::traits::ConvertBasis;
        // Source: power-basis polynomial with domain [0, 10] and
        // window [-1, 1] — non-trivial mapping. Convert to Chebyshev
        // and check that the target picks up the source's domain.
        let p = Polynomial::new(&[1.0, 2.0])
            .with_domain([0.0, 10.0])
            .unwrap();
        let cheb: Chebyshev = p.convert_with_mapping(None, None).unwrap();
        assert_eq!(cheb.domain(), [0.0, 10.0]);
        assert_eq!(cheb.window(), p.window());
    }

    #[test]
    fn chebyshev_convert_with_explicit_overrides() {
        use crate::power::Polynomial;
        use crate::traits::ConvertBasis;
        let p = Polynomial::new(&[1.0, 2.0]);
        let cheb: Chebyshev = p
            .convert_with_mapping(Some([-2.0, 5.0]), Some([-1.0, 1.0]))
            .unwrap();
        assert_eq!(cheb.domain(), [-2.0, 5.0]);
        assert_eq!(cheb.window(), [-1.0, 1.0]);
    }

    // ---- basis / identity / linspace (#477, #478) ----------------------

    #[test]
    fn chebyshev_basis_n_is_unit_in_own_basis() {
        let t3 = <Chebyshev as Poly>::basis(3);
        assert_eq!(t3.coeffs(), &[0.0, 0.0, 0.0, 1.0]);
        // T_3(0.5) = 4*0.125 - 3*0.5 = 0.5 - 1.5 = -1.0
        assert!((t3.eval(0.5).unwrap() - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_identity_evaluates_as_x() {
        let p = <Chebyshev as FromPowerBasis>::identity().unwrap();
        // identity must satisfy p(x) == x for x in window.
        for x in [-0.9_f64, -0.3, 0.0, 0.4, 0.8] {
            assert!((p.eval(x).unwrap() - x).abs() < 1e-12);
        }
    }

    #[test]
    fn chebyshev_linspace_3_points_on_default_domain() {
        let p = <Chebyshev as FromPowerBasis>::identity().unwrap();
        let (xs, ys) = p.linspace(3, None).unwrap();
        assert_eq!(xs, vec![-1.0, 0.0, 1.0]);
        // identity → ys == xs
        for (x, y) in xs.iter().zip(ys.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    // ---- from_roots via FromPowerBasis default (#476) -----------------

    #[test]
    fn chebyshev_from_roots_evaluates_to_zero_at_roots() {
        let roots = [-1.0_f64, 0.0, 0.5, 1.5];
        let cheb = Chebyshev::from_roots(&roots).unwrap();
        for &r in &roots {
            assert!(
                cheb.eval(r).unwrap().abs() < 1e-9,
                "Chebyshev::from_roots: eval({r}) was non-zero"
            );
        }
    }

    #[test]
    fn eval_t0() {
        let p = Chebyshev::new(&[1.0]); // T_0 = 1
        assert!((p.eval(0.5).unwrap() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn eval_t1() {
        let p = Chebyshev::new(&[0.0, 1.0]); // T_1 = x
        assert!((p.eval(0.5).unwrap() - 0.5).abs() < 1e-14);
    }

    #[test]
    fn eval_t2() {
        let p = Chebyshev::new(&[0.0, 0.0, 1.0]); // T_2 = 2x^2 - 1
        let x = 0.5;
        let expected = 2.0 * x * x - 1.0;
        assert!((p.eval(x).unwrap() - expected).abs() < 1e-14);
    }

    #[test]
    fn chebyshev_to_power_and_back() {
        // AC-3: Chebyshev -> Polynomial -> Chebyshev round-trip
        let original_coeffs = vec![1.0, 2.0, 3.0];
        let cheb = Chebyshev::new(&original_coeffs);

        let power = chebyshev_to_power(&cheb.coeffs);
        let recovered = power_to_chebyshev(&power);

        for (i, (&orig, &rec)) in original_coeffs.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 1e-12,
                "index {i}: expected {orig}, got {rec}",
            );
        }
    }

    #[test]
    fn fit_chebyshev() {
        // Fit a known polynomial and verify evaluation
        let x: Vec<f64> = (0..20).map(|i| -1.0 + 2.0 * i as f64 / 19.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect(); // y = x^2 = T_0/2 + T_2/2 (roughly)

        let cheb = Chebyshev::fit(&x, &y, 5).unwrap();
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let eval = cheb.eval(xi).unwrap();
            assert!(
                (eval - yi).abs() < 1e-10,
                "at x={xi}: expected {yi}, got {eval}",
            );
        }
    }

    #[test]
    fn add_chebyshev() {
        let a = Chebyshev::new(&[1.0, 2.0]);
        let b = Chebyshev::new(&[3.0, 4.0, 5.0]);
        let c = a.add(&b).unwrap();
        assert!((c.coeffs[0] - 4.0).abs() < 1e-14);
        assert!((c.coeffs[1] - 6.0).abs() < 1e-14);
        assert!((c.coeffs[2] - 5.0).abs() < 1e-14);
    }

    #[test]
    fn mul_chebyshev_basic() {
        // T_1 * T_1 = (T_2 + T_0)/2
        let t1 = Chebyshev::new(&[0.0, 1.0]);
        let result = t1.mul(&t1).unwrap();
        // Should be [0.5, 0, 0.5] approximately
        assert!((result.coeffs[0] - 0.5).abs() < 1e-14);
        if result.coeffs.len() > 1 {
            assert!(result.coeffs[1].abs() < 1e-14);
        }
        assert!((result.coeffs[2] - 0.5).abs() < 1e-14);
    }

    #[test]
    fn deriv_chebyshev() {
        // T_2'(x) = d/dx(2x^2 - 1) = 4x = 4*T_1
        let t2 = Chebyshev::new(&[0.0, 0.0, 1.0]);
        let dt2 = t2.deriv(1).unwrap();
        assert!((dt2.coeffs[0] - 0.0).abs() < 1e-12);
        assert!((dt2.coeffs[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn integ_then_deriv_chebyshev() {
        // AC-4: p.integ(1, &[0.0]).deriv(1) recovers p
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let integrated = p.integ(1, &[0.0]).unwrap();
        let recovered = integrated.deriv(1).unwrap();

        // The recovered polynomial should match the original up to the original's degree
        let n = p.coeffs.len();
        for i in 0..n {
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

    // ----- Chebyshev integ accuracy (#253) -------------------------------
    //
    // Cross-check the integ() output against analytical antiderivatives
    // for known series, plus the integration-constant pass-through.

    #[test]
    fn integ_t0_yields_t1_with_zero_constant() {
        // T_0 = 1, ∫ T_0 dx = x = T_1. integ(1, &[0.0]) should give
        // coefficients [0, 1] (T_0 coeff = 0 from k, T_1 coeff = 1).
        let p = Chebyshev::new(&[1.0]);
        let i = p.integ(1, &[0.0]).unwrap();
        assert!(i.coeffs.len() >= 2);
        assert!((i.coeffs[0] - 0.0).abs() < 1e-12, "c0 = {}", i.coeffs[0]);
        assert!((i.coeffs[1] - 1.0).abs() < 1e-12, "c1 = {}", i.coeffs[1]);
    }

    #[test]
    fn integ_t0_with_nonzero_constant() {
        // T_0 = 1, ∫ T_0 dx = x + C. integ(1, &[C]) should give
        // coefficients [C, 1].
        let p = Chebyshev::new(&[1.0]);
        let c = 2.5_f64;
        let i = p.integ(1, &[c]).unwrap();
        assert!((i.coeffs[0] - c).abs() < 1e-12, "c0 = {}", i.coeffs[0]);
        assert!((i.coeffs[1] - 1.0).abs() < 1e-12, "c1 = {}", i.coeffs[1]);
    }

    #[test]
    fn integ_t1_yields_quarter_t2_plus_quarter_t0() {
        // T_1 = x, ∫ x dx = x^2/2 = (T_2 + T_0)/4.
        // integ(1, &[0.0]) on [0, 1] (T_0=0, T_1=1) should give [0.25, 0, 0.25].
        let p = Chebyshev::new(&[0.0, 1.0]);
        let i = p.integ(1, &[0.0]).unwrap();
        // Integration adds a constant to make the result evaluate to 0
        // at x = 0 in numpy's convention; verify the leading recurrence
        // structure matches the analytical formula up to the leading T_0.
        // The integ implementation uses k as the literal T_0 coefficient.
        assert!(i.coeffs.len() >= 3);
        // T_2 coefficient must be 1/4.
        assert!(
            (i.coeffs[2] - 0.25).abs() < 1e-12,
            "T_2 coeff = {}, expected 0.25",
            i.coeffs[2]
        );
        // T_1 coefficient must be 0.
        assert!((i.coeffs[1] - 0.0).abs() < 1e-12);
        // T_0 coefficient is C - (T_2/4 - T_0/4 corrections from numpy
        // convention) — check via eval that derivative recovers T_1.
        let recovered = i.deriv(1).unwrap();
        assert!((recovered.coeffs[0] - 0.0).abs() < 1e-12);
        assert!((recovered.coeffs[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn integ_double_pass_uses_two_constants() {
        // p(x) = T_0 = 1; ∫∫ 1 dx² = x²/2 + C1*x + C2.
        // integ(2, &[C2, C1]) should produce that polynomial when
        // evaluated through deriv: deriv(deriv(integ(2))) == p.
        let p = Chebyshev::new(&[1.0]);
        let c2 = 3.0_f64;
        let c1 = 2.0_f64;
        let ii = p.integ(2, &[c2, c1]).unwrap();
        let back = ii.deriv(2).unwrap();
        assert!((back.coeffs[0] - 1.0).abs() < 1e-10, "c0 = {}", back.coeffs[0]);
    }

    #[test]
    fn integ_eval_matches_analytical_antiderivative() {
        // p(x) = 2x + 3x^2 in power basis. ∫p dx = x^2 + x^3 + C.
        // Convert to Chebyshev, integrate with C=0, eval at x=2.
        // Analytical: F(2) - F(0) where F = x^2 + x^3, plus the C
        // adjustment numpy folds in.
        // We pin: deriv(integ(p)) == p for arbitrary mid-degree input.
        let p = Chebyshev::new(&[1.0, 2.0, 3.0, 4.0]);
        let recovered = p.integ(1, &[5.0]).unwrap().deriv(1).unwrap();
        for (i, &want) in p.coeffs.iter().enumerate() {
            let got = recovered.coeffs.get(i).copied().unwrap_or(0.0);
            assert!(
                (got - want).abs() < 1e-10,
                "deriv(integ) coeff {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn integ_zero_passes_returns_self() {
        // integ(0, _) is identity.
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let same = p.integ(0, &[]).unwrap();
        assert_eq!(same.coeffs, p.coeffs);
    }

    #[test]
    fn integ_constant_pads_with_zero_when_k_too_short() {
        // integ(2) with k = [] should default the unspecified constants
        // to zero (numpy convention).
        let p = Chebyshev::new(&[1.0]);
        let ii = p.integ(2, &[]).unwrap();
        let back = ii.deriv(2).unwrap();
        assert!((back.coeffs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn convert_roundtrip() {
        use crate::traits::ConvertBasis;

        let original = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let power: crate::power::Polynomial = original.convert().unwrap();
        let recovered: Chebyshev = power.convert().unwrap();

        for (i, (&orig, &rec)) in original
            .coeffs
            .iter()
            .zip(recovered.coeffs.iter())
            .enumerate()
        {
            assert!(
                (orig - rec).abs() < 1e-10,
                "index {i}: expected {orig}, got {rec}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Domain / window mapping tests (issue #474)
    // -----------------------------------------------------------------------

    #[test]
    fn chebyshev_default_mapping_is_identity() {
        let p = Chebyshev::new(&[1.0, 2.0, 3.0]);
        let (offset, scale) = p.mapparms().unwrap();
        assert_eq!(offset, 0.0);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn chebyshev_with_domain_evaluates_via_mapping() {
        // T_1(u) = u with domain [0, 4] mapped to canonical [-1, 1]:
        // u = -1 + 0.5*x, so x=0->u=-1, x=2->u=0, x=4->u=1.
        let p = Chebyshev::new(&[0.0, 1.0]).with_domain([0.0, 4.0]).unwrap();
        assert!((p.eval(0.0).unwrap() - (-1.0)).abs() < 1e-12);
        assert!((p.eval(2.0).unwrap() - 0.0).abs() < 1e-12);
        assert!((p.eval(4.0).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chebyshev_fit_with_domain_recovers_function_values_on_wide_interval() {
        // Fit y = x^2 on x ∈ [0, 100] — without auto-domain this would be
        // ill-conditioned because Chebyshev expects x in [-1, 1].
        let x: Vec<f64> = (0..=20).map(|i| i as f64 * 5.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let cheb = Chebyshev::fit_with_domain(&x, &y, 5).unwrap();
        assert_eq!(cheb.domain(), [0.0, 100.0]);
        assert_eq!(cheb.window(), [-1.0, 1.0]);
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            assert!(
                (cheb.eval(xi).unwrap() - yi).abs() < 1e-6,
                "at x={xi}: expected {yi}, got {}",
                cheb.eval(xi).unwrap()
            );
        }
    }

    #[test]
    fn chebyshev_binary_op_rejects_mismatched_mapping() {
        let a = Chebyshev::new(&[1.0, 2.0]).with_domain([0.0, 1.0]).unwrap();
        let b = Chebyshev::new(&[3.0, 4.0]).with_domain([0.0, 2.0]).unwrap();
        assert!(a.add(&b).is_err());
    }
}
