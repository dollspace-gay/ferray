// ferray-polynomial: f32 power-basis polynomial (proof-of-concept generic
// lift; tracks #612).
//
// `PolynomialF32` mirrors the existing f64-only `Polynomial` for the
// power basis. It provides eval / deriv / integ / arithmetic / trim /
// truncate / degree directly on f32; least-squares fitting and root-
// finding internally cast through f64 since the underlying linear
// algebra (faer eigensolver, Vandermonde solve) is f64-only.
//
// This is the smallest concrete demonstration of the planned
// generic-Float lift: the same structural pattern (per-T type with
// shared trait surface) maps mechanically to the other 5 bases. The
// existing `Polynomial` is untouched so callers keep their exact API.
//
// f64 ↔ f32 bridges: [`PolynomialF32::to_f64`] / [`PolynomialF32::from_f64`]
// are explicit conversions — no `From` impls because the precision loss
// should be visible at call sites.

use ferray_core::error::{FerrayError, FerrayResult};
use num_complex::Complex;

use crate::fitting::{least_squares_fit, power_vandermonde};
use crate::power::Polynomial;
use crate::roots::find_roots_from_power_coeffs;
use crate::traits::Poly;

/// f32 power-basis polynomial.
///
/// Domain/window mapping support is included for parity with
/// [`Polynomial`]; the canonical case is `domain == window == [-1.0, 1.0]`
/// (identity mapping).
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialF32 {
    coeffs: Vec<f32>,
    domain: [f32; 2],
    window: [f32; 2],
}

const DEFAULT_DOMAIN: [f32; 2] = [-1.0, 1.0];
const DEFAULT_WINDOW: [f32; 2] = [-1.0, 1.0];

impl PolynomialF32 {
    /// Create from coefficients in ascending order.
    #[must_use]
    pub fn new(coeffs: &[f32]) -> Self {
        let coeffs = if coeffs.is_empty() {
            vec![0.0]
        } else {
            coeffs.to_vec()
        };
        Self {
            coeffs,
            domain: DEFAULT_DOMAIN,
            window: DEFAULT_WINDOW,
        }
    }

    /// Borrow the coefficient slice.
    #[must_use]
    pub fn coeffs(&self) -> &[f32] {
        &self.coeffs
    }

    /// Polynomial degree (length - 1, with trailing exact zeros stripped).
    #[must_use]
    pub fn degree(&self) -> usize {
        let mut deg = self.coeffs.len().saturating_sub(1);
        while deg > 0 && self.coeffs[deg] == 0.0 {
            deg -= 1;
        }
        deg
    }

    /// Set the input domain `[a, b]`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `domain[0] == domain[1]`.
    pub fn with_domain(mut self, domain: [f32; 2]) -> FerrayResult<Self> {
        if (domain[1] - domain[0]).abs() < f32::EPSILON {
            return Err(FerrayError::invalid_value(
                "PolynomialF32::with_domain: domain must have positive width",
            ));
        }
        self.domain = domain;
        Ok(self)
    }

    /// Set the canonical window `[c, d]`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `window[0] == window[1]`.
    pub fn with_window(mut self, window: [f32; 2]) -> FerrayResult<Self> {
        if (window[1] - window[0]).abs() < f32::EPSILON {
            return Err(FerrayError::invalid_value(
                "PolynomialF32::with_window: window must have positive width",
            ));
        }
        self.window = window;
        Ok(self)
    }

    fn mapparms(&self) -> (f32, f32) {
        // u = offset + scale * x, mapping domain → window
        let (a, b) = (self.domain[0], self.domain[1]);
        let (c, d) = (self.window[0], self.window[1]);
        let scale = (d - c) / (b - a);
        let offset = c - scale * a;
        (offset, scale)
    }

    /// Evaluate at `x` via Horner's method, applying the domain → window
    /// affine map first.
    #[must_use]
    pub fn eval(&self, x: f32) -> f32 {
        let (offset, scale) = self.mapparms();
        let u = scale.mul_add(x, offset);
        let mut acc = 0.0_f32;
        for &c in self.coeffs.iter().rev() {
            acc = acc.mul_add(u, c);
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
                coeffs = vec![0.0];
                break;
            }
            let mut next = Vec::with_capacity(coeffs.len() - 1);
            for (i, &c) in coeffs.iter().enumerate().skip(1) {
                next.push(c * i as f32);
            }
            coeffs = next;
        }
        if coeffs.is_empty() {
            coeffs = vec![0.0];
        }
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Integrate `m` times. `k` supplies the integration constants
    /// (defaults to `0.0` for any short slice).
    #[must_use]
    pub fn integ(&self, m: usize, k: &[f32]) -> Self {
        if m == 0 {
            return self.clone();
        }
        let mut coeffs = self.coeffs.clone();
        for step in 0..m {
            let constant = k.get(step).copied().unwrap_or(0.0);
            let mut next = Vec::with_capacity(coeffs.len() + 1);
            next.push(constant);
            for (i, &c) in coeffs.iter().enumerate() {
                next.push(c / (i + 1) as f32);
            }
            coeffs = next;
        }
        Self {
            coeffs,
            domain: self.domain,
            window: self.window,
        }
    }

    /// Trim trailing coefficients with magnitude `<= tol`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `tol < 0.0`.
    pub fn trim(&self, tol: f32) -> FerrayResult<Self> {
        if tol < 0.0 {
            return Err(FerrayError::invalid_value(
                "PolynomialF32::trim: tolerance must be non-negative",
            ));
        }
        let mut last = self.coeffs.len();
        while last > 1 && self.coeffs[last - 1].abs() <= tol {
            last -= 1;
        }
        Ok(Self {
            coeffs: self.coeffs[..last].to_vec(),
            domain: self.domain,
            window: self.window,
        })
    }

    /// Truncate to the given number of leading terms.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size == 0`.
    pub fn truncate(&self, size: usize) -> FerrayResult<Self> {
        if size == 0 {
            return Err(FerrayError::invalid_value(
                "PolynomialF32::truncate: size must be at least 1",
            ));
        }
        let len = size.min(self.coeffs.len());
        Ok(Self {
            coeffs: self.coeffs[..len].to_vec(),
            domain: self.domain,
            window: self.window,
        })
    }

    /// Pointwise sum of two polynomials with the same mapping.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` on mapping mismatch.
    pub fn add(&self, other: &Self) -> FerrayResult<Self> {
        self.check_same_mapping(other)?;
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0.0_f32; n];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] += c;
        }
        Ok(Self {
            coeffs: out,
            domain: self.domain,
            window: self.window,
        })
    }

    /// Pointwise difference.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` on mapping mismatch.
    pub fn sub(&self, other: &Self) -> FerrayResult<Self> {
        self.check_same_mapping(other)?;
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut out = vec![0.0_f32; n];
        for (i, &c) in self.coeffs.iter().enumerate() {
            out[i] += c;
        }
        for (i, &c) in other.coeffs.iter().enumerate() {
            out[i] -= c;
        }
        Ok(Self {
            coeffs: out,
            domain: self.domain,
            window: self.window,
        })
    }

    /// Convolution-style polynomial multiplication.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` on mapping mismatch.
    pub fn mul(&self, other: &Self) -> FerrayResult<Self> {
        self.check_same_mapping(other)?;
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Ok(Self {
                coeffs: vec![0.0],
                domain: self.domain,
                window: self.window,
            });
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut out = vec![0.0_f32; n];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                out[i + j] = a.mul_add(b, out[i + j]);
            }
        }
        Ok(Self {
            coeffs: out,
            domain: self.domain,
            window: self.window,
        })
    }

    fn check_same_mapping(&self, other: &Self) -> FerrayResult<()> {
        if self.domain != other.domain || self.window != other.window {
            return Err(FerrayError::invalid_value(format!(
                "PolynomialF32: mapping mismatch — self has domain={:?} window={:?}, \
                 other has domain={:?} window={:?}",
                self.domain, self.window, other.domain, other.window
            )));
        }
        Ok(())
    }

    /// Least-squares polynomial fit. Internally promotes x/y to f64 for
    /// the linear-algebra solve, then casts the resulting coefficients
    /// back to f32 (matching numpy.polyfit's float32 path).
    ///
    /// # Errors
    /// Returns whatever `least_squares_fit` reports (singular system,
    /// length mismatch, etc.).
    pub fn fit(x: &[f32], y: &[f32], deg: usize) -> FerrayResult<Self> {
        if x.len() != y.len() {
            return Err(FerrayError::invalid_value(format!(
                "PolynomialF32::fit: x and y lengths differ ({} vs {})",
                x.len(),
                y.len()
            )));
        }
        if x.is_empty() {
            return Err(FerrayError::invalid_value(
                "PolynomialF32::fit: x and y must not be empty",
            ));
        }
        let xf64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let yf64: Vec<f64> = y.iter().map(|&v| f64::from(v)).collect();
        let v = power_vandermonde(&xf64, deg);
        let coeffs64 = least_squares_fit(&v, x.len(), deg + 1, &yf64, None)?;
        let coeffs: Vec<f32> = coeffs64.iter().map(|&c| c as f32).collect();
        Ok(Self::new(&coeffs))
    }

    /// Find the roots of the polynomial as `Vec<Complex<f32>>`.
    /// Internally promotes coefficients to f64 (the eigensolver is
    /// f64-only) and demotes the result.
    ///
    /// # Errors
    /// Returns whatever `find_roots_from_power_coeffs` reports.
    pub fn roots(&self) -> FerrayResult<Vec<Complex<f32>>> {
        let c64: Vec<f64> = self.coeffs.iter().map(|&c| f64::from(c)).collect();
        let r64 = find_roots_from_power_coeffs(&c64)?;
        Ok(r64
            .into_iter()
            .map(|c| Complex::new(c.re as f32, c.im as f32))
            .collect())
    }

    /// Convert to the f64 [`Polynomial`].
    ///
    /// Coefficients widen losslessly; domain/window are widened too. The
    /// resulting f64 polynomial is suitable for any operation that
    /// requires the existing f64-only API.
    #[must_use]
    pub fn to_f64(&self) -> Polynomial {
        let coeffs: Vec<f64> = self.coeffs.iter().map(|&c| f64::from(c)).collect();
        let p = Polynomial::new(&coeffs);
        let domain = [f64::from(self.domain[0]), f64::from(self.domain[1])];
        let window = [f64::from(self.window[0]), f64::from(self.window[1])];
        // with_domain/with_window only fail on degenerate widths, which
        // we already prevent in our own constructors; unwrapping here
        // matches the structural invariant.
        p.with_domain(domain)
            .and_then(|q| q.with_window(window))
            .expect("PolynomialF32 invariants ensure non-degenerate domain/window")
    }

    /// Cast a [`Polynomial`] (f64) down to [`PolynomialF32`].
    ///
    /// This is a precision-narrowing operation; the explicit name is
    /// deliberate so the loss is visible at call sites.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if domain/window don't fit
    /// in f32 (NaN-producing infinities or values outside f32 range).
    pub fn from_f64(p: &Polynomial) -> FerrayResult<Self> {
        let coeffs: Vec<f32> = p.coeffs().iter().map(|&c| c as f32).collect();
        let d = p.domain();
        let w = p.window();
        let domain = [d[0] as f32, d[1] as f32];
        let window = [w[0] as f32, w[1] as f32];
        for v in [domain[0], domain[1], window[0], window[1]] {
            if !v.is_finite() {
                return Err(FerrayError::invalid_value(
                    "PolynomialF32::from_f64: domain or window not representable in f32",
                ));
            }
        }
        Ok(Self {
            coeffs,
            domain,
            window,
        })
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn eval_constant() {
        let p = PolynomialF32::new(&[3.5]);
        assert_eq!(p.eval(0.0), 3.5);
        assert_eq!(p.eval(100.0), 3.5);
    }

    #[test]
    fn eval_quadratic() {
        // 1 + 2x + 3x^2 at x = 2 → 1 + 4 + 12 = 17
        let p = PolynomialF32::new(&[1.0, 2.0, 3.0]);
        assert!(approx(p.eval(2.0), 17.0, 1e-5));
    }

    #[test]
    fn deriv_quadratic() {
        // d/dx(1 + 2x + 3x^2) = 2 + 6x
        let p = PolynomialF32::new(&[1.0, 2.0, 3.0]);
        let d = p.deriv(1);
        assert_eq!(d.coeffs(), &[2.0, 6.0]);
    }

    #[test]
    fn integ_then_deriv_roundtrips() {
        let p = PolynomialF32::new(&[1.0, 2.0, 3.0]);
        let i = p.integ(1, &[]);
        let back = i.deriv(1);
        // Trim leading zero from leading-term floor.
        for (a, b) in back.coeffs().iter().zip(p.coeffs().iter()) {
            assert!(approx(*a, *b, 1e-5));
        }
    }

    #[test]
    fn arithmetic_basic() {
        let a = PolynomialF32::new(&[1.0, 2.0, 3.0]);
        let b = PolynomialF32::new(&[4.0, 5.0]);
        let sum = a.add(&b).unwrap();
        assert_eq!(sum.coeffs(), &[5.0, 7.0, 3.0]);
        let diff = a.sub(&b).unwrap();
        assert_eq!(diff.coeffs(), &[-3.0, -3.0, 3.0]);
        // (1+2x+3x^2)(4+5x) = 4 + 13x + 22x^2 + 15x^3
        let prod = a.mul(&b).unwrap();
        assert!(approx(prod.coeffs()[0], 4.0, 1e-5));
        assert!(approx(prod.coeffs()[1], 13.0, 1e-5));
        assert!(approx(prod.coeffs()[2], 22.0, 1e-5));
        assert!(approx(prod.coeffs()[3], 15.0, 1e-5));
    }

    #[test]
    fn trim_removes_trailing_zeros() {
        let p = PolynomialF32::new(&[1.0, 2.0, 0.0, 0.0]);
        let t = p.trim(0.0).unwrap();
        assert_eq!(t.coeffs(), &[1.0, 2.0]);
    }

    #[test]
    fn truncate_keeps_leading() {
        let p = PolynomialF32::new(&[1.0, 2.0, 3.0, 4.0]);
        let t = p.truncate(2).unwrap();
        assert_eq!(t.coeffs(), &[1.0, 2.0]);
    }

    #[test]
    fn fit_linear_recovers_coeffs() {
        // y = 1 + 2x sampled at x = 0, 1, 2, 3, 4 gives perfect linear fit
        let xs: Vec<f32> = (0..5).map(|i| i as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| 2.0_f32.mul_add(x, 1.0)).collect();
        let p = PolynomialF32::fit(&xs, &ys, 1).unwrap();
        assert!(approx(p.coeffs()[0], 1.0, 1e-3));
        assert!(approx(p.coeffs()[1], 2.0, 1e-3));
    }

    #[test]
    fn roots_real_quadratic() {
        // x^2 - 5x + 6 = (x-2)(x-3)
        let p = PolynomialF32::new(&[6.0, -5.0, 1.0]);
        let mut r = p.roots().unwrap();
        r.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        assert!(approx(r[0].re, 2.0, 1e-3));
        assert!(approx(r[1].re, 3.0, 1e-3));
        assert!(r[0].im.abs() < 1e-3);
        assert!(r[1].im.abs() < 1e-3);
    }

    #[test]
    fn to_f64_and_back_roundtrip() {
        let p32 = PolynomialF32::new(&[1.0, 2.0, 3.0]);
        let p64 = p32.to_f64();
        assert_eq!(p64.coeffs(), &[1.0_f64, 2.0, 3.0]);
        let back = PolynomialF32::from_f64(&p64).unwrap();
        assert_eq!(back.coeffs(), p32.coeffs());
    }

    #[test]
    fn mapping_mismatch_rejected() {
        let a = PolynomialF32::new(&[1.0, 2.0])
            .with_domain([0.0, 10.0])
            .unwrap();
        let b = PolynomialF32::new(&[3.0, 4.0]);
        // a has [0, 10] domain; b has the default [-1, 1] — mismatch.
        assert!(a.add(&b).is_err());
    }
}
