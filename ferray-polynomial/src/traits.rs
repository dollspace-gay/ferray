// ferray-polynomial: Poly, ToPowerBasis, FromPowerBasis traits (REQ-7, REQ-8, REQ-9, REQ-10, REQ-11)

use ferray_core::error::FerrayError;
use num_complex::Complex;

use crate::mapping::mapparms;

/// Common trait for polynomial types in all bases.
///
/// Every polynomial class (power, Chebyshev, Legendre, Laguerre, Hermite,
/// `HermiteE`) implements this trait, providing evaluation, calculus,
/// arithmetic, root-finding, and fitting.
pub trait Poly: Clone + Sized {
    /// Evaluate the polynomial at a single point.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if computation fails.
    fn eval(&self, x: f64) -> Result<f64, FerrayError>;

    /// Evaluate the polynomial at multiple points.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if computation fails.
    fn eval_many(&self, x: &[f64]) -> Result<Vec<f64>, FerrayError> {
        x.iter().map(|&xi| self.eval(xi)).collect()
    }

    /// Differentiate the polynomial `m` times.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `m` is invalid.
    fn deriv(&self, m: usize) -> Result<Self, FerrayError>;

    /// Integrate the polynomial `m` times with integration constants `k`.
    ///
    /// `k` should have exactly `m` elements. If fewer are provided,
    /// zeros are used for the missing constants.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if computation fails.
    fn integ(&self, m: usize, k: &[f64]) -> Result<Self, FerrayError>;

    /// Return the roots of the polynomial as complex values.
    ///
    /// Uses companion matrix eigenvalues via `ferray-linalg`.
    ///
    /// # Errors
    /// Returns an error if root-finding fails or linalg is unavailable.
    fn roots(&self) -> Result<Vec<Complex<f64>>, FerrayError>;

    /// Return the degree of the polynomial.
    fn degree(&self) -> usize;

    /// Return the coefficients of this polynomial.
    fn coeffs(&self) -> &[f64];

    /// Remove trailing coefficients below tolerance.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `tol` is negative.
    fn trim(&self, tol: f64) -> Result<Self, FerrayError>;

    /// Truncate the polynomial to the given number of terms.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `size` is zero.
    fn truncate(&self, size: usize) -> Result<Self, FerrayError>;

    /// Add two polynomials of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn add(&self, other: &Self) -> Result<Self, FerrayError>;

    /// Subtract another polynomial of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn sub(&self, other: &Self) -> Result<Self, FerrayError>;

    /// Multiply two polynomials of the same basis.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn mul(&self, other: &Self) -> Result<Self, FerrayError>;

    /// Raise the polynomial to the `n`-th power.
    ///
    /// # Errors
    /// Returns an error if the operation fails.
    fn pow(&self, n: usize) -> Result<Self, FerrayError>;

    /// Divide this polynomial by another, returning (quotient, remainder).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the divisor is zero.
    fn divmod(&self, other: &Self) -> Result<(Self, Self), FerrayError>;

    /// Least-squares polynomial fit of degree `deg` to the given data.
    ///
    /// # Errors
    /// Returns an error if the fit fails (e.g., singular matrix).
    fn fit(x: &[f64], y: &[f64], deg: usize) -> Result<Self, FerrayError>;

    /// Weighted least-squares polynomial fit.
    ///
    /// # Errors
    /// Returns an error if the fit fails.
    fn fit_weighted(x: &[f64], y: &[f64], deg: usize, w: &[f64]) -> Result<Self, FerrayError>;

    /// Construct the polynomial from the given coefficients.
    fn from_coeffs(coeffs: &[f64]) -> Self;

    // -----------------------------------------------------------------------
    // Domain / window mapping (issue #474)
    //
    // NumPy's polynomial classes carry a `domain` (where x lives) and a
    // `window` (the canonical interval for the basis). When evaluating, x is
    // affine-mapped from `domain` to `window` first. This is critical for
    // numerical stability when fitting on non-canonical intervals.
    // -----------------------------------------------------------------------

    /// Return the input domain `[a, b]` for this polynomial.
    ///
    /// `eval(x)` for `x` outside `[a, b]` is still defined (the affine
    /// extrapolation is applied) but may be ill-conditioned.
    fn domain(&self) -> [f64; 2];

    /// Return the canonical window `[c, d]` for this basis.
    ///
    /// For most bases this is `[-1, 1]`. Laguerre uses `[0, 1]`. The Power
    /// basis nominally uses `[-1, 1]` but in practice many users leave the
    /// domain identical to the window for an identity mapping.
    fn window(&self) -> [f64; 2];

    /// Compute the affine map parameters `(offset, scale)` such that
    /// evaluating the polynomial at `x` is the same as evaluating its
    /// canonical-basis form at `u = offset + scale * x`.
    ///
    /// Returns `(0, 1)` when `domain == window` (the default mapping).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the domain is degenerate.
    fn mapparms(&self) -> Result<(f64, f64), FerrayError> {
        mapparms(self.domain(), self.window())
    }

    /// Replace this polynomial's `(domain, window)` mapping in one
    /// shot, returning a new polynomial with the new mapping (#483).
    ///
    /// Each basis impl typically forwards to its inherent
    /// `with_domain` / `with_window` methods. The trait method exists
    /// so generic code (notably [`ConvertBasis::convert_with_mapping`])
    /// can carry mapping context across basis conversion.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `domain` or `window` is
    /// degenerate.
    fn with_mapping(
        self,
        domain: [f64; 2],
        window: [f64; 2],
    ) -> Result<Self, FerrayError>;

    /// Polynomial composition: returns `r` such that
    /// `r(x) == self(q(x))` (#485, #733).
    ///
    /// Generic default that routes through the power basis: convert
    /// both `self` and `q` to power coefficients, run Horner-style
    /// composition there, then convert back to this basis. The
    /// inherent `Polynomial::compose` shipped earlier (#485) is more
    /// efficient for the power basis itself; this default covers
    /// Chebyshev / Hermite / HermiteE / Legendre / Laguerre.
    ///
    /// `self` and `q` must share the same domain/window mapping.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` on mapping mismatch, plus any
    /// error propagated from basis conversions.
    fn compose(&self, q: &Self) -> Result<Self, FerrayError>
    where
        Self: ToPowerBasis + FromPowerBasis,
    {
        if self.domain() != q.domain() || self.window() != q.window() {
            return Err(FerrayError::invalid_value(
                "compose: polynomials must share the same domain and window",
            ));
        }
        let p_power = self.to_power_basis()?;
        let q_power = q.to_power_basis()?;
        if p_power.is_empty() {
            return Self::from_power_basis(&[]);
        }
        let mut r = vec![*p_power.last().unwrap()];
        for &c in p_power.iter().rev().skip(1) {
            let mut next = vec![0.0_f64; r.len() + q_power.len() - 1];
            for (i, &ri) in r.iter().enumerate() {
                for (j, &qj) in q_power.iter().enumerate() {
                    next[i + j] += ri * qj;
                }
            }
            next[0] += c;
            r = next;
        }
        let target = Self::from_power_basis(&r)?;
        target.with_mapping(self.domain(), self.window())
    }

    /// Integrate `m` times with full numpy-parity knobs (#482, #732).
    ///
    /// Generalises [`integ`](Self::integ) by also accepting:
    /// - `lbnd`: lower bound. Each step adjusts the integration
    ///   constant so the integrated polynomial evaluates to
    ///   `k[step]` at `lbnd` (or 0 if `step >= k.len()`).
    /// - `scl`: per-step divisor applied after each integration
    ///   (matches numpy's coefficient-scaling semantic, not a true
    ///   variable change).
    ///
    /// For `lbnd = 0` and `scl = 1` the result is identical to
    /// [`integ`](Self::integ). The default impl works for every
    /// basis whose 0-degree polynomial evaluates to 1 — Power,
    /// Chebyshev, Hermite, HermiteE, Legendre, Laguerre. Mirrors
    /// `numpy.polynomial.<basis>.<basis>int(c, m, k, lbnd, scl)`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if `scl` is zero, plus
    /// any error propagated from `integ`, `eval`, or
    /// `with_mapping`.
    fn integ_with_bounds(
        &self,
        m: usize,
        k: &[f64],
        lbnd: f64,
        scl: f64,
    ) -> Result<Self, FerrayError> {
        if scl == 0.0 {
            return Err(FerrayError::invalid_value(
                "integ_with_bounds: scl must be non-zero",
            ));
        }
        if m == 0 {
            return Ok(self.clone());
        }
        let domain = self.domain();
        let window = self.window();
        let mut current = self.clone();
        for step in 0..m {
            let integrated = current.integ(1, &[0.0])?;
            let scaled: Vec<f64> = integrated.coeffs().iter().map(|c| c / scl).collect();
            let probe = Self::from_coeffs(&scaled).with_mapping(domain, window)?;
            let target = if step < k.len() { k[step] } else { 0.0 };
            let value_at_lbnd = probe.eval(lbnd)?;
            let mut next = scaled;
            next[0] += target - value_at_lbnd;
            current = Self::from_coeffs(&next).with_mapping(domain, window)?;
        }
        Ok(current)
    }

    /// Build the `deg`-th basis polynomial in this type's basis
    /// (#478).
    ///
    /// For every basis, this is the polynomial whose coefficients are
    /// `[0, 0, ..., 0, 1]` with the `1` at position `deg`. In the
    /// power basis that's `x^deg`; in Chebyshev it's `T_deg(x)`;
    /// in Hermite it's `H_deg(x)`; etc.
    ///
    /// The result uses the type's default domain/window. Chain
    /// `with_domain` / `with_window` to override.
    fn basis(deg: usize) -> Self {
        let mut coeffs = vec![0.0_f64; deg + 1];
        coeffs[deg] = 1.0;
        Self::from_coeffs(&coeffs)
    }

    /// Sample the polynomial at `n` evenly-spaced x-values across its
    /// domain (or the supplied `domain` override) and return both the
    /// x-values and the y-values (#477).
    ///
    /// Equivalent to `numpy.polynomial.<basis>.<basis>.linspace(n, domain)`,
    /// commonly used for plotting.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if `n < 2` or evaluation fails.
    fn linspace(
        &self,
        n: usize,
        domain: Option<[f64; 2]>,
    ) -> Result<(Vec<f64>, Vec<f64>), FerrayError> {
        if n < 2 {
            return Err(FerrayError::invalid_value(
                "linspace requires n >= 2",
            ));
        }
        let [lo, hi] = domain.unwrap_or_else(|| self.domain());
        let step = (hi - lo) / (n as f64 - 1.0);
        let xs: Vec<f64> = (0..n).map(|i| lo + step * i as f64).collect();
        let ys = self.eval_many(&xs)?;
        Ok((xs, ys))
    }
}

/// Trait for converting a polynomial to its power basis representation.
pub trait ToPowerBasis: Poly {
    /// Convert this polynomial to power basis coefficients.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn to_power_basis(&self) -> Result<Vec<f64>, FerrayError>;
}

/// Trait for constructing a polynomial from power basis coefficients.
pub trait FromPowerBasis: Poly {
    /// Create this polynomial type from power basis coefficients.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn from_power_basis(coeffs: &[f64]) -> Result<Self, FerrayError>;

    /// Identity polynomial: `p(x) = x` represented in this basis
    /// (#478).
    ///
    /// Built by converting the power-basis polynomial `[0, 1]` to the
    /// target basis. For Power / Chebyshev / Legendre / HermiteE this
    /// is just `[0, 1]`; for Hermite (physicist) it becomes `[0, 0.5]`
    /// because `H_1(x) = 2x`; for Laguerre it becomes `[1, -1]`
    /// because `L_1(x) = 1 - x`.
    ///
    /// The result uses the type's default domain/window. Chain
    /// `with_domain` / `with_window` to override.
    ///
    /// # Errors
    /// Returns an error if power-basis conversion fails.
    fn identity() -> Result<Self, FerrayError> {
        Self::from_power_basis(&[0.0, 1.0])
    }

    /// Construct this polynomial type from a list of roots (#476).
    ///
    /// Builds `(x - r0)*(x - r1)*...*(x - r_{n-1})` in the power
    /// basis, then converts to this basis. An empty `roots` slice
    /// yields the constant polynomial `1`, matching
    /// `numpy.polynomial.<basis>.<basis>fromroots`.
    ///
    /// # Errors
    /// Returns an error if power-basis conversion fails for the
    /// computed coefficients.
    fn from_roots(roots: &[f64]) -> Result<Self, FerrayError> {
        let mut coeffs = vec![1.0_f64];
        for &r in roots {
            let n = coeffs.len();
            let mut next = vec![0.0_f64; n + 1];
            for i in 0..n {
                next[i] -= r * coeffs[i];
                next[i + 1] += coeffs[i];
            }
            coeffs = next;
        }
        Self::from_power_basis(&coeffs)
    }
}

/// Extension trait providing `.convert::<TargetType>()` for basis conversion.
///
/// This uses power basis as a canonical pivot: source -> power -> target.
/// This avoids the N^2 pairwise conversion problem and the coherence
/// issues with blanket `From` impls.
pub trait ConvertBasis: ToPowerBasis {
    /// Convert this polynomial to a different basis type.
    ///
    /// The target inherits its basis's default domain/window. Use
    /// [`convert_with_mapping`](Self::convert_with_mapping) to carry
    /// the source's mapping (or supply explicit overrides) to the
    /// target — the equivalent of numpy's
    /// `convert(domain, kind, window)`.
    ///
    /// # Errors
    /// Returns an error if conversion fails.
    fn convert<T: FromPowerBasis>(&self) -> Result<T, FerrayError> {
        let power_coeffs = self.to_power_basis()?;
        T::from_power_basis(&power_coeffs)
    }

    /// Convert to a different basis type, propagating domain/window
    /// (#483).
    ///
    /// `domain` and `window` default to the source polynomial's
    /// values when `None`. The target's coefficients are computed via
    /// the power-basis pivot just like [`convert`](Self::convert),
    /// then `with_mapping` applies the chosen domain/window.
    ///
    /// Equivalent to numpy's
    /// `numpy.polynomial.<basis>.<basis>.convert(domain, kind, window)`.
    ///
    /// # Errors
    /// Returns an error if conversion or mapping application fails
    /// (e.g. degenerate domain/window).
    fn convert_with_mapping<T: FromPowerBasis>(
        &self,
        domain: Option<[f64; 2]>,
        window: Option<[f64; 2]>,
    ) -> Result<T, FerrayError> {
        let power_coeffs = self.to_power_basis()?;
        let target = T::from_power_basis(&power_coeffs)?;
        let d = domain.unwrap_or_else(|| self.domain());
        let w = window.unwrap_or_else(|| self.window());
        target.with_mapping(d, w)
    }
}

// Blanket implementation: anything that implements ToPowerBasis automatically
// gets ConvertBasis.
impl<P: ToPowerBasis> ConvertBasis for P {}
