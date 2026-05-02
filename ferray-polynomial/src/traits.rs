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
    /// # Errors
    /// Returns an error if conversion fails.
    fn convert<T: FromPowerBasis>(&self) -> Result<T, FerrayError> {
        let power_coeffs = self.to_power_basis()?;
        T::from_power_basis(&power_coeffs)
    }
}

// Blanket implementation: anything that implements ToPowerBasis automatically
// gets ConvertBasis.
impl<P: ToPowerBasis> ConvertBasis for P {}
