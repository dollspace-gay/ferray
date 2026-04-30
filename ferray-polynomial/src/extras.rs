// ferray-polynomial: numpy.polynomial extras
//
// This module groups the NumPy `polynomial.*` free functions that don't
// fit cleanly inside any one basis: per-basis `*line` constructors,
// `*fromroots` builders, `polyvalfromroots`, multivariate
// evaluation/Vandermonde helpers (`polyval2d`, `polyval3d`, `polygrid2d`,
// `polygrid3d`, `polyvander2d`, `polyvander3d`), Gauss-quadrature points
// and weights, Chebyshev-only extras (`chebpts1`, `chebpts2`,
// `chebweight`, `chebinterpolate`), basis-conversion free functions
// (`cheb2poly`, `poly2cheb`, etc.), and `*mulx`.
//
// All functions are f64-only — see issue #612 for the planned generic
// lift across all 6 bases.

use ferray_core::error::{FerrayError, FerrayResult};

use crate::chebyshev::Chebyshev;
use crate::hermite::Hermite;
use crate::hermite_e::HermiteE;
use crate::laguerre::Laguerre;
use crate::legendre::Legendre;
use crate::power::Polynomial;
use crate::traits::{FromPowerBasis, Poly, ToPowerBasis};

// ===========================================================================
// *line — degree-1 helpers
// ===========================================================================

/// Power-basis polynomial `c0 + c1 * x` (i.e. `[c0, c1]`).
///
/// Equivalent to `numpy.polynomial.polynomial.polyline(c0, c1)`.
#[must_use]
pub fn polyline(c0: f64, c1: f64) -> Polynomial {
    Polynomial::new(&[c0, c1])
}

/// Chebyshev-basis polynomial `c0 + c1 * T_1(x)` (i.e. `[c0, c1]`).
#[must_use]
pub fn chebline(c0: f64, c1: f64) -> Chebyshev {
    Chebyshev::new(&[c0, c1])
}

/// Legendre-basis polynomial `c0 + c1 * P_1(x)` (i.e. `[c0, c1]`).
#[must_use]
pub fn legline(c0: f64, c1: f64) -> Legendre {
    Legendre::new(&[c0, c1])
}

/// Laguerre-basis polynomial `c0 + c1 * L_1(x)` — note `L_1(x) = 1 - x`,
/// so the resulting polynomial in power form is `(c0 + c1) - c1 * x`.
#[must_use]
pub fn lagline(c0: f64, c1: f64) -> Laguerre {
    Laguerre::new(&[c0, c1])
}

/// Hermite (physicist's) polynomial `c0 + c1 * H_1(x)` where `H_1(x) = 2x`.
#[must_use]
pub fn hermline(c0: f64, c1: f64) -> Hermite {
    Hermite::new(&[c0, c1])
}

/// HermiteE (probabilist's) polynomial `c0 + c1 * He_1(x)` where `He_1(x) = x`.
#[must_use]
pub fn hermeline(c0: f64, c1: f64) -> HermiteE {
    HermiteE::new(&[c0, c1])
}

// ===========================================================================
// polyfromroots — power-basis polynomial from a list of (real) roots
// ===========================================================================

/// Construct a power-basis polynomial whose roots are the given values.
///
/// Returns the polynomial `(x - r_0)(x - r_1)...(x - r_{n-1})` in
/// ascending-coefficient form. Equivalent to
/// `numpy.polynomial.polynomial.polyfromroots`.
///
/// An empty `roots` slice yields the constant polynomial `1`.
#[must_use]
pub fn polyfromroots(roots: &[f64]) -> Polynomial {
    let mut coeffs = vec![1.0];
    for &r in roots {
        // Multiply current poly by (x - r): new_coeffs[i] = -r * old[i] + old[i-1]
        let n = coeffs.len();
        let mut next = vec![0.0; n + 1];
        for i in 0..n {
            next[i] += -r * coeffs[i];
            next[i + 1] += coeffs[i];
        }
        coeffs = next;
    }
    Polynomial::new(&coeffs)
}

/// Construct a Chebyshev polynomial from roots — power-basis pivot.
///
/// Computes the power-basis form via [`polyfromroots`], then converts.
/// Equivalent to `numpy.polynomial.chebyshev.chebfromroots`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the basis conversion fails.
pub fn chebfromroots(roots: &[f64]) -> FerrayResult<Chebyshev> {
    let p = polyfromroots(roots);
    Chebyshev::from_power_basis(p.coeffs())
}

/// Legendre-basis polynomial from roots — power-basis pivot.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the basis conversion fails.
pub fn legfromroots(roots: &[f64]) -> FerrayResult<Legendre> {
    let p = polyfromroots(roots);
    Legendre::from_power_basis(p.coeffs())
}

/// Laguerre-basis polynomial from roots — power-basis pivot.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the basis conversion fails.
pub fn lagfromroots(roots: &[f64]) -> FerrayResult<Laguerre> {
    let p = polyfromroots(roots);
    Laguerre::from_power_basis(p.coeffs())
}

/// Hermite (physicist's) polynomial from roots — power-basis pivot.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the basis conversion fails.
pub fn hermfromroots(roots: &[f64]) -> FerrayResult<Hermite> {
    let p = polyfromroots(roots);
    Hermite::from_power_basis(p.coeffs())
}

/// HermiteE (probabilist's) polynomial from roots — power-basis pivot.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the basis conversion fails.
pub fn hermefromroots(roots: &[f64]) -> FerrayResult<HermiteE> {
    let p = polyfromroots(roots);
    HermiteE::from_power_basis(p.coeffs())
}

// ===========================================================================
// polyvalfromroots — direct evaluation from roots, no coefficient expansion
// ===========================================================================

/// Evaluate a polynomial defined by its roots at `x` directly via the
/// product `Π (x - r_i)`. More numerically stable than expanding to
/// power-basis coefficients first when the polynomial has many close
/// roots.
///
/// Equivalent to `numpy.polynomial.polynomial.polyvalfromroots`.
#[must_use]
pub fn polyvalfromroots(x: f64, roots: &[f64]) -> f64 {
    roots.iter().fold(1.0, |acc, &r| acc * (x - r))
}

// ===========================================================================
// Multivariate power-basis: polyval2d / polyval3d / polygrid2d / polygrid3d
// ===========================================================================

#[inline]
fn poly_eval_1d(c: &[f64], x: f64) -> f64 {
    let mut acc = 0.0;
    for &ci in c.iter().rev() {
        acc = acc * x + ci;
    }
    acc
}

/// Evaluate a 2-D power-series polynomial `Σ_{i,j} c[i,j] * x^i * y^j` at
/// the points `(xs[k], ys[k])`.
///
/// `coeffs` is row-major over `(i, j)` with `nx` rows of `ny` columns:
/// `coeffs[i * ny + j]` is the coefficient of `x^i * y^j`.
///
/// Returns one value per element of `xs`/`ys`. Equivalent to
/// `numpy.polynomial.polynomial.polyval2d`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `xs.len() != ys.len()` or
///   `coeffs.len() != nx * ny`.
pub fn polyval2d(
    xs: &[f64],
    ys: &[f64],
    coeffs: &[f64],
    nx: usize,
    ny: usize,
) -> FerrayResult<Vec<f64>> {
    if xs.len() != ys.len() {
        return Err(FerrayError::shape_mismatch(format!(
            "polyval2d: xs.len()={} != ys.len()={}",
            xs.len(),
            ys.len()
        )));
    }
    if coeffs.len() != nx * ny {
        return Err(FerrayError::shape_mismatch(format!(
            "polyval2d: coeffs.len()={} != nx*ny={}",
            coeffs.len(),
            nx * ny
        )));
    }
    let mut out = Vec::with_capacity(xs.len());
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        // Evaluate inner-y first per row, then outer-x via Horner.
        let mut row_evals = Vec::with_capacity(nx);
        for i in 0..nx {
            let row = &coeffs[i * ny..(i + 1) * ny];
            row_evals.push(poly_eval_1d(row, y));
        }
        out.push(poly_eval_1d(&row_evals, x));
    }
    Ok(out)
}

/// Evaluate a 3-D power-series polynomial `Σ_{i,j,k} c[i,j,k] * x^i * y^j * z^k`
/// at the points `(xs[m], ys[m], zs[m])`.
///
/// `coeffs` is row-major over `(i, j, k)` with shape `(nx, ny, nz)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if input lengths or coefficient count
///   are inconsistent.
pub fn polyval3d(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    coeffs: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> FerrayResult<Vec<f64>> {
    if xs.len() != ys.len() || xs.len() != zs.len() {
        return Err(FerrayError::shape_mismatch(
            "polyval3d: xs/ys/zs lengths must match",
        ));
    }
    if coeffs.len() != nx * ny * nz {
        return Err(FerrayError::shape_mismatch(format!(
            "polyval3d: coeffs.len()={} != nx*ny*nz={}",
            coeffs.len(),
            nx * ny * nz
        )));
    }
    let mut out = Vec::with_capacity(xs.len());
    for ((&x, &y), &z) in xs.iter().zip(ys.iter()).zip(zs.iter()) {
        // Inner z, then y, then x.
        let mut yx = Vec::with_capacity(nx);
        for i in 0..nx {
            let mut yslice = Vec::with_capacity(ny);
            for j in 0..ny {
                let off = (i * ny + j) * nz;
                yslice.push(poly_eval_1d(&coeffs[off..off + nz], z));
            }
            yx.push(poly_eval_1d(&yslice, y));
        }
        out.push(poly_eval_1d(&yx, x));
    }
    Ok(out)
}

/// Evaluate a 2-D power-series polynomial on the Cartesian grid
/// `xs × ys`. Returns a row-major vector of shape `(xs.len(), ys.len())`.
///
/// Equivalent to `numpy.polynomial.polynomial.polygrid2d`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `coeffs.len() != nx * ny`.
pub fn polygrid2d(
    xs: &[f64],
    ys: &[f64],
    coeffs: &[f64],
    nx: usize,
    ny: usize,
) -> FerrayResult<Vec<f64>> {
    if coeffs.len() != nx * ny {
        return Err(FerrayError::shape_mismatch(format!(
            "polygrid2d: coeffs.len()={} != nx*ny={}",
            coeffs.len(),
            nx * ny
        )));
    }
    let mut out = Vec::with_capacity(xs.len() * ys.len());
    for &x in xs {
        for &y in ys {
            // Reuse polyval2d's per-point logic.
            let mut row_evals = Vec::with_capacity(nx);
            for i in 0..nx {
                let row = &coeffs[i * ny..(i + 1) * ny];
                row_evals.push(poly_eval_1d(row, y));
            }
            out.push(poly_eval_1d(&row_evals, x));
        }
    }
    Ok(out)
}

/// Evaluate a 3-D power-series polynomial on the Cartesian grid
/// `xs × ys × zs`. Returns a row-major vector of shape
/// `(xs.len(), ys.len(), zs.len())`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `coeffs.len() != nx * ny * nz`.
pub fn polygrid3d(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    coeffs: &[f64],
    nx: usize,
    ny: usize,
    nz: usize,
) -> FerrayResult<Vec<f64>> {
    if coeffs.len() != nx * ny * nz {
        return Err(FerrayError::shape_mismatch(format!(
            "polygrid3d: coeffs.len()={} != nx*ny*nz={}",
            coeffs.len(),
            nx * ny * nz
        )));
    }
    let mut out = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &x in xs {
        for &y in ys {
            for &z in zs {
                let mut yx = Vec::with_capacity(nx);
                for i in 0..nx {
                    let mut yslice = Vec::with_capacity(ny);
                    for j in 0..ny {
                        let off = (i * ny + j) * nz;
                        yslice.push(poly_eval_1d(&coeffs[off..off + nz], z));
                    }
                    yx.push(poly_eval_1d(&yslice, y));
                }
                out.push(poly_eval_1d(&yx, x));
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// polyvander2d / polyvander3d
// ===========================================================================

/// 2-D power-basis Vandermonde: row-major `(npoints, (degx+1)*(degy+1))`,
/// with the column index `i*(degy+1) + j` corresponding to `x^i * y^j`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `xs.len() != ys.len()`.
pub fn polyvander2d(xs: &[f64], ys: &[f64], degx: usize, degy: usize) -> FerrayResult<Vec<f64>> {
    if xs.len() != ys.len() {
        return Err(FerrayError::shape_mismatch(format!(
            "polyvander2d: xs.len()={} != ys.len()={}",
            xs.len(),
            ys.len()
        )));
    }
    let n = xs.len();
    let cols = (degx + 1) * (degy + 1);
    let mut out = vec![0.0; n * cols];
    for k in 0..n {
        let mut xpows = Vec::with_capacity(degx + 1);
        let mut ypows = Vec::with_capacity(degy + 1);
        let mut p = 1.0;
        for _ in 0..=degx {
            xpows.push(p);
            p *= xs[k];
        }
        let mut q = 1.0;
        for _ in 0..=degy {
            ypows.push(q);
            q *= ys[k];
        }
        for i in 0..=degx {
            for j in 0..=degy {
                out[k * cols + i * (degy + 1) + j] = xpows[i] * ypows[j];
            }
        }
    }
    Ok(out)
}

/// 3-D power-basis Vandermonde: row-major
/// `(npoints, (degx+1)*(degy+1)*(degz+1))`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if input lengths don't match.
pub fn polyvander3d(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    degx: usize,
    degy: usize,
    degz: usize,
) -> FerrayResult<Vec<f64>> {
    if xs.len() != ys.len() || xs.len() != zs.len() {
        return Err(FerrayError::shape_mismatch(
            "polyvander3d: xs/ys/zs lengths must match",
        ));
    }
    let n = xs.len();
    let cols = (degx + 1) * (degy + 1) * (degz + 1);
    let mut out = vec![0.0; n * cols];
    for k in 0..n {
        let mut xpows = Vec::with_capacity(degx + 1);
        let mut ypows = Vec::with_capacity(degy + 1);
        let mut zpows = Vec::with_capacity(degz + 1);
        let mut p = 1.0;
        for _ in 0..=degx {
            xpows.push(p);
            p *= xs[k];
        }
        let mut q = 1.0;
        for _ in 0..=degy {
            ypows.push(q);
            q *= ys[k];
        }
        let mut r = 1.0;
        for _ in 0..=degz {
            zpows.push(r);
            r *= zs[k];
        }
        for i in 0..=degx {
            for j in 0..=degy {
                for m in 0..=degz {
                    out[k * cols + i * (degy + 1) * (degz + 1) + j * (degz + 1) + m] =
                        xpows[i] * ypows[j] * zpows[m];
                }
            }
        }
    }
    Ok(out)
}

// ===========================================================================
// Chebyshev-only extras
// ===========================================================================

/// Roots of the Chebyshev polynomial T_n: `chebpts1(n)`.
///
/// Returns the `n` Chebyshev-Gauss nodes
/// `cos((2k - 1) * pi / (2n))` for `k = 1..=n`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn chebpts1(n: usize) -> FerrayResult<Vec<f64>> {
    if n == 0 {
        return Err(FerrayError::invalid_value("chebpts1: n must be > 0"));
    }
    let mut pts = Vec::with_capacity(n);
    for k in 1..=n {
        let theta = ((2 * k - 1) as f64) * std::f64::consts::PI / (2.0 * n as f64);
        pts.push(theta.cos());
    }
    pts.iter_mut()
        .for_each(|x| *x = if x.abs() < f64::EPSILON { 0.0 } else { *x });
    Ok(pts)
}

/// Chebyshev-Gauss-Lobatto points `chebpts2(n)`.
///
/// Returns the `n` extrema `cos(k * pi / (n - 1))` for `k = 0..n`.
/// For `n == 1` returns `[0.0]`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn chebpts2(n: usize) -> FerrayResult<Vec<f64>> {
    if n == 0 {
        return Err(FerrayError::invalid_value("chebpts2: n must be > 0"));
    }
    if n == 1 {
        return Ok(vec![0.0]);
    }
    let mut pts = Vec::with_capacity(n);
    let denom = (n - 1) as f64;
    for k in 0..n {
        let theta = (k as f64) * std::f64::consts::PI / denom;
        pts.push(theta.cos());
    }
    Ok(pts)
}

/// Chebyshev weight function on `[-1, 1]`: `w(x) = 1 / sqrt(1 - x^2)`.
///
/// Returns `f64::INFINITY` at the endpoints.
#[must_use]
pub fn chebweight(x: f64) -> f64 {
    let denom = (1.0 - x * x).sqrt();
    if denom <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / denom
    }
}

/// Chebyshev interpolation: build a Chebyshev polynomial of degree `deg`
/// that interpolates `f` at the `deg + 1` Chebyshev-Gauss nodes.
///
/// Equivalent to `numpy.polynomial.chebyshev.chebinterpolate(f, deg)`.
/// Computes the coefficients via the Discrete Chebyshev Transform.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `deg == 0` (need at least 1 point).
pub fn chebinterpolate<F>(f: F, deg: usize) -> FerrayResult<Chebyshev>
where
    F: Fn(f64) -> f64,
{
    if deg == 0 {
        return Err(FerrayError::invalid_value(
            "chebinterpolate: deg must be > 0",
        ));
    }
    let n = deg + 1;
    let pts = chebpts1(n)?;
    let fs: Vec<f64> = pts.iter().copied().map(&f).collect();
    // DCT-II onto Chebyshev coefficients.
    let mut c = vec![0.0; n];
    let pi_n = std::f64::consts::PI / n as f64;
    for (k, slot) in c.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &fj) in fs.iter().enumerate() {
            let theta = ((2 * j + 1) as f64) * (k as f64) * pi_n / 2.0;
            acc += fj * theta.cos();
        }
        *slot = if k == 0 {
            acc / n as f64
        } else {
            2.0 * acc / n as f64
        };
    }
    Ok(Chebyshev::new(&c))
}

// ===========================================================================
// Gauss quadrature: chebgauss / leggauss / hermgauss / hermegauss / lagguass
// ===========================================================================

/// `n`-point Chebyshev-Gauss quadrature: returns `(points, weights)` such
/// that `Σ w_k f(x_k) ≈ ∫_{-1}^{1} f(x) / sqrt(1 - x^2) dx`.
///
/// All weights are `pi / n`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn chebgauss(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    let pts = chebpts1(n)?;
    let w = std::f64::consts::PI / n as f64;
    Ok((pts, vec![w; n]))
}

/// `n`-point Legendre-Gauss quadrature: returns `(points, weights)` for
/// `∫_{-1}^{1} f(x) dx ≈ Σ w_k f(x_k)`.
///
/// Uses Newton iteration on the Legendre polynomial roots, with weights
/// `w_k = 2 / ((1 - x_k^2) * (P'_n(x_k))^2)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn leggauss(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(FerrayError::invalid_value("leggauss: n must be > 0"));
    }
    if n == 1 {
        return Ok((vec![0.0], vec![2.0]));
    }
    let mut points = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n {
        // Initial guess (asymptotic): cos(pi (i + 0.75) / (n + 0.5))
        let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        // Newton iteration on P_n(x).
        for _ in 0..100 {
            // Forward recurrence to compute P_n and P_{n-1}.
            let mut p_prev = 1.0;
            let mut p_curr = x;
            for k in 1..n {
                let kf = k as f64;
                let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
                p_prev = p_curr;
                p_curr = p_next;
            }
            // P'_n(x) = n * (x * P_n - P_{n-1}) / (x^2 - 1)
            let dp = nf * (x * p_curr - p_prev) / (x * x - 1.0);
            let dx = p_curr / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        points[i] = x;
        // Recompute P_{n-1}(x) for the weight formula.
        let mut p_prev = 1.0;
        let mut p_curr = x;
        for k in 1..n {
            let kf = k as f64;
            let p_next = ((2.0 * kf + 1.0) * x * p_curr - kf * p_prev) / (kf + 1.0);
            p_prev = p_curr;
            p_curr = p_next;
        }
        let dp = nf * (x * p_curr - p_prev) / (x * x - 1.0);
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    // Sort points in ascending order (Newton iteration may produce reversed order).
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| points[a].partial_cmp(&points[b]).unwrap());
    let pts_sorted: Vec<f64> = idx.iter().map(|&i| points[i]).collect();
    let w_sorted: Vec<f64> = idx.iter().map(|&i| weights[i]).collect();
    Ok((pts_sorted, w_sorted))
}

/// `n`-point Hermite (physicist's) Gauss quadrature: returns `(points, weights)`
/// for `∫_{-∞}^{∞} f(x) exp(-x^2) dx ≈ Σ w_k f(x_k)`.
///
/// Computed via Newton iteration on the Hermite polynomial roots, using
/// the standard physicist's recurrence `H_{n+1} = 2 x H_n - 2 n H_{n-1}`.
/// Weights: `w_k = 2^(n-1) * n! * sqrt(pi) / (n^2 * H_{n-1}(x_k)^2)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn hermgauss(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(FerrayError::invalid_value("hermgauss: n must be > 0"));
    }
    let nf = n as f64;
    let mut points = vec![0.0; n];
    let mut weights = vec![0.0; n];
    // Orthonormal Hermite polynomial seed: H̃_0(x) = π^(-1/4).
    let h0 = std::f64::consts::PI.powf(-0.25);
    for i in 0..n {
        // Asymptotic initial guess (Stroud & Secrest).
        let z = if i == 0 {
            (2.0 * nf + 1.0).sqrt() - 1.85575 * (2.0 * nf + 1.0).powf(-1.0 / 6.0)
        } else if i == 1 {
            points[0] - 1.14 * nf.powf(0.426) / points[0]
        } else if i == 2 {
            1.86_f64.mul_add(points[1], -0.86 * points[0])
        } else if i == 3 {
            1.91_f64.mul_add(points[2], -0.91 * points[1])
        } else {
            2.0_f64.mul_add(points[i - 1], -points[i - 2])
        };
        let mut x = z;
        let mut h_prev = 0.0;
        for _ in 0..100 {
            let mut p1 = h0;
            let mut p2 = 0.0;
            for k in 0..n {
                let kf = k as f64;
                let p3 = p2;
                p2 = p1;
                // H_{k+1}(x) = sqrt(2/(k+1)) * x * H_k - sqrt(k/(k+1)) * H_{k-1}
                p1 = x * (2.0 / (kf + 1.0)).sqrt() * p2 - (kf / (kf + 1.0)).sqrt() * p3;
            }
            h_prev = p2;
            // Newton step using the orthonormal-recurrence form.
            let dp = (2.0 * nf).sqrt() * h_prev;
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        points[i] = x;
        // Orthonormal Hermite recurrence: w_k = 1 / (n * H̃_{n-1}(x_k)²).
        weights[i] = 1.0 / (nf * h_prev * h_prev);
    }
    // Sort ascending.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| points[a].partial_cmp(&points[b]).unwrap());
    let pts_sorted: Vec<f64> = idx.iter().map(|&i| points[i]).collect();
    let w_sorted: Vec<f64> = idx.iter().map(|&i| weights[i]).collect();
    Ok((pts_sorted, w_sorted))
}

/// `n`-point HermiteE (probabilist's) Gauss quadrature: returns
/// `(points, weights)` for `∫_{-∞}^{∞} f(x) exp(-x^2/2) dx ≈ Σ w_k f(x_k)`.
///
/// Related to [`hermgauss`] by the variable substitution `x = sqrt(2) y`:
/// nodes scale by `sqrt(2)` and weights by `sqrt(2)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn hermegauss(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    let (pts, w) = hermgauss(n)?;
    let s = std::f64::consts::SQRT_2;
    Ok((
        pts.into_iter().map(|x| x * s).collect(),
        w.into_iter().map(|wi| wi * s).collect(),
    ))
}

/// `n`-point Laguerre-Gauss quadrature: returns `(points, weights)` for
/// `∫_{0}^{∞} f(x) exp(-x) dx ≈ Σ w_k f(x_k)`.
///
/// Uses Newton iteration on the Laguerre polynomial roots, weights via
/// `w_k = x_k / ((n+1)^2 * L_{n+1}(x_k)^2)`.
///
/// Note: NumPy's spelling has a typo `lagguass`; ferray exposes both
/// `lagguass` (for parity) and the corrected `laggauss`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n == 0`.
pub fn laggauss(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    if n == 0 {
        return Err(FerrayError::invalid_value("laggauss: n must be > 0"));
    }
    let nf = n as f64;
    let mut points = vec![0.0; n];
    let mut weights = vec![0.0; n];
    for i in 0..n {
        // Initial guess.
        let mut x = if i == 0 {
            3.0 / (1.0 + 2.4 * nf)
        } else if i == 1 {
            points[0] + 15.0 / (1.0 + 2.5 * nf)
        } else {
            let ai = (i - 1) as f64;
            (1.0 + 2.55 * ai) / (1.9 * ai) * (points[i - 1] - points[i - 2]) + points[i - 1]
        };
        // Newton iterate.
        let mut l_n = 0.0;
        let mut l_n1 = 0.0;
        for _ in 0..100 {
            // Forward recurrence: L_0 = 1, L_1 = 1 - x,
            //   L_{k+1} = ((2k+1 - x) L_k - k L_{k-1}) / (k+1)
            let mut p_prev = 1.0;
            let mut p_curr = 1.0 - x;
            if n == 1 {
                p_curr = 1.0 - x;
            }
            for k in 1..n {
                let kf = k as f64;
                let p_next = ((2.0 * kf + 1.0 - x) * p_curr - kf * p_prev) / (kf + 1.0);
                p_prev = p_curr;
                p_curr = p_next;
            }
            l_n = p_curr;
            l_n1 = p_prev;
            // L'_n(x) = n * (L_n - L_{n-1}) / x
            let dl = nf * (l_n - l_n1) / x;
            let dx = l_n / dl;
            x -= dx;
            if dx.abs() < 1e-14 * x.abs().max(1.0) {
                break;
            }
        }
        points[i] = x;
        // Weight formula:
        //   w_k = x_k / ((n+1) L_{n+1}(x_k))^2
        // Compute L_{n+1} via one more recurrence step.
        let l_n_plus_1 = ((2.0 * nf + 1.0 - x) * l_n - nf * l_n1) / (nf + 1.0);
        let denom = (nf + 1.0) * l_n_plus_1;
        weights[i] = x / (denom * denom);
    }
    Ok((points, weights))
}

/// NumPy-spelling alias for [`laggauss`] (the upstream name has a typo).
///
/// # Errors
/// Returns the same errors as [`laggauss`].
pub fn lagguass(n: usize) -> FerrayResult<(Vec<f64>, Vec<f64>)> {
    laggauss(n)
}

// ===========================================================================
// Basis-conversion free functions
// ===========================================================================

/// Convert power-basis coefficients to Chebyshev-basis coefficients.
///
/// Equivalent to `numpy.polynomial.chebyshev.poly2cheb`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn poly2cheb(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    let cheb = Chebyshev::from_power_basis(coeffs)?;
    Ok(cheb.coeffs().to_vec())
}

/// Convert Chebyshev-basis coefficients to power-basis coefficients.
///
/// Equivalent to `numpy.polynomial.chebyshev.cheb2poly`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn cheb2poly(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    let cheb = Chebyshev::new(coeffs);
    cheb.to_power_basis()
}

/// Convert power-basis coefficients to Legendre-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn poly2leg(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    let leg = Legendre::from_power_basis(coeffs)?;
    Ok(leg.coeffs().to_vec())
}

/// Convert Legendre-basis coefficients to power-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn leg2poly(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Legendre::new(coeffs).to_power_basis()
}

/// Convert power-basis coefficients to Laguerre-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn poly2lag(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Ok(Laguerre::from_power_basis(coeffs)?.coeffs().to_vec())
}

/// Convert Laguerre-basis coefficients to power-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn lag2poly(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Laguerre::new(coeffs).to_power_basis()
}

/// Convert power-basis coefficients to Hermite-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn poly2herm(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Ok(Hermite::from_power_basis(coeffs)?.coeffs().to_vec())
}

/// Convert Hermite-basis coefficients to power-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn herm2poly(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Hermite::new(coeffs).to_power_basis()
}

/// Convert power-basis coefficients to HermiteE-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn poly2herme(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    Ok(HermiteE::from_power_basis(coeffs)?.coeffs().to_vec())
}

/// Convert HermiteE-basis coefficients to power-basis coefficients.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if conversion fails.
pub fn herme2poly(coeffs: &[f64]) -> FerrayResult<Vec<f64>> {
    HermiteE::new(coeffs).to_power_basis()
}

// ===========================================================================
// *mulx — multiply by x in each basis (returns new coeffs vector)
// ===========================================================================

/// Multiply a power-basis polynomial by `x`: shift coefficients up by one.
///
/// Equivalent to `numpy.polynomial.polynomial.polymulx`.
#[must_use]
pub fn polymulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0, 0.0];
    }
    let mut out = vec![0.0; coeffs.len() + 1];
    for (i, &c) in coeffs.iter().enumerate() {
        out[i + 1] = c;
    }
    out
}

/// Multiply a Chebyshev polynomial by `x` using the recurrence
/// `x * T_k = (T_{k+1} + T_{k-1}) / 2` (with `T_{-1} = 0`).
#[must_use]
pub fn chebmulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    let n = coeffs.len();
    let mut out = vec![0.0; n + 1];
    // c_0 * T_0 * x = c_0 * T_1
    out[1] = coeffs[0];
    if n > 1 {
        // c_1 * T_1 * x = c_1 * (T_2 + T_0) / 2
        out[0] += coeffs[1] / 2.0;
        out[2] += coeffs[1] / 2.0;
    }
    for k in 2..n {
        out[k + 1] += coeffs[k] / 2.0;
        out[k - 1] += coeffs[k] / 2.0;
    }
    out
}

/// Multiply a Legendre polynomial by `x` using
/// `x * P_k = ((k+1) P_{k+1} + k P_{k-1}) / (2k+1)`.
#[must_use]
pub fn legmulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    let n = coeffs.len();
    let mut out = vec![0.0; n + 1];
    for k in 0..n {
        let kf = k as f64;
        let denom = 2.0 * kf + 1.0;
        if k + 1 < n + 1 {
            out[k + 1] += coeffs[k] * (kf + 1.0) / denom;
        }
        if k > 0 {
            out[k - 1] += coeffs[k] * kf / denom;
        }
    }
    out
}

/// Multiply a Laguerre polynomial by `x`: `x L_k = -(k+1) L_{k+1} +
/// (2k+1) L_k - k L_{k-1}`.
#[must_use]
pub fn lagmulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    let n = coeffs.len();
    let mut out = vec![0.0; n + 1];
    for k in 0..n {
        let kf = k as f64;
        out[k] += coeffs[k] * (2.0 * kf + 1.0);
        out[k + 1] += -coeffs[k] * (kf + 1.0);
        if k > 0 {
            out[k - 1] += -coeffs[k] * kf;
        }
    }
    out
}

/// Multiply a Hermite polynomial by `x`: `x H_k = H_{k+1}/2 + k H_{k-1}`.
#[must_use]
pub fn hermmulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    let n = coeffs.len();
    let mut out = vec![0.0; n + 1];
    for k in 0..n {
        out[k + 1] += coeffs[k] / 2.0;
        if k > 0 {
            out[k - 1] += coeffs[k] * k as f64;
        }
    }
    out
}

/// Multiply a HermiteE polynomial by `x`: `x He_k = He_{k+1} + k He_{k-1}`.
#[must_use]
pub fn hermemulx(coeffs: &[f64]) -> Vec<f64> {
    if coeffs.is_empty() {
        return vec![0.0];
    }
    let n = coeffs.len();
    let mut out = vec![0.0; n + 1];
    for k in 0..n {
        out[k + 1] += coeffs[k];
        if k > 0 {
            out[k - 1] += coeffs[k] * k as f64;
        }
    }
    out
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn approx_slice(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx(*x, *y, tol))
    }

    // -- *line --

    #[test]
    fn polyline_basic() {
        let p = polyline(2.0, 3.0);
        assert_eq!(p.coeffs(), &[2.0, 3.0]);
    }

    #[test]
    fn chebline_legline_lagline_basic() {
        assert_eq!(chebline(1.0, 2.0).coeffs(), &[1.0, 2.0]);
        assert_eq!(legline(1.0, 2.0).coeffs(), &[1.0, 2.0]);
        assert_eq!(lagline(1.0, 2.0).coeffs(), &[1.0, 2.0]);
        assert_eq!(hermline(1.0, 2.0).coeffs(), &[1.0, 2.0]);
        assert_eq!(hermeline(1.0, 2.0).coeffs(), &[1.0, 2.0]);
    }

    // -- polyfromroots --

    #[test]
    fn polyfromroots_simple_roots() {
        // (x - 1)(x - 2) = x^2 - 3x + 2 → coeffs ascending: [2, -3, 1]
        let p = polyfromroots(&[1.0, 2.0]);
        assert!(approx_slice(p.coeffs(), &[2.0, -3.0, 1.0], 1e-12));
    }

    #[test]
    fn polyfromroots_empty() {
        let p = polyfromroots(&[]);
        assert_eq!(p.coeffs(), &[1.0]);
    }

    #[test]
    fn polyvalfromroots_matches_expansion() {
        let roots = [1.0, 2.0, 3.0];
        let p = polyfromroots(&roots);
        for &x in &[-2.0, 0.0, 1.5, 4.0] {
            let direct = polyvalfromroots(x, &roots);
            let via_p = p.eval(x).unwrap();
            assert!(approx(direct, via_p, 1e-10), "{direct} vs {via_p}");
        }
    }

    // -- polyval2d / polygrid2d --

    #[test]
    fn polyval2d_constant() {
        // c[0,0] = 5 → result is always 5
        let r = polyval2d(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[5.0], 1, 1).unwrap();
        assert_eq!(r, vec![5.0, 5.0, 5.0]);
    }

    #[test]
    fn polyval2d_xy() {
        // f(x, y) = x * y → coeffs[1, 1] = 1, others zero, layout (nx=2, ny=2)
        let coeffs = vec![0.0, 0.0, 0.0, 1.0]; // (i*ny + j): (1,1) at index 3
        let r = polyval2d(&[2.0, 3.0], &[5.0, 7.0], &coeffs, 2, 2).unwrap();
        assert!(approx_slice(&r, &[10.0, 21.0], 1e-12));
    }

    #[test]
    fn polygrid2d_dims() {
        // c[0,0] = 1 → 1 everywhere, expect xs.len() * ys.len() values.
        let r = polygrid2d(&[0.0, 1.0], &[0.0, 1.0, 2.0], &[1.0], 1, 1).unwrap();
        assert_eq!(r.len(), 6);
        assert!(r.iter().all(|&v| (v - 1.0).abs() < 1e-12));
    }

    // -- polyvander2d --

    #[test]
    fn polyvander2d_basic() {
        // degx=1, degy=1, so 4 columns: 1, y, x, x*y
        let v = polyvander2d(&[2.0], &[3.0], 1, 1).unwrap();
        assert!(approx_slice(&v, &[1.0, 3.0, 2.0, 6.0], 1e-12));
    }

    // -- chebpts1 / chebpts2 / chebweight --

    #[test]
    fn chebpts1_count_and_range() {
        let pts = chebpts1(5).unwrap();
        assert_eq!(pts.len(), 5);
        for &p in &pts {
            assert!(p > -1.0 && p < 1.0);
        }
    }

    #[test]
    fn chebpts2_endpoints() {
        let pts = chebpts2(5).unwrap();
        assert_eq!(pts.len(), 5);
        // chebpts2 includes the endpoints.
        assert!(approx(pts[0], 1.0, 1e-12));
        assert!(approx(pts[4], -1.0, 1e-12));
    }

    #[test]
    fn chebweight_at_zero_is_one() {
        assert!(approx(chebweight(0.0), 1.0, 1e-12));
    }

    #[test]
    fn chebinterpolate_constant() {
        let p = chebinterpolate(|_| 3.0, 4).unwrap();
        for &x in &[-0.5, 0.0, 0.5] {
            assert!(approx(p.eval(x).unwrap(), 3.0, 1e-10));
        }
    }

    // -- Gauss quadrature --

    #[test]
    fn chebgauss_constant_integral() {
        // ∫_{-1}^{1} 1 / sqrt(1-x^2) dx = pi
        let (_, w) = chebgauss(8).unwrap();
        let s: f64 = w.iter().sum();
        assert!(approx(s, std::f64::consts::PI, 1e-12));
    }

    #[test]
    fn leggauss_constant_integral() {
        // ∫_{-1}^{1} dx = 2
        let (_, w) = leggauss(8).unwrap();
        let s: f64 = w.iter().sum();
        assert!(approx(s, 2.0, 1e-12));
    }

    #[test]
    fn leggauss_polynomial_2x_squared() {
        // ∫_{-1}^{1} x^2 dx = 2/3
        let (pts, w) = leggauss(8).unwrap();
        let s: f64 = pts.iter().zip(w.iter()).map(|(x, w)| x * x * w).sum();
        assert!(approx(s, 2.0 / 3.0, 1e-10));
    }

    #[test]
    fn hermgauss_constant_integral() {
        // ∫_{-∞}^{∞} exp(-x^2) dx = sqrt(pi)
        let (_, w) = hermgauss(8).unwrap();
        let s: f64 = w.iter().sum();
        assert!(approx(s, std::f64::consts::PI.sqrt(), 1e-10));
    }

    #[test]
    fn laggauss_constant_integral() {
        // ∫_{0}^{∞} exp(-x) dx = 1
        let (_, w) = laggauss(8).unwrap();
        let s: f64 = w.iter().sum();
        assert!(approx(s, 1.0, 1e-10));
    }

    #[test]
    fn laggauss_x_integral() {
        // ∫_{0}^{∞} x exp(-x) dx = 1
        let (pts, w) = laggauss(8).unwrap();
        let s: f64 = pts.iter().zip(w.iter()).map(|(x, w)| x * w).sum();
        assert!(approx(s, 1.0, 1e-10));
    }

    #[test]
    fn hermegauss_constant_integral() {
        // ∫_{-∞}^{∞} exp(-x^2/2) dx = sqrt(2*pi)
        let (_, w) = hermegauss(8).unwrap();
        let s: f64 = w.iter().sum();
        assert!(approx(s, (2.0 * std::f64::consts::PI).sqrt(), 1e-10));
    }

    // -- basis conversion --

    #[test]
    fn poly2cheb_then_back_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 0.5];
        let cheb = poly2cheb(&original).unwrap();
        let back = cheb2poly(&cheb).unwrap();
        assert!(approx_slice(&back, &original, 1e-10));
    }

    #[test]
    fn poly2leg_roundtrip() {
        let original = vec![1.0, -2.0, 3.0];
        let leg = poly2leg(&original).unwrap();
        let back = leg2poly(&leg).unwrap();
        assert!(approx_slice(&back, &original, 1e-10));
    }

    // -- mulx --

    #[test]
    fn polymulx_shifts_coeffs() {
        // (1 + 2x + 3x^2) * x = x + 2x^2 + 3x^3
        let r = polymulx(&[1.0, 2.0, 3.0]);
        assert_eq!(r, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn chebmulx_consistency_with_eval() {
        // Verify x * T(x) evaluated equals (T(x)) * x at random points.
        let coeffs = [1.0, 2.0, -1.0];
        let p = Chebyshev::new(&coeffs);
        let mulx = chebmulx(&coeffs);
        let p_mulx = Chebyshev::new(&mulx);
        for &x in &[-0.5, 0.0, 0.3, 0.9] {
            let lhs = p_mulx.eval(x).unwrap();
            let rhs = x * p.eval(x).unwrap();
            assert!(approx(lhs, rhs, 1e-10), "x={x}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn legmulx_consistency_with_eval() {
        let coeffs = [1.0, 2.0, -1.0, 0.5];
        let p = Legendre::new(&coeffs);
        let mulx = legmulx(&coeffs);
        let p_mulx = Legendre::new(&mulx);
        for &x in &[-0.5, 0.0, 0.3, 0.9] {
            let lhs = p_mulx.eval(x).unwrap();
            let rhs = x * p.eval(x).unwrap();
            assert!(approx(lhs, rhs, 1e-10));
        }
    }

    #[test]
    fn hermemulx_consistency_with_eval() {
        let coeffs = [1.0, 2.0, -1.0];
        let p = HermiteE::new(&coeffs);
        let mulx = hermemulx(&coeffs);
        let p_mulx = HermiteE::new(&mulx);
        for &x in &[-1.0, 0.0, 0.5] {
            let lhs = p_mulx.eval(x).unwrap();
            let rhs = x * p.eval(x).unwrap();
            assert!(
                approx(lhs, rhs, 1e-10),
                "x={x}: lhs={lhs}, rhs={rhs}, mulx={mulx:?}"
            );
        }
    }
}
