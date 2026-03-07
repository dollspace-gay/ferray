// ferray-polynomial: Least-squares fitting via Vandermonde matrices (REQ-9)
//
// Implements polynomial fitting using the normal equations approach.
// For a Vandermonde-like matrix V, solves V^T W V c = V^T W y
// where W is a diagonal weight matrix.

use ferray_core::error::FerrayError;

/// Build a Vandermonde matrix for power basis fitting.
///
/// Given data points `x` of length `n` and degree `deg`, produces an
/// n x (deg+1) matrix where row i is `[1, x_i, x_i^2, ..., x_i^deg]`.
///
/// Returned in row-major order.
pub fn power_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        v[i * ncols] = 1.0;
        for j in 1..ncols {
            v[i * ncols + j] = v[i * ncols + (j - 1)] * x[i];
        }
    }
    v
}

/// Build a Vandermonde-like matrix for Chebyshev basis fitting.
///
/// Row i is `[T_0(x_i), T_1(x_i), ..., T_deg(x_i)]` where T_j is the
/// j-th Chebyshev polynomial of the first kind.
pub fn chebyshev_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        if ncols > 0 {
            v[i * ncols] = 1.0; // T_0(x) = 1
        }
        if ncols > 1 {
            v[i * ncols + 1] = x[i]; // T_1(x) = x
        }
        for j in 2..ncols {
            // T_n(x) = 2x T_{n-1}(x) - T_{n-2}(x)
            v[i * ncols + j] = 2.0 * x[i] * v[i * ncols + (j - 1)] - v[i * ncols + (j - 2)];
        }
    }
    v
}

/// Build a Vandermonde-like matrix for Legendre basis fitting.
pub fn legendre_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        if ncols > 0 {
            v[i * ncols] = 1.0; // P_0(x) = 1
        }
        if ncols > 1 {
            v[i * ncols + 1] = x[i]; // P_1(x) = x
        }
        for j in 2..ncols {
            // P_n(x) = ((2n-1)*x*P_{n-1}(x) - (n-1)*P_{n-2}(x)) / n
            let jf = j as f64;
            v[i * ncols + j] = ((2.0 * jf - 1.0) * x[i] * v[i * ncols + (j - 1)]
                - (jf - 1.0) * v[i * ncols + (j - 2)])
                / jf;
        }
    }
    v
}

/// Build a Vandermonde-like matrix for Laguerre basis fitting.
pub fn laguerre_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        if ncols > 0 {
            v[i * ncols] = 1.0; // L_0(x) = 1
        }
        if ncols > 1 {
            v[i * ncols + 1] = 1.0 - x[i]; // L_1(x) = 1 - x
        }
        for j in 2..ncols {
            // L_n(x) = ((2n-1-x)*L_{n-1}(x) - (n-1)*L_{n-2}(x)) / n
            let jf = j as f64;
            v[i * ncols + j] = ((2.0 * jf - 1.0 - x[i]) * v[i * ncols + (j - 1)]
                - (jf - 1.0) * v[i * ncols + (j - 2)])
                / jf;
        }
    }
    v
}

/// Build a Vandermonde-like matrix for physicist's Hermite basis fitting.
pub fn hermite_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        if ncols > 0 {
            v[i * ncols] = 1.0; // H_0(x) = 1
        }
        if ncols > 1 {
            v[i * ncols + 1] = 2.0 * x[i]; // H_1(x) = 2x
        }
        for j in 2..ncols {
            // H_n(x) = 2x*H_{n-1}(x) - 2(n-1)*H_{n-2}(x)
            let jf = j as f64;
            v[i * ncols + j] =
                2.0 * x[i] * v[i * ncols + (j - 1)] - 2.0 * (jf - 1.0) * v[i * ncols + (j - 2)];
        }
    }
    v
}

/// Build a Vandermonde-like matrix for probabilist's Hermite basis fitting.
pub fn hermite_e_vandermonde(x: &[f64], deg: usize) -> Vec<f64> {
    let n = x.len();
    let ncols = deg + 1;
    let mut v = vec![0.0; n * ncols];
    for i in 0..n {
        if ncols > 0 {
            v[i * ncols] = 1.0; // He_0(x) = 1
        }
        if ncols > 1 {
            v[i * ncols + 1] = x[i]; // He_1(x) = x
        }
        for j in 2..ncols {
            // He_n(x) = x*He_{n-1}(x) - (n-1)*He_{n-2}(x)
            let jf = j as f64;
            v[i * ncols + j] = x[i] * v[i * ncols + (j - 1)] - (jf - 1.0) * v[i * ncols + (j - 2)];
        }
    }
    v
}

/// Solve the least-squares problem V c = y using the normal equations.
///
/// Given a Vandermonde(-like) matrix V (n x m, row-major), data y (n),
/// and optional weights w (n), solves for the coefficient vector c (m).
///
/// Uses the normal equations: (V^T W V) c = V^T W y
/// where W = diag(w) if provided, or identity otherwise.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if dimensions don't match.
/// Returns `FerrayError::SingularMatrix` if the normal equations matrix is singular.
pub fn least_squares_fit(
    v: &[f64],
    n: usize,
    m: usize,
    y: &[f64],
    w: Option<&[f64]>,
) -> Result<Vec<f64>, FerrayError> {
    if v.len() != n * m {
        return Err(FerrayError::invalid_value(format!(
            "Vandermonde matrix has {} elements, expected {}",
            v.len(),
            n * m
        )));
    }
    if y.len() != n {
        return Err(FerrayError::invalid_value(format!(
            "y has {} elements, expected {}",
            y.len(),
            n
        )));
    }
    if let Some(w) = w {
        if w.len() != n {
            return Err(FerrayError::invalid_value(format!(
                "weights have {} elements, expected {}",
                w.len(),
                n
            )));
        }
    }

    // Compute V^T W V (m x m) and V^T W y (m)
    let mut vtv = vec![0.0; m * m];
    let mut vty = vec![0.0; m];

    for i in 0..n {
        let wi = w.map_or(1.0, |w| w[i]);
        for j in 0..m {
            let vij = v[i * m + j];
            vty[j] += vij * wi * y[i];
            for k in j..m {
                let vik = v[i * m + k];
                vtv[j * m + k] += vij * wi * vik;
            }
        }
    }
    // Fill lower triangle (symmetric)
    for j in 0..m {
        for k in (j + 1)..m {
            vtv[k * m + j] = vtv[j * m + k];
        }
    }

    // Solve vtv * c = vty using Cholesky decomposition
    // Since V^T W V is positive semidefinite (positive definite if V has full column rank),
    // Cholesky is appropriate.
    cholesky_solve(&vtv, m, &vty)
}

/// Solve A x = b using Cholesky decomposition where A is symmetric positive definite.
///
/// `a` is an m x m matrix in row-major order.
///
/// # Errors
/// Returns `FerrayError::SingularMatrix` if A is not positive definite.
fn cholesky_solve(a: &[f64], m: usize, b: &[f64]) -> Result<Vec<f64>, FerrayError> {
    // Cholesky decomposition: A = L L^T
    let mut l = vec![0.0; m * m];

    for i in 0..m {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * m + k] * l[j * m + k];
            }
            if i == j {
                let diag = a[i * m + i] - sum;
                if diag <= 0.0 {
                    // Fall back to LU-like solver for ill-conditioned systems
                    return lu_solve(a, m, b);
                }
                l[i * m + j] = diag.sqrt();
            } else {
                let ljj = l[j * m + j];
                if ljj.abs() < f64::EPSILON * 1e6 {
                    return lu_solve(a, m, b);
                }
                l[i * m + j] = (a[i * m + j] - sum) / ljj;
            }
        }
    }

    // Forward solve: L y = b
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i * m + j] * y[j];
        }
        let lii = l[i * m + i];
        if lii.abs() < f64::EPSILON * 1e6 {
            return lu_solve(a, m, b);
        }
        y[i] = (b[i] - sum) / lii;
    }

    // Back solve: L^T x = y
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..m {
            sum += l[j * m + i] * x[j];
        }
        let lii = l[i * m + i];
        if lii.abs() < f64::EPSILON * 1e6 {
            return lu_solve(a, m, b);
        }
        x[i] = (y[i] - sum) / lii;
    }

    Ok(x)
}

/// Solve A x = b using LU decomposition with partial pivoting.
///
/// Fallback for cases where Cholesky fails (ill-conditioned systems).
///
/// # Errors
/// Returns `FerrayError::SingularMatrix` if A is singular.
fn lu_solve(a: &[f64], m: usize, b: &[f64]) -> Result<Vec<f64>, FerrayError> {
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..m).collect();

    for k in 0..m {
        // Find pivot
        let mut max_val = lu[perm[k] * m + k].abs();
        let mut max_idx = k;
        for i in (k + 1)..m {
            let v = lu[perm[i] * m + k].abs();
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        if max_val < f64::EPSILON * 1e6 {
            return Err(FerrayError::SingularMatrix {
                message: "normal equations matrix is singular or nearly singular".to_string(),
            });
        }
        perm.swap(k, max_idx);

        let pivot = lu[perm[k] * m + k];
        for i in (k + 1)..m {
            let factor = lu[perm[i] * m + k] / pivot;
            lu[perm[i] * m + k] = factor;
            for j in (k + 1)..m {
                let val = lu[perm[k] * m + j];
                lu[perm[i] * m + j] -= factor * val;
            }
        }
    }

    // Forward substitution (L y = P b)
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut sum = 0.0;
        for j in 0..i {
            sum += lu[perm[i] * m + j] * y[j];
        }
        y[i] = b[perm[i]] - sum;
    }

    // Back substitution (U x = y)
    let mut x = vec![0.0; m];
    for i in (0..m).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..m {
            sum += lu[perm[i] * m + j] * x[j];
        }
        let diag = lu[perm[i] * m + i];
        if diag.abs() < f64::EPSILON * 1e10 {
            return Err(FerrayError::SingularMatrix {
                message: "normal equations matrix is singular".to_string(),
            });
        }
        x[i] = (y[i] - sum) / diag;
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_vandermonde_basic() {
        let x = vec![1.0, 2.0, 3.0];
        let v = power_vandermonde(&x, 2);
        // row 0: [1, 1, 1]
        // row 1: [1, 2, 4]
        // row 2: [1, 3, 9]
        assert_eq!(v.len(), 9);
        assert!((v[0] - 1.0).abs() < 1e-14);
        assert!((v[1] - 1.0).abs() < 1e-14);
        assert!((v[2] - 1.0).abs() < 1e-14);
        assert!((v[3] - 1.0).abs() < 1e-14);
        assert!((v[4] - 2.0).abs() < 1e-14);
        assert!((v[5] - 4.0).abs() < 1e-14);
        assert!((v[6] - 1.0).abs() < 1e-14);
        assert!((v[7] - 3.0).abs() < 1e-14);
        assert!((v[8] - 9.0).abs() < 1e-14);
    }

    #[test]
    fn fit_line() {
        // y = 2x + 1, fit with degree 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let v = power_vandermonde(&x, 1);
        let c = least_squares_fit(&v, x.len(), 2, &y, None).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-10, "c[0] = {}", c[0]);
        assert!((c[1] - 2.0).abs() < 1e-10, "c[1] = {}", c[1]);
    }

    #[test]
    fn fit_quadratic() {
        // y = x^2 - 3x + 2
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.5).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi - 3.0 * xi + 2.0).collect();
        let v = power_vandermonde(&x, 2);
        let c = least_squares_fit(&v, x.len(), 3, &y, None).unwrap();
        assert!((c[0] - 2.0).abs() < 1e-10, "c[0] = {}", c[0]);
        assert!((c[1] - (-3.0)).abs() < 1e-10, "c[1] = {}", c[1]);
        assert!((c[2] - 1.0).abs() < 1e-10, "c[2] = {}", c[2]);
    }

    #[test]
    fn fit_weighted() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // uniform weights
        let v = power_vandermonde(&x, 1);
        let c = least_squares_fit(&v, x.len(), 2, &y, Some(&w)).unwrap();
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn chebyshev_vandermonde_basic() {
        let x = vec![0.0, 0.5, 1.0];
        let v = chebyshev_vandermonde(&x, 2);
        // T_0(x)=1, T_1(x)=x, T_2(x)=2x^2-1
        // row 0: [1, 0, -1]
        // row 1: [1, 0.5, -0.5]
        // row 2: [1, 1, 1]
        assert!((v[0] - 1.0).abs() < 1e-14); // T_0(0) = 1
        assert!((v[1] - 0.0).abs() < 1e-14); // T_1(0) = 0
        assert!((v[2] - (-1.0)).abs() < 1e-14); // T_2(0) = -1
    }
}
