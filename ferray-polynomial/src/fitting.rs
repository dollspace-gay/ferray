// ferray-polynomial: Least-squares fitting via Vandermonde matrices (REQ-9)
//
// Implements polynomial fitting using the normal equations approach.
// For a Vandermonde-like matrix V, solves V^T W V c = V^T W y
// where W is a diagonal weight matrix.

use ferray_core::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_core::error::FerrayError;

/// Build a Vandermonde matrix for power basis fitting.
///
/// Given data points `x` of length `n` and degree `deg`, produces an
/// n x (deg+1) matrix where row i is `[1, x_i, x_i^2, ..., x_i^deg]`.
///
/// Returned in row-major order.
#[must_use]
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
/// Row i is `[T_0(x_i), T_1(x_i), ..., T_deg(x_i)]` where `T_j` is the
/// j-th Chebyshev polynomial of the first kind.
#[must_use]
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
            v[i * ncols + j] =
                (2.0 * x[i]).mul_add(v[i * ncols + (j - 1)], -v[i * ncols + (j - 2)]);
        }
    }
    v
}

/// Build a Vandermonde-like matrix for Legendre basis fitting.
#[must_use]
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
            v[i * ncols + j] = (2.0f64.mul_add(jf, -1.0) * x[i]).mul_add(
                v[i * ncols + (j - 1)],
                -((jf - 1.0) * v[i * ncols + (j - 2)]),
            ) / jf;
        }
    }
    v
}

/// Build a Vandermonde-like matrix for Laguerre basis fitting.
#[must_use]
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
            v[i * ncols + j] = (2.0f64.mul_add(jf, -1.0) - x[i]).mul_add(
                v[i * ncols + (j - 1)],
                -((jf - 1.0) * v[i * ncols + (j - 2)]),
            ) / jf;
        }
    }
    v
}

/// Build a Vandermonde-like matrix for physicist's Hermite basis fitting.
#[must_use]
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
            v[i * ncols + j] = (2.0 * x[i]).mul_add(
                v[i * ncols + (j - 1)],
                -(2.0 * (jf - 1.0) * v[i * ncols + (j - 2)]),
            );
        }
    }
    v
}

/// Build a Vandermonde-like matrix for probabilist's Hermite basis fitting.
#[must_use]
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
            v[i * ncols + j] = x[i].mul_add(
                v[i * ncols + (j - 1)],
                -((jf - 1.0) * v[i * ncols + (j - 2)]),
            );
        }
    }
    v
}

/// Solve the least-squares problem V c = y using Householder QR decomposition.
///
/// Given a Vandermonde(-like) matrix V (n x m, row-major), data y (n),
/// and optional weights w (n), solves for the coefficient vector c (m).
///
/// Uses thin QR decomposition on the (optionally weighted) system, which
/// avoids forming the normal equations V^T V (whose condition number is
/// the square of V's). This matches `NumPy`'s approach of using an
/// SVD/QR-based lstsq rather than the normal equations.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if dimensions don't match.
/// Returns `FerrayError::SingularMatrix` if the system is rank-deficient.
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
    // NaN/Inf inputs would silently produce garbage coefficients
    // through the normal-equations solve, #252.
    if let Some((i, _)) = v.iter().enumerate().find(|(_, c)| !c.is_finite()) {
        return Err(FerrayError::invalid_value(format!(
            "Vandermonde matrix entry at index {i} is not finite"
        )));
    }
    if let Some((i, _)) = y.iter().enumerate().find(|(_, c)| !c.is_finite()) {
        return Err(FerrayError::invalid_value(format!(
            "y has a non-finite value at index {i}"
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
        // #256: weights are squared via sqrt() to build the weighted
        // system; a non-finite or negative weight would silently produce
        // NaN. The is_finite branch also covers #252 for the weight
        // column. Reject up front so the caller gets a useful error
        // instead of "polynomial fit returned NaN coefficients".
        for (i, &wi) in w.iter().enumerate() {
            if !wi.is_finite() || wi < 0.0 {
                return Err(FerrayError::invalid_value(format!(
                    "weights must be non-negative finite numbers; w[{i}] = {wi}"
                )));
            }
        }
    }

    // Build the weighted system: A = sqrt(W) * V, b = sqrt(W) * y
    let mut a_data = vec![0.0; n * m];
    let mut b_data = y.to_vec();
    for i in 0..n {
        let sw = w.map_or(1.0, |w| w[i].sqrt());
        for j in 0..m {
            a_data[i * m + j] = v[i * m + j] * sw;
        }
        b_data[i] *= sw;
    }

    // Keep copies for iterative refinement
    let a_data_copy = a_data.clone();
    let b_data_copy = b_data.clone();

    // Use ferray-linalg's SVD-based lstsq for maximum numerical accuracy.
    // This matches NumPy's polyfit which delegates to numpy.linalg.lstsq.
    let a_arr = Array::<f64, Ix2>::from_vec(Ix2::new([n, m]), a_data)
        .map_err(|e| FerrayError::invalid_value(format!("failed to build A matrix: {e}")))?;
    let b_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), b_data)
        .map_err(|e| FerrayError::invalid_value(format!("failed to build b vector: {e}")))?;

    let (x_arr, _residuals, _rank, _sv) = ferray_linalg::lstsq(&a_arr, &b_arr, None)?;
    let mut x = x_arr.as_slice().unwrap().to_vec();

    // Iterative refinement: compute residual r = b − A*x, solve A*dx = r,
    // update x += dx. Two passes typically recover full f64 precision from
    // the initial SVD solve.
    for _ in 0..2 {
        let mut residual = b_data_copy.clone();
        for i in 0..n {
            let mut ax_i = 0.0;
            for j in 0..m {
                ax_i += a_data_copy[i * m + j] * x[j];
            }
            residual[i] -= ax_i;
        }
        let r_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), residual)
            .map_err(|e| FerrayError::invalid_value(format!("residual build failed: {e}")))?;
        if let Ok((dx_arr, _, _, _)) = ferray_linalg::lstsq(&a_arr, &r_arr, None) {
            let dx = dx_arr.as_slice().unwrap();
            for j in 0..m {
                x[j] += dx[j];
            }
        }
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
    fn fit_weighted_negative_weight_errors() {
        // Issue #256: a negative weight would silently produce NaN via
        // sqrt(); the validation now rejects it up front.
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let w = vec![1.0, -1.0, 1.0];
        let v = power_vandermonde(&x, 1);
        assert!(least_squares_fit(&v, x.len(), 2, &y, Some(&w)).is_err());
    }

    #[test]
    fn fit_weighted_nan_weight_errors() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        let w = vec![1.0, f64::NAN, 1.0];
        let v = power_vandermonde(&x, 1);
        assert!(least_squares_fit(&v, x.len(), 2, &y, Some(&w)).is_err());
    }

    // ----- Non-uniform weighting (#255) ---------------------------------
    //
    // The existing fit_weighted test used uniform [1,1,1,1,1] weights,
    // which exercises only the same code path as unweighted fit. Add
    // tests where the weighting actually affects the solution.

    #[test]
    fn fit_weighted_outlier_suppression() {
        // Inliers on y = 2x + 1; outlier at x=4 (an extreme x) where
        // it has leverage on the slope. Uniform weights skew both
        // intercept and slope; down-weighting the outlier recovers
        // the inlier line.
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_with_outlier = vec![1.0, 3.0, 5.0, 7.0, 100.0];
        let v = power_vandermonde(&x, 1);

        // Uniform weights — outlier at x=4 has leverage; slope should
        // be far from 2.
        let unif = vec![1.0; 5];
        let c_unif = least_squares_fit(&v, x.len(), 2, &y_with_outlier, Some(&unif)).unwrap();
        assert!(
            (c_unif[1] - 2.0).abs() > 5.0,
            "uniform fit should be skewed by leverage outlier; got slope {}",
            c_unif[1]
        );

        // Down-weight the outlier — line snaps back to y = 2x + 1.
        let mut down = vec![1.0; 5];
        down[4] = 1e-9;
        let c_down = least_squares_fit(&v, x.len(), 2, &y_with_outlier, Some(&down)).unwrap();
        assert!(
            (c_down[0] - 1.0).abs() < 1e-3,
            "down-weighted intercept = {} != 1.0",
            c_down[0]
        );
        assert!(
            (c_down[1] - 2.0).abs() < 1e-3,
            "down-weighted slope = {} != 2.0",
            c_down[1]
        );
    }

    #[test]
    fn fit_weighted_zero_weight_excludes_point() {
        // Setting w[i] = 0 must produce the same coefficients as
        // dropping that point from x and y entirely.
        let x_full = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y_full = vec![1.0, 3.0, 999.0, 7.0, 9.0];
        let w_zero = vec![1.0, 1.0, 0.0, 1.0, 1.0];
        let v_full = power_vandermonde(&x_full, 1);
        let c_zero = least_squares_fit(&v_full, 5, 2, &y_full, Some(&w_zero)).unwrap();

        // Compare to fit on only the 4 inliers.
        let x_sub = vec![0.0, 1.0, 3.0, 4.0];
        let y_sub = vec![1.0, 3.0, 7.0, 9.0];
        let v_sub = power_vandermonde(&x_sub, 1);
        let c_sub = least_squares_fit(&v_sub, 4, 2, &y_sub, None).unwrap();

        assert!(
            (c_zero[0] - c_sub[0]).abs() < 1e-10,
            "intercept: weighted={}, dropped={}",
            c_zero[0],
            c_sub[0]
        );
        assert!(
            (c_zero[1] - c_sub[1]).abs() < 1e-10,
            "slope: weighted={}, dropped={}",
            c_zero[1],
            c_sub[1]
        );
    }

    #[test]
    fn fit_weighted_scaling_invariance() {
        // Multiplying all weights by a positive constant must not
        // change the coefficients (the scalar factors out of both
        // sides of the normal equations).
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.5, 2.7, 5.1, 7.4, 9.8];
        let w_unit = vec![0.5, 1.0, 2.0, 1.5, 0.8];
        let w_scaled: Vec<f64> = w_unit.iter().map(|&w| w * 7.5).collect();

        let v = power_vandermonde(&x, 2);
        let c_unit = least_squares_fit(&v, 5, 3, &y, Some(&w_unit)).unwrap();
        let c_scaled = least_squares_fit(&v, 5, 3, &y, Some(&w_scaled)).unwrap();

        for (i, (&a, &b)) in c_unit.iter().zip(c_scaled.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-9,
                "coeff {i}: unit={a}, scaled={b}",
                a = a,
                b = b
            );
        }
    }

    #[test]
    fn fit_weighted_emphasizes_specific_point() {
        // Heavily over-weighting a single point should pull the fit
        // line to pass nearly through that point.
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![10.0, 0.0, 0.0, 0.0]; // the y[0] is a "must hit" target.
        let v = power_vandermonde(&x, 1);

        // Weights pin the target.
        let w = vec![1e6, 1.0, 1.0, 1.0];
        let c = least_squares_fit(&v, 4, 2, &y, Some(&w)).unwrap();

        // Predicted y at x=0 should be near 10.
        let pred_at_0 = c[0]; // intercept.
        assert!(
            (pred_at_0 - 10.0).abs() < 1e-3,
            "high-weight point not pinned: pred(0) = {pred_at_0}"
        );
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
