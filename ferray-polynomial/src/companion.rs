// ferray-polynomial: Companion matrix construction for root-finding (REQ-12)
//
// The companion matrix of a monic polynomial p(x) = x^n + c_{n-1}x^{n-1} + ... + c_0
// has eigenvalues equal to the roots of p(x).

use ferray_core::error::FerrayError;

/// Build a companion matrix for root-finding from power basis coefficients.
///
/// Given coefficients `[c_0, c_1, ..., c_n]` where `c_n != 0`, constructs
/// the companion matrix whose eigenvalues are the roots of the polynomial.
///
/// The companion matrix is an n x n matrix (where n = degree) of the form:
///
/// ```text
///  [  0   0   ...  0  -c_0/c_n ]
///  [  1   0   ...  0  -c_1/c_n ]
///  [  0   1   ...  0  -c_2/c_n ]
///  [  ...                      ]
///  [  0   0   ...  1  -c_{n-1}/c_n ]
/// ```
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if:
/// - coefficients are empty
/// - leading coefficient is zero
/// - polynomial has degree 0
pub fn companion_matrix(coeffs: &[f64]) -> Result<Vec<f64>, FerrayError> {
    if coeffs.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot build companion matrix from empty coefficients",
        ));
    }

    // Trim trailing near-zero coefficients to find the true degree
    let mut n = coeffs.len();
    while n > 1 && coeffs[n - 1].abs() < f64::EPSILON * 100.0 {
        n -= 1;
    }

    let deg = n - 1;
    if deg == 0 {
        return Err(FerrayError::invalid_value(
            "cannot find roots of a constant polynomial (degree 0)",
        ));
    }

    let leading = coeffs[n - 1];
    if leading.abs() < f64::EPSILON * 100.0 {
        return Err(FerrayError::invalid_value(
            "leading coefficient is effectively zero",
        ));
    }

    // Build companion matrix in row-major order (deg x deg)
    let mut mat = vec![0.0_f64; deg * deg];

    // Sub-diagonal ones
    for i in 1..deg {
        mat[i * deg + (i - 1)] = 1.0;
    }

    // Last column: -c_i / c_n
    for i in 0..deg {
        mat[i * deg + (deg - 1)] = -coeffs[i] / leading;
    }

    Ok(mat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn companion_matrix_quadratic() {
        // p(x) = x^2 - 3x + 2 = (x-1)(x-2), coefficients [2, -3, 1]
        let mat = companion_matrix(&[2.0, -3.0, 1.0]).unwrap();
        // 2x2 matrix
        assert_eq!(mat.len(), 4);
        // Expected companion:
        // [0, -2]
        // [1,  3]
        assert!((mat[0] - 0.0).abs() < 1e-14);
        assert!((mat[1] - (-2.0)).abs() < 1e-14);
        assert!((mat[2] - 1.0).abs() < 1e-14);
        assert!((mat[3] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn companion_matrix_cubic() {
        // p(x) = x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        // coefficients [-6, 11, -6, 1]
        let mat = companion_matrix(&[-6.0, 11.0, -6.0, 1.0]).unwrap();
        assert_eq!(mat.len(), 9); // 3x3
    }

    #[test]
    fn companion_matrix_constant_err() {
        assert!(companion_matrix(&[5.0]).is_err());
    }

    #[test]
    fn companion_matrix_empty_err() {
        assert!(companion_matrix(&[]).is_err());
    }
}
