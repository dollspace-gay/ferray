// ferray-linalg: LU decomposition (REQ-14)
//
// Wraps faer::PartialPivLu. Returns (P, L, U) with partial pivoting.

use ferray_core::array::owned::Array;
use ferray_core::dimension::Ix2;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::faer_bridge;

/// Compute the LU decomposition with partial pivoting.
///
/// Returns `(P, L, U)` such that `P @ A = L @ U` where:
/// - `P` is a permutation matrix (m x m)
/// - `L` is lower triangular with unit diagonal (m x min(m,n))
/// - `U` is upper triangular (min(m,n) x n)
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not at least 1x1.
pub fn lu(
    a: &Array<f64, Ix2>,
) -> FerrayResult<(Array<f64, Ix2>, Array<f64, Ix2>, Array<f64, Ix2>)> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    if m == 0 || n == 0 {
        return Err(FerrayError::shape_mismatch("LU requires non-empty matrix"));
    }

    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat.as_ref().partial_piv_lu();

    let l_mat = decomp.L().to_owned();
    let u_mat = decomp.U().to_owned();
    let perm = decomp.P();

    // Build permutation matrix from the permutation
    let perm_fwd = perm.arrays().0;
    let mut p_data = vec![0.0; m * m];
    for i in 0..m {
        let j = perm_fwd[i];
        p_data[i * m + j] = 1.0;
    }

    let p_arr = Array::from_vec(Ix2::new([m, m]), p_data)?;
    let l_arr = faer_bridge::faer_to_array2(&l_mat)?;
    let u_arr = faer_bridge::faer_to_array2(&u_mat)?;

    Ok((p_arr, l_arr, u_arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        c
    }

    #[test]
    fn lu_3x3_reconstructs() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
        )
        .unwrap();
        let (p, l, u) = lu(&a).unwrap();

        let ps = p.as_slice().unwrap();
        let ls = l.as_slice().unwrap();
        let us = u.as_slice().unwrap();

        // P * A = L * U
        let n = 3;
        let pa = matmul(ps, a.as_slice().unwrap(), n, n, n);
        let lu_result = matmul(ls, us, l.shape()[0], l.shape()[1], u.shape()[1]);

        for i in 0..n * n {
            assert!(
                (pa[i] - lu_result[i]).abs() < 1e-10,
                "PA[{}] = {} != LU[{}] = {}",
                i,
                pa[i],
                i,
                lu_result[i]
            );
        }
    }
}
