// ferray-linalg: SVD decomposition (REQ-10)
//
// Wraps faer::Svd. Returns (U, S, Vt).

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::faer_bridge;

/// Compute the Singular Value Decomposition of a matrix.
///
/// Returns `(U, S, Vt)` where `A = U * diag(S) * Vt`.
///
/// If `full_matrices` is true, U is (m, m) and Vt is (n, n).
/// Otherwise, U is (m, min(m,n)) and Vt is (min(m,n), n).
///
/// S is always a 1D array of length min(m, n) in nonincreasing order.
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD computation fails to converge.
pub fn svd(
    a: &Array<f64, Ix2>,
    full_matrices: bool,
) -> FerrayResult<(Array<f64, Ix2>, Array<f64, Ix1>, Array<f64, Ix2>)> {
    let mat = faer_bridge::array2_to_faer(a);

    let decomp = if full_matrices {
        mat.as_ref().svd()
    } else {
        mat.as_ref().thin_svd()
    };

    let decomp = decomp.map_err(|e| FerrayError::InvalidValue {
        message: format!("SVD failed to converge: {e:?}"),
    })?;

    let u = decomp.U().to_owned();
    let v = decomp.V().to_owned();
    let s_diag = decomp.S();

    // S is a diagonal; extract values
    let min_dim = a.shape()[0].min(a.shape()[1]);
    let mut s_vals = Vec::with_capacity(min_dim);
    for i in 0..min_dim {
        s_vals.push(s_diag.column_vector()[i]);
    }

    // V from faer is V, but numpy returns Vt = V^T
    let (vn, vk) = v.shape();
    let mut vt_data = Vec::with_capacity(vk * vn);
    for i in 0..vk {
        for j in 0..vn {
            vt_data.push(v[(j, i)]);
        }
    }

    let u_arr = faer_bridge::faer_to_array2(&u)?;
    let s_arr = Array::from_vec(Ix1::new([s_vals.len()]), s_vals)?;
    let vt_arr = Array::from_vec(Ix2::new([vk, vn]), vt_data)?;

    Ok((u_arr, s_arr, vt_arr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    #[test]
    fn svd_thin_reconstructs() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let (u, s, vt) = svd(&a, false).unwrap();

        let us = u.as_slice().unwrap();
        let ss = s.as_slice().unwrap();
        let vts = vt.as_slice().unwrap();

        let m = 3;
        let n = 2;
        let k = ss.len(); // min(3,2)=2

        // Reconstruct A = U * diag(S) * Vt
        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0;
                for p in 0..k {
                    val += us[i * k + p] * ss[p] * vts[p * n + j];
                }
                let expected = a.as_slice().unwrap()[i * n + j];
                assert!(
                    (val - expected).abs() < 1e-10,
                    "U*S*Vt[{},{}] = {} != {}",
                    i,
                    j,
                    val,
                    expected
                );
            }
        }
    }

    #[test]
    fn svd_full_shapes() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let (u, s, vt) = svd(&a, true).unwrap();
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vt.shape(), &[2, 2]);
    }

    #[test]
    fn svd_singular_values_nonnegative() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let (_u, s, _vt) = svd(&a, false).unwrap();
        for &val in s.as_slice().unwrap() {
            assert!(val >= 0.0, "singular value {} should be >= 0", val);
        }
    }
}
