// ferray-linalg: QR decomposition (REQ-9)
//
// Wraps faer::Qr. Supports Reduced and Complete modes.

use ferray_core::array::owned::Array;
use ferray_core::dimension::Ix2;
use ferray_core::error::FerrayResult;

use crate::faer_bridge;
use crate::scalar::LinalgFloat;

/// Mode for QR decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QrMode {
    /// Reduced QR: Q is (m, min(m,n)), R is (min(m,n), n).
    Reduced,
    /// Complete QR: Q is (m, m), R is (m, n).
    Complete,
}

/// Compute the QR decomposition of a matrix.
///
/// Returns `(Q, R)` where `A = Q * R`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the input is not 2D.
pub fn qr<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    mode: QrMode,
) -> FerrayResult<(Array<T, Ix2>, Array<T, Ix2>)> {
    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat.as_ref().qr();

    match mode {
        QrMode::Reduced => {
            let q = decomp.compute_thin_Q();
            let r_mat = decomp.thin_R().to_owned();
            let q_arr = faer_bridge::faer_to_array2(&q)?;
            let r_arr = faer_bridge::faer_to_array2(&r_mat)?;
            Ok((q_arr, r_arr))
        }
        QrMode::Complete => {
            let q = decomp.compute_Q();
            let r_full = decomp.R().to_owned();
            let q_arr = faer_bridge::faer_to_array2(&q)?;
            let r_arr = faer_bridge::faer_to_array2(&r_full)?;
            Ok((q_arr, r_arr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    fn matmul_check(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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
    fn qr_reduced_reconstructs() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([4, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 1.0],
        )
        .unwrap();
        let (q, r) = qr(&a, QrMode::Reduced).unwrap();

        let qs = q.as_slice().unwrap();
        let rs = r.as_slice().unwrap();
        let (m, n) = (4, 3);
        let k = q.shape()[1]; // min(m,n)=3
        let reconstructed = matmul_check(qs, rs, m, k, n);

        let orig = a.as_slice().unwrap();
        for i in 0..m * n {
            assert!(
                (reconstructed[i] - orig[i]).abs() < 1e-10,
                "Q*R[{}] = {} != {}",
                i,
                reconstructed[i],
                orig[i]
            );
        }
    }

    #[test]
    fn qr_f32() {
        let a = Array::<f32, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        )
        .unwrap();
        let (q, r) = qr(&a, QrMode::Reduced).unwrap();
        assert_eq!(q.shape(), &[3, 3]);
        assert_eq!(r.shape(), &[3, 3]);
    }

    #[test]
    fn qr_complete_q_is_square() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([4, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 2.0, 1.0, 1.0],
        )
        .unwrap();
        let (q, _r) = qr(&a, QrMode::Complete).unwrap();
        assert_eq!(q.shape(), &[4, 4]);
    }

    #[test]
    fn qr_q_orthogonal() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
        )
        .unwrap();
        let (q, _r) = qr(&a, QrMode::Complete).unwrap();
        let qs = q.as_slice().unwrap();
        let n = 3;
        // Check Q^T * Q ≈ I
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += qs[k * n + i] * qs[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "Q^T*Q[{},{}] = {} != {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }
}
