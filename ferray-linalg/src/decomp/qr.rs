// ferray-linalg: QR decomposition (REQ-9)
//
// Wraps faer::Qr. Supports Reduced and Complete modes.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch;
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

/// Batched QR decomposition for arrays of shape `(..., M, N)`.
///
/// Returns `(Q, R)` stacks with the input's batch dimensions preserved:
/// for `Reduced` mode, `Q` has shape `(..., M, K)` and `R` has shape
/// `(..., K, N)` where `K = min(M, N)`; for `Complete` mode, `Q` has shape
/// `(..., M, M)` and `R` has shape `(..., M, N)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the array has fewer than 2 dims.
pub fn qr_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    mode: QrMode,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, IxDyn>)> {
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "qr_batched: a must have at least 2 dimensions",
        ));
    }
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let (q, r) = qr(&a2, mode)?;
        return Ok((
            Array::from_vec(IxDyn::new(q.shape()), q.iter().copied().collect())?,
            Array::from_vec(IxDyn::new(r.shape()), r.iter().copied().collect())?,
        ));
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let k = m.min(n);
    let (q_m, q_n, r_m, r_n) = match mode {
        QrMode::Reduced => (m, k, k, n),
        QrMode::Complete => (m, m, m, n),
    };

    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    use rayon::prelude::*;
    let results: Vec<FerrayResult<(Vec<T>, Vec<T>)>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), slice.to_vec())?;
            let (q, r) = qr(&mat, mode)?;
            Ok((
                q.iter().copied().collect::<Vec<T>>(),
                r.iter().copied().collect::<Vec<T>>(),
            ))
        })
        .collect();

    let mut q_flat: Vec<T> = Vec::with_capacity(num_batches * q_m * q_n);
    let mut r_flat: Vec<T> = Vec::with_capacity(num_batches * r_m * r_n);
    for res in results {
        let (q, r) = res?;
        q_flat.extend(q);
        r_flat.extend(r);
    }

    let mut q_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
    q_shape.push(q_m);
    q_shape.push(q_n);
    let mut r_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
    r_shape.push(r_m);
    r_shape.push(r_n);
    Ok((
        Array::from_vec(IxDyn::new(&q_shape), q_flat)?,
        Array::from_vec(IxDyn::new(&r_shape), r_flat)?,
    ))
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
                    "Q^T*Q[{i},{j}] = {dot} != {expected}"
                );
            }
        }
    }

    #[test]
    fn qr_batched_reduced_3d() {
        // Two 3x2 matrices; reduced QR gives Q shape (...,3,2), R shape (...,2,2).
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3, 2]), data.clone()).unwrap();
        let (q, r) = qr_batched(&a, QrMode::Reduced).unwrap();
        assert_eq!(q.shape(), &[2, 3, 2]);
        assert_eq!(r.shape(), &[2, 2, 2]);

        // Verify Q @ R ≈ A per batch.
        let q_data: Vec<f64> = q.iter().copied().collect();
        let r_data: Vec<f64> = r.iter().copied().collect();
        for b in 0..2 {
            let q_off = b * 6;
            let r_off = b * 4;
            let a_off = b * 6;
            for i in 0..3 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for p in 0..2 {
                        sum += q_data[q_off + i * 2 + p] * r_data[r_off + p * 2 + j];
                    }
                    assert!(
                        (sum - data[a_off + i * 2 + j]).abs() < 1e-10,
                        "Q@R != A at batch {b} [{i},{j}]"
                    );
                }
            }
        }
    }
}
