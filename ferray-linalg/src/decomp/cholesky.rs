// ferray-linalg: Cholesky decomposition (REQ-8)
//
// Wraps faer's LLT decomposition. Returns lower triangular L where A = L L^T.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch::{self, faer_to_vec, slice_to_faer};
use crate::faer_bridge;
use crate::scalar::LinalgFloat;

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower triangular matrix `L` such that `A = L * L^T`.
/// See [`cholesky_upper`] for the `NumPy` `upper=True` variant that returns
/// the upper triangular factor `U` such that `A = U^T * U`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if the matrix is not positive definite.
pub fn cholesky<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix2>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "cholesky requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = faer_bridge::array2_to_faer(a);
    let llt = mat
        .as_ref()
        .llt(faer::Side::Lower)
        .map_err(|_| FerrayError::SingularMatrix {
            message: "matrix is not positive definite".to_string(),
        })?;
    let l = llt.L();
    faer_bridge::faer_to_array2(&l.to_owned())
}

/// Compute the upper-triangular Cholesky decomposition.
///
/// Returns the upper triangular matrix `U` such that `A = U^T * U`.
/// Equivalent to `np.linalg.cholesky(a, upper=True)` (#409).
///
/// Implemented as a transpose of the lower-triangular factor produced by
/// [`cholesky`]: `U = L^T`. The underlying decomposition runs exactly
/// once; only the result is reshaped in a single pass.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if the matrix is not positive definite.
pub fn cholesky_upper<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix2>> {
    let l = cholesky(a)?;
    let n = l.shape()[0];
    let l_data = l.as_slice().ok_or_else(|| {
        FerrayError::invalid_value("cholesky_upper: lower factor was not contiguous")
    })?;
    // U[i][j] = L[j][i] — single-pass swap into a fresh buffer.
    let mut u_data = vec![<T as ferray_core::dtype::Element>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            u_data[i * n + j] = l_data[j * n + i];
        }
    }
    Array::from_vec(Ix2::new([n, n]), u_data)
}

/// Batched Cholesky decomposition for 3D+ arrays.
///
/// Applies Cholesky along the last two dimensions, parallelized via Rayon.
/// Returns lower-triangular factors — see [`cholesky_upper_batched`] for
/// the `NumPy` `upper=True` variant (#564).
pub fn cholesky_batched<T: LinalgFloat>(a: &Array<T, IxDyn>) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let result = cholesky(&a2)?;
        return Array::from_vec(IxDyn::new(shape), result.iter().copied().collect());
    }

    let results = batch::apply_batched_2d(a, |m, n, data| {
        if m != n {
            return Err(FerrayError::shape_mismatch(format!(
                "cholesky requires square matrices, got {m}x{n}"
            )));
        }
        let mat = slice_to_faer(m, n, data);
        let llt = mat
            .as_ref()
            .llt(faer::Side::Lower)
            .map_err(|_| FerrayError::SingularMatrix {
                message: "matrix is not positive definite".to_string(),
            })?;
        Ok(faer_to_vec(&llt.L().to_owned()))
    })?;

    let flat: Vec<T> = results.into_iter().flatten().collect();
    Array::from_vec(IxDyn::new(shape), flat)
}

/// Batched upper-triangular Cholesky decomposition.
///
/// Equivalent to `np.linalg.cholesky(a, upper=True)` on a stack of
/// matrices (`shape = (..., N, N)`). Each batch element is decomposed
/// into the upper triangular factor `U` such that `A = U^T * U`.
///
/// Implemented by calling [`cholesky_batched`] to get the per-batch
/// lower factors and then transposing each matrix in place: for a
/// factor at offset `off` in the flat result, `U[i, j] = L[j, i]` for
/// all (i, j). The underlying decomposition runs exactly once.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last two dims are not square.
/// - `FerrayError::SingularMatrix` if any batch is not positive definite.
pub fn cholesky_upper_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
) -> FerrayResult<Array<T, IxDyn>> {
    let lower = cholesky_batched(a)?;
    let shape = lower.shape().to_vec();
    if shape.len() < 2 {
        return Ok(lower);
    }
    let n = shape[shape.len() - 1];
    // Flat mat_size = n * n. Total batches = product of leading dims.
    let mat_size = n * n;
    let num_batches: usize = shape[..shape.len() - 2].iter().product::<usize>().max(1);

    let lower_data: Vec<T> = lower.iter().copied().collect();
    let mut upper_data = vec![<T as ferray_core::dtype::Element>::zero(); num_batches * mat_size];
    for b in 0..num_batches {
        let off = b * mat_size;
        for i in 0..n {
            for j in 0..n {
                // U[i, j] = L[j, i]
                upper_data[off + i * n + j] = lower_data[off + j * n + i];
            }
        }
    }
    Array::from_vec(IxDyn::new(&shape), upper_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    #[test]
    fn cholesky_2x2() {
        // A = [[4, 2], [2, 3]]
        // L = [[2, 0], [1, sqrt(2)]]
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a).unwrap();
        let ld = l.as_slice().unwrap();

        // Verify L * L^T = A
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += ld[i * n + k] * ld[j * n + k];
                }
                let expected = a.as_slice().unwrap()[i * n + j];
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "L*L^T[{i},{j}] = {sum} != {expected}"
                );
            }
        }
    }

    #[test]
    fn cholesky_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0f32, 2.0, 2.0, 3.0]).unwrap();
        let l = cholesky(&a).unwrap();
        let ld = l.as_slice().unwrap();
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += ld[i * n + k] * ld[j * n + k];
                }
                let expected = a.as_slice().unwrap()[i * n + j];
                assert!(
                    (sum - expected).abs() < 1e-5,
                    "L*L^T[{i},{j}] = {sum} != {expected}"
                );
            }
        }
    }

    #[test]
    fn cholesky_not_positive_definite() {
        // Not positive definite
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![-1.0, 0.0, 0.0, -1.0]).unwrap();
        let result = cholesky(&a);
        assert!(result.is_err());
    }

    #[test]
    fn cholesky_non_square() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(cholesky(&a).is_err());
    }

    // ----- cholesky_batched coverage (#219) -----

    #[test]
    fn cholesky_batched_2d_matches_unbatched() {
        // 2-D input goes through the early-return shortcut.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l_batched = cholesky_batched(&a).unwrap();
        let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let l_unbatched = cholesky(&a2).unwrap();
        for (a, b) in l_batched.iter().zip(l_unbatched.iter()) {
            assert!((a - b).abs() < 1e-12, "2D path mismatch");
        }
    }

    #[test]
    fn cholesky_batched_3d_two_matrices() {
        // Two stacked 2x2 SPD matrices: A1 = [[4,2],[2,3]],
        // A2 = [[9,3],[3,5]]. Verify each batch is correctly factored.
        let data = vec![
            // batch 0
            4.0, 2.0, 2.0, 3.0, // batch 1
            9.0, 3.0, 3.0, 5.0,
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let l = cholesky_batched(&a).unwrap();
        assert_eq!(l.shape(), &[2, 2, 2]);

        // For each batch, reconstruct L*L^T and compare to A.
        let l_data: Vec<f64> = l.iter().copied().collect();
        let a_data: Vec<f64> = a.iter().copied().collect();
        for batch in 0..2 {
            let off = batch * 4;
            for i in 0..2 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += l_data[off + i * 2 + k] * l_data[off + j * 2 + k];
                    }
                    let expected = a_data[off + i * 2 + j];
                    assert!(
                        (sum - expected).abs() < 1e-10,
                        "batch {batch} L*L^T[{i},{j}] = {sum} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_batched_rejects_non_spd_in_one_batch() {
        // First batch SPD, second batch indefinite — should error.
        let data = vec![4.0, 2.0, 2.0, 3.0, -1.0, 0.0, 0.0, -1.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        assert!(cholesky_batched(&a).is_err());
    }

    // ---- cholesky_upper (#409) ----

    #[test]
    fn cholesky_upper_2x2_reconstructs_input() {
        // A = [[4, 2], [2, 3]]; U such that U^T * U == A.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let u = cholesky_upper(&a).unwrap();
        let ud = u.as_slice().unwrap();
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                // (U^T * U)[i, j] = sum_k U[k, i] * U[k, j]
                let mut sum = 0.0;
                for k in 0..n {
                    sum += ud[k * n + i] * ud[k * n + j];
                }
                let expected = a.as_slice().unwrap()[i * n + j];
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "U^T*U[{i},{j}] = {sum} != {expected}"
                );
            }
        }
    }

    #[test]
    fn cholesky_upper_is_transpose_of_lower() {
        // L from the default cholesky and U from cholesky_upper must
        // satisfy U[i, j] == L[j, i] exactly.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![
                4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0, //
            ],
        )
        .unwrap();
        let l = cholesky(&a).unwrap();
        let u = cholesky_upper(&a).unwrap();
        let ld = l.as_slice().unwrap();
        let ud = u.as_slice().unwrap();
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (ud[i * n + j] - ld[j * n + i]).abs() < 1e-14,
                    "U[{i},{j}]={} != L[{j},{i}]={}",
                    ud[i * n + j],
                    ld[j * n + i]
                );
            }
        }
    }

    #[test]
    fn cholesky_upper_is_upper_triangular() {
        // All entries strictly below the diagonal must be zero.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0],
        )
        .unwrap();
        let u = cholesky_upper(&a).unwrap();
        let ud = u.as_slice().unwrap();
        for i in 1..3 {
            for j in 0..i {
                assert!(
                    ud[i * 3 + j].abs() < 1e-14,
                    "U[{i},{j}]={} should be zero",
                    ud[i * 3 + j]
                );
            }
        }
    }

    #[test]
    fn cholesky_upper_rejects_non_square() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(cholesky_upper(&a).is_err());
    }

    #[test]
    fn cholesky_upper_rejects_non_positive_definite() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![-1.0, 0.0, 0.0, -1.0]).unwrap();
        assert!(cholesky_upper(&a).is_err());
    }

    // ---- cholesky_upper_batched (#564) ----

    #[test]
    fn cholesky_upper_batched_stack_reconstructs_each_batch() {
        // Two stacked SPD matrices: A1 = [[4,2],[2,3]], A2 = [[9,3],[3,5]].
        // For each batch, verify U^T * U == A.
        let data = vec![4.0, 2.0, 2.0, 3.0, 9.0, 3.0, 3.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data.clone()).unwrap();
        let u = cholesky_upper_batched(&a).unwrap();
        assert_eq!(u.shape(), &[2, 2, 2]);

        let u_data: Vec<f64> = u.iter().copied().collect();
        for batch in 0..2 {
            let off = batch * 4;
            for i in 0..2 {
                for j in 0..2 {
                    // (U^T * U)[i, j] = sum_k U[k, i] * U[k, j]
                    let mut sum = 0.0;
                    for k in 0..2 {
                        sum += u_data[off + k * 2 + i] * u_data[off + k * 2 + j];
                    }
                    let expected = data[off + i * 2 + j];
                    assert!(
                        (sum - expected).abs() < 1e-10,
                        "batch {batch} U^T*U[{i},{j}] = {sum} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_upper_batched_2d_input_matches_unbatched() {
        // 2-D input routes through the early-return path in
        // cholesky_batched. Verify it still produces the correct U.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let u_batched = cholesky_upper_batched(&a).unwrap();
        let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![4.0, 2.0, 2.0, 3.0]).unwrap();
        let u_unbatched = cholesky_upper(&a2).unwrap();
        for (a, b) in u_batched.iter().zip(u_unbatched.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn cholesky_upper_batched_is_transpose_of_lower_batched() {
        // U from cholesky_upper_batched must be the per-batch transpose
        // of L from cholesky_batched.
        let data = vec![4.0, 2.0, 2.0, 3.0, 9.0, 3.0, 3.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let l = cholesky_batched(&a).unwrap();
        let u = cholesky_upper_batched(&a).unwrap();
        let l_data: Vec<f64> = l.iter().copied().collect();
        let u_data: Vec<f64> = u.iter().copied().collect();
        for batch in 0..2 {
            let off = batch * 4;
            for i in 0..2 {
                for j in 0..2 {
                    assert!(
                        (u_data[off + i * 2 + j] - l_data[off + j * 2 + i]).abs() < 1e-14,
                        "batch {batch} U[{i},{j}] != L[{j},{i}]"
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_upper_batched_each_block_is_upper_triangular() {
        let data = vec![
            4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0, // 3x3 batch 0
            9.0, 3.0, 0.0, 3.0, 5.0, 1.0, 0.0, 1.0, 4.0, // 3x3 batch 1
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3, 3]), data).unwrap();
        let u = cholesky_upper_batched(&a).unwrap();
        let u_data: Vec<f64> = u.iter().copied().collect();
        for batch in 0..2 {
            let off = batch * 9;
            for i in 1..3 {
                for j in 0..i {
                    assert!(
                        u_data[off + i * 3 + j].abs() < 1e-14,
                        "batch {batch} U[{i},{j}]={} should be zero",
                        u_data[off + i * 3 + j]
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_upper_batched_rejects_non_spd_in_one_batch() {
        let data = vec![4.0, 2.0, 2.0, 3.0, -1.0, 0.0, 0.0, -1.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        assert!(cholesky_upper_batched(&a).is_err());
    }
}
