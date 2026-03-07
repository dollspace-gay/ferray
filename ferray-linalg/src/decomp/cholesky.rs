// ferray-linalg: Cholesky decomposition (REQ-8)
//
// Wraps faer's LLT decomposition. Returns lower triangular L where A = L L^T.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch::{self, faer_to_vec, slice_to_faer};
use crate::faer_bridge;

/// Compute the Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower triangular matrix `L` such that `A = L * L^T`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the matrix is not square.
/// - `FerrayError::SingularMatrix` if the matrix is not positive definite.
pub fn cholesky(a: &Array<f64, Ix2>) -> FerrayResult<Array<f64, Ix2>> {
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

/// Batched Cholesky decomposition for 3D+ arrays.
///
/// Applies Cholesky along the last two dimensions, parallelized via Rayon.
pub fn cholesky_batched(a: &Array<f64, IxDyn>) -> FerrayResult<Array<f64, IxDyn>> {
    let shape = a.shape();
    if shape.len() == 2 {
        let a2 = Array::<f64, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            a.iter().copied().collect(),
        )?;
        let result = cholesky(&a2)?;
        return Array::from_vec(IxDyn::new(shape), result.iter().copied().collect());
    }

    let results = batch::apply_batched_2d(a, |m, n, data| {
        if m != n {
            return Err(FerrayError::shape_mismatch(format!(
                "cholesky requires square matrices, got {}x{}",
                m, n
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

    let flat: Vec<f64> = results.into_iter().flatten().collect();
    Array::from_vec(IxDyn::new(shape), flat)
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
                    "L*L^T[{},{}] = {} != {}",
                    i,
                    j,
                    sum,
                    expected
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
}
