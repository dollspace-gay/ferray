// ferray-linalg: Eigendecomposition (REQ-11, REQ-12, REQ-13)
//
// eig, eigh, eigvals, eigvalsh via faer.
// eig returns complex eigenvalues/eigenvectors for general square matrices.
// eigh returns real eigenvalues for symmetric/Hermitian matrices.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_core::error::{FerrayError, FerrayResult};
use num_complex::Complex;

use crate::faer_bridge;
use crate::scalar::LinalgFloat;

/// Compute eigenvalues and right eigenvectors of a general square matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues is a 1D array
/// of `Complex<T>` and eigenvectors is a 2D array of `Complex<T>`.
/// The columns of eigenvectors are the right eigenvectors.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if eigendecomposition fails.
pub fn eig<T: LinalgFloat>(
    a: &Array<T, Ix2>,
) -> FerrayResult<(Array<Complex<T>, Ix1>, Array<Complex<T>, Ix2>)>
where
    Complex<T>: ferray_core::dtype::Element,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "eig requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat
        .as_ref()
        .eigen()
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("eigendecomposition failed: {e:?}"),
        })?;

    // Extract eigenvalues from Diag
    let s = decomp.S();
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(s.column_vector()[i]);
    }

    // Extract eigenvectors
    let u = decomp.U();
    let mut eigvecs = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            eigvecs.push(u[(i, j)]);
        }
    }

    let vals = Array::from_vec(Ix1::new([n]), eigenvalues)?;
    let vecs = Array::from_vec(Ix2::new([n, n]), eigvecs)?;
    Ok((vals, vecs))
}

/// Compute eigenvalues and eigenvectors of a symmetric (Hermitian) matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues is a 1D array
/// of real `T` in nondecreasing order, and eigenvectors is a 2D real matrix
/// whose columns are orthonormal eigenvectors.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if eigendecomposition fails.
pub fn eigh<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<(Array<T, Ix1>, Array<T, Ix2>)> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "eigh requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];
    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat
        .as_ref()
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("symmetric eigendecomposition failed: {e:?}"),
        })?;

    let s = decomp.S();
    let mut eigenvalues = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues.push(s.column_vector()[i]);
    }

    let u = decomp.U();
    let u_owned = u.to_owned();
    let vecs = faer_bridge::faer_to_array2(&u_owned)?;

    let vals = Array::from_vec(Ix1::new([n]), eigenvalues)?;
    Ok((vals, vecs))
}

/// Compute eigenvalues of a general square matrix.
///
/// Returns a 1D array of `Complex<T>` eigenvalues.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if computation fails.
pub fn eigvals<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<Complex<T>, Ix1>>
where
    Complex<T>: ferray_core::dtype::Element,
{
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "eigvals requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = faer_bridge::array2_to_faer(a);
    let vals = mat
        .as_ref()
        .eigenvalues()
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("eigenvalue computation failed: {e:?}"),
        })?;
    Array::from_vec(Ix1::new([vals.len()]), vals)
}

/// Compute eigenvalues of a symmetric (Hermitian) matrix.
///
/// Returns a 1D array of real `T` eigenvalues in nondecreasing order.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if computation fails.
pub fn eigvalsh<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix1>> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "eigvalsh requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let mat = faer_bridge::array2_to_faer(a);
    let vals = mat
        .as_ref()
        .self_adjoint_eigenvalues(faer::Side::Lower)
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("symmetric eigenvalue computation failed: {e:?}"),
        })?;
    Array::from_vec(Ix1::new([vals.len()]), vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    #[test]
    fn eigh_symmetric_2x2() {
        // Symmetric matrix [[2, 1], [1, 2]]
        // Eigenvalues: 1, 3
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (vals, vecs) = eigh(&a).unwrap();

        let vs = vals.as_slice().unwrap();
        assert!((vs[0] - 1.0).abs() < 1e-10);
        assert!((vs[1] - 3.0).abs() < 1e-10);

        // Check orthogonality of eigenvectors
        let es = vecs.as_slice().unwrap();
        let dot = es[0] * es[1] + es[2] * es[3];
        assert!(dot.abs() < 1e-10);
    }

    #[test]
    fn eigh_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0f32, 1.0, 1.0, 2.0]).unwrap();
        let (vals, _vecs) = eigh(&a).unwrap();
        let vs = vals.as_slice().unwrap();
        assert!((vs[0] - 1.0f32).abs() < 1e-5);
        assert!((vs[1] - 3.0f32).abs() < 1e-5);
    }

    #[test]
    fn eigvals_3x3() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        )
        .unwrap();
        let vals = eigvals(&a).unwrap();
        let mut real_parts: Vec<f64> = vals.iter().map(|c| c.re).collect();
        real_parts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((real_parts[0] - 1.0).abs() < 1e-10);
        assert!((real_parts[1] - 2.0).abs() < 1e-10);
        assert!((real_parts[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn eigvalsh_sorted() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
        )
        .unwrap();
        let vals = eigvalsh(&a).unwrap();
        let vs = vals.as_slice().unwrap();
        // Should be sorted nondecreasing
        assert!(vs[0] <= vs[1]);
        assert!(vs[1] <= vs[2]);
    }

    #[test]
    fn eig_non_square_error() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(eig(&a).is_err());
    }
}
