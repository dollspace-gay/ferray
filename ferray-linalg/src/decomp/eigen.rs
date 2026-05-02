// ferray-linalg: Eigendecomposition (REQ-11, REQ-12, REQ-13)
//
// eig, eigh, eigvals, eigvalsh via faer.
// eig returns complex eigenvalues/eigenvectors for general square matrices.
// eigh returns real eigenvalues for symmetric/Hermitian matrices.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use num_complex::Complex;

use crate::batch;
use crate::faer_bridge;
use crate::scalar::LinalgFloat;

/// Which triangle of a symmetric/Hermitian matrix to read during
/// [`eigh`] / [`eigvalsh`] and their batched / UPLO-explicit variants.
///
/// Matches `NumPy`'s `UPLO='L'` / `UPLO='U'` parameter on
/// `np.linalg.eigh` (#410). The lower triangle is the default — it
/// matches ferray's historical behaviour and `NumPy`'s default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPLO {
    /// Read the lower triangle (default). Equivalent to `UPLO='L'`.
    Lower,
    /// Read the upper triangle. Equivalent to `UPLO='U'`.
    Upper,
}

impl UPLO {
    /// Translate to the faer `Side` enum used by the underlying solver.
    #[inline]
    const fn to_faer_side(self) -> faer::Side {
        match self {
            Self::Lower => faer::Side::Lower,
            Self::Upper => faer::Side::Upper,
        }
    }
}

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
/// Reads the lower triangle — see [`eigh_uplo`] for the
/// upper-triangle variant.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if eigendecomposition fails.
pub fn eigh<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<(Array<T, Ix1>, Array<T, Ix2>)> {
    eigh_uplo(a, UPLO::Lower)
}

/// Compute eigenvalues and eigenvectors of a symmetric (Hermitian) matrix,
/// reading the specified triangle of the input.
///
/// Equivalent to `np.linalg.eigh(a, UPLO=...)` (#410). The lower
/// triangle is ferray's historical default (and also `NumPy`'s default);
/// pass [`UPLO::Upper`] to read the upper triangle instead.
///
/// For a truly-symmetric matrix the two triangles should produce
/// numerically-identical results; the parameter matters when the input
/// is "assumed symmetric" and the two halves differ (e.g. a matrix
/// built by filling only one triangle and leaving the other untouched).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if eigendecomposition fails.
pub fn eigh_uplo<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    uplo: UPLO,
) -> FerrayResult<(Array<T, Ix1>, Array<T, Ix2>)> {
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
        .self_adjoint_eigen(uplo.to_faer_side())
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
/// Reads the lower triangle — see [`eigvalsh_uplo`] for the
/// upper-triangle variant.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if computation fails.
pub fn eigvalsh<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix1>> {
    eigvalsh_uplo(a, UPLO::Lower)
}

/// Compute eigenvalues of a symmetric (Hermitian) matrix, reading the
/// specified triangle.
///
/// Equivalent to `np.linalg.eigvalsh(a, UPLO=...)` (#410).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if matrix is not square.
/// - `FerrayError::InvalidValue` if computation fails.
pub fn eigvalsh_uplo<T: LinalgFloat>(a: &Array<T, Ix2>, uplo: UPLO) -> FerrayResult<Array<T, Ix1>> {
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
        .self_adjoint_eigenvalues(uplo.to_faer_side())
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("symmetric eigenvalue computation failed: {e:?}"),
        })?;
    Array::from_vec(Ix1::new([vals.len()]), vals)
}

/// Batched symmetric eigendecomposition for arrays of shape `(..., N, N)`.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues has shape
/// `(..., N)` and eigenvectors has shape `(..., N, N)`, matching `NumPy`.
/// Reads the lower triangle of each matrix — see [`eigh_batched_uplo`]
/// for the UPLO-explicit variant (#564).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last two dims are not square.
/// - `FerrayError::InvalidValue` if decomposition fails for any batch.
pub fn eigh_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, IxDyn>)> {
    eigh_batched_uplo(a, UPLO::Lower)
}

/// Batched symmetric eigendecomposition with an explicit [`UPLO`] choice.
///
/// Equivalent to `np.linalg.eigh(a, UPLO=...)` on a stack of matrices.
/// Each batch element reads the specified triangle via
/// [`eigh_uplo`], so any asymmetry between the two halves of the input
/// affects every batch element identically.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last two dims are not square.
/// - `FerrayError::InvalidValue` if decomposition fails for any batch.
pub fn eigh_batched_uplo<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    uplo: UPLO,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, IxDyn>)> {
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "eigh_batched: a must have at least 2 dimensions",
        ));
    }
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let (vals, vecs) = eigh_uplo(&a2, uplo)?;
        return Ok((
            Array::from_vec(IxDyn::new(vals.shape()), vals.iter().copied().collect())?,
            Array::from_vec(IxDyn::new(vecs.shape()), vecs.iter().copied().collect())?,
        ));
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let n = shape[shape.len() - 1];
    if shape[shape.len() - 2] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "eigh_batched: matrices must be square, got {}x{}",
            shape[shape.len() - 2],
            n
        )));
    }

    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = n * n;

    use rayon::prelude::*;
    let results: Vec<FerrayResult<(Vec<T>, Vec<T>)>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([n, n]), slice.to_vec())?;
            let (vals, vecs) = eigh_uplo(&mat, uplo)?;
            Ok((
                vals.iter().copied().collect::<Vec<T>>(),
                vecs.iter().copied().collect::<Vec<T>>(),
            ))
        })
        .collect();

    let mut vals_flat: Vec<T> = Vec::with_capacity(num_batches * n);
    let mut vecs_flat: Vec<T> = Vec::with_capacity(num_batches * n * n);
    for res in results {
        let (vs, ve) = res?;
        vals_flat.extend(vs);
        vecs_flat.extend(ve);
    }

    let mut vals_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
    vals_shape.push(n);
    let vecs_shape: Vec<usize> = shape.to_vec();
    Ok((
        Array::from_vec(IxDyn::new(&vals_shape), vals_flat)?,
        Array::from_vec(IxDyn::new(&vecs_shape), vecs_flat)?,
    ))
}

/// Batched symmetric eigenvalues-only for arrays of shape `(..., N, N)`.
///
/// Returns an array with shape `(..., N)` containing eigenvalues in
/// ascending order per batch element. Reads the lower triangle — see
/// [`eigvalsh_batched_uplo`] for the UPLO-explicit variant (#564).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the last two dims are not square.
/// - `FerrayError::InvalidValue` if computation fails for any batch.
pub fn eigvalsh_batched<T: LinalgFloat>(a: &Array<T, IxDyn>) -> FerrayResult<Array<T, IxDyn>> {
    eigvalsh_batched_uplo(a, UPLO::Lower)
}

/// Batched symmetric eigenvalues-only with an explicit [`UPLO`] choice.
pub fn eigvalsh_batched_uplo<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    uplo: UPLO,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "eigvalsh_batched: a must have at least 2 dimensions",
        ));
    }
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let vals = eigvalsh_uplo(&a2, uplo)?;
        return Array::from_vec(IxDyn::new(vals.shape()), vals.iter().copied().collect());
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let n = shape[shape.len() - 1];
    if shape[shape.len() - 2] != n {
        return Err(FerrayError::shape_mismatch(format!(
            "eigvalsh_batched: matrices must be square, got {}x{}",
            shape[shape.len() - 2],
            n
        )));
    }

    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = n * n;

    use rayon::prelude::*;
    let results: Vec<FerrayResult<Vec<T>>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([n, n]), slice.to_vec())?;
            let vals = eigvalsh_uplo(&mat, uplo)?;
            Ok(vals.iter().copied().collect::<Vec<T>>())
        })
        .collect();

    let mut flat: Vec<T> = Vec::with_capacity(num_batches * n);
    for res in results {
        flat.extend(res?);
    }

    let mut out_shape: Vec<usize> = shape[..shape.len() - 2].to_vec();
    out_shape.push(n);
    Array::from_vec(IxDyn::new(&out_shape), flat)
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
        let dot = es[0].mul_add(es[1], es[2] * es[3]);
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

    // ---- Batched variants (#412) ----

    #[test]
    fn eigh_batched_stack() {
        // Two 2x2 symmetric matrices.
        // [[2,1],[1,2]] -> eigenvalues {1, 3}
        // [[4,0],[0,5]] -> eigenvalues {4, 5}
        let data = vec![2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let (vals, vecs) = eigh_batched(&a).unwrap();
        assert_eq!(vals.shape(), &[2, 2]);
        assert_eq!(vecs.shape(), &[2, 2, 2]);

        let v = vals.as_slice().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 3.0).abs() < 1e-10);
        assert!((v[2] - 4.0).abs() < 1e-10);
        assert!((v[3] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn eigvalsh_batched_stack() {
        let data = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let vals = eigvalsh_batched(&a).unwrap();
        assert_eq!(vals.shape(), &[2, 2]);
        let v: Vec<f64> = vals.iter().copied().collect();
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[1] - 2.0).abs() < 1e-10);
        assert!((v[2] - 3.0).abs() < 1e-10);
        assert!((v[3] - 4.0).abs() < 1e-10);
    }

    // ---- UPLO parameter (#410) ----

    #[test]
    fn eigh_uplo_lower_matches_default_eigh() {
        // The default eigh always uses Lower; eigh_uplo(a, Lower) must
        // produce identical results.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (v_default, e_default) = eigh(&a).unwrap();
        let (v_lower, e_lower) = eigh_uplo(&a, UPLO::Lower).unwrap();
        assert_eq!(v_default.as_slice().unwrap(), v_lower.as_slice().unwrap());
        assert_eq!(e_default.as_slice().unwrap(), e_lower.as_slice().unwrap());
    }

    #[test]
    fn eigh_uplo_symmetric_input_same_for_both_triangles() {
        // For a truly-symmetric input the two triangles carry identical
        // data, so eigh_uplo(Upper) and eigh_uplo(Lower) must produce
        // the same eigenvalues (up to numerical noise).
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0],
        )
        .unwrap();
        let (v_lower, _) = eigh_uplo(&a, UPLO::Lower).unwrap();
        let (v_upper, _) = eigh_uplo(&a, UPLO::Upper).unwrap();
        for (l, u) in v_lower
            .as_slice()
            .unwrap()
            .iter()
            .zip(v_upper.as_slice().unwrap())
        {
            assert!((l - u).abs() < 1e-10, "lower {l} vs upper {u}");
        }
    }

    #[test]
    fn eigh_uplo_asymmetric_lower_vs_upper_differ() {
        // An asymmetric matrix: Lower reads only the lower triangle
        // (interpreting the rest as its mirror); Upper reads only the
        // upper triangle. The two should produce different eigenvalues
        // because they're actually factoring different symmetric
        // matrices derived from the input.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![
                3.0, 10.0, // Lower reads (0,0)=3 and (1,0)=0
                0.0, 5.0, // Upper reads (0,0)=3, (0,1)=10, (1,1)=5
            ],
        )
        .unwrap();
        let (v_lower, _) = eigh_uplo(&a, UPLO::Lower).unwrap();
        let (v_upper, _) = eigh_uplo(&a, UPLO::Upper).unwrap();
        // Lower triangle = [[3, 0], [0, 5]] → eigenvalues {3, 5}
        assert!((v_lower.as_slice().unwrap()[0] - 3.0).abs() < 1e-10);
        assert!((v_lower.as_slice().unwrap()[1] - 5.0).abs() < 1e-10);
        // Upper triangle = [[3, 10], [10, 5]] → eigenvalues != {3, 5}
        let vu = v_upper.as_slice().unwrap();
        assert!(
            (vu[0] - 3.0).abs() > 0.1 || (vu[1] - 5.0).abs() > 0.1,
            "upper eigenvalues should differ from lower: got {vu:?}"
        );
    }

    #[test]
    fn eigvalsh_uplo_lower_matches_default() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let v_default = eigvalsh(&a).unwrap();
        let v_lower = eigvalsh_uplo(&a, UPLO::Lower).unwrap();
        assert_eq!(v_default.as_slice().unwrap(), v_lower.as_slice().unwrap());
    }

    #[test]
    fn eigvalsh_uplo_symmetric_upper_matches_lower() {
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0],
        )
        .unwrap();
        let v_lower = eigvalsh_uplo(&a, UPLO::Lower).unwrap();
        let v_upper = eigvalsh_uplo(&a, UPLO::Upper).unwrap();
        for (l, u) in v_lower
            .as_slice()
            .unwrap()
            .iter()
            .zip(v_upper.as_slice().unwrap())
        {
            assert!((l - u).abs() < 1e-10);
        }
    }

    #[test]
    fn eigh_uplo_rejects_non_square() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(eigh_uplo(&a, UPLO::Lower).is_err());
        assert!(eigh_uplo(&a, UPLO::Upper).is_err());
    }

    #[test]
    fn eigvalsh_uplo_rejects_non_square() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(eigvalsh_uplo(&a, UPLO::Lower).is_err());
        assert!(eigvalsh_uplo(&a, UPLO::Upper).is_err());
    }

    // ---- batched UPLO (#564) ----

    #[test]
    fn eigh_batched_uplo_lower_matches_default() {
        // Stack of two symmetric matrices; Lower variant matches the
        // default (non-UPLO) eigh_batched bit-exact.
        let data = vec![2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let (v_default, e_default) = eigh_batched(&a).unwrap();
        let (v_lower, e_lower) = eigh_batched_uplo(&a, UPLO::Lower).unwrap();
        assert_eq!(v_default.as_slice().unwrap(), v_lower.as_slice().unwrap());
        assert_eq!(e_default.as_slice().unwrap(), e_lower.as_slice().unwrap());
    }

    #[test]
    fn eigh_batched_uplo_symmetric_input_same_for_both_triangles() {
        // Truly-symmetric batched input: Upper and Lower produce the
        // same eigenvalues within numerical noise.
        let data = vec![2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let (v_lower, _) = eigh_batched_uplo(&a, UPLO::Lower).unwrap();
        let (v_upper, _) = eigh_batched_uplo(&a, UPLO::Upper).unwrap();
        for (l, u) in v_lower.iter().copied().zip(v_upper.iter().copied()) {
            assert!((l - u).abs() < 1e-10);
        }
    }

    #[test]
    fn eigh_batched_uplo_asymmetric_lower_differs_from_upper() {
        // Each batch element is asymmetric: Lower and Upper read
        // different data, so they should differ in at least one batch.
        //
        // Batch 0: [[3, 10], [0, 5]]
        //   Lower sees [[3, 0], [0, 5]] → eigenvalues {3, 5}
        //   Upper sees [[3, 10], [10, 5]] → eigenvalues different
        // Batch 1: same shape pattern.
        let data = vec![3.0, 10.0, 0.0, 5.0, 3.0, 10.0, 0.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let (v_lower, _) = eigh_batched_uplo(&a, UPLO::Lower).unwrap();
        let (v_upper, _) = eigh_batched_uplo(&a, UPLO::Upper).unwrap();
        let vl = v_lower.as_slice().unwrap();
        let vu = v_upper.as_slice().unwrap();
        // Lower → {3, 5} per batch
        assert!((vl[0] - 3.0).abs() < 1e-10);
        assert!((vl[1] - 5.0).abs() < 1e-10);
        // Upper must differ from Lower somewhere
        let any_diff = vl.iter().zip(vu.iter()).any(|(l, u)| (l - u).abs() > 0.1);
        assert!(
            any_diff,
            "upper batched should differ from lower on asymmetric input"
        );
    }

    #[test]
    fn eigh_batched_uplo_2d_input_goes_through_unbatched_path() {
        // 2-D input: the shape.len() == 2 early return routes to
        // eigh_uplo directly. Verify the result matches the non-batched
        // eigh_uplo.
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (v_batched, _) = eigh_batched_uplo(&a, UPLO::Upper).unwrap();

        let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (v_unbatched, _) = eigh_uplo(&a2, UPLO::Upper).unwrap();
        assert_eq!(
            v_batched.iter().copied().collect::<Vec<_>>(),
            v_unbatched.as_slice().unwrap().to_vec()
        );
    }

    #[test]
    fn eigvalsh_batched_uplo_lower_matches_default() {
        let data = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let v_default = eigvalsh_batched(&a).unwrap();
        let v_lower = eigvalsh_batched_uplo(&a, UPLO::Lower).unwrap();
        assert_eq!(v_default.as_slice().unwrap(), v_lower.as_slice().unwrap());
    }

    #[test]
    fn eigvalsh_batched_uplo_symmetric_matches_across_triangles() {
        let data = vec![2.0, 1.0, 1.0, 2.0, 4.0, 0.0, 0.0, 5.0];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2, 2]), data).unwrap();
        let v_lower = eigvalsh_batched_uplo(&a, UPLO::Lower).unwrap();
        let v_upper = eigvalsh_batched_uplo(&a, UPLO::Upper).unwrap();
        for (l, u) in v_lower.iter().copied().zip(v_upper.iter().copied()) {
            assert!((l - u).abs() < 1e-10);
        }
    }

    // ----- Non-symmetric eig correctness (#221) -------------------------
    //
    // eig() previously had only the non-square error test. These tests
    // verify the spectral identity A*v_j == lambda_j*v_j on:
    //   - upper-triangular real-spectrum matrices,
    //   - 3x3 mixed real-spectrum matrices,
    //   - rotation matrices with complex eigenvalues,
    // plus a full reconstruction A == V*Lambda*V^-1 check.

    /// Verify A*v_j == lambda_j*v_j for each (eigenvalue, eigenvector)
    /// pair. Eigenvectors are stored row-major with column j at indices
    /// `[i * n + j]`.
    fn assert_eig_identity(a: &[f64], vals: &[Complex<f64>], vecs: &[Complex<f64>], n: usize) {
        for j in 0..n {
            let lambda = vals[j];
            for i in 0..n {
                let mut av = Complex::new(0.0, 0.0);
                for k in 0..n {
                    av += Complex::new(a[i * n + k], 0.0) * vecs[k * n + j];
                }
                let expected = lambda * vecs[i * n + j];
                let diff = av - expected;
                assert!(
                    diff.norm() < 1e-9,
                    "eigenpair {j}: (A*v - lambda*v)[{i}] = {diff} (||={})",
                    diff.norm()
                );
            }
        }
    }

    #[test]
    fn eig_upper_triangular_real_spectrum() {
        // Eigenvalues are exactly the diagonal entries (2, 3) for an
        // upper-triangular matrix.
        let a_data = vec![2.0, 1.0, 0.0, 3.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), a_data.clone()).unwrap();
        let (vals, vecs) = eig(&a).unwrap();

        let vs = vals.as_slice().unwrap();
        let mut real_parts: Vec<f64> = vs.iter().map(|c| c.re).collect();
        real_parts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert!((real_parts[0] - 2.0).abs() < 1e-10);
        assert!((real_parts[1] - 3.0).abs() < 1e-10);
        for c in vs {
            assert!(c.im.abs() < 1e-10);
        }

        assert_eig_identity(&a_data, vs, vecs.as_slice().unwrap(), 2);
    }

    #[test]
    fn eig_non_symmetric_3x3_real_spectrum() {
        // Block triangular: spectrum {1, 4, 6}. Verifies the identity
        // even when the matrix is non-normal.
        let a_data = vec![1.0, 2.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 3]), a_data.clone()).unwrap();
        let (vals, vecs) = eig(&a).unwrap();

        let vs = vals.as_slice().unwrap();
        let mut real_parts: Vec<f64> = vs.iter().map(|c| c.re).collect();
        real_parts.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (got, want) in real_parts.iter().zip([1.0, 4.0, 6.0].iter()) {
            assert!(
                (got - want).abs() < 1e-10,
                "eigenvalue {got} != expected {want}"
            );
        }

        assert_eig_identity(&a_data, vs, vecs.as_slice().unwrap(), 3);
    }

    #[test]
    fn eig_rotation_matrix_complex_spectrum() {
        // 90-degree 2-D rotation has eigenvalues +i and -i.
        let a_data = vec![0.0, -1.0, 1.0, 0.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), a_data.clone()).unwrap();
        let (vals, vecs) = eig(&a).unwrap();

        let vs = vals.as_slice().unwrap();
        // |im| of each eigenvalue must be 1, real part 0.
        for c in vs {
            assert!(c.re.abs() < 1e-10, "real part should be 0, got {}", c.re);
            assert!(
                (c.im.abs() - 1.0).abs() < 1e-10,
                "|im| should be 1, got {}",
                c.im
            );
        }
        // The imaginary parts must come in conjugate pairs (+1, -1).
        let im_sum: f64 = vs.iter().map(|c| c.im).sum();
        assert!(im_sum.abs() < 1e-10);

        assert_eig_identity(&a_data, vs, vecs.as_slice().unwrap(), 2);
    }

    #[test]
    fn eigvals_complex_spectrum() {
        // eigvals() should return the same complex spectrum as eig().0
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![0.0, -1.0, 1.0, 0.0]).unwrap();
        let vals = eigvals(&a).unwrap();
        let vs = vals.as_slice().unwrap();
        let im_sum: f64 = vs.iter().map(|c| c.im).sum();
        assert!(
            im_sum.abs() < 1e-10,
            "complex eigenvalues should be conjugate pairs"
        );
        for c in vs {
            assert!(c.re.abs() < 1e-10);
            assert!((c.im.abs() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn eig_reconstruction_diagonalizable() {
        // For a diagonalizable A, V * Lambda * V^-1 == A. Use a 2x2
        // shear with distinct real eigenvalues so V is invertible.
        let a_data = vec![3.0, 1.0, 0.0, 2.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), a_data.clone()).unwrap();
        let (vals, vecs) = eig(&a).unwrap();
        let vs = vals.as_slice().unwrap();
        let es = vecs.as_slice().unwrap();
        let n = 2;

        // Compute V * Lambda (column-scale V by eigenvalues).
        let mut vl = vec![Complex::new(0.0, 0.0); n * n];
        for i in 0..n {
            for j in 0..n {
                vl[i * n + j] = es[i * n + j] * vs[j];
            }
        }

        // Invert V (2x2 closed form). det = es[0]*es[3] - es[1]*es[2].
        let det = es[0] * es[3] - es[1] * es[2];
        assert!(det.norm() > 1e-12, "V must be invertible");
        let inv = [es[3] / det, -es[1] / det, -es[2] / det, es[0] / det];

        // (V*Lambda) * V^-1 should reconstruct A.
        for i in 0..n {
            for j in 0..n {
                let mut acc = Complex::new(0.0, 0.0);
                for k in 0..n {
                    acc += vl[i * n + k] * inv[k * n + j];
                }
                let expected = Complex::new(a_data[i * n + j], 0.0);
                let diff = acc - expected;
                assert!(
                    diff.norm() < 1e-9,
                    "reconstruction[{i},{j}] off by {} (got {acc}, want {expected})",
                    diff.norm()
                );
            }
        }
    }
}
