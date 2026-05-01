// ferray-linalg: SVD decomposition (REQ-10)
//
// Wraps faer::Svd. Returns (U, S, Vt).

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::batch;
use crate::faer_bridge;
use crate::scalar::LinalgFloat;

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
pub fn svd<T: LinalgFloat>(
    a: &Array<T, Ix2>,
    full_matrices: bool,
) -> FerrayResult<(Array<T, Ix2>, Array<T, Ix1>, Array<T, Ix2>)> {
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

/// SVD of a symmetric/Hermitian matrix via eigendecomposition (#407).
///
/// Equivalent to `numpy.linalg.svd(a, hermitian=True)`: routes through
/// `eigh` instead of full SVD, which is roughly 2x faster for square
/// symmetric inputs because the eigendecomposition kernel exploits
/// symmetry and only operates on one triangle.
///
/// For a symmetric `A = Q * Λ * Qᵀ` the SVD is constructed as:
/// 1. Singular values `s = |Λ|`, sorted in descending order.
/// 2. `U = Q` with column signs flipped for negative eigenvalues.
/// 3. `Vt = Qᵀ`.
///
/// **Caller responsibility.** This function does not validate that
/// `a` is actually symmetric — passing an asymmetric matrix produces
/// undefined output (the eigendecomposition reads only one triangle).
/// If you're unsure, use [`svd`] instead.
///
/// Returns `(U, S, Vt)` matching the [`svd`] output convention.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `a` is not square.
/// - `FerrayError::InvalidValue` if eigendecomposition fails.
pub fn svd_hermitian<T: LinalgFloat>(
    a: &Array<T, Ix2>,
) -> FerrayResult<(Array<T, Ix2>, Array<T, Ix1>, Array<T, Ix2>)> {
    let shape = a.shape();
    if shape[0] != shape[1] {
        return Err(FerrayError::shape_mismatch(format!(
            "svd_hermitian requires a square matrix, got {}x{}",
            shape[0], shape[1]
        )));
    }
    let n = shape[0];

    // eigh returns ascending eigenvalues; we want |λ| in descending
    // order to match SVD convention.
    let (eigenvalues, eigenvectors) = super::eigen::eigh(a)?;

    // Build a (|λ|, original_index) list, sort by |λ| descending.
    let evals: Vec<T> = eigenvalues.iter().copied().collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        let ai = num_traits::Float::abs(evals[i]);
        let aj = num_traits::Float::abs(evals[j]);
        aj.partial_cmp(&ai)
            .unwrap_or(core::cmp::Ordering::Equal)
    });

    let evec_data: Vec<T> = eigenvectors.iter().copied().collect();

    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();

    let mut s_vals = Vec::with_capacity(n);
    let mut u_data = vec![zero; n * n];
    let mut vt_data = vec![zero; n * n];

    for (col, &orig) in order.iter().enumerate() {
        let lambda = evals[orig];
        s_vals.push(num_traits::Float::abs(lambda));
        let sign = if lambda < zero { -one } else { one };
        for row in 0..n {
            // U[row, col] = sign * Q[row, orig]
            u_data[row * n + col] = sign * evec_data[row * n + orig];
            // Vt[col, row] = Q[row, orig] (transpose of Q's orig column)
            vt_data[col * n + row] = evec_data[row * n + orig];
        }
    }

    let u_arr = Array::from_vec(Ix2::new([n, n]), u_data)?;
    let s_arr = Array::from_vec(Ix1::new([n]), s_vals)?;
    let vt_arr = Array::from_vec(Ix2::new([n, n]), vt_data)?;
    Ok((u_arr, s_arr, vt_arr))
}

/// Return only the singular values of a matrix.
///
/// Equivalent to `numpy.linalg.svdvals(a)`. Faster than full `svd`
/// because U and Vt are never materialized — only the diagonal S is
/// extracted from the faer decomposition (#406).
///
/// # Errors
/// - `FerrayError::InvalidValue` if SVD computation fails to converge.
pub fn svdvals<T: LinalgFloat>(a: &Array<T, Ix2>) -> FerrayResult<Array<T, Ix1>> {
    let mat = faer_bridge::array2_to_faer(a);
    let decomp = mat
        .as_ref()
        .thin_svd()
        .map_err(|e| FerrayError::InvalidValue {
            message: format!("svdvals failed to converge: {e:?}"),
        })?;
    let s_diag = decomp.S();
    let min_dim = a.shape()[0].min(a.shape()[1]);
    let mut s_vals = Vec::with_capacity(min_dim);
    for i in 0..min_dim {
        s_vals.push(s_diag.column_vector()[i]);
    }
    Array::from_vec(Ix1::new([s_vals.len()]), s_vals)
}

/// Batched SVD for arrays of shape `(..., M, N)`.
///
/// Returns `(U, S, Vt)` with batch dimensions preserved. In thin mode
/// (`full_matrices = false`): `U` has shape `(..., M, K)`, `S` has shape
/// `(..., K)`, `Vt` has shape `(..., K, N)` where `K = min(M, N)`. In
/// full mode: `U` has shape `(..., M, M)`, `Vt` has shape `(..., N, N)`.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if the array has fewer than 2 dims.
/// - `FerrayError::InvalidValue` if SVD fails for any batch element.
pub fn svd_batched<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    full_matrices: bool,
) -> FerrayResult<(Array<T, IxDyn>, Array<T, IxDyn>, Array<T, IxDyn>)> {
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "svd_batched: a must have at least 2 dimensions",
        ));
    }
    if shape.len() == 2 {
        let a2 =
            Array::<T, Ix2>::from_vec(Ix2::new([shape[0], shape[1]]), a.iter().copied().collect())?;
        let (u, s, vt) = svd(&a2, full_matrices)?;
        return Ok((
            Array::from_vec(IxDyn::new(u.shape()), u.iter().copied().collect())?,
            Array::from_vec(IxDyn::new(s.shape()), s.iter().copied().collect())?,
            Array::from_vec(IxDyn::new(vt.shape()), vt.iter().copied().collect())?,
        ));
    }

    let num_batches = batch::batch_count(shape, 2)?;
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let k = m.min(n);
    let (u_rows, u_cols, vt_rows, vt_cols) = if full_matrices {
        (m, m, n, n)
    } else {
        (m, k, k, n)
    };

    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    use rayon::prelude::*;
    let results: Vec<FerrayResult<(Vec<T>, Vec<T>, Vec<T>)>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let slice = &data[b * mat_size..(b + 1) * mat_size];
            let mat = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), slice.to_vec())?;
            let (u, s, vt) = svd(&mat, full_matrices)?;
            Ok((
                u.iter().copied().collect::<Vec<T>>(),
                s.iter().copied().collect::<Vec<T>>(),
                vt.iter().copied().collect::<Vec<T>>(),
            ))
        })
        .collect();

    let mut u_flat: Vec<T> = Vec::with_capacity(num_batches * u_rows * u_cols);
    let mut s_flat: Vec<T> = Vec::with_capacity(num_batches * k);
    let mut vt_flat: Vec<T> = Vec::with_capacity(num_batches * vt_rows * vt_cols);
    for res in results {
        let (u, s, vt) = res?;
        u_flat.extend(u);
        s_flat.extend(s);
        vt_flat.extend(vt);
    }

    let batch_dims = &shape[..shape.len() - 2];
    let mut u_shape: Vec<usize> = batch_dims.to_vec();
    u_shape.push(u_rows);
    u_shape.push(u_cols);
    let mut s_shape: Vec<usize> = batch_dims.to_vec();
    s_shape.push(k);
    let mut vt_shape: Vec<usize> = batch_dims.to_vec();
    vt_shape.push(vt_rows);
    vt_shape.push(vt_cols);
    Ok((
        Array::from_vec(IxDyn::new(&u_shape), u_flat)?,
        Array::from_vec(IxDyn::new(&s_shape), s_flat)?,
        Array::from_vec(IxDyn::new(&vt_shape), vt_flat)?,
    ))
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
                    "U*S*Vt[{i},{j}] = {val} != {expected}"
                );
            }
        }
    }

    #[test]
    fn svd_f32() {
        let a =
            Array::<f32, Ix2>::from_vec(Ix2::new([3, 2]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let (u, s, vt) = svd(&a, false).unwrap();

        let us = u.as_slice().unwrap();
        let ss = s.as_slice().unwrap();
        let vts = vt.as_slice().unwrap();

        let m = 3;
        let n = 2;
        let k = ss.len();

        for i in 0..m {
            for j in 0..n {
                let mut val = 0.0f32;
                for p in 0..k {
                    val += us[i * k + p] * ss[p] * vts[p * n + j];
                }
                let expected = a.as_slice().unwrap()[i * n + j];
                assert!(
                    (val - expected).abs() < 1e-4,
                    "U*S*Vt[{i},{j}] = {val} != {expected}"
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
            assert!(val >= 0.0, "singular value {val} should be >= 0");
        }
    }

    #[test]
    fn svd_batched_thin_reconstructs() {
        // Two 3x2 matrices stacked.
        let data: Vec<f64> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
            2.0, 0.0, 0.0, 1.0, 1.0, 1.0,
        ];
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3, 2]), data.clone()).unwrap();
        let (u, s, vt) = svd_batched(&a, false).unwrap();
        assert_eq!(u.shape(), &[2, 3, 2]);
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(vt.shape(), &[2, 2, 2]);

        // Verify U * diag(S) * Vt ≈ A per batch.
        let u_data: Vec<f64> = u.iter().copied().collect();
        let s_data: Vec<f64> = s.iter().copied().collect();
        let vt_data: Vec<f64> = vt.iter().copied().collect();
        for b in 0..2 {
            let u_off = b * 6;
            let s_off = b * 2;
            let vt_off = b * 4;
            let a_off = b * 6;
            for i in 0..3 {
                for j in 0..2 {
                    let mut sum = 0.0;
                    for p in 0..2 {
                        sum += u_data[u_off + i * 2 + p]
                            * s_data[s_off + p]
                            * vt_data[vt_off + p * 2 + j];
                    }
                    assert!(
                        (sum - data[a_off + i * 2 + j]).abs() < 1e-10,
                        "U diag(S) Vt != A at batch {b} [{i},{j}]: {} vs {}",
                        sum,
                        data[a_off + i * 2 + j]
                    );
                }
            }
        }
    }

    // ----- svd_hermitian (#407) -----------------------------------------

    #[test]
    fn svd_hermitian_2x2_diagonal() {
        // Diagonal symmetric matrix [[3, 0], [0, -2]]:
        // eigenvalues 3, -2 → singular values 3, 2 (sorted descending).
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![3.0, 0.0, 0.0, -2.0]).unwrap();
        let (_u, s, _vt) = svd_hermitian(&a).unwrap();
        assert_eq!(s.shape(), &[2]);
        let svals: Vec<f64> = s.iter().copied().collect();
        assert!((svals[0] - 3.0).abs() < 1e-10);
        assert!((svals[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn svd_hermitian_singular_values_match_full_svd() {
        // [[2, 1], [1, 2]] is symmetric. Both paths must agree on
        // singular values up to ordering / numerical noise.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let (_u_h, s_h, _vt_h) = svd_hermitian(&a).unwrap();
        let (_u_f, s_f, _vt_f) = svd(&a, false).unwrap();
        let mut sh: Vec<f64> = s_h.iter().copied().collect();
        let mut sf: Vec<f64> = s_f.iter().copied().collect();
        sh.sort_by(|x, y| y.partial_cmp(x).unwrap());
        sf.sort_by(|x, y| y.partial_cmp(x).unwrap());
        for (a, b) in sh.iter().zip(sf.iter()) {
            assert!((a - b).abs() < 1e-10, "{} vs {}", a, b);
        }
    }

    #[test]
    fn svd_hermitian_reconstructs_a() {
        // For a symmetric A, U * diag(s) * Vt should equal A.
        let a_data = vec![4.0_f64, 1.0, 1.0, 3.0];
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), a_data.clone()).unwrap();
        let (u, s, vt) = svd_hermitian(&a).unwrap();
        let u_d: Vec<f64> = u.iter().copied().collect();
        let s_d: Vec<f64> = s.iter().copied().collect();
        let vt_d: Vec<f64> = vt.iter().copied().collect();
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += u_d[i * n + k] * s_d[k] * vt_d[k * n + j];
                }
                assert!(
                    (sum - a_data[i * n + j]).abs() < 1e-10,
                    "reconstruct[{i},{j}] = {sum}, expected {}",
                    a_data[i * n + j]
                );
            }
        }
    }

    #[test]
    fn svd_hermitian_rejects_non_square() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let result = svd_hermitian(&a);
        assert!(result.is_err());
    }

    #[test]
    fn svd_hermitian_negative_eigenvalues_become_positive_singular_values() {
        // Negative-definite symmetric matrix: every eigenvalue is
        // negative, every singular value must be positive.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![-3.0, 0.0, 0.0, -5.0]).unwrap();
        let (_u, s, _vt) = svd_hermitian(&a).unwrap();
        for &v in s.as_slice().unwrap() {
            assert!(v >= 0.0, "singular value {v} should be ≥ 0");
        }
    }
}
