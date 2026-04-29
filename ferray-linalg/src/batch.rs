// ferray-linalg: Batched dispatch for stacked (3D+) arrays
//
// All linalg functions that accept 2D arrays also accept 3D+ stacked arrays.
// The operation applies along the last two axes, parallelized via Rayon.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use rayon::prelude::*;

use crate::scalar::LinalgFloat;

/// Compute the number of batches from a shape, given that the last `tail_dims`
/// dimensions are the per-element shape.
pub fn batch_count(shape: &[usize], tail_dims: usize) -> FerrayResult<usize> {
    if shape.len() < tail_dims {
        return Err(FerrayError::shape_mismatch(format!(
            "expected at least {}D array, got {}D",
            tail_dims,
            shape.len()
        )));
    }
    Ok(shape[..shape.len() - tail_dims]
        .iter()
        .product::<usize>()
        .max(1))
}

/// Extract the i-th 2D matrix from a batched array (last 2 dims are matrix dims).
/// Data is assumed to be in row-major (C) order.
pub fn extract_batch_matrix<T: Copy>(data: &[T], shape: &[usize], batch_idx: usize) -> Vec<T> {
    let ndim = shape.len();
    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let mat_size = m * n;
    let offset = batch_idx * mat_size;
    data[offset..offset + mat_size].to_vec()
}

/// Apply a function to each 2D matrix in a batched array, collecting results
/// into a flat Vec. The function receives (m, n, `matrix_data`) and returns
/// a result vector for that batch.
///
/// Results are computed in parallel using Rayon.
pub fn apply_batched_2d<T, F>(a: &Array<T, IxDyn>, f: F) -> FerrayResult<Vec<Vec<T>>>
where
    T: LinalgFloat,
    F: Fn(usize, usize, &[T]) -> FerrayResult<Vec<T>> + Send + Sync,
{
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "expected at least 2D array for batched matrix operation",
        ));
    }
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let num_batches = batch_count(shape, 2)?;
    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    let results: Vec<FerrayResult<Vec<T>>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let offset = b * mat_size;
            let slice = &data[offset..offset + mat_size];
            f(m, n, slice)
        })
        .collect();

    results.into_iter().collect()
}

/// Apply a scalar-returning function to each 2D matrix in a batched array.
/// Returns an Array1<T> with one scalar per batch.
pub fn apply_batched_scalar<T, F>(a: &Array<T, IxDyn>, f: F) -> FerrayResult<Array<T, Ix1>>
where
    T: LinalgFloat,
    F: Fn(usize, usize, &[T]) -> FerrayResult<T> + Send + Sync,
{
    let shape = a.shape();
    if shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "expected at least 2D array for batched matrix operation",
        ));
    }
    let m = shape[shape.len() - 2];
    let n = shape[shape.len() - 1];
    let num_batches = batch_count(shape, 2)?;
    let data: Vec<T> = a.iter().copied().collect();
    let mat_size = m * n;

    let results: Vec<FerrayResult<T>> = (0..num_batches)
        .into_par_iter()
        .map(|b| {
            let offset = b * mat_size;
            let slice = &data[offset..offset + mat_size];
            f(m, n, slice)
        })
        .collect();

    let scalars: FerrayResult<Vec<T>> = results.into_iter().collect();
    let scalars = scalars?;
    Array::from_vec(Ix1::new([scalars.len()]), scalars)
}

/// Apply a function that takes two 2D matrices (a, b) from batched arrays.
/// Both arrays must have the same batch dimensions.
pub fn apply_batched_2d_pair<T, F>(
    a: &Array<T, IxDyn>,
    b: &Array<T, IxDyn>,
    f: F,
) -> FerrayResult<Vec<Vec<T>>>
where
    T: LinalgFloat,
    F: Fn(usize, usize, &[T], usize, usize, &[T]) -> FerrayResult<Vec<T>> + Send + Sync,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(FerrayError::shape_mismatch(
            "expected at least 2D arrays for batched matrix operation",
        ));
    }
    let a_batch = &a_shape[..a_shape.len() - 2];
    let b_batch = &b_shape[..b_shape.len() - 2];
    if a_batch != b_batch {
        return Err(FerrayError::shape_mismatch(format!(
            "batch dimensions must match: {a_batch:?} vs {b_batch:?}"
        )));
    }
    let am = a_shape[a_shape.len() - 2];
    let an = a_shape[a_shape.len() - 1];
    let bm = b_shape[b_shape.len() - 2];
    let bn = b_shape[b_shape.len() - 1];
    let num_batches = batch_count(a_shape, 2)?;
    let a_data: Vec<T> = a.iter().copied().collect();
    let b_data: Vec<T> = b.iter().copied().collect();
    let a_mat_size = am * an;
    let b_mat_size = bm * bn;

    let results: Vec<FerrayResult<Vec<T>>> = (0..num_batches)
        .into_par_iter()
        .map(|idx| {
            let a_offset = idx * a_mat_size;
            let b_offset = idx * b_mat_size;
            let a_slice = &a_data[a_offset..a_offset + a_mat_size];
            let b_slice = &b_data[b_offset..b_offset + b_mat_size];
            f(am, an, a_slice, bm, bn, b_slice)
        })
        .collect();

    results.into_iter().collect()
}

/// Helper: create a `faer::Mat` from a flat row-major slice.
pub fn slice_to_faer<T: LinalgFloat>(m: usize, n: usize, data: &[T]) -> faer::Mat<T> {
    faer::Mat::from_fn(m, n, |i, j| data[i * n + j])
}

/// Helper: extract flat row-major data from a `faer::Mat`.
#[must_use]
pub fn faer_to_vec<T: LinalgFloat>(mat: &faer::Mat<T>) -> Vec<T> {
    let (m, n) = mat.shape();
    let mut v = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            v.push(mat[(i, j)]);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_count_2d() {
        assert_eq!(batch_count(&[3, 4], 2).unwrap(), 1);
    }

    #[test]
    fn batch_count_3d() {
        assert_eq!(batch_count(&[10, 3, 4], 2).unwrap(), 10);
    }

    #[test]
    fn batch_count_4d() {
        assert_eq!(batch_count(&[5, 10, 3, 4], 2).unwrap(), 50);
    }

    #[test]
    fn slice_faer_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mat = slice_to_faer(2, 3, &data);
        let back = faer_to_vec(&mat);
        assert_eq!(data, back);
    }

    #[test]
    fn slice_faer_roundtrip_f32() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mat = slice_to_faer(2, 2, &data);
        let back = faer_to_vec(&mat);
        assert_eq!(data, back);
    }
}
