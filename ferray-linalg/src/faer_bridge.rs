// ferray-linalg: Conversion between ferray NdArray and faer::Mat
//
// Generic over LinalgFloat (f32, f64). Zero-copy where memory layouts match
// (both C-contiguous), otherwise copies into a contiguous buffer before calling faer.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::scalar::LinalgFloat;

/// Convert a 2D ferray Array<T, Ix2> to a faer::Mat<T>.
///
/// Convert a 2D ferray Array to faer's column-major Mat.
///
/// Uses `as_slice()` for C-contiguous arrays to avoid an intermediate
/// allocation — the data is indexed directly from the contiguous buffer.
/// Non-contiguous arrays fall back to iterator-based collection.
pub fn array2_to_faer<T: LinalgFloat>(a: &Array<T, Ix2>) -> faer::Mat<T> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    if let Some(slice) = a.as_slice() {
        // C-contiguous: index directly into the slice (row-major -> col-major)
        faer::Mat::from_fn(m, n, |i, j| slice[i * n + j])
    } else {
        // Non-contiguous: collect in logical order first
        let data: Vec<T> = a.iter().copied().collect();
        faer::Mat::from_fn(m, n, |i, j| data[i * n + j])
    }
}

/// Convert a 2D ferray Array<T, D> to a faer::Mat<T> using iter-based
/// indexing for any layout.
pub fn array2_to_faer_general<T: LinalgFloat, D: Dimension>(
    a: &Array<T, D>,
) -> FerrayResult<faer::Mat<T>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::shape_mismatch(format!(
            "expected 2D array, got {}D",
            shape.len()
        )));
    }
    let (m, n) = (shape[0], shape[1]);
    if let Some(slice) = a.as_slice() {
        Ok(faer::Mat::from_fn(m, n, |i, j| slice[i * n + j]))
    } else {
        let data: Vec<T> = a.iter().copied().collect();
        Ok(faer::Mat::from_fn(m, n, |i, j| data[i * n + j]))
    }
}

/// Convert a faer::Mat<T> back to a ferray Array<T, Ix2>.
pub fn faer_to_array2<T: LinalgFloat>(mat: &faer::Mat<T>) -> FerrayResult<Array<T, Ix2>> {
    let (m, n) = mat.shape();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([m, n]), data)
}

/// Convert a faer::Mat<T> to a ferray Array<T, IxDyn>.
pub fn faer_to_arrayd<T: LinalgFloat>(mat: &faer::Mat<T>) -> FerrayResult<Array<T, IxDyn>> {
    let (m, n) = mat.shape();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat[(i, j)]);
        }
    }
    Array::from_vec(IxDyn::new(&[m, n]), data)
}

/// Convert a 1D ferray array to a faer column vector (Mat with 1 column).
pub fn array1_to_faer_col<T: LinalgFloat>(a: &Array<T, Ix1>) -> faer::Mat<T> {
    let n = a.shape()[0];
    let data: Vec<T> = a.iter().copied().collect();
    faer::Mat::from_fn(n, 1, |i, _| data[i])
}

/// Convert a faer column vector (Mat with 1 column) to a ferray Array<T, Ix1>.
pub fn faer_col_to_array1<T: LinalgFloat>(mat: &faer::Mat<T>) -> FerrayResult<Array<T, Ix1>> {
    let m = mat.nrows();
    let mut data = Vec::with_capacity(m);
    for i in 0..m {
        data.push(mat[(i, 0)]);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Extract a 2D subview from a dynamic-rank array, given batch indices.
/// The last two dimensions are the matrix dimensions.
pub fn extract_matrix_from_batch<T: LinalgFloat>(
    a: &Array<T, IxDyn>,
    batch_idx: usize,
) -> FerrayResult<faer::Mat<T>> {
    let shape = a.shape();
    let ndim = shape.len();
    if ndim < 2 {
        return Err(FerrayError::shape_mismatch(
            "array must have at least 2 dimensions for matrix operations",
        ));
    }
    let m = shape[ndim - 2];
    let n = shape[ndim - 1];
    let matrix_size = m * n;
    let data: Vec<T> = a.iter().copied().collect();
    let offset = batch_idx * matrix_size;
    if offset + matrix_size > data.len() {
        return Err(FerrayError::shape_mismatch("batch index out of bounds"));
    }
    Ok(faer::Mat::from_fn(m, n, |i, j| data[offset + i * n + j]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_array2_faer() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mat = array2_to_faer(&a);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        assert_eq!(mat[(0, 0)], 1.0);
        assert_eq!(mat[(0, 2)], 3.0);
        assert_eq!(mat[(1, 0)], 4.0);

        let b = faer_to_array2(&mat).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn roundtrip_array1_faer_col() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let col = array1_to_faer_col(&a);
        assert_eq!(col.nrows(), 3);
        assert_eq!(col.ncols(), 1);
        let b = faer_col_to_array1(&col).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn roundtrip_f32() {
        let a = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let mat = array2_to_faer(&a);
        assert_eq!(mat[(0, 0)], 1.0f32);
        assert_eq!(mat[(1, 1)], 4.0f32);
        let b = faer_to_array2(&mat).unwrap();
        assert_eq!(a, b);
    }
}
