// ferray-linalg: Conversion between ferray NdArray and faer::Mat
//
// Zero-copy where memory layouts match (both C-contiguous), otherwise copies
// into a contiguous buffer before calling faer.

use ferray_core::array::owned::Array;
use ferray_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};

/// Convert a 2D ferray Array<f64, Ix2> to a faer::Mat<f64>.
///
/// If the array is C-contiguous (row-major), we copy the data into faer's
/// column-major layout. faer always stores data column-major.
pub fn array2_to_faer(a: &Array<f64, Ix2>) -> faer::Mat<f64> {
    let shape = a.shape();
    let (m, n) = (shape[0], shape[1]);
    faer::Mat::from_fn(m, n, |i, j| {
        // Use iterator-based access for safety with any layout
        let idx = i * n + j;
        let flat: Vec<f64> = if let Some(s) = a.as_slice() {
            // Fast path: contiguous data
            return s[idx];
        } else {
            a.iter().copied().collect()
        };
        flat[idx]
    })
}

/// Convert a 2D ferray Array<f64, Ix2> to a faer::Mat<f64> using iter-based
/// indexing for any layout.
pub fn array2_to_faer_general<D: Dimension>(a: &Array<f64, D>) -> FerrayResult<faer::Mat<f64>> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::shape_mismatch(format!(
            "expected 2D array, got {}D",
            shape.len()
        )));
    }
    let (m, n) = (shape[0], shape[1]);
    let data: Vec<f64> = a.iter().copied().collect();
    Ok(faer::Mat::from_fn(m, n, |i, j| data[i * n + j]))
}

/// Convert a faer::Mat<f64> back to a ferray Array<f64, Ix2>.
pub fn faer_to_array2(mat: &faer::Mat<f64>) -> FerrayResult<Array<f64, Ix2>> {
    let (m, n) = mat.shape();
    let mut data = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            data.push(mat[(i, j)]);
        }
    }
    Array::from_vec(Ix2::new([m, n]), data)
}

/// Convert a faer::Mat<f64> to a ferray Array<f64, IxDyn>.
pub fn faer_to_arrayd(mat: &faer::Mat<f64>) -> FerrayResult<Array<f64, IxDyn>> {
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
pub fn array1_to_faer_col(a: &Array<f64, Ix1>) -> faer::Mat<f64> {
    let n = a.shape()[0];
    let data: Vec<f64> = a.iter().copied().collect();
    faer::Mat::from_fn(n, 1, |i, _| data[i])
}

/// Convert a faer column vector (Mat with 1 column) to a ferray Array<f64, Ix1>.
pub fn faer_col_to_array1(mat: &faer::Mat<f64>) -> FerrayResult<Array<f64, Ix1>> {
    let m = mat.nrows();
    let mut data = Vec::with_capacity(m);
    for i in 0..m {
        data.push(mat[(i, 0)]);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Extract a 2D subview from a dynamic-rank array, given batch indices.
/// The last two dimensions are the matrix dimensions.
pub fn extract_matrix_from_batch(
    a: &Array<f64, IxDyn>,
    batch_idx: usize,
) -> FerrayResult<faer::Mat<f64>> {
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
    let data: Vec<f64> = a.iter().copied().collect();
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
}
