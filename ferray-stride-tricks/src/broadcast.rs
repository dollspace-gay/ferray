// ferray-stride-tricks: Broadcasting helpers (REQ-2, REQ-3, REQ-4)
//
// These functions are thin forwarding wrappers over
// `ferray_core::dimension::broadcast`. No separate implementation
// lives here — they exist purely so users who pull in
// `ferray-stride-tricks` for `as_strided`/`sliding_window_view` get a
// NumPy-style broadcasting surface at the same import path without
// needing to reach into `ferray-core::dimension::broadcast`. #528
// verified this is not a second copy of the code.

use ferray_core::dimension::broadcast as core_broadcast;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::FerrayResult;
use ferray_core::{Array, ArrayView, Element};

/// Broadcast an array to a target shape via zero-copy stride manipulation.
///
/// The returned view uses stride-0 tricks to virtually expand size-1
/// dimensions — no data is copied. This is equivalent to `NumPy`'s
/// `numpy.broadcast_to`.
///
/// # Errors
///
/// Returns `FerrayError::BroadcastFailure` if the array's shape cannot be
/// broadcast to `target_shape` (e.g., a non-1 dimension differs from the
/// target).
///
/// # Examples
///
/// ```
/// # use ferray_core::{Array, dimension::Ix1};
/// # use ferray_stride_tricks::broadcast_to;
/// let a = Array::<f64, Ix1>::ones(Ix1::new([3])).unwrap();
/// let v = broadcast_to(&a, &[4, 3]).unwrap();
/// assert_eq!(v.shape(), &[4, 3]);
/// assert_eq!(v.size(), 12);
/// ```
pub fn broadcast_to<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    target_shape: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    core_broadcast::broadcast_to(array, target_shape)
}

/// Broadcast multiple arrays to a common shape.
///
/// Returns a vector of `ArrayView<IxDyn>` views, all sharing the same
/// broadcast shape. No data is copied.
///
/// # Errors
///
/// Returns `FerrayError::BroadcastFailure` if the shapes are incompatible.
///
/// # Examples
///
/// ```
/// # use ferray_core::{Array, dimension::Ix2};
/// # use ferray_stride_tricks::broadcast_arrays;
/// let a = Array::<f64, Ix2>::ones(Ix2::new([4, 1])).unwrap();
/// let b = Array::<f64, Ix2>::ones(Ix2::new([1, 3])).unwrap();
/// let arrays = [a, b];
/// let views = broadcast_arrays(&arrays).unwrap();
/// assert_eq!(views[0].shape(), &[4, 3]);
/// assert_eq!(views[1].shape(), &[4, 3]);
/// ```
pub fn broadcast_arrays<T: Element, D: Dimension>(
    arrays: &[Array<T, D>],
) -> FerrayResult<Vec<ArrayView<'_, T, IxDyn>>> {
    core_broadcast::broadcast_arrays(arrays)
}

/// Compute the broadcast result shape from multiple shapes, without
/// allocating any arrays.
///
/// Follows `NumPy`'s broadcasting rules: shapes are right-aligned, size-1
/// dimensions stretch, and mismatched non-1 dimensions are errors.
///
/// # Errors
///
/// Returns `FerrayError::BroadcastFailure` if any pair of shapes is
/// incompatible.
///
/// # Examples
///
/// ```
/// # use ferray_stride_tricks::broadcast_shapes;
/// let result = broadcast_shapes(&[&[3, 1][..], &[1, 4][..]]).unwrap();
/// assert_eq!(result, vec![3, 4]);
/// ```
pub fn broadcast_shapes(shapes: &[&[usize]]) -> FerrayResult<Vec<usize>> {
    core_broadcast::broadcast_shapes_multi(shapes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    #[test]
    fn broadcast_to_1d_to_2d() {
        let a = Array::<f64, Ix1>::ones(Ix1::new([3])).unwrap();
        let v = broadcast_to(&a, &[4, 3]).unwrap();
        assert_eq!(v.shape(), &[4, 3]);
        assert_eq!(v.size(), 12);
        // All rows identical
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0; 12]);
    }

    #[test]
    fn broadcast_to_no_copy() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = broadcast_to(&a, &[100, 3]).unwrap();
        assert_eq!(v.as_ptr(), a.as_ptr());
    }

    #[test]
    fn broadcast_to_incompatible() {
        let a = Array::<f64, Ix1>::ones(Ix1::new([3])).unwrap();
        assert!(broadcast_to(&a, &[4, 5]).is_err());
    }

    #[test]
    fn broadcast_arrays_two() {
        let a = Array::<f64, Ix2>::ones(Ix2::new([4, 1])).unwrap();
        let b = Array::<f64, Ix2>::ones(Ix2::new([1, 3])).unwrap();
        let arrays = [a, b];
        let views = broadcast_arrays(&arrays).unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].shape(), &[4, 3]);
        assert_eq!(views[1].shape(), &[4, 3]);
    }

    #[test]
    fn broadcast_arrays_empty() {
        let views = broadcast_arrays::<f64, Ix1>(&[]).unwrap();
        assert!(views.is_empty());
    }

    #[test]
    fn broadcast_shapes_basic() {
        let result = broadcast_shapes(&[&[3, 1][..], &[1, 4][..]]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    #[test]
    fn broadcast_shapes_three() {
        let result = broadcast_shapes(&[&[2, 1][..], &[3][..], &[1, 3][..]]).unwrap();
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn broadcast_shapes_incompatible() {
        assert!(broadcast_shapes(&[&[3][..], &[4][..]]).is_err());
    }

    #[test]
    fn broadcast_shapes_empty() {
        let result = broadcast_shapes(&[]).unwrap();
        assert_eq!(result, Vec::<usize>::new());
    }
}
