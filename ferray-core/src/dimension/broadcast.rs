// ferray-core: Broadcasting logic (REQ-9, REQ-10, REQ-11)
//
// Implements NumPy's full broadcasting rules:
//   1. Prepend 1s to shape of lower-dim array
//   2. Stretch size-1 dimensions
//   3. Error on size mismatch where neither is 1
//
// Broadcasting NEVER materializes the expanded array — it uses virtual
// expansion via strides (setting stride = 0 for broadcast dimensions).

use ndarray::ShapeBuilder;

use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::dimension::{Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

/// Compute the broadcast shape from two shapes, following NumPy rules.
///
/// The result shape has `max(a.len(), b.len())` dimensions. Shorter shapes
/// are left-padded with 1s. For each axis, the result dimension is the
/// larger of the two inputs; if neither is 1 and they differ, an error
/// is returned.
///
/// # Examples
/// ```
/// # use ferray_core::dimension::broadcast::broadcast_shapes;
/// let result = broadcast_shapes(&[4, 3], &[3]).unwrap();
/// assert_eq!(result, vec![4, 3]);
///
/// let result = broadcast_shapes(&[2, 1, 4], &[3, 4]).unwrap();
/// assert_eq!(result, vec![2, 3, 4]);
/// ```
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if shapes are incompatible.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> FerrayResult<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];

    for i in 0..ndim {
        let da = if i < ndim - a.len() {
            1
        } else {
            a[i - (ndim - a.len())]
        };
        let db = if i < ndim - b.len() {
            1
        } else {
            b[i - (ndim - b.len())]
        };

        if da == db {
            result[i] = da;
        } else if da == 1 {
            result[i] = db;
        } else if db == 1 {
            result[i] = da;
        } else {
            return Err(FerrayError::broadcast_failure(a, b));
        }
    }
    Ok(result)
}

/// Compute the broadcast shape from multiple shapes.
///
/// This is the N-ary version of [`broadcast_shapes`]. It folds pairwise
/// over all input shapes.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if any pair is incompatible.
pub fn broadcast_shapes_multi(shapes: &[&[usize]]) -> FerrayResult<Vec<usize>> {
    if shapes.is_empty() {
        return Ok(vec![]);
    }
    let mut result = shapes[0].to_vec();
    for &s in &shapes[1..] {
        result = broadcast_shapes(&result, s)?;
    }
    Ok(result)
}

/// Compute the strides for broadcasting a source shape to a target shape.
///
/// For dimensions where the source has size 1 but the target is larger,
/// the stride is set to 0 (virtual expansion). For matching dimensions,
/// the original stride is preserved. The source shape is left-padded with
/// 1s (stride 0) as needed.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if the source cannot be broadcast
/// to the target (i.e., a source dimension is neither 1 nor equal to target).
pub fn broadcast_strides(
    src_shape: &[usize],
    src_strides: &[isize],
    target_shape: &[usize],
) -> FerrayResult<Vec<isize>> {
    let tndim = target_shape.len();
    let sndim = src_shape.len();

    if tndim < sndim {
        return Err(FerrayError::shape_mismatch(format!(
            "cannot broadcast shape {:?} to shape {:?}: target has fewer dimensions",
            src_shape, target_shape
        )));
    }

    let pad = tndim - sndim;
    let mut out_strides = vec![0isize; tndim];

    for i in 0..tndim {
        if i < pad {
            // Prepended dimension: virtual, stride = 0
            out_strides[i] = 0;
        } else {
            let si = i - pad;
            let src_dim = src_shape[si];
            let tgt_dim = target_shape[i];

            if src_dim == tgt_dim {
                out_strides[i] = src_strides[si];
            } else if src_dim == 1 {
                // Broadcast: virtual expansion
                out_strides[i] = 0;
            } else {
                return Err(FerrayError::shape_mismatch(format!(
                    "cannot broadcast dimension {} (size {}) to size {}",
                    si, src_dim, tgt_dim
                )));
            }
        }
    }

    Ok(out_strides)
}

/// Broadcast an array to a target shape, returning a view.
///
/// The returned view uses stride-0 tricks to virtually expand size-1
/// dimensions — no data is copied. The view borrows from the source array.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if the array cannot be broadcast
/// to the given shape.
pub fn broadcast_to<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    target_shape: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    let src_shape = array.shape();
    let src_strides = array.strides();

    // Validate broadcast compatibility
    let result_shape = broadcast_shapes(src_shape, target_shape)?;
    if result_shape != target_shape {
        return Err(FerrayError::shape_mismatch(format!(
            "cannot broadcast shape {:?} to shape {:?}",
            src_shape, target_shape
        )));
    }

    let new_strides = broadcast_strides(src_shape, src_strides, target_shape)?;

    // Validate all strides are non-negative before casting to usize.
    // Negative strides (from reversed/transposed views) cannot be represented
    // as usize and would wrap to huge values, causing out-of-bounds access.
    for (i, &s) in new_strides.iter().enumerate() {
        if s < 0 {
            return Err(FerrayError::shape_mismatch(format!(
                "cannot broadcast array with negative stride {} on axis {}; \
                 make the array contiguous first",
                s, i
            )));
        }
    }

    // Build ndarray view with computed strides
    let nd_shape = ndarray::IxDyn(target_shape);
    let nd_strides = ndarray::IxDyn(&new_strides.iter().map(|&s| s as usize).collect::<Vec<_>>());

    // Use from_shape_ptr with the broadcast strides
    let ptr = array.as_ptr();
    // SAFETY: broadcast strides are validated non-negative above and ensure
    // we only access valid memory from the source array. Stride-0 dimensions
    // repeat the same element.
    let nd_view = unsafe { ndarray::ArrayView::from_shape_ptr(nd_shape.strides(nd_strides), ptr) };

    Ok(ArrayView::from_ndarray(nd_view))
}

/// Broadcast an `ArrayView` to a target shape, returning a new view.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if the view cannot be broadcast.
pub fn broadcast_view_to<'a, T: Element, D: Dimension>(
    view: &ArrayView<'a, T, D>,
    target_shape: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    let src_shape = view.shape();
    let src_strides = view.strides();

    let result_shape = broadcast_shapes(src_shape, target_shape)?;
    if result_shape != target_shape {
        return Err(FerrayError::shape_mismatch(format!(
            "cannot broadcast shape {:?} to shape {:?}",
            src_shape, target_shape
        )));
    }

    let new_strides = broadcast_strides(src_shape, src_strides, target_shape)?;

    for (i, &s) in new_strides.iter().enumerate() {
        if s < 0 {
            return Err(FerrayError::shape_mismatch(format!(
                "cannot broadcast view with negative stride {} on axis {}; \
                 make the array contiguous first",
                s, i
            )));
        }
    }

    let nd_shape = ndarray::IxDyn(target_shape);
    let nd_strides = ndarray::IxDyn(&new_strides.iter().map(|&s| s as usize).collect::<Vec<_>>());

    let ptr = view.as_ptr();
    // SAFETY: strides validated non-negative above; broadcast strides ensure
    // only valid source memory is accessed.
    let nd_view = unsafe { ndarray::ArrayView::from_shape_ptr(nd_shape.strides(nd_strides), ptr) };

    Ok(ArrayView::from_ndarray(nd_view))
}

/// Broadcast multiple arrays to a common shape.
///
/// Returns a vector of `ArrayView<IxDyn>` views, all sharing the same
/// broadcast shape. No data is copied.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if shapes are incompatible.
pub fn broadcast_arrays<'a, T: Element, D: Dimension>(
    arrays: &'a [Array<T, D>],
) -> FerrayResult<Vec<ArrayView<'a, T, IxDyn>>> {
    if arrays.is_empty() {
        return Ok(vec![]);
    }

    // Compute common broadcast shape
    let shapes: Vec<&[usize]> = arrays.iter().map(|a| a.shape()).collect();
    let target = broadcast_shapes_multi(&shapes)?;

    // Broadcast each array to the common shape
    let mut result = Vec::with_capacity(arrays.len());
    for arr in arrays {
        result.push(broadcast_to(arr, &target)?);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Methods on Array for broadcasting
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> Array<T, D> {
    /// Broadcast this array to the given shape, returning a dynamic-rank view.
    ///
    /// Uses stride-0 tricks for virtual expansion — no data is copied.
    ///
    /// # Errors
    /// Returns `FerrayError::BroadcastFailure` if the array cannot be broadcast
    /// to the target shape.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> FerrayResult<ArrayView<'_, T, IxDyn>> {
        broadcast_to(self, target_shape)
    }
}

impl<'a, T: Element, D: Dimension> ArrayView<'a, T, D> {
    /// Broadcast this view to the given shape, returning a dynamic-rank view.
    ///
    /// # Errors
    /// Returns `FerrayError::BroadcastFailure` if the view cannot be broadcast.
    pub fn broadcast_to(&self, target_shape: &[usize]) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
        let src_shape = self.shape();
        let src_strides = self.strides();

        let result_shape = broadcast_shapes(src_shape, target_shape)?;
        if result_shape != target_shape {
            return Err(FerrayError::shape_mismatch(format!(
                "cannot broadcast shape {:?} to shape {:?}",
                src_shape, target_shape
            )));
        }

        let new_strides = broadcast_strides(src_shape, src_strides, target_shape)?;

        for (i, &s) in new_strides.iter().enumerate() {
            if s < 0 {
                return Err(FerrayError::shape_mismatch(format!(
                    "cannot broadcast view with negative stride {} on axis {}; \
                     make the array contiguous first",
                    s, i
                )));
            }
        }

        let nd_shape = ndarray::IxDyn(target_shape);
        let nd_strides =
            ndarray::IxDyn(&new_strides.iter().map(|&s| s as usize).collect::<Vec<_>>());

        let ptr = self.as_ptr();
        // SAFETY: strides validated non-negative above; broadcast strides ensure
        // only valid source memory is accessed.
        let nd_view =
            unsafe { ndarray::ArrayView::from_shape_ptr(nd_shape.strides(nd_strides), ptr) };

        Ok(ArrayView::from_ndarray(nd_view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2, Ix3};

    // -----------------------------------------------------------------------
    // broadcast_shapes tests
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_shapes_same() {
        assert_eq!(broadcast_shapes(&[3, 4], &[3, 4]).unwrap(), vec![3, 4]);
    }

    #[test]
    fn broadcast_shapes_scalar() {
        assert_eq!(broadcast_shapes(&[3, 4], &[]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[], &[5]).unwrap(), vec![5]);
    }

    #[test]
    fn broadcast_shapes_prepend_ones() {
        // (4,3) + (3,) -> (4,3)
        assert_eq!(broadcast_shapes(&[4, 3], &[3]).unwrap(), vec![4, 3]);
    }

    #[test]
    fn broadcast_shapes_stretch_ones() {
        // (4,1) * (4,3) -> (4,3)
        assert_eq!(broadcast_shapes(&[4, 1], &[4, 3]).unwrap(), vec![4, 3]);
    }

    #[test]
    fn broadcast_shapes_3d() {
        // (2,1,4) + (3,4) -> (2,3,4)
        assert_eq!(
            broadcast_shapes(&[2, 1, 4], &[3, 4]).unwrap(),
            vec![2, 3, 4]
        );
    }

    #[test]
    fn broadcast_shapes_both_ones() {
        // (1,3) + (2,1) -> (2,3)
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
    }

    #[test]
    fn broadcast_shapes_incompatible() {
        assert!(broadcast_shapes(&[3], &[4]).is_err());
        assert!(broadcast_shapes(&[2, 3], &[4, 3]).is_err());
    }

    #[test]
    fn broadcast_shapes_multi_test() {
        let result = broadcast_shapes_multi(&[&[2, 1], &[3], &[1, 3]]).unwrap();
        assert_eq!(result, vec![2, 3]);
    }

    #[test]
    fn broadcast_shapes_multi_empty() {
        assert_eq!(broadcast_shapes_multi(&[]).unwrap(), vec![]);
    }

    // -----------------------------------------------------------------------
    // broadcast_strides tests
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_strides_identity() {
        let strides = broadcast_strides(&[3, 4], &[3, 4], &[3, 4]).unwrap();
        assert_eq!(strides, vec![3, 4]);
    }

    #[test]
    fn broadcast_strides_expand_ones() {
        // shape (1,4) with strides (4,1) -> target (3,4)
        let strides = broadcast_strides(&[1, 4], &[4, 1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn broadcast_strides_prepend() {
        // shape (4,) with strides (1,) -> target (3, 4)
        let strides = broadcast_strides(&[4], &[1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
    }

    // -----------------------------------------------------------------------
    // broadcast_to tests
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_to_1d_to_2d() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = broadcast_to(&arr, &[4, 3]).unwrap();
        assert_eq!(view.shape(), &[4, 3]);
        assert_eq!(view.size(), 12);

        // All rows should be the same
        let data: Vec<f64> = view.iter().copied().collect();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn broadcast_to_column_to_2d() {
        // (3,1) -> (3,4)
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = broadcast_to(&arr, &[3, 4]).unwrap();
        assert_eq!(view.shape(), &[3, 4]);

        let data: Vec<f64> = view.iter().copied().collect();
        assert_eq!(
            data,
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn broadcast_to_no_materialization() {
        // Verify that broadcast_to does NOT copy data
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = broadcast_to(&arr, &[1000, 3]).unwrap();
        assert_eq!(view.shape(), &[1000, 3]);
        // The view shares the same base pointer
        assert_eq!(view.as_ptr(), arr.as_ptr());
    }

    #[test]
    fn broadcast_to_incompatible() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(broadcast_to(&arr, &[4, 5]).is_err());
    }

    #[test]
    fn broadcast_to_scalar() {
        // (1,) -> (5,)
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![42.0]).unwrap();
        let view = broadcast_to(&arr, &[5]).unwrap();
        assert_eq!(view.shape(), &[5]);
        let data: Vec<f64> = view.iter().copied().collect();
        assert_eq!(data, vec![42.0; 5]);
    }

    // -----------------------------------------------------------------------
    // broadcast_arrays tests
    // -----------------------------------------------------------------------

    #[test]
    fn broadcast_arrays_test() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([4, 1]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![10.0, 20.0, 30.0]).unwrap();
        let arrays = [a, b];
        let views = broadcast_arrays(&arrays).unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].shape(), &[4, 3]);
        assert_eq!(views[1].shape(), &[4, 3]);
    }

    // -----------------------------------------------------------------------
    // Method tests
    // -----------------------------------------------------------------------

    #[test]
    fn array_broadcast_to_method() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = arr.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
    }

    #[test]
    fn broadcast_3d() {
        // (2,1,4) + (3,4) -> (2,3,4)
        let a =
            Array::<i32, Ix3>::from_vec(Ix3::new([2, 1, 4]), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let view = a.broadcast_to(&[2, 3, 4]).unwrap();
        assert_eq!(view.shape(), &[2, 3, 4]);
        assert_eq!(view.size(), 24);
    }

    #[test]
    fn broadcast_to_same_shape() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let view = arr.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(view.shape(), &[2, 3]);
    }

    #[test]
    fn broadcast_to_cannot_shrink() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), vec![1.0; 12]).unwrap();
        assert!(arr.broadcast_to(&[3]).is_err());
    }
}
