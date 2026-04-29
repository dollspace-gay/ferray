// ferray-core: Advanced (fancy) indexing (REQ-13, REQ-15)
//
// - index_select: integer-array indexing along an axis → copies
// - boolean_index: boolean-mask indexing → copies (always 1-D)
// - boolean_index_assign: masked assignment (a[mask] = value)
//
// All advanced indexing operations return COPIES, not views.

use super::normalize_index;
use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::dimension::{Axis, Dimension, Ix1, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ---------------------------------------------------------------------------
// index_select
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> Array<T, D> {
    /// Select elements along an axis using an array of indices.
    ///
    /// This is advanced (fancy) indexing — it always returns a **copy**.
    /// The result has the same number of dimensions as the input, but
    /// the size along `axis` is replaced by `indices.len()`.
    ///
    /// Negative indices are supported (counting from end).
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis >= ndim`
    /// - `IndexOutOfBounds` if any index is out of range
    pub fn index_select(&self, axis: Axis, indices: &[isize]) -> FerrayResult<Array<T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        let axis_size = self.shape()[ax];

        // Normalize all indices
        let normalized: Vec<usize> = indices
            .iter()
            .map(|&idx| normalize_index(idx, axis_size, ax))
            .collect::<FerrayResult<Vec<_>>>()?;

        let dyn_view = self.inner.view().into_dyn();
        let nd_axis = ndarray::Axis(ax);
        let selected = dyn_view.select(nd_axis, &normalized);
        Ok(Array::from_ndarray(selected))
    }

    /// Select elements using a boolean mask.
    ///
    /// Returns a 1-D array containing elements where `mask` is `true`.
    /// This is always a **copy**.
    ///
    /// The mask must be broadcastable to the array's shape, or have the
    /// same total number of elements. When the mask is 1-D and the array
    /// is N-D, the mask is applied to the flattened array.
    ///
    /// # Errors
    /// - `ShapeMismatch` if mask shape is incompatible
    pub fn boolean_index(&self, mask: &Array<bool, D>) -> FerrayResult<Array<T, Ix1>> {
        if self.shape() != mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "boolean index mask shape {:?} does not match array shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }

        let data: Vec<T> = self
            .inner
            .iter()
            .zip(mask.inner.iter())
            .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
            .collect();

        let len = data.len();
        Array::from_vec(Ix1::new([len]), data)
    }

    /// Boolean indexing with a flat mask (1-D mask on N-D array).
    ///
    /// The mask is applied to the flattened (row-major) array.
    ///
    /// # Errors
    /// - `ShapeMismatch` if `mask.len() != self.size()`
    pub fn boolean_index_flat(&self, mask: &Array<bool, Ix1>) -> FerrayResult<Array<T, Ix1>> {
        if mask.size() != self.size() {
            return Err(FerrayError::shape_mismatch(format!(
                "flat boolean mask length {} does not match array size {}",
                mask.size(),
                self.size()
            )));
        }

        let data: Vec<T> = self
            .inner
            .iter()
            .zip(mask.inner.iter())
            .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
            .collect();

        let len = data.len();
        Array::from_vec(Ix1::new([len]), data)
    }

    /// Assign a scalar value to elements selected by a boolean mask.
    ///
    /// Equivalent to `a[mask] = value` in `NumPy`.
    ///
    /// # Errors
    /// - `ShapeMismatch` if mask shape differs from array shape
    pub fn boolean_index_assign(&mut self, mask: &Array<bool, D>, value: T) -> FerrayResult<()> {
        if self.shape() != mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "boolean index mask shape {:?} does not match array shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }

        for (elem, &m) in self.inner.iter_mut().zip(mask.inner.iter()) {
            if m {
                *elem = value.clone();
            }
        }
        Ok(())
    }

    /// Assign values from an array to elements selected by a boolean mask.
    ///
    /// `values` must have exactly as many elements as `mask` has `true`
    /// entries.
    ///
    /// # Errors
    /// - `ShapeMismatch` if mask shape differs or values length mismatches
    pub fn boolean_index_assign_array(
        &mut self,
        mask: &Array<bool, D>,
        values: &Array<T, Ix1>,
    ) -> FerrayResult<()> {
        if self.shape() != mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "boolean index mask shape {:?} does not match array shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }

        let true_count = mask.inner.iter().filter(|&&m| m).count();
        if values.size() != true_count {
            return Err(FerrayError::shape_mismatch(format!(
                "values array has {} elements but mask has {} true entries",
                values.size(),
                true_count
            )));
        }

        let mut val_iter = values.inner.iter();
        for (elem, &m) in self.inner.iter_mut().zip(mask.inner.iter()) {
            if m {
                if let Some(v) = val_iter.next() {
                    *elem = v.clone();
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ArrayView advanced indexing
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> ArrayView<'_, T, D> {
    /// Select elements along an axis using an array of indices (copy).
    pub fn index_select(&self, axis: Axis, indices: &[isize]) -> FerrayResult<Array<T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        let axis_size = self.shape()[ax];

        let normalized: Vec<usize> = indices
            .iter()
            .map(|&idx| normalize_index(idx, axis_size, ax))
            .collect::<FerrayResult<Vec<_>>>()?;

        let dyn_view = self.inner.clone().into_dyn();
        let nd_axis = ndarray::Axis(ax);
        let selected = dyn_view.select(nd_axis, &normalized);
        Ok(Array::from_ndarray(selected))
    }

    /// Select elements using a boolean mask (copy).
    pub fn boolean_index(&self, mask: &Array<bool, D>) -> FerrayResult<Array<T, Ix1>> {
        if self.shape() != mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "boolean index mask shape {:?} does not match view shape {:?}",
                mask.shape(),
                self.shape()
            )));
        }

        let data: Vec<T> = self
            .inner
            .iter()
            .zip(mask.inner.iter())
            .filter_map(|(val, &m)| if m { Some(val.clone()) } else { None })
            .collect();

        let len = data.len();
        Array::from_vec(Ix1::new([len]), data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    // -----------------------------------------------------------------------
    // index_select
    // -----------------------------------------------------------------------

    #[test]
    fn index_select_rows() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([4, 3]), (0..12).collect()).unwrap();
        let sel = arr.index_select(Axis(0), &[0, 2, 3]).unwrap();
        assert_eq!(sel.shape(), &[3, 3]);
        let data: Vec<i32> = sel.iter().copied().collect();
        assert_eq!(data, vec![0, 1, 2, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn index_select_columns() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let sel = arr.index_select(Axis(1), &[0, 2]).unwrap();
        assert_eq!(sel.shape(), &[3, 2]);
        let data: Vec<i32> = sel.iter().copied().collect();
        assert_eq!(data, vec![0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn index_select_negative() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let sel = arr.index_select(Axis(0), &[-1, -3]).unwrap();
        assert_eq!(sel.shape(), &[2]);
        let data: Vec<i32> = sel.iter().copied().collect();
        assert_eq!(data, vec![50, 30]);
    }

    #[test]
    fn index_select_out_of_bounds() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(arr.index_select(Axis(0), &[3]).is_err());
        assert!(arr.index_select(Axis(0), &[-4]).is_err());
    }

    #[test]
    fn index_select_returns_copy() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let sel = arr.index_select(Axis(0), &[0, 1]).unwrap();
        // Should be a different allocation
        assert_ne!(sel.as_ptr() as usize, arr.as_ptr() as usize);
    }

    #[test]
    fn index_select_duplicate_indices() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let sel = arr.index_select(Axis(0), &[1, 1, 0, 2, 2]).unwrap();
        assert_eq!(sel.shape(), &[5]);
        let data: Vec<i32> = sel.iter().copied().collect();
        assert_eq!(data, vec![20, 20, 10, 30, 30]);
    }

    #[test]
    fn index_select_empty() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let sel = arr.index_select(Axis(0), &[]).unwrap();
        assert_eq!(sel.shape(), &[0]);
    }

    // -----------------------------------------------------------------------
    // boolean_index
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_index_1d() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let selected = arr.boolean_index(&mask).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.as_slice().unwrap(), &[10, 30, 50]);
    }

    #[test]
    fn boolean_index_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, false, true, false, true, false],
        )
        .unwrap();
        let selected = arr.boolean_index(&mask).unwrap();
        assert_eq!(selected.shape(), &[3]);
        assert_eq!(selected.as_slice().unwrap(), &[1, 3, 5]);
    }

    #[test]
    fn boolean_index_all_false() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let selected = arr.boolean_index(&mask).unwrap();
        assert_eq!(selected.shape(), &[0]);
    }

    #[test]
    fn boolean_index_shape_mismatch() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([2]), vec![true, false]).unwrap();
        assert!(arr.boolean_index(&mask).is_err());
    }

    #[test]
    fn boolean_index_returns_copy() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        let selected = arr.boolean_index(&mask).unwrap();
        assert_ne!(selected.as_ptr() as usize, arr.as_ptr() as usize);
    }

    // -----------------------------------------------------------------------
    // boolean_index_flat
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_index_flat_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(
            Ix1::new([6]),
            vec![false, true, false, true, false, true],
        )
        .unwrap();
        let selected = arr.boolean_index_flat(&mask).unwrap();
        assert_eq!(selected.as_slice().unwrap(), &[2, 4, 6]);
    }

    #[test]
    fn boolean_index_flat_wrong_size() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        assert!(arr.boolean_index_flat(&mask).is_err());
    }

    // -----------------------------------------------------------------------
    // boolean_index_assign
    // -----------------------------------------------------------------------

    #[test]
    fn boolean_assign_scalar() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        arr.boolean_index_assign(&mask, 0).unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0, 2, 0, 4, 0]);
    }

    #[test]
    fn boolean_assign_array() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, true, false, true, false])
                .unwrap();
        let values = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![99, 88]).unwrap();
        arr.boolean_index_assign_array(&mask, &values).unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[1, 99, 3, 88, 5]);
    }

    #[test]
    fn boolean_assign_array_wrong_count() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, false]).unwrap();
        let values = Array::<i32, Ix1>::from_vec(Ix1::new([1]), vec![99]).unwrap();
        assert!(arr.boolean_index_assign_array(&mask, &values).is_err());
    }

    #[test]
    fn boolean_assign_2d() {
        let mut arr =
            Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, false, false, true, false],
        )
        .unwrap();
        arr.boolean_index_assign(&mask, -1).unwrap();
        let data: Vec<i32> = arr.iter().copied().collect();
        assert_eq!(data, vec![1, -1, 3, 4, -1, 6]);
    }

    // -----------------------------------------------------------------------
    // ArrayView advanced indexing
    // -----------------------------------------------------------------------

    #[test]
    fn view_index_select() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let v = arr.view();
        let sel = v.index_select(Axis(1), &[0, 3]).unwrap();
        assert_eq!(sel.shape(), &[3, 2]);
        let data: Vec<i32> = sel.iter().copied().collect();
        assert_eq!(data, vec![0, 3, 4, 7, 8, 11]);
    }

    #[test]
    fn view_boolean_index() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let v = arr.view();
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, false, true]).unwrap();
        let selected = v.boolean_index(&mask).unwrap();
        assert_eq!(selected.as_slice().unwrap(), &[10, 40]);
    }
}
