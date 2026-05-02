// ferray-core: Basic indexing (REQ-12, REQ-14)
//
// Integer + slice indexing returning views (zero-copy).
// insert_axis / remove_axis for dimension manipulation.
//
// These are implemented as methods on Array, ArrayView, ArrayViewMut.
// The s![] macro is out of scope (Agent 1d).

use crate::array::owned::Array;
use crate::array::view::ArrayView;
use crate::array::view_mut::ArrayViewMut;
use crate::dimension::{Axis, Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

use super::normalize_index;

/// A slice specification for one axis, mirroring Python's `start:stop:step`.
///
/// All fields are optional (represented by `None`), matching `NumPy` behaviour:
/// - `start`: defaults to 0 (or end-1 if step < 0)
/// - `stop`: defaults to size (or before-start if step < 0)
/// - `step`: defaults to 1; must not be 0
#[derive(Debug, Clone, Copy)]
pub struct SliceSpec {
    /// Start index (inclusive). Negative counts from end.
    pub start: Option<isize>,
    /// Stop index (exclusive). Negative counts from end.
    pub stop: Option<isize>,
    /// Step size. Must not be zero.
    pub step: Option<isize>,
}

impl SliceSpec {
    /// Create a new full-range slice (equivalent to `:`).
    #[must_use]
    pub const fn full() -> Self {
        Self {
            start: None,
            stop: None,
            step: None,
        }
    }

    /// Create a slice `start:stop` with step 1.
    #[must_use]
    pub const fn new(start: isize, stop: isize) -> Self {
        Self {
            start: Some(start),
            stop: Some(stop),
            step: None,
        }
    }

    /// Create a slice `start:stop:step`.
    #[must_use]
    pub const fn with_step(start: isize, stop: isize, step: isize) -> Self {
        Self {
            start: Some(start),
            stop: Some(stop),
            step: Some(step),
        }
    }

    /// Validate that the step is not zero.
    fn validate(&self) -> FerrayResult<()> {
        if self.step == Some(0) {
            return Err(FerrayError::invalid_value("slice step cannot be zero"));
        }
        Ok(())
    }

    /// Convert to an ndarray Slice.
    #[allow(clippy::wrong_self_convention)]
    fn to_ndarray_slice(&self) -> ndarray::Slice {
        ndarray::Slice::new(self.start.unwrap_or(0), self.stop, self.step.unwrap_or(1))
    }

    /// Convert to an ndarray `SliceInfoElem` (used by s![] macro integration).
    #[allow(dead_code, clippy::wrong_self_convention)]
    pub(crate) fn to_ndarray_elem(&self) -> ndarray::SliceInfoElem {
        ndarray::SliceInfoElem::Slice {
            start: self.start.unwrap_or(0),
            end: self.stop,
            step: self.step.unwrap_or(1),
        }
    }
}

// ---------------------------------------------------------------------------
// Array methods — basic indexing
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> Array<T, D> {
    /// Index into the array along a given axis, removing that axis.
    ///
    /// Equivalent to `a[i]` for axis 0, or `a[:, i]` for axis 1, etc.
    /// Returns a view with one fewer dimension (dynamic-rank).
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis >= ndim`
    /// - `IndexOutOfBounds` if `index` is out of range (supports negative)
    pub fn index_axis(&self, axis: Axis, index: isize) -> FerrayResult<ArrayView<'_, T, D::Smaller>>
    where
        D::NdarrayDim:
            ndarray::RemoveAxis<Smaller = <D::Smaller as crate::dimension::Dimension>::NdarrayDim>,
    {
        // #349: return type now preserves rank — Array<T, Ix2>::index_axis
        // produces ArrayView<T, Ix1> instead of the old IxDyn fallback.
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        let size = self.shape()[ax];
        let idx = normalize_index(index, size, ax)?;

        let nd_axis = ndarray::Axis(ax);
        let sub = self.inner.index_axis(nd_axis, idx);
        Ok(ArrayView::from_ndarray(sub))
    }

    /// Slice the array along a given axis, returning a view.
    ///
    /// The returned view shares data with the source (zero-copy).
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis >= ndim`
    /// - `InvalidValue` if step is zero
    pub fn slice_axis(&self, axis: Axis, spec: SliceSpec) -> FerrayResult<ArrayView<'_, T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        spec.validate()?;

        let nd_axis = ndarray::Axis(ax);
        let nd_slice = spec.to_ndarray_slice();
        let sliced = self.inner.slice_axis(nd_axis, nd_slice);
        let dyn_view = sliced.into_dyn();
        Ok(ArrayView::from_ndarray(dyn_view))
    }

    /// Slice the array along a given axis, returning a mutable view.
    ///
    /// # Errors
    /// Same as [`slice_axis`](Self::slice_axis).
    pub fn slice_axis_mut(
        &mut self,
        axis: Axis,
        spec: SliceSpec,
    ) -> FerrayResult<ArrayViewMut<'_, T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        spec.validate()?;

        let nd_axis = ndarray::Axis(ax);
        let nd_slice = spec.to_ndarray_slice();
        let sliced = self.inner.slice_axis_mut(nd_axis, nd_slice);
        let dyn_view = sliced.into_dyn();
        Ok(ArrayViewMut::from_ndarray(dyn_view))
    }

    /// Multi-axis slicing: apply a slice specification to each axis.
    ///
    /// `specs` must have length equal to `ndim()`. For axes you don't
    /// want to slice, pass `SliceSpec::full()`.
    ///
    /// # Errors
    /// - `InvalidValue` if `specs.len() != ndim()`
    /// - Any errors from individual axis slicing
    pub fn slice_multi(&self, specs: &[SliceSpec]) -> FerrayResult<ArrayView<'_, T, IxDyn>> {
        let ndim = self.ndim();
        if specs.len() != ndim {
            return Err(FerrayError::invalid_value(format!(
                "expected {} slice specs, got {}",
                ndim,
                specs.len()
            )));
        }

        for spec in specs {
            spec.validate()?;
        }

        // Apply axis-by-axis slicing using move variants to preserve lifetimes
        let mut result = self.inner.view().into_dyn();
        for (ax, spec) in specs.iter().enumerate() {
            let nd_axis = ndarray::Axis(ax);
            let nd_slice = spec.to_ndarray_slice();
            result = result.slice_axis_move(nd_axis, nd_slice).into_dyn();
        }
        Ok(ArrayView::from_ndarray(result))
    }

    /// Insert a new axis of length 1 at the given position.
    ///
    /// This is equivalent to `np.expand_dims` or `np.newaxis`.
    /// Returns a view with rank one greater than `Self` (#349) —
    /// `Array<T, Ix2>::insert_axis` produces `ArrayView<T, Ix3>`.
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis > ndim`
    pub fn insert_axis(&self, axis: Axis) -> FerrayResult<ArrayView<'_, T, D::Larger>>
    where
        D::NdarrayDim:
            ndarray::Dimension<Larger = <D::Larger as crate::dimension::Dimension>::NdarrayDim>,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax > ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim + 1));
        }
        let expanded = self.inner.view().insert_axis(ndarray::Axis(ax));
        Ok(ArrayView::from_ndarray(expanded))
    }

    /// Remove an axis of length 1.
    ///
    /// This is equivalent to `np.squeeze` for a single axis.
    /// Returns a view with rank one less than `Self` (#349).
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis >= ndim`
    /// - `InvalidValue` if the axis has size != 1
    pub fn remove_axis(&self, axis: Axis) -> FerrayResult<ArrayView<'_, T, D::Smaller>>
    where
        D::NdarrayDim:
            ndarray::RemoveAxis<Smaller = <D::Smaller as crate::dimension::Dimension>::NdarrayDim>,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        if self.shape()[ax] != 1 {
            return Err(FerrayError::invalid_value(format!(
                "cannot remove axis {} with size {} (must be 1)",
                ax,
                self.shape()[ax]
            )));
        }
        let squeezed = self.inner.view().index_axis_move(ndarray::Axis(ax), 0);
        Ok(ArrayView::from_ndarray(squeezed))
    }

    /// Index into the array with a flat (linear) index.
    ///
    /// Elements are ordered in row-major (C) order. The flat index is
    /// converted to a multi-dimensional index via unravel, then stride
    /// arithmetic computes the physical offset in O(ndim) time.
    ///
    /// # Errors
    /// Returns `IndexOutOfBounds` if the index is out of range.
    pub fn flat_index(&self, index: isize) -> FerrayResult<&T> {
        let size = self.size();
        let idx = normalize_index(index, size, 0)?;

        // Unravel flat index to multi-dim index in row-major order,
        // then compute the physical offset via strides.
        let shape = self.shape();
        let strides = self.inner.strides();
        let base_ptr = self.inner.as_ptr();
        let ndim = shape.len();

        let mut remaining = idx;
        let mut offset: isize = 0;
        for d in 0..ndim {
            let dim_stride: usize = shape[d + 1..].iter().product::<usize>().max(1);
            let coord = remaining / dim_stride;
            remaining %= dim_stride;
            offset += coord as isize * strides[d];
        }

        // SAFETY: idx is validated in-bounds via normalize_index, and the
        // unravel produces coordinates within each axis, so the computed
        // offset is within the array's data allocation.
        Ok(unsafe { &*base_ptr.offset(offset) })
    }

    /// Get a reference to a single element by multi-dimensional index.
    ///
    /// Supports negative indices (counting from end).
    ///
    /// # Errors
    /// - `InvalidValue` if `indices.len() != ndim()`
    /// - `IndexOutOfBounds` if any index is out of range
    pub fn get(&self, indices: &[isize]) -> FerrayResult<&T> {
        let ndim = self.ndim();
        if indices.len() != ndim {
            return Err(FerrayError::invalid_value(format!(
                "expected {} indices, got {}",
                ndim,
                indices.len()
            )));
        }

        // Compute the flat offset manually
        let shape = self.shape();
        let strides = self.inner.strides();
        let base_ptr = self.inner.as_ptr();

        let mut offset: isize = 0;
        for (ax, &idx) in indices.iter().enumerate() {
            let pos = normalize_index(idx, shape[ax], ax)?;
            offset += pos as isize * strides[ax];
        }

        // SAFETY: all indices are validated in-bounds, so the computed
        // offset is within the array's data allocation.
        Ok(unsafe { &*base_ptr.offset(offset) })
    }

    /// Get a mutable reference to a single element by multi-dimensional index.
    ///
    /// # Errors
    /// Same as [`get`](Self::get).
    pub fn get_mut(&mut self, indices: &[isize]) -> FerrayResult<&mut T> {
        let ndim = self.ndim();
        if indices.len() != ndim {
            return Err(FerrayError::invalid_value(format!(
                "expected {} indices, got {}",
                ndim,
                indices.len()
            )));
        }

        let shape = self.shape().to_vec();
        let strides: Vec<isize> = self.inner.strides().to_vec();
        let base_ptr = self.inner.as_mut_ptr();

        let mut offset: isize = 0;
        for (ax, &idx) in indices.iter().enumerate() {
            let pos = normalize_index(idx, shape[ax], ax)?;
            offset += pos as isize * strides[ax];
        }

        // SAFETY: we have &mut self so exclusive access is guaranteed,
        // and all indices are validated in-bounds.
        Ok(unsafe { &mut *base_ptr.offset(offset) })
    }
}

// ---------------------------------------------------------------------------
// ArrayView methods — basic indexing
// ---------------------------------------------------------------------------

impl<'a, T: Element, D: Dimension> ArrayView<'a, T, D> {
    /// Index into the view along a given axis, removing that axis.
    /// Returns a view with rank one less than `Self` (#349).
    pub fn index_axis(&self, axis: Axis, index: isize) -> FerrayResult<ArrayView<'a, T, D::Smaller>>
    where
        D::NdarrayDim:
            ndarray::RemoveAxis<Smaller = <D::Smaller as crate::dimension::Dimension>::NdarrayDim>,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        let size = self.shape()[ax];
        let idx = normalize_index(index, size, ax)?;

        let nd_axis = ndarray::Axis(ax);
        // clone() on ArrayView is cheap (it's Copy-like)
        let sub = self.inner.clone().index_axis_move(nd_axis, idx);
        Ok(ArrayView::from_ndarray(sub))
    }

    /// Slice the view along a given axis.
    pub fn slice_axis(&self, axis: Axis, spec: SliceSpec) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        spec.validate()?;

        let nd_axis = ndarray::Axis(ax);
        let nd_slice = spec.to_ndarray_slice();
        // slice_axis on a cloned view preserves the 'a lifetime
        let sliced = self.inner.clone().slice_axis_move(nd_axis, nd_slice);
        let dyn_view = sliced.into_dyn();
        Ok(ArrayView::from_ndarray(dyn_view))
    }

    /// Insert a new axis of length 1 at the given position.
    /// Returns a view with rank one greater than `Self` (#349).
    pub fn insert_axis(&self, axis: Axis) -> FerrayResult<ArrayView<'a, T, D::Larger>>
    where
        D::NdarrayDim:
            ndarray::Dimension<Larger = <D::Larger as crate::dimension::Dimension>::NdarrayDim>,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax > ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim + 1));
        }
        let expanded = self.inner.clone().insert_axis(ndarray::Axis(ax));
        Ok(ArrayView::from_ndarray(expanded))
    }

    /// Remove an axis of length 1.
    /// Returns a view with rank one less than `Self` (#349).
    pub fn remove_axis(&self, axis: Axis) -> FerrayResult<ArrayView<'a, T, D::Smaller>>
    where
        D::NdarrayDim:
            ndarray::RemoveAxis<Smaller = <D::Smaller as crate::dimension::Dimension>::NdarrayDim>,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        if self.shape()[ax] != 1 {
            return Err(FerrayError::invalid_value(format!(
                "cannot remove axis {} with size {} (must be 1)",
                ax,
                self.shape()[ax]
            )));
        }

        let squeezed = self.inner.clone().index_axis_move(ndarray::Axis(ax), 0);
        Ok(ArrayView::from_ndarray(squeezed))
    }

    /// Get a reference to a single element by multi-dimensional index.
    pub fn get(&self, indices: &[isize]) -> FerrayResult<&'a T> {
        let ndim = self.ndim();
        if indices.len() != ndim {
            return Err(FerrayError::invalid_value(format!(
                "expected {} indices, got {}",
                ndim,
                indices.len()
            )));
        }

        let shape = self.shape();
        let strides = self.inner.strides();
        let base_ptr = self.inner.as_ptr();

        let mut offset: isize = 0;
        for (ax, &idx) in indices.iter().enumerate() {
            let pos = normalize_index(idx, shape[ax], ax)?;
            offset += pos as isize * strides[ax];
        }

        // SAFETY: indices validated in-bounds; the pointer is valid for 'a.
        Ok(unsafe { &*base_ptr.offset(offset) })
    }
}

// ---------------------------------------------------------------------------
// ArrayViewMut methods — basic indexing
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> ArrayViewMut<'_, T, D> {
    /// Slice the mutable view along a given axis.
    pub fn slice_axis_mut(
        &mut self,
        axis: Axis,
        spec: SliceSpec,
    ) -> FerrayResult<ArrayViewMut<'_, T, IxDyn>> {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        spec.validate()?;

        let nd_axis = ndarray::Axis(ax);
        let nd_slice = spec.to_ndarray_slice();
        let sliced = self.inner.slice_axis_mut(nd_axis, nd_slice);
        let dyn_view = sliced.into_dyn();
        Ok(ArrayViewMut::from_ndarray(dyn_view))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2, Ix3};

    // -----------------------------------------------------------------------
    // Normalization
    // -----------------------------------------------------------------------

    #[test]
    fn normalize_positive_in_bounds() {
        assert_eq!(normalize_index(2, 5, 0).unwrap(), 2);
    }

    #[test]
    fn normalize_negative() {
        assert_eq!(normalize_index(-1, 5, 0).unwrap(), 4);
        assert_eq!(normalize_index(-5, 5, 0).unwrap(), 0);
    }

    #[test]
    fn normalize_out_of_bounds() {
        assert!(normalize_index(5, 5, 0).is_err());
        assert!(normalize_index(-6, 5, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // index_axis
    // -----------------------------------------------------------------------

    #[test]
    fn index_axis_row() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let row = arr.index_axis(Axis(0), 1).unwrap();
        assert_eq!(row.shape(), &[4]);
        let data: Vec<i32> = row.iter().copied().collect();
        assert_eq!(data, vec![4, 5, 6, 7]);
    }

    #[test]
    fn index_axis_column() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let col = arr.index_axis(Axis(1), 2).unwrap();
        assert_eq!(col.shape(), &[3]);
        let data: Vec<i32> = col.iter().copied().collect();
        assert_eq!(data, vec![2, 6, 10]);
    }

    #[test]
    fn index_axis_negative() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let row = arr.index_axis(Axis(0), -1).unwrap();
        let data: Vec<i32> = row.iter().copied().collect();
        assert_eq!(data, vec![8, 9, 10, 11]);
    }

    #[test]
    fn index_axis_out_of_bounds() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        assert!(arr.index_axis(Axis(0), 3).is_err());
        assert!(arr.index_axis(Axis(2), 0).is_err());
    }

    #[test]
    fn index_axis_is_zero_copy() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let row = arr.index_axis(Axis(0), 0).unwrap();
        assert_eq!(row.as_ptr(), arr.as_ptr());
    }

    // -----------------------------------------------------------------------
    // slice_axis
    // -----------------------------------------------------------------------

    #[test]
    fn slice_axis_basic() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let sliced = arr.slice_axis(Axis(0), SliceSpec::new(1, 4)).unwrap();
        assert_eq!(sliced.shape(), &[3]);
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![20, 30, 40]);
    }

    #[test]
    fn slice_axis_step() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![0, 1, 2, 3, 4, 5]).unwrap();
        let sliced = arr
            .slice_axis(Axis(0), SliceSpec::with_step(0, 6, 2))
            .unwrap();
        assert_eq!(sliced.shape(), &[3]);
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![0, 2, 4]);
    }

    #[test]
    fn slice_axis_negative_step() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 2, 3, 4]).unwrap();
        // Reverse the entire array: range [0, len) traversed backwards
        // ndarray interprets Slice::new(start, end, step) where start/end define
        // a forward range and negative step reverses traversal within it.
        let spec = SliceSpec {
            start: None,
            stop: None,
            step: Some(-1),
        };
        let sliced = arr.slice_axis(Axis(0), spec).unwrap();
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn slice_axis_negative_step_partial() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 2, 3, 4]).unwrap();
        // Range [1, 4) traversed backwards with step -1: [3, 2, 1]
        let sliced = arr
            .slice_axis(Axis(0), SliceSpec::with_step(1, 4, -1))
            .unwrap();
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![3, 2, 1]);
    }

    #[test]
    fn slice_axis_full() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let sliced = arr.slice_axis(Axis(0), SliceSpec::full()).unwrap();
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn slice_axis_2d_rows() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([4, 3]), (0..12).collect()).unwrap();
        let sliced = arr.slice_axis(Axis(0), SliceSpec::new(1, 3)).unwrap();
        assert_eq!(sliced.shape(), &[2, 3]);
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn slice_axis_is_zero_copy() {
        let arr =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let sliced = arr.slice_axis(Axis(0), SliceSpec::new(1, 4)).unwrap();
        unsafe {
            assert_eq!(*sliced.as_ptr(), 2.0);
        }
    }

    #[test]
    fn slice_axis_zero_step_error() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(
            arr.slice_axis(Axis(0), SliceSpec::with_step(0, 3, 0))
                .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // slice_multi
    // -----------------------------------------------------------------------

    #[test]
    fn slice_multi_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([4, 5]), (0..20).collect()).unwrap();
        let sliced = arr
            .slice_multi(&[SliceSpec::new(1, 3), SliceSpec::new(0, 4)])
            .unwrap();
        assert_eq!(sliced.shape(), &[2, 4]);
    }

    #[test]
    fn slice_multi_wrong_count() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), (0..6).collect()).unwrap();
        assert!(arr.slice_multi(&[SliceSpec::full()]).is_err());
    }

    // -----------------------------------------------------------------------
    // insert_axis / remove_axis
    // -----------------------------------------------------------------------

    #[test]
    fn insert_axis_at_front() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expanded = arr.insert_axis(Axis(0)).unwrap();
        assert_eq!(expanded.shape(), &[1, 3]);
    }

    #[test]
    fn insert_axis_at_end() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let expanded = arr.insert_axis(Axis(1)).unwrap();
        assert_eq!(expanded.shape(), &[3, 1]);
    }

    #[test]
    fn insert_axis_out_of_bounds() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(arr.insert_axis(Axis(3)).is_err());
    }

    #[test]
    fn remove_axis_single() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let squeezed = arr.remove_axis(Axis(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3]);
        let data: Vec<f64> = squeezed.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn remove_axis_not_one() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        assert!(arr.remove_axis(Axis(0)).is_err());
    }

    #[test]
    fn remove_axis_out_of_bounds() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(arr.remove_axis(Axis(1)).is_err());
    }

    // -----------------------------------------------------------------------
    // flat_index
    // -----------------------------------------------------------------------

    #[test]
    fn flat_index_positive() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(*arr.flat_index(0).unwrap(), 1);
        assert_eq!(*arr.flat_index(3).unwrap(), 4);
        assert_eq!(*arr.flat_index(5).unwrap(), 6);
    }

    #[test]
    fn flat_index_negative() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        assert_eq!(*arr.flat_index(-1).unwrap(), 50);
        assert_eq!(*arr.flat_index(-5).unwrap(), 10);
    }

    #[test]
    fn flat_index_out_of_bounds() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(arr.flat_index(3).is_err());
        assert!(arr.flat_index(-4).is_err());
    }

    // -----------------------------------------------------------------------
    // get / get_mut
    // -----------------------------------------------------------------------

    #[test]
    fn get_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        assert_eq!(*arr.get(&[0, 0]).unwrap(), 0);
        assert_eq!(*arr.get(&[1, 2]).unwrap(), 6);
        assert_eq!(*arr.get(&[2, 3]).unwrap(), 11);
    }

    #[test]
    fn get_negative_indices() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        assert_eq!(*arr.get(&[-1, -1]).unwrap(), 11);
        assert_eq!(*arr.get(&[-3, 0]).unwrap(), 0);
    }

    #[test]
    fn get_wrong_ndim() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), (0..6).collect()).unwrap();
        assert!(arr.get(&[0]).is_err());
        assert!(arr.get(&[0, 0, 0]).is_err());
    }

    #[test]
    fn get_mut_modify() {
        let mut arr =
            Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        *arr.get_mut(&[1, 2]).unwrap() = 99;
        assert_eq!(*arr.get(&[1, 2]).unwrap(), 99);
    }

    // -----------------------------------------------------------------------
    // ArrayView basic indexing
    // -----------------------------------------------------------------------

    #[test]
    fn view_index_axis() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let v = arr.view();
        let row = v.index_axis(Axis(0), 1).unwrap();
        let data: Vec<i32> = row.iter().copied().collect();
        assert_eq!(data, vec![4, 5, 6, 7]);
    }

    #[test]
    fn view_slice_axis() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let v = arr.view();
        let sliced = v.slice_axis(Axis(0), SliceSpec::new(1, 4)).unwrap();
        let data: Vec<i32> = sliced.iter().copied().collect();
        assert_eq!(data, vec![20, 30, 40]);
    }

    #[test]
    fn view_insert_remove_axis() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = arr.view();
        let expanded = v.insert_axis(Axis(0)).unwrap();
        assert_eq!(expanded.shape(), &[1, 4]);
        let squeezed = expanded.remove_axis(Axis(0)).unwrap();
        assert_eq!(squeezed.shape(), &[4]);
    }

    #[test]
    fn view_get() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = arr.view();
        assert_eq!(*v.get(&[1, 2]).unwrap(), 6);
    }

    // -----------------------------------------------------------------------
    // ArrayViewMut slice
    // -----------------------------------------------------------------------

    #[test]
    fn view_mut_slice_axis() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        {
            let mut vm = arr.view_mut();
            let mut sliced = vm.slice_axis_mut(Axis(0), SliceSpec::new(1, 3)).unwrap();
            if let Some(s) = sliced.as_slice_mut() {
                s[0] = 20;
                s[1] = 30;
            }
        }
        assert_eq!(arr.as_slice().unwrap(), &[1, 20, 30, 4, 5]);
    }

    // -----------------------------------------------------------------------
    // 3D indexing
    // -----------------------------------------------------------------------

    #[test]
    fn index_axis_3d() {
        let arr = Array::<i32, Ix3>::from_vec(Ix3::new([2, 3, 4]), (0..24).collect()).unwrap();
        let plane = arr.index_axis(Axis(0), 1).unwrap();
        assert_eq!(plane.shape(), &[3, 4]);
        assert_eq!(*plane.get(&[0, 0]).unwrap(), 12);
    }

    // ---- Rank-preserving indexing (#349) ------------------------------
    //
    // index_axis on a fixed-rank Array now returns a fixed-rank
    // ArrayView with one fewer axis; insert_axis adds one axis. These
    // tests pin the *static type* of the result via explicit type
    // annotations (compile-time check, not just runtime shape).

    #[test]
    fn index_axis_2d_returns_ix1() {
        let arr = Array::<f64, crate::dimension::Ix2>::from_vec(
            crate::dimension::Ix2::new([3, 4]),
            (0..12).map(|i| i as f64).collect(),
        )
        .unwrap();
        // The variable's type is explicit — must compile to Ix1, not IxDyn.
        let row: ArrayView<'_, f64, crate::dimension::Ix1> = arr.index_axis(Axis(0), 1).unwrap();
        assert_eq!(row.shape(), &[4]);
    }

    #[test]
    fn index_axis_3d_returns_ix2() {
        let arr = Array::<i32, Ix3>::from_vec(Ix3::new([2, 3, 4]), (0..24).collect()).unwrap();
        let plane: ArrayView<'_, i32, crate::dimension::Ix2> = arr.index_axis(Axis(0), 0).unwrap();
        assert_eq!(plane.shape(), &[3, 4]);
    }

    #[test]
    fn insert_axis_ix2_returns_ix3() {
        let arr = Array::<f64, crate::dimension::Ix2>::from_vec(
            crate::dimension::Ix2::new([2, 3]),
            (0..6).map(|i| i as f64).collect(),
        )
        .unwrap();
        let expanded: ArrayView<'_, f64, Ix3> = arr.insert_axis(Axis(0)).unwrap();
        assert_eq!(expanded.shape(), &[1, 2, 3]);
    }

    #[test]
    fn remove_axis_ix3_returns_ix2() {
        let arr =
            Array::<f64, Ix3>::from_vec(Ix3::new([1, 3, 4]), (0..12).map(|i| i as f64).collect())
                .unwrap();
        let squeezed: ArrayView<'_, f64, crate::dimension::Ix2> = arr.remove_axis(Axis(0)).unwrap();
        assert_eq!(squeezed.shape(), &[3, 4]);
    }

    #[test]
    fn index_axis_chains_preserve_rank_at_each_step() {
        // Ix3 → Ix2 → Ix1 → Ix0 chain.
        let arr = Array::<i32, Ix3>::from_vec(Ix3::new([2, 3, 4]), (0..24).collect()).unwrap();
        let plane: ArrayView<'_, i32, crate::dimension::Ix2> = arr.index_axis(Axis(0), 1).unwrap();
        let row: ArrayView<'_, i32, crate::dimension::Ix1> = plane.index_axis(Axis(0), 1).unwrap();
        let scalar: ArrayView<'_, i32, crate::dimension::Ix0> = row.index_axis(Axis(0), 2).unwrap();
        assert_eq!(scalar.shape(), &[] as &[usize]);
    }
}
