// ferray-core: Iterator implementations (REQ-37)

use crate::dimension::{Axis, Dimension, Ix1, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

use super::owned::Array;
use super::view::ArrayView;
use super::view_mut::ArrayViewMut;

// ---------------------------------------------------------------------------
// Element iteration for Array<T, D>
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> Array<T, D> {
    /// Iterate over all elements in logical (row-major) order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
    }

    /// Mutably iterate over all elements in logical order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> + '_ {
        self.inner.iter_mut()
    }

    /// Iterate with multi-dimensional indices.
    ///
    /// Yields `(Vec<usize>, &T)` pairs in logical order. The index vector
    /// has one entry per dimension.
    pub fn indexed_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> + '_ {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        self.inner.iter().enumerate().map(move |(flat_idx, val)| {
            let mut idx = vec![0usize; ndim];
            let mut rem = flat_idx;
            // Convert flat index to multi-dimensional (assuming C-order iteration)
            for (d, s) in shape.iter().enumerate().rev() {
                if *s > 0 {
                    idx[d] = rem % s;
                    rem /= s;
                }
            }
            (idx, val)
        })
    }

    /// Flat iterator — same as `iter()` but emphasises logical-order traversal.
    pub fn flat(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
    }

    /// Iterate over lanes (1-D slices) along the given axis.
    ///
    /// For a 2-D array with `axis=1`, this yields each row.
    /// For `axis=0`, this yields each column.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn lanes(
        &self,
        axis: Axis,
    ) -> FerrayResult<impl Iterator<Item = ArrayView<'_, T, Ix1>> + '_> {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis.index(), ndim));
        }
        let nd_axis = ndarray::Axis(axis.index());
        Ok(self
            .inner
            .lanes(nd_axis)
            .into_iter()
            .map(|lane| ArrayView::from_ndarray(lane)))
    }

    /// Iterate over sub-arrays along the given axis.
    ///
    /// For a 3-D array with shape `[2,3,4]` and `axis=0`, this yields
    /// two 2-D views each of shape `[3,4]`.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn axis_iter(
        &self,
        axis: Axis,
    ) -> FerrayResult<impl Iterator<Item = ArrayView<'_, T, IxDyn>> + '_>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis.index(), ndim));
        }
        let nd_axis = ndarray::Axis(axis.index());
        Ok(self.inner.axis_iter(nd_axis).map(|sub| {
            let dyn_view = sub.into_dyn();
            ArrayView::from_ndarray(dyn_view)
        }))
    }

    /// Mutably iterate over sub-arrays along the given axis.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn axis_iter_mut(
        &mut self,
        axis: Axis,
    ) -> FerrayResult<impl Iterator<Item = ArrayViewMut<'_, T, IxDyn>> + '_>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis.index(), ndim));
        }
        let nd_axis = ndarray::Axis(axis.index());
        Ok(self.inner.axis_iter_mut(nd_axis).map(|sub| {
            let dyn_view = sub.into_dyn();
            ArrayViewMut::from_ndarray(dyn_view)
        }))
    }
}

// ---------------------------------------------------------------------------
// Consuming iterator
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> IntoIterator for Array<T, D> {
    type Item = T;
    type IntoIter = ndarray::iter::IntoIter<T, D::NdarrayDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<'a, T: Element, D: Dimension> IntoIterator for &'a Array<T, D> {
    type Item = &'a T;
    type IntoIter = ndarray::iter::Iter<'a, T, D::NdarrayDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter()
    }
}

impl<'a, T: Element, D: Dimension> IntoIterator for &'a mut Array<T, D> {
    type Item = &'a mut T;
    type IntoIter = ndarray::iter::IterMut<'a, T, D::NdarrayDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter_mut()
    }
}

// ---------------------------------------------------------------------------
// ArrayView iteration
// ---------------------------------------------------------------------------

impl<'a, T: Element, D: Dimension> ArrayView<'a, T, D> {
    /// Iterate over all elements in logical order.
    pub fn iter(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
    }

    /// Flat iterator.
    pub fn flat(&self) -> impl Iterator<Item = &T> + '_ {
        self.inner.iter()
    }

    /// Iterate with multi-dimensional indices.
    pub fn indexed_iter(&self) -> impl Iterator<Item = (Vec<usize>, &T)> + '_ {
        let shape = self.shape().to_vec();
        let ndim = shape.len();
        self.inner.iter().enumerate().map(move |(flat_idx, val)| {
            let mut idx = vec![0usize; ndim];
            let mut rem = flat_idx;
            for (d, s) in shape.iter().enumerate().rev() {
                if *s > 0 {
                    idx[d] = rem % s;
                    rem /= s;
                }
            }
            (idx, val)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn iter_elements() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let collected: Vec<f64> = arr.iter().copied().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn iter_mut_elements() {
        let mut arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        for x in arr.iter_mut() {
            *x *= 2.0;
        }
        assert_eq!(arr.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn into_iter_consuming() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let collected: Vec<i32> = arr.into_iter().collect();
        assert_eq!(collected, vec![10, 20, 30]);
    }

    #[test]
    fn indexed_iter_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let items: Vec<_> = arr.indexed_iter().collect();
        assert_eq!(items.len(), 6);
        assert_eq!(items[0], (vec![0, 0], &1));
        assert_eq!(items[1], (vec![0, 1], &2));
        assert_eq!(items[3], (vec![1, 0], &4));
    }

    #[test]
    fn flat_iterator() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let flat: Vec<f64> = arr.flat().copied().collect();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn lanes_axis1() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let rows: Vec<Vec<f64>> = arr
            .lanes(Axis(1))
            .unwrap()
            .map(|lane| lane.iter().copied().collect())
            .collect();
        assert_eq!(rows, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn lanes_out_of_bounds() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 4])).unwrap();
        assert!(arr.lanes(Axis(2)).is_err());
    }

    #[test]
    fn axis_iter_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 2]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let rows: Vec<Vec<i32>> = arr
            .axis_iter(Axis(0))
            .unwrap()
            .map(|sub| sub.iter().copied().collect())
            .collect();
        assert_eq!(rows, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);
    }

    #[test]
    fn axis_iter_out_of_bounds() {
        let arr = Array::<f64, Ix1>::zeros(Ix1::new([5])).unwrap();
        assert!(arr.axis_iter(Axis(1)).is_err());
    }

    #[test]
    fn axis_iter_mut_modify() {
        let mut arr =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        for mut row in arr.axis_iter_mut(Axis(0)).unwrap() {
            if let Some(s) = row.as_slice_mut() {
                for v in s.iter_mut() {
                    *v *= 10.0;
                }
            }
        }
        assert_eq!(
            arr.as_slice().unwrap(),
            &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        );
    }

    #[test]
    fn for_loop_borrow() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let mut sum = 0;
        for &x in &arr {
            sum += x;
        }
        assert_eq!(sum, 60);
    }
}
