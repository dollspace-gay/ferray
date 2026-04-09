// ferray-core: Closure-based operations (REQ-38)
//   mapv, mapv_inplace, zip_mut_with, fold_axis
//   as_standard_layout, as_fortran_layout (#351)

use crate::dimension::{Axis, Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

use super::cow::CowArray;
use super::owned::Array;
use super::view::ArrayView;

impl<T: Element, D: Dimension> Array<T, D> {
    /// Apply a closure to every element, returning a new array.
    ///
    /// The closure receives each element by value (cloned) and must return
    /// the same type. For type-changing maps, collect via iterators.
    pub fn mapv(&self, f: impl Fn(T) -> T) -> Self {
        let inner = self.inner.mapv(&f);
        Self::from_ndarray(inner)
    }

    /// Apply a closure to every element in place.
    pub fn mapv_inplace(&mut self, f: impl Fn(T) -> T) {
        self.inner.mapv_inplace(&f);
    }

    /// Zip this array mutably with another array of the same shape,
    /// applying a closure to each pair of elements.
    ///
    /// The closure receives `(&mut T, &T)` — the first element is from
    /// `self` and can be modified, the second is from `other`.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if shapes differ.
    pub fn zip_mut_with(
        &mut self,
        other: &Array<T, D>,
        f: impl Fn(&mut T, &T),
    ) -> FerrayResult<()> {
        if self.shape() != other.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "cannot zip arrays with shapes {:?} and {:?}",
                self.shape(),
                other.shape(),
            )));
        }
        self.inner.zip_mut_with(&other.inner, |a, b| f(a, b));
        Ok(())
    }

    /// Fold (reduce) along the given axis.
    ///
    /// `init` provides the initial accumulator value for each lane.
    /// The closure receives `(accumulator, &element)` and must return
    /// the new accumulator.
    ///
    /// Returns an array with one fewer dimension (the folded axis removed).
    /// The result is always returned as a dynamic-rank array.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
    pub fn fold_axis(
        &self,
        axis: Axis,
        init: T,
        fold: impl FnMut(&T, &T) -> T,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis.index(), ndim));
        }
        let nd_axis = ndarray::Axis(axis.index());
        let mut fold = fold;
        let result = self.inner.fold_axis(nd_axis, init, |acc, x| fold(acc, x));
        let dyn_result = result.into_dyn();
        Ok(Array::from_ndarray(dyn_result))
    }

    /// Apply a closure elementwise, producing an array of a different type.
    ///
    /// Unlike `mapv` which preserves the element type, this allows
    /// mapping to a different `Element` type.
    pub fn map_to<U: Element>(&self, f: impl Fn(T) -> U) -> Array<U, D> {
        let inner = self.inner.mapv(&f);
        Array::from_ndarray(inner)
    }

    /// Convert this array to a dynamic-rank `Array<T, IxDyn>`.
    ///
    /// This is a cheap conversion (re-wraps the same data) and is needed
    /// to call functions like [`crate::manipulation::concatenate`] which
    /// take dynamic-rank slices.
    pub fn to_dyn(&self) -> Array<T, IxDyn> {
        let dyn_inner = self.inner.clone().into_dyn();
        Array::<T, IxDyn>::from_ndarray(dyn_inner)
    }

    /// Consume this array and return a dynamic-rank `Array<T, IxDyn>`.
    ///
    /// Like [`Array::to_dyn`] but takes ownership to avoid the clone.
    pub fn into_dyn(self) -> Array<T, IxDyn> {
        let dyn_inner = self.inner.into_dyn();
        Array::<T, IxDyn>::from_ndarray(dyn_inner)
    }

    /// Return a C-contiguous (row-major) version of this array, copying
    /// only if the current layout is not already C-contiguous (#351).
    ///
    /// Equivalent to NumPy's `np.ascontiguousarray`. The returned
    /// [`CowArray`] borrows from `self` when no copy is needed, so this
    /// is a zero-cost guard before BLAS calls, SIMD loops, or FFI that
    /// require row-major storage.
    pub fn as_standard_layout(&self) -> CowArray<'_, T, D> {
        // Delegate to the inner ndarray check: is_standard_layout returns
        // true for C-contiguous arrays, including the 1-D / size-0 / size-1
        // edge cases where the crate's tri-state `MemoryLayout` enum can't
        // represent "both C and F at once".
        if self.inner.is_standard_layout() {
            CowArray::Borrowed(self.view())
        } else {
            // `iter()` walks in logical (row-major) order regardless of
            // the underlying stride pattern, so collecting produces a
            // C-contiguous flat buffer.
            let data: Vec<T> = self.iter().cloned().collect();
            let owned = Array::from_vec(self.dim().clone(), data)
                .expect("from_vec: data length was just built from self.iter()");
            CowArray::Owned(owned)
        }
    }

    /// Return a Fortran-contiguous (column-major) version of this array,
    /// copying only if the current layout is not already F-contiguous (#351).
    ///
    /// Equivalent to NumPy's `np.asfortranarray`. The returned
    /// [`CowArray`] borrows from `self` when no copy is needed. 1-D arrays
    /// are borrowed because they are trivially both C- and F-contiguous.
    pub fn as_fortran_layout(&self) -> CowArray<'_, T, D> {
        // The transpose of an F-contiguous array is C-contiguous, so
        // `t().is_standard_layout()` is the canonical F-contig check and
        // correctly handles 1-D / size-0 / size-1 edge cases.
        if self.inner.t().is_standard_layout() {
            CowArray::Borrowed(self.view())
        } else {
            // `t()` reverses all axes; iterating the reversed view in
            // logical (row-major) order yields the original elements in
            // column-major order, which is exactly what `from_vec_f`
            // expects for Fortran-layout storage.
            let data: Vec<T> = self.inner.t().iter().cloned().collect();
            let owned = Array::from_vec_f(self.dim().clone(), data)
                .expect("from_vec_f: data length was just built from self.inner.t().iter()");
            CowArray::Owned(owned)
        }
    }
}

// ---------------------------------------------------------------------------
// ArrayView methods
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> ArrayView<'_, T, D> {
    /// Apply a closure to every element, returning a new owned array.
    pub fn mapv(&self, f: impl Fn(T) -> T) -> Array<T, D> {
        let inner = self.inner.mapv(&f);
        Array::from_ndarray(inner)
    }

    /// Fold along an axis.
    pub fn fold_axis(
        &self,
        axis: Axis,
        init: T,
        fold: impl FnMut(&T, &T) -> T,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        if axis.index() >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis.index(), ndim));
        }
        let nd_axis = ndarray::Axis(axis.index());
        let mut fold = fold;
        let result = self.inner.fold_axis(nd_axis, init, |acc, x| fold(acc, x));
        let dyn_result = result.into_dyn();
        Ok(Array::from_ndarray(dyn_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};
    use crate::layout::MemoryLayout;

    #[test]
    fn mapv_double() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let doubled = arr.mapv(|x| x * 2.0);
        assert_eq!(doubled.as_slice().unwrap(), &[2.0, 4.0, 6.0, 8.0]);
        // Original unchanged
        assert_eq!(arr.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn mapv_inplace_negate() {
        let mut arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, -2.0, 3.0]).unwrap();
        arr.mapv_inplace(|x| -x);
        assert_eq!(arr.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn zip_mut_with_add() {
        let mut a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        a.zip_mut_with(&b, |x, y| *x += y).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn zip_mut_with_shape_mismatch() {
        let mut a = Array::<f64, Ix1>::zeros(Ix1::new([3])).unwrap();
        let b = Array::<f64, Ix1>::zeros(Ix1::new([5])).unwrap();
        assert!(a.zip_mut_with(&b, |_, _| {}).is_err());
    }

    // ---- #351: as_standard_layout / as_fortran_layout ----

    #[test]
    fn as_standard_layout_borrows_when_already_c_contig() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(a.layout(), MemoryLayout::C);
        let cow = a.as_standard_layout();
        assert!(cow.is_borrowed(), "C-contig input must borrow, not copy");
        assert_eq!(cow.shape(), &[2, 3]);
        assert_eq!(cow.layout(), MemoryLayout::C);
    }

    #[test]
    fn as_standard_layout_copies_f_contig_input_to_c() {
        let a = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        // Logical shape is (2, 3), but storage is F-order.
        assert_eq!(a.layout(), MemoryLayout::Fortran);
        let cow = a.as_standard_layout();
        assert!(cow.is_owned(), "F-contig input must be copied to C-contig");
        assert_eq!(cow.shape(), &[2, 3]);
        assert_eq!(cow.layout(), MemoryLayout::C);
        // Logical element order must be preserved.
        let owned = cow.into_owned();
        assert_eq!(owned.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_fortran_layout_borrows_when_already_f_contig() {
        let a = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
            .unwrap();
        assert_eq!(a.layout(), MemoryLayout::Fortran);
        let cow = a.as_fortran_layout();
        assert!(cow.is_borrowed(), "F-contig input must borrow, not copy");
        assert_eq!(cow.shape(), &[2, 3]);
        assert_eq!(cow.layout(), MemoryLayout::Fortran);
    }

    #[test]
    fn as_fortran_layout_copies_c_contig_input_to_f() {
        // [[1,2,3],[4,5,6]] — C-contig logical layout.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(a.layout(), MemoryLayout::C);
        let cow = a.as_fortran_layout();
        assert!(cow.is_owned(), "C-contig input must be copied to F-contig");
        assert_eq!(cow.shape(), &[2, 3]);
        assert_eq!(cow.layout(), MemoryLayout::Fortran);
        // Logical element values by row-major walk must match the original.
        let owned = cow.into_owned();
        let logical: Vec<f64> = owned.iter().copied().collect();
        assert_eq!(logical, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn layout_roundtrip_preserves_values() {
        // C -> F -> C returns the same logical contents.
        let original = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12i32).collect()).unwrap();
        let f_cow = original.as_fortran_layout();
        let f_owned = f_cow.into_owned();
        assert_eq!(f_owned.layout(), MemoryLayout::Fortran);
        let c_cow = f_owned.as_standard_layout();
        let c_owned = c_cow.into_owned();
        assert_eq!(c_owned.layout(), MemoryLayout::C);
        assert_eq!(c_owned.as_slice().unwrap(), original.as_slice().unwrap());
    }

    #[test]
    fn as_standard_layout_1d_always_borrows() {
        // Any 1-D array is both C and F contiguous; must borrow.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(a.as_standard_layout().is_borrowed());
        assert!(a.as_fortran_layout().is_borrowed());
    }

    #[test]
    fn fold_axis_sum_rows() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // Sum along axis 1 (sum each row)
        let sums = arr.fold_axis(Axis(1), 0.0, |acc, &x| *acc + x).unwrap();
        assert_eq!(sums.shape(), &[2]);
        let data: Vec<f64> = sums.iter().copied().collect();
        assert_eq!(data, vec![6.0, 15.0]);
    }

    #[test]
    fn fold_axis_sum_cols() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // Sum along axis 0 (sum each column)
        let sums = arr.fold_axis(Axis(0), 0.0, |acc, &x| *acc + x).unwrap();
        assert_eq!(sums.shape(), &[3]);
        let data: Vec<f64> = sums.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn fold_axis_out_of_bounds() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([2, 3])).unwrap();
        assert!(arr.fold_axis(Axis(2), 0.0, |a, _| *a).is_err());
    }

    #[test]
    fn map_to_different_type() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.5, 2.7, 3.1]).unwrap();
        let ints: Array<i32, Ix1> = arr.map_to(|x| x as i32);
        assert_eq!(ints.as_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn view_mapv() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = arr.view();
        let doubled = v.mapv(|x| x * 2.0);
        assert_eq!(doubled.as_slice().unwrap(), &[2.0, 4.0, 6.0]);
    }
}
