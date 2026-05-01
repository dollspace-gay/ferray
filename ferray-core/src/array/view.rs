// ferray-core: Immutable array view — ArrayView<'a, T, D> (REQ-3)

use ndarray::ShapeBuilder;

use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::layout::MemoryLayout;

use super::ArrayFlags;
use super::owned::Array;

/// An immutable, borrowed view into an existing array's data.
///
/// This is a zero-copy slice — no data is cloned. The lifetime `'a`
/// ties this view to the source array.
pub struct ArrayView<'a, T: Element, D: Dimension> {
    pub(crate) inner: ndarray::ArrayView<'a, T, D::NdarrayDim>,
    pub(crate) dim: D,
}

impl<'a, T: Element, D: Dimension> ArrayView<'a, T, D> {
    /// Create from an ndarray view. Crate-internal.
    pub(crate) fn from_ndarray(inner: ndarray::ArrayView<'a, T, D::NdarrayDim>) -> Self {
        let dim = D::from_ndarray_dim(&inner.raw_dim());
        Self { inner, dim }
    }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Whether the view has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Strides as a slice.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.inner.strides()
    }

    /// Raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Try to get a contiguous slice.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.inner.as_slice()
    }

    /// Memory layout.
    pub fn layout(&self) -> MemoryLayout {
        crate::layout::classify_layout(
            self.inner.is_standard_layout(),
            self.dim.as_slice(),
            self.inner.strides(),
        )
    }

    /// Return a reference to the internal dimension descriptor.
    #[inline]
    pub const fn dim(&self) -> &D {
        &self.dim
    }

    /// Convert this view into an owned array by cloning all elements.
    pub fn to_owned(&self) -> Array<T, D> {
        Array::from_ndarray(self.inner.to_owned())
    }

    /// Convert to a flat `Vec<T>` in logical (row-major) order.
    ///
    /// Unlike `as_slice()` which requires contiguous memory, this works on
    /// strided/non-contiguous views by iterating elements in logical order.
    pub fn to_vec_flat(&self) -> Vec<T> {
        self.inner.iter().cloned().collect()
    }

    /// Array flags for this view.
    pub fn flags(&self) -> ArrayFlags {
        let layout = self.layout();
        ArrayFlags {
            c_contiguous: layout.is_c_contiguous(),
            f_contiguous: layout.is_f_contiguous(),
            owndata: false,
            writeable: false,
            // #345: views may originate from raw pointers (`from_shape_ptr`);
            // verify the head-of-buffer pointer satisfies T's alignment.
            aligned: (self.inner.as_ptr() as usize) % core::mem::align_of::<T>() == 0,
        }
    }
}

use crate::dimension::IxDyn;

impl<T: Element> ArrayView<'_, T, IxDyn> {
    /// Construct a dynamic-rank view from a raw pointer, shape, and strides.
    ///
    /// This is the primary escape hatch for crates that need to build views
    /// with custom stride patterns (e.g., `ferray-stride-tricks`).
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is valid for reads for the entire region described by `shape`
    ///   and `strides`.
    /// - The lifetime `'a` does not outlive the allocation that `ptr` points
    ///   into.
    /// - No mutable reference to the same memory region exists for the
    ///   duration of `'a`.
    /// - `strides` are given in units of elements (not bytes).
    pub unsafe fn from_shape_ptr(ptr: *const T, shape: &[usize], strides: &[usize]) -> Self {
        let nd_shape = ndarray::IxDyn(shape);
        let nd_strides = ndarray::IxDyn(strides);
        let nd_view =
            unsafe { ndarray::ArrayView::from_shape_ptr(nd_shape.strides(nd_strides), ptr) };
        Self::from_ndarray(nd_view)
    }
}

impl<T: Element, D: Dimension> Clone for ArrayView<'_, T, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dim: self.dim.clone(),
        }
    }
}

// Create an ArrayView from an owned Array
impl<T: Element, D: Dimension> Array<T, D> {
    /// Create an immutable view of this array.
    pub fn view(&self) -> ArrayView<'_, T, D> {
        ArrayView::from_ndarray(self.inner.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;

    #[test]
    fn view_from_owned() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let v = arr.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.size(), 6);
        assert!(!v.flags().owndata);
        assert!(!v.flags().writeable);
    }

    #[test]
    fn view_shares_data() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let v = arr.view();
        // Same pointer
        assert_eq!(arr.as_ptr(), v.as_ptr());
    }

    #[test]
    fn view_to_owned() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let v = arr.view();
        let owned = v.to_owned();
        assert_eq!(owned.shape(), arr.shape());
        assert_eq!(owned.as_slice().unwrap(), arr.as_slice().unwrap());
        // But different allocations
        assert_ne!(owned.as_ptr(), arr.as_ptr());
    }
}
