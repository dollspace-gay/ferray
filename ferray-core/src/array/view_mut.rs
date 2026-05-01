// ferray-core: Mutable array view — ArrayViewMut<'a, T, D> (REQ-3)

use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::layout::MemoryLayout;

use super::ArrayFlags;
use super::owned::Array;

/// A mutable, borrowed view into an existing array's data.
///
/// This is a zero-copy mutable slice. The lifetime `'a` ties this view
/// to the source array, and Rust's borrow checker ensures exclusivity.
pub struct ArrayViewMut<'a, T: Element, D: Dimension> {
    pub(crate) inner: ndarray::ArrayViewMut<'a, T, D::NdarrayDim>,
    pub(crate) dim: D,
}

impl<'a, T: Element, D: Dimension> ArrayViewMut<'a, T, D> {
    /// Create from an ndarray mutable view. Crate-internal.
    pub(crate) fn from_ndarray(inner: ndarray::ArrayViewMut<'a, T, D::NdarrayDim>) -> Self {
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

    /// Mutable raw pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    /// Try to get a contiguous slice.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.inner.as_slice()
    }

    /// Try to get a contiguous mutable slice.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.inner.as_slice_mut()
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

    /// Convert to a flat `Vec<T>` in logical (row-major) order.
    pub fn to_vec_flat(&self) -> Vec<T> {
        self.inner.iter().cloned().collect()
    }

    /// Array flags for this mutable view.
    pub fn flags(&self) -> ArrayFlags {
        let layout = self.layout();
        ArrayFlags {
            c_contiguous: layout.is_c_contiguous(),
            f_contiguous: layout.is_f_contiguous(),
            owndata: false,
            writeable: true,
            // #345: same alignment story as the immutable view path.
            aligned: (self.inner.as_ptr() as usize) % core::mem::align_of::<T>() == 0,
        }
    }
}

// Create an ArrayViewMut from an owned Array
impl<T: Element, D: Dimension> Array<T, D> {
    /// Create a mutable view of this array.
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T, D> {
        ArrayViewMut::from_ndarray(self.inner.view_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;

    #[test]
    fn view_mut_from_owned() {
        let mut arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = arr.view_mut();
        assert_eq!(v.shape(), &[3]);
        assert!(v.flags().writeable);
        assert!(!v.flags().owndata);
    }

    #[test]
    fn view_mut_modify() {
        let mut arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        {
            let mut v = arr.view_mut();
            if let Some(s) = v.as_slice_mut() {
                s[0] = 99.0;
            }
        }
        assert_eq!(arr.as_slice().unwrap()[0], 99.0);
    }
}
