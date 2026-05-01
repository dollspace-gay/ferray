// ferray-core: CowArray<'a, T, D> — owned-or-borrowed (REQ-3)

use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::layout::MemoryLayout;

use super::ArrayFlags;
use super::owned::Array;
use super::view::ArrayView;

/// A copy-on-write array that is either a borrowed view or an owned array.
///
/// This is useful when a function might or might not need to allocate:
/// it can accept borrowed data and only clone if mutation is required.
pub enum CowArray<'a, T: Element, D: Dimension> {
    /// Borrowed — refers to data owned by another array.
    Borrowed(ArrayView<'a, T, D>),
    /// Owned — has its own data buffer.
    Owned(Array<T, D>),
}

impl<'a, T: Element, D: Dimension> CowArray<'a, T, D> {
    /// Create a `CowArray` from a borrowed view.
    pub const fn from_view(view: ArrayView<'a, T, D>) -> Self {
        Self::Borrowed(view)
    }

    /// Create a `CowArray` from an owned array.
    pub const fn from_owned(arr: Array<T, D>) -> Self {
        Self::Owned(arr)
    }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Borrowed(v) => v.shape(),
            Self::Owned(a) => a.shape(),
        }
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.ndim(),
            Self::Owned(a) => a.ndim(),
        }
    }

    /// Total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        match self {
            Self::Borrowed(v) => v.size(),
            Self::Owned(a) => a.size(),
        }
    }

    /// Whether the array has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Memory layout.
    pub fn layout(&self) -> MemoryLayout {
        match self {
            Self::Borrowed(v) => v.layout(),
            Self::Owned(a) => a.layout(),
        }
    }

    /// Whether this is a borrowed (view) variant.
    pub const fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed(_))
    }

    /// Whether this is an owned variant.
    pub const fn is_owned(&self) -> bool {
        matches!(self, Self::Owned(_))
    }

    /// Convert to an owned array, cloning if currently borrowed.
    pub fn into_owned(self) -> Array<T, D> {
        match self {
            Self::Borrowed(v) => v.to_owned(),
            Self::Owned(a) => a,
        }
    }

    /// Ensure this is the owned variant, cloning if necessary,
    /// and return a mutable reference to the owned array.
    pub fn to_mut(&mut self) -> &mut Array<T, D> {
        if let Self::Borrowed(v) = self {
            *self = Self::Owned(v.to_owned());
        }
        match self {
            Self::Owned(a) => a,
            Self::Borrowed(_) => unreachable!(),
        }
    }

    /// Get a read-only view of the data.
    ///
    /// If this is a borrowed variant, returns a view with the same lifetime
    /// as `&self`. If owned, returns a view borrowing from `self`.
    pub fn view(&self) -> ArrayView<'_, T, D> {
        match self {
            Self::Borrowed(v) => {
                // Reborrow the inner ndarray view with &self lifetime
                ArrayView::from_ndarray(v.inner.view())
            }
            Self::Owned(a) => a.view(),
        }
    }

    /// Raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        match self {
            Self::Borrowed(v) => v.as_ptr(),
            Self::Owned(a) => a.as_ptr(),
        }
    }

    /// Array flags.
    pub fn flags(&self) -> ArrayFlags {
        match self {
            Self::Borrowed(v) => v.flags(),
            Self::Owned(a) => {
                let layout = a.layout();
                ArrayFlags {
                    c_contiguous: layout.is_c_contiguous(),
                    f_contiguous: layout.is_f_contiguous(),
                    owndata: true,
                    writeable: true,
                    // CowArray::Owned wraps an Array (Vec<T> path) — aligned (#345).
                    aligned: true,
                }
            }
        }
    }
}

impl<T: Element, D: Dimension> Clone for CowArray<'_, T, D> {
    fn clone(&self) -> Self {
        match self {
            Self::Borrowed(v) => Self::Borrowed(v.clone()),
            Self::Owned(a) => Self::Owned(a.clone()),
        }
    }
}

impl<T: Element, D: Dimension> From<Array<T, D>> for CowArray<'_, T, D> {
    fn from(arr: Array<T, D>) -> Self {
        Self::Owned(arr)
    }
}

impl<'a, T: Element, D: Dimension> From<ArrayView<'a, T, D>> for CowArray<'a, T, D> {
    fn from(view: ArrayView<'a, T, D>) -> Self {
        Self::Borrowed(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;

    #[test]
    fn cow_from_view() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = arr.view();
        let cow = CowArray::from_view(view);
        assert!(cow.is_borrowed());
        assert!(!cow.is_owned());
        assert_eq!(cow.shape(), &[3]);
    }

    #[test]
    fn cow_from_owned() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let cow = CowArray::from_owned(arr);
        assert!(cow.is_owned());
        assert!(!cow.is_borrowed());
    }

    #[test]
    fn cow_to_mut_clones_when_borrowed() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = arr.view();
        let mut cow = CowArray::from_view(view);

        assert!(cow.is_borrowed());
        let owned = cow.to_mut();
        assert_eq!(owned.shape(), &[3]);
        assert!(cow.is_owned());
    }

    #[test]
    fn cow_into_owned() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let view = arr.view();
        let cow = CowArray::from_view(view);
        let owned = cow.into_owned();
        assert_eq!(owned.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
