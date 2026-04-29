// ferray-core: Owned array — Array<T, D> (REQ-1, REQ-3, REQ-5)

use ndarray::ShapeBuilder;

use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};
use crate::layout::MemoryLayout;

/// An owned, heap-allocated N-dimensional array.
///
/// This is the primary array type in ferray — analogous to `numpy.ndarray`
/// with full ownership of its data buffer.
///
/// `T` is the element type (must implement [`Element`]) and `D` describes
/// the dimensionality ([`Ix1`], [`Ix2`], ..., [`IxDyn`]).
///
/// [`Ix1`]: crate::dimension::Ix1
/// [`Ix2`]: crate::dimension::Ix2
/// [`IxDyn`]: crate::dimension::IxDyn
pub struct Array<T: Element, D: Dimension> {
    /// The internal ndarray storage. This is never exposed publicly.
    pub(crate) inner: ndarray::Array<T, D::NdarrayDim>,
    /// Cached dimension (our own Dimension type).
    pub(crate) dim: D,
}

impl<T: Element, D: Dimension> Array<T, D> {
    // -- Construction helpers (crate-internal) --

    /// Wrap an existing `ndarray::Array`. Crate-internal.
    pub(crate) fn from_ndarray(inner: ndarray::Array<T, D::NdarrayDim>) -> Self {
        let dim = D::from_ndarray_dim(&inner.raw_dim());
        Self { inner, dim }
    }

    /// Unwrap to the internal `ndarray::Array`. Crate-internal.
    /// Consume this ferray array and return the underlying
    /// [`ndarray::Array`], preserving shape and layout.
    ///
    /// This is the escape hatch for interop with crates that already
    /// work with ndarray (notably `numpy`/`PyO3`'s
    /// [`PyArray::from_owned_array`], which avoids a reshape round-trip).
    /// Calls through to a move — no allocation or copy.
    pub fn into_ndarray(self) -> ndarray::Array<T, D::NdarrayDim> {
        self.inner
    }

    // -- Public construction --

    /// Create a new array filled with the given value.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the shape has zero dimensions
    /// but `D` is a fixed-rank type with nonzero rank, or vice versa.
    pub fn from_elem(dim: D, elem: T) -> FerrayResult<Self> {
        let nd_dim = dim.to_ndarray_dim();
        let inner = ndarray::Array::from_elem(nd_dim, elem);
        Ok(Self { inner, dim })
    }

    /// Create a new array filled with zeros.
    pub fn zeros(dim: D) -> FerrayResult<Self> {
        Self::from_elem(dim, T::zero())
    }

    /// Create a new array filled with ones.
    pub fn ones(dim: D) -> FerrayResult<Self> {
        Self::from_elem(dim, T::one())
    }

    /// Create an array from a flat vector and a shape.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if `data.len()` does not equal
    /// the product of the shape dimensions.
    pub fn from_vec(dim: D, data: Vec<T>) -> FerrayResult<Self> {
        let expected = dim.size();
        if data.len() != expected {
            return Err(FerrayError::shape_mismatch(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                dim.as_slice(),
                expected,
            )));
        }
        let nd_dim = dim.to_ndarray_dim();
        let inner = ndarray::Array::from_shape_vec(nd_dim, data)
            .map_err(|e| FerrayError::shape_mismatch(format!("ndarray shape error: {e}")))?;
        Ok(Self { inner, dim })
    }

    /// Create an array from a flat vector with Fortran (column-major) layout.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if lengths don't match.
    pub fn from_vec_f(dim: D, data: Vec<T>) -> FerrayResult<Self> {
        let expected = dim.size();
        if data.len() != expected {
            return Err(FerrayError::shape_mismatch(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                dim.as_slice(),
                expected,
            )));
        }
        let nd_dim = dim.to_ndarray_dim();
        let inner = ndarray::Array::from_shape_vec(nd_dim.f(), data)
            .map_err(|e| FerrayError::shape_mismatch(format!("ndarray shape error: {e}")))?;
        let dim = D::from_ndarray_dim(&inner.raw_dim());
        Ok(Self { inner, dim })
    }

    /// Create a 1-D array from an iterator.
    ///
    /// This only makes sense for `D = Ix1`; for other dimensions,
    /// collect first and use `from_vec`.
    pub fn from_iter_1d(iter: impl IntoIterator<Item = T>) -> FerrayResult<Self>
    where
        D: Dimension<NdarrayDim = ndarray::Ix1>,
    {
        let inner = ndarray::Array::from_iter(iter);
        let dim = D::from_ndarray_dim(&inner.raw_dim());
        Ok(Self { inner, dim })
    }

    /// Return the memory layout of this array.
    pub fn layout(&self) -> MemoryLayout {
        crate::layout::classify_layout(
            self.inner.is_standard_layout(),
            self.dim.as_slice(),
            self.inner.strides(),
        )
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Strides as a slice (in units of elements, not bytes).
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.inner.strides()
    }

    /// Total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Whether the array has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return a raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.inner.as_ptr()
    }

    /// Return a mutable raw pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }

    /// Return the data as a contiguous slice, if the layout allows it.
    pub fn as_slice(&self) -> Option<&[T]> {
        self.inner.as_slice()
    }

    /// Return the data as a contiguous mutable slice, if the layout allows it.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        self.inner.as_slice_mut()
    }

    /// Return a reference to the internal dimension descriptor.
    #[inline]
    pub const fn dim(&self) -> &D {
        &self.dim
    }
}

// REQ-5: ndarray conversions.
//
// Historically we exposed blanket `From<ndarray::Array>` / `From<Array>`
// impls here, which let downstream users implicitly move values between
// `ferray::Array` and `ndarray::Array` via `.into()`. That leaked
// ndarray into the public API surface even for users who never wanted
// to depend on ndarray types (see issue #76). The inward wrapper lives
// on as a crate-internal constructor (`Array::from_ndarray`) and the
// outward escape hatch is still available via the public
// `into_ndarray` method — both are called explicitly so the ndarray
// dependency is visible at every use site.

impl<T: Element, D: Dimension> Clone for Array<T, D> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dim: self.dim.clone(),
        }
    }
}

impl<T: Element + PartialEq, D: Dimension> PartialEq for Array<T, D> {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl<T: Element + Eq, D: Dimension> Eq for Array<T, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2, IxDyn};

    #[test]
    fn create_zeros() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 4])).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.size(), 12);
        assert_eq!(arr.ndim(), 2);
        assert!(!arr.is_empty());
    }

    #[test]
    fn create_from_vec() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        assert_eq!(arr.shape(), &[4]);
        assert_eq!(arr.as_slice().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn create_from_vec_shape_mismatch() {
        let res = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0]);
        assert!(res.is_err());
    }

    #[test]
    fn from_iter_1d() {
        let arr = Array::<f64, Ix1>::from_iter_1d((0..5).map(|x| x as f64)).unwrap();
        assert_eq!(arr.shape(), &[5]);
    }

    #[test]
    fn layout_c_contiguous() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 4])).unwrap();
        assert_eq!(arr.layout(), MemoryLayout::C);
    }

    #[test]
    fn from_vec_f_order() {
        let arr =
            Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.layout(), MemoryLayout::Fortran);
    }

    #[test]
    fn clone_array() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn ndarray_roundtrip() {
        let original = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), original.clone()).unwrap();
        // Use the explicit methods rather than `Into`/`From` so the
        // ndarray dependency is not implicit in the public API (#76).
        let nd: ndarray::Array<f64, ndarray::Ix2> = arr.into_ndarray();
        let arr2: Array<f64, Ix2> = Array::from_ndarray(nd);
        assert_eq!(arr2.as_slice().unwrap(), &original[..]);
    }

    #[test]
    fn dynamic_rank() {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0; 6]).unwrap();
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.shape(), &[2, 3]);
    }
}
