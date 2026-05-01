// ferray-core: Array introspection properties (REQ-35, REQ-36)

use crate::dimension::Dimension;
use crate::dtype::{DType, Element};
use crate::error::{FerrayError, FerrayResult};

use super::ArrayFlags;
use super::owned::Array;
use super::view::ArrayView;

// ---------------------------------------------------------------------------
// REQ-35: Core introspection for Array<T, D>
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> Array<T, D> {
    /// Size in bytes of a single element.
    #[inline]
    pub const fn itemsize(&self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Total size in bytes of all elements (size * itemsize).
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Runtime dtype tag for this array's element type.
    #[inline]
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    // -- REQ-36 additional properties --

    /// Transposed view (zero-copy). Reverses the axes.
    ///
    /// This is the equivalent of `NumPy`'s `.T` property.
    pub fn t(&self) -> ArrayView<'_, T, D> {
        let transposed = self.inner.view().reversed_axes();
        ArrayView::from_ndarray(transposed)
    }

    /// Deep copy of this array.
    #[must_use]
    pub fn copy(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            dim: self.dim.clone(),
        }
    }

    /// Convert to a flat `Vec<T>` in logical (row-major) order.
    pub fn to_vec_flat(&self) -> Vec<T> {
        self.inner.iter().cloned().collect()
    }

    /// Return the raw bytes of the array data.
    ///
    /// Only succeeds if the array is contiguous; returns an error otherwise.
    /// Requires `T: Copy` to guarantee no padding bytes exist.
    pub fn to_bytes(&self) -> FerrayResult<&[u8]>
    where
        T: Copy,
    {
        let slice = self.inner.as_slice().ok_or_else(|| {
            FerrayError::invalid_value("array is not contiguous; cannot produce byte slice")
        })?;
        let ptr = slice.as_ptr().cast::<u8>();
        let len = std::mem::size_of_val(slice);
        // SAFETY: the slice is contiguous and alive for 'self lifetime;
        // reinterpreting as bytes is always safe for Copy-like types.
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }

    /// Return flags describing memory properties.
    pub fn flags(&self) -> ArrayFlags {
        let layout = self.layout();
        ArrayFlags {
            c_contiguous: layout.is_c_contiguous(),
            f_contiguous: layout.is_f_contiguous(),
            owndata: true,
            writeable: true,
            // Owned arrays go through Vec<T>, whose allocator
            // guarantees alignment for T (#345).
            aligned: true,
        }
    }
}

// ---------------------------------------------------------------------------
// REQ-35: Core introspection for ArrayView<'a, T, D>
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> ArrayView<'_, T, D> {
    /// Size in bytes of a single element.
    #[inline]
    pub const fn itemsize(&self) -> usize {
        std::mem::size_of::<T>()
    }

    /// Total size in bytes of all elements.
    #[inline]
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Runtime dtype tag.
    #[inline]
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// Transposed view (zero-copy).
    pub fn t(&self) -> ArrayView<'_, T, D> {
        let transposed = self.inner.clone().reversed_axes();
        ArrayView::from_ndarray(transposed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn introspect_basics() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.size(), 6);
        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr.itemsize(), 8); // f64 = 8 bytes
        assert_eq!(arr.nbytes(), 48); // 6 * 8
        assert_eq!(arr.dtype(), DType::F64);
        assert!(!arr.is_empty());
    }

    #[test]
    fn introspect_empty() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert!(arr.is_empty());
        assert_eq!(arr.size(), 0);
        assert_eq!(arr.nbytes(), 0);
    }

    #[test]
    fn flags_owned() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 4])).unwrap();
        let f = arr.flags();
        assert!(f.c_contiguous);
        assert!(f.owndata);
        assert!(f.writeable);
        assert!(f.aligned, "owned arrays go through Vec<T> — always aligned");
    }

    #[test]
    fn flags_owned_aligned_for_various_dtypes() {
        // #345: owned arrays of every supported dtype must report
        // aligned=true. Vec<T>'s allocator guarantees this.
        let f64_arr = Array::<f64, Ix1>::zeros(Ix1::new([8])).unwrap();
        assert!(f64_arr.flags().aligned);
        let i32_arr = Array::<i32, Ix1>::zeros(Ix1::new([8])).unwrap();
        assert!(i32_arr.flags().aligned);
        let u8_arr = Array::<u8, Ix1>::zeros(Ix1::new([8])).unwrap();
        assert!(u8_arr.flags().aligned);
    }

    #[test]
    fn transpose_view() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let t = arr.t();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(t.size(), 6);
    }

    #[test]
    fn copy_is_independent() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let copy = arr.copy();
        assert_eq!(copy.as_slice().unwrap(), arr.as_slice().unwrap());
        assert_ne!(copy.as_ptr(), arr.as_ptr());
    }

    #[test]
    fn to_vec_flat() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
        assert_eq!(arr.to_vec_flat(), vec![1, 2, 3, 4]);
    }

    #[test]
    fn to_bytes_contiguous() {
        let arr = Array::<u8, Ix1>::from_vec(Ix1::new([4]), vec![0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        let bytes = arr.to_bytes().unwrap();
        assert_eq!(bytes, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn view_introspection() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let v = arr.view();
        assert_eq!(v.itemsize(), 8);
        assert_eq!(v.nbytes(), 48);
        assert_eq!(v.dtype(), DType::F64);
    }

    #[test]
    fn view_transpose() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let v = arr.view();
        let vt = v.t();
        assert_eq!(vt.shape(), &[3, 2]);
    }
}
