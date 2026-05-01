// ferray-core: ArcArray<T, D> — reference-counted with copy-on-write (REQ-3, REQ-4)

use std::sync::Arc;

use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::layout::MemoryLayout;

use super::ArrayFlags;
use super::owned::Array;
use super::view::ArrayView;

/// A reference-counted N-dimensional array with copy-on-write semantics.
///
/// Multiple `ArcArray` instances can share the same underlying buffer.
/// When a mutation is requested and the reference count is greater than 1,
/// the buffer is cloned first (copy-on-write). Views derived from an
/// `ArcArray` observe the data at creation time; subsequent mutations to
/// the source (which trigger a `CoW` clone) do not affect existing views.
pub struct ArcArray<T: Element, D: Dimension> {
    /// Shared data buffer. Using Arc<Vec<T>> + shape/strides for `CoW` support.
    data: Arc<Vec<T>>,
    /// Shape of this array.
    dim: D,
    /// Strides in element counts.
    strides: Vec<isize>,
    /// Offset into the data buffer (for views into sub-regions).
    offset: usize,
}

impl<T: Element, D: Dimension> ArcArray<T, D> {
    /// Create an `ArcArray` from an owned `Array`.
    ///
    /// Preserves the original memory layout (C or Fortran) when possible.
    /// Non-contiguous arrays are converted to C-contiguous.
    pub fn from_owned(arr: Array<T, D>) -> Self {
        let dim = arr.dim.clone();
        let original_strides: Vec<isize> = arr.inner.strides().to_vec();

        if arr.inner.is_standard_layout() {
            // C-contiguous: take data directly, preserve strides
            let data = arr.inner.into_raw_vec_and_offset().0;
            Self {
                data: Arc::new(data),
                dim,
                strides: original_strides,
                offset: 0,
            }
        } else {
            // Non-standard layout: check if F-contiguous
            let shape = dim.as_slice();
            let is_f = crate::layout::detect_layout(shape, &original_strides)
                == crate::layout::MemoryLayout::Fortran;

            if is_f {
                // Fortran-contiguous: data is contiguous, just different stride order.
                // Extract the raw vec — for F-order the data is contiguous in memory.
                let data = arr.inner.into_raw_vec_and_offset().0;
                Self {
                    data: Arc::new(data),
                    dim,
                    strides: original_strides,
                    offset: 0,
                }
            } else {
                // Truly non-contiguous (e.g., strided slice): make C-contiguous
                let contiguous = arr.inner.as_standard_layout().into_owned();
                let data = contiguous.into_raw_vec_and_offset().0;
                let strides = compute_c_strides(shape);
                Self {
                    data: Arc::new(data),
                    dim,
                    strides,
                    offset: 0,
                }
            }
        }
    }

    /// Shape as a slice.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Total number of elements.
    #[inline]
    pub fn size(&self) -> usize {
        self.dim.size()
    }

    /// Whether the array has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Strides as a slice.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Memory layout.
    pub fn layout(&self) -> MemoryLayout {
        crate::layout::detect_layout(self.dim.as_slice(), &self.strides)
    }

    /// Return a reference to the dimension descriptor.
    #[inline]
    pub const fn dim(&self) -> &D {
        &self.dim
    }

    /// Number of shared references to the underlying buffer.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Whether this is the sole owner of the data (refcount == 1).
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.data) == 1
    }

    /// Get a slice of the data for this array.
    pub fn as_slice(&self) -> &[T] {
        &self.data[self.offset..self.offset + self.size()]
    }

    /// Raw pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Create an immutable view of the data.
    ///
    /// The view borrows from this `ArcArray` and will see the data as it
    /// exists at creation time. If the `ArcArray` is later mutated (triggering
    /// a `CoW` clone), the view continues to see the old data.
    pub fn view(&self) -> ArrayView<'_, T, D> {
        let nd_dim = self.dim.to_ndarray_dim();
        let slice = self.as_slice();
        let nd_view = ndarray::ArrayView::from_shape(nd_dim, slice)
            .expect("ArcArray data should be consistent with shape");
        ArrayView::from_ndarray(nd_view)
    }

    /// Ensure exclusive ownership of the data buffer (`CoW`).
    ///
    /// If the reference count is > 1, this clones the buffer so that
    /// mutations will not affect other holders.
    fn make_unique(&mut self) {
        if Arc::strong_count(&self.data) > 1 {
            let slice = &self.data[self.offset..self.offset + self.size()];
            self.data = Arc::new(slice.to_vec());
            self.offset = 0;
        }
    }

    /// Get a mutable slice of the data, performing a `CoW` clone if necessary.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        self.make_unique();
        let size = self.size();
        let offset = self.offset;
        Arc::get_mut(&mut self.data)
            .expect("make_unique should ensure refcount == 1")
            .get_mut(offset..offset + size)
            .expect("offset + size should be in bounds")
    }

    /// Apply a function to each element, performing `CoW` if needed.
    pub fn mapv_inplace(&mut self, f: impl Fn(T) -> T) {
        self.make_unique();
        let size = self.size();
        let offset = self.offset;
        let data = Arc::get_mut(&mut self.data).expect("unique after make_unique");
        for elem in &mut data[offset..offset + size] {
            *elem = f(elem.clone());
        }
    }

    /// Convert to an owned `Array`, cloning if shared.
    pub fn into_owned(self) -> Array<T, D> {
        let data: Vec<T> = if self.offset == 0 && self.data.len() == self.size() {
            match Arc::try_unwrap(self.data) {
                Ok(v) => v,
                Err(arc) => arc[..].to_vec(),
            }
        } else {
            self.data[self.offset..self.offset + self.size()].to_vec()
        };
        Array::from_vec(self.dim, data).expect("data should match shape")
    }

    /// Deep copy — always creates a new independent buffer.
    #[must_use]
    pub fn copy(&self) -> Self {
        let data = self.as_slice().to_vec();
        Self {
            data: Arc::new(data),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            offset: 0,
        }
    }

    /// Array flags.
    pub fn flags(&self) -> ArrayFlags {
        let layout = self.layout();
        ArrayFlags {
            c_contiguous: layout.is_c_contiguous(),
            f_contiguous: layout.is_f_contiguous(),
            owndata: true, // ArcArray conceptually owns (shared ownership)
            writeable: true,
            // ArcArray wraps an Arc<ndarray::ArrayD<T>> which goes
            // through the Vec<T> allocator path — always aligned (#345).
            aligned: true,
        }
    }
}

impl<T: Element, D: Dimension> Clone for ArcArray<T, D> {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }
}

impl<T: Element, D: Dimension> From<Array<T, D>> for ArcArray<T, D> {
    fn from(arr: Array<T, D>) -> Self {
        Self::from_owned(arr)
    }
}

/// Compute C-contiguous strides for a given shape.
fn compute_c_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0isize; ndim];
    strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    #[test]
    fn arc_from_owned() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let arc = ArcArray::from_owned(arr);
        assert_eq!(arc.shape(), &[3]);
        assert_eq!(arc.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(arc.ref_count(), 1);
    }

    #[test]
    fn arc_clone_shares() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let arc1 = ArcArray::from_owned(arr);
        let arc2 = arc1.clone();
        assert_eq!(arc1.ref_count(), 2);
        assert_eq!(arc2.ref_count(), 2);
        assert_eq!(arc1.as_ptr(), arc2.as_ptr());
    }

    #[test]
    fn arc_cow_on_mutation() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let arc1 = ArcArray::from_owned(arr);
        let mut arc2 = arc1.clone();

        // Before mutation, they share data
        assert_eq!(arc1.as_ptr(), arc2.as_ptr());
        assert_eq!(arc1.ref_count(), 2);

        // Mutate arc2 — this triggers CoW
        arc2.as_slice_mut()[0] = 99.0;

        // After mutation, data is separate
        assert_ne!(arc1.as_ptr(), arc2.as_ptr());
        assert_eq!(arc1.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(arc2.as_slice(), &[99.0, 2.0, 3.0]);
        assert_eq!(arc1.ref_count(), 1);
        assert_eq!(arc2.ref_count(), 1);
    }

    #[test]
    fn arc_view_sees_old_data_after_cow() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mut arc = ArcArray::from_owned(arr);
        let arc_clone = arc.clone();

        // Create a view from the clone (borrows the shared data)
        let view = arc_clone.view();
        assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

        // Mutate the original arc — triggers CoW
        arc.as_slice_mut()[0] = 99.0;

        // The view still sees the old data
        assert_eq!(view.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
        // But arc has the new data
        assert_eq!(arc.as_slice(), &[99.0, 2.0, 3.0]);
    }

    #[test]
    fn arc_unique_no_clone() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mut arc = ArcArray::from_owned(arr);
        let ptr_before = arc.as_ptr();

        // Sole owner — mutation should NOT clone
        arc.as_slice_mut()[0] = 99.0;
        assert_eq!(arc.as_ptr(), ptr_before);
        assert_eq!(arc.as_slice(), &[99.0, 2.0, 3.0]);
    }

    #[test]
    fn arc_into_owned() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let arc = ArcArray::from_owned(arr);
        let owned = arc.into_owned();
        assert_eq!(owned.shape(), &[2, 3]);
    }

    #[test]
    fn arc_mapv_inplace() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let mut arc = ArcArray::from_owned(arr);
        arc.mapv_inplace(|x| x * 2.0);
        assert_eq!(arc.as_slice(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn arc_copy_is_independent() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let arc = ArcArray::from_owned(arr);
        let copy = arc.copy();
        assert_ne!(arc.as_ptr(), copy.as_ptr());
        assert_eq!(arc.ref_count(), 1); // original not shared with copy
        assert_eq!(copy.ref_count(), 1);
    }

    // ----- non-contiguous source coverage (#129) -----

    #[test]
    fn arc_from_owned_after_transpose_is_standard_layout() {
        // Transpose produces an F-contiguous view; going through
        // `as_standard_layout` in manipulation::transpose materializes
        // a standard-layout copy, and wrapping that in ArcArray must
        // yield a usable shared array whose `as_slice` succeeds.
        use crate::dimension::Ix2;
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let transposed = crate::manipulation::transpose(&arr, None).unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);
        let arc = ArcArray::<f64, crate::dimension::IxDyn>::from_owned(transposed);
        // ArcArray::as_slice returns the underlying contiguous buffer
        // in row-major order; the transposed data should be
        // [1, 4, 2, 5, 3, 6] after as_standard_layout.
        assert_eq!(arc.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn arc_from_owned_with_broadcast_to_materializes_data() {
        // broadcast_to produces a stride-0 view that, after
        // `as_standard_layout`, becomes a proper contiguous array
        // with duplicated rows. ArcArray should accept it.
        use crate::dimension::Ix1;
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = crate::manipulation::broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        let arc = ArcArray::<f64, crate::dimension::IxDyn>::from_owned(b);
        // Both rows should be `[1, 2, 3]`.
        assert_eq!(arc.as_slice(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
}
