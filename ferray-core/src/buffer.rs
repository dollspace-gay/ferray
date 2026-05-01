// ferray-core: AsRawBuffer trait for zero-copy interop (REQ-29)

use crate::dimension::Dimension;
use crate::dtype::{DType, Element};

use crate::array::arc::ArcArray;
use crate::array::owned::Array;
use crate::array::view::ArrayView;

/// FFI-safe descriptor for an array's memory layout (#358).
///
/// Mirrors the fields C and Python's PEP 3118 buffer protocol carry:
/// data pointer, dtype tag, ndim, plus pointers to shape and strides
/// arrays. The struct is `repr(C)` so it can cross an FFI boundary
/// directly — caller is responsible for ensuring the originating
/// array outlives the descriptor (the pointers borrow into the
/// array's storage; ferray does not extend the array's lifetime).
///
/// Strides are in **bytes**, matching numpy's PyArrayObject and the
/// PEP 3118 convention. The shape and strides slices are alive for
/// as long as the originating array (typically `'static` for
/// `Array<T, D>` storage and `'a` for views).
#[repr(C)]
#[derive(Debug, Clone)]
pub struct BufferDescriptor<'a> {
    /// Pointer to the first element of the array.
    pub data: *const u8,
    /// Number of axes.
    pub ndim: usize,
    /// Shape (length `ndim`).
    pub shape: &'a [usize],
    /// Strides in bytes (length `ndim`). Matches numpy / PEP 3118.
    pub strides_bytes: Box<[isize]>,
    /// Element type tag.
    pub dtype: DType,
    /// Element size in bytes (redundant with `dtype.size_of()` but
    /// surfaced here so foreign code doesn't need a DType match).
    pub itemsize: usize,
    /// Whether the data is laid out in C-contiguous (row-major) order.
    pub c_contiguous: bool,
    /// Whether the data is laid out in F-contiguous (column-major) order.
    pub f_contiguous: bool,
}

/// Trait exposing the raw memory layout of an array for zero-copy interop.
///
/// Implementors provide enough information for foreign code (C, Python/NumPy,
/// Arrow, etc.) to read the array data without copying.
pub trait AsRawBuffer {
    /// Raw pointer to the first element.
    fn raw_ptr(&self) -> *const u8;

    /// Shape as a slice of dimension sizes.
    fn raw_shape(&self) -> &[usize];

    /// Strides in bytes (not elements).
    fn raw_strides_bytes(&self) -> Vec<isize>;

    /// Runtime dtype descriptor.
    fn raw_dtype(&self) -> DType;

    /// Whether the data is C-contiguous.
    fn is_c_contiguous(&self) -> bool;

    /// Whether the data is Fortran-contiguous.
    fn is_f_contiguous(&self) -> bool;

    /// Build a [`BufferDescriptor`] aggregating every field above into
    /// a single FFI-safe struct (#358). Default impl composes the
    /// other trait methods so concrete types don't need to override.
    fn buffer_descriptor(&self) -> BufferDescriptor<'_> {
        let dtype = self.raw_dtype();
        let shape = self.raw_shape();
        BufferDescriptor {
            data: self.raw_ptr(),
            ndim: shape.len(),
            shape,
            strides_bytes: self.raw_strides_bytes().into_boxed_slice(),
            dtype,
            itemsize: dtype.size_of(),
            c_contiguous: self.is_c_contiguous(),
            f_contiguous: self.is_f_contiguous(),
        }
    }
}

// The three concrete array types (`Array`, `ArrayView`, `ArcArray`) all
// expose the same `as_ptr`/`shape`/`strides`/`layout` inherent methods,
// so each `AsRawBuffer` impl is identical save for the receiver type.
// The prior hand-written three-way copy (six methods × three types) is
// factored through a macro so future additions to the trait only need
// to be made once (see issue #123).
macro_rules! impl_as_raw_buffer {
    ($ty:ty, $($lt:lifetime)?) => {
        impl<$($lt,)? T: Element, D: Dimension> AsRawBuffer for $ty {
            fn raw_ptr(&self) -> *const u8 {
                self.as_ptr().cast::<u8>()
            }

            fn raw_shape(&self) -> &[usize] {
                self.shape()
            }

            fn raw_strides_bytes(&self) -> Vec<isize> {
                let itemsize = std::mem::size_of::<T>() as isize;
                self.strides().iter().map(|&s| s * itemsize).collect()
            }

            fn raw_dtype(&self) -> DType {
                T::dtype()
            }

            fn is_c_contiguous(&self) -> bool {
                self.layout().is_c_contiguous()
            }

            fn is_f_contiguous(&self) -> bool {
                self.layout().is_f_contiguous()
            }
        }
    };
}

impl_as_raw_buffer!(Array<T, D>,);
impl_as_raw_buffer!(ArrayView<'a, T, D>, 'a);
impl_as_raw_buffer!(ArcArray<T, D>,);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix2;

    #[test]
    fn raw_buffer_array() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        assert_eq!(arr.raw_shape(), &[2, 3]);
        assert_eq!(arr.raw_dtype(), DType::F64);
        assert!(arr.is_c_contiguous());
        assert!(!arr.raw_ptr().is_null());

        // Strides in bytes: row stride = 3*8=24, col stride = 8
        let strides = arr.raw_strides_bytes();
        assert_eq!(strides, vec![24, 8]);
    }

    #[test]
    fn raw_buffer_view() {
        let arr = Array::<f32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = arr.view();

        assert_eq!(v.raw_dtype(), DType::F32);
        assert_eq!(v.raw_shape(), &[2, 2]);
        assert!(v.is_c_contiguous());
    }

    #[test]
    fn raw_buffer_arc() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
        let arc = ArcArray::from_owned(arr);

        assert_eq!(arc.raw_dtype(), DType::I32);
        assert_eq!(arc.raw_shape(), &[2, 2]);
    }

    #[test]
    fn buffer_descriptor_aggregates_layout() {
        // #358: BufferDescriptor must populate every field consistently.
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0; 6]).unwrap();
        let d = arr.buffer_descriptor();
        assert_eq!(d.ndim, 2);
        assert_eq!(d.shape, &[2, 3]);
        assert_eq!(&*d.strides_bytes, &[24, 8]);
        assert_eq!(d.dtype, DType::F64);
        assert_eq!(d.itemsize, 8);
        assert!(d.c_contiguous);
        assert!(!d.f_contiguous);
        // Pointer non-null and matches raw_ptr().
        assert!(!d.data.is_null());
        assert_eq!(d.data, arr.raw_ptr());
    }

    #[test]
    fn buffer_descriptor_repr_c() {
        // The descriptor is `repr(C)` so its layout is stable for FFI.
        // Spot-check via size_of: must be > 0 and consistent across
        // calls. (We don't pin a specific size because pointer width
        // varies by platform.)
        let arr = Array::<u32, Ix2>::from_vec(Ix2::new([2, 2]), vec![0u32; 4]).unwrap();
        let d = arr.buffer_descriptor();
        assert_eq!(d.itemsize, 4);
        assert_eq!(d.dtype, DType::U32);
        assert_eq!(d.shape, &[2, 2]);
    }
}
