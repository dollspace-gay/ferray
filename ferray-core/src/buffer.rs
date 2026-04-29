// ferray-core: AsRawBuffer trait for zero-copy interop (REQ-29)

use crate::dimension::Dimension;
use crate::dtype::{DType, Element};

use crate::array::arc::ArcArray;
use crate::array::owned::Array;
use crate::array::view::ArrayView;

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
}
