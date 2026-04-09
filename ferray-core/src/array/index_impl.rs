//! `Index` and `IndexMut` impls for `Array<T, IxN>` with `[usize; N]` indices.
//!
//! Enables `arr[[i, j]]` syntax for element access, matching ndarray's ergonomics.
//! Panics on out-of-bounds (standard `Index` trait semantics).

use std::ops::{Index, IndexMut};

use crate::dtype::Element;

use super::Array;

/// Compute the flat row-major offset for the given indices and shape.
///
/// # Panics
/// Panics if any index is out of bounds.
#[inline]
fn flat_offset(indices: &[usize], shape: &[usize]) -> usize {
    debug_assert_eq!(indices.len(), shape.len());
    let mut offset = 0;
    for (i, (&idx, &dim)) in indices.iter().zip(shape.iter()).enumerate() {
        assert!(
            idx < dim,
            "index out of bounds: axis {i} index {idx} >= dimension {dim}"
        );
        offset = offset * dim + idx;
    }
    offset
}

macro_rules! impl_index {
    ($ix:ident, $n:expr) => {
        impl<T: Element> Index<[usize; $n]> for Array<T, crate::dimension::$ix> {
            type Output = T;

            #[inline]
            fn index(&self, idx: [usize; $n]) -> &T {
                let offset = flat_offset(&idx, self.shape());
                // to_vec_flat iterates in row-major order, and as_slice returns
                // the contiguous buffer when the array is C-contiguous (the default).
                // For non-contiguous arrays, fall back to the strides.
                if let Some(slice) = self.as_slice() {
                    &slice[offset]
                } else {
                    // Non-contiguous: compute via strides
                    let strides = self.strides();
                    let mut raw_offset: isize = 0;
                    for (&i, &s) in idx.iter().zip(strides.iter()) {
                        raw_offset += i as isize * s;
                    }
                    unsafe { &*self.as_ptr().offset(raw_offset) }
                }
            }
        }

        impl<T: Element> IndexMut<[usize; $n]> for Array<T, crate::dimension::$ix> {
            #[inline]
            fn index_mut(&mut self, idx: [usize; $n]) -> &mut T {
                // Bounds check first, then use raw pointer to avoid borrow conflicts
                let strides = self.strides().to_vec();
                let shape = self.shape().to_vec();
                let _ = flat_offset(&idx, &shape); // panics if out of bounds
                let mut raw_offset: isize = 0;
                for (&i, &s) in idx.iter().zip(strides.iter()) {
                    raw_offset += i as isize * s;
                }
                // SAFETY: all indices validated in-bounds by flat_offset
                unsafe { &mut *self.as_mut_ptr().offset(raw_offset) }
            }
        }
    };
}

impl_index!(Ix1, 1);
impl_index!(Ix2, 2);
impl_index!(Ix3, 3);
impl_index!(Ix4, 4);
impl_index!(Ix5, 5);
impl_index!(Ix6, 6);

// IxDyn uses &[usize] slice indexing instead of fixed-size arrays
impl<T: Element> Index<&[usize]> for Array<T, crate::dimension::IxDyn> {
    type Output = T;

    #[inline]
    fn index(&self, idx: &[usize]) -> &T {
        assert_eq!(
            idx.len(),
            self.ndim(),
            "index dimension mismatch: got {} indices for {}D array",
            idx.len(),
            self.ndim()
        );
        let offset = flat_offset(idx, self.shape());
        if let Some(slice) = self.as_slice() {
            &slice[offset]
        } else {
            let strides = self.strides();
            let mut raw_offset: isize = 0;
            for (&i, &s) in idx.iter().zip(strides.iter()) {
                raw_offset += i as isize * s;
            }
            unsafe { &*self.as_ptr().offset(raw_offset) }
        }
    }
}

impl<T: Element> IndexMut<&[usize]> for Array<T, crate::dimension::IxDyn> {
    #[inline]
    fn index_mut(&mut self, idx: &[usize]) -> &mut T {
        assert_eq!(
            idx.len(),
            self.ndim(),
            "index dimension mismatch: got {} indices for {}D array",
            idx.len(),
            self.ndim()
        );
        let strides = self.strides().to_vec();
        let shape = self.shape().to_vec();
        let _ = flat_offset(idx, &shape); // panics if out of bounds
        let mut raw_offset: isize = 0;
        for (&i, &s) in idx.iter().zip(strides.iter()) {
            raw_offset += i as isize * s;
        }
        // SAFETY: all indices validated in-bounds by flat_offset
        unsafe { &mut *self.as_mut_ptr().offset(raw_offset) }
    }
}

#[cfg(test)]
mod tests {
    use crate::dimension::{Ix1, Ix2, Ix3, Ix4, IxDyn};

    use super::*;

    #[test]
    fn index_1d() {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        assert_eq!(arr[[0]], 10.0);
        assert_eq!(arr[[3]], 40.0);
    }

    #[test]
    fn index_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(arr[[0, 0]], 1);
        assert_eq!(arr[[0, 2]], 3);
        assert_eq!(arr[[1, 0]], 4);
        assert_eq!(arr[[1, 2]], 6);
    }

    #[test]
    fn index_3d() {
        // 2x2x2 = 8 elements
        let arr = Array::<f32, Ix3>::from_vec(
            Ix3::new([2, 2, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        assert_eq!(arr[[0, 0, 0]], 1.0);
        assert_eq!(arr[[0, 0, 1]], 2.0);
        assert_eq!(arr[[1, 1, 1]], 8.0);
    }

    #[test]
    fn index_4d() {
        // 2x2x2x2 = 16 elements
        let data: Vec<i32> = (0..16).collect();
        let arr = Array::<i32, Ix4>::from_vec(Ix4::new([2, 2, 2, 2]), data).unwrap();
        assert_eq!(arr[[0, 0, 0, 0]], 0);
        assert_eq!(arr[[0, 0, 0, 1]], 1);
        assert_eq!(arr[[1, 1, 1, 1]], 15);
    }

    #[test]
    fn index_mut_2d() {
        let mut arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![0; 6]).unwrap();
        arr[[0, 1]] = 42;
        arr[[1, 2]] = 99;
        assert_eq!(arr[[0, 1]], 42);
        assert_eq!(arr[[1, 2]], 99);
        assert_eq!(arr[[0, 0]], 0);
    }

    #[test]
    fn index_dyn() {
        let arr =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert_eq!(arr[&[0, 0][..]], 1.0);
        assert_eq!(arr[&[1, 2][..]], 6.0);
    }

    #[test]
    fn index_mut_dyn() {
        let mut arr = Array::<i32, IxDyn>::from_vec(IxDyn::new(&[3]), vec![0, 0, 0]).unwrap();
        arr[&[1][..]] = 77;
        assert_eq!(arr[&[1][..]], 77);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn index_out_of_bounds() {
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        let _ = arr[[2, 0]]; // row 2 doesn't exist
    }

    #[test]
    #[should_panic(expected = "index dimension mismatch")]
    fn index_dyn_wrong_ndim() {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        let _ = arr[&[0][..]]; // 1 index for 2D array
    }
}
