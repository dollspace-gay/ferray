// ferray-stride-tricks: Low-level view construction via custom strides.
//
// Implements `numpy.lib.stride_tricks`: sliding window views, broadcast
// views, and the safe/unsafe `as_strided` family.
//
// # Overview
//
// | Function                | Safety | Copies data? |
// |------------------------|--------|--------------|
// | `sliding_window_view`  | safe   | no           |
// | `broadcast_to`         | safe   | no           |
// | `broadcast_arrays`     | safe   | no           |
// | `broadcast_shapes`     | safe   | n/a          |
// | `as_strided`           | safe   | no           |
// | `as_strided_unchecked` | unsafe | no           |

// Stride manipulation routinely converts between `isize` byte strides and
// `usize` element offsets, and divides shape products that may exceed
// `i64::MAX` only on bug paths already validated upstream. Property
// tests assert exact float equality on view roundtrips.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod as_strided;
pub mod broadcast;
pub mod overlap_check;
pub mod sliding_window;

// Re-export primary public functions at crate root for ergonomics.

pub use as_strided::{StridedSource, as_strided, as_strided_unchecked};
pub use broadcast::{broadcast_arrays, broadcast_shapes, broadcast_to};
pub use sliding_window::{sliding_window_view, sliding_window_view_axis};

#[cfg(test)]
mod integration_tests {
    //! Integration tests covering the acceptance criteria (AC-1 through AC-6).

    use ferray_core::Array;
    use ferray_core::dimension::{Ix1, Ix2};

    use crate::{
        as_strided, as_strided_unchecked, broadcast_arrays, broadcast_shapes, broadcast_to,
        sliding_window_view,
    };

    // AC-1: sliding_window_view(&[1,2,3,4,5], (3,)) returns views
    //        [[1,2,3], [2,3,4], [3,4,5]]
    #[test]
    fn ac1_sliding_window_view() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = sliding_window_view(&a, &[3]).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
    }

    // AC-2: broadcast_to(&ones(3), (4, 3)) returns a (4,3) view without
    //        allocation
    #[test]
    fn ac2_broadcast_to_no_alloc() {
        let a = Array::<f64, Ix1>::ones(Ix1::new([3])).unwrap();
        let v = broadcast_to(&a, &[4, 3]).unwrap();
        assert_eq!(v.shape(), &[4, 3]);
        assert_eq!(v.size(), 12);
        // No allocation: view shares the same base pointer
        assert_eq!(v.as_ptr(), a.as_ptr());
        // All elements are 1.0
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0; 12]);
    }

    // AC-3: broadcast_shapes(&[(3,1), (1,4)]) returns (3,4)
    #[test]
    fn ac3_broadcast_shapes() {
        let result = broadcast_shapes(&[&[3, 1][..], &[1, 4][..]]).unwrap();
        assert_eq!(result, vec![3, 4]);
    }

    // AC-4: as_strided with non-overlapping strides succeeds; overlapping
    //        strides returns Err
    #[test]
    fn ac4_as_strided_safe_vs_overlapping() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Non-overlapping: 2x3 with standard row-major strides
        let v = as_strided(&a, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Overlapping: sliding window pattern (strides [1,1])
        let a5 = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let err = as_strided(&a5, &[3, 3], &[1, 1]);
        assert!(err.is_err());
    }

    // AC-5: as_strided_unchecked compiles only in unsafe block
    // (This is enforced by the `unsafe fn` signature; attempting to call
    // it outside an unsafe block will produce a compile error.)
    #[test]
    fn ac5_as_strided_unchecked_requires_unsafe() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        // This call requires `unsafe` — removing the unsafe block would
        // fail to compile.
        let v = unsafe { as_strided_unchecked(&a, &[3, 3], &[1, 1]).unwrap() };
        assert_eq!(v.shape(), &[3, 3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
    }

    // AC-6: cargo test -p ferray-stride-tricks passes (this file!),
    //        cargo clippy clean (checked by CI / manual run).

    // Additional integration: broadcast_arrays
    #[test]
    fn broadcast_arrays_integration() {
        let a = Array::<f64, Ix2>::ones(Ix2::new([4, 1])).unwrap();
        let b = Array::<f64, Ix2>::ones(Ix2::new([1, 3])).unwrap();
        let arrays = [a, b];
        let views = broadcast_arrays(&arrays).unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].shape(), &[4, 3]);
        assert_eq!(views[1].shape(), &[4, 3]);
    }

    // Additional: sliding_window_view is truly zero-copy
    #[test]
    fn sliding_window_is_zero_copy() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([10]), (0..10).map(f64::from).collect()).unwrap();
        let v = sliding_window_view(&a, &[4]).unwrap();
        // Shape: (7, 4)
        assert_eq!(v.shape(), &[7, 4]);
        // Same base pointer
        assert_eq!(v.as_ptr(), a.as_ptr());
    }

    // Additional: as_strided with non-contiguous but non-overlapping strides
    #[test]
    fn as_strided_skip_elements() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([10]), vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            .unwrap();
        // Take every 3rd element: shape (3,), stride (3,)
        // offsets: 0, 3, 6 — within buffer of 10, non-overlapping
        let v = as_strided(&a, &[3], &[3]).unwrap();
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![0, 3, 6]);
    }

    // --- Tests for various stride patterns ---

    #[test]
    fn as_strided_stride_zero_broadcast() {
        // stride 0 broadcasts a single value — only valid with unchecked
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![42.0]).unwrap();
        let v = unsafe { as_strided_unchecked(&a, &[5], &[0]).unwrap() };
        assert_eq!(v.shape(), &[5]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![42.0; 5]);
    }

    #[test]
    fn as_strided_2d_column_broadcast() {
        // (3,) -> (3, 4) via stride [1, 0] — each row repeats the element
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let v = unsafe { as_strided_unchecked(&a, &[3, 4], &[1, 0]).unwrap() };
        assert_eq!(v.shape(), &[3, 4]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![10, 10, 10, 10, 20, 20, 20, 20, 30, 30, 30, 30]);
    }

    #[test]
    fn sliding_window_2d() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        // 2x2 sliding window on 3x4 -> shape (2, 3, 2, 2)
        let v = sliding_window_view(&a, &[2, 2]).unwrap();
        assert_eq!(v.shape(), &[2, 3, 2, 2]);
        // First window (top-left 2x2): [[0,1],[4,5]]
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(&data[0..4], &[0, 1, 4, 5]);
    }

    #[test]
    fn sliding_window_single_element_window() {
        // Window size 1 along each axis: output has double ndim but same data
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let v = sliding_window_view(&a, &[1]).unwrap();
        assert_eq!(v.shape(), &[4, 1]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![10, 20, 30, 40]);
    }

    #[test]
    fn sliding_window_full_window() {
        // Window covers entire array: output has shape (1, n)
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = sliding_window_view(&a, &[5]).unwrap();
        assert_eq!(v.shape(), &[1, 5]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn overlap_check_large_stride_no_overlap() {
        // Large strides that skip elements: no overlap
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([20]), (0..20).map(f64::from).collect()).unwrap();
        let v = as_strided(&a, &[2, 3], &[10, 3]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![0.0, 3.0, 6.0, 10.0, 13.0, 16.0]);
    }
}
