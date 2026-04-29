// Property-based tests for ferray-stride-tricks
//
// Tests mathematical invariants of stride-trick operations using proptest.

// Property tests sample integer sizes and assert exact float equality on
// view roundtrip invariants by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2};

use ferray_stride_tricks::{broadcast_arrays, broadcast_shapes, broadcast_to, sliding_window_view};

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. sliding_window_view shape is [n-w+1, w] for 1D
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sliding_window_shape(
        n in 2usize..=50,
        w_offset in 0usize..=49,
    ) {
        let w = (w_offset % n) + 1; // w in 1..=n
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let view = sliding_window_view(&arr, &[w]).unwrap();
        prop_assert_eq!(view.shape(), &[n - w + 1, w]);
    }

    // -----------------------------------------------------------------------
    // 2. each window[i] contains the correct elements from source
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sliding_window_elements(
        n in 2usize..=50,
        w_offset in 0usize..=49,
    ) {
        let w = (w_offset % n) + 1;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let view = sliding_window_view(&arr, &[w]).unwrap();
        let num_windows = n - w + 1;
        let flat: Vec<f64> = view.iter().copied().collect();
        for i in 0..num_windows {
            for j in 0..w {
                let actual = flat[i * w + j];
                let expected = (i + j) as f64;
                prop_assert!(
                    (actual - expected).abs() < f64::EPSILON,
                    "window[{}][{}]: expected {}, got {}",
                    i, j, expected, actual,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 3. number of windows == n - w + 1
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sliding_window_count(
        n in 2usize..=50,
        w_offset in 0usize..=49,
    ) {
        let w = (w_offset % n) + 1;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let view = sliding_window_view(&arr, &[w]).unwrap();
        let num_windows = view.shape()[0];
        prop_assert_eq!(num_windows, n - w + 1);
    }

    // -----------------------------------------------------------------------
    // 4. broadcast_to(arr, shape) has the target shape
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_to_shape(
        n in 1usize..=20,
        rows in 1usize..=20,
    ) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let target = [rows, n];
        let view = broadcast_to(&arr, &target).unwrap();
        prop_assert_eq!(view.shape(), &target[..]);
    }

    // -----------------------------------------------------------------------
    // 5. broadcast values replicate correctly (check specific indices)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_to_values(
        n in 1usize..=20,
        rows in 1usize..=20,
    ) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let view = broadcast_to(&arr, &[rows, n]).unwrap();
        let flat: Vec<f64> = view.iter().copied().collect();
        for r in 0..rows {
            for c in 0..n {
                let actual = flat[r * n + c];
                let expected = c as f64;
                prop_assert!(
                    (actual - expected).abs() < f64::EPSILON,
                    "broadcast[{}][{}]: expected {}, got {}",
                    r, c, expected, actual,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 6. broadcast_shapes is commutative
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_shapes_commutative(
        a in 1usize..=20,
        b in 1usize..=20,
    ) {
        // Only test compatible shapes: at least one must be 1, or they must be equal.
        let (sa, sb) = if a == b {
            (vec![a], vec![b])
        } else {
            (vec![1, a], vec![b, 1])
        };
        let forward = broadcast_shapes(&[&sa[..], &sb[..]]);
        let reverse = broadcast_shapes(&[&sb[..], &sa[..]]);
        match (forward, reverse) {
            (Ok(f), Ok(r)) => prop_assert_eq!(f, r),
            (Err(_), Err(_)) => {} // both fail — fine
            _ => prop_assert!(false, "commutativity violated: one succeeded and the other failed"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. broadcast_shapes is idempotent: broadcast_shapes([a, a]) == a
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_shapes_idempotent(
        d1 in 1usize..=20,
        d2 in 1usize..=20,
    ) {
        let shape = vec![d1, d2];
        let result = broadcast_shapes(&[&shape[..], &shape[..]]).unwrap();
        prop_assert_eq!(result, shape);
    }

    // -----------------------------------------------------------------------
    // 8. broadcast_arrays: all output arrays share the same shape
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_arrays_same_shape(
        r in 1usize..=10,
        c in 1usize..=10,
    ) {
        let a = Array::<f64, Ix2>::ones(Ix2::new([r, 1])).unwrap();
        let b = Array::<f64, Ix2>::ones(Ix2::new([1, c])).unwrap();
        let arrays = [a, b];
        let views = broadcast_arrays(&arrays).unwrap();
        prop_assert_eq!(views.len(), 2);
        prop_assert_eq!(views[0].shape(), views[1].shape());
        prop_assert_eq!(views[0].shape(), &[r, c]);
    }

    // -----------------------------------------------------------------------
    // 9. adjacent windows overlap by w-1 elements
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sliding_window_overlap(
        n in 3usize..=50,
        w_raw in 0usize..=49,
    ) {
        let w = (w_raw % (n - 1)) + 2; // w in 2..=n
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let view = sliding_window_view(&arr, &[w]).unwrap();
        let num_windows = n - w + 1;
        let flat: Vec<f64> = view.iter().copied().collect();

        // For each pair of adjacent windows, the last w-1 elements of window[i]
        // must equal the first w-1 elements of window[i+1].
        for i in 0..(num_windows - 1) {
            for j in 1..w {
                let tail = flat[i * w + j];
                let head = flat[(i + 1) * w + (j - 1)];
                prop_assert!(
                    (tail - head).abs() < f64::EPSILON,
                    "overlap mismatch at windows ({}, {}), offset {}: {} vs {}",
                    i, i + 1, j, tail, head,
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 10. broadcast_to preserves data: broadcast [v] to [n] has all same value
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_to_preserves_data(
        val in -100.0f64..100.0,
        n in 1usize..=50,
    ) {
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![val]).unwrap();
        let view = broadcast_to(&arr, &[n]).unwrap();
        prop_assert_eq!(view.shape(), &[n]);
        let flat: Vec<f64> = view.iter().copied().collect();
        for (idx, &elem) in flat.iter().enumerate() {
            prop_assert!(
                (elem - val).abs() < f64::EPSILON,
                "broadcast scalar to [{}]: index {} expected {}, got {}",
                n, idx, val, elem,
            );
        }
    }

    // -----------------------------------------------------------------------
    // 11. sliding_window_view rejects window > array length
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sliding_window_rejects_too_large(
        n in 2usize..=50,
        extra in 1usize..=20,
    ) {
        let w = n + extra; // w > n, always invalid
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let result = sliding_window_view(&arr, &[w]);
        prop_assert!(result.is_err(), "expected error for w={} > n={}", w, n);
    }

    // -----------------------------------------------------------------------
    // 12. broadcasting [1] with [n] gives [n]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_shapes_scalar(
        n in 1usize..=50,
    ) {
        let result = broadcast_shapes(&[&[1usize][..], &[n][..]]).unwrap();
        prop_assert_eq!(result, vec![n]);
    }
}
