// Property-based tests for ferray-core
//
// Tests mathematical invariants of core array operations using proptest.

// Property tests sample integer sizes and assert exact float equality on
// roundtrip / shape invariants by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::creation::{array, linspace, ones, zeros};
use ferray_core::dimension::broadcast::broadcast_shapes;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::indexing::basic::SliceSpec;
use ferray_core::manipulation::{flatten, reshape, transpose};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate a valid 1-D shape (non-zero).
fn shape_1d() -> impl Strategy<Value = usize> {
    1usize..=50
}

/// Generate a valid 2-D shape where both dims > 0 and total <= 200.
fn shape_2d() -> impl Strategy<Value = (usize, usize)> {
    (1usize..=20, 1usize..=20)
}

/// Generate a Vec<f64> of a given length with reasonable values.
fn vec_f64(len: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(-100.0f64..100.0, len)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. reshape preserves element count
    // -----------------------------------------------------------------------
    #[test]
    fn prop_reshape_preserves_element_count(
        rows in 1usize..=10,
        cols in 1usize..=10,
    ) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = array(Ix2::new([rows, cols]), data).unwrap();

        // Reshape to 1-D
        let b = reshape(&a, &[n]).unwrap();
        prop_assert_eq!(b.size(), a.size());

        // Reshape to different 2-D
        if n >= 2 {
            // Find a factor of n
            let mut factor = 1;
            for f in 2..=n {
                if n % f == 0 {
                    factor = f;
                    break;
                }
            }
            let other = n / factor;
            let c = reshape(&a, &[factor, other]).unwrap();
            prop_assert_eq!(c.size(), a.size());
        }
    }

    // -----------------------------------------------------------------------
    // 2. transpose is involutory for 2-D arrays
    // -----------------------------------------------------------------------
    #[test]
    fn prop_transpose_involutory((rows, cols) in shape_2d()) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = array(Ix2::new([rows, cols]), data).unwrap();

        let t1 = transpose(&a, None).unwrap();
        let t2 = transpose(&t1, None).unwrap();

        // t2 should have same shape as a
        prop_assert_eq!(t2.shape(), a.shape());

        // t2 elements should equal a elements
        let orig: Vec<f64> = a.iter().copied().collect();
        let back: Vec<f64> = t2.iter().copied().collect();
        prop_assert_eq!(orig, back);
    }

    // -----------------------------------------------------------------------
    // 3. flatten preserves total element count
    // -----------------------------------------------------------------------
    #[test]
    fn prop_flatten_preserves_elements((rows, cols) in shape_2d()) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = array(Ix2::new([rows, cols]), data).unwrap();

        let flat = flatten(&a).unwrap();
        prop_assert_eq!(flat.size(), n);
        prop_assert_eq!(flat.shape(), &[n]);

        // The flattened elements should be the same as iteration order
        let orig: Vec<f64> = a.iter().copied().collect();
        let flat_data: Vec<f64> = flat.iter().copied().collect();
        prop_assert_eq!(orig, flat_data);
    }

    // -----------------------------------------------------------------------
    // 4. broadcast_shapes is commutative
    // -----------------------------------------------------------------------
    #[test]
    fn prop_broadcast_shapes_commutative(
        a_shape in proptest::collection::vec(1usize..=5, 1..=4),
        b_shape in proptest::collection::vec(1usize..=5, 1..=4),
    ) {
        let result_ab = broadcast_shapes(&a_shape, &b_shape);
        let result_ba = broadcast_shapes(&b_shape, &a_shape);

        match (result_ab, result_ba) {
            (Ok(ab), Ok(ba)) => prop_assert_eq!(ab, ba),
            (Err(_), Err(_)) => { /* both fail, consistent */ }
            _ => prop_assert!(false, "broadcast_shapes not commutative"),
        }
    }

    // -----------------------------------------------------------------------
    // 5. zeros creates all-zero arrays
    // -----------------------------------------------------------------------
    #[test]
    fn prop_zeros_all_zero(n in shape_1d()) {
        let a = zeros::<f64, Ix1>(Ix1::new([n])).unwrap();
        prop_assert_eq!(a.size(), n);
        for &v in a.iter() {
            prop_assert_eq!(v, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // 6. ones creates all-one arrays
    // -----------------------------------------------------------------------
    #[test]
    fn prop_ones_all_one(n in shape_1d()) {
        let a = ones::<f64, Ix1>(Ix1::new([n])).unwrap();
        prop_assert_eq!(a.size(), n);
        for &v in a.iter() {
            prop_assert_eq!(v, 1.0);
        }
    }

    // -----------------------------------------------------------------------
    // 7. slice [0..n] returns n elements
    // -----------------------------------------------------------------------
    #[test]
    fn prop_slice_returns_correct_count(
        total in 2usize..=50,
        end in 1usize..=50,
    ) {
        let end = end.min(total);
        let data: Vec<f64> = (0..total).map(|i| i as f64).collect();
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([total]), data).unwrap();

        let spec = SliceSpec::new(0, end as isize);
        let view = a.slice_axis(ferray_core::Axis(0), spec).unwrap();
        prop_assert_eq!(view.size(), end);
    }

    // -----------------------------------------------------------------------
    // 8. reshape with incompatible shape fails
    // -----------------------------------------------------------------------
    #[test]
    fn prop_reshape_incompatible_shape_fails(
        n in 2usize..=50,
    ) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = array(Ix1::new([n]), data).unwrap();

        // Try to reshape to a size that doesn't match
        let bad_size = n + 1;
        let result = reshape(&a, &[bad_size]);
        prop_assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // 9. linspace produces correct number of elements
    // -----------------------------------------------------------------------
    #[test]
    fn prop_linspace_count(
        num in 0usize..=100,
        start in -100.0f64..100.0,
        stop in -100.0f64..100.0,
    ) {
        let a = linspace(start, stop, num, true).unwrap();
        prop_assert_eq!(a.size(), num);
    }

    // -----------------------------------------------------------------------
    // 10. zeros and ones have the same shape
    // -----------------------------------------------------------------------
    #[test]
    fn prop_zeros_ones_same_shape((rows, cols) in shape_2d()) {
        let z = zeros::<f64, Ix2>(Ix2::new([rows, cols])).unwrap();
        let o = ones::<f64, Ix2>(Ix2::new([rows, cols])).unwrap();
        prop_assert_eq!(z.shape(), o.shape());
    }

    // -----------------------------------------------------------------------
    // 11. reshape then flatten gives same elements in same order
    // -----------------------------------------------------------------------
    #[test]
    fn prop_reshape_flatten_roundtrip(
        rows in 1usize..=10,
        cols in 1usize..=10,
    ) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = array(Ix1::new([n]), data.clone()).unwrap();

        let reshaped = reshape(&a, &[rows, cols]).unwrap();
        let flat = flatten(&reshaped).unwrap();
        let flat_data: Vec<f64> = flat.iter().copied().collect();
        prop_assert_eq!(data, flat_data);
    }

    // -----------------------------------------------------------------------
    // 12. from_vec roundtrip preserves data
    // -----------------------------------------------------------------------
    #[test]
    fn prop_from_vec_preserves_data(data in vec_f64(20)) {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([20]), data.clone()).unwrap();
        let stored: Vec<f64> = a.iter().copied().collect();
        prop_assert_eq!(data, stored);
    }

    // -----------------------------------------------------------------------
    // 13. dynamic-rank arrays preserve shape and data
    // -----------------------------------------------------------------------
    #[test]
    fn prop_dynamic_rank_preserves(
        rows in 1usize..=10,
        cols in 1usize..=10,
    ) {
        let n = rows * cols;
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[rows, cols]), data.clone()).unwrap();
        prop_assert_eq!(a.shape(), &[rows, cols]);
        prop_assert_eq!(a.ndim(), 2);
        let stored: Vec<f64> = a.iter().copied().collect();
        prop_assert_eq!(data, stored);
    }
}
