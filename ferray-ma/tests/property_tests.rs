// Property-based tests for ferray-ma
//
// Tests invariants of masked array operations using proptest.

// Property tests sample integer sizes and assert exact float equality on
// roundtrip / mask invariants by design.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;

use ferray_ma::mask_ops::count_masked;
use ferray_ma::{MaskedArray, masked_add, masked_equal, masked_invalid, masked_mul, masked_sub};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn mask1(data: Vec<bool>) -> Array<bool, Ix1> {
    let n = data.len();
    Array::<bool, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. count() + count_masked() == size()
    // -----------------------------------------------------------------------
    #[test]
    fn prop_count_plus_masked_eq_size(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let ma = MaskedArray::new(arr1(data), mask1(mask_data)).unwrap();
        let count = ma.count().unwrap();
        let masked = count_masked(&ma).unwrap();
        prop_assert_eq!(
            count + masked, n,
            "count({}) + count_masked({}) != size({})", count, masked, n
        );
    }

    // -----------------------------------------------------------------------
    // 2. masked_add mask == union of input masks
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mask_propagation_add(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let mask_a: Vec<bool> = data.iter().map(|x| *x > 30.0).collect();
        let mask_b: Vec<bool> = data.iter().map(|x| *x < -30.0).collect();
        let a = MaskedArray::new(arr1(data), mask1(mask_a.clone())).unwrap();
        let b = MaskedArray::new(arr1(vec![1.0; n]), mask1(mask_b.clone())).unwrap();
        let result = masked_add(&a, &b).unwrap();
        let result_mask: Vec<bool> = result.mask().iter().copied().collect();
        let expected: Vec<bool> = mask_a.iter().zip(mask_b.iter()).map(|(a, b)| *a || *b).collect();
        prop_assert_eq!(result_mask, expected, "add mask != union of input masks");
    }

    // -----------------------------------------------------------------------
    // 3. masked_sub mask == union of input masks
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mask_propagation_sub(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let mask_a: Vec<bool> = data.iter().map(|x| *x > 30.0).collect();
        let mask_b: Vec<bool> = data.iter().map(|x| *x < -30.0).collect();
        let a = MaskedArray::new(arr1(data), mask1(mask_a.clone())).unwrap();
        let b = MaskedArray::new(arr1(vec![1.0; n]), mask1(mask_b.clone())).unwrap();
        let result = masked_sub(&a, &b).unwrap();
        let result_mask: Vec<bool> = result.mask().iter().copied().collect();
        let expected: Vec<bool> = mask_a.iter().zip(mask_b.iter()).map(|(a, b)| *a || *b).collect();
        prop_assert_eq!(result_mask, expected, "sub mask != union of input masks");
    }

    // -----------------------------------------------------------------------
    // 4. masked_mul mask == union of input masks
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mask_propagation_mul(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let mask_a: Vec<bool> = data.iter().map(|x| *x > 30.0).collect();
        let mask_b: Vec<bool> = data.iter().map(|x| *x < -30.0).collect();
        let a = MaskedArray::new(arr1(data), mask1(mask_a.clone())).unwrap();
        let b = MaskedArray::new(arr1(vec![1.0; n]), mask1(mask_b.clone())).unwrap();
        let result = masked_mul(&a, &b).unwrap();
        let result_mask: Vec<bool> = result.mask().iter().copied().collect();
        let expected: Vec<bool> = mask_a.iter().zip(mask_b.iter()).map(|(a, b)| *a || *b).collect();
        prop_assert_eq!(result_mask, expected, "mul mask != union of input masks");
    }

    // -----------------------------------------------------------------------
    // 5. from_data produces all-false mask
    // -----------------------------------------------------------------------
    #[test]
    fn prop_from_data_no_mask(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let ma = MaskedArray::from_data(arr1(data)).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        prop_assert_eq!(mask_vals, vec![false; n], "from_data should produce all-false mask");
    }

    // -----------------------------------------------------------------------
    // 6. masked_equal masks exactly the positions where data == val
    // -----------------------------------------------------------------------
    #[test]
    fn prop_masked_equal_masks_target(data in proptest::collection::vec(-10.0f64..10.0, 2..=20)) {
        // Round to get some repeats, pick 0.0 as the target
        let rounded: Vec<f64> = data.iter().map(|x| x.round()).collect();
        let target = 0.0_f64;
        let arr = arr1(rounded.clone());
        let ma = masked_equal(&arr, target).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        for (i, v) in rounded.iter().enumerate() {
            prop_assert_eq!(
                mask_vals[i], *v == target,
                "masked_equal mismatch at index {}: value={}, mask={}", i, v, mask_vals[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 7. compressed().len() == count()
    // -----------------------------------------------------------------------
    #[test]
    fn prop_compressed_length(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let ma = MaskedArray::new(arr1(data), mask1(mask_data)).unwrap();
        let compressed = ma.compressed().unwrap();
        let count = ma.count().unwrap();
        prop_assert_eq!(
            compressed.size(), count,
            "compressed len {} != count {}", compressed.size(), count
        );
    }

    // -----------------------------------------------------------------------
    // 8. filled(fill_val) at unmasked positions == original data
    // -----------------------------------------------------------------------
    #[test]
    fn prop_filled_preserves_unmasked(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let ma = MaskedArray::new(arr1(data.clone()), mask1(mask_data.clone())).unwrap();
        let fill_val = -999.0_f64;
        let filled = ma.filled(fill_val).unwrap();
        let filled_vals: Vec<f64> = filled.iter().copied().collect();
        for (i, (fv, m)) in filled_vals.iter().zip(mask_data.iter()).enumerate() {
            if !m {
                prop_assert!(
                    (*fv - data[i]).abs() < 1e-15,
                    "filled changed unmasked value at index {}: {} vs {}", i, fv, data[i]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 9. mean of masked array == mean of compressed array
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mean_ignores_masked(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let ma = MaskedArray::new(arr1(data), mask1(mask_data)).unwrap();
        let ma_mean = ma.mean().unwrap();
        let compressed = ma.compressed().unwrap();
        let comp_vals: Vec<f64> = compressed.iter().copied().collect();
        if comp_vals.is_empty() {
            prop_assert!(ma_mean.is_nan(), "mean of all-masked should be NaN");
        } else {
            let comp_sum: f64 = comp_vals.iter().sum();
            let comp_mean = comp_sum / comp_vals.len() as f64;
            let diff = (ma_mean - comp_mean).abs();
            let scale = ma_mean.abs().max(1.0);
            prop_assert!(
                diff / scale < 1e-10,
                "masked mean {} != compressed mean {}", ma_mean, comp_mean
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. sort() puts unmasked elements in non-decreasing order
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sort_unmasked_monotone(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let ma = MaskedArray::new(arr1(data), mask1(mask_data)).unwrap();
        let sorted = ma.sort().unwrap();
        let sorted_data: Vec<f64> = sorted.data().iter().copied().collect();
        let sorted_mask: Vec<bool> = sorted.mask().iter().copied().collect();
        // Collect unmasked values from sorted result
        let unmasked: Vec<f64> = sorted_data.iter()
            .zip(sorted_mask.iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .collect();
        for i in 1..unmasked.len() {
            prop_assert!(
                unmasked[i] >= unmasked[i - 1],
                "sort not monotone at index {}: {} < {}", i, unmasked[i], unmasked[i - 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 11. harden_mask prevents unmask via set_mask with all-false
    // -----------------------------------------------------------------------
    #[test]
    fn prop_harden_mask_prevents_unmask(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        let n = data.len();
        let mask_data: Vec<bool> = data.iter().map(|x| *x > 50.0).collect();
        let original_mask = mask_data.clone();
        let mut ma = MaskedArray::new(arr1(data), mask1(mask_data)).unwrap();
        ma.harden_mask().unwrap();
        // Try to clear all mask bits
        let all_false = mask1(vec![false; n]);
        ma.set_mask(all_false).unwrap();
        let result_mask: Vec<bool> = ma.mask().iter().copied().collect();
        // Hard mask: union of old mask and all-false == old mask
        prop_assert_eq!(
            result_mask, original_mask,
            "harden_mask did not preserve mask under set_mask(all-false)"
        );
    }

    // -----------------------------------------------------------------------
    // 12. masked_invalid masks NaN positions
    // -----------------------------------------------------------------------
    #[test]
    fn prop_masked_invalid_masks_nan(data in proptest::collection::vec(-100.0f64..100.0, 2..=20)) {
        // Insert a NaN at every position where value > 80
        let modified: Vec<f64> = data.iter()
            .map(|x| if *x > 80.0 { f64::NAN } else { *x })
            .collect();
        let arr = arr1(modified.clone());
        let ma = masked_invalid(&arr).unwrap();
        let mask_vals: Vec<bool> = ma.mask().iter().copied().collect();
        for (i, v) in modified.iter().enumerate() {
            if v.is_nan() {
                prop_assert!(
                    mask_vals[i],
                    "masked_invalid did not mask NaN at index {}", i
                );
            } else if v.is_finite() {
                prop_assert!(
                    !mask_vals[i],
                    "masked_invalid incorrectly masked finite value {} at index {}", v, i
                );
            }
        }
    }
}
