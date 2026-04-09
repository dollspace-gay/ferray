// Property-based tests for ferray-stats
//
// Tests mathematical invariants of statistical functions using proptest.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

use ferray_stats::sorting::{SortKind, sort};
use ferray_stats::{argmax, argmin, cumsum, max, mean, min, std_, sum, unique, var};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn scalar_val(a: &Array<f64, ferray_core::IxDyn>) -> f64 {
    *a.iter().next().unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. mean(constant_array) == constant
    // -----------------------------------------------------------------------
    #[test]
    fn prop_mean_constant(c in -100.0f64..100.0, n in 1usize..=50) {
        let a = arr1(vec![c; n]);
        let m = mean(&a, None).unwrap();
        let val = scalar_val(&m);
        prop_assert!((val - c).abs() < 1e-10, "mean of constant {} array = {}", c, val);
    }

    // -----------------------------------------------------------------------
    // 2. var(constant_array) == 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_var_constant(c in -100.0f64..100.0, n in 2usize..=50) {
        let a = arr1(vec![c; n]);
        let v = var(&a, None, 0).unwrap();
        let val = scalar_val(&v);
        prop_assert!(val.abs() < 1e-10, "var of constant {} array = {}", c, val);
    }

    // -----------------------------------------------------------------------
    // 3. std(constant_array) == 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_std_constant(c in -100.0f64..100.0, n in 2usize..=50) {
        let a = arr1(vec![c; n]);
        let s = std_(&a, None, 0).unwrap();
        let val = scalar_val(&s);
        prop_assert!(val.abs() < 1e-10, "std of constant {} array = {}", c, val);
    }

    // -----------------------------------------------------------------------
    // 4. sum(a) == len(a) * mean(a) (approximately)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sum_eq_len_times_mean(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let n = data.len();
        let a = arr1(data);
        let s = sum(&a, None).unwrap();
        let m = mean(&a, None).unwrap();
        let s_val = scalar_val(&s);
        let m_val = scalar_val(&m);
        let expected = n as f64 * m_val;
        let diff = (s_val - expected).abs();
        let scale = s_val.abs().max(1.0);
        prop_assert!(diff / scale < 1e-10, "sum={} != len*mean={}", s_val, expected);
    }

    // -----------------------------------------------------------------------
    // 5. sort is idempotent: sort(sort(a)) == sort(a)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sort_idempotent(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data);
        let sorted1 = sort(&a, None, SortKind::Stable).unwrap();
        let sorted2 = sort(&sorted1, None, SortKind::Stable).unwrap();
        let v1: Vec<f64> = sorted1.iter().copied().collect();
        let v2: Vec<f64> = sorted2.iter().copied().collect();
        prop_assert_eq!(v1, v2);
    }

    // -----------------------------------------------------------------------
    // 6. sort output is monotonically non-decreasing
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sort_monotone(data in proptest::collection::vec(-100.0f64..100.0, 2..=50)) {
        let a = arr1(data);
        let sorted = sort(&a, None, SortKind::Stable).unwrap();
        let vals: Vec<f64> = sorted.iter().copied().collect();
        for i in 1..vals.len() {
            prop_assert!(
                vals[i] >= vals[i - 1],
                "sort not monotone at index {}: {} < {}",
                i, vals[i], vals[i - 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 7. unique output has no duplicates
    // -----------------------------------------------------------------------
    #[test]
    fn prop_unique_no_duplicates(data in proptest::collection::vec(-20.0f64..20.0, 1..=50)) {
        // Round to integers so we get some duplicates
        let rounded: Vec<f64> = data.iter().map(|x| x.round()).collect();
        let a = arr1(rounded);
        let result = unique(&a, false, false, false).unwrap();
        let vals: Vec<f64> = result.values.iter().copied().collect();
        for i in 1..vals.len() {
            prop_assert!(
                vals[i] != vals[i - 1],
                "unique has duplicates at index {}: {} == {}",
                i, vals[i], vals[i - 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 8. unique output is sorted
    // -----------------------------------------------------------------------
    #[test]
    fn prop_unique_sorted(data in proptest::collection::vec(-50.0f64..50.0, 1..=50)) {
        let rounded: Vec<f64> = data.iter().map(|x| x.round()).collect();
        let a = arr1(rounded);
        let result = unique(&a, false, false, false).unwrap();
        let vals: Vec<f64> = result.values.iter().copied().collect();
        for i in 1..vals.len() {
            prop_assert!(
                vals[i] >= vals[i - 1],
                "unique not sorted: {} < {} at index {}",
                vals[i], vals[i - 1], i
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. argmin indexes the minimum value
    // -----------------------------------------------------------------------
    #[test]
    fn prop_argmin_indexes_min(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data.clone());
        let ami = argmin(&a, None).unwrap();
        let idx = *ami.iter().next().unwrap() as usize;
        let mn = min(&a, None).unwrap();
        let min_val = scalar_val(&mn);
        prop_assert!(
            (data[idx] - min_val).abs() < 1e-15,
            "argmin index {} has value {} but min is {}",
            idx, data[idx], min_val
        );
    }

    // -----------------------------------------------------------------------
    // 10. argmax indexes the maximum value
    // -----------------------------------------------------------------------
    #[test]
    fn prop_argmax_indexes_max(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data.clone());
        let amx = argmax(&a, None).unwrap();
        let idx = *amx.iter().next().unwrap() as usize;
        let mx = max(&a, None).unwrap();
        let max_val = scalar_val(&mx);
        prop_assert!(
            (data[idx] - max_val).abs() < 1e-15,
            "argmax index {} has value {} but max is {}",
            idx, data[idx], max_val
        );
    }

    // -----------------------------------------------------------------------
    // 11. cumsum last element equals sum
    // -----------------------------------------------------------------------
    #[test]
    fn prop_cumsum_last_eq_sum(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data);
        let cs = cumsum(&a, None).unwrap();
        let s = sum(&a, None).unwrap();
        let cs_last = *cs.iter().last().unwrap();
        let s_val = scalar_val(&s);
        let diff = (cs_last - s_val).abs();
        let scale = s_val.abs().max(1.0);
        prop_assert!(
            diff / scale < 1e-10,
            "cumsum last = {}, sum = {}, diff = {}",
            cs_last, s_val, diff
        );
    }

    // -----------------------------------------------------------------------
    // 12. min(a) <= mean(a) <= max(a) for non-empty arrays
    // -----------------------------------------------------------------------
    #[test]
    fn prop_min_le_mean_le_max(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data);
        let mn = min(&a, None).unwrap();
        let mx = max(&a, None).unwrap();
        let m = mean(&a, None).unwrap();
        let min_val = scalar_val(&mn);
        let max_val = scalar_val(&mx);
        let mean_val = scalar_val(&m);
        prop_assert!(
            min_val <= mean_val + 1e-10,
            "min {} > mean {}",
            min_val, mean_val
        );
        prop_assert!(
            mean_val <= max_val + 1e-10,
            "mean {} > max {}",
            mean_val, max_val
        );
    }

    // -----------------------------------------------------------------------
    // 13. sum of sorted == sum of original
    // -----------------------------------------------------------------------
    #[test]
    fn prop_sum_invariant_under_sort(data in proptest::collection::vec(-100.0f64..100.0, 1..=50)) {
        let a = arr1(data);
        let sorted = sort(&a, None, SortKind::Stable).unwrap();
        let s1 = sum(&a, None).unwrap();
        let s2 = sum(&sorted, None).unwrap();
        let v1 = scalar_val(&s1);
        let v2 = scalar_val(&s2);
        let scale = v1.abs().max(1.0);
        prop_assert!(
            (v1 - v2).abs() / scale < 1e-10,
            "sum changed after sort: {} vs {}",
            v1, v2
        );
    }
}
