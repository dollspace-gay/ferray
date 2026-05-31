//! Regression coverage for #918 — `ferray_stats::sort` / `argsort` and the
//! partition / lexsort / searchsorted family must place NaN LAST (total
//! order), matching `numpy`'s `np.sort` convention rather than the old
//! `partial_cmp(...).unwrap_or(Equal)` behaviour that treated NaN as equal
//! to everything (leaving the array unsorted).
//!
//! Expected values are taken from live numpy 2.4:
//!   np.sort([3., nan, 1.])            -> [1., 3., nan]
//!   np.sort([nan, nan, 1., 2.])       -> [1., 2., nan, nan]
//!   np.argsort([3., nan, 1.])         -> [2, 0, 1]
//!   np.sort([inf, -inf, 0., nan, 1.]) -> [-inf, 0., 1., inf, nan]
//!   np.sort([[3,nan,1],[nan,2,0]], axis=1) -> [[1,3,nan],[0,2,nan]]
//!   np.sort([5,2,8,1])                -> [1,2,5,8]  (integers unaffected)

use ferray_core::{Array, Ix1, Ix2};
use ferray_stats::{SortKind, argsort, sort};

#[test]
fn sort_nan_last_quick_f64() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![3.0, f64::NAN, 1.0]).unwrap();
    let s = sort(&a, None, SortKind::Quick).unwrap();
    let d: Vec<f64> = s.iter().copied().collect();
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 3.0);
    assert!(d[2].is_nan());
}

#[test]
fn sort_nan_last_stable_multiple() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![f64::NAN, f64::NAN, 1.0, 2.0]).unwrap();
    let s = sort(&a, None, SortKind::Stable).unwrap();
    let d: Vec<f64> = s.iter().copied().collect();
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 2.0);
    assert!(d[2].is_nan() && d[3].is_nan());
}

#[test]
fn argsort_nan_index_last() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![3.0, f64::NAN, 1.0]).unwrap();
    let idx = argsort(&a, None).unwrap();
    let d: Vec<u64> = idx.iter().copied().collect();
    assert_eq!(d, vec![2, 0, 1]);
}

#[test]
fn sort_2d_axis1_nan_last() {
    let a = Array::<f64, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![3.0, f64::NAN, 1.0, f64::NAN, 2.0, 0.0],
    )
    .unwrap();
    let s = sort(&a, Some(1), SortKind::Quick).unwrap();
    let d: Vec<f64> = s.iter().copied().collect();
    // row 0 -> [1, 3, nan]
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 3.0);
    assert!(d[2].is_nan());
    // row 1 -> [0, 2, nan]
    assert_eq!(d[3], 0.0);
    assert_eq!(d[4], 2.0);
    assert!(d[5].is_nan());
}

#[test]
fn sort_f32_nan_last() {
    let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![3.0, f32::NAN, 1.0]).unwrap();
    let s = sort(&a, None, SortKind::Quick).unwrap();
    let d: Vec<f32> = s.iter().copied().collect();
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 3.0);
    assert!(d[2].is_nan());
}

#[test]
fn sort_inf_order_then_nan() {
    let a = Array::<f64, Ix1>::from_vec(
        Ix1::new([5]),
        vec![f64::INFINITY, f64::NEG_INFINITY, 0.0, f64::NAN, 1.0],
    )
    .unwrap();
    let s = sort(&a, None, SortKind::Quick).unwrap();
    let d: Vec<f64> = s.iter().copied().collect();
    assert_eq!(d[0], f64::NEG_INFINITY);
    assert_eq!(d[1], 0.0);
    assert_eq!(d[2], 1.0);
    assert_eq!(d[3], f64::INFINITY);
    assert!(d[4].is_nan());
}

#[test]
fn negative_nan_also_last() {
    // np.sort([1., nan, -nan, 2.]) -> [1., 2., nan, nan]; both NaN signs last.
    let a =
        Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, -f64::NAN, 2.0]).unwrap();
    let s = sort(&a, None, SortKind::Quick).unwrap();
    let d: Vec<f64> = s.iter().copied().collect();
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 2.0);
    assert!(d[2].is_nan() && d[3].is_nan());
}

#[test]
fn integer_sort_unregressed() {
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![5, 2, 8, 1]).unwrap();
    let s = sort(&a, None, SortKind::Quick).unwrap();
    assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![1, 2, 5, 8]);
}
