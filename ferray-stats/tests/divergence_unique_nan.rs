//! Regression coverage for #977 — `ferray_stats::unique` (and the
//! `unique_values` / `unique_counts` Array-API aliases that route through it)
//! must sort floats NaN-LAST (total order) and collapse all NaN to a SINGLE
//! NaN (`equal_nan=True`, numpy's default), matching `numpy.unique` rather
//! than the old `partial_cmp(...).unwrap_or(Equal)` behaviour that treated
//! NaN as equal to everything — leaving the result unsorted with NaN not
//! deduplicated.
//!
//! Expected values are taken from live numpy 2.4:
//!   np.unique([3., nan, 1., nan, 1.])             -> [1., 3., nan]
//!   np.unique([1., nan, 1., nan])                 -> [1., nan]
//!   np.unique([3., nan, 1., nan, 1.], rc=True)    -> ([1.,3.,nan], [2,1,2])
//!   np.unique([inf, nan, 1., nan, -inf])          -> [-inf, 1., inf, nan]
//!   np.unique([1., nan, -nan, 2.])                -> [1., 2., nan]  (one nan)
//!   np.unique([5, 2, 8, 1, 2])                    -> [1, 2, 5, 8]   (int unchanged)

use ferray_core::{Array, Ix1};
use ferray_stats::{unique, unique_counts, unique_values};

#[test]
fn unique_nan_collapsed_sorted_last_f64() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, f64::NAN, 1.0, f64::NAN, 1.0])
        .unwrap();
    let v = unique_values(&a).unwrap();
    let d: Vec<f64> = v.iter().copied().collect();
    // np.unique([3.,nan,1.,nan,1.]) -> [1., 3., nan]  (one NaN, sorted last)
    assert_eq!(d.len(), 3);
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 3.0);
    assert!(d[2].is_nan());
}

#[test]
fn unique_nan_single_collapse() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 1.0, f64::NAN]).unwrap();
    let v = unique_values(&a).unwrap();
    let d: Vec<f64> = v.iter().copied().collect();
    // np.unique([1.,nan,1.,nan]) -> [1., nan]
    assert_eq!(d.len(), 2);
    assert_eq!(d[0], 1.0);
    assert!(d[1].is_nan());
}

#[test]
fn unique_counts_nan_collapsed() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, f64::NAN, 1.0, f64::NAN, 1.0])
        .unwrap();
    let (vals, counts) = unique_counts(&a).unwrap();
    let v: Vec<f64> = vals.iter().copied().collect();
    let c: Vec<u64> = counts.iter().copied().collect();
    // np.unique([3.,nan,1.,nan,1.], return_counts=True) -> ([1.,3.,nan],[2,1,2])
    assert_eq!(v.len(), 3);
    assert_eq!(v[0], 1.0);
    assert_eq!(v[1], 3.0);
    assert!(v[2].is_nan());
    assert_eq!(c, vec![2, 1, 2]);
}

#[test]
fn unique_inf_orders_before_nan() {
    let a = Array::<f64, Ix1>::from_vec(
        Ix1::new([5]),
        vec![f64::INFINITY, f64::NAN, 1.0, f64::NAN, f64::NEG_INFINITY],
    )
    .unwrap();
    let v = unique_values(&a).unwrap();
    let d: Vec<f64> = v.iter().copied().collect();
    // np.unique([inf, nan, 1., nan, -inf]) -> [-inf, 1., inf, nan]
    assert_eq!(d.len(), 4);
    assert_eq!(d[0], f64::NEG_INFINITY);
    assert_eq!(d[1], 1.0);
    assert_eq!(d[2], f64::INFINITY);
    assert!(d[3].is_nan());
}

#[test]
fn unique_negative_nan_collapses() {
    let a =
        Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, -f64::NAN, 2.0]).unwrap();
    let v = unique_values(&a).unwrap();
    let d: Vec<f64> = v.iter().copied().collect();
    // np.unique([1., nan, -nan, 2.]) -> [1., 2., nan]  (both NaN signs collapse to one)
    assert_eq!(d.len(), 3);
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 2.0);
    assert!(d[2].is_nan());
}

#[test]
fn unique_f32_nan_collapsed_sorted_last() {
    let a = Array::<f32, Ix1>::from_vec(Ix1::new([5]), vec![3.0, f32::NAN, 1.0, f32::NAN, 1.0])
        .unwrap();
    let v = unique_values(&a).unwrap();
    let d: Vec<f32> = v.iter().copied().collect();
    assert_eq!(d.len(), 3);
    assert_eq!(d[0], 1.0);
    assert_eq!(d[1], 3.0);
    assert!(d[2].is_nan());
}

#[test]
fn unique_integer_unregressed() {
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![5, 2, 8, 1, 2]).unwrap();
    let v = unique_values(&a).unwrap();
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1, 2, 5, 8]);
}

#[test]
fn unique_inverse_with_nan_reconstructs() {
    // np.unique([3.,nan,1.,nan,1.], return_inverse=True):
    //   values = [1.,3.,nan]; inverse = [1,2,0,2,0]  (values[inverse] == input modulo NaN)
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, f64::NAN, 1.0, f64::NAN, 1.0])
        .unwrap();
    let r = unique(&a, false, true, false).unwrap();
    let inv: Vec<u64> = r.inverse.unwrap().iter().copied().collect();
    assert_eq!(inv, vec![1, 2, 0, 2, 0]);
}
