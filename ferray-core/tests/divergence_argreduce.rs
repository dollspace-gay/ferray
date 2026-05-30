//! Divergence pins for ferray-core argmin / argmax / integer-mean→f64
//! (REQ-40, REQ-41, REQ-42).
//!
//! Expected values are taken from the live numpy 2.4.5 oracle (R-CHAR-3 — never
//! literal-copied from the ferray side). Each `EXPECT` comment records the exact
//! `python3 -c "import numpy as np; ..."` invocation and its printed result, so
//! the constant is traceable to numpy, not to ferray.
//!
//! NumPy contract:
//!   - argmax/argmin: first-occurrence on ties (numpy/_core/fromnumeric.py:
//!     1261-1262, :1361-1362), NaN-first (any NaN → index of first NaN),
//!     result dtype `intp` (`i64` here). Empty flattened → numpy raises
//!     ValueError; ferray returns `None` (the Option analog `min`/`max` use,
//!     mapped to ValueError at the ferray-python boundary).
//!   - integer/bool mean → f64 (numpy/_core/_methods.py:124-127).

use ferray_core::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2};

fn i64_1d(data: Vec<i64>) -> Array<i64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

fn f64_1d(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

// ---------------------------------------------------------------------------
// REQ-40 / REQ-41: argmax / argmin — flattened, first-occurrence, NaN-first
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_basic() {
    // EXPECT: np.argmax([3,1,4,1,5,9,2]) == 5 ; np.argmin([3,1,4,1,5,9,2]) == 1
    let a = i64_1d(vec![3, 1, 4, 1, 5, 9, 2]);
    assert_eq!(a.argmax(), Some(5));
    assert_eq!(a.argmin(), Some(1));
}

#[test]
fn argmin_first_occurrence_on_ties() {
    // EXPECT: np.argmin([3,1,1,2]) == 1  (first of the tied minimum `1`s)
    let a = i64_1d(vec![3, 1, 1, 2]);
    assert_eq!(a.argmin(), Some(1));
}

#[test]
fn argmax_first_occurrence_on_ties() {
    // EXPECT: np.argmax([1,3,3,2]) == 1  (first of the tied maximum `3`s)
    let a = i64_1d(vec![1, 3, 3, 2]);
    assert_eq!(a.argmax(), Some(1));
}

#[test]
fn argmax_argmin_nan_first() {
    // EXPECT: np.argmax([1.0, nan, 3.0, nan]) == 1
    //         np.argmin([1.0, nan, 3.0])      == 1
    let a = f64_1d(vec![1.0, f64::NAN, 3.0, f64::NAN]);
    assert_eq!(a.argmax(), Some(1));
    let b = f64_1d(vec![1.0, f64::NAN, 3.0]);
    assert_eq!(b.argmin(), Some(1));
}

#[test]
fn argmax_nan_at_front() {
    // EXPECT: np.argmax([nan, 5.0, 2.0]) == 0  (NaN-first wins immediately)
    let a = f64_1d(vec![f64::NAN, 5.0, 2.0]);
    assert_eq!(a.argmax(), Some(0));
    assert_eq!(a.argmin(), Some(0));
}

#[test]
fn argmax_argmin_empty_returns_none() {
    // numpy raises ValueError ("attempt to get argmax of an empty sequence");
    // ferray's Option analog returns None (mapped to ValueError at the python
    // boundary, mirroring how min/max already do it).
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    assert_eq!(a.argmax(), None);
    assert_eq!(a.argmin(), None);
}

#[test]
fn argmax_argmin_axis_2d() {
    // a = [[1,5,3],[4,2,6]]
    // EXPECT: np.argmax(a, axis=0) == [1,0,1] ; axis=1 == [1,2]
    //         np.argmin(a, axis=0) == [0,1,0] ; axis=1 == [0,1]
    let a = Array::<i64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 5, 3, 4, 2, 6]).unwrap();

    let mx0 = a.argmax_axis(Axis(0)).unwrap();
    assert_eq!(mx0.shape(), &[3]);
    assert_eq!(mx0.iter().copied().collect::<Vec<_>>(), vec![1, 0, 1]);

    let mx1 = a.argmax_axis(Axis(1)).unwrap();
    assert_eq!(mx1.iter().copied().collect::<Vec<_>>(), vec![1, 2]);

    let mn0 = a.argmin_axis(Axis(0)).unwrap();
    assert_eq!(mn0.iter().copied().collect::<Vec<_>>(), vec![0, 1, 0]);

    let mn1 = a.argmin_axis(Axis(1)).unwrap();
    assert_eq!(mn1.iter().copied().collect::<Vec<_>>(), vec![0, 1]);
}

#[test]
fn argmax_axis_out_of_bounds_errors() {
    let a = Array::<i64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 5, 3, 4, 2, 6]).unwrap();
    assert!(a.argmax_axis(Axis(2)).is_err());
    assert!(a.argmin_axis(Axis(5)).is_err());
}

// ---------------------------------------------------------------------------
// REQ-42: integer / bool mean → f64
// ---------------------------------------------------------------------------

#[test]
fn mean_i32_returns_f64() {
    // EXPECT: np.mean(np.array([1,2,3], np.int32)) == 2.0, dtype float64
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    let m: f64 = a.mean().unwrap();
    assert_eq!(m, 2.0_f64);
}

#[test]
fn mean_u8_returns_f64() {
    // EXPECT: np.mean(np.array([2,4,6,8], np.uint8)) == 5.0 (float64)
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([4]), vec![2, 4, 6, 8]).unwrap();
    let m: f64 = a.mean().unwrap();
    assert_eq!(m, 5.0_f64);
}

#[test]
fn mean_bool_returns_f64() {
    // EXPECT: np.mean([True,False,True]) == 0.6666666666666666 (float64)
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
    let m: f64 = a.mean().unwrap();
    assert!((m - 2.0_f64 / 3.0_f64).abs() < 1e-15);
}

#[test]
fn mean_f32_stays_f32() {
    // EXPECT: np.mean(np.array([1,2,3], np.float32)) == 2.0, dtype float32
    let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    let m: f32 = a.mean().unwrap();
    assert_eq!(m, 2.0_f32);
}

#[test]
fn mean_axis_int_returns_f64() {
    // a = [[1,2,3],[4,5,6]] int32
    // EXPECT: np.mean(a, axis=0) == [2.5,3.5,4.5], dtype float64
    let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let m: Array<f64, _> = a.mean_axis(Axis(0)).unwrap();
    assert_eq!(m.iter().copied().collect::<Vec<_>>(), vec![2.5, 3.5, 4.5]);
}

#[test]
fn mean_empty_returns_none() {
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    assert_eq!(a.mean(), None);
}
