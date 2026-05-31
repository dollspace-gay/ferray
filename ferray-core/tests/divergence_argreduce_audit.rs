//! Re-audit pins for ferray-core argmin / argmax / integer-mean→f64
//! (builder commit f62cb4b0, #784). These are the cases the builder's
//! `divergence_argreduce.rs` did NOT cover.
//!
//! Expected values come from the live numpy 2.4.5 oracle (R-CHAR-3 — never
//! literal-copied from the ferray side). Each `EXPECT` comment records the
//! exact `python3 -c "import numpy as np; ..."` invocation and printed result.
//!
//! NumPy contract:
//!   - argmax/argmin: first-occurrence on ties (numpy/_core/fromnumeric.py:
//!     1261-1262, :1361-1362), NaN-first — when ANY NaN is present the index of
//!     the FIRST NaN is returned, regardless of argmax vs argmin. All-NaN input
//!     therefore yields index 0. Result dtype `intp` (`int64`).
//!   - integer/bool mean → float64, value AND dtype (numpy/_core/_methods.py:
//!     124-127), including mean_axis lanes.

use ferray_core::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2, Ix3};

fn f64_1d(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

// ---------------------------------------------------------------------------
// all-NaN array — both directions. The builder pinned NaN-at-front and
// NaN-interleaved-with-numbers, but never an array that is ALL NaN.
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_all_nan_returns_zero() {
    // EXPECT: np.argmax([nan,nan]) == 0 ; np.argmin([nan,nan]) == 0
    let a = f64_1d(vec![f64::NAN, f64::NAN]);
    assert_eq!(a.argmax(), Some(0));
    assert_eq!(a.argmin(), Some(0));
    // three-element all-NaN, still first NaN (index 0).
    // EXPECT: np.argmax([nan,nan,nan]) == 0 ; np.argmin([nan,nan,nan]) == 0
    let b = f64_1d(vec![f64::NAN, f64::NAN, f64::NAN]);
    assert_eq!(b.argmax(), Some(0));
    assert_eq!(b.argmin(), Some(0));
}

// ---------------------------------------------------------------------------
// argmin NaN-first when the NaN is NOT the global minimum direction. Builder
// tested argmin NaN at idx 1 for [1,nan,3] but not the standalone min case
// where the numeric minimum precedes the NaN.
// ---------------------------------------------------------------------------

#[test]
fn argmin_nan_first_overrides_numeric_min() {
    // EXPECT: np.argmin([nan,5.0,2.0]) == 0 (NaN-first wins over numeric min 2.0)
    let a = f64_1d(vec![f64::NAN, 5.0, 2.0]);
    assert_eq!(a.argmin(), Some(0));
    // EXPECT: np.argmax([5.0, nan, 2.0]) == 1
    let b = f64_1d(vec![5.0, f64::NAN, 2.0]);
    assert_eq!(b.argmax(), Some(1));
}

// ---------------------------------------------------------------------------
// -0.0 vs 0.0 — numpy treats them equal, so first-occurrence wins. Builder
// never tested signed zero.
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_signed_zero_first_occurrence() {
    // EXPECT: np.argmin([-0.0,0.0]) == 0 ; np.argmax([0.0,-0.0]) == 0
    assert_eq!(f64_1d(vec![-0.0, 0.0]).argmin(), Some(0));
    assert_eq!(f64_1d(vec![0.0, -0.0]).argmax(), Some(0));
    // EXPECT: np.argmax([-0.0,0.0]) == 0 ; np.argmin([0.0,-0.0]) == 0
    assert_eq!(f64_1d(vec![-0.0, 0.0]).argmax(), Some(0));
    assert_eq!(f64_1d(vec![0.0, -0.0]).argmin(), Some(0));
}

// ---------------------------------------------------------------------------
// all-equal ties — first occurrence both directions. Builder tested mixed-tie
// arrays but not a fully-uniform array.
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_all_equal_first_occurrence() {
    // EXPECT: np.argmin([1,1,1]) == 0 ; np.argmax([5,5,5]) == 0
    let a = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![1, 1, 1]).unwrap();
    assert_eq!(a.argmin(), Some(0));
    assert_eq!(a.argmax(), Some(0));
    let b = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![5, 5, 5]).unwrap();
    assert_eq!(b.argmin(), Some(0));
    assert_eq!(b.argmax(), Some(0));
}

// ---------------------------------------------------------------------------
// axis form with a NaN inside a lane — both axes, both directions. Builder's
// only axis test was all-integer (no NaN).
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_axis_nan_in_lane() {
    // a = [[1.0, nan],[nan, 2.0]]
    // EXPECT: np.argmax(a,axis=0)==[1,0]; axis=1==[1,0]
    //         np.argmin(a,axis=0)==[1,0]; axis=1==[1,0]
    let a =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, f64::NAN, f64::NAN, 2.0]).unwrap();

    assert_eq!(
        a.argmax_axis(Axis(0))
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(
        a.argmax_axis(Axis(1))
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(
        a.argmin_axis(Axis(0))
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(
        a.argmin_axis(Axis(1))
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
}

// ---------------------------------------------------------------------------
// 3-D axis form — shape (reduced axis removed) and per-lane indices for the
// middle axis. Builder only exercised 2-D.
// ---------------------------------------------------------------------------

#[test]
fn argmax_axis_3d_shape_and_indices() {
    // a shape (2,3,2):
    //   [[[1,2],[3,4],[5,6]],[[7,8],[9,1],[2,3]]]
    // EXPECT: np.argmax(a,axis=1)==[[2,2],[1,0]] shape (2,2)
    //         np.argmax(a,axis=2)==[[1,1,1],[1,0,1]] shape (2,3)
    let a = Array::<i64, Ix3>::from_vec(
        Ix3::new([2, 3, 2]),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3],
    )
    .unwrap();

    let m1 = a.argmax_axis(Axis(1)).unwrap();
    assert_eq!(m1.shape(), &[2, 2]);
    assert_eq!(m1.iter().copied().collect::<Vec<_>>(), vec![2, 2, 1, 0]);

    let m2 = a.argmax_axis(Axis(2)).unwrap();
    assert_eq!(m2.shape(), &[2, 3]);
    assert_eq!(
        m2.iter().copied().collect::<Vec<_>>(),
        vec![1, 1, 1, 1, 0, 1]
    );
}

// ---------------------------------------------------------------------------
// negative numbers — sanity that comparison direction is right.
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_negative_values() {
    // EXPECT: np.argmax([-3,-1,-7]) == 1 ; np.argmin([-3,-1,-7]) == 2
    let a = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![-3, -1, -7]).unwrap();
    assert_eq!(a.argmax(), Some(1));
    assert_eq!(a.argmin(), Some(2));
}

// ---------------------------------------------------------------------------
// single-element array.
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_single_element() {
    // EXPECT: np.argmax([7]) == 0 ; np.argmin([7]) == 0
    let a = Array::<i64, Ix1>::from_vec(Ix1::new([1]), vec![7]).unwrap();
    assert_eq!(a.argmax(), Some(0));
    assert_eq!(a.argmin(), Some(0));
}

// ---------------------------------------------------------------------------
// mean_axis dtype for narrow int (i8) and bool — value must be f64 and the
// arithmetic must be correct. Builder's mean_axis test used i32 only.
// ---------------------------------------------------------------------------

#[test]
fn mean_axis_i8_returns_f64_values() {
    // a = [[1,2],[3,4]] int8
    // EXPECT: np.mean(a,axis=0) == [2.0,3.0] dtype float64
    let a = Array::<i8, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
    let m: Array<f64, _> = a.mean_axis(Axis(0)).unwrap();
    assert_eq!(
        m.iter().copied().collect::<Vec<_>>(),
        vec![2.0_f64, 3.0_f64]
    );
}

#[test]
fn mean_axis_bool_returns_f64_values() {
    // a = [[True,False],[True,True]]
    // EXPECT: np.mean(a,axis=0) == [1.0, 0.5] dtype float64
    let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 2]), vec![true, false, true, true]).unwrap();
    let m: Array<f64, _> = a.mean_axis(Axis(0)).unwrap();
    assert_eq!(
        m.iter().copied().collect::<Vec<_>>(),
        vec![1.0_f64, 0.5_f64]
    );
}

// ---------------------------------------------------------------------------
// mean value correctness for plain integer flattened (2.5 — the .5 proves the
// f64 division, not integer truncation).
// ---------------------------------------------------------------------------

#[test]
fn mean_int_value_is_fractional_f64() {
    // EXPECT: np.mean(np.array([1,2,3,4],np.int32)) == 2.5
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
    let m: f64 = a.mean().unwrap();
    assert_eq!(m, 2.5_f64);
    // negative ints. EXPECT: np.mean([-1,-2,-3,-4]) == -2.5
    let b = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![-1, -2, -3, -4]).unwrap();
    let m2: f64 = b.mean().unwrap();
    assert_eq!(m2, -2.5_f64);
}

// ---------------------------------------------------------------------------
// NaN at the END of the array — NaN-first must still win even when it is the
// last element scanned (it fails partial_cmp so it can never be beaten).
// ---------------------------------------------------------------------------

#[test]
fn argmax_argmin_nan_at_end() {
    // EXPECT: np.argmin([2.0,5.0,nan]) == 2 ; np.argmax([2.0,5.0,nan]) == 2
    let a = f64_1d(vec![2.0, 5.0, f64::NAN]);
    assert_eq!(a.argmin(), Some(2));
    assert_eq!(a.argmax(), Some(2));
}

// ---------------------------------------------------------------------------
// mean_axis along axis=1 (reduce the trailing axis) — f64 dtype + values.
// ---------------------------------------------------------------------------

#[test]
fn mean_axis1_int_returns_f64() {
    // a = [[1,2,3],[4,5,6]] int32
    // EXPECT: np.mean(a,axis=1) == [2.0,5.0] dtype float64
    let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let m: Array<f64, _> = a.mean_axis(Axis(1)).unwrap();
    assert_eq!(
        m.iter().copied().collect::<Vec<_>>(),
        vec![2.0_f64, 5.0_f64]
    );
}
