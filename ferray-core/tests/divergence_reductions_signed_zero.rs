//! Divergence pins: ferray's `Array::min`/`max` (and the axis forms) diverge
//! from NumPy on the signed-zero tie case.
//!
//! NumPy's `np.max`/`np.min` are `maximum.reduce`/`minimum.reduce`. The scalar
//! `np.maximum(a, b)` returns `b` when `a == b` (the *later* operand wins ties),
//! and since `+0.0 == -0.0`, the reduction keeps the LAST signed zero in the
//! tie chain. ferray's `reduce_step` (ferray-core/src/array/reductions.rs:313)
//! keeps the accumulator (the FIRST element) on `Ordering::Equal`, so it keeps
//! the FIRST signed zero.
//!
//! Live numpy 2.4.4 oracle (`ferray-python/.venv`):
//!   np.max([0.0, -0.0]) == -0.0   (signbit True)   ferray -> 0.0
//!   np.max([-0.0, 0.0]) ==  0.0   (signbit False)  ferray -> -0.0
//!   np.min([0.0, -0.0]) == -0.0   (signbit True)   ferray -> 0.0
//!   np.min([-0.0, 0.0]) ==  0.0   (signbit False)  ferray -> -0.0
//!
//! Upstream: `numpy/_core/fromnumeric.py` `amax`/`amin` -> `umr_maximum`/
//! `umr_minimum` (`numpy/_core/_methods.py:38-44`); the maximum/minimum ufunc
//! "last operand wins on equal" loop. The observable signed-zero sign bit
//! is the divergence.
//!
//! These are NOT marked `#[ignore]`: the sign bit of a reduction result is an
//! observable, NEP-defined property. Tracking: filed as a blocker.

use ferray_core::Array;
use ferray_core::dimension::{Axis, Ix1, Ix2};

/// Divergence: `Array::max` keeps the FIRST element on a signed-zero tie;
/// NumPy keeps the LAST. `np.max([0.0, -0.0])` is `-0.0` (negative zero),
/// ferray returns `+0.0`.
/// Upstream: numpy/_core/_methods.py:38 `umr_maximum` (maximum.reduce, last
/// operand wins ties). Oracle: numpy 2.4.4 `float(np.max([0.0,-0.0])) == -0.0`,
/// `math.copysign(1, x) == -1`.
#[test]
fn divergence_max_signed_zero_keeps_last() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, -0.0]).unwrap();
    let got = a.max().unwrap();
    // numpy returns -0.0 -> the sign bit must be set.
    assert!(
        got.is_sign_negative(),
        "max([0.0, -0.0]) must be -0.0 (numpy), got {got} sign_negative={}",
        got.is_sign_negative()
    );
}

/// Divergence: the other tie order. `np.max([-0.0, 0.0])` is `+0.0`; ferray
/// keeps the first element `-0.0`.
/// Oracle: numpy 2.4.4 `math.copysign(1, np.max([-0.0,0.0])) == +1`.
#[test]
fn divergence_max_signed_zero_other_order() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![-0.0, 0.0]).unwrap();
    let got = a.max().unwrap();
    assert!(
        got.is_sign_positive(),
        "max([-0.0, 0.0]) must be +0.0 (numpy), got sign_negative={}",
        got.is_sign_negative()
    );
}

/// Divergence: `Array::min` mirror. `np.min([0.0, -0.0])` is `-0.0`; ferray
/// keeps the first `+0.0`.
/// Oracle: numpy 2.4.4 `math.copysign(1, np.min([0.0,-0.0])) == -1`.
#[test]
fn divergence_min_signed_zero_keeps_last() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![0.0, -0.0]).unwrap();
    let got = a.min().unwrap();
    assert!(
        got.is_sign_negative(),
        "min([0.0, -0.0]) must be -0.0 (numpy), got sign_negative={}",
        got.is_sign_negative()
    );
}

/// Divergence: axis form. `np.max([[0.0,-0.0],[-0.0,0.0]], axis=0)` has
/// signbit `[True, False]` (column 0 keeps the last `-0.0`, column 1 keeps
/// the last `0.0`). ferray's `max_axis` keeps the first per lane.
/// Oracle: numpy 2.4.4 `np.signbit(np.max(a, axis=0)).tolist() == [True, False]`.
#[test]
fn divergence_max_axis_signed_zero() {
    // Row-major [[0.0, -0.0], [-0.0, 0.0]]
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![0.0, -0.0, -0.0, 0.0]).unwrap();
    let m = a.max_axis(Axis(0)).unwrap();
    let vals: Vec<f64> = m.iter().copied().collect();
    // Column 0: max(0.0 [row0], -0.0 [row1]) -> numpy keeps last -> -0.0
    // Column 1: max(-0.0 [row0], 0.0 [row1]) -> numpy keeps last -> +0.0
    assert!(
        vals[0].is_sign_negative() && vals[1].is_sign_positive(),
        "max(axis=0) signbits must be [neg, pos] (numpy), got [{}, {}]",
        vals[0].is_sign_negative(),
        vals[1].is_sign_negative()
    );
}
