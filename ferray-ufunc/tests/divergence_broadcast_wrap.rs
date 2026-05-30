//! Divergence pins: `add_broadcast` / `subtract_broadcast` /
//! `multiply_broadcast` on integer overflow vs NumPy 2.4.5.
//!
//! NumPy's `add`/`subtract`/`multiply` ufuncs register fixed-width integer
//! loops (`numpy/_core/code_generators/generate_umath.py:355-403`,
//! `TD(no_bool_times_obj, dispatch=[..., ('loops_autovec', ints)])`). Those
//! integer loops are modular (two's-complement wrap), never overflow-checked:
//!
//!   np.add(np.array([100, 100], np.int8), np.array([100], np.int8)) == [-56, -56]
//!   np.subtract(np.array([-100,-100], np.int8), np.array([100], np.int8)) == [56, 56]
//!   np.multiply(np.array([100, 100], np.int8), np.array([100], np.int8)) == [16, 16]
//!   np.add(np.array([200, 200], np.uint8), np.array([100], np.uint8)) == [44, 44]
//!
//! Every expected value below is from a live `numpy` 2.4.5 call (R-CHAR-3),
//! reproduced in each test's doc comment. The same-shape `add`/`subtract`/
//! `multiply` were fixed in #88 to wrap via the `WrappingArith` trait; these
//! broadcasting siblings used raw `x + y` / `x - y` / `x * y`, which PANIC in
//! debug builds on integer overflow instead of wrapping like NumPy.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

fn i8_arr(data: Vec<i8>) -> Array<i8, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

fn u8_arr(data: Vec<u8>) -> Array<u8, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

/// `np.add(np.array([100,100],np.int8), np.array([100],np.int8))` == `[-56,-56]`.
/// int8 100 + 100 = 200, wraps to -56 (mod 256). Broadcasts [2] against [1].
#[test]
fn add_broadcast_int8_wraps_like_numpy() {
    let a = i8_arr(vec![100, 100]);
    let b = i8_arr(vec![100]);
    let result = ferray_ufunc::add_broadcast(&a, &b).unwrap();
    assert_eq!(result.as_slice().unwrap(), &[-56i8, -56]);
}

/// `np.subtract(np.array([-100,-100],np.int8), np.array([100],np.int8))` == `[56,56]`.
/// int8 -100 - 100 = -200, wraps to 56 (mod 256). Broadcasts [2] against [1].
#[test]
fn subtract_broadcast_int8_wraps_like_numpy() {
    let a = i8_arr(vec![-100, -100]);
    let b = i8_arr(vec![100]);
    let result = ferray_ufunc::subtract_broadcast(&a, &b).unwrap();
    assert_eq!(result.as_slice().unwrap(), &[56i8, 56]);
}

/// `np.multiply(np.array([100,100],np.int8), np.array([100],np.int8))` == `[16,16]`.
/// int8 100 * 100 = 10000, wraps to 16 (mod 256). Broadcasts [2] against [1].
#[test]
fn multiply_broadcast_int8_wraps_like_numpy() {
    let a = i8_arr(vec![100, 100]);
    let b = i8_arr(vec![100]);
    let result = ferray_ufunc::multiply_broadcast(&a, &b).unwrap();
    assert_eq!(result.as_slice().unwrap(), &[16i8, 16]);
}

/// `np.add(np.array([200,200],np.uint8), np.array([100],np.uint8))` == `[44,44]`.
/// uint8 200 + 100 = 300, wraps to 44 (mod 256). Broadcasts [2] against [1].
#[test]
fn add_broadcast_uint8_wraps_like_numpy() {
    let a = u8_arr(vec![200, 200]);
    let b = u8_arr(vec![100]);
    let result = ferray_ufunc::add_broadcast(&a, &b).unwrap();
    assert_eq!(result.as_slice().unwrap(), &[44u8, 44]);
}

/// Floats are unaffected by the wrap: `add_broadcast` on f64 stays ordinary
/// IEEE addition (`WrappingArith` for floats is plain `+`).
/// `np.add(np.array([1.5,2.5]), np.array([0.5]))` == `[2.0, 3.0]`.
#[test]
fn add_broadcast_f64_unchanged() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.5, 2.5]).unwrap();
    let b = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![0.5]).unwrap();
    let result = ferray_ufunc::add_broadcast(&a, &b).unwrap();
    assert_eq!(result.as_slice().unwrap(), &[2.0f64, 3.0]);
}
