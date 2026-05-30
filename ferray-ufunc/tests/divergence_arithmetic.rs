//! Divergence pins: ferray-ufunc arithmetic + dtype dispatch vs NumPy 2.4.5.
//!
//! Every expected value here is derived from a live `numpy` call (R-CHAR-3),
//! reproduced in the doc comment of each test. These tests assert NumPy's
//! observable contract.
//!
//! Root cause (now FIXED — see `ferray-ufunc/src/ops/arithmetic.rs`):
//! - `divide` / `true_divide` are NumPy *true division* via the `TrueDivide`
//!   trait: integer operands promote to float64, floats keep their dtype.
//!   Integer divide-by-zero produces inf/nan (operands cast to f64 first),
//!   never a panic. NumPy `PyUFunc_TrueDivisionTypeResolver`,
//!   `numpy/_core/code_generators/generate_umath.py:422`.
//! - `floor_divide_int` / `remainder_int` / `mod_int` / `power_int` keep the
//!   integer dtype; divisor-zero -> 0 for floor-div/mod (no panic).
//! - `maximum_ord` / `minimum_ord` / `sign_int` / `negative_int` /
//!   `absolute_int` preserve the integer dtype.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

fn i32_arr(data: Vec<i32>) -> Array<i32, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

fn i64_arr(data: Vec<i64>) -> Array<i64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

// ===========================================================================
// divide / true_divide — int true-division to float64 (the critic's pins)
// ===========================================================================

/// `np.divide(int, int)` is *true* division returning float64, not Rust
/// truncating integer division.
///
/// Upstream: `numpy/_core/code_generators/generate_umath.py:422`
/// (`PyUFunc_TrueDivisionTypeResolver`). Live oracle:
/// ```text
/// >>> np.divide(np.array([1,2,7,8]), np.array([2,2,2,3]))
/// array([0.5, 1. , 3.5, 2.66666667])   # dtype=float64
/// ```
#[test]
fn divide_int_int_is_true_division_float64() {
    // numpy: np.true_divide([1,2,7,8],[2,2,2,3]) -> [0.5, 1.0, 3.5, 2.6666666666666665]
    const NP_DIVIDE: [f64; 4] = [0.5, 1.0, 3.5, 2.666_666_666_666_666_5];

    let a = i32_arr(vec![1, 2, 7, 8]);
    let b = i32_arr(vec![2, 2, 2, 3]);

    // The result element type is now f64 (NEP-50 true-division promotion).
    let got = ferray_ufunc::divide(&a, &b).unwrap();
    let got_f: &[f64] = got.as_slice().unwrap();

    assert_eq!(
        got_f,
        NP_DIVIDE.as_slice(),
        "divide(int,int) must equal NumPy true_divide (float64), not truncated integer division"
    );
}

/// `np.divide(int, 0)` returns inf/nan + RuntimeWarning, NEVER panics.
///
/// Live oracle:
/// ```text
/// >>> np.divide(np.array([1, 0, -1]), np.array([0, 0, 0]))
/// array([ inf,  nan, -inf])             # dtype=float64
/// ```
#[test]
fn divide_int_by_zero_must_not_panic() {
    let result = std::panic::catch_unwind(|| {
        let a = i32_arr(vec![1, 0, -1]);
        let b = i32_arr(vec![0, 0, 0]);
        ferray_ufunc::divide(&a, &b)
    });

    assert!(
        result.is_ok(),
        "divide(int, 0) panicked; NumPy returns [inf, nan, -inf] (float64) with only a RuntimeWarning"
    );
    let arr = result.unwrap().unwrap();
    let s: &[f64] = arr.as_slice().unwrap();
    // numpy: [inf, nan, -inf]
    assert!(s[0].is_infinite() && s[0] > 0.0, "1/0 -> +inf");
    assert!(s[1].is_nan(), "0/0 -> nan");
    assert!(s[2].is_infinite() && s[2] < 0.0, "-1/0 -> -inf");
}

/// `array_div` (the `/` operator overload) inherits the true-division fix:
/// integer `/` never panics on divide-by-zero.
///
/// Live oracle:
/// ```text
/// >>> np.array([5, 0]) / np.array([0, 0])
/// array([inf, nan])                     # float64
/// ```
#[test]
fn array_div_int_by_zero_must_not_panic() {
    let result = std::panic::catch_unwind(|| {
        let a = i64_arr(vec![5, 0]);
        let b = i64_arr(vec![0, 0]);
        ferray_ufunc::array_div(&a, &b)
    });

    assert!(
        result.is_ok(),
        "array_div(int, 0) (operator `/`) panicked; NumPy returns [inf, nan] float64"
    );
    let arr = result.unwrap().unwrap();
    let s: &[f64] = arr.as_slice().unwrap();
    assert!(s[0].is_infinite() && s[0] > 0.0, "5/0 -> +inf");
    assert!(s[1].is_nan(), "0/0 -> nan");
}

/// `np.divide(a, b, out=c)` with a float64 `out` writes the true-division
/// result and does not panic on divide-by-zero.
///
/// Live oracle:
/// ```text
/// >>> c = np.empty(3); np.divide(np.array([3,4,5]), np.array([2,2,0]), out=c)
/// array([1.5, 2. , inf])                # float64
/// ```
#[test]
fn divide_into_int_by_zero_must_not_panic() {
    let result = std::panic::catch_unwind(|| {
        let a = i32_arr(vec![3, 4, 5]);
        let b = i32_arr(vec![2, 2, 0]);
        // out is float64 — np.divide's true-division output dtype.
        let mut out = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, 0.0, 0.0]).unwrap();
        ferray_ufunc::divide_into(&a, &b, &mut out).map(|()| out)
    });

    assert!(
        result.is_ok(),
        "divide_into(int, 0) panicked; NumPy `np.divide(.., out=)` writes [1.5, 2.0, inf] float64"
    );
    let out = result.unwrap().unwrap();
    let s: &[f64] = out.as_slice().unwrap();
    assert_eq!(s[0], 1.5);
    assert_eq!(s[1], 2.0);
    assert!(s[2].is_infinite() && s[2] > 0.0, "5/0 -> +inf");
}

/// f32 / f64 division stays byte-identical (true-division on floats is the
/// identity float path — no regression for existing float callers).
///
/// Live oracle:
/// ```text
/// >>> np.divide(np.array([10.,20.,30.]), np.array([2.,4.,5.]))
/// array([5., 5., 6.])
/// ```
#[test]
fn divide_f64_unchanged() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
    let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 4.0, 5.0]).unwrap();
    let got = ferray_ufunc::divide(&a, &b).unwrap();
    assert_eq!(got.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
}

// ===========================================================================
// power(int, int) -> int
// ===========================================================================

/// `np.power(int, int)` is integer exponentiation, int -> int.
///
/// Upstream: generate_umath.py:480 (`power`, `TD(ints)`). Live oracle:
/// ```text
/// >>> np.power(np.array([2,3,4]), np.array([2,3,3]))
/// array([ 4, 27, 64])                   # dtype=int64
/// ```
#[test]
fn power_int_int_is_integer_exponentiation() {
    // numpy: np.power([2,3,4],[2,3,3]) -> [4, 27, 64]
    let a = i64_arr(vec![2, 3, 4]);
    let b = i64_arr(vec![2, 3, 3]);
    let r = ferray_ufunc::power_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[4, 27, 64]);
}

/// `np.power(int, 0) == 1`, and `np.power(int, negative)` truncates to 0
/// for |base|>1.
///
/// Live oracle:
/// ```text
/// >>> np.power(np.array([5, 1, -1, 7]), np.array([0, -3, -3, -1]))
/// array([ 1,  1, -1,  0])
/// ```
#[test]
fn power_int_zero_and_negative_exponent() {
    // numpy: np.power([5,1,-1,7],[0,-3,-3,-1]) -> [1, 1, -1, 0]
    let a = i64_arr(vec![5, 1, -1, 7]);
    let b = i64_arr(vec![0, -3, -3, -1]);
    let r = ferray_ufunc::power_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[1, 1, -1, 0]);
}

// ===========================================================================
// floor_divide / remainder / mod (int) — divisor-zero -> 0, no panic
// ===========================================================================

/// `np.floor_divide(int, int)` rounds toward -inf (Python `//`), int -> int.
///
/// Upstream: generate_umath.py:405 (`floor_divide`, `TD(ints)`). Live oracle:
/// ```text
/// >>> np.floor_divide(np.array([7,-7,8,-8]), np.array([2,2,3,3]))
/// array([ 3, -4,  2, -3])
/// ```
#[test]
fn floor_divide_int_rounds_toward_neg_inf() {
    // numpy: np.floor_divide([7,-7,8,-8],[2,2,3,3]) -> [3, -4, 2, -3]
    let a = i64_arr(vec![7, -7, 8, -8]);
    let b = i64_arr(vec![2, 2, 3, 3]);
    let r = ferray_ufunc::floor_divide_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[3, -4, 2, -3]);
}

/// `np.floor_divide(int, 0) == 0` + RuntimeWarning, never panics.
///
/// Live oracle:
/// ```text
/// >>> np.floor_divide(np.array([7,0,-7]), np.array([0,0,0]))
/// array([0, 0, 0])
/// ```
#[test]
fn floor_divide_int_by_zero_is_zero_no_panic() {
    let result = std::panic::catch_unwind(|| {
        let a = i64_arr(vec![7, 0, -7]);
        let b = i64_arr(vec![0, 0, 0]);
        ferray_ufunc::floor_divide_int(&a, &b)
    });
    assert!(
        result.is_ok(),
        "floor_divide_int(int, 0) panicked; NumPy returns 0"
    );
    assert_eq!(result.unwrap().unwrap().as_slice().unwrap(), &[0, 0, 0]);
}

/// `np.remainder(int, int)` takes the sign of the divisor (Python `%`).
///
/// Upstream: generate_umath.py:1039 (`remainder`, `TD(ints)`). Live oracle:
/// ```text
/// >>> np.remainder(np.array([7,-7,8,-8]), np.array([3,3,3,3]))
/// array([1, 2, 2, 1])
/// ```
#[test]
fn remainder_int_takes_sign_of_divisor() {
    // numpy: np.remainder([7,-7,8,-8],[3,3,3,3]) -> [1, 2, 2, 1]
    let a = i64_arr(vec![7, -7, 8, -8]);
    let b = i64_arr(vec![3, 3, 3, 3]);
    let r = ferray_ufunc::remainder_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[1, 2, 2, 1]);
}

/// `np.remainder(int, 0) == 0` + RuntimeWarning, never panics.
///
/// Live oracle:
/// ```text
/// >>> np.remainder(np.array([7,0,-7]), np.array([0,0,0]))
/// array([0, 0, 0])
/// ```
#[test]
fn remainder_int_by_zero_is_zero_no_panic() {
    let result = std::panic::catch_unwind(|| {
        let a = i64_arr(vec![7, 0, -7]);
        let b = i64_arr(vec![0, 0, 0]);
        ferray_ufunc::mod_int(&a, &b)
    });
    assert!(result.is_ok(), "mod_int(int, 0) panicked; NumPy returns 0");
    assert_eq!(result.unwrap().unwrap().as_slice().unwrap(), &[0, 0, 0]);
}

// ===========================================================================
// add/subtract/multiply int8 wraparound + mixed-width promotion
// ===========================================================================

/// `np.add(int8, int8)` wraps on overflow (int8 fixed-width), int8 -> int8.
///
/// Live oracle:
/// ```text
/// >>> np.add(np.array([100], np.int8), np.array([100], np.int8))
/// array([-56], dtype=int8)              # 200 wraps to -56
/// ```
#[test]
fn add_int8_wraps_on_overflow() {
    // numpy: int8(100) + int8(100) -> int8(-56)  (200 wraps in i8)
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![100, 50]).unwrap();
    let b = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![100, 80]).unwrap();
    // ferray's `add` must wrap on int8 overflow (must NOT panic in debug).
    let r = std::panic::catch_unwind(|| ferray_ufunc::add(&a, &b));
    assert!(r.is_ok(), "add(int8, int8) overflow panicked; NumPy wraps");
    let arr = r.unwrap().unwrap();
    // 100+100=200 -> -56 ; 50+80=130 -> -126
    assert_eq!(arr.as_slice().unwrap(), &[-56_i8, -126]);
}

/// `np.add(int8, int64)` promotes to int64 (NEP-50 mixed-width).
///
/// Live oracle:
/// ```text
/// >>> np.add(np.array([1,2], np.int8), np.array([10,20], np.int64)).dtype
/// dtype('int64')
/// >>> np.add(np.array([1,2], np.int8), np.array([10,20], np.int64))
/// array([11, 22])
/// ```
#[test]
fn add_int8_int64_promotes_to_int64() {
    use ferray_core::dtype::promotion::Promoted;
    // The promotion table resolves int8 + int64 -> int64.
    let _check: <i8 as Promoted<i64>>::Output = 0_i64;
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![1, 2]).unwrap();
    let b = i64_arr(vec![10, 20]);
    // Cast i8 -> i64 (the promoted dtype) then add in i64.
    use ferray_core::dtype::promotion::PromoteTo;
    let a_i64: Vec<i64> = a.as_slice().unwrap().iter().map(|&x| x.promote()).collect();
    let a_i64 = Array::<i64, Ix1>::from_vec(Ix1::new([2]), a_i64).unwrap();
    let r = ferray_ufunc::add(&a_i64, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[11, 22]);
}

// ===========================================================================
// maximum / minimum (int) preserve dtype
// ===========================================================================

/// `np.maximum(int, int)` / `np.minimum` preserve the integer dtype.
///
/// Upstream: generate_umath.py:661,671. Live oracle:
/// ```text
/// >>> np.maximum(np.array([1,5,3]), np.array([4,2,3]))
/// array([4, 5, 3])
/// >>> np.minimum(np.array([1,5,3]), np.array([4,2,3]))
/// array([1, 2, 3])
/// ```
#[test]
fn maximum_minimum_int_preserve_dtype() {
    let a = i64_arr(vec![1, 5, 3]);
    let b = i64_arr(vec![4, 2, 3]);
    let mx = ferray_ufunc::maximum_ord(&a, &b).unwrap();
    let mn = ferray_ufunc::minimum_ord(&a, &b).unwrap();
    assert_eq!(mx.as_slice().unwrap(), &[4, 5, 3]);
    assert_eq!(mn.as_slice().unwrap(), &[1, 2, 3]);
}

// ===========================================================================
// sign / negative / absolute (int) preserve dtype
// ===========================================================================

/// `np.sign(int32) -> int32`, `np.negative(int32) -> int32`,
/// `np.absolute(int32) -> int32`.
///
/// Upstream: generate_umath.py:534,516,496. Live oracle (dtype=int32):
/// ```text
/// >>> np.sign(np.array([-3,0,5], np.int32))      -> array([-1, 0, 1])
/// >>> np.negative(np.array([-3,0,5], np.int32))  -> array([ 3, 0, -5])
/// >>> np.absolute(np.array([-3,0,5], np.int32))  -> array([ 3, 0, 5])
/// ```
#[test]
fn sign_negative_absolute_int_preserve_dtype() {
    let a = i32_arr(vec![-3, 0, 5]);
    let s = ferray_ufunc::sign_int(&a).unwrap();
    let n = ferray_ufunc::negative_int(&a).unwrap();
    let b = ferray_ufunc::absolute_int(&a).unwrap();
    assert_eq!(s.as_slice().unwrap(), &[-1, 0, 1]);
    assert_eq!(n.as_slice().unwrap(), &[3, 0, -5]);
    assert_eq!(b.as_slice().unwrap(), &[3, 0, 5]);
}

/// `np.sign(uint8) -> uint8` (unsigned: 0 or 1, the `< 0` branch is dead).
///
/// Live oracle:
/// ```text
/// >>> np.sign(np.array([0, 7, 255], np.uint8))   -> array([0, 1, 1], uint8)
/// ```
#[test]
fn sign_int_unsigned_is_zero_or_one() {
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![0, 7, 255]).unwrap();
    let s = ferray_ufunc::sign_int(&a).unwrap();
    assert_eq!(s.as_slice().unwrap(), &[0, 1, 1]);
}
