//! Re-audit of builder commit `a1bfe1cb` (#785): ferray-ufunc integer-dtype
//! arithmetic + NEP-50 promotion.
//!
//! Two parts:
//!   (A) VALIDATION tests — these PASS if the builder's same-shape integer
//!       ops are numpy-faithful. They re-derive expected values from a live
//!       numpy 2.4.5 oracle (reproduced in each doc comment; R-CHAR-3 — never
//!       literal-copied from the ferray side).
//!   (B) PIN tests — these FAIL against the current target. They pin a real
//!       divergence: `divide_broadcast` on integer dtypes (1) panics on a
//!       zero divisor and (2) truncates toward zero / keeps the integer dtype
//!       instead of returning float64, contradicting NumPy's
//!       `PyUFunc_TrueDivisionTypeResolver`
//!       (numpy/_core/code_generators/generate_umath.py:419-422).
//!
//! Every expected value below was produced by running the matching `np.*`
//! call under numpy 2.4.5 (the project's live oracle).

use ferray_core::Array;
use ferray_core::dimension::{Ix1, IxDyn};

fn i64_arr(data: Vec<i64>) -> Array<i64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

// ===========================================================================
// (A) VALIDATION — should PASS (builder is correct)
// ===========================================================================

/// VALIDATE: `np.divide(int, int)` is true division -> float64.
/// Live oracle: `np.divide([7,8],[2,3]) -> [3.5, 2.6666666666666665]` (float64).
/// numpy/_core/code_generators/generate_umath.py:419 `'divide'` uses
/// `'PyUFunc_TrueDivisionTypeResolver'`.
#[test]
fn validate_divide_int_true_division_float64() {
    const NP: [f64; 2] = [3.5, 2.666_666_666_666_666_5];
    let a = i64_arr(vec![7, 8]);
    let b = i64_arr(vec![2, 3]);
    let got = ferray_ufunc::divide(&a, &b).unwrap();
    assert_eq!(got.as_slice().unwrap(), NP.as_slice());
}

/// VALIDATE: `np.floor_divide(int, int)` floors toward -inf for ALL sign
/// combinations (NOT Rust truncation toward zero).
/// Live oracle:
/// `np.floor_divide([7,-7,7,-7],[3,3,-3,-3]) -> [ 2, -3, -3,  2]`.
/// generate_umath.py:404 `'floor_divide'`, `TD(ints, cfunc_alias='divide')`.
#[test]
fn validate_floor_divide_int_floors_all_sign_combos() {
    // numpy floors toward -inf; Rust `/` truncates toward zero, so the
    // builder's sign-correction is the load-bearing part being validated.
    const NP: [i64; 4] = [2, -3, -3, 2];
    let a = i64_arr(vec![7, -7, 7, -7]);
    let b = i64_arr(vec![3, 3, -3, -3]);
    let r = ferray_ufunc::floor_divide_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), NP.as_slice());
}

/// VALIDATE: `np.remainder(int, int)` takes the sign of the DIVISOR
/// (Python `%`), not the dividend (Rust `%`).
/// Live oracle:
/// `np.remainder([7,-7,7,-7],[3,3,-3,-3]) -> [ 1,  2, -2, -1]`.
/// generate_umath.py:1035 `'remainder'`, `'PyUFunc_RemainderTypeResolver'`.
#[test]
fn validate_remainder_int_takes_divisor_sign_all_combos() {
    // Rust `%` would give [1, -1, 1, -1]; numpy follows the divisor sign.
    const NP: [i64; 4] = [1, 2, -2, -1];
    let a = i64_arr(vec![7, -7, 7, -7]);
    let b = i64_arr(vec![3, 3, -3, -3]);
    let r = ferray_ufunc::remainder_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), NP.as_slice());
}

/// VALIDATE: `np.power(int, int)` is integer exponentiation, int -> int.
/// Live oracle: `np.power([2,3],[3,2]) -> [8, 9]` (int64).
/// generate_umath.py:480 `'power'`, `TD(ints)`.
#[test]
fn validate_power_int() {
    const NP: [i64; 2] = [8, 9];
    let a = i64_arr(vec![2, 3]);
    let b = i64_arr(vec![3, 2]);
    let r = ferray_ufunc::power_int(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), NP.as_slice());
}

/// VALIDATE: int8 add wraps on overflow (fixed-width modular arithmetic).
/// Live oracle: `np.add(int8(100), int8(100)) -> int8(-56)` (200 wraps).
#[test]
fn validate_add_int8_wraps() {
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([1]), vec![100]).unwrap();
    let b = Array::<i8, Ix1>::from_vec(Ix1::new([1]), vec![100]).unwrap();
    let r = ferray_ufunc::add(&a, &b).unwrap();
    assert_eq!(r.as_slice().unwrap(), &[-56_i8]);
}

/// VALIDATE: NEP-50 mixed-width int promotion int8+int64 -> int64 value+dtype.
/// Live oracle: `np.add(int8[1,2], int64[10,20]) -> int64[11, 22]`.
#[test]
fn validate_add_int8_int64_promotes_to_int64() {
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![1, 2]).unwrap();
    let b = i64_arr(vec![10, 20]);
    let c = ferray_ufunc::add_promoted(&a, &b).unwrap();
    // Compile-time dtype check: output element type is i64.
    let slice: &[i64] = c.as_slice().unwrap();
    assert_eq!(slice, &[11, 22]);
}

/// VALIDATE: NEP-50 int16+int32 -> int32; uint8+int8 -> int16.
/// Live oracle:
/// `np.add(int16[100,200], int32[1,2]) -> int32[101, 202]`
/// `np.add(uint8[1,2], int8[10,20]) -> int16[11, 22]`.
#[test]
fn validate_mixed_width_promotion_dtypes() {
    let a = Array::<i16, Ix1>::from_vec(Ix1::new([2]), vec![100, 200]).unwrap();
    let b = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![1, 2]).unwrap();
    let c = ferray_ufunc::add_promoted(&a, &b).unwrap();
    let c_slice: &[i32] = c.as_slice().unwrap();
    assert_eq!(c_slice, &[101, 202]);

    let u = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![1, 2]).unwrap();
    let s = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![10, 20]).unwrap();
    let d = ferray_ufunc::add_promoted(&u, &s).unwrap();
    let d_slice: &[i16] = d.as_slice().unwrap();
    assert_eq!(d_slice, &[11, 22]);
}

// ===========================================================================
// (B) PINS — should FAIL (real divergences)
// ===========================================================================

/// PIN: `divide_broadcast` on an integer dtype with a zero divisor PANICS,
/// but NumPy's broadcasting `np.divide` returns float64 [inf, nan] with only
/// a RuntimeWarning.
///
/// Divergence: `ferray_ufunc::divide_broadcast` (ops/arithmetic.rs
/// `divide_broadcast`) is generic over `T: Div<Output = T>` and applies the
/// raw kernel `|x, y| x / y`. For integer `T` this is Rust integer `/`, which
/// PANICS on a zero divisor (attempt to divide by zero) and otherwise returns
/// a truncated INTEGER — neither matches NumPy.
///
/// Upstream: numpy/_core/code_generators/generate_umath.py:419 `'divide'`
/// uses `'PyUFunc_TrueDivisionTypeResolver'`, so integer operands promote to
/// float64 and integer `x/0` yields inf/nan, never an error.
///
/// Live oracle:
/// ```text
/// >>> np.divide(np.array([5,0], np.int64), np.array([0], np.int64))
/// array([inf, nan])   # dtype=float64, only RuntimeWarning
/// ```
#[test]
fn pin_divide_broadcast_int_zero_divisor_panics() {
    let outcome = std::panic::catch_unwind(|| {
        let a = i64_arr(vec![5, 0]);
        let b = i64_arr(vec![0]); // length-1, broadcasts against length-2
        ferray_ufunc::divide_broadcast(&a, &b)
    });
    // NumPy does NOT panic here; it returns float64 [inf, nan]. The target
    // panics on integer divide-by-zero. This assertion pins the divergence:
    // it FAILS today (outcome is Err) and will PASS once divide_broadcast
    // is a true-division float path like `divide`.
    assert!(
        outcome.is_ok(),
        "divide_broadcast(int, 0) panicked; NumPy np.divide broadcasting \
         returns float64 [inf, nan] with only a RuntimeWarning \
         (generate_umath.py:419 PyUFunc_TrueDivisionTypeResolver)"
    );
}

/// PIN: even with a NON-zero integer divisor, `divide_broadcast` truncates
/// toward zero and keeps the integer dtype, but NumPy returns float64 with
/// the exact quotient.
///
/// Divergence: `ferray_ufunc::divide_broadcast::<i64,_,_>([7,8],[2])` returns
/// `Array<i64, IxDyn>` == `[3, 4]` (truncated), while NumPy returns
/// `Array<f64>` == `[3.5, 4.0]`.
///
/// Upstream: generate_umath.py:419 — `np.divide` is true division; integer
/// operands promote to float64.
///
/// Live oracle:
/// ```text
/// >>> np.divide(np.array([7,8], np.int64), np.array([2], np.int64))
/// array([3.5, 4. ])   # dtype=float64
/// ```
#[test]
fn pin_divide_broadcast_int_truncates_instead_of_true_division() {
    // Before the #786 fix the target returned the truncated integer
    // quotient [3, 4]; after it, `divide_broadcast` is true division so the
    // output element type is f64 and the value is NumPy's [3.5, 4.0].
    const NP: [f64; 2] = [3.5, 4.0];
    let a = i64_arr(vec![7, 8]);
    let b = i64_arr(vec![2]);
    // i64 operands promote to f64 (NEP-50 true division), so the result is
    // `Array<f64, IxDyn>`, asserted directly against the numpy oracle.
    let got: Array<f64, IxDyn> = ferray_ufunc::divide_broadcast(&a, &b).unwrap();
    assert_eq!(
        got.as_slice().unwrap(),
        NP.as_slice(),
        "divide_broadcast(int, int) truncated to integer; NumPy np.divide is \
         true division returning float64 [3.5, 4.0] \
         (generate_umath.py:419 PyUFunc_TrueDivisionTypeResolver)"
    );
}
