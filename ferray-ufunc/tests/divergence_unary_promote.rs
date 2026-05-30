//! Divergence pins: NumPy unary float ufuncs promote INTEGER (and bool)
//! input arrays to a FLOAT output dtype; the rounding family
//! floor/ceil/trunc/round keep integer input INTEGER. ferray's unary float
//! ops were all `T: Element + Float`-bounded, so they rejected integer input
//! *at compile time* — there was no int->float promotion path nor an
//! int-identity rounding path. The builder shipped:
//!
//!   * REQ-23 — `ferray_ufunc::*_promote` (sqrt_promote, exp_promote,
//!     rint_promote, sin_promote, …) accept integer/bool input and promote
//!     to the smallest safe-cast float (bool/int8/uint8 -> float16,
//!     int16/uint16 -> float32, int32/int64/uint32/uint64 -> float64).
//!   * REQ-24 — `ferray_ufunc::{floor,ceil,trunc,round,fix,around}_int`
//!     return integer/bool input UNCHANGED in the input dtype.
//!
//! The output element type for `*_promote` is `<T as PromoteFloat>::Out`,
//! so the dtype assertions below are *compile-time enforced* by the
//! explicitly-typed result bindings (e.g. `let out: Array<f64, Ix1> = …`),
//! and `<T as PromoteFloat>::Out::dtype()` is additionally asserted at
//! runtime.
//!
//! ## Upstream contract (live numpy 2.4.5; cross-checked vs
//! ## `numpy/_core/code_generators/generate_umath.py`)
//!
//! | op family                                   | int64 input -> | float loop cite |
//! |---------------------------------------------|----------------|-----------------|
//! | exp exp2 expm1                              | float64        | generate_umath.py:908,917,925 |
//! | log log2 log10 log1p                        | float64        | generate_umath.py:933,942,950,958 |
//! | sqrt cbrt                                    | float64        | generate_umath.py:966,975 |
//! | sin cos tan                                 | float64        | generate_umath.py:865,855,875 |
//! | arcsin arccos arctan sinh cosh tanh         | float64        | (TD(inexact,...)) |
//! | fabs                                        | float64        | generate_umath.py:1003 (TD(flts)) |
//! | rint                                        | float64        | generate_umath.py:1021-1029 (NO TD(bints)) |
//! | floor ceil trunc round                      | **int64 identity** | generate_umath.py:987,997,1015 (TD(bints) FIRST) |
//!
//! Smallest-safe-cast float (confirmed live, numpy 2.4.5):
//!   * bool/int8/uint8 -> **float16**  (`np.sqrt(np.array([True])).dtype == float16`)
//!   * int16/uint16    -> **float32**  (`np.sqrt(np.array([4],np.int16)).dtype == float32`)
//!   * int32/int64/uint32/uint64 -> **float64**
//!
//! All expected values below are live-numpy results (R-CHAR-3): never
//! literal-copied from ferray, which produced no value at all for int input.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::dtype::{DType, Element};
use ferray_ufunc::PromoteFloat;

// ---------------------------------------------------------------------------
// Constructors for typed input arrays.
// ---------------------------------------------------------------------------

fn af(v: Vec<f64>) -> Array<f64, Ix1> {
    Array::<f64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn ai64(v: Vec<i64>) -> Array<i64, Ix1> {
    Array::<i64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn ai32(v: Vec<i32>) -> Array<i32, Ix1> {
    Array::<i32, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn ai16(v: Vec<i16>) -> Array<i16, Ix1> {
    Array::<i16, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn au16(v: Vec<u16>) -> Array<u16, Ix1> {
    Array::<u16, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

// ---------------------------------------------------------------------------
// POSITIVE BASELINE (preserved invariant) — a *float64* input to each unary
// op yields a float64 result whose values match the live-numpy oracle. The
// builder's int-promote path must NOT disturb this float64-in/float64-out
// contract. Not tautological: the expected constants are numpy results.
// ---------------------------------------------------------------------------

/// Baseline: `ferray sqrt` on f64 `[1,2,4]` matches
/// `np.sqrt(np.array([1.,2.,4.])) == [1.0, 1.4142135623730951, 2.0]`.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "numpy sqrt(2) oracle constant, not a std consts approximation"
)]
fn baseline_sqrt_f64_matches_numpy() {
    let out = ferray_ufunc::sqrt(&af(vec![1.0, 2.0, 4.0])).unwrap();
    assert_eq!(
        out.as_slice().unwrap(),
        &[1.0_f64, 1.414_213_562_373_095_1, 2.0]
    );
}

/// Baseline: `ferray exp` on f64 `[1,2,4]` matches
/// `np.exp([1.,2.,4.]) == [2.718281828459045, 7.38905609893065, 54.598150033144236]`.
#[test]
#[allow(
    clippy::approx_constant,
    reason = "numpy exp(1)=e oracle constant, not a std consts approximation"
)]
fn baseline_exp_f64_matches_numpy() {
    let out = ferray_ufunc::exp(&af(vec![1.0, 2.0, 4.0])).unwrap();
    let s = out.as_slice().unwrap();
    assert!((s[0] - 2.718_281_828_459_045).abs() < 1e-12);
    assert!((s[1] - 7.389_056_098_930_65).abs() < 1e-11);
    assert!((s[2] - 54.598_150_033_144_236).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// REQ-23 — int -> float promotion. The `*_promote` family now compiles and
// runs on integer input. Output dtype is enforced by the explicitly-typed
// result binding AND asserted at runtime via the static `Out::dtype()`.
// ---------------------------------------------------------------------------

/// `np.sqrt(int64 [1,2,4])` -> float64 `[1.0, 1.4142135623730951, 2.0]`
/// (generate_umath.py:966-974: sqrt registers only float loops; int
/// safe-casts to float64). Was MISSING (compile-gated `T: Float`).
#[test]
// The 1.414… literal is the live `np.sqrt(2)` oracle value (R-CHAR-3), not
// an approximation of `std::f64::consts::SQRT_2` — clippy can't tell.
#[allow(
    clippy::approx_constant,
    reason = "numpy sqrt(2) oracle constant, not a std consts approximation"
)]
fn divergence_sqrt_int_promotes_to_f64() {
    // numpy 2.4.5: np.sqrt(np.array([1,2,4])).tolist()
    const NUMPY_SQRT_INT_124: [f64; 3] = [1.0, 1.414_213_562_373_095_1, 2.0];
    // Compile-time dtype proof: `<i64 as PromoteFloat>::Out == f64`.
    assert_eq!(<i64 as PromoteFloat>::Out::dtype(), DType::F64);
    let out: Array<f64, Ix1> = ferray_ufunc::sqrt_promote(&ai64(vec![1, 2, 4])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &NUMPY_SQRT_INT_124);
}

/// `np.exp(int64 [1,2,4])` -> float64
/// `[2.718281828459045, 7.38905609893065, 54.598150033144236]`
/// (generate_umath.py:908-916). Was MISSING (`T: Float + CrMath`).
#[test]
#[allow(
    clippy::approx_constant,
    reason = "numpy exp(1)=e oracle constant, not a std consts approximation"
)]
fn divergence_exp_int_promotes_to_f64() {
    const NUMPY_EXP_INT_124: [f64; 3] = [
        2.718_281_828_459_045,
        7.389_056_098_930_65,
        54.598_150_033_144_236,
    ];
    let out: Array<f64, Ix1> = ferray_ufunc::exp_promote(&ai64(vec![1, 2, 4])).unwrap();
    let s = out.as_slice().unwrap();
    assert!((s[0] - NUMPY_EXP_INT_124[0]).abs() < 1e-12);
    assert!((s[1] - NUMPY_EXP_INT_124[1]).abs() < 1e-11);
    assert!((s[2] - NUMPY_EXP_INT_124[2]).abs() < 1e-10);
}

/// `np.rint(int64 [1,2,4])` -> float64 `[1.0,2.0,4.0]`. Unlike
/// floor/ceil/trunc, `rint` registers NO `TD(bints)`
/// (generate_umath.py:1021-1029), so int input promotes to float64.
#[test]
fn divergence_rint_int_promotes_to_f64() {
    const NUMPY_RINT_INT_124: [f64; 3] = [1.0, 2.0, 4.0];
    let out: Array<f64, Ix1> = ferray_ufunc::rint_promote(&ai64(vec![1, 2, 4])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &NUMPY_RINT_INT_124);
}

/// Smallest-safe-cast-float dtype kind mapping for the f32-output kinds.
/// `np.sqrt(int16 [4,9]).dtype == float32`, `np.sqrt(uint16 ...).dtype ==
/// float32` (numpy resolves int16/uint16 against float32, its smallest
/// safe-cast float). Values: `np.sqrt([4,9]) == [2.0, 3.0]`.
#[test]
fn divergence_sqrt_int16_uint16_promotes_to_f32() {
    assert_eq!(<i16 as PromoteFloat>::Out::dtype(), DType::F32);
    assert_eq!(<u16 as PromoteFloat>::Out::dtype(), DType::F32);
    // numpy 2.4.5: np.sqrt(np.array([4,9],np.int16)).tolist() == [2.0, 3.0]
    let out_i16: Array<f32, Ix1> = ferray_ufunc::sqrt_promote(&ai16(vec![4, 9])).unwrap();
    assert_eq!(out_i16.as_slice().unwrap(), &[2.0_f32, 3.0]);
    let out_u16: Array<f32, Ix1> = ferray_ufunc::sqrt_promote(&au16(vec![4, 9])).unwrap();
    assert_eq!(out_u16.as_slice().unwrap(), &[2.0_f32, 3.0]);
}

/// Smallest-safe-cast-float dtype kind mapping for the f64-output kinds.
/// `np.sqrt(int32 [4,9]).dtype == float64` (int32/int64/uint32/uint64 all
/// resolve to float64). Distinct from int16->float32 above.
#[test]
fn divergence_sqrt_int32_promotes_to_f64() {
    assert_eq!(<i32 as PromoteFloat>::Out::dtype(), DType::F64);
    assert_eq!(<u32 as PromoteFloat>::Out::dtype(), DType::F64);
    assert_eq!(<i64 as PromoteFloat>::Out::dtype(), DType::F64);
    assert_eq!(<u64 as PromoteFloat>::Out::dtype(), DType::F64);
    // numpy 2.4.5: np.sqrt(np.array([4,9],np.int32)).tolist() == [2.0, 3.0]
    let out: Array<f64, Ix1> = ferray_ufunc::sqrt_promote(&ai32(vec![4, 9])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[2.0_f64, 3.0]);
}

/// `sin`/`cos` promote integer input to float64 too (the whole trig family
/// shares the no-`TD(bints)` float-loop registration). `np.sin(int64 [0]) ==
/// [0.0]`, `np.cos(int64 [0]) == [1.0]`.
#[test]
fn divergence_sin_cos_int_promote_to_f64() {
    let s: Array<f64, Ix1> = ferray_ufunc::sin_promote(&ai64(vec![0, 1])).unwrap();
    let c: Array<f64, Ix1> = ferray_ufunc::cos_promote(&ai64(vec![0, 1])).unwrap();
    // numpy 2.4.5: np.sin([0,1]) == [0.0, 0.8414709848078965]
    assert!((s.as_slice().unwrap()[0] - 0.0).abs() < 1e-15);
    assert!((s.as_slice().unwrap()[1] - 0.841_470_984_807_896_5).abs() < 1e-12);
    // np.cos([0,1]) == [1.0, 0.5403023058681398]
    assert!((c.as_slice().unwrap()[0] - 1.0).abs() < 1e-15);
    assert!((c.as_slice().unwrap()[1] - 0.540_302_305_868_139_8).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// REQ-23 — bool/int8/uint8 -> float16 (feature-gated: `half::f16` exists only
// under the `f16` feature, so this pin is gated to match). Run with
// `cargo test -p ferray-ufunc --features f16`.
// ---------------------------------------------------------------------------

/// `np.sqrt(bool [T,F,T])` -> **float16** `[1.0, 0.0, 1.0]`, and
/// `np.sqrt(uint8)` -> float16 (numpy picks the smallest safe-cast float:
/// bool/int8/uint8 -> float16). The output element type is `half::f16`.
#[cfg(feature = "f16")]
#[test]
fn divergence_sqrt_bool_promotes_to_f16() {
    use half::f16;
    assert_eq!(<bool as PromoteFloat>::Out::dtype(), DType::F16);
    assert_eq!(<u8 as PromoteFloat>::Out::dtype(), DType::F16);
    assert_eq!(<i8 as PromoteFloat>::Out::dtype(), DType::F16);
    // numpy 2.4.5: np.sqrt(np.array([True,False,True])).tolist() == [1.0,0.0,1.0]
    let b = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
    let out: Array<f16, Ix1> = ferray_ufunc::sqrt_promote(&b).unwrap();
    let vals: Vec<f32> = out
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    assert_eq!(vals, vec![1.0_f32, 0.0, 1.0]);
    // np.sqrt(np.array([4,9], np.uint8)).dtype == float16, vals [2.0, 3.0]
    let u = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![4u8, 9]).unwrap();
    let out_u: Array<f16, Ix1> = ferray_ufunc::sqrt_promote(&u).unwrap();
    let vals_u: Vec<f32> = out_u
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    assert_eq!(vals_u, vec![2.0_f32, 3.0]);
}

// ---------------------------------------------------------------------------
// REQ-24 — floor/ceil/trunc/round INT-IDENTITY (opposite of rint). Integer
// input returns the values unchanged in the INPUT dtype.
// ---------------------------------------------------------------------------

/// `np.floor(int64 [1,2,4])` returns **int64 identity** `[1,2,4]`, NOT
/// float64 — floor registers `TD(bints)` FIRST (generate_umath.py:1011-1019);
/// same for ceil (:983-991) and trunc (:993-1001). Was MISSING.
#[test]
fn divergence_floor_int_is_identity_int64() {
    // numpy 2.4.5: np.floor(np.array([1,2,4])).tolist() == [1,2,4], dtype int64
    const NUMPY_FLOOR_INT_124: [i64; 3] = [1, 2, 4];
    let out: Array<i64, Ix1> = ferray_ufunc::floor_int(&ai64(vec![1, 2, 4])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &NUMPY_FLOOR_INT_124);
    // dtype preserved as int64.
    assert_eq!(<i64 as Element>::dtype(), DType::I64);
}

/// floor/ceil/trunc/round/fix/around all int-identity on int64 input.
/// numpy: `np.ceil([1,2,4]) == np.trunc([1,2,4]) == np.round([1,2,4]) ==
/// [1,2,4]` (all int64), distinct from `np.rint([1,2,4])` -> float64.
#[test]
fn divergence_round_family_int_identity() {
    let v = vec![1i64, -2, 3, -4];
    assert_eq!(
        ferray_ufunc::floor_int(&ai64(v.clone()))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
    assert_eq!(
        ferray_ufunc::ceil_int(&ai64(v.clone()))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
    assert_eq!(
        ferray_ufunc::trunc_int(&ai64(v.clone()))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
    assert_eq!(
        ferray_ufunc::round_int(&ai64(v.clone()))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
    assert_eq!(
        ferray_ufunc::fix_int(&ai64(v.clone()))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
    assert_eq!(
        ferray_ufunc::around_int(&ai64(v))
            .unwrap()
            .as_slice()
            .unwrap(),
        &[1, -2, 3, -4]
    );
}

/// REQ-24 preserves the INPUT dtype, not a forced int64: `np.floor(int32
/// [5]).dtype == int32`, `np.ceil(uint16 ...).dtype == uint16`. Contrast with
/// REQ-23 where every int promotes to a *float* kind.
#[test]
fn divergence_floor_preserves_input_int_dtype() {
    // np.floor(np.array([5,6], np.int32)).dtype == int32
    let out: Array<i32, Ix1> = ferray_ufunc::floor_int(&ai32(vec![5, 6])).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[5_i32, 6]);
    assert_eq!(<i32 as Element>::dtype(), DType::I32);
    // np.ceil(np.array([7,8], np.uint16)).dtype == uint16
    let out_u: Array<u16, Ix1> = ferray_ufunc::ceil_int(&au16(vec![7, 8])).unwrap();
    assert_eq!(out_u.as_slice().unwrap(), &[7_u16, 8]);
}

/// The contract split made explicit: `rint` PROMOTES int->float64 (REQ-23)
/// while `floor` keeps int->int64 (REQ-24) on the SAME input. This is the
/// exact divergence the design doc calls out (numpy `rint` has no
/// `TD(bints)`; `floor` registers it first).
#[test]
fn divergence_rint_vs_floor_int_contract_split() {
    let input = ai64(vec![1, 2, 4]);
    let rinted: Array<f64, Ix1> = ferray_ufunc::rint_promote(&input).unwrap();
    let floored: Array<i64, Ix1> = ferray_ufunc::floor_int(&input).unwrap();
    // rint -> float64 [1.0,2.0,4.0]; floor -> int64 [1,2,4].
    assert_eq!(rinted.as_slice().unwrap(), &[1.0_f64, 2.0, 4.0]);
    assert_eq!(floored.as_slice().unwrap(), &[1_i64, 2, 4]);
}
