//! Divergence pins: BINARY ufunc dtype contracts that ferray-ufunc gets
//! wrong relative to NumPy 2.4.5.
//!
//! Two families are audited here:
//!
//!  1. **gcd / lcm — registered INTEGER-ONLY in NumPy, BACKWARDS in ferray.**
//!     NumPy registers `gcd`/`lcm` with `TD(ints)` only
//!     (`numpy/_core/code_generators/generate_umath.py:1156-1169`): integer
//!     input works, FLOAT input raises `TypeError` ("ufunc 'gcd' did not
//!     contain a loop with signature matching types"). ferray inverts this:
//!     the public `ferray_ufunc::gcd` / `lcm` are bound `T: Element + Float`
//!     (`ops/arithmetic.rs` `pub fn gcd` / `pub fn lcm`), so they ACCEPT a
//!     float array and silently compute a value, while INTEGER input does
//!     not compile against `gcd`/`lcm` at all (it must route to the separate
//!     `gcd_int`/`lcm_int`). The public `gcd`/`lcm` symbol therefore has the
//!     wrong domain: float-accepted (should reject), int-rejected (should
//!     accept). The test below pins the float-accepted half — it runs
//!     ferray's `gcd`/`lcm` on a float array and shows it returns `Ok`,
//!     whereas NumPy raises `TypeError`.
//!
//!  2. **hypot / arctan2 / logaddexp / logaddexp2 / copysign / nextafter —
//!     int input promotes to float (REQ-25, now SHIPPED).**
//!     NumPy registers these `TD(flts)` only (`generate_umath.py:1057-1063`
//!     hypot, `:1030-1037` arctan2, `:710-721` logaddexp/logaddexp2,
//!     `:1107-1119` copysign/nextafter). Because no integer loop is
//!     registered, NumPy's ufunc resolver SAFE-CASTS integer input up to the
//!     smallest float loop: `np.hypot([3,4],[4,3]) -> float64 [5.,5.]`,
//!     `np.hypot(int16,int16) -> float32`, `np.arctan2([1,1],[1,1]) ->
//!     float64`. ferray's `hypot`/`arctan2`/`logaddexp`/`logaddexp2`/
//!     `copysign`/`nextafter` are bound `T: Element + Float`; the builder added
//!     the binary `*_promote` siblings (`hypot_promote`/`arctan2_promote`/
//!     `logaddexp_promote`/`logaddexp2_promote`/`copysign_promote`/
//!     `nextafter_promote`) that reuse the REQ-23 `PromoteFloat` smallest-safe-
//!     cast-float kind rule for two same-type operands. The pins below run
//!     those promote entry points on integer/bool input.
//!
//! All expected values are live-numpy 2.4.5 results (R-CHAR-3), never
//! copied from the ferray side.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

fn af(v: Vec<f64>) -> Array<f64, Ix1> {
    Array::<f64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn ai64(v: Vec<i64>) -> Array<i64, Ix1> {
    Array::<i64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

// ---------------------------------------------------------------------------
// DIVERGENCE 1 — gcd/lcm domain is BACKWARDS (float accepted, int rejected).
//
// NumPy: `np.gcd(np.array([12.0,8.0]), np.array([8.0,4.0]))` -> TypeError
//        (gcd registered TD(ints) only; generate_umath.py:1156-1161).
// ferray: `ferray_ufunc::gcd` is `T: Element + Float`, so the same float
//        call COMPILES and returns `Ok([4.0, 4.0])`.
//
// The public `gcd` symbol must reject float and accept int (matching numpy);
// ferray's public `gcd` does the opposite. This test FAILS because ferray
// returns a value (Ok) where numpy has no valid loop. We pin it by asserting
// the operation should not have been a valid float op — i.e. the call that
// numpy rejects must not silently succeed on the public `gcd` entry point.
// ---------------------------------------------------------------------------

/// Divergence: `ferray_ufunc::gcd` on a FLOAT array succeeds, but
/// `np.gcd(float_array, float_array)` raises `TypeError` — gcd is registered
/// INTEGER-ONLY in NumPy (`numpy/_core/code_generators/generate_umath.py:1156`
/// `'gcd': Ufunc(... TD(ints) ...)`). NumPy: float gcd -> TypeError.
/// ferray: float gcd -> Ok. The public `gcd` domain is backwards.
#[test]
fn divergence_gcd_rejects_float_like_numpy() {
    // np.gcd on float -> TypeError (verified live, numpy 2.4.5):
    //   np.gcd(np.array([12.,8.]), np.array([8.,4.])) raises TypeError.
    // The public ferray `gcd` MUST mirror numpy's integer-only domain: a
    // float array is NOT a valid gcd input. ferray currently accepts it.
    let a = af(vec![12.0, 8.0]);
    let b = af(vec![8.0, 4.0]);
    let result = ferray_ufunc::gcd(&a, &b);
    // numpy has no float gcd loop -> the operation is invalid. The public
    // entry point must NOT silently succeed on float input. This assertion
    // pins the numpy contract (float gcd is rejected) and FAILS today
    // because ferray returns Ok.
    assert!(
        result.is_err(),
        "numpy np.gcd(float,float) raises TypeError (gcd is integer-only, \
         generate_umath.py:1156 TD(ints)); ferray's public `gcd` must reject \
         float input, but it returned Ok({:?})",
        result.ok().map(|r| r.as_slice().unwrap().to_vec())
    );
}

/// Divergence: `ferray_ufunc::lcm` on a FLOAT array succeeds, but
/// `np.lcm(float_array, float_array)` raises `TypeError` — lcm is registered
/// INTEGER-ONLY in NumPy
/// (`numpy/_core/code_generators/generate_umath.py:1163` `'lcm': Ufunc(...
/// TD(ints) ...)`). NumPy: float lcm -> TypeError. ferray: float lcm -> Ok.
#[test]
fn divergence_lcm_rejects_float_like_numpy() {
    // np.lcm(np.array([4.,6.]), np.array([6.,8.])) raises TypeError (live
    // numpy 2.4.5). The public ferray `lcm` must reject float input too.
    let a = af(vec![4.0, 6.0]);
    let b = af(vec![6.0, 8.0]);
    let result = ferray_ufunc::lcm(&a, &b);
    assert!(
        result.is_err(),
        "numpy np.lcm(float,float) raises TypeError (lcm is integer-only, \
         generate_umath.py:1163 TD(ints)); ferray's public `lcm` must reject \
         float input, but it returned Ok({:?})",
        result.ok().map(|r| r.as_slice().unwrap().to_vec())
    );
}

// ---------------------------------------------------------------------------
// POSITIVE BASELINE — gcd_int / lcm_int on INTEGER input match numpy's
// integer gcd/lcm. This is the domain the PUBLIC `gcd`/`lcm` *should* accept.
// These pass today (the int variants are correct); they document that the
// fix is to make the public `gcd`/`lcm` symbol route integer input here.
// Expected values are live numpy: np.gcd([12,15],[8,20]) == [4,5];
// np.lcm([4,6],[6,8]) == [12,24].
// ---------------------------------------------------------------------------

/// Baseline (passes): `ferray_ufunc::gcd_int` on i64 matches
/// `np.gcd([12,15],[8,20]) == [4,5]`. The public `gcd` must accept this
/// integer input (it currently does not compile against `gcd`).
#[test]
fn baseline_gcd_int_matches_numpy() {
    let a = ai64(vec![12, 15]);
    let b = ai64(vec![8, 20]);
    let out = ferray_ufunc::gcd_int(&a, &b).unwrap();
    // np.gcd([12,15],[8,20]) -> array([4, 5]) (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[4, 5]);
}

/// Baseline (passes): `ferray_ufunc::lcm_int` on i64 matches
/// `np.lcm([4,6],[6,8]) == [12,24]`.
#[test]
fn baseline_lcm_int_matches_numpy() {
    let a = ai64(vec![4, 6]);
    let b = ai64(vec![6, 8]);
    let out = ferray_ufunc::lcm_int(&a, &b).unwrap();
    // np.lcm([4,6],[6,8]) -> array([12, 24]) (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[12, 24]);
}

// ---------------------------------------------------------------------------
// REQ-25 — BINARY float-ufunc int/bool -> float promotion. The builder shipped
// the `*_promote` family (hypot_promote/arctan2_promote/logaddexp_promote/
// logaddexp2_promote/copysign_promote/nextafter_promote), reusing the REQ-23
// `PromoteFloat` smallest-safe-cast-float kind rule for two same-type operands:
// bool/int8/uint8 -> float16, int16/uint16 -> float32,
// int32/int64/uint32/uint64 -> float64. These ops register only float loops
// with NO `TD(bints)`, so numpy safe-casts integer input up:
//   np.hypot([3,4],[4,3]) -> float64 [5.,5.]; np.arctan2([1,1],[1,1]) ->
//   float64; np.copysign([1,2],[-1,1]) -> float64 [-1.,2.];
//   np.nextafter(int,int) -> float64.
// The output element type is `<T as PromoteFloat>::Out`, so the dtype below is
// compile-time enforced by the explicitly-typed result binding AND asserted at
// runtime via the static `Out::dtype()`. All expected values are live numpy
// 2.4.5 (R-CHAR-3).
// ---------------------------------------------------------------------------

use ferray_core::dtype::{DType, Element};
use ferray_ufunc::PromoteFloat;

fn ai16(v: Vec<i16>) -> Array<i16, Ix1> {
    Array::<i16, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

/// AC-19: `hypot_promote(int64 [3,4], int64 [4,3])` -> float64 `[5.0, 5.0]`
/// (numpy: `np.hypot([3,4],[4,3]).tolist() == [5.0, 5.0]`, dtype float64;
/// `generate_umath.py:1057-1063`, no `TD(bints)`). Was MISSING (compile-gated
/// `T: Float`).
#[test]
fn divergence_hypot_int_promotes_to_f64() {
    assert_eq!(<i64 as PromoteFloat>::Out::dtype(), DType::F64);
    let out: Array<f64, Ix1> =
        ferray_ufunc::hypot_promote(&ai64(vec![3, 4]), &ai64(vec![4, 3])).unwrap();
    // np.hypot([3,4],[4,3]).tolist() == [5.0, 5.0] (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[5.0_f64, 5.0]);
}

/// AC-19: `copysign_promote(int64 [1,2], int64 [-1,1])` -> float64
/// `[-1.0, 2.0]` (numpy: `np.copysign([1,2],[-1,1]).tolist() == [-1.0, 2.0]`,
/// dtype float64; `generate_umath.py:1107-1113`).
#[test]
fn divergence_copysign_int_promotes_to_f64() {
    let out: Array<f64, Ix1> =
        ferray_ufunc::copysign_promote(&ai64(vec![1, 2]), &ai64(vec![-1, 1])).unwrap();
    // np.copysign([1,2],[-1,1]).tolist() == [-1.0, 2.0] (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[-1.0_f64, 2.0]);
}

/// `arctan2_promote(int64 [1,1], int64 [1,1])` -> float64
/// `[π/4, π/4] == [0.7853981633974483, …]` (numpy: `np.arctan2([1,1],[1,1])`
/// dtype float64; `generate_umath.py:1030-1037`).
#[test]
#[allow(
    clippy::approx_constant,
    reason = "numpy arctan2(1,1)=π/4 oracle constant, not a std consts approximation"
)]
fn divergence_arctan2_int_promotes_to_f64() {
    let out: Array<f64, Ix1> =
        ferray_ufunc::arctan2_promote(&ai64(vec![1, 1]), &ai64(vec![1, 1])).unwrap();
    // np.arctan2([1,1],[1,1]).tolist() == [0.7853981633974483, 0.7853981633974483]
    let s = out.as_slice().unwrap();
    assert!((s[0] - 0.785_398_163_397_448_3).abs() < 1e-15);
    assert!((s[1] - 0.785_398_163_397_448_3).abs() < 1e-15);
}

/// `logaddexp_promote(int64 [0,0], int64 [0,0])` -> float64
/// `[ln 2, ln 2] == [0.6931471805599453, …]` (numpy:
/// `np.logaddexp([0,0],[0,0])` dtype float64; `generate_umath.py:710-715`).
#[test]
#[allow(
    clippy::approx_constant,
    reason = "numpy logaddexp(0,0)=ln2 oracle constant, not a std consts approximation"
)]
fn divergence_logaddexp_int_promotes_to_f64() {
    let out: Array<f64, Ix1> =
        ferray_ufunc::logaddexp_promote(&ai64(vec![0, 0]), &ai64(vec![0, 0])).unwrap();
    // np.logaddexp([0,0],[0,0]).tolist() == [0.6931471805599453, 0.6931471805599453]
    let s = out.as_slice().unwrap();
    assert!((s[0] - 0.693_147_180_559_945_3).abs() < 1e-15);
    assert!((s[1] - 0.693_147_180_559_945_3).abs() < 1e-15);
}

/// `logaddexp2_promote(int64 [0,0], int64 [0,0])` -> float64 `[1.0, 1.0]`
/// (numpy: `np.logaddexp2([0,0],[0,0]).tolist() == [1.0, 1.0]`, dtype float64;
/// `generate_umath.py:716-721`).
#[test]
fn divergence_logaddexp2_int_promotes_to_f64() {
    let out: Array<f64, Ix1> =
        ferray_ufunc::logaddexp2_promote(&ai64(vec![0, 0]), &ai64(vec![0, 0])).unwrap();
    // np.logaddexp2([0,0],[0,0]).tolist() == [1.0, 1.0] (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[1.0_f64, 1.0]);
}

/// `nextafter_promote(int64 [1], int64 [2])` -> float64
/// `[1.0000000000000002]` (numpy: `np.nextafter(np.array([1]),np.array([2]))`
/// dtype float64; the next f64 after 1.0 towards 2.0;
/// `generate_umath.py:1114-1119`).
#[test]
fn divergence_nextafter_int_promotes_to_f64() {
    let out: Array<f64, Ix1> =
        ferray_ufunc::nextafter_promote(&ai64(vec![1]), &ai64(vec![2])).unwrap();
    // np.nextafter([1],[2])[0] == 1.0000000000000002 (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[1.000_000_000_000_000_2_f64]);
}

/// AC-20: int16 input promotes to **float32** (the REQ-23 smallest-safe-cast
/// kind rule, now applied to two operands). `np.hypot(int16 [3], int16 [4])`
/// dtype float32, value `[5.0]`.
#[test]
fn divergence_hypot_int16_promotes_to_f32() {
    assert_eq!(<i16 as PromoteFloat>::Out::dtype(), DType::F32);
    let out: Array<f32, Ix1> = ferray_ufunc::hypot_promote(&ai16(vec![3]), &ai16(vec![4])).unwrap();
    // np.hypot(np.int16([3]), np.int16([4])).tolist() == [5.0], dtype float32.
    assert_eq!(out.as_slice().unwrap(), &[5.0_f32]);
}

/// AC-20: bool/uint8 input promotes to **float16** (feature-gated: `half::f16`
/// exists only under `f16`). `np.hypot(uint8 [3], uint8 [4])` dtype float16
/// value `[5.0]`; `np.hypot(bool [T,F], bool [F,T])` dtype float16 `[1.0, 1.0]`.
#[cfg(feature = "f16")]
#[test]
fn divergence_hypot_bool_u8_promotes_to_f16() {
    use half::f16;
    assert_eq!(<u8 as PromoteFloat>::Out::dtype(), DType::F16);
    assert_eq!(<bool as PromoteFloat>::Out::dtype(), DType::F16);
    let au8 = |v: Vec<u8>| Array::<u8, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap();
    let out_u: Array<f16, Ix1> = ferray_ufunc::hypot_promote(&au8(vec![3]), &au8(vec![4])).unwrap();
    // np.hypot(np.uint8([3]), np.uint8([4])).tolist() == [5.0], dtype float16.
    let vals_u: Vec<f32> = out_u
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    assert_eq!(vals_u, vec![5.0_f32]);
    let ab = |v: Vec<bool>| Array::<bool, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap();
    let out_b: Array<f16, Ix1> =
        ferray_ufunc::hypot_promote(&ab(vec![true, false]), &ab(vec![false, true])).unwrap();
    // np.hypot([T,F],[F,T]).tolist() == [1.0, 1.0], dtype float16.
    let vals_b: Vec<f32> = out_b
        .as_slice()
        .unwrap()
        .iter()
        .map(|&x| x.to_f32())
        .collect();
    assert_eq!(vals_b, vec![1.0_f32, 1.0]);
}

/// Shape mismatch propagates as an error from the promoted binary path (the
/// underlying same-type binary kernel requires same-shape operands).
#[test]
fn divergence_hypot_promote_shape_mismatch_errors() {
    let r = ferray_ufunc::hypot_promote(&ai64(vec![1, 2, 3]), &ai64(vec![1, 2]));
    assert!(r.is_err());
}

// ---------------------------------------------------------------------------
// POSITIVE BASELINE (preserved invariant) — the float64-in/float64-out
// contract the REQ-25 `*_promote` path must NOT disturb (it never touches the
// existing `T: Float` kernels). Expected values are live numpy 2.4.5.
// ---------------------------------------------------------------------------

/// Baseline (passes): `ferray_ufunc::hypot` on f64 matches
/// `np.hypot([3.,4.],[4.,3.]) == [5.,5.]` — the float kernel is byte-identical
/// after the promote family lands.
#[test]
fn baseline_hypot_f64_matches_numpy() {
    let a = af(vec![3.0, 4.0]);
    let b = af(vec![4.0, 3.0]);
    let out = ferray_ufunc::hypot(&a, &b).unwrap();
    // np.hypot([3.,4.],[4.,3.]) -> array([5., 5.]) (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[5.0, 5.0]);
}

/// Baseline (passes): `ferray_ufunc::copysign` on f64 matches
/// `np.copysign([1.,2.],[-1.,1.]) == [-1.,2.]`.
#[test]
fn baseline_copysign_f64_matches_numpy() {
    let a = af(vec![1.0, 2.0]);
    let b = af(vec![-1.0, 1.0]);
    let out = ferray_ufunc::copysign(&a, &b).unwrap();
    // np.copysign([1.,2.],[-1.,1.]) -> array([-1., 2.]) (live numpy 2.4.5).
    assert_eq!(out.as_slice().unwrap(), &[-1.0, 2.0]);
}

// ---------------------------------------------------------------------------
// COMPARISON spot-check (VALIDATION, expected to PASS) —
// greater/less/equal/not_equal on int+int and NaN match numpy's bool output.
// NumPy registers these comparison ufuncs with bool output
// (`generate_umath.py:543-614`, `TD(bints, out='?')` + `TD(inexact ..., out='?')`).
// ferray uses Rust's native `<`/`==`, which already give IEEE NaN semantics
// (NaN != NaN; all NaN orderings false). These tests confirm NO divergence —
// expected values are live numpy 2.4.5.
// ---------------------------------------------------------------------------

/// Validation (passes): `ferray_ufunc::greater` on i64 matches
/// `np.greater([1,2,3],[3,2,1]) == [F,F,T]`.
#[test]
fn comparison_greater_int_matches_numpy() {
    let a = ai64(vec![1, 2, 3]);
    let b = ai64(vec![3, 2, 1]);
    let out = ferray_ufunc::greater(&a, &b).unwrap();
    // np.greater([1,2,3],[3,2,1]) -> array([False, False,  True]).
    assert_eq!(out.as_slice().unwrap(), &[false, false, true]);
}

/// Validation (passes): `ferray_ufunc::equal` on i64 matches
/// `np.equal([1,2,3],[1,0,3]) == [T,F,T]`.
#[test]
fn comparison_equal_int_matches_numpy() {
    let a = ai64(vec![1, 2, 3]);
    let b = ai64(vec![1, 0, 3]);
    let out = ferray_ufunc::equal(&a, &b).unwrap();
    // np.equal([1,2,3],[1,0,3]) -> array([ True, False,  True]).
    assert_eq!(out.as_slice().unwrap(), &[true, false, true]);
}

/// Validation (passes): NaN comparisons match numpy IEEE semantics —
/// `np.equal(nan,nan)==False`, `np.not_equal(nan,nan)==True`,
/// `np.greater(nan,nan)==False`, `np.less(nan,1.)==False`.
#[test]
fn comparison_nan_matches_numpy() {
    let nan = af(vec![f64::NAN]);
    let nan2 = af(vec![f64::NAN]);
    let one = af(vec![1.0]);
    // np.equal(nan,nan) -> False
    assert_eq!(
        ferray_ufunc::equal(&nan, &nan2)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[false]
    );
    // np.not_equal(nan,nan) -> True
    assert_eq!(
        ferray_ufunc::not_equal(&nan, &nan2)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[true]
    );
    // np.greater(nan,nan) -> False
    assert_eq!(
        ferray_ufunc::greater(&nan, &nan2)
            .unwrap()
            .as_slice()
            .unwrap(),
        &[false]
    );
    // np.less(nan,1.) -> False
    assert_eq!(
        ferray_ufunc::less(&nan, &one).unwrap().as_slice().unwrap(),
        &[false]
    );
}
