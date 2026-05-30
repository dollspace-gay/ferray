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
//!     int input MUST promote to float; ferray is compile-gated float-only.**
//!     NumPy registers these `TD(flts)` only (`generate_umath.py:1057-1062`
//!     hypot, `:1030-1037` arctan2, `:710-721` logaddexp/logaddexp2,
//!     `:1107-1118` copysign/nextafter). Because no integer loop is
//!     registered, NumPy's ufunc resolver SAFE-CASTS integer input up to the
//!     smallest float loop: `np.hypot([3,4],[4,3]) -> float64 [5.,5.]`,
//!     `np.hypot(int16,int16) -> float32`, `np.arctan2([1,1],[1,1]) ->
//!     float64`. ferray's `hypot`/`arctan2`/`logaddexp`/`logaddexp2`/
//!     `copysign`/`nextafter` are all bound `T: Element + Float` with NO
//!     int-promoting `*_promote` sibling (the unary family has `sqrt_promote`
//!     etc.; the binary float family has none). So `ferray_ufunc::hypot` on
//!     an `Array<i64>` DOES NOT COMPILE. This is a MISSING COMPONENT, not a
//!     runtime mis-value, and is documented (not pinned) below — there is no
//!     callable entry point to assert against. A commented-out call shows the
//!     compile gate.
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
// MISSING COMPONENT (documented, not pinnable at runtime) —
// hypot/arctan2/logaddexp/logaddexp2/copysign/nextafter int->float promotion.
//
// NumPy promotes integer input to float for these (TD(flts) only; the
// resolver safe-casts int up): np.hypot([3,4],[4,3]) -> float64 [5.,5.];
// np.arctan2([1,1],[1,1]) -> float64; np.copysign([1,2],[-1,1]) -> float64
// [-1.,2.]; np.nextafter(int,int) -> float64. ferray's hypot/arctan2/
// logaddexp/logaddexp2/copysign/nextafter are bound `T: Element + Float`
// with NO `*_promote` sibling, so an `Array<i64>` argument does not compile.
// There is no runtime entry point to assert against; the builder must add a
// binary float-promote family (analogous to the unary `sqrt_promote` ...).
//
// The line below, if uncommented, fails to COMPILE — proving the gate:
//
//   let _ = ferray_ufunc::hypot(&ai64(vec![3, 4]), &ai64(vec![4, 3]));
//   // error[E0277]: the trait bound `i64: num_traits::Float` is not satisfied
//
// A float baseline IS callable and correct (pins the preserved invariant):
// ---------------------------------------------------------------------------

/// Baseline (passes): `ferray_ufunc::hypot` on f64 matches
/// `np.hypot([3.,4.],[4.,3.]) == [5.,5.]`. The int-input promotion path that
/// numpy provides has NO ferray entry point (missing component).
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
