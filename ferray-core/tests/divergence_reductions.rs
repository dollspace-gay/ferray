// Divergence pins for ferray-core reductions (sum, prod, min, max, mean).
//
// AUDIT TARGET: ferray-core/src/array/reductions.rs
//
// Each test asserts that ferray-core's reduction matches the LIVE numpy 2.4.5
// oracle (value + dtype). Expected values are numpy-derived (cited inline),
// NEVER copied from the ferray side (R-CHAR-3).
//
// These are FAILING tests pinning real divergences from NumPy. They do NOT
// fix production code.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

// ---------------------------------------------------------------------------
// DIVERGENCE 1 (HIGHEST SEVERITY — silent wrong value / panic):
// Integer accumulator promotion.
//
// numpy: `sum`/`prod` of an integer dtype with LESS precision than the default
// platform integer accumulate into (and return) the platform integer:
//   - signed  -> int64
//   - unsigned -> uint64
// Cite: numpy/_core/fromnumeric.py:2321-2327 (sum dtype docstring); the same
// rule for prod at fromnumeric.py:3310-3312. Implemented by `umr_sum =
// um.add.reduce` / `umr_prod = um.multiply.reduce` in
// numpy/_core/_methods.py:20-21.
//
// ferray: `Array::<T>::sum()` / `prod()` accumulate in `T` itself
// (`acc = T::zero(); acc = acc + x` — reductions.rs `pub fn sum`). For narrow
// integer T the running sum overflows T: in debug builds this PANICS
// ("attempt to add with overflow"); in release it silently WRAPS. Either way
// the result diverges from numpy's promoted, non-overflowing value.
// ---------------------------------------------------------------------------

#[test]
fn sum_i8_promotes_to_platform_int_no_overflow() {
    // numpy: np.sum(np.array([100,100,100], dtype=np.int8)) == 300, dtype int64
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([3]), vec![100, 100, 100]).unwrap();
    // The `: i64` annotation pins the promoted RESULT DTYPE: if `sum()`
    // returned `i8` (the pre-#780 behavior) this would not compile, and the
    // i8 accumulation would have panicked/wrapped at 44 (300 mod 256).
    let got: i64 = a.sum();
    assert_eq!(
        got, 300,
        "numpy int8 sum promotes to int64 = 300; ferray overflows i8"
    );
}

#[test]
fn prod_i8_promotes_to_platform_int_no_overflow() {
    // numpy: np.prod(np.array([100,100], dtype=np.int8)) == 10000, dtype int64
    let a = Array::<i8, Ix1>::from_vec(Ix1::new([2]), vec![100, 100]).unwrap();
    let got: i64 = a.prod();
    assert_eq!(
        got, 10000,
        "numpy int8 prod promotes to int64 = 10000; ferray overflows i8"
    );
}

#[test]
fn sum_i16_promotes_to_platform_int_no_overflow() {
    // numpy: np.sum(np.array([20000,20000], dtype=np.int16)) == 40000, int64
    let a = Array::<i16, Ix1>::from_vec(Ix1::new([2]), vec![20000, 20000]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(
        got, 40000,
        "numpy int16 sum promotes to int64 = 40000; ferray overflows i16"
    );
}

#[test]
fn sum_u8_promotes_to_unsigned_platform_int_no_overflow() {
    // numpy: np.sum(np.array([200,200], dtype=np.uint8)) == 400, dtype uint64
    let a = Array::<u8, Ix1>::from_vec(Ix1::new([2]), vec![200, 200]).unwrap();
    let got: u64 = a.sum();
    assert_eq!(
        got, 400,
        "numpy uint8 sum promotes to uint64 = 400; ferray overflows u8"
    );
}

#[test]
fn prod_i32_promotes_to_platform_int_no_overflow() {
    // numpy: np.prod(np.array([100000,100000], dtype=np.int32)) == 10_000_000_000, int64
    let a = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![100000, 100000]).unwrap();
    let got: i64 = a.prod();
    assert_eq!(
        got, 10_000_000_000,
        "numpy int32 prod promotes to int64; ferray overflows i32"
    );
}

#[test]
fn sum_i64_stays_i64_no_change() {
    // numpy: np.sum(np.array([1,2,3], dtype=np.int64)).dtype == int64 (no promotion).
    // The accumulator mapping leaves >= platform-int dtypes unchanged.
    let a = Array::<i64, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(got, 6, "numpy int64 sum stays int64 = 6");
}

#[test]
fn sum_f32_stays_f32_no_promotion() {
    // numpy: np.sum(np.array([1,2,3], dtype=np.float32)).dtype == float32 (no promotion).
    let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    let got: f32 = a.sum();
    assert_eq!(got, 6.0, "numpy float32 sum stays float32 = 6.0");
}

// ---------------------------------------------------------------------------
// DIVERGENCE 2: bool sum/prod promote to int64, counting True.
//
// numpy: np.sum(bool array) has dtype int64 and counts True; np.prod has
// dtype int64. Cite: numpy/_core/fromnumeric.py:2321-2327 (bool has less
// precision than platform int -> platform int) + numpy/_core/_methods.py:20-21
// (umr_sum/umr_prod). Live oracle:
//   np.sum(np.array([True, True, True])) == 3, dtype int64.
// ---------------------------------------------------------------------------

#[test]
fn sum_bool_counts_true_as_int64() {
    // numpy: np.sum(np.array([True, False, True, True])) == 3, dtype int64.
    let a = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, true]).unwrap();
    let got: i64 = a.sum();
    assert_eq!(got, 3, "numpy bool sum counts True as int64 = 3");
}

#[test]
fn prod_bool_promotes_to_int64() {
    // numpy: np.prod(np.array([True, True, True])) == 1, dtype int64;
    //        np.prod(np.array([True, False])) == 0, dtype int64.
    let all_true = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
    let with_false = Array::<bool, Ix1>::from_vec(Ix1::new([2]), vec![true, false]).unwrap();
    let got_true: i64 = all_true.prod();
    let got_false: i64 = with_false.prod();
    assert_eq!(got_true, 1, "numpy bool prod of all-true == 1, int64");
    assert_eq!(got_false, 0, "numpy bool prod with a false == 0, int64");
}

// NOTE on cumsum / cumprod: numpy promotes narrow-int cumulative reductions
// the same way (signed -> int64, unsigned -> uint64;
// numpy/_core/fromnumeric.py:2692-2693, 2769-2770). In ferray these live as
// free functions in `ferray-stats` (`cumsum`/`cumprod`, returning Array<T, D>
// with axis support), NOT as Array methods — so their promotion fix belongs to
// a SEPARATE ferray-stats blocker, not #780 (which owns ferray-core's
// `Array::sum`/`prod` accumulator). #780 ships the `ReduceAcc` accumulator
// mapping that the stats fix can reuse.

// ---------------------------------------------------------------------------
// #782 RESOLUTION — layer mis-pin (closed as the correct Rust analog, R-DEV-4).
//
// numpy: np.min/np.max of an empty array RAISE ValueError ("zero-size array to
// reduction operation minimum which has no identity") — cite numpy/_core/
// _methods.py:39-45 (_amin/_amax via um.minimum.reduce, no identity for empty).
//
// The numpy contract is USER-OBSERVABLE at the Python layer, and ferray ALREADY
// satisfies it there: `import ferray as fr; fr.min(fr.array([]))` raises
// ValueError("cannot compute min of empty array") — verified live against numpy
// 2.4.5 (both raise ValueError). The ferray-python binding maps ferray-core's
// `Option::None` -> PyValueError.
//
// ferray-core's `Array::min()/max() -> Option<T>` returning `None` for empty is
// the IDIOMATIC RUST ANALOG (R-DEV-4: deviate from a literal "raise" transcription
// and use Option for "no value exists") — NOT a divergence. Forcing Err here would
// break the idiomatic Rust API for zero user-observable benefit. The two tests
// below now assert that correct contract (None for empty) and double as the
// regression coverage proving the binding's error path has a well-defined source.
// ---------------------------------------------------------------------------

#[test]
fn min_empty_returns_none_rust_analog() {
    // ferray-core contract: empty min -> None (the Rust analog). numpy's
    // ValueError is enforced one layer up in ferray-python (verified live).
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    assert!(
        a.min().is_none(),
        "ferray-core min of empty is the None Rust analog; numpy's ValueError is enforced at the ferray-python boundary"
    );
}

#[test]
fn max_empty_returns_none_rust_analog() {
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
    assert!(
        a.max().is_none(),
        "ferray-core max of empty is the None Rust analog; numpy's ValueError is enforced at the ferray-python boundary"
    );
}

// ---------------------------------------------------------------------------
// MISSING COMPONENTS (no non-tautological #[test] possible — capability absent;
// reported in prose, NOT faked with a contrived assertion):
//
//  * argmin / argmax           — absent from ferray-core entirely. numpy returns
//    first-occurrence index as intp/int64, raises ValueError on empty, and (for
//    argmin/argmax) returns the index of the FIRST NaN when NaN is present.
//    Cite: numpy/_core/fromnumeric.py (argmin/argmax registrations).
//  * cumsum / cumprod          — live as ferray-stats free functions
//    (`ferray_stats::cumsum`/`cumprod`, Array<T, D>); their narrow-int
//    promotion is a SEPARATE ferray-stats blocker. #780 ships the `ReduceAcc`
//    accumulator mapping that fix reuses. Cite: fromnumeric.py:2692-2693.
//  * bool sum / prod -> int64  — SHIPPED in #780 (ReduceAcc<bool> -> i64);
//    pinned above (sum_bool_counts_true_as_int64 / prod_bool_promotes_to_int64).
//  * integer-array mean -> f64 — `mean()` is `T: Float`-bounded, so
//    `Array::<i32>::mean()` does not compile. numpy means an int array as f64.
//    Cite: numpy/_core/_methods.py:115-132.
// ---------------------------------------------------------------------------
