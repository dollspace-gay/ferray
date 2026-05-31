//! Divergence pins: ferray-stats reductions / descriptive stats vs numpy 2.4.5.
//!
//! Authored by acto-critic. Each test asserts the LIVE-numpy-2.4.5 result
//! (R-CHAR-3: expected values traced to a numpy `file:line` symbolic constant,
//! NEVER copied from the ferray side). Tests that currently FAIL pin a real
//! divergence the generator must fix DOWN in ferray-stats (or its ufunc/core
//! dependency that ferray-stats re-exports).
//!
//! Root-cause map (per AUDIT TARGET):
//!   - sum/prod narrow-int -> int64 promotion: `reductions::{sum,prod}` are
//!     `T -> T` (no `ReduceAcc`). ferray-core grew `ReduceAcc` (commit
//!     c13a3f20) whose own commit message says "ferray-stats ... reuses
//!     ReduceAcc" — i.e. the stats crate has NOT yet adopted it.
//!   - cumsum/cumprod narrow-int -> int64: `reductions::{cumsum,cumprod}`
//!     re-export `ferray_ufunc::{cumsum,cumprod}`, both `T -> T`.
//!   - var/std ddof>=n -> inf (not raise): `reductions::var` returns
//!     `Err(invalid_value("ddof >= number of elements"))`; numpy clamps
//!     `rcount = max(n-ddof, 0)` -> 0 and divides -> +inf
//!     (numpy/_core/_methods.py:204).
//!
//! Numpy oracle commands (reproduce expected values):
//!   np.cumsum(np.array([100,100,100],dtype=np.int8))         -> [100 200 300] int64
//!   np.cumprod(np.array([5,5,5,5],dtype=np.int8))            -> [5 25 125 625] int64
//!   np.sum(np.array([100,100,100],dtype=np.int8))            -> 300 int64
//!   np.prod(np.array([5,5,5,5,5,5],dtype=np.int8))           -> 15625 int64
//!   np.var([1.,2.,3.], ddof=3)                               -> inf
//!   np.std([1.,2.,3.], ddof=3)                               -> inf

use std::panic::{AssertUnwindSafe, catch_unwind};

use ferray_core::{Array, Ix1};
use ferray_stats::{cumprod, cumsum, prod, std_, sum, var};

// ---------------------------------------------------------------------------
// DIVERGENCE 1 — sum(int8) overflow: must promote accumulator+result to int64.
//
// numpy contract: numpy/_core/fromnumeric.py:2321-2327 (sum docstring) —
//   "if `a` has an integer dtype of less precision than the default platform
//    integer ... the platform integer is used". So np.sum(int8) -> int64,
//   value never overflows.
// Live oracle: np.sum(np.array([100,100,100],dtype=np.int8)) == 300 (int64).
//
// ferray: `sum::<i8>` accumulates in i8. 100+100+100 overflows i8 (max 127):
//   debug build => panic ("attempt to add with overflow");
//   release build => wraps to 44. Either way != 300, and the result dtype is
//   i8 not i64. Caught with catch_unwind so the harness reports the divergence
//   instead of aborting.
// ---------------------------------------------------------------------------
#[test]
fn divergence_sum_int8_promotes_to_int64() {
    // Live numpy: np.sum(np.array([100,100,100],dtype=np.int8)) -> 300
    const NUMPY_SUM: i64 = 300;

    let a = Array::<i8, Ix1>::from_vec(Ix1::new([3]), vec![100, 100, 100]).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| {
        // sum now promotes i8 -> i64; the result element is already `i64`, and
        // the explicit annotation pins the promoted dtype.
        let r = sum(&a, None).unwrap();
        let v: i64 = *r.iter().next().unwrap();
        v
    }));

    match got {
        Ok(v) => assert_eq!(
            v, NUMPY_SUM,
            "sum(int8 [100,100,100]) wrapped/overflowed to {v}; numpy promotes to int64 = {NUMPY_SUM}"
        ),
        Err(_) => panic!(
            "sum(int8 [100,100,100]) PANICKED on i8 overflow; numpy promotes to int64 = {NUMPY_SUM}"
        ),
    }
}

// ---------------------------------------------------------------------------
// DIVERGENCE 2 — prod(int8) overflow: must promote accumulator+result to int64.
//
// numpy contract: numpy/_core/fromnumeric.py:3306-3312 (prod docstring), same
//   narrow-int promotion rule as sum.
// Live oracle: np.prod(np.array([5,5,5,5,5,5],dtype=np.int8)) == 15625 (int64).
//
// ferray: `prod::<i8>` accumulates in i8. 5^6 = 15625 overflows i8 — debug
//   panic / release wrap. Result dtype i8 not i64.
// ---------------------------------------------------------------------------
#[test]
fn divergence_prod_int8_promotes_to_int64() {
    // Live numpy: np.prod(np.array([5,5,5,5,5,5],dtype=np.int8)) -> 15625
    const NUMPY_PROD: i64 = 15625;

    let a = Array::<i8, Ix1>::from_vec(Ix1::new([6]), vec![5, 5, 5, 5, 5, 5]).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| {
        // prod now promotes i8 -> i64; the result element is already `i64`.
        let r = prod(&a, None).unwrap();
        let v: i64 = *r.iter().next().unwrap();
        v
    }));

    match got {
        Ok(v) => assert_eq!(
            v, NUMPY_PROD,
            "prod(int8 5^6) wrapped/overflowed to {v}; numpy promotes to int64 = {NUMPY_PROD}"
        ),
        Err(_) => {
            panic!("prod(int8 5^6) PANICKED on i8 overflow; numpy promotes to int64 = {NUMPY_PROD}")
        }
    }
}

// ---------------------------------------------------------------------------
// DIVERGENCE 3 — cumsum(int8) overflow: must promote to int64.
//
// numpy contract: numpy/_core/fromnumeric.py:2854 (cumsum) — accumulator
//   promotion identical to sum. Live oracle:
//   np.cumsum(np.array([100,100,100],dtype=np.int8)) == [100,200,300] int64.
//
// ferray: `cumsum::<i8>` (re-export of ferray_ufunc::cumsum, `T->T`, fold
//   `a+b` on i8). Second step 100+100=200 overflows i8 — debug panic / release
//   wrap (-56). Result dtype i8 not int64.
// ---------------------------------------------------------------------------
#[test]
fn divergence_cumsum_int8_promotes_to_int64() {
    // Live numpy: np.cumsum(np.array([100,100,100],dtype=np.int8)) -> [100,200,300]
    const NUMPY_CUMSUM: [i64; 3] = [100, 200, 300];

    let a = Array::<i8, Ix1>::from_vec(Ix1::new([3]), vec![100, 100, 100]).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| {
        // cumsum now promotes i8 -> i64; collecting into `Vec<i64>` also pins
        // the promoted result dtype (a non-promoting cumsum would be `Vec<i8>`
        // and fail to type-check here).
        let r = cumsum(&a, None).unwrap();
        r.iter().copied().collect::<Vec<i64>>()
    }));

    match got {
        Ok(v) => assert_eq!(
            v.as_slice(),
            NUMPY_CUMSUM.as_slice(),
            "cumsum(int8 [100,100,100]) = {v:?}; numpy promotes to int64 = {NUMPY_CUMSUM:?}"
        ),
        Err(_) => {
            panic!("cumsum(int8 [100,100,100]) PANICKED on i8 overflow; numpy = {NUMPY_CUMSUM:?}")
        }
    }
}

// ---------------------------------------------------------------------------
// DIVERGENCE 4 — cumprod(int8) overflow: must promote to int64.
//
// numpy contract: numpy/_core/fromnumeric.py (cumprod) — same promotion.
// Live oracle: np.cumprod(np.array([5,5,5,5],dtype=np.int8)) == [5,25,125,625] int64.
//
// ferray: `cumprod::<i8>` (ferray_ufunc::cumprod, `T->T`). Fourth step
//   125*5=625 overflows i8 — debug panic / release wrap. dtype i8 not int64.
// ---------------------------------------------------------------------------
#[test]
fn divergence_cumprod_int8_promotes_to_int64() {
    // Live numpy: np.cumprod(np.array([5,5,5,5],dtype=np.int8)) -> [5,25,125,625]
    const NUMPY_CUMPROD: [i64; 4] = [5, 25, 125, 625];

    let a = Array::<i8, Ix1>::from_vec(Ix1::new([4]), vec![5, 5, 5, 5]).unwrap();
    let got = catch_unwind(AssertUnwindSafe(|| {
        // cumprod now promotes i8 -> i64; the `Vec<i64>` collect also pins the
        // promoted result dtype.
        let r = cumprod(&a, None).unwrap();
        r.iter().copied().collect::<Vec<i64>>()
    }));

    match got {
        Ok(v) => assert_eq!(
            v.as_slice(),
            NUMPY_CUMPROD.as_slice(),
            "cumprod(int8 [5,5,5,5]) = {v:?}; numpy promotes to int64 = {NUMPY_CUMPROD:?}"
        ),
        Err(_) => {
            panic!("cumprod(int8 [5,5,5,5]) PANICKED on i8 overflow; numpy = {NUMPY_CUMPROD:?}")
        }
    }
}

// ---------------------------------------------------------------------------
// DIVERGENCE 5 — var(ddof >= n) must return +inf, not raise.
//
// numpy contract: numpy/_core/_methods.py:204 —
//   `rcount = um.maximum(rcount - ddof, 0)` then true_divide(ret, rcount):
//   when ddof >= n, rcount clamps to 0, division by zero yields +inf (plus a
//   RuntimeWarning). numpy does NOT raise.
// Live oracle: np.var([1.,2.,3.], ddof=3) == inf ; ddof=5 == inf.
//
// ferray: `reductions::var` returns
//   Err(invalid_value("ddof >= number of elements, variance undefined"))
//   for n <= ddof — a hard error where numpy returns a finite-typed inf.
// ---------------------------------------------------------------------------
#[test]
fn divergence_var_ddof_ge_n_returns_inf() {
    // Live numpy: np.var([1.,2.,3.], ddof=3) -> inf (RuntimeWarning, no raise)
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();

    let r = var(&a, None, 3).expect(
        "var(ddof=3) returned Err; numpy returns +inf (numpy/_core/_methods.py:204 clamps \
         rcount to 0 then true_divide -> inf)",
    );
    let v = *r.iter().next().unwrap();
    assert!(
        v.is_infinite() && v > 0.0,
        "var([1,2,3], ddof=3) = {v}; numpy = +inf"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 6 — std(ddof >= n) must return +inf, not raise.
//
// numpy contract: numpy/_core/_methods.py:217-225 — std = sqrt(_var(...)),
//   and _var returns inf for ddof>=n; sqrt(inf) == inf.
// Live oracle: np.std([1.,2.,3.], ddof=3) == inf.
//
// ferray: `std_` calls `var(a, axis, ddof)?` which Errs for ddof>=n, so std_
//   propagates the same hard error instead of returning inf.
// ---------------------------------------------------------------------------
#[test]
fn divergence_std_ddof_ge_n_returns_inf() {
    // Live numpy: np.std([1.,2.,3.], ddof=3) -> inf
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();

    let r = std_(&a, None, 3).expect(
        "std(ddof=3) returned Err; numpy returns +inf (sqrt of _var's inf, \
         numpy/_core/_methods.py:217-225)",
    );
    let v = *r.iter().next().unwrap();
    assert!(
        v.is_infinite() && v > 0.0,
        "std([1,2,3], ddof=3) = {v}; numpy = +inf"
    );
}

// ===========================================================================
// NaN-propagation divergences (acto-critic iteration, 2026-05-31).
//
// Imports kept local to this block so the original header (DIV 1-6) stays
// untouched. numpy oracle: numpy 2.4.4 via ferray-python/.venv/bin/python.
// ===========================================================================
use ferray_core::Ix2;
use ferray_stats::{argmax, argmin, max, min, ptp};

// ---------------------------------------------------------------------------
// DIVERGENCE 7 — max(NaN present) must PROPAGATE NaN, not drop it.
//
// numpy contract: numpy/_core/fromnumeric.py:3076-3078 (max docstring) —
//   "NaN values are propagated, that is if at least one item is NaN, the
//    corresponding max value will be NaN as well. To ignore NaN values
//    (MATLAB behavior), please use nanmax." np.max is _wrapreduction over
//   np.maximum (fromnumeric.py:3124-3125), and np.maximum propagates NaN.
// Live oracle: np.max(np.array([1.0, np.nan, 3.0])) -> nan.
//
// ferray: `reductions::max` nan_max(a,b) returns `a` in the "unordered"
//   else-branch (mod.rs:665-673). Reducing [1, nan, 3] folds
//   nan_max(nan_max(1,nan)=1, 3)=3 — the NaN is silently dropped, giving 3.
// ---------------------------------------------------------------------------
#[test]
fn divergence_max_propagates_nan() {
    // Live numpy: np.max([1.0, nan, 3.0]) -> nan
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    let r = max(&a, None).unwrap();
    let v = *r.iter().next().unwrap();
    assert!(
        v.is_nan(),
        "max([1, nan, 3]) = {v}; numpy propagates NaN -> nan \
         (numpy/_core/fromnumeric.py:3076)"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 8 — min(NaN present) must PROPAGATE NaN, not drop it.
//
// numpy contract: numpy/_core/fromnumeric.py:3214-3216 (min docstring) —
//   "NaN values are propagated, that is if at least one item is NaN, the
//    corresponding min value will be NaN as well ... please use nanmin."
//   np.min is _wrapreduction over np.minimum, which propagates NaN.
// Live oracle: np.min(np.array([1.0, np.nan, 3.0])) -> nan ;
//              np.min(np.array([np.nan, 1.0, 3.0])) -> nan.
//
// ferray: `reductions::min` nan_min else-branch returns `a` (mod.rs:622-632),
//   so the NaN is dropped whenever a non-NaN precedes it in the fold.
// ---------------------------------------------------------------------------
#[test]
fn divergence_min_propagates_nan() {
    // Live numpy: np.min([1.0, nan, 3.0]) -> nan
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    let r = min(&a, None).unwrap();
    let v = *r.iter().next().unwrap();
    assert!(
        v.is_nan(),
        "min([1, nan, 3]) = {v}; numpy propagates NaN -> nan \
         (numpy/_core/fromnumeric.py:3214)"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 9 — min/max along an axis must propagate NaN per lane.
//
// numpy contract: same NaN-propagation rule (fromnumeric.py:3076 / 3214)
//   applies lane-wise.
// Live oracle:
//   m = [[1, nan, 3], [4, 5, nan]]
//   np.min(m, axis=1) -> [nan, nan]
//   np.max(m, axis=1) -> [nan, nan]
// ---------------------------------------------------------------------------
#[test]
fn divergence_min_max_axis_propagate_nan() {
    let m = Array::<f64, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN],
    )
    .unwrap();
    let lo = min(&m, Some(1)).unwrap();
    let hi = max(&m, Some(1)).unwrap();
    let lo_v: Vec<f64> = lo.iter().copied().collect();
    let hi_v: Vec<f64> = hi.iter().copied().collect();
    // numpy: both lanes contain a NaN -> [nan, nan] for min and max.
    assert!(
        lo_v.iter().all(|x| x.is_nan()),
        "min(axis=1) = {lo_v:?}; numpy = [nan, nan] (fromnumeric.py:3214)"
    );
    assert!(
        hi_v.iter().all(|x| x.is_nan()),
        "max(axis=1) = {hi_v:?}; numpy = [nan, nan] (fromnumeric.py:3076)"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 10 — ptp(NaN present) must be NaN.
//
// numpy contract: np.ptp = max - min (fromnumeric.py:2922 def ptp), and both
//   max and min propagate NaN, so ptp of a slice containing NaN is NaN.
// Live oracle: np.ptp(np.array([1.0, np.nan, 3.0])) -> nan.
//
// ferray: `reductions::ptp` = max(a) - min(a) (mod.rs:814-823); because the
//   broken nan_min/nan_max drop NaN, ptp returns 3 - 1 = 2 instead of NaN.
// ---------------------------------------------------------------------------
#[test]
fn divergence_ptp_propagates_nan() {
    // Live numpy: np.ptp([1.0, nan, 3.0]) -> nan
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    let r = ptp(&a, None).unwrap();
    let v = *r.iter().next().unwrap();
    assert!(
        v.is_nan(),
        "ptp([1, nan, 3]) = {v}; numpy = nan (max - min, both propagate NaN)"
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE 11 — argmin with a NaN returns the FIRST NaN's index.
//
// numpy contract: argmin reduces with np.minimum, which propagates NaN, so a
//   NaN compares as "the smallest" and argmin returns the index of the first
//   NaN. First-occurrence tie rule: numpy/_core/fromnumeric.py:1362
//   ("the indices corresponding to the first occurrence are returned").
// Live oracle:
//   np.argmin([1.0, nan, 3.0]) -> 1
//   np.argmin([nan, 1.0, 3.0]) -> 0
//
// ferray: `reductions::argmin` reduces with `if av <= bv` (mod.rs:716). With a
//   NaN, `av <= bv` is false, so it walks PAST the NaN to the next element,
//   yielding 2 (for [1,nan,3]) and 1 (for [nan,1,3]) — neither equals numpy.
// ---------------------------------------------------------------------------
#[test]
fn divergence_argmin_returns_first_nan_index() {
    // Live numpy: np.argmin([1.0, nan, 3.0]) -> 1
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    let got = *argmin(&a, None).unwrap().iter().next().unwrap();
    assert_eq!(
        got, 1u64,
        "argmin([1, nan, 3]) = {got}; numpy = 1 (NaN propagates as the minimum, \
         fromnumeric.py:1362)"
    );

    // Live numpy: np.argmin([nan, 1.0, 3.0]) -> 0
    let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![f64::NAN, 1.0, 3.0]).unwrap();
    let got_b = *argmin(&b, None).unwrap().iter().next().unwrap();
    assert_eq!(got_b, 0u64, "argmin([nan, 1, 3]) = {got_b}; numpy = 0");
}

// ---------------------------------------------------------------------------
// DIVERGENCE 12 — argmax with a NaN returns the FIRST NaN's index.
//
// numpy contract: argmax reduces with np.maximum (propagates NaN), so a NaN
//   compares as "the largest" and argmax returns the index of the first NaN.
//   First-occurrence tie rule: numpy/_core/fromnumeric.py:1262.
// Live oracle:
//   np.argmax([1.0, nan, 3.0]) -> 1
//   np.argmax([nan, 1.0, 3.0]) -> 0
//
// ferray: `reductions::argmax` reduces with `if av >= bv` (mod.rs:756); the
//   NaN comparison is false so it skips past the NaN -> 2 (for [1,nan,3]) and
//   2 (for [nan,1,3]).
// ---------------------------------------------------------------------------
#[test]
fn divergence_argmax_returns_first_nan_index() {
    // Live numpy: np.argmax([1.0, nan, 3.0]) -> 1
    let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
    let got = *argmax(&a, None).unwrap().iter().next().unwrap();
    assert_eq!(
        got, 1u64,
        "argmax([1, nan, 3]) = {got}; numpy = 1 (NaN propagates as the maximum, \
         fromnumeric.py:1262)"
    );

    // Live numpy: np.argmax([nan, 1.0, 3.0]) -> 0
    let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![f64::NAN, 1.0, 3.0]).unwrap();
    let got_b = *argmax(&b, None).unwrap().iter().next().unwrap();
    assert_eq!(got_b, 0u64, "argmax([nan, 1, 3]) = {got_b}; numpy = 0");
}

// ---------------------------------------------------------------------------
// DIVERGENCE 13 — argmin/argmax along an axis return the first-NaN index per
// lane.
//
// Live oracle:
//   m = [[1, nan, 3], [4, 5, nan]]
//   np.argmin(m, axis=1) -> [1, 2]
//   np.argmax(m, axis=1) -> [1, 2]
// ---------------------------------------------------------------------------
#[test]
fn divergence_argmin_argmax_axis_first_nan() {
    let m = Array::<f64, Ix2>::from_vec(
        Ix2::new([2, 3]),
        vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN],
    )
    .unwrap();
    let amin: Vec<u64> = argmin(&m, Some(1)).unwrap().iter().copied().collect();
    let amax: Vec<u64> = argmax(&m, Some(1)).unwrap().iter().copied().collect();
    // numpy: NaN is the extremum in each lane -> index of the NaN.
    assert_eq!(
        amin,
        vec![1u64, 2u64],
        "argmin(axis=1) = {amin:?}; numpy = [1, 2]"
    );
    assert_eq!(
        amax,
        vec![1u64, 2u64],
        "argmax(axis=1) = {amax:?}; numpy = [1, 2]"
    );
}
