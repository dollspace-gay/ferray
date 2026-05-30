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
