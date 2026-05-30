//! Divergence pins: `var_into`/`std_into` (and regression pins for
//! `nanvar`/`nanstd`) with `ddof >= n` vs numpy 2.4.5.
//!
//! Sibling of blocker #794 (which fixed `var`/`std_`). Authored as a
//! pin-then-fix for blocker #795.
//!
//! R-CHAR-3: every expected value is traced to a live numpy 2.4.5 call /
//! a numpy `file:line`, NEVER copied from the ferray side.
//!
//! Numpy contracts:
//!
//! `var`/`var_into` (=`numpy.var`, `_core/_methods.py:_var` ~L204):
//! `rcount = um.maximum(rcount - ddof, 0)` then true-divide. So `ddof >= n`
//! clamps the divisor to 0 and IEEE-754 float division yields +inf
//! (sum_sq > 0) or NaN (sum_sq == 0) — NEVER an error. Live:
//! `np.var([1.,2.,3.], ddof=3) == inf`, `np.var([1.,2.,3.], ddof=5) == inf`,
//! `np.var([5.,5.,5.], ddof=3) == nan` (sum_sq == 0).
//!
//! `nanvar`/`nanstd` (=`numpy.nanvar`, `lib/_nanfunctions_impl.py:1864-1873`):
//! `dof = cnt - ddof; var = var/dof; isbad = (dof <= 0);
//! var = _copyto(var, np.nan, isbad)` — numpy EXPLICITLY replaces the
//! inf/neg/nan result with NaN wherever `dof <= 0`. So nanvar/nanstd return
//! NaN (NOT inf) for `ddof >= valid-count`. Live:
//! `np.nanvar([1.,2.,3.], ddof=3) == nan`,
//! `np.nanvar([1.,2.,3.], ddof=5) == nan`,
//! `np.nanstd([1.,2.,3.], ddof=3) == nan`,
//! `np.nanvar([1.,2.,nan], ddof=2) == nan`.
//!
//! Surface paths exercised (named for the surface gate):
//!   - `ferray_stats::reductions::var_into`
//!   - `ferray_stats::reductions::std_into`
//!   - `ferray_stats::reductions::nanvar`
//!   - `ferray_stats::reductions::nanstd`

use ferray_core::{Array, Ix1, IxDyn};
use ferray_stats::{nanstd, nanvar, std_into, var_into};

fn arr1(v: Vec<f64>) -> Array<f64, Ix1> {
    Array::<f64, Ix1>::from_vec(Ix1::new([v.len()]), v).unwrap()
}

fn scalar_out() -> Array<f64, IxDyn> {
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap()
}

// ---------------------------------------------------------------------------
// DIVERGENCE — var_into ddof>=n must clamp the divisor to 0 and divide ->
// +inf (sum_sq > 0), NOT return Err. Mirrors the #794 fix to `var`.
// Live numpy: np.var([1.,2.,3.], ddof=3) == inf
// ---------------------------------------------------------------------------
#[test]
fn divergence_var_into_ddof_eq_n_is_inf() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let mut out = scalar_out();
    var_into(&a, None, 3, &mut out).expect("var_into must not error for ddof>=n (numpy: inf)");
    let got: f64 = *out.iter().next().unwrap();
    assert!(
        got.is_infinite() && got.is_sign_positive(),
        "var_into([1,2,3], ddof=3) = {got}, expected +inf (np.var(...,ddof=3)==inf)"
    );
}

// Live numpy: np.var([1.,2.,3.], ddof=5) == inf
#[test]
fn divergence_var_into_ddof_gt_n_is_inf() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let mut out = scalar_out();
    var_into(&a, None, 5, &mut out).expect("var_into must not error for ddof>n (numpy: inf)");
    let got: f64 = *out.iter().next().unwrap();
    assert!(
        got.is_infinite() && got.is_sign_positive(),
        "var_into([1,2,3], ddof=5) = {got}, expected +inf (np.var(...,ddof=5)==inf)"
    );
}

// Live numpy: np.var([5.,5.,5.], ddof=3) == nan  (sum_sq == 0 -> 0/0 = nan)
#[test]
fn divergence_var_into_ddof_eq_n_zero_numerator_is_nan() {
    let a = arr1(vec![5.0, 5.0, 5.0]);
    let mut out = scalar_out();
    var_into(&a, None, 3, &mut out).expect("var_into must not error for ddof>=n (numpy: nan)");
    let got: f64 = *out.iter().next().unwrap();
    assert!(
        got.is_nan(),
        "var_into([5,5,5], ddof=3) = {got}, expected NaN (np.var(...,ddof=3)==nan)"
    );
}

// std_into delegates to var_into, so it inherits the clamp. std of +inf
// variance is +inf. Live numpy: np.std([1.,2.,3.], ddof=3) == inf
#[test]
fn divergence_std_into_ddof_eq_n_is_inf() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let mut out = scalar_out();
    std_into(&a, None, 3, &mut out).expect("std_into must not error for ddof>=n (numpy: inf)");
    let got: f64 = *out.iter().next().unwrap();
    assert!(
        got.is_infinite() && got.is_sign_positive(),
        "std_into([1,2,3], ddof=3) = {got}, expected +inf (np.std(...,ddof=3)==inf)"
    );
}

// Axis variant: np.var([[1.,2.,3.]], axis=1, ddof=3) == [inf]
#[test]
fn divergence_var_into_axis_ddof_eq_axislen_is_inf() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![0.0]).unwrap();
    var_into(&a, Some(1), 3, &mut out).expect("var_into axis must not error for ddof>=axislen");
    let got: f64 = *out.iter().next().unwrap();
    assert!(
        got.is_infinite() && got.is_sign_positive(),
        "var_into([[1,2,3]], axis=1, ddof=3) = {got}, expected +inf"
    );
}

// ---------------------------------------------------------------------------
// REGRESSION pins — nanvar/nanstd return NaN (NOT inf) for ddof >= valid-count.
// numpy lib/_nanfunctions_impl.py:1864-1873 explicitly _copyto(var, nan, dof<=0).
// These lock the already-correct ferray behavior so a future "make it inf"
// edit can't silently regress it.
// ---------------------------------------------------------------------------

// Live numpy: np.nanvar([1.,2.,3.], ddof=3) == nan
#[test]
fn nanvar_ddof_eq_count_is_nan() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let v = nanvar(&a, None, 3).expect("nanvar must not error for ddof>=count");
    let got: f64 = *v.iter().next().unwrap();
    assert!(
        got.is_nan(),
        "nanvar([1,2,3], ddof=3) = {got}, expected NaN (np.nanvar(...,ddof=3)==nan)"
    );
}

// Live numpy: np.nanvar([1.,2.,3.], ddof=5) == nan
#[test]
fn nanvar_ddof_gt_count_is_nan() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let v = nanvar(&a, None, 5).expect("nanvar must not error for ddof>count");
    let got: f64 = *v.iter().next().unwrap();
    assert!(
        got.is_nan(),
        "nanvar([1,2,3], ddof=5) = {got}, expected NaN (np.nanvar(...,ddof=5)==nan)"
    );
}

// Live numpy: np.nanstd([1.,2.,3.], ddof=3) == nan
#[test]
fn nanstd_ddof_eq_count_is_nan() {
    let a = arr1(vec![1.0, 2.0, 3.0]);
    let v = nanstd(&a, None, 3).expect("nanstd must not error for ddof>=count");
    let got: f64 = *v.iter().next().unwrap();
    assert!(
        got.is_nan(),
        "nanstd([1,2,3], ddof=3) = {got}, expected NaN (np.nanstd(...,ddof=3)==nan)"
    );
}

// Live numpy: np.nanvar([1.,2.,nan], ddof=2) == nan  (valid-count 2, ddof 2)
#[test]
fn nanvar_with_nan_ddof_eq_validcount_is_nan() {
    let a = arr1(vec![1.0, 2.0, f64::NAN]);
    let v = nanvar(&a, None, 2).expect("nanvar must not error for ddof>=valid-count");
    let got: f64 = *v.iter().next().unwrap();
    assert!(
        got.is_nan(),
        "nanvar([1,2,nan], ddof=2) = {got}, expected NaN (np.nanvar(...,ddof=2)==nan)"
    );
}
