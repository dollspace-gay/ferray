//! ACToR critic — pinned LIBRARY divergences of `ferray-ma` (Rust API)
//! from `numpy.ma`. Each expected value is traceable to a numpy
//! `ma/core.py file:line` symbolic constant OR a live `numpy.ma` oracle
//! call (recorded inline), NEVER literal-copied from the ferray side
//! (R-CHAR-3).
//!
//! Oracle session (numpy 2.4.5):
//!   >>> import numpy.ma as ma
//!   >>> ma.array([1.,2.,3.]).fill_value            -> np.float64(1e+20)
//!   >>> am = ma.array([1.,2.,3.,4.], mask=[1,1,1,1])
//!   >>> am.sum() is ma.masked                      -> True
//!   >>> am.sum().filled() ... (masked singleton)
//!   >>> ma.array([1.,2.,3.], mask=[0,1,0]).filled().tolist() -> [1.0, 1e+20, 3.0]
//!
//! These pin OBSERVABLE-VALUE divergences in the f64 model. The
//! `masked`/`nomask` Python singletons have no direct Rust analog; the
//! observable consequence (what `filled()`/`sum()` yields) is pinned.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_ma::MaskedArray;

fn ma1(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
    let n = data.len();
    let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
    let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask).unwrap();
    MaskedArray::new(d, m).unwrap()
}

/// Divergence: `ferray_ma::MaskedArray::fill_value` default diverges from
/// numpy's `default_filler['f']` at `numpy/ma/core.py:166` (`'f': 1.e20`).
///
/// Upstream: `ma.array([1.,2.,3.]).fill_value` == `np.float64(1e+20)`
///           (live oracle, numpy 2.4.5).
/// Target:   `MaskedArray::new(...).fill_value()` == `0.0`
///           (`masked_array.rs:121` sets `fill_value: T::zero()`).
///
/// The `1e20` literal here is the numpy `default_filler['f']` symbolic
/// constant, NOT copied from ferray (ferray returns 0.0).
#[test]
fn divergence_default_fill_value_float64_is_1e20() {
    let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, false, false]);
    // numpy/ma/core.py:166  default_filler 'f': 1.e20
    let numpy_default_filler_f: f64 = 1.0e20;
    assert_eq!(
        ma.fill_value(),
        numpy_default_filler_f,
        "ferray default fill_value diverges from numpy default_filler['f']=1e20"
    );
}

/// Divergence: `filled()` with the DEFAULT fill_value substitutes the
/// wrong value at masked positions.
///
/// Upstream: `ma.array([1.,2.,3.], mask=[0,1,0]).filled().tolist()`
///           == `[1.0, 1e+20, 3.0]` (live oracle; masked slot gets the
///           default fill_value `default_filler['f']`, core.py:166).
/// Target:   `ma.filled_default()` == `[1.0, 0.0, 3.0]`
///           (`filled.rs:39` + default fill_value `T::zero()`).
#[test]
fn divergence_filled_default_uses_1e20_not_zero() {
    let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
    let filled = ma.filled_default().unwrap();
    // Expected from live numpy.ma: masked slot -> default_filler['f'] = 1e20.
    let numpy_default_filler_f: f64 = 1.0e20;
    assert_eq!(
        filled.as_slice().unwrap(),
        &[1.0, numpy_default_filler_f, 3.0],
        "filled() default must substitute numpy default fill_value 1e20, not 0.0"
    );
}

/// Divergence: all-masked `sum()` diverges from numpy's `masked`-singleton
/// reduction path at `numpy/ma/core.py:5249` (`elif newmask: result = masked`).
///
/// Upstream: `am = ma.array([1.,2.,3.,4.], mask=[1,1,1,1]); am.sum()`
///           returns the `ma.masked` singleton (`am.sum() is ma.masked`
///           == True). The observable scalar is masked, NOT 0.0; filling
///           it yields the fill_value, not 0.0.
/// Target:   `ma.sum()` returns `Ok(0.0)` (`reductions.rs:243` folds the
///           empty unmasked set to `zero`), losing the masked status.
///
/// The f64 model has no `masked` singleton, but numpy's contract is that
/// an all-masked sum is NOT the additive identity 0.0 — it is masked.
/// This test pins that 0.0 is the WRONG observable for an all-masked sum.
#[test]
fn divergence_all_masked_sum_is_masked_not_zero() {
    let ma = ma1(vec![1.0, 2.0, 3.0, 4.0], vec![true, true, true, true]);
    let s = ma.sum().unwrap();
    // numpy: am.sum() is ma.masked  (NOT the additive identity 0.0).
    // count of unmasked == 0, so there is no defined numeric sum.
    assert_eq!(
        ma.count().unwrap(),
        0,
        "precondition: array is fully masked"
    );
    assert!(
        s.is_nan(),
        "all-masked sum must NOT be the bare additive identity 0.0 \
         (numpy returns the `masked` singleton, core.py:5249); \
         got {s}"
    );
}
