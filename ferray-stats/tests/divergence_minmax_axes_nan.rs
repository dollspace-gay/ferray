//! Divergence pins: `ferray_stats::reductions::{min_axes, max_axes}` must
//! PROPAGATE NaN along the reduced axis/axes, matching `numpy.min`/`numpy.max`.
//!
//! Authored by acto-critic (2026-05-31). The inline `nan_min`/`nan_max`
//! closures in `reductions/mod.rs` (~:1279 / :1311) carry the OLD broken
//! else-branch that DROPS NaN: a finite value PRECEDING a NaN wins the
//! reduction, so a NaN that is not the first element of a lane is silently
//! discarded. This is the same bug already fixed in the scalar `min`/`max`
//! reductions (#1058 — see `divergence_reductions.rs::DIVERGENCE 7/8`), but
//! `min_axes`/`max_axes` were not updated.
//!
//! numpy contract: any reduction whose reduced slice contains a NaN yields
//! NaN along that axis (NaN propagation through the `<`/`>` comparisons of
//! `np.minimum`/`np.maximum`; numpy/_core/fromnumeric.py:83 `ufunc.reduce(...)`
//! over the `minimum`/`maximum` ufuncs, whose binary op returns NaN if either
//! operand is NaN).
//!
//! Expected values taken from LIVE numpy 2.4.4 (ferray-python/.venv/bin/python):
//!   m = [[1., nan], [3., 4.]]
//!   np.max(m, axis=1) -> [nan, 4.]   (lane [1,nan]: NaN wins; lane [3,4]: 4)
//!   np.max(m, axis=0) -> [3.,  nan]  (lane [nan,4]: NaN wins)
//!   np.min(m, axis=1) -> [nan, 3.]
//!   np.min(m, axis=0) -> [1.,  nan]
//!   np.min([1., nan]) -> nan ; np.max([1., nan]) -> nan
//!
//! The broken `nan_min(1.0, NaN)`: `1.0 <= NaN` is false, `1.0 > NaN` is
//! false, so the `else` arm returns `a == 1.0` — NaN DROPPED. Mirror for
//! `nan_max`. NaN-FIRST lanes happen to propagate (the `else` returns the
//! NaN `a`), so only the finite-then-NaN lanes expose the divergence.

use ferray_core::{Array, Ix2};
use ferray_stats::reductions::{max_axes, min_axes};

// ---------------------------------------------------------------------------
// DIVERGENCE — max_axes drops a NaN that follows a finite value in the lane.
//
// Live numpy: np.max([[1, nan],[3, 4]], axis=1) -> [nan, 4.]
//             np.max([[1, nan],[3, 4]], axis=0) -> [3., nan]
// ---------------------------------------------------------------------------
#[test]
fn divergence_max_axes_propagates_nan() {
    let m =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();

    // axis=1: lane [1, nan] -> nan ; lane [3, 4] -> 4
    let r1: Vec<f64> = max_axes(&m, Some(&[1]), false)
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert!(
        r1[0].is_nan(),
        "max_axes(axis=1)[0] = {}; numpy np.max([[1,nan],[3,4]],axis=1)[0] = nan \
         (finite-then-NaN lane drops NaN — reductions/mod.rs nan_max else-branch)",
        r1[0]
    );
    assert_eq!(
        r1[1], 4.0,
        "max_axes(axis=1)[1] = {}; numpy = 4.0",
        r1[1]
    );

    // axis=0: lane [1, 3] -> 3 ; lane [nan, 4] -> nan (NaN-first, still nan)
    let r0: Vec<f64> = max_axes(&m, Some(&[0]), false)
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(r0[0], 3.0, "max_axes(axis=0)[0] = {}; numpy = 3.0", r0[0]);
    assert!(
        r0[1].is_nan(),
        "max_axes(axis=0)[1] = {}; numpy = nan",
        r0[1]
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE — min_axes drops a NaN that follows a finite value in the lane.
//
// Live numpy: np.min([[1, nan],[3, 4]], axis=1) -> [nan, 3.]
//             np.min([[1, nan],[3, 4]], axis=0) -> [1., nan]
// ---------------------------------------------------------------------------
#[test]
fn divergence_min_axes_propagates_nan() {
    let m =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, f64::NAN, 3.0, 4.0]).unwrap();

    // axis=1: lane [1, nan] -> nan ; lane [3, 4] -> 3
    let r1: Vec<f64> = min_axes(&m, Some(&[1]), false)
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert!(
        r1[0].is_nan(),
        "min_axes(axis=1)[0] = {}; numpy np.min([[1,nan],[3,4]],axis=1)[0] = nan \
         (finite-then-NaN lane drops NaN — reductions/mod.rs nan_min else-branch)",
        r1[0]
    );
    assert_eq!(
        r1[1], 3.0,
        "min_axes(axis=1)[1] = {}; numpy = 3.0",
        r1[1]
    );

    // axis=0: lane [1, 3] -> 1 ; lane [nan, 4] -> nan (NaN-first, still nan)
    let r0: Vec<f64> = min_axes(&m, Some(&[0]), false)
        .unwrap()
        .iter()
        .copied()
        .collect();
    assert_eq!(r0[0], 1.0, "min_axes(axis=0)[0] = {}; numpy = 1.0", r0[0]);
    assert!(
        r0[1].is_nan(),
        "min_axes(axis=0)[1] = {}; numpy = nan",
        r0[1]
    );
}

// ---------------------------------------------------------------------------
// DIVERGENCE — full reduction (axes=None) over a finite-then-NaN buffer.
//
// Live numpy: np.min([1., nan]) -> nan ; np.max([1., nan]) -> nan
// A single lane [1, nan] (row-major flatten) drops the NaN under the broken
// else-branch, yielding the finite value instead of NaN.
// ---------------------------------------------------------------------------
#[test]
fn divergence_minmax_axes_full_reduction_propagates_nan() {
    // 1x2 so the flattened lane is exactly [1.0, NaN].
    let m = Array::<f64, Ix2>::from_vec(Ix2::new([1, 2]), vec![1.0, f64::NAN]).unwrap();

    let mn = *min_axes(&m, None, false).unwrap().iter().next().unwrap();
    let mx = *max_axes(&m, None, false).unwrap().iter().next().unwrap();
    assert!(
        mn.is_nan(),
        "min_axes(None) over [1, nan] = {mn}; numpy np.min([1., nan]) = nan"
    );
    assert!(
        mx.is_nan(),
        "max_axes(None) over [1, nan] = {mx}; numpy np.max([1., nan]) = nan"
    );
}
