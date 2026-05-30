//! Divergence pins: ferray-core `manipulation` vs NumPy 2.4.5 (live oracle).
//!
//! Each test asserts the ferray-core result equals the numpy-derived
//! expected value (shape + data). Expected values come from a live numpy
//! call documented in the doc-comment (R-CHAR-3 — never literal-copied
//! from the ferray side). These are FAILING tests pinning real divergence;
//! they go green only when the owning production code is fixed.
//!
//! Surface under audit: `ferray-core/src/manipulation/mod.rs` and
//! `ferray-core/src/manipulation/extended.rs`.

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::manipulation;
use ferray_core::manipulation::extended as ext;

fn dyn_arr(shape: &[usize], data: Vec<f64>) -> Array<f64, IxDyn> {
    Array::from_vec(IxDyn::new(shape), data).unwrap()
}

/// DIVERGENCE 1 — `squeeze` over an all-length-1 array must collapse to a
/// 0-D array (shape `()`, ndim 0), not a 1-D `[1]`.
///
/// numpy oracle (numpy 2.4.5):
///   `np.squeeze(np.ones((1,1,1))).shape == ()`  and `.ndim == 0`
/// Upstream: numpy/_core/fromnumeric.py:1597 `def squeeze(a, axis=None)`
/// — delegates to `ndarray.squeeze`, which removes ALL length-1 axes,
/// yielding a 0-D scalar array when every axis is length 1.
///
/// ferray actual: `manipulation::squeeze(&a, None)` returns shape `[1]`
/// (ndim 1) because `mod.rs` special-cases the empty-result shape to
/// `vec![1]` instead of `vec![]`.
#[test]
fn divergence_squeeze_all_ones_is_zero_d() {
    let a = dyn_arr(&[1, 1, 1], vec![5.0]);
    let r = manipulation::squeeze(&a, None).unwrap();
    // numpy: shape () , ndim 0
    assert_eq!(r.ndim(), 0, "squeeze of (1,1,1) must be 0-D like numpy");
    let empty: &[usize] = &[];
    assert_eq!(r.shape(), empty, "squeeze of (1,1,1) must have shape ()");
}

/// DIVERGENCE 2 — `insert` along an axis with a 1-D `values` array of
/// length matching the inserted slice must insert ONE slice (broadcasting
/// the values across that slice), not `values.len()` slices.
///
/// numpy oracle (numpy 2.4.5):
///   `np.insert([[1,2],[3,4]], 1, [10,20], axis=0)`
///     -> array([[ 1,  2],
///               [10, 20],
///               [ 3,  4]])   shape (3, 2)
/// Upstream: numpy/lib/_function_base_impl.py `def insert(arr, obj, values,
/// axis=None)` — a scalar `obj` inserts a single sub-array; `values` is
/// broadcast to the shape of that sub-array `(2,)`.
///
/// ferray actual: `ext::insert(&a, 1, &vals, 0)` returns shape `[4, 2]`
/// with data `[1,2,10,20,10,20,3,4]` — it interprets `values.size()` (2)
/// as the count of inserted slices and tiles the values, producing two
/// extra rows instead of one.
#[test]
fn divergence_insert_axis0_broadcasts_single_slice() {
    let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let vals = dyn_arr(&[2], vec![10.0, 20.0]);
    let r = ext::insert(&a, 1, &vals, 0).unwrap();
    assert_eq!(r.shape(), &[3, 2], "insert axis=0 must add a single row");
    let data: Vec<f64> = r.iter().copied().collect();
    // numpy: [[1,2],[10,20],[3,4]]
    assert_eq!(data, vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0]);
}
