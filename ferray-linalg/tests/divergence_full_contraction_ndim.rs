//! ACToR critic divergence pins: full-contraction `tensordot` / 1-D `vecdot`
//! must produce a 0-dimensional (scalar) result — shape `()`, `ndim() == 0` —
//! matching NumPy. ferray-linalg instead pads the empty out_shape with a
//! spurious length-1 axis, returning shape `[1]` (`ndim() == 1`).
//!
//! Oracle (numpy 2.4.5, derived live, R-CHAR-3):
//!   np.tensordot([[1,2],[3,4]], [[1,2],[3,4]]).shape == ()   # ndim 0, value 30.0
//!   np.tensordot(...,  axes=1).shape == (2, 2)               # guard: NOT collapsed
//!   np.vecdot([1,2,3], [4,5,6]).shape == ()                  # ndim 0, value 32.0
//!
//! The crate already supports 0-D results (`dot` 1Dx1D builds
//! `IxDyn::new(&[])` at `products/mod.rs:84`), so the 0-D assertion below is
//! constructible — the divergence is purely the `out_shape.push(1)` padding,
//! not a missing 0-D capability.

use ferray_core::array::owned::Array;
use ferray_core::dimension::IxDyn;
use ferray_linalg::{TensordotAxes, tensordot, vecdot};

/// Divergence: `ferray_linalg::tensordot` (full contraction, default
/// `axes=2`) diverges from `numpy/_core/numeric.py:1218`
/// (`return res.reshape(olda + oldb)` — both `olda`/`oldb` empty → shape `()`)
/// for two 2-D matrices.
/// Upstream `np.tensordot([[1,2],[3,4]],[[1,2],[3,4]])` -> shape `()`, ndim 0,
/// value 30.0; ferray returns shape `[1]`, ndim 1
/// (`products/tensordot.rs:138-142`, `out_shape.push(1)`).
/// Tracking: #<crosslink>
#[test]
fn divergence_tensordot_full_contraction_is_0d() {
    // [[1,2],[3,4]] row-major.
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    // Default tensordot is axes=2 (contract last 2 of a with first 2 of b) —
    // a FULL contraction of two 2-D arrays, leaving no free axes.
    let c = tensordot(&a, &b, TensordotAxes::Scalar(2)).unwrap();

    // numpy oracle: scalar result, shape () / ndim 0.
    assert_eq!(c.ndim(), 0, "full tensordot must be 0-D (numpy shape ())");
    assert_eq!(c.shape(), &[] as &[usize], "full tensordot shape must be ()");

    // numpy oracle value: 1*1 + 2*2 + 3*3 + 4*4 = 30.0
    let v: Vec<f64> = c.iter().copied().collect();
    assert_eq!(v.len(), 1, "exactly one scalar element");
    assert!((v[0] - 30.0).abs() < 1e-10, "value must be 30.0, got {}", v[0]);
}

/// Guard (must stay GREEN through any fix): a NON-full tensordot of two 2-D
/// matrices (`axes=1`) still produces a >=1-D result. Mirrors
/// `np.tensordot(a, b, axes=1).shape == (2, 2)`. This pins the fixer against
/// over-collapsing every tensordot to 0-D.
#[test]
fn guard_tensordot_axes1_keeps_2d_shape() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

    let c = tensordot(&a, &b, TensordotAxes::Scalar(1)).unwrap();
    assert_eq!(c.ndim(), 2, "axes=1 tensordot of two 2-D arrays stays 2-D");
    assert_eq!(c.shape(), &[2, 2], "numpy shape (2, 2)");
}

/// Divergence: `ferray_linalg::vecdot` of two 1-D vectors diverges from
/// `numpy.vecdot` (`numpy/linalg/_linalg.py:3604` -> `_core_vecdot`; result
/// shape `()` when the operands are 1-D).
/// Upstream `np.vecdot([1,2,3],[4,5,6])` -> shape `()`, ndim 0, value 32.0;
/// ferray returns shape `[1]`, ndim 1 (`products/mod.rs:1002-1006`,
/// `IxDyn::new(&[1])`).
/// Tracking: #<crosslink>
#[test]
fn divergence_vecdot_1d_is_0d() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![4.0, 5.0, 6.0]).unwrap();

    let c = vecdot(&a, &b, None).unwrap();

    // numpy oracle: scalar result, shape () / ndim 0.
    assert_eq!(c.ndim(), 0, "1-D vecdot must be 0-D (numpy shape ())");
    assert_eq!(c.shape(), &[] as &[usize], "1-D vecdot shape must be ()");

    // numpy oracle value: 1*4 + 2*5 + 3*6 = 32.0
    let v: Vec<f64> = c.iter().copied().collect();
    assert_eq!(v.len(), 1, "exactly one scalar element");
    assert!((v[0] - 32.0).abs() < 1e-10, "value must be 32.0, got {}", v[0]);
}
