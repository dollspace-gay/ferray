//! Conformance tests for ferray-window's functional utilities.
//!
//! These items are higher-order combinators (vectorize, piecewise,
//! apply_along_axis, apply_over_axes, sum_axis_keepdims). numpy/scipy
//! have direct equivalents only for some of them, and those equivalents
//! are themselves trivial transforms — for example, numpy.vectorize is
//! documented in its own docstring as "provided primarily for convenience,
//! not for performance, ... essentially a for loop". The contract that
//! actually matters for these utilities is the algebraic identity they
//! satisfy: `vectorize(identity)(x) == x`, `apply_along_axis(sum, axis, x)`
//! collapses one axis and yields the same totals as `numpy.sum(x, axis)`,
//! and so on. Each test asserts that contract, and each test mentions the
//! user-facing crate-root path and canonical `ferray_window::functional::*`
//! path so the surface-coverage gate finds both.

use ferray_core::dimension::{Axis, Ix1, Ix2, IxDyn};
use ferray_core::{Array, error::FerrayResult};
use ferray_test_oracle::{TOL_REDUCTION_F64_ABS, assert_close_f64_slice};

// ---------------------------------------------------------------------------
// ferray_window::vectorize
// ---------------------------------------------------------------------------

/// Covers: ferray_window::vectorize, ferray_window::functional::vectorize
///
/// COVERAGE NOTE: numpy.vectorize exists but is documented as a thin "for
/// loop" wrapper around a Python callable; comparing two scalar-mapping
/// for-loops adds no signal. Test asserts the two algebraic identities that
/// define vectorize: (1) `vectorize(identity)(x) == x` (round-trip), and
/// (2) `vectorize(f)(x)` equals `x.mapv(f)` (the underlying primitive).
#[test]
fn vectorize_satisfies_identity_and_mapv_equivalence() {
    // Identity round-trip
    let id = ferray_window::vectorize(|x: f64| x);
    let input =
        Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![-2.0, -0.5, 0.0, 1.5, 3.0]).unwrap();
    let out = id(&input).expect("ferray_window::vectorize identity");
    assert_close_f64_slice(
        out.as_slice().unwrap(),
        input.as_slice().unwrap(),
        0.0,
        TOL_REDUCTION_F64_ABS,
    );

    // mapv equivalence on a non-trivial function
    let f = |x: f64| x.sin() + x.powi(2);
    let vf = ferray_window::vectorize(f);
    let via_vectorize = vf(&input).expect("ferray_window::vectorize map");
    let via_mapv = input.mapv(f);
    assert_close_f64_slice(
        via_vectorize.as_slice().unwrap(),
        via_mapv.as_slice().unwrap(),
        0.0,
        TOL_REDUCTION_F64_ABS,
    );

    // 2-D case: vectorize is generic over dimension
    let m2 =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let sq = ferray_window::vectorize(|x: f64| x * x);
    let out2 = sq(&m2).expect("ferray_window::vectorize 2d");
    assert_eq!(out2.shape(), &[2, 3]);
    assert_close_f64_slice(
        out2.as_slice().unwrap(),
        &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0],
        0.0,
        TOL_REDUCTION_F64_ABS,
    );
}

// ---------------------------------------------------------------------------
// ferray_window::piecewise
// ---------------------------------------------------------------------------

/// Covers: ferray_window::piecewise, ferray_window::functional::piecewise
///
/// Algebraic identity: numpy.piecewise(x, [True_everywhere], [f]) == f(x)
/// elementwise. Also asserts the first-match-wins semantics and the
/// no-condition-met default behaviour, which is the contract numpy.piecewise
/// also implements.
#[test]
fn piecewise_matches_documented_semantics() {
    // Build a sign function: negative -> -1, zero -> 0 (default), positive -> +1
    let x = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![-2.0, -0.5, 0.0, 1.5, 3.0]).unwrap();
    let cond_neg =
        Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, true, false, false, false]).unwrap();
    let cond_pos =
        Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![false, false, false, true, true]).unwrap();
    let neg: &dyn Fn(f64) -> f64 = &|_| -1.0;
    let pos: &dyn Fn(f64) -> f64 = &|_| 1.0;
    let sign = ferray_window::piecewise(&x, &[cond_neg, cond_pos], &[neg, pos], 0.0)
        .expect("ferray_window::piecewise sign");
    assert_close_f64_slice(
        sign.as_slice().unwrap(),
        &[-1.0, -1.0, 0.0, 1.0, 1.0],
        0.0,
        TOL_REDUCTION_F64_ABS,
    );

    // First-match-wins identity: both conds true everywhere -> first applies.
    let c1 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
    let c2 = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
    let y = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    let f1: &dyn Fn(f64) -> f64 = &|v| v * 10.0;
    let f2: &dyn Fn(f64) -> f64 = &|v| v * 100.0;
    let result = ferray_window::piecewise(&y, &[c1, c2], &[f1, f2], 0.0)
        .expect("ferray_window::piecewise first-match");
    assert_close_f64_slice(
        result.as_slice().unwrap(),
        &[10.0, 20.0, 30.0],
        0.0,
        TOL_REDUCTION_F64_ABS,
    );

    // No conditions true -> default everywhere.
    let c_none = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
    let any: &dyn Fn(f64) -> f64 = &|v| v * 10.0;
    let result_default =
        ferray_window::piecewise(&y, &[c_none], &[any], -42.0).expect("piecewise default");
    assert_close_f64_slice(
        result_default.as_slice().unwrap(),
        &[-42.0, -42.0, -42.0],
        0.0,
        TOL_REDUCTION_F64_ABS,
    );
}

// ---------------------------------------------------------------------------
// ferray_window::apply_along_axis
// ---------------------------------------------------------------------------

/// Covers: ferray_window::apply_along_axis,
/// ferray_window::functional::apply_along_axis
///
/// Asserts equivalence with numpy.apply_along_axis when the inner function
/// is `numpy.sum`: applying `sum` along axis 0 of a (2, 3) array yields the
/// column sums; along axis 1 yields the row sums. These outputs are
/// independent of any library-specific implementation detail.
#[test]
fn apply_along_axis_matches_numpy_sum_axis() {
    let m =
        Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // Column sums (axis 0): numpy.sum(m, axis=0) == [5, 7, 9]
    let col_sums = ferray_window::apply_along_axis(
        |col: &Array<f64, Ix1>| -> FerrayResult<f64> { Ok(col.iter().copied().sum::<f64>()) },
        Axis(0),
        &m,
    )
    .expect("ferray_window::apply_along_axis axis=0");
    assert_eq!(col_sums.shape(), &[3]);
    let col_data: Vec<f64> = col_sums.iter().copied().collect();
    assert_close_f64_slice(&col_data, &[5.0, 7.0, 9.0], 0.0, TOL_REDUCTION_F64_ABS);

    // Row sums (axis 1): numpy.sum(m, axis=1) == [6, 15]
    let row_sums = ferray_window::apply_along_axis(
        |row: &Array<f64, Ix1>| -> FerrayResult<f64> { Ok(row.iter().copied().sum::<f64>()) },
        Axis(1),
        &m,
    )
    .expect("ferray_window::apply_along_axis axis=1");
    assert_eq!(row_sums.shape(), &[2]);
    let row_data: Vec<f64> = row_sums.iter().copied().collect();
    assert_close_f64_slice(&row_data, &[6.0, 15.0], 0.0, TOL_REDUCTION_F64_ABS);
}

// ---------------------------------------------------------------------------
// ferray_window::apply_over_axes
// ---------------------------------------------------------------------------

/// Covers: ferray_window::apply_over_axes,
/// ferray_window::functional::apply_over_axes
///
/// numpy.apply_over_axes(numpy.sum, m, [0, 1]) reduces along axis 0 with
/// keepdims, then along axis 1 with keepdims, producing a (1, 1) array with
/// the grand total. Asserts that identity for a (2, 3) array of 1..=6
/// (total 21).
#[test]
fn apply_over_axes_matches_numpy_double_sum() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let total = ferray_window::apply_over_axes(ferray_window::sum_axis_keepdims, &a, &[0, 1])
        .expect("ferray_window::apply_over_axes double sum");
    assert_eq!(total.shape(), &[1, 1]);
    let total_data: Vec<f64> = total.iter().copied().collect();
    assert_close_f64_slice(&total_data, &[21.0], 0.0, TOL_REDUCTION_F64_ABS);

    // Single-axis reduction matches numpy.sum(a, axis=0, keepdims=True)
    let along0 = ferray_window::apply_over_axes(ferray_window::sum_axis_keepdims, &a, &[0])
        .expect("ferray_window::apply_over_axes axis 0");
    assert_eq!(along0.shape(), &[1, 3]);
    let along0_data: Vec<f64> = along0.iter().copied().collect();
    assert_close_f64_slice(&along0_data, &[5.0, 7.0, 9.0], 0.0, TOL_REDUCTION_F64_ABS);
}

// ---------------------------------------------------------------------------
// ferray_window::sum_axis_keepdims
// ---------------------------------------------------------------------------

/// Covers: ferray_window::sum_axis_keepdims,
/// ferray_window::functional::sum_axis_keepdims
///
/// Equivalent to numpy.sum(a, axis=k, keepdims=True). Asserts shape (axis
/// retained as size 1) and values for both axes of a (2, 3) array of 1..=6.
#[test]
fn sum_axis_keepdims_matches_numpy_sum_with_keepdims() {
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    // numpy.sum(a, axis=0, keepdims=True) -> shape (1, 3), [[5, 7, 9]]
    let s0 = ferray_window::sum_axis_keepdims(&a, Axis(0))
        .expect("ferray_window::sum_axis_keepdims axis 0");
    assert_eq!(s0.shape(), &[1, 3]);
    let s0_data: Vec<f64> = s0.iter().copied().collect();
    assert_close_f64_slice(&s0_data, &[5.0, 7.0, 9.0], 0.0, TOL_REDUCTION_F64_ABS);

    // numpy.sum(a, axis=1, keepdims=True) -> shape (2, 1), [[6], [15]]
    let s1 = ferray_window::sum_axis_keepdims(&a, Axis(1))
        .expect("ferray_window::sum_axis_keepdims axis 1");
    assert_eq!(s1.shape(), &[2, 1]);
    let s1_data: Vec<f64> = s1.iter().copied().collect();
    assert_close_f64_slice(&s1_data, &[6.0, 15.0], 0.0, TOL_REDUCTION_F64_ABS);
}
