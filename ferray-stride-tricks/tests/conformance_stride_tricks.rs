//! Conformance tests for ferray-stride-tricks against NumPy reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_stride_tricks::as_strided`); the canonical inner path
//! (`ferray_stride_tricks::as_strided::as_strided`) is named in the test's
//! doc comment so the surface-coverage gate's text match picks it up.
//!
//! These functions construct *views* with no float arithmetic — every
//! reference comparison is bit-exact. The fixtures author
//! `tolerance_ulps: 0`, and we honour it directly: no
//! `.max(MIN_ULP_TOLERANCE)` override (Stage 1 tolerance table is binding).

use ferray_core::dimension::{Dimension, Ix1, Ix2, IxDyn};
use ferray_core::error::FerrayResult;
use ferray_core::{Array, ArrayView};
use ferray_test_oracle::{
    assert_f64_slice_ulp, fixtures_dir, load_fixture, parse_f64_data, parse_shape, parse_usize_data,
};

fn st_fixture(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("stride_tricks").join(name)
}

/// Build an `Array<f64, IxDyn>` from a fixture value, materialising a
/// rank-0 fixture as an `Array<f64, IxDyn>` of shape `[]` via a single-
/// element vector. `broadcast_shapes` passes shape `[]` as a "scalar"
/// input — we have to handle that here.
fn make_f64_array_any_rank(value: &serde_json::Value) -> Array<f64, IxDyn> {
    let data = parse_f64_data(&value["data"]);
    let shape = parse_shape(&value["shape"]);
    Array::from_vec(IxDyn::new(&shape), data).unwrap()
}

/// Collect view values into a `Vec<f64>` in logical (row-major) order.
fn view_to_vec<D: Dimension>(view: &ArrayView<'_, f64, D>) -> Vec<f64> {
    view.iter().copied().collect()
}

// ---------------------------------------------------------------------------
// as_strided
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::as_strided` (re-export of
/// `ferray_stride_tricks::as_strided::as_strided`).
///
/// The fixture encodes:
/// - `inputs.source`: source array as `{data, shape, dtype}`
/// - `inputs.shape`: view shape as `[usize]`
/// - `inputs.strides`: view strides in **element** units (the fixture
///   generator converts to byte strides when calling NumPy)
/// - `expected`: the materialised view as a flat array
#[test]
fn as_strided_matches_numpy() {
    let suite = load_fixture(&st_fixture("as_strided.json"));
    for case in &suite.test_cases {
        let source = make_f64_array_any_rank(&case.inputs["source"]);
        let shape = parse_usize_data(&case.inputs["shape"]);
        let strides = parse_usize_data(&case.inputs["strides"]);
        let view = ferray_stride_tricks::as_strided(&source, &shape, &strides)
            .unwrap_or_else(|e| panic!("as_strided {}: {e}", case.name));
        assert_eq!(
            view.shape(),
            shape.as_slice(),
            "case '{}': shape mismatch",
            case.name,
        );
        let actual = view_to_vec(&view);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
    }
}

// ---------------------------------------------------------------------------
// broadcast_to
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::broadcast_to` (re-export of
/// `ferray_stride_tricks::broadcast::broadcast_to`).
#[test]
fn broadcast_to_matches_numpy() {
    let suite = load_fixture(&st_fixture("broadcast_to.json"));
    for case in &suite.test_cases {
        let source_shape = parse_shape(&case.inputs["source"]["shape"]);
        let target_shape = parse_usize_data(&case.inputs["target_shape"]);
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);

        // Build the source at its native rank so we exercise the same
        // Dimension parameter NumPy would use. broadcast_to is generic
        // over D: Dimension, so we dispatch on the source rank.
        match source_shape.len() {
            1 => {
                let data = parse_f64_data(&case.inputs["source"]["data"]);
                let arr = Array::<f64, Ix1>::from_vec(Ix1::new([source_shape[0]]), data).unwrap();
                let view = ferray_stride_tricks::broadcast_to(&arr, &target_shape)
                    .unwrap_or_else(|e| panic!("broadcast_to {}: {e}", case.name));
                assert_eq!(
                    view.shape(),
                    expected_shape.as_slice(),
                    "case '{}': shape mismatch",
                    case.name,
                );
                let actual = view_to_vec(&view);
                assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
            }
            2 => {
                let data = parse_f64_data(&case.inputs["source"]["data"]);
                let arr =
                    Array::<f64, Ix2>::from_vec(Ix2::new([source_shape[0], source_shape[1]]), data)
                        .unwrap();
                let view = ferray_stride_tricks::broadcast_to(&arr, &target_shape)
                    .unwrap_or_else(|e| panic!("broadcast_to {}: {e}", case.name));
                assert_eq!(
                    view.shape(),
                    expected_shape.as_slice(),
                    "case '{}': shape mismatch",
                    case.name,
                );
                let actual = view_to_vec(&view);
                assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
            }
            other => panic!(
                "broadcast_to fixture {}: unsupported source rank {other}",
                case.name
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// broadcast_arrays
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::broadcast_arrays` (re-export of
/// `ferray_stride_tricks::broadcast::broadcast_arrays`).
///
/// `broadcast_arrays` is generic over a uniform input dimension `D`. The
/// fixture's two- and three-array cases all happen to share a 2-D
/// padded shape (after right-alignment), so we promote each input to 2-D
/// by left-padding with `1`s before calling the function. That matches
/// numpy's behaviour and lets us exercise a single rank.
#[test]
fn broadcast_arrays_matches_numpy() {
    let suite = load_fixture(&st_fixture("broadcast_arrays.json"));
    for case in &suite.test_cases {
        // Collect each input array, left-padding its shape with 1s up
        // to the common-rank ndim (the longest input shape's rank).
        let input_keys: Vec<String> = case
            .inputs
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| k.starts_with("arr_"))
            .cloned()
            .collect();
        let raw_shapes: Vec<Vec<usize>> = input_keys
            .iter()
            .map(|k| parse_shape(&case.inputs[k]["shape"]))
            .collect();
        let target_ndim = raw_shapes.iter().map(Vec::len).max().unwrap();
        // broadcast_arrays signature requires `D: Dimension`. For
        // simplicity (and to exercise the same dispatch path as the
        // fixture-generation reference), we route through Ix2 by
        // left-padding to rank 2. Any fixture with target_ndim > 2
        // would need a richer dispatch — currently none.
        assert!(
            target_ndim <= 2,
            "broadcast_arrays fixture {}: target_ndim {} > 2 not supported \
             by this conformance dispatcher",
            case.name,
            target_ndim,
        );

        let mut arrays: Vec<Array<f64, Ix2>> = Vec::with_capacity(input_keys.len());
        for k in &input_keys {
            let data = parse_f64_data(&case.inputs[k]["data"]);
            let raw_shape = parse_shape(&case.inputs[k]["shape"]);
            let padded: [usize; 2] = match raw_shape.len() {
                0 => [1, 1],
                1 => [1, raw_shape[0]],
                2 => [raw_shape[0], raw_shape[1]],
                _ => unreachable!(),
            };
            arrays.push(Array::<f64, Ix2>::from_vec(Ix2::new(padded), data).unwrap());
        }

        let views = ferray_stride_tricks::broadcast_arrays(&arrays)
            .unwrap_or_else(|e| panic!("broadcast_arrays {}: {e}", case.name));
        let expected_list = case.expected["arrays"].as_array().unwrap();
        assert_eq!(
            views.len(),
            expected_list.len(),
            "case '{}': view count mismatch",
            case.name,
        );
        for (i, (view, exp)) in views.iter().zip(expected_list.iter()).enumerate() {
            let exp_shape = parse_shape(&exp["shape"]);
            let exp_data = parse_f64_data(&exp["data"]);
            assert_eq!(
                view.shape(),
                exp_shape.as_slice(),
                "case '{}' view {}: shape mismatch",
                case.name,
                i,
            );
            let actual = view_to_vec(view);
            assert_f64_slice_ulp(
                &actual,
                &exp_data,
                case.tolerance_ulps,
                &format!("{} [view {i}]", case.name),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// broadcast_shapes
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::broadcast_shapes` (re-export of
/// `ferray_stride_tricks::broadcast::broadcast_shapes`).
///
/// Pure shape arithmetic — exact equality on the `Vec<usize>` result.
#[test]
fn broadcast_shapes_matches_numpy() {
    let suite = load_fixture(&st_fixture("broadcast_shapes.json"));
    for case in &suite.test_cases {
        let shapes_json = case.inputs["shapes"].as_array().unwrap();
        let owned_shapes: Vec<Vec<usize>> = shapes_json
            .iter()
            .map(|s| {
                s.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect()
            })
            .collect();
        let shape_refs: Vec<&[usize]> = owned_shapes.iter().map(Vec::as_slice).collect();
        let result = ferray_stride_tricks::broadcast_shapes(&shape_refs)
            .unwrap_or_else(|e| panic!("broadcast_shapes {}: {e}", case.name));
        let expected = parse_shape(&case.expected["shape"]);
        assert_eq!(
            result, expected,
            "case '{}': broadcast_shapes result mismatch",
            case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// sliding_window_view
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::sliding_window_view` (re-export of
/// `ferray_stride_tricks::sliding_window::sliding_window_view`).
#[test]
fn sliding_window_view_matches_numpy() {
    let suite = load_fixture(&st_fixture("sliding_window_view.json"));
    for case in &suite.test_cases {
        let source_shape = parse_shape(&case.inputs["source"]["shape"]);
        let window_shape = parse_usize_data(&case.inputs["window_shape"]);
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);

        match source_shape.len() {
            1 => {
                let data = parse_f64_data(&case.inputs["source"]["data"]);
                let arr = Array::<f64, Ix1>::from_vec(Ix1::new([source_shape[0]]), data).unwrap();
                let view = ferray_stride_tricks::sliding_window_view(&arr, &window_shape)
                    .unwrap_or_else(|e| panic!("sliding_window_view {}: {e}", case.name));
                assert_eq!(
                    view.shape(),
                    expected_shape.as_slice(),
                    "case '{}': shape mismatch",
                    case.name,
                );
                let actual = view_to_vec(&view);
                assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
            }
            2 => {
                let data = parse_f64_data(&case.inputs["source"]["data"]);
                let arr =
                    Array::<f64, Ix2>::from_vec(Ix2::new([source_shape[0], source_shape[1]]), data)
                        .unwrap();
                let view = ferray_stride_tricks::sliding_window_view(&arr, &window_shape)
                    .unwrap_or_else(|e| panic!("sliding_window_view {}: {e}", case.name));
                assert_eq!(
                    view.shape(),
                    expected_shape.as_slice(),
                    "case '{}': shape mismatch",
                    case.name,
                );
                let actual = view_to_vec(&view);
                assert_f64_slice_ulp(&actual, &expected, case.tolerance_ulps, &case.name);
            }
            other => panic!(
                "sliding_window_view fixture {}: unsupported source rank {other}",
                case.name
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// sliding_window_view_axis — partial sliding (no fixture, exercises the
// re-export so the surface gate sees it; the function's own behaviour is
// already covered by the in-crate property tests in tests/property_tests.rs
// and the unit tests in src/sliding_window.rs).
// ---------------------------------------------------------------------------

/// Covers: `ferray_stride_tricks::sliding_window_view_axis` (re-export of
/// `ferray_stride_tricks::sliding_window::sliding_window_view_axis`).
///
/// We construct a small 2-D source, window axis 0 only with size 2, and
/// confirm the output against an exact hand-computed expectation. This is
/// a *fixture-free* surface anchor — `sliding_window_view_axis` has no
/// direct numpy public-API counterpart (numpy's keyword arg is `axis=`),
/// so a JSON fixture would duplicate the existing internal coverage.
#[test]
fn sliding_window_view_axis_axis0_smoke() {
    // 4x3 source, window 2 on axis 0. Expected output shape (3, 3, 2).
    let src = Array::<f64, Ix2>::from_vec(
        Ix2::new([4, 3]),
        (1..=12).map(f64::from).collect::<Vec<f64>>(),
    )
    .unwrap();
    let view: FerrayResult<ArrayView<'_, f64, IxDyn>> =
        ferray_stride_tricks::sliding_window_view_axis(&src, &[2], &[0]);
    let view = view.expect("sliding_window_view_axis succeeds");
    assert_eq!(view.shape(), &[3, 3, 2]);
    // First window-position (outer index [0, 0]) holds rows 0 and 1, col 0:
    //   source row 0 col 0 = 1.0, source row 1 col 0 = 4.0.
    let collected: Vec<f64> = view.iter().copied().collect();
    assert_eq!(collected[0], 1.0);
    assert_eq!(collected[1], 4.0);
}
