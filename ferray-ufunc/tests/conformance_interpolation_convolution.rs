//! Conformance tests for interpolation, convolution, and NaN extrema
//! reductions against NumPy fixture outputs.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_f64_value,
    parse_shape,
};

type ArrF64 = Array<f64, IxDyn>;
type ArrF64Ix1 = Array<f64, Ix1>;
type AxisReduceFn = fn(&ArrF64, usize, bool) -> ferray_core::error::FerrayResult<ArrF64>;
type AxesReduceFn = fn(&ArrF64, &[usize], bool) -> ferray_core::error::FerrayResult<ArrF64>;
type AllReduceFn = fn(&ArrF64) -> f64;

// Surface coverage anchors for canonical paths exercised by this file:
// ferray_ufunc::ConvolveMode
// ferray_ufunc::convolve
// ferray_ufunc::fftconvolve
// ferray_ufunc::interp
// ferray_ufunc::interp_one
// ferray_ufunc::nan_max_reduce
// ferray_ufunc::nan_max_reduce_all
// ferray_ufunc::nan_max_reduce_axes
// ferray_ufunc::nan_min_reduce
// ferray_ufunc::nan_min_reduce_all
// ferray_ufunc::nan_min_reduce_axes
// ferray_ufunc::ops::convolution::ConvolveMode
// ferray_ufunc::ops::convolution::convolve
// ferray_ufunc::ops::convolution::fftconvolve
// ferray_ufunc::ops::interpolation::interp
// ferray_ufunc::ops::interpolation::interp_one
// ferray_ufunc::ops::arithmetic::nan_max_reduce
// ferray_ufunc::ops::arithmetic::nan_max_reduce_all
// ferray_ufunc::ops::arithmetic::nan_max_reduce_axes
// ferray_ufunc::ops::arithmetic::nan_min_reduce
// ferray_ufunc::ops::arithmetic::nan_min_reduce_all
// ferray_ufunc::ops::arithmetic::nan_min_reduce_axes

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn make_f64_ix1(value: &ferray_test_oracle::serde_json::Value) -> ArrF64Ix1 {
    let data = parse_f64_data(&value["data"]);
    let shape = parse_shape(&value["shape"]);
    assert_eq!(shape.len(), 1, "expected 1-D fixture input");
    ArrF64Ix1::from_vec(Ix1::new([shape[0]]), data).unwrap()
}

fn assert_ix1_matches(
    name: &str,
    case_name: &str,
    result: &ArrF64Ix1,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_data = parse_f64_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_f64_slice_ulp(
        result.as_slice().expect("contiguous"),
        &expected_data,
        tolerance_ulps,
        case_name,
    );
}

fn assert_dyn_matches(
    name: &str,
    case_name: &str,
    result: &ArrF64,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_data = parse_f64_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_f64_slice_ulp(
        result.as_slice().expect("contiguous"),
        &expected_data,
        tolerance_ulps,
        case_name,
    );
}

fn axis_value(value: &ferray_test_oracle::serde_json::Value) -> usize {
    value.as_u64().expect("axis integer") as usize
}

fn axes_value(value: &ferray_test_oracle::serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("axis list")
        .iter()
        .map(|v| v.as_u64().expect("axis integer") as usize)
        .collect()
}

fn keepdims_value(value: &ferray_test_oracle::serde_json::Value) -> bool {
    value.as_bool().expect("keepdims bool")
}

fn convolve_mode(value: &ferray_test_oracle::serde_json::Value) -> ferray_ufunc::ConvolveMode {
    match value.as_str().expect("convolve mode string") {
        "full" => ferray_ufunc::ConvolveMode::Full,
        "same" => ferray_ufunc::ConvolveMode::Same,
        "valid" => ferray_ufunc::ConvolveMode::Valid,
        mode => panic!("unsupported convolve mode {mode}"),
    }
}

/// Covers: `ferray_ufunc::interp` and
/// `ferray_ufunc::ops::interpolation::interp`.
#[test]
fn interp_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("interp.json"));
    for case in &suite.test_cases {
        let x = make_f64_ix1(&case.inputs["x"]);
        let xp = make_f64_ix1(&case.inputs["xp"]);
        let fp = make_f64_ix1(&case.inputs["fp"]);
        let result = ferray_ufunc::interp(&x, &xp, &fp)
            .unwrap_or_else(|e| panic!("interp {}: {e}", case.name));
        assert_ix1_matches(
            "interp",
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

/// Covers: `ferray_ufunc::interp_one` and
/// `ferray_ufunc::ops::interpolation::interp_one`.
#[test]
fn interp_one_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("interp_one.json"));
    for case in &suite.test_cases {
        let xi = parse_f64_value(&case.inputs["xi"]);
        let xp = parse_f64_data(&case.inputs["xp"]["data"]);
        let fp = parse_f64_data(&case.inputs["fp"]["data"]);
        let result = ferray_ufunc::interp_one(xi, &xp, &fp)
            .unwrap_or_else(|e| panic!("interp_one {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(expected.len(), 1, "interp_one expected scalar");
        assert_f64_slice_ulp(&[result], &expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers: `ferray_ufunc::ConvolveMode`, `ferray_ufunc::convolve`, and
/// `ferray_ufunc::ops::convolution::{ConvolveMode, convolve}`.
#[test]
fn convolve_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("convolve.json"));
    for case in &suite.test_cases {
        let a = make_f64_ix1(&case.inputs["a"]);
        let v = make_f64_ix1(&case.inputs["v"]);
        let mode = convolve_mode(&case.inputs["mode"]);
        let result = ferray_ufunc::convolve(&a, &v, mode)
            .unwrap_or_else(|e| panic!("convolve {}: {e}", case.name));
        assert_ix1_matches(
            "convolve",
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

/// Covers: `ferray_ufunc::fftconvolve` and
/// `ferray_ufunc::ops::convolution::fftconvolve`.
#[cfg(feature = "fft-convolve")]
#[test]
fn fftconvolve_matches_scipy() {
    let suite = load_fixture(&ufunc_fixture("fftconvolve.json"));
    for case in &suite.test_cases {
        let a = make_f64_ix1(&case.inputs["a"]);
        let v = make_f64_ix1(&case.inputs["v"]);
        let mode = convolve_mode(&case.inputs["mode"]);
        let result = ferray_ufunc::fftconvolve(&a, &v, mode)
            .unwrap_or_else(|e| panic!("fftconvolve {}: {e}", case.name));
        assert_ix1_matches(
            "fftconvolve",
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_axis_reduce(fixture: &str, name: &str, f: AxisReduceFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let axis = axis_value(&case.inputs["axis"]);
        let keepdims = keepdims_value(&case.inputs["keepdims"]);
        let result =
            f(&arr, axis, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_dyn_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_axes_reduce(fixture: &str, name: &str, f: AxesReduceFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let axes = axes_value(&case.inputs["axis"]);
        let keepdims = keepdims_value(&case.inputs["keepdims"]);
        let result =
            f(&arr, &axes, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_dyn_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_all_reduce(fixture: &str, name: &str, f: AllReduceFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let result = f(&arr);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_eq!(
            expected.len(),
            1,
            "{name} case {} expected scalar",
            case.name
        );
        assert_f64_slice_ulp(&[result], &expected, case.tolerance_ulps, &case.name);
    }
}

/// Covers: `ferray_ufunc::nan_max_reduce` and
/// `ferray_ufunc::ops::arithmetic::nan_max_reduce`.
#[test]
fn nan_max_reduce_matches_numpy() {
    run_nan_axis_reduce(
        "nan_max_reduce.json",
        "nan_max_reduce",
        ferray_ufunc::nan_max_reduce,
    );
}

/// Covers: `ferray_ufunc::nan_max_reduce_all` and
/// `ferray_ufunc::ops::arithmetic::nan_max_reduce_all`.
#[test]
fn nan_max_reduce_all_matches_numpy() {
    run_nan_all_reduce(
        "nan_max_reduce_all.json",
        "nan_max_reduce_all",
        ferray_ufunc::nan_max_reduce_all,
    );
}

/// Covers: `ferray_ufunc::nan_max_reduce_axes` and
/// `ferray_ufunc::ops::arithmetic::nan_max_reduce_axes`.
#[test]
fn nan_max_reduce_axes_matches_numpy() {
    run_nan_axes_reduce(
        "nan_max_reduce_axes.json",
        "nan_max_reduce_axes",
        ferray_ufunc::nan_max_reduce_axes,
    );
}

/// Covers: `ferray_ufunc::nan_min_reduce` and
/// `ferray_ufunc::ops::arithmetic::nan_min_reduce`.
#[test]
fn nan_min_reduce_matches_numpy() {
    run_nan_axis_reduce(
        "nan_min_reduce.json",
        "nan_min_reduce",
        ferray_ufunc::nan_min_reduce,
    );
}

/// Covers: `ferray_ufunc::nan_min_reduce_all` and
/// `ferray_ufunc::ops::arithmetic::nan_min_reduce_all`.
#[test]
fn nan_min_reduce_all_matches_numpy() {
    run_nan_all_reduce(
        "nan_min_reduce_all.json",
        "nan_min_reduce_all",
        ferray_ufunc::nan_min_reduce_all,
    );
}

/// Covers: `ferray_ufunc::nan_min_reduce_axes` and
/// `ferray_ufunc::ops::arithmetic::nan_min_reduce_axes`.
#[test]
fn nan_min_reduce_axes_matches_numpy() {
    run_nan_axes_reduce(
        "nan_min_reduce_axes.json",
        "nan_min_reduce_axes",
        ferray_ufunc::nan_min_reduce_axes,
    );
}
