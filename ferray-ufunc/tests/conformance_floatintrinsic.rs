//! Conformance tests for ferray-ufunc's float intrinsic / rounding functions
//! against NumPy reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_ufunc::floor`); the canonical inner path
//! (`ferray_ufunc::ops::rounding::floor` or
//! `ferray_ufunc::ops::floatintrinsic::clip`) is named in the doc comment.
//!
//! Uses ULP-based comparison for NaN/inf-aware matching.

use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_f64_value,
    parse_shape, should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type UnaryFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn run_unary(fixture: &str, name: &str, f: UnaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Rounding family
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::floor` (re-export of
/// `ferray_ufunc::ops::rounding::floor`).
#[test]
fn floor_matches_numpy() {
    run_unary("floor.json", "floor", ferray_ufunc::floor);
}

/// Covers: `ferray_ufunc::ceil` (re-export of
/// `ferray_ufunc::ops::rounding::ceil`).
#[test]
fn ceil_matches_numpy() {
    run_unary("ceil.json", "ceil", ferray_ufunc::ceil);
}

/// Covers: `ferray_ufunc::trunc` (re-export of
/// `ferray_ufunc::ops::rounding::trunc`).
#[test]
fn trunc_matches_numpy() {
    run_unary("trunc.json", "trunc", ferray_ufunc::trunc);
}

/// Covers: `ferray_ufunc::rint` (re-export of
/// `ferray_ufunc::ops::rounding::rint`).
#[test]
fn rint_matches_numpy() {
    run_unary("rint.json", "rint", ferray_ufunc::rint);
}

/// Covers: `ferray_ufunc::fix` (re-export of
/// `ferray_ufunc::ops::rounding::fix`).
#[test]
fn fix_matches_numpy() {
    run_unary("fix.json", "fix", ferray_ufunc::fix);
}

/// Covers: `ferray_ufunc::round` (re-export of
/// `ferray_ufunc::ops::rounding::round`). The round fixture has cases with
/// `decimals` parameters; ferray's `round` takes only the array (banker's
/// rounding), so we restrict to decimals=0 cases.
#[test]
fn round_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("round.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let decimals = case
            .inputs
            .get("decimals")
            .and_then(ferray_test_oracle::serde_json::Value::as_i64)
            .unwrap_or(0);
        if decimals != 0 {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_ufunc::round(&arr).expect(&case.name);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Float intrinsics
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::clip` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::clip`).
#[test]
fn clip_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("clip.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let a_min = parse_f64_value(&case.inputs["a_min"]);
        let a_max = parse_f64_value(&case.inputs["a_max"]);
        let result = ferray_ufunc::clip(&arr, a_min, a_max).expect(&case.name);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::nan_to_num` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::nan_to_num`).
#[test]
fn nan_to_num_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("nan_to_num.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let nan = case
            .inputs
            .get("nan")
            .and_then(ferray_test_oracle::serde_json::Value::as_f64)
            .unwrap_or(0.0);
        let posinf = case
            .inputs
            .get("posinf")
            .and_then(ferray_test_oracle::serde_json::Value::as_f64)
            .unwrap_or(f64::MAX);
        let neginf = case
            .inputs
            .get("neginf")
            .and_then(ferray_test_oracle::serde_json::Value::as_f64)
            .unwrap_or(f64::MIN);
        let result = ferray_ufunc::nan_to_num(&arr, Some(nan), Some(posinf), Some(neginf))
            .expect(&case.name);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}
