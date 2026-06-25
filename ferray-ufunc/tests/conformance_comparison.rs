//! Conformance tests for ferray-ufunc's comparison / min-max functions
//! against NumPy reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_ufunc::maximum`); the canonical inner path
//! (`ferray_ufunc::ops::floatintrinsic::maximum`) is named in the test's
//! doc comment.
//!
//! Uses ULP-based comparison for NaN/inf-aware matching.

use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_bool_data, parse_f64_data,
    parse_f64_value, parse_shape, should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type BinaryFn = fn(&ArrF64, &ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type BoolBinaryFn = fn(
    &ArrF64,
    &ArrF64,
) -> ferray_core::error::FerrayResult<
    ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
>;
type IsCloseFn = fn(
    &ArrF64,
    &ArrF64,
    f64,
    f64,
    bool,
) -> ferray_core::error::FerrayResult<
    ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
>;
type AllCloseFn = fn(&ArrF64, &ArrF64, f64, f64) -> ferray_core::error::FerrayResult<bool>;
type BoolScalarFn = fn(&ArrF64, &ArrF64) -> bool;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn run_binary(fixture: &str, name: &str, f: BinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let shape = parse_shape(&a["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let result = f(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

fn run_bool_binary(fixture: &str, name: &str, f: BoolBinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let shape = parse_shape(&a["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let result = f(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} mismatch",
            case.name
        );
    }
}

fn run_isclose(fixture: &str, name: &str, f: IsCloseFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let rtol = parse_f64_value(&case.inputs["rtol"]);
        let atol = parse_f64_value(&case.inputs["atol"]);
        let equal_nan = case.inputs["equal_nan"].as_bool().unwrap_or(false);
        let result = f(&arr_a, &arr_b, rtol, atol, equal_nan)
            .unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} mismatch",
            case.name
        );
    }
}

fn run_allclose(fixture: &str, name: &str, f: AllCloseFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let rtol = parse_f64_value(&case.inputs["rtol"]);
        let atol = parse_f64_value(&case.inputs["atol"]);
        let result =
            f(&arr_a, &arr_b, rtol, atol).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = case.expected["data"].as_bool().expect("scalar bool");
        assert_eq!(result, expected, "{name} case {} mismatch", case.name);
    }
}

fn run_bool_scalar(fixture: &str, name: &str, f: BoolScalarFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let result = f(&arr_a, &arr_b);
        let expected = case.expected["data"].as_bool().expect("scalar bool");
        assert_eq!(result, expected, "{name} case {} mismatch", case.name);
    }
}

/// Covers: `ferray_ufunc::equal` (re-export of
/// `ferray_ufunc::ops::comparison::equal`).
#[test]
fn equal_matches_numpy() {
    run_bool_binary("equal.json", "equal", ferray_ufunc::equal);
}

/// Covers: `ferray_ufunc::equal_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::equal_broadcast`) with NumPy `equal`
/// fixtures.
#[test]
fn equal_broadcast_matches_numpy() {
    run_bool_binary(
        "equal.json",
        "equal_broadcast",
        ferray_ufunc::equal_broadcast,
    );
}

/// Covers: `ferray_ufunc::not_equal` (re-export of
/// `ferray_ufunc::ops::comparison::not_equal`).
#[test]
fn not_equal_matches_numpy() {
    run_bool_binary("not_equal.json", "not_equal", ferray_ufunc::not_equal);
}

/// Covers: `ferray_ufunc::not_equal_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::not_equal_broadcast`) with NumPy
/// `not_equal` fixtures.
#[test]
fn not_equal_broadcast_matches_numpy() {
    run_bool_binary(
        "not_equal.json",
        "not_equal_broadcast",
        ferray_ufunc::not_equal_broadcast,
    );
}

/// Covers: `ferray_ufunc::less` (re-export of
/// `ferray_ufunc::ops::comparison::less`).
#[test]
fn less_matches_numpy() {
    run_bool_binary("less.json", "less", ferray_ufunc::less);
}

/// Covers: `ferray_ufunc::less_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::less_broadcast`) with NumPy `less`
/// fixtures.
#[test]
fn less_broadcast_matches_numpy() {
    run_bool_binary("less.json", "less_broadcast", ferray_ufunc::less_broadcast);
}

/// Covers: `ferray_ufunc::less_equal` (re-export of
/// `ferray_ufunc::ops::comparison::less_equal`).
#[test]
fn less_equal_matches_numpy() {
    run_bool_binary("less_equal.json", "less_equal", ferray_ufunc::less_equal);
}

/// Covers: `ferray_ufunc::less_equal_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::less_equal_broadcast`) with NumPy
/// `less_equal` fixtures.
#[test]
fn less_equal_broadcast_matches_numpy() {
    run_bool_binary(
        "less_equal.json",
        "less_equal_broadcast",
        ferray_ufunc::less_equal_broadcast,
    );
}

/// Covers: `ferray_ufunc::greater` (re-export of
/// `ferray_ufunc::ops::comparison::greater`).
#[test]
fn greater_matches_numpy() {
    run_bool_binary("greater.json", "greater", ferray_ufunc::greater);
}

/// Covers: `ferray_ufunc::greater_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::greater_broadcast`) with NumPy `greater`
/// fixtures.
#[test]
fn greater_broadcast_matches_numpy() {
    run_bool_binary(
        "greater.json",
        "greater_broadcast",
        ferray_ufunc::greater_broadcast,
    );
}

/// Covers: `ferray_ufunc::greater_equal` (re-export of
/// `ferray_ufunc::ops::comparison::greater_equal`).
#[test]
fn greater_equal_matches_numpy() {
    run_bool_binary(
        "greater_equal.json",
        "greater_equal",
        ferray_ufunc::greater_equal,
    );
}

/// Covers: `ferray_ufunc::greater_equal_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::greater_equal_broadcast`) with NumPy
/// `greater_equal` fixtures.
#[test]
fn greater_equal_broadcast_matches_numpy() {
    run_bool_binary(
        "greater_equal.json",
        "greater_equal_broadcast",
        ferray_ufunc::greater_equal_broadcast,
    );
}

/// Covers: `ferray_ufunc::isclose` (re-export of
/// `ferray_ufunc::ops::comparison::isclose`).
#[test]
fn isclose_matches_numpy() {
    run_isclose("isclose.json", "isclose", ferray_ufunc::isclose);
}

/// Covers: `ferray_ufunc::isclose_broadcast` (re-export of
/// `ferray_ufunc::ops::comparison::isclose_broadcast`) with NumPy
/// `isclose` fixtures.
#[test]
fn isclose_broadcast_matches_numpy() {
    run_isclose(
        "isclose.json",
        "isclose_broadcast",
        ferray_ufunc::isclose_broadcast,
    );
}

/// Covers: `ferray_ufunc::allclose` (re-export of
/// `ferray_ufunc::ops::comparison::allclose`).
#[test]
fn allclose_matches_numpy() {
    run_allclose("allclose.json", "allclose", ferray_ufunc::allclose);
}

/// Covers: `ferray_ufunc::array_equal` (re-export of
/// `ferray_ufunc::ops::comparison::array_equal`).
#[test]
fn array_equal_matches_numpy() {
    run_bool_scalar("array_equal.json", "array_equal", ferray_ufunc::array_equal);
}

/// Covers: `ferray_ufunc::array_equiv` (re-export of
/// `ferray_ufunc::ops::comparison::array_equiv`).
#[test]
fn array_equiv_matches_numpy() {
    run_bool_scalar("array_equiv.json", "array_equiv", ferray_ufunc::array_equiv);
}

/// Covers: `ferray_ufunc::maximum` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::maximum`).
#[test]
fn maximum_matches_numpy() {
    run_binary("maximum.json", "maximum", ferray_ufunc::maximum);
}

/// Covers: `ferray_ufunc::minimum` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::minimum`).
#[test]
fn minimum_matches_numpy() {
    run_binary("minimum.json", "minimum", ferray_ufunc::minimum);
}

/// Covers: `ferray_ufunc::fmax` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::fmax`).
#[test]
fn fmax_matches_numpy() {
    run_binary("fmax.json", "fmax", ferray_ufunc::fmax);
}

/// Covers: `ferray_ufunc::fmin` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::fmin`).
#[test]
fn fmin_matches_numpy() {
    run_binary("fmin.json", "fmin", ferray_ufunc::fmin);
}
