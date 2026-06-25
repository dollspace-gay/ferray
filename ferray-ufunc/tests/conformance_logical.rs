//! Conformance tests for ferray-ufunc's logical functions against NumPy
//! reference outputs.
//!
//! The tests call crate-root re-exports and name the canonical
//! `ferray_ufunc::ops::logical::*` paths so the surface coverage gate can
//! prove both the user-facing and inner items.

use ferray_test_oracle::{load_fixture, make_f64_array, parse_bool_data, parse_shape};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type BoolUnaryFn = fn(
    &ArrF64,
) -> ferray_core::error::FerrayResult<
    ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
>;
type BoolBinaryFn = fn(
    &ArrF64,
    &ArrF64,
) -> ferray_core::error::FerrayResult<
    ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
>;
type BoolScalarFn = fn(&ArrF64) -> bool;
type BoolAxisFn = fn(
    &ArrF64,
    usize,
) -> ferray_core::error::FerrayResult<
    ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn run_bool_unary(fixture: &str, name: &str, f: BoolUnaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let arr = make_f64_array(input);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_bool_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            result.shape(),
            expected_shape.as_slice(),
            "{name} case {} shape mismatch",
            case.name
        );
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} data mismatch",
            case.name
        );
    }
}

fn run_bool_binary(fixture: &str, name: &str, f: BoolBinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr_a = make_f64_array(&case.inputs["a"]);
        let arr_b = make_f64_array(&case.inputs["b"]);
        let result = f(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_bool_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            result.shape(),
            expected_shape.as_slice(),
            "{name} case {} shape mismatch",
            case.name
        );
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} data mismatch",
            case.name
        );
    }
}

fn run_bool_scalar(fixture: &str, name: &str, f: BoolScalarFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let result = f(&arr);
        let expected = parse_bool_data(&case.expected["data"])[0];
        assert_eq!(result, expected, "{name} case {} mismatch", case.name);
    }
}

fn run_bool_axis(fixture: &str, name: &str, f: BoolAxisFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let axis = case.inputs["axis"].as_u64().expect("axis") as usize;
        let result = f(&arr, axis).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_bool_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            result.shape(),
            expected_shape.as_slice(),
            "{name} case {} shape mismatch",
            case.name
        );
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} data mismatch",
            case.name
        );
    }
}

/// Covers: `ferray_ufunc::logical_and` (re-export of
/// `ferray_ufunc::ops::logical::logical_and`).
///
/// Numeric truthiness in this fixture also exercises
/// `ferray_ufunc::Logical` / `ferray_ufunc::ops::logical::Logical`.
#[test]
fn logical_and_matches_numpy() {
    run_bool_binary("logical_and.json", "logical_and", ferray_ufunc::logical_and);
}

/// Covers: `ferray_ufunc::logical_or` (re-export of
/// `ferray_ufunc::ops::logical::logical_or`).
#[test]
fn logical_or_matches_numpy() {
    run_bool_binary("logical_or.json", "logical_or", ferray_ufunc::logical_or);
}

/// Covers: `ferray_ufunc::logical_xor` (re-export of
/// `ferray_ufunc::ops::logical::logical_xor`).
#[test]
fn logical_xor_matches_numpy() {
    run_bool_binary("logical_xor.json", "logical_xor", ferray_ufunc::logical_xor);
}

/// Covers: `ferray_ufunc::logical_not` (re-export of
/// `ferray_ufunc::ops::logical::logical_not`).
#[test]
fn logical_not_matches_numpy() {
    run_bool_unary("logical_not.json", "logical_not", ferray_ufunc::logical_not);
}

/// Covers: `ferray_ufunc::all` (re-export of
/// `ferray_ufunc::ops::logical::all`).
#[test]
fn all_matches_numpy() {
    run_bool_scalar("all.json", "all", ferray_ufunc::all);
}

/// Covers: `ferray_ufunc::any` (re-export of
/// `ferray_ufunc::ops::logical::any`).
#[test]
fn any_matches_numpy() {
    run_bool_scalar("any.json", "any", ferray_ufunc::any);
}

/// Covers: `ferray_ufunc::all_axis` (re-export of
/// `ferray_ufunc::ops::logical::all_axis`).
#[test]
fn all_axis_matches_numpy() {
    run_bool_axis("all_axis.json", "all_axis", ferray_ufunc::all_axis);
}

/// Covers: `ferray_ufunc::any_axis` (re-export of
/// `ferray_ufunc::ops::logical::any_axis`).
#[test]
fn any_axis_matches_numpy() {
    run_bool_axis("any_axis.json", "any_axis", ferray_ufunc::any_axis);
}
