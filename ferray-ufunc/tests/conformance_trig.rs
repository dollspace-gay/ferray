//! Conformance tests for ferray-ufunc's trigonometric functions against
//! NumPy reference outputs.
//!
//! Each test loads one fixture file and asserts every f64 test case against
//! the corresponding ferray-ufunc re-export. Tests call the user-facing path
//! at the crate root (e.g. `ferray_ufunc::sin`); the inner canonical path
//! (e.g. `ferray_ufunc::ops::trig::sin`) is mentioned in a doc comment on
//! each test so the surface-coverage gate's text match catches it too.
//!
//! Comparison uses ULP tolerance taken directly from the fixture's
//! `tolerance_ulps` field — fixtures are authored against the Stage 1
//! tolerance table (transcendentals: 1e-12 rel f64, which corresponds to
//! a per-case ULP budget the fixture generator selected). No `max()`
//! floor is applied: a fixture with `tolerance_ulps: 0` means bit-exact.
//! NaN/infinity handling is delegated to `assert_f64_slice_ulp` which
//! already treats NaN==NaN as a match.

use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_shape,
    should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type UnaryFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type BinaryFn = fn(&ArrF64, &ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;

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

/// Covers: `ferray_ufunc::sin` (re-export of `ferray_ufunc::ops::trig::sin`).
#[test]
fn sin_matches_numpy() {
    run_unary("sin.json", "sin", ferray_ufunc::sin);
}

/// Covers: `ferray_ufunc::cos` (re-export of `ferray_ufunc::ops::trig::cos`).
#[test]
fn cos_matches_numpy() {
    run_unary("cos.json", "cos", ferray_ufunc::cos);
}

/// Covers: `ferray_ufunc::tan` (re-export of `ferray_ufunc::ops::trig::tan`).
#[test]
fn tan_matches_numpy() {
    run_unary("tan.json", "tan", ferray_ufunc::tan);
}

/// Covers: `ferray_ufunc::arcsin` (re-export of
/// `ferray_ufunc::ops::trig::arcsin`).
#[test]
fn arcsin_matches_numpy() {
    run_unary("arcsin.json", "arcsin", ferray_ufunc::arcsin);
}

/// Covers: `ferray_ufunc::arccos` (re-export of
/// `ferray_ufunc::ops::trig::arccos`).
#[test]
fn arccos_matches_numpy() {
    run_unary("arccos.json", "arccos", ferray_ufunc::arccos);
}

/// Covers: `ferray_ufunc::arctan` (re-export of
/// `ferray_ufunc::ops::trig::arctan`).
#[test]
fn arctan_matches_numpy() {
    run_unary("arctan.json", "arctan", ferray_ufunc::arctan);
}

/// Covers: `ferray_ufunc::arctan2` (re-export of
/// `ferray_ufunc::ops::trig::arctan2`).
#[test]
fn arctan2_matches_numpy() {
    run_binary("arctan2.json", "arctan2", ferray_ufunc::arctan2);
}

/// Covers: `ferray_ufunc::sinh` (re-export of `ferray_ufunc::ops::trig::sinh`).
#[test]
fn sinh_matches_numpy() {
    run_unary("sinh.json", "sinh", ferray_ufunc::sinh);
}

/// Covers: `ferray_ufunc::cosh` (re-export of `ferray_ufunc::ops::trig::cosh`).
#[test]
fn cosh_matches_numpy() {
    run_unary("cosh.json", "cosh", ferray_ufunc::cosh);
}

/// Covers: `ferray_ufunc::tanh` (re-export of `ferray_ufunc::ops::trig::tanh`).
#[test]
fn tanh_matches_numpy() {
    run_unary("tanh.json", "tanh", ferray_ufunc::tanh);
}

/// Covers: `ferray_ufunc::arcsinh` (re-export of
/// `ferray_ufunc::ops::trig::arcsinh`).
#[test]
fn arcsinh_matches_numpy() {
    run_unary("arcsinh.json", "arcsinh", ferray_ufunc::arcsinh);
}

/// Covers: `ferray_ufunc::arccosh` (re-export of
/// `ferray_ufunc::ops::trig::arccosh`).
#[test]
fn arccosh_matches_numpy() {
    run_unary("arccosh.json", "arccosh", ferray_ufunc::arccosh);
}

/// Covers: `ferray_ufunc::arctanh` (re-export of
/// `ferray_ufunc::ops::trig::arctanh`).
#[test]
fn arctanh_matches_numpy() {
    run_unary("arctanh.json", "arctanh", ferray_ufunc::arctanh);
}
