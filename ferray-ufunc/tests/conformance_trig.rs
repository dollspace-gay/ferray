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
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_f64_value,
    parse_shape, should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type UnaryFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type UnaryIntoFn = fn(&ArrF64, &mut ArrF64) -> ferray_core::error::FerrayResult<()>;
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

fn run_unary_into(fixture: &str, name: &str, f: UnaryIntoFn) {
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
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        let mut out = ArrF64::from_vec(
            ferray_core::dimension::IxDyn::new(&expected_shape),
            vec![0.0; expected.len()],
        )
        .unwrap();
        f(&arr, &mut out).unwrap_or_else(|e| panic!("{name}_into {}: {e}", case.name));
        assert_f64_slice_ulp(
            out.as_slice().expect("contiguous"),
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

fn assert_array_matches(
    name: &str,
    case_name: &str,
    result: &ArrF64,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    let expected_data = parse_f64_data(&expected["data"]);
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

/// Covers: `ferray_ufunc::sin` (re-export of `ferray_ufunc::ops::trig::sin`).
#[test]
fn sin_matches_numpy() {
    run_unary("sin.json", "sin", ferray_ufunc::sin);
}

/// Covers: `ferray_ufunc::sin_into` (`ferray_ufunc::ops::trig::sin_into`)
/// with NumPy `sin` fixtures.
#[test]
fn sin_into_matches_numpy() {
    run_unary_into("sin.json", "sin", ferray_ufunc::sin_into);
}

/// Covers: `ferray_ufunc::cos` (re-export of `ferray_ufunc::ops::trig::cos`).
#[test]
fn cos_matches_numpy() {
    run_unary("cos.json", "cos", ferray_ufunc::cos);
}

/// Covers: `ferray_ufunc::cos_into` (`ferray_ufunc::ops::trig::cos_into`)
/// with NumPy `cos` fixtures.
#[test]
fn cos_into_matches_numpy() {
    run_unary_into("cos.json", "cos", ferray_ufunc::cos_into);
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

/// Covers: `ferray_ufunc::hypot` (re-export of
/// `ferray_ufunc::ops::trig::hypot`).
#[test]
fn hypot_matches_numpy() {
    run_binary("hypot.json", "hypot", ferray_ufunc::hypot);
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

/// Covers: `ferray_ufunc::degrees` (re-export of
/// `ferray_ufunc::ops::trig::degrees`).
#[test]
fn degrees_matches_numpy() {
    run_unary("degrees.json", "degrees", ferray_ufunc::degrees);
}

/// Covers: `ferray_ufunc::rad2deg` (re-export of
/// `ferray_ufunc::ops::trig::rad2deg`).
#[test]
fn rad2deg_matches_numpy() {
    run_unary("rad2deg.json", "rad2deg", ferray_ufunc::rad2deg);
}

/// Covers: `ferray_ufunc::radians` (re-export of
/// `ferray_ufunc::ops::trig::radians`).
#[test]
fn radians_matches_numpy() {
    run_unary("radians.json", "radians", ferray_ufunc::radians);
}

/// Covers: `ferray_ufunc::deg2rad` (re-export of
/// `ferray_ufunc::ops::trig::deg2rad`).
#[test]
fn deg2rad_matches_numpy() {
    run_unary("deg2rad.json", "deg2rad", ferray_ufunc::deg2rad);
}

/// Covers: `ferray_ufunc::unwrap` (re-export of
/// `ferray_ufunc::ops::trig::unwrap`) against NumPy `unwrap`.
#[test]
fn unwrap_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("unwrap.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let arr = make_f64_array(input);
        let discont = case.inputs.get("discont").map(parse_f64_value);
        let result = ferray_ufunc::unwrap(&arr, discont)
            .unwrap_or_else(|e| panic!("unwrap {}: {e}", case.name));
        assert_array_matches(
            "unwrap",
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}
