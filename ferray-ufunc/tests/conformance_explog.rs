//! Conformance tests for ferray-ufunc's exponential and logarithmic
//! functions against NumPy reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_ufunc::exp`); the canonical inner path
//! (`ferray_ufunc::ops::explog::exp`) is mentioned in a doc comment on
//! each test so the surface-coverage gate's text match picks it up too.
//!
//! Uses ULP-based comparison with the fixture's `tolerance_ulps` as the
//! authoritative budget — no workspace-wide floor is applied. NaN/inf
//! semantics come from `assert_f64_slice_ulp` directly.

use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_shape,
    should_skip_f64,
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

/// Covers: `ferray_ufunc::exp` (re-export of
/// `ferray_ufunc::ops::explog::exp`).
#[test]
fn exp_matches_numpy() {
    run_unary("exp.json", "exp", ferray_ufunc::exp);
}

/// Covers: `ferray_ufunc::exp2` (re-export of
/// `ferray_ufunc::ops::explog::exp2`).
#[test]
fn exp2_matches_numpy() {
    run_unary("exp2.json", "exp2", ferray_ufunc::exp2);
}

/// Covers: `ferray_ufunc::expm1` (re-export of
/// `ferray_ufunc::ops::explog::expm1`).
#[test]
fn expm1_matches_numpy() {
    run_unary("expm1.json", "expm1", ferray_ufunc::expm1);
}

/// Covers: `ferray_ufunc::log` (re-export of
/// `ferray_ufunc::ops::explog::log`).
#[test]
fn log_matches_numpy() {
    run_unary("log.json", "log", ferray_ufunc::log);
}

/// Covers: `ferray_ufunc::log2` (re-export of
/// `ferray_ufunc::ops::explog::log2`).
#[test]
fn log2_matches_numpy() {
    run_unary("log2.json", "log2", ferray_ufunc::log2);
}

/// Covers: `ferray_ufunc::log10` (re-export of
/// `ferray_ufunc::ops::explog::log10`).
#[test]
fn log10_matches_numpy() {
    run_unary("log10.json", "log10", ferray_ufunc::log10);
}

/// Covers: `ferray_ufunc::log1p` (re-export of
/// `ferray_ufunc::ops::explog::log1p`).
#[test]
fn log1p_matches_numpy() {
    run_unary("log1p.json", "log1p", ferray_ufunc::log1p);
}
