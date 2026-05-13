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
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_shape,
    should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type BinaryFn = fn(&ArrF64, &ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;

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
