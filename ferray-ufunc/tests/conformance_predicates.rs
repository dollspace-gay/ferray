//! Conformance tests for ferray-ufunc's boolean predicate functions against
//! NumPy reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_ufunc::isnan`); the canonical inner path
//! (`ferray_ufunc::ops::floatintrinsic::isnan`) is named in the test's doc
//! comment so the surface-coverage gate's text match picks it up too.

use ferray_test_oracle::{
    load_fixture, make_f64_array, parse_shape, should_skip_f64,
};

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn parse_bool_data(value: &ferray_test_oracle::serde_json::Value) -> Vec<bool> {
    value
        .as_array()
        .expect("expected data array")
        .iter()
        .map(|v| {
            if let Some(b) = v.as_bool() {
                b
            } else if let Some(n) = v.as_i64() {
                n != 0
            } else if let Some(n) = v.as_f64() {
                n != 0.0
            } else {
                panic!("expected bool-coercible value, got {v:?}");
            }
        })
        .collect()
}

fn unary_bool_check<F>(fixture: &str, name: &str, f: F)
where
    F: Fn(
        &ferray_core::Array<f64, ferray_core::dimension::IxDyn>,
    ) -> ferray_core::error::FerrayResult<
        ferray_core::Array<bool, ferray_core::dimension::IxDyn>,
    >,
{
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
        let expected = parse_bool_data(&case.expected["data"]);
        let actual = result.as_slice().expect("contiguous");
        assert_eq!(
            actual,
            expected.as_slice(),
            "{name} case {} mismatch",
            case.name
        );
    }
}

/// Covers: `ferray_ufunc::isnan` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::isnan`).
#[test]
fn isnan_matches_numpy() {
    unary_bool_check("isnan.json", "isnan", ferray_ufunc::isnan);
}

/// Covers: `ferray_ufunc::isinf` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::isinf`).
#[test]
fn isinf_matches_numpy() {
    unary_bool_check("isinf.json", "isinf", ferray_ufunc::isinf);
}

/// Covers: `ferray_ufunc::isfinite` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::isfinite`).
#[test]
fn isfinite_matches_numpy() {
    unary_bool_check("isfinite.json", "isfinite", ferray_ufunc::isfinite);
}
