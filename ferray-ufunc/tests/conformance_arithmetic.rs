//! Conformance tests for ferray-ufunc's arithmetic functions against NumPy
//! reference outputs.
//!
//! Each test calls the user-facing re-export at the crate root (e.g.
//! `ferray_ufunc::add`); the canonical inner path
//! (`ferray_ufunc::ops::arithmetic::add`) is named in the test's doc comment
//! so the surface-coverage gate's text match picks it up.
//!
//! Uses ULP-based comparison so NaN and inf semantics match oracle.rs.

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

// ---------------------------------------------------------------------------
// Unary arithmetic
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::absolute` (re-export of
/// `ferray_ufunc::ops::arithmetic::absolute`).
#[test]
fn absolute_matches_numpy() {
    run_unary("absolute.json", "absolute", ferray_ufunc::absolute);
}

/// Covers: `ferray_ufunc::negative` (re-export of
/// `ferray_ufunc::ops::arithmetic::negative`).
#[test]
fn negative_matches_numpy() {
    run_unary("negative.json", "negative", ferray_ufunc::negative);
}

/// Covers: `ferray_ufunc::sqrt` (re-export of
/// `ferray_ufunc::ops::arithmetic::sqrt`).
#[test]
fn sqrt_matches_numpy() {
    run_unary("sqrt.json", "sqrt", ferray_ufunc::sqrt);
}

/// Covers: `ferray_ufunc::cbrt` (re-export of
/// `ferray_ufunc::ops::arithmetic::cbrt`).
#[test]
fn cbrt_matches_numpy() {
    run_unary("cbrt.json", "cbrt", ferray_ufunc::cbrt);
}

/// Covers: `ferray_ufunc::square` (re-export of
/// `ferray_ufunc::ops::arithmetic::square`).
#[test]
fn square_matches_numpy() {
    run_unary("square.json", "square", ferray_ufunc::square);
}

/// Covers: `ferray_ufunc::reciprocal` (re-export of
/// `ferray_ufunc::ops::arithmetic::reciprocal`).
#[test]
fn reciprocal_matches_numpy() {
    run_unary("reciprocal.json", "reciprocal", ferray_ufunc::reciprocal);
}

/// Covers: `ferray_ufunc::sinc` (re-export of
/// `ferray_ufunc::ops::special::sinc`).
#[test]
fn sinc_matches_numpy() {
    run_unary("sinc.json", "sinc", ferray_ufunc::sinc);
}

// ---------------------------------------------------------------------------
// Binary arithmetic
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::add` (re-export of
/// `ferray_ufunc::ops::arithmetic::add`).
#[test]
fn add_matches_numpy() {
    run_binary("add.json", "add", ferray_ufunc::add);
}

/// Covers: `ferray_ufunc::subtract` (re-export of
/// `ferray_ufunc::ops::arithmetic::subtract`).
#[test]
fn subtract_matches_numpy() {
    run_binary("subtract.json", "subtract", ferray_ufunc::subtract);
}

/// Covers: `ferray_ufunc::multiply` (re-export of
/// `ferray_ufunc::ops::arithmetic::multiply`).
#[test]
fn multiply_matches_numpy() {
    run_binary("multiply.json", "multiply", ferray_ufunc::multiply);
}

/// Covers: `ferray_ufunc::divide` (re-export of
/// `ferray_ufunc::ops::arithmetic::divide`).
#[test]
fn divide_matches_numpy() {
    run_binary("divide.json", "divide", ferray_ufunc::divide);
}

/// Covers: `ferray_ufunc::power` (re-export of
/// `ferray_ufunc::ops::arithmetic::power`).
#[test]
fn power_matches_numpy() {
    run_binary("power.json", "power", ferray_ufunc::power);
}

/// Covers: `ferray_ufunc::remainder` (re-export of
/// `ferray_ufunc::ops::arithmetic::remainder`).
#[test]
fn remainder_matches_numpy() {
    run_binary("remainder.json", "remainder", ferray_ufunc::remainder);
}

/// Covers: `ferray_ufunc::mod_` (re-export of
/// `ferray_ufunc::ops::arithmetic::mod_`).
#[test]
fn mod_matches_numpy() {
    run_binary("mod.json", "mod_", ferray_ufunc::mod_);
}

/// Covers: `ferray_ufunc::heaviside` (re-export of
/// `ferray_ufunc::ops::arithmetic::heaviside`).
#[test]
fn heaviside_matches_numpy() {
    run_binary("heaviside.json", "heaviside", ferray_ufunc::heaviside);
}
