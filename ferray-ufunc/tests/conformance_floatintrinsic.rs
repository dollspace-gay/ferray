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
    assert_f64_slice_ulp, load_fixture, make_f64_array, make_i64_array, parse_f64_data,
    parse_f64_value, parse_i64_data, parse_shape, should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type ArrI32 = ferray_core::Array<i32, ferray_core::dimension::IxDyn>;
type ArrI64 = ferray_core::Array<i64, ferray_core::dimension::IxDyn>;
type UnaryFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type BinaryFn = fn(&ArrF64, &ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn parse_i32_data(value: &ferray_test_oracle::serde_json::Value) -> Vec<i32> {
    parse_i64_data(value)
        .into_iter()
        .map(|n| i32::try_from(n).expect("fixture value fits i32"))
        .collect()
}

fn make_i32_array(value: &ferray_test_oracle::serde_json::Value) -> ArrI32 {
    let data = parse_i32_data(&value["data"]);
    let shape = parse_shape(&value["shape"]);
    ArrI32::from_vec(ferray_core::dimension::IxDyn::new(&shape), data).unwrap()
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

/// Covers: `ferray_ufunc::around` (re-export of
/// `ferray_ufunc::ops::rounding::around`).
#[test]
fn around_matches_numpy() {
    run_unary("around.json", "around", ferray_ufunc::around);
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

/// Covers: `ferray_ufunc::clip_ord` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::clip_ord`) against integer NumPy
/// `clip` fixtures.
#[test]
fn clip_ord_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("clip_ord.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let arr: ArrI64 = make_i64_array(input);
        let a_min = case.inputs["a_min"].as_i64().expect("a_min integer");
        let a_max = case.inputs["a_max"].as_i64().expect("a_max integer");
        let result = ferray_ufunc::clip_ord(&arr, a_min, a_max)
            .unwrap_or_else(|e| panic!("clip_ord {}: {e}", case.name));
        let expected = parse_i64_data(&case.expected["data"]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "clip_ord case {} mismatch",
            case.name
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

/// Covers: `ferray_ufunc::copysign` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::copysign`).
#[test]
fn copysign_matches_numpy() {
    run_binary("copysign.json", "copysign", ferray_ufunc::copysign);
}

/// Covers: `ferray_ufunc::float_power` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::float_power`).
#[test]
fn float_power_matches_numpy() {
    run_binary("float_power.json", "float_power", ferray_ufunc::float_power);
}

/// Covers: `ferray_ufunc::nextafter` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::nextafter`).
#[test]
fn nextafter_matches_numpy() {
    run_binary("nextafter.json", "nextafter", ferray_ufunc::nextafter);
}

/// Covers: `ferray_ufunc::spacing` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::spacing`).
#[test]
fn spacing_matches_numpy() {
    run_unary("spacing.json", "spacing", ferray_ufunc::spacing);
}

/// Covers: `ferray_ufunc::ldexp` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::ldexp`).
#[test]
fn ldexp_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("ldexp.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let exponent = make_i32_array(&case.inputs["n"]);
        let result = ferray_ufunc::ldexp(&arr, &exponent)
            .unwrap_or_else(|e| panic!("ldexp {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::frexp` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::frexp`).
#[test]
fn frexp_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("frexp.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let (mantissa, exponent) =
            ferray_ufunc::frexp(&arr).unwrap_or_else(|e| panic!("frexp {}: {e}", case.name));
        let expected_mantissa = parse_f64_data(&case.expected["mantissa"]["data"]);
        let expected_exponent = parse_i32_data(&case.expected["exponent"]["data"]);
        assert_f64_slice_ulp(
            mantissa.as_slice().expect("contiguous"),
            &expected_mantissa,
            case.tolerance_ulps,
            &format!("{} mantissa", case.name),
        );
        assert_eq!(
            exponent.as_slice().expect("contiguous"),
            expected_exponent.as_slice(),
            "frexp case {} exponent mismatch",
            case.name
        );
    }
}

/// Covers: `ferray_ufunc::modf` (re-export of
/// `ferray_ufunc::ops::floatintrinsic::modf`).
#[test]
fn modf_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("modf.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let (fractional, integral) =
            ferray_ufunc::modf(&arr).unwrap_or_else(|e| panic!("modf {}: {e}", case.name));
        let expected_fractional = parse_f64_data(&case.expected["fractional"]["data"]);
        let expected_integral = parse_f64_data(&case.expected["integral"]["data"]);
        assert_f64_slice_ulp(
            fractional.as_slice().expect("contiguous"),
            &expected_fractional,
            case.tolerance_ulps,
            &format!("{} fractional", case.name),
        );
        assert_f64_slice_ulp(
            integral.as_slice().expect("contiguous"),
            &expected_integral,
            case.tolerance_ulps,
            &format!("{} integral", case.name),
        );
    }
}
