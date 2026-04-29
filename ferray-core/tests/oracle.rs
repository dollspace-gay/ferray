//! Oracle tests: validate ferray-core creation and manipulation against `NumPy` fixtures.

// Oracle tests cross fixture data through `f64` JSON values, recover bit
// patterns, and ULP-compare against NumPy reference outputs — both the
// casts and the float comparisons are part of the contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    clippy::similar_names
)]

use ferray_core::Array;
use ferray_core::creation;
use ferray_core::dimension::{Ix2, IxDyn};
use ferray_test_oracle::*;

fn core_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("core").join(name)
}

// ---------------------------------------------------------------------------
// Creation functions
// ---------------------------------------------------------------------------

#[test]
fn oracle_linspace() {
    let suite = load_fixture(&core_path("linspace.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let start = parse_f64_value(&case.inputs["start"]);
        let stop = parse_f64_value(&case.inputs["stop"]);
        let num = case.inputs["num"].as_u64().unwrap() as usize;
        let result = creation::linspace::<f64>(start, stop, num, true).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_arange() {
    let suite = load_fixture(&core_path("arange.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let start = parse_f64_value(&case.inputs["start"]);
        let stop = parse_f64_value(&case.inputs["stop"]);
        let step = case.inputs.get("step").map_or(1.0, parse_f64_value);
        let result = creation::arange(start, stop, step).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_zeros() {
    let suite = load_fixture(&core_path("zeros.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        if shape.is_empty() {
            continue;
        }
        let result = creation::zeros::<f64, IxDyn>(IxDyn::new(&shape)).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_ones() {
    let suite = load_fixture(&core_path("ones.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        if shape.is_empty() {
            continue;
        }
        let result = creation::ones::<f64, IxDyn>(IxDyn::new(&shape)).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_full() {
    let suite = load_fixture(&core_path("full.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&case.inputs["shape"]);
        if shape.is_empty() {
            continue;
        }
        let fill_value = parse_f64_value(&case.inputs["fill_value"]);
        let result = creation::full::<f64, IxDyn>(IxDyn::new(&shape), fill_value).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_eye() {
    let suite = load_fixture(&core_path("eye.json"));
    for case in &suite.test_cases {
        let dtype = case
            .inputs
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("float64");
        if dtype != "float64" {
            continue;
        }
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let m = case
            .inputs
            .get("m")
            .and_then(ferray_test_oracle::serde_json::Value::as_u64)
            .unwrap_or(n as u64) as usize;
        let k = case
            .inputs
            .get("k")
            .and_then(ferray_test_oracle::serde_json::Value::as_i64)
            .unwrap_or(0) as isize;
        let result = creation::eye::<f64>(n, m, k).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

// ---------------------------------------------------------------------------
// Shape manipulation
// ---------------------------------------------------------------------------

#[test]
fn oracle_transpose() {
    let suite = load_fixture(&core_path("transpose.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.len() != 2 {
            continue;
        }
        let arr = Array::<f64, Ix2>::from_vec(
            Ix2::new([shape[0], shape[1]]),
            parse_f64_data(&input["data"]),
        )
        .unwrap();
        let result = ferray_core::manipulation::transpose(&arr, None).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_reshape() {
    let suite = load_fixture(&core_path("reshape.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let new_shape_val = &case.inputs["new_shape"];
        let new_shape: Vec<usize> = new_shape_val
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let result = ferray_core::manipulation::reshape(&arr, &new_shape).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        // Data should be identical (reshape doesn't change data order)
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_flatten() {
    let suite = load_fixture(&core_path("flatten.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_core::manipulation::flatten(&arr).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_squeeze() {
    let suite = load_fixture(&core_path("squeeze.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case
            .inputs
            .get("axis")
            .and_then(ferray_test_oracle::serde_json::Value::as_u64)
            .map(|v| v as usize);
        let result = ferray_core::manipulation::squeeze(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_expand_dims() {
    let suite = load_fixture(&core_path("expand_dims.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let axis = case.inputs["axis"].as_u64().unwrap() as usize;
        let result = ferray_core::manipulation::expand_dims(&arr, axis).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        // Data identical, only shape changes
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}
