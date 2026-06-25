//! Conformance tests for first-class ufunc objects and generic ufunc methods
//! against NumPy's `ufunc.reduce` / `accumulate` / `outer` / `at` APIs.

use ferray_core::Array;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_test_oracle::{
    assert_f64_slice_ulp, load_fixture, make_f64_array, parse_f64_data, parse_f64_value,
    parse_shape, parse_usize_data,
};

type ArrF64 = Array<f64, IxDyn>;
type ArrF64Ix1 = Array<f64, Ix1>;

// Surface coverage anchors for canonical paths exercised by this file:
// ferray_ufunc::ufunc_methods::accumulate_axis
// ferray_ufunc::ufunc_methods::at
// ferray_ufunc::ufunc_methods::outer
// ferray_ufunc::ufunc_methods::reduce_all
// ferray_ufunc::ufunc_methods::reduce_axes
// ferray_ufunc::ufunc_methods::reduce_axis
// ferray_ufunc::ufunc_methods::reduce_axis_keepdims
// ferray_ufunc::ufunc_object::Ufunc::accumulate
// ferray_ufunc::ufunc_object::Ufunc::at
// ferray_ufunc::ufunc_object::Ufunc::call
// ferray_ufunc::ufunc_object::Ufunc::identity
// ferray_ufunc::ufunc_object::Ufunc::name
// ferray_ufunc::ufunc_object::Ufunc::new
// ferray_ufunc::ufunc_object::Ufunc::outer
// ferray_ufunc::ufunc_object::Ufunc::reduce
// ferray_ufunc::ufunc_object::add_ufunc
// ferray_ufunc::ufunc_object::divide_ufunc
// ferray_ufunc::ufunc_object::multiply_ufunc
// ferray_ufunc::ufunc_object::subtract_ufunc

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn make_f64_ix1(value: &ferray_test_oracle::serde_json::Value) -> ArrF64Ix1 {
    let data = parse_f64_data(&value["data"]);
    let shape = parse_shape(&value["shape"]);
    assert_eq!(shape.len(), 1, "expected 1-D fixture input");
    ArrF64Ix1::from_vec(Ix1::new([shape[0]]), data).unwrap()
}

fn axis_value(value: &ferray_test_oracle::serde_json::Value) -> usize {
    value.as_u64().expect("axis integer") as usize
}

fn axes_value(value: &ferray_test_oracle::serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("axis list")
        .iter()
        .map(|v| v.as_u64().expect("axis integer") as usize)
        .collect()
}

fn assert_array_matches(
    name: &str,
    case_name: &str,
    result: &ArrF64,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_data = parse_f64_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
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

fn assert_scalar_matches(
    name: &str,
    case_name: &str,
    result: f64,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_data = parse_f64_data(&expected["data"]);
    assert_eq!(
        expected_data.len(),
        1,
        "{name} case {case_name} expected scalar"
    );
    assert_f64_slice_ulp(&[result], &expected_data, tolerance_ulps, case_name);
}

fn add_op(a: f64, b: f64) -> f64 {
    a + b
}

fn subtract_op(a: f64, b: f64) -> f64 {
    a - b
}

fn multiply_op(a: f64, b: f64) -> f64 {
    a * b
}

/// Covers: `ferray_ufunc::reduce_all`,
/// `ferray_ufunc::reduce_axis`,
/// `ferray_ufunc::reduce_axis_keepdims`,
/// `ferray_ufunc::reduce_axes`,
/// `ferray_ufunc::accumulate_axis`,
/// `ferray_ufunc::ufunc_outer`,
/// `ferray_ufunc::at`, and the canonical
/// `ferray_ufunc::ufunc_methods::{reduce_all, reduce_axis,
/// reduce_axis_keepdims, reduce_axes, accumulate_axis, outer, at}` paths.
#[test]
fn generic_ufunc_methods_match_numpy() {
    let suite = load_fixture(&ufunc_fixture("ufunc_methods.json"));
    for case in &suite.test_cases {
        let method = case.inputs["method"].as_str().expect("method string");
        let op = case.inputs["op"].as_str().expect("op string");
        match (method, op) {
            ("reduce_all", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let result = ferray_ufunc::reduce_all(&arr, 0.0, add_op);
                assert_scalar_matches(
                    "reduce_all",
                    &case.name,
                    result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("reduce_axis", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::reduce_axis(&arr, axis, 0.0, add_op)
                    .unwrap_or_else(|e| panic!("reduce_axis {}: {e}", case.name));
                assert_array_matches(
                    "reduce_axis",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("reduce_axis_keepdims", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let keepdims = case.inputs["keepdims"].as_bool().expect("keepdims bool");
                let result = ferray_ufunc::reduce_axis_keepdims(&arr, axis, 0.0, keepdims, add_op)
                    .unwrap_or_else(|e| panic!("reduce_axis_keepdims {}: {e}", case.name));
                assert_array_matches(
                    "reduce_axis_keepdims",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("reduce_axes" | "reduce_axes_keepdims", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axes = axes_value(&case.inputs["axis"]);
                let keepdims = case
                    .inputs
                    .get("keepdims")
                    .and_then(ferray_test_oracle::serde_json::Value::as_bool)
                    .unwrap_or(false);
                let result = ferray_ufunc::reduce_axes(&arr, &axes, 0.0, keepdims, add_op)
                    .unwrap_or_else(|e| panic!("reduce_axes {}: {e}", case.name));
                assert_array_matches(
                    "reduce_axes",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("accumulate_axis", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::accumulate_axis(&arr, axis, add_op)
                    .unwrap_or_else(|e| panic!("accumulate_axis add {}: {e}", case.name));
                assert_array_matches(
                    "accumulate_axis",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("accumulate_axis", "subtract") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::accumulate_axis(&arr, axis, subtract_op)
                    .unwrap_or_else(|e| panic!("accumulate_axis subtract {}: {e}", case.name));
                assert_array_matches(
                    "accumulate_axis",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("outer", "add") => {
                let a = make_f64_ix1(&case.inputs["a"]);
                let b = make_f64_ix1(&case.inputs["b"]);
                let result = ferray_ufunc::ufunc_outer(&a, &b, add_op)
                    .unwrap_or_else(|e| panic!("outer add {}: {e}", case.name));
                assert_array_matches(
                    "outer",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("outer", "multiply") => {
                let a = make_f64_ix1(&case.inputs["a"]);
                let b = make_f64_ix1(&case.inputs["b"]);
                let result = ferray_ufunc::ufunc_outer(&a, &b, multiply_op)
                    .unwrap_or_else(|e| panic!("outer multiply {}: {e}", case.name));
                assert_array_matches(
                    "outer",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("at", "add") => {
                let mut arr = make_f64_ix1(&case.inputs["x"]);
                let indices = parse_usize_data(&case.inputs["indices"]["data"]);
                let values = parse_f64_data(&case.inputs["values"]["data"]);
                ferray_ufunc::at(&mut arr, &indices, &values, add_op)
                    .unwrap_or_else(|e| panic!("at {}: {e}", case.name));
                assert_f64_slice_ulp(
                    arr.as_slice().expect("contiguous"),
                    &parse_f64_data(&case.expected["data"]),
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            _ => {}
        }
    }
}

/// Covers: `ferray_ufunc::Ufunc`, `ferray_ufunc::add_ufunc`,
/// `ferray_ufunc::subtract_ufunc`, `ferray_ufunc::multiply_ufunc`,
/// `ferray_ufunc::divide_ufunc`, the canonical
/// `ferray_ufunc::ufunc_object::{Ufunc, add_ufunc, subtract_ufunc,
/// multiply_ufunc, divide_ufunc}` paths, and
/// `ferray_ufunc::ufunc_object::Ufunc::{new, name, identity, call, reduce,
/// accumulate, outer, at}`.
#[test]
fn first_class_ufunc_object_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("ufunc_methods.json"));
    for case in &suite.test_cases {
        let method = case.inputs["method"].as_str().expect("method string");
        let op = case.inputs["op"].as_str().expect("op string");
        match (method, op) {
            ("metadata", "add") => {
                let add = ferray_ufunc::add_ufunc::<f64>();
                assert_eq!(add.name(), case.expected["name"].as_str().unwrap());
                assert_eq!(add.identity(), parse_f64_value(&case.expected["identity"]));
            }
            ("metadata", "multiply") => {
                let multiply = ferray_ufunc::multiply_ufunc::<f64>();
                assert_eq!(multiply.name(), case.expected["name"].as_str().unwrap());
                assert_eq!(
                    multiply.identity(),
                    parse_f64_value(&case.expected["identity"])
                );
            }
            ("custom_reduce", "maximum") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let max = ferray_ufunc::Ufunc::new("maximum", f64::NEG_INFINITY, f64::max);
                let result = max
                    .reduce(&arr, axis)
                    .unwrap_or_else(|e| panic!("custom max reduce {}: {e}", case.name));
                assert_array_matches(
                    "custom max reduce",
                    &case.name,
                    &result,
                    &case.expected,
                    case.tolerance_ulps,
                );
            }
            ("call", "add") => {
                let a = make_f64_array(&case.inputs["a"]);
                let b = make_f64_array(&case.inputs["b"]);
                let result = ferray_ufunc::add_ufunc::<f64>()
                    .call(&a, &b)
                    .unwrap_or_else(|e| panic!("add call {}: {e}", case.name));
                assert_array_matches("add call", &case.name, &result, &case.expected, 0);
            }
            ("call", "subtract") => {
                let a = make_f64_array(&case.inputs["a"]);
                let b = make_f64_array(&case.inputs["b"]);
                let result = ferray_ufunc::subtract_ufunc::<f64>()
                    .call(&a, &b)
                    .unwrap_or_else(|e| panic!("subtract call {}: {e}", case.name));
                assert_array_matches("subtract call", &case.name, &result, &case.expected, 0);
            }
            ("call", "multiply") => {
                let a = make_f64_array(&case.inputs["a"]);
                let b = make_f64_array(&case.inputs["b"]);
                let result = ferray_ufunc::multiply_ufunc::<f64>()
                    .call(&a, &b)
                    .unwrap_or_else(|e| panic!("multiply call {}: {e}", case.name));
                assert_array_matches("multiply call", &case.name, &result, &case.expected, 0);
            }
            ("call", "divide") => {
                let a = make_f64_array(&case.inputs["a"]);
                let b = make_f64_array(&case.inputs["b"]);
                let result = ferray_ufunc::divide_ufunc::<f64>()
                    .call(&a, &b)
                    .unwrap_or_else(|e| panic!("divide call {}: {e}", case.name));
                assert_array_matches("divide call", &case.name, &result, &case.expected, 0);
            }
            ("reduce_axis", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::add_ufunc::<f64>()
                    .reduce(&arr, axis)
                    .unwrap_or_else(|e| panic!("add reduce {}: {e}", case.name));
                assert_array_matches("add reduce", &case.name, &result, &case.expected, 0);
            }
            ("accumulate_axis", "add") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::add_ufunc::<f64>()
                    .accumulate(&arr, axis)
                    .unwrap_or_else(|e| panic!("add accumulate {}: {e}", case.name));
                assert_array_matches("add accumulate", &case.name, &result, &case.expected, 0);
            }
            ("accumulate_axis", "subtract") => {
                let arr = make_f64_array(&case.inputs["x"]);
                let axis = axis_value(&case.inputs["axis"]);
                let result = ferray_ufunc::subtract_ufunc::<f64>()
                    .accumulate(&arr, axis)
                    .unwrap_or_else(|e| panic!("subtract accumulate {}: {e}", case.name));
                assert_array_matches(
                    "subtract accumulate",
                    &case.name,
                    &result,
                    &case.expected,
                    0,
                );
            }
            ("outer", "add") => {
                let a = make_f64_ix1(&case.inputs["a"]);
                let b = make_f64_ix1(&case.inputs["b"]);
                let result = ferray_ufunc::add_ufunc::<f64>()
                    .outer(&a, &b)
                    .unwrap_or_else(|e| panic!("add outer {}: {e}", case.name));
                assert_array_matches("add outer", &case.name, &result, &case.expected, 0);
            }
            ("outer", "multiply") => {
                let a = make_f64_ix1(&case.inputs["a"]);
                let b = make_f64_ix1(&case.inputs["b"]);
                let result = ferray_ufunc::multiply_ufunc::<f64>()
                    .outer(&a, &b)
                    .unwrap_or_else(|e| panic!("multiply outer {}: {e}", case.name));
                assert_array_matches("multiply outer", &case.name, &result, &case.expected, 0);
            }
            ("at", "add") => {
                let mut arr = make_f64_ix1(&case.inputs["x"]);
                let indices = parse_usize_data(&case.inputs["indices"]["data"]);
                let values = parse_f64_data(&case.inputs["values"]["data"]);
                ferray_ufunc::add_ufunc::<f64>()
                    .at(&mut arr, &indices, &values)
                    .unwrap_or_else(|e| panic!("add at {}: {e}", case.name));
                assert_f64_slice_ulp(
                    arr.as_slice().expect("contiguous"),
                    &parse_f64_data(&case.expected["data"]),
                    case.tolerance_ulps,
                    &case.name,
                );
            }
            _ => {}
        }
    }
}
