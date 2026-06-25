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
    assert_f64_slice_ulp, load_fixture, make_f64_array, make_i64_array, parse_f64_data,
    parse_f64_value, parse_i64_data, parse_shape, should_skip_f64,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type ArrF64Ix1 = ferray_core::Array<f64, ferray_core::dimension::Ix1>;
type ArrI32 = ferray_core::Array<i32, ferray_core::dimension::IxDyn>;
type ArrI64 = ferray_core::Array<i64, ferray_core::dimension::IxDyn>;
type UnaryFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type BinaryFn = fn(&ArrF64, &ArrF64) -> ferray_core::error::FerrayResult<ArrF64>;
type I64BinaryFn = fn(&ArrI64, &ArrI64) -> ferray_core::error::FerrayResult<ArrI64>;
type UnaryIntoFn = fn(&ArrF64, &mut ArrF64) -> ferray_core::error::FerrayResult<()>;
type BinaryIntoFn = fn(&ArrF64, &ArrF64, &mut ArrF64) -> ferray_core::error::FerrayResult<()>;
type AxisReduceFn = fn(&ArrF64, usize) -> ferray_core::error::FerrayResult<ArrF64>;
type AxisReduceKeepdimsFn = fn(&ArrF64, usize, bool) -> ferray_core::error::FerrayResult<ArrF64>;
type AxesReduceFn = fn(&ArrF64, &[usize], bool) -> ferray_core::error::FerrayResult<ArrF64>;
type ReduceAllFn = fn(&ArrF64) -> f64;
type OuterFn = fn(&ArrF64Ix1, &ArrF64Ix1) -> ferray_core::error::FerrayResult<ArrF64>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn make_f64_ix1(value: &ferray_test_oracle::serde_json::Value) -> ArrF64Ix1 {
    let data = parse_f64_data(&value["data"]);
    let shape = parse_shape(&value["shape"]);
    assert_eq!(shape.len(), 1, "expected 1-D fixture input");
    ArrF64Ix1::from_vec(ferray_core::dimension::Ix1::new([shape[0]]), data).unwrap()
}

fn make_i32_array(value: &ferray_test_oracle::serde_json::Value) -> ArrI32 {
    let data = parse_i64_data(&value["data"])
        .into_iter()
        .map(|x| i32::try_from(x).expect("fixture value fits i32"))
        .collect();
    let shape = parse_shape(&value["shape"]);
    ArrI32::from_vec(ferray_core::dimension::IxDyn::new(&shape), data).unwrap()
}

fn assert_f64_array_matches(
    name: &str,
    case_name: &str,
    result: &ArrF64,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    let expected_data = parse_f64_data(&expected["data"]);
    assert_f64_slice_ulp(
        result.as_slice().expect("contiguous"),
        &expected_data,
        tolerance_ulps,
        case_name,
    );
}

fn optional_f64_vec(value: Option<&ferray_test_oracle::serde_json::Value>) -> Option<Vec<f64>> {
    value.map(|v| parse_f64_data(&v["data"]))
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

fn run_axis_unary<F>(fixture: &str, name: &str, f: F)
where
    F: Fn(&ArrF64, Option<usize>) -> ferray_core::error::FerrayResult<ArrF64>,
{
    let suite = load_fixture(&ufunc_fixture(fixture));
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
            .map(|axis| axis as usize);
        let result = f(&arr, axis).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_eq!(
            result.shape(),
            expected_shape.as_slice(),
            "{name} case {} shape mismatch",
            case.name
        );
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
        assert_f64_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn method_case(name: &str) -> ferray_test_oracle::TestCase {
    let suite = load_fixture(&ufunc_fixture("ufunc_methods.json"));
    suite
        .test_cases
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing ufunc_methods case {name}"))
}

fn axis_from_case(case: &ferray_test_oracle::TestCase) -> usize {
    case.inputs["axis"].as_u64().expect("axis integer") as usize
}

fn keepdims_from_case(case: &ferray_test_oracle::TestCase) -> bool {
    case.inputs
        .get("keepdims")
        .and_then(ferray_test_oracle::serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn axes_from_case(case: &ferray_test_oracle::TestCase) -> Vec<usize> {
    case.inputs["axis"]
        .as_array()
        .expect("axes list")
        .iter()
        .map(|axis| axis.as_u64().expect("axis integer") as usize)
        .collect()
}

fn run_axis_reduce_case(case_name: &str, name: &str, f: AxisReduceFn) {
    let case = method_case(case_name);
    let arr = make_f64_array(&case.inputs["x"]);
    let axis = axis_from_case(&case);
    let result = f(&arr, axis).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
    assert_f64_array_matches(
        name,
        &case.name,
        &result,
        &case.expected,
        case.tolerance_ulps,
    );
}

fn run_axis_reduce_keepdims_case(case_name: &str, name: &str, f: AxisReduceKeepdimsFn) {
    let case = method_case(case_name);
    let arr = make_f64_array(&case.inputs["x"]);
    let axis = axis_from_case(&case);
    let keepdims = keepdims_from_case(&case);
    let result = f(&arr, axis, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
    assert_f64_array_matches(
        name,
        &case.name,
        &result,
        &case.expected,
        case.tolerance_ulps,
    );
}

fn run_axes_reduce_case(case_name: &str, name: &str, f: AxesReduceFn) {
    let case = method_case(case_name);
    let arr = make_f64_array(&case.inputs["x"]);
    let axes = axes_from_case(&case);
    let keepdims = keepdims_from_case(&case);
    let result = f(&arr, &axes, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
    assert_f64_array_matches(
        name,
        &case.name,
        &result,
        &case.expected,
        case.tolerance_ulps,
    );
}

fn run_reduce_all_case(case_name: &str, _name: &str, f: ReduceAllFn) {
    let case = method_case(case_name);
    let arr = make_f64_array(&case.inputs["x"]);
    let result = f(&arr);
    let expected = parse_f64_data(&case.expected["data"]);
    assert_f64_slice_ulp(&[result], &expected, case.tolerance_ulps, &case.name);
}

fn run_outer_case(case_name: &str, name: &str, f: OuterFn) {
    let case = method_case(case_name);
    let a = make_f64_ix1(&case.inputs["a"]);
    let b = make_f64_ix1(&case.inputs["b"]);
    let result = f(&a, &b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
    assert_f64_array_matches(
        name,
        &case.name,
        &result,
        &case.expected,
        case.tolerance_ulps,
    );
}

fn run_nan_axis_reduce(fixture: &str, name: &str, f: AxisReduceKeepdimsFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let axis = axis_from_case(case);
        let keepdims = keepdims_from_case(case);
        let result =
            f(&arr, axis, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f64_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_axes_reduce(fixture: &str, name: &str, f: AxesReduceFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let axes = case.inputs["axes"]
            .as_array()
            .expect("axes list")
            .iter()
            .map(|axis| axis.as_u64().expect("axis integer") as usize)
            .collect::<Vec<_>>();
        let keepdims = keepdims_from_case(case);
        let result =
            f(&arr, &axes, keepdims).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f64_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_reduce_all(fixture: &str, _name: &str, f: ReduceAllFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let result = f(&arr);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(&[result], &expected, case.tolerance_ulps, &case.name);
    }
}

fn run_binary_error(fixture: &str, name: &str, f: BinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        assert!(
            matches!(
                case.expected["error_type"].as_str(),
                Some("TypeError" | "UFuncTypeError")
            ),
            "{name} case {} fixture should record NumPy TypeError",
            case.name
        );
        let arr_a = make_f64_array(&case.inputs["a"]);
        let arr_b = make_f64_array(&case.inputs["b"]);
        assert!(
            f(&arr_a, &arr_b).is_err(),
            "{name} case {} should reject the same float domain NumPy rejects",
            case.name
        );
    }
}

fn run_i64_binary(fixture: &str, name: &str, f: I64BinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr_a = make_i64_array(&case.inputs["a"]);
        let arr_b = make_i64_array(&case.inputs["b"]);
        let result = f(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected = parse_i64_data(&case.expected["data"]);
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

fn run_binary_into(fixture: &str, name: &str, f: BinaryIntoFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let shape_a = parse_shape(&a["shape"]);
        let shape_b = parse_shape(&b["shape"]);
        if shape_a.is_empty() || shape_a != shape_b {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let expected = parse_f64_data(&case.expected["data"]);
        let expected_shape = parse_shape(&case.expected["shape"]);
        let mut out = ArrF64::from_vec(
            ferray_core::dimension::IxDyn::new(&expected_shape),
            vec![0.0; expected.len()],
        )
        .unwrap();
        f(&arr_a, &arr_b, &mut out).unwrap_or_else(|e| panic!("{name}_into {}: {e}", case.name));
        assert_f64_slice_ulp(
            out.as_slice().expect("contiguous"),
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

/// Covers: `ferray_ufunc::absolute_into`
/// (`ferray_ufunc::ops::arithmetic::absolute_into`) with NumPy `absolute`
/// fixtures.
#[test]
fn absolute_into_matches_numpy() {
    run_unary_into("absolute.json", "absolute", ferray_ufunc::absolute_into);
}

/// Covers: `ferray_ufunc::fabs` (re-export of
/// `ferray_ufunc::ops::arithmetic::fabs`).
#[test]
fn fabs_matches_numpy() {
    run_unary("fabs.json", "fabs", ferray_ufunc::fabs);
}

/// Covers: `ferray_ufunc::negative` (re-export of
/// `ferray_ufunc::ops::arithmetic::negative`).
#[test]
fn negative_matches_numpy() {
    run_unary("negative.json", "negative", ferray_ufunc::negative);
}

/// Covers: `ferray_ufunc::positive` (re-export of
/// `ferray_ufunc::ops::arithmetic::positive`).
#[test]
fn positive_matches_numpy() {
    run_unary("positive.json", "positive", ferray_ufunc::positive);
}

/// Covers: `ferray_ufunc::sign` (re-export of
/// `ferray_ufunc::ops::arithmetic::sign`).
#[test]
fn sign_matches_numpy() {
    run_unary("sign.json", "sign", ferray_ufunc::sign);
}

/// Covers: `ferray_ufunc::negative_into`
/// (`ferray_ufunc::ops::arithmetic::negative_into`) with NumPy `negative`
/// fixtures.
#[test]
fn negative_into_matches_numpy() {
    run_unary_into("negative.json", "negative", ferray_ufunc::negative_into);
}

/// Covers: `ferray_ufunc::sqrt` (re-export of
/// `ferray_ufunc::ops::arithmetic::sqrt`).
#[test]
fn sqrt_matches_numpy() {
    run_unary("sqrt.json", "sqrt", ferray_ufunc::sqrt);
}

/// Covers: `ferray_ufunc::sqrt_into`
/// (`ferray_ufunc::ops::arithmetic::sqrt_into`) with NumPy `sqrt` fixtures.
#[test]
fn sqrt_into_matches_numpy() {
    run_unary_into("sqrt.json", "sqrt", ferray_ufunc::sqrt_into);
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

/// Covers: `ferray_ufunc::square_into`
/// (`ferray_ufunc::ops::arithmetic::square_into`) with NumPy `square`
/// fixtures.
#[test]
fn square_into_matches_numpy() {
    run_unary_into("square.json", "square", ferray_ufunc::square_into);
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

/// Covers: `ferray_ufunc::i0` (re-export of
/// `ferray_ufunc::ops::special::i0`) and the scalar kernel
/// `ferray_ufunc::ops::special::bessel_i0_scalar`.
#[test]
fn i0_matches_numpy() {
    run_unary("i0.json", "i0", ferray_ufunc::i0);

    let suite = load_fixture(&ufunc_fixture("i0.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let values = parse_f64_data(&input["data"]);
        let expected = parse_f64_data(&case.expected["data"]);
        for (idx, (value, expected)) in values.iter().zip(expected.iter()).enumerate() {
            let result = ferray_ufunc::ops::special::bessel_i0_scalar(*value);
            assert_f64_slice_ulp(
                &[result],
                &[*expected],
                case.tolerance_ulps,
                &format!("{} scalar {idx}", case.name),
            );
        }
    }
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

/// Covers: `ferray_ufunc::add_into`
/// (`ferray_ufunc::ops::arithmetic::add_into`) with same-shape NumPy `add`
/// fixtures.
#[test]
fn add_into_matches_numpy() {
    run_binary_into("add.json", "add", ferray_ufunc::add_into);
}

/// Covers: `ferray_ufunc::add_broadcast` and
/// `ferray_ufunc::ops::arithmetic::add_broadcast` with NumPy broadcasted
/// `add` fixtures.
#[test]
fn add_broadcast_matches_numpy() {
    run_binary("add.json", "add_broadcast", ferray_ufunc::add_broadcast);
    run_binary(
        "add.json",
        "ops::arithmetic::add_broadcast",
        ferray_ufunc::ops::arithmetic::add_broadcast,
    );
}

/// Covers: `ferray_ufunc::subtract` (re-export of
/// `ferray_ufunc::ops::arithmetic::subtract`).
#[test]
fn subtract_matches_numpy() {
    run_binary("subtract.json", "subtract", ferray_ufunc::subtract);
}

/// Covers: `ferray_ufunc::subtract_into`
/// (`ferray_ufunc::ops::arithmetic::subtract_into`) with same-shape NumPy
/// `subtract` fixtures.
#[test]
fn subtract_into_matches_numpy() {
    run_binary_into("subtract.json", "subtract", ferray_ufunc::subtract_into);
}

/// Covers: `ferray_ufunc::subtract_broadcast` and
/// `ferray_ufunc::ops::arithmetic::subtract_broadcast` with NumPy broadcasted
/// `subtract` fixtures.
#[test]
fn subtract_broadcast_matches_numpy() {
    run_binary(
        "subtract.json",
        "subtract_broadcast",
        ferray_ufunc::subtract_broadcast,
    );
    run_binary(
        "subtract.json",
        "ops::arithmetic::subtract_broadcast",
        ferray_ufunc::ops::arithmetic::subtract_broadcast,
    );
}

/// Covers: `ferray_ufunc::multiply` (re-export of
/// `ferray_ufunc::ops::arithmetic::multiply`).
#[test]
fn multiply_matches_numpy() {
    run_binary("multiply.json", "multiply", ferray_ufunc::multiply);
}

/// Covers: `ferray_ufunc::multiply_into`
/// (`ferray_ufunc::ops::arithmetic::multiply_into`) with same-shape NumPy
/// `multiply` fixtures.
#[test]
fn multiply_into_matches_numpy() {
    run_binary_into("multiply.json", "multiply", ferray_ufunc::multiply_into);
}

/// Covers: `ferray_ufunc::multiply_broadcast` and
/// `ferray_ufunc::ops::arithmetic::multiply_broadcast` with NumPy broadcasted
/// `multiply` fixtures.
#[test]
fn multiply_broadcast_matches_numpy() {
    run_binary(
        "multiply.json",
        "multiply_broadcast",
        ferray_ufunc::multiply_broadcast,
    );
    run_binary(
        "multiply.json",
        "ops::arithmetic::multiply_broadcast",
        ferray_ufunc::ops::arithmetic::multiply_broadcast,
    );
}

/// Covers: `ferray_ufunc::divide` (re-export of
/// `ferray_ufunc::ops::arithmetic::divide`).
#[test]
fn divide_matches_numpy() {
    run_binary("divide.json", "divide", ferray_ufunc::divide);
}

/// Covers: `ferray_ufunc::true_divide` (re-export of
/// `ferray_ufunc::ops::arithmetic::true_divide`).
#[test]
fn true_divide_matches_numpy() {
    run_binary("true_divide.json", "true_divide", ferray_ufunc::true_divide);
}

/// Covers: `ferray_ufunc::divide_into`
/// (`ferray_ufunc::ops::arithmetic::divide_into`) with same-shape NumPy
/// `divide` fixtures.
#[test]
fn divide_into_matches_numpy() {
    run_binary_into("divide.json", "divide", ferray_ufunc::divide_into);
}

/// Covers: `ferray_ufunc::divide_broadcast` and
/// `ferray_ufunc::ops::arithmetic::divide_broadcast` with NumPy broadcasted
/// `divide` fixtures.
#[test]
fn divide_broadcast_matches_numpy() {
    run_binary(
        "divide.json",
        "divide_broadcast",
        ferray_ufunc::divide_broadcast,
    );
    run_binary(
        "divide.json",
        "ops::arithmetic::divide_broadcast",
        ferray_ufunc::ops::arithmetic::divide_broadcast,
    );
}

/// Covers: `ferray_ufunc::floor_divide` (re-export of
/// `ferray_ufunc::ops::arithmetic::floor_divide`).
#[test]
fn floor_divide_matches_numpy() {
    run_binary(
        "floor_divide.json",
        "floor_divide",
        ferray_ufunc::floor_divide,
    );
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

// ---------------------------------------------------------------------------
// Operator-style arithmetic wrappers
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::array_add` and
/// `ferray_ufunc::operator_overloads::array_add` with NumPy `add` fixtures.
#[test]
fn array_add_matches_numpy() {
    run_binary("add.json", "array_add", ferray_ufunc::array_add);
}

/// Covers: `ferray_ufunc::array_sub` and
/// `ferray_ufunc::operator_overloads::array_sub` with NumPy `subtract`
/// fixtures.
#[test]
fn array_sub_matches_numpy() {
    run_binary("subtract.json", "array_sub", ferray_ufunc::array_sub);
}

/// Covers: `ferray_ufunc::array_mul` and
/// `ferray_ufunc::operator_overloads::array_mul` with NumPy `multiply`
/// fixtures.
#[test]
fn array_mul_matches_numpy() {
    run_binary("multiply.json", "array_mul", ferray_ufunc::array_mul);
}

/// Covers: `ferray_ufunc::array_div` and
/// `ferray_ufunc::operator_overloads::array_div` with NumPy `divide`
/// fixtures.
#[test]
fn array_div_matches_numpy() {
    run_binary("divide.json", "array_div", ferray_ufunc::array_div);
}

/// Covers: `ferray_ufunc::array_rem` and
/// `ferray_ufunc::operator_overloads::array_rem` with NumPy `remainder`
/// fixtures.
#[test]
fn array_rem_matches_numpy() {
    run_binary("remainder.json", "array_rem", ferray_ufunc::array_rem);
}

/// Covers: `ferray_ufunc::array_neg` and
/// `ferray_ufunc::operator_overloads::array_neg` with NumPy `negative`
/// fixtures.
#[test]
fn array_neg_matches_numpy() {
    run_unary("negative.json", "array_neg", ferray_ufunc::array_neg);
}

/// Covers: `ferray_ufunc::mod_` (re-export of
/// `ferray_ufunc::ops::arithmetic::mod_`).
#[test]
fn mod_matches_numpy() {
    run_binary("mod.json", "mod_", ferray_ufunc::mod_);
}

/// Covers: `ferray_ufunc::fmod` (re-export of
/// `ferray_ufunc::ops::arithmetic::fmod`).
#[test]
fn fmod_matches_numpy() {
    run_binary("fmod.json", "fmod", ferray_ufunc::fmod);
}

/// Covers: `ferray_ufunc::divmod` (re-export of
/// `ferray_ufunc::ops::arithmetic::divmod`).
#[test]
fn divmod_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("divmod.json"));
    for case in &suite.test_cases {
        let a = &case.inputs["a"];
        let b = &case.inputs["b"];
        if should_skip_f64(a) || should_skip_f64(b) {
            continue;
        }
        let arr_a = make_f64_array(a);
        let arr_b = make_f64_array(b);
        let (quotient, remainder) = ferray_ufunc::divmod(&arr_a, &arr_b)
            .unwrap_or_else(|e| panic!("divmod {}: {e}", case.name));
        let expected_quotient = parse_f64_data(&case.expected["quotient"]["data"]);
        let expected_remainder = parse_f64_data(&case.expected["remainder"]["data"]);
        assert_f64_slice_ulp(
            quotient.as_slice().expect("contiguous"),
            &expected_quotient,
            case.tolerance_ulps,
            &format!("{} quotient", case.name),
        );
        assert_f64_slice_ulp(
            remainder.as_slice().expect("contiguous"),
            &expected_remainder,
            case.tolerance_ulps,
            &format!("{} remainder", case.name),
        );
    }
}

/// Covers: `ferray_ufunc::gcd` (re-export of
/// `ferray_ufunc::ops::arithmetic::gcd`) rejecting the float input domain that
/// NumPy rejects with `TypeError`.
#[test]
fn gcd_rejects_float_like_numpy() {
    run_binary_error("gcd.json", "gcd", ferray_ufunc::gcd);
}

/// Covers: `ferray_ufunc::gcd_int` (re-export of
/// `ferray_ufunc::ops::arithmetic::gcd_int`) with NumPy `gcd` integer
/// fixtures.
#[test]
fn gcd_int_matches_numpy() {
    run_i64_binary("gcd_int.json", "gcd_int", ferray_ufunc::gcd_int);
}

/// Covers: `ferray_ufunc::lcm` (re-export of
/// `ferray_ufunc::ops::arithmetic::lcm`) rejecting the float input domain that
/// NumPy rejects with `TypeError`.
#[test]
fn lcm_rejects_float_like_numpy() {
    run_binary_error("lcm.json", "lcm", ferray_ufunc::lcm);
}

/// Covers: `ferray_ufunc::lcm_int` (re-export of
/// `ferray_ufunc::ops::arithmetic::lcm_int`) with NumPy `lcm` integer
/// fixtures.
#[test]
fn lcm_int_matches_numpy() {
    run_i64_binary("lcm_int.json", "lcm_int", ferray_ufunc::lcm_int);
}

/// Covers: `ferray_ufunc::heaviside` (re-export of
/// `ferray_ufunc::ops::arithmetic::heaviside`).
#[test]
fn heaviside_matches_numpy() {
    run_binary("heaviside.json", "heaviside", ferray_ufunc::heaviside);
}

/// Covers mixed-type promoted arithmetic re-exports
/// `ferray_ufunc::add_promoted`, `ferray_ufunc::subtract_promoted`,
/// `ferray_ufunc::multiply_promoted`, `ferray_ufunc::divide_promoted`, and
/// canonical paths `ferray_ufunc::promoted::add_promoted`,
/// `ferray_ufunc::promoted::subtract_promoted`,
/// `ferray_ufunc::promoted::multiply_promoted`, and
/// `ferray_ufunc::promoted::divide_promoted`.
#[test]
fn promoted_arithmetic_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("promoted_arithmetic.json"));
    for case in &suite.test_cases {
        let a = make_i32_array(&case.inputs["a"]);
        let b = make_f64_array(&case.inputs["b"]);
        let op = case.inputs["op"].as_str().expect("promoted op name");

        let root_result = match op {
            "add_promoted" => ferray_ufunc::add_promoted(&a, &b),
            "subtract_promoted" => ferray_ufunc::subtract_promoted(&a, &b),
            "multiply_promoted" => ferray_ufunc::multiply_promoted(&a, &b),
            "divide_promoted" => ferray_ufunc::divide_promoted(&a, &b),
            other => panic!("unexpected promoted op {other}"),
        }
        .unwrap_or_else(|e| panic!("{op} root {}: {e}", case.name));
        assert_f64_array_matches(
            op,
            &case.name,
            &root_result,
            &case.expected,
            case.tolerance_ulps,
        );

        let canonical_result = match op {
            "add_promoted" => ferray_ufunc::promoted::add_promoted(&a, &b),
            "subtract_promoted" => ferray_ufunc::promoted::subtract_promoted(&a, &b),
            "multiply_promoted" => ferray_ufunc::promoted::multiply_promoted(&a, &b),
            "divide_promoted" => ferray_ufunc::promoted::divide_promoted(&a, &b),
            other => panic!("unexpected promoted op {other}"),
        }
        .unwrap_or_else(|e| panic!("{op} canonical {}: {e}", case.name));
        assert_f64_array_matches(
            op,
            &format!("{} canonical", case.name),
            &canonical_result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

/// Covers: `ferray_ufunc::add_reduce` and
/// `ferray_ufunc::ops::arithmetic::add_reduce`.
#[test]
fn add_reduce_matches_numpy() {
    run_axis_reduce_case("add_reduce_axis", "add_reduce", ferray_ufunc::add_reduce);
    run_axis_reduce_case(
        "add_reduce_axis",
        "ops::arithmetic::add_reduce",
        ferray_ufunc::ops::arithmetic::add_reduce,
    );
}

/// Covers: `ferray_ufunc::add_reduce_keepdims` and
/// `ferray_ufunc::ops::arithmetic::add_reduce_keepdims`.
#[test]
fn add_reduce_keepdims_matches_numpy() {
    run_axis_reduce_keepdims_case(
        "add_reduce_axis_keepdims",
        "add_reduce_keepdims",
        ferray_ufunc::add_reduce_keepdims,
    );
    run_axis_reduce_keepdims_case(
        "add_reduce_axis_keepdims",
        "ops::arithmetic::add_reduce_keepdims",
        ferray_ufunc::ops::arithmetic::add_reduce_keepdims,
    );
}

/// Covers: `ferray_ufunc::add_reduce_axes` and
/// `ferray_ufunc::ops::arithmetic::add_reduce_axes`.
#[test]
fn add_reduce_axes_matches_numpy() {
    for case_name in ["add_reduce_axes", "add_reduce_axes_keepdims"] {
        run_axes_reduce_case(case_name, "add_reduce_axes", ferray_ufunc::add_reduce_axes);
        run_axes_reduce_case(
            case_name,
            "ops::arithmetic::add_reduce_axes",
            ferray_ufunc::ops::arithmetic::add_reduce_axes,
        );
    }
}

/// Covers: `ferray_ufunc::add_reduce_all` and
/// `ferray_ufunc::ops::arithmetic::add_reduce_all`.
#[test]
fn add_reduce_all_matches_numpy() {
    run_reduce_all_case(
        "add_reduce_all",
        "add_reduce_all",
        ferray_ufunc::add_reduce_all,
    );
    run_reduce_all_case(
        "add_reduce_all",
        "ops::arithmetic::add_reduce_all",
        ferray_ufunc::ops::arithmetic::add_reduce_all,
    );
}

/// Covers: `ferray_ufunc::add_accumulate` and
/// `ferray_ufunc::ops::arithmetic::add_accumulate`.
#[test]
fn add_accumulate_matches_numpy() {
    run_axis_reduce_case(
        "add_accumulate_axis",
        "add_accumulate",
        ferray_ufunc::add_accumulate,
    );
    run_axis_reduce_case(
        "add_accumulate_axis",
        "ops::arithmetic::add_accumulate",
        ferray_ufunc::ops::arithmetic::add_accumulate,
    );
}

/// Covers: `ferray_ufunc::multiply_outer` and
/// `ferray_ufunc::ops::arithmetic::multiply_outer`.
#[test]
fn multiply_outer_matches_numpy() {
    run_outer_case(
        "multiply_outer",
        "multiply_outer",
        ferray_ufunc::multiply_outer,
    );
    run_outer_case(
        "multiply_outer",
        "ops::arithmetic::multiply_outer",
        ferray_ufunc::ops::arithmetic::multiply_outer,
    );
}

/// Covers: `ferray_ufunc::cumsum` (re-export of
/// `ferray_ufunc::ops::arithmetic::cumsum`).
#[test]
fn cumsum_matches_numpy() {
    run_axis_unary("cumsum.json", "cumsum", ferray_ufunc::cumsum);
}

/// Covers: `ferray_ufunc::cumprod` (re-export of
/// `ferray_ufunc::ops::arithmetic::cumprod`).
#[test]
fn cumprod_matches_numpy() {
    run_axis_unary("cumprod.json", "cumprod", ferray_ufunc::cumprod);
}

/// Covers: `ferray_ufunc::cumulative_sum` (re-export of
/// `ferray_ufunc::ops::arithmetic::cumulative_sum`).
#[test]
fn cumulative_sum_matches_numpy() {
    run_axis_unary(
        "cumulative_sum.json",
        "cumulative_sum",
        ferray_ufunc::cumulative_sum,
    );
}

/// Covers: `ferray_ufunc::cumulative_prod` (re-export of
/// `ferray_ufunc::ops::arithmetic::cumulative_prod`).
#[test]
fn cumulative_prod_matches_numpy() {
    run_axis_unary(
        "cumulative_prod.json",
        "cumulative_prod",
        ferray_ufunc::cumulative_prod,
    );
}

/// Covers: `ferray_ufunc::nancumsum` (re-export of
/// `ferray_ufunc::ops::arithmetic::nancumsum`).
#[test]
fn nancumsum_matches_numpy() {
    run_axis_unary("nancumsum.json", "nancumsum", ferray_ufunc::nancumsum);
}

/// Covers: `ferray_ufunc::nancumprod` (re-export of
/// `ferray_ufunc::ops::arithmetic::nancumprod`).
#[test]
fn nancumprod_matches_numpy() {
    run_axis_unary("nancumprod.json", "nancumprod", ferray_ufunc::nancumprod);
}

/// Covers: `ferray_ufunc::nan_add_reduce` and
/// `ferray_ufunc::ops::arithmetic::nan_add_reduce`.
#[test]
fn nan_add_reduce_matches_numpy() {
    run_nan_axis_reduce(
        "nan_add_reduce.json",
        "nan_add_reduce",
        ferray_ufunc::nan_add_reduce,
    );
    run_nan_axis_reduce(
        "nan_add_reduce.json",
        "ops::arithmetic::nan_add_reduce",
        ferray_ufunc::ops::arithmetic::nan_add_reduce,
    );
}

/// Covers: `ferray_ufunc::nan_add_reduce_axes` and
/// `ferray_ufunc::ops::arithmetic::nan_add_reduce_axes`.
#[test]
fn nan_add_reduce_axes_matches_numpy() {
    run_nan_axes_reduce(
        "nan_add_reduce_axes.json",
        "nan_add_reduce_axes",
        ferray_ufunc::nan_add_reduce_axes,
    );
    run_nan_axes_reduce(
        "nan_add_reduce_axes.json",
        "ops::arithmetic::nan_add_reduce_axes",
        ferray_ufunc::ops::arithmetic::nan_add_reduce_axes,
    );
}

/// Covers: `ferray_ufunc::nan_add_reduce_all` and
/// `ferray_ufunc::ops::arithmetic::nan_add_reduce_all`.
#[test]
fn nan_add_reduce_all_matches_numpy() {
    run_nan_reduce_all(
        "nan_add_reduce_all.json",
        "nan_add_reduce_all",
        ferray_ufunc::nan_add_reduce_all,
    );
    run_nan_reduce_all(
        "nan_add_reduce_all.json",
        "ops::arithmetic::nan_add_reduce_all",
        ferray_ufunc::ops::arithmetic::nan_add_reduce_all,
    );
}

/// Covers: `ferray_ufunc::nan_multiply_reduce` and
/// `ferray_ufunc::ops::arithmetic::nan_multiply_reduce`.
#[test]
fn nan_multiply_reduce_matches_numpy() {
    run_nan_axis_reduce(
        "nan_multiply_reduce.json",
        "nan_multiply_reduce",
        ferray_ufunc::nan_multiply_reduce,
    );
    run_nan_axis_reduce(
        "nan_multiply_reduce.json",
        "ops::arithmetic::nan_multiply_reduce",
        ferray_ufunc::ops::arithmetic::nan_multiply_reduce,
    );
}

/// Covers: `ferray_ufunc::nan_multiply_reduce_axes` and
/// `ferray_ufunc::ops::arithmetic::nan_multiply_reduce_axes`.
#[test]
fn nan_multiply_reduce_axes_matches_numpy() {
    run_nan_axes_reduce(
        "nan_multiply_reduce_axes.json",
        "nan_multiply_reduce_axes",
        ferray_ufunc::nan_multiply_reduce_axes,
    );
    run_nan_axes_reduce(
        "nan_multiply_reduce_axes.json",
        "ops::arithmetic::nan_multiply_reduce_axes",
        ferray_ufunc::ops::arithmetic::nan_multiply_reduce_axes,
    );
}

/// Covers: `ferray_ufunc::nan_multiply_reduce_all` and
/// `ferray_ufunc::ops::arithmetic::nan_multiply_reduce_all`.
#[test]
fn nan_multiply_reduce_all_matches_numpy() {
    run_nan_reduce_all(
        "nan_multiply_reduce_all.json",
        "nan_multiply_reduce_all",
        ferray_ufunc::nan_multiply_reduce_all,
    );
    run_nan_reduce_all(
        "nan_multiply_reduce_all.json",
        "ops::arithmetic::nan_multiply_reduce_all",
        ferray_ufunc::ops::arithmetic::nan_multiply_reduce_all,
    );
}

/// Covers: `ferray_ufunc::diff` (re-export of
/// `ferray_ufunc::ops::arithmetic::diff`).
#[test]
fn diff_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("diff.json"));
    for case in &suite.test_cases {
        let arr = make_f64_ix1(&case.inputs["x"]);
        let n = case.inputs["n"].as_u64().expect("n integer") as usize;
        let result =
            ferray_ufunc::diff(&arr, n).unwrap_or_else(|e| panic!("diff {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::ediff1d` (re-export of
/// `ferray_ufunc::ops::arithmetic::ediff1d`).
#[test]
fn ediff1d_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("ediff1d.json"));
    for case in &suite.test_cases {
        let arr = make_f64_ix1(&case.inputs["x"]);
        let to_end = optional_f64_vec(case.inputs.get("to_end"));
        let to_begin = optional_f64_vec(case.inputs.get("to_begin"));
        let result = ferray_ufunc::ediff1d(&arr, to_end.as_deref(), to_begin.as_deref())
            .unwrap_or_else(|e| panic!("ediff1d {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::gradient` (re-export of
/// `ferray_ufunc::ops::arithmetic::gradient`).
#[test]
fn gradient_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("gradient.json"));
    for case in &suite.test_cases {
        let arr = make_f64_ix1(&case.inputs["x"]);
        let spacing = case.inputs.get("spacing").map(parse_f64_value);
        let result = ferray_ufunc::gradient(&arr, spacing)
            .unwrap_or_else(|e| panic!("gradient {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::cross` (re-export of
/// `ferray_ufunc::ops::arithmetic::cross`).
#[test]
fn cross_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("cross.json"));
    for case in &suite.test_cases {
        let a = make_f64_ix1(&case.inputs["a"]);
        let b = make_f64_ix1(&case.inputs["b"]);
        let result =
            ferray_ufunc::cross(&a, &b).unwrap_or_else(|e| panic!("cross {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps,
            &case.name,
        );
    }
}

/// Covers: `ferray_ufunc::trapezoid` (re-export of
/// `ferray_ufunc::ops::arithmetic::trapezoid`).
#[test]
fn trapezoid_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("trapezoid.json"));
    for case in &suite.test_cases {
        let y = make_f64_ix1(&case.inputs["y"]);
        let x = case.inputs.get("x").map(make_f64_ix1);
        let dx = case.inputs.get("dx").map(parse_f64_value);
        let result = ferray_ufunc::trapezoid(&y, x.as_ref(), dx)
            .unwrap_or_else(|e| panic!("trapezoid {}: {e}", case.name));
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(&[result], &expected, case.tolerance_ulps, &case.name);
    }
}
