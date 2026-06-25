#![cfg(feature = "f16")]

//! Feature-gated conformance tests for ferray-ufunc f16 entry points against
//! NumPy float16 fixture outputs.

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_test_oracle::{load_fixture, parse_bool_data, parse_f32_data, parse_shape};

type ArrF16 = Array<half::f16, IxDyn>;
type BoolArr = Array<bool, IxDyn>;
type UnaryF16Fn = fn(&ArrF16) -> ferray_core::error::FerrayResult<ArrF16>;
type UnaryF16ToBoolFn = fn(&ArrF16) -> ferray_core::error::FerrayResult<BoolArr>;
type BinaryF16Fn = fn(&ArrF16, &ArrF16) -> ferray_core::error::FerrayResult<ArrF16>;
type ClipF16Fn = fn(&ArrF16, half::f16, half::f16) -> ferray_core::error::FerrayResult<ArrF16>;
type NanToNumF16Fn = fn(
    &ArrF16,
    Option<half::f16>,
    Option<half::f16>,
    Option<half::f16>,
) -> ferray_core::error::FerrayResult<ArrF16>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn make_f16_array(value: &ferray_test_oracle::serde_json::Value) -> ArrF16 {
    let data = parse_f32_data(&value["data"])
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let shape = parse_shape(&value["shape"]);
    ArrF16::from_vec(IxDyn::new(&shape), data).unwrap()
}

fn half_ulp_key(value: half::f16) -> i32 {
    let bits = i32::from(value.to_bits());
    if bits & 0x8000 != 0 {
        0x8000 - (bits & 0x7fff)
    } else {
        0x8000 + bits
    }
}

fn assert_half_ulp(actual: half::f16, expected: half::f16, tolerance_ulps: u64, context: &str) {
    if actual == expected || (actual.is_nan() && expected.is_nan()) {
        return;
    }
    let diff = (half_ulp_key(actual) - half_ulp_key(expected)).abs() as u64;
    assert!(
        diff <= tolerance_ulps,
        "{context}: f16 ULP diff {diff} > {tolerance_ulps}; actual={} ({:#06x}) expected={} ({:#06x})",
        actual.to_f32(),
        actual.to_bits(),
        expected.to_f32(),
        expected.to_bits()
    );
}

fn assert_f16_array_matches(
    name: &str,
    case_name: &str,
    result: &ArrF16,
    expected: &ferray_test_oracle::serde_json::Value,
    tolerance_ulps: u64,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    let expected_data: Vec<half::f16> = parse_f32_data(&expected["data"])
        .into_iter()
        .map(half::f16::from_f32)
        .collect();
    let actual = result.as_slice().expect("contiguous");
    assert_eq!(
        actual.len(),
        expected_data.len(),
        "{name} case {case_name} length mismatch"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected_data.iter()).enumerate() {
        assert_half_ulp(
            a,
            e,
            tolerance_ulps,
            &format!("{name} case {case_name} element {i}"),
        );
    }
}

fn assert_bool_array_matches(
    name: &str,
    case_name: &str,
    result: &BoolArr,
    expected: &ferray_test_oracle::serde_json::Value,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    let expected_data = parse_bool_data(&expected["data"]);
    let actual = result.as_slice().expect("contiguous");
    assert_eq!(
        actual,
        expected_data.as_slice(),
        "{name} case {case_name} bool data mismatch"
    );
}

fn scalar_f16(value: &ferray_test_oracle::serde_json::Value) -> half::f16 {
    let data = parse_f32_data(&value["data"]);
    assert_eq!(data.len(), 1, "expected scalar f16 fixture input");
    half::f16::from_f32(data[0])
}

fn optional_f16_input(
    inputs: &ferray_test_oracle::serde_json::Value,
    key: &str,
) -> Option<half::f16> {
    inputs.get(key).map(scalar_f16)
}

fn run_unary_f16(fixture: &str, name: &str, f: UnaryF16Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f16_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f16_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_unary_f16_to_bool(fixture: &str, name: &str, f: UnaryF16ToBoolFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f16_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_bool_array_matches(name, &case.name, &result, &case.expected);
    }
}

fn run_binary_f16(fixture: &str, name: &str, f: BinaryF16Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let a = make_f16_array(&case.inputs["a"]);
        let b = make_f16_array(&case.inputs["b"]);
        let result = f(&a, &b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f16_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_clip_f16(fixture: &str, name: &str, f: ClipF16Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f16_array(&case.inputs["x"]);
        let a_min = scalar_f16(&case.inputs["a_min"]);
        let a_max = scalar_f16(&case.inputs["a_max"]);
        let result = f(&arr, a_min, a_max).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f16_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

fn run_nan_to_num_f16(fixture: &str, name: &str, f: NanToNumF16Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f16_array(&case.inputs["x"]);
        let nan = optional_f16_input(&case.inputs, "nan");
        let posinf = optional_f16_input(&case.inputs, "posinf");
        let neginf = optional_f16_input(&case.inputs, "neginf");
        let result =
            f(&arr, nan, posinf, neginf).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_f16_array_matches(
            name,
            &case.name,
            &result,
            &case.expected,
            case.tolerance_ulps,
        );
    }
}

macro_rules! unary_f16_match {
    ($test_name:ident, $fixture:expr, $func:path) => {
        #[test]
        fn $test_name() {
            run_unary_f16($fixture, stringify!($test_name), $func);
        }
    };
}

macro_rules! unary_f16_bool_match {
    ($test_name:ident, $fixture:expr, $func:path) => {
        #[test]
        fn $test_name() {
            run_unary_f16_to_bool($fixture, stringify!($test_name), $func);
        }
    };
}

macro_rules! binary_f16_match {
    ($test_name:ident, $fixture:expr, $func:path) => {
        #[test]
        fn $test_name() {
            run_binary_f16($fixture, stringify!($test_name), $func);
        }
    };
}

unary_f16_match!(
    absolute_f16_matches_numpy,
    "absolute_f16.json",
    ferray_ufunc::absolute_f16
);
unary_f16_match!(
    negative_f16_matches_numpy,
    "negative_f16.json",
    ferray_ufunc::negative_f16
);
unary_f16_match!(
    sqrt_f16_matches_numpy,
    "sqrt_f16.json",
    ferray_ufunc::sqrt_f16
);
unary_f16_match!(
    cbrt_f16_matches_numpy,
    "cbrt_f16.json",
    ferray_ufunc::cbrt_f16
);
unary_f16_match!(
    square_f16_matches_numpy,
    "square_f16.json",
    ferray_ufunc::square_f16
);
unary_f16_match!(
    reciprocal_f16_matches_numpy,
    "reciprocal_f16.json",
    ferray_ufunc::reciprocal_f16
);
unary_f16_match!(
    sign_f16_matches_numpy,
    "sign_f16.json",
    ferray_ufunc::sign_f16
);
unary_f16_match!(exp_f16_matches_numpy, "exp_f16.json", ferray_ufunc::exp_f16);
unary_f16_match!(
    exp2_f16_matches_numpy,
    "exp2_f16.json",
    ferray_ufunc::exp2_f16
);
unary_f16_match!(
    expm1_f16_matches_numpy,
    "expm1_f16.json",
    ferray_ufunc::expm1_f16
);
unary_f16_match!(log_f16_matches_numpy, "log_f16.json", ferray_ufunc::log_f16);
unary_f16_match!(
    log2_f16_matches_numpy,
    "log2_f16.json",
    ferray_ufunc::log2_f16
);
unary_f16_match!(
    log10_f16_matches_numpy,
    "log10_f16.json",
    ferray_ufunc::log10_f16
);
unary_f16_match!(
    log1p_f16_matches_numpy,
    "log1p_f16.json",
    ferray_ufunc::log1p_f16
);
unary_f16_match!(
    floor_f16_matches_numpy,
    "floor_f16.json",
    ferray_ufunc::floor_f16
);
unary_f16_match!(
    ceil_f16_matches_numpy,
    "ceil_f16.json",
    ferray_ufunc::ceil_f16
);
unary_f16_match!(
    trunc_f16_matches_numpy,
    "trunc_f16.json",
    ferray_ufunc::trunc_f16
);
unary_f16_match!(
    round_f16_matches_numpy,
    "round_f16.json",
    ferray_ufunc::round_f16
);
unary_f16_match!(sin_f16_matches_numpy, "sin_f16.json", ferray_ufunc::sin_f16);
unary_f16_match!(cos_f16_matches_numpy, "cos_f16.json", ferray_ufunc::cos_f16);
unary_f16_match!(tan_f16_matches_numpy, "tan_f16.json", ferray_ufunc::tan_f16);
unary_f16_match!(
    arcsin_f16_matches_numpy,
    "arcsin_f16.json",
    ferray_ufunc::arcsin_f16
);
unary_f16_match!(
    arccos_f16_matches_numpy,
    "arccos_f16.json",
    ferray_ufunc::arccos_f16
);
unary_f16_match!(
    arctan_f16_matches_numpy,
    "arctan_f16.json",
    ferray_ufunc::arctan_f16
);
unary_f16_match!(
    sinh_f16_matches_numpy,
    "sinh_f16.json",
    ferray_ufunc::sinh_f16
);
unary_f16_match!(
    cosh_f16_matches_numpy,
    "cosh_f16.json",
    ferray_ufunc::cosh_f16
);
unary_f16_match!(
    tanh_f16_matches_numpy,
    "tanh_f16.json",
    ferray_ufunc::tanh_f16
);
unary_f16_match!(
    arcsinh_f16_matches_numpy,
    "arcsinh_f16.json",
    ferray_ufunc::arcsinh_f16
);
unary_f16_match!(
    arccosh_f16_matches_numpy,
    "arccosh_f16.json",
    ferray_ufunc::arccosh_f16
);
unary_f16_match!(
    arctanh_f16_matches_numpy,
    "arctanh_f16.json",
    ferray_ufunc::arctanh_f16
);
unary_f16_match!(
    degrees_f16_matches_numpy,
    "degrees_f16.json",
    ferray_ufunc::degrees_f16
);
unary_f16_match!(
    radians_f16_matches_numpy,
    "radians_f16.json",
    ferray_ufunc::radians_f16
);
unary_f16_match!(
    sinc_f16_matches_numpy,
    "sinc_f16.json",
    ferray_ufunc::sinc_f16
);
unary_f16_bool_match!(
    isnan_f16_matches_numpy,
    "isnan_f16.json",
    ferray_ufunc::isnan_f16
);
unary_f16_bool_match!(
    isinf_f16_matches_numpy,
    "isinf_f16.json",
    ferray_ufunc::isinf_f16
);
unary_f16_bool_match!(
    isfinite_f16_matches_numpy,
    "isfinite_f16.json",
    ferray_ufunc::isfinite_f16
);

binary_f16_match!(add_f16_matches_numpy, "add_f16.json", ferray_ufunc::add_f16);
binary_f16_match!(
    subtract_f16_matches_numpy,
    "subtract_f16.json",
    ferray_ufunc::subtract_f16
);
binary_f16_match!(
    multiply_f16_matches_numpy,
    "multiply_f16.json",
    ferray_ufunc::multiply_f16
);
binary_f16_match!(
    divide_f16_matches_numpy,
    "divide_f16.json",
    ferray_ufunc::divide_f16
);
binary_f16_match!(
    power_f16_matches_numpy,
    "power_f16.json",
    ferray_ufunc::power_f16
);
binary_f16_match!(
    floor_divide_f16_matches_numpy,
    "floor_divide_f16.json",
    ferray_ufunc::floor_divide_f16
);
binary_f16_match!(
    remainder_f16_matches_numpy,
    "remainder_f16.json",
    ferray_ufunc::remainder_f16
);
binary_f16_match!(
    arctan2_f16_matches_numpy,
    "arctan2_f16.json",
    ferray_ufunc::arctan2_f16
);
binary_f16_match!(
    hypot_f16_matches_numpy,
    "hypot_f16.json",
    ferray_ufunc::hypot_f16
);
binary_f16_match!(
    maximum_f16_matches_numpy,
    "maximum_f16.json",
    ferray_ufunc::maximum_f16
);
binary_f16_match!(
    minimum_f16_matches_numpy,
    "minimum_f16.json",
    ferray_ufunc::minimum_f16
);

#[test]
fn clip_f16_matches_numpy() {
    run_clip_f16("clip_f16.json", "clip_f16", ferray_ufunc::clip_f16);
}

#[test]
fn nan_to_num_f16_matches_numpy() {
    run_nan_to_num_f16(
        "nan_to_num_f16.json",
        "nan_to_num_f16",
        ferray_ufunc::nan_to_num_f16,
    );
}

#[test]
fn canonical_floatintrinsic_f16_paths_match_numpy() {
    run_clip_f16(
        "clip_f16.json",
        "ferray_ufunc::ops::floatintrinsic::clip_f16",
        ferray_ufunc::ops::floatintrinsic::clip_f16,
    );
    run_unary_f16_to_bool(
        "isfinite_f16.json",
        "ferray_ufunc::ops::floatintrinsic::isfinite_f16",
        ferray_ufunc::ops::floatintrinsic::isfinite_f16,
    );
    run_unary_f16_to_bool(
        "isinf_f16.json",
        "ferray_ufunc::ops::floatintrinsic::isinf_f16",
        ferray_ufunc::ops::floatintrinsic::isinf_f16,
    );
    run_unary_f16_to_bool(
        "isnan_f16.json",
        "ferray_ufunc::ops::floatintrinsic::isnan_f16",
        ferray_ufunc::ops::floatintrinsic::isnan_f16,
    );
    run_binary_f16(
        "maximum_f16.json",
        "ferray_ufunc::ops::floatintrinsic::maximum_f16",
        ferray_ufunc::ops::floatintrinsic::maximum_f16,
    );
    run_binary_f16(
        "minimum_f16.json",
        "ferray_ufunc::ops::floatintrinsic::minimum_f16",
        ferray_ufunc::ops::floatintrinsic::minimum_f16,
    );
    run_nan_to_num_f16(
        "nan_to_num_f16.json",
        "ferray_ufunc::ops::floatintrinsic::nan_to_num_f16",
        ferray_ufunc::ops::floatintrinsic::nan_to_num_f16,
    );
}
