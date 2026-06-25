//! Conformance tests for ferray-ufunc's bitwise functions against NumPy
//! reference outputs.
//!
//! The tests call crate-root re-exports and name the canonical
//! `ferray_ufunc::ops::bitwise::*` paths so the surface coverage gate can
//! prove both the user-facing and inner items.

use ferray_test_oracle::{
    get_dtype, load_fixture, make_bool_array, make_i64_array, make_u32_array, parse_bool_data,
    parse_i64_data, parse_shape, parse_u32_data,
};

type ArrBool = ferray_core::Array<bool, ferray_core::dimension::IxDyn>;
type ArrI64 = ferray_core::Array<i64, ferray_core::dimension::IxDyn>;
type ArrU32 = ferray_core::Array<u32, ferray_core::dimension::IxDyn>;
type BoolBinaryFn = fn(&ArrBool, &ArrBool) -> ferray_core::error::FerrayResult<ArrBool>;
type BoolUnaryFn = fn(&ArrBool) -> ferray_core::error::FerrayResult<ArrBool>;
type I64BinaryFn = fn(&ArrI64, &ArrI64) -> ferray_core::error::FerrayResult<ArrI64>;
type I64ShiftFn = fn(&ArrI64, &ArrU32) -> ferray_core::error::FerrayResult<ArrI64>;
type I64UnaryFn = fn(&ArrI64) -> ferray_core::error::FerrayResult<ArrI64>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn assert_i64_result(
    name: &str,
    case_name: &str,
    result: &ArrI64,
    expected: &ferray_test_oracle::serde_json::Value,
) {
    let expected_data = parse_i64_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        result.as_slice().expect("contiguous"),
        expected_data.as_slice(),
        "{name} case {case_name} data mismatch"
    );
}

fn assert_bool_result(
    name: &str,
    case_name: &str,
    result: &ArrBool,
    expected: &ferray_test_oracle::serde_json::Value,
) {
    let expected_data = parse_bool_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        result.as_slice().expect("contiguous"),
        expected_data.as_slice(),
        "{name} case {case_name} data mismatch"
    );
}

fn assert_u32_result(
    name: &str,
    case_name: &str,
    result: &ArrU32,
    expected: &ferray_test_oracle::serde_json::Value,
) {
    let expected_data = parse_u32_data(&expected["data"]);
    let expected_shape = parse_shape(&expected["shape"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        result.as_slice().expect("contiguous"),
        expected_data.as_slice(),
        "{name} case {case_name} data mismatch"
    );
}

fn run_bitwise_binary(fixture: &str, name: &str, f_i64: I64BinaryFn, f_bool: BoolBinaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let input_a = &case.inputs["a"];
        let input_b = &case.inputs["b"];
        match (get_dtype(input_a), get_dtype(input_b)) {
            ("int64", "int64") => {
                let arr_a = make_i64_array(input_a);
                let arr_b = make_i64_array(input_b);
                let result =
                    f_i64(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
                assert_i64_result(name, &case.name, &result, &case.expected);
            }
            ("bool", "bool") => {
                let arr_a = make_bool_array(input_a);
                let arr_b = make_bool_array(input_b);
                let result =
                    f_bool(&arr_a, &arr_b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
                assert_bool_result(name, &case.name, &result, &case.expected);
            }
            (a, b) => panic!("{name} case {} unsupported dtypes {a}/{b}", case.name),
        }
    }
}

fn run_bitwise_unary(fixture: &str, name: &str, f_i64: I64UnaryFn, f_bool: BoolUnaryFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        match get_dtype(input) {
            "int64" => {
                let arr = make_i64_array(input);
                let result = f_i64(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
                assert_i64_result(name, &case.name, &result, &case.expected);
            }
            "bool" => {
                let arr = make_bool_array(input);
                let result = f_bool(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
                assert_bool_result(name, &case.name, &result, &case.expected);
            }
            dtype => panic!("{name} case {} unsupported dtype {dtype}", case.name),
        }
    }
}

fn run_shift(fixture: &str, name: &str, f: I64ShiftFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_i64_array(&case.inputs["a"]);
        let shift = make_u32_array(&case.inputs["b"]);
        let result = f(&arr, &shift).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_i64_result(name, &case.name, &result, &case.expected);
    }
}

fn run_bitwise_count() {
    let suite = load_fixture(&ufunc_fixture("bitwise_count.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let result = match get_dtype(input) {
            "int64" => ferray_ufunc::bitwise_count(&make_i64_array(input))
                .unwrap_or_else(|e| panic!("bitwise_count {}: {e}", case.name)),
            "uint32" => ferray_ufunc::bitwise_count(&make_u32_array(input))
                .unwrap_or_else(|e| panic!("bitwise_count {}: {e}", case.name)),
            "bool" => ferray_ufunc::bitwise_count(&make_bool_array(input))
                .unwrap_or_else(|e| panic!("bitwise_count {}: {e}", case.name)),
            dtype => panic!("bitwise_count case {} unsupported dtype {dtype}", case.name),
        };
        assert_u32_result("bitwise_count", &case.name, &result, &case.expected);
    }
}

/// Covers: `ferray_ufunc::bitwise_and` (re-export of
/// `ferray_ufunc::ops::bitwise::bitwise_and`).
///
/// Integer and boolean cases also exercise `ferray_ufunc::BitwiseOps` /
/// `ferray_ufunc::ops::bitwise::BitwiseOps`.
#[test]
fn bitwise_and_matches_numpy() {
    run_bitwise_binary(
        "bitwise_and.json",
        "bitwise_and",
        ferray_ufunc::bitwise_and,
        ferray_ufunc::bitwise_and,
    );
}

/// Covers: `ferray_ufunc::bitwise_or` (re-export of
/// `ferray_ufunc::ops::bitwise::bitwise_or`).
#[test]
fn bitwise_or_matches_numpy() {
    run_bitwise_binary(
        "bitwise_or.json",
        "bitwise_or",
        ferray_ufunc::bitwise_or,
        ferray_ufunc::bitwise_or,
    );
}

/// Covers: `ferray_ufunc::bitwise_xor` (re-export of
/// `ferray_ufunc::ops::bitwise::bitwise_xor`).
#[test]
fn bitwise_xor_matches_numpy() {
    run_bitwise_binary(
        "bitwise_xor.json",
        "bitwise_xor",
        ferray_ufunc::bitwise_xor,
        ferray_ufunc::bitwise_xor,
    );
}

/// Covers: `ferray_ufunc::bitwise_not` (re-export of
/// `ferray_ufunc::ops::bitwise::bitwise_not`).
#[test]
fn bitwise_not_matches_numpy() {
    run_bitwise_unary(
        "bitwise_not.json",
        "bitwise_not",
        ferray_ufunc::bitwise_not,
        ferray_ufunc::bitwise_not,
    );
}

/// Covers: `ferray_ufunc::invert` (re-export of
/// `ferray_ufunc::ops::bitwise::invert`).
#[test]
fn invert_matches_numpy() {
    run_bitwise_unary(
        "invert.json",
        "invert",
        ferray_ufunc::invert,
        ferray_ufunc::invert,
    );
}

/// Covers: `ferray_ufunc::left_shift` (re-export of
/// `ferray_ufunc::ops::bitwise::left_shift`).
///
/// Shift fixtures also exercise `ferray_ufunc::ShiftOps` /
/// `ferray_ufunc::ops::bitwise::ShiftOps`.
#[test]
fn left_shift_matches_numpy() {
    run_shift("left_shift.json", "left_shift", ferray_ufunc::left_shift);
}

/// Covers: `ferray_ufunc::right_shift` (re-export of
/// `ferray_ufunc::ops::bitwise::right_shift`).
#[test]
fn right_shift_matches_numpy() {
    run_shift("right_shift.json", "right_shift", ferray_ufunc::right_shift);
}

// ---------------------------------------------------------------------------
// Operator-style bitwise wrappers
// ---------------------------------------------------------------------------

/// Covers: `ferray_ufunc::array_bitand` and
/// `ferray_ufunc::operator_overloads::array_bitand` with NumPy `bitwise_and`
/// fixtures.
#[test]
fn array_bitand_matches_numpy() {
    run_bitwise_binary(
        "bitwise_and.json",
        "array_bitand",
        ferray_ufunc::array_bitand,
        ferray_ufunc::array_bitand,
    );
}

/// Covers: `ferray_ufunc::array_bitor` and
/// `ferray_ufunc::operator_overloads::array_bitor` with NumPy `bitwise_or`
/// fixtures.
#[test]
fn array_bitor_matches_numpy() {
    run_bitwise_binary(
        "bitwise_or.json",
        "array_bitor",
        ferray_ufunc::array_bitor,
        ferray_ufunc::array_bitor,
    );
}

/// Covers: `ferray_ufunc::array_bitxor` and
/// `ferray_ufunc::operator_overloads::array_bitxor` with NumPy `bitwise_xor`
/// fixtures.
#[test]
fn array_bitxor_matches_numpy() {
    run_bitwise_binary(
        "bitwise_xor.json",
        "array_bitxor",
        ferray_ufunc::array_bitxor,
        ferray_ufunc::array_bitxor,
    );
}

/// Covers: `ferray_ufunc::array_bitnot` and
/// `ferray_ufunc::operator_overloads::array_bitnot` with NumPy `bitwise_not`
/// fixtures.
#[test]
fn array_bitnot_matches_numpy() {
    run_bitwise_unary(
        "bitwise_not.json",
        "array_bitnot",
        ferray_ufunc::array_bitnot,
        ferray_ufunc::array_bitnot,
    );
}

/// Covers: `ferray_ufunc::array_shl` and
/// `ferray_ufunc::operator_overloads::array_shl` with NumPy `left_shift`
/// fixtures.
#[test]
fn array_shl_matches_numpy() {
    run_shift("left_shift.json", "array_shl", ferray_ufunc::array_shl);
}

/// Covers: `ferray_ufunc::array_shr` and
/// `ferray_ufunc::operator_overloads::array_shr` with NumPy `right_shift`
/// fixtures.
#[test]
fn array_shr_matches_numpy() {
    run_shift("right_shift.json", "array_shr", ferray_ufunc::array_shr);
}

/// Covers: `ferray_ufunc::bitwise_count` (re-export of
/// `ferray_ufunc::ops::bitwise::bitwise_count`).
///
/// Signed, unsigned, and boolean fixtures also exercise
/// `ferray_ufunc::BitwiseCount` /
/// `ferray_ufunc::ops::bitwise::BitwiseCount`.
#[test]
fn bitwise_count_matches_numpy() {
    run_bitwise_count();
}
