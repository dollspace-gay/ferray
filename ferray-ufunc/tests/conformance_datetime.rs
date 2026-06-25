//! Conformance tests for ferray-ufunc datetime64/timedelta64 kernels against
//! NumPy reference outputs.
//!
//! NumPy stores datetime64/timedelta64 arrays as i64 ticks plus a dtype unit.
//! Ferray mirrors that split as typed tick arrays plus explicit `TimeUnit`
//! arguments for promoted kernels, so these tests compare raw ticks and unit
//! tags directly.

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::dtype::{DateTime64, TimeUnit, Timedelta64};
use ferray_core::error::FerrayResult;
use ferray_test_oracle::serde_json::Value;
use ferray_test_oracle::{load_fixture, parse_bool_data, parse_i64_data, parse_shape};

type DateTimeArray = Array<DateTime64, IxDyn>;
type TimedeltaArray = Array<Timedelta64, IxDyn>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn unit_value(value: &Value) -> TimeUnit {
    TimeUnit::from_descr_suffix(value["unit"].as_str().expect("time unit string"))
        .expect("supported time unit")
}

fn make_datetime_array(value: &Value) -> (DateTimeArray, TimeUnit) {
    let data = parse_i64_data(&value["data"])
        .into_iter()
        .map(DateTime64)
        .collect();
    let shape = parse_shape(&value["shape"]);
    let arr = Array::from_vec(IxDyn::new(&shape), data).unwrap();
    (arr, unit_value(value))
}

fn make_timedelta_array(value: &Value) -> (TimedeltaArray, TimeUnit) {
    let data = parse_i64_data(&value["data"])
        .into_iter()
        .map(Timedelta64)
        .collect();
    let shape = parse_shape(&value["shape"]);
    let arr = Array::from_vec(IxDyn::new(&shape), data).unwrap();
    (arr, unit_value(value))
}

fn datetime_ticks(arr: &DateTimeArray) -> Vec<i64> {
    arr.as_slice()
        .expect("contiguous datetime result")
        .iter()
        .map(|v| v.0)
        .collect()
}

fn timedelta_ticks(arr: &TimedeltaArray) -> Vec<i64> {
    arr.as_slice()
        .expect("contiguous timedelta result")
        .iter()
        .map(|v| v.0)
        .collect()
}

fn assert_datetime_matches(name: &str, case_name: &str, result: &DateTimeArray, expected: &Value) {
    let expected_shape = parse_shape(&expected["shape"]);
    let expected_data = parse_i64_data(&expected["data"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        datetime_ticks(result),
        expected_data,
        "{name} case {case_name} data mismatch"
    );
}

fn assert_timedelta_matches(
    name: &str,
    case_name: &str,
    result: &TimedeltaArray,
    expected: &Value,
) {
    let expected_shape = parse_shape(&expected["shape"]);
    let expected_data = parse_i64_data(&expected["data"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        timedelta_ticks(result),
        expected_data,
        "{name} case {case_name} data mismatch"
    );
}

fn assert_bool_matches(name: &str, case_name: &str, result: &Array<bool, IxDyn>, expected: &Value) {
    let expected_shape = parse_shape(&expected["shape"]);
    let expected_data = parse_bool_data(&expected["data"]);
    assert_eq!(
        result.shape(),
        expected_shape.as_slice(),
        "{name} case {case_name} shape mismatch"
    );
    assert_eq!(
        result.as_slice().expect("contiguous bool result"),
        expected_data.as_slice(),
        "{name} case {case_name} data mismatch"
    );
}

fn run_datetime_isnat(
    fixture: &str,
    name: &str,
    f: fn(&DateTimeArray) -> FerrayResult<Array<bool, IxDyn>>,
) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let (arr, _) = make_datetime_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_bool_matches(name, &case.name, &result, &case.expected);
    }
}

fn run_timedelta_isnat(
    fixture: &str,
    name: &str,
    f: fn(&TimedeltaArray) -> FerrayResult<Array<bool, IxDyn>>,
) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let (arr, _) = make_timedelta_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_bool_matches(name, &case.name, &result, &case.expected);
    }
}

fn run_datetime_datetime_to_timedelta(
    fixture: &str,
    name: &str,
    f: fn(&DateTimeArray, &DateTimeArray) -> FerrayResult<TimedeltaArray>,
) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let (a, _) = make_datetime_array(&case.inputs["a"]);
        let (b, _) = make_datetime_array(&case.inputs["b"]);
        let result = f(&a, &b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_timedelta_matches(name, &case.name, &result, &case.expected);
    }
}

fn run_datetime_timedelta_to_datetime(
    fixture: &str,
    name: &str,
    f: fn(&DateTimeArray, &TimedeltaArray) -> FerrayResult<DateTimeArray>,
) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let (a, _) = make_datetime_array(&case.inputs["a"]);
        let (b, _) = make_timedelta_array(&case.inputs["b"]);
        let result = f(&a, &b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_datetime_matches(name, &case.name, &result, &case.expected);
    }
}

fn run_timedelta_timedelta(
    fixture: &str,
    name: &str,
    f: fn(&TimedeltaArray, &TimedeltaArray) -> FerrayResult<TimedeltaArray>,
) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let (a, _) = make_timedelta_array(&case.inputs["a"]);
        let (b, _) = make_timedelta_array(&case.inputs["b"]);
        let result = f(&a, &b).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        assert_timedelta_matches(name, &case.name, &result, &case.expected);
    }
}

/// Covers: `ferray_ufunc::isnat_datetime`
/// (`ferray_ufunc::ops::datetime::isnat_datetime`).
#[test]
fn isnat_datetime_matches_numpy() {
    run_datetime_isnat(
        "isnat_datetime.json",
        "isnat_datetime",
        ferray_ufunc::isnat_datetime,
    );
}

/// Covers: `ferray_ufunc::isnat_timedelta`
/// (`ferray_ufunc::ops::datetime::isnat_timedelta`).
#[test]
fn isnat_timedelta_matches_numpy() {
    run_timedelta_isnat(
        "isnat_timedelta.json",
        "isnat_timedelta",
        ferray_ufunc::isnat_timedelta,
    );
}

/// Covers: `ferray_ufunc::sub_datetime`
/// (`ferray_ufunc::ops::datetime::sub_datetime`).
#[test]
fn sub_datetime_matches_numpy() {
    run_datetime_datetime_to_timedelta(
        "sub_datetime.json",
        "sub_datetime",
        ferray_ufunc::sub_datetime,
    );
}

/// Covers: `ferray_ufunc::add_datetime_timedelta`
/// (`ferray_ufunc::ops::datetime::add_datetime_timedelta`).
#[test]
fn add_datetime_timedelta_matches_numpy() {
    run_datetime_timedelta_to_datetime(
        "add_datetime_timedelta.json",
        "add_datetime_timedelta",
        ferray_ufunc::add_datetime_timedelta,
    );
}

/// Covers: `ferray_ufunc::sub_datetime_timedelta`
/// (`ferray_ufunc::ops::datetime::sub_datetime_timedelta`).
#[test]
fn sub_datetime_timedelta_matches_numpy() {
    run_datetime_timedelta_to_datetime(
        "sub_datetime_timedelta.json",
        "sub_datetime_timedelta",
        ferray_ufunc::sub_datetime_timedelta,
    );
}

/// Covers: `ferray_ufunc::add_timedelta`
/// (`ferray_ufunc::ops::datetime::add_timedelta`).
#[test]
fn add_timedelta_matches_numpy() {
    run_timedelta_timedelta(
        "add_timedelta.json",
        "add_timedelta",
        ferray_ufunc::add_timedelta,
    );
}

/// Covers: `ferray_ufunc::sub_timedelta`
/// (`ferray_ufunc::ops::datetime::sub_timedelta`).
#[test]
fn sub_timedelta_matches_numpy() {
    run_timedelta_timedelta(
        "sub_timedelta.json",
        "sub_timedelta",
        ferray_ufunc::sub_timedelta,
    );
}

/// Covers: `ferray_ufunc::sub_datetime_promoted`
/// (`ferray_ufunc::ops::datetime::sub_datetime_promoted`).
#[test]
fn sub_datetime_promoted_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("sub_datetime_promoted.json"));
    for case in &suite.test_cases {
        let (a, unit_a) = make_datetime_array(&case.inputs["a"]);
        let (b, unit_b) = make_datetime_array(&case.inputs["b"]);
        let (result, unit) = ferray_ufunc::sub_datetime_promoted(&a, unit_a, &b, unit_b)
            .unwrap_or_else(|e| {
                panic!("sub_datetime_promoted {}: {e}", case.name);
            });
        assert_eq!(
            unit,
            unit_value(&case.expected),
            "case {} unit mismatch",
            case.name
        );
        assert_timedelta_matches("sub_datetime_promoted", &case.name, &result, &case.expected);
    }
}

/// Covers: `ferray_ufunc::add_datetime_timedelta_promoted`
/// (`ferray_ufunc::ops::datetime::add_datetime_timedelta_promoted`).
#[test]
fn add_datetime_timedelta_promoted_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("add_datetime_timedelta_promoted.json"));
    for case in &suite.test_cases {
        let (a, unit_a) = make_datetime_array(&case.inputs["a"]);
        let (b, unit_b) = make_timedelta_array(&case.inputs["b"]);
        let (result, unit) = ferray_ufunc::add_datetime_timedelta_promoted(&a, unit_a, &b, unit_b)
            .unwrap_or_else(|e| panic!("add_datetime_timedelta_promoted {}: {e}", case.name));
        assert_eq!(
            unit,
            unit_value(&case.expected),
            "case {} unit mismatch",
            case.name
        );
        assert_datetime_matches(
            "add_datetime_timedelta_promoted",
            &case.name,
            &result,
            &case.expected,
        );
    }
}

/// Covers: `ferray_ufunc::add_timedelta_promoted`
/// (`ferray_ufunc::ops::datetime::add_timedelta_promoted`).
#[test]
fn add_timedelta_promoted_matches_numpy() {
    let suite = load_fixture(&ufunc_fixture("add_timedelta_promoted.json"));
    for case in &suite.test_cases {
        let (a, unit_a) = make_timedelta_array(&case.inputs["a"]);
        let (b, unit_b) = make_timedelta_array(&case.inputs["b"]);
        let (result, unit) = ferray_ufunc::add_timedelta_promoted(&a, unit_a, &b, unit_b)
            .unwrap_or_else(|e| {
                panic!("add_timedelta_promoted {}: {e}", case.name);
            });
        assert_eq!(
            unit,
            unit_value(&case.expected),
            "case {} unit mismatch",
            case.name
        );
        assert_timedelta_matches(
            "add_timedelta_promoted",
            &case.name,
            &result,
            &case.expected,
        );
    }
}
