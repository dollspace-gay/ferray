//! Conformance tests for ferray-ufunc floating-point error policy state.

use ferray_test_oracle::load_fixture;
use ferray_ufunc::errstate::{
    FpErrorClass, FpErrorState, check_fp_errors, geterr, record_fp_event, seterr, take_fp_events,
    with_errstate,
};

type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::Ix1>;
type ArrI32 = ferray_core::Array<i32, ferray_core::dimension::Ix1>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn errstate_case(name: &str) -> ferray_test_oracle::TestCase {
    let suite = load_fixture(&ufunc_fixture("errstate.json"));
    suite
        .test_cases
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing errstate fixture case {name}"))
}

fn class_for_numpy_key(key: &str) -> FpErrorClass {
    match key {
        "divide" => FpErrorClass::DivideByZero,
        "over" => FpErrorClass::Overflow,
        "under" => FpErrorClass::Underflow,
        "invalid" => FpErrorClass::Invalid,
        other => panic!("unexpected NumPy errstate key {other}"),
    }
}

fn state_for_numpy_value(value: &str) -> FpErrorState {
    match value {
        "ignore" => FpErrorState::Ignore,
        "warn" => FpErrorState::Warn,
        "raise" => FpErrorState::Raise,
        other => panic!("unexpected NumPy errstate value {other}"),
    }
}

fn apply_expected_policy(policy: &ferray_test_oracle::serde_json::Value) {
    for (key, value) in policy.as_object().expect("policy object") {
        let state = state_for_numpy_value(value.as_str().expect("policy string"));
        seterr(class_for_numpy_key(key), state);
    }
}

fn snapshot_policy() -> std::collections::BTreeMap<&'static str, FpErrorState> {
    [
        ("divide", geterr(FpErrorClass::DivideByZero)),
        ("over", geterr(FpErrorClass::Overflow)),
        ("under", geterr(FpErrorClass::Underflow)),
        ("invalid", geterr(FpErrorClass::Invalid)),
    ]
    .into_iter()
    .collect()
}

fn expected_policy(
    policy: &ferray_test_oracle::serde_json::Value,
) -> std::collections::BTreeMap<&'static str, FpErrorState> {
    let mut out = std::collections::BTreeMap::new();
    for key in ["divide", "over", "under", "invalid"] {
        let value = policy[key].as_str().expect("policy value");
        out.insert(key, state_for_numpy_value(value));
    }
    out
}

fn reset_errstate() {
    let case = errstate_case("default");
    apply_expected_policy(&case.expected);
    let _ = take_fp_events();
    let _ = check_fp_errors();
}

/// Covers: `ferray_ufunc::errstate::FpErrorClass`,
/// `ferray_ufunc::errstate::FpErrorState`, and
/// `ferray_ufunc::errstate::geterr`.
#[test]
fn default_errstate_matches_numpy_geterr_fixture() {
    reset_errstate();
    let case = errstate_case("default");
    assert_eq!(snapshot_policy(), expected_policy(&case.expected));
}

/// Covers: `ferray_ufunc::errstate::seterr` and
/// `ferray_ufunc::errstate::geterr`.
#[test]
fn seterr_returns_previous_and_sets_current_policy() {
    reset_errstate();
    let case = errstate_case("default");
    for key in ["divide", "over", "under", "invalid"] {
        let class = class_for_numpy_key(key);
        let expected_previous =
            state_for_numpy_value(case.expected[key].as_str().expect("policy value"));
        let previous = seterr(class, FpErrorState::Raise);
        assert_eq!(previous, expected_previous, "{key} previous policy");
        assert_eq!(geterr(class), FpErrorState::Raise, "{key} current policy");
        seterr(class, previous);
    }
    assert_eq!(snapshot_policy(), expected_policy(&case.expected));
}

/// Covers: `ferray_ufunc::errstate::ErrstateGuard` through the scoped
/// `ferray_ufunc::errstate::with_errstate` API.
#[test]
fn with_errstate_restores_like_numpy_errstate_fixture() {
    reset_errstate();
    let case = errstate_case("scoped_over_raise_invalid_ignore");
    let inner = with_errstate(
        &[
            (FpErrorClass::Overflow, FpErrorState::Raise),
            (FpErrorClass::Invalid, FpErrorState::Ignore),
        ],
        snapshot_policy,
    );
    assert_eq!(inner, expected_policy(&case.expected["inner"]));
    assert_eq!(snapshot_policy(), expected_policy(&case.expected["outer"]));
}

/// Covers: `ferray_ufunc::errstate::record_fp_event`,
/// `ferray_ufunc::errstate::take_fp_events`, and
/// `ferray_ufunc::errstate::check_fp_errors`.
#[test]
fn recorded_fp_events_follow_warn_ignore_raise_policy_meanings() {
    reset_errstate();
    let case = errstate_case("policy_meanings");
    assert_eq!(case.expected["warn"], "record");
    assert_eq!(case.expected["ignore"], "drop");
    assert_eq!(case.expected["raise"], "defer_error");

    seterr(FpErrorClass::Overflow, FpErrorState::Warn);
    record_fp_event(FpErrorClass::Overflow);
    record_fp_event(FpErrorClass::Overflow);
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::Overflow, FpErrorClass::Overflow]
    );
    assert!(take_fp_events().is_empty());

    seterr(FpErrorClass::Underflow, FpErrorState::Ignore);
    record_fp_event(FpErrorClass::Underflow);
    assert!(take_fp_events().is_empty());

    seterr(FpErrorClass::Invalid, FpErrorState::Raise);
    record_fp_event(FpErrorClass::Invalid);
    let err = check_fp_errors().unwrap_err();
    assert!(err.to_string().contains("Invalid"));
    assert!(check_fp_errors().is_ok());
}

/// Covers kernel integration for `ferray_ufunc::divide` and
/// `ferray_ufunc::divide_broadcast`: divide-by-zero and invalid events are
/// recorded by the ufunc kernels themselves, matching NumPy errstate's
/// operation-driven behavior rather than requiring manual `record_fp_event`.
#[test]
fn divide_kernels_emit_divide_and_invalid_events() {
    reset_errstate();
    let a = ArrF64::from_vec(ferray_core::dimension::Ix1::new([2]), vec![1.0, 0.0]).unwrap();
    let b = ArrF64::from_vec(ferray_core::dimension::Ix1::new([2]), vec![0.0, 0.0]).unwrap();

    let _ = ferray_ufunc::divide(&a, &b).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );

    let scalar_zero = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![0.0]).unwrap();
    let _ = ferray_ufunc::divide_broadcast(&a, &scalar_zero).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );
}

/// Covers integer true-division integration. NumPy's integer
/// `true_divide` promotes to float output but still drives divide/invalid
/// errstate events for `x / 0` and `0 / 0`.
#[test]
fn integer_true_divide_emits_zero_denominator_events() {
    reset_errstate();
    let a = ArrI32::from_vec(ferray_core::dimension::Ix1::new([2]), vec![2, 0]).unwrap();
    let b = ArrI32::from_vec(ferray_core::dimension::Ix1::new([2]), vec![0, 0]).unwrap();

    let _ = ferray_ufunc::divide(&a, &b).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );
}

/// Covers overflow and underflow event integration for arithmetic kernels.
#[test]
fn multiply_kernel_emits_overflow_and_underflow_events() {
    reset_errstate();
    seterr(FpErrorClass::Underflow, FpErrorState::Warn);

    let huge = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![f64::MAX]).unwrap();
    let two = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![2.0]).unwrap();
    let _ = ferray_ufunc::multiply(&huge, &two).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Overflow]);

    let tiny = ArrF64::from_vec(
        ferray_core::dimension::Ix1::new([1]),
        vec![f64::MIN_POSITIVE],
    )
    .unwrap();
    let half = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![0.5]).unwrap();
    let _ = ferray_ufunc::multiply(&tiny, &half).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Underflow]);
}

/// Covers divide-by-zero and invalid domain integration for log kernels.
#[test]
fn log_kernels_emit_divide_and_invalid_events() {
    reset_errstate();
    let input =
        ArrF64::from_vec(ferray_core::dimension::Ix1::new([3]), vec![0.0, -1.0, 1.0]).unwrap();

    let _ = ferray_ufunc::log(&input).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );

    let log1p_input = ArrF64::from_vec(
        ferray_core::dimension::Ix1::new([3]),
        vec![-1.0, -2.0, -0.5],
    )
    .unwrap();
    let _ = ferray_ufunc::log1p(&log1p_input).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );
}

/// Covers overflow and underflow event integration for exp kernels.
#[test]
fn exp_kernel_emits_overflow_and_underflow_events() {
    reset_errstate();
    seterr(FpErrorClass::Underflow, FpErrorState::Warn);
    let input =
        ArrF64::from_vec(ferray_core::dimension::Ix1::new([2]), vec![1000.0, -1000.0]).unwrap();

    let _ = ferray_ufunc::exp(&input).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::Overflow, FpErrorClass::Underflow]
    );
}

/// Covers invalid-domain integration for trigonometric kernels.
#[test]
fn trig_kernels_emit_invalid_and_divide_domain_events() {
    reset_errstate();

    let inf = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![f64::INFINITY]).unwrap();
    let _ = ferray_ufunc::sin(&inf).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Invalid]);

    let outside_unit = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![2.0]).unwrap();
    let _ = ferray_ufunc::arcsin(&outside_unit).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Invalid]);

    let below_one = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![0.5]).unwrap();
    let _ = ferray_ufunc::arccosh(&below_one).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Invalid]);

    let atanh_input =
        ArrF64::from_vec(ferray_core::dimension::Ix1::new([2]), vec![1.0, 2.0]).unwrap();
    let _ = ferray_ufunc::arctanh(&atanh_input).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![FpErrorClass::DivideByZero, FpErrorClass::Invalid]
    );
}

/// Covers overflow/underflow integration for hyperbolic and scale kernels.
#[test]
fn hyperbolic_and_scale_kernels_emit_range_events() {
    reset_errstate();
    seterr(FpErrorClass::Underflow, FpErrorState::Warn);

    let huge = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![1000.0]).unwrap();
    let _ = ferray_ufunc::sinh(&huge).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Overflow]);

    let scale_huge =
        ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![f64::MAX]).unwrap();
    let _ = ferray_ufunc::degrees(&scale_huge).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Overflow]);

    let tiny = ArrF64::from_vec(
        ferray_core::dimension::Ix1::new([1]),
        vec![f64::MIN_POSITIVE],
    )
    .unwrap();
    let exponents = ferray_core::Array::<i32, ferray_core::dimension::Ix1>::from_vec(
        ferray_core::dimension::Ix1::new([1]),
        vec![-1075],
    )
    .unwrap();
    let _ = ferray_ufunc::ldexp(&tiny, &exponents).unwrap();
    assert_eq!(take_fp_events(), vec![FpErrorClass::Underflow]);
}

/// Covers `float_power` integration for divide-by-zero, overflow, and invalid
/// events.
#[test]
fn float_power_kernel_emits_domain_and_range_events() {
    reset_errstate();
    let bases = ArrF64::from_vec(
        ferray_core::dimension::Ix1::new([3]),
        vec![0.0, f64::MAX, -1.0],
    )
    .unwrap();
    let exponents =
        ArrF64::from_vec(ferray_core::dimension::Ix1::new([3]), vec![-1.0, 2.0, 0.5]).unwrap();

    let _ = ferray_ufunc::float_power(&bases, &exponents).unwrap();
    assert_eq!(
        take_fp_events(),
        vec![
            FpErrorClass::DivideByZero,
            FpErrorClass::Overflow,
            FpErrorClass::Invalid,
        ]
    );
}

/// Covers the `Raise` policy being fed by actual ufunc kernels.
#[test]
fn ufunc_events_feed_raise_policy_queue() {
    reset_errstate();
    seterr(FpErrorClass::DivideByZero, FpErrorState::Raise);
    let a = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![1.0]).unwrap();
    let b = ArrF64::from_vec(ferray_core::dimension::Ix1::new([1]), vec![0.0]).unwrap();

    let _ = ferray_ufunc::divide(&a, &b).unwrap();
    let err = check_fp_errors().unwrap_err();
    assert!(err.to_string().contains("DivideByZero"));
    assert!(check_fp_errors().is_ok());
}
