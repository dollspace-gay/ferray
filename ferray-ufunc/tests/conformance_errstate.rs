//! Conformance tests for ferray-ufunc floating-point error policy state.

use ferray_test_oracle::load_fixture;
use ferray_ufunc::errstate::{
    FpErrorClass, FpErrorState, check_fp_errors, geterr, record_fp_event, seterr, take_fp_events,
    with_errstate,
};

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
