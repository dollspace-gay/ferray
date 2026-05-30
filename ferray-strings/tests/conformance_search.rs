//! Conformance tests for ferray-strings: search, regex_ops, classify.
//!
//! Tolerance is fixture-strict (always 0 — string ops are bit-exact).
//!
//! Each test names the user-facing crate-root re-export path AND the
//! inner canonical module path in its doc comment so the
//! surface-coverage gate's text-match picks both up.

#![allow(clippy::cast_possible_truncation)]

use ferray_strings::{self, StringArray1};
use ferray_test_oracle::{
    fixtures_dir, load_fixture, parse_bool_data, parse_string_data,
};

fn strings_fx(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("strings").join(name)
}

fn make_array(value: &ferray_test_oracle::serde_json::Value) -> StringArray1 {
    let data = parse_string_data(&value["data"]);
    ferray_strings::array(
        &data.iter().map(String::as_str).collect::<Vec<_>>(),
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// search::find / count / startswith / endswith / replace
// search::rfind / index / rindex
// ---------------------------------------------------------------------------

/// Covers: `ferray_strings::find` and `ferray_strings::search::find`.
#[test]
fn find_matches_numpy() {
    let suite = load_fixture(&strings_fx("find.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let sub = case.inputs["substr"].as_str().unwrap();
        let out = ferray_strings::find(&arr, sub).unwrap();
        let expected: Vec<i64> = case.expected["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(out.as_slice().unwrap(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::count` and `ferray_strings::search::count`.
#[test]
fn count_matches_numpy() {
    let suite = load_fixture(&strings_fx("count.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let sub = case.inputs["substr"].as_str().unwrap();
        let out = ferray_strings::count(&arr, sub).unwrap();
        let expected: Vec<i64> = case.expected["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        assert_eq!(out.as_slice().unwrap(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::startswith` and `ferray_strings::search::startswith`.
#[test]
fn startswith_matches_numpy() {
    let suite = load_fixture(&strings_fx("startswith.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let prefix = case.inputs["substr"].as_str().unwrap();
        let out = ferray_strings::startswith(&arr, prefix).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(out.as_slice().unwrap(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::endswith` and `ferray_strings::search::endswith`.
#[test]
fn endswith_matches_numpy() {
    let suite = load_fixture(&strings_fx("endswith.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let suffix = case.inputs["substr"].as_str().unwrap();
        let out = ferray_strings::endswith(&arr, suffix).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(out.as_slice().unwrap(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::replace` and `ferray_strings::search::replace`.
#[test]
fn replace_matches_numpy() {
    let suite = load_fixture(&strings_fx("replace.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let old = case.inputs["old"].as_str().unwrap();
        let new = case.inputs["new"].as_str().unwrap();
        let count = case
            .inputs
            .get("count")
            .and_then(ferray_test_oracle::serde_json::Value::as_u64)
            .map(|v| v as usize);
        let out = ferray_strings::replace(&arr, old, new, count).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        assert_eq!(out.as_slice(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::rfind` and `ferray_strings::search::rfind`.
#[test]
fn rfind_returns_rightmost_index() {
    let arr = ferray_strings::array(&["abcabc", "aaa", "no"]).unwrap();
    let out = ferray_strings::rfind(&arr, "abc").unwrap();
    assert_eq!(out.as_slice().unwrap(), &[3_i64, -1, -1]);
}

/// Covers: `ferray_strings::index` and `ferray_strings::search::index`.
#[test]
fn index_returns_position_or_errors() {
    let arr = ferray_strings::array(&["hello", "help"]).unwrap();
    let out = ferray_strings::index(&arr, "he").unwrap();
    assert_eq!(out.as_slice().unwrap(), &[0_i64, 0]);

    let missing = ferray_strings::array(&["xyz"]).unwrap();
    assert!(ferray_strings::index(&missing, "he").is_err());
}

/// Covers: `ferray_strings::rindex` and `ferray_strings::search::rindex`.
#[test]
fn rindex_returns_rightmost_position_or_errors() {
    let arr = ferray_strings::array(&["abcabc"]).unwrap();
    let out = ferray_strings::rindex(&arr, "abc").unwrap();
    assert_eq!(out.as_slice().unwrap(), &[3_i64]);

    let missing = ferray_strings::array(&["xyz"]).unwrap();
    assert!(ferray_strings::rindex(&missing, "abc").is_err());
}

// ---------------------------------------------------------------------------
// regex_ops::match_ / match_compiled / extract / extract_compiled
// (also covers the crate-root `Regex` re-export from the `regex` crate)
// ---------------------------------------------------------------------------

/// Covers: `ferray_strings::match_` and `ferray_strings::regex_ops::match_`.
#[test]
fn match_matches_numpy() {
    let suite = load_fixture(&strings_fx("match.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let pattern = case.inputs["pattern"].as_str().unwrap();
        let out = ferray_strings::match_(&arr, pattern).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(out.as_slice().unwrap(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::match_compiled` and
/// `ferray_strings::regex_ops::match_compiled`, plus the crate-root
/// re-export `ferray_strings::Regex` of `regex::Regex`.
#[test]
fn match_compiled_reuses_pattern() {
    let arr = ferray_strings::array(&["abc1", "x", "y2"]).unwrap();
    let re = ferray_strings::Regex::new(r"\d").unwrap();
    let out = ferray_strings::match_compiled(&arr, &re).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, false, true]);
}

/// Covers: `ferray_strings::extract` and `ferray_strings::regex_ops::extract`.
#[test]
fn extract_matches_numpy() {
    let suite = load_fixture(&strings_fx("extract.json"));
    for case in &suite.test_cases {
        assert_eq!(case.tolerance_ulps, 0);
        let arr = make_array(&case.inputs["x"]);
        let pattern = case.inputs["pattern"].as_str().unwrap();
        let out = ferray_strings::extract(&arr, pattern).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        assert_eq!(out.as_slice(), expected.as_slice(), "case '{}'", case.name);
    }
}

/// Covers: `ferray_strings::extract_compiled` and
/// `ferray_strings::regex_ops::extract_compiled`.
#[test]
fn extract_compiled_reuses_pattern() {
    let arr = ferray_strings::array(&["abc123", "def", "ghi456"]).unwrap();
    let re = ferray_strings::Regex::new(r"(\d+)").unwrap();
    let out = ferray_strings::extract_compiled(&arr, &re).unwrap();
    assert_eq!(out.as_slice(), &["123", "", "456"]);
}

// ---------------------------------------------------------------------------
// classify::isalpha / isdigit / isspace / isupper / islower / isalnum /
// isnumeric / isdecimal / istitle
// ---------------------------------------------------------------------------

/// Covers: `ferray_strings::isalpha` and `ferray_strings::classify::isalpha`.
#[test]
fn isalpha_identifies_alphabetic_strings() {
    let arr = ferray_strings::array(&["abc", "abc1", "", "café"]).unwrap();
    let out = ferray_strings::isalpha(&arr).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, false, false, true]);
}

/// Covers: `ferray_strings::isdigit` and `ferray_strings::classify::isdigit`.
#[test]
fn isdigit_identifies_digit_strings() {
    let arr = ferray_strings::array(&["123", "12a", "", "0"]).unwrap();
    let out = ferray_strings::isdigit(&arr).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, false, false, true]);
}

/// Covers: `ferray_strings::isspace` and `ferray_strings::classify::isspace`.
#[test]
fn isspace_identifies_whitespace_strings() {
    let arr = ferray_strings::array(&[" ", "\t\n", "", "a "]).unwrap();
    let out = ferray_strings::isspace(&arr).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, true, false, false]);
}

/// Covers: `ferray_strings::isupper` and `ferray_strings::classify::isupper`.
#[test]
fn isupper_identifies_uppercase_strings() {
    let arr = ferray_strings::array(&["ABC", "AbC", "123", ""]).unwrap();
    let out = ferray_strings::isupper(&arr).unwrap();
    // numpy semantics: non-alphabetic only is false; mixed is false.
    let slc = out.as_slice().unwrap();
    assert!(slc[0]);
    assert!(!slc[1]);
}

/// Covers: `ferray_strings::islower` and `ferray_strings::classify::islower`.
#[test]
fn islower_identifies_lowercase_strings() {
    let arr = ferray_strings::array(&["abc", "AbC", "abc1"]).unwrap();
    let out = ferray_strings::islower(&arr).unwrap();
    let slc = out.as_slice().unwrap();
    assert!(slc[0]);
    assert!(!slc[1]);
}

/// Covers: `ferray_strings::isalnum` and `ferray_strings::classify::isalnum`.
#[test]
fn isalnum_identifies_alphanumeric_strings() {
    let arr = ferray_strings::array(&["abc123", "abc!", "", "9"]).unwrap();
    let out = ferray_strings::isalnum(&arr).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, false, false, true]);
}

/// Covers: `ferray_strings::isnumeric` and `ferray_strings::classify::isnumeric`.
#[test]
fn isnumeric_identifies_numeric_strings() {
    // ferray's `isnumeric` accepts ascii digits plus `. + -`; reject
    // alpha and empty strings.
    let arr = ferray_strings::array(&["123", "12a", "", "0"]).unwrap();
    let out = ferray_strings::isnumeric(&arr).unwrap();
    let slc = out.as_slice().unwrap();
    assert!(slc[0]);
    assert!(!slc[1]);
    assert!(!slc[2]);
    assert!(slc[3]);
}

/// Covers: `ferray_strings::isdecimal` and `ferray_strings::classify::isdecimal`.
#[test]
fn isdecimal_identifies_decimal_strings() {
    let arr = ferray_strings::array(&["123", "12a", "", "0"]).unwrap();
    let out = ferray_strings::isdecimal(&arr).unwrap();
    assert_eq!(out.as_slice().unwrap(), &[true, false, false, true]);
}

/// Covers: `ferray_strings::istitle` and `ferray_strings::classify::istitle`.
#[test]
fn istitle_identifies_title_case_strings() {
    let arr = ferray_strings::array(&["Hello World", "hello world", "Hello", ""]).unwrap();
    let out = ferray_strings::istitle(&arr).unwrap();
    let slc = out.as_slice().unwrap();
    assert!(slc[0]);
    assert!(!slc[1]);
}
