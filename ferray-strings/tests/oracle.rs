//! Oracle tests: validate ferray-strings against `NumPy` fixture outputs.
//!
//! String operations are exact (no ULP tolerance needed), but we use the
//! oracle framework for consistent fixture loading.

// Oracle tests cross JSON fixture indices through `usize` and tolerate
// the same dtype roundtrips as the rest of the oracle corpus.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use ferray_strings::{self, StringArray1};
use ferray_test_oracle::*;

fn strings_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("strings").join(name)
}

fn make_string_array(value: &ferray_test_oracle::serde_json::Value) -> StringArray1 {
    let data = parse_string_data(&value["data"]);
    ferray_strings::array(
        &data
            .iter()
            .map(std::string::String::as_str)
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

macro_rules! string_case_oracle {
    ($test_name:ident, $file:expr, $func:expr) => {
        #[test]
        fn $test_name() {
            let suite = load_fixture(&strings_path($file));
            for case in &suite.test_cases {
                let input = &case.inputs["x"];
                let arr = make_string_array(input);
                let result = $func(&arr).unwrap();
                let expected = parse_string_data(&case.expected["data"]);
                let result_slice = result.as_slice();
                assert_eq!(
                    result_slice.len(),
                    expected.len(),
                    "case '{}': length mismatch",
                    case.name
                );
                for (i, (a, e)) in result_slice.iter().zip(expected.iter()).enumerate() {
                    assert_eq!(
                        a, e,
                        "case '{}' [element {i}]: actual={a:?}, expected={e:?}",
                        case.name
                    );
                }
            }
        }
    };
}

// Case manipulation
string_case_oracle!(oracle_upper, "upper.json", ferray_strings::upper);
string_case_oracle!(oracle_lower, "lower.json", ferray_strings::lower);
string_case_oracle!(
    oracle_capitalize,
    "capitalize.json",
    ferray_strings::capitalize
);
string_case_oracle!(oracle_title, "title.json", ferray_strings::title);

// Stripping
#[test]
fn oracle_strip() {
    let suite = load_fixture(&strings_path("strip.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let chars = case.inputs.get("chars").and_then(|v| v.as_str());
        let result = ferray_strings::strip(&arr, chars).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_lstrip() {
    let suite = load_fixture(&strings_path("lstrip.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let chars = case.inputs.get("chars").and_then(|v| v.as_str());
        let result = ferray_strings::lstrip(&arr, chars).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_rstrip() {
    let suite = load_fixture(&strings_path("rstrip.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let chars = case.inputs.get("chars").and_then(|v| v.as_str());
        let result = ferray_strings::rstrip(&arr, chars).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

// Search operations
#[test]
fn oracle_find() {
    let suite = load_fixture(&strings_path("find.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let sub = case.inputs["substr"].as_str().unwrap();
        let result = ferray_strings::find(&arr, sub).unwrap();
        let expected: Vec<i64> = case.expected["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let result_slice = result.as_slice().unwrap();
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_count() {
    let suite = load_fixture(&strings_path("count.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let sub = case.inputs["substr"].as_str().unwrap();
        let result = ferray_strings::count(&arr, sub).unwrap();
        let expected: Vec<u64> = case.expected["data"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        let result_slice = result.as_slice().unwrap();
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_startswith() {
    let suite = load_fixture(&strings_path("startswith.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let prefix = case.inputs["substr"].as_str().unwrap();
        let result = ferray_strings::startswith(&arr, prefix).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_endswith() {
    let suite = load_fixture(&strings_path("endswith.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let suffix = case.inputs["substr"].as_str().unwrap();
        let result = ferray_strings::endswith(&arr, suffix).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

#[test]
fn oracle_replace() {
    let suite = load_fixture(&strings_path("replace.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let old = case.inputs["old"].as_str().unwrap();
        let new = case.inputs["new"].as_str().unwrap();
        let count = case
            .inputs
            .get("count")
            .and_then(ferray_test_oracle::serde_json::Value::as_u64)
            .map(|v| v as usize);
        let result = ferray_strings::replace(&arr, old, new, count).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "case '{}' [element {i}]", case.name);
        }
    }
}

// ----- Alignment / padding (#284) ------------------------------------

#[test]
fn oracle_center() {
    let suite = load_fixture(&strings_path("center.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let width = case.inputs["width"].as_u64().unwrap() as usize;
        let result = ferray_strings::center(&arr, width, ' ').unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "center case '{}' [{i}]", case.name);
        }
    }
}

#[test]
fn oracle_ljust() {
    let suite = load_fixture(&strings_path("ljust.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let width = case.inputs["width"].as_u64().unwrap() as usize;
        let result = ferray_strings::ljust(&arr, width).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "ljust case '{}' [{i}]", case.name);
        }
    }
}

#[test]
fn oracle_rjust() {
    let suite = load_fixture(&strings_path("rjust.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let width = case.inputs["width"].as_u64().unwrap() as usize;
        let result = ferray_strings::rjust(&arr, width).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "rjust case '{}' [{i}]", case.name);
        }
    }
}

#[test]
fn oracle_zfill() {
    let suite = load_fixture(&strings_path("zfill.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let width = case.inputs["width"].as_u64().unwrap() as usize;
        let result = ferray_strings::zfill(&arr, width).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "zfill case '{}' [{i}]", case.name);
        }
    }
}

// ----- Concatenation (#284) ------------------------------------------

#[test]
fn oracle_add() {
    let suite = load_fixture(&strings_path("add.json"));
    for case in &suite.test_cases {
        let x = make_string_array(&case.inputs["x"]);
        let y = make_string_array(&case.inputs["y"]);
        let result = ferray_strings::add(&x, &y).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "add case '{}' [{i}]", case.name);
        }
    }
}

#[test]
fn oracle_multiply() {
    let suite = load_fixture(&strings_path("multiply.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let result = ferray_strings::multiply(&arr, n).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "multiply case '{}' [{i}]", case.name);
        }
    }
}

// ----- Regex (#284) ---------------------------------------------------

#[test]
fn oracle_match() {
    let suite = load_fixture(&strings_path("match.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let pattern = case.inputs["pattern"].as_str().unwrap();
        let result = ferray_strings::match_(&arr, pattern).unwrap();
        let expected = parse_bool_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        for (i, (&a, &e)) in result_slice.iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "match case '{}' [{i}]", case.name);
        }
    }
}

#[test]
fn oracle_extract() {
    let suite = load_fixture(&strings_path("extract.json"));
    for case in &suite.test_cases {
        let arr = make_string_array(&case.inputs["x"]);
        let pattern = case.inputs["pattern"].as_str().unwrap();
        let result = ferray_strings::extract(&arr, pattern).unwrap();
        let expected = parse_string_data(&case.expected["data"]);
        for (i, (a, e)) in result.as_slice().iter().zip(expected.iter()).enumerate() {
            assert_eq!(a, e, "extract case '{}' [{i}]", case.name);
        }
    }
}
