// ferrum-strings: Regex operations (REQ-12, REQ-13)
//
// Implements match_ and extract using the `regex` crate.

use ferrum_core::Array;
use ferrum_core::dimension::{Dimension, Ix1};
use ferrum_core::error::{FerrumError, FerrumResult};
use regex::Regex;

use crate::string_array::{StringArray, StringArray1};

/// Test whether each string element matches the given regex pattern.
///
/// Returns an `Array<bool>` where each element indicates whether the
/// corresponding string contains a match for the pattern.
///
/// # Errors
/// Returns `FerrumError::InvalidValue` if the regex pattern is invalid.
/// Returns an error if the internal array construction fails.
pub fn match_<D: Dimension>(a: &StringArray<D>, pattern: &str) -> FerrumResult<Array<bool, Ix1>> {
    let re = Regex::new(pattern)
        .map_err(|e| FerrumError::invalid_value(format!("invalid regex pattern: {e}")))?;

    let data: Vec<bool> = a.map_to_vec(|s| re.is_match(s));
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Extract the first capture group from each string element.
///
/// For each string, finds the first match of the pattern and returns
/// the contents of capture group 1. If there is no match or no capture
/// group, an empty string is returned.
///
/// The pattern must contain at least one capture group `(...)`.
///
/// # Errors
/// Returns `FerrumError::InvalidValue` if the regex pattern is invalid.
/// Returns an error if the internal array construction fails.
pub fn extract<D: Dimension>(a: &StringArray<D>, pattern: &str) -> FerrumResult<StringArray1> {
    let re = Regex::new(pattern)
        .map_err(|e| FerrumError::invalid_value(format!("invalid regex pattern: {e}")))?;

    let data: Vec<String> = a
        .iter()
        .map(|s| {
            re.captures(s)
                .and_then(|caps| caps.get(1))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default()
        })
        .collect();

    let dim = Ix1::new([data.len()]);
    StringArray1::from_vec(dim, data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_match_basic() {
        let a = array(&["hello123", "world", "foo42"]).unwrap();
        let result = match_(&a, r"\d+").unwrap();
        let data = result.as_slice().unwrap();
        assert_eq!(data, &[true, false, true]);
    }

    #[test]
    fn test_match_full_pattern() {
        let a = array(&["abc", "def", "abcdef"]).unwrap();
        let result = match_(&a, r"^abc$").unwrap();
        let data = result.as_slice().unwrap();
        assert_eq!(data, &[true, false, false]);
    }

    #[test]
    fn test_match_invalid_regex() {
        let a = array(&["hello"]).unwrap();
        let result = match_(&a, r"[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_extract_capture_group() {
        let a = array(&["hello123world", "foo42bar", "nodigits"]).unwrap();
        let result = extract(&a, r"(\d+)").unwrap();
        assert_eq!(result.as_slice(), &["123", "42", ""]);
    }

    #[test]
    fn test_extract_named_group() {
        let a = array(&["user:alice", "user:bob", "invalid"]).unwrap();
        let result = extract(&a, r"user:(\w+)").unwrap();
        assert_eq!(result.as_slice(), &["alice", "bob", ""]);
    }

    #[test]
    fn test_extract_no_match() {
        let a = array(&["no match here"]).unwrap();
        let result = extract(&a, r"(\d+)").unwrap();
        assert_eq!(result.as_slice(), &[""]);
    }

    #[test]
    fn test_extract_invalid_regex() {
        let a = array(&["hello"]).unwrap();
        let result = extract(&a, r"[invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_match_and_extract_ac5() {
        // AC-5: Regex match_ and extract work correctly with capture groups
        let a = array(&["abc123", "def", "ghi456"]).unwrap();

        let matched = match_(&a, r"\d+").unwrap();
        let matched_data = matched.as_slice().unwrap();
        assert_eq!(matched_data, &[true, false, true]);

        let extracted = extract(&a, r"([a-z]+)(\d+)").unwrap();
        // "def" has no digits, so no match => empty string
        assert_eq!(extracted.as_slice(), &["abc", "", "ghi"]);
    }

    #[test]
    fn test_extract_empty_string() {
        let a = array(&["", "abc"]).unwrap();
        let result = extract(&a, r"(abc)").unwrap();
        assert_eq!(result.as_slice(), &["", "abc"]);
    }
}
