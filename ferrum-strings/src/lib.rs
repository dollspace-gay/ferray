// ferrum-strings: Vectorized string operations on arrays of strings
//
// Implements `numpy.strings` (NumPy 2.0+): vectorized elementwise string
// operations on arrays of strings with broadcasting. Covers case manipulation,
// alignment/padding, stripping, find/replace, splitting/joining, and regex
// support. Operates on `StringArray` — a separate array type backed by
// `Vec<String>`.

//! # ferrum-strings
//!
//! Vectorized string operations on arrays of strings, analogous to
//! `numpy.strings` in NumPy 2.0+.
//!
//! The primary type is [`StringArray`], a specialized N-dimensional array
//! backed by `Vec<String>`. Since `String` does not implement
//! [`ferrum_core::Element`], this type is separate from `NdArray<T, D>`.
//!
//! # Quick Start
//!
//! ```ignore
//! use ferrum_strings::*;
//!
//! let a = array(&["hello", "world"]).unwrap();
//! let b = upper(&a).unwrap();
//! assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
//! ```

pub mod align;
pub mod case;
pub mod concat;
pub mod regex_ops;
pub mod search;
pub mod split_join;
pub mod string_array;
pub mod strip;

// Re-export types
pub use string_array::{StringArray, StringArray1, StringArray2, array};

// Re-export operations for flat namespace (like numpy.strings.upper etc.)
pub use align::{center, ljust, rjust, zfill};
pub use case::{capitalize, lower, title, upper};
pub use concat::{add, multiply};
pub use regex_ops::{extract, match_};
pub use search::{count, endswith, find, replace, startswith};
pub use split_join::{join, join_array, split};
pub use strip::{lstrip, rstrip, strip};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn ac1_upper() {
        // AC-1: strings::upper(&["hello", "world"]) produces ["HELLO", "WORLD"]
        let a = array(&["hello", "world"]).unwrap();
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
    }

    #[test]
    fn ac2_add_broadcast_scalar() {
        // AC-2: strings::add broadcasts a scalar string against an array correctly
        let a = array(&["hello", "world"]).unwrap();
        let b = array(&["!"]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &["hello!", "world!"]);
    }

    #[test]
    fn ac3_find_indices() {
        // AC-3: strings::find(&a, "ll") returns correct indices
        let a = array(&["hello", "world"]).unwrap();
        let b = find(&a, "ll").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[2_i64, -1_i64]);
    }

    #[test]
    fn ac4_split() {
        // AC-4: strings::split(&["a-b", "c-d"], "-") returns [vec!["a","b"], vec!["c","d"]]
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
            ]
        );
    }

    #[test]
    fn ac5_regex() {
        // AC-5: Regex match_ and extract work correctly with capture groups
        let a = array(&["abc123", "def", "ghi456"]).unwrap();

        let matched = match_(&a, r"\d+").unwrap();
        let matched_data = matched.as_slice().unwrap();
        assert_eq!(matched_data, &[true, false, true]);

        let extracted = extract(&a, r"(\d+)").unwrap();
        assert_eq!(extracted.as_slice(), &["123", "", "456"]);
    }

    #[test]
    fn full_pipeline() {
        // End-to-end: strip, upper, add suffix, search
        let raw = array(&["  Hello  ", " World "]).unwrap();
        let stripped = strip(&raw, None).unwrap();
        let uppered = upper(&stripped).unwrap();
        let suffix = array(&["!"]).unwrap();
        let result = add(&uppered, &suffix).unwrap();
        assert_eq!(result.as_slice(), &["HELLO!", "WORLD!"]);

        let has_excl = endswith(&result, "!").unwrap();
        let data = has_excl.as_slice().unwrap();
        assert_eq!(data, &[true, true]);
    }

    #[test]
    fn case_round_trip() {
        let a = array(&["Hello World"]).unwrap();
        let low = lower(&a).unwrap();
        let titled = title(&low).unwrap();
        assert_eq!(titled.as_slice(), &["Hello World"]);
    }

    #[test]
    fn alignment_operations() {
        let a = array(&["hi"]).unwrap();
        let c = center(&a, 6, '-').unwrap();
        assert_eq!(c.as_slice(), &["--hi--"]);

        let l = ljust(&a, 6).unwrap();
        assert_eq!(l.as_slice(), &["hi    "]);

        let r = rjust(&a, 6).unwrap();
        assert_eq!(r.as_slice(), &["    hi"]);

        let z = zfill(&array(&["42"]).unwrap(), 5).unwrap();
        assert_eq!(z.as_slice(), &["00042"]);
    }

    #[test]
    fn strip_operations() {
        let a = array(&["  hello  "]).unwrap();
        assert_eq!(strip(&a, None).unwrap().as_slice(), &["hello"]);
        assert_eq!(lstrip(&a, None).unwrap().as_slice(), &["hello  "]);
        assert_eq!(rstrip(&a, None).unwrap().as_slice(), &["  hello"]);
    }

    #[test]
    fn search_operations() {
        let a = array(&["hello world", "foo bar"]).unwrap();
        let c = count(&a, "o").unwrap();
        let data = c.as_slice().unwrap();
        // "hello world" has 2 'o's, "foo bar" has 2 'o's
        assert_eq!(data, &[2_u64, 2]);
    }

    #[test]
    fn replace_operation() {
        let a = array(&["hello world"]).unwrap();
        let b = replace(&a, "world", "rust", None).unwrap();
        assert_eq!(b.as_slice(), &["hello rust"]);
    }

    #[test]
    fn multiply_operation() {
        let a = array(&["ab"]).unwrap();
        let b = multiply(&a, 3).unwrap();
        assert_eq!(b.as_slice(), &["ababab"]);
    }

    #[test]
    fn join_operation() {
        let parts = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];
        let result = join("-", &parts).unwrap();
        assert_eq!(result.as_slice(), &["a-b", "c-d"]);
    }

    #[test]
    fn capitalize_operation() {
        let a = array(&["hello world", "RUST"]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello world", "Rust"]);
    }

    #[test]
    fn string_array_2d() {
        let a = StringArray2::from_rows(&[&["a", "b"], &["c", "d"]]).unwrap();
        assert_eq!(a.shape(), &[2, 2]);
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &["A", "B", "C", "D"]);
        assert_eq!(b.shape(), &[2, 2]);
    }
}
