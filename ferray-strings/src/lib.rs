// ferray-strings: Vectorized string operations on arrays of strings
//
// Implements `numpy.strings` (NumPy 2.0+): vectorized elementwise string
// operations on arrays of strings with broadcasting. Covers case manipulation,
// alignment/padding, stripping, find/replace, splitting/joining, and regex
// support. Operates on `StringArray` — a separate array type backed by
// `Vec<String>`.

//! # ferray-strings
//!
//! Vectorized string operations on arrays of strings, analogous to
//! `numpy.strings` in `NumPy` 2.0+.
//!
//! The primary type is [`StringArray`], a specialized N-dimensional array
//! backed by `Vec<String>`. Since `String` does not implement
//! [`ferray_core::Element`], this type is separate from `NdArray<T, D>`.
//!
//! # Quick Start
//!
//! ```ignore
//! use ferray_strings::*;
//!
//! let a = array(&["hello", "world"]).unwrap();
//! let b = upper(&a).unwrap();
//! assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
//! ```

// Workspace convention: every public function returns FerrayResult<T> and
// the FerrayError variants are documented once on the type, not on every
// returning function.
#![allow(
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::items_after_statements,
    clippy::option_if_let_else,
    clippy::too_long_first_doc_paragraph,
    clippy::needless_pass_by_value,
    clippy::match_same_arms
)]

pub mod align;
pub mod case;
pub mod classify;
pub mod concat;
pub mod extras;
pub mod regex_ops;
pub mod search;
pub mod serde_impl;
pub mod split_join;
pub mod str_ops;
pub mod string_array;
pub mod strip;

// Re-export types
pub use string_array::{StringArray, StringArray1, StringArray2, array};

// Re-export operations for flat namespace (like numpy.strings.upper etc.)
pub use align::{center, ljust, ljust_with, rjust, rjust_with, zfill};
pub use case::{capitalize, lower, title, upper};
pub use classify::{
    isalnum, isalpha, isdecimal, isdigit, islower, isnumeric, isspace, istitle, isupper,
};
pub use concat::{add, add_same, multiply};
pub use extras::{decode, encode, expandtabs, mod_, partition, rpartition, slice, translate};
pub use regex_ops::{extract, extract_compiled, match_, match_compiled};
// Re-export `regex::Regex` so callers of `match_compiled`/`extract_compiled`
// don't have to add a direct `regex` dependency to construct one.
pub use regex::Regex;
pub use search::{count, endswith, find, index, replace, rfind, rindex, startswith};
pub use split_join::{join, join_array, split, split_ragged};
pub use str_ops::{equal, greater, greater_equal, less, less_equal, not_equal, str_len, swapcase};
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
        // AC-4 (#277): strings::split returns a 2-D StringArray of
        // shape (n_inputs, max_parts). split_ragged keeps the
        // ragged Vec<Vec<String>> form for callers that need it.
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.as_slice(), &["a", "b", "c", "d"]);
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

    // -----------------------------------------------------------------
    // #520 — shape preservation for ND StringArrays across every op.
    //
    // Every operation that takes `StringArray<D>` must thread `D`
    // through the output. These tests pin that contract on a 2x2
    // input so regressions (e.g. an accidental `Ix1::new([len])` in
    // the output constructor) are caught immediately.
    // -----------------------------------------------------------------

    fn two_by_two(vals: &[&str; 4]) -> crate::StringArray2 {
        crate::StringArray2::from_rows(&[&[vals[0], vals[1]], &[vals[2], vals[3]]]).unwrap()
    }

    #[test]
    fn shape_preserved_case_ops_2d() {
        let a = two_by_two(&["Hello", "World", "foo", "Bar"]);
        assert_eq!(upper(&a).unwrap().shape(), &[2, 2]);
        assert_eq!(lower(&a).unwrap().shape(), &[2, 2]);
        assert_eq!(capitalize(&a).unwrap().shape(), &[2, 2]);
        assert_eq!(title(&a).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_align_ops_2d() {
        let a = two_by_two(&["a", "bb", "ccc", "dddd"]);
        assert_eq!(center(&a, 6, ' ').unwrap().shape(), &[2, 2]);
        assert_eq!(ljust(&a, 6).unwrap().shape(), &[2, 2]);
        assert_eq!(rjust(&a, 6).unwrap().shape(), &[2, 2]);
        assert_eq!(zfill(&a, 6).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_strip_ops_2d() {
        let a = two_by_two(&["  a  ", "  b  ", "  c  ", "  d  "]);
        assert_eq!(strip(&a, None).unwrap().shape(), &[2, 2]);
        assert_eq!(lstrip(&a, None).unwrap().shape(), &[2, 2]);
        assert_eq!(rstrip(&a, None).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_concat_ops_2d() {
        let a = two_by_two(&["ab", "cd", "ef", "gh"]);
        // `add` currently flattens to IxDyn regardless of input rank
        // (it supports cross-rank broadcasting), so it's not expected
        // to preserve D. Just verify the total element count.
        let b = two_by_two(&["!", "!", "!", "!"]);
        let ab = add(&a, &b).unwrap();
        assert_eq!(ab.shape(), &[2, 2]);
        // multiply preserves D
        assert_eq!(multiply(&a, 2).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_search_ops_2d() {
        let a = two_by_two(&["hello", "help", "world", "word"]);
        // Each of these returns Array<T, D> — verify D is preserved.
        assert_eq!(find(&a, "ell").unwrap().shape(), &[2, 2]);
        assert_eq!(count(&a, "l").unwrap().shape(), &[2, 2]);
        assert_eq!(startswith(&a, "he").unwrap().shape(), &[2, 2]);
        assert_eq!(endswith(&a, "d").unwrap().shape(), &[2, 2]);
        assert_eq!(replace(&a, "l", "L", None).unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_regex_ops_2d() {
        let a = two_by_two(&["abc123", "x", "y42", "zzz"]);
        // match_ preserves D; extract flattens by design (ragged results).
        assert_eq!(match_(&a, r"\d+").unwrap().shape(), &[2, 2]);
    }

    #[test]
    fn shape_preserved_case_ops_3d() {
        // Just to confirm we aren't special-casing Ix2 somewhere — bump to Ix3.
        use ferray_core::dimension::Ix3;
        let data: Vec<String> = (0..8).map(|i| format!("s{i}")).collect();
        let a = crate::StringArray::<Ix3>::from_vec(Ix3::new([2, 2, 2]), data).unwrap();
        assert_eq!(upper(&a).unwrap().shape(), &[2, 2, 2]);
        assert_eq!(lower(&a).unwrap().shape(), &[2, 2, 2]);
    }

    // --- Unicode / multi-byte character tests ---

    #[test]
    fn unicode_upper_lower() {
        let a = array(&["café", "naïve", "über"]).unwrap();
        let u = upper(&a).unwrap();
        assert_eq!(u.as_slice(), &["CAFÉ", "NAÏVE", "ÜBER"]);
        let l = lower(&u).unwrap();
        assert_eq!(l.as_slice(), &["café", "naïve", "über"]);
    }

    #[test]
    fn unicode_capitalize() {
        let a = array(&["ñoño", "straße"]).unwrap();
        let c = capitalize(&a).unwrap();
        assert_eq!(c.as_slice()[0], "Ñoño");
        // Rust's capitalize of "straße" -> "Straße"
        assert_eq!(c.as_slice()[1], "Straße");
    }

    #[test]
    fn unicode_find() {
        let a = array(&["日本語テスト", "こんにちは"]).unwrap();
        let r = find(&a, "テスト").unwrap();
        let data = r.as_slice().unwrap();
        assert_eq!(data[0], 3); // "テスト" starts at byte position, but find uses char position...
        // Actually, find returns the byte index via str::find. Check:
        // "日本語テスト".find("テスト") returns byte offset 9
        // But our find should return character index or byte index?
        // Let's just verify it finds it (>= 0) vs not found (-1)
        assert!(data[0] >= 0); // found
        assert_eq!(data[1], -1); // not found
    }

    #[test]
    fn unicode_strip() {
        let a = array(&["  héllo  ", "  wörld  "]).unwrap();
        let s = strip(&a, None).unwrap();
        assert_eq!(s.as_slice(), &["héllo", "wörld"]);
    }

    #[test]
    fn unicode_replace() {
        let a = array(&["café latte"]).unwrap();
        let r = replace(&a, "café", "tea", None).unwrap();
        assert_eq!(r.as_slice(), &["tea latte"]);
    }

    #[test]
    fn emoji_operations() {
        let a = array(&["hello 🌍", "rust 🦀"]).unwrap();
        let u = upper(&a).unwrap();
        assert_eq!(u.as_slice(), &["HELLO 🌍", "RUST 🦀"]);
        let c = count(&a, "🌍").unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1, 0]);
    }

    #[test]
    fn cjk_characters() {
        let a = array(&["你好世界", "こんにちは"]).unwrap();
        let starts = startswith(&a, "你好").unwrap();
        assert_eq!(starts.as_slice().unwrap(), &[true, false]);
        let ends = endswith(&a, "世界").unwrap();
        assert_eq!(ends.as_slice().unwrap(), &[true, false]);
    }

    // ----- Empty array tests (#282) -----

    #[test]
    fn empty_array_upper() {
        let a = StringArray1::from_vec(ferray_core::dimension::Ix1::new([0]), vec![]).unwrap();
        let u = upper(&a).unwrap();
        assert_eq!(u.len(), 0);
    }

    #[test]
    fn empty_array_str_len() {
        let a = StringArray1::from_vec(ferray_core::dimension::Ix1::new([0]), vec![]).unwrap();
        let l = str_len(&a).unwrap();
        assert_eq!(l.size(), 0);
    }

    #[test]
    fn empty_array_find() {
        let a = StringArray1::from_vec(ferray_core::dimension::Ix1::new([0]), vec![]).unwrap();
        let f = find(&a, "x").unwrap();
        assert_eq!(f.size(), 0);
    }
}
