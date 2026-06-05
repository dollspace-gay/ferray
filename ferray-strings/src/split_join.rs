// ferray-strings: Split and join operations (REQ-11)
//
// Implements split and join — elementwise on StringArray.
//
// ## REQ status
//
// SHIPPED:
//   - REQ-11 split/join — `split`, `rsplit`, `splitlines` (`pub fn`) produce
//     a rectangular 2-D `StringArray2` (right-padded with empty strings to
//     the widest row); `split_ragged` (`pub fn`) keeps the ragged
//     `Vec<Vec<String>>` form for callers that need it. `join` /
//     `join_array` (`pub fn`) concatenate parts with a separator. An empty
//     separator is rejected (CPython/`np.char.split` raise on it), and
//     `splitlines` recognizes the full CPython line-boundary set. Matches
//     `numpy.strings.split`/`rsplit`/`splitlines`/`join`.
//
// Consumers (non-test): re-exported from the crate root
// (`ferray-strings/src/lib.rs` `pub use split_join::{join, join_array,
// rsplit, split, split_ragged, splitlines}`) and bound at the Python surface
// in `ferray-python/src/char.rs` (`fs::split_ragged` backs the `split` shim,
// `fs::rsplit` the `rsplit` shim, `fs::splitlines` the `splitlines` shim, and
// the `join` shim), which back `numpy.char`/`numpy.strings`.

use ferray_core::dimension::{Dimension, Ix1, Ix2};
use ferray_core::error::{FerrayError, FerrayResult};

use crate::string_array::{StringArray, StringArray1, StringArray2};

/// Reject the empty separator: Rust's `str::split("")` returns the
/// surprising `["", "a", "b", "c", ""]` pattern (one empty token per
/// boundary including the ends), but numpy's `np.char.split` raises
/// ValueError on an empty separator. Reject up front to match (#283).
fn validate_separator(sep: &str) -> FerrayResult<()> {
    if sep.is_empty() {
        return Err(FerrayError::invalid_value(
            "split separator must not be empty",
        ));
    }
    Ok(())
}

/// Split each string element by the given separator.
///
/// Returns a 2-D `StringArray` of shape `(n_inputs, max_parts)` where row
/// `i` contains the parts produced by splitting element `i`. Rows shorter
/// than `max_parts` are padded with empty strings (#277). Use
/// [`split_ragged`] when you need the unpadded `Vec<Vec<String>>` form.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn split<D: Dimension>(a: &StringArray<D>, sep: &str) -> FerrayResult<StringArray2> {
    validate_separator(sep)?;
    let parts: Vec<Vec<String>> = a
        .iter()
        .map(|s| s.split(sep).map(String::from).collect())
        .collect();
    let n_inputs = parts.len();
    let max_parts = parts.iter().map(Vec::len).max().unwrap_or(0);
    let mut flat: Vec<String> = Vec::with_capacity(n_inputs * max_parts);
    for row in &parts {
        for j in 0..max_parts {
            flat.push(row.get(j).cloned().unwrap_or_default());
        }
    }
    StringArray2::from_vec(Ix2::new([n_inputs, max_parts]), flat)
}

/// Right-to-left counterpart of [`split`] (#515).
///
/// Splits each element on `sep` starting from the right. With the
/// optional `maxsplit` cap, only the rightmost `maxsplit` separators
/// produce splits — leading remainder is kept as one piece. Mirrors
/// `numpy.strings.rsplit`.
///
/// # Errors
/// Returns an error if `sep` is empty or array construction fails.
pub fn rsplit<D: Dimension>(
    a: &StringArray<D>,
    sep: &str,
    maxsplit: Option<usize>,
) -> FerrayResult<StringArray2> {
    validate_separator(sep)?;
    let parts: Vec<Vec<String>> = a
        .iter()
        .map(|s| match maxsplit {
            None => s.rsplit(sep).map(String::from).collect::<Vec<_>>(),
            Some(n) => s.rsplitn(n + 1, sep).map(String::from).collect::<Vec<_>>(),
        })
        .map(|mut v| {
            v.reverse();
            v
        })
        .collect();
    let n_inputs = parts.len();
    let max_parts = parts.iter().map(Vec::len).max().unwrap_or(0);
    let mut flat: Vec<String> = Vec::with_capacity(n_inputs * max_parts);
    for row in &parts {
        for j in 0..max_parts {
            flat.push(row.get(j).cloned().unwrap_or_default());
        }
    }
    StringArray2::from_vec(Ix2::new([n_inputs, max_parts]), flat)
}

/// Split each element on universal newlines (#515). Equivalent to
/// `numpy.strings.splitlines`.
///
/// Returns a 2-D `StringArray` shaped `(n_inputs, max_lines)` with
/// trailing empty padding when the per-element line count differs.
/// `keepends = true` retains the line terminator on each kept line,
/// matching Python/NumPy behavior.
///
/// # Errors
/// Returns an error if array construction fails.
pub fn splitlines<D: Dimension>(a: &StringArray<D>, keepends: bool) -> FerrayResult<StringArray2> {
    let parts: Vec<Vec<String>> = a
        .iter()
        .map(|s| split_universal_newlines(s, keepends))
        .collect();
    let n_inputs = parts.len();
    let max_lines = parts.iter().map(Vec::len).max().unwrap_or(0);
    let mut flat: Vec<String> = Vec::with_capacity(n_inputs * max_lines);
    for row in &parts {
        for j in 0..max_lines {
            flat.push(row.get(j).cloned().unwrap_or_default());
        }
    }
    StringArray2::from_vec(Ix2::new([n_inputs, max_lines]), flat)
}

/// Whether `c` is a line boundary for `splitlines`, matching Python
/// `str.splitlines` / `numpy.char.splitlines`.
///
/// Python breaks on the full universal-newline set, derived live
/// (`("a" + sep + "b").splitlines()` has length > 1): `\n`, `\v` (U+000B),
/// `\f` (U+000C), `\r`, U+001C/001D/001E (FS/GS/RS), U+0085 (NEL),
/// U+2028 (LS), U+2029 (PS) — exactly ten boundaries (note: NOT U+001F, and
/// `\r\n` is a single boundary, handled by the caller). Upstream:
/// numpy/_core/strings.py:1528 splitlines -> `str.splitlines`.
fn is_line_boundary(c: char) -> bool {
    matches!(
        c,
        '\n' | '\u{0B}'
            | '\u{0C}'
            | '\r'
            | '\u{1C}'
            | '\u{1D}'
            | '\u{1E}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// Universal-newline split: a `\r\n` pair is a single boundary; every other
/// boundary character in [`is_line_boundary`] terminates a line on its own.
/// The result mirrors Python's `str.splitlines`.
fn split_universal_newlines(s: &str, keepends: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut line_start = 0usize;
    let mut chars = s.char_indices().peekable();
    while let Some((idx, c)) = chars.next() {
        if !is_line_boundary(c) {
            continue;
        }
        // `\r\n` collapses into one boundary; consume the trailing `\n`.
        let mut eol_end = idx + c.len_utf8();
        if c == '\r'
            && let Some(&(_, '\n')) = chars.peek()
        {
            chars.next();
            eol_end += '\n'.len_utf8();
        }
        let line_end = if keepends { eol_end } else { idx };
        out.push(s[line_start..line_end].to_string());
        line_start = eol_end;
    }
    if line_start < s.len() {
        out.push(s[line_start..].to_string());
    }
    out
}

/// Ragged-result variant of [`split`]: returns a `Vec<Vec<String>>` so
/// callers that need the unpadded splits per element don't have to
/// strip empty padding from the 2-D result (#277).
///
/// # Errors
/// Returns an error only for internal failures.
pub fn split_ragged<D: Dimension>(a: &StringArray<D>, sep: &str) -> FerrayResult<Vec<Vec<String>>> {
    validate_separator(sep)?;
    let result: Vec<Vec<String>> = a
        .iter()
        .map(|s| s.split(sep).map(String::from).collect())
        .collect();
    Ok(result)
}

/// Join a collection of string vectors using the given separator.
///
/// Each element in the input is a `Vec<String>` which is joined into
/// a single string. Returns a 1-D `StringArray`.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn join(sep: &str, items: &[Vec<String>]) -> FerrayResult<StringArray1> {
    let data: Vec<String> = items.iter().map(|parts| parts.join(sep)).collect();
    let dim = Ix1::new([data.len()]);
    StringArray1::from_vec(dim, data)
}

/// Join each string element of a `StringArray` using the given separator.
///
/// This variant takes a `StringArray` and joins all elements into a single
/// string. Returns a 1-D `StringArray` with one element.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn join_array<D: Dimension>(sep: &str, a: &StringArray<D>) -> FerrayResult<StringArray1> {
    let joined: String = a
        .iter()
        .map(std::string::String::as_str)
        .collect::<Vec<&str>>()
        .join(sep);
    let dim = Ix1::new([1]);
    StringArray1::from_vec(dim, vec![joined])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_split() {
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let s = result.as_slice();
        assert_eq!(s, &["a", "b", "c", "d"]);
    }

    // ---- rsplit / splitlines (#515) ------------------------------------

    #[test]
    fn rsplit_basic_no_limit() {
        let a = array(&["a-b-c", "x-y"]).unwrap();
        let r = rsplit(&a, "-", None).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        let s = r.as_slice();
        // Trailing empty pads (matches `split`'s padding convention).
        // Row 0: ["a","b","c"], Row 1: ["x","y",""]
        assert_eq!(s, &["a", "b", "c", "x", "y", ""]);
    }

    #[test]
    fn rsplit_with_maxsplit_one() {
        // maxsplit=1: only the rightmost separator splits.
        let a = array(&["a-b-c-d"]).unwrap();
        let r = rsplit(&a, "-", Some(1)).unwrap();
        assert_eq!(r.shape(), &[1, 2]);
        let s = r.as_slice();
        assert_eq!(s, &["a-b-c", "d"]);
    }

    #[test]
    fn splitlines_with_lf_and_crlf() {
        let a = array(&["one\ntwo\r\nthree", "single"]).unwrap();
        let r = splitlines(&a, false).unwrap();
        // Row 0 has 3 lines, row 1 has 1 line. Padding to 3.
        assert_eq!(r.shape(), &[2, 3]);
        let s = r.as_slice();
        assert_eq!(s, &["one", "two", "three", "single", "", ""]);
    }

    #[test]
    fn splitlines_keepends_retains_terminator() {
        let a = array(&["x\ny\r\nz"]).unwrap();
        let r = splitlines(&a, true).unwrap();
        let s = r.as_slice();
        assert_eq!(s, &["x\n", "y\r\n", "z"]);
    }

    #[test]
    fn splitlines_handles_solo_carriage_return() {
        let a = array(&["a\rb"]).unwrap();
        let r = splitlines(&a, false).unwrap();
        let s = r.as_slice();
        assert_eq!(s, &["a", "b"]);
    }

    // ---- Universal-newline boundary divergence regression (#917) -------
    // Python `str.splitlines` / `numpy.char.splitlines` break on the full
    // set; ferray previously only broke on \n/\r/\r\n. Expected values mirror
    // e.g. `"a\x0bb\x0cc".splitlines() == ["a", "b", "c"]` (R-CHAR-3). These
    // exercise the boundary logic directly via `split_universal_newlines`.

    #[test]
    fn splitlines_breaks_on_vtab_and_formfeed() {
        // "a\x0bb\x0cc".splitlines() == ["a", "b", "c"]
        assert_eq!(
            split_universal_newlines("a\u{0B}b\u{0C}c", false),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn splitlines_breaks_on_c0_separators() {
        // "a\x1cb\x1dc\x1ed".splitlines() == ["a", "b", "c", "d"]
        // (FS/GS/RS are boundaries; US \x1f is NOT, per Python.)
        assert_eq!(
            split_universal_newlines("a\u{1C}b\u{1D}c\u{1E}d", false),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn splitlines_unit_separator_is_not_a_boundary() {
        // "a\x1fb".splitlines() == ["a\x1fb"] — U+001F is NOT a boundary.
        assert_eq!(
            split_universal_newlines("a\u{1F}b", false),
            vec!["a\u{1F}b".to_string()]
        );
    }

    #[test]
    fn splitlines_breaks_on_nel_and_unicode_separators() {
        // "a\x85b c d".splitlines() == ["a", "b", "c", "d"]
        // (NEL U+0085, LINE SEP U+2028, PARA SEP U+2029.)
        assert_eq!(
            split_universal_newlines("a\u{85}b\u{2028}c\u{2029}d", false),
            vec![
                "a".to_string(),
                "b".to_string(),
                "c".to_string(),
                "d".to_string()
            ]
        );
    }

    #[test]
    fn splitlines_keepends_retains_unicode_terminators() {
        // keepends keeps each (multi-byte) terminator on its line, matching
        // `"a\x85b c".splitlines(True) == ["a\x85", "b ", "c"]`.
        assert_eq!(
            split_universal_newlines("a\u{85}b\u{2028}c", true),
            vec![
                "a\u{85}".to_string(),
                "b\u{2028}".to_string(),
                "c".to_string()
            ]
        );
    }

    #[test]
    fn test_split_multiple_parts() {
        let a = array(&["a-b-c"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(result.as_slice(), &["a", "b", "c"]);
    }

    #[test]
    fn test_split_no_separator_found() {
        let a = array(&["hello"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        assert_eq!(result.as_slice(), &["hello"]);
    }

    #[test]
    fn test_split_pads_short_rows_with_empty_strings() {
        // #277: rows shorter than max_parts must be padded with "".
        let a = array(&["a-b", "x-y-z"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: ["a", "b", ""] (padded), Row 1: ["x", "y", "z"]
        assert_eq!(result.as_slice(), &["a", "b", "", "x", "y", "z"]);
    }

    #[test]
    fn test_split_ragged_returns_unpadded() {
        // #277: split_ragged keeps the per-element variable length.
        let a = array(&["a-b", "x-y-z"]).unwrap();
        let result = split_ragged(&a, "-").unwrap();
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["x".to_string(), "y".to_string(), "z".to_string()],
            ]
        );
    }

    #[test]
    fn test_join() {
        let items = vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["c".to_string(), "d".to_string()],
        ];
        let result = join("-", &items).unwrap();
        assert_eq!(result.as_slice(), &["a-b", "c-d"]);
    }

    #[test]
    fn test_join_array() {
        let a = array(&["hello", "world"]).unwrap();
        let result = join_array(" ", &a).unwrap();
        assert_eq!(result.as_slice(), &["hello world"]);
    }

    #[test]
    fn test_split_ac4() {
        // AC-4: strings::split_ragged(&["a-b", "c-d"], "-") returns
        // [vec!["a","b"], vec!["c","d"]] — the ragged form preserves
        // the original AC behavior.
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split_ragged(&a, "-").unwrap();
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
            ]
        );
    }

    // ----- Empty separator rejection (#283) ------------------------------

    #[test]
    fn test_split_empty_separator_errs() {
        // #283: Rust's str::split("") returns the surprising
        // ["", "a", "b", "c", ""] pattern with empty tokens around
        // every char boundary. numpy's np.char.split raises ValueError
        // for an empty separator. Match numpy's strict path.
        let a = array(&["abc", "def"]).unwrap();
        let err = split(&a, "").unwrap_err();
        assert!(
            err.to_string().contains("separator must not be empty"),
            "expected empty-separator error, got: {err}"
        );
    }

    #[test]
    fn test_split_ragged_empty_separator_errs() {
        let a = array(&["abc"]).unwrap();
        assert!(split_ragged(&a, "").is_err());
    }

    #[test]
    fn test_split_single_char_separator_works() {
        // Sanity check: a single-char separator still splits correctly
        // — the validation gates only the empty-string case.
        let a = array(&["a,b,c"]).unwrap();
        let result = split_ragged(&a, ",").unwrap();
        assert_eq!(result[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_multichar_separator_works() {
        // Multi-character separator: "::" should split exactly on the
        // 2-byte sequence, not on each byte.
        let a = array(&["a::b::c"]).unwrap();
        let result = split_ragged(&a, "::").unwrap();
        assert_eq!(result[0], vec!["a", "b", "c"]);
    }
}
