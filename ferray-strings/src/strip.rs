// ferray-strings: Stripping operations (REQ-7)
//
// Implements strip, lstrip, rstrip — elementwise on StringArray.

use ferray_core::dimension::Dimension;
use ferray_core::error::FerrayResult;

use crate::string_array::StringArray;

/// Build a char-set predicate from `chars`. Shared by strip/lstrip/rstrip
/// so the `char_set: Vec<char>` construction lives in one place (#280).
fn char_set_predicate(chars: &str) -> impl Fn(char) -> bool + '_ {
    let char_set: Vec<char> = chars.chars().collect();
    move |c: char| char_set.contains(&c)
}

/// Whitespace predicate for the default (no-`chars`) strip path, matching
/// Python `str.strip` / `numpy.char.strip` (numpy/_core/strings.py:1034 strip
/// -> `str.strip`).
///
/// Python strips every character for which `str.isspace()` is true. That set
/// differs from Rust's `char::is_whitespace` (the Unicode `White_Space`
/// property) by exactly the four C0 information separators
/// U+001C..U+001F (FS/GS/RS/US), which CPython treats as whitespace
/// (Objects/unicodectype.c) but Unicode `White_Space` excludes. Derived live:
/// `[hex(c) for c in range(0x110000) if (chr(c)).strip() == '']` has 29
/// entries; Rust `is_whitespace` covers all but those four, with no extras.
fn is_python_strip_whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1C}'..='\u{1F}')
}

/// Strip leading and trailing characters from each string element.
///
/// If `chars` is `None`, strips whitespace. Otherwise strips any character
/// present in the `chars` string.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn strip<D: Dimension>(
    a: &StringArray<D>,
    chars: Option<&str>,
) -> FerrayResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim_matches(is_python_strip_whitespace).to_string()),
        Some(ch) => {
            let pred = char_set_predicate(ch);
            a.map(|s| s.trim_matches(&pred).to_string())
        }
    }
}

/// Strip leading characters from each string element.
///
/// If `chars` is `None`, strips leading whitespace. Otherwise strips any
/// character present in the `chars` string from the left.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn lstrip<D: Dimension>(
    a: &StringArray<D>,
    chars: Option<&str>,
) -> FerrayResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim_start_matches(is_python_strip_whitespace).to_string()),
        Some(ch) => {
            let pred = char_set_predicate(ch);
            a.map(|s| s.trim_start_matches(&pred).to_string())
        }
    }
}

/// Strip trailing characters from each string element.
///
/// If `chars` is `None`, strips trailing whitespace. Otherwise strips any
/// character present in the `chars` string from the right.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn rstrip<D: Dimension>(
    a: &StringArray<D>,
    chars: Option<&str>,
) -> FerrayResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim_end_matches(is_python_strip_whitespace).to_string()),
        Some(ch) => {
            let pred = char_set_predicate(ch);
            a.map(|s| s.trim_end_matches(&pred).to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_strip_whitespace() {
        let a = array(&["  hello  ", "\tworld\n"]).unwrap();
        let b = strip(&a, None).unwrap();
        assert_eq!(b.as_slice(), &["hello", "world"]);
    }

    #[test]
    fn test_strip_chars() {
        let a = array(&["xxhelloxx", "xyworldyx"]).unwrap();
        let b = strip(&a, Some("xy")).unwrap();
        assert_eq!(b.as_slice(), &["hello", "world"]);
    }

    #[test]
    fn test_lstrip_whitespace() {
        let a = array(&["  hello  ", "\tworld\n"]).unwrap();
        let b = lstrip(&a, None).unwrap();
        assert_eq!(b.as_slice(), &["hello  ", "world\n"]);
    }

    #[test]
    fn test_lstrip_chars() {
        let a = array(&["xxhello", "xyhello"]).unwrap();
        let b = lstrip(&a, Some("xy")).unwrap();
        assert_eq!(b.as_slice(), &["hello", "hello"]);
    }

    #[test]
    fn test_rstrip_whitespace() {
        let a = array(&["  hello  ", "\tworld\n"]).unwrap();
        let b = rstrip(&a, None).unwrap();
        assert_eq!(b.as_slice(), &["  hello", "\tworld"]);
    }

    #[test]
    fn test_rstrip_chars() {
        let a = array(&["helloxx", "worldyx"]).unwrap();
        let b = rstrip(&a, Some("xy")).unwrap();
        assert_eq!(b.as_slice(), &["hello", "world"]);
    }

    #[test]
    fn test_strip_empty_string() {
        let a = array(&["", "   "]).unwrap();
        let b = strip(&a, None).unwrap();
        assert_eq!(b.as_slice(), &["", ""]);
    }

    // ---- C0 separator whitespace divergence regression (#916) ----------
    // Python `str.strip` / `numpy.char.strip` treat U+001C..U+001F as
    // whitespace; Rust `str::trim` does not. Expected values mirror
    // `("\x1chi\x1c").strip() == "hi"` etc. (R-CHAR-3).

    #[test]
    fn test_strip_c0_separators() {
        for ch in ['\u{1C}', '\u{1D}', '\u{1E}', '\u{1F}'] {
            let s = format!("{ch}hi{ch}");
            let a = array(&[s.as_str()]).unwrap();
            let b = strip(&a, None).unwrap();
            assert_eq!(
                b.as_slice(),
                &["hi"],
                "strip failed for U+{:04X}",
                ch as u32
            );
        }
    }

    #[test]
    fn test_lstrip_c0_separators() {
        for ch in ['\u{1C}', '\u{1D}', '\u{1E}', '\u{1F}'] {
            let s = format!("{ch}ab");
            let a = array(&[s.as_str()]).unwrap();
            let b = lstrip(&a, None).unwrap();
            assert_eq!(
                b.as_slice(),
                &["ab"],
                "lstrip failed for U+{:04X}",
                ch as u32
            );
        }
    }

    #[test]
    fn test_rstrip_c0_separators() {
        for ch in ['\u{1C}', '\u{1D}', '\u{1E}', '\u{1F}'] {
            let s = format!("ab{ch}");
            let a = array(&[s.as_str()]).unwrap();
            let b = rstrip(&a, None).unwrap();
            assert_eq!(
                b.as_slice(),
                &["ab"],
                "rstrip failed for U+{:04X}",
                ch as u32
            );
        }
    }

    #[test]
    fn test_strip_unicode_whitespace_still_works() {
        // Guard: the previously-converged Unicode whitespace
        // (U+0085 NEL, U+00A0 NBSP, U+000B VT, U+000C FF) still strips,
        // matching `("\x85\xa0\x0b\x0c hi \t\n\r ").strip() == "hi"`.
        let a = array(&["\u{85}\u{A0}\u{0B}\u{0C} hi \t\n\r "]).unwrap();
        let b = strip(&a, None).unwrap();
        assert_eq!(b.as_slice(), &["hi"]);
    }
}
