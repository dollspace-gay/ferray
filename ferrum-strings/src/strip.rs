// ferrum-strings: Stripping operations (REQ-7)
//
// Implements strip, lstrip, rstrip — elementwise on StringArray.

use ferrum_core::dimension::Dimension;
use ferrum_core::error::FerrumResult;

use crate::string_array::StringArray;

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
) -> FerrumResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim().to_string()),
        Some(ch) => {
            let char_set: Vec<char> = ch.chars().collect();
            a.map(|s| s.trim_matches(|c: char| char_set.contains(&c)).to_string())
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
) -> FerrumResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim_start().to_string()),
        Some(ch) => {
            let char_set: Vec<char> = ch.chars().collect();
            a.map(|s| {
                s.trim_start_matches(|c: char| char_set.contains(&c))
                    .to_string()
            })
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
) -> FerrumResult<StringArray<D>> {
    match chars {
        None => a.map(|s| s.trim_end().to_string()),
        Some(ch) => {
            let char_set: Vec<char> = ch.chars().collect();
            a.map(|s| {
                s.trim_end_matches(|c: char| char_set.contains(&c))
                    .to_string()
            })
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
}
