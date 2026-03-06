// ferrum-strings: Case manipulation operations (REQ-5)
//
// Implements upper, lower, capitalize, title — elementwise on StringArray.

use ferrum_core::dimension::Dimension;
use ferrum_core::error::FerrumResult;

use crate::string_array::StringArray;

/// Convert each string element to uppercase.
///
/// # Errors
/// Returns an error if the internal array construction fails.
///
/// # Examples
/// ```ignore
/// let a = strings::array(&["hello", "world"]).unwrap();
/// let b = strings::upper(&a).unwrap();
/// assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
/// ```
pub fn upper<D: Dimension>(a: &StringArray<D>) -> FerrumResult<StringArray<D>> {
    a.map(|s| s.to_uppercase())
}

/// Convert each string element to lowercase.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn lower<D: Dimension>(a: &StringArray<D>) -> FerrumResult<StringArray<D>> {
    a.map(|s| s.to_lowercase())
}

/// Capitalize each string element (first character uppercase, rest lowercase).
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn capitalize<D: Dimension>(a: &StringArray<D>) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let mut chars = s.chars();
        match chars.next() {
            None => String::new(),
            Some(first) => {
                let upper: String = first.to_uppercase().collect();
                let rest: String = chars.as_str().to_lowercase();
                format!("{upper}{rest}")
            }
        }
    })
}

/// Title-case each string element (first letter of each word uppercase,
/// rest lowercase).
///
/// Words are separated by whitespace.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn title<D: Dimension>(a: &StringArray<D>) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let mut result = String::with_capacity(s.len());
        let mut capitalize_next = true;
        for ch in s.chars() {
            if ch.is_whitespace() {
                result.push(ch);
                capitalize_next = true;
            } else if capitalize_next {
                for upper in ch.to_uppercase() {
                    result.push(upper);
                }
                capitalize_next = false;
            } else {
                for lower in ch.to_lowercase() {
                    result.push(lower);
                }
            }
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_upper() {
        let a = array(&["hello", "world"]).unwrap();
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &["HELLO", "WORLD"]);
    }

    #[test]
    fn test_lower() {
        let a = array(&["HELLO", "World"]).unwrap();
        let b = lower(&a).unwrap();
        assert_eq!(b.as_slice(), &["hello", "world"]);
    }

    #[test]
    fn test_capitalize() {
        let a = array(&["hello world", "fOO BAR", ""]).unwrap();
        let b = capitalize(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello world", "Foo bar", ""]);
    }

    #[test]
    fn test_title() {
        let a = array(&["hello world", "foo bar baz"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello World", "Foo Bar Baz"]);
    }

    #[test]
    fn test_title_mixed_case() {
        let a = array(&["hELLO wORLD"]).unwrap();
        let b = title(&a).unwrap();
        assert_eq!(b.as_slice(), &["Hello World"]);
    }

    #[test]
    fn test_upper_empty() {
        let a = array(&[""]).unwrap();
        let b = upper(&a).unwrap();
        assert_eq!(b.as_slice(), &[""]);
    }
}
