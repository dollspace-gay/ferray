// ferray-strings: String classification functions (is* family)
//
// Elementwise boolean classification matching NumPy's
// `numpy.strings.isalpha`, `numpy.strings.isdigit`, etc.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::error::FerrayResult;

use crate::string_array::StringArray;

/// Classify each element by applying `pred` to the string content.
fn classify<D: Dimension>(
    a: &StringArray<D>,
    pred: impl Fn(&str) -> bool,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().map(|s| pred(s)).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Return `true` where every character is alphabetic and the string is
/// non-empty. Matches `numpy.strings.isalpha`.
pub fn isalpha<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(|c| c.is_alphabetic()))
}

/// Return `true` where every character is a digit and the string is
/// non-empty. Matches `numpy.strings.isdigit`.
pub fn isdigit<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
}

/// Return `true` where every character is whitespace and the string is
/// non-empty. Matches `numpy.strings.isspace`.
pub fn isspace<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(|c| c.is_whitespace()))
}

/// Return `true` where every character is uppercase and the string is
/// non-empty. Matches `numpy.strings.isupper`.
pub fn isupper<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        !s.is_empty() && s.chars().any(char::is_uppercase) && !s.chars().any(char::is_lowercase)
    })
}

/// Return `true` where every character is lowercase and the string is
/// non-empty. Matches `numpy.strings.islower`.
pub fn islower<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        !s.is_empty() && s.chars().any(char::is_lowercase) && !s.chars().any(char::is_uppercase)
    })
}

/// Return `true` where every character is alphanumeric and the string
/// is non-empty. Matches `numpy.strings.isalnum`.
pub fn isalnum<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()))
}

/// Return `true` where the string could be a valid numeric literal.
/// Matches `numpy.strings.isnumeric` (simplified — checks for
/// ASCII digit + decimal point + sign characters).
pub fn isnumeric<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '+' || c == '-')
    })
}

/// Return `true` where the string is titlecased (first letter of each
/// word is uppercase, the rest lowercase). Matches
/// `numpy.strings.istitle`.
pub fn istitle<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        if s.is_empty() {
            return false;
        }
        let mut expect_upper = true;
        for c in s.chars() {
            if c.is_alphabetic() {
                if expect_upper && !c.is_uppercase() {
                    return false;
                }
                if !expect_upper && !c.is_lowercase() {
                    return false;
                }
                expect_upper = false;
            } else {
                expect_upper = true;
            }
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_isalpha() {
        let a = array(&["hello", "hello123", "", "HELLO"]).unwrap();
        let r = isalpha(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, true]);
    }

    #[test]
    fn test_isdigit() {
        let a = array(&["123", "12.3", "", "abc"]).unwrap();
        let r = isdigit(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn test_isspace() {
        let a = array(&["  ", "\t\n", "", "a b"]).unwrap();
        let r = isspace(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, true, false, false]);
    }

    #[test]
    fn test_isupper() {
        let a = array(&["HELLO", "Hello", "hello", ""]).unwrap();
        let r = isupper(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn test_islower() {
        let a = array(&["hello", "Hello", "HELLO", ""]).unwrap();
        let r = islower(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn test_isalnum() {
        let a = array(&["abc123", "abc 123", "", "abc"]).unwrap();
        let r = isalnum(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, true]);
    }

    #[test]
    fn test_istitle() {
        let a = array(&["Hello World", "hello world", "HELLO WORLD", ""]).unwrap();
        let r = istitle(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }
}
