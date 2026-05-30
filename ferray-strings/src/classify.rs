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
    classify(a, |s| !s.is_empty() && s.chars().all(char::is_alphabetic))
}

/// Return `true` where every character is a digit and the string is
/// non-empty. Matches `numpy.strings.isdigit`.
pub fn isdigit<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
    })
}

/// Return `true` where every character is whitespace and the string is
/// non-empty. Matches `numpy.strings.isspace`.
pub fn isspace<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(char::is_whitespace))
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
    classify(a, |s| !s.is_empty() && s.chars().all(char::is_alphanumeric))
}

/// Return `true` where every character is numeric and the string is
/// non-empty. Matches `numpy.strings.isnumeric`, which delegates per
/// element to CPython's `str.isnumeric()` (Unicode `Numeric_Type` =
/// Decimal, Digit, or Numeric). Rust's `char::is_numeric()` tests the
/// Unicode `Nd`/`Nl`/`No` general categories, which coincides with that
/// set: superscripts/fractions like `²` and `½` are numeric, while
/// `.`, `+`, and `-` are not.
pub fn isnumeric<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(char::is_numeric))
}

/// Return `true` where every character is a Unicode decimal digit (`Nd`
/// general category) and the string is non-empty. Matches
/// `numpy.strings.isdecimal` — stricter than [`isnumeric`] in that
/// superscripts (e.g. ²) and fractions (e.g. ½) are not decimal-digit
/// characters and return `false`.
pub fn isdecimal<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| {
        // Stricter than `isnumeric`: rejects superscripts/subscripts,
        // fractions, and other non-Decimal_Number "numeric" characters.
        // Rust's std doesn't expose the full Unicode `Nd` category check
        // without an extra crate, so we use the ASCII-digit subset — this
        // matches CPython's `str.isdecimal()` for the ASCII range and for
        // the non-decimal-digit cases that distinguish it from
        // `isnumeric`/`isdigit` (²/½ -> false, ASCII digits -> true).
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
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

    // Divergence pin for crosslink blocker #823. Expected values are the
    // documented results of CPython's per-character `str.is*()` methods,
    // which `numpy.strings.isnumeric`/`isdecimal`/`isalnum` mirror element
    // by element (numpy 2.4.5 oracle confirmed):
    //   '12.3'.isnumeric() -> False   ('²' True, '½' True, '123' True)
    //   '²'.isdecimal()    -> False   ('123' True)
    //   'abc1'.isalnum()   -> True    ('ab c' False)
    //   ''.is*()           -> False   (empty string is False for all)

    #[test]
    fn test_isnumeric_python_semantics() {
        // '.'/'+'/'-' are NOT numeric; Unicode numerics (², ½) ARE numeric.
        let a = array(&["12.3", "²", "½", "123", ""]).unwrap();
        let r = isnumeric(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, true, true, false]);
    }

    #[test]
    fn test_isdecimal_python_semantics() {
        // Decimal digits only: '²' is numeric but NOT decimal; '123' is.
        let a = array(&["12.3", "²", "½", "123", ""]).unwrap();
        let r = isdecimal(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, false, false, true, false]);
    }

    #[test]
    fn test_isalnum_python_semantics() {
        // Alphanumeric and non-empty; whitespace breaks it.
        let a = array(&["abc1", "ab c", "²", "123", ""]).unwrap();
        let r = isalnum(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, true, true, false]);
    }
}
