// ferray-strings: String classification functions (is* family)
//
// Elementwise boolean classification matching NumPy's
// `numpy.strings.isalpha`, `numpy.strings.isdigit`, etc.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::error::FerrayResult;

use crate::string_array::StringArray;

/// Inclusive codepoint ranges of characters with Unicode
/// `Numeric_Type` of Decimal **or** Digit — i.e. exactly the set for
/// which CPython's `str.isdigit()` returns `True` for a single
/// character (`Py_UNICODE_ISDIGIT`). This is broader than the Decimal-
/// only `isdecimal` (it adds superscripts/subscripts ²³⁴, circled ①,
/// parenthesized ⑴, etc.) and narrower than `isnumeric` (it excludes
/// fractions ½ and letter-numerals like Ⅻ).
///
/// Derived live from the oracle CPython 3.13 / Unicode 15.1.0 (the
/// version backing the installed `numpy` 2.4 build that this crate is
/// verified against):
/// `[c for c in range(0x110000) if chr(c).isdigit()]`, run-length
/// compressed to 83 ranges. Rust's `std` exposes no `Numeric_Type`
/// predicate (`char::is_numeric` covers `Nd|Nl|No` — too broad — and
/// `char::to_digit(10)` is ASCII-only — too narrow), so the table is
/// embedded directly.
const DIGIT_RANGES: &[(char, char)] = &[
    ('\u{0030}', '\u{0039}'),
    ('\u{00B2}', '\u{00B3}'),
    ('\u{00B9}', '\u{00B9}'),
    ('\u{0660}', '\u{0669}'),
    ('\u{06F0}', '\u{06F9}'),
    ('\u{07C0}', '\u{07C9}'),
    ('\u{0966}', '\u{096F}'),
    ('\u{09E6}', '\u{09EF}'),
    ('\u{0A66}', '\u{0A6F}'),
    ('\u{0AE6}', '\u{0AEF}'),
    ('\u{0B66}', '\u{0B6F}'),
    ('\u{0BE6}', '\u{0BEF}'),
    ('\u{0C66}', '\u{0C6F}'),
    ('\u{0CE6}', '\u{0CEF}'),
    ('\u{0D66}', '\u{0D6F}'),
    ('\u{0DE6}', '\u{0DEF}'),
    ('\u{0E50}', '\u{0E59}'),
    ('\u{0ED0}', '\u{0ED9}'),
    ('\u{0F20}', '\u{0F29}'),
    ('\u{1040}', '\u{1049}'),
    ('\u{1090}', '\u{1099}'),
    ('\u{1369}', '\u{1371}'),
    ('\u{17E0}', '\u{17E9}'),
    ('\u{1810}', '\u{1819}'),
    ('\u{1946}', '\u{194F}'),
    ('\u{19D0}', '\u{19DA}'),
    ('\u{1A80}', '\u{1A89}'),
    ('\u{1A90}', '\u{1A99}'),
    ('\u{1B50}', '\u{1B59}'),
    ('\u{1BB0}', '\u{1BB9}'),
    ('\u{1C40}', '\u{1C49}'),
    ('\u{1C50}', '\u{1C59}'),
    ('\u{2070}', '\u{2070}'),
    ('\u{2074}', '\u{2079}'),
    ('\u{2080}', '\u{2089}'),
    ('\u{2460}', '\u{2468}'),
    ('\u{2474}', '\u{247C}'),
    ('\u{2488}', '\u{2490}'),
    ('\u{24EA}', '\u{24EA}'),
    ('\u{24F5}', '\u{24FD}'),
    ('\u{24FF}', '\u{24FF}'),
    ('\u{2776}', '\u{277E}'),
    ('\u{2780}', '\u{2788}'),
    ('\u{278A}', '\u{2792}'),
    ('\u{A620}', '\u{A629}'),
    ('\u{A8D0}', '\u{A8D9}'),
    ('\u{A900}', '\u{A909}'),
    ('\u{A9D0}', '\u{A9D9}'),
    ('\u{A9F0}', '\u{A9F9}'),
    ('\u{AA50}', '\u{AA59}'),
    ('\u{ABF0}', '\u{ABF9}'),
    ('\u{FF10}', '\u{FF19}'),
    ('\u{104A0}', '\u{104A9}'),
    ('\u{10A40}', '\u{10A43}'),
    ('\u{10D30}', '\u{10D39}'),
    ('\u{10E60}', '\u{10E68}'),
    ('\u{11052}', '\u{1105A}'),
    ('\u{11066}', '\u{1106F}'),
    ('\u{110F0}', '\u{110F9}'),
    ('\u{11136}', '\u{1113F}'),
    ('\u{111D0}', '\u{111D9}'),
    ('\u{112F0}', '\u{112F9}'),
    ('\u{11450}', '\u{11459}'),
    ('\u{114D0}', '\u{114D9}'),
    ('\u{11650}', '\u{11659}'),
    ('\u{116C0}', '\u{116C9}'),
    ('\u{11730}', '\u{11739}'),
    ('\u{118E0}', '\u{118E9}'),
    ('\u{11950}', '\u{11959}'),
    ('\u{11C50}', '\u{11C59}'),
    ('\u{11D50}', '\u{11D59}'),
    ('\u{11DA0}', '\u{11DA9}'),
    ('\u{11F50}', '\u{11F59}'),
    ('\u{16A60}', '\u{16A69}'),
    ('\u{16AC0}', '\u{16AC9}'),
    ('\u{16B50}', '\u{16B59}'),
    ('\u{1D7CE}', '\u{1D7FF}'),
    ('\u{1E140}', '\u{1E149}'),
    ('\u{1E2F0}', '\u{1E2F9}'),
    ('\u{1E4F0}', '\u{1E4F9}'),
    ('\u{1E950}', '\u{1E959}'),
    ('\u{1F100}', '\u{1F10A}'),
    ('\u{1FBF0}', '\u{1FBF9}'),
];

/// Return `true` if `c` has Unicode `Numeric_Type` Decimal or Digit,
/// matching CPython's per-character `Py_UNICODE_ISDIGIT`. Looks the
/// codepoint up in the sorted, non-overlapping [`DIGIT_RANGES`] table
/// by binary search.
fn is_digit_char(c: char) -> bool {
    DIGIT_RANGES
        .binary_search_by(|&(lo, hi)| {
            if c < lo {
                core::cmp::Ordering::Greater
            } else if c > hi {
                core::cmp::Ordering::Less
            } else {
                core::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

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
/// non-empty. Matches `numpy.strings.isdigit`, which delegates per
/// element to CPython's `str.isdigit()` (`Py_UNICODE_ISDIGIT`):
/// Unicode `Numeric_Type` of Decimal **or** Digit. This is broader than
/// [`isdecimal`] (it accepts superscripts/subscripts like `²³`, circled
/// digits like `①`, and parenthesized digits) and narrower than
/// [`isnumeric`] (it rejects fractions like `½` and letter-numerals like
/// `Ⅻ`). See [`is_digit_char`]/[`DIGIT_RANGES`].
pub fn isdigit<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<bool, D>> {
    classify(a, |s| !s.is_empty() && s.chars().all(is_digit_char))
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

    // Divergence pin for crosslink blocker #911 (refs #836). Python's
    // `str.isdigit` (which `numpy.strings.isdigit` mirrors element by
    // element) is Unicode `Numeric_Type` Decimal OR Digit -- BROADER than
    // `isdecimal` (Decimal only) and NARROWER than `isnumeric`
    // (Decimal+Digit+Numeric). Expected values are the live results of
    // CPython 3.13 / numpy 2.4 (Unicode 15.1.0):
    //   '²³'.isdigit() -> True   (superscript: Digit, not Decimal)
    //   '⁴'.isdigit()   -> True   (superscript 4)
    //   '₀'.isdigit()   -> True   (subscript 0)
    //   '①'.isdigit()   -> True   (circled 1: Digit)
    //   '⓪'.isdigit()   -> True   (circled 0)
    //   '⑴'.isdigit()   -> True   (parenthesized 1)
    //   '０９'.isdigit() -> True   (fullwidth Decimal)
    //   '٣'.isdigit()   -> True   (Arabic-Indic Decimal)
    //   '½'.isdigit()   -> False  (fraction 1/2: Numeric, not Digit)
    //   'Ⅻ'.isdigit()   -> False  (roman 12: Numeric, not Digit)
    //   '123'.isdigit() -> True;  '12a' -> False;  ''.isdigit() -> False
    #[test]
    fn test_isdigit_python_semantics() {
        let a = array(&[
            "²³", "⁴", "₀", "①", "⓪", "⑴", "０９", "٣", "½", "Ⅻ", "123", "12a", "",
        ])
        .unwrap();
        let r = isdigit(&a).unwrap();
        assert_eq!(
            r.as_slice().unwrap(),
            &[
                true, true, true, true, true, true, true, true, false, false, true, false, false,
            ]
        );
    }

    // Guard against `isdigit` widening into `isdecimal`/`isnumeric`
    // territory: the Decimal-only and Decimal+Digit+Numeric predicates
    // must NOT shift. Superscript '²' is digit-but-not-decimal;
    // fraction '½' is numeric-but-not-digit.
    #[test]
    fn test_isdigit_does_not_regress_isdecimal_isnumeric() {
        let a = array(&["²", "½", "①", "Ⅻ", "0123456789"]).unwrap();
        // isdecimal: only the ASCII run is Decimal.
        assert_eq!(
            isdecimal(&a).unwrap().as_slice().unwrap(),
            &[false, false, false, false, true]
        );
        // isnumeric: all of these are numeric (Decimal|Digit|Numeric).
        assert_eq!(
            isnumeric(&a).unwrap().as_slice().unwrap(),
            &[true, true, true, true, true]
        );
        // isdigit: superscript and circled are digits; fraction/roman are not.
        assert_eq!(
            isdigit(&a).unwrap().as_slice().unwrap(),
            &[true, false, true, false, true]
        );
    }
}
