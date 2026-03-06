// ferrum-strings: Alignment and padding operations (REQ-6)
//
// Implements center, ljust, rjust, zfill — elementwise on StringArray.

use ferrum_core::dimension::Dimension;
use ferrum_core::error::FerrumResult;

use crate::string_array::StringArray;

/// Center each string in a field of the given width, padded with `fillchar`.
///
/// If the string is already longer than `width`, it is returned unchanged.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn center<D: Dimension>(
    a: &StringArray<D>,
    width: usize,
    fillchar: char,
) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let char_count = s.chars().count();
        if char_count >= width {
            return s.to_string();
        }
        let total_pad = width - char_count;
        let left_pad = total_pad / 2;
        let right_pad = total_pad - left_pad;
        let mut result = String::with_capacity(s.len() + total_pad);
        for _ in 0..left_pad {
            result.push(fillchar);
        }
        result.push_str(s);
        for _ in 0..right_pad {
            result.push(fillchar);
        }
        result
    })
}

/// Left-justify each string in a field of the given width, padded with spaces.
///
/// If the string is already longer than `width`, it is returned unchanged.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn ljust<D: Dimension>(a: &StringArray<D>, width: usize) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let char_count = s.chars().count();
        if char_count >= width {
            return s.to_string();
        }
        let pad = width - char_count;
        let mut result = String::with_capacity(s.len() + pad);
        result.push_str(s);
        for _ in 0..pad {
            result.push(' ');
        }
        result
    })
}

/// Right-justify each string in a field of the given width, padded with spaces.
///
/// If the string is already longer than `width`, it is returned unchanged.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn rjust<D: Dimension>(a: &StringArray<D>, width: usize) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let char_count = s.chars().count();
        if char_count >= width {
            return s.to_string();
        }
        let pad = width - char_count;
        let mut result = String::with_capacity(s.len() + pad);
        for _ in 0..pad {
            result.push(' ');
        }
        result.push_str(s);
        result
    })
}

/// Pad each string on the left with zeros to fill the given width.
///
/// If the string starts with a sign (`+` or `-`), the sign is placed
/// before the zeros. If the string is already longer than `width`,
/// it is returned unchanged.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn zfill<D: Dimension>(a: &StringArray<D>, width: usize) -> FerrumResult<StringArray<D>> {
    a.map(|s| {
        let char_count = s.chars().count();
        if char_count >= width {
            return s.to_string();
        }
        let pad = width - char_count;
        let (sign, rest) = if s.starts_with('+') || s.starts_with('-') {
            (&s[..1], &s[1..])
        } else {
            ("", s)
        };
        let mut result = String::with_capacity(s.len() + pad);
        result.push_str(sign);
        for _ in 0..pad {
            result.push('0');
        }
        result.push_str(rest);
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_center() {
        let a = array(&["hi", "x"]).unwrap();
        let b = center(&a, 6, '*').unwrap();
        assert_eq!(b.as_slice(), &["**hi**", "**x***"]);
    }

    #[test]
    fn test_center_no_pad_needed() {
        let a = array(&["hello"]).unwrap();
        let b = center(&a, 3, ' ').unwrap();
        assert_eq!(b.as_slice(), &["hello"]);
    }

    #[test]
    fn test_ljust() {
        let a = array(&["hi", "hello"]).unwrap();
        let b = ljust(&a, 6).unwrap();
        assert_eq!(b.as_slice(), &["hi    ", "hello "]);
    }

    #[test]
    fn test_ljust_no_pad_needed() {
        let a = array(&["hello"]).unwrap();
        let b = ljust(&a, 3).unwrap();
        assert_eq!(b.as_slice(), &["hello"]);
    }

    #[test]
    fn test_rjust() {
        let a = array(&["hi", "hello"]).unwrap();
        let b = rjust(&a, 6).unwrap();
        assert_eq!(b.as_slice(), &["    hi", " hello"]);
    }

    #[test]
    fn test_zfill() {
        let a = array(&["42", "-17", "+5", "abc"]).unwrap();
        let b = zfill(&a, 5).unwrap();
        assert_eq!(b.as_slice(), &["00042", "-0017", "+0005", "00abc"]);
    }

    #[test]
    fn test_zfill_no_pad_needed() {
        let a = array(&["12345"]).unwrap();
        let b = zfill(&a, 3).unwrap();
        assert_eq!(b.as_slice(), &["12345"]);
    }
}
