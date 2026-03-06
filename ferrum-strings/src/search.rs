// ferrum-strings: Search operations (REQ-8, REQ-9, REQ-10)
//
// Implements find, count, startswith, endswith, replace — elementwise on StringArray.

use ferrum_core::Array;
use ferrum_core::dimension::{Dimension, Ix1};
use ferrum_core::error::FerrumResult;

use crate::string_array::StringArray;

/// Find the lowest index of `sub` in each string element.
///
/// Returns an `Array<i64>` where each element is the index of the first
/// occurrence of `sub`, or -1 if not found.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn find<D: Dimension>(a: &StringArray<D>, sub: &str) -> FerrumResult<Array<i64, Ix1>> {
    let data: Vec<i64> = a.map_to_vec(|s| {
        match s.find(sub) {
            Some(byte_idx) => {
                // Convert byte index to character index
                s[..byte_idx].chars().count() as i64
            }
            None => -1,
        }
    });
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Count non-overlapping occurrences of `sub` in each string element.
///
/// Returns an `Array<u64>` with the count for each element.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn count<D: Dimension>(a: &StringArray<D>, sub: &str) -> FerrumResult<Array<u64, Ix1>> {
    let data: Vec<u64> = a.map_to_vec(|s| s.matches(sub).count() as u64);
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Test whether each string element starts with the given prefix.
///
/// Returns an `Array<bool>` indicating the result for each element.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn startswith<D: Dimension>(
    a: &StringArray<D>,
    prefix: &str,
) -> FerrumResult<Array<bool, Ix1>> {
    let data: Vec<bool> = a.map_to_vec(|s| s.starts_with(prefix));
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Test whether each string element ends with the given suffix.
///
/// Returns an `Array<bool>` indicating the result for each element.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn endswith<D: Dimension>(a: &StringArray<D>, suffix: &str) -> FerrumResult<Array<bool, Ix1>> {
    let data: Vec<bool> = a.map_to_vec(|s| s.ends_with(suffix));
    let dim = Ix1::new([data.len()]);
    Array::from_vec(dim, data)
}

/// Replace occurrences of `old` with `new` in each string element.
///
/// If `max_count` is `Some(n)`, only the first `n` occurrences are replaced.
/// If `None`, all occurrences are replaced.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn replace<D: Dimension>(
    a: &StringArray<D>,
    old: &str,
    new: &str,
    max_count: Option<usize>,
) -> FerrumResult<StringArray<D>> {
    a.map(|s| match max_count {
        None => s.replace(old, new),
        Some(n) => s.replacen(old, new, n),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_find() {
        let a = array(&["hello", "world", "bell"]).unwrap();
        let b = find(&a, "ll").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[2, -1, 2]);
    }

    #[test]
    fn test_find_at_start() {
        let a = array(&["abc", "def"]).unwrap();
        let b = find(&a, "abc").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[0, -1]);
    }

    #[test]
    fn test_find_empty_sub() {
        let a = array(&["hello"]).unwrap();
        let b = find(&a, "").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[0]);
    }

    #[test]
    fn test_count() {
        let a = array(&["abcabc", "abc", "xyz"]).unwrap();
        let b = count(&a, "abc").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[2_u64, 1, 0]);
    }

    #[test]
    fn test_startswith() {
        let a = array(&["hello", "world", "help"]).unwrap();
        let b = startswith(&a, "hel").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[true, false, true]);
    }

    #[test]
    fn test_endswith() {
        let a = array(&["hello", "world", "bello"]).unwrap();
        let b = endswith(&a, "llo").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[true, false, true]);
    }

    #[test]
    fn test_replace_all() {
        let a = array(&["aabbcc", "aabba"]).unwrap();
        let b = replace(&a, "aa", "XX", None).unwrap();
        assert_eq!(b.as_slice(), &["XXbbcc", "XXbba"]);
    }

    #[test]
    fn test_replace_with_count() {
        let a = array(&["ababab"]).unwrap();
        let b = replace(&a, "ab", "X", Some(2)).unwrap();
        assert_eq!(b.as_slice(), &["XXab"]);
    }

    #[test]
    fn test_find_ac3() {
        // AC-3: strings::find(&a, "ll") returns correct indices (2 for "hello", -1 for "world")
        let a = array(&["hello", "world"]).unwrap();
        let b = find(&a, "ll").unwrap();
        let data = b.as_slice().unwrap();
        assert_eq!(data, &[2, -1]);
    }
}
