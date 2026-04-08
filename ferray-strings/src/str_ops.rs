// ferray-strings: Miscellaneous string operations
//
// str_len, swapcase, and elementwise comparison functions.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::error::FerrayResult;

use crate::string_array::StringArray;

/// Return the length of each string element. Matches
/// `numpy.strings.str_len` (#518).
pub fn str_len<D: Dimension>(a: &StringArray<D>) -> FerrayResult<Array<u64, D>> {
    let data: Vec<u64> = a.iter().map(|s| s.len() as u64).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Swap the case of each character in every element. Matches
/// `numpy.strings.swapcase` (#515).
pub fn swapcase<D: Dimension>(a: &StringArray<D>) -> FerrayResult<StringArray<D>> {
    a.map(|s| {
        s.chars()
            .map(|c| {
                if c.is_uppercase() {
                    c.to_lowercase().collect::<String>()
                } else if c.is_lowercase() {
                    c.to_uppercase().collect::<String>()
                } else {
                    c.to_string()
                }
            })
            .collect()
    })
}

// ---------------------------------------------------------------------------
// Elementwise string comparison (#516)
// ---------------------------------------------------------------------------

/// Elementwise string equality. Both arrays must have the same shape.
pub fn equal<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x == y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise string inequality.
pub fn not_equal<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x != y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise lexicographic less-than.
pub fn less<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x < y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise lexicographic greater-than.
pub fn greater<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x > y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise lexicographic less-or-equal.
pub fn less_equal<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x <= y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise lexicographic greater-or-equal.
pub fn greater_equal<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<Array<bool, D>> {
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| x >= y).collect();
    Array::from_vec(a.dim().clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_str_len() {
        let a = array(&["hello", "", "abc", "hi"]).unwrap();
        let r = str_len(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5, 0, 3, 2]);
    }

    #[test]
    fn test_swapcase() {
        let a = array(&["Hello World", "ABC", "abc", "123"]).unwrap();
        let r = swapcase(&a).unwrap();
        assert_eq!(
            r.as_slice(),
            &["hELLO wORLD", "abc", "ABC", "123"]
        );
    }

    #[test]
    fn test_equal() {
        let a = array(&["abc", "def", "ghi"]).unwrap();
        let b = array(&["abc", "xyz", "ghi"]).unwrap();
        let r = equal(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, true]);
    }

    #[test]
    fn test_less() {
        let a = array(&["abc", "xyz"]).unwrap();
        let b = array(&["abd", "abc"]).unwrap();
        let r = less(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false]);
    }

    #[test]
    fn test_greater() {
        let a = array(&["xyz", "abc"]).unwrap();
        let b = array(&["abc", "xyz"]).unwrap();
        let r = greater(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false]);
    }
}
