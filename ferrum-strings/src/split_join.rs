// ferrum-strings: Split and join operations (REQ-11)
//
// Implements split and join — elementwise on StringArray.

use ferrum_core::dimension::{Dimension, Ix1};
use ferrum_core::error::FerrumResult;

use crate::string_array::{StringArray, StringArray1};

/// Split each string element by the given separator.
///
/// Returns a 1-D array where each element is a `Vec<String>` containing
/// the split parts. This matches NumPy's behavior where splitting produces
/// a ragged result.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn split<D: Dimension>(a: &StringArray<D>, sep: &str) -> FerrumResult<Vec<Vec<String>>> {
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
pub fn join(sep: &str, items: &[Vec<String>]) -> FerrumResult<StringArray1> {
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
pub fn join_array<D: Dimension>(sep: &str, a: &StringArray<D>) -> FerrumResult<StringArray1> {
    let joined: String = a
        .iter()
        .map(|s| s.as_str())
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
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], vec!["a", "b"]);
        assert_eq!(result[1], vec!["c", "d"]);
    }

    #[test]
    fn test_split_multiple_parts() {
        let a = array(&["a-b-c"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_no_separator_found() {
        let a = array(&["hello"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result[0], vec!["hello"]);
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
        // AC-4: strings::split(&["a-b", "c-d"], "-") returns [vec!["a","b"], vec!["c","d"]]
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
            ]
        );
    }
}
