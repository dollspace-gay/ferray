// ferray-strings: Split and join operations (REQ-11)
//
// Implements split and join — elementwise on StringArray.

use ferray_core::dimension::{Dimension, Ix1, Ix2};
use ferray_core::error::FerrayResult;

use crate::string_array::{StringArray, StringArray1, StringArray2};

/// Split each string element by the given separator.
///
/// Returns a 2-D `StringArray` of shape `(n_inputs, max_parts)` where row
/// `i` contains the parts produced by splitting element `i`. Rows shorter
/// than `max_parts` are padded with empty strings (#277). Use
/// [`split_ragged`] when you need the unpadded `Vec<Vec<String>>` form.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn split<D: Dimension>(a: &StringArray<D>, sep: &str) -> FerrayResult<StringArray2> {
    let parts: Vec<Vec<String>> = a
        .iter()
        .map(|s| s.split(sep).map(String::from).collect())
        .collect();
    let n_inputs = parts.len();
    let max_parts = parts.iter().map(Vec::len).max().unwrap_or(0);
    let mut flat: Vec<String> = Vec::with_capacity(n_inputs * max_parts);
    for row in &parts {
        for j in 0..max_parts {
            flat.push(row.get(j).cloned().unwrap_or_default());
        }
    }
    StringArray2::from_vec(Ix2::new([n_inputs, max_parts]), flat)
}

/// Ragged-result variant of [`split`]: returns a `Vec<Vec<String>>` so
/// callers that need the unpadded splits per element don't have to
/// strip empty padding from the 2-D result (#277).
///
/// # Errors
/// Returns an error only for internal failures.
pub fn split_ragged<D: Dimension>(
    a: &StringArray<D>,
    sep: &str,
) -> FerrayResult<Vec<Vec<String>>> {
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
pub fn join(sep: &str, items: &[Vec<String>]) -> FerrayResult<StringArray1> {
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
pub fn join_array<D: Dimension>(sep: &str, a: &StringArray<D>) -> FerrayResult<StringArray1> {
    let joined: String = a
        .iter()
        .map(std::string::String::as_str)
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
        assert_eq!(result.shape(), &[2, 2]);
        let s = result.as_slice();
        assert_eq!(s, &["a", "b", "c", "d"]);
    }

    #[test]
    fn test_split_multiple_parts() {
        let a = array(&["a-b-c"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(result.as_slice(), &["a", "b", "c"]);
    }

    #[test]
    fn test_split_no_separator_found() {
        let a = array(&["hello"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        assert_eq!(result.as_slice(), &["hello"]);
    }

    #[test]
    fn test_split_pads_short_rows_with_empty_strings() {
        // #277: rows shorter than max_parts must be padded with "".
        let a = array(&["a-b", "x-y-z"]).unwrap();
        let result = split(&a, "-").unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        // Row 0: ["a", "b", ""] (padded), Row 1: ["x", "y", "z"]
        assert_eq!(result.as_slice(), &["a", "b", "", "x", "y", "z"]);
    }

    #[test]
    fn test_split_ragged_returns_unpadded() {
        // #277: split_ragged keeps the per-element variable length.
        let a = array(&["a-b", "x-y-z"]).unwrap();
        let result = split_ragged(&a, "-").unwrap();
        assert_eq!(result, vec![
            vec!["a".to_string(), "b".to_string()],
            vec!["x".to_string(), "y".to_string(), "z".to_string()],
        ]);
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
        // AC-4: strings::split_ragged(&["a-b", "c-d"], "-") returns
        // [vec!["a","b"], vec!["c","d"]] — the ragged form preserves
        // the original AC behavior.
        let a = array(&["a-b", "c-d"]).unwrap();
        let result = split_ragged(&a, "-").unwrap();
        assert_eq!(
            result,
            vec![
                vec!["a".to_string(), "b".to_string()],
                vec!["c".to_string(), "d".to_string()],
            ]
        );
    }
}
