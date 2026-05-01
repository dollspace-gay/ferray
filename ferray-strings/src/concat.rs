// ferray-strings: Concatenation and repetition operations (REQ-3, REQ-4)
//
// Implements add (elementwise concat with broadcasting) and multiply (repeat).

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::FerrayResult;

use crate::string_array::{StringArray, broadcast_binary};

/// Elementwise string concatenation with broadcasting.
///
/// Concatenates corresponding elements of `a` and `b`. If shapes differ,
/// NumPy-style broadcasting is applied (e.g., a scalar string is broadcast
/// against an array).
///
/// The result is always a dynamic-rank `StringArray<IxDyn>`.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if shapes are incompatible.
pub fn add<Da: Dimension, Db: Dimension>(
    a: &StringArray<Da>,
    b: &StringArray<Db>,
) -> FerrayResult<StringArray<IxDyn>> {
    let (out_shape, pairs) = broadcast_binary(a, b)?;
    let a_data = a.as_slice();
    let b_data = b.as_slice();

    let data: Vec<String> = pairs
        .map(|(ia, ib)| format!("{}{}", a_data[ia], b_data[ib]))
        .collect();

    StringArray::from_vec(IxDyn::new(&out_shape), data)
}

/// Same-dimension elementwise string concatenation.
///
/// Like [`add`] but both inputs must have the same shape — no
/// broadcasting is performed, and the result preserves the static
/// dimension type. Use this when you know the shapes match and want
/// to keep `StringArray<Ix1>` instead of getting `StringArray<IxDyn>`
/// (#163).
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn add_same<D: Dimension>(
    a: &StringArray<D>,
    b: &StringArray<D>,
) -> FerrayResult<StringArray<D>> {
    if a.shape() != b.shape() {
        return Err(ferray_core::error::FerrayError::shape_mismatch(format!(
            "add_same: shapes {:?} and {:?} must be identical",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<String> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| format!("{x}{y}"))
        .collect();
    StringArray::from_vec(a.dim().clone(), data)
}

/// Repeat each string element `n` times.
///
/// # Errors
/// Returns an error if the internal array construction fails.
pub fn multiply<D: Dimension>(a: &StringArray<D>, n: usize) -> FerrayResult<StringArray<D>> {
    a.map(|s| s.repeat(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_array::array;

    #[test]
    fn test_add_same_shape() {
        let a = array(&["hello", "foo"]).unwrap();
        let b = array(&[" world", " bar"]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &["hello world", "foo bar"]);
    }

    #[test]
    fn test_add_broadcast_scalar() {
        // AC-2: strings::add broadcasts a scalar string against an array correctly
        let a = array(&["hello", "world"]).unwrap();
        let b = array(&["!"]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &["hello!", "world!"]);
    }

    #[test]
    fn test_add_broadcast_scalar_left() {
        let a = array(&[">> "]).unwrap();
        let b = array(&["hello", "world"]).unwrap();
        let c = add(&a, &b).unwrap();
        assert_eq!(c.as_slice(), &[">> hello", ">> world"]);
    }

    #[test]
    fn test_add_incompatible_shapes() {
        let a = array(&["a", "b", "c"]).unwrap();
        let b = array(&["x", "y"]).unwrap();
        assert!(add(&a, &b).is_err());
    }

    #[test]
    fn test_multiply() {
        let a = array(&["ab", "cd"]).unwrap();
        let b = multiply(&a, 3).unwrap();
        assert_eq!(b.as_slice(), &["ababab", "cdcdcd"]);
    }

    #[test]
    fn test_multiply_zero() {
        let a = array(&["hello"]).unwrap();
        let b = multiply(&a, 0).unwrap();
        assert_eq!(b.as_slice(), &[""]);
    }

    #[test]
    fn test_multiply_one() {
        let a = array(&["hello"]).unwrap();
        let b = multiply(&a, 1).unwrap();
        assert_eq!(b.as_slice(), &["hello"]);
    }
}
