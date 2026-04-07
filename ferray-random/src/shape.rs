// ferray-random: IntoShape trait for flexible shape arguments
//
// Mirrors NumPy's convention where `size=` accepts either an int or a
// tuple of ints. Distribution methods take `shape: impl IntoShape` so
// callers can write:
//
//     rng.random(10)           // 1-D, length 10
//     rng.random([3, 4])       // 2-D, shape (3, 4)
//     rng.random(&[2, 3, 4])   // 3-D, shape (2, 3, 4)
//
// See: https://github.com/dollspace-gay/ferray/issues/440

use ferray_core::FerrayError;

/// Convert a size argument into a concrete shape vector.
///
/// Implemented for:
/// - `usize` — 1-D shape `[n]`
/// - `&[usize]` — arbitrary-rank shape
/// - `[usize; N]` (any N, via const generics)
/// - `&[usize; N]`
/// - `Vec<usize>`
///
/// A zero-element shape (e.g. `&[]`) is rejected: distributions need at
/// least one axis. A shape containing a zero axis (e.g. `[0, 4]`) is also
/// rejected, matching the original 1-D `size == 0` check.
pub trait IntoShape {
    /// Consume `self` and return the shape as a `Vec<usize>`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the resulting shape is empty
    /// or contains a zero-sized axis.
    fn into_shape(self) -> Result<Vec<usize>, FerrayError>;
}

fn validate_shape(shape: Vec<usize>) -> Result<Vec<usize>, FerrayError> {
    if shape.is_empty() {
        return Err(FerrayError::invalid_value(
            "shape must have at least one axis",
        ));
    }
    if shape.contains(&0) {
        return Err(FerrayError::invalid_value(format!(
            "shape must not contain zero-sized axes, got {shape:?}"
        )));
    }
    Ok(shape)
}

impl IntoShape for usize {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(vec![self])
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(self.to_vec())
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(self.to_vec())
    }
}

impl<const N: usize> IntoShape for &[usize; N] {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(self.to_vec())
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(self)
    }
}

impl IntoShape for &Vec<usize> {
    fn into_shape(self) -> Result<Vec<usize>, FerrayError> {
        validate_shape(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_from_usize() {
        assert_eq!(10usize.into_shape().unwrap(), vec![10]);
    }

    #[test]
    fn shape_from_slice() {
        let s: &[usize] = &[3, 4];
        assert_eq!(s.into_shape().unwrap(), vec![3, 4]);
    }

    #[test]
    fn shape_from_array() {
        assert_eq!([2, 3, 4].into_shape().unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn shape_from_array_ref() {
        let a = [2, 3, 4];
        assert_eq!((&a).into_shape().unwrap(), vec![2, 3, 4]);
    }

    #[test]
    fn shape_from_vec() {
        assert_eq!(vec![2, 3].into_shape().unwrap(), vec![2, 3]);
    }

    #[test]
    fn empty_shape_rejected() {
        let s: &[usize] = &[];
        assert!(s.into_shape().is_err());
    }

    #[test]
    fn zero_axis_rejected() {
        assert!(0usize.into_shape().is_err());
        assert!([3, 0, 4].into_shape().is_err());
    }
}
