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
/// - `usize` — 1-D shape `[n]` (including `n == 0` for an empty array)
/// - `&[usize]` — arbitrary-rank shape
/// - `[usize; N]` (any N, via const generics)
/// - `&[usize; N]`
/// - `Vec<usize>`
///
/// Zero-axis shapes (e.g. `0usize`, `[3, 0, 4]`) are now permitted and
/// produce an empty array, matching `NumPy`'s `np.random.uniform(size=0)`
/// behaviour (#264, #455). The only rejected shape is a totally
/// rank-empty `&[]`, which would correspond to a 0-d scalar — that is
/// not yet wired through the distribution machinery.
pub trait IntoShape {
    /// Consume `self` and return the shape as a `Vec<usize>`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if the resulting shape has
    /// zero rank (i.e. an empty shape slice).
    fn into_shape(self) -> Result<Vec<usize>, FerrayError>;
}

fn validate_shape(shape: Vec<usize>) -> Result<Vec<usize>, FerrayError> {
    if shape.is_empty() {
        return Err(FerrayError::invalid_value(
            "shape must have at least one axis",
        ));
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
    fn zero_axis_allowed() {
        // NumPy allows size=0 / size=(3, 0, 4) and returns an empty
        // array. ferray now matches that (#264, #455).
        assert_eq!(0usize.into_shape().unwrap(), vec![0]);
        assert_eq!([3, 0, 4].into_shape().unwrap(), vec![3, 0, 4]);
    }
}
