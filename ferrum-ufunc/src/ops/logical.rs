// ferrum-ufunc: Logical functions
//
// logical_and, logical_or, logical_xor, logical_not, all, any

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;

use crate::helpers::{binary_map_op, unary_map_op};

/// Trait for types that can be interpreted as boolean for logical ops.
pub trait Logical {
    /// Return true if the value is "truthy" (nonzero).
    fn is_truthy(&self) -> bool;
}

impl Logical for bool {
    #[inline]
    fn is_truthy(&self) -> bool {
        *self
    }
}

macro_rules! impl_logical_numeric {
    ($($ty:ty),*) => {
        $(
            impl Logical for $ty {
                #[inline]
                fn is_truthy(&self) -> bool {
                    *self != 0 as $ty
                }
            }
        )*
    };
}

impl_logical_numeric!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

impl Logical for f32 {
    #[inline]
    fn is_truthy(&self) -> bool {
        *self != 0.0
    }
}

impl Logical for f64 {
    #[inline]
    fn is_truthy(&self) -> bool {
        *self != 0.0
    }
}

impl Logical for num_complex::Complex<f32> {
    #[inline]
    fn is_truthy(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

impl Logical for num_complex::Complex<f64> {
    #[inline]
    fn is_truthy(&self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}

/// Elementwise logical AND.
pub fn logical_and<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
) -> FerrumResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() && y.is_truthy())
}

/// Elementwise logical OR.
pub fn logical_or<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
) -> FerrumResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() || y.is_truthy())
}

/// Elementwise logical XOR.
pub fn logical_xor<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
) -> FerrumResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() ^ y.is_truthy())
}

/// Elementwise logical NOT.
pub fn logical_not<T, D>(input: &Array<T, D>) -> FerrumResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    unary_map_op(input, |x| !x.is_truthy())
}

/// Test whether all elements are truthy.
pub fn all<T, D>(input: &Array<T, D>) -> bool
where
    T: Element + Logical,
    D: Dimension,
{
    input.iter().all(|x| x.is_truthy())
}

/// Test whether any element is truthy.
pub fn any<T, D>(input: &Array<T, D>) -> bool
where
    T: Element + Logical,
    D: Dimension,
{
    input.iter().any(|x| x.is_truthy())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::Ix1;

    fn arr1_bool(data: Vec<bool>) -> Array<bool, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_logical_and() {
        let a = arr1_bool(vec![true, true, false, false]);
        let b = arr1_bool(vec![true, false, true, false]);
        let r = logical_and(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn test_logical_or() {
        let a = arr1_bool(vec![true, true, false, false]);
        let b = arr1_bool(vec![true, false, true, false]);
        let r = logical_or(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, true, true, false]);
    }

    #[test]
    fn test_logical_xor() {
        let a = arr1_bool(vec![true, true, false, false]);
        let b = arr1_bool(vec![true, false, true, false]);
        let r = logical_xor(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, true, false]);
    }

    #[test]
    fn test_logical_not() {
        let a = arr1_bool(vec![true, false, true]);
        let r = logical_not(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_logical_and_numeric() {
        let a = arr1_i32(vec![1, 1, 0, 0]);
        let b = arr1_i32(vec![1, 0, 1, 0]);
        let r = logical_and(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false, false]);
    }

    #[test]
    fn test_all() {
        let a = arr1_bool(vec![true, true, true]);
        assert!(all(&a));
        let b = arr1_bool(vec![true, false, true]);
        assert!(!all(&b));
    }

    #[test]
    fn test_any() {
        let a = arr1_bool(vec![false, false, true]);
        assert!(any(&a));
        let b = arr1_bool(vec![false, false, false]);
        assert!(!any(&b));
    }

    #[test]
    fn test_all_numeric() {
        let a = arr1_i32(vec![1, 2, 3]);
        assert!(all(&a));
        let b = arr1_i32(vec![1, 0, 3]);
        assert!(!all(&b));
    }
}
