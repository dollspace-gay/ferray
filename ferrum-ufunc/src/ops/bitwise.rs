// ferrum-ufunc: Bitwise functions
//
// bitwise_and, bitwise_or, bitwise_xor, bitwise_not, invert,
// left_shift, right_shift

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;

use crate::helpers::{binary_float_op, unary_float_op};

/// Trait for types that support bitwise operations.
pub trait BitwiseOps:
    std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitXor<Output = Self>
    + std::ops::Not<Output = Self>
    + Copy
{
}

/// Trait for types that support shift operations in addition to bitwise ops.
pub trait ShiftOps: BitwiseOps + std::ops::Shl<u32, Output = Self> + std::ops::Shr<u32, Output = Self> {}

macro_rules! impl_bitwise_ops {
    ($($ty:ty),*) => {
        $(impl BitwiseOps for $ty {})*
    };
}

macro_rules! impl_shift_ops {
    ($($ty:ty),*) => {
        $(impl ShiftOps for $ty {})*
    };
}

impl_bitwise_ops!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128, bool);
impl_shift_ops!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// Elementwise bitwise AND.
pub fn bitwise_and<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x & y)
}

/// Elementwise bitwise OR.
pub fn bitwise_or<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x | y)
}

/// Elementwise bitwise XOR.
pub fn bitwise_xor<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x ^ y)
}

/// Elementwise bitwise NOT.
pub fn bitwise_not<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    unary_float_op(input, |x| !x)
}

/// Alias for [`bitwise_not`].
pub fn invert<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    bitwise_not(input)
}

/// Elementwise left shift.
///
/// Each element of `a` is shifted left by the corresponding element of `b`.
pub fn left_shift<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(ferrum_core::error::FerrumError::shape_mismatch(format!(
            "left_shift: shapes {:?} and {:?} do not match",
            a.shape(), b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &s)| x << s).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Elementwise right shift.
///
/// Each element of `a` is shifted right by the corresponding element of `b`.
pub fn right_shift<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(ferrum_core::error::FerrumError::shape_mismatch(format!(
            "right_shift: shapes {:?} and {:?} do not match",
            a.shape(), b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &s)| x >> s).collect();
    Array::from_vec(a.dim().clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::Ix1;

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_u32(data: Vec<u32>) -> Array<u32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_u8(data: Vec<u8>) -> Array<u8, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_bitwise_and() {
        let a = arr1_i32(vec![0b1100, 0b1010]);
        let b = arr1_i32(vec![0b1010, 0b1010]);
        let r = bitwise_and(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1000, 0b1010]);
    }

    #[test]
    fn test_bitwise_or() {
        let a = arr1_i32(vec![0b1100, 0b1010]);
        let b = arr1_i32(vec![0b1010, 0b0101]);
        let r = bitwise_or(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1110, 0b1111]);
    }

    #[test]
    fn test_bitwise_xor() {
        let a = arr1_i32(vec![0b1100, 0b1010]);
        let b = arr1_i32(vec![0b1010, 0b1010]);
        let r = bitwise_xor(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b0110, 0b0000]);
    }

    #[test]
    fn test_bitwise_not() {
        let a = arr1_u8(vec![0b0000_1111]);
        let r = bitwise_not(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1111_0000]);
    }

    #[test]
    fn test_invert() {
        let a = arr1_u8(vec![0b0000_1111]);
        let r = invert(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1111_0000]);
    }

    #[test]
    fn test_left_shift() {
        let a = arr1_i32(vec![1, 2, 4]);
        let s = arr1_u32(vec![1, 2, 3]);
        let r = left_shift(&a, &s).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2, 8, 32]);
    }

    #[test]
    fn test_right_shift() {
        let a = arr1_i32(vec![8, 16, 32]);
        let s = arr1_u32(vec![1, 2, 3]);
        let r = right_shift(&a, &s).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4, 4, 4]);
    }
}
