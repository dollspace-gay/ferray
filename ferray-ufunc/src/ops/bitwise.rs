// ferray-ufunc: Bitwise functions
//
// bitwise_and, bitwise_or, bitwise_xor, bitwise_not, invert,
// left_shift, right_shift

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

use crate::helpers::{binary_float_op, binary_mixed_op, unary_float_op};

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
pub trait ShiftOps:
    BitwiseOps + std::ops::Shl<u32, Output = Self> + std::ops::Shr<u32, Output = Self>
{
}

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
pub fn bitwise_and<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x & y)
}

/// Elementwise bitwise OR.
pub fn bitwise_or<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x | y)
}

/// Elementwise bitwise XOR.
pub fn bitwise_xor<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| x ^ y)
}

/// Elementwise bitwise NOT.
pub fn bitwise_not<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    unary_float_op(input, |x| !x)
}

/// Alias for [`bitwise_not`].
pub fn invert<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    bitwise_not(input)
}

/// Elementwise left shift with NumPy broadcasting.
///
/// Each element of `a` is shifted left by the corresponding element of `b`.
pub fn left_shift<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    binary_mixed_op(a, b, |x, s| x << s)
}

/// Elementwise right shift with NumPy broadcasting.
///
/// Each element of `a` is shifted right by the corresponding element of `b`.
pub fn right_shift<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    binary_mixed_op(a, b, |x, s| x >> s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

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

    // -----------------------------------------------------------------------
    // Broadcasting tests for bitwise ops (issue #379)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitwise_and_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![0xFF, 0x0F]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 2]), vec![0x0F, 0xF0]).unwrap();
        let r = bitwise_and(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // 0xFF & 0x0F = 0x0F, 0xFF & 0xF0 = 0xF0, 0x0F & 0x0F = 0x0F, 0x0F & 0xF0 = 0x00
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![0x0F, 0xF0, 0x0F, 0x00]
        );
    }

    #[test]
    fn test_bitwise_or_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<u8, Ix2>::from_vec(Ix2::new([2, 1]), vec![0b1010, 0b0101]).unwrap();
        let b = Array::<u8, Ix2>::from_vec(Ix2::new([1, 2]), vec![0b0011, 0b1100]).unwrap();
        let r = bitwise_or(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![0b1011, 0b1110, 0b0111, 0b1101]
        );
    }

    #[test]
    fn test_left_shift_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![1, 2]).unwrap();
        let s = Array::<u32, Ix2>::from_vec(Ix2::new([1, 3]), vec![0, 1, 2]).unwrap();
        let r = left_shift(&a, &s).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        // 1 << {0,1,2} = {1,2,4}, 2 << {0,1,2} = {2,4,8}
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![1, 2, 4, 2, 4, 8]);
    }

    #[test]
    fn test_right_shift_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 1]), vec![16, 64]).unwrap();
        let s = Array::<u32, Ix2>::from_vec(Ix2::new([1, 3]), vec![0, 1, 2]).unwrap();
        let r = right_shift(&a, &s).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![16, 8, 4, 64, 32, 16]
        );
    }
}
