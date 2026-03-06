// ferrum-ufunc: Operator-style convenience functions
//
// REQ-9: +, -, *, /, % on arrays — provided as functions since the orphan rule
// prevents implementing std::ops traits on ferrum_core::Array from this crate.
// The operator trait impls themselves should live in ferrum-core; these functions
// serve as the underlying implementations that ferrum-core can delegate to.
//
// REQ-13: &, |, ^, !, <<, >> on integer arrays — same approach.
//
// Users can call these directly: `ufunc::array_add(&a, &b)` etc.

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::ops::bitwise::{BitwiseOps, ShiftOps};

/// Array addition (delegates to `arithmetic::add`).
pub fn array_add<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    crate::ops::arithmetic::add(a, b)
}

/// Array subtraction (delegates to `arithmetic::subtract`).
pub fn array_sub<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + std::ops::Sub<Output = T> + Copy,
    D: Dimension,
{
    crate::ops::arithmetic::subtract(a, b)
}

/// Array multiplication (delegates to `arithmetic::multiply`).
pub fn array_mul<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    crate::ops::arithmetic::multiply(a, b)
}

/// Array division (delegates to `arithmetic::divide`).
pub fn array_div<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + std::ops::Div<Output = T> + Copy,
    D: Dimension,
{
    crate::ops::arithmetic::divide(a, b)
}

/// Array remainder (delegates to `arithmetic::remainder`).
pub fn array_rem<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ops::arithmetic::remainder(a, b)
}

/// Array negation (delegates to `arithmetic::negative`).
pub fn array_neg<T, D>(a: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    crate::ops::arithmetic::negative(a)
}

/// Array bitwise AND (delegates to `bitwise::bitwise_and`).
pub fn array_bitand<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    crate::ops::bitwise::bitwise_and(a, b)
}

/// Array bitwise OR (delegates to `bitwise::bitwise_or`).
pub fn array_bitor<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    crate::ops::bitwise::bitwise_or(a, b)
}

/// Array bitwise XOR (delegates to `bitwise::bitwise_xor`).
pub fn array_bitxor<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    crate::ops::bitwise::bitwise_xor(a, b)
}

/// Array bitwise NOT (delegates to `bitwise::bitwise_not`).
pub fn array_bitnot<T, D>(a: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + BitwiseOps,
    D: Dimension,
{
    crate::ops::bitwise::bitwise_not(a)
}

/// Array left shift (delegates to `bitwise::left_shift`).
pub fn array_shl<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    crate::ops::bitwise::left_shift(a, b)
}

/// Array right shift (delegates to `bitwise::right_shift`).
pub fn array_shr<T, D>(a: &Array<T, D>, b: &Array<u32, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + ShiftOps,
    D: Dimension,
{
    crate::ops::bitwise::right_shift(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::Ix1;

    fn arr1_f64(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_u32(data: Vec<u32>) -> Array<u32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    // AC-8: Operator functions produce identical results to ufunc functions

    #[test]
    fn test_array_add() {
        let a = arr1_f64(vec![1.0, 2.0, 3.0]);
        let b = arr1_f64(vec![4.0, 5.0, 6.0]);
        let r = array_add(&a, &b).unwrap();
        let r2 = crate::ops::arithmetic::add(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), r2.as_slice().unwrap());
    }

    #[test]
    fn test_array_sub() {
        let a = arr1_f64(vec![5.0, 7.0, 9.0]);
        let b = arr1_f64(vec![1.0, 2.0, 3.0]);
        let r = array_sub(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_array_mul() {
        let a = arr1_f64(vec![2.0, 3.0]);
        let b = arr1_f64(vec![4.0, 5.0]);
        let r = array_mul(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[8.0, 15.0]);
    }

    #[test]
    fn test_array_div() {
        let a = arr1_f64(vec![10.0, 20.0]);
        let b = arr1_f64(vec![2.0, 5.0]);
        let r = array_div(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 4.0]);
    }

    #[test]
    fn test_array_rem() {
        let a = arr1_f64(vec![7.0, 10.0]);
        let b = arr1_f64(vec![3.0, 4.0]);
        let r = array_rem(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_array_neg() {
        let a = arr1_f64(vec![1.0, -2.0, 3.0]);
        let r = array_neg(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_array_bitand() {
        let a = arr1_i32(vec![0b1100, 0b1010]);
        let b = arr1_i32(vec![0b1010, 0b1010]);
        let r = array_bitand(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1000, 0b1010]);
    }

    #[test]
    fn test_array_bitor() {
        let a = arr1_i32(vec![0b1100, 0b1010]);
        let b = arr1_i32(vec![0b1010, 0b0101]);
        let r = array_bitor(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1110, 0b1111]);
    }

    #[test]
    fn test_array_bitxor() {
        let a = arr1_i32(vec![0b1100]);
        let b = arr1_i32(vec![0b1010]);
        let r = array_bitxor(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b0110]);
    }

    #[test]
    fn test_array_bitnot() {
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([1]), vec![0b0000_1111]).unwrap();
        let r = array_bitnot(&a).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[0b1111_0000]);
    }

    #[test]
    fn test_array_shl() {
        let a = arr1_i32(vec![1, 2, 4]);
        let s = arr1_u32(vec![1, 2, 3]);
        let r = array_shl(&a, &s).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[2, 8, 32]);
    }

    #[test]
    fn test_array_shr() {
        let a = arr1_i32(vec![8, 16, 32]);
        let s = arr1_u32(vec![1, 2, 3]);
        let r = array_shr(&a, &s).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[4, 4, 4]);
    }
}
