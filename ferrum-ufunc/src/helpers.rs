// ferrum-ufunc: Common helper functions for ufunc implementations
//
// Provides generic unary/binary operation wrappers that handle contiguous
// vs non-contiguous arrays, SIMD dispatch, and broadcasting.

use ferrum_core::dimension::{Dimension, IxDyn};
use ferrum_core::dimension::broadcast::broadcast_shapes;
use ferrum_core::dtype::Element;
use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::Array;

/// Apply a unary function elementwise, preserving dimension.
/// Works for any `T: Element + Float` (or any Copy type with the given fn).
#[inline]
pub fn unary_float_op<T, D>(
    input: &Array<T, D>,
    f: impl Fn(T) -> T,
) -> FerrumResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    let data: Vec<T> = input.iter().map(|&x| f(x)).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a unary function that maps T -> U, preserving dimension.
#[inline]
pub fn unary_map_op<T, U, D>(
    input: &Array<T, D>,
    f: impl Fn(T) -> U,
) -> FerrumResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    let data: Vec<U> = input.iter().map(|&x| f(x)).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a binary function elementwise on same-shape arrays.
#[inline]
pub fn binary_float_op<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    f: impl Fn(T, T) -> T,
) -> FerrumResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrumError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Apply a binary function that maps (T, T) -> U, preserving dimension.
#[inline]
pub fn binary_map_op<T, U, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    f: impl Fn(T, T) -> U,
) -> FerrumResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrumError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<U> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Apply a binary function with broadcasting support.
/// Returns an IxDyn array with the broadcast shape.
pub fn binary_broadcast_op<T, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
    f: impl Fn(T, T) -> T,
) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D1: Dimension,
    D2: Dimension,
{
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_view = a.broadcast_to(&shape)?;
    let b_view = b.broadcast_to(&shape)?;
    let data: Vec<T> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Array::from_vec(IxDyn::from(&shape[..]), data)
}

/// Apply a binary function with broadcasting, mapping to output type U.
pub fn binary_broadcast_map_op<T, U, D1, D2>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
    f: impl Fn(T, T) -> U,
) -> FerrumResult<Array<U, IxDyn>>
where
    T: Element + Copy,
    U: Element,
    D1: Dimension,
    D2: Dimension,
{
    let shape = broadcast_shapes(a.shape(), b.shape())?;
    let a_view = a.broadcast_to(&shape)?;
    let b_view = b.broadcast_to(&shape)?;
    let data: Vec<U> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Array::from_vec(IxDyn::from(&shape[..]), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::{Ix1, Ix2};

    #[test]
    fn unary_op_works() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 4.0, 9.0]).unwrap();
        let r = unary_float_op(&a, f64::sqrt).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn binary_op_same_shape() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![4.0, 5.0, 6.0]).unwrap();
        let r = binary_float_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn binary_op_shape_mismatch() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![4.0, 5.0]).unwrap();
        assert!(binary_float_op(&a, &b, |x, y| x + y).is_err());
    }

    #[test]
    fn binary_broadcast_works() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![1.0, 2.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let r = binary_broadcast_op(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        let s: Vec<f64> = r.iter().copied().collect();
        assert_eq!(s, vec![11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }
}
