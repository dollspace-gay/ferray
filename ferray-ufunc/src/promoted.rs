//! Mixed-type (promoted) variants of arithmetic ufuncs.
//!
//! NumPy's ufunc type resolver automatically promotes operands to a
//! common result type — `np.add(int32_arr, float64_arr)` just works.
//! ferray-core already carries the compile-time promotion machinery
//! (`Promoted` for result-type resolution, `PromoteTo` for element
//! casting), so this module is glue: cast each operand to the promoted
//! output type, then invoke the existing same-type ufunc.
//!
//! Scope: the four basic arithmetic ops (add / subtract / multiply /
//! divide) and elementwise max/min. These cover the vast majority of
//! NumPy calls with mixed dtypes. Unary ufuncs don't need promotion
//! because there's only one operand, and the float-only transcendentals
//! (sin, exp, log, …) can't meaningfully accept integer inputs without
//! first promoting them — callers that need that can `.cast::<f64>()`
//! explicitly.
//!
//! All helpers here take same-shape inputs (no broadcasting on the
//! promoted path — composing promotion with broadcasting is not hard
//! but is deferred until there's a concrete caller that wants it).

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::dtype::promotion::{Promoted, PromoteTo};
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

use crate::helpers::binary_elementwise_op;

/// Cast every element of `a` from `A` to the target type `Out`, producing
/// a fresh array. This is the tiny bridge that lets a mixed-type op
/// route both operands through a same-shape same-type kernel.
#[inline]
fn cast_array<A, Out, D>(a: &Array<A, D>) -> FerrayResult<Array<Out, D>>
where
    A: Element + Copy + PromoteTo<Out>,
    Out: Element + Copy,
    D: Dimension,
{
    if let Some(slice) = a.as_slice() {
        let data: Vec<Out> = slice.iter().map(|&x| x.promote()).collect();
        Array::from_vec(a.dim().clone(), data)
    } else {
        let data: Vec<Out> = a.iter().map(|&x| x.promote()).collect();
        Array::from_vec(a.dim().clone(), data)
    }
}

// ---------------------------------------------------------------------------
// Add / Subtract / Multiply / Divide
// ---------------------------------------------------------------------------

/// Elementwise addition with NumPy-style type promotion.
///
/// `add_promoted(Array<i32>, Array<f64>)` promotes the i32 to f64
/// (per the `Promoted` trait) and returns `Array<f64>`.
///
/// Both inputs must have the same shape. For broadcasting, cast
/// explicitly and use the existing [`crate::add`].
///
/// # Errors
/// Returns [`FerrayError::ShapeMismatch`] if shapes differ.
pub fn add_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + Float,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "add_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, |x, y| x + y)
}

/// Elementwise subtraction with NumPy-style type promotion.
pub fn subtract_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + Float,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "subtract_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, |x, y| x - y)
}

/// Elementwise multiplication with NumPy-style type promotion.
pub fn multiply_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + Float,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "multiply_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, |x, y| x * y)
}

/// Elementwise division with NumPy-style type promotion.
pub fn divide_promoted<A, B, D>(
    a: &Array<A, D>,
    b: &Array<B, D>,
) -> FerrayResult<Array<<A as Promoted<B>>::Output, D>>
where
    A: Element + Copy + Promoted<B> + PromoteTo<<A as Promoted<B>>::Output>,
    B: Element + Copy + PromoteTo<<A as Promoted<B>>::Output>,
    <A as Promoted<B>>::Output: Element + Copy + Float,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "divide_promoted: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let a_cast = cast_array::<A, <A as Promoted<B>>::Output, D>(a)?;
    let b_cast = cast_array::<B, <A as Promoted<B>>::Output, D>(b)?;
    binary_elementwise_op(&a_cast, &b_cast, |x, y| x / y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    #[test]
    fn add_i32_f64_promotes_to_f64() {
        // np.add(int32_arr, float64_arr) → float64
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1i32, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 1.5, 2.5]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        // Type-check: the return type is Array<f64, Ix1>
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[1.5, 3.5, 5.5]);
    }

    #[test]
    fn add_f32_f64_promotes_to_f64() {
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 0.5, 0.5]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn subtract_i16_f32_promotes_to_f32() {
        let a = Array::<i16, Ix1>::from_vec(Ix1::new([3]), vec![10i16, 20, 30]).unwrap();
        let b = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.5f32, 2.5, 3.5]).unwrap();
        let c = subtract_promoted(&a, &b).unwrap();
        let slice: &[f32] = c.as_slice().unwrap();
        assert_eq!(slice, &[8.5, 17.5, 26.5]);
    }

    #[test]
    fn multiply_u8_f64_promotes_to_f64() {
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([3]), vec![2u8, 3, 4]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.5, 0.5, 0.5]).unwrap();
        let c = multiply_promoted(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1.0, 1.5, 2.0]);
    }

    #[test]
    fn divide_f32_f64_promotes_to_f64() {
        let a = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![10.0f32, 20.0, 30.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, 4.0, 5.0]).unwrap();
        let c = divide_promoted(&a, &b).unwrap();
        let slice: &[f64] = c.as_slice().unwrap();
        assert_eq!(slice, &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn same_type_path_is_identity() {
        // f64 + f64 should still work (Promoted<f64>::Output = f64).
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![4.0, 5.0, 6.0]).unwrap();
        let c = add_promoted(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn promoted_2d_shape_preserved() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )
        .unwrap();
        let c = add_promoted(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(
            c.as_slice().unwrap(),
            &[1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        );
    }

    #[test]
    fn shape_mismatch_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1i32, 2, 3]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(add_promoted(&a, &b).is_err());
    }
}
