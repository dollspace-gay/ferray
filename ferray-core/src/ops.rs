// ferray-core: Operator overloading for Array<T, D>
//
// Implements std::ops::{Add, Sub, Mul, Div, Rem, Neg} with
// Output = FerrayResult<Array<T, D>>.
//
// Users write `(a + b)?` to get the result, maintaining the zero-panic
// guarantee while enabling natural math syntax.
//
// These operate elementwise on same-shape arrays. For broadcasting,
// use the functions in ferray-ufunc (e.g. ferray::add(&a, &b)).
//
// See: https://github.com/dollspace-gay/ferray/issues/7

use crate::array::owned::Array;
use crate::dimension::Dimension;
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

/// Elementwise binary operation on two same-shape arrays.
///
/// Returns an error if the shapes don't match.
fn elementwise_binary<T, D, F>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    op: F,
    op_name: &str,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "operator {}: shapes {:?} and {:?} are not compatible",
            op_name,
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Implement a binary operator for all ownership combinations of Array.
///
/// Generates impls for:
///   &Array op &Array
///   Array  op Array
///   Array  op &Array
///   &Array op Array
macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $op_fn:expr, $op_name:expr) => {
        // &Array op &Array
        impl<T, D> std::ops::$trait<&Array<T, D>> for &Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: &Array<T, D>) -> Self::Output {
                elementwise_binary(self, rhs, $op_fn, $op_name)
            }
        }

        // Array op Array
        impl<T, D> std::ops::$trait<Array<T, D>> for Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: Array<T, D>) -> Self::Output {
                elementwise_binary(&self, &rhs, $op_fn, $op_name)
            }
        }

        // Array op &Array
        impl<T, D> std::ops::$trait<&Array<T, D>> for Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: &Array<T, D>) -> Self::Output {
                elementwise_binary(&self, rhs, $op_fn, $op_name)
            }
        }

        // &Array op Array
        impl<T, D> std::ops::$trait<Array<T, D>> for &Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: Array<T, D>) -> Self::Output {
                elementwise_binary(self, &rhs, $op_fn, $op_name)
            }
        }
    };
}

impl_binary_op!(Add, add, |a, b| a + b, "+");
impl_binary_op!(Sub, sub, |a, b| a - b, "-");
impl_binary_op!(Mul, mul, |a, b| a * b, "*");
impl_binary_op!(Div, div, |a, b| a / b, "/");
impl_binary_op!(Rem, rem, |a, b| a % b, "%");

// ---------------------------------------------------------------------------
// Scalar-array operations: Array op scalar and scalar op Array
// ---------------------------------------------------------------------------

/// Implement scalar-array binary operators (Array op T and &Array op T).
macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $op_fn:expr) => {
        // &Array op scalar
        impl<T, D> std::ops::$trait<T> for &Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: T) -> Self::Output {
                let data: Vec<T> = self.iter().map(|&x| $op_fn(x, rhs)).collect();
                Array::from_vec(self.dim().clone(), data)
            }
        }

        // Array op scalar
        impl<T, D> std::ops::$trait<T> for Array<T, D>
        where
            T: Element + Copy + std::ops::$trait<Output = T>,
            D: Dimension,
        {
            type Output = FerrayResult<Array<T, D>>;

            fn $method(self, rhs: T) -> Self::Output {
                (&self).$method(rhs)
            }
        }
    };
}

impl_scalar_op!(Add, add, |a, b| a + b);
impl_scalar_op!(Sub, sub, |a, b| a - b);
impl_scalar_op!(Mul, mul, |a, b| a * b);
impl_scalar_op!(Div, div, |a, b| a / b);
impl_scalar_op!(Rem, rem, |a, b| a % b);

// Unary negation: -&Array and -Array
impl<T, D> std::ops::Neg for &Array<T, D>
where
    T: Element + Copy + std::ops::Neg<Output = T>,
    D: Dimension,
{
    type Output = FerrayResult<Array<T, D>>;

    fn neg(self) -> Self::Output {
        let data: Vec<T> = self.iter().map(|&x| -x).collect();
        Array::from_vec(self.dim().clone(), data)
    }
}

impl<T, D> std::ops::Neg for Array<T, D>
where
    T: Element + Copy + std::ops::Neg<Output = T>,
    D: Dimension,
{
    type Output = FerrayResult<Array<T, D>>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::Ix1;

    fn arr(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_add_ref_ref() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![4.0, 5.0, 6.0]);
        let c = (&a + &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_owned_owned() {
        let a = arr(vec![1.0, 2.0]);
        let b = arr(vec![3.0, 4.0]);
        let c = (a + b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[4.0, 6.0]);
    }

    #[test]
    fn test_add_mixed() {
        let a = arr(vec![1.0, 2.0]);
        let b = arr(vec![3.0, 4.0]);
        let c = (a + &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[4.0, 6.0]);

        let d = arr(vec![10.0, 20.0]);
        let e = (&b + d).unwrap();
        assert_eq!(e.as_slice().unwrap(), &[13.0, 24.0]);
    }

    #[test]
    fn test_sub() {
        let a = arr(vec![5.0, 7.0]);
        let b = arr(vec![1.0, 2.0]);
        let c = (&a - &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn test_mul() {
        let a = arr(vec![2.0, 3.0]);
        let b = arr(vec![4.0, 5.0]);
        let c = (&a * &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[8.0, 15.0]);
    }

    #[test]
    fn test_div() {
        let a = arr(vec![10.0, 20.0]);
        let b = arr(vec![2.0, 5.0]);
        let c = (&a / &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 4.0]);
    }

    #[test]
    fn test_rem() {
        let a = arr_i32(vec![7, 10]);
        let b = arr_i32(vec![3, 4]);
        let c = (&a % &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1, 2]);
    }

    #[test]
    fn test_neg() {
        let a = arr(vec![1.0, -2.0, 3.0]);
        let b = (-&a).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_neg_owned() {
        let a = arr(vec![1.0, -2.0]);
        let b = (-a).unwrap();
        assert_eq!(b.as_slice().unwrap(), &[-1.0, 2.0]);
    }

    #[test]
    fn test_shape_mismatch_errors() {
        let a = arr(vec![1.0, 2.0]);
        let b = arr(vec![1.0, 2.0, 3.0]);
        let result = &a + &b;
        assert!(result.is_err());
    }

    // --- Scalar-array operations ---

    #[test]
    fn test_add_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let c = (&a + 10.0).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_sub_scalar() {
        let a = arr(vec![10.0, 20.0, 30.0]);
        let c = (&a - 5.0).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 15.0, 25.0]);
    }

    #[test]
    fn test_mul_scalar() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let c = (&a * 3.0).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[3.0, 6.0, 9.0]);
    }

    #[test]
    fn test_div_scalar() {
        let a = arr(vec![10.0, 20.0, 30.0]);
        let c = (&a / 10.0).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_rem_scalar() {
        let a = arr_i32(vec![7, 10, 15]);
        let c = (&a % 4).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[3, 2, 3]);
    }

    #[test]
    fn test_scalar_op_owned() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let c = (a + 10.0).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_chained_ops() {
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![4.0, 5.0, 6.0]);
        let c = arr(vec![10.0, 10.0, 10.0]);
        // (a + b)? * c)?
        let result = (&(&a + &b).unwrap() * &c).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[50.0, 70.0, 90.0]);
    }
}
