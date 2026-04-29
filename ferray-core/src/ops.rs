// ferray-core: Operator overloading for Array<T, D>
//
// Implements std::ops::{Add, Sub, Mul, Div, Rem, Neg} with
// Output = FerrayResult<Array<T, D>>.
//
// Users write `(a + b)?` to get the result, maintaining the zero-panic
// guarantee while enabling natural math syntax.
//
// Broadcasting: when both operands share the same dimension type `D`,
// the operators broadcast along compatible axes (NumPy rules). Cross-rank
// broadcasting (e.g. Ix1 + Ix2) is exposed via `add_broadcast` etc., which
// return `Array<T, IxDyn>` because the result rank cannot be expressed in
// the type system without specialization.
//
// See: https://github.com/dollspace-gay/ferray/issues/7
// Broadcasting: https://github.com/dollspace-gay/ferray/issues/346

use crate::array::owned::Array;
use crate::dimension::Dimension;
use crate::dimension::IxDyn;
use crate::dimension::broadcast::{broadcast_shapes, broadcast_to};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

/// Elementwise binary operation on two same-D arrays, with `NumPy` broadcasting.
///
/// - If shapes match exactly, takes the fast path (zip iter, no broadcast).
/// - Otherwise, broadcasts both inputs to the common shape and applies `op`.
///
/// Both inputs share dimension type `D`, so the broadcast result also has
/// rank `D::NDIM` (or, for `IxDyn`, the maximum of the two ranks). The result
/// is reconstructed via [`Dimension::from_dim_slice`].
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the shapes are not broadcast-compatible.
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
    // Fast path: identical shapes — no broadcasting needed.
    if a.shape() == b.shape() {
        let data: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| op(x, y)).collect();
        return Array::from_vec(a.dim().clone(), data);
    }

    // Broadcasting path.
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "operator {}: shapes {:?} and {:?} are not broadcast-compatible",
            op_name,
            a.shape(),
            b.shape()
        ))
    })?;

    let a_view = broadcast_to(a, &target_shape)?;
    let b_view = broadcast_to(b, &target_shape)?;

    let data: Vec<T> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();

    let result_dim = D::from_dim_slice(&target_shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!(
            "operator {op_name}: cannot represent broadcast result shape {target_shape:?} as the input dimension type"
        ))
    })?;

    Array::from_vec(result_dim, data)
}

/// Cross-rank broadcasting helper: apply a binary op to two arrays with
/// possibly different dimension types, returning a dynamic-rank result.
///
/// This is the primitive behind [`Array::add_broadcast`], [`Array::sub_broadcast`],
/// etc. Always returns `Array<T, IxDyn>` because the result rank depends
/// on input shapes at runtime.
fn elementwise_binary_dyn<T, D1, D2, F>(
    a: &Array<T, D1>,
    b: &Array<T, D2>,
    op: F,
    op_name: &str,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D1: Dimension,
    D2: Dimension,
    F: Fn(T, T) -> T,
{
    let target_shape = broadcast_shapes(a.shape(), b.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "{}: shapes {:?} and {:?} are not broadcast-compatible",
            op_name,
            a.shape(),
            b.shape()
        ))
    })?;

    let a_view = broadcast_to(a, &target_shape)?;
    let b_view = broadcast_to(b, &target_shape)?;

    let data: Vec<T> = a_view
        .iter()
        .zip(b_view.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();

    Array::from_vec(IxDyn::from(&target_shape[..]), data)
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
    type Output = FerrayResult<Self>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

// ---------------------------------------------------------------------------
// Cross-rank broadcasting methods
//
// These methods accept an operand of any dimension type and return a
// dynamic-rank result. They handle the case where the std::ops operators
// can't apply because the type system requires both operands to share `D`.
//
// Example: a (Ix1, shape (3,)) + b (Ix2, shape (2,1)) -> Array<T, IxDyn> shape (2,3)
// ---------------------------------------------------------------------------

impl<T, D> Array<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Elementwise add with `NumPy` broadcasting across arbitrary ranks.
    ///
    /// Returns a dynamic-rank `Array<T, IxDyn>` so that mixed-rank inputs
    /// (e.g. 1D + 2D) can produce a result whose rank is determined at
    /// runtime. For same-rank inputs prefer the `+` operator.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if shapes are not broadcast-compatible.
    pub fn add_broadcast<D2: Dimension>(
        &self,
        other: &Array<T, D2>,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        T: std::ops::Add<Output = T>,
    {
        elementwise_binary_dyn(self, other, |x, y| x + y, "add_broadcast")
    }

    /// Elementwise subtract with `NumPy` broadcasting across arbitrary ranks.
    ///
    /// See [`Array::add_broadcast`] for details.
    pub fn sub_broadcast<D2: Dimension>(
        &self,
        other: &Array<T, D2>,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        T: std::ops::Sub<Output = T>,
    {
        elementwise_binary_dyn(self, other, |x, y| x - y, "sub_broadcast")
    }

    /// Elementwise multiply with `NumPy` broadcasting across arbitrary ranks.
    ///
    /// See [`Array::add_broadcast`] for details.
    pub fn mul_broadcast<D2: Dimension>(
        &self,
        other: &Array<T, D2>,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        T: std::ops::Mul<Output = T>,
    {
        elementwise_binary_dyn(self, other, |x, y| x * y, "mul_broadcast")
    }

    /// Elementwise divide with `NumPy` broadcasting across arbitrary ranks.
    ///
    /// See [`Array::add_broadcast`] for details.
    pub fn div_broadcast<D2: Dimension>(
        &self,
        other: &Array<T, D2>,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        T: std::ops::Div<Output = T>,
    {
        elementwise_binary_dyn(self, other, |x, y| x / y, "div_broadcast")
    }

    /// Elementwise remainder with `NumPy` broadcasting across arbitrary ranks.
    ///
    /// See [`Array::add_broadcast`] for details.
    pub fn rem_broadcast<D2: Dimension>(
        &self,
        other: &Array<T, D2>,
    ) -> FerrayResult<Array<T, IxDyn>>
    where
        T: std::ops::Rem<Output = T>,
    {
        elementwise_binary_dyn(self, other, |x, y| x % y, "rem_broadcast")
    }
}

// ---------------------------------------------------------------------------
// In-place operators (#348)
//
// NumPy supports `arr += 5`, `arr *= other` — mutation without allocation.
// We split these into two groups:
//
// 1. Scalar in-place: `std::ops::*Assign<T> for Array<T, D>`. Always safe
//    (scalar ops never fail), implemented via `mapv_inplace`.
//
// 2. Array-array in-place: inherent fallible methods `add_inplace`,
//    `sub_inplace`, `mul_inplace`, `div_inplace`, `rem_inplace` returning
//    `FerrayResult<()>`. We cannot implement `std::ops::AddAssign<&Array>`
//    because the trait signature returns `()`, leaving no channel for
//    shape-mismatch errors — incompatible with ferray's zero-panic rule.
//    Use the `*_inplace` methods instead of `arr += other`.
//
//    The RHS is broadcast to `self.shape()` (NumPy semantics for in-place
//    arithmetic: the destination shape is fixed and the source must be
//    broadcastable to it).
// ---------------------------------------------------------------------------

/// Implement a scalar `*Assign` operator trait using `mapv_inplace`.
macro_rules! impl_scalar_op_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T, D> std::ops::$trait<T> for Array<T, D>
        where
            T: Element + Copy + std::ops::$trait,
            D: Dimension,
        {
            fn $method(&mut self, rhs: T) {
                self.mapv_inplace(|mut x| {
                    x $op rhs;
                    x
                });
            }
        }
    };
}

impl_scalar_op_assign!(AddAssign, add_assign, +=);
impl_scalar_op_assign!(SubAssign, sub_assign, -=);
impl_scalar_op_assign!(MulAssign, mul_assign, *=);
impl_scalar_op_assign!(DivAssign, div_assign, /=);
impl_scalar_op_assign!(RemAssign, rem_assign, %=);

/// Shared implementation for in-place array-array ops with broadcasting.
///
/// Fast path on identical shapes uses `zip_mut_with`. Otherwise the RHS is
/// broadcast into `self.shape()` via [`broadcast_to`] and then zipped with
/// `self.iter_mut()` in logical order.
fn inplace_binary<T, D, F>(
    lhs: &mut Array<T, D>,
    rhs: &Array<T, D>,
    op: F,
    op_name: &str,
) -> FerrayResult<()>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    // Fast path: identical shapes — delegate to zip_mut_with.
    if lhs.shape() == rhs.shape() {
        return lhs.zip_mut_with(rhs, |a, b| *a = op(*a, *b));
    }

    // Broadcasting path: rhs must broadcast into lhs.shape() (the destination
    // shape is fixed for in-place operations). Any shape change on the LHS
    // would require a reallocation, defeating the purpose of `*_inplace`.
    let target_shape: Vec<usize> = lhs.shape().to_vec();
    let rhs_view = broadcast_to(rhs, &target_shape).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "{}: shape {:?} cannot be broadcast into destination shape {:?}",
            op_name,
            rhs.shape(),
            target_shape
        ))
    })?;

    for (a, b) in lhs.iter_mut().zip(rhs_view.iter()) {
        *a = op(*a, *b);
    }
    Ok(())
}

impl<T, D> Array<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// In-place elementwise add: `self[i] += other[i]`. `other` is broadcast
    /// into `self.shape()`; the destination shape never changes.
    ///
    /// Prefer this over `self = (&self + &other)?` for large arrays — it
    /// avoids allocating a new result buffer.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if `other.shape()` cannot be
    /// broadcast into `self.shape()`.
    pub fn add_inplace(&mut self, other: &Self) -> FerrayResult<()>
    where
        T: std::ops::Add<Output = T>,
    {
        inplace_binary(self, other, |a, b| a + b, "add_inplace")
    }

    /// In-place elementwise subtract. See [`Array::add_inplace`].
    pub fn sub_inplace(&mut self, other: &Self) -> FerrayResult<()>
    where
        T: std::ops::Sub<Output = T>,
    {
        inplace_binary(self, other, |a, b| a - b, "sub_inplace")
    }

    /// In-place elementwise multiply. See [`Array::add_inplace`].
    pub fn mul_inplace(&mut self, other: &Self) -> FerrayResult<()>
    where
        T: std::ops::Mul<Output = T>,
    {
        inplace_binary(self, other, |a, b| a * b, "mul_inplace")
    }

    /// In-place elementwise divide. See [`Array::add_inplace`].
    pub fn div_inplace(&mut self, other: &Self) -> FerrayResult<()>
    where
        T: std::ops::Div<Output = T>,
    {
        inplace_binary(self, other, |a, b| a / b, "div_inplace")
    }

    /// In-place elementwise remainder. See [`Array::add_inplace`].
    pub fn rem_inplace(&mut self, other: &Self) -> FerrayResult<()>
    where
        T: std::ops::Rem<Output = T>,
    {
        inplace_binary(self, other, |a, b| a % b, "rem_inplace")
    }
}

// ---------------------------------------------------------------------------
// copyto: broadcasted elementwise assignment (#352)
//
// NumPy parity for `np.copyto(dst, src)`: copies values from `src` into `dst`,
// broadcasting `src` into `dst.shape()`. The destination shape is fixed — a
// copyto never reallocates `dst`. `casting=` is not modeled because ferray's
// type system already enforces matching element types; use `astype()` before
// copyto if conversion is desired.
//
// Kept separate from the in-place arithmetic ops because copyto is the
// "fundamental building block" called out in the issue: it has to handle
// cross-rank src/dst pairs (e.g. `Ix1` into `Ix2` via broadcasting), whereas
// `*_inplace` stay within one dimension type `D` to match std::ops::*Assign.
// ---------------------------------------------------------------------------

/// Copy values from `src` into `dst`, broadcasting `src` to `dst.shape()`.
///
/// Equivalent to `np.copyto(dst, src)` without the `casting=` or `where=`
/// parameters. `src` may have any dimension type — it only needs to be
/// broadcast-compatible with `dst.shape()`. The destination shape is fixed,
/// so `src` can never grow the destination.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if `src.shape()` cannot be broadcast
/// into `dst.shape()`. On error `dst` is left untouched.
pub fn copyto<T, D1, D2>(dst: &mut Array<T, D1>, src: &Array<T, D2>) -> FerrayResult<()>
where
    T: Element,
    D1: Dimension,
    D2: Dimension,
{
    // Fast path: identical shapes — straight iter-zip, no broadcast machinery.
    if dst.shape() == src.shape() {
        for (d, s) in dst.iter_mut().zip(src.iter()) {
            *d = s.clone();
        }
        return Ok(());
    }

    // Broadcasting path: src must broadcast into dst.shape(). We validate up
    // front so that a shape error leaves dst completely untouched.
    let target_shape: Vec<usize> = dst.shape().to_vec();
    let src_view = broadcast_to(src, &target_shape).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "copyto: source shape {:?} cannot be broadcast into destination shape {:?}",
            src.shape(),
            target_shape
        ))
    })?;

    for (d, s) in dst.iter_mut().zip(src_view.iter()) {
        *d = s.clone();
    }
    Ok(())
}

/// Copy values from `src` into `dst` where `mask` is `true`, broadcasting
/// both `src` and `mask` to `dst.shape()`.
///
/// Equivalent to `np.copyto(dst, src, where=mask)`. This is the generalized
/// form of `NumPy`'s `where=` ufunc parameter (#353): positions where the mask
/// is `false` are left untouched in `dst`. All three shapes must be
/// broadcast-compatible with `dst.shape()`; the destination shape is fixed,
/// so neither `src` nor `mask` can grow `dst`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if either `src.shape()` or
/// `mask.shape()` cannot be broadcast into `dst.shape()`. On error `dst` is
/// left untouched (broadcast validation happens before any writes).
pub fn copyto_where<T, D1, D2, D3>(
    dst: &mut Array<T, D1>,
    src: &Array<T, D2>,
    mask: &Array<bool, D3>,
) -> FerrayResult<()>
where
    T: Element,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    // Validate + materialize broadcast views BEFORE writing, so that a shape
    // mismatch can't leave dst in a partially-updated state.
    let target_shape: Vec<usize> = dst.shape().to_vec();

    let src_view = broadcast_to(src, &target_shape).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "copyto_where: source shape {:?} cannot be broadcast into destination shape {:?}",
            src.shape(),
            target_shape
        ))
    })?;

    let mask_view = broadcast_to(mask, &target_shape).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "copyto_where: mask shape {:?} cannot be broadcast into destination shape {:?}",
            mask.shape(),
            target_shape
        ))
    })?;

    for ((d, s), &m) in dst.iter_mut().zip(src_view.iter()).zip(mask_view.iter()) {
        if m {
            *d = s.clone();
        }
    }
    Ok(())
}

impl<T, D> Array<T, D>
where
    T: Element,
    D: Dimension,
{
    /// Copy values from `src` into `self`, broadcasting `src` to `self.shape()`.
    ///
    /// See [`copyto`] for the free-function form and full semantics.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if `src` cannot be broadcast into
    /// `self.shape()`. On error `self` is left untouched.
    pub fn copy_from<D2: Dimension>(&mut self, src: &Array<T, D2>) -> FerrayResult<()> {
        copyto(self, src)
    }

    /// Copy values from `src` into `self` at positions where `mask` is `true`,
    /// broadcasting both inputs to `self.shape()`.
    ///
    /// See [`copyto_where`] for the free-function form and full semantics.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if `src` or `mask` cannot be
    /// broadcast into `self.shape()`. On error `self` is left untouched.
    pub fn copy_from_where<D2: Dimension, D3: Dimension>(
        &mut self,
        src: &Array<T, D2>,
        mask: &Array<bool, D3>,
    ) -> FerrayResult<()> {
        copyto_where(self, src, mask)
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

    // -----------------------------------------------------------------------
    // Broadcasting tests (issue #346)
    // -----------------------------------------------------------------------

    use crate::dimension::{Ix2, Ix3, IxDyn};

    #[test]
    fn test_broadcast_2d_row_plus_column() {
        // (3, 1) + (1, 4) -> (3, 4) — both Ix2
        let col = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 2.0, 3.0]).unwrap();
        let row =
            Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let result = (&col + &row).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[
                11.0, 21.0, 31.0, 41.0, // row 1
                12.0, 22.0, 32.0, 42.0, // row 2
                13.0, 23.0, 33.0, 43.0, // row 3
            ]
        );
    }

    #[test]
    fn test_broadcast_2d_stretch_one_axis() {
        // (3, 4) + (1, 4) -> (3, 4)
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 4]), vec![100.0, 200.0, 300.0, 400.0])
            .unwrap();
        let result = (&a + &b).unwrap();
        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[
                101.0, 202.0, 303.0, 404.0, 105.0, 206.0, 307.0, 408.0, 109.0, 210.0, 311.0, 412.0,
            ]
        );
    }

    #[test]
    fn test_broadcast_3d_with_2d_axis() {
        // (2, 3, 4) - (1, 3, 4) -> (2, 3, 4) — both Ix3
        let a =
            Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), (1..=24).map(|i| i as f64).collect())
                .unwrap();
        let b =
            Array::<f64, Ix3>::from_vec(Ix3::new([1, 3, 4]), (1..=12).map(|i| i as f64).collect())
                .unwrap();
        let result = (&a - &b).unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
        // First 12 elements: a[0..12] - b
        let first_half: Vec<f64> = (1..=12).map(|_| 0.0).collect();
        assert_eq!(&result.as_slice().unwrap()[..12], &first_half[..]);
        // Second 12 elements: a[12..24] - b == 12 each
        let second_half: Vec<f64> = (0..12).map(|_| 12.0).collect();
        assert_eq!(&result.as_slice().unwrap()[12..], &second_half[..]);
    }

    #[test]
    fn test_broadcast_incompatible_shapes_error() {
        // (3,) + (4,) — incompatible
        let a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let result = &a + &b;
        assert!(result.is_err());

        // (3, 4) + (3, 5) — incompatible
        let c = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), vec![0.0; 12]).unwrap();
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([3, 5]), vec![0.0; 15]).unwrap();
        assert!((&c + &d).is_err());
    }

    #[test]
    fn test_broadcast_mul_2d() {
        // (3, 1) * (1, 3) -> (3, 3) outer-product style
        let col = Array::<i32, Ix2>::from_vec(Ix2::new([3, 1]), vec![1, 2, 3]).unwrap();
        let row = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![10, 20, 30]).unwrap();
        let result = (&col * &row).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[10, 20, 30, 20, 40, 60, 30, 60, 90]
        );
    }

    // -----------------------------------------------------------------------
    // Cross-rank broadcasting via `add_broadcast` etc.
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_broadcast_1d_plus_2d() {
        // (3,) + (2, 3) -> (2, 3)
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let m =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let result = v.add_broadcast(&m).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[11.0, 22.0, 33.0, 41.0, 52.0, 63.0]
        );
    }

    #[test]
    fn test_add_broadcast_1d_plus_column() {
        // (3,) + (2, 1) -> (2, 3) — the canonical NumPy example from issue #346
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let col = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![10.0, 20.0]).unwrap();
        let result = v.add_broadcast(&col).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[11.0, 12.0, 13.0, 21.0, 22.0, 23.0]
        );
    }

    #[test]
    fn test_sub_broadcast_2d_minus_1d() {
        // (2, 3) - (3,) -> (2, 3)
        let m =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let result = m.sub_broadcast(&v).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[9.0, 18.0, 27.0, 39.0, 48.0, 57.0]
        );
    }

    #[test]
    fn test_mul_broadcast_returns_dyn() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![10.0, 20.0]).unwrap();
        let result: Array<f64, IxDyn> = a.mul_broadcast(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[10.0, 20.0, 30.0, 20.0, 40.0, 60.0]
        );
    }

    #[test]
    fn test_div_broadcast_incompatible() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(a.div_broadcast(&b).is_err());
    }

    #[test]
    fn test_rem_broadcast_2d() {
        let a =
            Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![10, 20, 30, 40, 50, 60]).unwrap();
        let b = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![3, 7, 11]).unwrap();
        let result = a.rem_broadcast(&b).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[10 % 3, 20 % 7, 30 % 11, 40 % 3, 50 % 7, 60 % 11]
        );
    }

    // ------------------------------------------------------------------
    // #348: in-place operators
    // ------------------------------------------------------------------

    #[test]
    fn scalar_add_assign_mutates_in_place() {
        let mut a = arr(vec![1.0, 2.0, 3.0]);
        a += 10.0;
        assert_eq!(a.as_slice().unwrap(), &[11.0, 12.0, 13.0]);
    }

    #[test]
    fn scalar_sub_mul_div_rem_assign() {
        let mut a = arr(vec![10.0, 20.0, 30.0]);
        a -= 1.0;
        assert_eq!(a.as_slice().unwrap(), &[9.0, 19.0, 29.0]);
        a *= 2.0;
        assert_eq!(a.as_slice().unwrap(), &[18.0, 38.0, 58.0]);
        a /= 2.0;
        assert_eq!(a.as_slice().unwrap(), &[9.0, 19.0, 29.0]);
        let mut b = arr_i32(vec![10, 11, 12]);
        b %= 3;
        assert_eq!(b.as_slice().unwrap(), &[1, 2, 0]);
    }

    #[test]
    fn scalar_assign_preserves_shape_ix2() {
        let mut a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        a += 1.0;
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.as_slice().unwrap(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn add_inplace_same_shape_fast_path() {
        let mut a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![10.0, 20.0, 30.0]);
        a.add_inplace(&b).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn sub_mul_div_rem_inplace_same_shape() {
        let mut a = arr(vec![10.0, 20.0, 30.0]);
        let b = arr(vec![1.0, 2.0, 3.0]);
        a.sub_inplace(&b).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[9.0, 18.0, 27.0]);
        a.mul_inplace(&b).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[9.0, 36.0, 81.0]);
        a.div_inplace(&b).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[9.0, 18.0, 27.0]);
        let mut c = arr_i32(vec![10, 20, 30]);
        let d = arr_i32(vec![3, 7, 11]);
        c.rem_inplace(&d).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[1, 6, 8]);
    }

    #[test]
    fn add_inplace_broadcasts_rhs_into_lhs_shape() {
        // (2, 3) += (1, 3) — RHS row broadcast across LHS rows.
        let mut a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![10.0, 20.0, 30.0]).unwrap();
        a.add_inplace(&b).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.as_slice().unwrap(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn add_inplace_broadcasts_column_into_rows() {
        // (2, 3) += (2, 1) — RHS column broadcast across LHS columns.
        let mut a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([2, 1]), vec![100.0, 200.0]).unwrap();
        a.add_inplace(&b).unwrap();
        assert_eq!(
            a.as_slice().unwrap(),
            &[101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
        );
    }

    #[test]
    fn add_inplace_rejects_incompatible_rhs() {
        // (3,) += (4,) — not broadcast-compatible; error must be Err, not panic.
        let mut a = arr(vec![1.0, 2.0, 3.0]);
        let b = arr(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(a.add_inplace(&b).is_err());
        // LHS must be untouched on error.
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn add_inplace_rejects_growing_shape() {
        // (1, 3) += (2, 3) — RHS bigger than LHS; the destination shape is
        // fixed for in-place operations, so this must error.
        let mut a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0; 6]).unwrap();
        assert!(a.add_inplace(&b).is_err());
        assert_eq!(a.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_same_shape_fast_path() {
        let mut dst = arr(vec![0.0, 0.0, 0.0]);
        let src = arr(vec![1.0, 2.0, 3.0]);
        copyto(&mut dst, &src).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_broadcasts_row_into_matrix() {
        // (2, 3) <= (1, 3)
        let mut dst = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        let src = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![10.0, 20.0, 30.0]).unwrap();
        copyto(&mut dst, &src).unwrap();
        assert_eq!(
            dst.as_slice().unwrap(),
            &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn copyto_broadcasts_cross_rank_src() {
        // (2, 3) <= (3,)  — a lower-rank src broadcasts against a higher-rank dst.
        let mut dst = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        let src = arr(vec![7.0, 8.0, 9.0]);
        copyto(&mut dst, &src).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[7.0, 8.0, 9.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn copyto_scalar_src_broadcasts_to_full_dst() {
        // (2, 3) <= () via a length-1 1D stand-in.
        let mut dst = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![0.0; 6]).unwrap();
        let src = arr(vec![42.0]);
        copyto(&mut dst, &src).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[42.0; 6]);
    }

    #[test]
    fn copyto_rejects_growing_dst() {
        // (1, 3) <= (2, 3) — src wants to grow dst; must error, dst untouched.
        let mut dst = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let src = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![99.0; 6]).unwrap();
        assert!(copyto(&mut dst, &src).is_err());
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_rejects_incompatible_shapes() {
        let mut dst = arr(vec![1.0, 2.0, 3.0]);
        let src = arr(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(copyto(&mut dst, &src).is_err());
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_method_form_equivalent_to_function() {
        let mut dst = arr(vec![0.0, 0.0, 0.0]);
        let src = arr(vec![1.0, 2.0, 3.0]);
        dst.copy_from(&src).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_works_for_non_copy_element_type_i64() {
        // Exercises the Clone-not-Copy path on the Element trait — copyto
        // must not require T: Copy.
        let mut dst = Array::<i64, Ix1>::from_vec(Ix1::new([4]), vec![0, 0, 0, 0]).unwrap();
        let src = Array::<i64, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        copyto(&mut dst, &src).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn copyto_where_same_shape_only_writes_masked_positions() {
        let mut dst = arr(vec![1.0, 2.0, 3.0, 4.0]);
        let src = arr(vec![10.0, 20.0, 30.0, 40.0]);
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        copyto_where(&mut dst, &src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[10.0, 2.0, 30.0, 4.0]);
    }

    #[test]
    fn copyto_where_broadcasts_mask_across_dst() {
        // (2, 3) <= (2, 3), mask = (1, 3) broadcasts across rows.
        let mut dst =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let src =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
                .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([1, 3]), vec![true, false, true]).unwrap();
        copyto_where(&mut dst, &src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[10.0, 2.0, 30.0, 40.0, 5.0, 60.0]);
    }

    #[test]
    fn copyto_where_broadcasts_scalar_src_with_mask() {
        // Conditional scalar fill: set dst[i,j] = 99.0 wherever mask is true.
        let mut dst =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let src = arr(vec![99.0]);
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, false, true, false, true, false],
        )
        .unwrap();
        copyto_where(&mut dst, &src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[99.0, 2.0, 99.0, 4.0, 99.0, 6.0]);
    }

    #[test]
    fn copyto_where_all_false_mask_is_noop() {
        let mut dst = arr(vec![1.0, 2.0, 3.0]);
        let original = dst.as_slice().unwrap().to_vec();
        let src = arr(vec![99.0, 99.0, 99.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        copyto_where(&mut dst, &src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &original[..]);
    }

    #[test]
    fn copyto_where_all_true_mask_matches_copyto() {
        let mut dst = arr(vec![0.0, 0.0, 0.0]);
        let src = arr(vec![1.0, 2.0, 3.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        copyto_where(&mut dst, &src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_where_rejects_incompatible_src_shape() {
        let mut dst = arr(vec![1.0, 2.0, 3.0]);
        let src = arr(vec![1.0, 2.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        assert!(copyto_where(&mut dst, &src, &mask).is_err());
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copyto_where_rejects_incompatible_mask_shape() {
        let mut dst = arr(vec![1.0, 2.0, 3.0]);
        let src = arr(vec![10.0, 20.0, 30.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true; 4]).unwrap();
        assert!(copyto_where(&mut dst, &src, &mask).is_err());
        // Validation happens before writes — dst untouched.
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn copy_from_where_method_form_equivalent() {
        let mut dst = arr(vec![1.0, 2.0, 3.0]);
        let src = arr(vec![10.0, 20.0, 30.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
        dst.copy_from_where(&src, &mask).unwrap();
        assert_eq!(dst.as_slice().unwrap(), &[1.0, 20.0, 3.0]);
    }

    #[test]
    fn div_inplace_by_zero_yields_ieee_sentinels() {
        // Pin current semantics: division-by-zero in-place produces IEEE
        // inf/NaN at the offending positions and does NOT error.
        let mut a = arr(vec![1.0, 2.0, 0.0]);
        let b = arr(vec![2.0, 0.0, 0.0]);
        a.div_inplace(&b).unwrap();
        let s = a.as_slice().unwrap();
        assert_eq!(s[0], 0.5);
        assert!(s[1].is_infinite() && s[1].is_sign_positive());
        assert!(s[2].is_nan());
    }
}
