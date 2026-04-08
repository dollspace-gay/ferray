//! First-class ufunc objects (#376).
//!
//! NumPy exposes every binary ufunc as a runtime object that carries
//! its own identity element and supports the standard methods —
//! `np.add.reduce(arr)`, `np.add.accumulate(arr)`, `np.add.outer(a, b)`,
//! `np.add.at(arr, idx, vals)`, and `np.add(a, b)` itself. These work
//! for *any* registered ufunc, so you don't have to write
//! `add_reduce`, `subtract_reduce`, `multiply_reduce` by hand.
//!
//! This module provides the Rust analogue: [`Ufunc<T, F>`] — a zero-cost
//! wrapper around a binary op `F: Fn(T, T) -> T` plus an identity
//! element (the seed for `.reduce()`). All five methods delegate to
//! the generic machinery already built for [`crate::ufunc_methods`]
//! (reduce_axis, accumulate_axis, outer, at) and to
//! [`crate::helpers::binary_elementwise_op`] for the elementwise call.
//!
//! # Example
//!
//! ```
//! use ferray_core::{Array, Ix1};
//! use ferray_ufunc::Ufunc;
//!
//! // Define once; use everywhere.
//! let max = Ufunc::new("max", f64::NEG_INFINITY, f64::max);
//!
//! let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
//! let m = max.reduce(&a, 0).unwrap();
//! assert_eq!(m.as_slice().unwrap(), &[5.0]);
//! ```
//!
//! The pre-registered ufuncs at the bottom of this module (`ADD`,
//! `SUBTRACT`, `MULTIPLY`, `DIVIDE`, `MAX`, `MIN`) cover the common
//! NumPy names. They're plain `fn()` returning a fresh `Ufunc` each
//! call to avoid hidden global state.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

use crate::helpers::binary_elementwise_op;
use crate::ufunc_methods::{accumulate_axis, at, outer, reduce_axis};

/// A first-class binary ufunc: a name, an identity element, and the
/// binary kernel. Construct with [`Ufunc::new`]; the name is used only
/// for error messages and introspection.
#[derive(Clone, Copy)]
pub struct Ufunc<T, F>
where
    T: Element + Copy,
    F: Fn(T, T) -> T,
{
    name: &'static str,
    identity: T,
    op: F,
}

impl<T, F> Ufunc<T, F>
where
    T: Element + Copy,
    F: Fn(T, T) -> T,
{
    /// Construct a new ufunc. `name` is only used for diagnostics;
    /// `identity` is the reduction seed (0 for `add`, 1 for `multiply`,
    /// `-inf` for `max`, etc.); `op` is the binary kernel.
    #[inline]
    pub const fn new(name: &'static str, identity: T, op: F) -> Self {
        Self { name, identity, op }
    }

    /// The ufunc's name, used for diagnostics.
    #[inline]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The identity element used as the seed for [`reduce`](Self::reduce).
    #[inline]
    pub fn identity(&self) -> T {
        self.identity
    }

    /// Elementwise call: `c[i] = op(a[i], b[i])`. Equivalent to
    /// `numpy_ufunc(a, b)`.
    ///
    /// Requires same-shape inputs; for broadcasting, use the
    /// free-function entry points in [`crate::ops`] which thread
    /// broadcast metadata through the helpers.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if shapes differ.
    pub fn call<D: Dimension>(
        &self,
        a: &Array<T, D>,
        b: &Array<T, D>,
    ) -> FerrayResult<Array<T, D>> {
        binary_elementwise_op(a, b, &self.op)
    }

    /// `ufunc.reduce(arr, axis)` — collapse `axis` by folding with
    /// `op`, seeded with `identity`.
    pub fn reduce<D: Dimension>(
        &self,
        a: &Array<T, D>,
        axis: usize,
    ) -> FerrayResult<Array<T, IxDyn>> {
        reduce_axis(a, axis, self.identity, &self.op)
    }

    /// `ufunc.accumulate(arr, axis)` — running fold along `axis`,
    /// output shape matches input.
    pub fn accumulate<D: Dimension>(
        &self,
        a: &Array<T, D>,
        axis: usize,
    ) -> FerrayResult<Array<T, D>> {
        accumulate_axis(a, axis, &self.op)
    }

    /// `ufunc.outer(a, b)` — 1-D × 1-D outer product with `op`.
    pub fn outer(
        &self,
        a: &Array<T, Ix1>,
        b: &Array<T, Ix1>,
    ) -> FerrayResult<Array<T, IxDyn>> {
        outer(a, b, &self.op)
    }

    /// `ufunc.at(arr, indices, values)` — unbuffered scatter-reduce
    /// in place. Duplicate indices are applied in the order they
    /// appear (matches NumPy semantics).
    pub fn at(
        &self,
        arr: &mut Array<T, Ix1>,
        indices: &[usize],
        values: &[T],
    ) -> FerrayResult<()> {
        at(arr, indices, values, &self.op)
    }
}

// ---------------------------------------------------------------------------
// Pre-registered ufuncs for the common NumPy names
//
// These are plain `fn()` constructors that return a fresh Ufunc —
// avoiding hidden global state while still providing the NumPy-style
// `Ufuncs::add::<f64>().reduce(&arr, 0)` ergonomics.
// ---------------------------------------------------------------------------

/// Build the `add` ufunc for a given float type (`identity = 0`).
pub fn add_ufunc<T>() -> Ufunc<T, fn(T, T) -> T>
where
    T: Element + Copy + std::ops::Add<Output = T>,
{
    fn add_kernel<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
        a + b
    }
    Ufunc::new("add", <T as Element>::zero(), add_kernel::<T> as fn(T, T) -> T)
}

/// Build the `subtract` ufunc (`identity = 0`).
pub fn subtract_ufunc<T>() -> Ufunc<T, fn(T, T) -> T>
where
    T: Element + Copy + std::ops::Sub<Output = T>,
{
    fn sub_kernel<T: std::ops::Sub<Output = T>>(a: T, b: T) -> T {
        a - b
    }
    Ufunc::new("subtract", <T as Element>::zero(), sub_kernel::<T> as fn(T, T) -> T)
}

/// Build the `multiply` ufunc (`identity = 1`).
pub fn multiply_ufunc<T>() -> Ufunc<T, fn(T, T) -> T>
where
    T: Element + Copy + std::ops::Mul<Output = T>,
{
    fn mul_kernel<T: std::ops::Mul<Output = T>>(a: T, b: T) -> T {
        a * b
    }
    Ufunc::new("multiply", <T as Element>::one(), mul_kernel::<T> as fn(T, T) -> T)
}

/// Build the `divide` ufunc (`identity = 1`).
pub fn divide_ufunc<T>() -> Ufunc<T, fn(T, T) -> T>
where
    T: Element + Copy + std::ops::Div<Output = T>,
{
    fn div_kernel<T: std::ops::Div<Output = T>>(a: T, b: T) -> T {
        a / b
    }
    Ufunc::new("divide", <T as Element>::one(), div_kernel::<T> as fn(T, T) -> T)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    use crate::test_util::arr1;

    #[test]
    fn add_ufunc_roundtrip() {
        let add = add_ufunc::<f64>();
        assert_eq!(add.name(), "add");
        assert_eq!(add.identity(), 0.0);

        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[10.0, 20.0, 30.0]);
        let c = add.call(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn add_ufunc_reduce() {
        let add = add_ufunc::<f64>();
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = add.reduce(&a, 0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0]);
    }

    #[test]
    fn add_ufunc_accumulate() {
        let add = add_ufunc::<f64>();
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = add.accumulate(&a, 0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn multiply_ufunc_reduce_is_product() {
        let mul = multiply_ufunc::<f64>();
        assert_eq!(mul.identity(), 1.0);
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = mul.reduce(&a, 0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[24.0]);
    }

    #[test]
    fn multiply_ufunc_outer() {
        let mul = multiply_ufunc::<f64>();
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[10.0, 20.0]);
        let r = mul.outer(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(
            r.as_slice().unwrap(),
            &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]
        );
    }

    #[test]
    fn subtract_ufunc_accumulate_is_running_diff() {
        // subtract.accumulate([10, 3, 2, 1]) = [10, 7, 5, 4]
        let sub = subtract_ufunc::<f64>();
        let a = arr1(&[10.0, 3.0, 2.0, 1.0]);
        let r = sub.accumulate(&a, 0).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0, 7.0, 5.0, 4.0]);
    }

    #[test]
    fn user_defined_max_ufunc() {
        // User builds a custom ufunc in one line.
        let max = Ufunc::new("max", f64::NEG_INFINITY, f64::max);
        let a = arr1(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let m = max.reduce(&a, 0).unwrap();
        assert_eq!(m.as_slice().unwrap(), &[9.0]);
    }

    #[test]
    fn add_ufunc_at_unbuffered_duplicates() {
        let add = add_ufunc::<f64>();
        let mut a = arr1(&[0.0, 0.0, 0.0]);
        add.at(&mut a, &[0, 0, 1, 2], &[1.0, 2.0, 5.0, 10.0])
            .unwrap();
        assert_eq!(a.as_slice().unwrap(), &[3.0, 5.0, 10.0]);
    }

    #[test]
    fn add_ufunc_reduce_2d_row_sums() {
        use ferray_core::dimension::Ix2;
        let add = add_ufunc::<f64>();
        // 2x3 array, reduce axis=1 → row sums
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let r = add.reduce(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn divide_ufunc_elementwise() {
        let div = divide_ufunc::<f64>();
        assert_eq!(div.identity(), 1.0);
        let a = arr1(&[10.0, 20.0, 30.0]);
        let b = arr1(&[2.0, 4.0, 5.0]);
        let c = div.call(&a, &b).unwrap();
        assert_eq!(c.as_slice().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn ufunc_is_copy() {
        // Copy trait lets callers freely pass the ufunc by value.
        let add = add_ufunc::<f64>();
        let add_copy = add;
        assert_eq!(add.name(), add_copy.name());
        let _ = Ix2::new([1, 1]); // silence unused-import warning for Ix2
    }
}
