// ferrum-ufunc: Common helper functions for ufunc implementations
//
// Provides generic unary/binary operation wrappers that handle contiguous
// vs non-contiguous arrays, SIMD dispatch, and broadcasting.

use ferrum_core::Array;
use ferrum_core::dimension::broadcast::broadcast_shapes;
use ferrum_core::dimension::{Dimension, IxDyn};
use ferrum_core::dtype::Element;
use ferrum_core::error::{FerrumError, FerrumResult};

/// Apply a unary function elementwise, preserving dimension.
/// Works for any `T: Element + Float` (or any Copy type with the given fn).
///
/// When the input is contiguous, operates directly on the underlying slice
/// for better auto-vectorization and cache locality.
#[inline]
pub fn unary_float_op<T, D>(input: &Array<T, D>, f: impl Fn(T) -> T) -> FerrumResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    // Fast path: contiguous array — iterate the flat slice directly.
    // Pre-allocate and use iter_mut().zip() for best auto-vectorization.
    if let Some(slice) = input.as_slice() {
        let n = slice.len();
        let mut data = vec![T::zero(); n];
        for (out, &inp) in data.iter_mut().zip(slice.iter()) {
            *out = f(inp);
        }
        Array::from_vec(input.dim().clone(), data)
    } else {
        let data: Vec<T> = input.iter().map(|&x| f(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Apply a unary operation using a pre-written slice-to-slice kernel.
///
/// This is used for operations like sqrt, abs, neg where we have optimized
/// SIMD implementations that operate on contiguous `f64` slices directly.
#[inline]
pub fn unary_slice_op_f64<D>(
    input: &Array<f64, D>,
    kernel: fn(&[f64], &mut [f64]),
    scalar_fallback: fn(f64) -> f64,
) -> FerrumResult<Array<f64, D>>
where
    D: Dimension,
{
    let n = input.size();
    if let Some(slice) = input.as_slice() {
        let mut data = vec![0.0f64; n];
        kernel(slice, &mut data);
        Array::from_vec(input.dim().clone(), data)
    } else {
        let mut data = Vec::with_capacity(n);
        for &x in input.iter() {
            data.push(scalar_fallback(x));
        }
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Apply a unary operation using a pre-written slice-to-slice kernel for f32.
#[inline]
pub fn unary_slice_op_f32<D>(
    input: &Array<f32, D>,
    kernel: fn(&[f32], &mut [f32]),
    scalar_fallback: fn(f32) -> f32,
) -> FerrumResult<Array<f32, D>>
where
    D: Dimension,
{
    let n = input.size();
    if let Some(slice) = input.as_slice() {
        let mut data = vec![0.0f32; n];
        kernel(slice, &mut data);
        Array::from_vec(input.dim().clone(), data)
    } else {
        let mut data = Vec::with_capacity(n);
        for &x in input.iter() {
            data.push(scalar_fallback(x));
        }
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Try to run a SIMD f64 kernel on a contiguous array.
///
/// Returns `None` if `T` is not `f64` or the array is not contiguous,
/// allowing the caller to fall back to the generic scalar path.
#[inline]
pub fn try_simd_f64_unary<T, D>(
    input: &Array<T, D>,
    kernel: fn(&[f64], &mut [f64]),
) -> Option<FerrumResult<Array<T, D>>>
where
    T: Element + Copy,
    D: Dimension,
{
    use std::any::TypeId;

    if TypeId::of::<T>() != TypeId::of::<f64>() {
        return None;
    }
    let slice = input.as_slice()?;
    let n = slice.len();
    // SAFETY: T is f64, verified by TypeId check above. f64 and T have
    // identical size, alignment, and bit representation.
    let f64_slice: &[f64] =
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const f64, n) };
    let mut output = vec![0.0f64; n];
    kernel(f64_slice, &mut output);
    // SAFETY: T is f64. Reinterpret Vec<f64> as Vec<T> without copying.
    let t_vec: Vec<T> = unsafe {
        let mut md = std::mem::ManuallyDrop::new(output);
        Vec::from_raw_parts(md.as_mut_ptr() as *mut T, n, n)
    };
    Some(Array::from_vec(input.dim().clone(), t_vec))
}

/// Apply a unary function that maps T -> U, preserving dimension.
#[inline]
pub fn unary_map_op<T, U, D>(input: &Array<T, D>, f: impl Fn(T) -> U) -> FerrumResult<Array<U, D>>
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

// ---------------------------------------------------------------------------
// f16 helpers — f32-promoted operations (feature-gated)
// ---------------------------------------------------------------------------

/// Apply a unary f32 function to an f16 array via promotion.
///
/// Each element is promoted to f32, the function is applied, and
/// the result is converted back to f16.
#[cfg(feature = "f16")]
#[inline]
pub fn unary_f16_op<D>(
    input: &Array<half::f16, D>,
    f: impl Fn(f32) -> f32,
) -> FerrumResult<Array<half::f16, D>>
where
    D: Dimension,
{
    let data: Vec<half::f16> = input
        .iter()
        .map(|&x| half::f16::from_f32(f(x.to_f32())))
        .collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a unary f32 function to an f16 array, returning a bool array.
#[cfg(feature = "f16")]
#[inline]
pub fn unary_f16_to_bool_op<D>(
    input: &Array<half::f16, D>,
    f: impl Fn(f32) -> bool,
) -> FerrumResult<Array<bool, D>>
where
    D: Dimension,
{
    let data: Vec<bool> = input.iter().map(|&x| f(x.to_f32())).collect();
    Array::from_vec(input.dim().clone(), data)
}

/// Apply a binary f32 function to two f16 arrays via promotion.
#[cfg(feature = "f16")]
#[inline]
pub fn binary_f16_op<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
    f: impl Fn(f32, f32) -> f32,
) -> FerrumResult<Array<half::f16, D>>
where
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrumError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<half::f16> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| half::f16::from_f32(f(x.to_f32(), y.to_f32())))
        .collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Apply a binary f32 function to two f16 arrays, returning a bool array.
#[cfg(feature = "f16")]
#[inline]
pub fn binary_f16_to_bool_op<D>(
    a: &Array<half::f16, D>,
    b: &Array<half::f16, D>,
    f: impl Fn(f32, f32) -> bool,
) -> FerrumResult<Array<bool, D>>
where
    D: Dimension,
{
    if a.shape() != b.shape() {
        return Err(FerrumError::shape_mismatch(format!(
            "binary op: shapes {:?} and {:?} do not match",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<bool> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| f(x.to_f32(), y.to_f32()))
        .collect();
    Array::from_vec(a.dim().clone(), data)
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
