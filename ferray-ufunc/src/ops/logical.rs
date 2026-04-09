// ferray-ufunc: Logical functions
//
// logical_and, logical_or, logical_xor, logical_not, all, any, all_axis,
// any_axis.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

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
pub fn logical_and<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() && y.is_truthy())
}

/// Elementwise logical OR.
pub fn logical_or<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() || y.is_truthy())
}

/// Elementwise logical XOR.
pub fn logical_xor<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + Logical + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x.is_truthy() ^ y.is_truthy())
}

/// Elementwise logical NOT.
pub fn logical_not<T, D>(input: &Array<T, D>) -> FerrayResult<Array<bool, D>>
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

/// Shared axis-reduction kernel for [`all_axis`] and [`any_axis`]. Collapses
/// `axis` by folding truthiness with `op`, short-circuiting as soon as the
/// accumulator reaches `stop_at` (the value the reduction can no longer
/// change away from).
fn reduce_truthy_axis<T, D, F>(
    input: &Array<T, D>,
    axis: usize,
    identity: bool,
    stop_at: bool,
    op: F,
) -> FerrayResult<Array<bool, IxDyn>>
where
    T: Element + Logical,
    D: Dimension,
    F: Fn(bool, &T) -> bool,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }

    let shape: Vec<usize> = input.shape().to_vec();
    let axis_len = shape[axis];
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..].iter().product();

    // Materialize in row-major (iteration) order so that
    // (outer, k, inner) indexing resolves into the flat buffer directly.
    // Uses Clone rather than Copy so the kernel works for any Element type.
    let data: Vec<T> = input.iter().cloned().collect();

    let mut out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    let mut result = vec![identity; out_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            let mut acc = identity;
            for k in 0..axis_len {
                let idx = outer * axis_len * inner_size + k * inner_size + inner;
                acc = op(acc, &data[idx]);
                if acc == stop_at {
                    break;
                }
            }
            result[out_idx] = acc;
        }
    }

    // NumPy collapses a single-axis input to a 0-D scalar; we expose it as a
    // length-1 1-D array because ferray has no Ix0 in the IxDyn constructor
    // path used here and users can .reshape(&[]) if they truly need scalar
    // rank.
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    Array::from_vec(IxDyn::from(&out_shape[..]), result)
}

/// Test whether all elements along `axis` are truthy, returning an array
/// with `axis` removed. Equivalent to `np.all(input, axis=axis)`.
///
/// Short-circuits per-slice: as soon as a `false` is found along the axis
/// for a given output position, the remainder of that slice is skipped.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= input.ndim()`.
pub fn all_axis<T, D>(input: &Array<T, D>, axis: usize) -> FerrayResult<Array<bool, IxDyn>>
where
    T: Element + Logical,
    D: Dimension,
{
    reduce_truthy_axis(input, axis, true, false, |acc, x| acc && x.is_truthy())
}

/// Test whether any element along `axis` is truthy, returning an array with
/// `axis` removed. Equivalent to `np.any(input, axis=axis)`.
///
/// Short-circuits per-slice: as soon as a `true` is found along the axis
/// for a given output position, the remainder of that slice is skipped.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= input.ndim()`.
pub fn any_axis<T, D>(input: &Array<T, D>, axis: usize) -> FerrayResult<Array<bool, IxDyn>>
where
    T: Element + Logical,
    D: Dimension,
{
    reduce_truthy_axis(input, axis, false, true, |acc, x| acc || x.is_truthy())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

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

    // -----------------------------------------------------------------------
    // Broadcasting tests for logical ops (issue #379)
    // -----------------------------------------------------------------------

    #[test]
    fn test_logical_and_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 1]), vec![true, false]).unwrap();
        let b = Array::<bool, Ix2>::from_vec(Ix2::new([1, 3]), vec![true, false, true]).unwrap();
        let r = logical_and(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![true, false, true, false, false, false]
        );
    }

    #[test]
    fn test_logical_or_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 1]), vec![true, false]).unwrap();
        let b = Array::<bool, Ix2>::from_vec(Ix2::new([1, 3]), vec![true, false, true]).unwrap();
        let r = logical_or(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, true, false, true]
        );
    }

    // -----------------------------------------------------------------------
    // Axis-aware all/any (#389)
    // -----------------------------------------------------------------------

    #[test]
    fn all_axis_2d_rows() {
        use ferray_core::dimension::Ix2;
        // [[T,T,T],[T,F,T]] — row 0 all true, row 1 has a false.
        let a = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, true, true, true, false, true],
        )
        .unwrap();
        let r = all_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[true, false]);
    }

    #[test]
    fn all_axis_2d_cols() {
        use ferray_core::dimension::Ix2;
        // [[T,T,F],[T,T,T]] — col 0 all T, col 1 all T, col 2 has F.
        let a = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, true, false, true, true, true],
        )
        .unwrap();
        let r = all_axis(&a, 0).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.as_slice().unwrap(), &[true, true, false]);
    }

    #[test]
    fn any_axis_2d_rows() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, false, false, false, true, false],
        )
        .unwrap();
        let r = any_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[false, true]);
    }

    #[test]
    fn any_axis_2d_cols() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, false, false, false, false],
        )
        .unwrap();
        let r = any_axis(&a, 0).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn all_axis_numeric_integer_input() {
        use ferray_core::dimension::Ix2;
        // Integer truthiness: 0 is false, nonzero is true.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 0, 6]).unwrap();
        let r = all_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[true, false]);
    }

    #[test]
    fn any_axis_numeric_float_input_with_nan() {
        use ferray_core::dimension::Ix1;
        // NaN is truthy (nonzero), Inf is truthy, 0 is falsy.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![0.0, 0.0, f64::NAN, 0.0]).unwrap();
        let r = any_axis(&a, 0).unwrap();
        // Reducing a 1-D array along its only axis produces a length-1 result.
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.as_slice().unwrap(), &[true]);
    }

    #[test]
    fn all_axis_empty_axis_returns_identity() {
        // np.all over an empty slice returns True (vacuously).
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 0]), vec![]).unwrap();
        let r = all_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[true, true]);
    }

    #[test]
    fn any_axis_empty_axis_returns_identity() {
        // np.any over an empty slice returns False.
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 0]), vec![]).unwrap();
        let r = any_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[false, false]);
    }

    #[test]
    fn all_axis_3d_middle_axis() {
        use ferray_core::dimension::Ix3;
        // shape (2, 3, 2): reduce axis=1 → shape (2, 2).
        // Layer 0:
        //   row0: T, T
        //   row1: T, F
        //   row2: T, T
        //   → per-col: T && T && T = T  ;  T && F && T = F
        // Layer 1:
        //   row0: T, T
        //   row1: T, T
        //   row2: T, T
        //   → T, T
        let data = vec![
            true, true, true, false, true, true, // layer 0
            true, true, true, true, true, true, // layer 1
        ];
        let a = Array::<bool, Ix3>::from_vec(Ix3::new([2, 3, 2]), data).unwrap();
        let r = all_axis(&a, 1).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        assert_eq!(r.as_slice().unwrap(), &[true, false, true, true]);
    }

    #[test]
    fn all_axis_out_of_bounds_errors() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![true; 6]).unwrap();
        assert!(all_axis(&a, 5).is_err());
    }

    #[test]
    fn any_axis_out_of_bounds_errors() {
        use ferray_core::dimension::Ix2;
        let a = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![true; 6]).unwrap();
        assert!(any_axis(&a, 2).is_err());
    }

    #[test]
    fn all_axis_short_circuit_correct_value() {
        // Regression: ensure that once we find a `false`, the output is
        // actually set to `false` (not the `true` identity seed).
        use ferray_core::dimension::Ix2;
        let a =
            Array::<bool, Ix2>::from_vec(Ix2::new([1, 4]), vec![false, true, true, true]).unwrap();
        let r = all_axis(&a, 1).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false]);
    }
}
