// ferray-window: Functional programming utilities
//
// Implements NumPy-equivalent functional utilities: vectorize, piecewise,
// apply_along_axis, and apply_over_axes.

use ferray_core::Array;
use ferray_core::dimension::{Axis, Dimension, Ix1, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Zero;

/// Wrap a scalar function to operate elementwise on arrays of any dimension.
///
/// Returns a closure that accepts `&Array<T, D>` and returns
/// `FerrayResult<Array<U, D>>`, applying `f` to every element.
///
/// This is NumPy's `np.vectorize` — in Rust it is essentially `.mapv()`
/// wrapped as a reusable callable.
///
/// Works with any dimension type: `Ix1`, `Ix2`, `IxDyn`, etc.
///
/// # Example
/// ```ignore
/// let square = vectorize(|x: f64| x * x);
/// let result = square(&input_array)?;
/// ```
pub fn vectorize<T, U, F, D>(f: F) -> impl Fn(&Array<T, D>) -> FerrayResult<Array<U, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
    F: Fn(T) -> U,
{
    move |input: &Array<T, D>| {
        let data: Vec<U> = input.iter().map(|&x| f(x)).collect();
        Array::from_vec(input.dim().clone(), data)
    }
}

/// Evaluate a piecewise-defined function.
///
/// For each element position, the first condition in `condlist` that is `true`
/// determines which function from `funclist` is applied. Elements where no
/// condition is true receive the `default` value.
///
/// This is equivalent to `numpy.piecewise(x, condlist, funclist)`.
///
/// # Arguments
/// * `x` - The input array.
/// * `condlist` - A slice of boolean arrays, each the same shape as `x`.
/// * `funclist` - A slice of function references, one per condition. The
///   slice element type is `&dyn Fn(T) -> T` (not `Box<dyn Fn>`) so
///   callers can pass stack-allocated closures without heap allocation
///   (#297).
/// * `default` - The default value for elements where no condition is true.
///
/// # Errors
/// - Returns `FerrayError::InvalidValue` if `condlist` and `funclist` have different lengths.
/// - Returns `FerrayError::ShapeMismatch` if any condition array has a different shape than `x`.
pub fn piecewise<T, D>(
    x: &Array<T, D>,
    condlist: &[Array<bool, D>],
    funclist: &[&dyn Fn(T) -> T],
    default: T,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if condlist.len() != funclist.len() {
        return Err(FerrayError::invalid_value(format!(
            "piecewise: condlist length ({}) must equal funclist length ({})",
            condlist.len(),
            funclist.len()
        )));
    }

    for (i, cond) in condlist.iter().enumerate() {
        if cond.shape() != x.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "piecewise: condlist[{i}] shape {:?} does not match x shape {:?}",
                cond.shape(),
                x.shape()
            )));
        }
    }

    let size = x.size();
    let mut result_data = vec![default; size];
    let x_data: Vec<T> = x.iter().copied().collect();

    // Collect all condition data upfront
    let cond_data: Vec<Vec<bool>> = condlist
        .iter()
        .map(|c| c.iter().copied().collect())
        .collect();

    // For each element, find the first matching condition
    for i in 0..size {
        for (j, cond) in cond_data.iter().enumerate() {
            if cond[i] {
                result_data[i] = funclist[j](x_data[i]);
                break;
            }
        }
    }

    Array::from_vec(x.dim().clone(), result_data)
}

/// Apply a function along one axis of an array.
///
/// The function receives 1-D slices (lanes) along the specified axis and
/// returns a scalar value. The result has one fewer dimension than the input
/// (the specified axis is removed).
///
/// This is equivalent to `numpy.apply_along_axis(func1d, axis, arr)` when
/// `func1d` returns a scalar.
///
/// # Arguments
/// * `func` - A function that takes a 1-D array view and returns a scalar result.
/// * `axis` - The axis along which to apply the function.
/// * `a` - The input array.
///
/// # Errors
/// - Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
/// - Propagates any error from the function or array construction.
pub fn apply_along_axis<T, D>(
    func: impl Fn(&Array<T, Ix1>) -> FerrayResult<T>,
    axis: Axis,
    a: &Array<T, D>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
{
    let ndim = a.ndim();
    let ax = axis.index();
    if ax >= ndim {
        return Err(FerrayError::axis_out_of_bounds(ax, ndim));
    }

    // Collect lanes along the axis, apply the function, collect results
    let lanes_iter = a.lanes(axis)?;
    let mut results = Vec::new();

    for lane in lanes_iter {
        // Convert the Ix1 ArrayView to an owned Array<T, Ix1>
        let owned_lane = lane.to_owned();
        let val = func(&owned_lane)?;
        results.push(val);
    }

    // Compute the result shape: input shape with the axis dimension removed
    let mut result_shape: Vec<usize> = a.shape().to_vec();
    result_shape.remove(ax);
    if result_shape.is_empty() {
        // 0-D result when input was 1-D
        result_shape.push(results.len());
    }

    Array::from_vec(IxDyn::new(&result_shape), results)
}

/// Apply a reducing function repeatedly over multiple axes.
///
/// The function is applied to the array over each specified axis in sequence.
/// After each application, the axis dimension is kept with size 1 (keepdims
/// semantics) to maintain dimensionality alignment for subsequent reductions.
///
/// This is equivalent to `numpy.apply_over_axes(func, a, axes)`. Generic
/// over any `T: Element + Copy` so f32/i32/etc. arrays work too — the
/// previous signature hard-coded `Array<f64, IxDyn>` for no real
/// reason (#182).
///
/// # Errors
/// - Returns `FerrayError::AxisOutOfBounds` if any axis is out of bounds.
/// - Propagates any error from the function.
pub fn apply_over_axes<T, F>(
    func: F,
    a: &Array<T, IxDyn>,
    axes: &[usize],
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    F: Fn(&Array<T, IxDyn>, Axis) -> FerrayResult<Array<T, IxDyn>>,
{
    let ndim = a.ndim();
    for &ax in axes {
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
    }

    let mut current = a.clone();
    for &ax in axes {
        current = func(&current, Axis(ax))?;
        // Ensure the result has the same number of dimensions as current
        // (keepdims semantics): if the function collapsed an axis, we don't
        // need to re-expand since we expect the function to keep dims.
    }

    Ok(current)
}

/// Helper: sum along an axis with keepdims semantics (keeps the axis as size 1).
///
/// This is useful as a `func` argument for [`apply_over_axes`]. Generic
/// over any `T: Element + Copy + Zero + Add` so it works for any
/// addable numeric type (#182).
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`.
pub fn sum_axis_keepdims<T>(a: &Array<T, IxDyn>, axis: Axis) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy + Zero + core::ops::Add<Output = T>,
{
    let ndim = a.ndim();
    let ax = axis.index();
    if ax >= ndim {
        return Err(FerrayError::axis_out_of_bounds(ax, ndim));
    }

    let reduced = a.fold_axis(axis, <T as Zero>::zero(), |acc, &x| *acc + x)?;

    // Reinsert the axis as size 1 (keepdims)
    let mut new_shape: Vec<usize> = reduced.shape().to_vec();
    new_shape.insert(ax, 1);
    let data: Vec<T> = reduced.iter().copied().collect();
    Array::from_vec(IxDyn::new(&new_shape), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_bool(data: Vec<bool>) -> Array<bool, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    // -----------------------------------------------------------------------
    // AC-4: vectorize(|x: f64| x.powi(2))(&array) produces element-squared
    // -----------------------------------------------------------------------
    #[test]
    fn vectorize_square_ac4() {
        let square = vectorize(|x: f64| x.powi(2));
        let input = arr1(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = square(&input).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 4.0, 9.0, 16.0, 25.0][..]);
    }

    #[test]
    fn vectorize_matches_mapv() {
        let f = |x: f64| x.sin();
        let vf = vectorize(f);
        let input = arr1(vec![0.0, 1.0, 2.0, 3.0]);
        let via_vectorize = vf(&input).unwrap();
        let via_mapv = input.mapv(f);
        assert_eq!(
            via_vectorize.as_slice().unwrap(),
            via_mapv.as_slice().unwrap()
        );
    }

    #[test]
    fn vectorize_2d_generic_dimension() {
        // vectorize is now generic over dimension — this used to live
        // in a separate `vectorize_nd` that has been removed (#293).
        let square = vectorize(|x: f64| x * x);
        let input =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let result = square(&input).unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.as_slice().unwrap(),
            &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0][..]
        );
    }

    #[test]
    fn vectorize_empty() {
        let f = vectorize(|x: f64| x + 1.0);
        let input = arr1(vec![]);
        let result = f(&input).unwrap();
        assert_eq!(result.shape(), &[0]);
    }

    // -----------------------------------------------------------------------
    // Piecewise tests
    // -----------------------------------------------------------------------
    #[test]
    fn piecewise_basic() {
        let x = arr1(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let cond_neg = arr1_bool(vec![true, true, false, false, false]);
        let cond_pos = arr1_bool(vec![false, false, false, true, true]);

        // piecewise now takes `&[&dyn Fn]` rather than `&[Box<dyn Fn>]`
        // so callers don't need a heap allocation per branch (#297).
        let neg: &dyn Fn(f64) -> f64 = &|v| -v;
        let pos: &dyn Fn(f64) -> f64 = &|v| v * 2.0;
        let result = piecewise(&x, &[cond_neg, cond_pos], &[neg, pos], 0.0).unwrap();

        let s = result.as_slice().unwrap();
        assert_eq!(s, &[2.0, 1.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn piecewise_first_match_wins() {
        let x = arr1(vec![1.0, 2.0, 3.0]);
        // Both conditions true for all elements
        let cond1 = arr1_bool(vec![true, true, true]);
        let cond2 = arr1_bool(vec![true, true, true]);

        let f1: &dyn Fn(f64) -> f64 = &|v| v * 10.0;
        let f2: &dyn Fn(f64) -> f64 = &|v| v * 100.0;
        let result = piecewise(&x, &[cond1, cond2], &[f1, f2], 0.0).unwrap();

        // First condition wins
        let s = result.as_slice().unwrap();
        assert_eq!(s, &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn piecewise_no_match_uses_default() {
        let x = arr1(vec![1.0, 2.0, 3.0]);
        let cond = arr1_bool(vec![false, false, false]);

        let f: &dyn Fn(f64) -> f64 = &|v| v * 10.0;
        let result = piecewise(&x, &[cond], &[f], -999.0).unwrap();

        let s = result.as_slice().unwrap();
        assert_eq!(s, &[-999.0, -999.0, -999.0]);
    }

    #[test]
    fn piecewise_length_mismatch() {
        let x = arr1(vec![1.0, 2.0]);
        let cond = arr1_bool(vec![true, false]);
        let f1: &dyn Fn(f64) -> f64 = &|v| v;
        let f2: &dyn Fn(f64) -> f64 = &|v| v;
        assert!(piecewise(&x, &[cond], &[f1, f2], 0.0).is_err());
    }

    #[test]
    fn piecewise_shape_mismatch() {
        let x = arr1(vec![1.0, 2.0]);
        let cond = arr1_bool(vec![true, false, true]); // wrong shape
        let f: &dyn Fn(f64) -> f64 = &|v| v;
        assert!(piecewise(&x, &[cond], &[f], 0.0).is_err());
    }

    // -----------------------------------------------------------------------
    // AC-5: apply_along_axis sum along axis 0 produces column sums
    // -----------------------------------------------------------------------
    #[test]
    fn apply_along_axis_col_sums_ac5() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let result = apply_along_axis(
            |col| {
                let sum: f64 = col.iter().sum();
                Ok(sum)
            },
            Axis(0),
            &m,
        )
        .unwrap();

        // Lanes along axis 0 yield columns: [1,4], [2,5], [3,6]
        // Sums: [5, 7, 9]
        assert_eq!(result.shape(), &[3]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn apply_along_axis_row_sums() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        let result = apply_along_axis(
            |row| {
                let sum: f64 = row.iter().sum();
                Ok(sum)
            },
            Axis(1),
            &m,
        )
        .unwrap();

        // Lanes along axis 1 yield rows: [1,2,3], [4,5,6]
        // Sums: [6, 15]
        assert_eq!(result.shape(), &[2]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![6.0, 15.0]);
    }

    #[test]
    fn apply_along_axis_1d() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let result = apply_along_axis(
            |lane| {
                let sum: f64 = lane.iter().sum();
                Ok(sum)
            },
            Axis(0),
            &a,
        )
        .unwrap();
        // Should return scalar-like (1 element)
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![6.0]);
    }

    #[test]
    fn apply_along_axis_out_of_bounds() {
        let m = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert!(apply_along_axis(|_| Ok(0.0), Axis(5), &m).is_err());
    }

    // -----------------------------------------------------------------------
    // apply_over_axes tests
    // -----------------------------------------------------------------------
    #[test]
    fn apply_over_axes_sum() {
        // 2x3 array, sum over axis 0 then axis 1
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        let result = apply_over_axes(sum_axis_keepdims, &a, &[0, 1]).unwrap();

        // After sum axis 0: shape [1, 3], values [5, 7, 9]
        // After sum axis 1: shape [1, 1], values [21]
        assert_eq!(result.shape(), &[1, 1]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![21.0]);
    }

    #[test]
    fn apply_over_axes_single_axis() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        let result = apply_over_axes(sum_axis_keepdims, &a, &[0]).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn apply_over_axes_out_of_bounds() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        assert!(apply_over_axes(sum_axis_keepdims, &a, &[5]).is_err());
    }

    // -----------------------------------------------------------------------
    // sum_axis_keepdims tests
    // -----------------------------------------------------------------------
    #[test]
    fn sum_axis_keepdims_basic() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        let result = sum_axis_keepdims(&a, Axis(0)).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn sum_axis_keepdims_axis1() {
        let a =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();

        let result = sum_axis_keepdims(&a, Axis(1)).unwrap();
        assert_eq!(result.shape(), &[2, 1]);
        let data: Vec<f64> = result.iter().copied().collect();
        assert_eq!(data, vec![6.0, 15.0]);
    }
}
