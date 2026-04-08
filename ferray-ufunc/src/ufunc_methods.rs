// ferray-ufunc: Generic ufunc methods (reduce / accumulate / outer / at).
//
// NumPy exposes these as unbound methods on every ufunc:
//
//   np.add.reduce(arr, axis=0)       → sum along axis
//   np.subtract.accumulate(arr, axis) → running differences
//   np.multiply.outer(a, b)          → outer product
//   np.add.at(arr, indices, b)       → unbuffered scattered in-place
//
// ferray-ufunc previously only hand-wrote the `add_reduce`,
// `add_accumulate`, and `multiply_outer` specializations. This module
// provides the *generic* machinery — one implementation per method — that
// works for any `Fn(T, T) -> T` op plus a start value. The existing
// special-case functions now delegate here, and callers who want
// `subtract.reduce` or `power.outer` can build them in one line.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

/// Borrow the input as a contiguous slice, copying into a Vec only if the
/// layout is non-contiguous. Shared by every method in this module.
fn contiguous_data<T, D>(input: &Array<T, D>) -> Vec<T>
where
    T: Element + Copy,
    D: Dimension,
{
    if let Some(s) = input.as_slice() {
        s.to_vec()
    } else {
        input.iter().copied().collect()
    }
}

/// Compute `(outer_size, axis_len, inner_size)` for iterating over an axis.
#[inline]
fn axis_layout(shape: &[usize], axis: usize) -> (usize, usize, usize) {
    let axis_len = shape[axis];
    let outer_size: usize = shape[..axis].iter().product();
    let inner_size: usize = shape[axis + 1..].iter().product();
    (outer_size, axis_len, inner_size)
}

/// Generic reduction along an axis: collapse `axis` by folding with `op`.
///
/// For each position in the output, the reduction starts from `identity`
/// and applies `op(acc, x)` for every element along `axis`. This is
/// equivalent to `np.<ufunc>.reduce(input, axis=...)`.
///
/// `identity` is the reduction seed (0 for add, 1 for multiply,
/// -infinity for max, etc.) — callers pick it.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis >= input.ndim()`.
pub fn reduce_axis<T, D, F>(
    input: &Array<T, D>,
    axis: usize,
    identity: T,
    op: F,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = input.shape().to_vec();
    let (outer_size, axis_len, inner_size) = axis_layout(&shape, axis);

    let mut out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    let data = contiguous_data(input);
    let mut result = vec![identity; out_size];

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let out_idx = outer * inner_size + inner;
            let mut acc = identity;
            for k in 0..axis_len {
                let idx = outer * axis_len * inner_size + k * inner_size + inner;
                acc = op(acc, data[idx]);
            }
            result[out_idx] = acc;
        }
    }

    if out_shape.is_empty() {
        out_shape.push(1);
    }
    Array::from_vec(IxDyn::from(&out_shape[..]), result)
}

/// Generic reduction along an axis with an optional `keepdims` flag.
///
/// Equivalent to calling [`reduce_axis`] followed by inserting a size-1 axis
/// at the reduced position when `keepdims` is `true`. Matches NumPy's
/// `np.<ufunc>.reduce(input, axis=axis, keepdims=keepdims)`.
///
/// `keepdims=false` is identical to [`reduce_axis`] — the reduced axis is
/// removed entirely. `keepdims=true` preserves the axis as a size-1
/// dimension so the result is broadcastable back against the original input
/// shape (the core use case — `arr - arr.sum(axis=1, keepdims=True)`).
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis >= input.ndim()`.
pub fn reduce_axis_keepdims<T, D, F>(
    input: &Array<T, D>,
    axis: usize,
    identity: T,
    keepdims: bool,
    op: F,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    // Delegate the core work to reduce_axis; we re-validate the axis here
    // too so we can produce the keepdims shape without materializing an
    // invalid intermediate.
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }

    let reduced = reduce_axis(input, axis, identity, op)?;
    if !keepdims {
        return Ok(reduced);
    }

    // Rebuild the shape with a size-1 axis re-inserted at `axis`. The core
    // `reduce_axis` also collapses 0-D results to length-1, which we
    // preserve here by inserting rather than replacing.
    let mut kept_shape: Vec<usize> = input.shape().to_vec();
    kept_shape[axis] = 1;
    let data: Vec<T> = reduced.iter().copied().collect();
    Array::from_vec(IxDyn::new(&kept_shape), data)
}

/// Generic cumulative reduction along an axis.
///
/// For each position in the output, the running accumulator starts at the
/// first element and applies `op(acc, x)` for every subsequent element
/// along `axis`. Output shape matches the input. Equivalent to
/// `np.<ufunc>.accumulate(input, axis=...)`.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis >= input.ndim()`.
pub fn accumulate_axis<T, D, F>(
    input: &Array<T, D>,
    axis: usize,
    op: F,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    let ndim = input.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = input.shape().to_vec();
    let (outer_size, axis_len, inner_size) = axis_layout(&shape, axis);

    let data = contiguous_data(input);
    let mut result = data.clone();

    // Walk along the axis, updating in place so we don't need to allocate
    // a running accumulator per lane.
    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base = outer * axis_len * inner_size + inner;
            for k in 1..axis_len {
                let prev = result[base + (k - 1) * inner_size];
                let cur = result[base + k * inner_size];
                result[base + k * inner_size] = op(prev, cur);
            }
        }
    }

    Array::from_vec(input.dim().clone(), result)
}

/// Generic outer product: `outer(a, b)[i, j] = op(a[i], b[j])`.
///
/// Takes two 1-D arrays and produces a 2-D `(a.size(), b.size())` result.
/// Equivalent to `np.<ufunc>.outer(a, b)` for 1-D inputs. Prefers
/// borrowing `a`/`b` directly when they are contiguous so contiguous
/// inputs avoid the two full-length `iter().copied().collect()` buffer
/// materializations the previous implementation performed (see #153).
pub fn outer<T, F>(
    a: &Array<T, ferray_core::dimension::Ix1>,
    b: &Array<T, ferray_core::dimension::Ix1>,
    op: F,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    F: Fn(T, T) -> T,
{
    let m = a.size();
    let n = b.size();
    let mut data = Vec::with_capacity(m * n);

    // Fast path: both inputs contiguous — iterate raw slices in place.
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        for &ai in a_slice {
            for &bj in b_slice {
                data.push(op(ai, bj));
            }
        }
    } else {
        // Fallback for non-contiguous views: iterate once via the
        // generic iterator. Still produces a single output buffer.
        for ai in a.iter().copied() {
            for bj in b.iter().copied() {
                data.push(op(ai, bj));
            }
        }
    }
    Array::from_vec(IxDyn::from(&[m, n][..]), data)
}

/// Generic unbuffered scatter-reduce: for each `(i, v)` pair, update
/// `arr[i] = op(arr[i], v)` in place. 1-D version — `at` in the general
/// case would need a multi-index parameter, which can be added if demand
/// appears.
///
/// Equivalent to `np.<ufunc>.at(arr, indices, values)` for 1-D `arr`.
/// Duplicate indices are applied in the order they appear (matching
/// NumPy's "unbuffered" semantics — unlike `arr[indices] = values` which
/// would only keep the last write at each index).
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `indices.len() != values.len()`.
/// - `FerrayError::InvalidValue` if any index is out of bounds or `arr`
///   is not contiguous.
pub fn at<T, F>(
    arr: &mut Array<T, ferray_core::dimension::Ix1>,
    indices: &[usize],
    values: &[T],
    op: F,
) -> FerrayResult<()>
where
    T: Element + Copy,
    F: Fn(T, T) -> T,
{
    if indices.len() != values.len() {
        return Err(FerrayError::shape_mismatch(format!(
            "at: indices has length {} but values has length {}",
            indices.len(),
            values.len()
        )));
    }
    let n = arr.size();
    let slice = arr.as_slice_mut().ok_or_else(|| {
        FerrayError::invalid_value("at: array must be contiguous (C-order)")
    })?;
    for (&i, &v) in indices.iter().zip(values.iter()) {
        if i >= n {
            return Err(FerrayError::invalid_value(format!(
                "at: index {i} out of bounds for length {n}"
            )));
        }
        slice[i] = op(slice[i], v);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix2;

    use crate::test_util::arr1;

    fn arr2(rows: usize, cols: usize, data: &[f64]) -> Array<f64, Ix2> {
        Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data.to_vec()).unwrap()
    }

    #[test]
    fn reduce_axis_add_1d() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = reduce_axis(&a, 0, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0]);
    }

    #[test]
    fn reduce_axis_add_2d_rows() {
        // Shape (2, 3); reducing axis 1 -> shape (2,), row sums
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axis(&a, 1, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_axis_add_2d_cols() {
        // Shape (2, 3); reducing axis 0 -> shape (3,), column sums
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axis(&a, 0, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[3]);
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_axis_multiply_product() {
        // reducing 1*2*3*4 = 24
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = reduce_axis(&a, 0, 1.0, |acc, x| acc * x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[24.0]);
    }

    #[test]
    fn reduce_axis_max() {
        let a = arr1(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let r = reduce_axis(&a, 0, f64::NEG_INFINITY, f64::max).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[9.0]);
    }

    #[test]
    fn reduce_axis_out_of_bounds() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(reduce_axis(&a, 1, 0.0, |x, y| x + y).is_err());
    }

    // ---- reduce_axis_keepdims (#394) ----

    #[test]
    fn reduce_axis_keepdims_2d_rows_preserves_axis() {
        // Shape (2, 3); reducing axis 1 with keepdims=true -> shape (2, 1).
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axis_keepdims(&a, 1, 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2, 1]);
        assert_eq!(r.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_axis_keepdims_2d_cols_preserves_axis() {
        // Shape (2, 3); reducing axis 0 with keepdims=true -> shape (1, 3).
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axis_keepdims(&a, 0, 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[1, 3]);
        assert_eq!(r.as_slice().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn reduce_axis_keepdims_false_matches_reduce_axis() {
        // keepdims=false must behave identically to the plain reduce_axis.
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let with_flag = reduce_axis_keepdims(&a, 1, 0.0, false, |acc, x| acc + x).unwrap();
        let without = reduce_axis(&a, 1, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(with_flag.shape(), without.shape());
        assert_eq!(with_flag.as_slice().unwrap(), without.as_slice().unwrap());
    }

    #[test]
    fn reduce_axis_keepdims_3d_middle_axis() {
        // Shape (2, 3, 4); reducing axis 1 with keepdims=true -> (2, 1, 4).
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = reduce_axis_keepdims(&a, 1, 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2, 1, 4]);
    }

    #[test]
    fn reduce_axis_keepdims_out_of_bounds_errors() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(reduce_axis_keepdims(&a, 5, 0.0, true, |x, y| x + y).is_err());
        assert!(reduce_axis_keepdims(&a, 5, 0.0, false, |x, y| x + y).is_err());
    }

    #[test]
    fn reduce_axis_keepdims_result_is_broadcastable() {
        // The whole point of keepdims: the result shape broadcasts back
        // against the input for row centering / normalization patterns.
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let row_sums =
            reduce_axis_keepdims(&a, 1, 0.0, true, |acc, x| acc + x).unwrap();
        // row_sums shape is (2, 1) which broadcasts against (2, 3).
        assert_eq!(row_sums.shape(), &[2, 1]);
        let row_sums_slice = row_sums.as_slice().unwrap();
        assert_eq!(row_sums_slice, &[6.0, 60.0]);
    }

    #[test]
    fn accumulate_axis_add_1d() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = accumulate_axis(&a, 0, |acc, x| acc + x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn accumulate_axis_multiply_1d() {
        let a = arr1(&[1.0, 2.0, 3.0, 4.0]);
        let r = accumulate_axis(&a, 0, |acc, x| acc * x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn accumulate_axis_subtract_running_diff() {
        // subtract.accumulate([10, 3, 2, 1]) = [10, 10-3, (10-3)-2, ((10-3)-2)-1]
        //                                    = [10, 7, 5, 4]
        let a = arr1(&[10.0, 3.0, 2.0, 1.0]);
        let r = accumulate_axis(&a, 0, |acc, x| acc - x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0, 7.0, 5.0, 4.0]);
    }

    #[test]
    fn accumulate_axis_2d_rows() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = accumulate_axis(&a, 1, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    #[test]
    fn accumulate_axis_out_of_bounds() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        assert!(accumulate_axis(&a, 1, |x, y| x + y).is_err());
    }

    #[test]
    fn outer_multiply() {
        let a = arr1(&[1.0, 2.0, 3.0]);
        let b = arr1(&[10.0, 20.0]);
        let r = outer(&a, &b, |x, y| x * y).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(
            r.as_slice().unwrap(),
            &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]
        );
    }

    #[test]
    fn outer_add() {
        let a = arr1(&[1.0, 2.0]);
        let b = arr1(&[10.0, 20.0, 30.0]);
        let r = outer(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.as_slice().unwrap(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn outer_power() {
        let a = arr1(&[2.0, 3.0]);
        let b = arr1(&[2.0, 3.0]);
        let r = outer(&a, &b, |x, y| x.powf(y)).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // 2^2=4, 2^3=8, 3^2=9, 3^3=27
        assert_eq!(r.as_slice().unwrap(), &[4.0, 8.0, 9.0, 27.0]);
    }

    #[test]
    fn at_add_unbuffered_duplicates() {
        // Duplicate index 0 -> both writes apply (adding twice), unlike
        // simple indexed assignment which would only keep the last.
        let mut a = arr1(&[0.0, 0.0, 0.0]);
        at(&mut a, &[0, 0, 1, 2], &[1.0, 2.0, 5.0, 10.0], |acc, x| acc + x)
            .unwrap();
        assert_eq!(a.as_slice().unwrap(), &[3.0, 5.0, 10.0]);
    }

    #[test]
    fn at_multiply() {
        let mut a = arr1(&[1.0, 1.0, 1.0, 1.0]);
        at(&mut a, &[1, 2, 2], &[5.0, 3.0, 4.0], |acc, x| acc * x).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[1.0, 5.0, 12.0, 1.0]);
    }

    #[test]
    fn at_length_mismatch_errors() {
        let mut a = arr1(&[0.0; 4]);
        assert!(at(&mut a, &[0, 1], &[1.0], |x, y| x + y).is_err());
    }

    #[test]
    fn at_index_out_of_bounds_errors() {
        let mut a = arr1(&[0.0; 3]);
        assert!(at(&mut a, &[5], &[1.0], |x, y| x + y).is_err());
    }
}
