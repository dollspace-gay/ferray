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
/// at the reduced position when `keepdims` is `true`. Matches `NumPy`'s
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

/// Generic reduction over multiple axes simultaneously.
///
/// Equivalent to `np.<ufunc>.reduce(input, axis=axes, keepdims=keepdims)`
/// where `axes` is a tuple of axes to collapse. Reduces all listed axes in
/// a single pass over the input — unlike chaining
/// `reduce_axis(reduce_axis(...))`, this never materializes an
/// intermediate buffer and is immune to the "axes shift after a reduction"
/// pitfall the issue calls out (#395).
///
/// `axes` may be empty: in that case the input is copied through unchanged
/// (matching `NumPy`'s `axis=()` semantics — no axes are reduced). When
/// `axes` covers every axis the result collapses to a single scalar; with
/// `keepdims=false` it is wrapped in a length-1 1-D array (the same
/// convention used by [`reduce_axis`]), and with `keepdims=true` it has
/// shape `(1, 1, …, 1)` matching the original rank.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if any axis is `>= input.ndim()`.
/// - `FerrayError::InvalidValue` if `axes` contains duplicates.
pub fn reduce_axes<T, D, F>(
    input: &Array<T, D>,
    axes: &[usize],
    identity: T,
    keepdims: bool,
    op: F,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    let ndim = input.ndim();
    let shape: Vec<usize> = input.shape().to_vec();

    // Validate axes: in-bounds and unique. We sort a local copy so the
    // public API can pass them in any order.
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    for window in sorted_axes.windows(2) {
        if window[0] == window[1] {
            return Err(FerrayError::invalid_value(format!(
                "reduce_axes: duplicate axis {}",
                window[0]
            )));
        }
    }
    for &ax in &sorted_axes {
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
    }

    // Build the index of "kept" axes (those NOT being reduced) once. We
    // use binary_search against sorted_axes to test membership in O(log k).
    let kept_axes: Vec<usize> = (0..ndim)
        .filter(|i| sorted_axes.binary_search(i).is_err())
        .collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&i| shape[i]).collect();

    // Output shape: drop reduced axes (keepdims=false) or set them to 1
    // (keepdims=true). Note that 0-D output is collapsed to a length-1
    // 1-D array on the keepdims=false path so callers always receive a
    // shape they can index into.
    let mut out_shape: Vec<usize> = if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &d)| {
                if sorted_axes.binary_search(&i).is_ok() {
                    1
                } else {
                    d
                }
            })
            .collect()
    } else {
        kept_dims.clone()
    };
    let out_size: usize = if keepdims {
        out_shape.iter().product()
    } else {
        kept_dims.iter().product::<usize>().max(1)
    };

    // Empty `axes` is a no-op reduction in NumPy: just return a copy of
    // the input (with the original shape).
    if sorted_axes.is_empty() {
        let data = contiguous_data(input);
        return Array::from_vec(IxDyn::new(&shape), data);
    }

    // C-order strides for the input. We'll decompose every flat index into
    // a multi-index and then collapse the kept axes back into an output
    // flat offset.
    let mut in_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }

    // C-order strides for the *output* axes — only the kept ones contribute.
    let mut out_strides = vec![1usize; kept_dims.len()];
    for i in (0..kept_dims.len().saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * kept_dims[i + 1];
    }

    let data = contiguous_data(input);
    let mut result = vec![identity; out_size];

    for (flat, &x) in data.iter().enumerate() {
        // Decompose `flat` into the input multi-index using in_strides,
        // then project to the output flat index by walking only the kept
        // axes. This single-pass scheme is the whole point — no
        // intermediate buffers and no per-reduction shape recomputation.
        let mut rem = flat;
        let mut out_flat = 0usize;
        let mut kept_pos = 0usize;
        for (i, &stride) in in_strides.iter().enumerate() {
            let idx = rem / stride;
            rem %= stride;
            if sorted_axes.binary_search(&i).is_err() {
                // i is a kept axis — accumulate its contribution to the
                // output flat index.
                if !out_strides.is_empty() {
                    out_flat += idx * out_strides[kept_pos];
                }
                kept_pos += 1;
            }
        }
        result[out_flat] = op(result[out_flat], x);
    }

    // Match reduce_axis: a fully-collapsed result becomes a length-1 1-D
    // array rather than a true 0-D so callers can always `.as_slice()`.
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    Array::from_vec(IxDyn::new(&out_shape), result)
}

/// Generic reduction over the entire array (the `axis=None` form).
///
/// Equivalent to `np.<ufunc>.reduce(input, axis=None)`. Folds every
/// element through `op` starting from `identity` and returns the single
/// scalar result. Use [`reduce_axes`] when you want axis-aware reductions
/// and a wrapped array result.
///
/// # Errors
/// Returns `FerrayError` only via the `Array::from_vec` round-trip in
/// [`contiguous_data`]; the reduction itself is infallible.
pub fn reduce_all<T, D, F>(input: &Array<T, D>, identity: T, op: F) -> T
where
    T: Element + Copy,
    D: Dimension,
    F: Fn(T, T) -> T,
{
    let mut acc = identity;
    if let Some(slice) = input.as_slice() {
        for &x in slice {
            acc = op(acc, x);
        }
    } else {
        for x in input.iter().copied() {
            acc = op(acc, x);
        }
    }
    acc
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
    let mut result = data;

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
/// `NumPy`'s "unbuffered" semantics — unlike `arr[indices] = values` which
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
    let slice = arr
        .as_slice_mut()
        .ok_or_else(|| FerrayError::invalid_value("at: array must be contiguous (C-order)"))?;
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
        let a = arr1([1.0, 2.0, 3.0, 4.0]);
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
        let a = arr1([1.0, 2.0, 3.0, 4.0]);
        let r = reduce_axis(&a, 0, 1.0, |acc, x| acc * x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[24.0]);
    }

    #[test]
    fn reduce_axis_max() {
        let a = arr1([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let r = reduce_axis(&a, 0, f64::NEG_INFINITY, f64::max).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[9.0]);
    }

    #[test]
    fn reduce_axis_out_of_bounds() {
        let a = arr1([1.0, 2.0, 3.0]);
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
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = reduce_axis_keepdims(&a, 1, 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2, 1, 4]);
    }

    #[test]
    fn reduce_axis_keepdims_out_of_bounds_errors() {
        let a = arr1([1.0, 2.0, 3.0]);
        assert!(reduce_axis_keepdims(&a, 5, 0.0, true, |x, y| x + y).is_err());
        assert!(reduce_axis_keepdims(&a, 5, 0.0, false, |x, y| x + y).is_err());
    }

    #[test]
    fn reduce_axis_keepdims_result_is_broadcastable() {
        // The whole point of keepdims: the result shape broadcasts back
        // against the input for row centering / normalization patterns.
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let row_sums = reduce_axis_keepdims(&a, 1, 0.0, true, |acc, x| acc + x).unwrap();
        // row_sums shape is (2, 1) which broadcasts against (2, 3).
        assert_eq!(row_sums.shape(), &[2, 1]);
        let row_sums_slice = row_sums.as_slice().unwrap();
        assert_eq!(row_sums_slice, &[6.0, 60.0]);
    }

    // ---- reduce_axes / reduce_all multi-axis reductions (#395) ----

    #[test]
    fn reduce_axes_single_axis_matches_reduce_axis() {
        // Single-axis reduce_axes must agree with the dedicated reduce_axis.
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let multi = reduce_axes(&a, &[1], 0.0, false, |acc, x| acc + x).unwrap();
        let single = reduce_axis(&a, 1, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(multi.shape(), single.shape());
        assert_eq!(multi.as_slice().unwrap(), single.as_slice().unwrap());
    }

    #[test]
    fn reduce_axes_two_axes_3d_collapse_to_vector() {
        // Shape (2, 3, 4); reduce axes (0, 2) → length-3 result with the
        // axis-1 sums of the corresponding 8-element strips.
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = reduce_axes(&a, &[0, 2], 0.0, false, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[3]);

        // Hand-checked: for j in 0..3, sum over i,k of a[i,j,k] = sum of
        // values where flat = i*12 + j*4 + k.
        let expected: Vec<f64> = (0..3)
            .map(|j| {
                let mut s = 0.0;
                for i in 0..2 {
                    for k in 0..4 {
                        s += f64::from(i * 12 + j * 4 + k);
                    }
                }
                s
            })
            .collect();
        assert_eq!(r.as_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn reduce_axes_unsorted_axes_input_works() {
        // Pass axes in reverse — the implementation should sort them
        // internally and produce the same result.
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r1 = reduce_axes(&a, &[0, 2], 0.0, false, |acc, x| acc + x).unwrap();
        let r2 = reduce_axes(&a, &[2, 0], 0.0, false, |acc, x| acc + x).unwrap();
        assert_eq!(r1.shape(), r2.shape());
        assert_eq!(r1.as_slice().unwrap(), r2.as_slice().unwrap());
    }

    #[test]
    fn reduce_axes_keepdims_preserves_reduced_axes_as_size_1() {
        // (2, 3, 4) reducing axes (0, 2) with keepdims=true → (1, 3, 1).
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let r = reduce_axes(&a, &[0, 2], 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[1, 3, 1]);
    }

    #[test]
    fn reduce_axes_all_axes_collapses_to_scalar() {
        // All axes → length-1 result (the keepdims=false collapse convention).
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axes(&a, &[0, 1], 0.0, false, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[1]);
        assert_eq!(r.as_slice().unwrap(), &[21.0]);
    }

    #[test]
    fn reduce_axes_all_axes_keepdims_gives_size_1_per_axis() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axes(&a, &[0, 1], 0.0, true, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[1, 1]);
        assert_eq!(r.as_slice().unwrap(), &[21.0]);
    }

    #[test]
    fn reduce_axes_empty_axes_is_identity_copy() {
        // axis=() in NumPy returns the input unchanged.
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = reduce_axes(&a, &[], 0.0, false, |acc, x| acc + x).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn reduce_axes_duplicate_axis_errors() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(reduce_axes(&a, &[1, 1], 0.0, false, |x, y| x + y).is_err());
    }

    #[test]
    fn reduce_axes_out_of_bounds_errors() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(reduce_axes(&a, &[5], 0.0, false, |x, y| x + y).is_err());
        assert!(reduce_axes(&a, &[0, 5], 0.0, false, |x, y| x + y).is_err());
    }

    #[test]
    fn reduce_axes_chained_via_single_pass_matches_sequential() {
        // Sanity check: chaining single-axis reductions in the right order
        // (descending so axes don't shift) must produce the same numbers as
        // the single-pass reduce_axes implementation.
        use ferray_core::dimension::Ix3;
        let data: Vec<f64> = (0..60).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([3, 4, 5]), data).unwrap();

        let single_pass = reduce_axes(&a, &[0, 2], 0.0, false, |acc, x| acc + x).unwrap();

        // Reduce axis 2 first (rightmost), then axis 0 of the result.
        let step1 = reduce_axis(&a, 2, 0.0, |acc, x| acc + x).unwrap();
        let step2 = reduce_axis(&step1, 0, 0.0, |acc, x| acc + x).unwrap();

        assert_eq!(single_pass.shape(), step2.shape());
        for (a, b) in single_pass
            .as_slice()
            .unwrap()
            .iter()
            .zip(step2.as_slice().unwrap().iter())
        {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
    }

    #[test]
    fn reduce_all_sums_full_array() {
        let a = arr2(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = reduce_all(&a, 0.0, |acc, x| acc + x);
        assert!((s - 21.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_all_max_returns_global_max() {
        let a = arr2(2, 3, &[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let m = reduce_all(&a, f64::NEG_INFINITY, f64::max);
        assert!((m - 9.0).abs() < 1e-12);
    }

    #[test]
    fn reduce_all_empty_array_returns_identity() {
        // 0-element array → reduction returns the identity.
        use ferray_core::dimension::Ix1;
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let s = reduce_all(&a, 0.0, |acc, x| acc + x);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn accumulate_axis_add_1d() {
        let a = arr1([1.0, 2.0, 3.0, 4.0]);
        let r = accumulate_axis(&a, 0, |acc, x| acc + x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn accumulate_axis_multiply_1d() {
        let a = arr1([1.0, 2.0, 3.0, 4.0]);
        let r = accumulate_axis(&a, 0, |acc, x| acc * x).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn accumulate_axis_subtract_running_diff() {
        // subtract.accumulate([10, 3, 2, 1]) = [10, 10-3, (10-3)-2, ((10-3)-2)-1]
        //                                    = [10, 7, 5, 4]
        let a = arr1([10.0, 3.0, 2.0, 1.0]);
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
        let a = arr1([1.0, 2.0, 3.0]);
        assert!(accumulate_axis(&a, 1, |x, y| x + y).is_err());
    }

    #[test]
    fn outer_multiply() {
        let a = arr1([1.0, 2.0, 3.0]);
        let b = arr1([10.0, 20.0]);
        let r = outer(&a, &b, |x, y| x * y).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.as_slice().unwrap(), &[10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn outer_add() {
        let a = arr1([1.0, 2.0]);
        let b = arr1([10.0, 20.0, 30.0]);
        let r = outer(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(r.as_slice().unwrap(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);
    }

    #[test]
    fn outer_power() {
        let a = arr1([2.0, 3.0]);
        let b = arr1([2.0, 3.0]);
        let r = outer(&a, &b, f64::powf).unwrap();
        assert_eq!(r.shape(), &[2, 2]);
        // 2^2=4, 2^3=8, 3^2=9, 3^3=27
        assert_eq!(r.as_slice().unwrap(), &[4.0, 8.0, 9.0, 27.0]);
    }

    #[test]
    fn at_add_unbuffered_duplicates() {
        // Duplicate index 0 -> both writes apply (adding twice), unlike
        // simple indexed assignment which would only keep the last.
        let mut a = arr1([0.0, 0.0, 0.0]);
        at(&mut a, &[0, 0, 1, 2], &[1.0, 2.0, 5.0, 10.0], |acc, x| {
            acc + x
        })
        .unwrap();
        assert_eq!(a.as_slice().unwrap(), &[3.0, 5.0, 10.0]);
    }

    #[test]
    fn at_multiply() {
        let mut a = arr1([1.0, 1.0, 1.0, 1.0]);
        at(&mut a, &[1, 2, 2], &[5.0, 3.0, 4.0], |acc, x| acc * x).unwrap();
        assert_eq!(a.as_slice().unwrap(), &[1.0, 5.0, 12.0, 1.0]);
    }

    #[test]
    fn at_length_mismatch_errors() {
        let mut a = arr1([0.0; 4]);
        assert!(at(&mut a, &[0, 1], &[1.0], |x, y| x + y).is_err());
    }

    #[test]
    fn at_index_out_of_bounds_errors() {
        let mut a = arr1([0.0; 3]);
        assert!(at(&mut a, &[5], &[1.0], |x, y| x + y).is_err());
    }
}
