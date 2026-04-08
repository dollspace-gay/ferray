// ferray-stats: Core reductions — sum, prod, min, max, argmin, argmax, mean, var, std (REQ-1, REQ-2)
//
// `cumulative` used to exist as an empty placeholder whose only content
// was a note that cumulative functions live in nan_aware / mod.rs; it's
// been deleted to stop pretending there's a dedicated module there (#162).

pub mod nan_aware;
pub mod quantile;

use std::any::TypeId;

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use crate::parallel;

/// Try SIMD-accelerated fused sum of squared differences if T is f64.
/// Returns sum((x - mean)²) without allocating an intermediate Vec.
#[inline]
fn try_simd_sum_sq_diff<T: Element + Copy + 'static>(data: &[T], mean: T) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: TypeId check guarantees T is f64. size_of::<T>() == size_of::<f64>().
        let f64_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let mean_f64: f64 = unsafe { std::mem::transmute_copy(&mean) };
        let result = parallel::simd_sum_sq_diff_f64(f64_slice, mean_f64);
        Some(unsafe { std::mem::transmute_copy(&result) })
    } else {
        None
    }
}

/// Try SIMD-accelerated pairwise sum if T is f64.
/// Returns the sum transmuted back to T, or None if T is not f64.
#[inline]
fn try_simd_pairwise_sum<T: Element + Copy + 'static>(data: &[T]) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: TypeId check guarantees T is f64. size_of::<T>() == size_of::<f64>().
        let f64_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, data.len()) };
        let result = parallel::pairwise_sum_f64(f64_slice);
        Some(unsafe { std::mem::transmute_copy(&result) })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Internal axis-reduction helper
// ---------------------------------------------------------------------------

/// Compute row-major strides for a given shape.
pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Flat index from a multi-index given row-major strides.
pub(crate) fn flat_index(multi: &[usize], strides: &[usize]) -> usize {
    multi.iter().zip(strides.iter()).map(|(i, s)| i * s).sum()
}

/// Increment a multi-index in row-major order. Returns false if overflowed.
pub(crate) fn increment_multi_index(multi: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..multi.len()).rev() {
        multi[d] += 1;
        if multi[d] < shape[d] {
            return true;
        }
        multi[d] = 0;
    }
    false
}

/// General axis reduction parameterized on the output element type.
///
/// Walks `data` (in row-major order with `shape`), gathers each lane
/// along `axis` into a temporary `Vec<T>`, and calls `f` to collapse it
/// to an output of type `U`. Returns the concatenated outputs in
/// row-major order for the shape with `axis` removed.
///
/// Used as the shared backbone for `reduce_axis_general` (T=T path) and
/// `reduce_axis_general_u64` (T=T, U=u64 path). The two used to have
/// copy-pasted bodies differing only in return type (#161).
pub(crate) fn reduce_axis_typed<T, U, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f: F,
) -> Vec<U>
where
    T: Copy,
    F: Fn(&[T]) -> U,
{
    let ndim = shape.len();
    let axis_len = shape[axis];
    let strides = compute_strides(shape);

    // Output shape: shape with axis removed
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    let mut result = Vec::with_capacity(out_size);
    let mut out_multi = vec![0usize; out_shape.len()];
    let mut in_multi = vec![0usize; ndim];
    let mut lane_vec: Vec<T> = Vec::with_capacity(axis_len);

    for _ in 0..out_size {
        // Build input multi-index by inserting axis position
        let mut out_dim = 0;
        for (d, idx) in in_multi.iter_mut().enumerate() {
            if d == axis {
                *idx = 0;
            } else {
                *idx = out_multi[out_dim];
                out_dim += 1;
            }
        }

        // Gather lane values
        lane_vec.clear();
        for k in 0..axis_len {
            in_multi[axis] = k;
            let idx = flat_index(&in_multi, &strides);
            lane_vec.push(data[idx]);
        }

        result.push(f(&lane_vec));

        // Increment output multi-index
        if !out_shape.is_empty() {
            increment_multi_index(&mut out_multi, &out_shape);
        }
    }

    result
}

/// Thin wrapper: `T -> T` reduction. Preserved for the many call sites
/// that pass `Fn(&[T]) -> T` kernels.
#[inline]
pub(crate) fn reduce_axis_general<T, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f: F,
) -> Vec<T>
where
    T: Copy,
    F: Fn(&[T]) -> T,
{
    reduce_axis_typed(data, shape, axis, f)
}

/// Validate axis parameter and return an error if out of bounds.
pub(crate) fn validate_axis(axis: usize, ndim: usize) -> FerrayResult<()> {
    if axis >= ndim {
        Err(FerrayError::axis_out_of_bounds(axis, ndim))
    } else {
        Ok(())
    }
}

/// Collect array data into a contiguous Vec in logical (row-major) order.
pub(crate) fn collect_data<T: Element + Copy, D: Dimension>(a: &Array<T, D>) -> Vec<T> {
    a.iter().copied().collect()
}

/// Borrow contiguous data or copy if strided. Avoids allocation for contiguous arrays.
pub(crate) enum DataRef<'a, T> {
    Borrowed(&'a [T]),
    Owned(Vec<T>),
}

impl<T> std::ops::Deref for DataRef<'_, T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            DataRef::Borrowed(s) => s,
            DataRef::Owned(v) => v,
        }
    }
}

/// Get a reference to contiguous data, or copy if strided.
pub(crate) fn borrow_data<'a, T: Element + Copy, D: Dimension>(
    a: &'a Array<T, D>,
) -> DataRef<'a, T> {
    if let Some(slice) = a.as_slice() {
        DataRef::Borrowed(slice)
    } else {
        DataRef::Owned(a.iter().copied().collect())
    }
}

/// Build an IxDyn result array from output shape and data.
pub(crate) fn make_result<T: Element>(
    out_shape: &[usize],
    data: Vec<T>,
) -> FerrayResult<Array<T, IxDyn>> {
    Array::from_vec(IxDyn::new(out_shape), data)
}

/// Compute the output shape when reducing along an axis.
pub(crate) fn output_shape(shape: &[usize], axis: usize) -> Vec<usize> {
    shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect()
}

// ---------------------------------------------------------------------------
// Multi-axis reduction helpers (issues #457 + #458)
// ---------------------------------------------------------------------------

/// Normalize the `axes: Option<&[usize]>` argument of a multi-axis reduction:
///
/// - `None` or `Some(&[])` expands to all axes `[0..ndim]` (reduce everything)
/// - Duplicate axes are an error (matches NumPy's `np.sum(a, axis=(0, 0))`)
/// - Any out-of-bounds axis is an error
///
/// Returns the sorted, unique axis list.
pub(crate) fn normalize_axes(axes: Option<&[usize]>, ndim: usize) -> FerrayResult<Vec<usize>> {
    let ax: Vec<usize> = match axes {
        None | Some([]) => (0..ndim).collect(),
        Some(s) => s.to_vec(),
    };
    for &a in &ax {
        if a >= ndim {
            return Err(FerrayError::axis_out_of_bounds(a, ndim));
        }
    }
    let mut sorted = ax.clone();
    sorted.sort_unstable();
    for w in sorted.windows(2) {
        if w[0] == w[1] {
            return Err(FerrayError::invalid_value(format!(
                "duplicate axis {} in reduction axes",
                w[0]
            )));
        }
    }
    Ok(sorted)
}

/// Compute the output shape when reducing over multiple axes.
///
/// `axes` must be sorted and unique (produced by [`normalize_axes`]).
/// With `keepdims = true`, reduced axes are replaced by size 1; otherwise
/// they are removed.
pub(crate) fn output_shape_axes(shape: &[usize], axes: &[usize], keepdims: bool) -> Vec<usize> {
    if keepdims {
        shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if axes.contains(&i) { 1 } else { s })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, &s)| s)
            .collect()
    }
}

/// Reduce over a set of axes, returning `(result_data, output_shape)`.
///
/// `axes` must be sorted and unique (typically produced by [`normalize_axes`]).
/// The caller supplies a reduction function `f` that collapses a lane of
/// reduced-axis values to a single scalar.
///
/// The output shape honors `keepdims`.
pub(crate) fn reduce_axes_general<T: Copy, F: Fn(&[T]) -> T>(
    data: &[T],
    shape: &[usize],
    axes: &[usize],
    keepdims: bool,
    f: F,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let strides = compute_strides(shape);

    // Partition axes into reduce / keep.
    let is_reduce: Vec<bool> = (0..ndim).map(|i| axes.contains(&i)).collect();
    let keep_axes: Vec<usize> = (0..ndim).filter(|i| !is_reduce[*i]).collect();
    let reduce_axes: Vec<usize> = (0..ndim).filter(|i| is_reduce[*i]).collect();
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&i| shape[i]).collect();
    let reduce_shape: Vec<usize> = reduce_axes.iter().map(|&i| shape[i]).collect();

    let out_size: usize = if keep_shape.is_empty() {
        1
    } else {
        keep_shape.iter().product()
    };
    let lane_size: usize = if reduce_shape.is_empty() {
        1
    } else {
        reduce_shape.iter().product()
    };

    let out_shape = output_shape_axes(shape, axes, keepdims);

    let mut result = Vec::with_capacity(out_size);
    let mut lane: Vec<T> = Vec::with_capacity(lane_size);
    let mut keep_multi = vec![0usize; keep_shape.len()];
    let mut reduce_multi = vec![0usize; reduce_shape.len()];
    let mut full_multi = vec![0usize; ndim];

    for _ in 0..out_size {
        // Fill kept-axis positions from keep_multi.
        for (i, &ax) in keep_axes.iter().enumerate() {
            full_multi[ax] = keep_multi[i];
        }

        // Gather values along all reduced axes.
        lane.clear();
        reduce_multi.iter_mut().for_each(|v| *v = 0);
        for _ in 0..lane_size {
            for (i, &ax) in reduce_axes.iter().enumerate() {
                full_multi[ax] = reduce_multi[i];
            }
            lane.push(data[flat_index(&full_multi, &strides)]);
            if !reduce_shape.is_empty() {
                increment_multi_index(&mut reduce_multi, &reduce_shape);
            }
        }

        result.push(f(&lane));

        if !keep_shape.is_empty() {
            increment_multi_index(&mut keep_multi, &keep_shape);
        }
    }

    (result, out_shape)
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------

/// Sum of array elements over a given axis, or over all elements if axis is None.
///
/// Equivalent to `numpy.sum`.
///
/// **Note:** Unlike NumPy, which auto-promotes `int32` sums to `int64`,
/// ferray returns the same type as the input. For large integer arrays
/// this may overflow. Use [`sum_as_f64`] for overflow-safe integer summation.
///
/// # Examples
/// ```ignore
/// let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let s = sum(&a, None).unwrap();
/// assert_eq!(s.iter().next(), Some(&10.0));
/// ```
pub fn sum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let total = try_simd_pairwise_sum(&data)
                .unwrap_or_else(|| parallel::parallel_sum(&data, <T as Element>::zero()));
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                try_simd_pairwise_sum(lane)
                    .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()))
            });
            make_result(&out_s, result)
        }
    }
}

/// Sum of array elements, returning `f64` regardless of input type.
///
/// This works on integer arrays (i32, u64, etc.) without overflow risk.
/// The result is always `Array<f64, IxDyn>`, matching NumPy's behavior
/// of promoting integer sums to a wider type.
pub fn sum_as_f64<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<f64, IxDyn>>
where
    T: Element + Copy + Send + Sync + num_traits::ToPrimitive,
    D: Dimension,
{
    match axis {
        None => {
            let total: f64 = a.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let f64_data: Vec<f64> = a.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
            let result = reduce_axis_general(&f64_data, shape, ax, |lane| {
                lane.iter().sum()
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// prod
// ---------------------------------------------------------------------------

/// Product of array elements over a given axis.
///
/// **Note:** Unlike NumPy, which auto-promotes integer products,
/// ferray returns the same type as the input. For large integer arrays
/// this may overflow.
/// Equivalent to `numpy.prod`.
pub fn prod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let total = parallel::parallel_prod(&data, <T as Element>::one());
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                lane.iter()
                    .copied()
                    .fold(<T as Element>::one(), |acc, x| acc * x)
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// min / max
// ---------------------------------------------------------------------------

/// Minimum value of array elements over a given axis.
///
/// Equivalent to `numpy.min` / `numpy.amin`.
pub fn min<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute min of empty array",
        ));
    }
    // NaN-propagating min: if either operand is NaN (comparison returns false
    // for both orderings), propagate NaN to match NumPy behavior.
    let nan_min = |a: T, b: T| -> T {
        if a <= b {
            a
        } else if a > b {
            b
        } else {
            // One of them is NaN — return whichever is unordered
            // (if a is NaN, a <= b and a > b are both false; return a)
            a
        }
    };
    let data = borrow_data(a);
    match axis {
        None => {
            let m = data.iter().copied().reduce(nan_min).unwrap();
            make_result(&[], vec![m])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                lane.iter().copied().reduce(nan_min).unwrap()
            });
            make_result(&out_s, result)
        }
    }
}

/// Maximum value of array elements over a given axis.
///
/// Equivalent to `numpy.max` / `numpy.amax`.
pub fn max<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute max of empty array",
        ));
    }
    // NaN-propagating max: same logic as min but reversed ordering.
    let nan_max = |a: T, b: T| -> T {
        if a >= b {
            a
        } else if a < b {
            b
        } else {
            a
        }
    };
    let data = borrow_data(a);
    match axis {
        None => {
            let m = data.iter().copied().reduce(nan_max).unwrap();
            make_result(&[], vec![m])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                lane.iter().copied().reduce(nan_max).unwrap()
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// argmin / argmax
// ---------------------------------------------------------------------------

/// Index of the minimum value. For axis=None, returns the flat index.
/// For axis=Some(ax), returns indices along that axis.
///
/// Equivalent to `numpy.argmin`.
pub fn argmin<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute argmin of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let idx = data
                .iter()
                .enumerate()
                .reduce(|(ai, av), (bi, bv)| if av <= bv { (ai, av) } else { (bi, bv) })
                .unwrap()
                .0 as u64;
            make_result(&[], vec![idx])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general_u64(&data, shape, ax, |lane| {
                lane.iter()
                    .enumerate()
                    .reduce(|(ai, av), (bi, bv)| if av <= bv { (ai, av) } else { (bi, bv) })
                    .unwrap()
                    .0 as u64
            });
            make_result(&out_s, result)
        }
    }
}

/// Index of the maximum value.
///
/// Equivalent to `numpy.argmax`.
pub fn argmax<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute argmax of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let idx = data
                .iter()
                .enumerate()
                .reduce(|(ai, av), (bi, bv)| if av >= bv { (ai, av) } else { (bi, bv) })
                .unwrap()
                .0 as u64;
            make_result(&[], vec![idx])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general_u64(&data, shape, ax, |lane| {
                lane.iter()
                    .enumerate()
                    .reduce(|(ai, av), (bi, bv)| if av >= bv { (ai, av) } else { (bi, bv) })
                    .unwrap()
                    .0 as u64
            });
            make_result(&out_s, result)
        }
    }
}

/// Thin wrapper: `T -> u64` reduction (for `argmin`/`argmax` /
/// `bincount` paths). Shares its body with `reduce_axis_general`
/// via `reduce_axis_typed`.
#[inline]
pub(crate) fn reduce_axis_general_u64<T, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    f: F,
) -> Vec<u64>
where
    T: Copy,
    F: Fn(&[T]) -> u64,
{
    reduce_axis_typed(data, shape, axis, f)
}

// ---------------------------------------------------------------------------
// mean
// ---------------------------------------------------------------------------

/// Mean of array elements over a given axis.
///
/// Equivalent to `numpy.mean`. The result is always floating-point.
pub fn mean<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute mean of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let n = T::from(data.len()).unwrap();
            let total = try_simd_pairwise_sum(&data)
                .unwrap_or_else(|| parallel::pairwise_sum(&data, <T as Element>::zero()));
            make_result(&[], vec![total / n])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let axis_len = shape[ax];
            let n = T::from(axis_len).unwrap();
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                let total = try_simd_pairwise_sum(lane)
                    .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()));
                total / n
            });
            make_result(&out_s, result)
        }
    }
}

/// Mean of array elements, returning `f64` regardless of input type.
///
/// This works on integer arrays (i32, u64, etc.) where [`mean`] would
/// fail because integers don't implement `Float`. The result is always
/// `Array<f64, IxDyn>`, matching NumPy's behavior of promoting integer
/// means to float64.
///
/// Equivalent to `numpy.mean` for integer inputs.
pub fn mean_as_f64<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<f64, IxDyn>>
where
    T: Element + Copy + Send + Sync + num_traits::ToPrimitive,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute mean of empty array",
        ));
    }
    match axis {
        None => {
            let n = a.size() as f64;
            let total: f64 = a.iter().map(|x| x.to_f64().unwrap_or(0.0)).sum();
            make_result(&[], vec![total / n])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let axis_len = shape[ax] as f64;
            let f64_data: Vec<f64> = a.iter().map(|x| x.to_f64().unwrap_or(0.0)).collect();
            let result = reduce_axis_general(&f64_data, shape, ax, |lane| {
                let total: f64 = lane.iter().sum();
                total / axis_len
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// var
// ---------------------------------------------------------------------------

/// Variance of array elements over a given axis.
///
/// `ddof` is the delta degrees of freedom (0 for population variance, 1 for sample).
/// Equivalent to `numpy.var`.
pub fn var<T, D>(a: &Array<T, D>, axis: Option<usize>, ddof: usize) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute variance of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let n = data.len();
            if n <= ddof {
                return Err(FerrayError::invalid_value(
                    "ddof >= number of elements, variance undefined",
                ));
            }
            let nf = T::from(n).unwrap();
            let mean_val = try_simd_pairwise_sum(&data)
                .unwrap_or_else(|| parallel::pairwise_sum(&data, <T as Element>::zero()))
                / nf;
            let sum_sq = try_simd_sum_sq_diff(&data, mean_val).unwrap_or_else(|| {
                data.iter().copied().fold(<T as Element>::zero(), |acc, x| {
                    let d = x - mean_val;
                    acc + d * d
                })
            });
            let var_val = sum_sq / T::from(n - ddof).unwrap();
            make_result(&[], vec![var_val])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let axis_len = shape[ax];
            if axis_len <= ddof {
                return Err(FerrayError::invalid_value(
                    "ddof >= axis length, variance undefined",
                ));
            }
            let nf = T::from(axis_len).unwrap();
            let denom = T::from(axis_len - ddof).unwrap();
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                let mean_val = try_simd_pairwise_sum(lane)
                    .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()))
                    / nf;
                let sum_sq = try_simd_sum_sq_diff(lane, mean_val).unwrap_or_else(|| {
                    lane.iter().copied().fold(<T as Element>::zero(), |acc, x| {
                        let d = x - mean_val;
                        acc + d * d
                    })
                });
                sum_sq / denom
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// std_
// ---------------------------------------------------------------------------

/// Standard deviation of array elements over a given axis.
///
/// `ddof` is the delta degrees of freedom.
/// Equivalent to `numpy.std`.
pub fn std_<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    let v = var(a, axis, ddof)?;
    let data: Vec<T> = v.iter().map(|x| x.sqrt()).collect();
    make_result(v.shape(), data)
}

// ---------------------------------------------------------------------------
// Multi-axis + keepdims public API (issues #457 + #458)
//
// These `*_axes` variants complement the existing single-axis reductions
// with two additional features:
//   1. `axes: Option<&[usize]>` — reduce over multiple axes at once
//      (None or empty slice means "reduce all axes")
//   2. `keepdims: bool` — preserve reduced axes as size 1 so the result
//      can broadcast back against the original array
//
// The existing single-axis `sum/prod/etc.` functions remain unchanged.
// ---------------------------------------------------------------------------

/// Multi-axis sum with optional `keepdims`.
///
/// Equivalent to `numpy.sum(a, axis=axes, keepdims=keepdims)`. If `axes` is
/// `None` or an empty slice, reduces over all axes.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` on any out-of-range axis, or
/// `FerrayError::InvalidValue` on duplicate axes.
pub fn sum_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let ax = normalize_axes(axes, a.ndim())?;
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, a.shape(), &ax, keepdims, |lane| {
        try_simd_pairwise_sum(lane)
            .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()))
    });
    make_result(&out_shape, result)
}

/// Multi-axis product with optional `keepdims`.
pub fn prod_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let ax = normalize_axes(axes, a.ndim())?;
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, a.shape(), &ax, keepdims, |lane| {
        lane.iter()
            .copied()
            .fold(<T as Element>::one(), |acc, x| acc * x)
    });
    make_result(&out_shape, result)
}

/// Multi-axis minimum with optional `keepdims`. NaN-propagating.
pub fn min_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute min of empty array",
        ));
    }
    let ax = normalize_axes(axes, a.ndim())?;
    let nan_min = |a: T, b: T| -> T {
        if a <= b {
            a
        } else if a > b {
            b
        } else {
            a // NaN propagates
        }
    };
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, a.shape(), &ax, keepdims, |lane| {
        lane.iter().copied().reduce(nan_min).unwrap()
    });
    make_result(&out_shape, result)
}

/// Multi-axis maximum with optional `keepdims`. NaN-propagating.
pub fn max_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute max of empty array",
        ));
    }
    let ax = normalize_axes(axes, a.ndim())?;
    let nan_max = |a: T, b: T| -> T {
        if a >= b {
            a
        } else if a < b {
            b
        } else {
            a // NaN propagates
        }
    };
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, a.shape(), &ax, keepdims, |lane| {
        lane.iter().copied().reduce(nan_max).unwrap()
    });
    make_result(&out_shape, result)
}

/// Multi-axis mean with optional `keepdims`.
pub fn mean_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute mean of empty array",
        ));
    }
    let ax = normalize_axes(axes, a.ndim())?;
    // Compute n = product of reduced-axis lengths.
    let shape = a.shape();
    let lane_len: usize = ax.iter().map(|&i| shape[i]).product();
    let n = T::from(lane_len).unwrap();
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, shape, &ax, keepdims, |lane| {
        let total = try_simd_pairwise_sum(lane)
            .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()));
        total / n
    });
    make_result(&out_shape, result)
}

/// Multi-axis variance with optional `keepdims` and Bessel correction `ddof`.
pub fn var_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    ddof: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute variance of empty array",
        ));
    }
    let ax = normalize_axes(axes, a.ndim())?;
    let shape = a.shape();
    let lane_len: usize = ax.iter().map(|&i| shape[i]).product();
    if lane_len <= ddof {
        return Err(FerrayError::invalid_value(
            "ddof >= reduced-axis length, variance undefined",
        ));
    }
    let nf = T::from(lane_len).unwrap();
    let denom = T::from(lane_len - ddof).unwrap();
    let data = borrow_data(a);
    let (result, out_shape) = reduce_axes_general(&data, shape, &ax, keepdims, |lane| {
        let mean_val = try_simd_pairwise_sum(lane)
            .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()))
            / nf;
        let sum_sq = try_simd_sum_sq_diff(lane, mean_val).unwrap_or_else(|| {
            lane.iter().copied().fold(<T as Element>::zero(), |acc, x| {
                let d = x - mean_val;
                acc + d * d
            })
        });
        sum_sq / denom
    });
    make_result(&out_shape, result)
}

/// Multi-axis standard deviation with optional `keepdims`.
pub fn std_axes<T, D>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
    ddof: usize,
    keepdims: bool,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    let v = var_axes(a, axes, ddof, keepdims)?;
    let data: Vec<T> = v.iter().map(|x| x.sqrt()).collect();
    make_result(v.shape(), data)
}

/// Single-axis argmin with optional `keepdims`.
///
/// NumPy's `argmin` only accepts a single axis (or `None`); this mirrors
/// that constraint. `keepdims` preserves the reduced axis as size 1.
pub fn argmin_keepdims<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    keepdims: bool,
) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute argmin of empty array",
        ));
    }
    let ndim = a.ndim();
    let ax_vec: Vec<usize> = match axis {
        None => (0..ndim).collect(),
        Some(ax) => {
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }
            vec![ax]
        }
    };
    let shape = a.shape();
    let data = borrow_data(a);
    let (result_f64, out_shape) =
        reduce_axes_general_u64(&data, shape, &ax_vec, keepdims, |lane| {
            lane.iter()
                .enumerate()
                .reduce(|(ai, av), (bi, bv)| if av <= bv { (ai, av) } else { (bi, bv) })
                .unwrap()
                .0 as u64
        });
    make_result(&out_shape, result_f64)
}

/// Single-axis argmax with optional `keepdims`.
pub fn argmax_keepdims<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    keepdims: bool,
) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute argmax of empty array",
        ));
    }
    let ndim = a.ndim();
    let ax_vec: Vec<usize> = match axis {
        None => (0..ndim).collect(),
        Some(ax) => {
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }
            vec![ax]
        }
    };
    let shape = a.shape();
    let data = borrow_data(a);
    let (result_u64, out_shape) =
        reduce_axes_general_u64(&data, shape, &ax_vec, keepdims, |lane| {
            lane.iter()
                .enumerate()
                .reduce(|(ai, av), (bi, bv)| if av >= bv { (ai, av) } else { (bi, bv) })
                .unwrap()
                .0 as u64
        });
    make_result(&out_shape, result_u64)
}

/// Multi-axis reduction that returns `u64` values (used by argmin/argmax variants).
pub(crate) fn reduce_axes_general_u64<T: Copy, F: Fn(&[T]) -> u64>(
    data: &[T],
    shape: &[usize],
    axes: &[usize],
    keepdims: bool,
    f: F,
) -> (Vec<u64>, Vec<usize>) {
    let ndim = shape.len();
    let strides = compute_strides(shape);

    let is_reduce: Vec<bool> = (0..ndim).map(|i| axes.contains(&i)).collect();
    let keep_axes: Vec<usize> = (0..ndim).filter(|i| !is_reduce[*i]).collect();
    let reduce_axes: Vec<usize> = (0..ndim).filter(|i| is_reduce[*i]).collect();
    let keep_shape: Vec<usize> = keep_axes.iter().map(|&i| shape[i]).collect();
    let reduce_shape: Vec<usize> = reduce_axes.iter().map(|&i| shape[i]).collect();

    let out_size: usize = if keep_shape.is_empty() {
        1
    } else {
        keep_shape.iter().product()
    };
    let lane_size: usize = if reduce_shape.is_empty() {
        1
    } else {
        reduce_shape.iter().product()
    };
    let out_shape = output_shape_axes(shape, axes, keepdims);

    let mut result = Vec::with_capacity(out_size);
    let mut lane: Vec<T> = Vec::with_capacity(lane_size);
    let mut keep_multi = vec![0usize; keep_shape.len()];
    let mut reduce_multi = vec![0usize; reduce_shape.len()];
    let mut full_multi = vec![0usize; ndim];

    for _ in 0..out_size {
        for (i, &ax) in keep_axes.iter().enumerate() {
            full_multi[ax] = keep_multi[i];
        }

        lane.clear();
        reduce_multi.iter_mut().for_each(|v| *v = 0);
        for _ in 0..lane_size {
            for (i, &ax) in reduce_axes.iter().enumerate() {
                full_multi[ax] = reduce_multi[i];
            }
            lane.push(data[flat_index(&full_multi, &strides)]);
            if !reduce_shape.is_empty() {
                increment_multi_index(&mut reduce_multi, &reduce_shape);
            }
        }

        result.push(f(&lane));

        if !keep_shape.is_empty() {
            increment_multi_index(&mut keep_multi, &keep_shape);
        }
    }

    (result, out_shape)
}

// ---------------------------------------------------------------------------
// Re-export cumulative operations from ferray-ufunc for discoverability
// ---------------------------------------------------------------------------

/// Cumulative sum along an axis (or flattened if axis is None).
///
/// Re-exported from `ferray_ufunc::cumsum` for convenience.
pub fn cumsum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    ferray_ufunc::cumsum(a, axis)
}

/// Cumulative product along an axis (or flattened if axis is None).
///
/// Re-exported from `ferray_ufunc::cumprod` for convenience.
pub fn cumprod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    ferray_ufunc::cumprod(a, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::{Ix1, Ix2};

    #[test]
    fn test_sum_1d_no_axis() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s = sum(&a, None).unwrap();
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.iter().next(), Some(&10.0));
    }

    #[test]
    fn test_sum_2d_axis0() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let s = sum(&a, Some(0)).unwrap();
        assert_eq!(s.shape(), &[3]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sum_2d_axis1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let s = sum(&a, Some(1)).unwrap();
        assert_eq!(s.shape(), &[2]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![6.0, 15.0]);
    }

    #[test]
    fn test_prod_1d() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let p = prod(&a, None).unwrap();
        assert_eq!(p.iter().next(), Some(&24.0));
    }

    #[test]
    fn test_min_max() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        let mn = min(&a, None).unwrap();
        let mx = max(&a, None).unwrap();
        assert_eq!(mn.iter().next(), Some(&1.0));
        assert_eq!(mx.iter().next(), Some(&4.0));
    }

    #[test]
    fn test_argmin_argmax() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        let ami = argmin(&a, None).unwrap();
        let amx = argmax(&a, None).unwrap();
        assert_eq!(ami.iter().next(), Some(&1u64));
        assert_eq!(amx.iter().next(), Some(&2u64));
    }

    #[test]
    fn test_mean() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let m = mean(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_var_population() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = var(&a, None, 0).unwrap();
        // var = ((1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2) / 4 = 1.25
        assert!((v.iter().next().unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn test_var_sample() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let v = var(&a, None, 1).unwrap();
        // var = 5.0 / 3.0 = 1.6666...
        assert!((v.iter().next().unwrap() - 5.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_std() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let s = std_(&a, None, 1).unwrap();
        let expected = (5.0_f64 / 3.0).sqrt();
        assert!((s.iter().next().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_sum_axis_out_of_bounds() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(sum(&a, Some(1)).is_err());
    }

    #[test]
    fn test_cumsum_reexport() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let cs = cumsum(&a, None).unwrap();
        assert_eq!(cs.as_slice().unwrap(), &[1, 3, 6, 10]);
    }

    #[test]
    fn test_cumprod_reexport() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let cp = cumprod(&a, None).unwrap();
        assert_eq!(cp.as_slice().unwrap(), &[1, 2, 6, 24]);
    }

    #[test]
    fn test_min_2d_axis0() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0])
            .unwrap();
        let m = min(&a, Some(0)).unwrap();
        let data: Vec<f64> = m.iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 2.0]);
    }

    #[test]
    fn test_argmin_2d_axis1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0])
            .unwrap();
        let ami = argmin(&a, Some(1)).unwrap();
        let data: Vec<u64> = ami.iter().copied().collect();
        assert_eq!(data, vec![1, 0]);
    }

    #[test]
    fn test_mean_2d_axis0() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let m = mean(&a, Some(0)).unwrap();
        let data: Vec<f64> = m.iter().copied().collect();
        assert_eq!(data, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_sum_integer() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let s = sum(&a, None).unwrap();
        assert_eq!(s.iter().next(), Some(&15));
    }

    #[test]
    fn test_mean_as_f64_integer() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let m = mean_as_f64(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_mean_as_f64_integer_axis() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let m = mean_as_f64(&a, Some(1)).unwrap();
        assert_eq!(m.shape(), &[2]);
        let data: Vec<f64> = m.iter().copied().collect();
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_as_f64_u8() {
        let a = Array::<u8, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let m = mean_as_f64(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 25.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_as_f64_integer() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let s = sum_as_f64(&a, None).unwrap();
        assert!((s.iter().next().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_as_f64_large_values() {
        // Values that would overflow i32 if summed as i32
        let a = Array::<i32, Ix1>::from_vec(
            Ix1::new([3]),
            vec![i32::MAX, i32::MAX, i32::MAX],
        )
        .unwrap();
        let s = sum_as_f64(&a, None).unwrap();
        let expected = 3.0 * (i32::MAX as f64);
        assert!((s.iter().next().unwrap() - expected).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // Multi-axis + keepdims tests (issues #457 + #458)
    // -----------------------------------------------------------------------

    use ferray_core::Ix3;

    fn arr2d(rows: usize, cols: usize, data: Vec<f64>) -> Array<f64, Ix2> {
        Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap()
    }

    fn arr3d(s0: usize, s1: usize, s2: usize, data: Vec<f64>) -> Array<f64, Ix3> {
        Array::<f64, Ix3>::from_vec(Ix3::new([s0, s1, s2]), data).unwrap()
    }

    #[test]
    fn test_normalize_axes_none_reduces_all() {
        let ax = normalize_axes(None, 3).unwrap();
        assert_eq!(ax, vec![0, 1, 2]);
    }

    #[test]
    fn test_normalize_axes_empty_reduces_all() {
        let ax = normalize_axes(Some(&[]), 3).unwrap();
        assert_eq!(ax, vec![0, 1, 2]);
    }

    #[test]
    fn test_normalize_axes_sorts() {
        let ax = normalize_axes(Some(&[2, 0]), 3).unwrap();
        assert_eq!(ax, vec![0, 2]);
    }

    #[test]
    fn test_normalize_axes_rejects_duplicate() {
        assert!(normalize_axes(Some(&[0, 0]), 3).is_err());
    }

    #[test]
    fn test_normalize_axes_rejects_out_of_bounds() {
        assert!(normalize_axes(Some(&[3]), 3).is_err());
    }

    #[test]
    fn test_sum_axes_single_axis_matches_legacy() {
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let legacy = sum(&a, Some(0)).unwrap();
        let new = sum_axes(&a, Some(&[0]), false).unwrap();
        assert_eq!(legacy.shape(), new.shape());
        let la: Vec<f64> = legacy.iter().copied().collect();
        let na: Vec<f64> = new.iter().copied().collect();
        assert_eq!(la, na);
    }

    #[test]
    fn test_sum_axes_keepdims_2d() {
        // sum((2, 3), axis=1, keepdims=True) -> shape (2, 1)
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = sum_axes(&a, Some(&[1]), true).unwrap();
        assert_eq!(s.shape(), &[2, 1]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![6.0, 15.0]);
    }

    #[test]
    fn test_sum_axes_keepdims_supports_broadcast_back() {
        // Canonical NumPy pattern: arr - arr.mean(axis=1, keepdims=True)
        // We do it with a 2x3 array.
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let m = mean_axes(&a, Some(&[1]), true).unwrap();
        assert_eq!(m.shape(), &[2, 1]);
        // Row 0 mean = 2.0, row 1 mean = 5.0
        let md: Vec<f64> = m.iter().copied().collect();
        assert_eq!(md, vec![2.0, 5.0]);
    }

    #[test]
    fn test_sum_axes_multi_axis_3d() {
        // shape (2, 3, 4), sum over axes (0, 2) -> shape (3,)
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = arr3d(2, 3, 4, data);
        let s = sum_axes(&a, Some(&[0, 2]), false).unwrap();
        assert_eq!(s.shape(), &[3]);
        // For each j in [0,1,2], sum over i in [0,1] and k in [0,1,2,3]:
        //   a[i,j,k] = i*12 + j*4 + k
        //   sum = (0*12 + j*4) + ... + (0*12 + j*4+3) + (1*12 + j*4) + ... + (1*12 + j*4+3)
        //       = 2*(4*j*4 + 6) + 2*12*1
        // For j=0: (0+1+2+3) + (12+13+14+15) = 6 + 54 = 60
        // For j=1: (4+5+6+7) + (16+17+18+19) = 22 + 70 = 92
        // For j=2: (8+9+10+11) + (20+21+22+23) = 38 + 86 = 124
        let d: Vec<f64> = s.iter().copied().collect();
        assert_eq!(d, vec![60.0, 92.0, 124.0]);
    }

    #[test]
    fn test_sum_axes_multi_axis_keepdims_3d() {
        // Same as above but keepdims -> shape (1, 3, 1)
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = arr3d(2, 3, 4, data);
        let s = sum_axes(&a, Some(&[0, 2]), true).unwrap();
        assert_eq!(s.shape(), &[1, 3, 1]);
        let d: Vec<f64> = s.iter().copied().collect();
        assert_eq!(d, vec![60.0, 92.0, 124.0]);
    }

    #[test]
    fn test_sum_axes_none_reduces_all() {
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = sum_axes(&a, None, false).unwrap();
        assert_eq!(s.shape(), &[]);
        assert_eq!(s.iter().next(), Some(&21.0));
    }

    #[test]
    fn test_sum_axes_none_keepdims() {
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = sum_axes(&a, None, true).unwrap();
        assert_eq!(s.shape(), &[1, 1]);
        assert_eq!(s.iter().next(), Some(&21.0));
    }

    #[test]
    fn test_prod_axes_keepdims() {
        let a = arr2d(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = prod_axes(&a, Some(&[1]), true).unwrap();
        assert_eq!(p.shape(), &[2, 1]);
        let d: Vec<f64> = p.iter().copied().collect();
        assert_eq!(d, vec![6.0, 120.0]);
    }

    #[test]
    fn test_min_max_axes_multi_axis() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = arr3d(2, 3, 4, data);
        let mn = min_axes(&a, Some(&[0, 2]), false).unwrap();
        let mx = max_axes(&a, Some(&[0, 2]), false).unwrap();
        // min over i=0..2, k=0..4 for each j: min value on i=0 has smallest
        // indices. For j=0, values are 0..3 and 12..15 -> min=0, max=15
        assert_eq!(mn.iter().copied().collect::<Vec<_>>(), vec![0.0, 4.0, 8.0]);
        assert_eq!(
            mx.iter().copied().collect::<Vec<_>>(),
            vec![15.0, 19.0, 23.0]
        );
    }

    #[test]
    fn test_mean_axes_multi_axis() {
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let a = arr3d(2, 3, 4, data);
        let m = mean_axes(&a, Some(&[0, 2]), false).unwrap();
        assert_eq!(m.shape(), &[3]);
        // For j=0: 60 / 8 = 7.5
        // For j=1: 92 / 8 = 11.5
        // For j=2: 124 / 8 = 15.5
        let d: Vec<f64> = m.iter().copied().collect();
        assert_eq!(d, vec![7.5, 11.5, 15.5]);
    }

    #[test]
    fn test_var_axes_single_axis_keepdims() {
        let a = arr2d(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Population variance along axis 1 (each row), keepdims = (2, 1)
        let v = var_axes(&a, Some(&[1]), 0, true).unwrap();
        assert_eq!(v.shape(), &[2, 1]);
        // Row 0: mean=2.5, var=((1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2)/4 = 1.25
        // Row 1: same distribution = 1.25
        let d: Vec<f64> = v.iter().copied().collect();
        assert!((d[0] - 1.25).abs() < 1e-12);
        assert!((d[1] - 1.25).abs() < 1e-12);
    }

    #[test]
    fn test_std_axes_multi_axis() {
        let a = arr2d(2, 4, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        // Population std over all: mean=4.5, var=5.25, std≈2.2913
        let s = std_axes(&a, None, 0, false).unwrap();
        assert_eq!(s.shape(), &[]);
        let v = *s.iter().next().unwrap();
        assert!((v - 5.25_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_argmin_keepdims_single_axis() {
        let a = arr2d(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0]);
        let am = argmin_keepdims(&a, Some(1), true).unwrap();
        assert_eq!(am.shape(), &[2, 1]);
        let d: Vec<u64> = am.iter().copied().collect();
        assert_eq!(d, vec![1, 0]);
    }

    #[test]
    fn test_argmax_keepdims_single_axis() {
        let a = arr2d(2, 3, vec![3.0, 1.0, 4.0, 1.0, 5.0, 2.0]);
        let am = argmax_keepdims(&a, Some(1), false).unwrap();
        assert_eq!(am.shape(), &[2]);
        let d: Vec<u64> = am.iter().copied().collect();
        assert_eq!(d, vec![2, 1]);
    }

    #[test]
    fn test_axes_out_of_bounds_error() {
        let a = arr2d(2, 3, vec![0.0; 6]);
        assert!(sum_axes(&a, Some(&[5]), false).is_err());
    }

    #[test]
    fn test_axes_duplicate_error() {
        let a = arr3d(2, 3, 4, vec![0.0; 24]);
        assert!(sum_axes(&a, Some(&[0, 0]), false).is_err());
    }
}
