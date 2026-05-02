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

/// Try SIMD-accelerated fused sum of squared differences for f64 or f32 (#173).
/// Returns sum((x - mean)²) without allocating an intermediate Vec.
#[inline]
fn try_simd_sum_sq_diff<T: Element + Copy + 'static>(data: &[T], mean: T) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: TypeId check guarantees T is f64. size_of::<T>() == size_of::<f64>().
        let f64_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f64>(), data.len()) };
        let mean_f64: f64 = unsafe { std::mem::transmute_copy(&mean) };
        let result = parallel::simd_sum_sq_diff_f64(f64_slice, mean_f64);
        Some(unsafe { std::mem::transmute_copy(&result) })
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: TypeId check guarantees T is f32. size_of::<T>() == size_of::<f32>().
        let f32_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len()) };
        let mean_f32: f32 = unsafe { std::mem::transmute_copy(&mean) };
        let result = parallel::simd_sum_sq_diff_f32(f32_slice, mean_f32);
        Some(unsafe { std::mem::transmute_copy(&result) })
    } else {
        None
    }
}

/// Try SIMD-accelerated pairwise sum for f64 or f32 (#173).
/// Returns the sum transmuted back to T, or None if T is not f64/f32.
#[inline]
fn try_simd_pairwise_sum<T: Element + Copy + 'static>(data: &[T]) -> Option<T> {
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: TypeId check guarantees T is f64. size_of::<T>() == size_of::<f64>().
        let f64_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f64>(), data.len()) };
        let result = parallel::pairwise_sum_f64(f64_slice);
        Some(unsafe { std::mem::transmute_copy(&result) })
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: TypeId check guarantees T is f32. size_of::<T>() == size_of::<f32>().
        let f32_slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<f32>(), data.len()) };
        let result = parallel::pairwise_sum_f32(f32_slice);
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
pub(crate) fn reduce_axis_typed<T, U, F>(data: &[T], shape: &[usize], axis: usize, f: F) -> Vec<U>
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
pub(crate) fn reduce_axis_general<T, F>(data: &[T], shape: &[usize], axis: usize, f: F) -> Vec<T>
where
    T: Copy,
    F: Fn(&[T]) -> T,
{
    reduce_axis_typed(data, shape, axis, f)
}

/// In-place variant of [`reduce_axis_typed`]: writes results directly into
/// `dst` without allocating a fresh `Vec<U>` to hold the output.
///
/// Used by the `*_into` reductions (#563) so callers that pre-allocate an
/// output buffer truly avoid every per-call allocation. The destination
/// slice must already have exactly `out_size` elements where `out_size` is
/// the product of `shape` with `axis` removed (or 1 if the result is
/// 0-D); callers should validate this via [`check_out_shape`] before
/// invoking the kernel.
pub(crate) fn reduce_axis_typed_into<T, U, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    dst: &mut [U],
    f: F,
) where
    T: Copy,
    F: Fn(&[T]) -> U,
{
    let ndim = shape.len();
    let axis_len = shape[axis];
    let strides = compute_strides(shape);

    // Output multi-index walks the shape with `axis` removed.
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();

    debug_assert_eq!(
        dst.len(),
        if out_shape.is_empty() {
            1
        } else {
            out_shape.iter().product::<usize>()
        },
        "reduce_axis_typed_into: dst length must match the reduction's output size"
    );

    let mut out_multi = vec![0usize; out_shape.len()];
    let mut in_multi = vec![0usize; ndim];
    // Reused per-lane buffer — one allocation total instead of one per
    // lane the way `reduce_axis_typed` does it via `Vec::push`.
    let mut lane_vec: Vec<T> = Vec::with_capacity(axis_len);

    for slot in dst.iter_mut() {
        // Build the input multi-index by inserting the axis position.
        let mut out_dim = 0;
        for (d, idx) in in_multi.iter_mut().enumerate() {
            if d == axis {
                *idx = 0;
            } else {
                *idx = out_multi[out_dim];
                out_dim += 1;
            }
        }

        // Gather lane values into the reusable buffer.
        lane_vec.clear();
        for k in 0..axis_len {
            in_multi[axis] = k;
            let idx = flat_index(&in_multi, &strides);
            lane_vec.push(data[idx]);
        }

        *slot = f(&lane_vec);

        if !out_shape.is_empty() {
            increment_multi_index(&mut out_multi, &out_shape);
        }
    }
}

/// In-place T-to-T reduction wrapper around [`reduce_axis_typed_into`].
#[inline]
pub(crate) fn reduce_axis_general_into<T, F>(
    data: &[T],
    shape: &[usize],
    axis: usize,
    dst: &mut [T],
    f: F,
) where
    T: Copy,
    F: Fn(&[T]) -> T,
{
    reduce_axis_typed_into(data, shape, axis, dst, f);
}

/// Validate axis parameter and return an error if out of bounds.
pub(crate) const fn validate_axis(axis: usize, ndim: usize) -> FerrayResult<()> {
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
pub(crate) fn borrow_data<T: Element + Copy, D: Dimension>(a: &Array<T, D>) -> DataRef<'_, T> {
    if let Some(slice) = a.as_slice() {
        DataRef::Borrowed(slice)
    } else {
        DataRef::Owned(a.iter().copied().collect())
    }
}

/// Build an `IxDyn` result array from output shape and data.
pub(crate) fn make_result<T: Element>(
    out_shape: &[usize],
    data: Vec<T>,
) -> FerrayResult<Array<T, IxDyn>> {
    Array::from_vec(IxDyn::new(out_shape), data)
}

/// Validate that `out` has the expected shape and is C-contiguous,
/// returning a mutable slice into its backing buffer.
///
/// Shared by every `*_into` reduction (#467, #563) so the validation
/// surface lives in exactly one place. Broadcasting is intentionally not
/// allowed because the destination shape is fixed by the input + axis
/// combination — accepting a mismatched destination would silently
/// re-route to a different reduction shape.
pub(crate) fn check_out_shape<'a, T: Element + Copy>(
    out: &'a mut Array<T, IxDyn>,
    expected_shape: &[usize],
    op_name: &str,
) -> FerrayResult<&'a mut [T]> {
    if out.shape() != expected_shape {
        return Err(FerrayError::shape_mismatch(format!(
            "{op_name}_into: out shape {:?} does not match expected reduction shape {:?}",
            out.shape(),
            expected_shape
        )));
    }
    out.as_slice_mut().ok_or_else(|| {
        FerrayError::invalid_value(format!("{op_name}_into: out must be C-contiguous"))
    })
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
/// - Duplicate axes are an error (matches `NumPy`'s `np.sum(a, axis=(0, 0))`)
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
    let mut sorted = ax;
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
        reduce_multi.fill(0);
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
/// **Note:** Unlike `NumPy`, which auto-promotes `int32` sums to `int64`,
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
/// The result is always `Array<f64, IxDyn>`, matching `NumPy`'s behavior
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
            let result = reduce_axis_general(&f64_data, shape, ax, |lane| lane.iter().sum());
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// prod
// ---------------------------------------------------------------------------

/// Product of array elements over a given axis.
///
/// **Note:** Unlike `NumPy`, which auto-promotes integer products,
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

/// Range (peak-to-peak) of array elements over a given axis.
///
/// Returns `max - min`. Analogous to `numpy.ptp(a, axis=...)`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array is empty.
pub fn ptp<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy + std::ops::Sub<Output = T>,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute ptp of empty array",
        ));
    }
    let lo = min(a, axis)?;
    let hi = max(a, axis)?;
    let lo_data: Vec<T> = lo.iter().copied().collect();
    let hi_data: Vec<T> = hi.iter().copied().collect();
    let result: Vec<T> = hi_data
        .into_iter()
        .zip(lo_data)
        .map(|(h, l)| h - l)
        .collect();
    make_result(lo.shape(), result)
}

/// Weighted average of array elements.
///
/// When `weights` is `None`, equivalent to [`mean`]. When `Some(w)`, computes
/// `sum(a * w) / sum(w)` along the given axis (or over all elements when
/// `axis=None`). The weights array must have the same shape as `a` (the
/// 1-D-along-axis broadcasting form supported by NumPy is intentionally
/// omitted here — call broadcast yourself first if you need it).
///
/// Analogous to `numpy.average`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if the array is empty.
/// - `FerrayError::ShapeMismatch` if `weights.shape() != a.shape()`.
/// - `FerrayError::InvalidValue` if the weight sum along an axis is zero.
pub fn average<T, D>(
    a: &Array<T, D>,
    weights: Option<&Array<T, D>>,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    let Some(w) = weights else {
        return mean(a, axis);
    };
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute average of empty array",
        ));
    }
    if a.shape() != w.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "average: weights shape {:?} differs from array shape {:?}",
            w.shape(),
            a.shape(),
        )));
    }
    let a_data = borrow_data(a);
    let w_data = borrow_data(w);
    match axis {
        None => {
            let mut wsum = <T as Element>::zero();
            let mut acc = <T as Element>::zero();
            for (&x, &wi) in a_data.iter().zip(w_data.iter()) {
                wsum = wsum + wi;
                acc = acc + x * wi;
            }
            if wsum == <T as Element>::zero() {
                return Err(FerrayError::invalid_value("average: weights sum to zero"));
            }
            make_result(&[], vec![acc / wsum])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            // Walk lanes for both arrays; we can reuse reduce_axis_general
            // by zipping data + weights into a tagged buffer. Simpler path:
            // build a side buffer of (a_lane, w_lane) per lane.
            let outer: usize = out_s.iter().product::<usize>().max(1);
            let lane_len = shape[ax];
            // Use the same lane-walk as reduce_axis_general. We replicate
            // the inner loop here so we can read both data + weights.
            let mut result = Vec::with_capacity(outer);
            // Compute strides for picking out the lane.
            let mut strides = vec![1usize; shape.len()];
            for i in (0..shape.len() - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
            let mut idx = vec![0usize; shape.len()];
            for _ in 0..outer {
                let mut wsum = <T as Element>::zero();
                let mut acc = <T as Element>::zero();
                for j in 0..lane_len {
                    idx[ax] = j;
                    let mut flat = 0usize;
                    for (d, &s) in idx.iter().zip(strides.iter()) {
                        flat += d * s;
                    }
                    let x = a_data[flat];
                    let wi = w_data[flat];
                    wsum = wsum + wi;
                    acc = acc + x * wi;
                }
                if wsum == <T as Element>::zero() {
                    return Err(FerrayError::invalid_value(
                        "average: weights sum to zero along axis",
                    ));
                }
                result.push(acc / wsum);
                // Advance multi-index over output dims (every dim except ax).
                idx[ax] = 0;
                for d in (0..shape.len()).rev() {
                    if d == ax {
                        continue;
                    }
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
            make_result(&out_s, result)
        }
    }
}

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
/// `Array<f64, IxDyn>`, matching `NumPy`'s behavior of promoting integer
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

/// Variance with integer-to-float64 promotion.
///
/// Sibling of [`mean_as_f64`] — accepts any `T: ToPrimitive` (including
/// `i64`/`i32`/`u64`/etc.) and returns an `Array<f64, IxDyn>`. Matches
/// `NumPy`'s behaviour of promoting integer variance to f64 (#170).
///
/// `ddof` is the delta degrees of freedom (0 for population, 1 for sample).
pub fn var_as_f64<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
) -> FerrayResult<Array<f64, IxDyn>>
where
    T: Element + Copy + Send + Sync + num_traits::ToPrimitive,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute variance of empty array",
        ));
    }
    // Promote each element to f64 once, then run the standard
    // float-typed `var` on the promoted array.
    let promoted: Vec<f64> = a
        .iter()
        .map(|v| {
            v.to_f64()
                .expect("ToPrimitive failed during var_as_f64 promotion")
        })
        .collect();
    let promoted_arr = Array::<f64, _>::from_vec(a.dim().clone(), promoted)?;
    var(&promoted_arr, axis, ddof)
}

/// Standard deviation with integer-to-float64 promotion (#170).
///
/// Sibling of [`mean_as_f64`] / [`var_as_f64`]; sqrt of the float-promoted
/// variance.
pub fn std_as_f64<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
) -> FerrayResult<Array<f64, IxDyn>>
where
    T: Element + Copy + Send + Sync + num_traits::ToPrimitive,
    D: Dimension,
{
    let v = var_as_f64(a, axis, ddof)?;
    let data: Vec<f64> = v.iter().map(|x| x.sqrt()).collect();
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

// ---------------------------------------------------------------------------
// `*_into` reductions: write into a caller-provided destination
//
// NumPy reductions accept an `out=` parameter that lets callers reuse a
// pre-allocated output buffer across repeated reductions on same-shaped
// data — common in streaming statistics, online ML metrics, etc. The
// ferray-stats reduction functions all allocate a fresh `Array<T, IxDyn>`
// internally, so we add `*_into` companion functions that take a
// `&mut Array<T, IxDyn>` destination and skip the allocation entirely.
//
// History:
//   #467 introduced the `*_into` API surface but the first implementation
//   still went through the allocating kernel and copied into `out`,
//   leaving one Vec materialization per call. #563 plumbs the destination
//   slice through the kernel itself via `reduce_axis_typed_into` so the
//   path is truly zero-alloc — only the per-lane scratch buffer
//   (allocated once per call, reused for every lane) and the input
//   contig-borrow allocation (only when the input is already non-contig)
//   remain.
// ---------------------------------------------------------------------------

/// Sum reduction writing into a pre-allocated destination.
///
/// Equivalent to `np.sum(a, axis=axis, out=out)`. The destination must
/// be C-contiguous and have exactly the shape that `sum(a, axis)` would
/// produce; broadcasting is not supported.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis` is out of range.
/// - `FerrayError::ShapeMismatch` if `out.shape()` does not match the
///   expected reduction shape.
/// - `FerrayError::InvalidValue` if `out` is not C-contiguous.
pub fn sum_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    T: Element + std::ops::Add<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let dst = check_out_shape(out, &[], "sum")?;
            let total = try_simd_pairwise_sum(&data)
                .unwrap_or_else(|| parallel::parallel_sum(&data, <T as Element>::zero()));
            dst[0] = total;
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let dst = check_out_shape(out, &out_s, "sum")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
                try_simd_pairwise_sum(lane)
                    .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()))
            });
            Ok(())
        }
    }
}

/// Product reduction writing into a pre-allocated destination.
///
/// See [`sum_into`] for the contract on `out`.
pub fn prod_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    T: Element + std::ops::Mul<Output = T> + Copy + Send + Sync,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let dst = check_out_shape(out, &[], "prod")?;
            dst[0] = parallel::parallel_prod(&data, <T as Element>::one());
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let dst = check_out_shape(out, &out_s, "prod")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
                lane.iter()
                    .copied()
                    .fold(<T as Element>::one(), |acc, x| acc * x)
            });
            Ok(())
        }
    }
}

/// Min reduction writing into a pre-allocated destination.
///
/// See [`sum_into`] for the contract on `out`. Empty input arrays return
/// `FerrayError::InvalidValue` because `min` of an empty set is undefined.
pub fn min_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    T: Element + PartialOrd + Copy + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute min of empty array",
        ));
    }
    // Same NaN-propagating reducer as `min` so the two paths agree.
    let nan_min = |a: T, b: T| -> T {
        if a <= b {
            a
        } else if a > b {
            b
        } else {
            a
        }
    };
    let data = borrow_data(a);
    match axis {
        None => {
            let dst = check_out_shape(out, &[], "min")?;
            dst[0] = data.iter().copied().reduce(nan_min).unwrap();
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let dst = check_out_shape(out, &out_s, "min")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
                lane.iter().copied().reduce(nan_min).unwrap()
            });
            Ok(())
        }
    }
}

/// Max reduction writing into a pre-allocated destination.
///
/// See [`sum_into`] for the contract on `out`. Empty input arrays return
/// `FerrayError::InvalidValue`.
pub fn max_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    T: Element + PartialOrd + Copy + Send + Sync,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute max of empty array",
        ));
    }
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
            let dst = check_out_shape(out, &[], "max")?;
            dst[0] = data.iter().copied().reduce(nan_max).unwrap();
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let dst = check_out_shape(out, &out_s, "max")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
                lane.iter().copied().reduce(nan_max).unwrap()
            });
            Ok(())
        }
    }
}

/// Mean reduction writing into a pre-allocated destination.
///
/// See [`sum_into`] for the contract on `out`. Empty inputs return
/// `FerrayError::InvalidValue`.
pub fn mean_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
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
            let dst = check_out_shape(out, &[], "mean")?;
            let n = T::from(data.len()).unwrap();
            let total = try_simd_pairwise_sum(&data)
                .unwrap_or_else(|| parallel::pairwise_sum(&data, <T as Element>::zero()));
            dst[0] = total / n;
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let axis_len = shape_vec[ax];
            let n = T::from(axis_len).unwrap();
            let dst = check_out_shape(out, &out_s, "mean")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
                let total = try_simd_pairwise_sum(lane)
                    .unwrap_or_else(|| parallel::pairwise_sum(lane, <T as Element>::zero()));
                total / n
            });
            Ok(())
        }
    }
}

/// Variance reduction writing into a pre-allocated destination.
///
/// `ddof` is the delta degrees of freedom. See [`sum_into`] for the
/// contract on `out`. Returns `FerrayError::InvalidValue` for empty
/// inputs or when `ddof >= n`.
pub fn var_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
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
            let dst = check_out_shape(out, &[], "var")?;
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
            dst[0] = sum_sq / T::from(n - ddof).unwrap();
            Ok(())
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let axis_len = shape_vec[ax];
            if axis_len <= ddof {
                return Err(FerrayError::invalid_value(
                    "ddof >= axis length, variance undefined",
                ));
            }
            let out_s = output_shape(&shape_vec, ax);
            let nf = T::from(axis_len).unwrap();
            let denom = T::from(axis_len - ddof).unwrap();
            let dst = check_out_shape(out, &out_s, "var")?;
            reduce_axis_general_into(&data, &shape_vec, ax, dst, |lane| {
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
            Ok(())
        }
    }
}

/// Standard deviation reduction writing into a pre-allocated destination.
///
/// Computes the variance directly into `out` and then takes the
/// element-wise square root in place — the std value never lives in any
/// intermediate buffer.
pub fn std_into<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
    out: &mut Array<T, IxDyn>,
) -> FerrayResult<()>
where
    T: Element + Float + Send + Sync,
    D: Dimension,
{
    var_into(a, axis, ddof, out)?;
    // Variance is now in `out`; sqrt each element in place.
    let dst = out
        .as_slice_mut()
        .ok_or_else(|| FerrayError::invalid_value("std_into: out must be C-contiguous"))?;
    for slot in dst.iter_mut() {
        *slot = slot.sqrt();
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `*_with` reductions: NumPy `initial=` and `where=` parameters (#459)
//
// NumPy reductions accept two extra parameters that ferray-stats was
// missing entirely:
//
//   - `initial`: starting value for the accumulator. Lets `np.sum([])`
//     return a non-zero seed, lets `np.min(arr, initial=999)` provide
//     a fallback for empty inputs, and lets cumulative-style code
//     thread an initial state through a reduction.
//
//   - `where`: same-shape boolean mask. Only positions where the mask
//     is `true` contribute to the reduction. This is the masked-array
//     equivalent without materializing a MaskedArray — `np.sum(arr,
//     where=arr > 0)` is the canonical "sum the positives" idiom.
//
// The `*_with` family adds these as `Option<T>` / `Option<&Array<bool,
// D>>` parameters. Mask broadcasting is intentionally not supported in
// this first cut — the mask must have exactly the same shape as the
// input. NumPy allows broadcast-compatible masks, but the same-shape
// path covers every common use case (predicates produced by comparison
// ufuncs against the same input always have a matching shape).
// ---------------------------------------------------------------------------

/// Prepare a `where` mask for use by a `*_with` reduction kernel.
///
/// Accepts an `Option<&Array<bool, IxDyn>>` so the mask can have any
/// rank independently of the input — callers with typed masks should
/// call `.to_dyn()` before passing in.
///
/// Returns:
/// - `Ok(None)` if the caller passed no mask.
/// - `Ok(Some(vec))` where `vec` is a row-major materialization of the
///   mask aligned with `a`'s flat logical order. For same-shape masks
///   this is a direct copy; for broadcast-compatible masks the mask is
///   broadcast into `a.shape()` first via
///   [`ferray_core::dimension::broadcast::broadcast_to`] (#565) and
///   then materialized.
/// - `Err(FerrayError::ShapeMismatch)` if the mask is not
///   broadcast-compatible with `a.shape()`.
fn prepare_where_mask<T, D>(
    a: &Array<T, D>,
    mask: Option<&Array<bool, IxDyn>>,
    op_name: &str,
) -> FerrayResult<Option<Vec<bool>>>
where
    T: Element,
    D: Dimension,
{
    use ferray_core::dimension::broadcast::broadcast_to;

    let Some(m) = mask else {
        return Ok(None);
    };

    // Fast path: the mask is already the same shape as the input.
    if m.shape() == a.shape() {
        return Ok(Some(m.iter().copied().collect()));
    }

    // Broadcast path: let the core machinery check compatibility and
    // produce a stride-tricked view at the target shape. We then
    // materialize it into a flat Vec<bool> aligned with `a`'s logical
    // row-major order.
    let view = broadcast_to(m, a.shape()).map_err(|_| {
        FerrayError::shape_mismatch(format!(
            "{op_name}: where mask shape {:?} is not broadcast-compatible with array shape {:?}",
            m.shape(),
            a.shape()
        ))
    })?;
    Ok(Some(view.iter().copied().collect()))
}

/// Walk a single axis-Some output position, gathering masked-true lane
/// values into `lane_buf` (cleared first). Used by every `*_with` axis
/// path so the lane-gather logic exists in one place.
///
/// `out_multi` is the output multi-index excluding `axis`. The function
/// reconstructs the corresponding input flat positions and consults the
/// mask in lockstep.
fn gather_lane_with_mask<T: Copy>(
    data: &[T],
    mask: Option<&[bool]>,
    shape: &[usize],
    strides: &[usize],
    axis: usize,
    out_multi: &[usize],
    lane_buf: &mut Vec<T>,
) {
    let ndim = shape.len();
    let axis_len = shape[axis];
    let mut in_multi = vec![0usize; ndim];
    let mut out_dim = 0;
    for (d, idx) in in_multi.iter_mut().enumerate() {
        if d == axis {
            *idx = 0;
        } else {
            *idx = out_multi[out_dim];
            out_dim += 1;
        }
    }
    lane_buf.clear();
    for k in 0..axis_len {
        in_multi[axis] = k;
        let idx = flat_index(&in_multi, strides);
        match mask {
            Some(m) if !m[idx] => {}
            _ => lane_buf.push(data[idx]),
        }
    }
}

/// Sum reduction with `initial` and `where` parameters.
///
/// Equivalent to `np.sum(a, axis=axis, initial=initial, where=where_mask)`.
///
/// `initial` defaults to `T::zero()` when `None`. `where_mask`, when
/// provided, must be broadcast-compatible with `a.shape()` (#565).
/// Callers with typed masks (e.g. `Array<bool, Ix1>`) should call
/// `.to_dyn()` before passing. Positions where the broadcast mask is
/// `false` are skipped.
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis` is out of range.
/// - `FerrayError::ShapeMismatch` if `where_mask` is not
///   broadcast-compatible with `a.shape()`.
pub fn sum_with<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    initial: Option<T>,
    where_mask: Option<&Array<bool, IxDyn>>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Add<Output = T> + Copy,
    D: Dimension,
{
    let init = initial.unwrap_or_else(<T as Element>::zero);
    let data = borrow_data(a);
    let mask_vec = prepare_where_mask(a, where_mask, "sum")?;
    let mask_slice: Option<&[bool]> = mask_vec.as_deref();

    match axis {
        None => {
            let total = match mask_slice {
                None => data.iter().copied().fold(init, |acc, x| acc + x),
                Some(mask) => data
                    .iter()
                    .zip(mask.iter())
                    .filter(|&(_, &m)| m)
                    .fold(init, |acc, (&x, _)| acc + x),
            };
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let strides = compute_strides(&shape_vec);

            let out_size: usize = if out_s.is_empty() {
                1
            } else {
                out_s.iter().product()
            };
            let mut result: Vec<T> = Vec::with_capacity(out_size);
            let mut out_multi = vec![0usize; out_s.len()];
            let mut lane_buf: Vec<T> = Vec::with_capacity(shape_vec[ax]);

            for _ in 0..out_size {
                gather_lane_with_mask(
                    &data,
                    mask_slice,
                    &shape_vec,
                    &strides,
                    ax,
                    &out_multi,
                    &mut lane_buf,
                );
                let lane_sum = lane_buf.iter().copied().fold(init, |acc, x| acc + x);
                result.push(lane_sum);
                if !out_s.is_empty() {
                    increment_multi_index(&mut out_multi, &out_s);
                }
            }
            make_result(&out_s, result)
        }
    }
}

/// Product reduction with `initial` and `where` parameters.
///
/// Equivalent to `np.prod(a, axis=axis, initial=initial, where=where_mask)`.
/// `initial` defaults to `T::one()` when `None`. `where_mask` is
/// broadcast-compatible with `a.shape()` (#565).
pub fn prod_with<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    initial: Option<T>,
    where_mask: Option<&Array<bool, IxDyn>>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + std::ops::Mul<Output = T> + Copy,
    D: Dimension,
{
    let init = initial.unwrap_or_else(<T as Element>::one);
    let data = borrow_data(a);
    let mask_vec = prepare_where_mask(a, where_mask, "prod")?;
    let mask_slice: Option<&[bool]> = mask_vec.as_deref();

    match axis {
        None => {
            let total = match mask_slice {
                None => data.iter().copied().fold(init, |acc, x| acc * x),
                Some(mask) => data
                    .iter()
                    .zip(mask.iter())
                    .filter(|&(_, &m)| m)
                    .fold(init, |acc, (&x, _)| acc * x),
            };
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let strides = compute_strides(&shape_vec);

            let out_size: usize = if out_s.is_empty() {
                1
            } else {
                out_s.iter().product()
            };
            let mut result: Vec<T> = Vec::with_capacity(out_size);
            let mut out_multi = vec![0usize; out_s.len()];
            let mut lane_buf: Vec<T> = Vec::with_capacity(shape_vec[ax]);

            for _ in 0..out_size {
                gather_lane_with_mask(
                    &data,
                    mask_slice,
                    &shape_vec,
                    &strides,
                    ax,
                    &out_multi,
                    &mut lane_buf,
                );
                let lane_prod = lane_buf.iter().copied().fold(init, |acc, x| acc * x);
                result.push(lane_prod);
                if !out_s.is_empty() {
                    increment_multi_index(&mut out_multi, &out_s);
                }
            }
            make_result(&out_s, result)
        }
    }
}

/// Min reduction with `initial` and `where` parameters.
///
/// Equivalent to `np.min(a, axis=axis, initial=initial, where=where_mask)`.
/// Unlike plain `min`, an empty input (or a fully-masked-out lane) is
/// allowed when `initial` is supplied — the result is `initial`. Without
/// `initial`, an empty lane is an error. `where_mask` is
/// broadcast-compatible with `a.shape()` (#565).
pub fn min_with<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    initial: Option<T>,
    where_mask: Option<&Array<bool, IxDyn>>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let nan_min = |a: T, b: T| -> T {
        if a <= b {
            a
        } else if a > b {
            b
        } else {
            a
        }
    };
    let data = borrow_data(a);
    let mask_vec = prepare_where_mask(a, where_mask, "min")?;
    let mask_slice: Option<&[bool]> = mask_vec.as_deref();

    let lane_min = |lane: &[T], initial: Option<T>| -> FerrayResult<T> {
        let mut iter = lane.iter().copied();
        let seed = match initial {
            Some(v) => Some(v),
            None => iter.next(),
        };
        match seed {
            Some(s) => Ok(iter.fold(s, nan_min)),
            None => Err(FerrayError::invalid_value(
                "min: empty lane and no initial value",
            )),
        }
    };

    match axis {
        None => {
            let total = match mask_slice {
                None => lane_min(&data, initial)?,
                Some(mask) => {
                    let filtered: Vec<T> = data
                        .iter()
                        .copied()
                        .zip(mask.iter().copied())
                        .filter_map(|(x, m)| if m { Some(x) } else { None })
                        .collect();
                    lane_min(&filtered, initial)?
                }
            };
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let strides = compute_strides(&shape_vec);

            let out_size: usize = if out_s.is_empty() {
                1
            } else {
                out_s.iter().product()
            };
            let mut result: Vec<T> = Vec::with_capacity(out_size);
            let mut out_multi = vec![0usize; out_s.len()];
            let mut lane_buf: Vec<T> = Vec::with_capacity(shape_vec[ax]);

            for _ in 0..out_size {
                gather_lane_with_mask(
                    &data,
                    mask_slice,
                    &shape_vec,
                    &strides,
                    ax,
                    &out_multi,
                    &mut lane_buf,
                );
                result.push(lane_min(&lane_buf, initial)?);
                if !out_s.is_empty() {
                    increment_multi_index(&mut out_multi, &out_s);
                }
            }
            make_result(&out_s, result)
        }
    }
}

/// Max reduction with `initial` and `where` parameters.
///
/// Symmetric to [`min_with`]. `where_mask` is broadcast-compatible with
/// `a.shape()` (#565).
pub fn max_with<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    initial: Option<T>,
    where_mask: Option<&Array<bool, IxDyn>>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
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
    let mask_vec = prepare_where_mask(a, where_mask, "max")?;
    let mask_slice: Option<&[bool]> = mask_vec.as_deref();

    let lane_max = |lane: &[T], initial: Option<T>| -> FerrayResult<T> {
        let mut iter = lane.iter().copied();
        let seed = match initial {
            Some(v) => Some(v),
            None => iter.next(),
        };
        match seed {
            Some(s) => Ok(iter.fold(s, nan_max)),
            None => Err(FerrayError::invalid_value(
                "max: empty lane and no initial value",
            )),
        }
    };

    match axis {
        None => {
            let total = match mask_slice {
                None => lane_max(&data, initial)?,
                Some(mask) => {
                    let filtered: Vec<T> = data
                        .iter()
                        .copied()
                        .zip(mask.iter().copied())
                        .filter_map(|(x, m)| if m { Some(x) } else { None })
                        .collect();
                    lane_max(&filtered, initial)?
                }
            };
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let strides = compute_strides(&shape_vec);

            let out_size: usize = if out_s.is_empty() {
                1
            } else {
                out_s.iter().product()
            };
            let mut result: Vec<T> = Vec::with_capacity(out_size);
            let mut out_multi = vec![0usize; out_s.len()];
            let mut lane_buf: Vec<T> = Vec::with_capacity(shape_vec[ax]);

            for _ in 0..out_size {
                gather_lane_with_mask(
                    &data,
                    mask_slice,
                    &shape_vec,
                    &strides,
                    ax,
                    &out_multi,
                    &mut lane_buf,
                );
                result.push(lane_max(&lane_buf, initial)?);
                if !out_s.is_empty() {
                    increment_multi_index(&mut out_multi, &out_s);
                }
            }
            make_result(&out_s, result)
        }
    }
}

/// Mean reduction with a `where` mask.
///
/// Equivalent to `np.mean(a, axis=axis, where=where_mask)`. The divisor
/// is the count of `true` positions in the (broadcast) mask, NOT the
/// lane length — fully-masked-out lanes return `T::nan()` (matching
/// `NumPy`'s "`RuntimeWarning`: Mean of empty slice" behavior, but without
/// the warning machinery). `initial` is intentionally not modeled
/// because the divisor for an "initial-bumped" mean is ambiguous in
/// `NumPy` too. `where_mask` is broadcast-compatible with `a.shape()`
/// (#565).
pub fn mean_where<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    where_mask: Option<&Array<bool, IxDyn>>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let data = borrow_data(a);
    let mask_vec = prepare_where_mask(a, where_mask, "mean")?;
    let mask_slice: Option<&[bool]> = mask_vec.as_deref();

    match axis {
        None => {
            let (sum, count) = match mask_slice {
                None => {
                    let total = data
                        .iter()
                        .copied()
                        .fold(<T as Element>::zero(), |acc, x| acc + x);
                    (total, data.len())
                }
                Some(mask) => {
                    let mut s = <T as Element>::zero();
                    let mut c = 0usize;
                    for (&x, &m) in data.iter().zip(mask.iter()) {
                        if m {
                            s = s + x;
                            c += 1;
                        }
                    }
                    (s, c)
                }
            };
            let result = if count == 0 {
                <T as Float>::nan()
            } else {
                sum / T::from(count).unwrap()
            };
            make_result(&[], vec![result])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape_vec = a.shape().to_vec();
            let out_s = output_shape(&shape_vec, ax);
            let strides = compute_strides(&shape_vec);

            let out_size: usize = if out_s.is_empty() {
                1
            } else {
                out_s.iter().product()
            };
            let mut result: Vec<T> = Vec::with_capacity(out_size);
            let mut out_multi = vec![0usize; out_s.len()];
            let mut lane_buf: Vec<T> = Vec::with_capacity(shape_vec[ax]);

            for _ in 0..out_size {
                gather_lane_with_mask(
                    &data,
                    mask_slice,
                    &shape_vec,
                    &strides,
                    ax,
                    &out_multi,
                    &mut lane_buf,
                );
                let lane_mean = if lane_buf.is_empty() {
                    <T as Float>::nan()
                } else {
                    let s = lane_buf
                        .iter()
                        .copied()
                        .fold(<T as Element>::zero(), |acc, x| acc + x);
                    s / T::from(lane_buf.len()).unwrap()
                };
                result.push(lane_mean);
                if !out_s.is_empty() {
                    increment_multi_index(&mut out_multi, &out_s);
                }
            }
            make_result(&out_s, result)
        }
    }
}

/// Single-axis argmin with optional `keepdims`.
///
/// `NumPy`'s `argmin` only accepts a single axis (or `None`); this mirrors
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
        reduce_multi.fill(0);
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
        let a =
            Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![i32::MAX, i32::MAX, i32::MAX]).unwrap();
        let s = sum_as_f64(&a, None).unwrap();
        let expected = 3.0 * f64::from(i32::MAX);
        assert!((s.iter().next().unwrap() - expected).abs() < 1.0);
    }

    // -----------------------------------------------------------------------
    // Multi-axis + keepdims tests (issues #457 + #458)
    // -----------------------------------------------------------------------

    use ferray_core::Ix3;

    fn arr1d(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
    }

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
        let data: Vec<f64> = (0..24).map(f64::from).collect();
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
        let data: Vec<f64> = (0..24).map(f64::from).collect();
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
        let data: Vec<f64> = (0..24).map(f64::from).collect();
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
        let data: Vec<f64> = (0..24).map(f64::from).collect();
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

    // ----- Infinity tests (#177) -----

    #[test]
    fn sum_with_infinity() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let s = sum(&a, None).unwrap();
        assert_eq!(s.iter().next().copied().unwrap(), f64::INFINITY);
    }

    #[test]
    fn sum_inf_minus_inf_is_nan() {
        let a = arr1d(vec![f64::INFINITY, f64::NEG_INFINITY]);
        let s = sum(&a, None).unwrap();
        assert!(s.iter().next().copied().unwrap().is_nan());
    }

    #[test]
    fn mean_with_infinity() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let m = mean(&a, None).unwrap();
        assert_eq!(m.iter().next().copied().unwrap(), f64::INFINITY);
    }

    #[test]
    fn min_with_neg_infinity() {
        let a = arr1d(vec![1.0, f64::NEG_INFINITY, 3.0]);
        let m = min(&a, None).unwrap();
        assert_eq!(m.iter().next().copied().unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn max_with_infinity() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let m = max(&a, None).unwrap();
        assert_eq!(m.iter().next().copied().unwrap(), f64::INFINITY);
    }

    #[test]
    fn prod_with_zero_and_infinity_is_nan() {
        // 0 * INF = NaN per IEEE 754
        let a = arr1d(vec![0.0, f64::INFINITY, 1.0]);
        let p = prod(&a, None).unwrap();
        assert!(p.iter().next().copied().unwrap().is_nan());
    }

    #[test]
    fn prod_with_infinity_propagates() {
        let a = arr1d(vec![2.0, f64::INFINITY, 3.0]);
        let p = prod(&a, None).unwrap();
        assert_eq!(p.iter().next().copied().unwrap(), f64::INFINITY);
    }

    #[test]
    fn var_with_infinity_is_nan() {
        // var includes (x - mean)^2 where mean is INF; (INF - INF)^2 = NaN.
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let v = var(&a, None, 0).unwrap();
        assert!(v.iter().next().copied().unwrap().is_nan());
    }

    #[test]
    fn std_with_infinity_is_nan() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let s = std_(&a, None, 0).unwrap();
        assert!(s.iter().next().copied().unwrap().is_nan());
    }

    #[test]
    fn argmin_finds_neg_infinity() {
        let a = arr1d(vec![1.0, f64::NEG_INFINITY, 3.0, -100.0]);
        let i = crate::reductions::argmin(&a, None).unwrap();
        assert_eq!(i.iter().next().copied().unwrap(), 1);
    }

    #[test]
    fn argmax_finds_infinity() {
        let a = arr1d(vec![1.0, f64::INFINITY, 1e300, 5.0]);
        let i = crate::reductions::argmax(&a, None).unwrap();
        assert_eq!(i.iter().next().copied().unwrap(), 1);
    }

    #[test]
    fn cumsum_propagates_infinity() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let c = cumsum(&a, None).unwrap();
        let v: Vec<f64> = c.iter().copied().collect();
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_infinite());
        assert!(v[2].is_infinite());
    }

    #[test]
    fn cumprod_inf_then_zero_yields_nan() {
        let a = arr1d(vec![2.0, f64::INFINITY, 0.0]);
        let c = cumprod(&a, None).unwrap();
        let v: Vec<f64> = c.iter().copied().collect();
        assert_eq!(v[0], 2.0);
        assert!(v[1].is_infinite());
        assert!(v[2].is_nan()); // INF * 0 = NaN
    }

    #[test]
    fn ptp_with_infinity_is_inf() {
        let a = arr1d(vec![1.0, f64::INFINITY, 3.0]);
        let p = ptp(&a, None).unwrap();
        assert!(p.iter().next().copied().unwrap().is_infinite());
    }

    // ----- Single-element var/std with ddof (#178) -----

    #[test]
    fn var_single_element_ddof0() {
        let a = arr1d(vec![5.0]);
        let v = var(&a, None, 0).unwrap();
        assert_eq!(v.iter().next().copied().unwrap(), 0.0);
    }

    #[test]
    fn var_single_element_ddof1_errors() {
        // ddof=1 with N=1 → ddof >= N, ferray errors instead of
        // returning NaN (stricter than NumPy which returns NaN).
        let a = arr1d(vec![5.0]);
        assert!(var(&a, None, 1).is_err());
    }

    #[test]
    fn std_single_element_ddof0() {
        let a = arr1d(vec![5.0]);
        let s = std_(&a, None, 0).unwrap();
        assert_eq!(s.iter().next().copied().unwrap(), 0.0);
    }

    #[test]
    fn std_single_element_ddof1_errors() {
        // Same shape as var: ddof=1 with N=1 hits the same ddof >= N
        // guard (std_ delegates to var internally). Confirms the
        // error path is plumbed through.
        let a = arr1d(vec![5.0]);
        assert!(std_(&a, None, 1).is_err());
    }

    #[test]
    fn var_two_elements_ddof1_population_to_sample() {
        // Sanity: ddof=0 vs ddof=1 on the same input — Bessel's correction
        // should give a 2x bigger variance for N=2.
        let a = arr1d(vec![1.0, 3.0]);
        let v0 = var(&a, None, 0).unwrap();
        let v1 = var(&a, None, 1).unwrap();
        let v0_val = v0.iter().next().copied().unwrap();
        let v1_val = v1.iter().next().copied().unwrap();
        assert!((v0_val - 1.0).abs() < 1e-12); // ((1-2)^2 + (3-2)^2) / 2 = 1.0
        assert!((v1_val - 2.0).abs() < 1e-12); // ((1-2)^2 + (3-2)^2) / 1 = 2.0
        // Sample variance (ddof=1) should be 2x population (ddof=0) for N=2.
        assert!((v1_val / v0_val - 2.0).abs() < 1e-12);
    }

    #[test]
    fn var_single_element_2d_ddof0() {
        // 1×1 array — single element along every axis
        use ferray_core::dimension::Ix2;
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 1]), vec![5.0]).unwrap();
        let v = var(&a, None, 0).unwrap();
        assert_eq!(v.iter().next().copied().unwrap(), 0.0);
    }

    #[test]
    fn var_single_element_axis_ddof_too_large_errors() {
        // 1×3 array with ddof=1 reduced along axis 0 (length 1) should error
        use ferray_core::dimension::Ix2;
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(var(&a, Some(0), 1).is_err());
    }

    // ---- *_into reductions (#467) ----

    #[test]
    fn sum_into_axis_writes_into_destination() {
        // (2, 3) sum along axis=1 → shape (2,)
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0; 2]).unwrap();
        sum_into(&a, Some(1), &mut out).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn sum_into_no_axis_writes_scalar_destination() {
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        sum_into(&a, None, &mut out).unwrap();
        assert_eq!(out.iter().next().copied().unwrap(), 10.0);
    }

    #[test]
    fn sum_into_rejects_wrong_shape() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // axis=1 reduces to shape (2,), but out has shape (3,)
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![0.0; 3]).unwrap();
        let err = sum_into(&a, Some(1), &mut out);
        assert!(err.is_err());
    }

    #[test]
    fn sum_into_rejects_axis_out_of_bounds() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![0.0]).unwrap();
        assert!(sum_into(&a, Some(5), &mut out).is_err());
    }

    #[test]
    fn prod_into_basic() {
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        prod_into(&a, None, &mut out).unwrap();
        assert_eq!(out.iter().next().copied().unwrap(), 24.0);
    }

    #[test]
    fn min_into_axis_basic() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0])
            .unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0; 2]).unwrap();
        min_into(&a, Some(1), &mut out).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[1.0, 3.0]);
    }

    #[test]
    fn max_into_axis_basic() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 5.0, 2.0, 4.0, 3.0, 6.0])
            .unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0; 2]).unwrap();
        max_into(&a, Some(1), &mut out).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn mean_into_axis_basic() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0; 2]).unwrap();
        mean_into(&a, Some(1), &mut out).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[2.0, 5.0]);
    }

    #[test]
    fn var_into_basic() {
        // Variance with ddof=0 of [1,2,3,4] = 1.25
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        var_into(&a, None, 0, &mut out).unwrap();
        assert!((out.iter().next().copied().unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn std_into_basic() {
        // sqrt(var) for [1,2,3,4] with ddof=0 = sqrt(1.25)
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        std_into(&a, None, 0, &mut out).unwrap();
        let expected = 1.25_f64.sqrt();
        assert!((out.iter().next().copied().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn into_reductions_can_reuse_destination_across_calls() {
        // The whole point of the out= API: a single allocation reused
        // across many reductions on same-shaped data.
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0; 2]).unwrap();
        for k in 0..3 {
            let base = f64::from(k);
            let a = Array::<f64, Ix2>::from_vec(
                Ix2::new([2, 3]),
                vec![
                    base + 1.0,
                    base + 2.0,
                    base + 3.0,
                    base + 4.0,
                    base + 5.0,
                    base + 6.0,
                ],
            )
            .unwrap();
            sum_into(&a, Some(1), &mut out).unwrap();
            assert_eq!(
                out.as_slice().unwrap(),
                &[3.0f64.mul_add(base, 6.0), 3.0f64.mul_add(base, 15.0)]
            );
        }
    }

    // ---- zero-alloc kernel regression tests (#563) ----

    #[test]
    fn sum_into_overwrites_existing_destination_garbage() {
        // The new kernel writes results directly into `out` rather than
        // routing through write_into's copy_from_slice; make sure existing
        // garbage is fully overwritten and never bleeds through.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![999.0, -999.0]).unwrap();
        sum_into(&a, Some(1), &mut out).unwrap();
        assert_eq!(out.as_slice().unwrap(), &[6.0, 15.0]);
    }

    #[test]
    fn reduce_axis_typed_into_matches_reduce_axis_typed() {
        // The in-place kernel must produce identical output to the
        // allocating kernel for any (shape, axis, fn).
        use super::{reduce_axis_typed, reduce_axis_typed_into};
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        for shape in [vec![24usize], vec![4, 6], vec![2, 3, 4], vec![2, 2, 2, 3]] {
            for ax in 0..shape.len() {
                let allocated: Vec<f64> =
                    reduce_axis_typed(&data, &shape, ax, |lane| lane.iter().sum());
                let mut dst = vec![0.0; allocated.len()];
                reduce_axis_typed_into(&data, &shape, ax, &mut dst, |lane| lane.iter().sum());
                assert_eq!(dst, allocated, "shape {shape:?} axis {ax}");
            }
        }
    }

    #[test]
    fn sum_into_3d_axis_correct() {
        use ferray_core::Ix3;
        // (2, 3, 4) reducing axis 1 → shape (2, 4)
        let data: Vec<f64> = (0..24).map(f64::from).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 4]), vec![0.0; 8]).unwrap();
        sum_into(&a, Some(1), &mut out).unwrap();
        // Hand check: out[i, k] = sum_{j} a[i, j, k] = sum_{j} (i*12 + j*4 + k)
        let expected: Vec<f64> = (0..2)
            .flat_map(|i| {
                (0..4).map(move |k| (0..3).map(|j| f64::from(i * 12 + j * 4 + k)).sum::<f64>())
            })
            .collect();
        assert_eq!(out.as_slice().unwrap(), expected.as_slice());
    }

    #[test]
    fn min_into_rejects_empty_input() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        assert!(min_into(&a, None, &mut out).is_err());
    }

    #[test]
    fn max_into_rejects_empty_input() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        assert!(max_into(&a, None, &mut out).is_err());
    }

    #[test]
    fn var_into_rejects_ddof_too_large() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        // n=3, ddof=3 → ddof >= n, must error.
        assert!(var_into(&a, None, 3, &mut out).is_err());
    }

    #[test]
    fn std_into_does_not_leave_variance_in_destination_on_success() {
        // var of [1,2,3,4] with ddof=0 = 1.25; std = sqrt(1.25) ≈ 1.118.
        // The destination must hold the std value, not the variance, after
        // a successful call.
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[]), vec![0.0]).unwrap();
        std_into(&a, None, 0, &mut out).unwrap();
        let got = out.iter().next().copied().unwrap();
        let expected = 1.25_f64.sqrt();
        assert!((got - expected).abs() < 1e-12);
        // Sanity: result is not the variance.
        assert!((got - 1.25).abs() > 1e-3);
    }

    // ---- *_with reductions: initial= and where= (#459) ----

    #[test]
    fn sum_with_initial_only() {
        // Initial = 100; sum becomes 100 + 1 + 2 + 3 + 4 = 110
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let r = sum_with(&a, None, Some(100.0), None).unwrap();
        assert_eq!(r.iter().next().copied().unwrap(), 110.0);
    }

    #[test]
    fn sum_with_where_mask_only() {
        // Sum positives only.
        let a = arr1d(vec![1.0, -2.0, 3.0, -4.0, 5.0]);
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let r = sum_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        assert_eq!(r.iter().next().copied().unwrap(), 9.0);
    }

    #[test]
    fn sum_with_initial_and_mask_combined() {
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0]);
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        let r = sum_with(&a, None, Some(50.0), Some(&mask.to_dyn())).unwrap();
        // 50 + 1 + 3 = 54
        assert_eq!(r.iter().next().copied().unwrap(), 54.0);
    }

    #[test]
    fn sum_with_axis_and_mask() {
        // (2, 3); mask zeroes out the first column; row sums become
        // [2+3, 5+6] = [5, 11].
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, true, true, false, true, true],
        )
        .unwrap();
        let r = sum_with(&a, Some(1), None, Some(&mask.to_dyn())).unwrap();
        assert_eq!(r.shape(), &[2]);
        assert_eq!(r.as_slice().unwrap(), &[5.0, 11.0]);
    }

    #[test]
    fn sum_with_axis_and_initial() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        // axis=1 with initial=10 → row sums become [10+6, 10+15] = [16, 25]
        let r = sum_with(&a, Some(1), Some(10.0), None).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[16.0, 25.0]);
    }

    #[test]
    fn sum_with_no_initial_no_mask_matches_legacy_sum() {
        // Sanity: omitting both knobs should match the existing sum().
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let legacy = sum(&a, Some(1)).unwrap();
        let with_form = sum_with(&a, Some(1), None, None).unwrap();
        assert_eq!(legacy.as_slice().unwrap(), with_form.as_slice().unwrap());
    }

    #[test]
    fn sum_with_rejects_mismatched_mask_shape() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true; 4]).unwrap();
        assert!(sum_with(&a, None, None, Some(&mask.to_dyn())).is_err());
    }

    #[test]
    fn prod_with_initial_only() {
        let a = arr1d(vec![2.0, 3.0, 4.0]);
        let r = prod_with(&a, None, Some(10.0), None).unwrap();
        // 10 * 2 * 3 * 4 = 240
        assert_eq!(r.iter().next().copied().unwrap(), 240.0);
    }

    #[test]
    fn prod_with_where_mask() {
        let a = arr1d(vec![2.0, 0.0, 3.0, 0.0, 4.0]);
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let r = prod_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // 2 * 3 * 4 = 24 (zero positions skipped)
        assert_eq!(r.iter().next().copied().unwrap(), 24.0);
    }

    #[test]
    fn min_with_initial_provides_fallback_for_empty_lane() {
        // Empty array + initial = the initial value.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let r = min_with(&a, None, Some(99.0), None).unwrap();
        assert_eq!(r.iter().next().copied().unwrap(), 99.0);
    }

    #[test]
    fn min_with_initial_caps_actual_min() {
        // Initial=0, data=[1,2,3] → min(0, 1, 2, 3) = 0.
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let r = min_with(&a, None, Some(0.0), None).unwrap();
        assert_eq!(r.iter().next().copied().unwrap(), 0.0);
    }

    #[test]
    fn min_with_where_mask_filters_then_reduces() {
        let a = arr1d(vec![5.0, 1.0, 4.0, 2.0, 3.0]);
        // Mask out positions 1 and 3 (values 1 and 2).
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let r = min_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // Filtered: [5, 4, 3] → min = 3
        assert_eq!(r.iter().next().copied().unwrap(), 3.0);
    }

    #[test]
    fn min_with_empty_lane_no_initial_errors() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert!(min_with(&a, None, None, None).is_err());
    }

    #[test]
    fn min_with_fully_masked_axis_lane_no_initial_errors() {
        // Each row fully masked out → error per lane.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        assert!(min_with(&a, Some(1), None, Some(&mask.to_dyn())).is_err());
    }

    #[test]
    fn min_with_fully_masked_axis_lane_with_initial_uses_initial() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([2, 3]), vec![false; 6]).unwrap();
        let r = min_with(&a, Some(1), Some(99.0), Some(&mask.to_dyn())).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[99.0, 99.0]);
    }

    #[test]
    fn max_with_initial_caps_actual_max() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let r = max_with(&a, None, Some(99.0), None).unwrap();
        assert_eq!(r.iter().next().copied().unwrap(), 99.0);
    }

    #[test]
    fn max_with_where_mask_filters_then_reduces() {
        let a = arr1d(vec![5.0, 10.0, 4.0, 20.0, 3.0]);
        // Mask out positions 1 and 3 (values 10 and 20).
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let r = max_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // Filtered: [5, 4, 3] → max = 5
        assert_eq!(r.iter().next().copied().unwrap(), 5.0);
    }

    #[test]
    fn mean_where_filters_and_divides_by_count() {
        // [1, 2, 3, 4, 5] with mask [T, F, T, F, T] → (1+3+5)/3 = 3.0
        let a = arr1d(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mask =
            Array::<bool, Ix1>::from_vec(Ix1::new([5]), vec![true, false, true, false, true])
                .unwrap();
        let r = mean_where(&a, None, Some(&mask.to_dyn())).unwrap();
        assert!((r.iter().next().copied().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn mean_where_no_mask_matches_legacy_mean() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let legacy = mean(&a, Some(1)).unwrap();
        let where_form = mean_where(&a, Some(1), None).unwrap();
        assert_eq!(legacy.as_slice().unwrap(), where_form.as_slice().unwrap());
    }

    #[test]
    fn mean_where_fully_masked_lane_returns_nan() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, true, true, false, false, false],
        )
        .unwrap();
        let r = mean_where(&a, Some(1), Some(&mask.to_dyn())).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 2.0).abs() < 1e-12);
        assert!(s[1].is_nan());
    }

    #[test]
    fn mean_where_axis_with_partial_mask() {
        // (2, 3); mask = [[T,T,F],[F,T,T]]
        // Row 0 mean: (1+2)/2 = 1.5
        // Row 1 mean: (5+6)/2 = 5.5
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, true, false, false, true, true],
        )
        .unwrap();
        let r = mean_where(&a, Some(1), Some(&mask.to_dyn())).unwrap();
        assert_eq!(r.shape(), &[2]);
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.5).abs() < 1e-12);
        assert!((s[1] - 5.5).abs() < 1e-12);
    }

    // ---- broadcast-compatible where masks (#565) ----

    #[test]
    fn sum_with_mask_broadcasts_ix1_into_ix2() {
        // (2, 3) input; mask of shape (3,) — broadcasts to (2, 3) with
        // every row identical. sum ignores the first column, so both
        // row sums skip that column's contribution.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask_1d = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, true]).unwrap();
        let r = sum_with(&a, None, None, Some(&mask_1d.to_dyn())).unwrap();
        // Sum with first column masked out: 2 + 3 + 5 + 6 = 16.0
        assert!((r.iter().next().copied().unwrap() - 16.0).abs() < 1e-12);
    }

    #[test]
    fn sum_with_mask_broadcasts_column_vector_across_rows() {
        // (3, 4) input; mask of shape (3, 1) — broadcasts across the
        // four columns, so each row is either fully kept or fully
        // masked out.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 4]),
            vec![
                1.0, 2.0, 3.0, 4.0, // row 0
                5.0, 6.0, 7.0, 8.0, // row 1
                9.0, 10.0, 11.0, 12.0, // row 2
            ],
        )
        .unwrap();
        let mask_col =
            Array::<bool, Ix2>::from_vec(Ix2::new([3, 1]), vec![true, false, true]).unwrap();
        let r = sum_with(&a, None, None, Some(&mask_col.to_dyn())).unwrap();
        // Rows 0 and 2 kept: 1+2+3+4 + 9+10+11+12 = 10 + 42 = 52
        assert!((r.iter().next().copied().unwrap() - 52.0).abs() < 1e-12);
    }

    #[test]
    fn sum_with_mask_broadcasts_row_vector_against_axis_reduction() {
        // (2, 3) reducing axis=1; mask shape (3,) broadcasts against
        // each row.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask_1d = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let r = sum_with(&a, Some(1), None, Some(&mask_1d.to_dyn())).unwrap();
        assert_eq!(r.shape(), &[2]);
        // Row 0: 1 + 3 = 4; row 1: 4 + 6 = 10
        assert_eq!(r.as_slice().unwrap(), &[4.0, 10.0]);
    }

    #[test]
    fn prod_with_mask_broadcasts_ix1() {
        // (2, 3); mask shape (3,) keeps columns 0 and 2.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
            .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let r = prod_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // Product of kept values: 2 * 4 * 5 * 7 = 280
        assert!((r.iter().next().copied().unwrap() - 280.0).abs() < 1e-12);
    }

    #[test]
    fn min_with_mask_broadcasts_ix1() {
        // Column mask; find min across the kept columns.
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![5.0, 1.0, 4.0, 2.0, 10.0, 3.0])
            .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let r = min_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // Kept values: 5, 4, 2, 3 → min = 2
        assert!((r.iter().next().copied().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn max_with_mask_broadcasts_ix1() {
        let a =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![5.0, 100.0, 4.0, 2.0, 200.0, 3.0])
                .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let r = max_with(&a, None, None, Some(&mask.to_dyn())).unwrap();
        // Kept values: 5, 4, 2, 3 → max = 5
        assert!((r.iter().next().copied().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mean_where_mask_broadcasts_ix1() {
        // (2, 3); mask (3,) keeps columns 0 and 2. Mean of 4 kept
        // values: (1 + 3 + 4 + 6) / 4 = 3.5
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let r = mean_where(&a, None, Some(&mask.to_dyn())).unwrap();
        assert!((r.iter().next().copied().unwrap() - 3.5).abs() < 1e-12);
    }

    #[test]
    fn with_mask_rejects_incompatible_broadcast_shape() {
        // Mask rank compatible but length wrong: shape (2,) against a
        // (2, 3) input cannot broadcast (the 2 aligns with the last
        // dim which is 3, not the first which is 2).
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let bad_mask = Array::<bool, Ix1>::from_vec(Ix1::new([2]), vec![true, false]).unwrap();
        assert!(sum_with(&a, None, None, Some(&bad_mask.to_dyn())).is_err());
    }

    #[test]
    fn sum_with_mask_broadcast_scalar_like_length_1() {
        // A shape-(1,) mask is the scalar case — broadcasts to every
        // position of the input, which for `true` is an identity and
        // for `false` yields 0 (everything masked out).
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let true_mask = Array::<bool, Ix1>::from_vec(Ix1::new([1]), vec![true]).unwrap();
        let r = sum_with(&a, None, None, Some(&true_mask.to_dyn())).unwrap();
        // All positions kept: 1+2+3+4+5+6 = 21
        assert!((r.iter().next().copied().unwrap() - 21.0).abs() < 1e-12);

        let false_mask = Array::<bool, Ix1>::from_vec(Ix1::new([1]), vec![false]).unwrap();
        let r2 = sum_with(&a, None, Some(100.0), Some(&false_mask.to_dyn())).unwrap();
        // Nothing kept, initial = 100 → result = 100
        assert!((r2.iter().next().copied().unwrap() - 100.0).abs() < 1e-12);
    }

    #[test]
    fn into_reductions_match_allocating_versions_3d_axis_2() {
        // Cross-check every *_into against its allocating sibling on a
        // 3-D input, axis = last dim — guards against an off-by-one in
        // the in-place index walker.
        use ferray_core::Ix3;
        let data: Vec<f64> = (0..24).map(|i| f64::from(i) + 0.5).collect();
        let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();

        let s_alloc = sum(&a, Some(2)).unwrap();
        let mut s_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        sum_into(&a, Some(2), &mut s_out).unwrap();
        assert_eq!(s_alloc.as_slice().unwrap(), s_out.as_slice().unwrap());

        let p_alloc = prod(&a, Some(2)).unwrap();
        let mut p_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        prod_into(&a, Some(2), &mut p_out).unwrap();
        assert_eq!(p_alloc.as_slice().unwrap(), p_out.as_slice().unwrap());

        let mn_alloc = min(&a, Some(2)).unwrap();
        let mut mn_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        min_into(&a, Some(2), &mut mn_out).unwrap();
        assert_eq!(mn_alloc.as_slice().unwrap(), mn_out.as_slice().unwrap());

        let mx_alloc = max(&a, Some(2)).unwrap();
        let mut mx_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        max_into(&a, Some(2), &mut mx_out).unwrap();
        assert_eq!(mx_alloc.as_slice().unwrap(), mx_out.as_slice().unwrap());

        let me_alloc = mean(&a, Some(2)).unwrap();
        let mut me_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        mean_into(&a, Some(2), &mut me_out).unwrap();
        assert_eq!(me_alloc.as_slice().unwrap(), me_out.as_slice().unwrap());

        let v_alloc = var(&a, Some(2), 0).unwrap();
        let mut v_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        var_into(&a, Some(2), 0, &mut v_out).unwrap();
        // var output may have small numerical drift between the two-pass
        // and one-shot kernels — compare element-wise within tolerance.
        for (a, b) in v_alloc
            .as_slice()
            .unwrap()
            .iter()
            .zip(v_out.as_slice().unwrap())
        {
            assert!((a - b).abs() < 1e-10);
        }

        let sd_alloc = std_(&a, Some(2), 0).unwrap();
        let mut sd_out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![0.0; 6]).unwrap();
        std_into(&a, Some(2), 0, &mut sd_out).unwrap();
        for (a, b) in sd_alloc
            .as_slice()
            .unwrap()
            .iter()
            .zip(sd_out.as_slice().unwrap())
        {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // -- ptp --

    #[test]
    fn test_ptp_1d() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 5.0, 3.0, 9.0, 2.0]).unwrap();
        let r = ptp(&a, None).unwrap();
        assert_eq!(r.iter().copied().next().unwrap(), 8.0);
    }

    #[test]
    fn test_ptp_2d_axis() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 5.0, 3.0, 7.0, 2.0, 9.0])
            .unwrap();
        let r = ptp(&a, Some(1)).unwrap();
        // row 0: max=5, min=1 → 4; row 1: max=9, min=2 → 7
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![4.0, 7.0]);
    }

    #[test]
    fn test_ptp_empty_errs() {
        let a: Array<f64, Ix1> = Array::from_vec(Ix1::new([0]), vec![]).unwrap();
        assert!(ptp(&a, None).is_err());
    }

    // -- average --

    #[test]
    fn test_average_unweighted_matches_mean() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let r = average(&a, None, None).unwrap();
        assert!((r.iter().copied().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_average_weighted() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let w = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 1.0, 1.0, 7.0]).unwrap();
        // (1+2+3+4*7) / (1+1+1+7) = 34/10 = 3.4
        let r = average(&a, Some(&w), None).unwrap();
        assert!((r.iter().copied().next().unwrap() - 3.4).abs() < 1e-12);
    }

    #[test]
    fn test_average_weighted_axis() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let w = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .unwrap();
        // With uniform weights, axis-1 average = row mean = [2, 5]
        let r = average(&a, Some(&w), Some(1)).unwrap();
        let data: Vec<f64> = r.iter().copied().collect();
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_average_weights_zero_sum_errs() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let w = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![0.0, 0.0, 0.0]).unwrap();
        assert!(average(&a, Some(&w), None).is_err());
    }

    #[test]
    fn test_average_shape_mismatch_errs() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let w = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0; 4]).unwrap();
        assert!(average(&a, Some(&w), None).is_err());
    }

    // ----------------------------------------------------------------------
    // var_as_f64 / std_as_f64 integer promotion (#170)
    // ----------------------------------------------------------------------

    #[test]
    fn var_as_f64_promotes_int_input() {
        let a = Array::<i64, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = var_as_f64(&a, None, 0).unwrap();
        // var([1,2,3,4,5]) with ddof=0 = 2.0
        assert!((v.iter().next().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn var_as_f64_ddof_1() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = var_as_f64(&a, None, 1).unwrap();
        // var([1,2,3,4,5]) with ddof=1 = 2.5
        assert!((v.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn std_as_f64_promotes_int_input() {
        let a = Array::<u32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let s = std_as_f64(&a, None, 0).unwrap();
        // std([1,2,3,4,5]) with ddof=0 = sqrt(2.0)
        assert!((s.iter().next().unwrap() - 2.0_f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn var_as_f64_axis_2d() {
        use ferray_core::dimension::Ix2;
        let a = Array::<i64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        // var along axis 0 of [[1,2,3],[4,5,6]] is [2.25, 2.25, 2.25]
        let v = var_as_f64(&a, Some(0), 0).unwrap();
        for x in v.iter() {
            assert!((x - 2.25).abs() < 1e-12);
        }
    }

    // ----------------------------------------------------------------------
    // f32 sibling tests (#185) — exercises the f32 SIMD path added in #173
    // and the generic Float-bound reduction paths.
    // ----------------------------------------------------------------------

    fn arr1d_f32(data: Vec<f32>) -> Array<f32, Ix1> {
        Array::<f32, Ix1>::from_vec(Ix1::new([data.len()]), data).unwrap()
    }

    #[test]
    fn sum_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = sum(&a, None).unwrap();
        assert!((s.iter().next().copied().unwrap() - 15.0).abs() < 1e-6);
    }

    #[test]
    fn sum_f32_large_for_simd() {
        // Big enough to actually exercise the SIMD pairwise sum kernel.
        let n = 4096;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
        let a = arr1d_f32(data);
        let s = sum(&a, None).unwrap();
        let expected = 0.1 * (n as f32) * ((n - 1) as f32) / 2.0;
        let got = s.iter().copied().next().unwrap();
        assert!((got - expected).abs() / expected < 1e-4);
    }

    #[test]
    fn mean_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let m = mean(&a, None).unwrap();
        assert!((m.iter().next().copied().unwrap() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn var_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let v = var(&a, None, 0).unwrap();
        assert!((v.iter().next().copied().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn var_f32_large_for_simd() {
        // Exercises the simd_sum_sq_diff_f32 kernel.
        let n = 4096;
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let a = arr1d_f32(data);
        let v = var(&a, None, 0).unwrap();
        let expected = 0.25 * ((n * n - 1) as f32) / 12.0;
        let got = v.iter().copied().next().unwrap();
        assert!((got - expected).abs() / expected < 1e-3);
    }

    #[test]
    fn std_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = std_(&a, None, 0).unwrap();
        assert!((s.iter().next().copied().unwrap() - 2.0_f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn min_max_f32_basic() {
        let a = arr1d_f32(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let mn = min(&a, None).unwrap();
        let mx = max(&a, None).unwrap();
        assert_eq!(mn.iter().next().copied().unwrap(), 1.0);
        assert_eq!(mx.iter().next().copied().unwrap(), 9.0);
    }

    #[test]
    fn prod_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0]);
        let p = prod(&a, None).unwrap();
        assert!((p.iter().next().copied().unwrap() - 24.0).abs() < 1e-6);
    }

    #[test]
    fn argmin_argmax_f32_basic() {
        let a = arr1d_f32(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let amin = argmin(&a, None).unwrap();
        let amax = argmax(&a, None).unwrap();
        assert_eq!(amin.iter().next().copied().unwrap(), 1);
        assert_eq!(amax.iter().next().copied().unwrap(), 5);
    }

    #[test]
    fn cumsum_f32_basic() {
        let a = arr1d_f32(vec![1.0, 2.0, 3.0, 4.0]);
        let c = cumsum(&a, None).unwrap();
        let v: Vec<f32> = c.iter().copied().collect();
        for (got, expected) in v.iter().zip(&[1.0_f32, 3.0, 6.0, 10.0]) {
            assert!((got - expected).abs() < 1e-6);
        }
    }
}
