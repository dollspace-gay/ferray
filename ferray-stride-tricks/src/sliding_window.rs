// ferray-stride-tricks: Sliding window view (REQ-1)
//
// Implements `numpy.lib.stride_tricks.sliding_window_view`.
// Returns an immutable view whose shape is (n - w + 1, ..., w0, w1, ...)
// by reusing the source array's memory with adjusted strides.

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, ArrayView, Element};

/// Create a read-only sliding window view of an array.
///
/// The returned view has shape `[n0 - w0 + 1, n1 - w1 + 1, ..., w0, w1, ...]`
/// where `n_i` is the size of the i-th axis of the source array and `w_i` is
/// the corresponding window size. The view shares memory with the source
/// array — no data is copied.
///
/// This is equivalent to `numpy.lib.stride_tricks.sliding_window_view`.
///
/// # Arguments
///
/// * `array` — the source array.
/// * `window_shape` — the window size along each axis. Must have the same
///   number of dimensions as `array`, and each window size must be at least 1
///   and at most the corresponding array dimension.
///
/// # Errors
///
/// Returns `FerrayError::InvalidValue` if:
/// - `window_shape` length does not match the array's number of dimensions.
/// - Any window dimension is 0 or exceeds the corresponding array dimension.
///
/// # Examples
///
/// ```
/// # use ferray_core::{Array, dimension::Ix1};
/// # use ferray_stride_tricks::sliding_window_view;
/// let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
/// let v = sliding_window_view(&a, &[3]).unwrap();
/// assert_eq!(v.shape(), &[3, 3]);
/// // First window: [1, 2, 3], second: [2, 3, 4], third: [3, 4, 5]
/// let data: Vec<i32> = v.iter().copied().collect();
/// assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
/// ```
pub fn sliding_window_view<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    window_shape: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    let src_shape = array.shape();
    let src_strides = array.strides();
    let ndim = src_shape.len();

    // Validate window_shape length.
    if window_shape.len() != ndim {
        return Err(FerrayError::invalid_value(format!(
            "window_shape length ({}) must match array ndim ({})",
            window_shape.len(),
            ndim,
        )));
    }

    // Validate each window dimension.
    for (i, (&w, &n)) in window_shape.iter().zip(src_shape.iter()).enumerate() {
        if w == 0 {
            return Err(FerrayError::invalid_value(format!(
                "window_shape[{i}] must be >= 1, got 0",
            )));
        }
        if w > n {
            return Err(FerrayError::invalid_value(format!(
                "window_shape[{i}] ({w}) exceeds array dimension {i} ({n})",
            )));
        }
    }

    // Build the output shape: [n0-w0+1, n1-w1+1, ..., w0, w1, ...]
    let mut out_shape = Vec::with_capacity(2 * ndim);
    for i in 0..ndim {
        out_shape.push(src_shape[i] - window_shape[i] + 1);
    }
    for &w in window_shape {
        out_shape.push(w);
    }

    // Build the output strides: [s0, s1, ..., s0, s1, ...]
    // The sliding axes use the original strides (stepping one position),
    // and the window axes also use the original strides (stepping within
    // the window).
    let mut out_strides = Vec::with_capacity(2 * ndim);
    for &s in src_strides {
        // Owned arrays (Array<T, D>) are always C-contiguous with non-negative
        // strides. If this ever fires, it indicates a bug in ferray-core.
        debug_assert!(
            s >= 0,
            "sliding_window_view: unexpected negative stride {s}"
        );
        out_strides.push(s as usize);
    }
    for &s in src_strides {
        out_strides.push(s as usize);
    }

    // SAFETY: The sliding window view only accesses memory within the
    // original array's allocation. Each window starts at a valid offset
    // and extends for exactly `window_shape[i]` elements along each axis.
    // The view is immutable, so no aliasing concerns arise.
    let ptr = array.as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, &out_shape, &out_strides) };

    Ok(view)
}

/// Sliding-window view restricted to a subset of axes (#523).
///
/// `axes` selects which axes of `array` to slide over. `window_shape`
/// must have the same length as `axes`. Other axes are kept at their
/// full size in the output.
///
/// Output shape: `[outer_dims..., window_shape...]` where each `outer_dim`
/// is `src_shape[i] - window_shape[axes.position(i)] + 1` for windowed
/// axes and `src_shape[i]` for non-windowed axes. Window axes appear at
/// the end in the order given by `axes`. This matches numpy's
/// `sliding_window_view(x, window_shape, axis=axes)`.
///
/// # Errors
/// - `FerrayError::InvalidValue` if `axes.len() != window_shape.len()`,
///   any axis is out of bounds, any window dimension is 0, or any
///   window exceeds its axis size.
pub fn sliding_window_view_axis<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    window_shape: &[usize],
    axes: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    let src_shape = array.shape();
    let src_strides = array.strides();
    let ndim = src_shape.len();

    if window_shape.len() != axes.len() {
        return Err(FerrayError::invalid_value(format!(
            "window_shape length ({}) must match axes length ({})",
            window_shape.len(),
            axes.len(),
        )));
    }
    for &ax in axes {
        if ax >= ndim {
            return Err(FerrayError::invalid_value(format!(
                "axis {ax} out of bounds for array with {ndim} dimensions"
            )));
        }
    }
    for (slot, (&ax, &w)) in axes.iter().zip(window_shape.iter()).enumerate() {
        if w == 0 {
            return Err(FerrayError::invalid_value(format!(
                "window_shape[{slot}] must be >= 1, got 0"
            )));
        }
        if w > src_shape[ax] {
            return Err(FerrayError::invalid_value(format!(
                "window_shape[{slot}] ({w}) exceeds array axis {ax} (size {})",
                src_shape[ax],
            )));
        }
    }

    // Outer (per-input-axis) shape: replace windowed axes with
    // src - w + 1, leave others as src.
    let mut out_shape = Vec::with_capacity(ndim + axes.len());
    for (i, &n) in src_shape.iter().enumerate() {
        if let Some(slot) = axes.iter().position(|&a| a == i) {
            out_shape.push(n - window_shape[slot] + 1);
        } else {
            out_shape.push(n);
        }
    }
    // Window axes (in axes order).
    for &w in window_shape {
        out_shape.push(w);
    }

    // Strides: outer = original src_strides, window = src_strides for
    // each axis in axes (preserves the within-window step).
    let mut out_strides = Vec::with_capacity(ndim + axes.len());
    for &s in src_strides {
        debug_assert!(s >= 0);
        out_strides.push(s as usize);
    }
    for &ax in axes {
        debug_assert!(src_strides[ax] >= 0);
        out_strides.push(src_strides[ax] as usize);
    }

    let ptr = array.as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, &out_shape, &out_strides) };
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

    // ---- sliding_window_view_axis (#523) -------------------------------

    #[test]
    fn axis_partial_window_on_2d_axis0() {
        // 4x3 array, window of length 2 on axis 0 only.
        // Output shape: (3, 3, 2). For each row i in 0..3, the row pair
        // (i, i+1) is preserved as the trailing window axis.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([4, 3]), (1..=12).collect()).unwrap();
        let v = sliding_window_view_axis(&a, &[2], &[0]).unwrap();
        assert_eq!(v.shape(), &[3, 3, 2]);
        // First "outer" position [0, 0] — should hold rows 0 and 1 col 0:
        // 1 (row 0 col 0) and 4 (row 1 col 0).
        let data: Vec<i32> = v.iter().copied().collect();
        // Verify first window at [0, 0, 0..2] is [1, 4].
        assert_eq!(data[0], 1);
        assert_eq!(data[1], 4);
    }

    #[test]
    fn axis_window_on_axis1_only() {
        // 2x5, window of length 3 on axis 1.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 5]), (1..=10).collect()).unwrap();
        let v = sliding_window_view_axis(&a, &[3], &[1]).unwrap();
        assert_eq!(v.shape(), &[2, 3, 3]);
    }

    #[test]
    fn axis_mismatch_lengths_errors() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 3]), (1..=9).collect()).unwrap();
        // 2 windows but 1 axis
        assert!(sliding_window_view_axis(&a, &[2, 2], &[0]).is_err());
    }

    #[test]
    fn axis_out_of_bounds_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), (1..=5).collect()).unwrap();
        assert!(sliding_window_view_axis(&a, &[3], &[5]).is_err());
    }

    #[test]
    fn axis_window_too_big_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(sliding_window_view_axis(&a, &[5], &[0]).is_err());
    }

    #[test]
    fn sliding_window_1d_size3() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = sliding_window_view(&a, &[3]).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
    }

    #[test]
    fn sliding_window_1d_full() {
        // Window size equals array size: single window
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![10.0, 20.0, 30.0]).unwrap();
        let v = sliding_window_view(&a, &[3]).unwrap();
        assert_eq!(v.shape(), &[1, 3]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn sliding_window_1d_size1() {
        // Window size 1: each element is its own window
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        let v = sliding_window_view(&a, &[1]).unwrap();
        assert_eq!(v.shape(), &[4, 1]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn sliding_window_2d() {
        // 3x4 array with 2x2 window -> (2, 3, 2, 2) output.
        // Issue #290: check element values, not just shape.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (1..=12).collect()).unwrap();
        let v = sliding_window_view(&a, &[2, 2]).unwrap();
        assert_eq!(v.shape(), &[2, 3, 2, 2]);

        // Spot-check a few windows. Source:
        //   1  2  3  4
        //   5  6  7  8
        //   9 10 11 12
        // Output shape (2, 3, 2, 2) means (win_row, win_col, row, col).
        // Row-major flat indexing: offset = (wr * 3 + wc) * 4.
        let data: Vec<i32> = v.iter().copied().collect();
        // Window (0, 0): rows 0-1, cols 0-1 -> [1, 2, 5, 6].
        assert_eq!(&data[0..4], &[1, 2, 5, 6]);
        // Window (0, 2): rows 0-1, cols 2-3 -> [3, 4, 7, 8]. Offset 8.
        assert_eq!(&data[8..12], &[3, 4, 7, 8]);
        // Window (1, 1): rows 1-2, cols 1-2 -> [6, 7, 10, 11]. Offset (1*3+1)*4 = 16.
        assert_eq!(&data[16..20], &[6, 7, 10, 11]);
        // Window (1, 2): rows 1-2, cols 2-3 -> [7, 8, 11, 12]. Offset 20.
        assert_eq!(&data[20..24], &[7, 8, 11, 12]);
    }

    #[test]
    fn sliding_window_zero_copy() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let v = sliding_window_view(&a, &[3]).unwrap();
        assert_eq!(v.as_ptr(), a.as_ptr());
    }

    #[test]
    fn sliding_window_wrong_ndim() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        assert!(sliding_window_view(&a, &[2, 2]).is_err());
    }

    #[test]
    fn sliding_window_zero_window() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        assert!(sliding_window_view(&a, &[0]).is_err());
    }

    #[test]
    fn sliding_window_exceeds_dim() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(sliding_window_view(&a, &[4]).is_err());
    }
}
