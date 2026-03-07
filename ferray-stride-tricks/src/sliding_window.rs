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
                "window_shape[{}] must be >= 1, got 0",
                i,
            )));
        }
        if w > n {
            return Err(FerrayError::invalid_value(format!(
                "window_shape[{}] ({}) exceeds array dimension {} ({})",
                i, w, i, n,
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};

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
        // 3x4 array with 2x2 window -> (2, 3, 2, 2) output
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (1..=12).collect()).unwrap();
        let v = sliding_window_view(&a, &[2, 2]).unwrap();
        assert_eq!(v.shape(), &[2, 3, 2, 2]);
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
