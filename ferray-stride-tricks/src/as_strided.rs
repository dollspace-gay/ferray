// ferray-stride-tricks: Safe and unsafe as_strided variants (REQ-5, REQ-6, REQ-7)
//
// `as_strided` validates that the requested strides do not produce
// overlapping memory accesses, while `as_strided_unchecked` skips that
// check for callers who need overlapping views (e.g. Toeplitz matrices).

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, ArrayView, Element};

use crate::overlap_check::{all_offsets_in_bounds, has_overlapping_strides};

/// Create a view of an array with the given shape and strides, after
/// validating that the strides do not produce overlapping memory accesses.
///
/// This is the safe variant of `as_strided_unchecked`. It performs two
/// runtime checks:
///
/// 1. **Bounds check** — every offset reachable through the new
///    (shape, strides) pair must lie within the source array's buffer.
/// 2. **Overlap check** — no two distinct logical indices may map to the
///    same physical element.
///
/// If either check fails, an error is returned.
///
/// Strides are given in units of elements (not bytes), and must be
/// non-negative.
///
/// # Errors
///
/// Returns `FerrayError::InvalidValue` if:
/// - `shape` and `strides` have different lengths.
/// - Any stride is negative (use `as_strided_unchecked` for advanced patterns).
/// - Any computed offset falls outside the source buffer.
/// - The strides produce overlapping memory accesses.
///
/// # Examples
///
/// ```
/// # use ferray_core::{Array, dimension::Ix1};
/// # use ferray_stride_tricks::as_strided;
/// let a = Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// // Reshape to 2x3 (non-overlapping):
/// let v = as_strided(&a, &[2, 3], &[3, 1]).unwrap();
/// assert_eq!(v.shape(), &[2, 3]);
/// let data: Vec<f64> = v.iter().copied().collect();
/// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn as_strided<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    shape: &[usize],
    strides: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    // Validate shape and strides have the same length.
    if shape.len() != strides.len() {
        return Err(FerrayError::invalid_value(format!(
            "shape length ({}) must equal strides length ({})",
            shape.len(),
            strides.len(),
        )));
    }

    // Convert usize strides to isize for the overlap check.
    let istrides: Vec<isize> = strides.iter().map(|&s| s as isize).collect();

    // Empty views are trivially valid.
    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        // SAFETY: zero elements means no memory access.
        let ptr = array.as_ptr();
        let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
        return Ok(view);
    }

    // Bounds check: all offsets must be within the source buffer.
    let buf_len = array.size();
    if !all_offsets_in_bounds(shape, &istrides, buf_len) {
        return Err(FerrayError::invalid_value(format!(
            "as_strided: strides {:?} with shape {:?} would access elements \
             outside the source buffer of length {}",
            strides, shape, buf_len,
        )));
    }

    // Overlap check: no two indices may map to the same element.
    if has_overlapping_strides(shape, &istrides) {
        return Err(FerrayError::invalid_value(format!(
            "as_strided: strides {:?} with shape {:?} produce overlapping \
             memory accesses; use as_strided_unchecked for overlapping views",
            strides, shape,
        )));
    }

    // SAFETY: bounds and overlap checks passed. All offsets are in-bounds
    // and distinct, so this is a valid immutable view.
    let ptr = array.as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
    Ok(view)
}

/// Create a view of an array with arbitrary shape and strides, without
/// checking for overlapping memory accesses.
///
/// This is the unsafe counterpart to [`as_strided`]. It still validates
/// that all offsets are within bounds, but it does **not** check for
/// overlapping strides. This makes it suitable for constructing views
/// like sliding windows and Toeplitz matrices where the same element is
/// intentionally accessed through multiple indices.
///
/// # Safety
///
/// The caller must uphold the following invariants:
///
/// 1. **No concurrent mutation** — for the entire lifetime of the returned
///    view, no mutable reference to any element that might overlap may
///    exist. Because the view is immutable, concurrent reads are safe, but
///    the caller must ensure no `&mut` alias exists.
///
/// 2. **Logical correctness** — the caller is responsible for ensuring the
///    stride pattern produces the intended semantics. The library cannot
///    verify this at runtime.
///
/// # Correct usage
///
/// ```
/// # use ferray_core::{Array, dimension::Ix1};
/// # use ferray_stride_tricks::as_strided_unchecked;
/// let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
/// // Sliding window of size 3: overlapping is intentional and safe here
/// // because the source is immutably borrowed for the view's lifetime.
/// let v = unsafe { as_strided_unchecked(&a, &[3, 3], &[1, 1]).unwrap() };
/// assert_eq!(v.shape(), &[3, 3]);
/// let data: Vec<i32> = v.iter().copied().collect();
/// assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
/// ```
///
/// # Incorrect usage (do NOT do this)
///
/// ```no_run
/// # use ferray_core::{Array, dimension::Ix1};
/// # use ferray_stride_tricks::as_strided_unchecked;
/// let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
/// let v = unsafe { as_strided_unchecked(&a, &[3, 3], &[1, 1]).unwrap() };
/// // BAD: mutating `a` while `v` is alive would be UB.
/// // drop(a); // <-- even moving `a` would invalidate `v`
/// // let _ = v.iter().count(); // <-- use-after-free
/// ```
///
/// # Errors
///
/// Returns `FerrayError::InvalidValue` if:
/// - `shape` and `strides` have different lengths.
/// - Any computed offset falls outside the source buffer.
pub unsafe fn as_strided_unchecked<'a, T: Element, D: Dimension>(
    array: &'a Array<T, D>,
    shape: &[usize],
    strides: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>> {
    // Validate shape and strides have the same length.
    if shape.len() != strides.len() {
        return Err(FerrayError::invalid_value(format!(
            "shape length ({}) must equal strides length ({})",
            shape.len(),
            strides.len(),
        )));
    }

    let istrides: Vec<isize> = strides.iter().map(|&s| s as isize).collect();

    // Empty views are trivially valid.
    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        let ptr = array.as_ptr();
        let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
        return Ok(view);
    }

    // Bounds check: all offsets must be within the source buffer.
    let buf_len = array.size();
    if !all_offsets_in_bounds(shape, &istrides, buf_len) {
        return Err(FerrayError::invalid_value(format!(
            "as_strided_unchecked: strides {:?} with shape {:?} would access \
             elements outside the source buffer of length {}",
            strides, shape, buf_len,
        )));
    }

    // SAFETY: bounds check passed. Overlap check is the caller's
    // responsibility per the Safety contract above.
    let ptr = array.as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    #[test]
    fn as_strided_contiguous_reshape() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = as_strided(&a, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_non_contiguous() {
        // Take every other element: shape (3,), stride (2,) from buffer of 6
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = as_strided(&a, &[3], &[2]).unwrap();
        assert_eq!(v.shape(), &[3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 3, 5]);
    }

    #[test]
    fn as_strided_rejects_overlap() {
        // Sliding window strides overlap
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let result = as_strided(&a, &[3, 3], &[1, 1]);
        assert!(result.is_err());
    }

    #[test]
    fn as_strided_rejects_out_of_bounds() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        // stride 2 with shape 3 needs offsets 0, 2, 4 — but buf_len is 3
        let result = as_strided(&a, &[3], &[2]);
        assert!(result.is_err());
    }

    #[test]
    fn as_strided_shape_stride_mismatch() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let result = as_strided(&a, &[2, 3], &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn as_strided_empty_shape() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = as_strided(&a, &[0], &[1]).unwrap();
        assert_eq!(v.shape(), &[0]);
        assert_eq!(v.size(), 0);
    }

    #[test]
    fn as_strided_zero_copy() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let v = as_strided(&a, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(v.as_ptr(), a.as_ptr());
    }

    // --- Unsafe variant tests ---

    #[test]
    fn as_strided_unchecked_overlapping() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = unsafe { as_strided_unchecked(&a, &[3, 3], &[1, 1]).unwrap() };
        assert_eq!(v.shape(), &[3, 3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
    }

    #[test]
    fn as_strided_unchecked_rejects_oob() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let result = unsafe { as_strided_unchecked(&a, &[3], &[2]) };
        assert!(result.is_err());
    }

    #[test]
    fn as_strided_unchecked_shape_stride_mismatch() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let result = unsafe { as_strided_unchecked(&a, &[2, 3], &[1]) };
        assert!(result.is_err());
    }

    #[test]
    fn as_strided_unchecked_broadcast_pattern() {
        // stride 0 on first axis: broadcast row
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let v = unsafe { as_strided_unchecked(&a, &[4, 3], &[0, 1]).unwrap() };
        assert_eq!(v.shape(), &[4, 3]);
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(
            data,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn as_strided_unchecked_empty() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = unsafe { as_strided_unchecked(&a, &[0], &[1]).unwrap() };
        assert_eq!(v.shape(), &[0]);
        assert_eq!(v.size(), 0);
    }
}
