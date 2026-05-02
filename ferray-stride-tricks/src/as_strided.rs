// ferray-stride-tricks: Safe and unsafe as_strided variants (REQ-5, REQ-6, REQ-7)
//
// `as_strided` validates that the requested strides do not produce
// overlapping memory accesses, while `as_strided_unchecked` skips that
// check for callers who need overlapping views (e.g. Toeplitz matrices).

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, ArrayView, Element};

use crate::overlap_check::{all_offsets_in_bounds, has_overlapping_strides};

/// A read-only source for [`as_strided`] / [`as_strided_unchecked`].
///
/// Implemented for both `&Array<T, D>` and `&ArrayView<'_, T, D>` so
/// callers can hand either type to the `as_strided` entry points
/// without `.to_owned()` materialization (#174, #525).
pub trait StridedSource<T: Element> {
    /// Raw pointer to the first element.
    fn strided_as_ptr(&self) -> *const T;
    /// Number of elements reachable from `strided_as_ptr`.
    fn strided_size(&self) -> usize;
}

impl<T: Element, D: Dimension> StridedSource<T> for Array<T, D> {
    #[inline]
    fn strided_as_ptr(&self) -> *const T {
        self.as_ptr()
    }
    #[inline]
    fn strided_size(&self) -> usize {
        self.size()
    }
}

impl<T: Element, D: Dimension> StridedSource<T> for ArrayView<'_, T, D> {
    #[inline]
    fn strided_as_ptr(&self) -> *const T {
        self.as_ptr()
    }
    #[inline]
    fn strided_size(&self) -> usize {
        self.size()
    }
}

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
/// non-negative. See [`as_strided_unchecked`] for views that need
/// negative strides or overlapping access patterns.
///
/// # View size vs. source buffer capacity
///
/// The returned `ArrayView::size()` reports the **logical** element
/// count `shape.iter().product()`, not the number of unique elements
/// touched or the capacity of the source buffer. If you need the
/// underlying buffer length you must keep a handle to the source
/// `Array` (#285). This mirrors ndarray's own convention.
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
pub fn as_strided<'a, T, S>(
    source: &'a S,
    shape: &[usize],
    strides: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>>
where
    T: Element,
    S: StridedSource<T>,
{
    let istrides = validate_and_bounds_check(source, shape, strides, "as_strided")?;

    // Overlap check: no two indices may map to the same element.
    // Empty views fall through here with an empty istrides vec — that's
    // fine, has_overlapping_strides returns false for zero elements.
    let n_elements: usize = shape.iter().product();
    if n_elements > 0 && has_overlapping_strides(shape, &istrides) {
        return Err(FerrayError::invalid_value(format!(
            "as_strided: strides {strides:?} with shape {shape:?} produce overlapping \
             memory accesses; use as_strided_unchecked for overlapping views",
        )));
    }

    // SAFETY: bounds and overlap checks passed. All offsets are in-bounds
    // and distinct, so this is a valid immutable view.
    let ptr = source.strided_as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
    Ok(view)
}

/// Signed-stride sibling of [`as_strided`] (#737).
///
/// Accepts `strides: &[isize]` so callers can construct reversed
/// views, transposed views with reversed axes, etc. `start_offset`
/// shifts the base pointer (in element units) before strides are
/// applied — necessary so negative strides land inside the buffer.
///
/// Bounds check: `start_offset + min_reachable_offset` must be in
/// `[0, buf_len)` and `start_offset + max_reachable_offset` must be
/// in `[0, buf_len)`. Overlap check rejects layouts where two
/// distinct multi-indices map to the same memory location.
///
/// # Errors
/// `FerrayError::InvalidValue` if dimensions don't match,
/// the stride pattern produces out-of-bounds offsets, or different
/// indices alias the same element.
pub fn as_strided_signed<'a, T, S>(
    source: &'a S,
    shape: &[usize],
    strides: &[isize],
    start_offset: usize,
) -> FerrayResult<ArrayView<'a, T, IxDyn>>
where
    T: Element,
    S: StridedSource<T>,
{
    validate_signed_bounds(source, shape, strides, start_offset, "as_strided_signed")?;
    let n_elements: usize = shape.iter().product();
    if n_elements > 0 && crate::overlap_check::has_overlapping_strides(shape, strides) {
        return Err(FerrayError::invalid_value(format!(
            "as_strided_signed: strides {strides:?} with shape {shape:?} produce \
             overlapping memory accesses; use as_strided_signed_unchecked for \
             overlapping views",
        )));
    }
    // Re-interpret each isize stride as usize via the bit pattern;
    // ndarray internally treats stride bytes as isize, so this
    // preserves the negative-stride semantics through the
    // ArrayView constructor.
    let strides_usize: Vec<usize> = strides.iter().map(|&s| s as usize).collect();
    // SAFETY: bounds + overlap checks passed. start_offset shifts
    // the base pointer to the location from which negative strides
    // reach into the source buffer.
    let base = source.strided_as_ptr();
    let ptr = unsafe { base.add(start_offset) };
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, &strides_usize) };
    Ok(view)
}

/// Unchecked sibling of [`as_strided_signed`].
///
/// Same as `as_strided_signed` but skips the overlap check;
/// callers must uphold the no-mutation / logical-correctness
/// contract documented on [`as_strided_unchecked`].
///
/// # Safety
/// See [`as_strided_unchecked`] — same invariants apply.
///
/// # Errors
/// `FerrayError::InvalidValue` if dimensions don't match or the
/// stride pattern produces out-of-bounds offsets.
pub unsafe fn as_strided_signed_unchecked<'a, T, S>(
    source: &'a S,
    shape: &[usize],
    strides: &[isize],
    start_offset: usize,
) -> FerrayResult<ArrayView<'a, T, IxDyn>>
where
    T: Element,
    S: StridedSource<T>,
{
    validate_signed_bounds(
        source,
        shape,
        strides,
        start_offset,
        "as_strided_signed_unchecked",
    )?;
    let strides_usize: Vec<usize> = strides.iter().map(|&s| s as usize).collect();
    let base = source.strided_as_ptr();
    // SAFETY: caller's contract.
    let ptr = unsafe { base.add(start_offset) };
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, &strides_usize) };
    Ok(view)
}

fn validate_signed_bounds<T, S>(
    source: &S,
    shape: &[usize],
    strides: &[isize],
    start_offset: usize,
    fn_name: &str,
) -> FerrayResult<()>
where
    T: Element,
    S: StridedSource<T>,
{
    if shape.len() != strides.len() {
        return Err(FerrayError::invalid_value(format!(
            "{fn_name}: shape length ({}) must equal strides length ({})",
            shape.len(),
            strides.len(),
        )));
    }
    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        return Ok(());
    }

    // Negative strides aren't yet supported through ferray-core's
    // safe `ArrayView::from_shape_ptr`; ndarray's raw-view path
    // would accept them but ferray-core doesn't yet expose that
    // constructor (#744). Reject with a clear diagnostic so users
    // see what's missing rather than a deep ndarray error.
    if strides.iter().any(|&s| s < 0) {
        return Err(FerrayError::invalid_value(format!(
            "{fn_name}: negative strides aren't supported yet — ferray-core needs \
             RawArrayView wiring (tracked in #744). strides={strides:?}"
        )));
    }

    // Compute min and max offsets reachable.
    let mut min_offset: isize = 0;
    let mut max_offset: isize = 0;
    for (i, &s) in strides.iter().enumerate() {
        if shape[i] == 0 {
            continue;
        }
        let extent = (shape[i] as isize - 1) * s;
        if extent > 0 {
            max_offset += extent;
        } else {
            min_offset += extent;
        }
    }
    let buf_len = source.strided_size() as isize;
    let lo = start_offset as isize + min_offset;
    let hi = start_offset as isize + max_offset;
    if lo < 0 || hi >= buf_len {
        return Err(FerrayError::invalid_value(format!(
            "{fn_name}: strides {strides:?} with shape {shape:?} and start_offset \
             {start_offset} would access elements outside the source buffer of length \
             {buf_len} (reachable range [{lo}, {hi}])"
        )));
    }
    Ok(())
}

/// Shared validation + bounds-check between [`as_strided`] and
/// [`as_strided_unchecked`]. Returns the signed stride vector on
/// success so the caller can feed it back into
/// [`has_overlapping_strides`] without re-collecting (#287).
fn validate_and_bounds_check<T, S>(
    source: &S,
    shape: &[usize],
    strides: &[usize],
    fn_name: &str,
) -> FerrayResult<Vec<isize>>
where
    T: Element,
    S: StridedSource<T>,
{
    if shape.len() != strides.len() {
        return Err(FerrayError::invalid_value(format!(
            "shape length ({}) must equal strides length ({})",
            shape.len(),
            strides.len(),
        )));
    }

    let istrides: Vec<isize> = strides.iter().map(|&s| s as isize).collect();

    // Empty views are trivially valid — skip the bounds check.
    let n_elements: usize = shape.iter().product();
    if n_elements == 0 {
        return Ok(istrides);
    }

    let buf_len = source.strided_size();
    if !all_offsets_in_bounds(shape, &istrides, buf_len) {
        return Err(FerrayError::invalid_value(format!(
            "{fn_name}: strides {strides:?} with shape {shape:?} would access elements \
             outside the source buffer of length {buf_len}",
        )));
    }

    Ok(istrides)
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
pub unsafe fn as_strided_unchecked<'a, T, S>(
    source: &'a S,
    shape: &[usize],
    strides: &[usize],
) -> FerrayResult<ArrayView<'a, T, IxDyn>>
where
    T: Element,
    S: StridedSource<T>,
{
    // Shared validation / bounds check with `as_strided` — we just
    // skip the overlap check (that's what `_unchecked` means, and the
    // caller has accepted the safety contract documented above).
    let _istrides = validate_and_bounds_check(source, shape, strides, "as_strided_unchecked")?;

    // SAFETY: bounds check passed. Overlap check is the caller's
    // responsibility per the Safety contract above.
    let ptr = source.strided_as_ptr();
    let view = unsafe { ArrayView::from_shape_ptr(ptr, shape, strides) };
    Ok(view)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    // ---- as_strided_signed (#737) --------------------------------------

    #[test]
    fn signed_with_positive_strides_works_like_as_strided() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let v = as_strided_signed(&a, &[2, 3], &[3, 1], 0).unwrap();
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn signed_with_start_offset_advances_base() {
        // Take elements [2, 3, 4] from [1, 2, 3, 4, 5] using
        // start_offset=1 and stride=1.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let v = as_strided_signed(&a, &[3], &[1], 1).unwrap();
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![2, 3, 4]);
    }

    #[test]
    fn signed_zero_stride_overlap_rejected_by_default() {
        // Stride 0 == every index aliases — checked variant rejects.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let err = as_strided_signed(&a, &[5], &[0], 0).unwrap_err();
        assert!(err.to_string().contains("overlapping"));
    }

    #[test]
    fn signed_unchecked_allows_zero_stride_broadcast() {
        // Stride 0 with the unchecked variant: broadcast over a
        // length-N shape yields N copies of the base element.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let v = unsafe { as_strided_signed_unchecked(&a, &[5], &[0], 0) }.unwrap();
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![10, 10, 10, 10, 10]);
    }

    #[test]
    fn signed_negative_stride_currently_rejected_with_diagnostic() {
        // Negative strides need ferray-core RawArrayView wiring
        // (tracked in #744). The signed entry rejects them with a
        // pointer to that issue so callers know what's blocked.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let err = as_strided_signed(&a, &[5], &[-1], 4).unwrap_err();
        assert!(err.to_string().contains("#744"), "got: {err}");
    }

    #[test]
    fn signed_offset_beyond_buffer_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        assert!(as_strided_signed(&a, &[2], &[1], 5).is_err());
    }

    #[test]
    fn signed_dimension_mismatch_errors() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        assert!(as_strided_signed(&a, &[2, 3], &[1], 0).is_err());
    }

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

    // ----- StridedSource on ArrayView (#174, #525) -----

    #[test]
    fn as_strided_accepts_array_view() {
        // The caller has a view (e.g. from a slice) and wants to
        // re-stride it without `.to_owned()`.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let view = a.view();
        let v = as_strided(&view, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn as_strided_view_zero_copy() {
        // The returned view should share the original buffer.
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([6]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = a.view();
        let v = as_strided(&view, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(v.as_ptr(), a.as_ptr());
    }

    #[test]
    fn as_strided_unchecked_accepts_array_view() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let view = a.view();
        let v = unsafe { as_strided_unchecked(&view, &[3, 3], &[1, 1]).unwrap() };
        let data: Vec<i32> = v.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 2, 3, 4, 3, 4, 5]);
    }

    // ----- Property tests (#289) -----

    use proptest::prelude::*;

    proptest! {
        /// Any contiguous row-major (shape, strides) pair on a buffer
        /// whose total size matches `product(shape)` must produce the
        /// same element values as the source array, in the same order.
        #[test]
        fn prop_as_strided_row_major_round_trip(
            rows in 1usize..8,
            cols in 1usize..8,
        ) {
            let n = rows * cols;
            let data: Vec<i32> = (0..n as i32).collect();
            let a = Array::<i32, Ix1>::from_vec(Ix1::new([n]), data.clone()).unwrap();
            // Row-major strides on a (rows, cols) reshape.
            let v = as_strided(&a, &[rows, cols], &[cols, 1]).unwrap();
            let collected: Vec<i32> = v.iter().copied().collect();
            prop_assert_eq!(collected, data);
        }

        /// Any "take every k-th element" 1-D strided view must produce
        /// the expected subset and never report overlap (the strides
        /// are always strictly larger than 1).
        #[test]
        fn prop_as_strided_stride_k_subset(
            buf in 1usize..32,
            k in 1usize..4,
        ) {
            let data: Vec<i32> = (0..buf as i32).collect();
            let a = Array::<i32, Ix1>::from_vec(Ix1::new([buf]), data).unwrap();
            // How many elements fit with stride k?
            let out_len = if buf == 0 { 0 } else { (buf - 1) / k + 1 };
            let v = as_strided(&a, &[out_len], &[k]).unwrap();
            let collected: Vec<i32> = v.iter().copied().collect();
            let expected: Vec<i32> = (0..out_len).map(|i| (i * k) as i32).collect();
            prop_assert_eq!(collected, expected);
        }

        /// A sliding window with stride (1, 1) and shape (n, n) on a
        /// 1-D buffer always overlaps and must be rejected by
        /// `as_strided`.
        #[test]
        fn prop_as_strided_rejects_sliding_overlap(
            window in 2usize..6,
        ) {
            let buf_len = window + 2;
            let data: Vec<i32> = (0..buf_len as i32).collect();
            let a = Array::<i32, Ix1>::from_vec(Ix1::new([buf_len]), data).unwrap();
            let result = as_strided(&a, &[window, window], &[1, 1]);
            prop_assert!(result.is_err());
        }
    }
}
