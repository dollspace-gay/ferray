// ferray-stride-tricks: Runtime overlap detection for safe as_strided
//
// Determines whether a given (shape, strides) pair produces overlapping
// memory accesses. Two distinct logical indices overlap when they map to
// the same physical offset from the base pointer.

/// Check whether a view described by `shape` and `strides` (in element
/// units) would access any element of the underlying buffer more than once.
///
/// Returns `true` if the strides produce overlapping accesses (i.e., two
/// distinct multi-dimensional indices map to the same memory offset).
///
/// Algorithm: try a cheap analytical check first (non-overlap is
/// decidable in O(ndim²) for views whose absolute strides are
/// "properly ordered" per the axis extents). Fall back to an exact
/// offset enumeration via HashSet for cases the analytical pass
/// can't prove. The previous version always enumerated and had a
/// dead "if ≤ 1M then enumerate else enumerate" branch (#286, #288).
///
/// # Arguments
/// * `shape` — the shape of the desired view.
/// * `strides` — strides in units of elements (may be negative).
pub(crate) fn has_overlapping_strides(shape: &[usize], strides: &[isize]) -> bool {
    debug_assert_eq!(shape.len(), strides.len());

    let ndim = shape.len();

    // Zero-dimensional or empty: no overlap possible.
    if ndim == 0 || shape.contains(&0) {
        return false;
    }

    // Total number of distinct logical indices.
    let n_elements: usize = shape.iter().product();

    // A single element can't overlap with itself.
    if n_elements <= 1 {
        return false;
    }

    // If any stride is 0, that dimension is broadcast (the same element is
    // accessed for every index along that axis). This immediately means
    // overlap unless that dimension has size <= 1.
    for (i, &s) in strides.iter().enumerate() {
        if s == 0 && shape[i] > 1 {
            return true;
        }
    }

    // Analytical fast path: when we can sort the non-trivial axes by
    // ascending `|stride|` and each subsequent stride is strictly
    // larger than the maximum offset reachable by the lower axes
    // (`(extent_so_far)`), overlap is impossible. This decides
    // contiguous, transposed, and strided-take patterns without
    // touching any per-element iteration (#288).
    if is_non_overlapping_by_stride_order(shape, strides) {
        return false;
    }

    // Fall back to exact offset enumeration for the stride patterns
    // the analytical pass can't dispatch (e.g., strides with a common
    // factor that produces aliasing). The enumeration is still O(n)
    // time + memory, but in practice only exotic views hit this path.
    has_overlapping_enumerate(shape, strides, n_elements)
}

/// Try to prove non-overlap analytically by sorting the axes by
/// ascending absolute stride and checking the "each axis strictly
/// larger than the span of all lower axes" invariant that C/F
/// contiguous-like layouts satisfy.
///
/// Returns `true` only when non-overlap is provably true; a `false`
/// return is inconclusive and the caller must enumerate. This is the
/// same pattern NumPy uses in `PyArray_NonZeroDimsNonOverlapping`
/// and gives O(ndim²) performance for the common cases.
fn is_non_overlapping_by_stride_order(shape: &[usize], strides: &[isize]) -> bool {
    // Drop size-1 axes; their stride doesn't matter.
    let mut axes: Vec<(usize, isize)> = shape
        .iter()
        .copied()
        .zip(strides.iter().copied())
        .filter(|&(n, _)| n > 1)
        .collect();
    // Sort by absolute stride ascending.
    axes.sort_by_key(|&(_, s)| s.unsigned_abs());

    // Walk the sorted axes. Each axis' absolute stride must be >= the
    // total span spanned by all previously-seen axes. The span of an
    // axis of length `n` with stride `|s|` is `(n - 1) * |s| + 1`.
    // We track the cumulative span starting from 1 (a single point).
    let mut cum_span: u128 = 1;
    for (n, s) in axes {
        let abs_s = s.unsigned_abs() as u128;
        if abs_s < cum_span {
            // Not provably non-overlapping — the enumeration path
            // can still decide correctly; return false to delegate.
            return false;
        }
        cum_span = (n as u128 - 1) * abs_s + 1;
    }
    true
}

/// Enumerate all offsets and check for duplicates.
fn has_overlapping_enumerate(shape: &[usize], strides: &[isize], n_elements: usize) -> bool {
    use std::collections::HashSet;

    let ndim = shape.len();
    let mut seen = HashSet::with_capacity(n_elements);

    // We iterate in row-major order using a counter vector.
    let mut index = vec![0usize; ndim];

    for _ in 0..n_elements {
        // Compute the physical offset for the current index.
        let offset: isize = index
            .iter()
            .zip(strides.iter())
            .map(|(&idx, &stride)| idx as isize * stride)
            .sum();

        if !seen.insert(offset) {
            return true;
        }

        // Increment the multi-index (row-major order: last axis increments first).
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                index[d] += 1;
                if index[d] >= shape[d] {
                    index[d] = 0;
                    // carry remains true
                } else {
                    carry = false;
                }
            }
        }
    }

    false
}

/// Validate that all offsets generated by `(shape, strides)` lie within
/// `[0, buf_len)` when measured from the base pointer.
///
/// Returns `true` if all offsets are in bounds, `false` otherwise.
pub(crate) fn all_offsets_in_bounds(shape: &[usize], strides: &[isize], buf_len: usize) -> bool {
    let ndim = shape.len();

    if ndim == 0 || shape.contains(&0) {
        return true;
    }

    // Compute the minimum and maximum offsets reachable.
    let mut min_offset: isize = 0;
    let mut max_offset: isize = 0;

    for i in 0..ndim {
        if shape[i] == 0 {
            continue;
        }
        let extent = (shape[i] as isize - 1) * strides[i];
        if extent > 0 {
            max_offset += extent;
        } else {
            min_offset += extent;
        }
    }

    // The offsets must all be non-negative and less than buf_len.
    min_offset >= 0 && max_offset >= 0 && (max_offset as usize) < buf_len
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_overlap_contiguous_1d() {
        // [0, 1, 2, 3, 4] — shape (5,), stride (1,)
        assert!(!has_overlapping_strides(&[5], &[1]));
    }

    #[test]
    fn no_overlap_contiguous_2d() {
        // 3x4 row-major: stride (4, 1)
        assert!(!has_overlapping_strides(&[3, 4], &[4, 1]));
    }

    #[test]
    fn overlap_broadcast_stride_zero() {
        // stride 0 on a dimension > 1 means overlap
        assert!(has_overlapping_strides(&[3, 4], &[0, 1]));
    }

    #[test]
    fn overlap_sliding_window() {
        // sliding window of size 3 over 5 elements: shape (3, 3), strides (1, 1)
        assert!(has_overlapping_strides(&[3, 3], &[1, 1]));
    }

    #[test]
    fn no_overlap_strided_non_contiguous() {
        // Take every other element: shape (3,), stride (2,) from buffer of 6
        assert!(!has_overlapping_strides(&[3], &[2]));
    }

    #[test]
    fn no_overlap_empty() {
        assert!(!has_overlapping_strides(&[0], &[1]));
        assert!(!has_overlapping_strides(&[3, 0], &[1, 1]));
    }

    #[test]
    fn no_overlap_scalar() {
        assert!(!has_overlapping_strides(&[1], &[1]));
        assert!(!has_overlapping_strides(&[1, 1], &[1, 1]));
    }

    #[test]
    fn overlap_negative_strides() {
        // shape (3,), stride (-1,): offsets are 0, -1, -2 — all distinct,
        // but they go negative. However for overlap detection we only care
        // about distinct vs. duplicate offsets.
        assert!(!has_overlapping_strides(&[3], &[-1]));
    }

    // ----- analytical non-overlap proof (#288) -----

    #[test]
    fn analytical_fast_path_contiguous_3d() {
        // 2 x 3 x 4, row-major: strides (12, 4, 1). The analytical
        // sort-by-|stride| check should accept this without entering
        // the HashSet enumeration.
        assert!(is_non_overlapping_by_stride_order(&[2, 3, 4], &[12, 4, 1]));
    }

    #[test]
    fn analytical_fast_path_transposed() {
        // A 4 x 3 transpose of a 3 x 4 row-major layout has strides
        // (1, 4). That's "outer axis has stride 1" which still sorts
        // correctly by |stride| and proves non-overlap.
        assert!(is_non_overlapping_by_stride_order(&[4, 3], &[1, 4]));
    }

    #[test]
    fn analytical_fast_path_rejects_sliding_window() {
        // shape (3, 3) strides (1, 1) is a sliding window → overlapping.
        // The analytical pass should not claim non-overlap here.
        assert!(!is_non_overlapping_by_stride_order(&[3, 3], &[1, 1]));
    }

    #[test]
    fn analytical_fast_path_single_axis() {
        assert!(is_non_overlapping_by_stride_order(&[5], &[1]));
        assert!(is_non_overlapping_by_stride_order(&[5], &[-1]));
    }

    #[test]
    fn analytical_fast_path_drops_size_one_axes() {
        // A trailing size-1 axis with arbitrary stride should not
        // confuse the analytical check.
        assert!(is_non_overlapping_by_stride_order(&[3, 1, 2], &[2, 99, 1]));
    }

    #[test]
    fn bounds_check_contiguous() {
        assert!(all_offsets_in_bounds(&[5], &[1], 5));
        assert!(!all_offsets_in_bounds(&[5], &[1], 4));
    }

    #[test]
    fn bounds_check_2d() {
        // 3x4, strides (4,1), buf_len=12
        assert!(all_offsets_in_bounds(&[3, 4], &[4, 1], 12));
        assert!(!all_offsets_in_bounds(&[3, 4], &[4, 1], 11));
    }

    #[test]
    fn bounds_check_empty() {
        assert!(all_offsets_in_bounds(&[0], &[1], 0));
    }

    #[test]
    fn bounds_check_negative_stride_out_of_bounds() {
        // Negative strides would go below 0
        assert!(!all_offsets_in_bounds(&[3], &[-1], 3));
    }
}
