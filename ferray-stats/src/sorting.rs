// ferray-stats: Sorting and searching — sort, argsort, searchsorted (REQ-11, REQ-12, REQ-13)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, Ix1, IxDyn};

use crate::parallel;
use crate::reductions::{compute_strides, flat_index, increment_multi_index};

// ---------------------------------------------------------------------------
// SortKind
// ---------------------------------------------------------------------------

/// Sorting algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortKind {
    /// Unstable quicksort (faster but does not preserve order of equal elements).
    Quick,
    /// Stable merge sort (preserves relative order of equal elements).
    Stable,
}

// ---------------------------------------------------------------------------
// Side (for searchsorted)
// ---------------------------------------------------------------------------

/// Side parameter for `searchsorted`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    /// Find the leftmost insertion point (first position where the value could be inserted).
    Left,
    /// Find the rightmost insertion point (last position where the value could be inserted).
    Right,
}

// ---------------------------------------------------------------------------
// sort
// ---------------------------------------------------------------------------

/// Sort an array along the given axis (or flattened if axis is None).
///
/// When `axis` is `None`, the array is flattened before sorting and a 1-D
/// array is returned. When an axis is given, the returned array has the
/// same shape as the input.
///
/// **Note:** NumPy's `np.sort(a)` defaults to `axis=-1` (last axis).
/// ferray's `sort(a, None, kind)` flattens instead. To match NumPy's
/// default, pass the last axis explicitly:
/// `sort(a, Some(a.ndim() - 1), kind)`.
///
/// Equivalent to `numpy.sort`.
pub fn sort<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    kind: SortKind,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy + Send + Sync,
    D: Dimension,
{
    match axis {
        None => {
            // Flatten and sort — return a 1-D array (NumPy behaviour)
            let mut data: Vec<T> = a.iter().copied().collect();
            let n = data.len();
            sort_slice(&mut data, kind);
            Array::from_vec(IxDyn::new(&[n]), data)
        }
        Some(ax) => {
            if ax >= a.ndim() {
                return Err(FerrayError::axis_out_of_bounds(ax, a.ndim()));
            }
            let shape = a.shape().to_vec();
            let ndim = shape.len();
            // Materialize once into a single buffer that we sort in
            // place — the previous code allocated `data` plus a full
            // `result = data.clone()` second copy (#171).
            let mut buf: Vec<T> = a.iter().copied().collect();
            let axis_len = shape[ax];

            // Last-axis fast path: lanes are already contiguous in
            // row-major order, so we can hand each `axis_len` window to
            // `sort_slice` directly with no gather/scatter.
            if ax == ndim - 1 {
                for chunk in buf.chunks_exact_mut(axis_len) {
                    sort_slice(chunk, kind);
                }
                return Array::from_vec(IxDyn::new(&shape), buf);
            }

            // General axis: gather a temporary lane, sort it, scatter
            // values back into the same buffer.
            let strides = compute_strides(&shape);
            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, &s)| s)
                .collect();
            let out_size: usize = if out_shape.is_empty() {
                1
            } else {
                out_shape.iter().product()
            };

            let mut out_multi = vec![0usize; out_shape.len()];
            // Re-used per-lane scratch buffers to avoid `axis_len`
            // re-allocations on every output position.
            let mut in_multi = vec![0usize; ndim];
            let mut lane: Vec<T> = Vec::with_capacity(axis_len);
            let mut lane_indices: Vec<usize> = Vec::with_capacity(axis_len);

            for _ in 0..out_size {
                // Build input multi-index template
                let mut out_dim = 0;
                for (d, slot) in in_multi.iter_mut().enumerate() {
                    if d == ax {
                        *slot = 0;
                    } else {
                        *slot = out_multi[out_dim];
                        out_dim += 1;
                    }
                }

                lane.clear();
                lane_indices.clear();
                for k in 0..axis_len {
                    in_multi[ax] = k;
                    let idx = flat_index(&in_multi, &strides);
                    lane.push(buf[idx]);
                    lane_indices.push(idx);
                }

                sort_slice(&mut lane, kind);

                // Scatter sorted values back into the in-place buffer.
                for (k, &idx) in lane_indices.iter().enumerate() {
                    buf[idx] = lane[k];
                }

                if !out_shape.is_empty() {
                    increment_multi_index(&mut out_multi, &out_shape);
                }
            }

            Array::from_vec(IxDyn::new(&shape), buf)
        }
    }
}

/// Sort a slice in place using the given algorithm.
fn sort_slice<T: PartialOrd + Copy + Send + Sync>(data: &mut [T], kind: SortKind) {
    match kind {
        SortKind::Quick => {
            parallel::parallel_sort(data);
        }
        SortKind::Stable => {
            parallel::parallel_sort_stable(data);
        }
    }
}

// ---------------------------------------------------------------------------
// argsort
// ---------------------------------------------------------------------------

/// Return the indices that would sort an array along the given axis.
///
/// When `axis` is `None`, the array is flattened before computing
/// indices and a 1-D array is returned.
///
/// Returns u64 indices.
///
/// **Note:** NumPy's `np.argsort(a)` defaults to `axis=-1` (last axis).
/// ferray's `argsort(a, None)` flattens instead. To match NumPy's
/// default, pass the last axis explicitly: `argsort(a, Some(a.ndim() - 1))`.
///
/// Equivalent to `numpy.argsort`.
pub fn argsort<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    match axis {
        None => {
            let data: Vec<T> = a.iter().copied().collect();
            let n = data.len();
            let mut indices: Vec<usize> = (0..n).collect();
            indices.sort_by(|&i, &j| {
                data[i]
                    .partial_cmp(&data[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let result: Vec<u64> = indices.into_iter().map(|i| i as u64).collect();
            Array::from_vec(IxDyn::new(&[n]), result)
        }
        Some(ax) => {
            if ax >= a.ndim() {
                return Err(FerrayError::axis_out_of_bounds(ax, a.ndim()));
            }
            let shape = a.shape().to_vec();
            let data: Vec<T> = a.iter().copied().collect();
            let strides = compute_strides(&shape);
            let ndim = shape.len();
            let axis_len = shape[ax];

            let out_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, &s)| s)
                .collect();
            let out_size: usize = if out_shape.is_empty() {
                1
            } else {
                out_shape.iter().product()
            };

            let mut result = vec![0u64; data.len()];
            let mut out_multi = vec![0usize; out_shape.len()];

            for _ in 0..out_size {
                let mut in_multi = Vec::with_capacity(ndim);
                let mut out_dim = 0;
                for d in 0..ndim {
                    if d == ax {
                        in_multi.push(0);
                    } else {
                        in_multi.push(out_multi[out_dim]);
                        out_dim += 1;
                    }
                }

                // Gather lane values and their axis-local indices
                let mut lane: Vec<(usize, T)> = Vec::with_capacity(axis_len);
                let mut lane_flat_indices: Vec<usize> = Vec::with_capacity(axis_len);
                for k in 0..axis_len {
                    in_multi[ax] = k;
                    let idx = flat_index(&in_multi, &strides);
                    lane.push((k, data[idx]));
                    lane_flat_indices.push(idx);
                }

                // Sort by value, tracking original axis-local index
                lane.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                // Scatter the original axis-local indices into the result
                for (k, &flat_idx) in lane_flat_indices.iter().enumerate() {
                    result[flat_idx] = lane[k].0 as u64;
                }

                if !out_shape.is_empty() {
                    increment_multi_index(&mut out_multi, &out_shape);
                }
            }

            Array::from_vec(IxDyn::new(&shape), result)
        }
    }
}

// ---------------------------------------------------------------------------
// partition / argpartition
// ---------------------------------------------------------------------------

/// Partial sort: rearrange elements so that `a[kth]` is the value that
/// would be there in a sorted array, all elements before it are `<=`,
/// and all elements after are `>=`. The relative order within the two
/// halves is undefined.
///
/// Equivalent to `numpy.partition(a, kth)`. Uses `select_nth_unstable`
/// for O(n) average-case performance (#466).
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `kth >= a.size()`.
pub fn partition<T>(a: &Array<T, Ix1>, kth: usize) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let n = a.size();
    if kth >= n {
        return Err(FerrayError::invalid_value(format!(
            "partition: kth={kth} out of range for array of size {n}"
        )));
    }
    let mut data: Vec<T> = a.iter().copied().collect();
    data.select_nth_unstable_by(kth, |x, y| {
        x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
    });
    Array::from_vec(Ix1::new([n]), data)
}

/// Return indices that would partition the array. The k-th element of
/// the result is the index of the k-th smallest element; elements
/// before index kth are indices of smaller-or-equal elements, and
/// elements after are indices of greater-or-equal.
///
/// Equivalent to `numpy.argpartition(a, kth)` (#466).
pub fn argpartition<T>(a: &Array<T, Ix1>, kth: usize) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let n = a.size();
    if kth >= n {
        return Err(FerrayError::invalid_value(format!(
            "argpartition: kth={kth} out of range for array of size {n}"
        )));
    }
    let data: Vec<T> = a.iter().copied().collect();
    let mut idx: Vec<u64> = (0..n as u64).collect();
    idx.select_nth_unstable_by(kth, |&a_i, &b_i| {
        let va = data[a_i as usize];
        let vb = data[b_i as usize];
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    Array::from_vec(Ix1::new([n]), idx)
}

// ---------------------------------------------------------------------------
// lexsort
// ---------------------------------------------------------------------------

/// Indirect stable sort using a sequence of keys.
///
/// `keys` is a list of 1-D arrays of the same length. The **last** key
/// in the list is the primary sort key (matching NumPy's
/// `numpy.lexsort` convention); ties are broken by the second-to-last
/// key, then the third-to-last, and so on. Returns a permutation
/// `idx` such that `keys[-1][idx]` is non-decreasing.
///
/// Implementation notes: the underlying sort is `sort_by` (stable),
/// applied once with a comparator that walks the keys from primary
/// (last) to secondary (earlier). This avoids the multi-pass stable
/// sort that NumPy historically used.
///
/// # Errors
/// - `FerrayError::InvalidValue` if `keys` is empty or the keys have
///   different lengths.
pub fn lexsort<T>(keys: &[&Array<T, Ix1>]) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    if keys.is_empty() {
        return Err(FerrayError::invalid_value(
            "lexsort: keys must contain at least one array",
        ));
    }
    let n = keys[0].size();
    for (i, k) in keys.iter().enumerate().skip(1) {
        if k.size() != n {
            return Err(FerrayError::invalid_value(format!(
                "lexsort: key {i} has length {}, expected {n}",
                k.size()
            )));
        }
    }

    // Materialize each key into a contiguous Vec so the comparator
    // can index into them directly without re-borrowing the array
    // iterator on every comparison.
    let key_data: Vec<Vec<T>> = keys.iter().map(|k| k.iter().copied().collect()).collect();

    let mut idx: Vec<u64> = (0..n as u64).collect();
    idx.sort_by(|&a, &b| {
        let ai = a as usize;
        let bi = b as usize;
        // Iterate keys from primary (last) to secondary (earlier).
        for k in key_data.iter().rev() {
            match k[ai].partial_cmp(&k[bi]).unwrap_or(std::cmp::Ordering::Equal) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        std::cmp::Ordering::Equal
    });

    Array::from_vec(Ix1::new([n]), idx)
}

// ---------------------------------------------------------------------------
// searchsorted
// ---------------------------------------------------------------------------

/// Find indices where elements should be inserted to maintain order.
///
/// `a` must be a sorted 1-D array. For each value in `v`, find the index
/// in `a` where it should be inserted. Returns u64 indices.
///
/// Equivalent to `numpy.searchsorted` (without `sorter`). For an
/// already-permuted view of an unsorted array, see [`searchsorted_with_sorter`].
pub fn searchsorted<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    side: Side,
) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let sorted: Vec<T> = a.iter().copied().collect();
    searchsorted_inner(&sorted, v, side)
}

/// Find indices where elements should be inserted to maintain order,
/// using `sorter` as a permutation that would sort `a`.
///
/// Mirrors `numpy.searchsorted(a, v, side, sorter)`. `a` may be in any
/// order; `sorter[i]` gives the index in `a` of the i-th smallest
/// element (i.e. `sorter` is the output of an `argsort` over `a`). The
/// returned indices are positions into the **sorted** view, matching
/// NumPy's behaviour. See issue #473.
///
/// # Errors
/// - `FerrayError::ShapeMismatch` if `sorter.len() != a.len()`.
/// - `FerrayError::InvalidValue` if `sorter` contains an out-of-range index.
pub fn searchsorted_with_sorter<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    side: Side,
    sorter: &Array<u64, Ix1>,
) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let n = a.size();
    if sorter.size() != n {
        return Err(FerrayError::shape_mismatch(format!(
            "searchsorted: sorter length {} does not match array length {}",
            sorter.size(),
            n
        )));
    }

    // Materialize `a` once and gather it in sorter order.
    let a_data: Vec<T> = a.iter().copied().collect();
    let mut sorted: Vec<T> = Vec::with_capacity(n);
    for &idx in sorter.iter() {
        let i = idx as usize;
        if i >= n {
            return Err(FerrayError::invalid_value(format!(
                "searchsorted: sorter index {i} out of range for array of length {n}"
            )));
        }
        sorted.push(a_data[i]);
    }

    searchsorted_inner(&sorted, v, side)
}

/// Shared binary-search core used by both [`searchsorted`] and
/// [`searchsorted_with_sorter`].
fn searchsorted_inner<T>(
    sorted: &[T],
    v: &Array<T, Ix1>,
    side: Side,
) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let mut result = Vec::with_capacity(v.size());
    for &val in v.iter() {
        let idx = match side {
            Side::Left => sorted.partition_point(|x| {
                x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Less) == std::cmp::Ordering::Less
            }),
            Side::Right => sorted.partition_point(|x| {
                x.partial_cmp(&val).unwrap_or(std::cmp::Ordering::Less)
                    != std::cmp::Ordering::Greater
            }),
        };
        result.push(idx as u64);
    }
    let n = result.len();
    Array::from_vec(Ix1::new([n]), result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Ix2;

    #[test]
    fn test_sort_1d() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap();
        let s = sort(&a, None, SortKind::Quick).unwrap();
        assert_eq!(s.shape(), &[5]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sort_stable_preserves_order() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![3, 1, 4, 1, 5]).unwrap();
        let s = sort(&a, None, SortKind::Stable).unwrap();
        assert_eq!(s.shape(), &[5]);
        let data: Vec<i32> = s.iter().copied().collect();
        assert_eq!(data, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn test_sort_2d_axis_none_returns_flat() {
        // Issue #91: sort(axis=None) should return a flat 1-D array
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![6.0, 4.0, 5.0, 3.0, 1.0, 2.0])
            .unwrap();
        let s = sort(&a, None, SortKind::Quick).unwrap();
        // Must be 1-D with 6 elements, not [2, 3]
        assert_eq!(s.shape(), &[6]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sort_2d_axis1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0])
            .unwrap();
        let s = sort(&a, Some(1), SortKind::Quick).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_sort_2d_axis0() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
            .unwrap();
        let s = sort(&a, Some(0), SortKind::Quick).unwrap();
        assert_eq!(s.shape(), &[2, 3]);
        let data: Vec<f64> = s.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_argsort_1d() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0, 1.0, 4.0, 2.0]).unwrap();
        let idx = argsort(&a, None).unwrap();
        assert_eq!(idx.shape(), &[4]);
        let data: Vec<u64> = idx.iter().copied().collect();
        assert_eq!(data, vec![1, 3, 0, 2]);
    }

    #[test]
    fn test_argsort_2d_axis1() {
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0])
            .unwrap();
        let idx = argsort(&a, Some(1)).unwrap();
        assert_eq!(idx.shape(), &[2, 3]);
        let data: Vec<u64> = idx.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 0, 1, 2, 0]);
    }

    #[test]
    fn test_searchsorted_left() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.5, 1.0, 5.5]).unwrap();
        let idx = searchsorted(&a, &v, Side::Left).unwrap();
        let data: Vec<u64> = idx.iter().copied().collect();
        assert_eq!(data, vec![2, 0, 5]);
    }

    #[test]
    fn test_searchsorted_right() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![2.0, 4.0]).unwrap();
        let idx = searchsorted(&a, &v, Side::Right).unwrap();
        let data: Vec<u64> = idx.iter().copied().collect();
        assert_eq!(data, vec![2, 4]);
    }

    // ----- searchsorted_with_sorter (#473) -----

    #[test]
    fn test_searchsorted_with_sorter_matches_pre_sorted() {
        // Unsorted `a` plus its argsort gives the same indices as
        // calling searchsorted on the pre-sorted array.
        let unsorted =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![3.0, 1.0, 5.0, 2.0, 4.0]).unwrap();
        // sorter so that unsorted[sorter] = [1.0, 2.0, 3.0, 4.0, 5.0]
        let sorter = Array::<u64, Ix1>::from_vec(Ix1::new([5]), vec![1, 3, 0, 4, 2]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.5, 1.0, 5.5]).unwrap();

        let idx = searchsorted_with_sorter(&unsorted, &v, Side::Left, &sorter).unwrap();
        assert_eq!(idx.iter().copied().collect::<Vec<_>>(), vec![2, 0, 5]);
    }

    #[test]
    fn test_searchsorted_with_sorter_length_mismatch_errors() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0, 1.0, 5.0, 2.0]).unwrap();
        let bad_sorter = Array::<u64, Ix1>::from_vec(Ix1::new([3]), vec![1, 3, 0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![2.5]).unwrap();
        assert!(searchsorted_with_sorter(&a, &v, Side::Left, &bad_sorter).is_err());
    }

    #[test]
    fn test_searchsorted_with_sorter_out_of_range_errors() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![3.0, 1.0, 5.0]).unwrap();
        let bad_sorter = Array::<u64, Ix1>::from_vec(Ix1::new([3]), vec![1, 99, 0]).unwrap();
        let v = Array::<f64, Ix1>::from_vec(Ix1::new([1]), vec![2.5]).unwrap();
        assert!(searchsorted_with_sorter(&a, &v, Side::Left, &bad_sorter).is_err());
    }

    // ----- lexsort (#469) -----

    #[test]
    fn test_lexsort_single_key_matches_argsort() {
        let k = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![3, 1, 4, 1, 5]).unwrap();
        let idx = lexsort(&[&k]).unwrap();
        // Sorted order: 1@idx1, 1@idx3, 3@idx0, 4@idx2, 5@idx4
        assert_eq!(idx.iter().copied().collect::<Vec<_>>(), vec![1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_lexsort_secondary_key_breaks_ties() {
        // Primary key (last in slice) sorts by ascending; ties resolved
        // by the earlier key. Match NumPy's lexsort([secondary, primary]).
        let secondary = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![20, 10, 40, 30]).unwrap();
        let primary = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 1, 2]).unwrap();
        let idx = lexsort(&[&secondary, &primary]).unwrap();
        // primary buckets:
        //   1 -> indices 0, 2 with secondary 20, 40 -> ordered as 0, 2
        //   2 -> indices 1, 3 with secondary 10, 30 -> ordered as 1, 3
        // result: [0, 2, 1, 3]
        assert_eq!(idx.iter().copied().collect::<Vec<_>>(), vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_lexsort_length_mismatch_errors() {
        let k1 = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let k2 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 2, 3, 4]).unwrap();
        assert!(lexsort(&[&k1, &k2]).is_err());
    }

    #[test]
    fn test_lexsort_empty_keys_errors() {
        let keys: &[&Array<i32, Ix1>] = &[];
        assert!(lexsort(keys).is_err());
    }

    #[test]
    fn test_sort_axis_out_of_bounds() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(sort(&a, Some(1), SortKind::Quick).is_err());
    }
}
