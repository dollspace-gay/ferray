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
/// array is returned (matching NumPy behaviour). When an axis is given, the
/// returned array has the same shape as the input.
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
            let data: Vec<T> = a.iter().copied().collect();
            let mut result = data.clone();
            let strides = compute_strides(&shape);

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

            let mut out_multi = vec![0usize; out_shape.len()];
            let ndim = shape.len();

            for _ in 0..out_size {
                // Build input multi-index template
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

                // Gather lane
                let mut lane: Vec<T> = Vec::with_capacity(axis_len);
                let mut lane_indices: Vec<usize> = Vec::with_capacity(axis_len);
                for k in 0..axis_len {
                    in_multi[ax] = k;
                    let idx = flat_index(&in_multi, &strides);
                    lane.push(data[idx]);
                    lane_indices.push(idx);
                }

                sort_slice(&mut lane, kind);

                // Scatter sorted values back
                for (k, &idx) in lane_indices.iter().enumerate() {
                    result[idx] = lane[k];
                }

                if !out_shape.is_empty() {
                    increment_multi_index(&mut out_multi, &out_shape);
                }
            }

            Array::from_vec(IxDyn::new(&shape), result)
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
/// indices and a 1-D array is returned (matching NumPy behaviour).
///
/// Returns u64 indices.
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
// searchsorted
// ---------------------------------------------------------------------------

/// Find indices where elements should be inserted to maintain order.
///
/// `a` must be a sorted 1-D array. For each value in `v`, find the index
/// in `a` where it should be inserted. Returns u64 indices.
///
/// Equivalent to `numpy.searchsorted`.
pub fn searchsorted<T>(
    a: &Array<T, Ix1>,
    v: &Array<T, Ix1>,
    side: Side,
) -> FerrayResult<Array<u64, Ix1>>
where
    T: Element + PartialOrd + Copy,
{
    let sorted: Vec<T> = a.iter().copied().collect();
    let values: Vec<T> = v.iter().copied().collect();

    let mut result = Vec::with_capacity(values.len());
    for &val in &values {
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

    #[test]
    fn test_sort_axis_out_of_bounds() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(sort(&a, Some(1), SortKind::Quick).is_err());
    }
}
