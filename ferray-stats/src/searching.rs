// ferray-stats: Searching — unique, nonzero, where_, count_nonzero (REQ-14, REQ-15, REQ-16, REQ-17)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, Ix1, IxDyn};

use crate::reductions::{
    borrow_data, make_result, output_shape, reduce_axis_general_u64, validate_axis,
};

// ---------------------------------------------------------------------------
// unique
// ---------------------------------------------------------------------------

/// Result from the `unique` function.
#[derive(Debug)]
pub struct UniqueResult<T: Element> {
    /// The sorted unique values.
    pub values: Array<T, Ix1>,
    /// If requested, the indices of the first occurrence of each unique value
    /// in the original array (as u64).
    pub indices: Option<Array<u64, Ix1>>,
    /// If requested, the inverse index array such that
    /// `values[inverse]` reconstructs the flattened input. Essential for
    /// label encoding (#463).
    pub inverse: Option<Array<u64, Ix1>>,
    /// If requested, the count of each unique value (as u64).
    pub counts: Option<Array<u64, Ix1>>,
}

/// Find the sorted unique elements of an array.
///
/// The input is flattened. Optionally returns:
/// - `return_index`: indices of the first occurrence of each unique value.
/// - `return_inverse`: an array of the same length as the flattened input,
///   where each entry is the index into `values` of the corresponding
///   original element. Satisfies `values[inverse] == flat_input`.
/// - `return_counts`: count of each unique value.
///
/// Equivalent to `numpy.unique`.
pub fn unique<T, D>(
    a: &Array<T, D>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> FerrayResult<UniqueResult<T>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let data: Vec<T> = a.iter().copied().collect();
    let n_data = data.len();

    // Create (value, original_index) pairs, then sort by value.
    let mut pairs: Vec<(T, usize)> = data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate. The inverse is built in lockstep: for each original
    // position `orig_idx`, `inverse[orig_idx]` is the u64 index into
    // `unique_vals` where that original's value ended up. We walk the
    // sorted pairs, advancing the unique position whenever we see a new
    // value, and use the sorted pair's `original_idx` to write into the
    // right inverse slot.
    let mut unique_vals = Vec::new();
    let mut unique_indices: Vec<u64> = Vec::new();
    let mut unique_counts: Vec<u64> = Vec::new();
    let mut inverse_vec: Vec<u64> = if return_inverse {
        vec![0u64; n_data]
    } else {
        Vec::new()
    };

    if !pairs.is_empty() {
        unique_vals.push(pairs[0].0);
        unique_indices.push(pairs[0].1 as u64);
        if return_inverse {
            inverse_vec[pairs[0].1] = 0;
        }
        let mut count = 1u64;
        let mut unique_pos: u64 = 0;

        for i in 1..pairs.len() {
            if pairs[i].0.partial_cmp(&pairs[i - 1].0) == Some(std::cmp::Ordering::Equal) {
                count += 1;
                // Keep the first occurrence index (smallest original index).
                let last = unique_indices.len() - 1;
                let new_idx = pairs[i].1 as u64;
                if new_idx < unique_indices[last] {
                    unique_indices[last] = new_idx;
                }
            } else {
                if return_counts {
                    unique_counts.push(count);
                }
                unique_vals.push(pairs[i].0);
                unique_indices.push(pairs[i].1 as u64);
                count = 1;
                unique_pos += 1;
            }
            if return_inverse {
                inverse_vec[pairs[i].1] = unique_pos;
            }
        }
        if return_counts {
            unique_counts.push(count);
        }
    }

    let n = unique_vals.len();
    let values = Array::from_vec(Ix1::new([n]), unique_vals)?;
    let indices = if return_index {
        Some(Array::from_vec(Ix1::new([n]), unique_indices)?)
    } else {
        None
    };
    let inverse = if return_inverse {
        Some(Array::from_vec(Ix1::new([n_data]), inverse_vec)?)
    } else {
        None
    };
    let counts = if return_counts {
        Some(Array::from_vec(Ix1::new([n]), unique_counts)?)
    } else {
        None
    };

    Ok(UniqueResult {
        values,
        indices,
        inverse,
        counts,
    })
}

/// Find unique hyperslices along an axis.
///
/// Equivalent to `numpy.unique(a, axis=axis)`: returns an array of the
/// same dimensionality as `a` with the `axis` dimension reduced to the
/// number of unique slices. Slices are compared element-wise (lex order)
/// and returned in sorted order, matching numpy's behavior (#464).
///
/// For axis-less unique-on-flattened, see [`unique`].
///
/// # Errors
/// - `FerrayError::AxisOutOfBounds` if `axis >= a.ndim()`.
pub fn unique_axis<T, D>(
    a: &Array<T, D>,
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let n = shape[axis];
    let inner_stride: usize = shape[axis + 1..].iter().product();
    let outer_size: usize = shape[..axis].iter().product();
    let block = n * inner_stride;
    let slice_len = outer_size * inner_stride;

    let data: Vec<T> = a.iter().copied().collect();

    // Empty axis: return the input unchanged.
    if n == 0 {
        return Array::<T, IxDyn>::from_vec(IxDyn::new(&shape), data);
    }

    // Gather each axis-slice in canonical (outer, inner) order.
    let mut slices: Vec<Vec<T>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = Vec::with_capacity(slice_len);
        for o in 0..outer_size {
            let base = o * block + i * inner_stride;
            s.extend_from_slice(&data[base..base + inner_stride]);
        }
        slices.push(s);
    }

    // Sort axis indices by lex comparison of their slices. Use
    // partial_cmp so floating-point T (NaN-tolerating) works the same
    // way it does in `unique`.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        slices[i]
            .iter()
            .zip(slices[j].iter())
            .find_map(|(a, b)| match a.partial_cmp(b) {
                None | Some(std::cmp::Ordering::Equal) => None,
                Some(c) => Some(c),
            })
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Dedupe consecutive equal slices.
    let mut kept: Vec<usize> = Vec::with_capacity(n);
    for &idx in &order {
        if let Some(&prev) = kept.last() {
            let equal = slices[idx]
                .iter()
                .zip(slices[prev].iter())
                .all(|(a, b)| a.partial_cmp(b) == Some(std::cmp::Ordering::Equal));
            if equal {
                continue;
            }
        }
        kept.push(idx);
    }

    let new_n = kept.len();
    let mut out_shape = shape.clone();
    out_shape[axis] = new_n;
    let new_block = new_n * inner_stride;
    let total: usize = out_shape.iter().product();
    let mut out_data: Vec<T> = vec![data[0]; total];
    for (out_i, &src_i) in kept.iter().enumerate() {
        for o in 0..outer_size {
            let src_base = o * block + src_i * inner_stride;
            let dst_base = o * new_block + out_i * inner_stride;
            out_data[dst_base..dst_base + inner_stride]
                .copy_from_slice(&data[src_base..src_base + inner_stride]);
        }
    }
    Array::<T, IxDyn>::from_vec(IxDyn::new(&out_shape), out_data)
}

// ---------------------------------------------------------------------------
// nonzero
// ---------------------------------------------------------------------------

/// Return the indices of non-zero elements.
///
/// Returns a vector of 1-D arrays (u64), one per dimension. For a 1-D input,
/// returns a single array of indices.
///
/// Equivalent to `numpy.nonzero`.
pub fn nonzero<T, D>(a: &Array<T, D>) -> FerrayResult<Vec<Array<u64, Ix1>>>
where
    T: Element + PartialEq + Copy,
    D: Dimension,
{
    let shape = a.shape();
    let ndim = shape.len();
    let zero = <T as Element>::zero();

    // Collect all multi-indices where element != 0
    let mut indices_per_dim: Vec<Vec<u64>> = vec![Vec::new(); ndim];

    // Compute strides for index conversion
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (flat_idx, &val) in a.iter().enumerate() {
        if val != zero {
            let mut rem = flat_idx;
            for d in 0..ndim {
                indices_per_dim[d].push((rem / strides[d]) as u64);
                rem %= strides[d];
            }
        }
    }

    let mut result = Vec::with_capacity(ndim);
    for idx_vec in indices_per_dim {
        let n = idx_vec.len();
        result.push(Array::from_vec(Ix1::new([n]), idx_vec)?);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// where_
// ---------------------------------------------------------------------------

/// Conditional element selection.
///
/// For each element, if the corresponding element of `condition` is non-zero,
/// select from `x`; otherwise select from `y`.
///
/// All three arrays must have the same shape.
///
/// Equivalent to `numpy.where`.
pub fn where_<T, D>(
    condition: &Array<bool, D>,
    x: &Array<T, D>,
    y: &Array<T, D>,
) -> FerrayResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if condition.shape() != x.shape() || condition.shape() != y.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "condition, x, y shapes must match: {:?}, {:?}, {:?}",
            condition.shape(),
            x.shape(),
            y.shape()
        )));
    }

    let result: Vec<T> = condition
        .iter()
        .zip(x.iter())
        .zip(y.iter())
        .map(|((&c, &xv), &yv)| if c { xv } else { yv })
        .collect();

    Array::from_vec(condition.dim().clone(), result)
}

/// One-argument form of `where`: return the indices where `condition`
/// is true, as a vector of 1-D index arrays (one per dimension).
///
/// Equivalent to `numpy.where(condition)` (single-argument form) or
/// `numpy.nonzero(condition.astype(int))`. Added for `NumPy` parity
/// (#166) — the three-argument form above is [`where_`].
pub fn where_condition<D: Dimension>(
    condition: &Array<bool, D>,
) -> FerrayResult<Vec<Array<u64, Ix1>>> {
    let shape = condition.shape();
    let ndim = shape.len();
    let mut indices_per_dim: Vec<Vec<u64>> = vec![Vec::new(); ndim];

    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for (flat_idx, &val) in condition.iter().enumerate() {
        if val {
            let mut rem = flat_idx;
            for d in 0..ndim {
                indices_per_dim[d].push((rem / strides[d]) as u64);
                rem %= strides[d];
            }
        }
    }

    indices_per_dim
        .into_iter()
        .map(|v| {
            let n = v.len();
            Array::from_vec(Ix1::new([n]), v)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// count_nonzero
// ---------------------------------------------------------------------------

/// Count the number of non-zero elements along a given axis.
///
/// Equivalent to `numpy.count_nonzero`.
pub fn count_nonzero<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + PartialEq + Copy,
    D: Dimension,
{
    let zero = <T as Element>::zero();
    let data = borrow_data(a);
    match axis {
        None => {
            let count = data.iter().filter(|&&x| x != zero).count() as u64;
            make_result(&[], vec![count])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general_u64(&data, shape, ax, |lane| {
                lane.iter().filter(|&&x| x != zero).count() as u64
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// Array API standard names: unique_values / unique_counts / unique_inverse / unique_all
// ---------------------------------------------------------------------------

/// Sorted unique values of the (flattened) array.
///
/// Array-API-standard alias for [`unique`] with no extra return arrays.
/// Equivalent to `numpy.unique_values(x)`.
pub fn unique_values<T, D>(a: &Array<T, D>) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    Ok(unique(a, false, false, false)?.values)
}

/// Sorted unique values and their occurrence counts.
///
/// Equivalent to `numpy.unique_counts(x)` — returns `(values, counts)`.
pub fn unique_counts<T, D>(a: &Array<T, D>) -> FerrayResult<(Array<T, Ix1>, Array<u64, Ix1>)>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let r = unique(a, false, false, true)?;
    Ok((r.values, r.counts.expect("return_counts requested")))
}

/// Sorted unique values and the inverse-index array.
///
/// Equivalent to `numpy.unique_inverse(x)` — returns `(values, inverse)`
/// where `values[inverse]` reconstructs the flattened input.
pub fn unique_inverse<T, D>(a: &Array<T, D>) -> FerrayResult<(Array<T, Ix1>, Array<u64, Ix1>)>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let r = unique(a, false, true, false)?;
    Ok((r.values, r.inverse.expect("return_inverse requested")))
}

/// Sorted unique values along with first-occurrence indices, inverse, and
/// counts.
///
/// Equivalent to `numpy.unique_all(x)` — returns
/// `(values, indices, inverse, counts)`. The four-tuple is the documented
/// Array-API contract; suppress clippy's complexity warning.
#[allow(clippy::type_complexity)]
pub fn unique_all<T, D>(
    a: &Array<T, D>,
) -> FerrayResult<(
    Array<T, Ix1>,
    Array<u64, Ix1>,
    Array<u64, Ix1>,
    Array<u64, Ix1>,
)>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let r = unique(a, true, true, true)?;
    Ok((
        r.values,
        r.indices.expect("return_index requested"),
        r.inverse.expect("return_inverse requested"),
        r.counts.expect("return_counts requested"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::{Ix1, Ix2};

    // ---- unique_axis (#464) --------------------------------------------

    #[test]
    fn unique_axis_rows_dedup() {
        // axis=0 on a 2D array dedupes rows.
        let a = Array::<i32, Ix2>::from_vec(
            Ix2::new([4, 3]),
            vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 7, 8, 9],
        )
        .unwrap();
        let u = unique_axis(&a, 0).unwrap();
        // Three unique rows: [1,2,3], [4,5,6], [7,8,9] in sorted order.
        assert_eq!(u.shape(), &[3, 3]);
        let s = u.as_slice().unwrap();
        assert_eq!(s, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn unique_axis_columns_dedup() {
        // axis=1 dedupes columns. Construct a matrix where columns 0 and 2
        // are identical.
        let a = Array::<i32, Ix2>::from_vec(
            Ix2::new([3, 4]),
            vec![1, 2, 1, 3, 4, 5, 4, 6, 7, 8, 7, 9],
        )
        .unwrap();
        let u = unique_axis(&a, 1).unwrap();
        // 3 unique columns: [1,4,7], [2,5,8], [3,6,9] sorted lex.
        assert_eq!(u.shape(), &[3, 3]);
        let s = u.as_slice().unwrap();
        assert_eq!(s, &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn unique_axis_all_distinct_keeps_count_but_sorts() {
        let a = Array::<i32, Ix2>::from_vec(
            Ix2::new([3, 2]),
            vec![3, 4, 1, 2, 5, 6],
        )
        .unwrap();
        let u = unique_axis(&a, 0).unwrap();
        assert_eq!(u.shape(), &[3, 2]);
        let s = u.as_slice().unwrap();
        assert_eq!(s, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn unique_axis_all_same_collapses() {
        let a = Array::<i32, Ix2>::from_vec(
            Ix2::new([4, 2]),
            vec![1, 2, 1, 2, 1, 2, 1, 2],
        )
        .unwrap();
        let u = unique_axis(&a, 0).unwrap();
        assert_eq!(u.shape(), &[1, 2]);
        let s = u.as_slice().unwrap();
        assert_eq!(s, &[1, 2]);
    }

    #[test]
    fn unique_axis_3d_axis0() {
        // Shape (4, 2, 2) with rows 0 and 2 the same hyperslice.
        let a = Array::<i32, Ix2>::from_vec(
            Ix2::new([4, 4]),
            vec![
                1, 2, 3, 4, // row 0
                5, 6, 7, 8, // row 1
                1, 2, 3, 4, // row 2 (== row 0)
                9, 0, 1, 2, // row 3
            ],
        )
        .unwrap();
        let u = unique_axis(&a, 0).unwrap();
        assert_eq!(u.shape(), &[3, 4]);
    }

    #[test]
    fn unique_axis_out_of_bounds() {
        let a =
            Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
        assert!(unique_axis(&a, 5).is_err());
    }

    #[test]
    fn test_unique_basic() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let u = unique(&a, false, false, false).unwrap();
        let data: Vec<i32> = u.values.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_unique_with_counts() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let u = unique(&a, false, false, true).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let cnts: Vec<u64> = u.counts.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 2, 3]);
        assert_eq!(cnts, vec![2, 2, 2]);
    }

    #[test]
    fn test_unique_with_index() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![5, 3, 3, 1, 5]).unwrap();
        let u = unique(&a, true, false, false).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let idxs: Vec<u64> = u.indices.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 3, 5]);
        assert_eq!(idxs, vec![3, 1, 0]);
    }

    // ---- return_inverse (#463) ----

    #[test]
    fn test_unique_inverse_reconstructs_input() {
        // The canonical label-encoding use case.
        let input = vec![3, 1, 2, 1, 3, 2];
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), input.clone()).unwrap();
        let u = unique(&a, false, true, false).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let inv: Vec<u64> = u.inverse.unwrap().iter().copied().collect();
        // Unique values must be sorted.
        assert_eq!(vals, vec![1, 2, 3]);
        // values[inverse] must reconstruct the flattened input.
        let reconstructed: Vec<i32> = inv.iter().map(|&i| vals[i as usize]).collect();
        assert_eq!(reconstructed, input);
    }

    #[test]
    fn test_unique_inverse_all_together() {
        // Request indices + inverse + counts in one call; each field must
        // independently match what a single-flag call would produce.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([7]), vec![2, 1, 2, 3, 1, 2, 3]).unwrap();
        let u = unique(&a, true, true, true).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let idxs: Vec<u64> = u.indices.unwrap().iter().copied().collect();
        let inv: Vec<u64> = u.inverse.unwrap().iter().copied().collect();
        let cnts: Vec<u64> = u.counts.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 2, 3]);
        assert_eq!(idxs, vec![1, 0, 3]); // first positions of 1, 2, 3
        assert_eq!(cnts, vec![2, 3, 2]);
        // Reconstruct via inverse.
        let reconstructed: Vec<i32> = inv.iter().map(|&i| vals[i as usize]).collect();
        assert_eq!(reconstructed, vec![2, 1, 2, 3, 1, 2, 3]);
    }

    #[test]
    fn test_unique_inverse_with_2d_flattens_first() {
        // NumPy's unique flattens the input; inverse has length
        // shape.iter().product(), indexing into the flat logical traversal.
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 1, 3, 2, 1]).unwrap();
        let u = unique(&a, false, true, false).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let inv: Vec<u64> = u.inverse.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 2, 3]);
        assert_eq!(inv.len(), 6);
        let flat: Vec<i32> = vec![1, 2, 1, 3, 2, 1];
        let reconstructed: Vec<i32> = inv.iter().map(|&i| vals[i as usize]).collect();
        assert_eq!(reconstructed, flat);
    }

    #[test]
    fn test_unique_inverse_empty_input() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let u = unique(&a, false, true, false).unwrap();
        assert_eq!(u.values.shape(), &[0]);
        let inv = u.inverse.unwrap();
        assert_eq!(inv.shape(), &[0]);
    }

    #[test]
    fn test_unique_inverse_single_value() {
        // Every element identical → all inverse entries point at position 0.
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![7, 7, 7, 7]).unwrap();
        let u = unique(&a, false, true, false).unwrap();
        let inv: Vec<u64> = u.inverse.unwrap().iter().copied().collect();
        assert_eq!(inv, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_unique_without_inverse_leaves_field_none() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 1]).unwrap();
        let u = unique(&a, false, false, false).unwrap();
        assert!(u.inverse.is_none());
    }

    #[test]
    fn test_nonzero_1d() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 0, 3, 0]).unwrap();
        let nz = nonzero(&a).unwrap();
        assert_eq!(nz.len(), 1);
        let data: Vec<u64> = nz[0].iter().copied().collect();
        assert_eq!(data, vec![1, 3]);
    }

    #[test]
    fn test_nonzero_2d() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![0, 1, 0, 3, 0, 5]).unwrap();
        let nz = nonzero(&a).unwrap();
        assert_eq!(nz.len(), 2);
        let rows: Vec<u64> = nz[0].iter().copied().collect();
        let cols: Vec<u64> = nz[1].iter().copied().collect();
        assert_eq!(rows, vec![0, 1, 1]);
        assert_eq!(cols, vec![1, 0, 2]);
    }

    #[test]
    fn test_where_basic() {
        let cond =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let r = where_(&cond, &x, &y).unwrap();
        let data: Vec<f64> = r.iter().copied().collect();
        assert_eq!(data, vec![1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_shape_mismatch() {
        let cond = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap();
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        assert!(where_(&cond, &x, &y).is_err());
    }

    #[test]
    fn test_count_nonzero_total() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 0, 3, 0]).unwrap();
        let c = count_nonzero(&a, None).unwrap();
        assert_eq!(c.iter().next(), Some(&2u64));
    }

    #[test]
    fn test_count_nonzero_axis() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![0, 1, 0, 3, 0, 5]).unwrap();
        let c = count_nonzero(&a, Some(0)).unwrap();
        let data: Vec<u64> = c.iter().copied().collect();
        assert_eq!(data, vec![1, 1, 1]);
    }

    #[test]
    fn test_count_nonzero_axis1() {
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![0, 1, 0, 3, 0, 5]).unwrap();
        let c = count_nonzero(&a, Some(1)).unwrap();
        let data: Vec<u64> = c.iter().copied().collect();
        assert_eq!(data, vec![1, 2]);
    }

    // -- Array API unique_* --

    #[test]
    fn test_unique_values_alias() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let v = unique_values(&a).unwrap();
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn test_unique_counts_alias() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let (v, c) = unique_counts(&a).unwrap();
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(c.iter().copied().collect::<Vec<_>>(), vec![2, 2, 2]);
    }

    #[test]
    fn test_unique_inverse_alias() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let (v, inv) = unique_inverse(&a).unwrap();
        // values = [1, 2, 3]; inverse maps each original to that index
        // 3→2, 1→0, 2→1, 1→0, 3→2, 2→1
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            inv.iter().copied().collect::<Vec<_>>(),
            vec![2, 0, 1, 0, 2, 1]
        );
    }

    #[test]
    fn test_unique_all_alias() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let (v, _idx, inv, c) = unique_all(&a).unwrap();
        assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            inv.iter().copied().collect::<Vec<_>>(),
            vec![2, 0, 1, 0, 2, 1]
        );
        assert_eq!(c.iter().copied().collect::<Vec<_>>(), vec![2, 2, 2]);
    }
}
