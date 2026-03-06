// ferrum-stats: Searching — unique, nonzero, where_, count_nonzero (REQ-14, REQ-15, REQ-16, REQ-17)

use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::{Array, Dimension, Element, Ix1, IxDyn};

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
    /// If requested, the count of each unique value (as u64).
    pub counts: Option<Array<u64, Ix1>>,
}

/// Find the sorted unique elements of an array.
///
/// The input is flattened. Optionally returns indices and/or counts.
///
/// Equivalent to `numpy.unique`.
pub fn unique<T, D>(
    a: &Array<T, D>,
    return_index: bool,
    return_counts: bool,
) -> FerrumResult<UniqueResult<T>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    let data: Vec<T> = a.iter().copied().collect();

    // Create (value, original_index) pairs, then sort by value
    let mut pairs: Vec<(T, usize)> = data
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| (v, i))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Deduplicate
    let mut unique_vals = Vec::new();
    let mut unique_indices: Vec<u64> = Vec::new();
    let mut unique_counts: Vec<u64> = Vec::new();

    if !pairs.is_empty() {
        unique_vals.push(pairs[0].0);
        unique_indices.push(pairs[0].1 as u64);
        let mut count = 1u64;

        for i in 1..pairs.len() {
            if pairs[i].0.partial_cmp(&pairs[i - 1].0) != Some(std::cmp::Ordering::Equal) {
                if return_counts {
                    unique_counts.push(count);
                }
                unique_vals.push(pairs[i].0);
                unique_indices.push(pairs[i].1 as u64);
                count = 1;
            } else {
                count += 1;
                // Keep the first occurrence index (smallest original index)
                let last = unique_indices.len() - 1;
                let new_idx = pairs[i].1 as u64;
                if new_idx < unique_indices[last] {
                    unique_indices[last] = new_idx;
                }
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
    let counts = if return_counts {
        Some(Array::from_vec(Ix1::new([n]), unique_counts)?)
    } else {
        None
    };

    Ok(UniqueResult {
        values,
        indices,
        counts,
    })
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
pub fn nonzero<T, D>(a: &Array<T, D>) -> FerrumResult<Vec<Array<u64, Ix1>>>
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
) -> FerrumResult<Array<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    if condition.shape() != x.shape() || condition.shape() != y.shape() {
        return Err(FerrumError::shape_mismatch(format!(
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

// ---------------------------------------------------------------------------
// count_nonzero
// ---------------------------------------------------------------------------

/// Count the number of non-zero elements along a given axis.
///
/// Equivalent to `numpy.count_nonzero`.
pub fn count_nonzero<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<u64, IxDyn>>
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{Ix1, Ix2};

    #[test]
    fn test_unique_basic() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let u = unique(&a, false, false).unwrap();
        let data: Vec<i32> = u.values.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3]);
    }

    #[test]
    fn test_unique_with_counts() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([6]), vec![3, 1, 2, 1, 3, 2]).unwrap();
        let u = unique(&a, false, true).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let cnts: Vec<u64> = u.counts.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 2, 3]);
        assert_eq!(cnts, vec![2, 2, 2]);
    }

    #[test]
    fn test_unique_with_index() {
        let a = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![5, 3, 3, 1, 5]).unwrap();
        let u = unique(&a, true, false).unwrap();
        let vals: Vec<i32> = u.values.iter().copied().collect();
        let idxs: Vec<u64> = u.indices.unwrap().iter().copied().collect();
        assert_eq!(vals, vec![1, 3, 5]);
        assert_eq!(idxs, vec![3, 1, 0]);
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
}
