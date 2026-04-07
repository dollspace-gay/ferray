// ferray-core: Extended indexing functions (REQ-15a)
//
// take, take_along_axis, put, put_along_axis, choose, compress, select,
// indices, ix_, diag_indices, diag_indices_from, tril_indices, triu_indices,
// tril_indices_from, triu_indices_from, ravel_multi_index, unravel_index,
// flatnonzero, fill_diagonal, ndindex (iterator), ndenumerate (iterator)
//
// Index-returning functions return Vec<Vec<usize>> or (Vec<usize>, Vec<usize>)
// because usize is not an Element type. Callers can wrap these into arrays
// of u64 or i64 if needed.

use crate::array::owned::Array;
use crate::dimension::{Axis, Dimension, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

/// Normalize a potentially negative index, returning an error on out-of-bounds.
fn normalize_index(index: isize, size: usize, axis: usize) -> FerrayResult<usize> {
    if index < 0 {
        let pos = size as isize + index;
        if pos < 0 {
            return Err(FerrayError::index_out_of_bounds(index, axis, size));
        }
        Ok(pos as usize)
    } else {
        let idx = index as usize;
        if idx >= size {
            return Err(FerrayError::index_out_of_bounds(index, axis, size));
        }
        Ok(idx)
    }
}

// ===========================================================================
// take / take_along_axis
// ===========================================================================

/// Take elements from an array along an axis.
///
/// Equivalent to `np.take(a, indices, axis)`. Returns a copy.
///
/// # Errors
/// - `AxisOutOfBounds` if `axis >= ndim`
/// - `IndexOutOfBounds` if any index is out of range
pub fn take<T: Element, D: Dimension>(
    a: &Array<T, D>,
    indices: &[isize],
    axis: Axis,
) -> FerrayResult<Array<T, IxDyn>> {
    a.index_select(axis, indices)
}

/// Take values from an array along an axis using an index slice.
///
/// Similar to `np.take_along_axis`. The `indices` slice contains
/// indices into `a` along the specified axis. The result replaces
/// the `axis` dimension with the indices dimension.
///
/// # Errors
/// - `AxisOutOfBounds` if `axis >= ndim`
/// - `IndexOutOfBounds` if any index is out of range
pub fn take_along_axis<T: Element, D: Dimension>(
    a: &Array<T, D>,
    indices: &[isize],
    axis: Axis,
) -> FerrayResult<Array<T, IxDyn>> {
    a.index_select(axis, indices)
}

// ===========================================================================
// put / put_along_axis
// ===========================================================================

impl<T: Element, D: Dimension> Array<T, D> {
    /// Put values into the flattened array at the given indices.
    ///
    /// Equivalent to `np.put(a, ind, v)`. Modifies the array in-place.
    /// Indices refer to the flattened (row-major) array.
    /// Values are cycled if fewer than indices.
    ///
    /// # Errors
    /// - `IndexOutOfBounds` if any index is out of range
    /// - `InvalidValue` if values is empty
    pub fn put(&mut self, indices: &[isize], values: &[T]) -> FerrayResult<()> {
        if values.is_empty() {
            return Err(FerrayError::invalid_value("values must not be empty"));
        }
        let size = self.size();
        let normalized: Vec<usize> = indices
            .iter()
            .map(|&idx| normalize_index(idx, size, 0))
            .collect::<FerrayResult<Vec<_>>>()?;

        let mut flat: Vec<&mut T> = self.inner.iter_mut().collect();

        for (i, &idx) in normalized.iter().enumerate() {
            let val_idx = i % values.len();
            *flat[idx] = values[val_idx].clone();
        }
        Ok(())
    }

    /// Put values along an axis at specified indices.
    ///
    /// For each index position along `axis`, assigns the values from the
    /// corresponding sub-array of `values`.
    ///
    /// # Errors
    /// - `AxisOutOfBounds` if `axis >= ndim`
    /// - `IndexOutOfBounds` if any index is out of range
    pub fn put_along_axis(
        &mut self,
        indices: &[isize],
        values: &Array<T, IxDyn>,
        axis: Axis,
    ) -> FerrayResult<()>
    where
        D::NdarrayDim: ndarray::RemoveAxis,
    {
        let ndim = self.ndim();
        let ax = axis.index();
        if ax >= ndim {
            return Err(FerrayError::axis_out_of_bounds(ax, ndim));
        }
        let axis_size = self.shape()[ax];

        let normalized: Vec<usize> = indices
            .iter()
            .map(|&idx| normalize_index(idx, axis_size, ax))
            .collect::<FerrayResult<Vec<_>>>()?;

        let nd_axis = ndarray::Axis(ax);
        let mut val_iter = values.inner.iter();

        for &idx in &normalized {
            let mut sub = self.inner.index_axis_mut(nd_axis, idx);
            for elem in sub.iter_mut() {
                if let Some(v) = val_iter.next() {
                    *elem = v.clone();
                }
            }
        }
        Ok(())
    }

    /// Fill the main diagonal of a 2-D (or N-D) array with a value.
    ///
    /// For N-D arrays, the diagonal consists of indices where all
    /// index values are equal: `a[i, i, ..., i]`.
    ///
    /// Equivalent to `np.fill_diagonal(a, val)`.
    pub fn fill_diagonal(&mut self, val: T) {
        let shape = self.shape().to_vec();
        if shape.is_empty() {
            return;
        }
        let min_dim = *shape.iter().min().unwrap_or(&0);
        let ndim = shape.len();

        for i in 0..min_dim {
            let idx: Vec<usize> = vec![i; ndim];
            let nd_idx = ndarray::IxDyn(&idx);
            let mut dyn_view = self.inner.view_mut().into_dyn();
            dyn_view[nd_idx] = val.clone();
        }
    }
}

// ===========================================================================
// choose
// ===========================================================================

/// Construct an array from an index array and a list of arrays to choose from.
///
/// Equivalent to `np.choose(a, choices)`. For each element in `index_arr`,
/// the value selects which choice array to pick from at that position.
/// Index values are given as `u64` to avoid the `usize` Element issue.
///
/// # Errors
/// - `IndexOutOfBounds` if any index in `index_arr` is >= `choices.len()`
/// - `ShapeMismatch` if choice arrays have different shapes from `index_arr`
pub fn choose<T: Element, D: Dimension>(
    index_arr: &Array<u64, D>,
    choices: &[Array<T, D>],
) -> FerrayResult<Array<T, IxDyn>> {
    if choices.is_empty() {
        return Err(FerrayError::invalid_value("choices must not be empty"));
    }

    let shape = index_arr.shape();
    for (i, c) in choices.iter().enumerate() {
        if c.shape() != shape {
            return Err(FerrayError::shape_mismatch(format!(
                "choice[{}] shape {:?} does not match index array shape {:?}",
                i,
                c.shape(),
                shape
            )));
        }
    }

    let n_choices = choices.len();
    let choice_iters: Vec<Vec<T>> = choices
        .iter()
        .map(|c| c.inner.iter().cloned().collect())
        .collect();

    let mut data = Vec::with_capacity(index_arr.size());
    for (pos, idx_val) in index_arr.inner.iter().enumerate() {
        let idx = *idx_val as usize;
        if idx >= n_choices {
            return Err(FerrayError::index_out_of_bounds(idx as isize, 0, n_choices));
        }
        data.push(choice_iters[idx][pos].clone());
    }

    let dyn_shape = IxDyn::new(shape);
    Array::from_vec(dyn_shape, data)
}

// ===========================================================================
// compress
// ===========================================================================

/// Select slices of an array along an axis where `condition` is true.
///
/// Equivalent to `np.compress(condition, a, axis)`.
///
/// # Errors
/// - `AxisOutOfBounds` if `axis >= ndim`
/// - `ShapeMismatch` if `condition.len()` exceeds axis size
pub fn compress<T: Element, D: Dimension>(
    condition: &[bool],
    a: &Array<T, D>,
    axis: Axis,
) -> FerrayResult<Array<T, IxDyn>> {
    let ndim = a.ndim();
    let ax = axis.index();
    if ax >= ndim {
        return Err(FerrayError::axis_out_of_bounds(ax, ndim));
    }
    let axis_size = a.shape()[ax];
    if condition.len() > axis_size {
        return Err(FerrayError::shape_mismatch(format!(
            "condition length {} exceeds axis size {}",
            condition.len(),
            axis_size
        )));
    }

    let indices: Vec<isize> = condition
        .iter()
        .enumerate()
        .filter_map(|(i, &c)| if c { Some(i as isize) } else { None })
        .collect();

    a.index_select(axis, &indices)
}

// ===========================================================================
// select
// ===========================================================================

/// Return an array drawn from elements in choicelist, depending on conditions.
///
/// Equivalent to `np.select(condlist, choicelist, default)`.
/// The first condition that is true determines which choice is used.
///
/// # Errors
/// - `InvalidValue` if condlist and choicelist have different lengths
/// - `ShapeMismatch` if shapes are incompatible
pub fn select<T: Element, D: Dimension>(
    condlist: &[Array<bool, D>],
    choicelist: &[Array<T, D>],
    default: T,
) -> FerrayResult<Array<T, IxDyn>> {
    if condlist.len() != choicelist.len() {
        return Err(FerrayError::invalid_value(format!(
            "condlist length {} != choicelist length {}",
            condlist.len(),
            choicelist.len()
        )));
    }
    if condlist.is_empty() {
        return Err(FerrayError::invalid_value(
            "condlist and choicelist must not be empty",
        ));
    }

    let shape = condlist[0].shape();
    for (i, (c, ch)) in condlist.iter().zip(choicelist.iter()).enumerate() {
        if c.shape() != shape || ch.shape() != shape {
            return Err(FerrayError::shape_mismatch(format!(
                "condlist[{}]/choicelist[{}] shape mismatch with reference shape {:?}",
                i, i, shape
            )));
        }
    }

    let size = condlist[0].size();
    let mut data = vec![default; size];

    // Process in reverse order so first matching condition wins
    for (cond, choice) in condlist.iter().zip(choicelist.iter()).rev() {
        for (i, (&c, v)) in cond.inner.iter().zip(choice.inner.iter()).enumerate() {
            if c {
                data[i] = v.clone();
            }
        }
    }

    let dyn_shape = IxDyn::new(shape);
    Array::from_vec(dyn_shape, data)
}

// ===========================================================================
// indices
// ===========================================================================

/// Return arrays representing the indices of a grid.
///
/// Equivalent to `np.indices(dimensions)`. Returns one `u64` array per
/// dimension, each with shape `dimensions`.
///
/// For example, `indices(&[2, 3])` returns two arrays of shape `[2, 3]`:
/// the first contains row indices, the second column indices.
pub fn indices(dimensions: &[usize]) -> FerrayResult<Vec<Array<u64, IxDyn>>> {
    let ndim = dimensions.len();
    let total: usize = dimensions.iter().product();

    let mut result = Vec::with_capacity(ndim);

    for ax in 0..ndim {
        let mut data = Vec::with_capacity(total);
        for flat_idx in 0..total {
            let mut rem = flat_idx;
            let mut idx_for_ax = 0;
            for (d, &dim_size) in dimensions.iter().enumerate().rev() {
                let coord = rem % dim_size;
                rem /= dim_size;
                if d == ax {
                    idx_for_ax = coord;
                }
            }
            data.push(idx_for_ax as u64);
        }
        let dim = IxDyn::new(dimensions);
        result.push(Array::from_vec(dim, data)?);
    }

    Ok(result)
}

// ===========================================================================
// ix_
// ===========================================================================

/// Construct an open mesh from multiple sequences.
///
/// Equivalent to `np.ix_(*args)`. Returns a list of arrays, each with
/// shape `(1, 1, ..., N, ..., 1)` where `N` is the length of that sequence
/// and it appears in the position corresponding to its argument index.
///
/// This is useful for constructing index arrays for cross-indexing.
pub fn ix_(sequences: &[&[u64]]) -> FerrayResult<Vec<Array<u64, IxDyn>>> {
    let ndim = sequences.len();
    let mut result = Vec::with_capacity(ndim);

    for (i, seq) in sequences.iter().enumerate() {
        let mut shape = vec![1usize; ndim];
        shape[i] = seq.len();

        let data = seq.to_vec();
        let dim = IxDyn::new(&shape);
        result.push(Array::from_vec(dim, data)?);
    }

    Ok(result)
}

// ===========================================================================
// diag_indices / diag_indices_from
// ===========================================================================

/// Return the indices to access the main diagonal of an n x n array.
///
/// Equivalent to `np.diag_indices(n, ndim=2)`. Returns `ndim` vectors,
/// each containing `[0, 1, ..., n-1]`.
pub fn diag_indices(n: usize, ndim: usize) -> Vec<Vec<usize>> {
    let data: Vec<usize> = (0..n).collect();
    vec![data; ndim]
}

/// Return the indices to access the main diagonal of the given array.
///
/// The array must be at least 2-D and square (all dimensions equal).
///
/// # Errors
/// - `InvalidValue` if the array has fewer than 2 dimensions
/// - `ShapeMismatch` if dimensions are not all equal
pub fn diag_indices_from<T: Element, D: Dimension>(
    a: &Array<T, D>,
) -> FerrayResult<Vec<Vec<usize>>> {
    let ndim = a.ndim();
    if ndim < 2 {
        return Err(FerrayError::invalid_value(
            "diag_indices_from requires at least 2 dimensions",
        ));
    }
    let shape = a.shape();
    let n = shape[0];
    for &s in &shape[1..] {
        if s != n {
            return Err(FerrayError::shape_mismatch(format!(
                "all dimensions must be equal for diag_indices_from, got {:?}",
                shape
            )));
        }
    }
    Ok(diag_indices(n, ndim))
}

// ===========================================================================
// tril_indices / triu_indices / tril_indices_from / triu_indices_from
// ===========================================================================

/// Return the indices for the lower triangle of an (n, m) array.
///
/// Equivalent to `np.tril_indices(n, k, m)`.
/// `k` is the diagonal offset: 0 = main diagonal, positive = above,
/// negative = below.
pub fn tril_indices(n: usize, k: isize, m: Option<usize>) -> (Vec<usize>, Vec<usize>) {
    let m = m.unwrap_or(n);
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    for i in 0..n {
        for j in 0..m {
            if (j as isize) <= (i as isize) + k {
                rows.push(i);
                cols.push(j);
            }
        }
    }

    (rows, cols)
}

/// Return the indices for the upper triangle of an (n, m) array.
///
/// Equivalent to `np.triu_indices(n, k, m)`.
pub fn triu_indices(n: usize, k: isize, m: Option<usize>) -> (Vec<usize>, Vec<usize>) {
    let m = m.unwrap_or(n);
    let mut rows = Vec::new();
    let mut cols = Vec::new();

    for i in 0..n {
        for j in 0..m {
            if (j as isize) >= (i as isize) + k {
                rows.push(i);
                cols.push(j);
            }
        }
    }

    (rows, cols)
}

/// Return the indices for the lower triangle of the given 2-D array.
///
/// # Errors
/// - `InvalidValue` if the array is not 2-D
pub fn tril_indices_from<T: Element, D: Dimension>(
    a: &Array<T, D>,
    k: isize,
) -> FerrayResult<(Vec<usize>, Vec<usize>)> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::invalid_value(
            "tril_indices_from requires a 2-D array",
        ));
    }
    Ok(tril_indices(shape[0], k, Some(shape[1])))
}

/// Return the indices for the upper triangle of the given 2-D array.
///
/// # Errors
/// - `InvalidValue` if the array is not 2-D
pub fn triu_indices_from<T: Element, D: Dimension>(
    a: &Array<T, D>,
    k: isize,
) -> FerrayResult<(Vec<usize>, Vec<usize>)> {
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrayError::invalid_value(
            "triu_indices_from requires a 2-D array",
        ));
    }
    Ok(triu_indices(shape[0], k, Some(shape[1])))
}

// ===========================================================================
// ravel_multi_index / unravel_index
// ===========================================================================

/// Convert a tuple of index arrays to a flat index array.
///
/// Equivalent to `np.ravel_multi_index(multi_index, dims)`.
/// Uses row-major (C) ordering.
///
/// # Errors
/// - `InvalidValue` if multi_index arrays have different lengths
/// - `IndexOutOfBounds` if any index is out of range for its dimension
#[allow(clippy::needless_range_loop)]
pub fn ravel_multi_index(multi_index: &[&[usize]], dims: &[usize]) -> FerrayResult<Vec<usize>> {
    if multi_index.len() != dims.len() {
        return Err(FerrayError::invalid_value(format!(
            "multi_index has {} components but dims has {} dimensions",
            multi_index.len(),
            dims.len()
        )));
    }
    if multi_index.is_empty() {
        return Ok(vec![]);
    }

    let n = multi_index[0].len();
    for (i, idx_arr) in multi_index.iter().enumerate() {
        if idx_arr.len() != n {
            return Err(FerrayError::invalid_value(format!(
                "multi_index[{}] has length {} but expected {}",
                i,
                idx_arr.len(),
                n
            )));
        }
    }

    // Compute strides for C-order
    let ndim = dims.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    let mut flat = Vec::with_capacity(n);
    #[allow(clippy::needless_range_loop)]
    for pos in 0..n {
        let mut linear = 0usize;
        for (d, &dim_size) in dims.iter().enumerate() {
            let coord = multi_index[d][pos];
            if coord >= dim_size {
                return Err(FerrayError::index_out_of_bounds(
                    coord as isize,
                    d,
                    dim_size,
                ));
            }
            linear += coord * strides[d];
        }
        flat.push(linear);
    }

    Ok(flat)
}

/// Convert flat indices to a tuple of coordinate arrays.
///
/// Equivalent to `np.unravel_index(indices, shape)`.
/// Uses row-major (C) ordering.
///
/// # Errors
/// - `IndexOutOfBounds` if any flat index >= product(shape)
pub fn unravel_index(flat_indices: &[usize], shape: &[usize]) -> FerrayResult<Vec<Vec<usize>>> {
    let total: usize = shape.iter().product();
    let ndim = shape.len();
    let n = flat_indices.len();

    let mut result: Vec<Vec<usize>> = vec![Vec::with_capacity(n); ndim];

    for &flat_idx in flat_indices {
        if flat_idx >= total {
            return Err(FerrayError::index_out_of_bounds(
                flat_idx as isize,
                0,
                total,
            ));
        }
        let mut rem = flat_idx;
        for (d, &dim_size) in shape.iter().enumerate().rev() {
            result[d].push(rem % dim_size);
            rem /= dim_size;
        }
    }

    Ok(result)
}

// ===========================================================================
// flatnonzero
// ===========================================================================

/// Return the indices of non-zero elements in the flattened array.
///
/// Equivalent to `np.flatnonzero(a)`. An element is "non-zero" if it
/// is not equal to the type's zero value.
pub fn flatnonzero<T: Element + PartialEq, D: Dimension>(a: &Array<T, D>) -> Vec<usize> {
    let zero = T::zero();
    a.inner
        .iter()
        .enumerate()
        .filter_map(|(i, val)| if *val != zero { Some(i) } else { None })
        .collect()
}

// ===========================================================================
// ndindex / ndenumerate iterators
// ===========================================================================

/// An iterator over all multi-dimensional indices for a given shape.
///
/// Equivalent to `np.ndindex(*shape)`. Yields indices in row-major order.
pub struct NdIndex {
    shape: Vec<usize>,
    current: Vec<usize>,
    done: bool,
}

impl NdIndex {
    fn new(shape: &[usize]) -> Self {
        let done = shape.contains(&0);
        Self {
            shape: shape.to_vec(),
            current: vec![0; shape.len()],
            done,
        }
    }
}

impl Iterator for NdIndex {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let result = self.current.clone();

        // Increment: rightmost dimension first (row-major / C-order)
        let mut carry = true;
        for i in (0..self.shape.len()).rev() {
            if carry {
                self.current[i] += 1;
                if self.current[i] >= self.shape[i] {
                    self.current[i] = 0;
                    carry = true;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            self.done = true;
        }

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let total: usize = self.shape.iter().product();
        // Compute how many we've already yielded
        let mut yielded = 0usize;
        let ndim = self.shape.len();
        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            yielded += self.current[i] * stride;
            stride *= self.shape[i];
        }
        let remaining = total - yielded;
        (remaining, Some(remaining))
    }
}

/// Create an iterator over all multi-dimensional indices for a shape.
///
/// Equivalent to `np.ndindex(*shape)`.
pub fn ndindex(shape: &[usize]) -> NdIndex {
    NdIndex::new(shape)
}

/// Create an iterator yielding `(index, &value)` pairs.
///
/// Equivalent to `np.ndenumerate(a)`.
pub fn ndenumerate<'a, T: Element, D: Dimension>(
    a: &'a Array<T, D>,
) -> impl Iterator<Item = (Vec<usize>, &'a T)> + 'a {
    let shape = a.shape().to_vec();
    let ndim = shape.len();
    a.inner.iter().enumerate().map(move |(flat_idx, val)| {
        let mut idx = vec![0usize; ndim];
        let mut rem = flat_idx;
        for (d, s) in shape.iter().enumerate().rev() {
            if *s > 0 {
                idx[d] = rem % s;
                rem /= s;
            }
        }
        (idx, val)
    })
}

// ---------------------------------------------------------------------------
// where_select — ternary selection (np.where equivalent)
// ---------------------------------------------------------------------------

/// Select elements from `x` or `y` depending on `condition`.
///
/// For each element position, returns `x[i]` where `condition[i]` is `true`,
/// and `y[i]` where `condition[i]` is `false`.
///
/// All three arrays must have the same shape.
///
/// Equivalent to `numpy.where(condition, x, y)`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn where_select<T: Element + Copy, D: Dimension>(
    condition: &Array<bool, D>,
    x: &Array<T, D>,
    y: &Array<T, D>,
) -> FerrayResult<Array<T, D>> {
    if condition.shape() != x.shape() || condition.shape() != y.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "where_select: condition shape {:?}, x shape {:?}, y shape {:?} must all match",
            condition.shape(),
            x.shape(),
            y.shape()
        )));
    }
    let data: Vec<T> = condition
        .iter()
        .zip(x.iter().zip(y.iter()))
        .map(|(&c, (&xi, &yi))| if c { xi } else { yi })
        .collect();
    Array::from_vec(x.dim().clone(), data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dimension::{Ix1, Ix2};

    // -----------------------------------------------------------------------
    // take
    // -----------------------------------------------------------------------

    #[test]
    fn take_1d() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let taken = take(&arr, &[0, 2, 4], Axis(0)).unwrap();
        assert_eq!(taken.shape(), &[3]);
        let data: Vec<i32> = taken.iter().copied().collect();
        assert_eq!(data, vec![10, 30, 50]);
    }

    #[test]
    fn take_2d_axis1() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let taken = take(&arr, &[0, 2], Axis(1)).unwrap();
        assert_eq!(taken.shape(), &[3, 2]);
        let data: Vec<i32> = taken.iter().copied().collect();
        assert_eq!(data, vec![0, 2, 4, 6, 8, 10]);
    }

    #[test]
    fn take_negative_indices() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let taken = take(&arr, &[-1, -3], Axis(0)).unwrap();
        let data: Vec<i32> = taken.iter().copied().collect();
        assert_eq!(data, vec![40, 20]);
    }

    // -----------------------------------------------------------------------
    // take_along_axis
    // -----------------------------------------------------------------------

    #[test]
    fn take_along_axis_basic() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let taken = take_along_axis(&arr, &[1, 3], Axis(1)).unwrap();
        assert_eq!(taken.shape(), &[3, 2]);
    }

    // -----------------------------------------------------------------------
    // put
    // -----------------------------------------------------------------------

    #[test]
    fn put_flat() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 0, 0, 0, 0]).unwrap();
        arr.put(&[1, 3], &[99, 88]).unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[0, 99, 0, 88, 0]);
    }

    #[test]
    fn put_cycling_values() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0; 5]).unwrap();
        arr.put(&[0, 1, 2, 3, 4], &[10, 20]).unwrap();
        assert_eq!(arr.as_slice().unwrap(), &[10, 20, 10, 20, 10]);
    }

    #[test]
    fn put_out_of_bounds() {
        let mut arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![0, 0, 0]).unwrap();
        assert!(arr.put(&[5], &[1]).is_err());
    }

    // -----------------------------------------------------------------------
    // fill_diagonal
    // -----------------------------------------------------------------------

    #[test]
    fn fill_diagonal_2d() {
        let mut arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 3]), vec![0; 9]).unwrap();
        arr.fill_diagonal(1);
        let data: Vec<i32> = arr.iter().copied().collect();
        assert_eq!(data, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
    }

    #[test]
    fn fill_diagonal_rectangular() {
        let mut arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 4]), vec![0; 8]).unwrap();
        arr.fill_diagonal(5);
        let data: Vec<i32> = arr.iter().copied().collect();
        assert_eq!(data, vec![5, 0, 0, 0, 0, 5, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // choose
    // -----------------------------------------------------------------------

    #[test]
    fn choose_basic() {
        let idx = Array::<u64, Ix1>::from_vec(Ix1::new([4]), vec![0, 1, 0, 1]).unwrap();
        let c0 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![10, 20, 30, 40]).unwrap();
        let c1 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![100, 200, 300, 400]).unwrap();
        let result = choose(&idx, &[c0, c1]).unwrap();
        let data: Vec<i32> = result.iter().copied().collect();
        assert_eq!(data, vec![10, 200, 30, 400]);
    }

    #[test]
    fn choose_out_of_bounds() {
        let idx = Array::<u64, Ix1>::from_vec(Ix1::new([2]), vec![0, 2]).unwrap();
        let c0 = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![1, 2]).unwrap();
        let c1 = Array::<i32, Ix1>::from_vec(Ix1::new([2]), vec![3, 4]).unwrap();
        assert!(choose(&idx, &[c0, c1]).is_err());
    }

    // -----------------------------------------------------------------------
    // compress
    // -----------------------------------------------------------------------

    #[test]
    fn compress_1d() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![10, 20, 30, 40, 50]).unwrap();
        let result = compress(&[true, false, true, false, true], &arr, Axis(0)).unwrap();
        let data: Vec<i32> = result.iter().copied().collect();
        assert_eq!(data, vec![10, 30, 50]);
    }

    #[test]
    fn compress_2d_axis0() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), (0..12).collect()).unwrap();
        let result = compress(&[true, false, true], &arr, Axis(0)).unwrap();
        assert_eq!(result.shape(), &[2, 4]);
        let data: Vec<i32> = result.iter().copied().collect();
        assert_eq!(data, vec![0, 1, 2, 3, 8, 9, 10, 11]);
    }

    // -----------------------------------------------------------------------
    // select
    // -----------------------------------------------------------------------

    #[test]
    fn select_basic() {
        let c1 =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, false, false]).unwrap();
        let c2 =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![false, true, false, false]).unwrap();
        let ch1 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![1, 1, 1, 1]).unwrap();
        let ch2 = Array::<i32, Ix1>::from_vec(Ix1::new([4]), vec![2, 2, 2, 2]).unwrap();
        let result = select(&[c1, c2], &[ch1, ch2], 0).unwrap();
        let data: Vec<i32> = result.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 0, 0]);
    }

    // -----------------------------------------------------------------------
    // indices
    // -----------------------------------------------------------------------

    #[test]
    fn indices_2d() {
        let idx = indices(&[2, 3]).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx[0].shape(), &[2, 3]);
        assert_eq!(idx[1].shape(), &[2, 3]);
        let rows: Vec<u64> = idx[0].iter().copied().collect();
        assert_eq!(rows, vec![0, 0, 0, 1, 1, 1]);
        let cols: Vec<u64> = idx[1].iter().copied().collect();
        assert_eq!(cols, vec![0, 1, 2, 0, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // ix_
    // -----------------------------------------------------------------------

    #[test]
    fn ix_basic() {
        let result = ix_(&[&[0, 1], &[2, 3, 4]]).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shape(), &[2, 1]);
        assert_eq!(result[1].shape(), &[1, 3]);
    }

    // -----------------------------------------------------------------------
    // diag_indices
    // -----------------------------------------------------------------------

    #[test]
    fn diag_indices_basic() {
        let idx = diag_indices(3, 2);
        assert_eq!(idx.len(), 2);
        assert_eq!(idx[0], vec![0, 1, 2]);
        assert_eq!(idx[1], vec![0, 1, 2]);
    }

    #[test]
    fn diag_indices_from_square() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([4, 4])).unwrap();
        let idx = diag_indices_from(&arr).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx[0].len(), 4);
    }

    #[test]
    fn diag_indices_from_not_square() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 4])).unwrap();
        assert!(diag_indices_from(&arr).is_err());
    }

    // -----------------------------------------------------------------------
    // tril_indices / triu_indices
    // -----------------------------------------------------------------------

    #[test]
    fn tril_indices_basic() {
        let (rows, cols) = tril_indices(3, 0, None);
        assert_eq!(rows, vec![0, 1, 1, 2, 2, 2]);
        assert_eq!(cols, vec![0, 0, 1, 0, 1, 2]);
    }

    #[test]
    fn triu_indices_basic() {
        let (rows, cols) = triu_indices(3, 0, None);
        assert_eq!(rows, vec![0, 0, 0, 1, 1, 2]);
        assert_eq!(cols, vec![0, 1, 2, 1, 2, 2]);
    }

    #[test]
    fn tril_indices_with_k() {
        let (rows, cols) = tril_indices(3, 1, None);
        assert_eq!(rows, vec![0, 0, 1, 1, 1, 2, 2, 2]);
        assert_eq!(cols, vec![0, 1, 0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn triu_indices_with_negative_k() {
        let (rows, cols) = triu_indices(3, -1, None);
        assert_eq!(rows, vec![0, 0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(cols, vec![0, 1, 2, 0, 1, 2, 1, 2]);
    }

    #[test]
    fn tril_indices_from_test() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 3])).unwrap();
        let (rows, _cols) = tril_indices_from(&arr, 0).unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn triu_indices_from_test() {
        let arr = Array::<f64, Ix2>::zeros(Ix2::new([3, 3])).unwrap();
        let (rows, _cols) = triu_indices_from(&arr, 0).unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn tril_indices_rectangular() {
        let (rows, cols) = tril_indices(3, 0, Some(4));
        assert_eq!(rows, vec![0, 1, 1, 2, 2, 2]);
        assert_eq!(cols, vec![0, 0, 1, 0, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // ravel_multi_index / unravel_index
    // -----------------------------------------------------------------------

    #[test]
    fn ravel_multi_index_basic() {
        let flat = ravel_multi_index(&[&[0, 1, 2], &[1, 2, 0]], &[3, 4]).unwrap();
        assert_eq!(flat, vec![1, 6, 8]);
    }

    #[test]
    fn ravel_multi_index_3d() {
        let flat = ravel_multi_index(&[&[0], &[1], &[2]], &[2, 3, 4]).unwrap();
        assert_eq!(flat, vec![6]);
    }

    #[test]
    fn ravel_multi_index_out_of_bounds() {
        assert!(ravel_multi_index(&[&[3]], &[3]).is_err());
    }

    #[test]
    fn unravel_index_basic() {
        let coords = unravel_index(&[1, 6, 8], &[3, 4]).unwrap();
        assert_eq!(coords[0], vec![0, 1, 2]);
        assert_eq!(coords[1], vec![1, 2, 0]);
    }

    #[test]
    fn unravel_index_out_of_bounds() {
        assert!(unravel_index(&[12], &[3, 4]).is_err());
    }

    #[test]
    fn ravel_unravel_roundtrip() {
        let dims = &[3, 4, 5];
        let a: &[usize] = &[1, 2];
        let b: &[usize] = &[2, 3];
        let c: &[usize] = &[3, 4];
        let multi: &[&[usize]] = &[a, b, c];
        let flat = ravel_multi_index(multi, dims).unwrap();
        let coords = unravel_index(&flat, dims).unwrap();
        assert_eq!(coords[0], vec![1, 2]);
        assert_eq!(coords[1], vec![2, 3]);
        assert_eq!(coords[2], vec![3, 4]);
    }

    // -----------------------------------------------------------------------
    // flatnonzero
    // -----------------------------------------------------------------------

    #[test]
    fn flatnonzero_basic() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![0, 1, 0, 3, 0]).unwrap();
        let nz = flatnonzero(&arr);
        assert_eq!(nz, vec![1, 3]);
    }

    #[test]
    fn flatnonzero_2d() {
        let arr = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![0, 1, 0, 2, 0, 3]).unwrap();
        let nz = flatnonzero(&arr);
        assert_eq!(nz, vec![1, 3, 5]);
    }

    #[test]
    fn flatnonzero_all_zero() {
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![0, 0, 0]).unwrap();
        let nz = flatnonzero(&arr);
        assert_eq!(nz.len(), 0);
    }

    // -----------------------------------------------------------------------
    // ndindex
    // -----------------------------------------------------------------------

    #[test]
    fn ndindex_2d() {
        let indices: Vec<Vec<usize>> = ndindex(&[2, 3]).collect();
        assert_eq!(indices.len(), 6);
        assert_eq!(indices[0], vec![0, 0]);
        assert_eq!(indices[1], vec![0, 1]);
        assert_eq!(indices[2], vec![0, 2]);
        assert_eq!(indices[3], vec![1, 0]);
        assert_eq!(indices[4], vec![1, 1]);
        assert_eq!(indices[5], vec![1, 2]);
    }

    #[test]
    fn ndindex_1d() {
        let indices: Vec<Vec<usize>> = ndindex(&[4]).collect();
        assert_eq!(indices.len(), 4);
        assert_eq!(indices[0], vec![0]);
        assert_eq!(indices[3], vec![3]);
    }

    #[test]
    fn ndindex_empty() {
        let indices: Vec<Vec<usize>> = ndindex(&[0]).collect();
        assert_eq!(indices.len(), 0);
    }

    #[test]
    fn ndindex_scalar() {
        let indices: Vec<Vec<usize>> = ndindex(&[]).collect();
        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], Vec::<usize>::new());
    }

    // -----------------------------------------------------------------------
    // ndenumerate
    // -----------------------------------------------------------------------

    #[test]
    fn ndenumerate_2d() {
        let arr =
            Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![10, 20, 30, 40, 50, 60]).unwrap();
        let items: Vec<(Vec<usize>, &i32)> = ndenumerate(&arr).collect();
        assert_eq!(items.len(), 6);
        assert_eq!(items[0], (vec![0, 0], &10));
        assert_eq!(items[1], (vec![0, 1], &20));
        assert_eq!(items[5], (vec![1, 2], &60));
    }

    // -----------------------------------------------------------------------
    // put_along_axis
    // -----------------------------------------------------------------------

    #[test]
    fn put_along_axis_basic() {
        let mut arr = Array::<i32, Ix2>::from_vec(Ix2::new([3, 4]), vec![0; 12]).unwrap();
        let values =
            Array::<i32, IxDyn>::from_vec(IxDyn::new(&[8]), vec![1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        arr.put_along_axis(&[0, 2], &values, Axis(0)).unwrap();
        let data: Vec<i32> = arr.iter().copied().collect();
        assert_eq!(data, vec![1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8]);
    }

    // -----------------------------------------------------------------------
    // where_
    // -----------------------------------------------------------------------

    #[test]
    fn where_basic() {
        let cond =
            Array::<bool, Ix1>::from_vec(Ix1::new([4]), vec![true, false, true, false]).unwrap();
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let result = where_select(&cond, &x, &y).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn where_all_true() {
        let cond = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, true]).unwrap();
        let x = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let y = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let result = where_select(&cond, &x, &y).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[1, 2, 3]);
    }

    #[test]
    fn where_all_false() {
        let cond = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, false, false]).unwrap();
        let x = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let y = Array::<i32, Ix1>::from_vec(Ix1::new([3]), vec![10, 20, 30]).unwrap();
        let result = where_select(&cond, &x, &y).unwrap();
        assert_eq!(result.as_slice().unwrap(), &[10, 20, 30]);
    }

    #[test]
    fn where_shape_mismatch() {
        let cond = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true; 3]).unwrap();
        let x = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0; 4]).unwrap();
        let y = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0; 3]).unwrap();
        assert!(where_select(&cond, &x, &y).is_err());
    }

    #[test]
    fn where_2d() {
        let cond = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 2]),
            vec![true, false, false, true],
        )
        .unwrap();
        let x = Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![1, 2, 3, 4]).unwrap();
        let y = Array::<i32, Ix2>::from_vec(Ix2::new([2, 2]), vec![10, 20, 30, 40]).unwrap();
        let result = where_select(&cond, &x, &y).unwrap();
        let data: Vec<i32> = result.iter().copied().collect();
        assert_eq!(data, vec![1, 20, 30, 4]);
    }
}
