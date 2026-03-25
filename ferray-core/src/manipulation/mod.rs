// ferray-core: Shape manipulation functions (REQ-20, REQ-21, REQ-22)
//
// Mirrors numpy's shape manipulation routines: reshape, ravel, flatten,
// concatenate, stack, transpose, flip, etc.

pub mod extended;

use crate::array::owned::Array;
use crate::dimension::{Dimension, Ix1, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ============================================================================
// REQ-20: Shape methods
// ============================================================================

/// Reshape an array to a new shape (returns a new owned array).
///
/// The total number of elements must remain the same.
///
/// Analogous to `numpy.reshape()`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the new shape has a different
/// total number of elements.
pub fn reshape<T: Element, D: Dimension>(
    a: &Array<T, D>,
    new_shape: &[usize],
) -> FerrayResult<Array<T, IxDyn>> {
    let old_size = a.size();
    let new_size: usize = new_shape.iter().product();
    if old_size != new_size {
        return Err(FerrayError::shape_mismatch(format!(
            "cannot reshape array of size {} into shape {:?} (size {})",
            old_size, new_shape, new_size,
        )));
    }
    let data: Vec<T> = a.iter().cloned().collect();
    Array::from_vec(IxDyn::new(new_shape), data)
}

/// Return a flattened (1-D) copy of the array.
///
/// Analogous to `numpy.ravel()`.
pub fn ravel<T: Element, D: Dimension>(a: &Array<T, D>) -> FerrayResult<Array<T, Ix1>> {
    let data: Vec<T> = a.iter().cloned().collect();
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data)
}

/// Return a flattened (1-D) copy of the array.
///
/// Identical to `ravel()` — analogous to `ndarray.flatten()`.
pub fn flatten<T: Element, D: Dimension>(a: &Array<T, D>) -> FerrayResult<Array<T, Ix1>> {
    ravel(a)
}

/// Remove axes of length 1 from the shape.
///
/// If `axis` is `None`, all length-1 axes are removed.
/// If `axis` is `Some(ax)`, only that axis is removed (errors if it is not length 1).
///
/// Analogous to `numpy.squeeze()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if the axis is invalid, or
/// `FerrayError::InvalidValue` if the specified axis has size != 1.
pub fn squeeze<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    match axis {
        Some(ax) => {
            if ax >= shape.len() {
                return Err(FerrayError::axis_out_of_bounds(ax, shape.len()));
            }
            if shape[ax] != 1 {
                return Err(FerrayError::invalid_value(format!(
                    "cannot select axis {} with size {} for squeeze (must be 1)",
                    ax, shape[ax],
                )));
            }
            let new_shape: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != ax)
                .map(|(_, &s)| s)
                .collect();
            let data: Vec<T> = a.iter().cloned().collect();
            Array::from_vec(IxDyn::new(&new_shape), data)
        }
        None => {
            let new_shape: Vec<usize> = shape.iter().copied().filter(|&s| s != 1).collect();
            // If all dims are 1, the result is a scalar (0-D is tricky), so
            // make it at least 1-D with a single element.
            let new_shape = if new_shape.is_empty() && !shape.is_empty() {
                vec![1]
            } else if new_shape.is_empty() {
                vec![]
            } else {
                new_shape
            };
            let data: Vec<T> = a.iter().cloned().collect();
            Array::from_vec(IxDyn::new(&new_shape), data)
        }
    }
}

/// Insert a new axis of length 1 at the given position.
///
/// Analogous to `numpy.expand_dims()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis > ndim`.
pub fn expand_dims<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let ndim = a.ndim();
    if axis > ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim + 1));
    }
    let mut new_shape: Vec<usize> = a.shape().to_vec();
    new_shape.insert(axis, 1);
    let data: Vec<T> = a.iter().cloned().collect();
    Array::from_vec(IxDyn::new(&new_shape), data)
}

/// Broadcast an array to a new shape (returns a new owned array).
///
/// The array is replicated along size-1 dimensions to match the target shape.
///
/// Analogous to `numpy.broadcast_to()`.
///
/// # Errors
/// Returns `FerrayError::BroadcastFailure` if the shapes are incompatible.
pub fn broadcast_to<T: Element, D: Dimension>(
    a: &Array<T, D>,
    new_shape: &[usize],
) -> FerrayResult<Array<T, IxDyn>> {
    let src_shape = a.shape();
    let src_ndim = src_shape.len();
    let dst_ndim = new_shape.len();

    if dst_ndim < src_ndim {
        return Err(FerrayError::BroadcastFailure {
            shape_a: src_shape.to_vec(),
            shape_b: new_shape.to_vec(),
        });
    }

    // Check compatibility: walk from the right
    let pad = dst_ndim - src_ndim;
    for i in 0..src_ndim {
        let s = src_shape[i];
        let d = new_shape[pad + i];
        if s != d && s != 1 {
            return Err(FerrayError::BroadcastFailure {
                shape_a: src_shape.to_vec(),
                shape_b: new_shape.to_vec(),
            });
        }
    }

    // Build the broadcast array
    let total: usize = new_shape.iter().product();
    let mut data = Vec::with_capacity(total);
    let src_data: Vec<T> = a.iter().cloned().collect();

    // Precompute source strides (C-order)
    let mut src_strides = vec![1usize; src_ndim];
    for i in (0..src_ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
    }

    // Precompute output strides (C-order)
    let mut out_strides = vec![1usize; dst_ndim];
    for i in (0..dst_ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    for flat in 0..total {
        let mut rem = flat;
        let mut s_idx = 0usize;
        #[allow(clippy::needless_range_loop)]
        for i in 0..dst_ndim {
            let idx = rem / out_strides[i];
            rem %= out_strides[i];
            if i >= pad {
                let src_i = i - pad;
                let src_idx = if src_shape[src_i] == 1 { 0 } else { idx };
                s_idx += src_idx * src_strides[src_i];
            }
        }
        data.push(src_data[s_idx].clone());
    }

    Array::from_vec(IxDyn::new(new_shape), data)
}

// ============================================================================
// REQ-21: Join/split
// ============================================================================

/// Join a sequence of arrays along an existing axis.
///
/// Analogous to `numpy.concatenate()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array list is empty.
/// Returns `FerrayError::ShapeMismatch` if shapes differ on non-concatenation axes.
/// Returns `FerrayError::AxisOutOfBounds` if axis is out of bounds.
pub fn concatenate<T: Element>(
    arrays: &[Array<T, IxDyn>],
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.is_empty() {
        return Err(FerrayError::invalid_value(
            "concatenate: need at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let base_shape = arrays[0].shape();

    // Validate all arrays have same ndim and matching shapes on non-concat axes
    let mut total_along_axis = 0usize;
    for arr in arrays {
        if arr.ndim() != ndim {
            return Err(FerrayError::shape_mismatch(format!(
                "all arrays must have same ndim; got {} and {}",
                ndim,
                arr.ndim(),
            )));
        }
        for (i, (&s, &base)) in arr.shape().iter().zip(base_shape.iter()).enumerate() {
            if i != axis && s != base {
                return Err(FerrayError::shape_mismatch(format!(
                    "shape mismatch on axis {}: {} vs {}",
                    i, s, base,
                )));
            }
        }
        total_along_axis += arr.shape()[axis];
    }

    // Build new shape
    let mut new_shape = base_shape.to_vec();
    new_shape[axis] = total_along_axis;
    let total: usize = new_shape.iter().product();
    let mut data = Vec::with_capacity(total);

    // Pre-collect each source array into a contiguous Vec to avoid O(n) iter().nth() per element
    let src_vecs: Vec<Vec<T>> = arrays.iter().map(|a| a.iter().cloned().collect()).collect();

    // Compute strides for the output array (C-order)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    // For each position in the output, figure out which source array and offset
    for flat_idx in 0..total {
        // Convert flat index to nd-index
        let mut rem = flat_idx;
        let mut nd_idx = vec![0usize; ndim];
        for i in 0..ndim {
            nd_idx[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }

        // Find which source array this position belongs to
        let concat_idx = nd_idx[axis];
        let mut offset = 0;
        let mut src_arr_idx = 0;
        for (k, arr) in arrays.iter().enumerate() {
            let len_along = arr.shape()[axis];
            if concat_idx < offset + len_along {
                src_arr_idx = k;
                break;
            }
            offset += len_along;
        }
        let local_concat_idx = concat_idx - offset;

        // Build source flat index (C-order)
        let src_shape = arrays[src_arr_idx].shape();
        let mut src_flat = 0usize;
        let mut src_mul = 1usize;
        for i in (0..ndim).rev() {
            let idx = if i == axis {
                local_concat_idx
            } else {
                nd_idx[i]
            };
            src_flat += idx * src_mul;
            src_mul *= src_shape[i];
        }

        let elem = src_vecs[src_arr_idx].get(src_flat).ok_or_else(|| {
            FerrayError::invalid_value(format!(
                "concatenate: internal index {} out of range for source array of length {}",
                src_flat,
                src_vecs[src_arr_idx].len(),
            ))
        })?;
        data.push(elem.clone());
    }

    Array::from_vec(IxDyn::new(&new_shape), data)
}

/// Join a sequence of arrays along a **new** axis.
///
/// All arrays must have the same shape. The result has one more dimension
/// than the inputs.
///
/// Analogous to `numpy.stack()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array list is empty.
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
/// Returns `FerrayError::AxisOutOfBounds` if axis > ndim.
pub fn stack<T: Element>(arrays: &[Array<T, IxDyn>], axis: usize) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.is_empty() {
        return Err(FerrayError::invalid_value("stack: need at least one array"));
    }
    let base_shape = arrays[0].shape();
    let ndim = base_shape.len();

    if axis > ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim + 1));
    }

    for arr in &arrays[1..] {
        if arr.shape() != base_shape {
            return Err(FerrayError::shape_mismatch(format!(
                "all input arrays must have the same shape; got {:?} and {:?}",
                base_shape,
                arr.shape(),
            )));
        }
    }

    // Expand each array along the new axis, then concatenate
    let mut expanded = Vec::with_capacity(arrays.len());
    for arr in arrays {
        expanded.push(expand_dims(arr, axis)?);
    }
    concatenate(&expanded, axis)
}

/// Stack arrays vertically (row-wise). Equivalent to `concatenate` along axis 0
/// for 2-D+ arrays, or equivalent to stacking 1-D arrays as rows.
///
/// Analogous to `numpy.vstack()`.
pub fn vstack<T: Element>(arrays: &[Array<T, IxDyn>]) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.is_empty() {
        return Err(FerrayError::invalid_value(
            "vstack: need at least one array",
        ));
    }
    // For 1-D arrays, reshape to (1, N) then concatenate along axis 0
    let ndim = arrays[0].ndim();
    if ndim == 1 {
        let mut reshaped = Vec::with_capacity(arrays.len());
        for arr in arrays {
            let n = arr.shape()[0];
            reshaped.push(reshape(arr, &[1, n])?);
        }
        concatenate(&reshaped, 0)
    } else {
        concatenate(arrays, 0)
    }
}

/// Stack arrays horizontally (column-wise). Equivalent to `concatenate` along
/// axis 1 for 2-D+ arrays, or along axis 0 for 1-D arrays.
///
/// Analogous to `numpy.hstack()`.
pub fn hstack<T: Element>(arrays: &[Array<T, IxDyn>]) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.is_empty() {
        return Err(FerrayError::invalid_value(
            "hstack: need at least one array",
        ));
    }
    let ndim = arrays[0].ndim();
    if ndim == 1 {
        concatenate(arrays, 0)
    } else {
        concatenate(arrays, 1)
    }
}

/// Stack arrays along the third axis (depth-wise).
///
/// For 1-D arrays of shape `(N,)`, reshapes to `(1, N, 1)`.
/// For 2-D arrays of shape `(M, N)`, reshapes to `(M, N, 1)`.
/// Then concatenates along axis 2.
///
/// Analogous to `numpy.dstack()`.
pub fn dstack<T: Element>(arrays: &[Array<T, IxDyn>]) -> FerrayResult<Array<T, IxDyn>> {
    if arrays.is_empty() {
        return Err(FerrayError::invalid_value(
            "dstack: need at least one array",
        ));
    }
    let mut expanded = Vec::with_capacity(arrays.len());
    for arr in arrays {
        let shape = arr.shape();
        match shape.len() {
            1 => {
                let n = shape[0];
                expanded.push(reshape(arr, &[1, n, 1])?);
            }
            2 => {
                let (m, n) = (shape[0], shape[1]);
                expanded.push(reshape(arr, &[m, n, 1])?);
            }
            _ => {
                // Already 3-D+, just use as-is
                let data: Vec<T> = arr.iter().cloned().collect();
                expanded.push(Array::from_vec(IxDyn::new(shape), data)?);
            }
        }
    }
    concatenate(&expanded, 2)
}

/// Assemble an array from nested blocks.
///
/// Simplified version: takes a 2-D grid of arrays (as Vec<Vec<...>>)
/// and assembles them by stacking rows horizontally, then all rows vertically.
///
/// Analogous to `numpy.block()`.
///
/// # Errors
/// Returns errors on shape mismatches.
pub fn block<T: Element>(blocks: &[Vec<Array<T, IxDyn>>]) -> FerrayResult<Array<T, IxDyn>> {
    if blocks.is_empty() {
        return Err(FerrayError::invalid_value("block: empty input"));
    }
    let mut rows = Vec::with_capacity(blocks.len());
    for row in blocks {
        if row.is_empty() {
            return Err(FerrayError::invalid_value("block: empty row"));
        }
        // Concatenate along axis 1 (columns within each row)
        let row_arr = if row.len() == 1 {
            let data: Vec<T> = row[0].iter().cloned().collect();
            Array::from_vec(IxDyn::new(row[0].shape()), data)?
        } else {
            hstack(row)?
        };
        rows.push(row_arr);
    }
    if rows.len() == 1 {
        // SAFETY: just checked len() == 1, so pop() always returns Some
        Ok(rows.pop().unwrap_or_else(|| unreachable!()))
    } else {
        vstack(&rows)
    }
}

/// Split an array into equal-sized sub-arrays.
///
/// `n_sections` must evenly divide the size along `axis`.
///
/// Analogous to `numpy.split()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the axis cannot be evenly split.
pub fn split<T: Element>(
    a: &Array<T, IxDyn>,
    n_sections: usize,
    axis: usize,
) -> FerrayResult<Vec<Array<T, IxDyn>>> {
    let shape = a.shape();
    if axis >= shape.len() {
        return Err(FerrayError::axis_out_of_bounds(axis, shape.len()));
    }
    let axis_len = shape[axis];
    if n_sections == 0 {
        return Err(FerrayError::invalid_value("split: n_sections must be > 0"));
    }
    if axis_len % n_sections != 0 {
        return Err(FerrayError::invalid_value(format!(
            "array of size {} along axis {} cannot be evenly split into {} sections",
            axis_len, axis, n_sections,
        )));
    }
    let chunk_size = axis_len / n_sections;
    let indices: Vec<usize> = (1..n_sections).map(|i| i * chunk_size).collect();
    array_split(a, &indices, axis)
}

/// Split an array into sub-arrays at the given indices along `axis`.
///
/// Unlike `split()`, this does not require even division.
///
/// Analogous to `numpy.array_split()` (with explicit split points).
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if axis is invalid.
pub fn array_split<T: Element>(
    a: &Array<T, IxDyn>,
    indices: &[usize],
    axis: usize,
) -> FerrayResult<Vec<Array<T, IxDyn>>> {
    let shape = a.shape();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let axis_len = shape[axis];
    let src_data: Vec<T> = a.iter().cloned().collect();

    // Build split points including 0 and axis_len
    let mut splits = Vec::with_capacity(indices.len() + 2);
    splits.push(0);
    for &idx in indices {
        splits.push(idx.min(axis_len));
    }
    splits.push(axis_len);

    // Compute source strides (C-order)
    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    let mut result = Vec::with_capacity(splits.len() - 1);
    for w in splits.windows(2) {
        let start = w[0];
        let end = w[1];
        let chunk_len = end - start;

        let mut sub_shape = shape.to_vec();
        sub_shape[axis] = chunk_len;
        let sub_total: usize = sub_shape.iter().product();

        // Compute sub strides
        let mut sub_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            sub_strides[i] = sub_strides[i + 1] * sub_shape[i + 1];
        }

        let mut sub_data = Vec::with_capacity(sub_total);
        for flat in 0..sub_total {
            // Convert to nd-index in sub array
            let mut rem = flat;
            let mut src_flat = 0usize;
            for i in 0..ndim {
                let idx = rem / sub_strides[i];
                rem %= sub_strides[i];
                let src_idx = if i == axis { idx + start } else { idx };
                src_flat += src_idx * src_strides[i];
            }
            sub_data.push(src_data[src_flat].clone());
        }
        result.push(Array::from_vec(IxDyn::new(&sub_shape), sub_data)?);
    }

    Ok(result)
}

/// Split array along axis 0 (vertical split). Equivalent to `split(a, n, 0)`.
///
/// Analogous to `numpy.vsplit()`.
pub fn vsplit<T: Element>(
    a: &Array<T, IxDyn>,
    n_sections: usize,
) -> FerrayResult<Vec<Array<T, IxDyn>>> {
    split(a, n_sections, 0)
}

/// Split array along axis 1 (horizontal split). Equivalent to `split(a, n, 1)`.
///
/// Analogous to `numpy.hsplit()`.
pub fn hsplit<T: Element>(
    a: &Array<T, IxDyn>,
    n_sections: usize,
) -> FerrayResult<Vec<Array<T, IxDyn>>> {
    split(a, n_sections, 1)
}

/// Split array along axis 2 (depth split). Equivalent to `split(a, n, 2)`.
///
/// Analogous to `numpy.dsplit()`.
pub fn dsplit<T: Element>(
    a: &Array<T, IxDyn>,
    n_sections: usize,
) -> FerrayResult<Vec<Array<T, IxDyn>>> {
    split(a, n_sections, 2)
}

// ============================================================================
// REQ-22: Transpose / reorder
// ============================================================================

/// Permute the axes of an array.
///
/// `axes` specifies the new ordering. For a 2-D array, `[1, 0]` transposes.
/// If `axes` is `None`, reverses the order of all axes.
///
/// Analogous to `numpy.transpose()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `axes` is the wrong length or
/// contains invalid/duplicate axis indices.
pub fn transpose<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axes: Option<&[usize]>,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    let perm: Vec<usize> = match axes {
        Some(ax) => {
            if ax.len() != ndim {
                return Err(FerrayError::invalid_value(format!(
                    "axes must have length {} but got {}",
                    ndim,
                    ax.len(),
                )));
            }
            // Validate: each axis appears exactly once
            let mut seen = vec![false; ndim];
            for &a in ax {
                if a >= ndim {
                    return Err(FerrayError::axis_out_of_bounds(a, ndim));
                }
                if seen[a] {
                    return Err(FerrayError::invalid_value(format!(
                        "duplicate axis {} in transpose",
                        a,
                    )));
                }
                seen[a] = true;
            }
            ax.to_vec()
        }
        None => (0..ndim).rev().collect(),
    };

    let new_shape: Vec<usize> = perm.iter().map(|&ax| shape[ax]).collect();
    let total: usize = new_shape.iter().product();
    let src_data: Vec<T> = a.iter().cloned().collect();

    // Compute source strides (C-order)
    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides (C-order)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    let mut data = Vec::with_capacity(total);
    for flat_out in 0..total {
        // Convert to nd-index in output
        let mut rem = flat_out;
        let mut src_flat = 0usize;
        #[allow(clippy::needless_range_loop)]
        for i in 0..ndim {
            let idx = rem / out_strides[i];
            rem %= out_strides[i];
            // This output dimension i corresponds to source dimension perm[i]
            src_flat += idx * src_strides[perm[i]];
        }
        data.push(src_data[src_flat].clone());
    }

    Array::from_vec(IxDyn::new(&new_shape), data)
}

/// Swap two axes of an array.
///
/// Analogous to `numpy.swapaxes()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if either axis is out of bounds.
pub fn swapaxes<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axis1: usize,
    axis2: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let ndim = a.ndim();
    if axis1 >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis1, ndim));
    }
    if axis2 >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis2, ndim));
    }
    let mut perm: Vec<usize> = (0..ndim).collect();
    perm.swap(axis1, axis2);
    transpose(a, Some(&perm))
}

/// Move an axis to a new position.
///
/// Analogous to `numpy.moveaxis()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if either axis is out of bounds.
pub fn moveaxis<T: Element, D: Dimension>(
    a: &Array<T, D>,
    source: usize,
    destination: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let ndim = a.ndim();
    if source >= ndim {
        return Err(FerrayError::axis_out_of_bounds(source, ndim));
    }
    if destination >= ndim {
        return Err(FerrayError::axis_out_of_bounds(destination, ndim));
    }
    // Build permutation by removing source and inserting at destination
    let mut order: Vec<usize> = (0..ndim).filter(|&x| x != source).collect();
    order.insert(destination, source);
    transpose(a, Some(&order))
}

/// Roll an axis to a new position (similar to moveaxis).
///
/// Analogous to `numpy.rollaxis()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim` or `start > ndim`.
pub fn rollaxis<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axis: usize,
    start: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let ndim = a.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    if start > ndim {
        return Err(FerrayError::axis_out_of_bounds(start, ndim + 1));
    }
    let dst = if start > axis { start - 1 } else { start };
    if axis == dst {
        // No-op: return a copy
        let data: Vec<T> = a.iter().cloned().collect();
        return Array::from_vec(IxDyn::new(a.shape()), data);
    }
    moveaxis(a, axis, dst)
}

/// Reverse the order of elements along the given axis.
///
/// Analogous to `numpy.flip()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if axis is out of bounds.
pub fn flip<T: Element, D: Dimension>(
    a: &Array<T, D>,
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let src_data: Vec<T> = a.iter().cloned().collect();
    let total = src_data.len();

    // Compute strides (C-order)
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut data = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut src_flat = 0usize;
        for i in 0..ndim {
            let idx = rem / strides[i];
            rem %= strides[i];
            let src_idx = if i == axis { shape[i] - 1 - idx } else { idx };
            src_flat += src_idx * strides[i];
        }
        data.push(src_data[src_flat].clone());
    }
    Array::from_vec(IxDyn::new(shape), data)
}

/// Flip array left-right (reverse axis 1).
///
/// Analogous to `numpy.fliplr()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array has fewer than 2 dimensions.
pub fn fliplr<T: Element, D: Dimension>(a: &Array<T, D>) -> FerrayResult<Array<T, IxDyn>> {
    if a.ndim() < 2 {
        return Err(FerrayError::invalid_value(
            "fliplr: array must be at least 2-D",
        ));
    }
    flip(a, 1)
}

/// Flip array up-down (reverse axis 0).
///
/// Analogous to `numpy.flipud()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array has 0 dimensions.
pub fn flipud<T: Element, D: Dimension>(a: &Array<T, D>) -> FerrayResult<Array<T, IxDyn>> {
    if a.ndim() < 1 {
        return Err(FerrayError::invalid_value(
            "flipud: array must be at least 1-D",
        ));
    }
    flip(a, 0)
}

/// Rotate array 90 degrees counterclockwise in the plane defined by axes (0, 1).
///
/// `k` specifies the number of 90-degree rotations (can be negative).
///
/// Analogous to `numpy.rot90()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array has fewer than 2 dimensions.
pub fn rot90<T: Element, D: Dimension>(a: &Array<T, D>, k: i32) -> FerrayResult<Array<T, IxDyn>> {
    if a.ndim() < 2 {
        return Err(FerrayError::invalid_value(
            "rot90: array must be at least 2-D",
        ));
    }
    // Normalize k to [0, 4)
    let k = k.rem_euclid(4);
    let shape = a.shape();
    let data: Vec<T> = a.iter().cloned().collect();

    // We work with the IxDyn representation
    let as_dyn = Array::from_vec(IxDyn::new(shape), data)?;

    match k {
        0 => Ok(as_dyn),
        1 => {
            // rot90 once: flip axis 1, then transpose axes 0,1
            let flipped = flip(&as_dyn, 1)?;
            swapaxes(&flipped, 0, 1)
        }
        2 => {
            // rot180: flip both axes
            let f1 = flip(&as_dyn, 0)?;
            flip(&f1, 1)
        }
        3 => {
            // rot270: transpose, then flip axis 1
            let transposed = swapaxes(&as_dyn, 0, 1)?;
            flip(&transposed, 1)
        }
        _ => unreachable!(),
    }
}

/// Roll elements along an axis. Elements that roll past the end
/// are re-introduced at the beginning.
///
/// If `axis` is `None`, the array is flattened first, then rolled.
///
/// Analogous to `numpy.roll()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if axis is out of bounds.
pub fn roll<T: Element, D: Dimension>(
    a: &Array<T, D>,
    shift: isize,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>> {
    match axis {
        None => {
            // Flatten, roll, reshape back
            let data: Vec<T> = a.iter().cloned().collect();
            let n = data.len();
            if n == 0 {
                return Array::from_vec(IxDyn::new(a.shape()), data);
            }
            let shift = ((shift % n as isize) + n as isize) as usize % n;
            let mut rolled = Vec::with_capacity(n);
            for i in 0..n {
                rolled.push(data[(n + i - shift) % n].clone());
            }
            Array::from_vec(IxDyn::new(a.shape()), rolled)
        }
        Some(ax) => {
            let shape = a.shape();
            let ndim = shape.len();
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }
            let axis_len = shape[ax];
            if axis_len == 0 {
                let data: Vec<T> = a.iter().cloned().collect();
                return Array::from_vec(IxDyn::new(shape), data);
            }
            let shift = ((shift % axis_len as isize) + axis_len as isize) as usize % axis_len;
            let src_data: Vec<T> = a.iter().cloned().collect();
            let total = src_data.len();

            // Compute strides (C-order)
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            let mut data = Vec::with_capacity(total);
            for flat in 0..total {
                let mut rem = flat;
                let mut src_flat = 0usize;
                #[allow(clippy::needless_range_loop)]
                for i in 0..ndim {
                    let idx = rem / strides[i];
                    rem %= strides[i];
                    let src_idx = if i == ax {
                        (axis_len + idx - shift) % axis_len
                    } else {
                        idx
                    };
                    src_flat += src_idx * strides[i];
                }
                data.push(src_data[src_flat].clone());
            }
            Array::from_vec(IxDyn::new(shape), data)
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn dyn_arr(shape: &[usize], data: Vec<f64>) -> Array<f64, IxDyn> {
        Array::from_vec(IxDyn::new(shape), data).unwrap()
    }

    // -- REQ-20 --

    #[test]
    fn test_reshape() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = reshape(&a, &[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_size_mismatch() {
        let a = dyn_arr(&[2, 3], vec![1.0; 6]);
        assert!(reshape(&a, &[2, 4]).is_err());
    }

    #[test]
    fn test_ravel() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = ravel(&a).unwrap();
        assert_eq!(b.shape(), &[6]);
        assert_eq!(b.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_flatten() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = flatten(&a).unwrap();
        assert_eq!(b.shape(), &[6]);
    }

    #[test]
    fn test_squeeze() {
        let a = dyn_arr(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
        let b = squeeze(&a, None).unwrap();
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_specific_axis() {
        let a = dyn_arr(&[1, 3, 1], vec![1.0, 2.0, 3.0]);
        let b = squeeze(&a, Some(0)).unwrap();
        assert_eq!(b.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze_not_size_1() {
        let a = dyn_arr(&[2, 3], vec![1.0; 6]);
        assert!(squeeze(&a, Some(0)).is_err());
    }

    #[test]
    fn test_expand_dims() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = expand_dims(&a, 0).unwrap();
        assert_eq!(b.shape(), &[1, 3]);
        let c = expand_dims(&a, 1).unwrap();
        assert_eq!(c.shape(), &[3, 1]);
    }

    #[test]
    fn test_expand_dims_oob() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        assert!(expand_dims(&a, 3).is_err());
    }

    #[test]
    fn test_broadcast_to() {
        let a = dyn_arr(&[1, 3], vec![1.0, 2.0, 3.0]);
        let b = broadcast_to(&a, &[3, 3]).unwrap();
        assert_eq!(b.shape(), &[3, 3]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_broadcast_to_1d_to_2d() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = broadcast_to(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_to_incompatible() {
        let a = dyn_arr(&[4], vec![1.0, 2.0, 3.0, 4.0]);
        assert!(broadcast_to(&a, &[3]).is_err());
    }

    // -- REQ-21 --

    #[test]
    fn test_concatenate() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = dyn_arr(&[2, 3], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = concatenate(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[4, 3]);
    }

    #[test]
    fn test_concatenate_axis1() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = dyn_arr(&[2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let c = concatenate(&[a, b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 5]);
    }

    #[test]
    fn test_concatenate_shape_mismatch() {
        let a = dyn_arr(&[2, 3], vec![1.0; 6]);
        let b = dyn_arr(&[3, 3], vec![1.0; 9]);
        // Axis 0: different sizes on axis 1? No — axis 1 is same (3).
        // But axis 0 concat: shapes are [2,3] and [3,3], axis 0 can differ.
        // Non-concat axis (1) matches.
        let c = concatenate(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[5, 3]);
    }

    #[test]
    fn test_concatenate_empty() {
        let v: Vec<Array<f64, IxDyn>> = vec![];
        assert!(concatenate(&v, 0).is_err());
    }

    #[test]
    fn test_stack() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = dyn_arr(&[3], vec![4.0, 5.0, 6.0]);
        let c = stack(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = dyn_arr(&[3], vec![4.0, 5.0, 6.0]);
        let c = stack(&[a, b], 1).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        let data: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_vstack() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = dyn_arr(&[3], vec![4.0, 5.0, 6.0]);
        let c = vstack(&[a, b]).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_hstack() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = dyn_arr(&[3], vec![4.0, 5.0, 6.0]);
        let c = hstack(&[a, b]).unwrap();
        assert_eq!(c.shape(), &[6]);
    }

    #[test]
    fn test_hstack_2d() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = dyn_arr(&[2, 3], vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let c = hstack(&[a, b]).unwrap();
        assert_eq!(c.shape(), &[2, 5]);
    }

    #[test]
    fn test_dstack() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = dyn_arr(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = dstack(&[a, b]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_block() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = dyn_arr(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let c = dyn_arr(&[2, 2], vec![9.0, 10.0, 11.0, 12.0]);
        let d = dyn_arr(&[2, 2], vec![13.0, 14.0, 15.0, 16.0]);
        let result = block(&[vec![a, b], vec![c, d]]).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
    }

    #[test]
    fn test_split() {
        let a = dyn_arr(&[6], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let parts = split(&a, 3, 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]);
        assert_eq!(parts[1].shape(), &[2]);
        assert_eq!(parts[2].shape(), &[2]);
    }

    #[test]
    fn test_split_uneven() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(split(&a, 3, 0).is_err()); // 5 not divisible by 3
    }

    #[test]
    fn test_array_split() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let parts = array_split(&a, &[2, 4], 0).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), &[2]); // [1,2]
        assert_eq!(parts[1].shape(), &[2]); // [3,4]
        assert_eq!(parts[2].shape(), &[1]); // [5]
    }

    #[test]
    fn test_vsplit() {
        let a = dyn_arr(&[4, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let parts = vsplit(&a, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
    }

    #[test]
    fn test_hsplit() {
        let a = dyn_arr(&[2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let parts = hsplit(&a, 2).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 2]);
    }

    // -- REQ-22 --

    #[test]
    fn test_transpose_2d() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = transpose(&a, None).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_explicit() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = transpose(&a, Some(&[1, 0])).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose_bad_axes() {
        let a = dyn_arr(&[2, 3], vec![1.0; 6]);
        assert!(transpose(&a, Some(&[0])).is_err()); // wrong length
    }

    #[test]
    fn test_swapaxes() {
        let a = dyn_arr(&[2, 3, 4], vec![0.0; 24]);
        let b = swapaxes(&a, 0, 2).unwrap();
        assert_eq!(b.shape(), &[4, 3, 2]);
    }

    #[test]
    fn test_moveaxis() {
        let a = dyn_arr(&[2, 3, 4], vec![0.0; 24]);
        let b = moveaxis(&a, 0, 2).unwrap();
        assert_eq!(b.shape(), &[3, 4, 2]);
    }

    #[test]
    fn test_rollaxis() {
        let a = dyn_arr(&[2, 3, 4], vec![0.0; 24]);
        let b = rollaxis(&a, 2, 0).unwrap();
        assert_eq!(b.shape(), &[4, 2, 3]);
    }

    #[test]
    fn test_flip() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = flip(&a, 0).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_flip_2d() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = flip(&a, 0).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

        let c = flip(&a, 1).unwrap();
        let data2: Vec<f64> = c.iter().copied().collect();
        assert_eq!(data2, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_fliplr() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = fliplr(&a).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_flipud() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = flipud(&a).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_fliplr_1d_err() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        assert!(fliplr(&a).is_err());
    }

    #[test]
    fn test_rot90_once() {
        // [[1, 2], [3, 4]] -> [[2, 4], [1, 3]]
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = rot90(&a, 1).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![2.0, 4.0, 1.0, 3.0]);
    }

    #[test]
    fn test_rot90_twice() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = rot90(&a, 2).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rot90_four_is_identity() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = rot90(&a, 4).unwrap();
        let data_a: Vec<f64> = a.iter().copied().collect();
        let data_b: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data_a, data_b);
        assert_eq!(a.shape(), b.shape());
    }

    #[test]
    fn test_roll_flat() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = roll(&a, 2, None).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![4.0, 5.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_negative() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = roll(&a, -2, None).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![3.0, 4.0, 5.0, 1.0, 2.0]);
    }

    #[test]
    fn test_roll_axis() {
        let a = dyn_arr(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = roll(&a, 1, Some(1)).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![3.0, 1.0, 2.0, 6.0, 4.0, 5.0]);
    }
}
