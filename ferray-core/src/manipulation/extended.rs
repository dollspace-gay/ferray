// ferray-core: Extended manipulation functions (REQ-22a)
//
// pad, tile, repeat, delete, insert, append, resize, trim_zeros

use crate::array::owned::Array;
use crate::dimension::{Dimension, Ix1, IxDyn};
use crate::dtype::Element;
use crate::error::{FerrayError, FerrayResult};

// ============================================================================
// Pad modes
// ============================================================================

/// Padding mode for [`pad`].
#[derive(Debug, Clone)]
pub enum PadMode<T: Element> {
    /// Pad with a constant value.
    Constant(T),
    /// Pad with the edge values of the array.
    Edge,
    /// Pad with the reflection of the array mirrored on the first and last
    /// values (does not repeat the edge).
    Reflect,
    /// Pad with the reflection of the array mirrored along the edge.
    Symmetric,
    /// Pad by wrapping the array around.
    Wrap,
}

/// Pad a 1-D array.
///
/// `pad_width` is `(before, after)` for the single axis.
///
/// Analogous to `numpy.pad()` for 1-D.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if the array is empty and the mode
/// requires elements (Edge, Reflect, Symmetric, Wrap).
pub fn pad_1d<T: Element>(
    a: &Array<T, Ix1>,
    pad_width: (usize, usize),
    mode: &PadMode<T>,
) -> FerrayResult<Array<T, Ix1>> {
    let n = a.shape()[0];
    let (before, after) = pad_width;
    let new_len = before + n + after;
    let src: Vec<T> = a.iter().cloned().collect();

    if n == 0 && !matches!(mode, PadMode::Constant(_)) {
        return Err(FerrayError::invalid_value(
            "pad: cannot use Edge/Reflect/Symmetric/Wrap mode on empty array",
        ));
    }

    let mut data = Vec::with_capacity(new_len);

    // Fill 'before' padding
    for i in 0..before {
        let val = match mode {
            PadMode::Constant(c) => c.clone(),
            PadMode::Edge => src[0].clone(),
            PadMode::Reflect => {
                // Reflect: index = before - 1 - i mapped into [1..n-1] reflected
                let idx = reflect_index(before as isize - 1 - i as isize, n);
                src[idx].clone()
            }
            PadMode::Symmetric => {
                // Symmetric: similar but includes edge
                let idx = symmetric_index(before as isize - 1 - i as isize, n);
                src[idx].clone()
            }
            PadMode::Wrap => {
                let idx = ((n as isize - (before as isize - i as isize) % n as isize) % n as isize)
                    as usize;
                src[idx].clone()
            }
        };
        data.push(val);
    }

    // Copy original data
    data.extend_from_slice(&src);

    // Fill 'after' padding
    for i in 0..after {
        let val = match mode {
            PadMode::Constant(c) => c.clone(),
            PadMode::Edge => src[n - 1].clone(),
            PadMode::Reflect => {
                let idx = reflect_index(n as isize + i as isize, n);
                src[idx].clone()
            }
            PadMode::Symmetric => {
                let idx = symmetric_index(n as isize + i as isize, n);
                src[idx].clone()
            }
            PadMode::Wrap => {
                let idx = i % n;
                src[idx].clone()
            }
        };
        data.push(val);
    }

    Array::from_vec(Ix1::new([new_len]), data)
}

/// Reflect index: maps indices outside [0, n) by reflecting at boundaries 1 and n-2.
/// This means the edge values are not repeated.
fn reflect_index(idx: isize, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let period = (n - 1) as isize * 2;
    let mut i = idx % period;
    if i < 0 {
        i += period;
    }
    if i >= n as isize {
        i = period - i;
    }
    i as usize
}

/// Symmetric index: maps indices outside [0, n) by reflecting at boundaries 0 and n-1.
/// The edge values are repeated.
fn symmetric_index(idx: isize, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 0;
    }
    let period = n as isize * 2;
    let mut i = idx % period;
    if i < 0 {
        i += period;
    }
    if i >= n as isize {
        i = period - 1 - i;
    }
    i.max(0) as usize
}

/// Pad an N-D array.
///
/// `pad_width` is a slice of `(before, after)` pairs, one per axis.
/// If it has fewer entries than the array's ndim, the last entry is repeated.
///
/// Analogous to `numpy.pad()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `pad_width` is empty.
pub fn pad<T: Element, D: Dimension>(
    a: &Array<T, D>,
    pad_width: &[(usize, usize)],
    mode: &PadMode<T>,
) -> FerrayResult<Array<T, IxDyn>> {
    if pad_width.is_empty() {
        return Err(FerrayError::invalid_value("pad: pad_width cannot be empty"));
    }

    let shape = a.shape();
    let ndim = shape.len();

    // Expand pad_width to ndim entries
    let pads: Vec<(usize, usize)> = (0..ndim)
        .map(|i| {
            if i < pad_width.len() {
                pad_width[i]
            } else {
                *pad_width.last().unwrap()
            }
        })
        .collect();

    // For multi-dimensional, we pad axis-by-axis starting from the last axis.
    // Convert to IxDyn first.
    let mut current_data: Vec<T> = a.iter().cloned().collect();
    let mut current_shape: Vec<usize> = shape.to_vec();

    for ax in (0..ndim).rev() {
        let (before, after) = pads[ax];
        if before == 0 && after == 0 {
            continue;
        }
        let axis_len = current_shape[ax];
        let new_axis_len = before + axis_len + after;

        // Compute strides
        let outer: usize = current_shape[..ax].iter().product();
        let inner: usize = current_shape[ax + 1..].iter().product();

        let new_total = outer * new_axis_len * inner;
        let mut new_data = Vec::with_capacity(new_total);

        for o in 0..outer {
            for j in 0..new_axis_len {
                for k in 0..inner {
                    let val = if j < before {
                        // Before padding
                        match mode {
                            PadMode::Constant(c) => c.clone(),
                            PadMode::Edge => {
                                let src_j = 0;
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Reflect => {
                                let src_j =
                                    reflect_index(before as isize - 1 - j as isize, axis_len);
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Symmetric => {
                                let src_j =
                                    symmetric_index(before as isize - 1 - j as isize, axis_len);
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Wrap => {
                                let src_j = ((axis_len as isize
                                    - (before as isize - j as isize) % axis_len as isize)
                                    % axis_len as isize)
                                    as usize;
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                        }
                    } else if j < before + axis_len {
                        // Original data
                        let src_j = j - before;
                        current_data[o * axis_len * inner + src_j * inner + k].clone()
                    } else {
                        // After padding
                        let after_idx = j - before - axis_len;
                        match mode {
                            PadMode::Constant(c) => c.clone(),
                            PadMode::Edge => {
                                let src_j = axis_len - 1;
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Reflect => {
                                let src_j = reflect_index(
                                    (axis_len as isize) + after_idx as isize,
                                    axis_len,
                                );
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Symmetric => {
                                let src_j = symmetric_index(
                                    (axis_len as isize) + after_idx as isize,
                                    axis_len,
                                );
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                            PadMode::Wrap => {
                                let src_j = after_idx % axis_len;
                                current_data[o * axis_len * inner + src_j * inner + k].clone()
                            }
                        }
                    };
                    new_data.push(val);
                }
            }
        }

        current_data = new_data;
        current_shape[ax] = new_axis_len;
    }

    Array::from_vec(IxDyn::new(&current_shape), current_data)
}

/// Construct an array by repeating `a` the number of times given by `reps`.
///
/// If `reps` has fewer entries than `a.ndim()`, it is prepended with 1s.
/// If `reps` has more entries, `a`'s shape is prepended with 1s.
///
/// Analogous to `numpy.tile()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `reps` is empty.
pub fn tile<T: Element, D: Dimension>(
    a: &Array<T, D>,
    reps: &[usize],
) -> FerrayResult<Array<T, IxDyn>> {
    if reps.is_empty() {
        return Err(FerrayError::invalid_value("tile: reps cannot be empty"));
    }

    let src_shape = a.shape();
    let src_ndim = src_shape.len();
    let reps_ndim = reps.len();
    let out_ndim = src_ndim.max(reps_ndim);

    // Pad shapes to out_ndim
    let mut padded_shape = vec![1usize; out_ndim];
    for i in 0..src_ndim {
        padded_shape[out_ndim - src_ndim + i] = src_shape[i];
    }
    let mut padded_reps = vec![1usize; out_ndim];
    for i in 0..reps_ndim {
        padded_reps[out_ndim - reps_ndim + i] = reps[i];
    }

    let out_shape: Vec<usize> = padded_shape
        .iter()
        .zip(padded_reps.iter())
        .map(|(&s, &r)| s * r)
        .collect();
    let total: usize = out_shape.iter().product();

    let src_data: Vec<T> = a.iter().cloned().collect();
    let mut data = Vec::with_capacity(total);

    // Compute strides for output and padded source
    let mut out_strides = vec![1usize; out_ndim];
    for i in (0..out_ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let mut src_strides = vec![1usize; out_ndim];
    for i in (0..out_ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * padded_shape[i + 1];
    }

    for flat in 0..total {
        let mut rem = flat;
        let mut src_flat = 0usize;
        for i in 0..out_ndim {
            let idx = rem / out_strides[i];
            rem %= out_strides[i];
            let src_idx = idx % padded_shape[i];
            src_flat += src_idx * src_strides[i];
        }
        // Map src_flat to the original (non-padded) source
        // Since we padded with 1s, the strides handle it.
        if src_flat < src_data.len() {
            data.push(src_data[src_flat].clone());
        } else {
            // This shouldn't happen if the math is right
            data.push(T::zero());
        }
    }

    Array::from_vec(IxDyn::new(&out_shape), data)
}

/// Repeat elements of an array.
///
/// If `axis` is `None`, the array is flattened first, then each element
/// is repeated `repeats` times.
///
/// Analogous to `numpy.repeat()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if the axis is out of bounds.
pub fn repeat<T: Element, D: Dimension>(
    a: &Array<T, D>,
    repeats: usize,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>> {
    match axis {
        None => {
            // Flatten and repeat each element
            let src: Vec<T> = a.iter().cloned().collect();
            let mut data = Vec::with_capacity(src.len() * repeats);
            for val in &src {
                for _ in 0..repeats {
                    data.push(val.clone());
                }
            }
            let n = data.len();
            Array::from_vec(IxDyn::new(&[n]), data)
        }
        Some(ax) => {
            let shape = a.shape();
            let ndim = shape.len();
            if ax >= ndim {
                return Err(FerrayError::axis_out_of_bounds(ax, ndim));
            }

            let mut new_shape = shape.to_vec();
            new_shape[ax] *= repeats;
            let total: usize = new_shape.iter().product();
            let src_data: Vec<T> = a.iter().cloned().collect();

            // Compute source strides (C-order)
            let mut src_strides = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                src_strides[i] = src_strides[i + 1] * shape[i + 1];
            }

            // Compute output strides
            let mut out_strides = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
            }

            let mut data = Vec::with_capacity(total);
            for flat in 0..total {
                let mut rem = flat;
                let mut src_flat = 0usize;
                for i in 0..ndim {
                    let idx = rem / out_strides[i];
                    rem %= out_strides[i];
                    let src_idx = if i == ax { idx / repeats } else { idx };
                    src_flat += src_idx * src_strides[i];
                }
                data.push(src_data[src_flat].clone());
            }

            Array::from_vec(IxDyn::new(&new_shape), data)
        }
    }
}

/// Delete sub-arrays along an axis.
///
/// `indices` specifies which indices along `axis` to remove.
///
/// Analogous to `numpy.delete()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if axis is out of bounds.
/// Returns `FerrayError::IndexOutOfBounds` if any index is out of range.
pub fn delete<T: Element, D: Dimension>(
    a: &Array<T, D>,
    indices: &[usize],
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let axis_len = shape[axis];

    // Validate indices
    for &idx in indices {
        if idx >= axis_len {
            return Err(FerrayError::IndexOutOfBounds {
                index: idx as isize,
                axis,
                size: axis_len,
            });
        }
    }

    let to_remove: std::collections::HashSet<usize> = indices.iter().copied().collect();
    let kept: Vec<usize> = (0..axis_len).filter(|i| !to_remove.contains(i)).collect();
    let new_axis_len = kept.len();

    let mut new_shape = shape.to_vec();
    new_shape[axis] = new_axis_len;
    let total: usize = new_shape.iter().product();
    let src_data: Vec<T> = a.iter().cloned().collect();

    // Compute source strides (C-order)
    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    let mut data = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut src_flat = 0usize;
        for i in 0..ndim {
            let idx = rem / out_strides[i];
            rem %= out_strides[i];
            let src_idx = if i == axis { kept[idx] } else { idx };
            src_flat += src_idx * src_strides[i];
        }
        data.push(src_data[src_flat].clone());
    }

    Array::from_vec(IxDyn::new(&new_shape), data)
}

/// Insert values along an axis before a given index.
///
/// `index` is the position before which to insert. `values` is a 1-D array
/// of values to insert (its length determines how many slices are added).
///
/// Analogous to `numpy.insert()`.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if axis is out of bounds.
/// Returns `FerrayError::IndexOutOfBounds` if `index > axis_len`.
pub fn insert<T: Element, D: Dimension>(
    a: &Array<T, D>,
    index: usize,
    values: &Array<T, IxDyn>,
    axis: usize,
) -> FerrayResult<Array<T, IxDyn>> {
    let shape = a.shape();
    let ndim = shape.len();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let axis_len = shape[axis];
    if index > axis_len {
        return Err(FerrayError::IndexOutOfBounds {
            index: index as isize,
            axis,
            size: axis_len + 1,
        });
    }

    let n_insert = values.size();
    let vals: Vec<T> = values.iter().cloned().collect();

    let mut new_shape = shape.to_vec();
    new_shape[axis] = axis_len + n_insert;
    let total: usize = new_shape.iter().product();
    let src_data: Vec<T> = a.iter().cloned().collect();

    // Compute strides
    let mut src_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    // Compute the "inner" size (product of dims after axis)
    let inner: usize = shape[axis + 1..].iter().product();

    let mut data = Vec::with_capacity(total);
    for flat in 0..total {
        let mut rem = flat;
        let mut nd_idx = vec![0usize; ndim];
        for i in 0..ndim {
            nd_idx[i] = rem / out_strides[i];
            rem %= out_strides[i];
        }

        let ax_idx = nd_idx[axis];
        if ax_idx >= index && ax_idx < index + n_insert {
            // This is an inserted value
            let insert_idx = ax_idx - index;
            // For multi-D insert, we tile the values along the inner dims
            let val_idx = (insert_idx * inner + nd_idx.get(axis + 1).copied().unwrap_or(0))
                % vals.len().max(1);
            data.push(vals[val_idx].clone());
        } else {
            // Original data
            let src_ax_idx = if ax_idx >= index + n_insert {
                ax_idx - n_insert
            } else {
                ax_idx
            };
            let mut src_flat = 0usize;
            for i in 0..ndim {
                let idx = if i == axis { src_ax_idx } else { nd_idx[i] };
                src_flat += idx * src_strides[i];
            }
            data.push(src_data[src_flat].clone());
        }
    }

    Array::from_vec(IxDyn::new(&new_shape), data)
}

/// Append values to the end of an array along an axis.
///
/// If `axis` is `None`, both arrays are flattened first.
///
/// Analogous to `numpy.append()`.
pub fn append<T: Element, D: Dimension>(
    a: &Array<T, D>,
    values: &Array<T, IxDyn>,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>> {
    match axis {
        None => {
            let mut data: Vec<T> = a.iter().cloned().collect();
            data.extend(values.iter().cloned());
            let n = data.len();
            Array::from_vec(IxDyn::new(&[n]), data)
        }
        Some(ax) => {
            let a_dyn = {
                let data: Vec<T> = a.iter().cloned().collect();
                Array::from_vec(IxDyn::new(a.shape()), data)?
            };
            let vals_dyn = {
                let data: Vec<T> = values.iter().cloned().collect();
                Array::from_vec(IxDyn::new(values.shape()), data)?
            };
            super::concatenate(&[a_dyn, vals_dyn], ax)
        }
    }
}

/// Resize an array to a new shape.
///
/// If the new size is larger, the array is filled by repeating its elements.
/// If smaller, the array is truncated.
///
/// Analogous to `numpy.resize()`.
pub fn resize<T: Element, D: Dimension>(
    a: &Array<T, D>,
    new_shape: &[usize],
) -> FerrayResult<Array<T, IxDyn>> {
    let src: Vec<T> = a.iter().cloned().collect();
    let new_size: usize = new_shape.iter().product();

    if src.is_empty() {
        // Fill with zeros
        let data = vec![T::zero(); new_size];
        return Array::from_vec(IxDyn::new(new_shape), data);
    }

    let mut data = Vec::with_capacity(new_size);
    for i in 0..new_size {
        data.push(src[i % src.len()].clone());
    }
    Array::from_vec(IxDyn::new(new_shape), data)
}

/// Trim leading and/or trailing zeros from a 1-D array.
///
/// `trim` can be `"f"` (front), `"b"` (back), or `"fb"` (both, default).
///
/// Analogous to `numpy.trim_zeros()`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `trim` contains invalid characters.
pub fn trim_zeros<T: Element + PartialEq>(
    a: &Array<T, Ix1>,
    trim: &str,
) -> FerrayResult<Array<T, Ix1>> {
    let data: Vec<T> = a.iter().cloned().collect();
    let zero = T::zero();

    let trim_front = trim.contains('f');
    let trim_back = trim.contains('b');

    if !trim.chars().all(|c| c == 'f' || c == 'b') {
        return Err(FerrayError::invalid_value(
            "trim_zeros: trim must contain only 'f' and/or 'b'",
        ));
    }

    let start = if trim_front {
        data.iter().position(|v| *v != zero).unwrap_or(data.len())
    } else {
        0
    };

    let end = if trim_back {
        data.iter()
            .rposition(|v| *v != zero)
            .map(|i| i + 1)
            .unwrap_or(start)
    } else {
        data.len()
    };

    let end = end.max(start);
    let trimmed: Vec<T> = data[start..end].to_vec();
    let n = trimmed.len();
    Array::from_vec(Ix1::new([n]), trimmed)
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

    fn arr1d(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    // -- pad --

    #[test]
    fn test_pad_1d_constant() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let b = pad_1d(&a, (2, 3), &PadMode::Constant(0.0)).unwrap();
        assert_eq!(b.shape(), &[8]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_pad_1d_edge() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let b = pad_1d(&a, (2, 2), &PadMode::Edge).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_pad_1d_wrap() {
        let a = arr1d(vec![1.0, 2.0, 3.0]);
        let b = pad_1d(&a, (2, 2), &PadMode::Wrap).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_pad_nd_constant() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = pad(&a, &[(1, 1), (1, 1)], &PadMode::Constant(0.0)).unwrap();
        assert_eq!(b.shape(), &[4, 4]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(
            data,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ]
        );
    }

    // -- tile --

    #[test]
    fn test_tile_1d() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = tile(&a, &[3]).unwrap();
        assert_eq!(b.shape(), &[9]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tile_2d() {
        let a = dyn_arr(&[2], vec![1.0, 2.0]);
        let b = tile(&a, &[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 6]);
    }

    // -- repeat --

    #[test]
    fn test_repeat_flat() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = repeat(&a, 2, None).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn test_repeat_axis() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = repeat(&a, 2, Some(0)).unwrap();
        assert_eq!(b.shape(), &[4, 2]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
    }

    // -- delete --

    #[test]
    fn test_delete() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = delete(&a, &[1, 3], 0).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_delete_2d() {
        let a = dyn_arr(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = delete(&a, &[1], 0).unwrap();
        assert_eq!(b.shape(), &[2, 2]);
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 5.0, 6.0]);
    }

    // -- insert --

    #[test]
    fn test_insert() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let vals = dyn_arr(&[2], vec![10.0, 20.0]);
        let b = insert(&a, 1, &vals, 0).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 10.0, 20.0, 2.0, 3.0]);
    }

    // -- append --

    #[test]
    fn test_append_flat() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let vals = dyn_arr(&[2], vec![4.0, 5.0]);
        let b = append(&a, &vals, None).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_append_axis() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let vals = dyn_arr(&[2, 1], vec![5.0, 6.0]);
        let b = append(&a, &vals, Some(1)).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    // -- resize --

    #[test]
    fn test_resize_larger() {
        let a = dyn_arr(&[3], vec![1.0, 2.0, 3.0]);
        let b = resize(&a, &[5]).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 1.0, 2.0]);
    }

    #[test]
    fn test_resize_smaller() {
        let a = dyn_arr(&[5], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = resize(&a, &[3]).unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_resize_2d() {
        let a = dyn_arr(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = resize(&a, &[3, 3]).unwrap();
        assert_eq!(b.shape(), &[3, 3]);
    }

    // -- trim_zeros --

    #[test]
    fn test_trim_zeros_both() {
        let a = arr1d(vec![0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);
        let b = trim_zeros(&a, "fb").unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_trim_zeros_front() {
        let a = arr1d(vec![0.0, 0.0, 1.0, 2.0, 0.0]);
        let b = trim_zeros(&a, "f").unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 0.0]);
    }

    #[test]
    fn test_trim_zeros_back() {
        let a = arr1d(vec![0.0, 1.0, 2.0, 0.0, 0.0]);
        let b = trim_zeros(&a, "b").unwrap();
        let data: Vec<f64> = b.iter().copied().collect();
        assert_eq!(data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_trim_zeros_all_zeros() {
        let a = arr1d(vec![0.0, 0.0, 0.0]);
        let b = trim_zeros(&a, "fb").unwrap();
        assert_eq!(b.shape(), &[0]);
    }

    #[test]
    fn test_trim_zeros_bad_mode() {
        let a = arr1d(vec![1.0, 2.0]);
        assert!(trim_zeros(&a, "x").is_err());
    }
}
