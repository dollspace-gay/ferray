//! ferray-ma: masked specialized algorithms mirroring `numpy.ma`.
//!
//! Each function mirrors its unmasked NumPy counterpart with explicit mask
//! propagation, matching the observable (data + mask) contract of the
//! corresponding `numpy.ma.*` routine (verified live against numpy 2.4.5):
//!
//! - [`ma_where`] — `numpy.ma.where(condition, x, y)` (`numpy/ma/core.py:7915`):
//!   `data = where(cf, xd, yd)`, `mask = where(cf, xm, ym)` then OR the
//!   condition mask, where `cf = filled(condition, False)`.
//! - [`ma_choose`] — `numpy.ma.choose(indices, choices)`
//!   (`numpy/ma/core.py:8007`): pick `data[indices]` / `mask[indices]` lane by
//!   lane, OR the index mask.
//! - [`ma_diff`] — `numpy.ma.diff(a, n, axis)` (`numpy/ma/core.py:7774`):
//!   `out[i] = a[i+1] - a[i]`; the result position is masked iff either
//!   operand is masked. `n` applies the difference recursively.
//! - [`ma_ediff1d`] — `numpy.ma.ediff1d(ary)`
//!   (`numpy/ma/extras.py:1229`): flattened first difference with optional
//!   `to_begin` / `to_end` (always unmasked, per numpy's `hstack`).
//! - [`ma_nonzero`] — `numpy.ma.nonzero(a)` (`MaskedArray.nonzero`,
//!   `numpy/ma/core.py:5049`): indices of elements that are non-zero AND
//!   unmasked (masked elements are treated as zero, `filled(self, 0)`).
//!
//! ## REQ status
//! - REQ-18 (ma_where) SHIPPED — `ma_where` in `algorithms.rs`; consumer
//!   `ferray-python/src/ma.rs::where_`.
//! - REQ-19 (ma_choose) SHIPPED — `ma_choose` in `algorithms.rs`; consumer
//!   `ferray-python/src/ma.rs::choose`.
//! - REQ-20 (ma_diff) SHIPPED — `ma_diff` in `algorithms.rs`; consumer
//!   `ferray-python/src/ma.rs::diff`.
//! - REQ-21 (ma_ediff1d) SHIPPED — `ma_ediff1d` in `algorithms.rs`; consumer
//!   `ferray-python/src/ma.rs::ediff1d`.
//! - REQ-22 (ma_nonzero) SHIPPED — `ma_nonzero` in `algorithms.rs`; consumer
//!   `ferray-python/src/ma.rs::nonzero`.

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Ix1};

use crate::MaskedArray;

/// Row-major strides for `shape`.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// `numpy.ma.where(condition, x, y)` — elementwise select with mask
/// propagation (`numpy/ma/core.py:7915`).
///
/// All three operands must share the same shape (ferray's f64 model does not
/// broadcast here). For each position `i`:
/// - `data[i] = if cond[i] { x.data[i] } else { y.data[i] }`, where a *masked*
///   condition counts as `False` (`cf = filled(condition, False)`), so the `y`
///   branch supplies the data.
/// - `mask[i] = (if cond[i] { x.mask[i] } else { y.mask[i] }) || cond.mask[i]`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if the three operands differ in shape,
/// or an internal array-construction error.
pub fn ma_where<D>(
    condition: &MaskedArray<bool, D>,
    x: &MaskedArray<f64, D>,
    y: &MaskedArray<f64, D>,
) -> FerrayResult<MaskedArray<f64, D>>
where
    D: Dimension,
{
    let shape = condition.shape();
    if x.shape() != shape || y.shape() != shape {
        return Err(FerrayError::shape_mismatch(format!(
            "ma_where: condition {:?}, x {:?}, y {:?} must share one shape",
            shape,
            x.shape(),
            y.shape()
        )));
    }
    let cd: Vec<bool> = condition.data().iter().copied().collect();
    let cm: Vec<bool> = condition.mask().iter().copied().collect();
    let xd: Vec<f64> = x.data().iter().copied().collect();
    let xm: Vec<bool> = x.mask().iter().copied().collect();
    let yd: Vec<f64> = y.data().iter().copied().collect();
    let ym: Vec<bool> = y.mask().iter().copied().collect();

    let n = cd.len();
    let mut out_data = Vec::with_capacity(n);
    let mut out_mask = Vec::with_capacity(n);
    for i in 0..n {
        // A masked condition is filled with False -> picks the y branch.
        let cf = cd[i] && !cm[i];
        if cf {
            out_data.push(xd[i]);
            out_mask.push(xm[i] || cm[i]);
        } else {
            out_data.push(yd[i]);
            out_mask.push(ym[i] || cm[i]);
        }
    }
    let data_arr = Array::from_vec(condition.data().dim().clone(), out_data)?;
    let mask_arr = Array::from_vec(condition.data().dim().clone(), out_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// `numpy.ma.choose(indices, choices)` — build a new array by selecting, at
/// each position, the element of `choices[indices[i]]` (`numpy/ma/core.py:8007`).
///
/// All choice arrays and `indices` share the same shape. The result is masked
/// at position `i` iff the selected choice is masked there OR `indices` is
/// masked there (a masked index is filled with 0, `c = filled(indices, 0)`).
/// An out-of-range index is an error (numpy's default `mode='raise'`).
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` for ragged shapes, `InvalidValue` for
/// an empty choice list or a non-integer index, or `IndexOutOfBounds` for an
/// out-of-range index.
pub fn ma_choose<D>(
    indices: &MaskedArray<f64, D>,
    choices: &[MaskedArray<f64, D>],
) -> FerrayResult<MaskedArray<f64, D>>
where
    D: Dimension,
{
    if choices.is_empty() {
        return Err(FerrayError::invalid_value(
            "ma_choose: choices must be a non-empty sequence",
        ));
    }
    let shape = indices.shape();
    for (k, c) in choices.iter().enumerate() {
        if c.shape() != shape {
            return Err(FerrayError::shape_mismatch(format!(
                "ma_choose: choices[{k}] shape {:?} != indices shape {:?}",
                c.shape(),
                shape
            )));
        }
    }
    let idx_d: Vec<f64> = indices.data().iter().copied().collect();
    let idx_m: Vec<bool> = indices.mask().iter().copied().collect();
    let ch_d: Vec<Vec<f64>> = choices
        .iter()
        .map(|c| c.data().iter().copied().collect())
        .collect();
    let ch_m: Vec<Vec<bool>> = choices
        .iter()
        .map(|c| c.mask().iter().copied().collect())
        .collect();

    let n = idx_d.len();
    let nchoices = choices.len();
    let mut out_data = Vec::with_capacity(n);
    let mut out_mask = Vec::with_capacity(n);
    for i in 0..n {
        // A masked index is filled with 0 (numpy's `c = filled(indices, 0)`).
        let raw = if idx_m[i] { 0.0 } else { idx_d[i] };
        if !(raw.is_finite() && raw.fract() == 0.0 && raw >= 0.0) {
            return Err(FerrayError::invalid_value(format!(
                "ma_choose: index {raw} is not a non-negative integer"
            )));
        }
        let k = raw as usize;
        if k >= nchoices {
            return Err(FerrayError::index_out_of_bounds(k as isize, 0, nchoices));
        }
        out_data.push(ch_d[k][i]);
        out_mask.push(ch_m[k][i] || idx_m[i]);
    }
    let data_arr = Array::from_vec(indices.data().dim().clone(), out_data)?;
    let mask_arr = Array::from_vec(indices.data().dim().clone(), out_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// One first-difference pass along `axis` of an IxDyn masked array.
/// `out[i] = a[i+1] - a[i]`; the result is masked iff either operand is masked.
fn diff_once(
    data: &[f64],
    mask: &[bool],
    shape: &[usize],
    axis: usize,
) -> FerrayResult<(Vec<f64>, Vec<bool>, Vec<usize>)> {
    let ndim = shape.len();
    let strides = row_major_strides(shape);
    let lane = shape[axis];

    let mut out_shape = shape.to_vec();
    // numpy: result is smaller by 1 along axis (after one pass).
    out_shape[axis] = lane.saturating_sub(1);
    let out_size: usize = out_shape.iter().product();

    let out_strides = row_major_strides(&out_shape);
    let mut out_data = vec![0.0f64; out_size];
    let mut out_mask = vec![false; out_size];

    // Walk every output multi-index, compute its two source positions.
    let mut multi = vec![0usize; ndim];
    for _ in 0..out_size {
        let out_flat: usize = multi
            .iter()
            .zip(out_strides.iter())
            .map(|(i, s)| i * s)
            .sum();
        // Source index of the lower operand (a[i]); upper is a[i+1].
        let lo_flat: usize = multi.iter().zip(strides.iter()).map(|(i, s)| i * s).sum();
        let hi_flat = lo_flat + strides[axis];
        out_data[out_flat] = data[hi_flat] - data[lo_flat];
        out_mask[out_flat] = mask[lo_flat] || mask[hi_flat];

        // Increment multi-index (row-major) over out_shape.
        for d in (0..ndim).rev() {
            multi[d] += 1;
            if multi[d] < out_shape[d] {
                break;
            }
            multi[d] = 0;
        }
    }
    Ok((out_data, out_mask, out_shape))
}

/// `numpy.ma.diff(a, n=1, axis=-1)` — the n-th discrete difference along
/// `axis`, preserving the mask (`numpy/ma/core.py:7774`).
///
/// `out[i] = a[i+1] - a[i]`; a result position is masked iff either of the two
/// operands feeding it is masked. For `n > 1` the difference is applied
/// recursively (each pass shrinks `axis` by 1). `n == 0` returns `a`
/// unchanged. `axis` is the usual signed axis (`-1` = last).
///
/// # Errors
/// Returns `FerrayError::InvalidValue` for a 0-D input,
/// `AxisOutOfBounds` for an out-of-range axis, or an internal error.
pub fn ma_diff(
    a: &MaskedArray<f64, IxDyn>,
    n: usize,
    axis: isize,
) -> FerrayResult<MaskedArray<f64, IxDyn>> {
    if a.ndim() == 0 {
        return Err(FerrayError::invalid_value(
            "ma_diff: input must be at least one dimensional",
        ));
    }
    let ndim = a.ndim();
    let axis_u = if axis < 0 {
        let adj = axis + ndim as isize;
        if adj < 0 {
            return Err(FerrayError::axis_out_of_bounds(axis.unsigned_abs(), ndim));
        }
        adj as usize
    } else {
        axis as usize
    };
    if axis_u >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis_u, ndim));
    }
    if n == 0 {
        return Ok(a.clone());
    }

    let mut data: Vec<f64> = a.data().iter().copied().collect();
    let mut mask: Vec<bool> = a.mask().iter().copied().collect();
    let mut shape: Vec<usize> = a.shape().to_vec();

    for _ in 0..n {
        if shape[axis_u] == 0 {
            break;
        }
        let (d, m, s) = diff_once(&data, &mask, &shape, axis_u)?;
        data = d;
        mask = m;
        shape = s;
    }

    let data_arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&shape), data)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(IxDyn::new(&shape), mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// `numpy.ma.ediff1d(ary, to_end=None, to_begin=None)` — first differences of
/// the flattened input with optional prepend/append values
/// (`numpy/ma/extras.py:1229`).
///
/// The interior differences `out[i] = flat[i+1] - flat[i]` carry the mask of
/// their operands. `to_begin` / `to_end` are plain (always-unmasked) values
/// stitched onto the front / back, matching numpy's `hstack` of unmasked
/// scalars.
///
/// # Errors
/// Returns an internal array-construction error only.
pub fn ma_ediff1d<D>(
    ary: &MaskedArray<f64, D>,
    to_begin: Option<&[f64]>,
    to_end: Option<&[f64]>,
) -> FerrayResult<MaskedArray<f64, Ix1>>
where
    D: Dimension,
{
    let flat_d: Vec<f64> = ary.data().iter().copied().collect();
    let flat_m: Vec<bool> = ary.mask().iter().copied().collect();

    let mut data: Vec<f64> = Vec::new();
    let mut mask: Vec<bool> = Vec::new();

    if let Some(begin) = to_begin {
        for &v in begin {
            data.push(v);
            mask.push(false);
        }
    }
    if flat_d.len() >= 2 {
        for i in 0..flat_d.len() - 1 {
            data.push(flat_d[i + 1] - flat_d[i]);
            mask.push(flat_m[i] || flat_m[i + 1]);
        }
    }
    if let Some(end) = to_end {
        for &v in end {
            data.push(v);
            mask.push(false);
        }
    }

    let len = data.len();
    let data_arr = Array::<f64, Ix1>::from_vec(Ix1::new([len]), data)?;
    let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([len]), mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// `numpy.ma.nonzero(a)` — for each dimension, the indices of elements that
/// are non-zero AND unmasked (`MaskedArray.nonzero`, `numpy/ma/core.py:5049`:
/// `filled(self, 0).nonzero()`; masked elements are treated as zero).
///
/// Returns a `Vec` with one `Array1<i64>` per dimension (length `ndim`), the
/// k-th holding the k-th coordinate of every selected element, in row-major
/// (C) order — exactly numpy's tuple-of-index-arrays layout.
///
/// # Errors
/// Returns an internal array-construction error only.
pub fn ma_nonzero<D>(a: &MaskedArray<f64, D>) -> FerrayResult<Vec<Array<i64, Ix1>>>
where
    D: Dimension,
{
    let shape = a.shape().to_vec();
    let ndim = shape.len().max(1);
    let data: Vec<f64> = a.data().iter().copied().collect();
    let mask: Vec<bool> = a.mask().iter().copied().collect();
    let strides = row_major_strides(&shape);

    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); ndim];
    for (flat, (&v, &m)) in data.iter().zip(mask.iter()).enumerate() {
        // Masked -> treated as zero -> never selected.
        if m || v == 0.0 {
            continue;
        }
        if shape.is_empty() {
            // 0-D non-zero scalar: numpy yields a single index ([0],).
            coords[0].push(0);
            continue;
        }
        let mut rem = flat;
        for d in 0..shape.len() {
            let c = rem / strides[d];
            rem %= strides[d];
            coords[d].push(c as i64);
        }
    }

    let mut out = Vec::with_capacity(ndim);
    for axis_coords in coords {
        let len = axis_coords.len();
        out.push(Array::<i64, Ix1>::from_vec(Ix1::new([len]), axis_coords)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;

    fn ma1(data: &[f64], mask: &[bool]) -> MaskedArray<f64, Ix1> {
        let n = data.len();
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data.to_vec()).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask.to_vec()).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    fn mb1(data: &[bool], mask: &[bool]) -> MaskedArray<bool, Ix1> {
        let n = data.len();
        let d = Array::<bool, Ix1>::from_vec(Ix1::new([n]), data.to_vec()).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask.to_vec()).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    fn dyn_ma(data: &[f64], mask: &[bool], shape: &[usize]) -> MaskedArray<f64, IxDyn> {
        let d = Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), data.to_vec()).unwrap();
        let m = Array::<bool, IxDyn>::from_vec(IxDyn::new(shape), mask.to_vec()).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // Expected values: numpy 2.4.5 live oracle (R-CHAR-3).
    // np.ma.where(np.ma.array([1.,0,1],mask=[0,1,0])>0, 10., 20.)
    //   -> data [10,20,10], mask [F,T,F]
    #[test]
    fn where_matches_numpy_scalar_branches() {
        let cond = mb1(&[true, false, true], &[false, true, false]);
        let x = ma1(&[10.0, 10.0, 10.0], &[false, false, false]);
        let y = ma1(&[20.0, 20.0, 20.0], &[false, false, false]);
        let out = ma_where(&cond, &x, &y).unwrap();
        assert_eq!(
            out.data().iter().copied().collect::<Vec<_>>(),
            vec![10.0, 20.0, 10.0]
        );
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    // np.ma.where(cond=[T,T,F](nomask), x=mask[0,1,0], y=mask[1,0,1])
    //   picks x[0](unmask), x[1](MASK), y[2](MASK) -> mask [F,T,T]
    #[test]
    fn where_propagates_source_mask() {
        let cond = mb1(&[true, true, false], &[false, false, false]);
        let x = ma1(&[1.0, 2.0, 3.0], &[false, true, false]);
        let y = ma1(&[4.0, 5.0, 6.0], &[true, false, true]);
        let out = ma_where(&cond, &x, &y).unwrap();
        assert_eq!(
            out.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 6.0]
        );
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, true]
        );
    }

    // np.ma.choose(np.ma.array([0,1,0],mask=[0,1,0]), ([10,20,30],[40,50,60]))
    //   -> data [10,--,30], mask [F,T,F] (masked index -> masked result)
    #[test]
    fn choose_matches_numpy() {
        let idx = ma1(&[0.0, 1.0, 0.0], &[false, true, false]);
        let c0 = ma1(&[10.0, 20.0, 30.0], &[false, false, false]);
        let c1 = ma1(&[40.0, 50.0, 60.0], &[false, false, false]);
        let out = ma_choose(&idx, &[c0, c1]).unwrap();
        assert_eq!(
            out.data().iter().copied().collect::<Vec<_>>(),
            vec![10.0, 20.0, 30.0]
        );
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    // masked choice propagates: choose([0,1,0], (mask[0,0,1],[...]))
    //   -> mask [F,F,T]
    #[test]
    fn choose_propagates_choice_mask() {
        let idx = ma1(&[0.0, 1.0, 0.0], &[false, false, false]);
        let c0 = ma1(&[10.0, 20.0, 30.0], &[false, false, true]);
        let c1 = ma1(&[40.0, 50.0, 60.0], &[false, false, false]);
        let out = ma_choose(&idx, &[c0, c1]).unwrap();
        assert_eq!(
            out.data().iter().copied().collect::<Vec<_>>(),
            vec![10.0, 50.0, 30.0]
        );
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, true]
        );
    }

    // a=[1,2,3,4,7,0,2,3], masked where a<2 -> diff n=1 mask
    //   [T,F,F,F,T,T,F]
    #[test]
    fn diff_n1_matches_numpy() {
        let data = [1.0, 2.0, 3.0, 4.0, 7.0, 0.0, 2.0, 3.0];
        let mask = [true, false, false, false, false, true, false, false];
        let m = dyn_ma(&data, &mask, &[8]);
        let out = ma_diff(&m, 1, -1).unwrap();
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, false, false, false, true, true, false]
        );
        // data at unmasked positions: 1,1,3,_,_,1 -> check [1] and [6].
        let d: Vec<f64> = out.data().iter().copied().collect();
        assert_eq!(d[1], 1.0);
        assert_eq!(d[6], 1.0);
    }

    // diff n=2 mask -> [T,F,F,T,T,T]
    #[test]
    fn diff_n2_matches_numpy() {
        let data = [1.0, 2.0, 3.0, 4.0, 7.0, 0.0, 2.0, 3.0];
        let mask = [true, false, false, false, false, true, false, false];
        let m = dyn_ma(&data, &mask, &[8]);
        let out = ma_diff(&m, 2, -1).unwrap();
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, false, false, true, true, true]
        );
    }

    // 2-D diff along axis=0: a=[[1,3,1,5,10],[0,1,5,6,8]] masked_equal(1)
    //   np.ma.diff(x,axis=0) -> mask [[T,T,T,F,F]]
    #[test]
    fn diff_axis0_2d_matches_numpy() {
        let data = [1.0, 3.0, 1.0, 5.0, 10.0, 0.0, 1.0, 5.0, 6.0, 8.0];
        let mask = [
            true, false, true, false, false, false, true, false, false, false,
        ];
        let m = dyn_ma(&data, &mask, &[2, 5]);
        let out = ma_diff(&m, 1, 0).unwrap();
        assert_eq!(out.shape(), &[1, 5]);
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, false, false]
        );
    }

    // np.ma.ediff1d(np.ma.array([1,2,3,4],mask=[0,1,0,0]))
    //   -> data [--, --, 1], mask [T,T,F]
    #[test]
    fn ediff1d_matches_numpy() {
        let m = ma1(&[1.0, 2.0, 3.0, 4.0], &[false, true, false, false]);
        let out = ma_ediff1d(&m, None, None).unwrap();
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, true, false]
        );
        let d: Vec<f64> = out.data().iter().copied().collect();
        assert_eq!(d[2], 1.0);
    }

    // ediff1d with to_begin=99, to_end=88 -> mask [F,T,T,F,F]
    #[test]
    fn ediff1d_to_begin_end() {
        let m = ma1(&[1.0, 2.0, 3.0, 4.0], &[false, true, false, false]);
        let out = ma_ediff1d(&m, Some(&[99.0]), Some(&[88.0])).unwrap();
        assert_eq!(
            out.data().iter().copied().collect::<Vec<_>>(),
            vec![99.0, 1.0, 1.0, 1.0, 88.0]
        );
        assert_eq!(
            out.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, false, false]
        );
    }

    // np.ma.nonzero(np.ma.array([0,1,0,2],mask=[0,0,1,0])) -> (array([1,3]),)
    #[test]
    fn nonzero_1d_treats_masked_as_zero() {
        let m = ma1(&[0.0, 1.0, 0.0, 2.0], &[false, false, true, false]);
        let out = ma_nonzero(&m).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].iter().copied().collect::<Vec<_>>(), vec![1, 3]);
    }

    // np.ma.nonzero([[0,1],[2,0]] mask [[0,0],[1,0]]) -> (array([0]),array([1]))
    #[test]
    fn nonzero_2d_matches_numpy() {
        let m = dyn_ma(&[0.0, 1.0, 2.0, 0.0], &[false, false, true, false], &[2, 2]);
        let out = ma_nonzero(&m).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].iter().copied().collect::<Vec<_>>(), vec![0]);
        assert_eq!(out[1].iter().copied().collect::<Vec<_>>(), vec![1]);
    }
}
