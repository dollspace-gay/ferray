// ferray-ma: numpy.ma extras
//
// Adds the high-value `numpy.ma` API surface that was missing:
//   - extra reductions: prod, cumsum, cumprod, argmin, argmax, median, ptp,
//     average, anom
//   - constructors: masked_all, masked_values, nomask
//   - manipulation: ma_concatenate, ma_clip, ma_diagonal, ma_repeat,
//     ma_atleast_{1,2,3}d, ma_expand_dims, ma_swapaxes
//   - mask manipulation: harden_mask, soften_mask, mask_or, make_mask,
//     make_mask_none
//   - linalg/set/corr: ma_dot, ma_inner, ma_outer, ma_trace,
//     ma_unique, ma_in1d, ma_isin
//   - functional: ma_apply_along_axis, ma_apply_over_axes, ma_vander
//   - fill-value protocol: default_fill_value, maximum_fill_value,
//     minimum_fill_value, common_fill_value
//   - bitwise/comparison ufuncs: ma_equal, ma_not_equal, ma_less,
//     ma_greater, ma_less_equal, ma_greater_equal, ma_logical_and,
//     ma_logical_or, ma_logical_not, ma_logical_xor
//   - class helpers: is_masked_array (alias isMA), getmaskarray, ids
//   - set ops: ma_unique_masked, ma_intersect1d, ma_union1d, ma_setdiff1d,
//     ma_setxor1d
//   - row/col suppression: ma_compress_rowcols/_rows/_cols, ma_mask_rowcols
//   - masked covariance/correlation: ma_cov, ma_corrcoef (y=None case)
//   - contiguity/run helpers: clump_masked, clump_unmasked,
//     flatnotmasked_edges/_contiguous, notmasked_edges(_axis2),
//     notmasked_contiguous_axis
//   - 2-D masked matmul: ma_dot_2d; multi-axis argsort: argsort_axis
//   - axis median + returned-weights average: ma_median_axis,
//     average_returned
//
// Items intentionally NOT covered (deferred — no REQ-N, genuine gaps):
//   - mvoid (record-void): requires structured-record dtype work in core
//   - mr_ / masked_singleton / masked_print_option: Python-class display
//     state with no Rust analog
//   - polyfit / convolve / correlate: would pull in ferray-polynomial /
//     ferray-stats; users can call those directly on the unmasked subset
//     via .compressed()
//   - ma_cov/ma_corrcoef two-variable-set `y` argument: the binding raises a
//     clear `ValueError` (REQ-29 covers only the `y=None` case)
//
// ## REQ status
//
// All extras REQs SHIPPED — audited, green. Each row's non-test production
// consumer is the matching `#[pyfunction]`/method in `ferray-python/src/ma.rs`,
// registered by `register_ma_module` in `ferray-python/src/lib.rs`. (REQ-18..22
// — `ma_where`/`ma_choose`/`ma_diff`/`ma_ediff1d`/`ma_nonzero` — live in
// `algorithms.rs`, not this file; their evidence is in `.design/ferray-ma.md`.)
//
// | REQ | Status | Evidence (impl in this file unless noted) |
// |-----|--------|------|
// | REQ-23 (`ma_vander`) | SHIPPED | `ma_vander`; consumer `ferray-python/src/ma.rs::vander`. |
// | REQ-24 (`ma_isin`/`ma_in1d`) | SHIPPED | `ma_isin`/`ma_in1d`; consumers `ferray-python/src/ma.rs::isin`/`in1d`. |
// | REQ-25 (set ops) | SHIPPED | `ma_intersect1d`/`ma_union1d`/`ma_setdiff1d`/`ma_setxor1d` (built on `ma_unique_masked`); consumers `ferray-python/src/ma.rs::intersect1d`/`union1d`/`setdiff1d`/`setxor1d` (`numpy/ma/extras.py:1317`/`:1463`/`:1485`/`:1350`). |
// | REQ-26 (`unique` masked slot) | SHIPPED | `ma_unique_masked`; consumed by the REQ-25 set ops (`numpy/ma/extras.py:1267`). |
// | REQ-27 (`compress_rowcols`) | SHIPPED | `ma_compress_rowcols`/`ma_compress_rows`/`ma_compress_cols`; consumers `ferray-python/src/ma.rs::compress_rowcols`/`compress_rows`/`compress_cols` (`numpy/ma/extras.py:920`/`:953`/`:991`). |
// | REQ-28 (`mask_rowcols`) | SHIPPED | `ma_mask_rowcols`; consumer `ferray-python/src/ma.rs::mask_rowcols` (`numpy/ma/extras.py:830`). |
// | REQ-29 (`cov`/`corrcoef`, y=None) | SHIPPED | `ma_cov`/`ma_corrcoef`; consumers `ferray-python/src/ma.rs::cov`/`corrcoef` (`numpy/ma/extras.py:1580`/`:1675`). The two-variable-set `y` arg remains a documented gap. |
// | REQ-30 (`clump_masked`/`clump_unmasked`) | SHIPPED | `clump_masked`/`clump_unmasked` (on `ezclump_true`); consumers `ferray-python/src/ma.rs::clump_masked`/`clump_unmasked` (`numpy/ma/extras.py:2189`/`:2235`). |
// | REQ-31 (`flatnotmasked_*`) | SHIPPED | `flatnotmasked_edges`/`flatnotmasked_contiguous`; consumers `ferray-python/src/ma.rs::flatnotmasked_edges`/`flatnotmasked_contiguous` (`numpy/ma/extras.py:1818`/`:1762`). |
// | REQ-32 (`notmasked_*`) | SHIPPED | `notmasked_edges`/`notmasked_edges_axis2`/`notmasked_contiguous_axis`; consumers `ferray-python/src/ma.rs::notmasked_edges`/`notmasked_contiguous` (`numpy/ma/extras.py:1977`/`:1925`). |
// | REQ-33 (2-D `ma.dot`) | SHIPPED | `MaskedArray<T, Ix2>::ma_dot_2d` (and the flat `ma_dot_flat`); consumer `ferray-python/src/ma.rs::dot` (`numpy/ma/core.py:8214`). |
// | REQ-14 (argsort axis) | SHIPPED | `argsort_axis` (endwith=True lane fill); consumer `ferray-python/src/ma.rs::argsort` (explicit-axis branch). |
// | REQ-38 (`median(axis=)`) | SHIPPED | `ma_median_axis`; consumer the explicit-axis branch of `ferray-python/src/ma.rs::median` (`numpy/ma/extras.py:678`). |
// | REQ-39 (`average(returned=)`) | SHIPPED | `MaskedArray::average_returned` (the plain `average` delegates to it); consumer the `returned` branch of `ferray-python/src/ma.rs::average` (`numpy/ma/extras.py:510`/`:670`). |
//
// Also SHIPPED here (REQ-4 extension, full-array reductions beyond
// `reductions.rs`): `prod`, `cumsum_flat`, `cumprod_flat`, `argmin`, `argmax`,
// `ptp`, `median`, `average`, `anom` — consumed by the matching `PyMaskedArray`
// methods in `ferray-python/src/ma.rs`. The fill-value protocol
// (`default_fill_value_f64`/`_f32`, `maximum_fill_value`, `minimum_fill_value`,
// `common_fill_value`) and the comparison/logical ufuncs (`ma_logical_and`/
// `_or`/`_xor`/`_not`) are likewise re-exported via `ferray-ma/src/lib.rs` and
// consumed by the `ferray.ma` module shims.

use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Ix1, Ix2};
use num_traits::Float;

use crate::MaskedArray;

// ===========================================================================
// Sentinel constants
// ===========================================================================

/// The "no mask" sentinel — equivalent to `numpy.ma.nomask`. Read this and
/// pass it as a mask to indicate "no elements are masked."
pub const NOMASK: bool = false;

// ===========================================================================
// Extra reductions on MaskedArray
// ===========================================================================

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + std::ops::Mul<Output = T> + num_traits::One,
    D: Dimension,
{
    /// Product of unmasked elements. Returns `T::one()` for the all-masked case.
    ///
    /// # Errors
    /// Returns an error only for internal failures.
    pub fn prod(&self) -> FerrayResult<T> {
        let one = num_traits::one::<T>();
        let p = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .fold(one, |acc, (v, _)| acc * *v);
        Ok(p)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + std::ops::Add<Output = T> + num_traits::Zero,
    D: Dimension,
{
    /// Cumulative sum over unmasked elements (mask propagates: each output
    /// position carries the same mask as the input). Masked positions
    /// contribute zero to the running total.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn cumsum_flat(&self) -> FerrayResult<MaskedArray<T, Ix1>> {
        let zero = num_traits::zero::<T>();
        let n = self.size();
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .scan(zero, |acc, (v, m)| {
                if !*m {
                    *acc = *acc + *v;
                }
                Some(*acc)
            })
            .collect();
        let mask: Vec<bool> = self.mask().iter().copied().collect();
        let data_arr = Array::from_vec(Ix1::new([n]), data)?;
        let mask_arr = Array::from_vec(Ix1::new([n]), mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + std::ops::Mul<Output = T> + num_traits::One,
    D: Dimension,
{
    /// Cumulative product over unmasked elements (mask propagates).
    /// Masked positions contribute one to the running product.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn cumprod_flat(&self) -> FerrayResult<MaskedArray<T, Ix1>> {
        let one = num_traits::one::<T>();
        let n = self.size();
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .scan(one, |acc, (v, m)| {
                if !*m {
                    *acc = *acc * *v;
                }
                Some(*acc)
            })
            .collect();
        let mask: Vec<bool> = self.mask().iter().copied().collect();
        let data_arr = Array::from_vec(Ix1::new([n]), data)?;
        let mask_arr = Array::from_vec(Ix1::new([n]), mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    /// Flat index of the minimum unmasked element.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if every element is masked.
    pub fn argmin(&self) -> FerrayResult<usize> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .enumerate()
            .filter(|(_, (_, m))| !**m)
            .min_by(|(_, (a, _)), (_, (b, _))| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| FerrayError::invalid_value("argmin: all elements are masked"))
    }

    /// Flat index of the maximum unmasked element.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if every element is masked.
    pub fn argmax(&self) -> FerrayResult<usize> {
        self.data()
            .iter()
            .zip(self.mask().iter())
            .enumerate()
            .filter(|(_, (_, m))| !**m)
            .max_by(|(_, (a, _)), (_, (b, _))| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .ok_or_else(|| FerrayError::invalid_value("argmax: all elements are masked"))
    }

    /// Peak-to-peak range over unmasked elements (`max - min`).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if every element is masked.
    pub fn ptp(&self) -> FerrayResult<T>
    where
        T: std::ops::Sub<Output = T>,
    {
        let mut iter = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v);
        let first = iter
            .next()
            .ok_or_else(|| FerrayError::invalid_value("ptp: all elements are masked"))?;
        let mut lo = first;
        let mut hi = first;
        for v in iter {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        Ok(hi - lo)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element
        + Copy
        + PartialOrd
        + num_traits::FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>,
    D: Dimension,
{
    /// Median of unmasked elements (interpolated between the two middle
    /// values for an even count).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if every element is masked.
    pub fn median(&self) -> FerrayResult<T> {
        let mut vals: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .filter(|(_, m)| !**m)
            .map(|(v, _)| *v)
            .collect();
        if vals.is_empty() {
            return Err(FerrayError::invalid_value(
                "median: all elements are masked",
            ));
        }
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = vals.len();
        if n % 2 == 1 {
            Ok(vals[n / 2])
        } else {
            let two = T::from_u8(2).unwrap();
            Ok((vals[n / 2 - 1] + vals[n / 2]) / two)
        }
    }
}

/// `numpy.ma.median(a, axis=axis)` (`numpy/ma/extras.py:678`) — per-lane median
/// of the unmasked elements along `axis`, returning an `(ndim - 1)`-D masked
/// array. A lane whose elements are all masked yields a masked output slot
/// (numpy's `median` returns the `masked` value for an all-masked lane).
///
/// Each lane's median is the middle unmasked value (odd count) or the average
/// of the two middle unmasked values (even count), matching the flat
/// [`MaskedArray::median`].
///
/// # Errors
/// Returns `FerrayError::axis_out_of_bounds` if `axis >= a.ndim()`, or an
/// internal construction error.
pub fn ma_median_axis<T>(
    a: &MaskedArray<T, IxDyn>,
    axis: usize,
) -> FerrayResult<MaskedArray<T, IxDyn>>
where
    T: Element
        + Copy
        + PartialOrd
        + num_traits::Zero
        + num_traits::FromPrimitive
        + std::ops::Add<Output = T>
        + std::ops::Div<Output = T>,
{
    let shape = a.shape().to_vec();
    if axis >= shape.len() {
        return Err(FerrayError::axis_out_of_bounds(axis, shape.len()));
    }
    let lane_len = shape[axis];
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let data: Vec<T> = a.data().iter().copied().collect();
    let mask: Vec<bool> = a.mask().iter().copied().collect();

    let mut out_data = Vec::with_capacity(out_size);
    let mut out_mask = Vec::with_capacity(out_size);

    // Iterate over every multi-index of the OUTPUT shape (axes other than
    // `axis`), reconstruct the base offset, then sweep the lane.
    let out_axes: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    let mut counter = vec![0usize; out_axes.len()];
    let two = T::from_u8(2).ok_or_else(|| {
        FerrayError::invalid_value("median: constant 2 not representable in element type")
    })?;
    for _ in 0..out_size {
        let mut base = 0usize;
        for (slot, &ax) in out_axes.iter().enumerate() {
            base += counter[slot] * strides[ax];
        }
        let mut vals: Vec<T> = Vec::with_capacity(lane_len);
        for k in 0..lane_len {
            let off = base + k * strides[axis];
            if !mask[off] {
                vals.push(data[off]);
            }
        }
        if vals.is_empty() {
            out_data.push(<T as num_traits::Zero>::zero());
            out_mask.push(true);
        } else {
            vals.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
            let n = vals.len();
            let med = if n % 2 == 1 {
                vals[n / 2]
            } else {
                (vals[n / 2 - 1] + vals[n / 2]) / two
            };
            out_data.push(med);
            out_mask.push(false);
        }

        // Increment the output multi-index (row-major over out_axes).
        for slot in (0..out_axes.len()).rev() {
            counter[slot] += 1;
            if counter[slot] < out_shape[slot] {
                break;
            }
            counter[slot] = 0;
        }
    }

    let dim = IxDyn::new(&out_shape);
    let data_arr = Array::<T, IxDyn>::from_vec(dim.clone(), out_data)?;
    let mask_arr = Array::<bool, IxDyn>::from_vec(dim, out_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + Float,
    D: Dimension,
{
    /// Weighted average of unmasked elements.
    ///
    /// `weights` must be `Some(Array<T, D>)` of the same shape as `self`,
    /// or `None` (delegates to [`MaskedArray::mean`]).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if weights shape differs, or
    /// `FerrayError::InvalidValue` if the weight sum over unmasked
    /// elements is zero (or every element is masked).
    pub fn average(&self, weights: Option<&Array<T, D>>) -> FerrayResult<T> {
        Ok(self.average_returned(weights)?.0)
    }

    /// Weighted average of unmasked elements together with the sum of the
    /// weights actually applied (the unmasked-weight total).
    ///
    /// Mirrors `numpy.ma.average(..., returned=True)`
    /// (`numpy/ma/extras.py:510`), which returns the tuple
    /// `(average, sum_of_weights)`. For the unweighted case (`weights=None`)
    /// numpy uses each element's implicit weight of one, so the second tuple
    /// element is the count of unmasked elements
    /// (`numpy/ma/extras.py:637` `scl = avg.dtype.type(a.count(axis))`).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if weights shape differs, or
    /// `FerrayError::InvalidValue` if the weight sum over unmasked
    /// elements is zero (or every element is masked).
    pub fn average_returned(&self, weights: Option<&Array<T, D>>) -> FerrayResult<(T, T)> {
        let zero = <T as num_traits::Zero>::zero();
        let Some(w) = weights else {
            // Unweighted: avg = mean, sum_of_weights = count of unmasked.
            let avg = self.mean()?;
            let count = self.mask().iter().filter(|m| !**m).count();
            let scl = <T as num_traits::NumCast>::from(count).ok_or_else(|| {
                FerrayError::invalid_value("average: unmasked count not representable")
            })?;
            return Ok((avg, scl));
        };
        if w.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "average: weights shape {:?} differs from array shape {:?}",
                w.shape(),
                self.shape(),
            )));
        }
        let mut wsum = zero;
        let mut acc = zero;
        for ((v, m), wi) in self.data().iter().zip(self.mask().iter()).zip(w.iter()) {
            if !*m {
                wsum = wsum + *wi;
                acc = acc + *v * *wi;
            }
        }
        if wsum == zero {
            return Err(FerrayError::invalid_value(
                "average: weight sum is zero (or all elements are masked)",
            ));
        }
        Ok((acc / wsum, wsum))
    }

    /// Anomaly array: `self - mean(self)`. Returns a masked array of the
    /// same shape with the same mask.
    ///
    /// Equivalent to `numpy.ma.anom`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` if every element is masked.
    pub fn anom(&self) -> FerrayResult<MaskedArray<T, D>> {
        let m = self.mean()?;
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .map(|(v, masked)| if *masked { *v } else { *v - m })
            .collect();
        let data_arr = Array::from_vec(self.data().dim().clone(), data)?;
        let mask_arr: Array<bool, D> = Array::from_vec(
            self.mask().dim().clone(),
            self.mask().iter().copied().collect(),
        )?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        Ok(out)
    }
}

// ===========================================================================
// Constructors
// ===========================================================================

/// Build a fully-masked array of the given shape, filled with `T::zero()`.
///
/// Equivalent to `numpy.ma.masked_all(shape)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn masked_all<T, D>(shape: D) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
{
    let data = Array::<T, D>::from_elem(shape.clone(), T::zero())?;
    let mask = Array::<bool, D>::from_elem(shape, true)?;
    MaskedArray::new(data, mask)
}

/// Build a masked-all array with the same shape as the reference.
///
/// Equivalent to `numpy.ma.masked_all_like(other)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn masked_all_like<T, U, D>(reference: &Array<U, D>) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    U: Element,
    D: Dimension,
{
    masked_all(reference.dim().clone())
}

/// Mask `arr` where each element approximately equals `value` within `rtol` /
/// `atol` (relative + absolute tolerance, NumPy semantics:
/// `|x - value| <= atol + rtol * |value|`).
///
/// Equivalent to `numpy.ma.masked_values(arr, value, rtol, atol)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn masked_values<T, D>(
    arr: &Array<T, D>,
    value: T,
    rtol: T,
    atol: T,
) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy + Float,
    D: Dimension,
{
    let threshold = atol + rtol * value.abs();
    let mask: Vec<bool> = arr
        .iter()
        .map(|x| (*x - value).abs() <= threshold)
        .collect();
    let mask_arr = Array::from_vec(arr.dim().clone(), mask)?;
    let data_arr = arr.clone();
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
    out.set_fill_value(value);
    Ok(out)
}

// ===========================================================================
// Mask manipulation: harden / soften / mask_or / make_mask / make_mask_none
// ===========================================================================

// Note: `harden_mask` / `soften_mask` already live on `MaskedArray` via
// `mask_ops.rs`. They're re-exported from this crate's root for parity
// with `numpy.ma.MaskedArray.harden_mask` / `soften_mask`.

/// Element-wise OR of two masks. Both must share the same shape.
///
/// Equivalent to `numpy.ma.mask_or(m1, m2)`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn mask_or<D: Dimension>(
    a: &Array<bool, D>,
    b: &Array<bool, D>,
) -> FerrayResult<Array<bool, D>> {
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "mask_or: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<bool> = a.iter().zip(b.iter()).map(|(x, y)| *x || *y).collect();
    Array::from_vec(a.dim().clone(), data)
}

/// Construct a boolean mask from a slice — convenience wrapper.
///
/// Equivalent to `numpy.ma.make_mask(values)` for the simple case of an
/// existing bool array.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn make_mask<D: Dimension>(values: &[bool], shape: D) -> FerrayResult<Array<bool, D>> {
    Array::from_vec(shape, values.to_vec())
}

/// Build an all-`false` mask of the given shape.
///
/// Equivalent to `numpy.ma.make_mask_none(shape)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn make_mask_none<D: Dimension>(shape: D) -> FerrayResult<Array<bool, D>> {
    Array::from_elem(shape, false)
}

// ===========================================================================
// Manipulation: concatenate / clip / repeat / atleast_*d / expand_dims /
// swapaxes / diag / diagonal
// ===========================================================================

/// Concatenate two masked arrays along an axis. Both must have the same
/// dimensionality. Mask is concatenated alongside data.
///
/// Equivalent to `numpy.ma.concatenate` for two-arg case.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes are incompatible.
pub fn ma_concatenate<T>(
    a: &MaskedArray<T, IxDyn>,
    b: &MaskedArray<T, IxDyn>,
    axis: usize,
) -> FerrayResult<MaskedArray<T, IxDyn>>
where
    T: Element + Copy,
{
    let cat_data =
        ferray_core::manipulation::concatenate(&[a.data().clone(), b.data().clone()], axis)?;
    let cat_mask =
        ferray_core::manipulation::concatenate(&[a.mask().clone(), b.mask().clone()], axis)?;
    MaskedArray::new(cat_data, cat_mask)
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    /// Clip each unmasked element to `[a_min, a_max]`. Masked elements
    /// pass through unchanged. Equivalent to `numpy.ma.clip`.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn clip(&self, a_min: T, a_max: T) -> FerrayResult<MaskedArray<T, D>> {
        let data: Vec<T> = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .map(|(v, m)| {
                if *m {
                    *v
                } else if *v < a_min {
                    a_min
                } else if *v > a_max {
                    a_max
                } else {
                    *v
                }
            })
            .collect();
        let data_arr = Array::from_vec(self.data().dim().clone(), data)?;
        let mask_arr: Array<bool, D> = Array::from_vec(
            self.mask().dim().clone(),
            self.mask().iter().copied().collect(),
        )?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        Ok(out)
    }
}

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Repeat each unmasked element `repeats` times along the flat axis.
    /// Always returns a 1-D masked array.
    ///
    /// Equivalent to `numpy.ma.repeat(arr, repeats)` with default flat axis.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn repeat(&self, repeats: usize) -> FerrayResult<MaskedArray<T, Ix1>> {
        let n = self.size() * repeats;
        let mut data = Vec::with_capacity(n);
        let mut mask = Vec::with_capacity(n);
        for (v, m) in self.data().iter().zip(self.mask().iter()) {
            for _ in 0..repeats {
                data.push(*v);
                mask.push(*m);
            }
        }
        let data_arr = Array::from_vec(Ix1::new([n]), data)?;
        let mask_arr = Array::from_vec(Ix1::new([n]), mask)?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        Ok(out)
    }

    /// Promote to at least 1-D. Scalar (0-D) becomes shape `(1,)`.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn atleast_1d(&self) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let shape = self.shape();
        let new_shape: Vec<usize> = if shape.is_empty() {
            vec![1]
        } else {
            shape.to_vec()
        };
        let data: Vec<T> = self.data().iter().copied().collect();
        let mask: Vec<bool> = self.mask().iter().copied().collect();
        let data_arr = Array::from_vec(IxDyn::new(&new_shape), data)?;
        let mask_arr = Array::from_vec(IxDyn::new(&new_shape), mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }

    /// Promote to at least 2-D. 0-D → (1,1); 1-D (N,) → (1, N).
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn atleast_2d(&self) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let shape = self.shape();
        let new_shape: Vec<usize> = match shape.len() {
            0 => vec![1, 1],
            1 => vec![1, shape[0]],
            _ => shape.to_vec(),
        };
        let data: Vec<T> = self.data().iter().copied().collect();
        let mask: Vec<bool> = self.mask().iter().copied().collect();
        let data_arr = Array::from_vec(IxDyn::new(&new_shape), data)?;
        let mask_arr = Array::from_vec(IxDyn::new(&new_shape), mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }

    /// Promote to at least 3-D. See [`atleast_3d`](crate::extras::MaskedArray::atleast_3d) for the rank ladder.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn atleast_3d(&self) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let shape = self.shape();
        let new_shape: Vec<usize> = match shape.len() {
            0 => vec![1, 1, 1],
            1 => vec![1, shape[0], 1],
            2 => vec![shape[0], shape[1], 1],
            _ => shape.to_vec(),
        };
        let data: Vec<T> = self.data().iter().copied().collect();
        let mask: Vec<bool> = self.mask().iter().copied().collect();
        let data_arr = Array::from_vec(IxDyn::new(&new_shape), data)?;
        let mask_arr = Array::from_vec(IxDyn::new(&new_shape), mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }

    /// Insert an axis of length 1 at the given position.
    ///
    /// # Errors
    /// Returns `FerrayError::AxisOutOfBounds` if `axis > ndim`.
    pub fn expand_dims(&self, axis: usize) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let data_exp = ferray_core::manipulation::expand_dims(self.data(), axis)?;
        let mask_exp = ferray_core::manipulation::expand_dims(self.mask(), axis)?;
        let mut out = MaskedArray::new(data_exp, mask_exp)?;
        out.set_fill_value(self.fill_value());
        Ok(out)
    }
}

impl<T> MaskedArray<T, Ix2>
where
    T: Element + Copy + num_traits::Zero,
{
    /// Extract the `k`-th diagonal of a 2-D masked array as a 1-D masked array.
    ///
    /// Equivalent to `numpy.ma.diagonal` for the 2-D case.
    ///
    /// # Errors
    /// Returns an error if internal array construction fails.
    pub fn diagonal(&self, k: isize) -> FerrayResult<MaskedArray<T, Ix1>> {
        let shape = self.shape();
        let (rows, cols) = (shape[0], shape[1]);
        let (start_r, start_c) = if k >= 0 {
            (0usize, k as usize)
        } else {
            (-k as usize, 0usize)
        };
        let mut data = Vec::new();
        let mut mask = Vec::new();
        let data_slice = self.data().as_slice();
        let mask_slice = self.mask().as_slice();
        let mut r = start_r;
        let mut c = start_c;
        while r < rows && c < cols {
            let flat = r * cols + c;
            // Use as_slice when available (C-contiguous), else fall back to iter().nth.
            if let (Some(ds), Some(ms)) = (data_slice, mask_slice) {
                data.push(ds[flat]);
                mask.push(ms[flat]);
            } else {
                data.push(*self.data().iter().nth(flat).expect("flat index in bounds"));
                mask.push(*self.mask().iter().nth(flat).expect("flat index in bounds"));
            }
            r += 1;
            c += 1;
        }
        let n = data.len();
        let data_arr = Array::from_vec(Ix1::new([n]), data)?;
        let mask_arr = Array::from_vec(Ix1::new([n]), mask)?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        Ok(out)
    }
}

// ===========================================================================
// Linalg-lite (mask-aware via fill-zero)
// ===========================================================================

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
    D: Dimension,
{
    /// Inner product of two masked arrays as flat 1-D vectors. Masked
    /// positions on either side contribute zero.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if total element counts differ.
    pub fn ma_dot_flat(&self, other: &MaskedArray<T, D>) -> FerrayResult<T> {
        if self.size() != other.size() {
            return Err(FerrayError::shape_mismatch(format!(
                "ma_dot_flat: lhs.size()={} != rhs.size()={}",
                self.size(),
                other.size(),
            )));
        }
        let zero = <T as num_traits::Zero>::zero();
        let s = self
            .data()
            .iter()
            .zip(self.mask().iter())
            .zip(other.data().iter().zip(other.mask().iter()))
            .fold(
                zero,
                |acc, ((a, ma), (b, mb))| {
                    if *ma || *mb { acc } else { acc + *a * *b }
                },
            );
        Ok(s)
    }
}

impl<T> MaskedArray<T, Ix2>
where
    T: Element + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
{
    /// Trace (sum of diagonal) of a 2-D masked array. Masked diagonal
    /// elements contribute zero.
    pub fn trace(&self, k: isize) -> FerrayResult<T> {
        let diag = self.diagonal(k)?;
        let zero = <T as num_traits::Zero>::zero();
        let s = diag
            .data()
            .iter()
            .zip(diag.mask().iter())
            .filter(|(_, m)| !**m)
            .fold(zero, |acc, (v, _)| acc + *v);
        Ok(s)
    }
}

// ===========================================================================
// Set ops on the unmasked subset
// ===========================================================================

/// Sorted unique values of `ma` as a masked array, including one trailing
/// masked slot iff the input contains any masked value.
///
/// Equivalent to `numpy.ma.unique(ma)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_unique<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    ma_unique_masked(ma)
}

/// Per-element membership test against unmasked `test_values`, matching
/// `numpy.ma.isin` / `numpy.ma.in1d` for Ferray's supported signature
/// (`numpy/ma/extras.py:1434` / `:1387`).
///
/// NumPy does not implement this as a simple value lookup. It first applies
/// `ma.unique(ar1, return_inverse=True)`, concatenates that with the unique
/// test values, stable-sorts the combined masked array, compares adjacent
/// sorted entries, then maps the flags back through `return_inverse`. This
/// preserves a few observable masked-array details: masked input values map to
/// an unmasked `False`, and when the largest unmatched unmasked value sits
/// immediately before the collapsed masked-unique sentinel, the output slot is
/// masked with an underlying `False` payload.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_isin<T, D>(
    ma: &MaskedArray<T, D>,
    test_values: &[T],
) -> FerrayResult<MaskedArray<bool, D>>
where
    T: Element + Copy + PartialEq + PartialOrd,
    D: Dimension,
{
    #[derive(Clone, Copy)]
    struct Entry<T> {
        value: Option<T>,
        ar1_unique_index: Option<usize>,
    }

    let mut unique_ar1 = Vec::<T>::new();
    let mut inverse = Vec::<usize>::with_capacity(ma.size());
    let mut masked_unique_index = None;

    for (value, masked) in ma.data().iter().zip(ma.mask().iter()) {
        if *masked {
            let index = *masked_unique_index.get_or_insert_with(|| unique_ar1.len());
            inverse.push(index);
        } else if let Some(index) = unique_ar1.iter().position(|candidate| candidate == value) {
            inverse.push(index);
        } else {
            unique_ar1.push(*value);
            inverse.push(unique_ar1.len() - 1);
        }
    }
    unique_ar1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_ar1.dedup_by(|a, b| *a == *b);

    if masked_unique_index.is_some() {
        let masked_index = unique_ar1.len();
        inverse.clear();
        for (value, masked) in ma.data().iter().zip(ma.mask().iter()) {
            if *masked {
                inverse.push(masked_index);
            } else {
                inverse.push(
                    unique_ar1
                        .iter()
                        .position(|candidate| candidate == value)
                        .expect("unmasked value exists in unique_ar1"),
                );
            }
        }
    } else {
        inverse.clear();
        for value in ma.data().iter() {
            inverse.push(
                unique_ar1
                    .iter()
                    .position(|candidate| candidate == value)
                    .expect("value exists in unique_ar1"),
            );
        }
    }

    let mut unique_test_values = test_values.to_vec();
    unique_test_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    unique_test_values.dedup_by(|a, b| *a == *b);

    let ar1_len = unique_ar1.len() + usize::from(masked_unique_index.is_some());
    let mut combined = Vec::<Entry<T>>::with_capacity(ar1_len + unique_test_values.len());
    combined.extend(unique_ar1.iter().enumerate().map(|(index, value)| Entry {
        value: Some(*value),
        ar1_unique_index: Some(index),
    }));
    if masked_unique_index.is_some() {
        combined.push(Entry {
            value: None,
            ar1_unique_index: Some(unique_ar1.len()),
        });
    }
    combined.extend(unique_test_values.iter().map(|value| Entry {
        value: Some(*value),
        ar1_unique_index: None,
    }));
    combined.sort_by(|left, right| match (left.value, right.value) {
        (Some(a), Some(b)) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    let mut unique_data = vec![false; ar1_len];
    let mut unique_mask = vec![false; ar1_len];
    for (position, entry) in combined.iter().enumerate() {
        let Some(index) = entry.ar1_unique_index else {
            continue;
        };
        let Some(value) = entry.value else {
            unique_data[index] = false;
            unique_mask[index] = false;
            continue;
        };
        if let Some(next) = combined.get(position + 1) {
            if let Some(next_value) = next.value {
                unique_data[index] = value == next_value;
                unique_mask[index] = false;
            } else {
                unique_data[index] = false;
                unique_mask[index] = true;
            }
        } else {
            unique_data[index] = false;
            unique_mask[index] = false;
        }
    }

    let data = inverse.iter().map(|&index| unique_data[index]).collect();
    let mask = inverse.iter().map(|&index| unique_mask[index]).collect();
    let data_arr = Array::from_vec(ma.data().dim().clone(), data)?;
    let mask_arr: Array<bool, D> = Array::from_vec(ma.mask().dim().clone(), mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// 1-D variant of [`ma_isin`].
///
/// Equivalent to `numpy.ma.in1d`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_in1d<T>(
    ma: &MaskedArray<T, Ix1>,
    test_values: &[T],
) -> FerrayResult<MaskedArray<bool, Ix1>>
where
    T: Element + Copy + PartialEq + PartialOrd,
{
    ma_isin(ma, test_values)
}

// ===========================================================================
// Masked set operations (numpy.ma.unique / intersect1d / union1d /
// setdiff1d / setxor1d). All operate on the *unmasked* unique values, with a
// single trailing `masked` element iff any input element was masked — exactly
// `numpy.ma.unique`'s observable contract (numpy/ma/extras.py:1267): masked
// values "are considered the same element (masked)" and collapse to one slot.
// ===========================================================================

/// Sorted unique unmasked values of `ma` plus one trailing **masked** slot iff
/// the input contained any masked element — i.e. `numpy.ma.unique(ma)`
/// (numpy/ma/extras.py:1267).
///
/// The returned [`MaskedArray<T, Ix1>`] carries a mask whose only `true` entry
/// (if present) is the final element. The data value behind that masked slot is
/// not observable; the first masked input value is reused as a placeholder.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_unique_masked<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let mut vals: Vec<T> = ma
        .data()
        .iter()
        .zip(ma.mask().iter())
        .filter(|(_, m)| !**m)
        .map(|(v, _)| *v)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a).partial_cmp(&*b) == Some(std::cmp::Ordering::Equal));

    let mut mask = vec![false; vals.len()];
    // A single trailing masked slot iff any input was masked.
    if let Some(pos) = ma.mask().iter().position(|&m| m)
        && let Some(placeholder) = ma.data().iter().nth(pos)
    {
        vals.push(*placeholder);
        mask.push(true);
    }
    let n = vals.len();
    let data_arr = Array::<T, Ix1>::from_vec(Ix1::new([n]), vals)?;
    let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
    out.set_fill_value(ma.fill_value());
    Ok(out)
}

/// Build a `MaskedArray<T, Ix1>` from explicit `(value, is_masked)` pairs.
fn build_ma_ix1<T>(pairs: Vec<(T, bool)>) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy,
{
    let n = pairs.len();
    let data: Vec<T> = pairs.iter().map(|(v, _)| *v).collect();
    let mask: Vec<bool> = pairs.iter().map(|(_, m)| *m).collect();
    let data_arr = Array::<T, Ix1>::from_vec(Ix1::new([n]), data)?;
    let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// Collect the unmasked unique values and the "any masked" flag of a
/// `MaskedArray` (the two ingredients every masked set op consumes).
fn unique_parts<T, D>(ma: &MaskedArray<T, D>) -> (Vec<T>, bool)
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let mut vals: Vec<T> = ma
        .data()
        .iter()
        .zip(ma.mask().iter())
        .filter(|(_, m)| !**m)
        .map(|(v, _)| *v)
        .collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a).partial_cmp(&*b) == Some(std::cmp::Ordering::Equal));
    let any_masked = ma.mask().iter().any(|&m| m);
    (vals, any_masked)
}

/// `numpy.ma.intersect1d(ar1, ar2)` — sorted unique values common to both
/// (numpy/ma/extras.py:1317). Masked values are considered equal to each other,
/// so a masked slot appears in the result iff **both** inputs were masked.
///
/// numpy concatenates `unique(ar1)` and `unique(ar2)`, sorts, and keeps an
/// element iff it equals its successor — i.e. it appears in both. With each
/// input contributing at most one masked slot, the masked slot survives iff
/// both contributed one.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_intersect1d<T, D>(
    ar1: &MaskedArray<T, D>,
    ar2: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let (v1, m1) = unique_parts(ar1);
    let (v2, m2) = unique_parts(ar2);
    // Intersection of two sorted unique unmasked sequences (merge walk).
    let mut out: Vec<(T, bool)> = Vec::new();
    let (mut i, mut j) = (0usize, 0usize);
    while i < v1.len() && j < v2.len() {
        match v1[i].partial_cmp(&v2[j]) {
            Some(std::cmp::Ordering::Equal) => {
                out.push((v1[i], false));
                i += 1;
                j += 1;
            }
            Some(std::cmp::Ordering::Less) => i += 1,
            Some(std::cmp::Ordering::Greater) => j += 1,
            None => {
                // NaN: skip the left element (NaN never equals anything).
                i += 1;
            }
        }
    }
    // Masked slot is common iff both inputs were masked.
    if m1 && m2 {
        let placeholder = first_masked_value(ar1).or_else(|| first_masked_value(ar2));
        if let Some(p) = placeholder {
            out.push((p, true));
        }
    }
    build_ma_ix1(out)
}

/// `numpy.ma.union1d(ar1, ar2)` — sorted unique union
/// (numpy/ma/extras.py:1463): `unique(concatenate((ar1, ar2)))`. A trailing
/// masked slot appears iff **either** input was masked.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_union1d<T, D>(
    ar1: &MaskedArray<T, D>,
    ar2: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let (v1, m1) = unique_parts(ar1);
    let (v2, m2) = unique_parts(ar2);
    let mut vals: Vec<T> = v1;
    vals.extend(v2);
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.dedup_by(|a, b| (*a).partial_cmp(&*b) == Some(std::cmp::Ordering::Equal));
    let mut out: Vec<(T, bool)> = vals.into_iter().map(|v| (v, false)).collect();
    if m1 || m2 {
        let placeholder = first_masked_value(ar1).or_else(|| first_masked_value(ar2));
        if let Some(p) = placeholder {
            out.push((p, true));
        }
    }
    build_ma_ix1(out)
}

/// `numpy.ma.setdiff1d(ar1, ar2)` — sorted unique values in `ar1` not in `ar2`
/// (numpy/ma/extras.py:1485): `unique(ar1)[in1d(unique(ar1), unique(ar2),
/// invert=True)]`. The masked slot of `ar1` survives iff `ar2` was **not**
/// masked (a masked `ar2` element removes the masked `ar1` slot, since masked
/// values are equal).
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_setdiff1d<T, D>(
    ar1: &MaskedArray<T, D>,
    ar2: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let (v1, m1) = unique_parts(ar1);
    let (v2, _m2) = unique_parts(ar2);
    let m2_masked = ar2.mask().iter().any(|&m| m);
    let mut out: Vec<(T, bool)> = Vec::new();
    for v in v1 {
        let present = v2
            .iter()
            .any(|w| v.partial_cmp(w) == Some(std::cmp::Ordering::Equal));
        if !present {
            out.push((v, false));
        }
    }
    if m1
        && !m2_masked
        && let Some(p) = first_masked_value(ar1)
    {
        out.push((p, true));
    }
    build_ma_ix1(out)
}

/// `numpy.ma.setxor1d(ar1, ar2)` — symmetric difference of the unique values
/// (numpy/ma/extras.py:1350): elements present in exactly one input. The masked
/// slot survives iff exactly one input was masked.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_setxor1d<T, D>(
    ar1: &MaskedArray<T, D>,
    ar2: &MaskedArray<T, D>,
) -> FerrayResult<MaskedArray<T, Ix1>>
where
    T: Element + Copy + PartialOrd,
    D: Dimension,
{
    let (v1, m1) = unique_parts(ar1);
    let (v2, m2) = unique_parts(ar2);
    let mut out: Vec<(T, bool)> = Vec::new();
    // Elements in v1 not in v2.
    for v in &v1 {
        if !v2
            .iter()
            .any(|w| v.partial_cmp(w) == Some(std::cmp::Ordering::Equal))
        {
            out.push((*v, false));
        }
    }
    // Elements in v2 not in v1.
    for v in &v2 {
        if !v1
            .iter()
            .any(|w| v.partial_cmp(w) == Some(std::cmp::Ordering::Equal))
        {
            out.push((*v, false));
        }
    }
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Masked slot present in exactly one input -> survives the xor.
    if m1 ^ m2 {
        let placeholder = first_masked_value(ar1).or_else(|| first_masked_value(ar2));
        if let Some(p) = placeholder {
            out.push((p, true));
        }
    }
    build_ma_ix1(out)
}

/// The first underlying value at a masked position, if any.
fn first_masked_value<T, D>(ma: &MaskedArray<T, D>) -> Option<T>
where
    T: Element + Copy,
    D: Dimension,
{
    ma.mask()
        .iter()
        .position(|&m| m)
        .and_then(|pos| ma.data().iter().nth(pos).copied())
}

// ===========================================================================
// Row/column suppression and masking (numpy.ma.compress_rowcols /
// compress_rows / compress_cols / mask_rowcols). 2-D only — numpy raises
// NotImplementedError otherwise (numpy/ma/extras.py:920 / :830).
// ===========================================================================

/// `numpy.ma.compress_rowcols(x, axis)` — suppress whole rows and/or columns of
/// a 2-D masked array that contain any masked value
/// (numpy/ma/extras.py:920, via `compress_nd`).
///
/// - `axis = None` → drop both rows and columns containing a masked value.
/// - `axis = Some(0)` → drop only rows.
/// - `axis = Some(1)` → drop only columns.
///
/// Returns a plain (unmasked) [`Array<T, Ix2>`] of the surviving data.
///
/// # Errors
/// Returns `FerrayError::invalid_value` if the input is not 2-D (numpy raises
/// `NotImplementedError`); or if internal array construction fails.
pub fn ma_compress_rowcols<T>(
    ma: &MaskedArray<T, Ix2>,
    axis: Option<usize>,
) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Copy,
{
    let shape = ma.data().shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let data: Vec<T> = ma.data().iter().copied().collect();
    let mask: Vec<bool> = ma.mask().iter().copied().collect();

    let drop_rows = matches!(axis, None | Some(0));
    let drop_cols = matches!(axis, None | Some(1));

    // Rows/cols to keep.
    let mut keep_row = vec![true; nrows];
    let mut keep_col = vec![true; ncols];
    if drop_rows {
        for (r, kr) in keep_row.iter_mut().enumerate() {
            *kr = !(0..ncols).any(|c| mask[r * ncols + c]);
        }
    }
    if drop_cols {
        for (c, kc) in keep_col.iter_mut().enumerate() {
            *kc = !(0..nrows).any(|r| mask[r * ncols + c]);
        }
    }

    let kept_rows: Vec<usize> = (0..nrows).filter(|&r| keep_row[r]).collect();
    let kept_cols: Vec<usize> = (0..ncols).filter(|&c| keep_col[c]).collect();

    let mut out: Vec<T> = Vec::with_capacity(kept_rows.len() * kept_cols.len());
    for &r in &kept_rows {
        for &c in &kept_cols {
            out.push(data[r * ncols + c]);
        }
    }
    Array::<T, Ix2>::from_vec(Ix2::new([kept_rows.len(), kept_cols.len()]), out)
}

/// `numpy.ma.compress_rows(a)` — suppress whole rows containing masked values;
/// equivalent to `compress_rowcols(a, 0)` (numpy/ma/extras.py:953).
///
/// # Errors
/// Propagates [`ma_compress_rowcols`] errors.
pub fn ma_compress_rows<T>(ma: &MaskedArray<T, Ix2>) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Copy,
{
    ma_compress_rowcols(ma, Some(0))
}

/// `numpy.ma.compress_cols(a)` — suppress whole columns containing masked
/// values; equivalent to `compress_rowcols(a, 1)` (numpy/ma/extras.py:991).
///
/// # Errors
/// Propagates [`ma_compress_rowcols`] errors.
pub fn ma_compress_cols<T>(ma: &MaskedArray<T, Ix2>) -> FerrayResult<Array<T, Ix2>>
where
    T: Element + Copy,
{
    ma_compress_rowcols(ma, Some(1))
}

/// `numpy.ma.mask_rowcols(a, axis)` — mask whole rows and/or columns of a 2-D
/// masked array that contain any masked value (numpy/ma/extras.py:830).
///
/// - `axis = None` → mask both rows and columns containing a masked value.
/// - `axis = Some(0)` → mask only rows.
/// - `axis = Some(1)` → mask only columns.
///
/// Returns a [`MaskedArray<T, Ix2>`] with the same data and an expanded mask.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_mask_rowcols<T>(
    ma: &MaskedArray<T, Ix2>,
    axis: Option<usize>,
) -> FerrayResult<MaskedArray<T, Ix2>>
where
    T: Element + Copy,
{
    let shape = ma.data().shape();
    let (nrows, ncols) = (shape[0], shape[1]);
    let data: Vec<T> = ma.data().iter().copied().collect();
    let mask: Vec<bool> = ma.mask().iter().copied().collect();

    let mask_rows = matches!(axis, None | Some(0));
    let mask_cols = matches!(axis, None | Some(1));

    let row_has_mask: Vec<bool> = (0..nrows)
        .map(|r| (0..ncols).any(|c| mask[r * ncols + c]))
        .collect();
    let col_has_mask: Vec<bool> = (0..ncols)
        .map(|c| (0..nrows).any(|r| mask[r * ncols + c]))
        .collect();

    let mut new_mask = mask.clone();
    for r in 0..nrows {
        for c in 0..ncols {
            if (mask_rows && row_has_mask[r]) || (mask_cols && col_has_mask[c]) {
                new_mask[r * ncols + c] = true;
            }
        }
    }

    let data_arr = Array::<T, Ix2>::from_vec(Ix2::new([nrows, ncols]), data)?;
    let mask_arr = Array::<bool, Ix2>::from_vec(Ix2::new([nrows, ncols]), new_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

// ===========================================================================
// Masked covariance / correlation (numpy.ma.cov / corrcoef). Common case
// y=None, computed over the unmasked pairs (numpy/ma/extras.py:1519
// `_covhelper`). Each row is centered by its own masked mean; the pairwise
// normalisation `fact[i,j]` is the count of jointly-unmasked observations,
// and an entry is masked where `fact <= 0`.
// ===========================================================================

/// `numpy.ma.cov(x, rowvar, bias, ddof)` for the `y=None` case
/// (numpy/ma/extras.py:1547). Returns the masked covariance matrix computed
/// over the unmasked observation pairs.
///
/// Each variable (row when `rowvar`, column otherwise) is centered by the mean
/// of its own unmasked observations. The `[i,j]` entry sums the product of the
/// centered values over the columns where **both** `i` and `j` are unmasked,
/// divided by `fact[i,j] = (#jointly-unmasked) - ddof`. When `ddof` is `None`,
/// it defaults to `0` if `bias`, else `1` (numpy/ma/extras.py:1648). An entry
/// is masked iff `fact[i,j] <= 0`.
///
/// # Errors
/// Returns `FerrayError::invalid_value` if the input is not 1-D or 2-D; or if
/// internal array construction fails.
pub fn ma_cov<D>(
    x: &MaskedArray<f64, D>,
    rowvar: bool,
    bias: bool,
    ddof: Option<usize>,
) -> FerrayResult<MaskedArray<f64, Ix2>>
where
    D: Dimension,
{
    let ddof_val = ddof.unwrap_or(if bias { 0 } else { 1 }) as f64;

    // Build the (nvars x nobs) data + not-mask matrices (rows = variables).
    let ndim = x.ndim();
    if ndim > 2 {
        return Err(FerrayError::invalid_value(
            "ma.cov requires a 1-D or 2-D masked array",
        ));
    }
    let shape = x.data().shape();
    let flat_data: Vec<f64> = x.data().iter().copied().collect();
    let flat_mask: Vec<bool> = x.mask().iter().copied().collect();

    // Build a row-major (nvars x nobs) data + mask matrix. numpy promotes 1-D
    // input to a single row (ndmin=2); for a single row, rowvar is forced True
    // (numpy/ma/extras.py: "if x.shape[0] == 1: rowvar = True").
    let (nvars, nobs, mat_data, mat_mask): (usize, usize, Vec<f64>, Vec<bool>) = if ndim <= 1 {
        let n = flat_data.len();
        (1, n, flat_data, flat_mask)
    } else {
        let (r, c) = (shape[0], shape[1]);
        let eff_rowvar = rowvar || r == 1;
        if eff_rowvar {
            (r, c, flat_data, flat_mask)
        } else {
            // Transpose so variables become rows.
            let mut td = vec![0.0f64; r * c];
            let mut tm = vec![false; r * c];
            for i in 0..c {
                for k in 0..r {
                    td[i * r + k] = flat_data[k * c + i];
                    tm[i * r + k] = flat_mask[k * c + i];
                }
            }
            (c, r, td, tm)
        }
    };

    // Center each variable by the mean of its own unmasked observations.
    let mut centered = vec![0.0f64; nvars * nobs];
    let mut notmask = vec![0.0f64; nvars * nobs];
    for i in 0..nvars {
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for k in 0..nobs {
            let idx = i * nobs + k;
            if !mat_mask[idx] {
                sum += mat_data[idx];
                cnt += 1.0;
                notmask[idx] = 1.0;
            }
        }
        let mean = if cnt > 0.0 { sum / cnt } else { 0.0 };
        for k in 0..nobs {
            let idx = i * nobs + k;
            // filled(x, 0): masked positions contribute 0 to the dot product.
            centered[idx] = if mat_mask[idx] {
                0.0
            } else {
                mat_data[idx] - mean
            };
        }
    }

    // cov[i,j] = (centered_i . centered_j) / (notmask_i . notmask_j - ddof),
    // masked where fact <= 0.
    let mut cov_data = vec![0.0f64; nvars * nvars];
    let mut cov_mask = vec![false; nvars * nvars];
    for i in 0..nvars {
        for j in i..nvars {
            let mut dot = 0.0;
            let mut fact = 0.0;
            for k in 0..nobs {
                dot += centered[i * nobs + k] * centered[j * nobs + k];
                fact += notmask[i * nobs + k] * notmask[j * nobs + k];
            }
            fact -= ddof_val;
            let (val, masked) = if fact <= 0.0 {
                (0.0, true)
            } else {
                (dot / fact, false)
            };
            cov_data[i * nvars + j] = val;
            cov_data[j * nvars + i] = val;
            cov_mask[i * nvars + j] = masked;
            cov_mask[j * nvars + i] = masked;
        }
    }

    let data_arr = Array::<f64, Ix2>::from_vec(Ix2::new([nvars, nvars]), cov_data)?;
    let mask_arr = Array::<bool, Ix2>::from_vec(Ix2::new([nvars, nvars]), cov_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// `numpy.ma.corrcoef(x, rowvar)` for the `y=None` case
/// (numpy/ma/extras.py:1672). The covariance matrix (computed with the default
/// `bias=False`, `ddof=None` normalisation) divided by the outer product of the
/// per-variable standard deviations `sqrt(diag(cov))`.
///
/// A `corr[i,j]` is masked iff `cov[i,j]` is masked or either `std[i]`/`std[j]`
/// is masked (numpy propagates the masked covariance through the division).
///
/// # Errors
/// Returns an error if [`ma_cov`] fails or internal array construction fails.
pub fn ma_corrcoef<D>(x: &MaskedArray<f64, D>, rowvar: bool) -> FerrayResult<MaskedArray<f64, Ix2>>
where
    D: Dimension,
{
    let cov = ma_cov(x, rowvar, false, None)?;
    let n = cov.data().shape()[0];
    let cov_data: Vec<f64> = cov.data().iter().copied().collect();
    let cov_mask: Vec<bool> = cov.mask().iter().copied().collect();

    // Per-variable std = sqrt(diag(cov)); masked iff the diagonal is masked.
    let mut std = vec![0.0f64; n];
    let mut std_masked = vec![false; n];
    for i in 0..n {
        std_masked[i] = cov_mask[i * n + i];
        std[i] = cov_data[i * n + i].sqrt();
    }

    let mut corr_data = vec![0.0f64; n * n];
    let mut corr_mask = vec![false; n * n];
    for i in 0..n {
        for j in 0..n {
            let d = std[i] * std[j];
            let masked = cov_mask[i * n + j] || std_masked[i] || std_masked[j] || d == 0.0;
            if masked {
                corr_mask[i * n + j] = true;
                corr_data[i * n + j] = 0.0;
            } else {
                // Clamp to [-1, 1] for numerical stability (matches the
                // ferray-stats corrcoef convention).
                let v = cov_data[i * n + j] / d;
                corr_data[i * n + j] = v.clamp(-1.0, 1.0);
            }
        }
    }

    let data_arr = Array::<f64, Ix2>::from_vec(Ix2::new([n, n]), corr_data)?;
    let mask_arr = Array::<bool, Ix2>::from_vec(Ix2::new([n, n]), corr_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

// ===========================================================================
// Functional helpers: apply_along_axis / apply_over_axes / vander
// ===========================================================================

/// Apply a function `f` to every 1-D slice along `axis`, collecting the
/// scalar result of each into an `(ndim - 1)`-D masked array.
///
/// `f` receives a [`MaskedArray<T, Ix1>`] view (rebuilt each call) and
/// returns a tuple `(value, masked)` indicating the reduction's output and
/// whether to mark the output position as masked.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= ndim`, or any error
/// from `f`.
pub fn ma_apply_along_axis<T, F>(
    ma: &MaskedArray<T, IxDyn>,
    axis: usize,
    mut f: F,
) -> FerrayResult<MaskedArray<T, IxDyn>>
where
    T: Element + Copy,
    F: FnMut(&MaskedArray<T, Ix1>) -> FerrayResult<(T, bool)>,
{
    let shape = ma.shape().to_vec();
    if axis >= shape.len() {
        return Err(FerrayError::axis_out_of_bounds(axis, shape.len()));
    }
    let lane_len = shape[axis];
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != axis)
        .map(|(_, &s)| s)
        .collect();
    let out_size: usize = out_shape.iter().product::<usize>().max(1);

    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let data: Vec<T> = ma.data().iter().copied().collect();
    let mask: Vec<bool> = ma.mask().iter().copied().collect();

    let mut out_data = Vec::with_capacity(out_size);
    let mut out_mask = Vec::with_capacity(out_size);
    let mut out_multi = vec![0usize; out_shape.len()];

    for _ in 0..out_size {
        // Build the lane.
        let mut lane_data = Vec::with_capacity(lane_len);
        let mut lane_mask = Vec::with_capacity(lane_len);
        let mut full_idx = vec![0usize; ndim];
        // Map out_multi -> full_idx (skipping axis).
        let mut oi = 0;
        for (d, slot) in full_idx.iter_mut().enumerate() {
            if d == axis {
                continue;
            }
            *slot = out_multi[oi];
            oi += 1;
        }
        for j in 0..lane_len {
            full_idx[axis] = j;
            let flat: usize = full_idx
                .iter()
                .zip(strides.iter())
                .map(|(i, s)| i * s)
                .sum();
            lane_data.push(data[flat]);
            lane_mask.push(mask[flat]);
        }
        let lane_data_arr = Array::from_vec(Ix1::new([lane_len]), lane_data)?;
        let lane_mask_arr = Array::from_vec(Ix1::new([lane_len]), lane_mask)?;
        let lane_ma = MaskedArray::new(lane_data_arr, lane_mask_arr)?;
        let (val, masked) = f(&lane_ma)?;
        out_data.push(val);
        out_mask.push(masked);

        // Advance multi-index over output dimensions.
        for d in (0..out_shape.len()).rev() {
            out_multi[d] += 1;
            if out_multi[d] < out_shape[d] {
                break;
            }
            out_multi[d] = 0;
        }
    }

    let data_arr = Array::from_vec(IxDyn::new(&out_shape), out_data)?;
    let mask_arr = Array::from_vec(IxDyn::new(&out_shape), out_mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

/// Apply `f` repeatedly, reducing over the given axes in succession.
///
/// Each axis in `axes` is reduced via [`ma_apply_along_axis`] in order.
/// When the function reduces rank, the axis is inserted back so the output
/// keeps NumPy's `apply_over_axes` rank-preserving shape contract.
///
/// # Errors
/// Returns errors from [`ma_apply_along_axis`].
pub fn ma_apply_over_axes<T, F>(
    ma: &MaskedArray<T, IxDyn>,
    axes: &[usize],
    mut f: F,
) -> FerrayResult<MaskedArray<T, IxDyn>>
where
    T: Element + Copy,
    F: FnMut(&MaskedArray<T, Ix1>) -> FerrayResult<(T, bool)>,
{
    let mut result = ma.clone();
    for &axis in axes {
        let reduced = ma_apply_along_axis(&result, axis, &mut f)?;
        result = if reduced.ndim() == result.ndim() {
            reduced
        } else {
            reduced.expand_dims(axis)?
        };
        if result.ndim() != ma.ndim() {
            return Err(FerrayError::invalid_value(
                "ma_apply_over_axes: function returned an array of the wrong rank",
            ));
        }
    }
    Ok(result)
}

/// Vandermonde matrix from a 1-D masked input, matching `numpy.ma.vander`.
///
/// numpy.ma.vander (`numpy/ma/extras.py:2216`):
/// ```text
/// _vander = np.vander(x, n)
/// m = getmask(x)
/// if m is not nomask:
///     _vander[m] = 0
/// return _vander
/// ```
/// The plain (unmasked) Vandermonde of the raw data is built first, then
/// **every masked row is set to all-zeros**, and the result carries **no
/// mask** (it is a plain ndarray returned through the `ma` namespace). Live
/// oracle (numpy 2.4.5): `np.ma.vander(np.ma.array([1.,2,3],mask=[0,1,0]),3)` →
/// `[[1,1,1],[0,0,0],[9,3,1]]`, `getmaskarray(...) == all False`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `x` is empty.
pub fn ma_vander<T>(x: &MaskedArray<T, Ix1>, n: Option<usize>) -> FerrayResult<MaskedArray<T, Ix2>>
where
    T: Element + Copy + std::ops::Mul<Output = T> + num_traits::One + num_traits::Zero,
{
    let m = x.size();
    if m == 0 {
        return Err(FerrayError::invalid_value(
            "ma_vander: input must not be empty",
        ));
    }
    let cols = n.unwrap_or(m);
    let xs: Vec<T> = x.data().iter().copied().collect();
    let xmask: Vec<bool> = x.mask().iter().copied().collect();
    let zero = num_traits::zero::<T>();
    let one = num_traits::one::<T>();
    let mut data = vec![one; m * cols];
    for (i, &xi) in xs.iter().enumerate() {
        // Build the plain Vandermonde row from the RAW data (numpy builds
        // `np.vander(x)` before masking, so unmasked rows use the true value).
        let mut acc = one;
        let mut powers = Vec::with_capacity(cols);
        for _ in 0..cols {
            powers.push(acc);
            acc = acc * xi;
        }
        for (j, p) in powers.iter().enumerate() {
            // NumPy's vander defaults to decreasing powers (left-to-right).
            data[i * cols + (cols - 1 - j)] = *p;
        }
        // numpy then zeros the ENTIRE row of any masked input (`_vander[m]=0`).
        if xmask[i] {
            for slot in data[i * cols..(i + 1) * cols].iter_mut() {
                *slot = zero;
            }
        }
    }
    let data_arr = Array::from_vec(Ix2::new([m, cols]), data)?;
    // nomask result — masked-array constructed from data only.
    MaskedArray::from_data(data_arr)
}

// ===========================================================================
// Fill-value protocol
// ===========================================================================

/// Default fill value for a dtype, mirroring NumPy:
/// - bool: `false`
/// - integer: `999_999` (capped at the type max for parity with modern numpy)
/// - float: `1e20` (numpy uses `1e20` for f64, `1e20` cast to f32)
/// - complex: `1e20 + 0j`
///
/// This function is type-erased at the boundary; the type-specific
/// equivalents are `T::fill_default_value`. ferray represents this via a
/// trait helper rather than a single function.
#[must_use]
pub fn default_fill_value_f64() -> f64 {
    1e20
}

/// Default fill value for f32.
#[must_use]
pub fn default_fill_value_f32() -> f32 {
    1e20_f32
}

/// Default fill value for bool.
#[must_use]
pub const fn default_fill_value_bool() -> bool {
    true
}

/// Default fill value for `i64`.
#[must_use]
pub const fn default_fill_value_i64() -> i64 {
    999_999
}

/// Maximum fill value for a Float type, matching `numpy.ma.maximum_fill_value`.
#[must_use]
pub fn maximum_fill_value<T: Float>() -> T {
    T::neg_infinity()
}

/// Minimum fill value for a Float type, matching `numpy.ma.minimum_fill_value`.
#[must_use]
pub fn minimum_fill_value<T: Float>() -> T {
    T::infinity()
}

/// Common fill value for two masked arrays — returns `a.fill_value()` if
/// both share the same fill value, else `T::zero()` (NumPy's fallback).
pub fn common_fill_value<T, D>(a: &MaskedArray<T, D>, b: &MaskedArray<T, D>) -> T
where
    T: Element + Copy + PartialEq,
    D: Dimension,
{
    if a.fill_value() == b.fill_value() {
        a.fill_value()
    } else {
        T::zero()
    }
}

// ===========================================================================
// Comparison and logical ufuncs (mask-aware)
// ===========================================================================

macro_rules! ma_cmp {
    ($name:ident, $op:tt, $bound:path) => {
        /// Element-wise comparison preserving mask union.
        ///
        /// # Errors
        /// Returns `FerrayError::ShapeMismatch` if shapes differ.
        pub fn $name<T, D>(
            a: &MaskedArray<T, D>,
            b: &MaskedArray<T, D>,
        ) -> FerrayResult<MaskedArray<bool, D>>
        where
            T: Element + Copy + $bound,
            D: Dimension,
        {
            if a.shape() != b.shape() {
                return Err(FerrayError::shape_mismatch(format!(
                    "{}: shapes {:?} and {:?} differ",
                    stringify!($name),
                    a.shape(),
                    b.shape(),
                )));
            }
            let data: Vec<bool> = a
                .data()
                .iter()
                .zip(a.mask().iter())
                .zip(b.data().iter().zip(b.mask().iter()))
                .map(|((x, ma), (y, mb))| {
                    if *ma || *mb {
                        *x != T::zero()
                    } else {
                        x $op y
                    }
                })
                .collect();
            let mask: Vec<bool> = a
                .mask()
                .iter()
                .zip(b.mask().iter())
                .map(|(x, y)| *x || *y)
                .collect();
            let data_arr = Array::from_vec(a.data().dim().clone(), data)?;
            let mask_arr: Array<bool, D> = Array::from_vec(a.mask().dim().clone(), mask)?;
            let mut out = MaskedArray::new(data_arr, mask_arr)?;
            out.set_typed_fill_value(a.fill_value());
            Ok(out)
        }
    };
}

ma_cmp!(ma_equal, ==, PartialEq);
ma_cmp!(ma_not_equal, !=, PartialEq);
ma_cmp!(ma_less, <, PartialOrd);
ma_cmp!(ma_greater, >, PartialOrd);
ma_cmp!(ma_less_equal, <=, PartialOrd);
ma_cmp!(ma_greater_equal, >=, PartialOrd);

/// Element-wise logical AND on bool MaskedArrays. Mask is unioned.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn ma_logical_and<D: Dimension>(
    a: &MaskedArray<bool, D>,
    b: &MaskedArray<bool, D>,
) -> FerrayResult<MaskedArray<bool, D>> {
    binary_bool(a, b, |x, y| x && y, "ma_logical_and")
}

/// Element-wise logical OR.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn ma_logical_or<D: Dimension>(
    a: &MaskedArray<bool, D>,
    b: &MaskedArray<bool, D>,
) -> FerrayResult<MaskedArray<bool, D>> {
    binary_bool(a, b, |x, y| x || y, "ma_logical_or")
}

/// Element-wise logical XOR.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if shapes differ.
pub fn ma_logical_xor<D: Dimension>(
    a: &MaskedArray<bool, D>,
    b: &MaskedArray<bool, D>,
) -> FerrayResult<MaskedArray<bool, D>> {
    binary_bool(a, b, |x, y| x ^ y, "ma_logical_xor")
}

/// Element-wise logical NOT. Mask is preserved.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_logical_not<D: Dimension>(
    a: &MaskedArray<bool, D>,
) -> FerrayResult<MaskedArray<bool, D>> {
    let data: Vec<bool> = a
        .data()
        .iter()
        .zip(a.mask().iter())
        .map(|(x, masked)| if *masked { *x } else { !*x })
        .collect();
    let data_arr = Array::from_vec(a.data().dim().clone(), data)?;
    let mask_arr: Array<bool, D> =
        Array::from_vec(a.mask().dim().clone(), a.mask().iter().copied().collect())?;
    MaskedArray::new(data_arr, mask_arr)
}

fn binary_bool<D: Dimension>(
    a: &MaskedArray<bool, D>,
    b: &MaskedArray<bool, D>,
    op: impl Fn(bool, bool) -> bool,
    name: &str,
) -> FerrayResult<MaskedArray<bool, D>> {
    if a.shape() != b.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "{name}: shapes {:?} and {:?} differ",
            a.shape(),
            b.shape()
        )));
    }
    let data: Vec<bool> = a
        .data()
        .iter()
        .zip(a.mask().iter())
        .zip(b.data().iter().zip(b.mask().iter()))
        .map(|((x, ma), (y, mb))| if *ma || *mb { *x } else { op(*x, *y) })
        .collect();
    let mask: Vec<bool> = a
        .mask()
        .iter()
        .zip(b.mask().iter())
        .map(|(x, y)| *x || *y)
        .collect();
    let data_arr = Array::from_vec(a.data().dim().clone(), data)?;
    let mask_arr: Array<bool, D> = Array::from_vec(a.mask().dim().clone(), mask)?;
    MaskedArray::new(data_arr, mask_arr)
}

// ===========================================================================
// Class helpers
// ===========================================================================

/// Whether `ma` is a [`MaskedArray`]. In Rust this is statically true, so
/// this always returns `true` — preserved for API parity with
/// `numpy.ma.isMaskedArray`.
#[must_use]
pub const fn is_masked_array<T, D>(_ma: &MaskedArray<T, D>) -> bool
where
    T: Element,
    D: Dimension,
{
    true
}

/// NumPy-spelling alias for [`is_masked_array`].
#[must_use]
pub const fn is_ma<T, D>(ma: &MaskedArray<T, D>) -> bool
where
    T: Element,
    D: Dimension,
{
    is_masked_array(ma)
}

/// Return the mask array, materializing the all-false sentinel form into a
/// real bool array. Equivalent to `numpy.ma.getmaskarray`.
///
/// Always returns an `Array<bool, D>`, never `nomask`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn getmaskarray<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element,
    D: Dimension,
{
    Array::from_vec(ma.mask().dim().clone(), ma.mask().iter().copied().collect())
}

/// Return a stable identity-pair `(data_ptr, mask_ptr)` for two
/// MaskedArrays, useful for cheap-equality checks.
///
/// Equivalent to `numpy.ma.MaskedArray.ids` — note that the mask sentinel
/// state still produces a valid pointer (the mask is materialized lazily).
#[must_use]
pub fn ids<T, D>(ma: &MaskedArray<T, D>) -> (*const u8, *const u8)
where
    T: Element,
    D: Dimension,
{
    let data_ptr: *const u8 = ma.data() as *const _ as *const u8;
    let mask_ptr: *const u8 = ma.mask() as *const _ as *const u8;
    (data_ptr, mask_ptr)
}

// ===========================================================================
// Mask-structure analysis: clump / notmasked run-length helpers
// (numpy/ma/extras.py). Slices are returned as `(start, stop)` half-open
// index pairs; the binding converts them to Python `slice` objects.
// ===========================================================================

/// Find the runs (clumps) of `true` values in a 1-D bool slice, returning each
/// as a half-open `(start, stop)` index pair.
///
/// This is the structural core shared by `clump_masked` / `clump_unmasked` /
/// `flatnotmasked_contiguous`, mirroring `numpy.ma.extras._ezclump`
/// (`numpy/ma/extras.py:2105`): it walks the boolean transitions and emits one
/// pair per contiguous run of `true`.
fn ezclump_true(mask: &[bool]) -> Vec<(usize, usize)> {
    let n = mask.len();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0usize;
    while i < n {
        if mask[i] {
            let start = i;
            while i < n && mask[i] {
                i += 1;
            }
            runs.push((start, i));
        } else {
            i += 1;
        }
    }
    runs
}

/// Flatten a masked array's mask into a row-major `Vec<bool>`.
fn flat_mask<T, D>(a: &MaskedArray<T, D>) -> Vec<bool>
where
    T: Element,
    D: Dimension,
{
    a.mask().iter().copied().collect()
}

/// `numpy.ma.clump_masked(a)` (`numpy/ma/extras.py:2189`) — the list of
/// half-open `(start, stop)` slices, one per contiguous run of **masked**
/// elements in a 1-D masked array. Returns an empty list when nothing is
/// masked (numpy returns `[]` for the `nomask` case).
#[must_use]
pub fn clump_masked<T, D>(a: &MaskedArray<T, D>) -> Vec<(usize, usize)>
where
    T: Element,
    D: Dimension,
{
    ezclump_true(&flat_mask(a))
}

/// `numpy.ma.clump_unmasked(a)` (`numpy/ma/extras.py:2235`) — the list of
/// half-open `(start, stop)` slices, one per contiguous run of **unmasked**
/// elements (numpy computes `_ezclump(~mask)`). With no mask, numpy yields the
/// single slice `[0, size)`.
#[must_use]
pub fn clump_unmasked<T, D>(a: &MaskedArray<T, D>) -> Vec<(usize, usize)>
where
    T: Element,
    D: Dimension,
{
    let inv: Vec<bool> = flat_mask(a).iter().map(|&m| !m).collect();
    ezclump_true(&inv)
}

/// `numpy.ma.flatnotmasked_edges(a)` (`numpy/ma/extras.py:1762`) — the
/// `[first, last]` indices of the unmasked values of a flattened masked array.
///
/// Returns `Some([0, size-1])` when nothing is masked, `Some([first, last])`
/// for the partially-masked case, and `None` when every element is masked
/// (matching numpy's `None` return).
#[must_use]
pub fn flatnotmasked_edges<T, D>(a: &MaskedArray<T, D>) -> Option<[usize; 2]>
where
    T: Element,
    D: Dimension,
{
    let m = flat_mask(a);
    let size = m.len();
    if size == 0 {
        return None;
    }
    if !m.iter().any(|&b| b) {
        return Some([0, size - 1]);
    }
    let first = m.iter().position(|&b| !b)?;
    let last = m.iter().rposition(|&b| !b)?;
    Some([first, last])
}

/// `numpy.ma.flatnotmasked_contiguous(a)` (`numpy/ma/extras.py:1818`) — the
/// list of half-open `(start, stop)` slices of contiguous **unmasked** regions
/// of a flattened masked array. With no mask, numpy returns the single slice
/// `[0, size)`; the all-masked case yields `[]`.
#[must_use]
pub fn flatnotmasked_contiguous<T, D>(a: &MaskedArray<T, D>) -> Vec<(usize, usize)>
where
    T: Element,
    D: Dimension,
{
    let inv: Vec<bool> = flat_mask(a).iter().map(|&m| !m).collect();
    ezclump_true(&inv)
}

/// `numpy.ma.notmasked_edges(a, axis=None)` (`numpy/ma/extras.py:1878`).
///
/// For `axis = None` (or a 1-D array) this is exactly
/// [`flatnotmasked_edges`]. The numpy multi-axis form returns per-axis
/// coordinate tuples; that form is a deferred extension (#835 follow-up) — this
/// entry point covers the flattened contract used by ferray's binding.
#[must_use]
pub fn notmasked_edges<T, D>(a: &MaskedArray<T, D>) -> Option<[usize; 2]>
where
    T: Element,
    D: Dimension,
{
    flatnotmasked_edges(a)
}

/// First/last unmasked coordinate tuples of a 2-D masked array along `axis`
/// (`numpy/ma/extras.py:1925` `notmasked_edges`, the explicit-`axis` branch).
///
/// numpy computes, per lane along `axis`, the min (first) and max (last)
/// coordinate of the unmasked positions, then compresses out fully-masked
/// lanes. The result is two `(rows, cols)` index tuples — the coordinate
/// arrays are always emitted in axis order (axis-0 indices, then axis-1
/// indices), independent of which `axis` selects the first/last extent.
///
/// Returns `(first, last)` where each is `(row_indices, col_indices)`. The
/// `row_indices` / `col_indices` vectors have one entry per lane (along the
/// axis orthogonal to `axis`) that contains at least one unmasked element.
///
/// # Errors
/// Returns `FerrayError::axis_out_of_bounds` if `axis >= 2`.
#[allow(
    clippy::type_complexity,
    reason = "mirrors numpy's two (rows, cols) tuples"
)]
pub fn notmasked_edges_axis2<T>(
    a: &MaskedArray<T, Ix2>,
    axis: usize,
) -> FerrayResult<((Vec<i64>, Vec<i64>), (Vec<i64>, Vec<i64>))>
where
    T: Element,
{
    if axis >= 2 {
        return Err(FerrayError::axis_out_of_bounds(axis, 2));
    }
    let shape = a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mask: Vec<bool> = a.mask().iter().copied().collect();

    // We iterate over lanes along `axis` for each fixed index on the orthogonal
    // axis; the first/last unmasked POSITION along `axis` defines the edge.
    let (mut first_rows, mut first_cols) = (Vec::new(), Vec::new());
    let (mut last_rows, mut last_cols) = (Vec::new(), Vec::new());
    let other_len = if axis == 0 { cols } else { rows };
    let axis_len = if axis == 0 { rows } else { cols };
    for o in 0..other_len {
        let mut first: Option<usize> = None;
        let mut last: Option<usize> = None;
        for k in 0..axis_len {
            let (r, c) = if axis == 0 { (k, o) } else { (o, k) };
            if !mask[r * cols + c] {
                if first.is_none() {
                    first = Some(k);
                }
                last = Some(k);
            }
        }
        if let (Some(f), Some(l)) = (first, last) {
            // Reconstruct the (row, col) coordinate of the first/last edge.
            let (fr_r, fr_c) = if axis == 0 { (f, o) } else { (o, f) };
            let (lr_r, lr_c) = if axis == 0 { (l, o) } else { (o, l) };
            first_rows.push(fr_r as i64);
            first_cols.push(fr_c as i64);
            last_rows.push(lr_r as i64);
            last_cols.push(lr_c as i64);
        }
    }
    Ok(((first_rows, first_cols), (last_rows, last_cols)))
}

/// `numpy.ma.notmasked_contiguous(a, axis=None)` (`numpy/ma/extras.py:1936`).
///
/// `axis = None` matches [`flatnotmasked_contiguous`] (numpy delegates to it
/// directly). For a 2-D array with an explicit `axis`, returns one slice list
/// per lane along the orthogonal axis (a list of lists), mirroring numpy's
/// per-lane `flatnotmasked_contiguous` sweep.
///
/// # Errors
/// Returns `FerrayError::axis_out_of_bounds` if `axis >= 2`, and a
/// shape-mismatch error if the array is not 2-D when an axis is given.
pub fn notmasked_contiguous_axis<T>(
    a: &MaskedArray<T, Ix2>,
    axis: usize,
) -> FerrayResult<Vec<Vec<(usize, usize)>>>
where
    T: Element,
{
    if axis >= 2 {
        return Err(FerrayError::axis_out_of_bounds(axis, 2));
    }
    let shape = a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let mask: Vec<bool> = a.mask().iter().copied().collect();
    // `other` is the axis we iterate over; for each fixed index on `other`
    // we sweep the full lane along `axis` and collect unmasked runs.
    let other = (axis + 1) % 2;
    let other_len = if other == 0 { rows } else { cols };
    let axis_len = if axis == 0 { rows } else { cols };
    let mut result: Vec<Vec<(usize, usize)>> = Vec::with_capacity(other_len);
    for o in 0..other_len {
        let mut lane: Vec<bool> = Vec::with_capacity(axis_len);
        for k in 0..axis_len {
            let (r, c) = if axis == 0 { (k, o) } else { (o, k) };
            lane.push(mask[r * cols + c]);
        }
        let inv: Vec<bool> = lane.iter().map(|&m| !m).collect();
        result.push(ezclump_true(&inv));
    }
    Ok(result)
}

// ===========================================================================
// 2-D masked matrix product (numpy/ma/core.py:8214, non-strict default)
// ===========================================================================

impl<T> MaskedArray<T, Ix2>
where
    T: Element + Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + num_traits::Zero,
{
    /// `numpy.ma.dot(a, b)` for 2-D operands, non-strict (the default)
    /// (`numpy/ma/core.py:8214`).
    ///
    /// numpy computes `d = dot(filled(a, 0), filled(b, 0))` (masked entries
    /// contribute zero) and masks the result where **no** valid pair
    /// contributed: `m = ~dot(~mask_a, ~mask_b)`. Concretely `out[i, j]` is
    /// masked iff for every `k`, `a[i, k]` is masked OR `b[k, j]` is masked.
    ///
    /// # Errors
    /// Returns `FerrayError::shape_mismatch` if `self.cols != other.rows`.
    pub fn ma_dot_2d(&self, other: &MaskedArray<T, Ix2>) -> FerrayResult<MaskedArray<T, Ix2>> {
        let a_shape = self.shape();
        let b_shape = other.shape();
        let (m, k1) = (a_shape[0], a_shape[1]);
        let (k2, n) = (b_shape[0], b_shape[1]);
        if k1 != k2 {
            return Err(FerrayError::shape_mismatch(format!(
                "ma_dot_2d: lhs.cols={k1} != rhs.rows={k2}"
            )));
        }
        let a_data: Vec<T> = self.data().iter().copied().collect();
        let a_mask: Vec<bool> = self.mask().iter().copied().collect();
        let b_data: Vec<T> = other.data().iter().copied().collect();
        let b_mask: Vec<bool> = other.mask().iter().copied().collect();
        let zero = <T as num_traits::Zero>::zero();

        let mut out_data = vec![zero; m * n];
        let mut out_mask = vec![false; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = zero;
                // `any_valid` tracks whether at least one k-pair was unmasked
                // on BOTH sides — mirrors `dot(~am, ~bm) != 0` (the result mask).
                let mut any_valid = false;
                for k in 0..k1 {
                    let am = a_mask[i * k1 + k];
                    let bm = b_mask[k * n + j];
                    // numpy's data is `dot(filled(a,0), filled(b,0))`: masked
                    // operands contribute a zero factor, but the sum still runs
                    // over every k.
                    let av = if am { zero } else { a_data[i * k1 + k] };
                    let bv = if bm { zero } else { b_data[k * n + j] };
                    acc = acc + av * bv;
                    if !am && !bm {
                        any_valid = true;
                    }
                }
                out_data[i * n + j] = acc;
                out_mask[i * n + j] = !any_valid;
            }
        }
        let data_arr = Array::<T, Ix2>::from_vec(Ix2::new([m, n]), out_data)?;
        let mask_arr = Array::<bool, Ix2>::from_vec(Ix2::new([m, n]), out_mask)?;
        MaskedArray::new(data_arr, mask_arr)
    }
}

// ===========================================================================
// Multi-axis masked argsort (numpy.ma.argsort, numpy/ma/core.py)
// ===========================================================================

impl<T, D> MaskedArray<T, D>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    /// `numpy.ma.argsort(a, axis)` along an explicit `axis`
    /// (`numpy/ma/core.py` `MaskedArray.argsort`).
    ///
    /// numpy fills masked positions with the per-dtype maximum fill value
    /// (`endwith=True`, the default) so they sort to the **end** of each lane,
    /// then returns the ordinary argsort of the filled data. Within a lane,
    /// indices are returned relative to that lane (0-based), exactly like
    /// `numpy.argsort(..., axis=axis)`. The result is a plain index array
    /// (no mask) of the input shape.
    ///
    /// # Errors
    /// Returns `FerrayError::axis_out_of_bounds` if `axis >= self.ndim()`.
    pub fn argsort_axis(&self, axis: usize) -> FerrayResult<Array<u64, IxDyn>> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(FerrayError::axis_out_of_bounds(axis, ndim));
        }
        let shape = self.shape().to_vec();
        let axis_len = shape[axis];
        let total: usize = shape.iter().product();
        let src_data: Vec<T> = self.data().iter().copied().collect();
        let src_mask: Vec<bool> = self.mask().iter().copied().collect();

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let mut out = vec![0u64; total];

        let outer_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
            .collect();
        let outer_size: usize = if outer_shape.is_empty() {
            1
        } else {
            outer_shape.iter().product()
        };

        let mut outer_multi = vec![0usize; outer_shape.len()];
        for _ in 0..outer_size {
            // Resolve the flat index of lane position `k` for the current
            // outer multi-index.
            let flat_of = |k: usize| -> usize {
                let mut flat = 0usize;
                let mut o = 0usize;
                for (i, &stride) in strides.iter().enumerate() {
                    if i == axis {
                        flat += stride * k;
                    } else {
                        flat += stride * outer_multi[o];
                        o += 1;
                    }
                }
                flat
            };

            // Stable argsort over lane positions: unmasked compare by value;
            // any masked entry sorts after every unmasked one (the `endwith`
            // fill-to-max behaviour), with masked entries keeping input order.
            let mut order: Vec<usize> = (0..axis_len).collect();
            order.sort_by(|&x, &y| {
                let fx = flat_of(x);
                let fy = flat_of(y);
                match (src_mask[fx], src_mask[fy]) {
                    (false, false) => src_data[fx]
                        .partial_cmp(&src_data[fy])
                        .unwrap_or(std::cmp::Ordering::Equal),
                    (false, true) => std::cmp::Ordering::Less,
                    (true, false) => std::cmp::Ordering::Greater,
                    (true, true) => std::cmp::Ordering::Equal,
                }
            });

            for (k, &pos) in order.iter().enumerate() {
                out[flat_of(k)] = pos as u64;
            }

            for i in (0..outer_shape.len()).rev() {
                outer_multi[i] += 1;
                if outer_multi[i] < outer_shape[i] {
                    break;
                }
                outer_multi[i] = 0;
            }
        }

        Array::<u64, IxDyn>::from_vec(IxDyn::new(&shape), out)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Ix1;

    fn arr1f(v: Vec<f64>) -> Array<f64, Ix1> {
        let n = v.len();
        Array::from_vec(Ix1::new([n]), v).unwrap()
    }

    fn mask1(v: Vec<bool>) -> Array<bool, Ix1> {
        let n = v.len();
        Array::from_vec(Ix1::new([n]), v).unwrap()
    }

    fn ma_f1(d: Vec<f64>, m: Vec<bool>) -> MaskedArray<f64, Ix1> {
        MaskedArray::new(arr1f(d), mask1(m)).unwrap()
    }

    #[test]
    fn prod_skips_masked() {
        let ma = ma_f1(vec![2.0, 3.0, 5.0, 7.0], vec![false, true, false, false]);
        let p = ma.prod().unwrap();
        assert!((p - 70.0).abs() < 1e-10); // 2 * 5 * 7
    }

    #[test]
    fn cumsum_propagates_mask() {
        let ma = ma_f1(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        let r = ma.cumsum_flat().unwrap();
        let mask: Vec<bool> = r.mask().iter().copied().collect();
        let data: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(mask, vec![false, true, false, false]);
        // Running sum: 1, (skipped→still 1, marked masked), 1+3=4, 4+4=8
        assert!((data[0] - 1.0).abs() < 1e-10);
        assert!((data[2] - 4.0).abs() < 1e-10);
        assert!((data[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn argmin_argmax_skip_masked() {
        let ma = ma_f1(vec![5.0, 1.0, 9.0, 3.0], vec![false, true, false, false]);
        // unmasked: 5, 9, 3 at positions 0, 2, 3 → min at 3, max at 2
        assert_eq!(ma.argmin().unwrap(), 3);
        assert_eq!(ma.argmax().unwrap(), 2);
    }

    #[test]
    fn ptp_basic() {
        let ma = ma_f1(vec![5.0, 1.0, 9.0, 3.0], vec![false, true, false, false]);
        // unmasked: 5, 9, 3 → ptp = 9 - 3 = 6
        assert!((ma.ptp().unwrap() - 6.0).abs() < 1e-10);
    }

    #[test]
    fn median_odd_and_even() {
        let odd = ma_f1(vec![3.0, 1.0, 4.0, 1.0, 5.0], vec![false; 5]);
        assert!((odd.median().unwrap() - 3.0).abs() < 1e-10);
        let even = ma_f1(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        assert!((even.median().unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn average_unweighted_matches_mean() {
        let ma = ma_f1(vec![2.0, 4.0, 6.0], vec![false; 3]);
        assert!((ma.average(None).unwrap() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn average_weighted_skips_masked() {
        let ma = ma_f1(vec![1.0, 100.0, 3.0], vec![false, true, false]);
        let w = arr1f(vec![1.0, 1.0, 3.0]);
        // unmasked weighted: (1*1 + 3*3) / (1 + 3) = 10/4 = 2.5
        assert!((ma.average(Some(&w)).unwrap() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn anom_centers_at_zero() {
        let ma = ma_f1(vec![1.0, 2.0, 3.0], vec![false; 3]);
        let a = ma.anom().unwrap();
        let data: Vec<f64> = a.data().iter().copied().collect();
        assert!((data[0] - (-1.0)).abs() < 1e-10);
        assert!((data[1] - 0.0).abs() < 1e-10);
        assert!((data[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn masked_all_is_fully_masked() {
        let ma: MaskedArray<f64, Ix1> = masked_all(Ix1::new([5])).unwrap();
        let mask: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask, vec![true; 5]);
    }

    #[test]
    fn masked_values_within_tol() {
        let arr = arr1f(vec![1.0, 1.0001, 2.0]);
        let r = masked_values(&arr, 1.0, 1e-3, 0.0).unwrap();
        let mask: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(mask, vec![true, true, false]);
        assert_eq!(r.fill_value(), 1.0);
    }

    #[test]
    fn harden_mask_blocks_clearing() {
        let mut ma = ma_f1(vec![1.0, 2.0], vec![false, true]);
        ma.harden_mask().unwrap();
        assert!(ma.is_hard_mask());
        // Try to clear the mask bit at index 1.
        ma.set_mask_flat(1, false).unwrap();
        let mask: Vec<bool> = ma.mask().iter().copied().collect();
        // Hardened: clear is a no-op.
        assert_eq!(mask, vec![false, true]);
        ma.soften_mask().unwrap();
        ma.set_mask_flat(1, false).unwrap();
        let mask2: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(mask2, vec![false, false]);
    }

    #[test]
    fn mask_or_unions() {
        let m1 = mask1(vec![true, false, false]);
        let m2 = mask1(vec![false, true, false]);
        let r = mask_or(&m1, &m2).unwrap();
        let v: Vec<bool> = r.iter().copied().collect();
        assert_eq!(v, vec![true, true, false]);
    }

    #[test]
    fn make_mask_none_is_all_false() {
        let m: Array<bool, Ix1> = make_mask_none(Ix1::new([3])).unwrap();
        let v: Vec<bool> = m.iter().copied().collect();
        assert_eq!(v, vec![false; 3]);
    }

    #[test]
    fn clip_unmasked_only() {
        let ma = ma_f1(vec![-5.0, 0.0, 5.0, 10.0], vec![false, false, false, true]);
        let r = ma.clip(-1.0, 3.0).unwrap();
        let data: Vec<f64> = r.data().iter().copied().collect();
        // Masked element passes through (10.0).
        assert_eq!(data, vec![-1.0, 0.0, 3.0, 10.0]);
    }

    #[test]
    fn repeat_propagates_mask() {
        let ma = ma_f1(vec![1.0, 2.0], vec![false, true]);
        let r = ma.repeat(3).unwrap();
        let data: Vec<f64> = r.data().iter().copied().collect();
        let mask: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(data, vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        assert_eq!(mask, vec![false, false, false, true, true, true]);
    }

    #[test]
    fn ma_unique_dedups_unmasked() {
        let ma = ma_f1(
            vec![3.0, 1.0, 2.0, 1.0, 3.0, 9.0],
            vec![false, false, false, false, false, true],
        );
        let v = ma_unique(&ma).unwrap();
        let data: Vec<f64> = v.data().iter().copied().collect();
        let mask: Vec<bool> = v.mask().iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 9.0]);
        assert_eq!(mask, vec![false, false, false, true]);
    }

    #[test]
    fn ma_isin_basic() {
        let ma = ma_f1(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        let r = ma_isin(&ma, &[2.0, 4.0]).unwrap();
        let data: Vec<bool> = r.data().iter().copied().collect();
        assert_eq!(data, vec![false, true, false, true]);
    }

    #[test]
    fn ma_dot_flat_skips_masked_pairs() {
        let a = ma_f1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let b = ma_f1(vec![4.0, 5.0, 6.0], vec![false, false, false]);
        // Pair (1,4): both ok → 4. Pair (2,5): a masked → skip. Pair (3,6): both ok → 18.
        // Total = 22.
        assert!((a.ma_dot_flat(&b).unwrap() - 22.0).abs() < 1e-10);
    }

    #[test]
    fn fill_value_protocol_constants() {
        assert_eq!(default_fill_value_f64(), 1e20);
        assert!(default_fill_value_bool());
        assert!(
            maximum_fill_value::<f64>().is_infinite()
                && maximum_fill_value::<f64>().is_sign_negative()
        );
        assert!(minimum_fill_value::<f64>().is_infinite());
    }

    #[test]
    fn common_fill_value_returns_shared_or_zero() {
        let a = ma_f1(vec![1.0, 2.0], vec![false, false]).with_fill_value(99.0);
        let b = ma_f1(vec![3.0, 4.0], vec![false, false]).with_fill_value(99.0);
        assert_eq!(common_fill_value(&a, &b), 99.0);
        let c = ma_f1(vec![5.0, 6.0], vec![false, false]).with_fill_value(0.5);
        assert_eq!(common_fill_value(&a, &c), 0.0); // fall back to zero
    }

    #[test]
    fn ma_equal_and_friends_union_mask() {
        let a = ma_f1(vec![1.0, 2.0, 3.0], vec![false, false, true]);
        let b = ma_f1(vec![1.0, 9.0, 3.0], vec![false, true, false]);
        let r = ma_equal(&a, &b).unwrap();
        let data: Vec<bool> = r.data().iter().copied().collect();
        let mask: Vec<bool> = r.mask().iter().copied().collect();
        assert_eq!(data, vec![true, true, true]);
        assert_eq!(mask, vec![false, true, true]);
    }

    #[test]
    fn ma_logical_and_basic() {
        let a = MaskedArray::new(
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, true, false]).unwrap(),
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap(),
        )
        .unwrap();
        let b = MaskedArray::new(
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![true, false, true]).unwrap(),
            Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false; 3]).unwrap(),
        )
        .unwrap();
        let r = ma_logical_and(&a, &b).unwrap();
        let data: Vec<bool> = r.data().iter().copied().collect();
        assert_eq!(data, vec![true, false, false]);
    }

    #[test]
    fn is_masked_array_always_true() {
        let ma = ma_f1(vec![1.0], vec![false]);
        assert!(is_masked_array(&ma));
        assert!(is_ma(&ma));
    }

    #[test]
    fn getmaskarray_materializes() {
        let ma = MaskedArray::<f64, Ix1>::from_data(arr1f(vec![1.0, 2.0])).unwrap();
        let m = getmaskarray(&ma).unwrap();
        let v: Vec<bool> = m.iter().copied().collect();
        assert_eq!(v, vec![false; 2]);
    }

    #[test]
    fn diagonal_main_and_offset() {
        use ferray_core::Ix2;
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([3, 3]), vec![false; 9]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let main = ma.diagonal(0).unwrap();
        let main_data: Vec<f64> = main.data().iter().copied().collect();
        assert_eq!(main_data, vec![1.0, 5.0, 9.0]);
        let upper = ma.diagonal(1).unwrap();
        let upper_data: Vec<f64> = upper.data().iter().copied().collect();
        assert_eq!(upper_data, vec![2.0, 6.0]);
    }

    #[test]
    fn trace_sums_diagonal() {
        use ferray_core::Ix2;
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([3, 3]),
            vec![1.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 9.0],
        )
        .unwrap();
        let mask = Array::<bool, Ix2>::from_vec(Ix2::new([3, 3]), vec![false; 9]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        assert!((ma.trace(0).unwrap() - 15.0).abs() < 1e-10);
    }

    #[test]
    fn ma_apply_along_axis_sum_lane() {
        use ferray_core::IxDyn;
        let data =
            Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                .unwrap();
        let mask = Array::<bool, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![false; 6]).unwrap();
        let ma = MaskedArray::new(data, mask).unwrap();
        let result = ma_apply_along_axis(&ma, 1, |lane| {
            let s: f64 = lane.data().iter().copied().sum();
            Ok((s, false))
        })
        .unwrap();
        let v: Vec<f64> = result.data().iter().copied().collect();
        assert_eq!(v, vec![6.0, 15.0]);
    }

    // -----------------------------------------------------------------------
    // Masked set ops (numpy.ma.intersect1d/union1d/setdiff1d/setxor1d).
    // Expected values from the numpy 2.4.5 oracle (R-CHAR-3):
    //   a = ma([1,2,3,4], mask=[0,1,0,0]); b = ma([3,4,5], mask=[0,0,1])
    //   intersect -> data[3,4,--] mask[F,F,T]
    //   union     -> data[1,3,4,--] mask[F,F,F,T]
    //   setdiff   -> data[1] mask[F]
    //   setxor    -> data[1] mask[F]
    //   unique(a) -> data[1,3,4,--] mask[F,F,F,T]
    // -----------------------------------------------------------------------
    fn ab() -> (MaskedArray<f64, Ix1>, MaskedArray<f64, Ix1>) {
        (
            ma_f1(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]),
            ma_f1(vec![3.0, 4.0, 5.0], vec![false, false, true]),
        )
    }

    fn data_mask(m: &MaskedArray<f64, Ix1>) -> (Vec<f64>, Vec<bool>) {
        (
            m.data().iter().copied().collect(),
            m.mask().iter().copied().collect(),
        )
    }

    #[test]
    fn ma_unique_masked_trails_one_masked_slot() -> FerrayResult<()> {
        let (a, _) = ab();
        let (data, mask) = data_mask(&ma_unique_masked(&a)?);
        // numpy: unmasked uniques [1,3,4] then one trailing masked slot.
        assert_eq!(&data[..3], &[1.0, 3.0, 4.0]);
        assert_eq!(mask, vec![false, false, false, true]);
        Ok(())
    }

    #[test]
    fn ma_intersect1d_common_with_both_masked() -> FerrayResult<()> {
        let (a, b) = ab();
        let (data, mask) = data_mask(&ma_intersect1d(&a, &b)?);
        assert_eq!(&data[..2], &[3.0, 4.0]);
        assert_eq!(mask, vec![false, false, true]);
        Ok(())
    }

    #[test]
    fn ma_union1d_all_uniques_plus_masked() -> FerrayResult<()> {
        let (a, b) = ab();
        let (data, mask) = data_mask(&ma_union1d(&a, &b)?);
        assert_eq!(&data[..3], &[1.0, 3.0, 4.0]);
        assert_eq!(mask, vec![false, false, false, true]);
        Ok(())
    }

    #[test]
    fn ma_setdiff1d_drops_masked_when_rhs_masked() -> FerrayResult<()> {
        let (a, b) = ab();
        let (data, mask) = data_mask(&ma_setdiff1d(&a, &b)?);
        // a's masked slot is removed because b is also masked.
        assert_eq!(data, vec![1.0]);
        assert_eq!(mask, vec![false]);
        Ok(())
    }

    #[test]
    fn ma_setxor1d_symmetric_difference() -> FerrayResult<()> {
        let (a, b) = ab();
        let (data, mask) = data_mask(&ma_setxor1d(&a, &b)?);
        // both masked -> masked slot cancels; only unmasked-unique-once is 1.
        assert_eq!(data, vec![1.0]);
        assert_eq!(mask, vec![false]);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // compress_rowcols / mask_rowcols (numpy.ma.extras).
    // Oracle: m = ma([[1,2,3],[4,5,6]], mask=[[0,0,1],[0,0,0]])
    //   compress_rowcols(None) -> [[4,5]]
    //   compress_rows          -> [[4,5,6]]
    //   compress_cols          -> [[1,2],[4,5]]
    //   mask_rowcols(None)     -> mask[[T,T,T],[F,F,T]]
    // -----------------------------------------------------------------------
    fn m23() -> FerrayResult<MaskedArray<f64, Ix2>> {
        let data =
            Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
        let mask = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, false, true, false, false, false],
        )?;
        MaskedArray::new(data, mask)
    }

    #[test]
    fn ma_compress_rowcols_both() -> FerrayResult<()> {
        let r = ma_compress_rowcols(&m23()?, None)?;
        assert_eq!(r.shape(), &[1, 2]);
        let v: Vec<f64> = r.iter().copied().collect();
        assert_eq!(v, vec![4.0, 5.0]);
        Ok(())
    }

    #[test]
    fn ma_compress_rows_and_cols() -> FerrayResult<()> {
        let rows = ma_compress_rows(&m23()?)?;
        assert_eq!(rows.shape(), &[1, 3]);
        assert_eq!(
            rows.iter().copied().collect::<Vec<f64>>(),
            vec![4.0, 5.0, 6.0]
        );
        let cols = ma_compress_cols(&m23()?)?;
        assert_eq!(cols.shape(), &[2, 2]);
        assert_eq!(
            cols.iter().copied().collect::<Vec<f64>>(),
            vec![1.0, 2.0, 4.0, 5.0]
        );
        Ok(())
    }

    #[test]
    fn ma_mask_rowcols_masks_whole_row_and_col() -> FerrayResult<()> {
        let r = ma_mask_rowcols(&m23()?, None)?;
        let mask: Vec<bool> = r.mask().iter().copied().collect();
        // row 0 and col 2 both masked: [[T,T,T],[F,F,T]]
        assert_eq!(mask, vec![true, true, true, false, false, true]);
        let data: Vec<f64> = r.data().iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Masked cov / corrcoef (numpy.ma.cov / corrcoef).
    // Oracle: m = ma([[1,2,3],[4,5,6]], mask=[[0,0,1],[0,0,0]])
    //   cov      -> [[0.5,0.5],[0.5,1.0]]
    //   corrcoef -> [[1.0, 0.70710678...],[0.70710678..., 1.0]]
    // -----------------------------------------------------------------------
    #[test]
    fn ma_cov_unmasked_pairs() -> FerrayResult<()> {
        let c = ma_cov(&m23()?, true, false, None)?;
        let data: Vec<f64> = c.data().iter().copied().collect();
        let mask: Vec<bool> = c.mask().iter().copied().collect();
        let expect = [0.5, 0.5, 0.5, 1.0];
        for (g, e) in data.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-12, "cov {g} != {e}");
        }
        assert_eq!(mask, vec![false; 4]);
        Ok(())
    }

    #[test]
    fn ma_corrcoef_normalized() -> FerrayResult<()> {
        let c = ma_corrcoef(&m23()?, true)?;
        let data: Vec<f64> = c.data().iter().copied().collect();
        let expect = [1.0, 0.7071067811865475, 0.7071067811865475, 1.0];
        for (g, e) in data.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-12, "corr {g} != {e}");
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Mask-structure analysis: clump / notmasked run-length helpers.
    //
    // Oracle (numpy 2.4.5) for `m = ma.array([1,2,3,4,5,6], mask=[0,0,1,1,0,0])`:
    //   clump_masked          -> [slice(2, 4)]
    //   clump_unmasked        -> [slice(0, 2), slice(4, 6)]
    //   notmasked_contiguous  -> [slice(0, 2), slice(4, 6)]
    //   notmasked_edges       -> [0, 5]
    //   flatnotmasked_edges   -> [0, 5]
    // -----------------------------------------------------------------------
    #[test]
    fn clump_run_length_matches_numpy() {
        let m = ma_f1(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, false, true, true, false, false],
        );
        assert_eq!(clump_masked(&m), vec![(2, 4)]);
        assert_eq!(clump_unmasked(&m), vec![(0, 2), (4, 6)]);
        assert_eq!(flatnotmasked_contiguous(&m), vec![(0, 2), (4, 6)]);
        assert_eq!(flatnotmasked_edges(&m), Some([0, 5]));
        assert_eq!(notmasked_edges(&m), Some([0, 5]));
    }

    #[test]
    fn clump_nomask_and_allmask_edges() {
        // numpy: clump_unmasked(no mask) -> [slice(0,n)], clump_masked -> [];
        // flatnotmasked_edges(no mask) -> [0, n-1].
        let none = ma_f1(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        assert_eq!(clump_masked(&none), vec![]);
        assert_eq!(clump_unmasked(&none), vec![(0, 3)]);
        assert_eq!(flatnotmasked_edges(&none), Some([0, 2]));
        // numpy: all masked -> clump_unmasked [], clump_masked [slice(0,n)],
        // flatnotmasked_contiguous [], flatnotmasked_edges None.
        let all = ma_f1(vec![1.0, 2.0, 3.0], vec![true, true, true]);
        assert_eq!(clump_unmasked(&all), vec![]);
        assert_eq!(clump_masked(&all), vec![(0, 3)]);
        assert_eq!(flatnotmasked_contiguous(&all), vec![]);
        assert_eq!(flatnotmasked_edges(&all), None);
    }

    #[test]
    fn clump_leading_and_trailing_masked() {
        // numpy for ma.masked_array(arange(10)); a[[0,1,2,6,8,9]] = masked:
        //   clump_masked   -> [slice(0,3), slice(6,7), slice(8,10)]
        //   clump_unmasked -> [slice(3,6), slice(7,8)]
        let mut mask = vec![false; 10];
        for &i in &[0usize, 1, 2, 6, 8, 9] {
            mask[i] = true;
        }
        let data: Vec<f64> = (0..10).map(|x| x as f64).collect();
        let m = ma_f1(data, mask);
        assert_eq!(clump_masked(&m), vec![(0, 3), (6, 7), (8, 10)]);
        assert_eq!(clump_unmasked(&m), vec![(3, 6), (7, 8)]);
    }

    #[test]
    fn notmasked_contiguous_axis_2d_matches_numpy() -> FerrayResult<()> {
        use ferray_core::Ix2;
        // numpy:
        //   a = arange(12).reshape(3,4); mask[1:,:-1]=1; mask[0,1]=1; mask[-1,0]=0
        //   notmasked_contiguous(ma, axis=0) ->
        //     [[slice(0,1), slice(2,3)], [], [slice(0,1)], [slice(0,3)]]
        //   notmasked_contiguous(ma, axis=1) ->
        //     [[slice(0,1), slice(2,4)], [slice(3,4)], [slice(0,1), slice(3,4)]]
        // mask matrix (row-major):
        //   [[0,1,0,0],[1,1,1,0],[0,1,1,0]]
        let mask = vec![
            false, true, false, false, true, true, true, false, false, true, true, false,
        ];
        let data: Vec<f64> = (0..12).map(|x| x as f64).collect();
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data)?;
        let mk = Array::<bool, Ix2>::from_vec(Ix2::new([3, 4]), mask)?;
        let m = MaskedArray::new(d, mk)?;
        let by0 = notmasked_contiguous_axis(&m, 0)?;
        assert_eq!(
            by0,
            vec![vec![(0, 1), (2, 3)], vec![], vec![(0, 1)], vec![(0, 3)]]
        );
        let by1 = notmasked_contiguous_axis(&m, 1)?;
        assert_eq!(
            by1,
            vec![vec![(0, 1), (2, 4)], vec![(3, 4)], vec![(0, 1), (3, 4)]]
        );
        Ok(())
    }

    // -----------------------------------------------------------------------
    // 2-D masked dot. Oracle (numpy 2.4.5):
    //   a = ma.array([[1.,2],[3,4]], mask=[[0,1],[0,0]])
    //   ma.dot(a, a).data -> [[1, 0], [15, 16]]  (masked entries -> 0 in product)
    //   ma.dot(a, a).mask -> [[False, True], [False, False]]
    // -----------------------------------------------------------------------
    #[test]
    fn ma_dot_2d_matches_numpy() -> FerrayResult<()> {
        use ferray_core::Ix2;
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0])?;
        let mk = Array::<bool, Ix2>::from_vec(Ix2::new([2, 2]), vec![false, true, false, false])?;
        let a = MaskedArray::new(d, mk)?;
        let out = a.ma_dot_2d(&a)?;
        let data: Vec<f64> = out.data().iter().copied().collect();
        let mask: Vec<bool> = out.mask().iter().copied().collect();
        // out[0,0]=1*1=1 (a[0,1] masked -> 0 factor); out[0,1] masked (every
        // contributing path traverses the masked a[0,1], data sums to 0).
        assert_eq!(data, vec![1.0, 0.0, 15.0, 16.0]);
        assert_eq!(mask, vec![false, true, false, false]);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Multi-axis argsort. Oracle (numpy 2.4.5):
    //   b = ma.array([[3,1,2],[6,5,4]], mask=[[0,0,1],[0,0,0]])
    //   ma.argsort(b, axis=1) -> [[1,0,2],[2,1,0]]
    //   ma.argsort(b, axis=0) -> [[0,0,1],[1,1,0]]
    // -----------------------------------------------------------------------
    #[test]
    fn argsort_axis_matches_numpy() -> FerrayResult<()> {
        use ferray_core::Ix2;
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), vec![3.0, 1.0, 2.0, 6.0, 5.0, 4.0])?;
        let mk = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![false, false, true, false, false, false],
        )?;
        let b = MaskedArray::new(d, mk)?;
        let by1 = b.argsort_axis(1)?;
        let v1: Vec<u64> = by1.iter().copied().collect();
        assert_eq!(v1, vec![1, 0, 2, 2, 1, 0]);
        let by0 = b.argsort_axis(0)?;
        let v0: Vec<u64> = by0.iter().copied().collect();
        assert_eq!(v0, vec![0, 0, 1, 1, 1, 0]);
        Ok(())
    }
}
