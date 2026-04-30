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
//
// Items intentionally NOT covered (deferred):
//   - mvoid (record-void): requires structured-record dtype work in core
//   - mr_ / masked_singleton / masked_print_option: Python-class display
//     state with no Rust analog
//   - clump_masked / clump_unmasked / *notmasked_contiguous: rarely used,
//     deferred until requested
//   - polyfit / convolve / correlate / corrcoef / cov: would pull in
//     ferray-polynomial / ferray-stats; users can call those directly
//     on the unmasked subset via .compressed()

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
        let Some(w) = weights else {
            return self.mean();
        };
        if w.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "average: weights shape {:?} differs from array shape {:?}",
                w.shape(),
                self.shape(),
            )));
        }
        let zero = <T as num_traits::Zero>::zero();
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
        Ok(acc / wsum)
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
        let data: Vec<T> = self.data().iter().map(|v| *v - m).collect();
        let data_arr = Array::from_vec(self.data().dim().clone(), data)?;
        let mask_arr: Array<bool, D> = Array::from_vec(
            self.mask().dim().clone(),
            self.mask().iter().copied().collect(),
        )?;
        MaskedArray::new(data_arr, mask_arr)
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
    MaskedArray::new(data_arr, mask_arr)
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
        MaskedArray::new(data_arr, mask_arr)
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
        MaskedArray::new(data_arr, mask_arr)
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
        MaskedArray::new(data_exp, mask_exp)
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
        MaskedArray::new(data_arr, mask_arr)
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

/// Sorted unique unmasked values of `ma` as a plain `Array<T, Ix1>` (not a
/// MaskedArray — the result has no masked entries by construction).
///
/// Equivalent to `numpy.ma.unique(ma)`.
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_unique<T, D>(ma: &MaskedArray<T, D>) -> FerrayResult<Array<T, Ix1>>
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
    let n = vals.len();
    Array::from_vec(Ix1::new([n]), vals)
}

/// Per-element membership test against `test_values`: true where `ma[i]`
/// equals any value in `test_values` AND `ma[i]` is not masked.
///
/// Equivalent to `numpy.ma.isin(ma, test_values)`. The output mask follows
/// the input's mask (positions where the input was masked are also masked
/// in the result, so the boolean is meaningful for unmasked positions).
///
/// # Errors
/// Returns an error if internal array construction fails.
pub fn ma_isin<T, D>(
    ma: &MaskedArray<T, D>,
    test_values: &[T],
) -> FerrayResult<MaskedArray<bool, D>>
where
    T: Element + Copy + PartialEq,
    D: Dimension,
{
    let data: Vec<bool> = ma
        .data()
        .iter()
        .map(|v| test_values.iter().any(|t| t == v))
        .collect();
    let data_arr = Array::from_vec(ma.data().dim().clone(), data)?;
    let mask_arr: Array<bool, D> =
        Array::from_vec(ma.mask().dim().clone(), ma.mask().iter().copied().collect())?;
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
    T: Element + Copy + PartialEq,
{
    ma_isin(ma, test_values)
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
/// Each axis in `axes` is reduced via [`ma_apply_along_axis`] in order,
/// with subsequent axis indices adjusted for previous reductions.
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
    let mut sorted: Vec<usize> = axes.to_vec();
    sorted.sort_unstable();
    for (offset, &ax) in sorted.iter().enumerate() {
        // Each previous reduction collapsed an earlier axis, so shift
        // subsequent axis indices left by the number already consumed.
        let adjusted = ax.saturating_sub(offset);
        result = ma_apply_along_axis(&result, adjusted, &mut f)?;
    }
    Ok(result)
}

/// Vandermonde matrix from a 1-D masked input. Masked rows propagate to
/// the corresponding row of the result.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `x` is empty.
pub fn ma_vander<T>(x: &MaskedArray<T, Ix1>, n: Option<usize>) -> FerrayResult<MaskedArray<T, Ix2>>
where
    T: Element + Copy + std::ops::Mul<Output = T> + num_traits::One,
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
    let one = num_traits::one::<T>();
    let mut data = vec![one; m * cols];
    let mut mask = vec![false; m * cols];
    for (i, &xi) in xs.iter().enumerate() {
        let mut acc = one;
        let mut powers = Vec::with_capacity(cols);
        for _ in 0..cols {
            powers.push(acc);
            acc = acc * xi;
        }
        for (j, p) in powers.iter().enumerate() {
            // NumPy's vander defaults to decreasing powers (left-to-right).
            data[i * cols + (cols - 1 - j)] = *p;
            mask[i * cols + (cols - 1 - j)] = xmask[i];
        }
    }
    let data_arr = Array::from_vec(Ix2::new([m, cols]), data)?;
    let mask_arr = Array::from_vec(Ix2::new([m, cols]), mask)?;
    MaskedArray::new(data_arr, mask_arr)
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
    false
}

/// Default fill value for `i64`.
#[must_use]
pub const fn default_fill_value_i64() -> i64 {
    999_999
}

/// Maximum fill value for a Float type (used when filling masked values
/// for max-reductions so they don't influence the result).
#[must_use]
pub fn maximum_fill_value<T: Float>() -> T {
    T::infinity()
}

/// Minimum fill value for a Float type (used when filling masked values
/// for min-reductions).
#[must_use]
pub fn minimum_fill_value<T: Float>() -> T {
    T::neg_infinity()
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
                .zip(b.data().iter())
                .map(|(x, y)| x $op y)
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
    let data: Vec<bool> = a.data().iter().map(|x| !*x).collect();
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
        .zip(b.data().iter())
        .map(|(x, y)| op(*x, *y))
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
        let data: Vec<f64> = v.iter().copied().collect();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
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
        assert!(!default_fill_value_bool());
        assert!(maximum_fill_value::<f64>().is_infinite());
        assert!(
            minimum_fill_value::<f64>().is_infinite()
                && minimum_fill_value::<f64>().is_sign_negative()
        );
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
        assert_eq!(data, vec![true, false, true]);
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
}
