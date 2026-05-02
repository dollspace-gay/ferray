// ferray-ma: Mask manipulation utilities (REQ-15, REQ-16, REQ-17)
//
// harden_mask, soften_mask, getmask, getdata, is_masked, count_masked

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

impl<T: Element, D: Dimension> MaskedArray<T, D> {
    /// Harden the mask: prevent subsequent assignments from clearing mask bits.
    ///
    /// After this call, any attempt to set a mask bit to `false` via
    /// `set_mask_flat` or `set_mask` will be silently ignored.
    ///
    /// # Errors
    /// This function does not currently error but returns `Result` for API
    /// consistency.
    pub const fn harden_mask(&mut self) -> FerrayResult<()> {
        self.hard_mask = true;
        Ok(())
    }

    /// Soften the mask: allow subsequent assignments to clear mask bits.
    ///
    /// # Errors
    /// This function does not currently error but returns `Result` for API
    /// consistency.
    pub const fn soften_mask(&mut self) -> FerrayResult<()> {
        self.hard_mask = false;
        Ok(())
    }
}

/// Return the mask array of a masked array.
///
/// This is equivalent to `ma.mask()` but provided as a free function
/// for API parity with `NumPy`'s `np.ma.getmask`.
///
/// # Errors
/// This function does not currently error but returns `Result` for API
/// consistency.
pub fn getmask<T: Element, D: Dimension>(ma: &MaskedArray<T, D>) -> FerrayResult<Array<bool, D>> {
    Ok(ma.mask().clone())
}

/// Return the underlying data array of a masked array.
///
/// This is equivalent to `ma.data()` but provided as a free function
/// for API parity with `NumPy`'s `np.ma.getdata`.
///
/// # Errors
/// This function does not currently error but returns `Result` for API
/// consistency.
pub fn getdata<T: Element + Copy, D: Dimension>(
    ma: &MaskedArray<T, D>,
) -> FerrayResult<Array<T, D>> {
    Ok(ma.data().clone())
}

/// Return `true` if any element in the masked array is masked.
///
/// # Errors
/// This function does not currently error but returns `Result` for API
/// consistency.
pub fn is_masked<T: Element, D: Dimension>(ma: &MaskedArray<T, D>) -> FerrayResult<bool> {
    Ok(ma.mask().iter().any(|m| *m))
}

/// Count the number of masked elements, optionally along an axis.
///
/// If `axis` is `None`, returns the total count of masked elements as a
/// single-element vector.
///
/// For axis-wise counting use [`count_masked_axis`].
///
/// # Errors
/// This function does not currently error but returns `Result` for API
/// consistency.
pub fn count_masked<T: Element, D: Dimension>(ma: &MaskedArray<T, D>) -> FerrayResult<usize> {
    let count = ma.mask().iter().filter(|m| **m).count();
    Ok(count)
}

/// Count masked elements along a specific axis (#268).
///
/// Reduces the array of masked booleans along `axis`, returning a
/// `usize` count for each remaining slice. The output shape drops
/// the reduced axis. Mirrors `numpy.ma.count_masked(a, axis=k)`.
///
/// The previous `count_masked(ma, axis: Option<usize>)` accepted an
/// `axis` argument but ignored it — a silent footgun where callers
/// got the total count back when they expected per-slice counts.
///
/// # Errors
/// Returns `FerrayError::AxisOutOfBounds` if `axis >= ma.ndim()`.
pub fn count_masked_axis<T: Element, D: Dimension>(
    ma: &MaskedArray<T, D>,
    axis: usize,
) -> FerrayResult<ferray_core::Array<u64, ferray_core::dimension::IxDyn>> {
    use ferray_core::dimension::IxDyn;

    let ndim = ma.ndim();
    if axis >= ndim {
        return Err(FerrayError::axis_out_of_bounds(axis, ndim));
    }
    let shape = ma.shape();
    let axis_len = shape[axis];

    // Output shape: drop the reduced axis.
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| if i == axis { None } else { Some(s) })
        .collect();
    let out_size: usize = if out_shape.is_empty() {
        1
    } else {
        out_shape.iter().product()
    };

    // Materialize the mask in row-major flat order so we can index by
    // computed flat indices regardless of the source memory layout.
    let mask_data: Vec<bool> = ma.mask().iter().copied().collect();
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    let mut out_data: Vec<u64> = Vec::with_capacity(out_size);
    let mut out_multi = vec![0usize; out_shape.len()];
    for _ in 0..out_size {
        // Reconstruct the source-side multi-index by inserting the
        // reduced-axis position back into the right spot.
        let mut count: u64 = 0;
        for k in 0..axis_len {
            let mut flat = 0usize;
            let mut out_idx = 0usize;
            for (i, &stride) in strides.iter().enumerate() {
                if i == axis {
                    flat += stride * k;
                } else {
                    flat += stride * out_multi[out_idx];
                    out_idx += 1;
                }
            }
            if mask_data[flat] {
                count += 1;
            }
        }
        out_data.push(count);

        // Increment the output multi-index in row-major order.
        for i in (0..out_shape.len()).rev() {
            out_multi[i] += 1;
            if out_multi[i] < out_shape[i] {
                break;
            }
            out_multi[i] = 0;
        }
    }

    let out_dim = if out_shape.is_empty() {
        IxDyn::new(&[])
    } else {
        IxDyn::new(&out_shape)
    };
    ferray_core::Array::from_vec(out_dim, out_data)
}
