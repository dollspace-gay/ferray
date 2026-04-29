// ferray-ma: Mask manipulation utilities (REQ-15, REQ-16, REQ-17)
//
// harden_mask, soften_mask, getmask, getdata, is_masked, count_masked

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;

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
/// single-element vector. If an axis is specified, this function currently
/// returns the total count (axis-specific counting for multi-dimensional
/// masked arrays is a future enhancement).
///
/// # Errors
/// This function does not currently error but returns `Result` for API
/// consistency.
pub fn count_masked<T: Element, D: Dimension>(
    ma: &MaskedArray<T, D>,
    _axis: Option<usize>,
) -> FerrayResult<usize> {
    let count = ma.mask().iter().filter(|m| **m).count();
    Ok(count)
}
