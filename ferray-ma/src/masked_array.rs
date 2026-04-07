// ferray-ma: MaskedArray<T, D> type (REQ-1, REQ-2, REQ-3)

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

/// A masked array that pairs data with a boolean mask.
///
/// Each element position has a corresponding mask bit:
/// - `true` means the element is **masked** (invalid / missing)
/// - `false` means the element is valid
///
/// All operations (arithmetic, reductions, ufuncs) respect the mask by
/// skipping masked elements.
///
/// The `fill_value` field is the replacement value for masked positions when
/// the masked array participates in operations or when [`MaskedArray::filled`]
/// is called without an explicit override. It defaults to `T::zero()`.
#[derive(Debug, Clone)]
pub struct MaskedArray<T: Element, D: Dimension> {
    /// The underlying data array.
    data: Array<T, D>,
    /// Boolean mask (`true` = masked/invalid).
    mask: Array<bool, D>,
    /// Whether the mask is hardened (cannot be cleared by assignment).
    pub(crate) hard_mask: bool,
    /// Replacement value for masked positions during operations and filling.
    /// Defaults to `T::zero()`.
    pub(crate) fill_value: T,
}

impl<T: Element, D: Dimension> MaskedArray<T, D> {
    /// Create a new masked array from data and mask arrays.
    ///
    /// The `fill_value` defaults to `T::zero()`. Use [`MaskedArray::with_fill_value`]
    /// to set a custom replacement value.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if data and mask shapes differ.
    pub fn new(data: Array<T, D>, mask: Array<bool, D>) -> FerrayResult<Self> {
        if data.shape() != mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "MaskedArray::new: data shape {:?} does not match mask shape {:?}",
                data.shape(),
                mask.shape()
            )));
        }
        Ok(Self {
            data,
            mask,
            hard_mask: false,
            fill_value: T::zero(),
        })
    }

    /// Create a masked array with no masked elements (all-false mask).
    ///
    /// # Errors
    /// Returns an error if the mask array cannot be created.
    pub fn from_data(data: Array<T, D>) -> FerrayResult<Self> {
        let mask = Array::<bool, D>::from_elem(data.dim().clone(), false)?;
        Ok(Self {
            data,
            mask,
            hard_mask: false,
            fill_value: T::zero(),
        })
    }

    /// Return the fill value used to replace masked positions.
    ///
    /// See [`MaskedArray::with_fill_value`] for setting it.
    #[inline]
    pub fn fill_value(&self) -> T
    where
        T: Copy,
    {
        self.fill_value
    }

    /// Set the fill value, returning the modified array.
    ///
    /// The fill value is used by [`MaskedArray::filled`] (when called
    /// without an explicit override) and by arithmetic operations as the
    /// replacement for masked positions in the result data.
    pub fn with_fill_value(mut self, fill_value: T) -> Self {
        self.fill_value = fill_value;
        self
    }

    /// Replace the fill value in place.
    pub fn set_fill_value(&mut self, fill_value: T) {
        self.fill_value = fill_value;
    }

    /// Return a reference to the underlying data array.
    #[inline]
    pub fn data(&self) -> &Array<T, D> {
        &self.data
    }

    /// Return a reference to the mask array.
    #[inline]
    pub fn mask(&self) -> &Array<bool, D> {
        &self.mask
    }

    /// Return a mutable reference to the underlying data array.
    #[inline]
    pub fn data_mut(&mut self) -> &mut Array<T, D> {
        &mut self.data
    }

    /// Return the shape of the masked array.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Return the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.data.ndim()
    }

    /// Return the total number of elements (including masked).
    #[inline]
    pub fn size(&self) -> usize {
        self.data.size()
    }

    /// Return the dimension descriptor.
    #[inline]
    pub fn dim(&self) -> &D {
        self.data.dim()
    }

    /// Return whether the mask is hardened.
    #[inline]
    pub fn is_hard_mask(&self) -> bool {
        self.hard_mask
    }

    /// Set a mask value at a flat index.
    ///
    /// If the mask is hardened, only `true` (masking) is allowed; attempts to
    /// clear a mask bit are silently ignored.
    ///
    /// # Errors
    /// Returns `FerrayError::IndexOutOfBounds` if `flat_idx >= size`.
    pub fn set_mask_flat(&mut self, flat_idx: usize, value: bool) -> FerrayResult<()> {
        let size = self.size();
        if flat_idx >= size {
            return Err(FerrayError::index_out_of_bounds(flat_idx as isize, 0, size));
        }
        if self.hard_mask && !value {
            // Hard mask: cannot clear mask bits
            return Ok(());
        }
        // Fast path: contiguous mask — direct O(1) slice indexing
        if let Some(slice) = self.mask.as_slice_mut() {
            slice[flat_idx] = value;
        } else {
            // Non-contiguous: fall back to iterator (rare case)
            if let Some(m) = self.mask.iter_mut().nth(flat_idx) {
                *m = value;
            }
        }
        Ok(())
    }

    /// Replace the mask with a new one.
    ///
    /// If the mask is hardened, only bits that are `true` in both the old and
    /// new masks (or newly set to `true`) are allowed; cleared bits are ignored.
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if shapes differ.
    pub fn set_mask(&mut self, new_mask: Array<bool, D>) -> FerrayResult<()> {
        if self.mask.shape() != new_mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "set_mask: mask shape {:?} does not match array shape {:?}",
                new_mask.shape(),
                self.mask.shape()
            )));
        }
        if self.hard_mask {
            // Union: keep old trues, add new trues, but never clear
            let merged: Vec<bool> = self
                .mask
                .iter()
                .zip(new_mask.iter())
                .map(|(old, new)| *old || *new)
                .collect();
            self.mask = Array::from_vec(self.mask.dim().clone(), merged)?;
        } else {
            self.mask = new_mask;
        }
        Ok(())
    }
}
