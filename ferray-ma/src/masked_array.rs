// ferray-ma: MaskedArray<T, D> type (REQ-1, REQ-2, REQ-3)

use std::sync::OnceLock;

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
///
/// # Nomask sentinel (#506)
///
/// When a [`MaskedArray`] is constructed via [`MaskedArray::from_data`]
/// the mask is logically "all-false" but is NOT allocated as a full
/// `Array<bool, D>` up front — the lazy `OnceLock` inside stores
/// nothing until the first call to [`MaskedArray::mask`]. For arrays
/// that never touch their mask (e.g. masked ops that short-circuit
/// via [`MaskedArray::has_real_mask`]), this saves a full bool-sized
/// allocation proportional to the data size.
///
/// The `.mask()` accessor still returns `&Array<bool, D>` so all
/// existing code continues to work unchanged; the cost is one
/// lazy allocation on first access. Hot-path code that wants to
/// avoid the materialization should check `has_real_mask()` first
/// and skip any mask work when it returns `false`.
pub struct MaskedArray<T: Element, D: Dimension> {
    /// The underlying data array.
    data: Array<T, D>,
    /// Boolean mask (`true` = masked/invalid). Lazily materialized
    /// when explicitly queried via [`MaskedArray::mask`] — a
    /// `from_data`-constructed array with no masked elements pays
    /// zero allocation cost until that first query.
    mask: OnceLock<Array<bool, D>>,
    /// `true` when a non-trivial mask has been explicitly provided
    /// (via [`MaskedArray::new`] or [`MaskedArray::set_mask`]),
    /// `false` when the array is in the nomask-sentinel state.
    ///
    /// Hot-path consumers should branch on this flag and skip the
    /// mask-iteration entirely when it is `false` — see
    /// [`MaskedArray::has_real_mask`].
    real_mask: bool,
    /// Whether the mask is hardened (cannot be cleared by assignment).
    pub(crate) hard_mask: bool,
    /// Replacement value for masked positions during operations and filling.
    /// Defaults to `T::zero()`.
    pub(crate) fill_value: T,
}

impl<T: Element, D: Dimension> std::fmt::Debug for MaskedArray<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaskedArray")
            .field("data", &self.data)
            .field("real_mask", &self.real_mask)
            .field("hard_mask", &self.hard_mask)
            .field("fill_value", &self.fill_value)
            .finish_non_exhaustive()
    }
}

impl<T: Element + Clone, D: Dimension> Clone for MaskedArray<T, D> {
    fn clone(&self) -> Self {
        // Clone the lazy state: if the original already materialized a
        // mask, the clone gets a fresh OnceLock already initialized to
        // a copy. If the original is still in the nomask-sentinel state
        // (OnceLock empty + `real_mask == false`), the clone stays in
        // the same state so it can also skip the allocation.
        let new_lock = OnceLock::new();
        if let Some(m) = self.mask.get() {
            let _ = new_lock.set(m.clone());
        }
        Self {
            data: self.data.clone(),
            mask: new_lock,
            real_mask: self.real_mask,
            hard_mask: self.hard_mask,
            fill_value: self.fill_value.clone(),
        }
    }
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
        let lock = OnceLock::new();
        let _ = lock.set(mask);
        Ok(Self {
            data,
            mask: lock,
            real_mask: true,
            hard_mask: false,
            fill_value: T::zero(),
        })
    }

    /// Create a masked array with no masked elements (all-false mask).
    ///
    /// Does NOT allocate the mask up front — the array is in the
    /// nomask-sentinel state (#506) until [`MaskedArray::mask`] is
    /// explicitly called. For code that only uses `data()` or
    /// short-circuits via [`MaskedArray::has_real_mask`], this saves
    /// a full-sized bool allocation.
    ///
    /// # Errors
    /// Always returns `Ok` — the `FerrayResult` is preserved for API
    /// parity with the previous eager implementation.
    pub fn from_data(data: Array<T, D>) -> FerrayResult<Self> {
        Ok(Self {
            data,
            mask: OnceLock::new(),
            real_mask: false,
            hard_mask: false,
            fill_value: T::zero(),
        })
    }

    /// Return `true` if this masked array holds a real (explicitly
    /// provided or materialized) mask. Returns `false` when the array
    /// is in the nomask-sentinel state and the mask is logically
    /// all-false.
    ///
    /// Hot-path iteration code should branch on this flag to skip
    /// mask scanning entirely when it returns `false` (#506).
    #[inline]
    pub fn has_real_mask(&self) -> bool {
        self.real_mask
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
    ///
    /// If the array is in the nomask-sentinel state (constructed via
    /// [`MaskedArray::from_data`] or otherwise) this lazily allocates
    /// a full all-false `Array<bool, D>` and caches it for subsequent
    /// calls. Use [`MaskedArray::has_real_mask`] to check whether the
    /// mask is known to be trivial first, and skip calling `.mask()`
    /// entirely on the hot path when you can.
    pub fn mask(&self) -> &Array<bool, D> {
        self.mask.get_or_init(|| {
            Array::<bool, D>::from_elem(self.data.dim().clone(), false)
                .expect("from_elem with matching dim cannot fail")
        })
    }

    /// Return a reference to the mask array if one has been
    /// materialized, or `None` when the array is still in the
    /// nomask-sentinel state.
    ///
    /// Unlike [`MaskedArray::mask`], this does NOT trigger lazy
    /// allocation — it's the fast-path query for hot code that
    /// wants to branch on whether any mask bits are set (#506).
    #[inline]
    pub fn mask_opt(&self) -> Option<&Array<bool, D>> {
        if self.real_mask {
            // A real mask was set via `new` or `set_mask`; the
            // OnceLock is guaranteed to be initialized.
            self.mask.get()
        } else {
            None
        }
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

    /// Internal helper: force the lazy nomask sentinel to materialize a
    /// concrete `Array<bool, D>` and return a mutable reference to it.
    ///
    /// After this call `real_mask` is `true` and `self.mask` is
    /// guaranteed to contain an initialized `Array<bool, D>`. Used by
    /// `set_mask_flat` and the crate-internal paths that need an owned
    /// mask to mutate.
    fn ensure_materialized_mut(&mut self) -> &mut Array<bool, D> {
        if !self.real_mask || self.mask.get().is_none() {
            // Create a fresh all-false mask and install it.
            let fresh = Array::<bool, D>::from_elem(self.data.dim().clone(), false)
                .expect("from_elem with matching dim cannot fail");
            // Replace the OnceLock entirely — get_mut() returns None
            // if the cell is empty, so we can't just .set() + .get_mut().
            self.mask = OnceLock::new();
            let _ = self.mask.set(fresh);
            self.real_mask = true;
        }
        self.mask
            .get_mut()
            .expect("OnceLock was just initialized above")
    }

    /// Set a mask value at a flat index.
    ///
    /// If the mask is hardened, only `true` (masking) is allowed; attempts to
    /// clear a mask bit are silently ignored.
    ///
    /// Setting a mask bit materializes the lazy nomask sentinel into a
    /// real mask array (#506) — if you set even one bit, the full
    /// `Array<bool, D>` is allocated.
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
        // Setting a nomask-sentinel to false is a no-op (mask is
        // already logically all-false); skip the allocation entirely.
        if !self.real_mask && !value {
            return Ok(());
        }
        let mask = self.ensure_materialized_mut();
        // Fast path: contiguous mask — direct O(1) slice indexing.
        if let Some(slice) = mask.as_slice_mut() {
            slice[flat_idx] = value;
        } else {
            // Non-contiguous: fall back to iterator (rare case).
            if let Some(m) = mask.iter_mut().nth(flat_idx) {
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
    /// Passing a new mask always materializes the array out of the
    /// nomask-sentinel state — the stored mask becomes the provided
    /// one (possibly unioned with the existing mask if hardened).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if shapes differ.
    pub fn set_mask(&mut self, new_mask: Array<bool, D>) -> FerrayResult<()> {
        if self.data.shape() != new_mask.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "set_mask: mask shape {:?} does not match array shape {:?}",
                new_mask.shape(),
                self.data.shape()
            )));
        }
        if self.hard_mask && self.real_mask {
            // Hard-mask union: merge the new mask with the existing
            // one, keeping any `true` bits and never clearing.
            let existing = self.mask.get().expect("real_mask implies OnceLock set");
            let merged: Vec<bool> = existing
                .iter()
                .zip(new_mask.iter())
                .map(|(old, new)| *old || *new)
                .collect();
            let merged_arr = Array::from_vec(self.data.dim().clone(), merged)?;
            self.mask = OnceLock::new();
            let _ = self.mask.set(merged_arr);
        } else {
            // Either not hardened or currently in the nomask sentinel
            // state — unconditionally install the new mask.
            self.mask = OnceLock::new();
            let _ = self.mask.set(new_mask);
        }
        self.real_mask = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;
    use ferray_core::dimension::Ix1;

    fn arr_f64(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr_bool(data: Vec<bool>) -> Array<bool, Ix1> {
        let n = data.len();
        Array::<bool, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
    }

    // ---- nomask sentinel (#506) ----

    #[test]
    fn from_data_starts_in_nomask_sentinel_state() {
        let ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        assert!(!ma.has_real_mask());
        assert!(ma.mask_opt().is_none());
    }

    #[test]
    fn new_with_explicit_mask_is_real_mask() {
        let ma = MaskedArray::new(
            arr_f64(vec![1.0, 2.0, 3.0]),
            arr_bool(vec![false, true, false]),
        )
        .unwrap();
        assert!(ma.has_real_mask());
        assert!(ma.mask_opt().is_some());
    }

    #[test]
    fn mask_accessor_lazily_materializes_nomask_sentinel() {
        let ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        // Before calling .mask(), the OnceLock is empty.
        assert!(ma.mask_opt().is_none());
        // After calling .mask(), we get a full all-false Array<bool, D>.
        let m = ma.mask();
        assert_eq!(m.shape(), &[3]);
        assert_eq!(m.iter().copied().collect::<Vec<_>>(), vec![false, false, false]);
        // Subsequent calls return the same cached array (no re-alloc).
        let m2 = ma.mask();
        assert_eq!(m as *const _, m2 as *const _);
        // BUT `has_real_mask` still reports `false` — the lazy
        // materialization doesn't promote the sentinel to a "real" mask
        // because the contents are still logically all-false. Hot-path
        // code can keep skipping.
        assert!(!ma.has_real_mask());
    }

    #[test]
    fn set_mask_flat_false_on_nomask_stays_zero_allocation() {
        // Setting a position to false on a nomask-sentinel array is a
        // no-op and should NOT materialize the mask.
        let mut ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        ma.set_mask_flat(1, false).unwrap();
        assert!(!ma.has_real_mask());
        assert!(ma.mask_opt().is_none());
    }

    #[test]
    fn set_mask_flat_true_on_nomask_materializes_and_promotes() {
        // Setting a position to true forces materialization and
        // promotes `real_mask` to true.
        let mut ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        ma.set_mask_flat(1, true).unwrap();
        assert!(ma.has_real_mask());
        let m: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, false]);
    }

    #[test]
    fn set_mask_promotes_and_keeps_provided_values() {
        let mut ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        assert!(!ma.has_real_mask());
        ma.set_mask(arr_bool(vec![true, false, true])).unwrap();
        assert!(ma.has_real_mask());
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, false, true]
        );
    }

    #[test]
    fn set_mask_shape_mismatch_errors() {
        let mut ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        assert!(ma.set_mask(arr_bool(vec![false; 4])).is_err());
    }

    #[test]
    fn clone_preserves_nomask_sentinel_state() {
        let ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        let cloned = ma.clone();
        assert!(!cloned.has_real_mask());
        assert!(cloned.mask_opt().is_none());
    }

    #[test]
    fn clone_after_materialization_copies_the_mask() {
        let ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        // Force materialization.
        let _ = ma.mask();
        let cloned = ma.clone();
        // The clone has the same mask contents (all-false).
        assert_eq!(
            cloned.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, false]
        );
    }

    #[test]
    fn clone_preserves_real_mask_state() {
        let ma = MaskedArray::new(
            arr_f64(vec![1.0, 2.0, 3.0]),
            arr_bool(vec![false, true, false]),
        )
        .unwrap();
        let cloned = ma.clone();
        assert!(cloned.has_real_mask());
        assert_eq!(
            cloned.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    #[test]
    fn hard_mask_union_on_real_mask() {
        let mut ma = MaskedArray::new(
            arr_f64(vec![1.0, 2.0, 3.0]),
            arr_bool(vec![true, false, false]),
        )
        .unwrap();
        ma.harden_mask().unwrap();
        // Try to clear position 0 and set position 2. With a hard
        // mask, the union keeps position 0's true bit.
        ma.set_mask(arr_bool(vec![false, false, true])).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, false, true]
        );
    }
}
