// ferray-ma: in-place flat assignment ops — `put` / `putmask` (REQ-36, REQ-37)
//
// These two operations mutate a `MaskedArray` in place by writing data and
// updating the mask at selected positions, honouring the array's hard-mask
// state. They mirror `numpy.ma.MaskedArray.put` (`numpy/ma/core.py:4837`) and
// `numpy.ma.putmask` (`numpy/ma/core.py:7537`) exactly.
//
// Both build on the existing in-place mutation surface in `masked_array.rs`
// (`data_mut`, `set_mask_flat`, `set_mask`, `ensure_materialized_mut`) — no
// new struct field is required because the hard-mask flag and its assignment
// effect already exist.

use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

/// Out-of-bounds index behaviour for [`MaskedArray::put`], mirroring numpy's
/// `mode` keyword (`{'raise', 'wrap', 'clip'}`, `numpy/ma/core.py:4852`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PutMode {
    /// `'raise'` — an out-of-range index returns `IndexOutOfBounds`.
    Raise,
    /// `'wrap'` — indices wrap around modulo the length (Python modulo, so a
    /// negative index wraps from the end).
    Wrap,
    /// `'clip'` — indices are clamped into `[0, len)`; negatives clip to `0`.
    Clip,
}

impl PutMode {
    /// Parse the numpy `mode` string. An unrecognised mode mirrors numpy's
    /// `ValueError` (raised by `np.put` for a bad mode).
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidValue` for any string other than
    /// `"raise"`, `"wrap"`, or `"clip"`.
    pub fn parse(s: &str) -> FerrayResult<Self> {
        match s {
            "raise" => Ok(Self::Raise),
            "wrap" => Ok(Self::Wrap),
            "clip" => Ok(Self::Clip),
            other => Err(FerrayError::invalid_value(format!(
                "put: clipmode '{other}' not understood (expected 'raise', 'wrap', or 'clip')"
            ))),
        }
    }
}

/// Resolve a possibly-negative index against `len` under `mode`, returning the
/// concrete flat position in `[0, len)`.
///
/// Mirrors numpy's `PyArray_PutTo` index resolution: `'raise'` accepts the
/// Python range `[-len, len)` and errors otherwise; `'wrap'` applies Python
/// modulo so any integer maps in-range; `'clip'` clamps, with negatives
/// (and any index into a zero-length array) clipping to `0`.
fn resolve_index(idx: isize, len: usize, mode: PutMode) -> FerrayResult<usize> {
    let len_i = len as isize;
    match mode {
        PutMode::Raise => {
            let resolved = if idx < 0 { idx + len_i } else { idx };
            if resolved < 0 || resolved >= len_i {
                return Err(FerrayError::index_out_of_bounds(idx, 0, len));
            }
            Ok(resolved as usize)
        }
        PutMode::Wrap => {
            if len == 0 {
                return Err(FerrayError::index_out_of_bounds(idx, 0, len));
            }
            Ok(idx.rem_euclid(len_i) as usize)
        }
        PutMode::Clip => {
            if len == 0 {
                return Err(FerrayError::index_out_of_bounds(idx, 0, len));
            }
            if idx < 0 {
                Ok(0)
            } else if idx >= len_i {
                Ok(len - 1)
            } else {
                Ok(idx as usize)
            }
        }
    }
}

impl<T: Element + Copy, D: Dimension> MaskedArray<T, D> {
    /// Set flat-indexed positions to corresponding values, mirroring
    /// `numpy.ma.MaskedArray.put` (`numpy/ma/core.py:4837`).
    ///
    /// `self._data.flat[indices[n]] = values[n]` for each `n`; if `values` is
    /// shorter than `indices` it repeats (numpy resizes `values` to the index
    /// shape). After the data write the mask is updated:
    ///
    /// - if `values_mask` is `None` (an unmasked `values`), every written
    ///   position is **unmasked** (`numpy/ma/core.py:4916`);
    /// - if `values_mask` is `Some`, the written positions take the
    ///   corresponding `values` mask bit (`:4918`).
    ///
    /// When the array is hard-masked, every `(index, value)` pair whose target
    /// position is currently masked is **dropped before the data write**
    /// (`numpy/ma/core.py:4899`): such positions are neither overwritten nor
    /// unmasked. Mirrors the live numpy oracle: after `a.harden_mask()`,
    /// `a.put([1,3],[10,30])` leaves a masked `a[1]` unchanged.
    ///
    /// `mode` controls out-of-bounds index behaviour; see [`PutMode`].
    ///
    /// # Errors
    /// Returns `FerrayError::IndexOutOfBounds` for an out-of-range index under
    /// [`PutMode::Raise`], `FerrayError::InvalidValue` when `values` is empty
    /// while `indices` is not (numpy raises on an empty `values` resize against
    /// a non-empty index shape), or any error from the underlying mask
    /// materialisation. An empty `indices` is a no-op.
    pub fn put(
        &mut self,
        indices: &[isize],
        values: &[T],
        values_mask: Option<&[bool]>,
        mode: PutMode,
    ) -> FerrayResult<()> {
        if indices.is_empty() {
            // Nothing to do; numpy's empty put is a no-op.
            return Ok(());
        }
        if values.is_empty() {
            return Err(FerrayError::invalid_value(
                "put: cannot place values from an empty `values` array into non-empty indices",
            ));
        }
        let len = self.size();

        // Resolve every index up-front so a `Raise` failure aborts before any
        // mutation (matching numpy, which validates the whole index array).
        let resolved: Vec<usize> = indices
            .iter()
            .map(|&i| resolve_index(i, len, mode))
            .collect::<FerrayResult<_>>()?;

        let hard = self.is_hard_mask() && self.has_real_mask();

        // Snapshot the current mask once (for the hard-mask drop test) so we
        // don't materialise it inside the loop.
        let current_mask: Option<Vec<bool>> = if hard {
            Some(self.mask().iter().copied().collect())
        } else {
            None
        };

        // Iterate (index_position, flat_index). `values` repeats cyclically
        // when shorter than `indices` (numpy resizes values to the index
        // shape — a cyclic tile for the common shorter-values case).
        for (n, &flat) in resolved.iter().enumerate() {
            // Hard-mask drop: skip pairs whose target is currently masked.
            if let Some(ref mask) = current_mask {
                if mask[flat] {
                    continue;
                }
            }
            let value = values[n % values.len()];
            // Data write.
            if let Some(slice) = self.data_mut() {
                slice[flat] = value;
            } else {
                return Err(FerrayError::invalid_value(
                    "put: underlying data is not contiguous; cannot place values",
                ));
            }
            // Mask update for this position.
            let new_mask_bit = match values_mask {
                None => false,
                Some(vm) => vm[n % vm.len()],
            };
            self.set_mask_flat(flat, new_mask_bit)?;
        }
        Ok(())
    }

    /// Set positions where `mask` is `true` from `values`, mirroring
    /// `numpy.ma.putmask` (`numpy/ma/core.py:7537`).
    ///
    /// The data is written unconditionally at every `mask`-true position
    /// (numpy's final `np.copyto(a._data, valdata, where=mask)`, `:7590`) —
    /// even hard-masked positions get their **data** overwritten (only the
    /// mask is protected). `values` is broadcast against `self` exactly as
    /// numpy's `copyto` does: it must be either length 1 (a scalar fill) or the
    /// same length as `self` (numpy raises `ValueError` on a mismatched
    /// non-scalar `values`, since `copyto` broadcasts rather than cycles).
    ///
    /// Mask update branches on hard-mask (`numpy/ma/core.py:7581`):
    ///
    /// - **soft + unmasked `values`**: written positions are unmasked (numpy
    ///   copies `getmaskarray(values)`, all-`False`, where `mask`);
    /// - **soft + masked `values`**: written positions take the `values` mask;
    /// - **hard + masked `values`**: the `values` mask is **unioned** into the
    ///   existing mask where `mask` is true — never cleared (`:7585`);
    /// - **hard + unmasked `values`**: the mask is left unchanged (`:7582`
    ///   `if valmask is not nomask`).
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if `mask.len()` differs from
    /// `self.size()`, or `FerrayError::InvalidValue` if `values` is neither
    /// length 1 nor length `self.size()` (numpy's broadcast failure).
    pub fn putmask(
        &mut self,
        mask: &[bool],
        values: &[T],
        values_mask: Option<&[bool]>,
    ) -> FerrayResult<()> {
        let len = self.size();
        if mask.len() != len {
            return Err(FerrayError::shape_mismatch(format!(
                "putmask: boolean mask length {} does not match array size {len}",
                mask.len()
            )));
        }
        if values.is_empty() {
            return Err(FerrayError::invalid_value(
                "putmask: `values` must not be empty",
            ));
        }
        // numpy's copyto broadcasts `values` against `self`: a length-1
        // `values` is a scalar fill; otherwise it must match `self` exactly.
        if values.len() != 1 && values.len() != len {
            return Err(FerrayError::invalid_value(format!(
                "putmask: could not broadcast values of length {} into array of size {len}",
                values.len()
            )));
        }
        let broadcast = |i: usize| -> T {
            if values.len() == 1 {
                values[0]
            } else {
                values[i]
            }
        };
        let vmask_bit = |i: usize| -> bool {
            match values_mask {
                None => false,
                Some(vm) => {
                    if vm.len() == 1 {
                        vm[0]
                    } else {
                        vm[i]
                    }
                }
            }
        };

        let hard = self.is_hard_mask() && self.has_real_mask();

        // ---- Mask update (numpy/ma/core.py:7576-7589) ----
        // Done before the data write; the two touch disjoint buffers so order
        // is immaterial.
        match (hard, values_mask) {
            // soft path: copy the values mask (or all-False) where `mask`.
            (false, _) => {
                for (i, &m) in mask.iter().enumerate() {
                    if m {
                        self.set_mask_flat(i, vmask_bit(i))?;
                    }
                }
            }
            // hard + masked values: union the values mask in where `mask`,
            // never clearing (set_mask_flat already refuses to clear on a hard
            // mask, so a `true` bit is added and a `false` bit is a no-op).
            (true, Some(_)) => {
                for (i, &m) in mask.iter().enumerate() {
                    if m && vmask_bit(i) {
                        self.set_mask_flat(i, true)?;
                    }
                }
            }
            // hard + unmasked values: mask unchanged (numpy's guard skips).
            (true, None) => {}
        }

        // ---- Data write (numpy/ma/core.py:7590) ----
        // Unconditional copyto where `mask`: even hard-masked positions get
        // their data overwritten (only the mask is protected on a hard mask).
        if let Some(slice) = self.data_mut() {
            for (i, &m) in mask.iter().enumerate() {
                if m {
                    slice[i] = broadcast(i);
                }
            }
        } else {
            return Err(FerrayError::invalid_value(
                "putmask: underlying data is not contiguous; cannot place values",
            ));
        }
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

    fn ma(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
        MaskedArray::new(arr_f64(data), arr_bool(mask)).unwrap()
    }

    fn data_of(m: &MaskedArray<f64, Ix1>) -> Vec<f64> {
        m.data().iter().copied().collect()
    }
    fn mask_of(m: &MaskedArray<f64, Ix1>) -> Vec<bool> {
        m.mask().iter().copied().collect()
    }

    // ---- put: basic write unmasks written positions ----
    // Oracle: np.ma.array([1,2,3,4], mask=[F,T,F,F]); a.put([1,3],[10,30])
    //   -> data [1,10,3,30], mask all False.
    #[test]
    fn put_basic_unmasks_written() {
        let mut a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        a.put(&[1, 3], &[10.0, 30.0], None, PutMode::Raise).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 10.0, 3.0, 30.0]);
        assert_eq!(mask_of(&a), vec![false, false, false, false]);
    }

    // ---- put: masked values propagate to the target mask ----
    // Oracle: a=[1,2,3,4] nomask; vals=ma([10,20], mask=[T,F]); a.put([0,1],vals)
    //   -> data [10,20,3,4], mask [T,F,F,F].
    #[test]
    fn put_masked_values_propagate() {
        let mut a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        a.put(&[0, 1], &[10.0, 20.0], Some(&[true, false]), PutMode::Raise)
            .unwrap();
        assert_eq!(data_of(&a), vec![10.0, 20.0, 3.0, 4.0]);
        assert_eq!(mask_of(&a), vec![true, false, false, false]);
    }

    // ---- put: values repeat (cycle) when shorter than indices ----
    // Oracle: a=[1..5] nomask; a.put([0,1,2,3],[9]) -> data [9,9,9,9,5].
    #[test]
    fn put_values_repeat() {
        let mut a = ma(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![false; 5]);
        a.put(&[0, 1, 2, 3], &[9.0], None, PutMode::Raise).unwrap();
        assert_eq!(data_of(&a), vec![9.0, 9.0, 9.0, 9.0, 5.0]);
    }

    // ---- put: mode wrap ----
    // Oracle: a=[1,2,3] nomask; a.put([5],[99],mode='wrap') (5%3=2) -> [1,2,99].
    #[test]
    fn put_mode_wrap() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        a.put(&[5], &[99.0], None, PutMode::Wrap).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 99.0]);
    }

    // ---- put: mode clip ----
    // Oracle: a=[1,2,3] nomask; a.put([5],[99],mode='clip') (->2) -> [1,2,99].
    #[test]
    fn put_mode_clip() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        a.put(&[5], &[99.0], None, PutMode::Clip).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 99.0]);
    }

    // ---- put: mode clip negative -> 0 ----
    // Oracle: a=[1,2,3] nomask; a.put([-5],[9],mode='clip') -> [9,2,3].
    #[test]
    fn put_mode_clip_negative() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        a.put(&[-5], &[9.0], None, PutMode::Clip).unwrap();
        assert_eq!(data_of(&a), vec![9.0, 2.0, 3.0]);
    }

    // ---- put: mode wrap negative (Python modulo) ----
    // Oracle: a=[1,2,3] nomask; a.put([-5],[9],mode='wrap') (-5 %3 ==1) -> [1,9,3].
    #[test]
    fn put_mode_wrap_negative() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        a.put(&[-5], &[9.0], None, PutMode::Wrap).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 9.0, 3.0]);
    }

    // ---- put: negative index under raise resolves from the end ----
    // Oracle: a=[1,2,3,4] nomask; a.put([-1,-2],[10,20]) -> [1,2,20,10].
    #[test]
    fn put_negative_index_raise() {
        let mut a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        a.put(&[-1, -2], &[10.0, 20.0], None, PutMode::Raise)
            .unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 20.0, 10.0]);
    }

    // ---- put: mode raise out of bounds errors ----
    // Oracle: a=[1,2,3] nomask; a.put([5],[99],mode='raise') -> IndexError.
    #[test]
    fn put_mode_raise_out_of_bounds() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        assert!(a.put(&[5], &[99.0], None, PutMode::Raise).is_err());
    }

    #[test]
    fn put_mode_raise_negative_out_of_bounds() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        assert!(a.put(&[-5], &[9.0], None, PutMode::Raise).is_err());
    }

    // ---- put: hard-mask suppression drops masked targets ----
    // Oracle: a=[1,2,3,4] mask=[F,T,F,F]; a.harden_mask(); a.put([1,3],[10,30])
    //   -> data [1,2,3,30], mask [F,T,F,F] (a[1] never written nor unmasked).
    #[test]
    fn put_hard_mask_suppresses_masked_target() {
        let mut a = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, false]);
        a.harden_mask().unwrap();
        a.put(&[1, 3], &[10.0, 30.0], None, PutMode::Raise).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 3.0, 30.0]);
        assert_eq!(mask_of(&a), vec![false, true, false, false]);
    }

    // ---- put: empty indices is a no-op ----
    #[test]
    fn put_empty_indices_noop() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![false; 3]);
        a.put(&[], &[99.0], None, PutMode::Raise).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 3.0]);
    }

    // ---- putmask: basic, clears mask where written ----
    // Oracle: x=ma([1,2,3,4], mask=[T,F,F,F]);
    //   np.ma.putmask(x,[T,F,T,F],[10,20,30,40])
    //   -> data [10,2,30,4], mask all False.
    #[test]
    fn putmask_basic_clears_mask() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![true, false, false, false]);
        x.putmask(
            &[true, false, true, false],
            &[10.0, 20.0, 30.0, 40.0],
            None,
        )
        .unwrap();
        assert_eq!(data_of(&x), vec![10.0, 2.0, 30.0, 4.0]);
        assert_eq!(mask_of(&x), vec![false, false, false, false]);
    }

    // ---- putmask: scalar values broadcast ----
    // Oracle: x=ma([1,2,3,4], nomask); np.ma.putmask(x,[T,F,T,F],99.)
    //   -> data [99,2,99,4].
    #[test]
    fn putmask_scalar_broadcast() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        x.putmask(&[true, false, true, false], &[99.0], None)
            .unwrap();
        assert_eq!(data_of(&x), vec![99.0, 2.0, 99.0, 4.0]);
        assert_eq!(mask_of(&x), vec![false, false, false, false]);
    }

    // ---- putmask: mismatched non-scalar values raises (copyto broadcast) ----
    // Oracle: np.ma.putmask(x, [T,T,T,T], [9,8]) -> ValueError (broadcast).
    #[test]
    fn putmask_mismatched_values_errors() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        assert!(
            x.putmask(&[true, true, true, true], &[9.0, 8.0], None)
                .is_err()
        );
    }

    // ---- putmask: mask length mismatch errors ----
    #[test]
    fn putmask_mask_length_mismatch_errors() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        assert!(x.putmask(&[true, false], &[9.0], None).is_err());
    }

    // ---- putmask: hard mask keeps existing mask; data still overwritten ----
    // Oracle: x=ma([1,2,3,4], mask=[T,F,F,F]); x.harden_mask();
    //   np.ma.putmask(x,[T,T,F,F],[10,20,30,40])
    //   -> data [10,20,3,4], mask [T,F,F,F] (a[0] data overwritten, mask kept).
    #[test]
    fn putmask_hard_keeps_mask_overwrites_data() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![true, false, false, false]);
        x.harden_mask().unwrap();
        x.putmask(
            &[true, true, false, false],
            &[10.0, 20.0, 30.0, 40.0],
            None,
        )
        .unwrap();
        assert_eq!(data_of(&x), vec![10.0, 20.0, 3.0, 4.0]);
        assert_eq!(mask_of(&x), vec![true, false, false, false]);
    }

    // ---- putmask: soft + masked values propagate (set/clear) ----
    // Oracle: x=ma([1,2,3,4], mask=[F,F,F,T]);
    //   vals=ma([10,20,30,40], mask=[T,F,F,F]);
    //   np.ma.putmask(x,[T,T,F,F],vals)
    //   -> data [10,20,3,4], mask [T,F,F,T].
    #[test]
    fn putmask_soft_masked_values_propagate() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false, false, false, true]);
        x.putmask(
            &[true, true, false, false],
            &[10.0, 20.0, 30.0, 40.0],
            Some(&[true, false, false, false]),
        )
        .unwrap();
        assert_eq!(data_of(&x), vec![10.0, 20.0, 3.0, 4.0]);
        assert_eq!(mask_of(&x), vec![true, false, false, true]);
    }

    // ---- putmask: hard + masked values union (never clears) ----
    // Oracle: x=ma([1,2,3,4], nomask); x.harden_mask();
    //   vals=ma([10,20,30,40], mask=[T,F,F,F]);
    //   np.ma.putmask(x,[T,T,F,F],vals)
    //   -> data [10,20,3,4], mask [T,F,F,F].
    #[test]
    fn putmask_hard_masked_values_union() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        x.harden_mask().unwrap();
        x.putmask(
            &[true, true, false, false],
            &[10.0, 20.0, 30.0, 40.0],
            Some(&[true, false, false, false]),
        )
        .unwrap();
        assert_eq!(data_of(&x), vec![10.0, 20.0, 3.0, 4.0]);
        assert_eq!(mask_of(&x), vec![true, false, false, false]);
    }

    // ---- putmask: hard pos already masked -> data still overwritten ----
    // Oracle: x=ma([1,2,3,4], mask=[T,F,F,F]); x.harden_mask();
    //   np.ma.putmask(x,[T,F,F,F],[10,20,30,40])
    //   -> data [10,2,3,4], mask [T,F,F,F].
    #[test]
    fn putmask_hard_masked_target_data_overwritten() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![true, false, false, false]);
        x.harden_mask().unwrap();
        x.putmask(&[true, false, false, false], &[10.0], None)
            .unwrap();
        assert_eq!(data_of(&x), vec![10.0, 2.0, 3.0, 4.0]);
        assert_eq!(mask_of(&x), vec![true, false, false, false]);
    }

    // ---- PutMode::from_str ----
    #[test]
    fn put_mode_parse() {
        assert_eq!(PutMode::parse("raise").unwrap(), PutMode::Raise);
        assert_eq!(PutMode::parse("wrap").unwrap(), PutMode::Wrap);
        assert_eq!(PutMode::parse("clip").unwrap(), PutMode::Clip);
        assert!(PutMode::parse("nonsense").is_err());
    }
}
