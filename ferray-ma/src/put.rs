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
    /// `self._data.flat[indices[n]] = values[n]` for each `n`. This is a
    /// faithful translation of numpy's two-phase delegation
    /// (`numpy/ma/core.py:4898-4921`): the **data** is written via
    /// `self._data.put(indices, values, mode)` (numpy `ndarray.put`, which
    /// *cycles* a short `values` and is a silent no-op for an empty `values`),
    /// then the **mask** is updated by a *separate* `put` whose value array is
    /// either a scalar `False` (unmasked `values`) or the `values` mask:
    ///
    /// - if `values_mask` is `None` (an unmasked `values`), every written
    ///   position is **unmasked** (`numpy/ma/core.py:4916`,
    ///   `m.put(indices, False, mode)`);
    /// - if `values_mask` is `Some`, the written positions take the
    ///   corresponding `values` mask bit, cycled like numpy's
    ///   `m.put(indices, values._mask, mode)` (`:4918`).
    ///
    /// Because the data and mask are two independent `ndarray.put` calls, an
    /// **empty `values` with non-empty `indices` writes no data** (the data
    /// `put` has nothing to iterate) but **still updates the mask** at every
    /// resolved position (the mask `put` uses the scalar `False` / values-mask,
    /// not the empty `values`). This matches the live oracle
    /// `a.put([0], [])` → data unchanged, `a[0]` unmasked.
    ///
    /// When the array is hard-masked **and carries a real mask**, numpy takes
    /// the hard-mask branch (`numpy/ma/core.py:4899-4905`): `values` is
    /// `resize`d to `indices.shape` — numpy's `ndarray.resize` **zero-pads**
    /// when growing (it does *not* cycle) — then every pair whose target is
    /// masked in the *original* mask is dropped (`indices = indices[~mask]`).
    /// The surviving `(index, zero-padded value)` pairs are then written 1:1.
    /// So a masked target is neither overwritten nor unmasked, and a short
    /// `values` zero-pads the tail rather than cycling.
    ///
    /// `mode` controls out-of-bounds index behaviour; see [`PutMode`].
    ///
    /// # Errors
    /// Returns `FerrayError::IndexOutOfBounds` for an out-of-range index under
    /// [`PutMode::Raise`], or any error from the underlying mask
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
        let len = self.size();

        // Resolve every index up-front so a `Raise` failure aborts before any
        // mutation (matching numpy, which validates the whole index array).
        let resolved: Vec<usize> = indices
            .iter()
            .map(|&i| resolve_index(i, len, mode))
            .collect::<FerrayResult<_>>()?;

        let hard = self.is_hard_mask() && self.has_real_mask();

        // Cycled values-mask bit at index `n` (numpy's `m.put(indices,
        // values._mask)` cycles a short mask). An unmasked or empty
        // values-mask yields `False`.
        let vmask_bit = |n: usize| -> bool {
            match values_mask {
                Some(vm) if !vm.is_empty() => vm[n % vm.len()],
                _ => false,
            }
        };

        if hard {
            // ---- Hard-mask branch (numpy/ma/core.py:4899-4905) ----
            // `values.resize(indices.shape)` zero-pads a short `values`; the
            // surviving pairs are those whose target is unmasked in the
            // *original* mask. Snapshot that mask once before any mutation.
            let original_mask: Vec<bool> = self.mask().iter().copied().collect();
            for (n, &flat) in resolved.iter().enumerate() {
                // Drop pairs whose target is masked in the original mask.
                if original_mask[flat] {
                    continue;
                }
                // Zero-pad: positions past the end of `values` write the
                // type's zero (numpy `ndarray.resize` pads with zeros).
                let value = if n < values.len() {
                    values[n]
                } else {
                    T::zero()
                };
                if let Some(slice) = self.data_mut() {
                    slice[flat] = value;
                } else {
                    return Err(FerrayError::invalid_value(
                        "put: underlying data is not contiguous; cannot place values",
                    ));
                }
                // Mask update for this surviving position. The hard mask cannot
                // be cleared, so an unmasked `values` (False) leaves a masked
                // target alone — but masked targets are already dropped above,
                // so only currently-unmasked positions reach here.
                self.set_mask_flat(flat, vmask_bit(n))?;
            }
            return Ok(());
        }

        // ---- Non-hard branch (numpy/ma/core.py:4907-4920) ----
        // Phase 1: `self._data.put(indices, values, mode)`. numpy's
        // `ndarray.put` cycles a short `values` and is a silent no-op for an
        // empty `values`, so skip the data write entirely when `values` is
        // empty (the mask phase still runs).
        if !values.is_empty() {
            for (n, &flat) in resolved.iter().enumerate() {
                let value = values[n % values.len()];
                if let Some(slice) = self.data_mut() {
                    slice[flat] = value;
                } else {
                    return Err(FerrayError::invalid_value(
                        "put: underlying data is not contiguous; cannot place values",
                    ));
                }
            }
        }

        // Phase 2: mask update (`numpy/ma/core.py:4910-4920`). A separate
        // `ndarray.put` whose value array is the scalar `False` (unmasked
        // `values`) or the cycled `values` mask — independent of whether the
        // data `values` was empty.
        for (n, &flat) in resolved.iter().enumerate() {
            self.set_mask_flat(flat, vmask_bit(n))?;
        }
        Ok(())
    }

    /// Set positions where `mask` is `true` from `values`, mirroring
    /// `numpy.ma.putmask` (`numpy/ma/core.py:7537`).
    ///
    /// The data is written unconditionally at every `mask`-true position
    /// (numpy's final `np.copyto(a._data, valdata, where=mask)`, `:7590`) —
    /// even hard-masked positions get their **data** overwritten (only the
    /// mask is protected). Both `mask` (the `where=` argument) and `values`
    /// are **broadcast** against `self` exactly as numpy's `copyto` does: a
    /// length-1 `mask`/`values` broadcasts across every position, otherwise it
    /// must match `self`'s length (numpy raises `ValueError` on any other
    /// length, since `copyto` broadcasts rather than cycles).
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
    /// Returns `FerrayError::ShapeMismatch` if `mask.len()` is neither 1 nor
    /// `self.size()` (numpy's `where=` broadcast failure), or
    /// `FerrayError::InvalidValue` if `values` is neither length 1 nor length
    /// `self.size()` (numpy's `values` broadcast failure).
    pub fn putmask(
        &mut self,
        mask: &[bool],
        values: &[T],
        values_mask: Option<&[bool]>,
    ) -> FerrayResult<()> {
        let len = self.size();
        // numpy's copyto broadcasts the `where=` mask against `self`: a
        // length-1 mask broadcasts across every position; otherwise it must
        // match `self` exactly (any other length is a broadcast error).
        if mask.len() != 1 && mask.len() != len {
            return Err(FerrayError::shape_mismatch(format!(
                "putmask: boolean mask length {} does not broadcast to array size {len}",
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
        // Broadcast the `where=` mask: a length-1 mask applies to all `len`
        // positions; otherwise it is read 1:1.
        let mask_bit = |i: usize| -> bool { if mask.len() == 1 { mask[0] } else { mask[i] } };
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
                for i in 0..len {
                    if mask_bit(i) {
                        self.set_mask_flat(i, vmask_bit(i))?;
                    }
                }
            }
            // hard + masked values: union the values mask in where `mask`,
            // never clearing (set_mask_flat already refuses to clear on a hard
            // mask, so a `true` bit is added and a `false` bit is a no-op).
            (true, Some(_)) => {
                for i in 0..len {
                    if mask_bit(i) && vmask_bit(i) {
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
            for (i, cell) in slice.iter_mut().enumerate() {
                if mask_bit(i) {
                    *cell = broadcast(i);
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

    // ---- put: empty values leaves DATA unchanged but unmasks targets (D3) ----
    // numpy delegates the data write to `ndarray.put(indices, values)`, a
    // silent no-op for an empty `values`; the mask is updated by a SEPARATE
    // `m.put(indices, False)` so the targeted positions are unmasked.
    // Oracle: a=np.ma.array([1.,2,3], mask=[1,0,0]); a.put([0], [])
    //   -> data [1,2,3] (unchanged), mask [F,F,F] (a[0] unmasked).
    #[test]
    fn put_empty_values_data_noop_unmasks_target() {
        let mut a = ma(vec![1.0, 2.0, 3.0], vec![true, false, false]);
        a.put(&[0], &[], None, PutMode::Raise).unwrap();
        assert_eq!(data_of(&a), vec![1.0, 2.0, 3.0]);
        assert_eq!(mask_of(&a), vec![false, false, false]);
    }

    // ---- put: hard-mask short values zero-pad, not cycle (D1) ----
    // numpy hard-mask branch: values.resize(indices.shape) ZERO-PADS, then
    // filters indices/values by the original ~mask.
    // Oracle: a=np.ma.array([1.,2,3,4,5], mask=[0,1,0,0,0]); a.harden_mask();
    //   a.put([0,1,2],[10,20]):
    //     resize([10,20]->(3,)) = [10,20,0]; ~mask over idx[0,1,2]=[F,T,F]
    //     -> keep idx[0,2], values[10,0] -> data [10,2,0,4,5], mask unchanged.
    #[test]
    fn put_hard_mask_short_values_zeropad() {
        let mut a = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, true, false, false, false],
        );
        a.harden_mask().unwrap();
        a.put(&[0, 1, 2], &[10.0, 20.0], None, PutMode::Raise)
            .unwrap();
        assert_eq!(data_of(&a), vec![10.0, 2.0, 0.0, 4.0, 5.0]);
        assert_eq!(mask_of(&a), vec![false, true, false, false, false]);
    }

    // ---- put: hard-mask, no masked target among indices, still zero-pads ----
    // Oracle: a=np.ma.array([1.,2,3,4,5], mask=[0,1,0,0,0]); a.harden_mask();
    //   a.put([0,2,4],[10,20]) -> resize->[10,20,0]; all ~mask kept
    //   -> data [10,2,20,4,0].
    #[test]
    fn put_hard_mask_zeropad_all_kept() {
        let mut a = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, true, false, false, false],
        );
        a.harden_mask().unwrap();
        a.put(&[0, 2, 4], &[10.0, 20.0], None, PutMode::Raise)
            .unwrap();
        assert_eq!(data_of(&a), vec![10.0, 2.0, 20.0, 4.0, 0.0]);
    }

    // ---- put: hard-mask, values longer than indices (resize truncates) ----
    // Oracle: a=np.ma.array([1.,2,3,4,5], mask=[0,1,0,0,0]); a.harden_mask();
    //   a.put([0,1,2],[10,20,30,40,50]) -> resize to (3,)=[10,20,30];
    //   keep idx[0,2] vals[10,30] -> data [10,2,30,4,5].
    #[test]
    fn put_hard_mask_long_values_truncate() {
        let mut a = ma(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![false, true, false, false, false],
        );
        a.harden_mask().unwrap();
        a.put(
            &[0, 1, 2],
            &[10.0, 20.0, 30.0, 40.0, 50.0],
            None,
            PutMode::Raise,
        )
        .unwrap();
        assert_eq!(data_of(&a), vec![10.0, 2.0, 30.0, 4.0, 5.0]);
    }

    // ---- putmask: basic, clears mask where written ----
    // Oracle: x=ma([1,2,3,4], mask=[T,F,F,F]);
    //   np.ma.putmask(x,[T,F,T,F],[10,20,30,40])
    //   -> data [10,2,30,4], mask all False.
    #[test]
    fn putmask_basic_clears_mask() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![true, false, false, false]);
        x.putmask(&[true, false, true, false], &[10.0, 20.0, 30.0, 40.0], None)
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

    // ---- putmask: length-1 mask broadcasts across all positions (D2) ----
    // numpy's `np.copyto(_data, valdata, where=mask)` broadcasts the `where=`
    // mask, so a length-1 True mask writes every position.
    // Oracle: a=np.ma.array([1.,2,3,4]); np.ma.putmask(a,[True],[9])
    //   -> [9,9,9,9].
    #[test]
    fn putmask_length1_mask_broadcasts() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        x.putmask(&[true], &[9.0], None).unwrap();
        assert_eq!(data_of(&x), vec![9.0, 9.0, 9.0, 9.0]);
    }

    // ---- putmask: length-1 False mask writes nothing ----
    // Oracle: a=np.ma.array([1.,2,3,4]); np.ma.putmask(a,[False],[9])
    //   -> [1,2,3,4].
    #[test]
    fn putmask_length1_false_mask_noop() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        x.putmask(&[false], &[9.0], None).unwrap();
        assert_eq!(data_of(&x), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ---- putmask: length-1 masked values broadcast their mask too (D2) ----
    // Oracle: a=np.ma.array([1.,2,3,4]); v=ma.array([9],mask=[1]);
    //   np.ma.putmask(a,[True],v) -> data [9,9,9,9], mask [T,T,T,T].
    #[test]
    fn putmask_length1_mask_broadcasts_values_mask() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![false; 4]);
        x.putmask(&[true], &[9.0], Some(&[true])).unwrap();
        assert_eq!(data_of(&x), vec![9.0, 9.0, 9.0, 9.0]);
        assert_eq!(mask_of(&x), vec![true, true, true, true]);
    }

    // ---- putmask: hard mask keeps existing mask; data still overwritten ----
    // Oracle: x=ma([1,2,3,4], mask=[T,F,F,F]); x.harden_mask();
    //   np.ma.putmask(x,[T,T,F,F],[10,20,30,40])
    //   -> data [10,20,3,4], mask [T,F,F,F] (a[0] data overwritten, mask kept).
    #[test]
    fn putmask_hard_keeps_mask_overwrites_data() {
        let mut x = ma(vec![1.0, 2.0, 3.0, 4.0], vec![true, false, false, false]);
        x.harden_mask().unwrap();
        x.putmask(&[true, true, false, false], &[10.0, 20.0, 30.0, 40.0], None)
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
