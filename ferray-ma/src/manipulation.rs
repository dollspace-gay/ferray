// ferray-ma: Shape manipulation and indexing for MaskedArray (#507, #508)
//
// Shape ops (reshape, ravel, transpose, flatten, squeeze) delegate to the
// corresponding ferray-core `manipulation` functions applied to both the
// data and mask in parallel. Indexing (index_axis, slice_axis, get,
// boolean_index) returns new MaskedArrays with the corresponding mask
// slices.

use ferray_core::Array;
use ferray_core::dimension::{Dimension, Ix1, IxDyn};
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::manipulation;

use crate::MaskedArray;

// ---------------------------------------------------------------------------
// Shape operations (#508)
// ---------------------------------------------------------------------------

impl<T: Element + Copy, D: Dimension> MaskedArray<T, D> {
    /// Return a new `MaskedArray` with the given shape.
    ///
    /// Equivalent to `numpy.ma.reshape(a, new_shape)`. The result is a
    /// dynamic-rank `MaskedArray<T, IxDyn>`. The total element count
    /// must match; mask and data are reshaped in lockstep so their
    /// logical positions remain aligned. The `fill_value` and
    /// `hard_mask` flag are preserved.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if `new_shape`'s product does not
    ///   equal `self.size()`.
    pub fn reshape(&self, new_shape: &[usize]) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let data = manipulation::reshape(self.data(), new_shape)?;
        let mask = manipulation::reshape(self.mask(), new_shape)?;
        let mut out = MaskedArray::new(data, mask)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }

    /// Return a 1-D `MaskedArray` with the same total element count.
    ///
    /// Equivalent to `numpy.ma.ravel(a)`. Preserves `fill_value` and
    /// `hard_mask`.
    pub fn ravel(&self) -> FerrayResult<MaskedArray<T, Ix1>> {
        let data = manipulation::ravel(self.data())?;
        let mask = manipulation::ravel(self.mask())?;
        let mut out = MaskedArray::new(data, mask)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }

    /// Alias for [`MaskedArray::ravel`].
    ///
    /// Equivalent to `numpy.ma.flatten(a)`.
    pub fn flatten(&self) -> FerrayResult<MaskedArray<T, Ix1>> {
        self.ravel()
    }

    /// Return a transposed `MaskedArray` by permuting axes.
    ///
    /// When `axes` is `None`, reverses the axis order (equivalent to
    /// `numpy.ma.transpose(a)` or `a.T`). When `Some(ax)`, uses the
    /// supplied permutation. Both data and mask are permuted together
    /// so their logical positions remain aligned.
    ///
    /// Returns a dynamic-rank result because the permutation is chosen
    /// at runtime.
    ///
    /// # Errors
    /// - `FerrayError::InvalidValue` if `axes` is the wrong length or
    ///   not a valid permutation.
    pub fn transpose(&self, axes: Option<&[usize]>) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let data = manipulation::transpose(self.data(), axes)?;
        let mask = manipulation::transpose(self.mask(), axes)?;
        let mut out = MaskedArray::new(data, mask)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }

    /// Return a transposed `MaskedArray` with reversed axis order.
    ///
    /// Shorthand for `self.transpose(None)`, equivalent to NumPy's
    /// `.T` property.
    pub fn t(&self) -> FerrayResult<MaskedArray<T, IxDyn>> {
        self.transpose(None)
    }

    /// Remove size-1 axes from the masked array.
    ///
    /// Equivalent to `numpy.ma.squeeze(a)`. When `axis` is `None`,
    /// removes every axis whose length is 1. When `Some(ax)`, only
    /// the specified axis (which must have length 1) is removed.
    ///
    /// The single-axis restriction mirrors `ferray_core::manipulation::squeeze`
    /// — chain multiple calls if you need to drop several axes.
    pub fn squeeze(&self, axis: Option<usize>) -> FerrayResult<MaskedArray<T, IxDyn>> {
        let data = manipulation::squeeze(self.data(), axis)?;
        let mask = manipulation::squeeze(self.mask(), axis)?;
        let mut out = MaskedArray::new(data, mask)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Indexing and slicing (#507)
// ---------------------------------------------------------------------------

impl<T: Element + Copy, D: Dimension> MaskedArray<T, D> {
    /// Get a single element by flat (row-major) index.
    ///
    /// Returns `Ok((value, is_masked))` where `is_masked` is `true`
    /// when the mask bit at that position is set. Returns an error if
    /// the index is out of bounds.
    ///
    /// Returning the raw `(T, bool)` pair lets callers decide what to
    /// do with masked values instead of forcing a fill-value
    /// substitution — if you want the NumPy-style masked scalar
    /// behavior, check `is_masked` and fall back to `self.fill_value()`.
    ///
    /// # Errors
    /// - `FerrayError::IndexOutOfBounds` if `flat_idx >= self.size()`.
    pub fn get_flat(&self, flat_idx: usize) -> FerrayResult<(T, bool)> {
        let size = self.size();
        if flat_idx >= size {
            return Err(FerrayError::index_out_of_bounds(flat_idx as isize, 0, size));
        }
        // Fast path: contiguous buffers give O(1) slice indexing.
        let value = if let Some(s) = self.data().as_slice() {
            s[flat_idx]
        } else {
            self.data().iter().nth(flat_idx).copied().unwrap()
        };
        let is_masked = if let Some(s) = self.mask().as_slice() {
            s[flat_idx]
        } else {
            self.mask().iter().nth(flat_idx).copied().unwrap()
        };
        Ok((value, is_masked))
    }

    /// Select elements where `bool_mask` is `true`.
    ///
    /// Equivalent to `a[bool_mask]` in NumPy boolean indexing. Returns
    /// a 1-D `MaskedArray` containing only positions where the
    /// supplied `bool_mask` is `true`; each selected position carries
    /// through both its value and its original mask bit. The
    /// `bool_mask` must have exactly the same shape as `self`.
    ///
    /// # Errors
    /// - `FerrayError::ShapeMismatch` if `bool_mask.shape() != self.shape()`.
    pub fn boolean_index(
        &self,
        bool_mask: &Array<bool, D>,
    ) -> FerrayResult<MaskedArray<T, Ix1>> {
        if bool_mask.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "boolean_index: selector shape {:?} does not match masked array shape {:?}",
                bool_mask.shape(),
                self.shape()
            )));
        }
        let mut picked_data: Vec<T> = Vec::new();
        let mut picked_mask: Vec<bool> = Vec::new();
        for ((&v, &m_bit), &sel) in self
            .data()
            .iter()
            .zip(self.mask().iter())
            .zip(bool_mask.iter())
        {
            if sel {
                picked_data.push(v);
                picked_mask.push(m_bit);
            }
        }
        let n = picked_data.len();
        let data_arr = Array::<T, Ix1>::from_vec(Ix1::new([n]), picked_data)?;
        let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), picked_mask)?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }

    /// Fancy index selection from a 1-D `MaskedArray`.
    ///
    /// Equivalent to `a[indices]` in NumPy fancy indexing (restricted
    /// to 1-D because higher-rank fancy indexing has NumPy-specific
    /// broadcasting semantics that would be easy to get subtly wrong).
    /// Returns a new 1-D `MaskedArray` whose elements are picked from
    /// `self` at the supplied flat positions.
    ///
    /// Each result position carries through both the selected value
    /// and its original mask bit.
    ///
    /// # Errors
    /// - `FerrayError::IndexOutOfBounds` if any index is out of range.
    pub fn take(&self, indices: &[usize]) -> FerrayResult<MaskedArray<T, Ix1>>
    where
        D: Dimension,
    {
        let size = self.size();
        let mut picked_data: Vec<T> = Vec::with_capacity(indices.len());
        let mut picked_mask: Vec<bool> = Vec::with_capacity(indices.len());
        // Walk the flat buffers once rather than calling get_flat in a
        // loop — for contiguous inputs this is O(1) per index.
        let data_slice = self.data().as_slice();
        let mask_slice = self.mask().as_slice();
        let data_fallback: Option<Vec<T>> = if data_slice.is_none() {
            Some(self.data().iter().copied().collect())
        } else {
            None
        };
        let mask_fallback: Option<Vec<bool>> = if mask_slice.is_none() {
            Some(self.mask().iter().copied().collect())
        } else {
            None
        };
        for &idx in indices {
            if idx >= size {
                return Err(FerrayError::index_out_of_bounds(idx as isize, 0, size));
            }
            let v = if let Some(s) = data_slice {
                s[idx]
            } else {
                data_fallback.as_ref().unwrap()[idx]
            };
            let m = if let Some(s) = mask_slice {
                s[idx]
            } else {
                mask_fallback.as_ref().unwrap()[idx]
            };
            picked_data.push(v);
            picked_mask.push(m);
        }
        let n = picked_data.len();
        let data_arr = Array::<T, Ix1>::from_vec(Ix1::new([n]), picked_data)?;
        let mask_arr = Array::<bool, Ix1>::from_vec(Ix1::new([n]), picked_mask)?;
        let mut out = MaskedArray::new(data_arr, mask_arr)?;
        out.set_fill_value(self.fill_value());
        out.hard_mask = self.hard_mask;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2, Ix3};

    fn ma2d(rows: usize, cols: usize, data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix2> {
        let d = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).unwrap();
        let m = Array::<bool, Ix2>::from_vec(Ix2::new([rows, cols]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // ---- reshape / ravel / transpose / squeeze (#508) ----

    #[test]
    fn reshape_2d_to_different_2d() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let r = ma.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        // Row-major order preserved.
        assert_eq!(
            r.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            r.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, false, true, false]
        );
    }

    #[test]
    fn reshape_2d_to_1d() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let r = ma.reshape(&[6]).unwrap();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(r.size(), 6);
    }

    #[test]
    fn reshape_mismatched_size_errors() {
        let ma = ma2d(2, 3, vec![1.0; 6], vec![false; 6]);
        assert!(ma.reshape(&[2, 4]).is_err());
    }

    #[test]
    fn reshape_preserves_fill_value_and_hard_mask() {
        let mut ma = ma2d(2, 3, vec![1.0; 6], vec![false; 6]);
        ma.set_fill_value(-99.0);
        ma.harden_mask().unwrap();
        let r = ma.reshape(&[3, 2]).unwrap();
        assert_eq!(r.fill_value(), -99.0);
        assert!(r.is_hard_mask());
    }

    #[test]
    fn ravel_2d_flattens_in_row_major() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let r = ma.ravel().unwrap();
        assert_eq!(r.shape(), &[6]);
        assert_eq!(
            r.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        assert_eq!(
            r.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, false, true, false]
        );
    }

    #[test]
    fn flatten_is_alias_for_ravel() {
        let ma = ma2d(2, 2, vec![1.0, 2.0, 3.0, 4.0], vec![false, true, false, true]);
        let r1 = ma.ravel().unwrap();
        let r2 = ma.flatten().unwrap();
        assert_eq!(
            r1.data().iter().copied().collect::<Vec<_>>(),
            r2.data().iter().copied().collect::<Vec<_>>()
        );
        assert_eq!(
            r1.mask().iter().copied().collect::<Vec<_>>(),
            r2.mask().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn transpose_swaps_2d() {
        // [[1,2,3],[4,5,6]] → [[1,4],[2,5],[3,6]]
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let t = ma.transpose(None).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        assert_eq!(
            t.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
        // Mask transposes too.
        assert_eq!(
            t.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, true, true, false, false]
        );
    }

    #[test]
    fn t_is_alias_for_transpose_none() {
        let ma = ma2d(2, 3, vec![1.0; 6], vec![false; 6]);
        let t1 = ma.transpose(None).unwrap();
        let t2 = ma.t().unwrap();
        assert_eq!(t1.shape(), t2.shape());
    }

    #[test]
    fn transpose_with_explicit_permutation() {
        // 3-D with explicit permutation [2, 0, 1] — 2x3x4 → 4x2x3
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let mask = vec![false; 24];
        let d =
            Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data).unwrap();
        let m = Array::<bool, Ix3>::from_vec(Ix3::new([2, 3, 4]), mask).unwrap();
        let ma = MaskedArray::new(d, m).unwrap();
        let t = ma.transpose(Some(&[2, 0, 1])).unwrap();
        assert_eq!(t.shape(), &[4, 2, 3]);
    }

    #[test]
    fn squeeze_removes_all_size_1_dims_when_axis_none() {
        // (1, 3, 1) → (3,)
        let d =
            Array::<f64, Ix3>::from_vec(Ix3::new([1, 3, 1]), vec![10.0, 20.0, 30.0]).unwrap();
        let m = Array::<bool, Ix3>::from_vec(Ix3::new([1, 3, 1]), vec![false, true, false])
            .unwrap();
        let ma = MaskedArray::new(d, m).unwrap();
        let s = ma.squeeze(None).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(
            s.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    #[test]
    fn squeeze_single_axis() {
        // (1, 3, 1), squeeze axis=0 → (3, 1)
        let d =
            Array::<f64, Ix3>::from_vec(Ix3::new([1, 3, 1]), vec![10.0, 20.0, 30.0]).unwrap();
        let m = Array::<bool, Ix3>::from_vec(Ix3::new([1, 3, 1]), vec![false, true, false])
            .unwrap();
        let ma = MaskedArray::new(d, m).unwrap();
        let s = ma.squeeze(Some(0)).unwrap();
        assert_eq!(s.shape(), &[3, 1]);
    }

    // ---- indexing (#507) ----

    #[test]
    fn get_flat_returns_value_and_mask_bit() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        // Row-major: position 1 is (0, 1) with value=2, masked=true.
        let (v, m) = ma.get_flat(1).unwrap();
        assert_eq!(v, 2.0);
        assert!(m);
        // Position 3 is (1, 0) with value=4, masked=false.
        let (v, m) = ma.get_flat(3).unwrap();
        assert_eq!(v, 4.0);
        assert!(!m);
    }

    #[test]
    fn get_flat_out_of_bounds_errors() {
        let ma = ma2d(2, 2, vec![1.0; 4], vec![false; 4]);
        assert!(ma.get_flat(4).is_err());
        assert!(ma.get_flat(99).is_err());
    }

    #[test]
    fn boolean_index_selects_unmasked_structure() {
        let ma = ma2d(
            2,
            3,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![false, true, false, false, true, false],
        );
        let selector = Array::<bool, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![true, true, false, false, true, true],
        )
        .unwrap();
        let picked = ma.boolean_index(&selector).unwrap();
        // Selected positions: 0, 1, 4, 5 → values 1, 2, 5, 6
        assert_eq!(
            picked.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 5.0, 6.0]
        );
        // Original mask bits at those positions: F, T, T, F
        assert_eq!(
            picked.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, false]
        );
    }

    #[test]
    fn boolean_index_rejects_wrong_shape() {
        let ma = ma2d(2, 3, vec![1.0; 6], vec![false; 6]);
        let wrong =
            Array::<bool, Ix2>::from_vec(Ix2::new([3, 2]), vec![false; 6]).unwrap();
        assert!(ma.boolean_index(&wrong).is_err());
    }

    #[test]
    fn take_fancy_index_picks_flat_positions() {
        let ma = ma2d(
            2,
            3,
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![false, true, false, false, false, true],
        );
        // Pick positions 0, 5, 2, 1 in that order.
        let r = ma.take(&[0, 5, 2, 1]).unwrap();
        assert_eq!(
            r.data().iter().copied().collect::<Vec<_>>(),
            vec![10.0, 60.0, 30.0, 20.0]
        );
        assert_eq!(
            r.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
    }

    #[test]
    fn take_out_of_bounds_errors() {
        let ma = ma2d(2, 2, vec![1.0; 4], vec![false; 4]);
        assert!(ma.take(&[0, 1, 5]).is_err());
    }

    #[test]
    fn take_with_repeated_indices() {
        let ma = ma2d(1, 3, vec![1.0, 2.0, 3.0], vec![false, false, true]);
        let r = ma.take(&[0, 0, 2, 2]).unwrap();
        assert_eq!(
            r.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 1.0, 3.0, 3.0]
        );
        assert_eq!(
            r.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, true, true]
        );
    }
}
