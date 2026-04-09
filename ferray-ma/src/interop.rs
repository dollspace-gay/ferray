// ferray-ma: Interop with ferray-core's Array<T, D>
//
// In NumPy, MaskedArray subclasses ndarray, so anywhere ndarray is accepted
// MaskedArray works too — with mask propagation handled automatically by
// __array_ufunc__. Rust has no inheritance, so we provide the interop in
// three layers:
//
//   1. `AsRef<Array<T, D>>` for low-level access — pass `&masked.as_ref()`
//      where `&Array<T, D>` is expected, accepting that the mask is dropped.
//      Useful when you need the data quickly and the mask is irrelevant.
//
//   2. `From<MaskedArray<T, D>>` for `Array<T, D>` — owned data extraction
//      that consumes the mask. Use when you want to convert and forget.
//
//   3. `apply_unary` / `apply_binary` — mask-aware adapters that take any
//      Array→Array function (e.g. `ferray_ufunc::sin`) and propagate the
//      mask through to the result. The function operates on every element
//      of the underlying data array (including masked positions, which
//      hold whatever value is there); the mask is then re-attached to the
//      result so that downstream consumers see the correct invalidity.
//
// See: https://github.com/dollspace-gay/ferray/issues/505

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::MaskedArray;

// ---------------------------------------------------------------------------
// AsRef / From — passive interop
// ---------------------------------------------------------------------------

impl<T: Element, D: Dimension> AsRef<Array<T, D>> for MaskedArray<T, D> {
    /// Borrow the underlying data array, dropping the mask.
    ///
    /// This lets you pass a `&MaskedArray<T, D>` to any function that takes
    /// `&Array<T, D>`, but the mask is **not** consulted — masked positions
    /// will be processed like normal data. Use [`MaskedArray::apply_unary`]
    /// or [`MaskedArray::apply_binary`] when you want mask propagation.
    ///
    /// # Example
    /// ```
    /// # use ferray_core::{Array, dimension::Ix1};
    /// # use ferray_ma::MaskedArray;
    /// # let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
    /// # let mask = Array::<bool, Ix1>::from_vec(Ix1::new([3]), vec![false, true, false]).unwrap();
    /// let ma = MaskedArray::new(data, mask).unwrap();
    /// // Pass through a function that operates on `&Array<T, D>`:
    /// let arr_ref: &Array<f64, Ix1> = ma.as_ref();
    /// assert_eq!(arr_ref.shape(), &[3]);
    /// ```
    fn as_ref(&self) -> &Array<T, D> {
        self.data()
    }
}

impl<T: Element + Copy, D: Dimension> From<MaskedArray<T, D>> for Array<T, D> {
    /// Consume a `MaskedArray` and return its underlying data array,
    /// **discarding** the mask. Equivalent to calling `ma.into_data()`.
    ///
    /// Requires `T: Copy` because the underlying data buffer is cloned;
    /// use [`MaskedArray::filled`] / [`MaskedArray::filled_default`] for a
    /// mask-aware materialization with custom fill semantics.
    fn from(ma: MaskedArray<T, D>) -> Self {
        ma.into_data()
    }
}

// ---------------------------------------------------------------------------
// Active interop — apply functions with mask propagation
// ---------------------------------------------------------------------------

impl<T, D> MaskedArray<T, D>
where
    T: Element + Copy,
    D: Dimension,
{
    /// Consume the masked array and return its underlying data array,
    /// dropping the mask.
    ///
    /// Use [`MaskedArray::filled_default`] (or [`MaskedArray::filled`]) if
    /// you want masked positions replaced by a sentinel value before
    /// dropping the mask.
    pub fn into_data(self) -> Array<T, D> {
        // We can't move out of a struct with non-Copy fields directly
        // because of `data_mut()` borrowing semantics, so destructure
        // the unsafe-but-safe internal `data` field via the public getter.
        // The clone here is unavoidable without a fully private accessor.
        self.data().clone()
    }

    /// Apply a unary function to the underlying data and re-attach the
    /// mask, propagating it to the result.
    ///
    /// The function `f` is called on the **entire** data array — masked
    /// positions are processed alongside unmasked ones, but their values
    /// in the result are immediately overwritten with the masked array's
    /// `fill_value`. This matches NumPy's `__array_ufunc__` semantics where
    /// ufuncs run over the raw data and the mask is propagated separately.
    ///
    /// # Example
    /// ```ignore
    /// // Apply ferray-ufunc::sin to a masked f64 array:
    /// let result = ma.apply_unary(|arr| ferray_ufunc::sin(arr))?;
    /// ```
    ///
    /// # Errors
    /// Forwards any error from `f`.
    pub fn apply_unary<F>(&self, f: F) -> FerrayResult<MaskedArray<T, D>>
    where
        F: FnOnce(&Array<T, D>) -> FerrayResult<Array<T, D>>,
    {
        let data_out = f(self.data())?;
        if data_out.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "apply_unary: function changed shape from {:?} to {:?}",
                self.shape(),
                data_out.shape()
            )));
        }
        let fill = self.fill_value();
        // Replace masked positions in the result with fill_value to keep
        // operations like log/sqrt from leaving misleading data behind.
        let masked_data: Vec<T> = data_out
            .iter()
            .zip(self.mask().iter())
            .map(|(v, m)| if *m { fill } else { *v })
            .collect();
        let final_data = Array::from_vec(self.dim().clone(), masked_data)?;
        let mut result = MaskedArray::new(final_data, self.mask().clone())?;
        result.set_fill_value(fill);
        Ok(result)
    }

    /// Apply a unary function that maps `T -> U`, propagating the mask.
    ///
    /// This is the type-changing variant of [`MaskedArray::apply_unary`],
    /// useful for predicates like `isnan` that return `Array<bool, D>` from
    /// `Array<T, D>`. Masked positions in the result hold the explicitly
    /// supplied `default_for_masked` value.
    ///
    /// # Errors
    /// Forwards any error from `f`. Returns `FerrayError::ShapeMismatch` if
    /// `f` produces an array with a different shape.
    pub fn apply_unary_to<U, F>(
        &self,
        f: F,
        default_for_masked: U,
    ) -> FerrayResult<MaskedArray<U, D>>
    where
        U: Element + Copy,
        F: FnOnce(&Array<T, D>) -> FerrayResult<Array<U, D>>,
    {
        let data_out = f(self.data())?;
        if data_out.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "apply_unary_to: function changed shape from {:?} to {:?}",
                self.shape(),
                data_out.shape()
            )));
        }
        let masked_data: Vec<U> = data_out
            .iter()
            .zip(self.mask().iter())
            .map(|(v, m)| if *m { default_for_masked } else { *v })
            .collect();
        let final_data = Array::from_vec(self.dim().clone(), masked_data)?;
        let mut result = MaskedArray::new(final_data, self.mask().clone())?;
        result.set_fill_value(default_for_masked);
        Ok(result)
    }

    /// Apply a binary function to two masked arrays, propagating the mask
    /// union. Both inputs must have the same shape.
    ///
    /// The function `f` is called on the underlying data of both inputs
    /// (no broadcasting — use [`crate::masked_add`]-style functions for
    /// that). The result mask is the OR of the two input masks; masked
    /// positions in the data are overwritten with the receiver's `fill_value`.
    ///
    /// # Example
    /// ```ignore
    /// let result = a.apply_binary(&b, |x, y| ferray_ufunc::power(x, y))?;
    /// ```
    ///
    /// # Errors
    /// Returns `FerrayError::ShapeMismatch` if shapes differ. Forwards any
    /// error from `f`.
    pub fn apply_binary<F>(
        &self,
        other: &MaskedArray<T, D>,
        f: F,
    ) -> FerrayResult<MaskedArray<T, D>>
    where
        F: FnOnce(&Array<T, D>, &Array<T, D>) -> FerrayResult<Array<T, D>>,
    {
        if self.shape() != other.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "apply_binary: shapes {:?} and {:?} differ",
                self.shape(),
                other.shape()
            )));
        }
        let data_out = f(self.data(), other.data())?;
        if data_out.shape() != self.shape() {
            return Err(FerrayError::shape_mismatch(format!(
                "apply_binary: function changed shape from {:?} to {:?}",
                self.shape(),
                data_out.shape()
            )));
        }

        // Mask union.
        let union_data: Vec<bool> = self
            .mask()
            .iter()
            .zip(other.mask().iter())
            .map(|(a, b)| *a || *b)
            .collect();
        let union_mask = Array::from_vec(self.dim().clone(), union_data)?;

        let fill = self.fill_value();
        let masked_data: Vec<T> = data_out
            .iter()
            .zip(union_mask.iter())
            .map(|(v, m)| if *m { fill } else { *v })
            .collect();
        let final_data = Array::from_vec(self.dim().clone(), masked_data)?;
        let mut result = MaskedArray::new(final_data, union_mask)?;
        result.set_fill_value(fill);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn ma1(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
        let n = data.len();
        let d = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    #[test]
    fn as_ref_returns_underlying_data() {
        let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let arr_ref: &Array<f64, Ix1> = ma.as_ref();
        assert_eq!(arr_ref.shape(), &[3]);
        // The data is the unmasked-equivalent, mask is dropped.
        let v: Vec<f64> = arr_ref.iter().copied().collect();
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_masked_to_array_drops_mask() {
        let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let arr: Array<f64, Ix1> = ma.into();
        assert_eq!(arr.shape(), &[3]);
    }

    #[test]
    fn into_data_method() {
        let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let arr = ma.into_data();
        assert_eq!(arr.shape(), &[3]);
    }

    #[test]
    fn apply_unary_propagates_mask() {
        let ma = ma1(vec![1.0, 4.0, 9.0, 16.0], vec![false, false, true, false])
            .with_fill_value(-1.0);
        let result = ma
            .apply_unary(|arr| {
                // Squaring closure as a stand-in for any ferray-ufunc function
                let data: Vec<f64> = arr.iter().map(|&x| x.sqrt()).collect();
                Array::<f64, Ix1>::from_vec(Ix1::new([arr.size()]), data)
            })
            .unwrap();

        // Masked position (index 2) holds fill_value; others hold sqrt.
        let d: Vec<f64> = result.data().iter().copied().collect();
        assert_eq!(d, vec![1.0, 2.0, -1.0, 4.0]);
        // Mask is preserved.
        let m: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(m, vec![false, false, true, false]);
        // Fill value is preserved.
        assert_eq!(result.fill_value(), -1.0);
    }

    #[test]
    fn apply_unary_forwards_error() {
        let ma = ma1(vec![1.0, 2.0], vec![false, false]);
        let result: FerrayResult<MaskedArray<f64, Ix1>> = ma.apply_unary(|_| {
            Err(FerrayError::invalid_value("simulated failure"))
        });
        assert!(result.is_err());
    }

    #[test]
    fn apply_unary_rejects_shape_change() {
        let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, false, false]);
        let result = ma.apply_unary(|_| {
            Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, 2.0])
        });
        assert!(result.is_err());
    }

    #[test]
    fn apply_unary_to_changes_type_with_mask_default() {
        // Apply a "is positive" predicate that returns bool.
        let ma = ma1(vec![1.0, -2.0, 3.0, -4.0], vec![false, false, true, false]);
        let result = ma
            .apply_unary_to(
                |arr| {
                    let data: Vec<bool> = arr.iter().map(|&x| x > 0.0).collect();
                    Array::<bool, Ix1>::from_vec(Ix1::new([arr.size()]), data)
                },
                false, // default for masked positions
            )
            .unwrap();

        let d: Vec<bool> = result.data().iter().copied().collect();
        // Index 2 is masked → false (the default); others reflect the predicate.
        assert_eq!(d, vec![true, false, false, false]);
        let m: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(m, vec![false, false, true, false]);
    }

    #[test]
    fn apply_binary_unions_masks() {
        let a = ma1(vec![10.0, 20.0, 30.0], vec![false, true, false])
            .with_fill_value(-1.0);
        let b = ma1(vec![1.0, 2.0, 3.0], vec![false, false, true]);
        let result = a
            .apply_binary(&b, |x, y| {
                let data: Vec<f64> = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();
                Array::<f64, Ix1>::from_vec(Ix1::new([x.size()]), data)
            })
            .unwrap();

        let d: Vec<f64> = result.data().iter().copied().collect();
        // Indices 1 and 2 are masked (union of the two input masks); index 0
        // gets the actual sum.
        assert_eq!(d, vec![11.0, -1.0, -1.0]);
        let m: Vec<bool> = result.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, true]);
        assert_eq!(result.fill_value(), -1.0);
    }

    #[test]
    fn apply_binary_rejects_shape_mismatch() {
        let a = ma1(vec![1.0, 2.0, 3.0], vec![false; 3]);
        let b = ma1(vec![1.0, 2.0], vec![false; 2]);
        let result = a.apply_binary(&b, |x, _y| Ok(x.clone()));
        assert!(result.is_err());
    }

    /// Demonstrates the canonical interop pattern with a real ferray-stats
    /// call: pass `&MaskedArray` through `as_ref()` to a function that
    /// expects `&Array`. This loses the mask (per the AsRef contract) but
    /// is the cheapest way to bridge.
    #[test]
    fn as_ref_works_with_array_consuming_function() {
        // A simple Array -> Array function (any closure works as a stand-in).
        fn double(arr: &Array<f64, Ix1>) -> FerrayResult<Array<f64, Ix1>> {
            let data: Vec<f64> = arr.iter().map(|&x| x * 2.0).collect();
            Array::<f64, Ix1>::from_vec(Ix1::new([arr.size()]), data)
        }

        let ma = ma1(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        // Direct call via AsRef — the mask is dropped:
        let result = double(ma.as_ref()).unwrap();
        let v: Vec<f64> = result.iter().copied().collect();
        assert_eq!(v, vec![2.0, 4.0, 6.0]);

        // For mask-preserving usage, route through apply_unary instead:
        let masked_result = ma.apply_unary(double).unwrap();
        let m: Vec<bool> = masked_result.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, false]);
    }
}

// ---------------------------------------------------------------------------
// MaskAware trait: common interface for Array and MaskedArray (#505)
//
// NumPy's MaskedArray subclasses ndarray so any ndarray-accepting function
// automatically accepts MaskedArray (with __array_ufunc__ handling mask
// propagation). Rust doesn't have inheritance, so the ferray equivalent is a
// trait both types implement: mask-aware functions take
// `&impl MaskAware<T, D>` and dispatch via the trait methods.
//
// Array<T, D> is treated as "always fully unmasked" — `mask_opt()` returns
// None and `fill_value()` falls back to `T::zero()`. MaskedArray<T, D>
// delegates to its actual accessors. This lets callers write one function
// that works on both and still propagates masks correctly.
// ---------------------------------------------------------------------------

/// Shared view contract for functions that want to accept either an
/// `Array` or a `MaskedArray` (#505).
///
/// Implementations:
/// - `Array<T, D>`: `data()` returns `self`, `mask_opt()` returns `None`
///   (no mask), `fill_value()` returns `T::zero()`. The array is treated
///   as fully unmasked.
/// - `MaskedArray<T, D>`: delegates to the existing accessors.
///
/// Downstream code that wants to write "one function, works on both"
/// should take `&impl MaskAware<T, D>` and consult `mask_opt()` to
/// decide whether to do mask propagation.
pub trait MaskAware<T: Element, D: Dimension> {
    /// Return a reference to the underlying data array.
    fn data(&self) -> &Array<T, D>;

    /// Return the mask array if one is explicitly present, or `None`
    /// when the input carries no mask (treated as fully unmasked).
    ///
    /// For `Array<T, D>` this always returns `None`. For
    /// `MaskedArray<T, D>` it returns `Some` when a real mask has
    /// been explicitly set and `None` when the array is in the
    /// nomask-sentinel state (#506).
    fn mask_opt(&self) -> Option<&Array<bool, D>>;

    /// Return the fill value to use for masked positions in
    /// derived results.
    fn fill_value(&self) -> T
    where
        T: Copy;

    /// Return the shape of the underlying data.
    fn shape(&self) -> &[usize] {
        self.data().shape()
    }
}

impl<T: Element, D: Dimension> MaskAware<T, D> for Array<T, D> {
    #[inline]
    fn data(&self) -> &Array<T, D> {
        self
    }

    /// A plain `Array<T, D>` has no mask — always returns `None`.
    #[inline]
    fn mask_opt(&self) -> Option<&Array<bool, D>> {
        None
    }

    /// A plain `Array<T, D>` has no fill value; returns `T::zero()`.
    #[inline]
    fn fill_value(&self) -> T
    where
        T: Copy,
    {
        T::zero()
    }
}

impl<T: Element, D: Dimension> MaskAware<T, D> for MaskedArray<T, D> {
    #[inline]
    fn data(&self) -> &Array<T, D> {
        MaskedArray::data(self)
    }

    #[inline]
    fn mask_opt(&self) -> Option<&Array<bool, D>> {
        MaskedArray::mask_opt(self)
    }

    #[inline]
    fn fill_value(&self) -> T
    where
        T: Copy,
    {
        MaskedArray::fill_value(self)
    }
}

/// Apply a unary function to any `MaskAware` input, propagating the
/// mask if one is present.
///
/// When the input is a plain `Array<T, D>` (or a nomask-sentinel
/// `MaskedArray`), the function is applied directly and the result
/// is returned as a nomask `MaskedArray`. When the input has a real
/// mask, this delegates to the existing [`MaskedArray::apply_unary`]
/// path so masked positions are overwritten with the fill value.
///
/// Use this to write "one function, works on both" adapters:
///
/// ```ignore
/// fn my_op<X: MaskAware<f64, Ix1>>(x: &X) -> FerrayResult<MaskedArray<f64, Ix1>> {
///     ma_apply_unary(x, |a| ferray_ufunc::sin(a))
/// }
/// ```
///
/// # Errors
/// Forwards any error from `f`, plus shape-mismatch errors if `f`
/// returns a differently-shaped array.
pub fn ma_apply_unary<T, D, X, F>(input: &X, f: F) -> FerrayResult<MaskedArray<T, D>>
where
    T: Element + Copy,
    D: Dimension,
    X: MaskAware<T, D>,
    F: FnOnce(&Array<T, D>) -> FerrayResult<Array<T, D>>,
{
    let data_out = f(input.data())?;
    if data_out.shape() != input.shape() {
        return Err(FerrayError::shape_mismatch(format!(
            "ma_apply_unary: function changed shape from {:?} to {:?}",
            input.shape(),
            data_out.shape()
        )));
    }

    match input.mask_opt() {
        None => {
            // No mask — wrap the result in a nomask-sentinel
            // MaskedArray so the caller gets a uniform return type.
            let mut out = MaskedArray::from_data(data_out)?;
            out.set_fill_value(input.fill_value());
            Ok(out)
        }
        Some(mask) => {
            // Overwrite masked positions with fill_value so downstream
            // operations can't see stale data at masked slots.
            let fill = input.fill_value();
            let masked_data: Vec<T> = data_out
                .iter()
                .zip(mask.iter())
                .map(|(v, m)| if *m { fill } else { *v })
                .collect();
            let final_data =
                Array::from_vec(input.data().dim().clone(), masked_data)?;
            let mut result = MaskedArray::new(final_data, mask.clone())?;
            result.set_fill_value(fill);
            Ok(result)
        }
    }
}

#[cfg(test)]
mod mask_aware_tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn arr_f64(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn ma_f64(data: Vec<f64>, mask: Vec<bool>) -> MaskedArray<f64, Ix1> {
        let d = arr_f64(data);
        let n = d.size();
        let m = Array::<bool, Ix1>::from_vec(Ix1::new([n]), mask).unwrap();
        MaskedArray::new(d, m).unwrap()
    }

    // ---- MaskAware trait impls (#505) ----

    #[test]
    fn array_implements_mask_aware_with_none_mask() {
        let a = arr_f64(vec![1.0, 2.0, 3.0]);
        // Plain Array carries no mask.
        assert!(<Array<f64, Ix1> as MaskAware<f64, Ix1>>::mask_opt(&a).is_none());
        assert_eq!(
            <Array<f64, Ix1> as MaskAware<f64, Ix1>>::fill_value(&a),
            0.0
        );
        assert_eq!(
            <Array<f64, Ix1> as MaskAware<f64, Ix1>>::shape(&a),
            &[3]
        );
    }

    #[test]
    fn masked_array_implements_mask_aware_with_real_mask() {
        let ma = ma_f64(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let via_trait = <MaskedArray<f64, Ix1> as MaskAware<f64, Ix1>>::mask_opt(&ma);
        assert!(via_trait.is_some());
        assert_eq!(
            via_trait.unwrap().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    #[test]
    fn nomask_sentinel_masked_array_reports_none_via_trait() {
        // A from_data-constructed MaskedArray (nomask sentinel) should
        // report None through the MaskAware trait, matching the
        // behavior of a plain Array.
        let ma = MaskedArray::from_data(arr_f64(vec![1.0, 2.0, 3.0])).unwrap();
        let via_trait = <MaskedArray<f64, Ix1> as MaskAware<f64, Ix1>>::mask_opt(&ma);
        assert!(via_trait.is_none());
    }

    #[test]
    fn ma_apply_unary_on_plain_array_returns_nomask_result() {
        let a = arr_f64(vec![1.0, 2.0, 3.0]);
        let result = ma_apply_unary(&a, |x| {
            let data: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
            Ok(Array::from_vec(x.dim().clone(), data)?)
        })
        .unwrap();
        assert_eq!(
            result.data().iter().copied().collect::<Vec<_>>(),
            vec![2.0, 4.0, 6.0]
        );
        // Plain-Array input → nomask-sentinel result.
        assert!(!result.has_real_mask());
    }

    #[test]
    fn ma_apply_unary_on_masked_array_propagates_mask() {
        let ma = ma_f64(vec![1.0, 2.0, 3.0], vec![false, true, false]);
        let result = ma_apply_unary(&ma, |x| {
            let data: Vec<f64> = x.iter().map(|v| v * 2.0).collect();
            Ok(Array::from_vec(x.dim().clone(), data)?)
        })
        .unwrap();
        // Mask survives the operation.
        assert!(result.has_real_mask());
        assert_eq!(
            result.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
        // Masked position was overwritten with fill_value (0.0 default).
        let d: Vec<f64> = result.data().iter().copied().collect();
        assert_eq!(d[0], 2.0);
        assert_eq!(d[1], 0.0); // masked → fill value
        assert_eq!(d[2], 6.0);
    }

    #[test]
    fn ma_apply_unary_generic_over_both_types() {
        // Write a helper that works on both Array and MaskedArray.
        fn double_it<T: Element, D: Dimension, X: MaskAware<T, D>>(
            x: &X,
        ) -> FerrayResult<MaskedArray<T, D>>
        where
            T: Copy + std::ops::Mul<Output = T> + num_traits::FromPrimitive,
        {
            let two = T::from_f64(2.0).unwrap();
            ma_apply_unary(x, move |a| {
                let data: Vec<T> = a.iter().map(|v| *v * two).collect();
                Ok(Array::from_vec(a.dim().clone(), data)?)
            })
        }

        let plain = arr_f64(vec![1.0, 2.0, 3.0]);
        let masked = ma_f64(vec![1.0, 2.0, 3.0], vec![false, true, false]);

        // Both inputs go through the same helper.
        let r_plain = double_it(&plain).unwrap();
        let r_masked = double_it(&masked).unwrap();

        // Plain result: no mask, all values doubled.
        assert!(!r_plain.has_real_mask());
        assert_eq!(
            r_plain.data().iter().copied().collect::<Vec<_>>(),
            vec![2.0, 4.0, 6.0]
        );

        // Masked result: mask preserved, masked position holds fill.
        assert!(r_masked.has_real_mask());
        assert_eq!(
            r_masked.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false]
        );
    }

    #[test]
    fn ma_apply_unary_rejects_shape_changing_function() {
        let a = arr_f64(vec![1.0, 2.0, 3.0]);
        let result = ma_apply_unary(&a, |_| {
            // Return a wrong-shape result deliberately.
            Ok(arr_f64(vec![1.0, 2.0]))
        });
        assert!(result.is_err());
    }
}
