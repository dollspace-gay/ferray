// ferray-ma: Masking constructors (REQ-7, REQ-8, REQ-9)
//
// masked_where, masked_invalid, masked_equal, masked_greater, masked_less,
// masked_not_equal, masked_greater_equal, masked_less_equal,
// masked_inside, masked_outside

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::MaskedArray;

/// Create a `MaskedArray` by masking elements where the condition array is `true`.
///
/// # Errors
/// Returns `FerrayError::ShapeMismatch` if `condition` and `data` have different shapes.
pub fn masked_where<T: Element + Copy, D: Dimension>(
    condition: &Array<bool, D>,
    data: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>> {
    MaskedArray::new(data.clone(), condition.clone())
}

/// Create a `MaskedArray` by masking NaN and Inf values.
///
/// The data is cloned as-is; masked positions retain their original
/// NaN/Inf values in the data array. If you want the data positions
/// replaced with a sentinel value too, use [`fix_invalid`] (#510).
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_invalid<T: Element + Float, D: Dimension>(
    data: &Array<T, D>,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| v.is_nan() || v.is_infinite()).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking NaN and Inf values AND replacing
/// them with `fill_value` in the underlying data.
///
/// Equivalent to `numpy.ma.fix_invalid(data, fill_value=fill_value)`
/// (#510). This is a strict superset of [`masked_invalid`]: the
/// result's mask is identical (positions where `x.is_nan() ||
/// x.is_infinite()`) but the data array has those positions replaced
/// with `fill_value` instead of the original NaN/Inf. The result's
/// `fill_value` is also set to `fill_value` so subsequent operations
/// that use the fill value behave consistently.
///
/// Cleaning a data array in a single pass is the primary use case —
/// after `fix_invalid`, the underlying `data()` is free of NaN/Inf
/// and can be passed to operations that would otherwise produce
/// NaN propagation.
///
/// # Errors
/// Returns an error only for internal failures (allocation).
pub fn fix_invalid<T: Element + Float, D: Dimension>(
    data: &Array<T, D>,
    fill_value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mut new_data: Vec<T> = Vec::with_capacity(data.size());
    let mut new_mask: Vec<bool> = Vec::with_capacity(data.size());
    for &v in data.iter() {
        if v.is_nan() || v.is_infinite() {
            new_data.push(fill_value);
            new_mask.push(true);
        } else {
            new_data.push(v);
            new_mask.push(false);
        }
    }
    let data_arr = Array::from_vec(data.dim().clone(), new_data)?;
    let mask_arr = Array::from_vec(data.dim().clone(), new_mask)?;
    let mut out = MaskedArray::new(data_arr, mask_arr)?;
    out.set_fill_value(fill_value);
    Ok(out)
}

/// Create a `MaskedArray` by masking elements equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_equal<T: Element + PartialEq + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v == value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements not equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_not_equal<T: Element + PartialEq + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v != value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements greater than `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_greater<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v > value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements less than `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_less<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v < value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements greater than or equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_greater_equal<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v >= value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements less than or equal to `value`.
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_less_equal<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    value: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let mask_data: Vec<bool> = data.iter().map(|v| *v <= value).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements inside the closed interval `[v1, v2]`.
///
/// Matches `numpy.ma.masked_inside`: if `v1 > v2` they are swapped so
/// the interval is always non-degenerate (#266).
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_inside<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    v1: T,
    v2: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let (lo, hi) = if v1 <= v2 { (v1, v2) } else { (v2, v1) };
    let mask_data: Vec<bool> = data.iter().map(|v| *v >= lo && *v <= hi).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

/// Create a `MaskedArray` by masking elements outside the closed interval `[v1, v2]`.
///
/// Matches `numpy.ma.masked_outside`: if `v1 > v2` they are swapped so
/// the interval is always non-degenerate (#266).
///
/// # Errors
/// Returns an error only for internal failures.
pub fn masked_outside<T: Element + PartialOrd + Copy, D: Dimension>(
    data: &Array<T, D>,
    v1: T,
    v2: T,
) -> FerrayResult<MaskedArray<T, D>> {
    let (lo, hi) = if v1 <= v2 { (v1, v2) } else { (v2, v1) };
    let mask_data: Vec<bool> = data.iter().map(|v| *v < lo || *v > hi).collect();
    let mask = Array::from_vec(data.dim().clone(), mask_data)?;
    MaskedArray::new(data.clone(), mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    // ---- fix_invalid (#510) ----

    #[test]
    fn fix_invalid_masks_and_replaces_nan_and_inf() {
        let data = Array::<f64, Ix1>::from_vec(
            Ix1::new([6]),
            vec![1.0, f64::NAN, 3.0, f64::INFINITY, f64::NEG_INFINITY, 6.0],
        )
        .unwrap();
        let ma = fix_invalid(&data, -99.0).unwrap();

        // Mask has the invalid positions set.
        let m: Vec<bool> = ma.mask().iter().copied().collect();
        assert_eq!(m, vec![false, true, false, true, true, false]);

        // Data has the invalid positions replaced with the fill value.
        let d: Vec<f64> = ma.data().iter().copied().collect();
        assert_eq!(d, vec![1.0, -99.0, 3.0, -99.0, -99.0, 6.0]);

        // Result's fill_value is set to the passed value, not the default.
        assert_eq!(ma.fill_value(), -99.0);
    }

    #[test]
    fn fix_invalid_preserves_valid_values() {
        // No NaN/Inf → mask is all-false and data matches input exactly.
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let ma = fix_invalid(&data, 0.0).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, false, false]
        );
        assert_eq!(
            ma.data().iter().copied().collect::<Vec<_>>(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn fix_invalid_all_nan_input() {
        let data =
            Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![f64::NAN, f64::NAN, f64::NAN]).unwrap();
        let ma = fix_invalid(&data, 0.0).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, true, true]
        );
        assert_eq!(
            ma.data().iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0]
        );
        // Crucially, NaN isn't in the data anymore — downstream ops
        // that compare against NaN won't propagate.
        assert!(ma.data().iter().all(|v| !v.is_nan()));
    }

    #[test]
    fn fix_invalid_vs_masked_invalid_data_difference() {
        // masked_invalid leaves NaN in the data; fix_invalid substitutes.
        let data = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
        let via_masked = masked_invalid(&data).unwrap();
        let via_fixed = fix_invalid(&data, -1.0).unwrap();

        // Masks are identical.
        assert_eq!(
            via_masked.mask().iter().copied().collect::<Vec<_>>(),
            via_fixed.mask().iter().copied().collect::<Vec<_>>()
        );

        // But the data differs: masked_invalid keeps NaN, fix_invalid
        // substitutes -1.0.
        assert!(via_masked.data().iter().nth(1).unwrap().is_nan());
        assert_eq!(*via_fixed.data().iter().nth(1).unwrap(), -1.0);
    }

    #[test]
    fn fix_invalid_2d_shape_preserved() {
        use ferray_core::dimension::Ix2;
        let data = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::INFINITY],
        )
        .unwrap();
        let ma = fix_invalid(&data, -1.0).unwrap();
        assert_eq!(ma.shape(), &[2, 3]);
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, false, false, false, true]
        );
    }

    // ----- masked_inside / masked_outside swap (#266) ---------------------

    #[test]
    fn masked_inside_canonical_order_masks_interior() {
        let data = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let ma = masked_inside(&data, 2, 4).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, true, true, true, false]
        );
    }

    #[test]
    fn masked_inside_swaps_when_v1_greater_than_v2() {
        // #266: numpy auto-swaps; masked_inside(data, 4, 2) must be
        // equivalent to masked_inside(data, 2, 4).
        let data = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let swapped = masked_inside(&data, 4, 2).unwrap();
        let canonical = masked_inside(&data, 2, 4).unwrap();
        assert_eq!(
            swapped.mask().iter().copied().collect::<Vec<_>>(),
            canonical.mask().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn masked_outside_canonical_order_masks_exterior() {
        let data = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let ma = masked_outside(&data, 2, 4).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![true, false, false, false, true]
        );
    }

    #[test]
    fn masked_outside_swaps_when_v1_greater_than_v2() {
        // #266: same swap behavior on the exterior path.
        let data = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let swapped = masked_outside(&data, 4, 2).unwrap();
        let canonical = masked_outside(&data, 2, 4).unwrap();
        assert_eq!(
            swapped.mask().iter().copied().collect::<Vec<_>>(),
            canonical.mask().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn masked_inside_equal_endpoints_masks_only_that_value() {
        // [v, v] is a degenerate interval — masking should still pick
        // up exact-equal entries.
        let data = Array::<i32, Ix1>::from_vec(Ix1::new([5]), vec![1, 2, 3, 4, 5]).unwrap();
        let ma = masked_inside(&data, 3, 3).unwrap();
        assert_eq!(
            ma.mask().iter().copied().collect::<Vec<_>>(),
            vec![false, false, true, false, false]
        );
    }
}
