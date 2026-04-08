// ferray-stats: Quantile-based reductions — median, percentile, quantile (REQ-1)
// Also nanmedian, nanpercentile (REQ-3)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use super::{collect_data, make_result, output_shape, reduce_axis_general, validate_axis};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a single quantile value from an unsorted slice using
/// `select_nth_unstable_by` rather than a full `sort_by`.
///
/// The selection algorithm gives an O(n) average-time path (quickselect)
/// instead of the O(n log n) full sort the previous implementation used
/// (#175). Linear interpolation between the two surrounding elements is
/// preserved exactly, matching NumPy's default `'linear'` method.
///
/// `data` is consumed: it is partitioned in place so the caller should
/// pass an owned buffer (or a clone they no longer need).
fn quantile_select<T: Float>(mut data: Vec<T>, q: T) -> T {
    let n = data.len();
    if n == 0 {
        return T::nan();
    }
    if n == 1 {
        return data[0];
    }

    let cmp = |a: &T, b: &T| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

    let idx_f = q * T::from(n - 1).unwrap();
    let lo = idx_f.floor();
    let lo_i = lo.to_usize().unwrap().min(n - 1);
    let frac = idx_f - lo;

    // First selection: place the lo_i-th smallest at position lo_i.
    data.select_nth_unstable_by(lo_i, cmp);
    let lo_val = data[lo_i];

    // Fast exit: exact hit on an integral index.
    if lo_i == n - 1 || frac == T::zero() {
        return lo_val;
    }

    // After the partial select, every element in `data[lo_i + 1..]` is
    // ordered-after `lo_val`; the smallest of them is the
    // `(lo_i + 1)`-th smallest element overall, which is the `hi_val`
    // we need for the linear interpolation.
    let hi_val = data[lo_i + 1..]
        .iter()
        .copied()
        .reduce(|a, b| match cmp(&a, &b) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => a,
            std::cmp::Ordering::Greater => b,
        })
        .unwrap_or(lo_val);

    lo_val * (T::one() - frac) + hi_val * frac
}

/// Compute quantile on a lane via the select-based fast path.
fn lane_quantile<T: Float>(lane: &[T], q: T) -> T {
    quantile_select(lane.to_vec(), q)
}

/// Compute quantile on a lane, excluding NaNs.
fn lane_nanquantile<T: Float>(lane: &[T], q: T) -> T {
    let filtered: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    if filtered.is_empty() {
        return T::nan();
    }
    quantile_select(filtered, q)
}

// ---------------------------------------------------------------------------
// quantile
// ---------------------------------------------------------------------------

/// Compute the q-th quantile of array data along a given axis.
///
/// `q` must be in \[0, 1\]. Uses linear interpolation (NumPy default method).
/// Equivalent to `numpy.quantile`.
pub fn quantile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if q < <T as Element>::zero() || q > <T as Element>::one() {
        return Err(FerrayError::invalid_value("quantile q must be in [0, 1]"));
    }
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute quantile of empty array",
        ));
    }
    let data = collect_data(a);
    match axis {
        None => {
            let val = lane_quantile(&data, q);
            make_result(&[], vec![val])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| lane_quantile(lane, q));
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// percentile
// ---------------------------------------------------------------------------

/// Compute the q-th percentile of array data along a given axis.
///
/// `q` must be in \[0, 100\].
/// Equivalent to `numpy.percentile`.
pub fn percentile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let hundred = T::from(100.0).unwrap();
    if q < <T as Element>::zero() || q > hundred {
        return Err(FerrayError::invalid_value(
            "percentile q must be in [0, 100]",
        ));
    }
    quantile(a, q / hundred, axis)
}

// ---------------------------------------------------------------------------
// median
// ---------------------------------------------------------------------------

/// Compute the median of array elements along a given axis.
///
/// Equivalent to `numpy.median`.
pub fn median<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let half = T::from(0.5).unwrap();
    quantile(a, half, axis)
}

// ---------------------------------------------------------------------------
// NaN-aware variants
// ---------------------------------------------------------------------------

/// Median, skipping NaN values.
///
/// Equivalent to `numpy.nanmedian`.
pub fn nanmedian<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let half = T::from(0.5).unwrap();
    nanquantile(a, half, axis)
}

/// Percentile, skipping NaN values.
///
/// Equivalent to `numpy.nanpercentile`.
pub fn nanpercentile<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let hundred = T::from(100.0).unwrap();
    if q < <T as Element>::zero() || q > hundred {
        return Err(FerrayError::invalid_value(
            "nanpercentile q must be in [0, 100]",
        ));
    }
    nanquantile(a, q / hundred, axis)
}

/// Quantile, skipping NaN values.
fn nanquantile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if q < <T as Element>::zero() || q > <T as Element>::one() {
        return Err(FerrayError::invalid_value("quantile q must be in [0, 1]"));
    }
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute nanquantile of empty array",
        ));
    }
    let data = collect_data(a);
    match axis {
        None => {
            let val = lane_nanquantile(&data, q);
            make_result(&[], vec![val])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| lane_nanquantile(lane, q));
            make_result(&out_s, result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Ix1;

    #[test]
    fn test_median_odd() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, 1.0, 3.0, 2.0, 4.0]).unwrap();
        let m = median(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_even() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![4.0, 1.0, 3.0, 2.0]).unwrap();
        let m = median(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_0_50_100() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let p0 = percentile(&a, 0.0, None).unwrap();
        let p50 = percentile(&a, 50.0, None).unwrap();
        let p100 = percentile(&a, 100.0, None).unwrap();
        assert!((p0.iter().next().unwrap() - 1.0).abs() < 1e-12);
        assert!((p50.iter().next().unwrap() - 3.0).abs() < 1e-12);
        assert!((p100.iter().next().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_bounds() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(quantile(&a, -0.1, None).is_err());
        assert!(quantile(&a, 1.1, None).is_err());
    }

    #[test]
    fn test_quantile_interpolation() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = quantile(&a, 0.25, None).unwrap();
        // index = 0.25 * 3 = 0.75, interp between 1.0 and 2.0 -> 1.75
        assert!((q.iter().next().unwrap() - 1.75).abs() < 1e-12);
    }

    #[test]
    fn test_nanmedian() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 3.0, 5.0]).unwrap();
        let m = nanmedian(&a, None).unwrap();
        // non-nan sorted: [1, 3, 5], median = 3.0
        assert!((m.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanpercentile() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 3.0, 5.0]).unwrap();
        let p = nanpercentile(&a, 50.0, None).unwrap();
        assert!((p.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanmedian_all_nan() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![f64::NAN, f64::NAN]).unwrap();
        let m = nanmedian(&a, None).unwrap();
        assert!(m.iter().next().unwrap().is_nan());
    }
}
