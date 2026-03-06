// ferrum-stats: Quantile-based reductions — median, percentile, quantile (REQ-1)
// Also nanmedian, nanpercentile (REQ-3)

use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use super::{collect_data, make_result, output_shape, reduce_axis_general, validate_axis};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute a single quantile value from a sorted slice using linear interpolation.
/// `q` must be in [0, 1].
fn quantile_sorted<T: Float>(sorted: &[T], q: T) -> T {
    let n = sorted.len();
    if n == 0 {
        return T::nan();
    }
    if n == 1 {
        return sorted[0];
    }
    let idx_f = q * T::from(n - 1).unwrap();
    let lo = idx_f.floor();
    let hi = idx_f.ceil();
    let lo_i = lo.to_usize().unwrap().min(n - 1);
    let hi_i = hi.to_usize().unwrap().min(n - 1);
    if lo_i == hi_i {
        sorted[lo_i]
    } else {
        let frac = idx_f - lo;
        sorted[lo_i] * (T::one() - frac) + sorted[hi_i] * frac
    }
}

/// Sort a mutable slice by partial_cmp, placing NaN at the end.
fn partial_sort<T: Float>(data: &mut [T]) {
    data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Sort and compute quantile from a lane.
fn lane_quantile<T: Float>(lane: &[T], q: T) -> T {
    let mut sorted: Vec<T> = lane.to_vec();
    partial_sort(&mut sorted);
    quantile_sorted(&sorted, q)
}

/// Sort (excluding NaN) and compute quantile from a lane.
fn lane_nanquantile<T: Float>(lane: &[T], q: T) -> T {
    let mut sorted: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    if sorted.is_empty() {
        return T::nan();
    }
    partial_sort(&mut sorted);
    quantile_sorted(&sorted, q)
}

// ---------------------------------------------------------------------------
// quantile
// ---------------------------------------------------------------------------

/// Compute the q-th quantile of array data along a given axis.
///
/// `q` must be in \[0, 1\]. Uses linear interpolation (NumPy default method).
/// Equivalent to `numpy.quantile`.
pub fn quantile<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if q < <T as Element>::zero() || q > <T as Element>::one() {
        return Err(FerrumError::invalid_value(
            "quantile q must be in [0, 1]"
        ));
    }
    if a.is_empty() {
        return Err(FerrumError::invalid_value("cannot compute quantile of empty array"));
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
pub fn percentile<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let hundred = T::from(100.0).unwrap();
    if q < <T as Element>::zero() || q > hundred {
        return Err(FerrumError::invalid_value(
            "percentile q must be in [0, 100]"
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
pub fn median<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
) -> FerrumResult<Array<T, IxDyn>>
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
pub fn nanmedian<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
) -> FerrumResult<Array<T, IxDyn>>
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
) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let hundred = T::from(100.0).unwrap();
    if q < <T as Element>::zero() || q > hundred {
        return Err(FerrumError::invalid_value(
            "nanpercentile q must be in [0, 100]"
        ));
    }
    nanquantile(a, q / hundred, axis)
}

/// Quantile, skipping NaN values.
fn nanquantile<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if q < <T as Element>::zero() || q > <T as Element>::one() {
        return Err(FerrumError::invalid_value(
            "quantile q must be in [0, 1]"
        ));
    }
    if a.is_empty() {
        return Err(FerrumError::invalid_value("cannot compute nanquantile of empty array"));
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
    use ferrum_core::Ix1;

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
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([4]),
            vec![1.0, f64::NAN, 3.0, 5.0],
        ).unwrap();
        let m = nanmedian(&a, None).unwrap();
        // non-nan sorted: [1, 3, 5], median = 3.0
        assert!((m.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanpercentile() {
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([4]),
            vec![1.0, f64::NAN, 3.0, 5.0],
        ).unwrap();
        let p = nanpercentile(&a, 50.0, None).unwrap();
        assert!((p.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanmedian_all_nan() {
        let a = Array::<f64, Ix1>::from_vec(
            Ix1::new([2]),
            vec![f64::NAN, f64::NAN],
        ).unwrap();
        let m = nanmedian(&a, None).unwrap();
        assert!(m.iter().next().unwrap().is_nan());
    }
}
