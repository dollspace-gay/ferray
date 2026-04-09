// ferray-stats: Quantile-based reductions — median, percentile, quantile (REQ-1)
// Also nanmedian, nanpercentile (REQ-3)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use super::{collect_data, make_result, output_shape, reduce_axis_general, validate_axis};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Interpolation method for [`quantile_with_method`] and its `percentile`
/// / `median` friends (#462).
///
/// Matches the five classic NumPy quantile methods. The remaining eight
/// methods NumPy supports (`'inverted_cdf'`, `'hazen'`, `'weibull'`,
/// `'median_unbiased'`, etc.) require specialized continuous-CDF
/// interpolators and can be added as follow-ups if demand appears.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// NumPy default: `lo_val * (1 - frac) + hi_val * frac`.
    Linear,
    /// Pick the element at `floor(q * (n - 1))` — the lower of the two
    /// bracketing sorted elements.
    Lower,
    /// Pick the element at `ceil(q * (n - 1))` — the upper of the two
    /// bracketing sorted elements.
    Higher,
    /// Pick the sorted element nearest to `q * (n - 1)`, with ties
    /// (frac = 0.5) broken to the even index (matches NumPy's
    /// round-half-to-even convention).
    Nearest,
    /// Average of the two bracketing sorted elements: `(lo + hi) / 2`.
    Midpoint,
}

/// Compute a single quantile value from an unsorted slice using
/// `select_nth_unstable_by` rather than a full `sort_by`.
///
/// The selection algorithm gives an O(n) average-time path (quickselect)
/// instead of the O(n log n) full sort the previous implementation used
/// (#175). For the default `Linear` method the result is exactly NumPy's
/// default `'linear'` output; the other four methods map directly to
/// the NumPy classics via [`QuantileMethod`].
///
/// `data` is consumed: it is partitioned in place so the caller should
/// pass an owned buffer (or a clone they no longer need).
fn quantile_select<T: Float>(mut data: Vec<T>, q: T, method: QuantileMethod) -> T {
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

    // Fast exit: exact hit on an integral index. All five methods agree
    // at integer positions.
    if lo_i == n - 1 || frac == T::zero() {
        return lo_val;
    }

    // After the partial select, every element in `data[lo_i + 1..]` is
    // ordered-after `lo_val`; the smallest of them is the
    // `(lo_i + 1)`-th smallest element overall, which is the `hi_val`
    // the non-Lower methods need.
    let hi_val = data[lo_i + 1..]
        .iter()
        .copied()
        .reduce(|a, b| match cmp(&a, &b) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => a,
            std::cmp::Ordering::Greater => b,
        })
        .unwrap_or(lo_val);

    match method {
        QuantileMethod::Linear => lo_val * (T::one() - frac) + hi_val * frac,
        QuantileMethod::Lower => lo_val,
        QuantileMethod::Higher => hi_val,
        QuantileMethod::Nearest => {
            // frac != 0 here (checked above). Compare against 0.5 and
            // break ties toward the even index to match NumPy's
            // round-half-to-even convention.
            let half = T::from(0.5).unwrap();
            if frac < half {
                lo_val
            } else if frac > half {
                hi_val
            } else if lo_i % 2 == 0 {
                lo_val
            } else {
                hi_val
            }
        }
        QuantileMethod::Midpoint => (lo_val + hi_val) / (T::one() + T::one()),
    }
}

/// Compute quantile on a lane using a caller-chosen interpolation method.
fn lane_quantile_with_method<T: Float>(lane: &[T], q: T, method: QuantileMethod) -> T {
    quantile_select(lane.to_vec(), q, method)
}

/// Compute quantile on a lane, excluding NaNs.
fn lane_nanquantile<T: Float>(lane: &[T], q: T) -> T {
    let filtered: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    if filtered.is_empty() {
        return T::nan();
    }
    quantile_select(filtered, q, QuantileMethod::Linear)
}

// ---------------------------------------------------------------------------
// quantile
// ---------------------------------------------------------------------------

/// Compute the q-th quantile of array data along a given axis.
///
/// `q` must be in \[0, 1\]. Uses linear interpolation (NumPy default method).
/// Equivalent to `numpy.quantile`. See [`quantile_with_method`] for the
/// variant that accepts a [`QuantileMethod`] selector (#462).
pub fn quantile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    quantile_with_method(a, q, axis, QuantileMethod::Linear)
}

/// Compute the q-th quantile of array data along a given axis using a
/// specific interpolation method.
///
/// Equivalent to `numpy.quantile(a, q, axis=axis, method=method)` for the
/// five classic methods exposed via [`QuantileMethod`]. `q` must be in
/// \[0, 1\]. Added for #462.
///
/// # Errors
/// - `FerrayError::InvalidValue` if `q` is outside \[0, 1\] or the array
///   is empty.
/// - `FerrayError::AxisOutOfBounds` if `axis` is out of range.
pub fn quantile_with_method<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
    method: QuantileMethod,
) -> FerrayResult<Array<T, IxDyn>>
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
            let val = lane_quantile_with_method(&data, q, method);
            make_result(&[], vec![val])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| {
                lane_quantile_with_method(lane, q, method)
            });
            make_result(&out_s, result)
        }
    }
}

// ---------------------------------------------------------------------------
// percentile
// ---------------------------------------------------------------------------

/// Compute the q-th percentile of array data along a given axis.
///
/// `q` must be in \[0, 100\]. Uses linear interpolation. See
/// [`percentile_with_method`] for the variant that accepts a
/// [`QuantileMethod`] selector.
///
/// Equivalent to `numpy.percentile`.
pub fn percentile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    percentile_with_method(a, q, axis, QuantileMethod::Linear)
}

/// Compute the q-th percentile of array data along a given axis using a
/// specific interpolation method.
///
/// `q` must be in \[0, 100\]. Equivalent to
/// `numpy.percentile(a, q, axis=axis, method=method)` for the five
/// classic methods exposed via [`QuantileMethod`].
pub fn percentile_with_method<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
    method: QuantileMethod,
) -> FerrayResult<Array<T, IxDyn>>
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
    quantile_with_method(a, q / hundred, axis, method)
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

/// Quantile, skipping NaN values. Equivalent to `numpy.nanquantile`
/// (#93 — was previously private, only accessible indirectly through
/// `nanmedian`/`nanpercentile`).
pub fn nanquantile<T, D>(a: &Array<T, D>, q: T, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
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

    // ---- quantile interpolation methods (#462) ----

    fn arr_1_5() -> Array<f64, Ix1> {
        Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap()
    }

    #[test]
    fn test_quantile_method_linear_matches_legacy() {
        // Default quantile uses Linear; explicit Linear must match.
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let legacy = quantile(&a, 0.25, None).unwrap();
        let with_flag =
            quantile_with_method(&a, 0.25, None, QuantileMethod::Linear).unwrap();
        assert_eq!(
            legacy.iter().next().unwrap(),
            with_flag.iter().next().unwrap()
        );
    }

    #[test]
    fn test_quantile_method_lower() {
        // n=5, q=0.25 → idx=1.0 → lo_i=1, frac=0.0 (integer index)
        // All methods agree: result = 2.0
        let a = arr_1_5();
        let q = quantile_with_method(&a, 0.25, None, QuantileMethod::Lower).unwrap();
        assert!((q.iter().next().unwrap() - 2.0).abs() < 1e-12);

        // n=4, q=0.25 → idx=0.75, lo_i=0, frac=0.75 → Lower = lo_val = 1.0
        let a4 = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = quantile_with_method(&a4, 0.25, None, QuantileMethod::Lower).unwrap();
        assert!((q.iter().next().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_higher() {
        // n=4, q=0.25 → idx=0.75 → Higher = hi_val = 2.0
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = quantile_with_method(&a, 0.25, None, QuantileMethod::Higher).unwrap();
        assert!((q.iter().next().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_nearest_round_down() {
        // n=4, q=0.2 → idx=0.6, frac=0.6 > 0.5 → pick hi_val (index 1) = 2.0
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = quantile_with_method(&a, 0.2, None, QuantileMethod::Nearest).unwrap();
        assert!((q.iter().next().unwrap() - 2.0).abs() < 1e-12);

        // n=4, q=0.1 → idx=0.3, frac=0.3 < 0.5 → pick lo_val (index 0) = 1.0
        let q2 = quantile_with_method(&a, 0.1, None, QuantileMethod::Nearest).unwrap();
        assert!((q2.iter().next().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_nearest_tie_even() {
        // n=5, q=0.125 → idx=0.5, frac=0.5, lo_i=0 (even) → pick lo_val = 1.0
        let a = arr_1_5();
        let q = quantile_with_method(&a, 0.125, None, QuantileMethod::Nearest).unwrap();
        assert!((q.iter().next().unwrap() - 1.0).abs() < 1e-12);

        // n=5, q=0.375 → idx=1.5, frac=0.5, lo_i=1 (odd) → pick hi_val = 3.0
        let q2 = quantile_with_method(&a, 0.375, None, QuantileMethod::Nearest).unwrap();
        assert!((q2.iter().next().unwrap() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_midpoint() {
        // n=4, q=0.25 → idx=0.75, lo_val=1.0, hi_val=2.0 → midpoint = 1.5
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let q = quantile_with_method(&a, 0.25, None, QuantileMethod::Midpoint).unwrap();
        assert!((q.iter().next().unwrap() - 1.5).abs() < 1e-12);

        // n=4, q=0.75 → idx=2.25, lo_val=3.0, hi_val=4.0 → midpoint = 3.5
        let q2 = quantile_with_method(&a, 0.75, None, QuantileMethod::Midpoint).unwrap();
        assert!((q2.iter().next().unwrap() - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_integer_index_all_agree() {
        // n=5, q=0.5 → idx=2.0, exactly on sorted[2]. All five methods
        // must return the same value.
        let a = arr_1_5();
        let linear =
            quantile_with_method(&a, 0.5, None, QuantileMethod::Linear).unwrap();
        let lower =
            quantile_with_method(&a, 0.5, None, QuantileMethod::Lower).unwrap();
        let higher =
            quantile_with_method(&a, 0.5, None, QuantileMethod::Higher).unwrap();
        let nearest =
            quantile_with_method(&a, 0.5, None, QuantileMethod::Nearest).unwrap();
        let midpoint =
            quantile_with_method(&a, 0.5, None, QuantileMethod::Midpoint).unwrap();
        let expected = 3.0;
        assert!((linear.iter().next().unwrap() - expected).abs() < 1e-12);
        assert!((lower.iter().next().unwrap() - expected).abs() < 1e-12);
        assert!((higher.iter().next().unwrap() - expected).abs() < 1e-12);
        assert!((nearest.iter().next().unwrap() - expected).abs() < 1e-12);
        assert!((midpoint.iter().next().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_method_axis_variant() {
        // Per-row quantile with a non-linear method.
        use ferray_core::Ix2;
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        // q=0.25, n=4, idx=0.75 → Lower picks lo_val (index 0).
        let r = quantile_with_method(&a, 0.25, Some(1), QuantileMethod::Lower).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[1.0, 10.0]);
    }

    #[test]
    fn test_percentile_with_method_50() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // q=50 (percentile) → 0.5 (quantile), n=4, idx=1.5
        // Linear: 2.5, Lower: 2.0, Higher: 3.0, Nearest (tie, lo_i=1 odd): 3.0, Midpoint: 2.5
        let lin =
            percentile_with_method(&a, 50.0, None, QuantileMethod::Linear).unwrap();
        let lo =
            percentile_with_method(&a, 50.0, None, QuantileMethod::Lower).unwrap();
        let hi =
            percentile_with_method(&a, 50.0, None, QuantileMethod::Higher).unwrap();
        let nr =
            percentile_with_method(&a, 50.0, None, QuantileMethod::Nearest).unwrap();
        let mp =
            percentile_with_method(&a, 50.0, None, QuantileMethod::Midpoint).unwrap();
        assert!((lin.iter().next().unwrap() - 2.5).abs() < 1e-12);
        assert!((lo.iter().next().unwrap() - 2.0).abs() < 1e-12);
        assert!((hi.iter().next().unwrap() - 3.0).abs() < 1e-12);
        assert!((nr.iter().next().unwrap() - 3.0).abs() < 1e-12);
        assert!((mp.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_with_method_rejects_out_of_range() {
        let a = arr_1_5();
        assert!(
            percentile_with_method(&a, -1.0, None, QuantileMethod::Linear).is_err()
        );
        assert!(
            percentile_with_method(&a, 101.0, None, QuantileMethod::Linear).is_err()
        );
    }
}
