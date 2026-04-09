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
/// / `median` friends.
///
/// Matches all 13 NumPy quantile methods (#462, #566). The continuous
/// methods use the Hyndman-Fan 1996 `(alpha, beta)` parameterization:
/// `virtual_index = n*q + alpha + q*(1 - alpha - beta) - 1` with
/// linear interpolation between the two bracketing sorted elements.
/// The discrete methods compute an integer index via method-specific
/// rules and return the exact sorted element (no interpolation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantileMethod {
    /// NumPy default. Continuous with `alpha = beta = 1`. Returns
    /// `lo_val * (1 - frac) + hi_val * frac`.
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
    /// Discrete method, Hyndman definition 1. Returns
    /// `sorted[ceil(n * q) - 1]` — a step function with jumps at
    /// `k / n`. NumPy's `'inverted_cdf'`.
    InvertedCdf,
    /// Discrete method, Hyndman definition 2. Same as `InvertedCdf`
    /// except that when `n * q` is an integer the result is the
    /// average of the two bracketing sorted elements. NumPy's
    /// `'averaged_inverted_cdf'`.
    AveragedInvertedCdf,
    /// Discrete method, Hyndman definition 3. `sorted[k]` where
    /// `k = round_half_to_even(n*q - 0.5)`. NumPy's
    /// `'closest_observation'`.
    ClosestObservation,
    /// Continuous method, Hyndman definition 4. `alpha = 0`, `beta = 1`.
    /// NumPy's `'interpolated_inverted_cdf'`.
    InterpolatedInvertedCdf,
    /// Continuous method, Hyndman definition 5. `alpha = beta = 0.5`.
    /// NumPy's `'hazen'`.
    Hazen,
    /// Continuous method, Hyndman definition 6. `alpha = beta = 0`.
    /// NumPy's `'weibull'`.
    Weibull,
    /// Continuous method, Hyndman definition 8. `alpha = beta = 1/3`.
    /// NumPy's `'median_unbiased'`.
    MedianUnbiased,
    /// Continuous method, Hyndman definition 9. `alpha = beta = 3/8`.
    /// NumPy's `'normal_unbiased'`.
    NormalUnbiased,
}

/// Compute a Hyndman-Fan virtual index for the continuous quantile
/// methods. Returns `(lo_i, gamma)` with `lo_i` clamped to `[0, n - 1]`
/// and `gamma` in `[0, 1]`.
///
/// `virtual_index = n * q + alpha + q * (1 - alpha - beta) - 1`
#[inline]
fn continuous_vidx<T: Float>(n: usize, q: T, alpha: T, beta: T) -> (usize, T) {
    let nf = T::from(n).unwrap();
    let zero = T::zero();
    let one = T::one();
    let n_minus_1 = T::from(n - 1).unwrap();

    let vidx = nf * q + alpha + q * (one - alpha - beta) - one;

    // Clamp the virtual index into the addressable range.
    let vidx_clamped = if vidx < zero {
        zero
    } else if vidx > n_minus_1 {
        n_minus_1
    } else {
        vidx
    };

    let lo = vidx_clamped.floor();
    let lo_i = lo.to_usize().unwrap_or(0).min(n - 1);
    let gamma = vidx_clamped - lo;
    (lo_i, gamma)
}

/// Round-half-to-even (banker's rounding) for a float.
#[inline]
fn round_half_to_even<T: Float>(x: T) -> T {
    let floor = x.floor();
    let frac = x - floor;
    let half = T::from(0.5).unwrap();
    if frac < half {
        floor
    } else if frac > half {
        floor + T::one()
    } else {
        // Exactly 0.5 — pick the even neighbor.
        let floor_i = floor.to_i64().unwrap_or(0);
        if floor_i.rem_euclid(2) == 0 {
            floor
        } else {
            floor + T::one()
        }
    }
}

/// Compute `(lo_i, gamma)` for a given quantile method, where the
/// result of the quantile is `(1 - gamma) * sorted[lo_i] + gamma *
/// sorted[lo_i + 1]`. For all discrete methods and for integer virtual
/// indices `gamma = 0`, which short-circuits to `sorted[lo_i]` and
/// avoids the second-pass scan for `hi_val`.
fn method_index_and_gamma<T: Float>(n: usize, q: T, method: QuantileMethod) -> (usize, T) {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let nf = T::from(n).unwrap();

    match method {
        // --- Continuous methods via (alpha, beta). Linear is (1, 1). ---
        QuantileMethod::Linear => continuous_vidx(n, q, one, one),
        QuantileMethod::Weibull => continuous_vidx(n, q, zero, zero),
        QuantileMethod::Hazen => continuous_vidx(n, q, half, half),
        QuantileMethod::InterpolatedInvertedCdf => continuous_vidx(n, q, zero, one),
        QuantileMethod::MedianUnbiased => {
            let third = T::from(1.0 / 3.0).unwrap();
            continuous_vidx(n, q, third, third)
        }
        QuantileMethod::NormalUnbiased => {
            let ae = T::from(3.0 / 8.0).unwrap();
            continuous_vidx(n, q, ae, ae)
        }

        // --- Old discrete classics: reuse the linear virtual index
        //     and apply their specific rounding rules.
        QuantileMethod::Lower => {
            let vidx = q * T::from(n - 1).unwrap();
            let lo_i = vidx.floor().to_usize().unwrap_or(0).min(n - 1);
            (lo_i, zero)
        }
        QuantileMethod::Higher => {
            let vidx = q * T::from(n - 1).unwrap();
            let lo = vidx.floor();
            let lo_i = lo.to_usize().unwrap_or(0).min(n - 1);
            let frac = vidx - lo;
            // If there's a fractional part, gamma=1 picks sorted[lo_i+1]
            // exactly. If not, lo_val itself is the ceiling.
            if frac > zero && lo_i + 1 < n {
                (lo_i, one)
            } else {
                (lo_i, zero)
            }
        }
        QuantileMethod::Nearest => {
            let vidx = q * T::from(n - 1).unwrap();
            let lo = vidx.floor();
            let lo_i = lo.to_usize().unwrap_or(0).min(n - 1);
            let frac = vidx - lo;
            if frac < half {
                (lo_i, zero)
            } else if frac > half {
                if lo_i + 1 < n {
                    (lo_i, one)
                } else {
                    (lo_i, zero)
                }
            } else {
                // Tie: round to the even lo_i.
                if lo_i % 2 == 0 || lo_i + 1 >= n {
                    (lo_i, zero)
                } else {
                    (lo_i, one)
                }
            }
        }
        QuantileMethod::Midpoint => {
            let vidx = q * T::from(n - 1).unwrap();
            let lo = vidx.floor();
            let lo_i = lo.to_usize().unwrap_or(0).min(n - 1);
            let frac = vidx - lo;
            if frac > zero && lo_i + 1 < n {
                (lo_i, half)
            } else {
                (lo_i, zero)
            }
        }

        // --- Discrete step-function methods ---
        QuantileMethod::InvertedCdf => {
            // k = ceil(n * q) - 1, clamped to [0, n - 1].
            let nq = nf * q;
            let k = if nq <= zero {
                0
            } else {
                nq.ceil()
                    .to_usize()
                    .unwrap_or(0)
                    .saturating_sub(1)
                    .min(n - 1)
            };
            (k, zero)
        }
        QuantileMethod::AveragedInvertedCdf => {
            // Same as InvertedCdf for non-integer n*q; for exact
            // integer n*q the result is (sorted[k-1] + sorted[k]) / 2.
            let nq = nf * q;
            let floor_nq = nq.floor();
            let is_integer = nq == floor_nq;
            if is_integer && nq > zero && nq < nf {
                // nq is in (0, n), so k = floor(nq) = floor(nq) - 1 + 1
                // and we average sorted[k-1] and sorted[k].
                let k = floor_nq.to_usize().unwrap_or(0);
                let lo_i = k.saturating_sub(1).min(n - 1);
                if lo_i + 1 < n {
                    (lo_i, half)
                } else {
                    (lo_i, zero)
                }
            } else {
                // Non-integer, or at the boundary — same as InvertedCdf.
                let k = if nq <= zero {
                    0
                } else {
                    nq.ceil()
                        .to_usize()
                        .unwrap_or(0)
                        .saturating_sub(1)
                        .min(n - 1)
                };
                (k, zero)
            }
        }
        QuantileMethod::ClosestObservation => {
            // k = round_half_to_even(n * q - 0.5), clamped to [0, n-1].
            let nq = nf * q;
            let adj = nq - half;
            let rounded = round_half_to_even(adj);
            let k = if rounded < zero {
                0
            } else {
                rounded.to_usize().unwrap_or(0).min(n - 1)
            };
            (k, zero)
        }
    }
}

/// Compute a single quantile value from an unsorted slice using
/// `select_nth_unstable_by` rather than a full `sort_by`.
///
/// The selection algorithm gives an O(n) average-time path (quickselect)
/// instead of the O(n log n) full sort the previous implementation used
/// (#175). All 13 NumPy quantile methods are supported: every method
/// produces a `(lo_i, gamma)` pair via [`method_index_and_gamma`] and
/// the kernel applies a single uniform interpolation formula.
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
    let (lo_i, gamma) = method_index_and_gamma(n, q, method);

    // First selection: place the lo_i-th smallest at position lo_i.
    data.select_nth_unstable_by(lo_i, cmp);
    let lo_val = data[lo_i];

    // Fast exit: no interpolation needed for discrete methods or
    // whenever the virtual index landed exactly on an integer position.
    if gamma == T::zero() || lo_i >= n - 1 {
        return lo_val;
    }

    // After the partial select, every element in `data[lo_i + 1..]` is
    // ordered-after `lo_val`; the smallest of them is the
    // `(lo_i + 1)`-th smallest element overall, which is the `hi_val`
    // the interpolation formula needs.
    let hi_val = data[lo_i + 1..]
        .iter()
        .copied()
        .reduce(|a, b| match cmp(&a, &b) {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => a,
            std::cmp::Ordering::Greater => b,
        })
        .unwrap_or(lo_val);

    (T::one() - gamma) * lo_val + gamma * hi_val
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
pub fn nanquantile<T, D>(
    a: &Array<T, D>,
    q: T,
    axis: Option<usize>,
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
        let with_flag = quantile_with_method(&a, 0.25, None, QuantileMethod::Linear).unwrap();
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
        let linear = quantile_with_method(&a, 0.5, None, QuantileMethod::Linear).unwrap();
        let lower = quantile_with_method(&a, 0.5, None, QuantileMethod::Lower).unwrap();
        let higher = quantile_with_method(&a, 0.5, None, QuantileMethod::Higher).unwrap();
        let nearest = quantile_with_method(&a, 0.5, None, QuantileMethod::Nearest).unwrap();
        let midpoint = quantile_with_method(&a, 0.5, None, QuantileMethod::Midpoint).unwrap();
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
        let lin = percentile_with_method(&a, 50.0, None, QuantileMethod::Linear).unwrap();
        let lo = percentile_with_method(&a, 50.0, None, QuantileMethod::Lower).unwrap();
        let hi = percentile_with_method(&a, 50.0, None, QuantileMethod::Higher).unwrap();
        let nr = percentile_with_method(&a, 50.0, None, QuantileMethod::Nearest).unwrap();
        let mp = percentile_with_method(&a, 50.0, None, QuantileMethod::Midpoint).unwrap();
        assert!((lin.iter().next().unwrap() - 2.5).abs() < 1e-12);
        assert!((lo.iter().next().unwrap() - 2.0).abs() < 1e-12);
        assert!((hi.iter().next().unwrap() - 3.0).abs() < 1e-12);
        assert!((nr.iter().next().unwrap() - 3.0).abs() < 1e-12);
        assert!((mp.iter().next().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_percentile_with_method_rejects_out_of_range() {
        let a = arr_1_5();
        assert!(percentile_with_method(&a, -1.0, None, QuantileMethod::Linear).is_err());
        assert!(percentile_with_method(&a, 101.0, None, QuantileMethod::Linear).is_err());
    }

    // ---- remaining 8 NumPy quantile methods (#566) ----
    //
    // Hand-verified expected values come from Hyndman & Fan 1996 / NumPy
    // source. For continuous methods the virtual index is
    //   vidx = n*q + alpha + q*(1 - alpha - beta) - 1
    // and the result is (1 - gamma) * sorted[lo_i] + gamma * sorted[lo_i+1]
    // with gamma = frac(vidx_clamped), lo_i = floor(vidx_clamped).

    fn arr_1_4() -> Array<f64, Ix1> {
        Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn test_quantile_weibull_q_half() {
        // n=4, q=0.5, alpha=beta=0:
        //   vidx = 4*0.5 + 0 + 0.5*(1 - 0 - 0) - 1 = 2 + 0.5 - 1 = 1.5
        //   lo_i=1, gamma=0.5 → 0.5*sorted[1] + 0.5*sorted[2] = 0.5*2 + 0.5*3 = 2.5
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::Weibull).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_weibull_q_quarter() {
        // n=4, q=0.25, alpha=beta=0:
        //   vidx = 4*0.25 + 0 + 0.25 - 1 = 0.25
        //   lo_i=0, gamma=0.25 → 0.75*1 + 0.25*2 = 1.25
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.25, None, QuantileMethod::Weibull).unwrap();
        assert!((q.iter().next().copied().unwrap() - 1.25).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_hazen_q_quarter() {
        // n=4, q=0.25, alpha=beta=0.5:
        //   vidx = 4*0.25 + 0.5 + 0.25*(1 - 1) - 1 = 1 + 0.5 - 1 = 0.5
        //   lo_i=0, gamma=0.5 → 0.5*1 + 0.5*2 = 1.5
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.25, None, QuantileMethod::Hazen).unwrap();
        assert!((q.iter().next().copied().unwrap() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_median_unbiased_q_half() {
        // n=4, q=0.5, alpha=beta=1/3:
        //   vidx = 4*0.5 + 1/3 + 0.5*(1 - 2/3) - 1
        //        = 2 + 1/3 + 0.5*(1/3) - 1
        //        = 2 + 1/3 + 1/6 - 1
        //        = 2 + 0.5 - 1 = 1.5
        //   lo_i=1, gamma=0.5 → 2.5 (same as Linear's median for n=4)
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::MedianUnbiased).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_normal_unbiased_q_half() {
        // n=4, q=0.5, alpha=beta=3/8:
        //   vidx = 4*0.5 + 3/8 + 0.5*(1 - 6/8) - 1
        //        = 2 + 0.375 + 0.5*0.25 - 1
        //        = 2 + 0.375 + 0.125 - 1
        //        = 1.5
        //   → 2.5 (matches Linear median at n=4)
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::NormalUnbiased).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_interpolated_inverted_cdf_q_half() {
        // n=4, q=0.5, alpha=0, beta=1:
        //   vidx = 4*0.5 + 0 + 0.5*(1 - 0 - 1) - 1
        //        = 2 + 0 + 0 - 1 = 1
        //   lo_i=1, gamma=0 → sorted[1] = 2
        // This is different from the median-family methods.
        let a = arr_1_4();
        let q =
            quantile_with_method(&a, 0.5, None, QuantileMethod::InterpolatedInvertedCdf).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_inverted_cdf_q_half() {
        // n=4, q=0.5: nq=2, ceil-1=1 → sorted[1] = 2
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::InvertedCdf).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_inverted_cdf_step_function() {
        // n=5, q values straddling the k/n steps:
        //   q=0.19 → nq=0.95 → ceil-1 = 0 → sorted[0] = 1
        //   q=0.21 → nq=1.05 → ceil-1 = 1 → sorted[1] = 2
        let a = arr_1_5();
        let q1 = quantile_with_method(&a, 0.19, None, QuantileMethod::InvertedCdf).unwrap();
        assert!((q1.iter().next().copied().unwrap() - 1.0).abs() < 1e-12);
        let q2 = quantile_with_method(&a, 0.21, None, QuantileMethod::InvertedCdf).unwrap();
        assert!((q2.iter().next().copied().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_averaged_inverted_cdf_integer_nq_averages() {
        // n=4, q=0.5 → nq=2 (integer) → average of sorted[1] and sorted[2]
        // = 0.5*2 + 0.5*3 = 2.5
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::AveragedInvertedCdf).unwrap();
        assert!((q.iter().next().copied().unwrap() - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_averaged_inverted_cdf_non_integer_nq_matches_inverted_cdf() {
        // n=5, q=0.3 → nq=1.5 (non-integer) → same as InvertedCdf:
        //   ceil(1.5) - 1 = 1 → sorted[1] = 2
        let a = arr_1_5();
        let q1 = quantile_with_method(&a, 0.3, None, QuantileMethod::AveragedInvertedCdf).unwrap();
        let q2 = quantile_with_method(&a, 0.3, None, QuantileMethod::InvertedCdf).unwrap();
        assert_eq!(
            q1.iter().next().copied().unwrap(),
            q2.iter().next().copied().unwrap()
        );
        assert!((q1.iter().next().copied().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_closest_observation_half_to_even() {
        // n=4, q=0.5:
        //   nq=2, adj=1.5, round_half_to_even(1.5) = 2 (2 is even) → k=2
        //   → sorted[2] = 3
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.5, None, QuantileMethod::ClosestObservation).unwrap();
        assert!((q.iter().next().copied().unwrap() - 3.0).abs() < 1e-12);

        // n=4, q=0.125:
        //   nq=0.5, adj=0, round_half_to_even(0) = 0 → k=0 → sorted[0] = 1
        let q2 = quantile_with_method(&a, 0.125, None, QuantileMethod::ClosestObservation).unwrap();
        assert!((q2.iter().next().copied().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_closest_observation_nq_0_875_rounds_up() {
        // n=4, q=0.875:
        //   nq=3.5, adj=3, round_half_to_even(3) = 3 → k=3 → sorted[3] = 4
        let a = arr_1_4();
        let q = quantile_with_method(&a, 0.875, None, QuantileMethod::ClosestObservation).unwrap();
        assert!((q.iter().next().copied().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_quantile_continuous_methods_agree_at_q_0_and_q_1() {
        // At q=0 all continuous methods should return the min; at q=1 all
        // should return the max (clamping). This is a sanity check that
        // the virtual-index clamp works in every branch.
        let a = arr_1_5();
        let methods = [
            QuantileMethod::Linear,
            QuantileMethod::Weibull,
            QuantileMethod::Hazen,
            QuantileMethod::InterpolatedInvertedCdf,
            QuantileMethod::MedianUnbiased,
            QuantileMethod::NormalUnbiased,
        ];
        for &m in &methods {
            let q0 = quantile_with_method(&a, 0.0, None, m).unwrap();
            let q1 = quantile_with_method(&a, 1.0, None, m).unwrap();
            assert!(
                (q0.iter().next().copied().unwrap() - 1.0).abs() < 1e-12,
                "method {m:?} at q=0 should be min"
            );
            assert!(
                (q1.iter().next().copied().unwrap() - 5.0).abs() < 1e-12,
                "method {m:?} at q=1 should be max"
            );
        }
    }

    #[test]
    fn test_quantile_discrete_methods_agree_at_q_1() {
        // At q=1.0 every method returns the max.
        let a = arr_1_5();
        let methods = [
            QuantileMethod::InvertedCdf,
            QuantileMethod::AveragedInvertedCdf,
            QuantileMethod::ClosestObservation,
        ];
        for &m in &methods {
            let q = quantile_with_method(&a, 1.0, None, m).unwrap();
            assert!(
                (q.iter().next().copied().unwrap() - 5.0).abs() < 1e-12,
                "method {m:?} at q=1 should be max"
            );
        }
    }

    #[test]
    fn test_quantile_all_13_methods_at_integer_index_agree() {
        // When the virtual index lands on a real integer position,
        // EVERY method should return that exact sorted element because
        // each method's dispatch produces gamma=0 or the continuous
        // formula yields fractional = 0. On n=5 with q=0.5, the linear
        // virtual index is exactly 2.0 → sorted[2] = 3.
        let a = arr_1_5();
        let all_methods = [
            QuantileMethod::Linear,
            QuantileMethod::Lower,
            QuantileMethod::Higher,
            QuantileMethod::Nearest,
            QuantileMethod::Midpoint,
            QuantileMethod::Weibull,
            QuantileMethod::Hazen,
            QuantileMethod::MedianUnbiased,
            QuantileMethod::NormalUnbiased,
            // These three use different virtual indices so they MAY
            // disagree at q=0.5 even for odd n; check them separately.
        ];
        for &m in &all_methods {
            let r = quantile_with_method(&a, 0.5, None, m).unwrap();
            assert!(
                (r.iter().next().copied().unwrap() - 3.0).abs() < 1e-12,
                "method {m:?} at odd n, q=0.5 should be 3.0"
            );
        }
    }

    #[test]
    fn test_quantile_method_axis_variant_weibull() {
        use ferray_core::Ix2;
        // (2, 4) rows; per-row Weibull quantile at q=0.5 on [1,2,3,4]
        // is 2.5, and on [10,20,30,40] is 25.0.
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 4]),
            vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        )
        .unwrap();
        let r = quantile_with_method(&a, 0.5, Some(1), QuantileMethod::Weibull).unwrap();
        assert_eq!(r.shape(), &[2]);
        let s = r.as_slice().unwrap();
        assert!((s[0] - 2.5).abs() < 1e-12);
        assert!((s[1] - 25.0).abs() < 1e-12);
    }
}
