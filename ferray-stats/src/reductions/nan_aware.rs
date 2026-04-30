// ferray-stats: NaN-aware reductions — nansum, nanprod, nanmin, nanmax, nanmean, nanvar, nanstd (REQ-3, REQ-4)
// Also nancumsum, nancumprod (REQ-2b)

use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use super::{
    borrow_data, make_result, output_shape, reduce_axis_general, reduce_axis_general_u64,
    validate_axis,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Iterate over the non-NaN elements of a lane without allocating a
/// filtered buffer. Every `lane_nan`* helper used to open-code this
/// filter; consolidating it keeps the "skip NaN" policy in one place
/// (#164).
#[inline]
fn non_nan<T: Float>(lane: &[T]) -> impl Iterator<Item = T> + '_ {
    lane.iter().copied().filter(|x| !x.is_nan())
}

/// Single-pass sum-and-count of non-NaN elements. Shared between
/// `lane_nanmean` and `lane_nanvar`.
#[inline]
fn nan_sum_count<T: Float>(lane: &[T]) -> (T, usize) {
    let mut sum = T::zero();
    let mut count = 0usize;
    for x in non_nan(lane) {
        sum = sum + x;
        count += 1;
    }
    (sum, count)
}

/// Sum a lane, skipping NaN values. Returns zero for all-NaN.
fn lane_nansum<T: Float>(lane: &[T]) -> T {
    non_nan(lane).fold(T::zero(), |a, b| a + b)
}

/// Product of a lane, skipping NaN values. Returns one for all-NaN.
fn lane_nanprod<T: Float>(lane: &[T]) -> T {
    non_nan(lane).fold(T::one(), |a, b| a * b)
}

/// Mean of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmean<T: Float>(lane: &[T]) -> T {
    let (sum, count) = nan_sum_count(lane);
    if count == 0 {
        T::nan()
    } else {
        sum / T::from(count).unwrap()
    }
}

/// Variance of a lane, skipping NaN values. Two-pass algorithm.
fn lane_nanvar<T: Float>(lane: &[T], ddof: usize) -> T {
    let (sum, count) = nan_sum_count(lane);
    if count <= ddof {
        return T::nan();
    }
    let mean = sum / T::from(count).unwrap();

    // Pass 2: sum of squared deviations
    let sq_sum: T = non_nan(lane)
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .fold(T::zero(), |a, b| a + b);
    sq_sum / T::from(count - ddof).unwrap()
}

/// Min of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmin<T: Float>(lane: &[T]) -> T {
    non_nan(lane)
        .reduce(|a, b| if a <= b { a } else { b })
        .unwrap_or_else(T::nan)
}

/// Max of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmax<T: Float>(lane: &[T]) -> T {
    non_nan(lane)
        .reduce(|a, b| if a >= b { a } else { b })
        .unwrap_or_else(T::nan)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sum of array elements, treating NaN as zero.
///
/// Equivalent to `numpy.nansum`.
pub fn nansum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let total = lane_nansum(&data);
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, lane_nansum);
            make_result(&out_s, result)
        }
    }
}

/// Product of array elements, treating NaN as one.
///
/// Equivalent to `numpy.nanprod`.
pub fn nanprod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let total = lane_nanprod(&data);
            make_result(&[], vec![total])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, lane_nanprod);
            make_result(&out_s, result)
        }
    }
}

/// Mean of array elements, skipping NaN. Returns NaN for all-NaN slices.
///
/// Equivalent to `numpy.nanmean`.
pub fn nanmean<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let m = lane_nanmean(&data);
            make_result(&[], vec![m])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, lane_nanmean);
            make_result(&out_s, result)
        }
    }
}

/// Variance of array elements, skipping NaN.
///
/// Equivalent to `numpy.nanvar`.
pub fn nanvar<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let data = borrow_data(a);
    match axis {
        None => {
            let v = lane_nanvar(&data, ddof);
            make_result(&[], vec![v])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, |lane| lane_nanvar(lane, ddof));
            make_result(&out_s, result)
        }
    }
}

/// Standard deviation of array elements, skipping NaN.
///
/// Equivalent to `numpy.nanstd`.
pub fn nanstd<T, D>(
    a: &Array<T, D>,
    axis: Option<usize>,
    ddof: usize,
) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    let v = nanvar(a, axis, ddof)?;
    let data: Vec<T> = v.iter().map(|x| x.sqrt()).collect();
    make_result(v.shape(), data)
}

/// Minimum of array elements, skipping NaN.
///
/// Equivalent to `numpy.nanmin`.
pub fn nanmin<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute nanmin of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let m = lane_nanmin(&data);
            make_result(&[], vec![m])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, lane_nanmin);
            make_result(&out_s, result)
        }
    }
}

/// Maximum of array elements, skipping NaN.
///
/// Equivalent to `numpy.nanmax`.
pub fn nanmax<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute nanmax of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let m = lane_nanmax(&data);
            make_result(&[], vec![m])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let out_s = output_shape(shape, ax);
            let result = reduce_axis_general(&data, shape, ax, lane_nanmax);
            make_result(&out_s, result)
        }
    }
}

/// Index of the minimum value, skipping NaN. Errors on all-NaN input.
///
/// Equivalent to `numpy.nanargmin`. For axis=None returns a flat-index
/// scalar; for axis=Some(ax) returns indices along that axis.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` for an empty array or an all-NaN
/// slice/lane (matching NumPy's `ValueError` from `nanargmin`).
pub fn nanargmin<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute nanargmin of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let pick = data
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_nan())
                .reduce(|(ai, av), (bi, bv)| if av <= bv { (ai, av) } else { (bi, bv) });
            let idx = pick
                .ok_or_else(|| FerrayError::invalid_value("nanargmin: all-NaN slice encountered"))?
                .0 as u64;
            make_result(&[], vec![idx])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            // The lane closure is infallible, so it returns `u64::MAX` for
            // an all-NaN lane and we surface the ValueError after the
            // reduction completes.
            let result = reduce_axis_general_u64(&data, shape, ax, |lane| {
                lane.iter()
                    .enumerate()
                    .filter(|(_, x)| !x.is_nan())
                    .reduce(|(ai, av), (bi, bv)| if av <= bv { (ai, av) } else { (bi, bv) })
                    .map_or(u64::MAX, |(i, _)| i as u64)
            });
            if result.contains(&u64::MAX) {
                return Err(FerrayError::invalid_value(
                    "nanargmin: all-NaN slice encountered along axis",
                ));
            }
            let out_s = output_shape(shape, ax);
            make_result(&out_s, result)
        }
    }
}

/// Index of the maximum value, skipping NaN. Errors on all-NaN input.
///
/// Equivalent to `numpy.nanargmax`. See [`nanargmin`] for axis semantics.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` for an empty array or an all-NaN
/// slice/lane.
pub fn nanargmax<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<u64, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrayError::invalid_value(
            "cannot compute nanargmax of empty array",
        ));
    }
    let data = borrow_data(a);
    match axis {
        None => {
            let pick = data
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_nan())
                .reduce(|(ai, av), (bi, bv)| if av >= bv { (ai, av) } else { (bi, bv) });
            let idx = pick
                .ok_or_else(|| FerrayError::invalid_value("nanargmax: all-NaN slice encountered"))?
                .0 as u64;
            make_result(&[], vec![idx])
        }
        Some(ax) => {
            validate_axis(ax, a.ndim())?;
            let shape = a.shape();
            let result = reduce_axis_general_u64(&data, shape, ax, |lane| {
                lane.iter()
                    .enumerate()
                    .filter(|(_, x)| !x.is_nan())
                    .reduce(|(ai, av), (bi, bv)| if av >= bv { (ai, av) } else { (bi, bv) })
                    .map_or(u64::MAX, |(i, _)| i as u64)
            });
            if result.contains(&u64::MAX) {
                return Err(FerrayError::invalid_value(
                    "nanargmax: all-NaN slice encountered along axis",
                ));
            }
            let out_s = output_shape(shape, ax);
            make_result(&out_s, result)
        }
    }
}

/// Cumulative sum, treating NaN as zero.
///
/// Re-exported from `ferray_ufunc::nancumsum`.
pub fn nancumsum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    ferray_ufunc::nancumsum(a, axis)
}

/// Cumulative product, treating NaN as one.
///
/// Re-exported from `ferray_ufunc::nancumprod`.
pub fn nancumprod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    ferray_ufunc::nancumprod(a, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Ix1;

    #[test]
    fn test_nanmean_basic() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
        let m = nanmean(&a, None).unwrap();
        assert!((m.iter().next().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanmean_all_nan() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![f64::NAN, f64::NAN]).unwrap();
        let m = nanmean(&a, None).unwrap();
        assert!(m.iter().next().unwrap().is_nan());
    }

    #[test]
    fn test_nansum_basic() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
        let s = nansum(&a, None).unwrap();
        assert!((s.iter().next().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_nansum_all_nan() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![f64::NAN, f64::NAN]).unwrap();
        let s = nansum(&a, None).unwrap();
        assert!((s.iter().next().unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanmin_nanmax() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![3.0, f64::NAN, 1.0, 4.0]).unwrap();
        let mn = nanmin(&a, None).unwrap();
        let mx = nanmax(&a, None).unwrap();
        assert!((mn.iter().next().unwrap() - 1.0).abs() < 1e-12);
        assert!((mx.iter().next().unwrap() - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_nanvar() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 3.0, 5.0]).unwrap();
        let v = nanvar(&a, None, 0).unwrap();
        // non-nan values: [1, 3, 5], mean=3, var = (4+0+4)/3 = 8/3
        let expected = 8.0 / 3.0;
        assert!((v.iter().next().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_nanstd() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([4]), vec![1.0, f64::NAN, 3.0, 5.0]).unwrap();
        let s = nanstd(&a, None, 0).unwrap();
        let expected = (8.0_f64 / 3.0).sqrt();
        assert!((s.iter().next().unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn test_nanprod() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, f64::NAN, 3.0]).unwrap();
        let p = nanprod(&a, None).unwrap();
        assert!((p.iter().next().unwrap() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_nancumsum() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, f64::NAN, 3.0]).unwrap();
        let cs = nancumsum(&a, None).unwrap();
        let data: Vec<f64> = cs.iter().copied().collect();
        assert!((data[0] - 1.0).abs() < 1e-12);
        assert!((data[1] - 1.0).abs() < 1e-12);
        assert!((data[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_nancumprod() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![2.0, f64::NAN, 3.0]).unwrap();
        let cp = nancumprod(&a, None).unwrap();
        let data: Vec<f64> = cp.iter().copied().collect();
        assert!((data[0] - 2.0).abs() < 1e-12);
        assert!((data[1] - 2.0).abs() < 1e-12);
        assert!((data[2] - 6.0).abs() < 1e-12);
    }

    // -- nanargmin / nanargmax --

    #[test]
    fn test_nanargmin_skips_nan() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, f64::NAN, 1.0, 3.0, f64::NAN])
            .unwrap();
        let r = nanargmin(&a, None).unwrap();
        assert_eq!(r.iter().copied().next().unwrap(), 2);
    }

    #[test]
    fn test_nanargmax_skips_nan() {
        let a = Array::<f64, Ix1>::from_vec(Ix1::new([5]), vec![5.0, f64::NAN, 1.0, 9.0, f64::NAN])
            .unwrap();
        let r = nanargmax(&a, None).unwrap();
        assert_eq!(r.iter().copied().next().unwrap(), 3);
    }

    #[test]
    fn test_nanargmin_all_nan_errs() {
        let a =
            Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![f64::NAN, f64::NAN, f64::NAN]).unwrap();
        assert!(nanargmin(&a, None).is_err());
    }

    #[test]
    fn test_nanargmax_2d_axis() {
        use ferray_core::Ix2;
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0, 4.0],
        )
        .unwrap();
        // Axis-0: per column, max-non-nan index
        // col 0: [1, NaN] → 0; col 1: [NaN, 5] → 1; col 2: [3, 4] → 1
        let r = nanargmax(&a, Some(0)).unwrap();
        assert_eq!(r.iter().copied().collect::<Vec<_>>(), vec![0, 1, 1]);
    }

    #[test]
    fn test_nanargmin_axis_lane_all_nan_errs() {
        use ferray_core::Ix2;
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![f64::NAN, 1.0, f64::NAN, 2.0])
            .unwrap();
        // Axis-1: row 0 = [NaN, 1] OK, row 1 = [NaN, 2] OK
        // Axis-0: col 0 = [NaN, NaN] → all-NaN
        assert!(nanargmin(&a, Some(0)).is_err());
        assert!(nanargmin(&a, Some(1)).is_ok());
    }
}
