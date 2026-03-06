// ferrum-stats: NaN-aware reductions — nansum, nanprod, nanmin, nanmax, nanmean, nanvar, nanstd (REQ-3, REQ-4)
// Also nancumsum, nancumprod (REQ-2b)

use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::{Array, Dimension, Element, IxDyn};
use num_traits::Float;

use super::{borrow_data, make_result, output_shape, reduce_axis_general, validate_axis};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Sum a lane, skipping NaN values. Returns zero for all-NaN.
fn lane_nansum<T: Float>(lane: &[T]) -> T {
    let non_nan: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    crate::parallel::pairwise_sum(&non_nan, T::zero())
}

/// Product of a lane, skipping NaN values. Returns one for all-NaN.
fn lane_nanprod<T: Float>(lane: &[T]) -> T {
    lane.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .fold(T::one(), |a, b| a * b)
}

/// Mean of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmean<T: Float>(lane: &[T]) -> T {
    let non_nan: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    if non_nan.is_empty() {
        T::nan()
    } else {
        crate::parallel::pairwise_sum(&non_nan, T::zero()) / T::from(non_nan.len()).unwrap()
    }
}

/// Variance of a lane, skipping NaN values.
fn lane_nanvar<T: Float>(lane: &[T], ddof: usize) -> T {
    let non_nan: Vec<T> = lane.iter().copied().filter(|x| !x.is_nan()).collect();
    let count = non_nan.len();
    if count <= ddof {
        return T::nan();
    }
    let mean = crate::parallel::pairwise_sum(&non_nan, T::zero()) / T::from(count).unwrap();
    let sq_diffs: Vec<T> = non_nan.iter().map(|&x| {
        let d = x - mean;
        d * d
    }).collect();
    crate::parallel::pairwise_sum(&sq_diffs, T::zero()) / T::from(count - ddof).unwrap()
}

/// Min of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmin<T: Float>(lane: &[T]) -> T {
    lane.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .reduce(|a, b| if a <= b { a } else { b })
        .unwrap_or_else(T::nan)
}

/// Max of a lane, skipping NaN values. Returns NaN for all-NaN.
fn lane_nanmax<T: Float>(lane: &[T]) -> T {
    lane.iter()
        .copied()
        .filter(|x| !x.is_nan())
        .reduce(|a, b| if a >= b { a } else { b })
        .unwrap_or_else(T::nan)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Sum of array elements, treating NaN as zero.
///
/// Equivalent to `numpy.nansum`.
pub fn nansum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, IxDyn>>
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
pub fn nanprod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, IxDyn>>
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
pub fn nanmean<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, IxDyn>>
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
) -> FerrumResult<Array<T, IxDyn>>
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
) -> FerrumResult<Array<T, IxDyn>>
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
pub fn nanmin<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrumError::invalid_value(
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
pub fn nanmax<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, IxDyn>>
where
    T: Element + Float,
    D: Dimension,
{
    if a.is_empty() {
        return Err(FerrumError::invalid_value(
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

/// Cumulative sum, treating NaN as zero.
///
/// Re-exported from `ferrum_ufunc::nancumsum`.
pub fn nancumsum<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    ferrum_ufunc::nancumsum(a, axis)
}

/// Cumulative product, treating NaN as one.
///
/// Re-exported from `ferrum_ufunc::nancumprod`.
pub fn nancumprod<T, D>(a: &Array<T, D>, axis: Option<usize>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    ferrum_ufunc::nancumprod(a, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::Ix1;

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
}
