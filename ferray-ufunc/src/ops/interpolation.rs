// ferray-ufunc: Interpolation
//
// interp — 1-D linear interpolation (like NumPy's np.interp)

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};
use num_traits::Float;

/// 1-D linear interpolation.
///
/// Given sample points `xp` and corresponding values `fp`, interpolate
/// at the query points `x`. Values outside the range of `xp` are clipped
/// to the boundary values of `fp`.
///
/// AC-12: `interp(2.5, [1,2,3], [3,2,0]) == 1.0`.
pub fn interp<T>(
    x: &Array<T, Ix1>,
    xp: &Array<T, Ix1>,
    fp: &Array<T, Ix1>,
) -> FerrayResult<Array<T, Ix1>>
where
    T: Element + Float,
{
    let xp_data: Vec<T> = xp.iter().copied().collect();
    let fp_data: Vec<T> = fp.iter().copied().collect();
    let n = xp_data.len();

    if n == 0 {
        return Err(FerrayError::invalid_value("interp: xp must be non-empty"));
    }
    if xp_data.len() != fp_data.len() {
        return Err(FerrayError::shape_mismatch(
            "interp: xp and fp must have the same length",
        ));
    }

    let result: Vec<T> = x
        .iter()
        .map(|&xi| interp_scalar(xi, &xp_data, &fp_data))
        .collect();

    Array::from_vec(Ix1::new([result.len()]), result)
}

/// Interpolate a single scalar value.
fn interp_scalar<T: Float>(xi: T, xp: &[T], fp: &[T]) -> T {
    let n = xp.len();
    if n == 1 {
        return fp[0];
    }

    // Clip to boundary
    if xi <= xp[0] {
        return fp[0];
    }
    if xi >= xp[n - 1] {
        return fp[n - 1];
    }

    // Binary search for the interval
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = usize::midpoint(lo, hi);
        if xp[mid] <= xi {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    // Linear interpolation between xp[lo] and xp[hi]
    let dx = xp[hi] - xp[lo];
    if dx == T::from(0.0).unwrap() {
        return fp[lo];
    }
    let t = (xi - xp[lo]) / dx;
    fp[lo] + t * (fp[hi] - fp[lo])
}

/// Convenience: interpolate a single scalar query point.
///
/// AC-12: `interp_one(2.5, [1,2,3], [3,2,0]) == 1.0`.
pub fn interp_one<T>(xi: T, xp: &[T], fp: &[T]) -> FerrayResult<T>
where
    T: Float,
{
    if xp.is_empty() {
        return Err(FerrayError::invalid_value("interp: xp must be non-empty"));
    }
    if xp.len() != fp.len() {
        return Err(FerrayError::shape_mismatch(
            "interp: xp and fp must have the same length",
        ));
    }
    Ok(interp_scalar(xi, xp, fp))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::arr1;

    #[test]
    fn test_interp_ac12() {
        // AC-12: interp(2.5, [1,2,3], [3,2,0]) == 1.0
        let x = arr1(vec![2.5]);
        let xp = arr1(vec![1.0, 2.0, 3.0]);
        let fp = arr1(vec![3.0, 2.0, 0.0]);
        let r = interp(&x, &xp, &fp).unwrap();
        assert!((r.as_slice().unwrap()[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_interp_boundary() {
        let x = arr1(vec![0.0, 4.0]);
        let xp = arr1(vec![1.0, 2.0, 3.0]);
        let fp = arr1(vec![10.0, 20.0, 30.0]);
        let r = interp(&x, &xp, &fp).unwrap();
        let s = r.as_slice().unwrap();
        // Below range: clipped to first value
        assert_eq!(s[0], 10.0);
        // Above range: clipped to last value
        assert_eq!(s[1], 30.0);
    }

    #[test]
    fn test_interp_exact_points() {
        let x = arr1(vec![1.0, 2.0, 3.0]);
        let xp = arr1(vec![1.0, 2.0, 3.0]);
        let fp = arr1(vec![10.0, 20.0, 30.0]);
        let r = interp(&x, &xp, &fp).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn test_interp_midpoints() {
        let x = arr1(vec![1.5, 2.5]);
        let xp = arr1(vec![1.0, 2.0, 3.0]);
        let fp = arr1(vec![0.0, 10.0, 20.0]);
        let r = interp(&x, &xp, &fp).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 5.0).abs() < 1e-12);
        assert!((s[1] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_interp_one_scalar() {
        let r = interp_one(2.5, &[1.0, 2.0, 3.0], &[3.0, 2.0, 0.0]).unwrap();
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_interp_single_point() {
        let x = arr1(vec![5.0]);
        let xp = arr1(vec![1.0]);
        let fp = arr1(vec![42.0]);
        let r = interp(&x, &xp, &fp).unwrap();
        assert_eq!(r.as_slice().unwrap()[0], 42.0);
    }

    #[test]
    fn test_interp_f32() {
        let x = Array::<f32, Ix1>::from_vec(Ix1::new([1]), vec![2.5f32]).unwrap();
        let xp = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![1.0f32, 2.0, 3.0]).unwrap();
        let fp = Array::<f32, Ix1>::from_vec(Ix1::new([3]), vec![3.0f32, 2.0, 0.0]).unwrap();
        let r = interp(&x, &xp, &fp).unwrap();
        assert!((r.as_slice().unwrap()[0] - 1.0).abs() < 1e-5);
    }
}
