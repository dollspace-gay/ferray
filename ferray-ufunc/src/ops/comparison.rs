// ferray-ufunc: Comparison functions
//
// equal, not_equal, less, less_equal, greater, greater_equal,
// array_equal, array_equiv, allclose, isclose

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::helpers::binary_map_op;

/// Elementwise equality test.
pub fn equal<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialEq + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x == y)
}

/// Elementwise inequality test.
pub fn not_equal<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialEq + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x != y)
}

/// Elementwise less-than test.
pub fn less<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x < y)
}

/// Elementwise less-than-or-equal test.
pub fn less_equal<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x <= y)
}

/// Elementwise greater-than test.
pub fn greater<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x > y)
}

/// Elementwise greater-than-or-equal test.
pub fn greater_equal<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<bool, D>>
where
    T: Element + PartialOrd + Copy,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| x >= y)
}

/// Test whether two arrays have the same shape and elements.
pub fn array_equal<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> bool
where
    T: Element + PartialEq,
    D: Dimension,
{
    if a.shape() != b.shape() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| x == y)
}

/// Test whether two arrays are element-wise equal within a tolerance,
/// or broadcastable to the same shape and element-wise equal.
///
/// For arrays of the same shape, this is the same as `array_equal`.
pub fn array_equiv<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> bool
where
    T: Element + PartialEq,
    D: Dimension,
{
    // For same-dimension arrays, just check equality
    array_equal(a, b)
}

/// Test whether two arrays are element-wise close within tolerances.
///
/// |a - b| <= atol + rtol * |b|
pub fn allclose<T, D>(a: &Array<T, D>, b: &Array<T, D>, rtol: T, atol: T) -> FerrayResult<bool>
where
    T: Element + Float,
    D: Dimension,
{
    let close = isclose(a, b, rtol, atol, false)?;
    Ok(close.iter().all(|&x| x))
}

/// Elementwise close-within-tolerance test.
///
/// |a - b| <= atol + rtol * |b|
///
/// If `equal_nan` is true, NaN values in corresponding positions are considered close.
pub fn isclose<T, D>(
    a: &Array<T, D>,
    b: &Array<T, D>,
    rtol: T,
    atol: T,
    equal_nan: bool,
) -> FerrayResult<Array<bool, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_map_op(a, b, |x, y| {
        if equal_nan && x.is_nan() && y.is_nan() {
            return true;
        }
        if x.is_nan() || y.is_nan() {
            return false;
        }
        (x - y).abs() <= atol + rtol * y.abs()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::Ix1;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    fn arr1_i32(data: Vec<i32>) -> Array<i32, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_equal() {
        let a = arr1_i32(vec![1, 2, 3]);
        let b = arr1_i32(vec![1, 5, 3]);
        let r = equal(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, true]);
    }

    #[test]
    fn test_not_equal() {
        let a = arr1_i32(vec![1, 2, 3]);
        let b = arr1_i32(vec![1, 5, 3]);
        let r = not_equal(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_less() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![2.0, 3.0, 3.0]);
        let r = less(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, false]);
    }

    #[test]
    fn test_less_equal() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![2.0, 3.0, 3.0]);
        let r = less_equal(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, true]);
    }

    #[test]
    fn test_greater() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![2.0, 3.0, 3.0]);
        let r = greater(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, false]);
    }

    #[test]
    fn test_greater_equal() {
        let a = arr1(vec![1.0, 5.0, 3.0]);
        let b = arr1(vec![2.0, 3.0, 3.0]);
        let r = greater_equal(&a, &b).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[false, true, true]);
    }

    #[test]
    fn test_array_equal() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![1.0, 2.0, 3.0]);
        let c = arr1(vec![1.0, 2.0, 4.0]);
        assert!(array_equal(&a, &b));
        assert!(!array_equal(&a, &c));
    }

    #[test]
    fn test_array_equal_different_shapes() {
        let a = arr1(vec![1.0, 2.0]);
        let b = arr1(vec![1.0, 2.0, 3.0]);
        assert!(!array_equal(&a, &b));
    }

    #[test]
    fn test_allclose() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![1.0 + 1e-9, 2.0 + 1e-9, 3.0 + 1e-9]);
        assert!(allclose(&a, &b, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_allclose_not_close() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![1.0, 2.0, 4.0]);
        assert!(!allclose(&a, &b, 1e-5, 1e-8).unwrap());
    }

    #[test]
    fn test_isclose() {
        let a = arr1(vec![1.0, 2.0, 3.0]);
        let b = arr1(vec![1.0, 2.1, 3.0]);
        let r = isclose(&a, &b, 1e-5, 1e-8, false).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, false, true]);
    }

    #[test]
    fn test_isclose_equal_nan() {
        let a = arr1(vec![f64::NAN, 1.0]);
        let b = arr1(vec![f64::NAN, 1.0]);
        let r = isclose(&a, &b, 1e-5, 1e-8, true).unwrap();
        assert_eq!(r.as_slice().unwrap(), &[true, true]);
    }

    // -----------------------------------------------------------------------
    // Broadcasting tests for comparison ops (issue #379)
    // -----------------------------------------------------------------------

    #[test]
    fn test_equal_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<i32, Ix2>::from_vec(Ix2::new([2, 3]), vec![1, 2, 3, 4, 2, 6]).unwrap();
        let b = Array::<i32, Ix2>::from_vec(Ix2::new([1, 3]), vec![1, 2, 3]).unwrap();
        let r = equal(&a, &b).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, false, true, false]
        );
    }

    #[test]
    fn test_less_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 1]), vec![1.0, 5.0, 10.0]).unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![3.0, 5.0, 7.0]).unwrap();
        let r = less(&a, &b).unwrap();
        assert_eq!(r.shape(), &[3, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![
                true, true, true, // 1 < {3,5,7}
                false, false, true, // 5 < {3,5,7}
                false, false, false, // 10 < {3,5,7}
            ]
        );
    }

    #[test]
    fn test_isclose_broadcasts() {
        use ferray_core::dimension::Ix2;
        let a = Array::<f64, Ix2>::from_vec(
            Ix2::new([2, 3]),
            vec![1.0, 2.0, 3.0, 1.0001, 2.0001, 3.0001],
        )
        .unwrap();
        let b = Array::<f64, Ix2>::from_vec(Ix2::new([1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
        let r = isclose(&a, &b, 1e-3, 1e-8, false).unwrap();
        assert_eq!(r.shape(), &[2, 3]);
        assert_eq!(
            r.iter().copied().collect::<Vec<_>>(),
            vec![true, true, true, true, true, true]
        );
    }
}
