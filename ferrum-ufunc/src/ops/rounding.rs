// ferrum-ufunc: Rounding functions
//
// round (banker's rounding!), floor, ceil, trunc, fix, rint, around

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::helpers::unary_float_op;

/// Banker's rounding: round half to even (AC-9).
///
/// `round(0.5) == 0`, `round(1.5) == 2`, `round(2.5) == 2`.
fn bankers_round<T: Float>(x: T) -> T {
    // Check if x is exactly at a .5 boundary
    let half = T::from(0.5).unwrap();
    let two = T::from(2.0).unwrap();

    // Get the fractional part: x - floor(x)
    let floored = x.floor();
    let frac = x - floored;

    // Check if fractional part is exactly 0.5
    if frac == half {
        // At exact .5 -- round to even
        let ceiled = x.ceil();
        // Check which of floor/ceil is even
        // A number is even if dividing by 2 and flooring gives back the same
        if (floored / two).floor() * two == floored {
            floored
        } else {
            ceiled
        }
    } else if frac == -half {
        // Negative half case: x is negative, frac = x - floor(x) could be 0.5 for negatives
        // Actually for negative numbers like -0.5: floor(-0.5) = -1, frac = -0.5 - (-1) = 0.5
        // So the above branch handles it. This branch is for safety.
        x.ceil()
    } else {
        // Not at a .5 boundary, standard rounding is fine
        x.round()
    }
}

/// Elementwise banker's rounding (round half to even).
///
/// This matches NumPy's `np.round` / `np.around` behavior.
/// AC-9: `round(0.5)==0`, `round(1.5)==2`.
pub fn round<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, bankers_round)
}

/// Alias for [`round`] -- matches NumPy's `around`.
pub fn around<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    round(input)
}

/// Alias for [`round`] -- matches NumPy's `rint`.
pub fn rint<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    round(input)
}

/// Elementwise floor (round toward negative infinity).
pub fn floor<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::floor)
}

/// Elementwise ceiling (round toward positive infinity).
pub fn ceil<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::ceil)
}

/// Elementwise truncation (round toward zero).
pub fn trunc<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::trunc)
}

/// Elementwise fix: round toward zero (same as trunc for real numbers).
pub fn fix<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    trunc(input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::dimension::Ix1;

    fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
        let n = data.len();
        Array::from_vec(Ix1::new([n]), data).unwrap()
    }

    #[test]
    fn test_bankers_round_half_to_even_ac9() {
        // AC-9: round(0.5)==0, round(1.5)==2
        let a = arr1(vec![0.5, 1.5, 2.5, 3.5, -0.5, -1.5]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);  // 0.5 -> 0 (even)
        assert_eq!(s[1], 2.0);  // 1.5 -> 2 (even)
        assert_eq!(s[2], 2.0);  // 2.5 -> 2 (even)
        assert_eq!(s[3], 4.0);  // 3.5 -> 4 (even)
        assert_eq!(s[4], 0.0);  // -0.5 -> 0 (even)
        assert_eq!(s[5], -2.0); // -1.5 -> -2 (even)
    }

    #[test]
    fn test_round_normal() {
        let a = arr1(vec![1.2, 2.7, -1.3, -2.8]);
        let r = round(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], 3.0);
        assert_eq!(s[2], -1.0);
        assert_eq!(s[3], -3.0);
    }

    #[test]
    fn test_floor() {
        let a = arr1(vec![1.7, -1.7, 0.0]);
        let r = floor(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -2.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_ceil() {
        let a = arr1(vec![1.2, -1.2, 0.0]);
        let r = ceil(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_trunc() {
        let a = arr1(vec![1.9, -1.9, 0.0]);
        let r = trunc(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 1.0);
        assert_eq!(s[1], -1.0);
        assert_eq!(s[2], 0.0);
    }

    #[test]
    fn test_fix() {
        let a = arr1(vec![2.9, -2.9]);
        let r = fix(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 2.0);
        assert_eq!(s[1], -2.0);
    }

    #[test]
    fn test_around_alias() {
        let a = arr1(vec![0.5, 1.5]);
        let r = around(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }

    #[test]
    fn test_rint_alias() {
        let a = arr1(vec![0.5, 1.5]);
        let r = rint(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 2.0);
    }
}
