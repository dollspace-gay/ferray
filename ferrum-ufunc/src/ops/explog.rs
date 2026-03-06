// ferrum-ufunc: Exponential and logarithmic functions
//
// exp, exp2, expm1, log, log2, log10, log1p, logaddexp, logaddexp2

use ferrum_core::dimension::Dimension;
use ferrum_core::dtype::Element;
use ferrum_core::error::FerrumResult;
use ferrum_core::Array;
use num_traits::Float;

use crate::helpers::{unary_float_op, binary_float_op};

/// Elementwise exponential (e^x).
pub fn exp<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::exp)
}

/// Elementwise 2^x.
pub fn exp2<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::exp2)
}

/// Elementwise exp(x) - 1, accurate near zero.
pub fn expm1<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::exp_m1)
}

/// Elementwise natural logarithm.
pub fn log<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::ln)
}

/// Elementwise base-2 logarithm.
pub fn log2<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::log2)
}

/// Elementwise base-10 logarithm.
pub fn log10<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::log10)
}

/// Elementwise ln(1 + x), accurate near zero.
pub fn log1p<T, D>(input: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op(input, T::ln_1p)
}

/// log(exp(a) + exp(b)), computed in a numerically stable way.
pub fn logaddexp<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    binary_float_op(a, b, |x, y| {
        let max = if x > y { x } else { y };
        let min = if x > y { y } else { x };
        max + (min - max).exp().ln_1p()
    })
}

/// log2(2^a + 2^b), computed in a numerically stable way.
pub fn logaddexp2<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrumResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let ln2 = T::from(std::f64::consts::LN_2).unwrap_or_else(|| <T as Element>::one());
    binary_float_op(a, b, |x, y| {
        let max = if x > y { x } else { y };
        let min = if x > y { y } else { x };
        max + ((min - max) * ln2).exp().ln_1p() / ln2
    })
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
    fn test_exp() {
        let a = arr1(vec![0.0, 1.0]);
        let r = exp(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_exp2() {
        let a = arr1(vec![0.0, 3.0, 10.0]);
        let r = exp2(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - 8.0).abs() < 1e-12);
        assert!((s[2] - 1024.0).abs() < 1e-9);
    }

    #[test]
    fn test_expm1() {
        let a = arr1(vec![0.0, 1e-15]);
        let r = expm1(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        // expm1 should be accurate near zero
        assert!((s[1] - 1e-15).abs() < 1e-25);
    }

    #[test]
    fn test_log() {
        let a = arr1(vec![1.0, std::f64::consts::E]);
        let r = log(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_log2() {
        let a = arr1(vec![1.0, 8.0, 1024.0]);
        let r = log2(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 3.0).abs() < 1e-12);
        assert!((s[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_log10() {
        let a = arr1(vec![1.0, 100.0, 1000.0]);
        let r = log10(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 2.0).abs() < 1e-12);
        assert!((s[2] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_log1p() {
        let a = arr1(vec![0.0, 1e-15]);
        let r = log1p(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1e-15).abs() < 1e-25);
    }

    #[test]
    fn test_logaddexp() {
        let a = arr1(vec![0.0]);
        let b = arr1(vec![0.0]);
        let r = logaddexp(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // log(e^0 + e^0) = log(2) ~ 0.693
        assert!((s[0] - std::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn test_logaddexp2() {
        let a = arr1(vec![0.0]);
        let b = arr1(vec![0.0]);
        let r = logaddexp2(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        // log2(2^0 + 2^0) = log2(2) = 1
        assert!((s[0] - 1.0).abs() < 1e-12);
    }
}
