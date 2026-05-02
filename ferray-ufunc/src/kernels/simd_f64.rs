// ferray-ufunc: SIMD kernels for f64 (REQ-17, REQ-18)

use crate::dispatch::{dispatch_binary_f64, dispatch_unary_f64};

/// SIMD-dispatched f64 sin.
#[inline]
pub fn sin_f64(input: &[f64], output: &mut [f64]) {
    dispatch_unary_f64(input, output, f64::sin);
}

/// SIMD-dispatched f64 cos.
#[inline]
pub fn cos_f64(input: &[f64], output: &mut [f64]) {
    dispatch_unary_f64(input, output, f64::cos);
}

/// SIMD-dispatched f64 exp.
#[inline]
pub fn exp_f64(input: &[f64], output: &mut [f64]) {
    dispatch_unary_f64(input, output, f64::exp);
}

/// SIMD-dispatched f64 ln.
#[inline]
pub fn log_f64(input: &[f64], output: &mut [f64]) {
    dispatch_unary_f64(input, output, f64::ln);
}

/// SIMD-dispatched f64 sqrt.
#[inline]
pub fn sqrt_f64(input: &[f64], output: &mut [f64]) {
    dispatch_unary_f64(input, output, f64::sqrt);
}

/// SIMD-dispatched f64 add.
#[inline]
pub fn add_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
    dispatch_binary_f64(a, b, output, |x, y| x + y);
}

/// SIMD-dispatched f64 mul.
#[inline]
pub fn mul_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
    dispatch_binary_f64(a, b, output, |x, y| x * y);
}

/// SIMD-dispatched f64 sub.
#[inline]
pub fn sub_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
    dispatch_binary_f64(a, b, output, |x, y| x - y);
}

/// SIMD-dispatched f64 div.
#[inline]
pub fn div_f64(a: &[f64], b: &[f64], output: &mut [f64]) {
    dispatch_binary_f64(a, b, output, |x, y| x / y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_f64_works() {
        let input = [0.0f64, std::f64::consts::FRAC_PI_2];
        let mut output = [0.0f64; 2];
        sin_f64(&input, &mut output);
        assert!((output[0]).abs() < 1e-12);
        assert!((output[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn add_f64_works() {
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        let mut out = [0.0f64; 3];
        add_f64(&a, &b, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }
}
