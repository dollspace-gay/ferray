// ferray-ufunc: SIMD kernels for f32 (REQ-17, REQ-18)
//
// Uses pulp for runtime CPU dispatch. For transcendental functions the
// current implementation delegates to scalar libm; the dispatch path is
// wired so that future polynomial-approximation SIMD kernels plug in here.

use crate::dispatch::{dispatch_binary_f32, dispatch_unary_f32};

/// SIMD-dispatched f32 sin.
#[inline]
pub fn sin_f32(input: &[f32], output: &mut [f32]) {
    dispatch_unary_f32(input, output, f32::sin);
}

/// SIMD-dispatched f32 cos.
#[inline]
pub fn cos_f32(input: &[f32], output: &mut [f32]) {
    dispatch_unary_f32(input, output, f32::cos);
}

/// SIMD-dispatched f32 exp.
#[inline]
pub fn exp_f32(input: &[f32], output: &mut [f32]) {
    dispatch_unary_f32(input, output, f32::exp);
}

/// SIMD-dispatched f32 ln.
#[inline]
pub fn log_f32(input: &[f32], output: &mut [f32]) {
    dispatch_unary_f32(input, output, f32::ln);
}

/// SIMD-dispatched f32 sqrt.
#[inline]
pub fn sqrt_f32(input: &[f32], output: &mut [f32]) {
    dispatch_unary_f32(input, output, f32::sqrt);
}

/// SIMD-dispatched f32 add.
#[inline]
pub fn add_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    dispatch_binary_f32(a, b, output, |x, y| x + y);
}

/// SIMD-dispatched f32 mul.
#[inline]
pub fn mul_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    dispatch_binary_f32(a, b, output, |x, y| x * y);
}

/// SIMD-dispatched f32 sub.
#[inline]
pub fn sub_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    dispatch_binary_f32(a, b, output, |x, y| x - y);
}

/// SIMD-dispatched f32 div.
#[inline]
pub fn div_f32(a: &[f32], b: &[f32], output: &mut [f32]) {
    dispatch_binary_f32(a, b, output, |x, y| x / y);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sin_f32_works() {
        let input = [0.0f32, std::f32::consts::FRAC_PI_2];
        let mut output = [0.0f32; 2];
        sin_f32(&input, &mut output);
        assert!((output[0]).abs() < 1e-6);
        assert!((output[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn add_f32_works() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let mut out = [0.0f32; 3];
        add_f32(&a, &b, &mut out);
        assert_eq!(out, [5.0, 7.0, 9.0]);
    }
}
