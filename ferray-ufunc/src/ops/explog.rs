// ferray-ufunc: Exponential and logarithmic functions
//
// exp, exp2, expm1, log, log2, log10, log1p, logaddexp, logaddexp2

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::cr_math::CrMath;
use crate::helpers::{
    binary_elementwise_op, unary_float_op, unary_float_op_compute, unary_float_op_into_compute,
};

/// Elementwise exponential (e^x).
pub fn exp<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_exp)
}

/// In-place `e^x` — `_into` counterpart of [`exp`]. Parallelizes along
/// the compute-bound threshold for transcendentals (100k elements).
pub fn exp_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_into_compute(input, out, "exp", T::cr_exp)
}

/// Fast elementwise exponential (e^x) with ≤1 ULP accuracy.
///
/// Uses an Even/Odd Remez decomposition that is ~30% faster than `exp()` (CORE-MATH)
/// while achieving faithful rounding (≤1 ULP). The default `exp()` is correctly
/// rounded (≤0.5 ULP) via CORE-MATH.
///
/// This function auto-vectorizes for SSE/AVX2/AVX-512/NEON with no lookup tables.
/// Subnormal outputs (x < -708.4) are flushed to zero.
///
/// For f64 arrays, uses the optimized batch kernel directly.
/// For f32 arrays, promotes to f64 internally (f32 has only 24 mantissa bits,
/// so the result is correctly rounded for all finite f32 inputs).
pub fn exp_fast<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T is f64 — reinterpret the array reference
        let f64_input = unsafe { &*(input as *const Array<T, D> as *const Array<f64, D>) };
        let n = f64_input.size();
        let result = if let Some(slice) = f64_input.as_slice() {
            let mut data = Vec::with_capacity(n);
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(n);
            }
            crate::dispatch::dispatch_exp_fast_f64(slice, &mut data);
            Array::from_vec(f64_input.dim().clone(), data)?
        } else {
            let data: Vec<f64> = f64_input
                .iter()
                .map(|&x| crate::fast_exp::exp_fast_f64(x))
                .collect();
            Array::from_vec(f64_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f64 at the top of this branch.
        Ok(unsafe { crate::helpers::reinterpret_array::<f64, T, D>(result) })
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        let f32_input = unsafe { &*(input as *const Array<T, D> as *const Array<f32, D>) };
        let n = f32_input.size();
        let result = if let Some(slice) = f32_input.as_slice() {
            let mut data = Vec::with_capacity(n);
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(n);
            }
            crate::dispatch::dispatch_exp_fast_f32(slice, &mut data);
            Array::from_vec(f32_input.dim().clone(), data)?
        } else {
            let data: Vec<f32> = f32_input
                .iter()
                .map(|&x| crate::fast_exp::exp_fast_f32(x))
                .collect();
            Array::from_vec(f32_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f32 at the top of this branch.
        Ok(unsafe { crate::helpers::reinterpret_array::<f32, T, D>(result) })
    } else {
        // Fallback for other float types: use libm exp
        unary_float_op(input, |x| x.exp())
    }
}

/// Elementwise 2^x.
pub fn exp2<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_exp2)
}

/// Elementwise exp(x) - 1, accurate near zero.
pub fn expm1<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_exp_m1)
}

/// Elementwise natural logarithm.
pub fn log<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_ln)
}

/// In-place natural logarithm — `_into` counterpart of [`log`].
pub fn log_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_into_compute(input, out, "log", T::cr_ln)
}

/// Elementwise base-2 logarithm.
pub fn log2<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_log2)
}

/// Elementwise base-10 logarithm.
pub fn log10<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_log10)
}

/// Elementwise ln(1 + x), accurate near zero.
pub fn log1p<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_ln_1p)
}

/// log(exp(a) + exp(b)), computed in a numerically stable way.
pub fn logaddexp<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    binary_elementwise_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            return T::nan();
        }
        let max = if x > y { x } else { y };
        let min = if x > y { y } else { x };
        max + (min - max).cr_exp().cr_ln_1p()
    })
}

/// log2(2^a + 2^b), computed in a numerically stable way.
pub fn logaddexp2<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let ln2 = T::from(std::f64::consts::LN_2).unwrap_or_else(|| <T as Element>::one());
    binary_elementwise_op(a, b, |x, y| {
        if x.is_nan() || y.is_nan() {
            return T::nan();
        }
        let max = if x > y { x } else { y };
        let min = if x > y { y } else { x };
        max + ((min - max) * ln2).cr_exp().cr_ln_1p() / ln2
    })
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted) — generated via the shared unary_f16_fn!
// macro (#142).
// ---------------------------------------------------------------------------

use crate::helpers::unary_f16_fn;

unary_f16_fn!(
    /// Elementwise exponential for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    exp_f16,
    f32::exp
);
unary_f16_fn!(
    /// Elementwise 2^x for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    exp2_f16,
    f32::exp2
);
unary_f16_fn!(
    /// Elementwise exp(x)-1 for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    expm1_f16,
    f32::exp_m1
);
unary_f16_fn!(
    /// Elementwise natural logarithm for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    log_f16,
    f32::ln
);
unary_f16_fn!(
    /// Elementwise base-2 logarithm for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    log2_f16,
    f32::log2
);
unary_f16_fn!(
    /// Elementwise base-10 logarithm for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    log10_f16,
    f32::log10
);
unary_f16_fn!(
    /// Elementwise ln(1+x) for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    log1p_f16,
    f32::ln_1p
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::arr1;

    #[test]
    fn test_exp() {
        let a = arr1(vec![0.0, 1.0]);
        let r = exp(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn test_exp_fast() {
        let a = arr1(vec![0.0, 1.0, -1.0, 10.0, -10.0]);
        let r = exp_fast(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-15);
        assert!((s[1] - std::f64::consts::E).abs() < 1e-14);
        assert!((s[2] - 1.0 / std::f64::consts::E).abs() < 1e-15);
        // Check ≤1.5 ULP vs libm
        for (i, &x) in [0.0, 1.0, -1.0, 10.0, -10.0].iter().enumerate() {
            let reference = x.exp();
            let ulp = (s[i] - reference).abs() / (reference.abs() * f64::EPSILON);
            assert!(ulp <= 1.5, "exp_fast({x}) ulp = {ulp}");
        }
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

    #[cfg(feature = "f16")]
    mod f16_tests {
        use super::*;

        fn arr1_f16(data: &[f32]) -> Array<half::f16, Ix1> {
            let n = data.len();
            let vals: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            Array::from_vec(Ix1::new([n]), vals).unwrap()
        }

        #[test]
        fn test_exp_f16() {
            let a = arr1_f16(&[0.0, 1.0]);
            let r = exp_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() - std::f32::consts::E).abs() < 0.02);
        }

        #[test]
        fn test_log_f16() {
            let a = arr1_f16(&[1.0, std::f32::consts::E]);
            let r = log_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_log2_f16() {
            let a = arr1_f16(&[1.0, 8.0]);
            let r = log2_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 3.0).abs() < 0.01);
        }
    }
}
