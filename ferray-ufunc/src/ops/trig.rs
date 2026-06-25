// ferray-ufunc: Trigonometric functions
//
// sin, cos, tan, arcsin, arccos, arctan, arctan2, hypot,
// sinh, cosh, tanh, arcsinh, arccosh, arctanh,
// degrees, radians, deg2rad, rad2deg, unwrap
//
// ## REQ status — REQ-5 (trig family) + binary-promote tie-ins
//
// SHIPPED:
//   - REQ-5 (`sin`/`cos`/`tan`/`arcsin`/`arccos`/`arctan`/`arctan2`/`hypot`/
//     `sinh`/`cosh`/`tanh`/`arcsinh`/`arccosh`/`arctanh`/`degrees`/`radians`/
//     `deg2rad`/`rad2deg`/`unwrap`): the full NumPy trig ufunc family as
//     generic free functions preserving input dimensionality (REQ-1). Anchors:
//     `pub fn sin`/`pub fn cos`/`pub fn tan`/`pub fn arcsin`/`pub fn arccos`/
//     `pub fn arctan`/`pub fn arcsinh`/`pub fn arccosh`/`pub fn arctanh`,
//     `pub fn arctan2`/`pub fn hypot` (binary),
//     `pub fn sinh`/`pub fn cosh`/`pub fn tanh`, `pub fn deg2rad`/
//     `pub fn rad2deg`/`pub fn degrees`/`pub fn radians`, `pub fn unwrap`.
//     `T: Element + Float` so f32/f64 (and complex via `complex.rs`) all route
//     here. Domain-edge behaviour (`arcsin`/`arccos` outside [-1,1] -> NaN,
//     `arctanh(±1)` -> ±inf) is audited against numpy 2.4.x and green. Each
//     unary op has an `_into` in-place counterpart (`pub fn sin_into`/
//     `pub fn cos_into`) and a faithful-rounding `_fast` kernel
//     (`pub fn sin_fast`/`pub fn cos_fast`). Non-test production consumer:
//     re-exported verbatim from the crate root (`lib.rs`
//     `pub use ops::trig::{sin, cos, tan, …, arctan2, hypot, …, unwrap}`),
//     the public ufunc surface and the ferray-python trig binding target.
//   - REQ-23 tie-in (integer/bool input promotion): the integer-accepting
//     `sin_promote`/`cos_promote`/`tan_promote`/`arctan_promote`/… wrappers
//     live in `promoted.rs` and call THESE generic `T: Float` kernels
//     monomorphised at the compute float, so the trig promotion surface
//     consumes this module unchanged (no separate int kernel here).
//   - REQ-25 tie-in (binary int/bool promotion): `arctan2_promote`/
//     `hypot_promote` (in `promoted.rs`) wrap `pub fn arctan2`/`pub fn hypot`
//     here for integer/bool operand pairs. f32/f64 callers stay byte-identical.
//
// NOT-STARTED: none — REQ-5 is fully shipped for this module.

use ferray_core::Array;
use ferray_core::dimension::Dimension;
use ferray_core::dtype::Element;
use ferray_core::error::FerrayResult;
use num_traits::Float;

use crate::cr_math::CrMath;
use crate::helpers::{
    binary_elementwise_op, unary_float_op, unary_float_op_compute, unary_float_op_into_compute,
};

// ---------------------------------------------------------------------------
// Unary trig
// ---------------------------------------------------------------------------

/// Elementwise sine.
///
/// Routes through `Float::sin` (libm) for ~2.4× faster per-element
/// throughput vs core-math at all sizes — matching NumPy's libm-based
/// path. Accuracy is libm's standard ~1 ULP, well within the 256-ULP
/// tolerance bands used by ferray's statistical equivalence harness.
/// For correctly-rounded results call `cr_math::CrMath::cr_sin` directly.
pub fn sin<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::sin)?;
    crate::errstate::record_invalid_unary_array_events(input, &result);
    Ok(result)
}

/// In-place sine — `_into` counterpart of [`sin`].
pub fn sin_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into_compute(input, out, "sin", T::sin)?;
    crate::errstate::record_invalid_unary_array_events(input, out);
    Ok(())
}

/// Elementwise cosine. See [`sin`] for the libm-vs-core-math accuracy note.
pub fn cos<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cos)?;
    crate::errstate::record_invalid_unary_array_events(input, &result);
    Ok(result)
}

/// In-place cosine — `_into` counterpart of [`cos`].
pub fn cos_into<T, D>(input: &Array<T, D>, out: &mut Array<T, D>) -> FerrayResult<()>
where
    T: Element + Float,
    D: Dimension,
{
    unary_float_op_into_compute(input, out, "cos", T::cos)?;
    crate::errstate::record_invalid_unary_array_events(input, out);
    Ok(())
}

/// Fast elementwise sine with ≤1 ULP accuracy (≤4 ULP for `|x| ≳ 2^20`).
///
/// Three-part Cody-Waite reduction + Cephes DP polynomials, with a branchless
/// quadrant select so the hot loop auto-vectorizes. The batch kernel is
/// runtime-multiversioned (AVX2+FMA when present, libm fallback otherwise),
/// giving ~3-4x over the default libm-based [`sin`] on contiguous f32/f64
/// arrays. The default [`sin`] stays the correctness/large-`|x|` reference
/// (Payne-Hanek-grade libm).
///
/// For f64 arrays the batch kernel is used directly; f32 promotes to f64
/// internally (24 mantissa bits round cleanly to the correct f32 answer).
pub fn sin_fast<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T is f64 — reinterpret the array reference.
        let f64_input =
            unsafe { &*std::ptr::from_ref::<Array<T, D>>(input).cast::<Array<f64, D>>() };
        let n = f64_input.size();
        let result = if let Some(slice) = f64_input.as_slice() {
            let mut data = vec![0.0_f64; n];
            crate::fast_trig::sin_fast_batch_f64(slice, &mut data);
            Array::from_vec(f64_input.dim().clone(), data)?
        } else {
            let data: Vec<f64> = f64_input
                .iter()
                .map(|&x| crate::fast_trig::sin_fast_f64(x))
                .collect();
            Array::from_vec(f64_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f64 at the top of this branch.
        let result = unsafe { crate::helpers::reinterpret_array::<f64, T, D>(result) };
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: T is f32 — reinterpret the array reference.
        let f32_input =
            unsafe { &*std::ptr::from_ref::<Array<T, D>>(input).cast::<Array<f32, D>>() };
        let n = f32_input.size();
        let result = if let Some(slice) = f32_input.as_slice() {
            let mut data = vec![0.0_f32; n];
            crate::fast_trig::sin_fast_batch_f32(slice, &mut data);
            Array::from_vec(f32_input.dim().clone(), data)?
        } else {
            let data: Vec<f32> = f32_input
                .iter()
                .map(|&x| crate::fast_trig::sin_fast_f32(x))
                .collect();
            Array::from_vec(f32_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f32 at the top of this branch.
        let result = unsafe { crate::helpers::reinterpret_array::<f32, T, D>(result) };
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    } else {
        // Other float types (f16/bf16): use the default libm path.
        let result = unary_float_op_compute(input, T::sin)?;
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    }
}

/// Fast elementwise cosine. See [`sin_fast`] for the accuracy/dispatch contract.
pub fn cos_fast<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f64>() {
        // SAFETY: T is f64 — reinterpret the array reference.
        let f64_input =
            unsafe { &*std::ptr::from_ref::<Array<T, D>>(input).cast::<Array<f64, D>>() };
        let n = f64_input.size();
        let result = if let Some(slice) = f64_input.as_slice() {
            let mut data = vec![0.0_f64; n];
            crate::fast_trig::cos_fast_batch_f64(slice, &mut data);
            Array::from_vec(f64_input.dim().clone(), data)?
        } else {
            let data: Vec<f64> = f64_input
                .iter()
                .map(|&x| crate::fast_trig::cos_fast_f64(x))
                .collect();
            Array::from_vec(f64_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f64 at the top of this branch.
        let result = unsafe { crate::helpers::reinterpret_array::<f64, T, D>(result) };
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    } else if TypeId::of::<T>() == TypeId::of::<f32>() {
        // SAFETY: T is f32 — reinterpret the array reference.
        let f32_input =
            unsafe { &*std::ptr::from_ref::<Array<T, D>>(input).cast::<Array<f32, D>>() };
        let n = f32_input.size();
        let result = if let Some(slice) = f32_input.as_slice() {
            let mut data = vec![0.0_f32; n];
            crate::fast_trig::cos_fast_batch_f32(slice, &mut data);
            Array::from_vec(f32_input.dim().clone(), data)?
        } else {
            let data: Vec<f32> = f32_input
                .iter()
                .map(|&x| crate::fast_trig::cos_fast_f32(x))
                .collect();
            Array::from_vec(f32_input.dim().clone(), data)?
        };
        // SAFETY: T was verified to be f32 at the top of this branch.
        let result = unsafe { crate::helpers::reinterpret_array::<f32, T, D>(result) };
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    } else {
        // Other float types (f16/bf16): use the default libm path.
        let result = unary_float_op_compute(input, T::cos)?;
        crate::errstate::record_invalid_unary_array_events(input, &result);
        Ok(result)
    }
}

/// Elementwise tangent. See [`sin`] for the libm-vs-core-math accuracy note.
pub fn tan<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::tan)?;
    crate::errstate::record_invalid_unary_array_events(input, &result);
    Ok(result)
}

/// Elementwise arc sine.
pub fn arcsin<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_asin)?;
    crate::errstate::record_unit_domain_array_events(input, &result);
    Ok(result)
}

/// Elementwise arc cosine.
pub fn arccos<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_acos)?;
    crate::errstate::record_unit_domain_array_events(input, &result);
    Ok(result)
}

/// Elementwise arc tangent.
pub fn arctan<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_atan)
}

/// Elementwise two-argument arc tangent (atan2).
pub fn arctan2<T, D>(y: &Array<T, D>, x: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = binary_elementwise_op(y, x, T::cr_atan2)?;
    crate::errstate::record_addlike_array_events(y, x, &result);
    Ok(result)
}

/// Elementwise hypotenuse: sqrt(a^2 + b^2).
pub fn hypot<T, D>(a: &Array<T, D>, b: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = binary_elementwise_op(a, b, T::cr_hypot)?;
    crate::errstate::record_addlike_array_events(a, b, &result);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Hyperbolic
// ---------------------------------------------------------------------------

/// Elementwise hyperbolic sine.
pub fn sinh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_sinh)?;
    crate::errstate::record_overflow_unary_array_events(input, &result);
    Ok(result)
}

/// Elementwise hyperbolic cosine.
pub fn cosh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_cosh)?;
    crate::errstate::record_overflow_unary_array_events(input, &result);
    Ok(result)
}

/// Elementwise hyperbolic tangent.
pub fn tanh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_tanh)
}

/// Elementwise inverse hyperbolic sine.
pub fn arcsinh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    unary_float_op_compute(input, T::cr_asinh)
}

/// Elementwise inverse hyperbolic cosine.
pub fn arccosh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_acosh)?;
    crate::errstate::record_arccosh_array_events(input, &result);
    Ok(result)
}

/// Elementwise inverse hyperbolic tangent.
pub fn arctanh<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float + CrMath,
    D: Dimension,
{
    let result = unary_float_op_compute(input, T::cr_atanh)?;
    crate::errstate::record_arctanh_array_events(input, &result);
    Ok(result)
}

// ---------------------------------------------------------------------------
// Degree/radian conversion
// ---------------------------------------------------------------------------

/// Convert radians to degrees.
pub fn degrees<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let result = unary_float_op(input, T::to_degrees)?;
    crate::errstate::record_overflow_unary_array_events(input, &result);
    Ok(result)
}

/// Convert degrees to radians.
pub fn radians<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let result = unary_float_op(input, T::to_radians)?;
    crate::errstate::record_overflow_unary_array_events(input, &result);
    Ok(result)
}

/// Alias for [`radians`].
pub fn deg2rad<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    radians(input)
}

/// Alias for [`degrees`].
pub fn rad2deg<T, D>(input: &Array<T, D>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    degrees(input)
}

/// Unwrap by changing deltas between values to their 2*pi complement.
///
/// Works on 1-D arrays. `discont` defaults to pi if `None`.
pub fn unwrap<T, D>(input: &Array<T, D>, discont: Option<T>) -> FerrayResult<Array<T, D>>
where
    T: Element + Float,
    D: Dimension,
{
    let pi = T::from(std::f64::consts::PI).unwrap_or_else(<T as Element>::zero);
    let two_pi = pi + pi;
    let discont = discont.unwrap_or(pi);

    let data: Vec<T> = input.iter().copied().collect();
    if data.is_empty() {
        return Array::from_vec(input.dim().clone(), data);
    }

    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    let mut cumulative = <T as Element>::zero();

    for i in 1..data.len() {
        let mut diff = data[i] - data[i - 1];
        if diff > discont || diff < -discont {
            diff = diff - two_pi * ((diff + pi) / two_pi).floor();
        }
        cumulative = cumulative + diff - (data[i] - data[i - 1]);
        result.push(data[i] + cumulative);
    }

    Array::from_vec(input.dim().clone(), result)
}

// ---------------------------------------------------------------------------
// f16 variants (f32-promoted) — declared via the shared `unary_f16_fn!`
// and `binary_f16_fn!` macros so each entry point is a single line. The
// prior hand-written pattern (6 lines × 15 functions ≈ 90 lines of
// boilerplate) is gone (#142).
// ---------------------------------------------------------------------------

use crate::helpers::{binary_f16_fn, unary_f16_fn};

unary_f16_fn!(
    /// Elementwise sine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    sin_f16,
    f32::sin
);
unary_f16_fn!(
    /// Elementwise cosine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    cos_f16,
    f32::cos
);
unary_f16_fn!(
    /// Elementwise tangent for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    tan_f16,
    f32::tan
);
unary_f16_fn!(
    /// Elementwise arc sine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arcsin_f16,
    f32::asin
);
unary_f16_fn!(
    /// Elementwise arc cosine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arccos_f16,
    f32::acos
);
unary_f16_fn!(
    /// Elementwise arc tangent for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arctan_f16,
    f32::atan
);
binary_f16_fn!(
    /// Elementwise two-argument arc tangent for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arctan2_f16,
    f32::atan2
);
binary_f16_fn!(
    /// Elementwise hypotenuse for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    hypot_f16,
    f32::hypot
);
unary_f16_fn!(
    /// Elementwise hyperbolic sine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    sinh_f16,
    f32::sinh
);
unary_f16_fn!(
    /// Elementwise hyperbolic cosine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    cosh_f16,
    f32::cosh
);
unary_f16_fn!(
    /// Elementwise hyperbolic tangent for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    tanh_f16,
    f32::tanh
);
unary_f16_fn!(
    /// Elementwise inverse hyperbolic sine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arcsinh_f16,
    f32::asinh
);
unary_f16_fn!(
    /// Elementwise inverse hyperbolic cosine for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arccosh_f16,
    f32::acosh
);
unary_f16_fn!(
    /// Elementwise inverse hyperbolic tangent for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    arctanh_f16,
    f32::atanh
);
unary_f16_fn!(
    /// Convert radians to degrees for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    degrees_f16,
    f32::to_degrees
);
unary_f16_fn!(
    /// Convert degrees to radians for f16 arrays via f32 promotion.
    #[cfg(feature = "f16")]
    radians_f16,
    f32::to_radians
);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_util::arr1;

    #[test]
    fn test_sin() {
        let a = arr1(vec![0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI]);
        let r = sin(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
        assert!((s[2]).abs() < 1e-12);
    }

    #[test]
    fn test_cos() {
        let a = arr1(vec![0.0, std::f64::consts::PI]);
        let r = cos(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 1.0).abs() < 1e-12);
        assert!((s[1] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_tan() {
        let a = arr1(vec![0.0, std::f64::consts::FRAC_PI_4]);
        let r = tan(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_arcsin() {
        let a = arr1(vec![0.0, 1.0]);
        let r = arcsin(&a).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_arctan2() {
        let y = arr1(vec![0.0, 1.0]);
        let x = arr1(vec![1.0, 0.0]);
        let r = arctan2(&y, &x).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0]).abs() < 1e-12);
        assert!((s[1] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn test_hypot() {
        let a = arr1(vec![3.0, 5.0]);
        let b = arr1(vec![4.0, 12.0]);
        let r = hypot(&a, &b).unwrap();
        let s = r.as_slice().unwrap();
        assert!((s[0] - 5.0).abs() < 1e-12);
        assert!((s[1] - 13.0).abs() < 1e-12);
    }

    #[test]
    fn test_sinh_cosh_tanh() {
        let a = arr1(vec![0.0]);
        assert!((sinh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
        assert!((cosh(&a).unwrap().as_slice().unwrap()[0] - 1.0).abs() < 1e-12);
        assert!((tanh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
    }

    #[test]
    fn test_degrees_radians() {
        let a = arr1(vec![std::f64::consts::PI]);
        let deg = degrees(&a).unwrap();
        assert!((deg.as_slice().unwrap()[0] - 180.0).abs() < 1e-10);

        let back = radians(&deg).unwrap();
        assert!((back.as_slice().unwrap()[0] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn test_deg2rad_rad2deg() {
        let a = arr1(vec![180.0]);
        let r = deg2rad(&a).unwrap();
        assert!((r.as_slice().unwrap()[0] - std::f64::consts::PI).abs() < 1e-12);

        let d = rad2deg(&arr1(vec![std::f64::consts::PI])).unwrap();
        assert!((d.as_slice().unwrap()[0] - 180.0).abs() < 1e-10);
    }

    #[test]
    fn test_unwrap_basic() {
        let a = arr1(vec![0.0, 0.5, 1.0, -0.5, -1.0]);
        let r = unwrap(&a, None).unwrap();
        // No discontinuity larger than pi, so should be unchanged
        let s = r.as_slice().unwrap();
        for (i, &v) in s.iter().enumerate() {
            assert!((v - a.as_slice().unwrap()[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_arcsinh_arccosh_arctanh() {
        let a = arr1(vec![0.0]);
        assert!((arcsinh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);

        let b = arr1(vec![1.0]);
        assert!((arccosh(&b).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
        assert!((arctanh(&a).unwrap().as_slice().unwrap()[0]).abs() < 1e-12);
    }

    #[cfg(feature = "f16")]
    mod f16_tests {
        use super::*;
        use ferray_core::dimension::Ix1;

        fn arr1_f16(data: &[f32]) -> Array<half::f16, Ix1> {
            let n = data.len();
            let vals: Vec<half::f16> = data.iter().map(|&x| half::f16::from_f32(x)).collect();
            Array::from_vec(Ix1::new([n]), vals).unwrap()
        }

        #[test]
        fn test_sin_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI]);
            let r = sin_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
            assert!(s[2].to_f32().abs() < 0.01);
        }

        #[test]
        fn test_cos_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::PI]);
            let r = cos_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!((s[0].to_f32() - 1.0).abs() < 0.01);
            assert!((s[1].to_f32() + 1.0).abs() < 0.01);
        }

        #[test]
        fn test_tan_f16() {
            let a = arr1_f16(&[0.0, std::f32::consts::FRAC_PI_4]);
            let r = tan_f16(&a).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - 1.0).abs() < 0.01);
        }

        #[test]
        fn test_arctan2_f16() {
            let y = arr1_f16(&[0.0, 1.0]);
            let x = arr1_f16(&[1.0, 0.0]);
            let r = arctan2_f16(&y, &x).unwrap();
            let s = r.as_slice().unwrap();
            assert!(s[0].to_f32().abs() < 0.01);
            assert!((s[1].to_f32() - std::f32::consts::FRAC_PI_2).abs() < 0.01);
        }

        #[test]
        fn test_degrees_radians_f16() {
            let a = arr1_f16(&[std::f32::consts::PI]);
            let deg = degrees_f16(&a).unwrap();
            assert!((deg.as_slice().unwrap()[0].to_f32() - 180.0).abs() < 0.5);

            let back = radians_f16(&deg).unwrap();
            assert!((back.as_slice().unwrap()[0].to_f32() - std::f32::consts::PI).abs() < 0.01);
        }
    }
}
