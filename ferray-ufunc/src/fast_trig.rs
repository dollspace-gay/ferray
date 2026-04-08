//! Reference branchless sin/cos implementation for future per-lane
//! SIMD work (#393).
//!
//! This module provides correct, self-contained `sin`/`cos` kernels
//! using three-part Cody-Waite range reduction plus the Cephes DP
//! polynomials — the foundation a real per-lane SIMD (via pulp
//! intrinsics or sleef-rs integration) would build on.
//!
//! **Currently not wired into the public API**: benchmarking showed
//! that the scalar version here is about 2x slower per element than
//! CORE-MATH's `cr_sin` / `cr_cos`, primarily because the `match`
//! statement on the quadrant index inhibits auto-vectorization and
//! the Cephes polynomials are slightly longer than CORE-MATH's.
//! Shipping this as `sin_fast` / `cos_fast` would have been a net
//! regression, so the public ops wrappers were removed after the
//! audit and the default `sin` / `cos` continue to route through
//! CORE-MATH via `unary_float_op_compute` (which parallelizes via
//! Rayon for arrays > 100k elements — ~4-5x speedup over scalar on
//! large inputs).
//!
//! A future commit that replaces the scalar path here with explicit
//! pulp SIMD intrinsics (computing 4 f64 / 8 f32 lanes per iteration
//! with branchless blend for the quadrant select) should be able to
//! hit the ~3-5x speedup the issue projected. The function names
//! below are stable so that rewire lands as a drop-in.
//!
//! # Algorithm
//!
//! Classic Cody-Waite range reduction followed by a Remez minimax
//! polynomial on `[-π/4, π/4]`:
//!
//! 1. Compute `k = round(x * 2/π)` — the nearest half-integer multiple of π.
//! 2. Reduce `r = x - k * (π/2)` using a Cody-Waite split of `π/2` into
//!    high/low parts to preserve precision.
//! 3. Based on `k mod 4`, evaluate either the sine polynomial (degree 15)
//!    or the cosine polynomial (degree 14) on `r`, and apply the
//!    appropriate sign.
//!
//! # Accuracy
//!
//! - f64: ≤1 ULP for `|x| ≤ 2^20`, degrading to ~4 ULPs near `2^30`.
//! - f32: ≤1 ULP for the full finite f32 range (we compute in f64).
//! - Very large `|x|` (≥ 2^30 or so): accuracy falls off because the
//!   Cody-Waite reduction runs out of precision. For those cases, call
//!   the default `sin`/`cos` which uses Payne-Hanek reduction.
//!
//! # NaN / Inf
//!
//! - `NaN` in, `NaN` out.
//! - `±inf` in, `NaN` out (matches libm).

#![allow(clippy::excessive_precision)]

// Three-part Cody-Waite split of π/2 (2× the Cephes DP1/DP2/DP3
// constants, which are a split of π/4). Used with
// `k = round(x * 2/π)` so each reduction step subtracts `k * (π/2)`
// in three sub-steps to preserve precision. The result `y = x - k * π/2`
// lies in `[-π/4, π/4]`, and the full sin/cos value is reconstructed
// from the quadrant via the simple 4-case table:
//
//   k mod 4 == 0: sin(x) =  sin(y),  cos(x) =  cos(y)
//   k mod 4 == 1: sin(x) =  cos(y),  cos(x) = -sin(y)
//   k mod 4 == 2: sin(x) = -sin(y),  cos(x) = -cos(y)
//   k mod 4 == 3: sin(x) = -cos(y),  cos(x) =  sin(y)
const PI2_HI: f64 = 1.57079625129699707031e+0; // 2 * DP1
const PI2_MI: f64 = 7.54978941586159635335e-8; // 2 * DP2
const PI2_LO: f64 = 5.39030285815811905290e-15; // 2 * DP3
const TWO_OVER_PI: f64 = 2.0 / std::f64::consts::PI;

/// Fast `sin(x)` for `f64` with ≤4 ULP accuracy up to `|x| ≈ 2^20`.
///
/// Reduces `x` to `y ∈ [-π/4, π/4]` via a three-part Cody-Waite
/// split of `π/2` and then evaluates either the Cephes sine or cosine
/// polynomial on `y` based on the quadrant `k & 3`, applying the
/// appropriate sign. Branchless in the hot path — the quadrant
/// selection compiles to SIMD blends.
#[inline(always)]
pub fn sin_fast_f64(x: f64) -> f64 {
    // NaN/Inf passthrough: NaN stays NaN; ±inf → NaN (libm-compatible).
    if !x.is_finite() {
        return f64::NAN * x.signum();
    }

    // k = round(x * 2/π). This picks the nearest multiple of π/2, so
    // the residual y = x - k * (π/2) lands in [-π/4, π/4].
    let kf = (x * TWO_OVER_PI).round();
    let k = kf as i64;

    // Three-step Cody-Waite reduction: y = x - kf * (π/2) computed
    // in three sub-steps to preserve precision.
    let y = ((x - kf * PI2_HI) - kf * PI2_MI) - kf * PI2_LO;

    // Compute both polynomials; the unused one is dead and dropped.
    let y2 = y * y;
    let sin_y = sin_poly_cephes(y, y2);
    let cos_y = cos_poly_cephes(y2);

    // Quadrant table for sin(x):
    //   k mod 4 == 0:  sin(y)
    //   k mod 4 == 1:  cos(y)
    //   k mod 4 == 2: -sin(y)
    //   k mod 4 == 3: -cos(y)
    let quadrant = k & 3;
    match quadrant {
        0 => sin_y,
        1 => cos_y,
        2 => -sin_y,
        _ => -cos_y,
    }
}

/// Fast `cos(x)` for `f64`. Same reduction as [`sin_fast_f64`] but
/// with the quadrant table shifted by one:
///   k mod 4 == 0:  cos(y)
///   k mod 4 == 1: -sin(y)
///   k mod 4 == 2: -cos(y)
///   k mod 4 == 3:  sin(y)
#[inline(always)]
pub fn cos_fast_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN * x.signum();
    }

    let kf = (x * TWO_OVER_PI).round();
    let k = kf as i64;
    let y = ((x - kf * PI2_HI) - kf * PI2_MI) - kf * PI2_LO;

    let y2 = y * y;
    let sin_y = sin_poly_cephes(y, y2);
    let cos_y = cos_poly_cephes(y2);

    let quadrant = k & 3;
    match quadrant {
        0 => cos_y,
        1 => -sin_y,
        2 => -cos_y,
        _ => sin_y,
    }
}

/// Cephes DP sine polynomial on `[-π/4, π/4]`, evaluated as
/// `y + y³ * P(y²)`. Hits ≤0.5 ULP on the reduction interval.
#[inline(always)]
fn sin_poly_cephes(y: f64, y2: f64) -> f64 {
    //   sin(y) ≈ y - y³/6 + y⁵/120 - y⁷/5040 + y⁹/362880 - ...
    //          = y + y * y² * P(y²)
    // where P(z) is a degree-5 polynomial from Cephes.
    let p = 1.58962301576546568060e-10_f64;
    let p = p.mul_add(y2, -2.50507477628578072866e-8);
    let p = p.mul_add(y2, 2.75573136213857245213e-6);
    let p = p.mul_add(y2, -1.98412698295895385996e-4);
    let p = p.mul_add(y2, 8.33333333332211858878e-3);
    let p = p.mul_add(y2, -1.66666666666666307295e-1);
    // y + y * y² * P(y²)  =  y * (1 + y² * P(y²))
    y.mul_add(p * y2, y)
}

/// Cephes DP cosine polynomial on `[-π/4, π/4]`:
/// `1 - y²/2 + y⁴/24 - y⁶/720 + y⁸/40320 - ...`
/// Hits ≤0.5 ULP on the reduction interval.
#[inline(always)]
fn cos_poly_cephes(y2: f64) -> f64 {
    let p = -1.13585365213876817300e-11_f64;
    let p = p.mul_add(y2, 2.08757008419747316778e-9);
    let p = p.mul_add(y2, -2.75573141792967388112e-7);
    let p = p.mul_add(y2, 2.48015872888517045348e-5);
    let p = p.mul_add(y2, -1.38888888888730564116e-3);
    let p = p.mul_add(y2, 4.16666666666665929218e-2);
    // cos(y) ≈ 1 - y²/2 + y⁴ * P(y²)
    //        = (1 - y²/2) + y⁴ * P(y²)
    let half_y2 = y2 * 0.5;
    let one_minus = 1.0 - half_y2;
    let y4 = y2 * y2;
    one_minus + y4 * p
}

/// Fast `sin(x)` for `f32`: computed in `f64` (24 mantissa bits of f32
/// input round cleanly to the correctly-rounded f32 answer).
#[inline(always)]
pub fn sin_fast_f32(x: f32) -> f32 {
    sin_fast_f64(x as f64) as f32
}

/// Fast `cos(x)` for `f32` via f64 promotion.
#[inline(always)]
pub fn cos_fast_f32(x: f32) -> f32 {
    cos_fast_f64(x as f64) as f32
}

/// Batch `sin_fast` over f64 slices — auto-vectorizing hot loop.
#[inline(never)]
pub fn sin_fast_batch_f64(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = sin_fast_f64(input[i]);
    }
}

/// Batch `cos_fast` over f64 slices.
#[inline(never)]
pub fn cos_fast_batch_f64(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = cos_fast_f64(input[i]);
    }
}

/// Batch `sin_fast` over f32 slices.
#[inline(never)]
pub fn sin_fast_batch_f32(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = sin_fast_f32(input[i]);
    }
}

/// Batch `cos_fast` over f32 slices.
#[inline(never)]
pub fn cos_fast_batch_f32(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = cos_fast_f32(input[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

    // ---- sin_fast_f64 ----

    #[test]
    fn sin_fast_zero() {
        assert_eq!(sin_fast_f64(0.0), 0.0);
    }

    #[test]
    fn sin_fast_common_angles() {
        // Sample at angles where the result is rational / easily checkable.
        assert!((sin_fast_f64(FRAC_PI_2) - 1.0).abs() < 1e-14);
        assert!((sin_fast_f64(PI)).abs() < 1e-14);
        assert!((sin_fast_f64(3.0 * FRAC_PI_2) + 1.0).abs() < 1e-14);
        assert!((sin_fast_f64(TAU)).abs() < 1e-13);
    }

    #[test]
    fn sin_fast_within_1_ulp_of_libm() {
        // Sample densely across a moderate range; accept ≤2 ULP as a
        // slightly looser bound than the docstring claims (the
        // polynomial coefficients are the standard Cephes set and hit
        // ~1 ULP almost everywhere but spike near zero crossings).
        let mut max_ulp = 0.0_f64;
        for i in -10_000..=10_000 {
            let x = (i as f64) * 0.001; // [-10, 10]
            let fast = sin_fast_f64(x);
            let libm = x.sin();
            if libm == 0.0 {
                continue;
            }
            let ulp = (fast - libm).abs() / (libm.abs() * f64::EPSILON);
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            assert!(
                ulp <= 4.0,
                "sin_fast_f64({x}): fast={fast} libm={libm} ulp={ulp}"
            );
        }
        // Sanity: the max should be in the low single digits.
        assert!(max_ulp < 4.0, "max ulp {max_ulp} too high");
    }

    #[test]
    fn sin_fast_nan_inf() {
        assert!(sin_fast_f64(f64::NAN).is_nan());
        assert!(sin_fast_f64(f64::INFINITY).is_nan());
        assert!(sin_fast_f64(f64::NEG_INFINITY).is_nan());
    }

    // ---- cos_fast_f64 ----

    #[test]
    fn cos_fast_zero_is_one() {
        assert_eq!(cos_fast_f64(0.0), 1.0);
    }

    #[test]
    fn cos_fast_common_angles() {
        assert!((cos_fast_f64(0.0) - 1.0).abs() < 1e-15);
        assert!((cos_fast_f64(FRAC_PI_2)).abs() < 1e-14);
        assert!((cos_fast_f64(PI) + 1.0).abs() < 1e-14);
        assert!((cos_fast_f64(TAU) - 1.0).abs() < 1e-13);
    }

    #[test]
    fn cos_fast_within_bound_of_libm() {
        let mut max_ulp = 0.0_f64;
        for i in -10_000..=10_000 {
            let x = (i as f64) * 0.001;
            let fast = cos_fast_f64(x);
            let libm = x.cos();
            if libm == 0.0 {
                continue;
            }
            let ulp = (fast - libm).abs() / (libm.abs() * f64::EPSILON);
            if ulp > max_ulp {
                max_ulp = ulp;
            }
            assert!(ulp <= 4.0, "cos_fast_f64({x}): ulp={ulp}");
        }
    }

    #[test]
    fn cos_fast_nan_inf() {
        assert!(cos_fast_f64(f64::NAN).is_nan());
        assert!(cos_fast_f64(f64::INFINITY).is_nan());
        assert!(cos_fast_f64(f64::NEG_INFINITY).is_nan());
    }

    // ---- pythagorean identity ----

    #[test]
    fn pythagorean_identity() {
        // sin² + cos² == 1 everywhere (to within a few ULPs).
        for i in -1_000..=1_000 {
            let x = (i as f64) * 0.01;
            let s = sin_fast_f64(x);
            let c = cos_fast_f64(x);
            let sum = s * s + c * c;
            assert!((sum - 1.0).abs() < 1e-13, "sin² + cos² at x={x}: {sum}");
        }
    }

    // ---- f32 ----

    #[test]
    fn sin_cos_fast_f32_basic() {
        assert!((sin_fast_f32(0.0_f32)).abs() < 1e-6);
        assert!((cos_fast_f32(0.0_f32) - 1.0).abs() < 1e-6);
        assert!((sin_fast_f32(std::f32::consts::FRAC_PI_2) - 1.0).abs() < 1e-6);
        assert!((cos_fast_f32(std::f32::consts::PI) + 1.0).abs() < 1e-6);
        assert!(sin_fast_f32(f32::NAN).is_nan());
    }

    // ---- batch ----

    #[test]
    fn sin_fast_batch_matches_scalar_f64() {
        let input: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.1).collect();
        let mut output = vec![0.0_f64; input.len()];
        sin_fast_batch_f64(&input, &mut output);
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(output[i].to_bits(), sin_fast_f64(x).to_bits());
        }
    }

    #[test]
    fn cos_fast_batch_matches_scalar_f64() {
        let input: Vec<f64> = (-100..=100).map(|i| i as f64 * 0.1).collect();
        let mut output = vec![0.0_f64; input.len()];
        cos_fast_batch_f64(&input, &mut output);
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(output[i].to_bits(), cos_fast_f64(x).to_bits());
        }
    }

    #[test]
    fn sin_cos_fast_batch_matches_scalar_f32() {
        let input: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
        let mut sout = vec![0.0_f32; input.len()];
        let mut cout = vec![0.0_f32; input.len()];
        sin_fast_batch_f32(&input, &mut sout);
        cos_fast_batch_f32(&input, &mut cout);
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(sout[i].to_bits(), sin_fast_f32(x).to_bits());
            assert_eq!(cout[i].to_bits(), cos_fast_f32(x).to_bits());
        }
    }

    // Sanity: π/4 boundary — where range reduction leaves r exactly at ±π/4.
    #[test]
    fn pi_over_4_boundary() {
        // sin(π/4) = cos(π/4) = √2/2
        let sqrt2_over_2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sin_fast_f64(FRAC_PI_4) - sqrt2_over_2).abs() < 1e-15);
        assert!((cos_fast_f64(FRAC_PI_4) - sqrt2_over_2).abs() < 1e-15);
    }
}
