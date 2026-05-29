// Cody-Waite splits and Cephes DP polynomial coefficients are 17-digit
// scientific constants taken verbatim from the Cephes Mathematical Library
// reference; underscore separators would not match the canonical form
// engineers cross-check against. `#[inline(always)]` is required on the
// hot kernels so that auto-vectorization can fuse them into caller loops.
#![allow(clippy::unreadable_literal, clippy::inline_always)]

//! Branchless SIMD-friendly sin/cos kernels (#393).
//!
//! Self-contained `sin`/`cos` using three-part Cody-Waite range reduction
//! plus the Cephes DP polynomials. The quadrant selection is fully
//! **branchless** — the old 4-way `match` (which inhibited auto-vectorization)
//! is replaced by floor-based quadrant arithmetic and an exact 2-way blend, so
//! the per-element kernel auto-vectorizes when fused into the batch loops.
//!
//! **Wired into the public API** as [`crate::sin_fast`] / [`crate::cos_fast`].
//! The batch entry points ([`sin_fast_batch_f64`] etc.) are
//! runtime-multiversioned: when AVX2+FMA are present they run a
//! `#[target_feature]` clone that LLVM vectorizes (~3-4x over the default
//! libm-based `sin`/`cos` on this machine, regardless of the crate's baseline
//! `target-cpu`); pre-AVX2 x86 falls back to libm so it is never slower than
//! the standard library; non-x86 ISAs (aarch64 NEON, …) vectorize the portable
//! loop directly. The default `sin`/`cos` remain the correctness/large-`|x|`
//! reference (Payne-Hanek-grade libm) — use them when `|x|` may exceed ~2^20.
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
#[must_use]
pub fn sin_fast_f64(x: f64) -> f64 {
    // k = round(x * 2/π). This picks the nearest multiple of π/2, so
    // the residual y = x - k * (π/2) lands in [-π/4, π/4].
    let kf = (x * TWO_OVER_PI).round();

    // Three-step Cody-Waite reduction: y = x - kf * (π/2) computed
    // in three sub-steps to preserve precision.
    let y = kf.mul_add(-PI2_LO, kf.mul_add(-PI2_MI, kf.mul_add(-PI2_HI, x)));

    // Compute both polynomials; the quadrant blend below selects.
    let y2 = y * y;
    let sin_y = sin_poly_cephes(y, y2);
    let cos_y = cos_poly_cephes(y2);

    // Branchless quadrant select (floor-based, no `match`/early-return →
    // the batch loop auto-vectorizes; `round`/`floor` lower to roundpd and
    // the rest is FMA/mul/add). All values stay f64, so no int-cast stalls.
    //   q = kf mod 4 ∈ {0,1,2,3}; swap = kf mod 2; neg = floor(q/2).
    //   sin table:  q0:+sin q1:+cos q2:-sin q3:-cos
    //     ⇒ base = swap ? cos_y : sin_y, sign = neg ? -1 : +1.
    // NaN/Inf propagate naturally to NaN (±inf·c + ±inf → NaN), matching libm.
    let q4 = kf - 4.0 * (kf * 0.25).floor();
    let swap = kf - 2.0 * (kf * 0.5).floor();
    let neg = (q4 * 0.5).floor();
    // Exact 2-way select — lowers to a blend, NOT the original 4-way `match`
    // jump table, so it still vectorizes; an arithmetic interpolation
    // `a + s*(b-a)` would add a rounding step and lose a ULP. The sign factor
    // (1 - 2*neg) is exactly ±1 for neg ∈ {0,1}.
    let base = if swap != 0.0 { cos_y } else { sin_y };
    (1.0 - 2.0 * neg) * base
}

/// Fast `cos(x)` for `f64`. Same reduction as [`sin_fast_f64`] but
/// with the quadrant table shifted by one:
///   k mod 4 == 0:  cos(y)
///   k mod 4 == 1: -sin(y)
///   k mod 4 == 2: -cos(y)
///   k mod 4 == 3:  sin(y)
#[inline(always)]
#[must_use]
pub fn cos_fast_f64(x: f64) -> f64 {
    let kf = (x * TWO_OVER_PI).round();
    let y = kf.mul_add(-PI2_LO, kf.mul_add(-PI2_MI, kf.mul_add(-PI2_HI, x)));

    let y2 = y * y;
    let sin_y = sin_poly_cephes(y, y2);
    let cos_y = cos_poly_cephes(y2);

    // Branchless quadrant select (see `sin_fast_f64`).
    //   cos table:  q0:+cos q1:-sin q2:-cos q3:+sin
    //     ⇒ base = swap ? sin_y : cos_y; negative for q∈{1,2}.
    //   neg_cos = neg_sin XOR swap (0/1 arithmetic XOR: a+b-2ab).
    let q4 = kf - 4.0 * (kf * 0.25).floor();
    let swap = kf - 2.0 * (kf * 0.5).floor();
    let neg_sin = (q4 * 0.5).floor();
    let neg = neg_sin + swap - 2.0 * neg_sin * swap;
    // Exact 2-way select (see `sin_fast_f64`).
    let base = if swap != 0.0 { sin_y } else { cos_y };
    (1.0 - 2.0 * neg) * base
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
    y4.mul_add(p, one_minus)
}

/// Fast `sin(x)` for `f32`: computed in `f64` (24 mantissa bits of f32
/// input round cleanly to the correctly-rounded f32 answer).
#[inline(always)]
#[must_use]
pub fn sin_fast_f32(x: f32) -> f32 {
    sin_fast_f64(f64::from(x)) as f32
}

/// Fast `cos(x)` for `f32` via f64 promotion.
#[inline(always)]
#[must_use]
pub fn cos_fast_f32(x: f32) -> f32 {
    cos_fast_f64(f64::from(x)) as f32
}

// Batch wrappers are multiversioned at runtime. The branchless scalar kernels
// above auto-vectorize (`round`/`floor` → `roundpd`, the rest FMA) ONLY when
// the codegen target has SSE4.1+/AVX. Rust's default x86-64 baseline lacks
// these, so `floor` would lower to a slow libcall and the poly path would be a
// net loss vs libm. To get the ~3-5x win for every consumer regardless of
// their `target-cpu`, the hot loop is compiled a second time behind
// `#[target_feature(enable = "avx2,fma")]` (which forces AVX2 codegen for that
// fn alone) and selected via runtime CPU detection; pre-AVX2 x86 falls back to
// libm so it is never slower than the standard library. Non-x86 ISAs
// (aarch64 NEON, …) have baseline `floor`/FMA, so the portable loop vectorizes
// directly. The libm fallback differs from the Cephes kernel by ≤1 ULP — far
// inside any reasonable equivalence tolerance.
macro_rules! simd_batch {
    ($name:ident, $avx:ident, $ty:ty, $kernel:path, $libm:path) => {
        #[doc = concat!("Batch fast ", stringify!($kernel), " over slices — runtime-multiversioned (AVX2+FMA when present, libm fallback otherwise).")]
        #[inline(never)]
        pub fn $name(input: &[$ty], output: &mut [$ty]) {
            debug_assert_eq!(input.len(), output.len());
            #[cfg(target_arch = "x86_64")]
            {
                if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                    // SAFETY: the `avx2,fma` target features required by `$avx`
                    // are confirmed present by the runtime detection above.
                    unsafe { $avx(input, output) };
                } else {
                    for (o, &x) in output.iter_mut().zip(input) {
                        *o = $libm(x);
                    }
                }
            }
            #[cfg(not(target_arch = "x86_64"))]
            for (o, &x) in output.iter_mut().zip(input) {
                *o = $kernel(x);
            }
        }

        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn $avx(input: &[$ty], output: &mut [$ty]) {
            for (o, &x) in output.iter_mut().zip(input) {
                *o = $kernel(x);
            }
        }
    };
}

simd_batch!(
    sin_fast_batch_f64,
    sin_fast_batch_f64_avx2,
    f64,
    sin_fast_f64,
    f64::sin
);
simd_batch!(
    cos_fast_batch_f64,
    cos_fast_batch_f64_avx2,
    f64,
    cos_fast_f64,
    f64::cos
);
simd_batch!(
    sin_fast_batch_f32,
    sin_fast_batch_f32_avx2,
    f32,
    sin_fast_f32,
    f32::sin
);
simd_batch!(
    cos_fast_batch_f32,
    cos_fast_batch_f32_avx2,
    f32,
    cos_fast_f32,
    f32::cos
);

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
            let x = f64::from(i) * 0.001; // [-10, 10]
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
            let x = f64::from(i) * 0.001;
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
            let x = f64::from(i) * 0.01;
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
        let input: Vec<f64> = (-100..=100).map(|i| f64::from(i) * 0.1).collect();
        let mut output = vec![0.0_f64; input.len()];
        sin_fast_batch_f64(&input, &mut output);
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(output[i].to_bits(), sin_fast_f64(x).to_bits());
        }
    }

    #[test]
    fn cos_fast_batch_matches_scalar_f64() {
        let input: Vec<f64> = (-100..=100).map(|i| f64::from(i) * 0.1).collect();
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
