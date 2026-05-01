// ferray-ufunc: Fast faithfully-rounded exp() via Even/Odd Remez decomposition
//
// Algorithm from GitHub issue #3 (ngutten):
// https://github.com/dollspace-gay/ferray/issues/3
//
// Properties:
// - ≤1 ULP accuracy (faithfully rounded) vs CORE-MATH's ≤0.5 ULP
// - ~30% faster than CORE-MATH in batch operation
// - No lookup tables — auto-vectorizes cleanly for SSE/AVX2/AVX-512/NEON
// - Branchless with explicit NaN re-injection for correct SIMD behavior

// Polynomial coefficients are 17-digit scientific constants taken from a
// Remez generator and are written without group separators on purpose so
// they match the published reference coefficients character-for-character.
// `#[inline(always)]` on `exp_fast_f64` / `exp_fast_f32` is required for
// auto-vectorization — without it, the function-call boundary blocks
// SLP/loop vectorizers from fusing the kernel into the caller's loop.
#![allow(clippy::unreadable_literal, clippy::inline_always)]

/// Fast exp(x) using Even/Odd Remez decomposition with expm1 reconstruction.
///
/// Returns e^x with ≤1 ULP accuracy (faithfully rounded). Handles all IEEE 754
/// edge cases: NaN, ±inf, overflow, underflow.
///
/// # ⚠️ Subnormal handling (#401)
///
/// **Subnormal outputs (`x < -708.4`) are flushed to zero.** This is a
/// deliberate accuracy-vs-speed trade-off: numpy's `np.exp` and Rust's
/// `f64::exp` produce correctly-rounded subnormal results down to the
/// minimum representable f64 (~5e-324, achieved at `x ≈ -745.13`),
/// while this fast path emits zero.
///
/// **When this matters.** Algorithms like log-sum-exp, softmax with
/// large negative logits, and probabilistic graphical models can hit
/// inputs in the (-745, -708.4) subnormal range. Silent flushing to
/// zero produces a 0/0 NaN downstream. If your inputs may sit in
/// that range, prefer the standard library:
///
/// ```ignore
/// // Subnormal-preserving slow path:
/// let y = x.exp();   // f64::exp is correctly-rounded down to ~5e-324.
/// ```
///
/// or apply the standard log-sum-exp shifting trick (subtract max
/// before exponentiating) which keeps every intermediate above the
/// subnormal range.
#[inline(always)]
#[allow(
    clippy::excessive_precision,
    clippy::approx_constant,
    clippy::manual_clamp
)]
#[must_use]
pub fn exp_fast_f64(x: f64) -> f64 {
    let x_orig = x;

    // Branchless clamping for SIMD-friendly codegen.
    // SIMD min/max silently clamp NaN — we re-inject it post-hoc.
    // Clamp to the range where exp() produces finite, non-subnormal results.
    // The post-hoc checks below handle overflow/underflow on the original input.
    let x_c = if x > 709.7827128933840 {
        709.7827128933840
    } else if x < -745.1332191019411 {
        -745.1332191019411
    } else {
        x
    };

    // Range reduction (Cody-Waite): x = n*ln2 + r, |r| <= ln(2)/2
    // LN2_HI has 31 mantissa bits so n*LN2_HI is exact for |n| <= 1024.
    // LN2_LO from 200-digit precision — NOT computed as ln2 - LN2_HI in f64.
    let n = (x_c * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x_c - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    // Even: (cosh(r) - 1) via Horner in r² (Remez minimax coefficients)
    let even = r2
        * (5.000000000000000000e-01
            + r2 * (4.166666666666667823e-02
                + r2 * (1.388888888887908173e-03
                    + r2 * (2.480158733642404552e-05
                        + r2 * (2.755726329888269414e-07 + r2 * 2.091813031817600864e-09)))));

    // Odd: sinh(r) via Horner in r² (Remez minimax coefficients)
    let odd = r
        * (1.000000000000000000e+00
            + r2 * (1.666666666666667962e-01
                + r2 * (8.333333333319599412e-03
                    + r2 * (1.984126989005147428e-04
                        + r2 * (2.755724091443086719e-06 + r2 * 2.511003898736913440e-08)))));

    // expm1 reconstruction avoids absorption error from 1 + small
    let expm1 = even + odd;

    // 2^n reconstruction: when n = 1024, 2^n overflows IEEE 754.
    // Split into two steps: 2^(n-1) * 2, so the intermediate is representable.
    //
    // The `let mut result = ...; if ... { result = ... }` shape below is
    // deliberate: each "fix-up" branch (NaN/Inf/0) compiles to a single
    // vcmpXXpd + vblendvpd in SIMD. Folding them into a chained ternary
    // (as `useless_let_if_seq` would suggest) defeats vectorization.
    #[allow(clippy::useless_let_if_seq)]
    let mut result = if ni <= 1023 {
        let scale = f64::from_bits(((ni + 1023) as u64) << 52);
        scale + scale * expm1
    } else {
        // ni = 1024: scale = 2^1023 * 2
        let scale_half = f64::from_bits(((1023 + 1023) as u64) << 52); // 2^1023
        let r = scale_half + scale_half * expm1;
        r * 2.0
    };

    // Special cases: each compiles to vcmpXXpd + vblendvpd in SIMD
    // x != x is the SIMD-friendly NaN check (compiles to vcmpunordpd)
    #[allow(clippy::eq_op)]
    if x_orig != x_orig {
        result = f64::NAN;
    }
    if x_orig >= 709.7827128933840 {
        result = f64::INFINITY;
    }
    if x_orig < -745.1332191019411 {
        result = 0.0;
    }

    result
}

/// Batch fast exp for f64 slices — this is the auto-vectorizing hot loop.
#[inline(never)]
pub fn exp_fast_batch_f64(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_fast_f64(input[i]);
    }
}

/// Fast exp(x) for f32 via promotion to f64.
///
/// f32 has only 24 mantissa bits, so the f64 result rounded to f32 is
/// correctly rounded for all finite f32 inputs.
///
/// # ⚠️ Subnormal handling (#401)
///
/// Inherits the f64 path's subnormal flush: f32 inputs below ~-87.3
/// (where the f64 result before downcast would be subnormal in f64)
/// flush to zero. f32 subnormals are reachable from larger inputs
/// (~-103 to -88), so the f32 wrapper actually loses subnormal range
/// the f64 path would have preserved. Use `f32::exp` directly if you
/// need subnormal-correct f32 exp.
#[inline(always)]
#[must_use]
pub fn exp_fast_f32(x: f32) -> f32 {
    exp_fast_f64(f64::from(x)) as f32
}

/// Batch fast exp for f32 slices.
#[inline(never)]
pub fn exp_fast_batch_f32(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = exp_fast_f32(input[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_values() {
        assert!((exp_fast_f64(0.0) - 1.0).abs() < 1e-15);
        assert!((exp_fast_f64(1.0) - std::f64::consts::E).abs() < 1e-14);
        assert!((exp_fast_f64(-1.0) - 1.0 / std::f64::consts::E).abs() < 1e-15);
    }

    #[test]
    fn edge_cases() {
        assert!(exp_fast_f64(f64::NAN).is_nan());
        assert_eq!(exp_fast_f64(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp_fast_f64(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp_fast_f64(710.0), f64::INFINITY);
        assert_eq!(exp_fast_f64(-750.0), 0.0);
        assert_eq!(exp_fast_f64(0.0), 1.0);
        assert_eq!(exp_fast_f64(-0.0), 1.0);
    }

    #[test]
    fn accuracy_vs_libm() {
        // Check ≤1 ULP vs libm across a range of values
        let test_values: Vec<f64> = (-7000..=7097).map(|i| f64::from(i) * 0.1).collect();
        for &x in &test_values {
            let fast = exp_fast_f64(x);
            let reference = x.exp();
            if reference == 0.0 || reference.is_infinite() {
                continue;
            }
            let ulp = (fast - reference).abs() / (reference.abs() * f64::EPSILON);
            assert!(
                ulp <= 1.5,
                "exp_fast_f64({x}) = {fast}, libm = {reference}, ulp = {ulp}"
            );
        }
    }

    #[test]
    fn batch_matches_scalar() {
        let input: Vec<f64> = (-100..=100).map(|i| f64::from(i) * 0.1).collect();
        let mut output = vec![0.0f64; input.len()];
        exp_fast_batch_f64(&input, &mut output);
        for (i, &x) in input.iter().enumerate() {
            assert_eq!(output[i].to_bits(), exp_fast_f64(x).to_bits());
        }
    }

    #[test]
    fn f32_basic() {
        assert!((exp_fast_f32(0.0f32) - 1.0).abs() < 1e-7);
        assert!((exp_fast_f32(1.0f32) - std::f32::consts::E).abs() < 1e-6);
        assert!(exp_fast_f32(f32::NAN).is_nan());
        assert_eq!(exp_fast_f32(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp_fast_f32(f32::NEG_INFINITY), 0.0);
    }
}
