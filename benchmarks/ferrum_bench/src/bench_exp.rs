// Benchmark: Even/Odd exp() vs CORE-MATH exp() vs libm exp()
//
// Tests the algorithm proposed in GitHub issue #3:
// https://github.com/dollspace-gay/ferrum/issues/3

use std::time::Instant;

// ---------------------------------------------------------------------------
// Even/Odd branchless exp (from the writeup)
// ---------------------------------------------------------------------------

#[inline(always)]
fn exp_evenodd(x: f64) -> f64 {
    // Handle overflow/underflow with early return
    // These comparisons return false for NaN, so NaN falls through
    if x > 709.7827128933840 {
        return f64::INFINITY;
    }
    if x < -708.3964185322641 {
        // Below this, result is < 0.5 * MIN_POSITIVE, so 0.0 (no subnormals)
        return 0.0;
    }

    // Range reduction (Cody-Waite)
    let n = (x * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    // Even: cosh(r) - 1 = r²/2! + r⁴/4! + ... + r¹²/12!
    let even = r2
        * (0.5
            + r2 * (4.16666666666666666e-02
                + r2 * (1.38888888888888889e-03
                    + r2 * (2.48015873015873016e-05
                        + r2 * (2.75573192239858907e-07 + r2 * 2.08767569878681145e-09)))));

    // Odd: sinh(r) = r + r³/3! + ... + r¹¹/11!
    let odd = r
        * (1.0
            + r2 * (1.66666666666666667e-01
                + r2 * (8.33333333333333333e-03
                    + r2 * (1.98412698412698413e-04
                        + r2 * (2.75573192239858907e-06 + r2 * 2.50521083854417188e-08)))));

    let exp_r = 1.0 + even + odd;

    // Reconstruct 2^n via IEEE 754 bit manipulation
    exp_r * f64::from_bits(((ni + 1023) as u64) << 52)
}

/// Even/Odd with Remez minimax coefficients (≤1 ULP target)
#[inline(always)]
fn exp_evenodd_remez(x: f64) -> f64 {
    if x > 709.7827128933840 {
        return f64::INFINITY;
    }
    if x < -708.3964185322641 {
        return 0.0;
    }

    let n = (x * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    // Remez-optimized even coefficients
    let even = r2
        * (5.00000000000000000000e-01
            + r2 * (4.16666666666666782315e-02
                + r2 * (1.38888888888790817261e-03
                    + r2 * (2.48015873364240455154e-05
                        + r2 * (2.75572632988826941427e-07
                            + r2 * 2.09181303181760086358e-09)))));

    // Remez-optimized odd coefficients
    let odd = r
        * (1.00000000000000000000e+00
            + r2 * (1.66666666666666796193e-01
                + r2 * (8.33333333331959941193e-03
                    + r2 * (1.98412698900514742825e-04
                        + r2 * (2.75572409144308671925e-06
                            + r2 * 2.51100389873691344004e-08)))));

    let exp_r = 1.0 + even + odd;
    exp_r * f64::from_bits(((ni + 1023) as u64) << 52)
}

/// Even/Odd Remez with subnormal result handling via two-step pow2i
#[inline(always)]
fn exp_evenodd_remez_subnorm(x: f64) -> f64 {
    if x > 709.7827128933840 {
        return f64::INFINITY;
    }
    if x < -745.1332191019411 {
        return 0.0;
    }

    let n = (x * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    let even = r2
        * (5.00000000000000000000e-01
            + r2 * (4.16666666666666782315e-02
                + r2 * (1.38888888888790817261e-03
                    + r2 * (2.48015873364240455154e-05
                        + r2 * (2.75572632988826941427e-07
                            + r2 * 2.09181303181760086358e-09)))));

    let odd = r
        * (1.00000000000000000000e+00
            + r2 * (1.66666666666666796193e-01
                + r2 * (8.33333333331959941193e-03
                    + r2 * (1.98412698900514742825e-04
                        + r2 * (2.75572409144308671925e-06
                            + r2 * 2.51100389873691344004e-08)))));

    let exp_r = 1.0 + even + odd;

    // Two-step pow2i for subnormal results:
    // When n < -1022, 2^n can't be represented directly.
    // Split: 2^n = 2^(n+1023) * 2^(-1023)
    if ni >= -1022 {
        exp_r * f64::from_bits(((ni + 1023) as u64) << 52)
    } else {
        // Scale in two steps to produce subnormals
        let scale1 = f64::from_bits(((ni + 1023 + 53) as u64) << 52); // 2^(n+53)
        let scale2 = f64::from_bits(((1023 - 53) as u64) << 52); // 2^(-53)
        exp_r * scale1 * scale2
    }
}

/// v2: Remez + expm1 reconstruction + NaN re-injection + branchless clamping
/// This is the recommended variant from the v2 writeup.
#[inline(always)]
fn exp_evenodd_v2(x: f64) -> f64 {
    let x_orig = x;

    // Branchless clamping (SIMD-friendly, but NaN gets clamped — fixed below)
    let x_c = if x > 709.782712893384 {
        709.782712893384
    } else if x < -745.1332191019411 {
        -745.1332191019411
    } else {
        x
    };

    // Range reduction (Cody-Waite)
    let n = (x_c * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x_c - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    // Even: cosh(r) - 1 (Remez minimax coefficients)
    let even = r2
        * (5.000000000000000000e-01
            + r2 * (4.166666666666667823e-02
                + r2 * (1.388888888887908173e-03
                    + r2 * (2.480158733642404552e-05
                        + r2 * (2.755726329888269414e-07
                            + r2 * 2.091813031817600864e-09)))));

    // Odd: sinh(r) (Remez minimax coefficients)
    let odd = r
        * (1.000000000000000000e+00
            + r2 * (1.666666666666667962e-01
                + r2 * (8.333333333319599412e-03
                    + r2 * (1.984126989005147428e-04
                        + r2 * (2.755724091443086719e-06
                            + r2 * 2.511003898736913440e-08)))));

    // expm1 reconstruction: avoids absorption error from 1 + small
    let expm1 = even + odd;
    let scale = f64::from_bits(((ni + 1023) as u64) << 52); // 2^n
    let mut result = scale + scale * expm1;

    // Special cases (each compiles to vcmpXXpd + vblendvpd in SIMD)
    // NaN re-injection: SIMD min/max silently clamp NaN, so we must fix it
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

/// v2 + subnormal: full production variant
#[inline(always)]
fn exp_evenodd_v2_subnorm(x: f64) -> f64 {
    let x_orig = x;

    let x_c = if x > 709.782712893384 {
        709.782712893384
    } else if x < -745.1332191019411 {
        -745.1332191019411
    } else {
        x
    };

    let n = (x_c * 1.44269504088896338700e+00).round();
    let ni = n as i64;
    let r = (x_c - n * 6.93147180369123816490e-01) - n * 1.90821492927058770002e-10;

    let r2 = r * r;

    let even = r2
        * (5.000000000000000000e-01
            + r2 * (4.166666666666667823e-02
                + r2 * (1.388888888887908173e-03
                    + r2 * (2.480158733642404552e-05
                        + r2 * (2.755726329888269414e-07
                            + r2 * 2.091813031817600864e-09)))));

    let odd = r
        * (1.000000000000000000e+00
            + r2 * (1.666666666666667962e-01
                + r2 * (8.333333333319599412e-03
                    + r2 * (1.984126989005147428e-04
                        + r2 * (2.755724091443086719e-06
                            + r2 * 2.511003898736913440e-08)))));

    let expm1 = even + odd;

    // Two-step pow2i for subnormal results
    let mut result = if ni >= -1022 {
        let scale = f64::from_bits(((ni + 1023) as u64) << 52);
        scale + scale * expm1
    } else {
        let scale1 = f64::from_bits(((ni + 1023 + 53) as u64) << 52);
        let scale2 = f64::from_bits(((1023 - 53) as u64) << 52);
        let s = scale1 * scale2;
        s + s * expm1
    };

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

/// Batch v2 (recommended)
#[inline(never)]
fn exp_v2_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_evenodd_v2(input[i]);
    }
}

/// Batch v2 + subnormal
#[inline(never)]
fn exp_v2_subnorm_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_evenodd_v2_subnorm(input[i]);
    }
}

/// Batch Even/Odd Taylor — this is what should auto-vectorize
#[inline(never)]
fn exp_evenodd_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_evenodd(input[i]);
    }
}

/// Batch Even/Odd Remez
#[inline(never)]
fn exp_remez_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_evenodd_remez(input[i]);
    }
}

/// Batch Even/Odd Remez with subnormal handling
#[inline(never)]
fn exp_remez_subnorm_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = exp_evenodd_remez_subnorm(input[i]);
    }
}

/// Batch CORE-MATH
#[inline(never)]
fn exp_coremath_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = core_math::exp(input[i]);
    }
}

/// Batch libm (std)
#[inline(never)]
fn exp_libm_batch(input: &[f64], output: &mut [f64]) {
    for i in 0..input.len() {
        output[i] = input[i].exp();
    }
}

// ---------------------------------------------------------------------------
// ULP measurement
// ---------------------------------------------------------------------------

fn ulp_error(got: f64, expected: f64) -> f64 {
    if got == expected {
        return 0.0;
    }
    if expected.is_infinite() || got.is_infinite() || expected.is_nan() || got.is_nan() {
        if (expected.is_infinite() && got.is_infinite() && expected.signum() == got.signum())
            || (expected.is_nan() && got.is_nan())
        {
            return 0.0;
        }
        return f64::INFINITY;
    }
    let diff = (got - expected).abs();
    let ulp = expected.abs() * f64::EPSILON;
    if ulp == 0.0 {
        diff / f64::MIN_POSITIVE
    } else {
        diff / ulp
    }
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

fn bench_fn(
    name: &str,
    input: &[f64],
    output: &mut [f64],
    f: fn(&[f64], &mut [f64]),
    iters: usize,
) -> f64 {
    // Warmup
    for _ in 0..100 {
        f(input, output);
    }

    let start = Instant::now();
    for _ in 0..iters {
        f(input, output);
        std::hint::black_box(&output);
    }
    let elapsed = start.elapsed();
    let ns_total = elapsed.as_nanos() as f64;
    let ns_per_elem = ns_total / (iters as f64 * input.len() as f64);
    let melem_s = 1000.0 / ns_per_elem;
    println!("  {name:<30} {ns_per_elem:6.2} ns/elem  {melem_s:7.1} Melem/s");
    ns_per_elem
}

pub fn run() {
    println!("\n=== exp() Benchmark: v1 vs v2 (expm1+NaN) vs CORE-MATH vs libm ===\n");

    let sizes = [256, 1024, 4096, 16384];
    let iters_for_size = |n: usize| -> usize {
        match n {
            ..=256 => 50000,
            ..=4096 => 10000,
            _ => 2000,
        }
    };

    // Focus on the distribution that exercises the full range
    let distributions: Vec<(&str, Box<dyn Fn(usize) -> Vec<f64>>)> = vec![
        (
            "uniform [-10, 10]",
            Box::new(|n| {
                (0..n)
                    .map(|i| -10.0 + 20.0 * (i as f64) / (n as f64 - 1.0).max(1.0))
                    .collect()
            }),
        ),
        (
            "uniform [-708, 709]",
            Box::new(|n| {
                (0..n)
                    .map(|i| -708.0 + 1417.0 * (i as f64) / (n as f64 - 1.0).max(1.0))
                    .collect()
            }),
        ),
    ];

    // Named batch functions — CORE-MATH first (truth reference)
    let variants: Vec<(&str, fn(&[f64], &mut [f64]))> = vec![
        ("CORE-MATH", exp_coremath_batch),
        ("Even/Odd Taylor (v1)", exp_evenodd_batch),
        ("Remez (v1)", exp_remez_batch),
        ("Remez+sub (v1)", exp_remez_subnorm_batch),
        ("v2 Remez+expm1+NaN", exp_v2_batch),
        ("v2 Remez+expm1+NaN+sub", exp_v2_subnorm_batch),
        ("libm", exp_libm_batch),
    ];

    for (dist_name, gen_input) in &distributions {
        println!("--- Distribution: {dist_name} ---\n");

        for &size in &sizes {
            let input = gen_input(size);
            let iters = iters_for_size(size);
            println!("  N = {size}");

            // Run CORE-MATH first as truth reference
            let mut truth = vec![0.0f64; size];
            exp_coremath_batch(&input, &mut truth);

            let mut cm_ns = 0.0f64;

            for &(name, func) in &variants {
                let mut output = vec![0.0f64; size];
                let ns = bench_fn(name, &input, &mut output, func, iters);

                if name == "CORE-MATH" {
                    cm_ns = ns;
                }

                // Accuracy vs CORE-MATH
                let mut max_ulp = 0.0f64;
                let mut sum_ulp = 0.0f64;
                for i in 0..size {
                    let ulp = ulp_error(output[i], truth[i]);
                    max_ulp = max_ulp.max(ulp);
                    sum_ulp += ulp;
                }
                let mean_ulp = sum_ulp / size as f64;
                let speedup = if ns > 0.0 { cm_ns / ns } else { 0.0 };

                if name != "CORE-MATH" && cm_ns > 0.0 {
                    println!(
                        "    -> vs CORE-MATH: {:.2}x faster, max {:.2} ULP, mean {:.4} ULP",
                        speedup, max_ulp, mean_ulp
                    );
                }
            }
            println!();
        }
    }

    // Edge case comparison: v1 vs v2 vs CORE-MATH
    println!("--- Edge Case Accuracy: v1 Remez vs v2 (expm1+NaN) ---\n");
    let edge_cases = [
        0.0, -0.0, 1.0, -1.0, 1e-15, -1e-15,
        709.0, -709.0, -708.0, -708.3, -708.39,
        709.7827128933840, -745.1332191019411,
        f64::NAN, f64::INFINITY, f64::NEG_INFINITY,
        f64::MIN_POSITIVE, -f64::MIN_POSITIVE,
    ];
    println!(
        "  {:<20} {:>22} {:>22} {:>8} {:>22} {:>8}",
        "input", "CORE-MATH", "v2+sub", "ULP", "v1 Remez+sub", "ULP"
    );
    for &x in &edge_cases {
        let cm = core_math::exp(x);
        let v2 = exp_evenodd_v2_subnorm(x);
        let v1 = exp_evenodd_remez_subnorm(x);
        let ulp_v2 = ulp_error(v2, cm);
        let ulp_v1 = ulp_error(v1, cm);
        let fmt_ulp = |u: f64| -> String {
            if u == 0.0 { "exact".into() }
            else if u.is_infinite() { "INF".into() }
            else { format!("{:.2}", u) }
        };
        println!(
            "  {:<20} {:>22.15e} {:>22.15e} {:>8} {:>22.15e} {:>8}",
            format!("{:.6e}", x), cm, v2, fmt_ulp(ulp_v2), v1, fmt_ulp(ulp_v1)
        );
    }
}
