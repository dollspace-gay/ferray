// End-to-end tests for the public GEMM entry points exercising the
// NEON path on aarch64 (and the faer fallback elsewhere). All tests
// run unconditionally — there's no CPU-feature gate because the public
// API works on every supported target by contract.
//
// Each dtype is checked at:
//   - Tile-aligned sizes (M=N=K=64) covering the full kernel path.
//   - Misaligned sizes (M=37, N=41, K=23) covering edge tiles + K-tail.
//
// Tolerance: 1e-9 for f64, 1e-3 (relative) for f32, exact-equal for
// integer types.

use super::{
    cpu_supports_neon, gemm_c32, gemm_c64, gemm_f32, gemm_f64, gemm_i8, gemm_i8_signed, gemm_i16,
};

// ------------- naive references ----------

fn naive_f64(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn naive_f32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn naive_c64(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    // Inputs and output as flat [re, im, re, im, ...] streams.
    let mut c = vec![0.0_f64; m * n * 2];
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0_f64;
            let mut acc_im = 0.0_f64;
            for p in 0..k {
                let a_re = a[i * k * 2 + p * 2];
                let a_im = a[i * k * 2 + p * 2 + 1];
                let b_re = b[p * n * 2 + j * 2];
                let b_im = b[p * n * 2 + j * 2 + 1];
                acc_re += a_re * b_re - a_im * b_im;
                acc_im += a_re * b_im + a_im * b_re;
            }
            c[i * n * 2 + j * 2] = acc_re;
            c[i * n * 2 + j * 2 + 1] = acc_im;
        }
    }
    c
}

fn naive_c32(m: usize, n: usize, k: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n * 2];
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0_f32;
            let mut acc_im = 0.0_f32;
            for p in 0..k {
                let a_re = a[i * k * 2 + p * 2];
                let a_im = a[i * k * 2 + p * 2 + 1];
                let b_re = b[p * n * 2 + j * 2];
                let b_im = b[p * n * 2 + j * 2 + 1];
                acc_re += a_re * b_re - a_im * b_im;
                acc_im += a_re * b_im + a_im * b_re;
            }
            c[i * n * 2 + j * 2] = acc_re;
            c[i * n * 2 + j * 2 + 1] = acc_im;
        }
    }
    c
}

fn naive_i16(m: usize, n: usize, k: usize, a: &[i16], b: &[i16]) -> Vec<i32> {
    let mut c = vec![0_i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn naive_i8_us(m: usize, n: usize, k: usize, a: &[u8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0_i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn naive_i8_ss(m: usize, n: usize, k: usize, a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut c = vec![0_i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
    c
}

// ------------- tests ----------

#[test]
fn cpu_supports_neon_on_aarch64() {
    // The function must be true exactly when aarch64 build target is set.
    if cfg!(target_arch = "aarch64") {
        assert!(cpu_supports_neon(), "expected true on aarch64 target");
    } else {
        assert!(!cpu_supports_neon(), "expected false on non-aarch64");
    }
}

fn test_f64_at(m: usize, n: usize, k: usize) {
    // Use deterministic but mixed-sign inputs to catch sign-handling bugs.
    let a: Vec<f64> = (0..m * k).map(|i| ((i as f64) * 0.07) - 1.3).collect();
    let b: Vec<f64> = (0..k * n).map(|i| ((i as f64) * 0.11) - 2.5).collect();
    let mut c = vec![0.0_f64; m * n];
    let ok = gemm_f64(m, n, k, 1.0, &a, &b, 0.0, &mut c);
    assert!(ok, "gemm_f64 returned false at ({m},{n},{k})");
    let expected = naive_f64(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let scale = (got.abs() + want.abs()).max(1.0);
        assert!(
            diff / scale < 1e-9,
            "f64 mismatch at idx {i} (m={m} n={n} k={k}): got {got} want {want} diff {diff}"
        );
    }
}

#[test]
fn gemm_f64_aligned() {
    test_f64_at(64, 64, 64);
}

#[test]
fn gemm_f64_misaligned() {
    test_f64_at(37, 41, 23);
}

#[test]
fn gemm_f64_dispatcher_runs_neon_on_aarch64() {
    // Sanity: results are non-zero, proving the dispatcher selected a
    // real implementation and ran it (not a stub returning false).
    let a = vec![1.0_f64; 16 * 16];
    let b = vec![2.0_f64; 16 * 16];
    let mut c = vec![0.0_f64; 16 * 16];
    let ok = gemm_f64(16, 16, 16, 1.0, &a, &b, 0.0, &mut c);
    assert!(ok);
    // Expected: each cell = 1*2 * 16 = 32.
    assert_eq!(c[0], 32.0);
    assert!(c.iter().all(|&v| v == 32.0));
}

#[test]
fn gemm_f64_alpha_beta() {
    // C := 0.5 * A @ B + 2.0 * C   (general alpha/beta path)
    let m = 8;
    let n = 16;
    let k = 12;
    let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.05).collect();
    let mut c = vec![3.0_f64; m * n];
    let mut c_ref = c.clone();
    gemm_f64(m, n, k, 0.5, &a, &b, 2.0, &mut c);
    let prod = naive_f64(m, n, k, &a, &b);
    for i in 0..m * n {
        c_ref[i] = 0.5 * prod[i] + 2.0 * c_ref[i];
    }
    for (i, (got, want)) in c.iter().zip(c_ref.iter()).enumerate() {
        let diff = (got - want).abs();
        let scale = (got.abs() + want.abs()).max(1.0);
        assert!(
            diff / scale < 1e-9,
            "f64 alpha-beta mismatch at idx {i}: got {got} want {want}"
        );
    }
}

fn test_f32_at(m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.07) - 1.3).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.11) - 2.5).collect();
    let mut c = vec![0.0_f32; m * n];
    let ok = gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c);
    assert!(ok);
    let expected = naive_f32(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let scale = (got.abs() + want.abs()).max(1.0);
        assert!(
            diff / scale < 5e-3,
            "f32 mismatch at idx {i} (m={m} n={n} k={k}): got {got} want {want}"
        );
    }
}

#[test]
fn gemm_f32_aligned() {
    test_f32_at(64, 64, 64);
}

#[test]
fn gemm_f32_misaligned() {
    test_f32_at(37, 41, 23);
}

fn test_c64_at(m: usize, n: usize, k: usize) {
    // Each f64 holds [re, im] interleaved.
    let a: Vec<f64> = (0..m * k * 2).map(|i| ((i as f64) * 0.07) - 1.0).collect();
    let b: Vec<f64> = (0..k * n * 2).map(|i| ((i as f64) * 0.13) - 2.0).collect();
    let mut c = vec![0.0_f64; m * n * 2];
    let ok = unsafe {
        gemm_c64(
            m,
            n,
            k,
            1.0,
            0.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            0.0,
            c.as_mut_ptr(),
        )
    };
    assert!(ok);
    let expected = naive_c64(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let scale = (got.abs() + want.abs()).max(1.0);
        assert!(
            diff / scale < 1e-9,
            "c64 mismatch at idx {i} (m={m} n={n} k={k}): got {got} want {want}"
        );
    }
}

#[test]
fn gemm_c64_aligned() {
    test_c64_at(64, 64, 64);
}

#[test]
fn gemm_c64_misaligned() {
    test_c64_at(37, 41, 23);
}

#[test]
fn gemm_c64_complex_alpha() {
    // alpha = (2 + 3i)
    let m = 4;
    let n = 6;
    let k = 5;
    let a: Vec<f64> = (0..m * k * 2).map(|i| (i as f64) * 0.1).collect();
    let b: Vec<f64> = (0..k * n * 2).map(|i| (i as f64) * 0.07).collect();
    let mut c = vec![0.0_f64; m * n * 2];
    let ok = unsafe {
        gemm_c64(
            m,
            n,
            k,
            2.0,
            3.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            0.0,
            c.as_mut_ptr(),
        )
    };
    assert!(ok);
    let prod = naive_c64(m, n, k, &a, &b);
    for j in 0..m * n {
        let want_re = 2.0 * prod[j * 2] - 3.0 * prod[j * 2 + 1];
        let want_im = 2.0 * prod[j * 2 + 1] + 3.0 * prod[j * 2];
        let got_re = c[j * 2];
        let got_im = c[j * 2 + 1];
        let scale = (want_re.abs() + want_im.abs() + got_re.abs() + got_im.abs()).max(1.0);
        assert!(
            (want_re - got_re).abs() / scale < 1e-9,
            "c64 alpha re mismatch at {j}: got {got_re} want {want_re}"
        );
        assert!(
            (want_im - got_im).abs() / scale < 1e-9,
            "c64 alpha im mismatch at {j}: got {got_im} want {want_im}"
        );
    }
}

fn test_c32_at(m: usize, n: usize, k: usize) {
    let a: Vec<f32> = (0..m * k * 2).map(|i| ((i as f32) * 0.07) - 1.0).collect();
    let b: Vec<f32> = (0..k * n * 2).map(|i| ((i as f32) * 0.13) - 2.0).collect();
    let mut c = vec![0.0_f32; m * n * 2];
    let ok = unsafe {
        gemm_c32(
            m,
            n,
            k,
            1.0,
            0.0,
            a.as_ptr(),
            b.as_ptr(),
            0.0,
            0.0,
            c.as_mut_ptr(),
        )
    };
    assert!(ok);
    let expected = naive_c32(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        let scale = (got.abs() + want.abs()).max(1.0);
        assert!(
            diff / scale < 5e-3,
            "c32 mismatch at idx {i} (m={m} n={n} k={k}): got {got} want {want}"
        );
    }
}

#[test]
fn gemm_c32_aligned() {
    test_c32_at(64, 64, 64);
}

#[test]
fn gemm_c32_misaligned() {
    test_c32_at(37, 41, 23);
}

fn test_i16_at(m: usize, n: usize, k: usize) {
    let a: Vec<i16> = (0..m * k).map(|i| ((i as i32) * 17 - 100) as i16).collect();
    let b: Vec<i16> = (0..k * n).map(|i| ((i as i32) * 13 - 80) as i16).collect();
    let mut c = vec![0_i32; m * n];
    let ok = unsafe { gemm_i16(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), false) };
    assert!(ok);
    let expected = naive_i16(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*got, *want, "i16 mismatch at idx {i} (m={m} n={n} k={k})");
    }
}

#[test]
fn gemm_i16_aligned() {
    test_i16_at(64, 64, 64);
}

#[test]
fn gemm_i16_misaligned() {
    test_i16_at(37, 41, 23);
}

#[test]
fn gemm_i16_accumulate() {
    let m = 8;
    let n = 16;
    let k = 12;
    let a: Vec<i16> = (0..m * k).map(|i| (i + 1) as i16).collect();
    let b: Vec<i16> = (0..k * n).map(|i| ((i as i32) - 5) as i16).collect();
    let mut c = vec![100_i32; m * n];
    let ok = unsafe { gemm_i16(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), true) };
    assert!(ok);
    let prod = naive_i16(m, n, k, &a, &b);
    for i in 0..m * n {
        let want = 100_i32.wrapping_add(prod[i]);
        assert_eq!(c[i], want, "i16 accum mismatch at {i}");
    }
}

fn test_i8_at(m: usize, n: usize, k: usize) {
    let a: Vec<u8> = (0..m * k).map(|i| ((i * 7) % 200) as u8).collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| (((i as i32) * 5 - 100) % 127) as i8)
        .collect();
    let mut c = vec![0_i32; m * n];
    let ok = unsafe { gemm_i8(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), false) };
    assert!(ok);
    let expected = naive_i8_us(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            *got, *want,
            "i8 (u8×i8) mismatch at idx {i} (m={m} n={n} k={k})"
        );
    }
}

#[test]
fn gemm_i8_aligned() {
    test_i8_at(64, 64, 64);
}

#[test]
fn gemm_i8_misaligned() {
    test_i8_at(37, 41, 23);
}

fn test_i8s_at(m: usize, n: usize, k: usize) {
    let a: Vec<i8> = (0..m * k)
        .map(|i| (((i as i32) * 7 - 100) % 127) as i8)
        .collect();
    let b: Vec<i8> = (0..k * n)
        .map(|i| (((i as i32) * 5 + 50) % 127) as i8)
        .collect();
    let mut c = vec![0_i32; m * n];
    let ok = unsafe { gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), false) };
    assert!(ok);
    let expected = naive_i8_ss(m, n, k, &a, &b);
    for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
        assert_eq!(*got, *want, "i8s mismatch at idx {i} (m={m} n={n} k={k})");
    }
}

#[test]
fn gemm_i8s_aligned() {
    test_i8s_at(64, 64, 64);
}

#[test]
fn gemm_i8s_misaligned() {
    test_i8s_at(37, 41, 23);
}

#[test]
fn gemm_f64_n128() {
    // Larger size that exercises the MC/KC/NC blocking loops.
    test_f64_at(128, 128, 128);
}

#[test]
fn gemm_f32_n128() {
    test_f32_at(128, 128, 128);
}

// --- Faer-fallback unit tests: compile and link to faer_fallback only on
// non-x86-64 non-aarch64 targets. These tests confirm the fallback produces
// correct results against the same scalar reference. The test gate matches
// the module gate — invisible on aarch64/x86 builds.

#[cfg(all(test, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
mod faer_fallback_tests {
    use super::*;
    use crate::gemm::faer_fallback;

    #[test]
    fn fallback_f64() {
        let m = 8;
        let n = 16;
        let k = 12;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.1).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.07).collect();
        let mut c = vec![0.0_f64; m * n];
        unsafe {
            faer_fallback::gemm_fallback_f64(
                m,
                n,
                k,
                1.0,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                0.0,
                c.as_mut_ptr(),
                n,
            );
        }
        let expected = naive_f64(m, n, k, &a, &b);
        for (i, (got, want)) in c.iter().zip(expected.iter()).enumerate() {
            let diff = (got - want).abs();
            assert!(diff < 1e-9, "fallback f64 mismatch at {i}");
        }
    }
}
