//! NEON GEMM benchmark — confirms the aarch64 NEON DGEMM/SGEMM kernel
//! is at least 2× faster than a scalar reference loop at N∈{128, 256, 512}.
//! Run on aarch64 with: cargo run --release --example neon_gemm_bench

use ferray_linalg::gemm::{cpu_supports_neon, gemm_f32, gemm_f64};
use std::time::Instant;

fn naive_dgemm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..n {
                acc += a[i * n + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn naive_sgemm(n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..n {
                acc += a[i * n + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn time_dgemm(n: usize, iters: usize) -> (f64, f64) {
    let a: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.001).sin()).collect();
    let b: Vec<f64> = (0..n * n).map(|i| ((i as f64) * 0.0017).cos()).collect();

    let mut c_neon = vec![0.0_f64; n * n];
    let mut c_naive = vec![0.0_f64; n * n];

    // Warm-up.
    gemm_f64(n, n, n, 1.0, &a, &b, 0.0, &mut c_neon);
    naive_dgemm(n, &a, &b, &mut c_naive);

    let t0 = Instant::now();
    for _ in 0..iters {
        gemm_f64(n, n, n, 1.0, &a, &b, 0.0, &mut c_neon);
    }
    let neon_secs = t0.elapsed().as_secs_f64() / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters.max(1) {
        naive_dgemm(n, &a, &b, &mut c_naive);
    }
    let naive_secs = t1.elapsed().as_secs_f64() / (iters.max(1) as f64);

    (neon_secs, naive_secs)
}

fn time_sgemm(n: usize, iters: usize) -> (f64, f64) {
    let a: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.001).sin()).collect();
    let b: Vec<f32> = (0..n * n).map(|i| ((i as f32) * 0.0017).cos()).collect();

    let mut c_neon = vec![0.0_f32; n * n];
    let mut c_naive = vec![0.0_f32; n * n];

    gemm_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c_neon);
    naive_sgemm(n, &a, &b, &mut c_naive);

    let t0 = Instant::now();
    for _ in 0..iters {
        gemm_f32(n, n, n, 1.0, &a, &b, 0.0, &mut c_neon);
    }
    let neon_secs = t0.elapsed().as_secs_f64() / (iters as f64);

    let t1 = Instant::now();
    for _ in 0..iters.max(1) {
        naive_sgemm(n, &a, &b, &mut c_naive);
    }
    let naive_secs = t1.elapsed().as_secs_f64() / (iters.max(1) as f64);

    (neon_secs, naive_secs)
}

fn main() {
    println!("cpu_supports_neon: {}", cpu_supports_neon());
    println!("\n=== DGEMM (f64) NEON vs scalar ===");
    println!(
        "{:>5}  {:>14}  {:>14}  {:>10}  {:>14}",
        "N", "NEON s/iter", "naive s/iter", "speedup", "NEON GFLOP/s"
    );
    for &n in &[128_usize, 256, 512] {
        let iters = match n {
            128 => 50,
            256 => 20,
            _ => 5,
        };
        let (neon, naive) = time_dgemm(n, iters);
        let flops = 2.0 * (n as f64).powi(3); // 2*M*N*K
        let neon_gflops = flops / neon / 1e9;
        let speedup = naive / neon;
        println!("{n:>5}  {neon:>14.6}  {naive:>14.6}  {speedup:>10.2}x  {neon_gflops:>14.2}");
    }
    println!("\n=== SGEMM (f32) NEON vs scalar ===");
    println!(
        "{:>5}  {:>14}  {:>14}  {:>10}  {:>14}",
        "N", "NEON s/iter", "naive s/iter", "speedup", "NEON GFLOP/s"
    );
    for &n in &[128_usize, 256, 512] {
        let iters = match n {
            128 => 100,
            256 => 30,
            _ => 5,
        };
        let (neon, naive) = time_sgemm(n, iters);
        let flops = 2.0 * (n as f64).powi(3);
        let neon_gflops = flops / neon / 1e9;
        let speedup = naive / neon;
        println!("{n:>5}  {neon:>14.6}  {naive:>14.6}  {speedup:>10.2}x  {neon_gflops:>14.2}");
    }
}
