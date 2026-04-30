// Standalone SGEMM benchmark for ferray-linalg.
// Compares ferray's hand-tuned SGEMM (gemm_f32 path) against a naive
// reference for correctness, and prints GFLOPs/sec at various sizes.

use std::time::Instant;

fn naive_sgemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("AVX2+FMA required");
        std::process::exit(1);
    }

    let warmup = 5;
    let iters = 30;
    let sizes = [128usize, 256, 384, 512, 768, 1024, 1536, 2048];

    println!("{:>5} {:>10} {:>10} {:>8}", "N", "us", "GFLOPs", "max_err");

    for &n in &sizes {
        let m = n;
        let k = n;
        // Generate inputs.
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.011).cos()).collect();
        let mut c = vec![0.0_f32; m * n];

        // Verify correctness for small N only (naive is O(N^3)).
        let mut max_err = 0.0_f32;
        if n <= 256 {
            ferray_linalg::gemm::gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c);
            let mut c_ref = vec![0.0_f32; m * n];
            naive_sgemm(m, n, k, &a, &b, &mut c_ref);
            for i in 0..m * n {
                let e = (c[i] - c_ref[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }

        // Warmup.
        for _ in 0..warmup {
            c.iter_mut().for_each(|x| *x = 0.0);
            ferray_linalg::gemm::gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c);
        }

        // Time iters.
        let mut times_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            c.iter_mut().for_each(|x| *x = 0.0);
            let t0 = Instant::now();
            ferray_linalg::gemm::gemm_f32(m, n, k, 1.0, &a, &b, 0.0, &mut c);
            let dt = t0.elapsed();
            times_us.push(dt.as_secs_f64() * 1e6);
        }
        times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times_us[iters / 2];
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let gflops = flops / median / 1e3;

        if n <= 256 {
            println!("{n:>5} {median:>9.0}us {gflops:>10.1} {max_err:>8.2e}");
        } else {
            println!("{n:>5} {median:>9.0}us {gflops:>10.1} {:>8}", "skip");
        }
    }
}
