// Standalone ZGEMM benchmark for ferray-linalg.
// Compares the hand-tuned Complex<f64> path (gemm_c64) against a naive
// reference for correctness and prints GFLOPs/sec at various sizes.
// Each complex FMA = 4 real FLOPs, so we count FLOPs as 8 * m * n * k.

use std::time::Instant;

fn naive_zgemm(m: usize, n: usize, k: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc_re = 0.0;
            let mut acc_im = 0.0;
            for p in 0..k {
                let ar = a[i * k * 2 + p * 2];
                let ai = a[i * k * 2 + p * 2 + 1];
                let br = b[p * n * 2 + j * 2];
                let bi = b[p * n * 2 + j * 2 + 1];
                acc_re += ar * br - ai * bi;
                acc_im += ar * bi + ai * br;
            }
            c[i * n * 2 + j * 2] = acc_re;
            c[i * n * 2 + j * 2 + 1] = acc_im;
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
    let sizes = [64usize, 128, 256, 384, 512, 768, 1024];

    println!("{:>5} {:>10} {:>10} {:>10}", "N", "us", "GFLOPs", "max_err");

    for &n in &sizes {
        let m = n;
        let k = n;
        // 2 doubles per complex
        let a: Vec<f64> = (0..m * k * 2).map(|i| ((i as f64) * 0.013).sin()).collect();
        let b: Vec<f64> = (0..k * n * 2).map(|i| ((i as f64) * 0.011).cos()).collect();
        let mut c = vec![0.0_f64; m * n * 2];

        // Verify correctness for small N only (naive is O(N^3)).
        let mut max_err = 0.0_f64;
        if n <= 128 {
            unsafe {
                ferray_linalg::gemm::gemm_c64(
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
                );
            }
            let mut c_ref = vec![0.0_f64; m * n * 2];
            naive_zgemm(m, n, k, &a, &b, &mut c_ref);
            for i in 0..m * n * 2 {
                let e = (c[i] - c_ref[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }

        for _ in 0..warmup {
            c.iter_mut().for_each(|x| *x = 0.0);
            unsafe {
                ferray_linalg::gemm::gemm_c64(
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
                );
            }
        }

        let mut times_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            c.iter_mut().for_each(|x| *x = 0.0);
            let t0 = Instant::now();
            unsafe {
                ferray_linalg::gemm::gemm_c64(
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
                );
            }
            times_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times_us[iters / 2];
        // Each complex multiply = 4 real FLOPs (2 muls + 2 muls + 1 add + 1 sub
        // for re; 2 muls + 1 add for im — that's 6 FLOPs but we count the
        // standard 8 m n k convention which counts 8 FLOPs per output
        // (= 4 real muls + 4 real adds for one complex FMA).
        let flops = 8.0 * (m as f64) * (n as f64) * (k as f64);
        let gflops = flops / median / 1e3;

        if n <= 128 {
            println!("{n:>5} {median:>9.0}us {gflops:>10.1} {max_err:>9.2e}");
        } else {
            println!("{n:>5} {median:>9.0}us {gflops:>10.1} {:>10}", "skip");
        }
    }
}
