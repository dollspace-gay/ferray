// Standalone i8 GEMM benchmark for ferray-linalg.
// Compares the hand-tuned u8 × i8 → i32 path (gemm_i8) against a naive
// reference for correctness and prints GOPS/sec at various sizes.
// (1 op = 1 i8 multiply-accumulate, the standard i8 BLAS convention.)

use std::time::Instant;

fn naive_i8gemm(m: usize, n: usize, k: usize, a: &[u8], b: &[i8], c: &mut [i32]) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0_i32;
            for p in 0..k {
                acc = acc.wrapping_add((a[i * k + p] as i32) * (b[p * n + j] as i32));
            }
            c[i * n + j] = acc;
        }
    }
}

fn main() {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("AVX2 required");
        std::process::exit(1);
    }
    let has_vnni = is_x86_feature_detected!("avxvnni");
    println!(
        "AVX-VNNI: {}",
        if has_vnni { "available" } else { "not present" }
    );
    println!();

    let warmup = 5;
    let iters = 30;
    let sizes = [64usize, 128, 256, 384, 512, 768, 1024];

    println!("{:>5} {:>10} {:>10} {:>10}", "N", "us", "GOPS", "max_err");

    for &n in &sizes {
        let m = n;
        let k = n;
        // u8 in [0, 127] (sign-bit safe for the i8 VPMADDUBSW path), i8 in [-100, 100].
        let a: Vec<u8> = (0..m * k)
            .map(|i| (((i * 7 + 3) as i32) & 0x7f) as u8)
            .collect();
        let b: Vec<i8> = (0..k * n)
            .map(|i| (((i as i32) * 11 - 50) % 200 - 100) as i8)
            .collect();
        let mut c = vec![0_i32; m * n];

        let mut max_err = 0_i32;
        if n <= 256 {
            unsafe {
                ferray_linalg::gemm::gemm_i8(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    false,
                );
            }
            let mut c_ref = vec![0_i32; m * n];
            naive_i8gemm(m, n, k, &a, &b, &mut c_ref);
            for i in 0..m * n {
                let e = (c[i] - c_ref[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
        }

        for _ in 0..warmup {
            c.iter_mut().for_each(|x| *x = 0);
            unsafe {
                ferray_linalg::gemm::gemm_i8(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    false,
                );
            }
        }

        let mut times_us = Vec::with_capacity(iters);
        for _ in 0..iters {
            c.iter_mut().for_each(|x| *x = 0);
            let t0 = Instant::now();
            unsafe {
                ferray_linalg::gemm::gemm_i8(
                    m,
                    n,
                    k,
                    a.as_ptr(),
                    b.as_ptr(),
                    c.as_mut_ptr(),
                    false,
                );
            }
            times_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times_us[iters / 2];
        // 1 op = 1 multiply-accumulate (the standard quantized GEMM convention).
        let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        let gops = ops / median / 1e3;

        if n <= 256 {
            println!("{n:>5} {median:>9.0}us {gops:>10.1} {max_err:>10}");
        } else {
            println!("{n:>5} {median:>9.0}us {gops:>10.1} {:>10}", "skip");
        }
    }
}
