// Quick perf check for gemm_i8_signed: dedicated kernel vs synthetic
// "widen-then-i16" baseline (the original wrapper implementation).
// Run as `cargo run --release --example bench_i8s -p ferray-linalg`.
use ferray_linalg::gemm::{gemm_i8_signed, gemm_i16};
use std::time::Instant;

fn run(m: usize, n: usize, k: usize, iters: usize) {
    let a: Vec<i8> = (0..m * k).map(|i| ((i as i32) * 7 - 60) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32) * 11 - 80) as i8).collect();
    let mut c = vec![0_i32; m * n];

    // Warmup.
    unsafe {
        gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), false);
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            gemm_i8_signed(m, n, k, a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), false);
        }
    }
    let dur_native = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        let a16: Vec<i16> = a.iter().map(|&v| v as i16).collect();
        let b16: Vec<i16> = b.iter().map(|&v| v as i16).collect();
        unsafe {
            gemm_i16(m, n, k, a16.as_ptr(), b16.as_ptr(), c.as_mut_ptr(), false);
        }
    }
    let dur_wrapper = t0.elapsed();

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64) * (iters as f64);
    println!(
        "{m:>4}x{n:>4}x{k:>4}  {iters:>4} iters | dedicated {:>7.1} GOPS  | widen+i16 {:>7.1} GOPS  | speedup {:>4.2}x",
        flops / dur_native.as_secs_f64() / 1e9,
        flops / dur_wrapper.as_secs_f64() / 1e9,
        dur_wrapper.as_secs_f64() / dur_native.as_secs_f64(),
    );
}

fn main() {
    let sizes = [
        (32, 32, 32, 2000),
        (64, 64, 64, 1000),
        (128, 128, 128, 200),
        (256, 256, 256, 100),
        (512, 512, 512, 20),
        (1024, 1024, 1024, 5),
    ];
    for &(m, n, k, iters) in &sizes {
        run(m, n, k, iters);
    }
}
