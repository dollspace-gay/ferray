//! Rough throughput microbenchmark for `ferray_linalg::matmul` and
//! `matrix_power`, verifying the faer-backed fast path.
//!
//! Run with: `cargo run -p ferray-linalg --release --example bench_matmul`

use ferray_core::{Array, IxDyn};
use std::time::Instant;

fn make_matrix(n: usize, seed: u64) -> Array<f64, IxDyn> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut data = Vec::with_capacity(n * n);
    for _ in 0..(n * n) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
        data.push(x);
    }
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n, n]), data).unwrap()
}

fn bench_matmul(n: usize) {
    let a = make_matrix(n, 1);
    let b = make_matrix(n, 2);

    // Warm up.
    let _ = ferray_linalg::matmul(&a, &b).unwrap();

    let iters = if n <= 128 { 20 } else { 5 };
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ferray_linalg::matmul(&a, &b).unwrap();
    }
    let elapsed = start.elapsed();
    let per_call = elapsed / iters as u32;
    // GFLOPS for matmul = 2 * n^3 per call
    let flops_per_call = 2.0 * (n as f64).powi(3);
    let gflops = flops_per_call / per_call.as_secs_f64() / 1e9;
    println!("matmul({n}x{n}): {per_call:?} per call, {gflops:.1} GFLOPS (avg of {iters} iters)");
}

fn bench_matrix_power(n: usize, power: i64) {
    use ferray_core::dimension::Ix2;
    let a_d = make_matrix(n, 3);
    let a =
        Array::<f64, Ix2>::from_vec(Ix2::new([n, n]), a_d.as_slice().unwrap().to_vec()).unwrap();

    let _ = ferray_linalg::matrix_power(&a, power).unwrap();

    let iters = 5;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = ferray_linalg::matrix_power(&a, power).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "matrix_power({n}x{n}, ^{power}): {:?} per call (avg of {iters} iters)",
        elapsed / iters as u32
    );
}

fn main() {
    for n in [64, 128, 256, 512, 1024] {
        bench_matmul(n);
    }
    println!();
    for n in [32, 128, 256] {
        bench_matrix_power(n, 16);
    }
}
