//! Throughput microbenchmark for ferray-ufunc's parallel dispatch.
//!
//! Measures elementwise ops at sizes above and below the parallel
//! threshold to sanity-check that large arrays now go through the Rayon
//! pool.
//!
//! Run with: `cargo run -p ferray-ufunc --release --example bench_parallel`

use ferray_core::{Array, IxDyn};
use ferray_ufunc::{add, exp, sqrt};
use std::time::Instant;

fn make(n: usize, seed: u64) -> Array<f64, IxDyn> {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = ((state >> 33) as f64) / ((1u64 << 31) as f64);
        data.push(x);
    }
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), data).unwrap()
}

fn bench_add(n: usize) {
    let a = make(n, 1);
    let b = make(n, 2);
    // warm up
    let _ = add(&a, &b).unwrap();
    let iters = if n <= 10_000 { 200 } else { 50 };
    let start = Instant::now();
    for _ in 0..iters {
        let _ = add(&a, &b).unwrap();
    }
    let elapsed = start.elapsed();
    let per = elapsed / iters as u32;
    let throughput = (n as f64) / per.as_secs_f64() / 1e9;
    println!(
        "add     n={n:>9}: {per:>10?} / call   ({throughput:5.2} Gelem/s)"
    );
}

fn bench_sqrt(n: usize) {
    let a = make(n, 3);
    let _ = sqrt(&a).unwrap();
    let iters = if n <= 10_000 { 200 } else { 50 };
    let start = Instant::now();
    for _ in 0..iters {
        let _ = sqrt(&a).unwrap();
    }
    let elapsed = start.elapsed();
    let per = elapsed / iters as u32;
    let throughput = (n as f64) / per.as_secs_f64() / 1e9;
    println!(
        "sqrt    n={n:>9}: {per:>10?} / call   ({throughput:5.2} Gelem/s)"
    );
}

fn bench_exp(n: usize) {
    let a = make(n, 4);
    let _ = exp(&a).unwrap();
    let iters = if n <= 10_000 { 200 } else { 30 };
    let start = Instant::now();
    for _ in 0..iters {
        let _ = exp(&a).unwrap();
    }
    let elapsed = start.elapsed();
    let per = elapsed / iters as u32;
    let throughput = (n as f64) / per.as_secs_f64() / 1e9;
    println!(
        "exp     n={n:>9}: {per:>10?} / call   ({throughput:5.2} Gelem/s)"
    );
}

fn main() {
    println!(
        "Thresholds: memory-bound = {}, compute-bound = {}",
        ferray_ufunc::parallel::THRESHOLD_MEMORY_BOUND,
        ferray_ufunc::parallel::THRESHOLD_COMPUTE_BOUND
    );
    println!();
    for n in [1_000, 10_000, 100_000, 999_999, 2_000_000, 10_000_000] {
        bench_add(n);
    }
    println!();
    for n in [1_000, 10_000, 100_000, 999_999, 2_000_000, 10_000_000] {
        bench_sqrt(n);
    }
    println!();
    for n in [1_000, 10_000, 99_999, 200_000, 1_000_000, 10_000_000] {
        bench_exp(n);
    }
}
