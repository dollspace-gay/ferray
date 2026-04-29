//! Allocating vs `_into` microbenchmark — quantifies the cost of the
//! implicit per-call Vec allocation that the `_into` variants remove.
//!
//! Run with: `cargo run -p ferray-ufunc --release --example bench_into`

// Benchmarks lift integer sizes into f64 throughput numbers as part of
// reporting; the casts are part of the bench output, not bugs.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]

use ferray_core::{Array, IxDyn};
use ferray_ufunc::{add, add_into, exp, exp_into};
use std::time::Instant;

fn zeros(n: usize) -> Array<f64, IxDyn> {
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), vec![0.0; n]).unwrap()
}

fn ramp(n: usize, start: f64) -> Array<f64, IxDyn> {
    Array::<f64, IxDyn>::from_vec(IxDyn::new(&[n]), (0..n).map(|i| start + i as f64).collect())
        .unwrap()
}

fn bench_add(n: usize) {
    let a = ramp(n, 0.0);
    let b = ramp(n, 1000.0);
    let iters = (10_000_000usize / n.max(1)).max(5);

    // Allocating variant: each call allocates a fresh output Vec.
    let start = Instant::now();
    for _ in 0..iters {
        let _ = add(&a, &b).unwrap();
    }
    let alloc_elapsed = start.elapsed();

    // In-place variant: caller owns the output buffer; zero allocation per call.
    let mut out = zeros(n);
    let start = Instant::now();
    for _ in 0..iters {
        add_into(&a, &b, &mut out).unwrap();
    }
    let into_elapsed = start.elapsed();

    let alloc_ns = alloc_elapsed.as_nanos() as f64 / iters as f64;
    let into_ns = into_elapsed.as_nanos() as f64 / iters as f64;
    let speedup = alloc_ns / into_ns;
    println!(
        "add  n={n:>8}:  alloc {alloc_ns:>10.1} ns/call   _into {into_ns:>10.1} ns/call   speedup {speedup:.2}x"
    );
}

fn bench_exp(n: usize) {
    let a = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[n]),
        (0..n).map(|i| (i as f64) * 1e-4).collect(),
    )
    .unwrap();
    let iters = (2_000_000usize / n.max(1)).max(5);

    let start = Instant::now();
    for _ in 0..iters {
        let _ = exp(&a).unwrap();
    }
    let alloc_elapsed = start.elapsed();

    let mut out = zeros(n);
    let start = Instant::now();
    for _ in 0..iters {
        exp_into(&a, &mut out).unwrap();
    }
    let into_elapsed = start.elapsed();

    let alloc_ns = alloc_elapsed.as_nanos() as f64 / iters as f64;
    let into_ns = into_elapsed.as_nanos() as f64 / iters as f64;
    let speedup = alloc_ns / into_ns;
    println!(
        "exp  n={n:>8}:  alloc {alloc_ns:>10.1} ns/call   _into {into_ns:>10.1} ns/call   speedup {speedup:.2}x"
    );
}

fn main() {
    println!("allocating vs _into, time per call (lower is better)");
    println!();
    for n in [100, 1000, 10_000, 100_000, 1_000_000] {
        bench_add(n);
    }
    println!();
    for n in [100, 1000, 10_000, 100_000, 1_000_000] {
        bench_exp(n);
    }
}
