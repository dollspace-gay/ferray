//! Rough throughput microbenchmark for the Ziggurat-based standard_normal.
//!
//! Run with: `cargo run -p ferray-random --release --example bench_normal`

use ferray_random::default_rng_seeded;
use std::time::Instant;

fn main() {
    let mut rng = default_rng_seeded(42);

    // Warm up the plan cache / CPU frequency.
    let _ = rng.standard_normal(10_000).unwrap();

    // f64 throughput.
    let n = 10_000_000usize;
    let start = Instant::now();
    let arr = rng.standard_normal(n).unwrap();
    let elapsed = start.elapsed();
    let ns_per_sample = elapsed.as_nanos() as f64 / n as f64;
    let msamples_per_sec = 1e9 / ns_per_sample / 1e6;
    println!(
        "standard_normal(f64, n={n}): {elapsed:?}  ({ns_per_sample:.2} ns/sample, {msamples_per_sec:.1} Msamples/s)"
    );
    // Use the result so LLVM cannot strip it.
    let sum: f64 = arr.as_slice().unwrap().iter().sum();
    println!("    sum = {sum:.3}");

    // f32 throughput.
    let start = Instant::now();
    let arr = rng.standard_normal_f32(n).unwrap();
    let elapsed = start.elapsed();
    let ns_per_sample = elapsed.as_nanos() as f64 / n as f64;
    let msamples_per_sec = 1e9 / ns_per_sample / 1e6;
    println!(
        "standard_normal(f32, n={n}): {elapsed:?}  ({ns_per_sample:.2} ns/sample, {msamples_per_sec:.1} Msamples/s)"
    );
    let sum: f32 = arr.as_slice().unwrap().iter().sum();
    println!("    sum = {sum:.3}");
}
