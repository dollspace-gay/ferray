// Quick bench comparing sin paths.
// Run as: cargo run --release --example bench_sin -p ferray-ufunc
use std::time::Instant;

fn time_loop(name: &str, iters: usize, n: usize, f: impl Fn(&[f64], &mut [f64])) {
    let input: Vec<f64> = (0..n).map(|i| (i as f64) * 0.013).collect();
    let mut out = vec![0.0_f64; n];

    // warmup
    for _ in 0..5 {
        f(&input, &mut out);
    }
    let t = Instant::now();
    for _ in 0..iters {
        f(&input, &mut out);
    }
    let dur = t.elapsed().as_secs_f64() / iters as f64 * 1e6;
    println!("  {name:>22} N={n:<7}  {dur:>7.2} us");
}

fn main() {
    for n in [1_000, 10_000, 100_000] {
        let iters = if n < 10_000 { 5000 } else { 200 };
        println!("--- N = {n} ---");
        time_loop("scalar core_math::sin", iters, n, |i, o| {
            for (oo, ii) in o.iter_mut().zip(i.iter()) {
                *oo = core_math::sin(*ii);
            }
        });
        time_loop("scalar f64::sin (libm)", iters, n, |i, o| {
            for (oo, ii) in o.iter_mut().zip(i.iter()) {
                *oo = ii.sin();
            }
        });
        time_loop(
            "fast_trig sin_fast_batch",
            iters,
            n,
            ferray_ufunc::fast_trig::sin_fast_batch_f64,
        );
        time_loop("scalar sin_fast (inlined)", iters, n, |i, o| {
            for (oo, ii) in o.iter_mut().zip(i.iter()) {
                *oo = ferray_ufunc::fast_trig::sin_fast_f64(*ii);
            }
        });
    }
}
