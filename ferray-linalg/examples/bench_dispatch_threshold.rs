// Benchmark to find the real ferray-gemm vs faer crossover for f64.
// Run as `cargo run --release --example bench_dispatch_threshold`.
use ferray_linalg::gemm::gemm_f64;
use std::time::Instant;

fn time_ferray(m: usize, n: usize, k: usize, iters: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut c = vec![0.0_f64; m * n];
    // Warmup
    for _ in 0..5 {
        gemm_f64(m, n, k, 1.0, a, b, 0.0, &mut c);
    }
    let t = Instant::now();
    for _ in 0..iters {
        gemm_f64(m, n, k, 1.0, a, b, 0.0, &mut c);
    }
    let dur = t.elapsed().as_secs_f64();
    dur / iters as f64 * 1e6
}

fn time_faer(m: usize, n: usize, k: usize, iters: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut c = vec![0.0_f64; m * n];
    // Warmup
    for _ in 0..5 {
        let a_ref = faer::MatRef::from_row_major_slice(a, m, k);
        let b_ref = faer::MatRef::from_row_major_slice(b, k, n);
        let c_mut = faer::MatMut::from_row_major_slice_mut(&mut c, m, n);
        faer::linalg::matmul::matmul(
            c_mut,
            faer::Accum::Replace,
            a_ref,
            b_ref,
            1.0,
            faer::Par::Seq,
        );
    }
    let t = Instant::now();
    for _ in 0..iters {
        let a_ref = faer::MatRef::from_row_major_slice(a, m, k);
        let b_ref = faer::MatRef::from_row_major_slice(b, k, n);
        let c_mut = faer::MatMut::from_row_major_slice_mut(&mut c, m, n);
        faer::linalg::matmul::matmul(
            c_mut,
            faer::Accum::Replace,
            a_ref,
            b_ref,
            1.0,
            faer::Par::Seq,
        );
    }
    let dur = t.elapsed().as_secs_f64();
    dur / iters as f64 * 1e6
}

fn time_naive(m: usize, n: usize, k: usize, iters: usize, a: &[f64], b: &[f64]) -> f64 {
    let mut c = vec![0.0_f64; m * n];
    let t = Instant::now();
    for _ in 0..iters {
        c.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..m {
            for p in 0..k {
                let a_ip = a[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }
    }
    let dur = t.elapsed().as_secs_f64();
    dur / iters as f64 * 1e6
}

fn main() {
    let sizes = [
        16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320,
    ];
    println!("    N   ferray_us   faer_us  naive_us  | best");
    for &n in &sizes {
        let iters = if n < 64 {
            5000
        } else if n < 128 {
            1000
        } else {
            200
        };
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.013 + 0.5).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.017 - 0.3).collect();
        let f = time_ferray(n, n, n, iters, &a, &b);
        let fa = time_faer(n, n, n, iters, &a, &b);
        let nv = time_naive(n, n, n, iters.min(200), &a, &b);
        let best = if f < fa && f < nv {
            "ferray"
        } else if fa < nv {
            "faer"
        } else {
            "naive"
        };
        println!("{:>5}  {:>9.2}  {:>9.2}  {:>9.2}  | {}", n, f, fa, nv, best);
    }
}
