//! ferray-bench — exhaustive benchmarking binary against NumPy.
//!
//! Wire protocol (batch mode):
//!   `echo '<batch_json>' | ferray-bench batch`
//!
//! Batch input:
//! ```json
//! {
//!   "warmup": 3,
//!   "iterations": 10,
//!   "tests": [
//!     {"function": "sin", "size": "1000", "data": [...]},
//!     {"function": "matmul", "size": "100x100:100x100", "data": [...]},
//!     {"function": "polyfit", "size": "100", "data": [...], "extra": "deg=3"},
//!     ...
//!   ]
//! }
//! ```
//!
//! `data` is a flat `Vec<f64>`. For multi-input operations, the buffer
//! is split according to `size`. The `size` string is parsed by
//! [`ShapeSpec`].
//!
//! Batch output: array of `{"function", "size", "times_ns": [...]}`.
//! Times are per-iteration nanoseconds, raw (no aggregation).

mod bench_exp;

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use std::io::Read;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ferray-bench batch                  (reads batch JSON from stdin)");
        eprintln!("       ferray-bench bench-exp              (run exp() algorithm comparison)");
        eprintln!("       ferray-bench <func> <size>          (single-call mode: reads [f64,..] from stdin, returns result JSON)");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "bench-exp" => bench_exp::run(),
        "batch" => run_batch_mode(),
        // Anything else is treated as a single-call invocation:
        //   ferray-bench <func> <size>
        // stdin: JSON array of f64; stdout: {"data":[..]} or {"data_real":..,"data_imag":..}.
        // This is the wire format expected by `statistical_equivalence.py`.
        func => {
            if args.len() < 3 {
                eprintln!("Single-call mode requires: ferray-bench <func> <size>");
                std::process::exit(1);
            }
            run_single_mode(func, &args[2]);
        }
    }
}

// ---------------------------------------------------------------------------
// ShapeSpec — parse the `size` string into a structured shape.
// ---------------------------------------------------------------------------

/// Encoded shape spec for the `size` field of a batch test.
///
/// Forms:
///   - `"1000"`             — 1-D array of 1000 elements (one input)
///   - `"100x50"`           — 2-D 100×50 array (one input)
///   - `"1000:1000"`        — two 1-D arrays of 1000 each (binary ufunc)
///   - `"100x100:100x100"`  — two 2-D matrices (matmul)
///   - `"1000:1000:1000"`   — three 1-D arrays (where(cond, x, y))
///   - `"100x100"` with extra payload — depends on `function`
#[derive(Debug)]
struct ShapeSpec {
    parts: Vec<Vec<usize>>,
}

impl ShapeSpec {
    fn parse(s: &str) -> Self {
        let parts = s
            .split(':')
            .map(|p| {
                p.split('x')
                    .map(|d| d.parse::<usize>().unwrap_or(0))
                    .collect::<Vec<_>>()
            })
            .collect();
        Self { parts }
    }

    /// Element count of part `i`.
    fn count(&self, i: usize) -> usize {
        self.parts[i].iter().product()
    }
}

// ---------------------------------------------------------------------------
// Batch mode I/O
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct BatchInput {
    warmup: usize,
    iterations: usize,
    tests: Vec<BatchTest>,
}

#[derive(serde::Deserialize)]
struct BatchTest {
    function: String,
    size: String,
    data: Vec<f64>,
    /// Optional `key=value` extras (e.g. `"deg=3"`, `"axis=0"`).
    #[serde(default)]
    extra: String,
}

#[derive(serde::Serialize)]
struct BatchResult {
    function: String,
    size: String,
    times_ns: Vec<u64>,
}

fn run_batch_mode() {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");
    let batch: BatchInput = serde_json::from_str(&input).expect("failed to parse batch JSON");

    // Global warmup so the Rayon pool, allocator pages, and FFT plan
    // cache are populated before the first measured iteration.
    {
        let dummy = vec![1.0_f64; 64];
        let _ = dispatch("sin", "64", &dummy, "");
        let _ = dispatch("sum", "64", &dummy, "");
        let _ = dispatch("matmul", "8x8:8x8", &vec![1.0_f64; 128], "");
        let _ = dispatch("fft", "64", &dummy, "");
    }

    let mut results = Vec::with_capacity(batch.tests.len());
    for t in &batch.tests {
        for _ in 0..batch.warmup {
            let _ = dispatch(&t.function, &t.size, &t.data, &t.extra);
        }
        let mut times = Vec::with_capacity(batch.iterations);
        for _ in 0..batch.iterations {
            let ns = dispatch(&t.function, &t.size, &t.data, &t.extra);
            times.push(ns);
        }
        results.push(BatchResult {
            function: t.function.clone(),
            size: t.size.clone(),
            times_ns: times,
        });
    }

    println!(
        "{}",
        serde_json::to_string(&results).expect("failed to serialize results")
    );
}

// ---------------------------------------------------------------------------
// Single-call mode: run one function on stdin data, emit result JSON.
//
// Wire format (used by statistical_equivalence.py):
//   stdin  : JSON array of f64 (the input data, possibly multi-input flattened)
//   stdout : {"data": [f64, ..]}                          (real result)
//          | {"data_real": [..], "data_imag": [..]}       (complex result, e.g. fft)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize)]
struct SingleResultReal {
    data: Vec<f64>,
}

#[derive(serde::Serialize)]
struct SingleResultComplex {
    data_real: Vec<f64>,
    data_imag: Vec<f64>,
}

fn run_single_mode(func: &str, size: &str) {
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");
    let data: Vec<f64> = serde_json::from_str(&input).expect("failed to parse input JSON");
    let spec = ShapeSpec::parse(size);

    match compute(func, &spec, &data) {
        ComputeResult::Real(v) => {
            println!(
                "{}",
                serde_json::to_string(&SingleResultReal { data: v })
                    .expect("failed to serialize result")
            );
        }
        ComputeResult::Complex(re, im) => {
            println!(
                "{}",
                serde_json::to_string(&SingleResultComplex {
                    data_real: re,
                    data_imag: im,
                })
                .expect("failed to serialize result")
            );
        }
    }
}

enum ComputeResult {
    Real(Vec<f64>),
    Complex(Vec<f64>, Vec<f64>),
}

fn compute(func: &str, spec: &ShapeSpec, data: &[f64]) -> ComputeResult {
    use ComputeResult::{Complex, Real};

    // Helper: run a unary ufunc and return the f64 result vector.
    fn unary_run<F>(data: &[f64], spec: &ShapeSpec, f: F) -> Vec<f64>
    where
        F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<Array<f64, Ix1>>,
    {
        let n = spec.count(0);
        let arr = make_1d(data, n);
        let out = f(&arr).expect("unary op failed");
        out.iter().copied().collect()
    }

    match func {
        // Trig
        "sin" => Real(unary_run(data, spec, ferray_ufunc::sin)),
        "cos" => Real(unary_run(data, spec, ferray_ufunc::cos)),
        "tan" => Real(unary_run(data, spec, ferray_ufunc::tan)),
        "arcsin" => Real(unary_run(data, spec, ferray_ufunc::arcsin)),
        "arccos" => Real(unary_run(data, spec, ferray_ufunc::arccos)),
        "arctan" => Real(unary_run(data, spec, ferray_ufunc::arctan)),
        "sinh" => Real(unary_run(data, spec, ferray_ufunc::sinh)),
        "cosh" => Real(unary_run(data, spec, ferray_ufunc::cosh)),
        "tanh" => Real(unary_run(data, spec, ferray_ufunc::tanh)),
        "arcsinh" => Real(unary_run(data, spec, ferray_ufunc::arcsinh)),
        "arccosh" => Real(unary_run(data, spec, ferray_ufunc::arccosh)),
        "arctanh" => Real(unary_run(data, spec, ferray_ufunc::arctanh)),

        // Exp / log
        "exp" => Real(unary_run(data, spec, ferray_ufunc::exp)),
        "exp2" => Real(unary_run(data, spec, ferray_ufunc::exp2)),
        "expm1" => Real(unary_run(data, spec, ferray_ufunc::expm1)),
        "log" => Real(unary_run(data, spec, ferray_ufunc::log)),
        "log2" => Real(unary_run(data, spec, ferray_ufunc::log2)),
        "log10" => Real(unary_run(data, spec, ferray_ufunc::log10)),
        "log1p" => Real(unary_run(data, spec, ferray_ufunc::log1p)),

        // Algebra
        "sqrt" => Real(unary_run(data, spec, ferray_ufunc::sqrt)),
        "cbrt" => Real(unary_run(data, spec, ferray_ufunc::cbrt)),
        "square" => Real(unary_run(data, spec, ferray_ufunc::square)),
        "abs" => Real(unary_run(data, spec, ferray_ufunc::absolute)),

        // Stats reductions — return as 1-element vec
        "sum" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::sum(&arr, None).expect("sum failed");
            Real(out.iter().copied().collect())
        }
        "mean" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::mean(&arr, None).expect("mean failed");
            Real(out.iter().copied().collect())
        }
        "var" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::var(&arr, None, 0).expect("var failed");
            Real(out.iter().copied().collect())
        }
        "std" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::std_(&arr, None, 0).expect("std failed");
            Real(out.iter().copied().collect())
        }
        "prod" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::prod(&arr, None).expect("prod failed");
            Real(out.iter().copied().collect())
        }
        "min" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::min(&arr, None).expect("min failed");
            Real(out.iter().copied().collect())
        }
        "max" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::max(&arr, None).expect("max failed");
            Real(out.iter().copied().collect())
        }
        "median" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_stats::median(&arr, None).expect("median failed");
            Real(out.iter().copied().collect())
        }

        // Matmul: size "RxC" — A is RxC, B is CxR (statistical_equivalence layout).
        "matmul" => {
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let na = r * c;
            let a = make_dyn(data, &[r, c]);
            let b = make_dyn(&data[na..], &[c, r]);
            let out = ferray_linalg::matmul(&a, &b).expect("matmul failed");
            Real(out.iter().copied().collect())
        }

        // FFT — complex output
        "fft" => {
            use num_complex::Complex;
            let n = spec.count(0);
            let cdata: Vec<Complex<f64>> =
                data[..n].iter().map(|&v| Complex::new(v, 0.0)).collect();
            let arr = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([n]), cdata).unwrap();
            let out = ferray_fft::fft(&arr, None, None, ferray_fft::FftNorm::Backward)
                .expect("fft failed");
            let (re, im): (Vec<f64>, Vec<f64>) = out.iter().map(|c| (c.re, c.im)).unzip();
            Complex(re, im)
        }
        "ifft" => {
            use num_complex::Complex;
            let n = spec.count(0);
            let cdata: Vec<Complex<f64>> =
                data[..n].iter().map(|&v| Complex::new(v, 0.0)).collect();
            let arr = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([n]), cdata).unwrap();
            let out = ferray_fft::ifft(&arr, None, None, ferray_fft::FftNorm::Backward)
                .expect("ifft failed");
            let (re, im): (Vec<f64>, Vec<f64>) = out.iter().map(|c| (c.re, c.im)).unzip();
            Complex(re, im)
        }
        "rfft" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            let out = ferray_fft::rfft(&arr, None, None, ferray_fft::FftNorm::Backward)
                .expect("rfft failed");
            let (re, im): (Vec<f64>, Vec<f64>) = out.iter().map(|c| (c.re, c.im)).unzip();
            Complex(re, im)
        }

        other => {
            eprintln!("ferray-bench single: unsupported function {other}");
            std::process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------

fn make_1d(data: &[f64], n: usize) -> Array<f64, Ix1> {
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data[..n].to_vec()).expect("from_vec 1d")
}

fn make_2d(data: &[f64], rows: usize, cols: usize) -> Array<f64, Ix2> {
    Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data[..rows * cols].to_vec())
        .expect("from_vec 2d")
}

fn make_dyn(data: &[f64], shape: &[usize]) -> Array<f64, IxDyn> {
    let n: usize = shape.iter().product();
    Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), data[..n].to_vec()).expect("from_vec dyn")
}

fn parse_kv(extra: &str, key: &str) -> Option<String> {
    for kv in extra.split(',') {
        let kv = kv.trim();
        if let Some(eq) = kv.find('=') {
            if &kv[..eq] == key {
                return Some(kv[eq + 1..].to_string());
            }
        }
    }
    None
}

fn parse_kv_usize(extra: &str, key: &str, default: usize) -> usize {
    parse_kv(extra, key)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[inline]
fn time_op<F: FnOnce()>(f: F) -> u64 {
    let start = Instant::now();
    f();
    start.elapsed().as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch(func: &str, size: &str, data: &[f64], extra: &str) -> u64 {
    let spec = ShapeSpec::parse(size);
    match func {
        // ---------- unary ufuncs ----------
        "sin" => unary(data, &spec, ferray_ufunc::sin),
        "cos" => unary(data, &spec, ferray_ufunc::cos),
        "tan" => unary(data, &spec, ferray_ufunc::tan),
        "arcsin" => unary(data, &spec, ferray_ufunc::arcsin),
        "arccos" => unary(data, &spec, ferray_ufunc::arccos),
        "arctan" => unary(data, &spec, ferray_ufunc::arctan),
        "sinh" => unary(data, &spec, ferray_ufunc::sinh),
        "cosh" => unary(data, &spec, ferray_ufunc::cosh),
        "tanh" => unary(data, &spec, ferray_ufunc::tanh),
        "arcsinh" => unary(data, &spec, ferray_ufunc::arcsinh),
        "arccosh" => unary(data, &spec, ferray_ufunc::arccosh),
        "arctanh" => unary(data, &spec, ferray_ufunc::arctanh),
        "exp" => unary(data, &spec, ferray_ufunc::exp),
        "exp2" => unary(data, &spec, ferray_ufunc::exp2),
        "expm1" => unary(data, &spec, ferray_ufunc::expm1),
        "log" => unary(data, &spec, ferray_ufunc::log),
        "log2" => unary(data, &spec, ferray_ufunc::log2),
        "log10" => unary(data, &spec, ferray_ufunc::log10),
        "log1p" => unary(data, &spec, ferray_ufunc::log1p),
        "sqrt" => unary(data, &spec, ferray_ufunc::sqrt),
        "cbrt" => unary(data, &spec, ferray_ufunc::cbrt),
        "square" => unary(data, &spec, ferray_ufunc::square),
        "abs" => unary(data, &spec, ferray_ufunc::absolute),
        "sign" => unary(data, &spec, ferray_ufunc::sign),
        "ceil" => unary(data, &spec, ferray_ufunc::ceil),
        "floor" => unary(data, &spec, ferray_ufunc::floor),
        "trunc" => unary(data, &spec, ferray_ufunc::trunc),
        "rint" => unary(data, &spec, ferray_ufunc::rint),
        "negative" => unary(data, &spec, ferray_ufunc::negative),
        "positive" => unary(data, &spec, ferray_ufunc::positive),
        "reciprocal" => unary(data, &spec, ferray_ufunc::reciprocal),
        "deg2rad" => unary(data, &spec, ferray_ufunc::deg2rad),
        "rad2deg" => unary(data, &spec, ferray_ufunc::rad2deg),
        "sinc" => unary(data, &spec, ferray_ufunc::sinc),
        "i0" => unary(data, &spec, ferray_ufunc::i0),

        // ---------- unary predicates returning bool ----------
        "isnan" => unary_to_bool(data, &spec, ferray_ufunc::isnan),
        "isinf" => unary_to_bool(data, &spec, ferray_ufunc::isinf),
        "isfinite" => unary_to_bool(data, &spec, ferray_ufunc::isfinite),
        "signbit" => unary_to_bool(data, &spec, ferray_ufunc::signbit),

        // ---------- binary ufuncs ----------
        "add" => binary(data, &spec, ferray_ufunc::add),
        "subtract" => binary(data, &spec, ferray_ufunc::subtract),
        "multiply" => binary(data, &spec, ferray_ufunc::multiply),
        "divide" => binary(data, &spec, ferray_ufunc::divide),
        "power" => binary(data, &spec, ferray_ufunc::power),
        "maximum" => binary(data, &spec, ferray_ufunc::maximum),
        "minimum" => binary(data, &spec, ferray_ufunc::minimum),
        "fmax" => binary(data, &spec, ferray_ufunc::fmax),
        "fmin" => binary(data, &spec, ferray_ufunc::fmin),
        "arctan2" => binary(data, &spec, ferray_ufunc::arctan2),
        "hypot" => binary(data, &spec, ferray_ufunc::hypot),
        "copysign" => binary(data, &spec, ferray_ufunc::copysign),
        "logaddexp" => binary(data, &spec, ferray_ufunc::logaddexp),

        // ---------- comparisons (return bool) ----------
        "greater" => binary_to_bool(data, &spec, ferray_ufunc::greater),
        "less" => binary_to_bool(data, &spec, ferray_ufunc::less),
        "equal" => binary_to_bool(data, &spec, ferray_ufunc::equal),
        "not_equal" => binary_to_bool(data, &spec, ferray_ufunc::not_equal),
        "greater_equal" => binary_to_bool(data, &spec, ferray_ufunc::greater_equal),
        "less_equal" => binary_to_bool(data, &spec, ferray_ufunc::less_equal),

        // ---------- cumulative ----------
        "cumsum" => unary(data, &spec, |a| ferray_ufunc::cumsum(a, None)),
        "cumprod" => unary(data, &spec, |a| ferray_ufunc::cumprod(a, None)),
        "diff" => unary_diff(data, &spec),
        "gradient" => unary_gradient(data, &spec),
        "trapezoid" => unary_to_scalar_f(data, &spec, |a| {
            ferray_ufunc::trapezoid(a, None, None)
        }),

        // ---------- reductions ----------
        "sum" => reduction(data, &spec, |a| ferray_stats::sum(a, None)),
        "prod" => reduction(data, &spec, |a| ferray_stats::prod(a, None)),
        "mean" => reduction(data, &spec, |a| ferray_stats::mean(a, None)),
        "var" => reduction(data, &spec, |a| ferray_stats::var(a, None, 0)),
        "std" => reduction(data, &spec, |a| ferray_stats::std_(a, None, 0)),
        "min" => reduction(data, &spec, |a| ferray_stats::min(a, None)),
        "max" => reduction(data, &spec, |a| ferray_stats::max(a, None)),
        "median" => reduction(data, &spec, |a| ferray_stats::median(a, None)),
        "percentile_50" => reduction(data, &spec, |a| {
            ferray_stats::percentile(a, 50.0, None)
        }),
        "argmin" => reduction_u64(data, &spec, |a| ferray_stats::argmin(a, None)),
        "argmax" => reduction_u64(data, &spec, |a| ferray_stats::argmax(a, None)),
        "ptp" => reduction(data, &spec, |a| ferray_stats::ptp(a, None)),
        "nansum" => reduction(data, &spec, |a| ferray_stats::nansum(a, None)),
        "nanmean" => reduction(data, &spec, |a| ferray_stats::nanmean(a, None)),
        "nanmax" => reduction(data, &spec, |a| ferray_stats::nanmax(a, None)),
        "nanmin" => reduction(data, &spec, |a| ferray_stats::nanmin(a, None)),
        "nanstd" => reduction(data, &spec, |a| ferray_stats::nanstd(a, None, 0)),
        "nanvar" => reduction(data, &spec, |a| ferray_stats::nanvar(a, None, 0)),
        "count_nonzero" => reduction_count_nonzero(data, &spec),

        // ---------- searching / sorting ----------
        "sort" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            time_op(|| {
                let _ = ferray_stats::sort(&arr, None, ferray_stats::SortKind::Quick);
            })
        }
        "argsort" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            time_op(|| {
                let _ = ferray_stats::argsort(&arr, None);
            })
        }
        "unique" => {
            let n = spec.count(0);
            let arr = make_1d(data, n);
            time_op(|| {
                let _ = ferray_stats::unique(&arr, false, false, false);
            })
        }
        "searchsorted" => {
            // size: "sorted_n:query_n"
            let sn = spec.count(0);
            let qn = spec.count(1);
            let sorted = make_1d(data, sn);
            let queries = make_1d(&data[sn..], qn);
            time_op(|| {
                let _ =
                    ferray_stats::searchsorted(&sorted, &queries, ferray_stats::Side::Left);
            })
        }
        "bincount" => {
            // data is u64-cast i64s for bincount. We simulate with positive ints.
            let n = spec.count(0);
            let ints: Vec<u64> = data[..n].iter().map(|x| (*x as u64).min(1023)).collect();
            let arr = Array::<u64, Ix1>::from_vec(Ix1::new([n]), ints).unwrap();
            time_op(|| {
                let _ = ferray_stats::bincount_u64(&arr, 0);
            })
        }

        // ---------- linalg ----------
        "matmul" => matmul(data, &spec),
        "dot_1d" => dot_1d(data, &spec),
        "inner" => inner(data, &spec),
        "outer" => outer(data, &spec),
        "trace" => {
            let r = spec.parts[0][0];
            let arr = make_2d(data, r, r);
            time_op(|| {
                let _ = ferray_linalg::trace(&arr);
            })
        }
        "det" => {
            let r = spec.parts[0][0];
            let arr = make_2d(data, r, r);
            time_op(|| {
                let _ = ferray_linalg::det(&arr);
            })
        }
        "inv" => {
            let r = spec.parts[0][0];
            let arr = make_2d(data, r, r);
            time_op(|| {
                let _ = ferray_linalg::inv(&arr);
            })
        }
        "solve" => {
            // size: NxN (matrix), data: Nx(N+1) (matrix + vector)
            let r = spec.parts[0][0];
            let mat = make_2d(data, r, r);
            let b = make_dyn(&data[r * r..], &[r]);
            time_op(|| {
                let _ = ferray_linalg::solve(&mat, &b);
            })
        }
        "qr" => {
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let arr = make_2d(data, r, c);
            time_op(|| {
                let _ = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced);
            })
        }
        "svd" => {
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let arr = make_2d(data, r, c);
            time_op(|| {
                let _ = ferray_linalg::svd(&arr, true);
            })
        }
        "cholesky" => {
            // expect SPD input
            let r = spec.parts[0][0];
            let arr = make_2d(data, r, r);
            time_op(|| {
                let _ = ferray_linalg::cholesky(&arr);
            })
        }
        "eigh" => {
            let r = spec.parts[0][0];
            let arr = make_2d(data, r, r);
            time_op(|| {
                let _ = ferray_linalg::eigh(&arr);
            })
        }
        "norm" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_linalg::vector_norm(
                    &arr,
                    ferray_linalg::NormOrder::L2,
                    None,
                    false,
                );
            })
        }

        // ---------- FFT ----------
        "fft" => fft_complex(data, &spec, false),
        "ifft" => fft_complex(data, &spec, true),
        "rfft" => fft_real(data, &spec),
        "fftshift" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_fft::fftshift(&arr, None);
            })
        }

        // ---------- creation ----------
        "zeros" => {
            let n = spec.count(0);
            time_op(|| {
                let _: Array<f64, Ix1> = Array::zeros(Ix1::new([n])).unwrap();
            })
        }
        "ones" => {
            let n = spec.count(0);
            time_op(|| {
                let _: Array<f64, Ix1> = ferray_core::creation::ones(Ix1::new([n])).unwrap();
            })
        }
        "arange" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_core::creation::arange(0.0_f64, n as f64, 1.0);
            })
        }
        "linspace" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_core::creation::linspace(0.0_f64, 1.0, n, true);
            })
        }
        "eye" => {
            let n = spec.count(0);
            time_op(|| {
                let _: Array<f64, Ix2> = ferray_core::creation::eye(n, n, 0).unwrap();
            })
        }

        // ---------- manipulation ----------
        "reshape" => {
            // size: source_shape (e.g., "100x100"); reshape to (rows*cols / 4, 4)
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let arr = make_dyn(data, &[r, c]);
            time_op(|| {
                let _ = ferray_core::manipulation::reshape(&arr, &[r * c / 4, 4]);
            })
        }
        "transpose" => {
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let arr = make_dyn(data, &[r, c]);
            time_op(|| {
                let _ = ferray_core::manipulation::transpose(&arr, None);
            })
        }
        "concatenate" => {
            // size: "n:n" — concat two 1-d arrays along axis 0
            let n = spec.count(0);
            let a = make_dyn(data, &[n]);
            let b = make_dyn(&data[n..], &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::concatenate(&[a.clone(), b.clone()], 0);
            })
        }
        "stack" => {
            let n = spec.count(0);
            let a = make_dyn(data, &[n]);
            let b = make_dyn(&data[n..], &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::stack(&[a.clone(), b.clone()], 0);
            })
        }
        "ravel" => {
            let r = spec.parts[0][0];
            let c = spec.parts[0][1];
            let arr = make_dyn(data, &[r, c]);
            time_op(|| {
                let _ = ferray_core::manipulation::ravel(&arr);
            })
        }
        "flip" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::flip(&arr, 0);
            })
        }
        "roll" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::roll(&arr, 3, None);
            })
        }
        "tile" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::extended::tile(&arr, &[2]);
            })
        }
        "repeat" => {
            let n = spec.count(0);
            let arr = make_dyn(data, &[n]);
            time_op(|| {
                let _ = ferray_core::manipulation::extended::repeat(&arr, 3, None);
            })
        }

        // ---------- random ----------
        "random_normal" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.standard_normal(n);
            })
        }
        "random_uniform" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.uniform(0.0, 1.0, n);
            })
        }
        "random_integers" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.integers(0, 1000, n);
            })
        }
        "random_exponential" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.exponential(1.0, n);
            })
        }
        "random_gamma" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.gamma(2.0, 1.0, n);
            })
        }
        "random_poisson" => {
            let n = spec.count(0);
            time_op(|| {
                let mut rng = ferray_random::default_rng_seeded(42);
                let _ = rng.poisson(3.0, n);
            })
        }

        // ---------- polynomial ----------
        "polyfit" => {
            // size: "n", extra: "deg=K". data first half is x, second half is y.
            let n = spec.count(0);
            let deg = parse_kv_usize(extra, "deg", 3);
            let xs = data[..n].to_vec();
            let ys = data[n..2 * n].to_vec();
            time_op(|| {
                use ferray_polynomial::Poly;
                let _ = ferray_polynomial::Polynomial::fit(&xs, &ys, deg);
            })
        }
        "polyval" => {
            // size: "n", extra: "deg=K"; first K+1 values are coefficients, rest are x.
            let n = spec.count(0);
            let deg = parse_kv_usize(extra, "deg", 3);
            let coeffs = data[..deg + 1].to_vec();
            let xs = data[deg + 1..deg + 1 + n].to_vec();
            time_op(|| {
                use ferray_polynomial::Poly;
                let p = ferray_polynomial::Polynomial::new(&coeffs);
                let _ = p.eval_many(&xs);
            })
        }
        "polyroots" => {
            // size: "n" — coefficients length
            let n = spec.count(0);
            let coeffs = data[..n].to_vec();
            time_op(|| {
                let _ = ferray_polynomial::roots::find_roots_from_power_coeffs(&coeffs);
            })
        }

        // ---------- window ----------
        "hamming" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_window::hamming(n);
            })
        }
        "hanning" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_window::hanning(n);
            })
        }
        "bartlett" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_window::bartlett(n);
            })
        }
        "blackman" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_window::blackman(n);
            })
        }
        "kaiser" => {
            let n = spec.count(0);
            time_op(|| {
                let _ = ferray_window::kaiser(n, 14.0);
            })
        }

        _ => {
            eprintln!("ferray-bench: unknown function {func}");
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Op runners
// ---------------------------------------------------------------------------

fn unary<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<Array<f64, Ix1>>,
{
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = f(&arr);
    })
}

fn unary_to_bool<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<Array<bool, Ix1>>,
{
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = f(&arr);
    })
}

fn unary_to_scalar_f<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<f64>,
{
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = f(&arr);
    })
}

fn binary<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(
        &Array<f64, Ix1>,
        &Array<f64, Ix1>,
    ) -> ferray_core::FerrayResult<Array<f64, Ix1>>,
{
    let n = spec.count(0);
    let a = make_1d(data, n);
    let b = make_1d(&data[n..], n);
    time_op(|| {
        let _ = f(&a, &b);
    })
}

fn binary_to_bool<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(
        &Array<f64, Ix1>,
        &Array<f64, Ix1>,
    ) -> ferray_core::FerrayResult<Array<bool, Ix1>>,
{
    let n = spec.count(0);
    let a = make_1d(data, n);
    let b = make_1d(&data[n..], n);
    time_op(|| {
        let _ = f(&a, &b);
    })
}

fn unary_diff(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = ferray_ufunc::diff(&arr, 1);
    })
}

fn unary_gradient(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = ferray_ufunc::gradient(&arr, Some(1.0));
    })
}

fn reduction<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<Array<f64, IxDyn>>,
{
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = f(&arr);
    })
}

fn reduction_u64<F>(data: &[f64], spec: &ShapeSpec, f: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferray_core::FerrayResult<Array<u64, IxDyn>>,
{
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = f(&arr);
    })
}

fn reduction_count_nonzero(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = ferray_stats::count_nonzero(&arr, None);
    })
}

fn matmul(data: &[f64], spec: &ShapeSpec) -> u64 {
    // size: "RxC:CxK" — A is RxC, B is CxK
    let r = spec.parts[0][0];
    let c = spec.parts[0][1];
    let k = spec.parts[1][1];
    let na = r * c;
    let a = make_dyn(data, &[r, c]);
    let b = make_dyn(&data[na..], &[c, k]);
    time_op(|| {
        let _ = ferray_linalg::matmul(&a, &b);
    })
}

fn dot_1d(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let a = make_dyn(data, &[n]);
    let b = make_dyn(&data[n..], &[n]);
    time_op(|| {
        let _ = ferray_linalg::dot(&a, &b);
    })
}

fn inner(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let a = make_dyn(data, &[n]);
    let b = make_dyn(&data[n..], &[n]);
    time_op(|| {
        let _ = ferray_linalg::inner(&a, &b);
    })
}

fn outer(data: &[f64], spec: &ShapeSpec) -> u64 {
    let m = spec.count(0);
    let n = spec.count(1);
    let a = make_dyn(data, &[m]);
    let b = make_dyn(&data[m..], &[n]);
    time_op(|| {
        let _ = ferray_linalg::outer(&a, &b);
    })
}

fn fft_complex(data: &[f64], spec: &ShapeSpec, inverse: bool) -> u64 {
    use num_complex::Complex;
    let n = spec.count(0);
    let cdata: Vec<Complex<f64>> = data[..n].iter().map(|&v| Complex::new(v, 0.0)).collect();
    let arr = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([n]), cdata).unwrap();
    if inverse {
        time_op(|| {
            let _ = ferray_fft::ifft(&arr, None, None, ferray_fft::FftNorm::Backward);
        })
    } else {
        time_op(|| {
            let _ = ferray_fft::fft(&arr, None, None, ferray_fft::FftNorm::Backward);
        })
    }
}

fn fft_real(data: &[f64], spec: &ShapeSpec) -> u64 {
    let n = spec.count(0);
    let arr = make_1d(data, n);
    time_op(|| {
        let _ = ferray_fft::rfft(&arr, None, None, ferray_fft::FftNorm::Backward);
    })
}
