//! Benchmark binary for statistical equivalence testing and speed benchmarking.
//!
//! **Single mode** (for accuracy tests):
//!   echo '[1.0, 2.0, 3.0]' | ferrum-bench <function> <size> [extra_args...]
//!
//! **Batch mode** (for speed tests — runs all benchmarks in one process):
//!   echo '<batch_json>' | ferrum-bench batch
//!
//! Batch JSON input format:
//! ```json
//! {
//!   "warmup": 3,
//!   "iterations": 10,
//!   "tests": [
//!     {"function": "sin", "size": "1000", "data": [1.0, 2.0, ...]},
//!     {"function": "matmul", "size": "10x10", "data": [...]},
//!     ...
//!   ]
//! }
//! ```
//!
//! Batch JSON output: array of `{"function", "size", "times_ns": [t1, t2, ...]}`.

mod bench_exp;

use ferrum_core::dimension::{Ix1, IxDyn};
use ferrum_core::Array;
use num_complex::Complex;
use std::io::Read;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: ferrum-bench <function> <size> [extra_args...]");
        eprintln!("       ferrum-bench batch  (reads batch JSON from stdin)");
        eprintln!("       ferrum-bench bench-exp  (run exp() algorithm comparison)");
        std::process::exit(1);
    }

    if args[1] == "bench-exp" {
        bench_exp::run();
        return;
    }

    if args[1] == "batch" {
        run_batch_mode();
        return;
    }

    if args.len() < 3 {
        eprintln!("Usage: ferrum-bench <function> <size> [extra_args...]");
        std::process::exit(1);
    }

    let func_name = &args[1];
    let size_str = &args[2];

    // Read input data from stdin
    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read stdin");

    let input_data: Vec<f64> = serde_json::from_str(&input).expect("Failed to parse input JSON");

    // Dispatch based on function name — each runner returns (json_value, elapsed_ns)
    let (result, elapsed_ns) = dispatch_single(func_name, size_str, &input_data);

    // Inject elapsed_ns into the JSON output
    let mut result_map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_value(result).expect("Expected JSON object");
    result_map.insert("elapsed_ns".to_string(), serde_json::Value::from(elapsed_ns));

    // Output as JSON
    println!(
        "{}",
        serde_json::to_string(&result_map).expect("Failed to serialize result")
    );
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn dispatch_single(
    func_name: &str,
    size_str: &str,
    input_data: &[f64],
) -> (serde_json::Value, u64) {
    match func_name {
        // ---- Ufunc: unary float ops ----
        "sin" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::sin),
        "cos" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::cos),
        "tan" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::tan),
        "exp" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::exp),
        "log" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::log),
        "log2" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::log2),
        "log10" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::log10),
        "sqrt" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::sqrt),
        "abs" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::absolute),
        "exp2" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::exp2),
        "expm1" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::expm1),
        "log1p" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::log1p),
        "arcsin" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::arcsin),
        "arccos" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::arccos),
        "arctan" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::arctan),
        "sinh" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::sinh),
        "cosh" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::cosh),
        "tanh" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::tanh),
        "cbrt" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::cbrt),
        "ceil" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::ceil),
        "floor" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::floor),
        "rint" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::rint),
        "sign" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::sign),
        "square" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::square),
        "negative" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::negative),
        "reciprocal" => run_unary_ufunc(input_data, size_str, ferrum_ufunc::reciprocal),

        // ---- Stats: reductions ----
        "mean" => run_reduction(input_data, size_str, |arr| ferrum_stats::mean(arr, None)),
        "var" => run_reduction(input_data, size_str, |arr| ferrum_stats::var(arr, None, 0)),
        "std" => run_reduction(input_data, size_str, |arr| ferrum_stats::std_(arr, None, 0)),
        "sum" => run_reduction(input_data, size_str, |arr| ferrum_stats::sum(arr, None)),

        // ---- Linalg: matmul ----
        "matmul" => run_matmul(input_data, size_str),

        // ---- FFT ----
        "fft" => run_fft(input_data, size_str),

        other => {
            eprintln!("Unknown function: {other}");
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Batch mode
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
        .expect("Failed to read stdin");

    let batch: BatchInput = serde_json::from_str(&input).expect("Failed to parse batch JSON");

    // Global warmup: trigger rayon thread pool + FFT plan cache + allocator
    // by running a small dummy operation of each type.
    {
        let dummy = vec![1.0f64; 64];
        let _ = dispatch_timed("sin", "64", &dummy);
        let _ = dispatch_timed("sum", "64", &dummy);
        let dummy_mat = vec![1.0f64; 128];
        let _ = dispatch_timed("matmul", "8x8", &dummy_mat);
        let _ = dispatch_timed("fft", "64", &dummy);
    }

    let mut results = Vec::with_capacity(batch.tests.len());

    for test in &batch.tests {
        // Per-test warmup
        for _ in 0..batch.warmup {
            let _ = dispatch_timed(&test.function, &test.size, &test.data);
        }

        // Timed iterations
        let mut times = Vec::with_capacity(batch.iterations);
        for _ in 0..batch.iterations {
            let ns = dispatch_timed(&test.function, &test.size, &test.data);
            times.push(ns);
        }

        results.push(BatchResult {
            function: test.function.clone(),
            size: test.size.clone(),
            times_ns: times,
        });
    }

    println!(
        "{}",
        serde_json::to_string(&results).expect("Failed to serialize batch results")
    );
}

/// Run a single timed invocation of a function, returning only elapsed_ns.
/// Does not serialize output data (speed-only mode).
fn dispatch_timed(func_name: &str, size_str: &str, input_data: &[f64]) -> u64 {
    match func_name {
        "sin" => time_unary_ufunc(input_data, ferrum_ufunc::sin),
        "cos" => time_unary_ufunc(input_data, ferrum_ufunc::cos),
        "tan" => time_unary_ufunc(input_data, ferrum_ufunc::tan),
        "exp" => time_unary_ufunc(input_data, ferrum_ufunc::exp),
        "log" => time_unary_ufunc(input_data, ferrum_ufunc::log),
        "log2" => time_unary_ufunc(input_data, ferrum_ufunc::log2),
        "log10" => time_unary_ufunc(input_data, ferrum_ufunc::log10),
        "sqrt" => time_unary_ufunc(input_data, ferrum_ufunc::sqrt),
        "abs" => time_unary_ufunc(input_data, ferrum_ufunc::absolute),
        "exp2" => time_unary_ufunc(input_data, ferrum_ufunc::exp2),
        "expm1" => time_unary_ufunc(input_data, ferrum_ufunc::expm1),
        "log1p" => time_unary_ufunc(input_data, ferrum_ufunc::log1p),
        "arcsin" => time_unary_ufunc(input_data, ferrum_ufunc::arcsin),
        "arccos" => time_unary_ufunc(input_data, ferrum_ufunc::arccos),
        "arctan" => time_unary_ufunc(input_data, ferrum_ufunc::arctan),
        "sinh" => time_unary_ufunc(input_data, ferrum_ufunc::sinh),
        "cosh" => time_unary_ufunc(input_data, ferrum_ufunc::cosh),
        "tanh" => time_unary_ufunc(input_data, ferrum_ufunc::tanh),
        "cbrt" => time_unary_ufunc(input_data, ferrum_ufunc::cbrt),
        "ceil" => time_unary_ufunc(input_data, ferrum_ufunc::ceil),
        "floor" => time_unary_ufunc(input_data, ferrum_ufunc::floor),
        "rint" => time_unary_ufunc(input_data, ferrum_ufunc::rint),
        "sign" => time_unary_ufunc(input_data, ferrum_ufunc::sign),
        "square" => time_unary_ufunc(input_data, ferrum_ufunc::square),
        "negative" => time_unary_ufunc(input_data, ferrum_ufunc::negative),
        "reciprocal" => time_unary_ufunc(input_data, ferrum_ufunc::reciprocal),
        "mean" => time_reduction(input_data, |arr| ferrum_stats::mean(arr, None)),
        "var" => time_reduction(input_data, |arr| ferrum_stats::var(arr, None, 0)),
        "std" => time_reduction(input_data, |arr| ferrum_stats::std_(arr, None, 0)),
        "sum" => time_reduction(input_data, |arr| ferrum_stats::sum(arr, None)),
        "matmul" => time_matmul(input_data, size_str),
        "fft" => time_fft(input_data),
        _ => 0,
    }
}

/// Time-only ufunc: creates array once, times only the computation.
fn time_unary_ufunc<F>(input_data: &[f64], func: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, Ix1>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");
    let start = Instant::now();
    let _ = func(&arr);
    start.elapsed().as_nanos() as u64
}

/// Time-only reduction.
fn time_reduction<F>(input_data: &[f64], func: F) -> u64
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, IxDyn>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");
    let start = Instant::now();
    let _ = func(&arr);
    start.elapsed().as_nanos() as u64
}

/// Time-only matmul.
fn time_matmul(input_data: &[f64], size_str: &str) -> u64 {
    let parts: Vec<&str> = size_str.split('x').collect();
    if parts.len() < 2 {
        return 0;
    }
    let rows: usize = parts[0].parse().unwrap_or(0);
    let cols: usize = parts[1].parse().unwrap_or(0);
    let n = rows * cols;
    if input_data.len() < 2 * n {
        return 0;
    }
    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[rows, cols]), input_data[..n].to_vec())
        .expect("Failed to create matrix A");
    let b = Array::<f64, IxDyn>::from_vec(
        IxDyn::new(&[cols, rows]),
        input_data[n..2 * n].to_vec(),
    )
    .expect("Failed to create matrix B");
    let start = Instant::now();
    let _ = ferrum_linalg::matmul(&a, &b);
    start.elapsed().as_nanos() as u64
}

/// Time-only FFT.
fn time_fft(input_data: &[f64]) -> u64 {
    let complex_data: Vec<Complex<f64>> = input_data
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    let dim = Ix1::new([complex_data.len()]);
    let arr = Array::<Complex<f64>, Ix1>::from_vec(dim, complex_data)
        .expect("Failed to create complex array");
    let start = Instant::now();
    let _ = ferrum_fft::fft(&arr, None, None, ferrum_fft::FftNorm::Backward);
    start.elapsed().as_nanos() as u64
}

// ---------------------------------------------------------------------------
// Single-mode runners (for accuracy / statistical equivalence tests)
// ---------------------------------------------------------------------------

/// Result structure for JSON output.
#[derive(serde::Serialize)]
struct BenchResult {
    function: String,
    shape: Vec<usize>,
    data: Vec<f64>,
    dtype: String,
}

/// Result structure for complex FFT output.
#[derive(serde::Serialize)]
struct BenchResultComplex {
    function: String,
    shape: Vec<usize>,
    data_real: Vec<f64>,
    data_imag: Vec<f64>,
    dtype: String,
}

/// Run a unary ufunc (sin, cos, exp, etc.) on a 1D f64 array.
fn run_unary_ufunc<F>(input_data: &[f64], _size_str: &str, func: F) -> (serde_json::Value, u64)
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, Ix1>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");

    let start = Instant::now();
    let result = func(&arr).expect("Ufunc failed");
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    let json = serde_json::to_value(BenchResult {
        function: "ufunc".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap();
    (json, elapsed_ns)
}

/// Run a stats reduction on a 1D f64 array.
fn run_reduction<F>(input_data: &[f64], _size_str: &str, func: F) -> (serde_json::Value, u64)
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, IxDyn>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");

    let start = Instant::now();
    let result = func(&arr).expect("Reduction failed");
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    let json = serde_json::to_value(BenchResult {
        function: "reduction".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap();
    (json, elapsed_ns)
}

/// Run matmul on a square matrix.
fn run_matmul(input_data: &[f64], size_str: &str) -> (serde_json::Value, u64) {
    let parts: Vec<&str> = size_str.split('x').collect();
    let rows: usize = parts[0].parse().expect("Invalid rows in size");
    let cols: usize = parts[1].parse().expect("Invalid cols in size");

    let n = rows * cols;
    if input_data.len() < 2 * n {
        eprintln!(
            "matmul: need {} elements for two {}x{} matrices, got {}",
            2 * n,
            rows,
            cols,
            input_data.len()
        );
        std::process::exit(1);
    }

    let a_data = input_data[..n].to_vec();
    let b_data = input_data[n..2 * n].to_vec();

    let a = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[rows, cols]), a_data)
        .expect("Failed to create matrix A");
    let b = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[cols, rows]), b_data)
        .expect("Failed to create matrix B");

    // Warmup: triggers faer/rayon thread pool initialization
    let _ = ferrum_linalg::matmul(&a, &b);

    let start = Instant::now();
    let result = ferrum_linalg::matmul(&a, &b).expect("matmul failed");
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    let json = serde_json::to_value(BenchResult {
        function: "matmul".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap();
    (json, elapsed_ns)
}

/// Run FFT on real input data, returns complex output.
fn run_fft(input_data: &[f64], _size_str: &str) -> (serde_json::Value, u64) {
    let complex_data: Vec<Complex<f64>> = input_data
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    let dim = Ix1::new([complex_data.len()]);
    let arr = Array::<Complex<f64>, Ix1>::from_vec(dim, complex_data)
        .expect("Failed to create complex array");

    // Warmup: triggers FftPlanner::new() and plan caching
    let _ = ferrum_fft::fft(&arr, None, None, ferrum_fft::FftNorm::Backward);

    let start = Instant::now();
    let result =
        ferrum_fft::fft(&arr, None, None, ferrum_fft::FftNorm::Backward).expect("FFT failed");
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let data_real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let data_imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    let shape = result.shape().to_vec();

    let json = serde_json::to_value(BenchResultComplex {
        function: "fft".to_string(),
        shape,
        data_real,
        data_imag,
        dtype: "complex128".to_string(),
    })
    .unwrap();
    (json, elapsed_ns)
}
