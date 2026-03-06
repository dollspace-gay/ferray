//! Benchmark binary for statistical equivalence testing.
//!
//! Accepts a function name and input specification via command-line args,
//! runs the corresponding ferrum operation, and outputs results as JSON
//! to stdout. Input data is read from stdin as a JSON array of f64.
//!
//! Usage:
//!   echo '[1.0, 2.0, 3.0]' | ferrum-bench <function> <size> [extra_args...]
//!
//! Supported functions:
//!   sin, cos, tan, exp, log, log2, log10, sqrt, abs, exp2, expm1, log1p,
//!   arcsin, arccos, arctan, sinh, cosh, tanh,
//!   mean, var, std, sum,
//!   matmul,
//!   fft

use ferrum_core::dimension::{Ix1, IxDyn};
use ferrum_core::Array;
use num_complex::Complex;
use std::io::Read;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: ferrum-bench <function> <size> [extra_args...]");
        eprintln!("Input data is read from stdin as a JSON array of f64.");
        eprintln!("Supported functions: sin, cos, tan, exp, log, log2, log10, sqrt,");
        eprintln!("  abs, exp2, expm1, log1p, arcsin, arccos, arctan, sinh, cosh, tanh,");
        eprintln!("  mean, var, std, sum, matmul, fft");
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

    // Dispatch based on function name, with timing
    let start = Instant::now();
    let result = match func_name.as_str() {
        // ---- Ufunc: unary float ops ----
        "sin" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::sin),
        "cos" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::cos),
        "tan" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::tan),
        "exp" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::exp),
        "log" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::log),
        "log2" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::log2),
        "log10" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::log10),
        "sqrt" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::sqrt),
        "abs" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::absolute),
        "exp2" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::exp2),
        "expm1" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::expm1),
        "log1p" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::log1p),
        "arcsin" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::arcsin),
        "arccos" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::arccos),
        "arctan" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::arctan),
        "sinh" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::sinh),
        "cosh" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::cosh),
        "tanh" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::tanh),
        "cbrt" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::cbrt),
        "ceil" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::ceil),
        "floor" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::floor),
        "rint" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::rint),
        "sign" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::sign),
        "square" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::square),
        "negative" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::negative),
        "reciprocal" => run_unary_ufunc(&input_data, size_str, ferrum_ufunc::reciprocal),

        // ---- Stats: reductions ----
        "mean" => run_reduction(&input_data, size_str, |arr| {
            ferrum_stats::mean(arr, None)
        }),
        "var" => run_reduction(&input_data, size_str, |arr| {
            ferrum_stats::var(arr, None, 0)
        }),
        "std" => run_reduction(&input_data, size_str, |arr| {
            ferrum_stats::std_(arr, None, 0)
        }),
        "sum" => run_reduction(&input_data, size_str, |arr| {
            ferrum_stats::sum(arr, None)
        }),

        // ---- Linalg: matmul ----
        "matmul" => run_matmul(&input_data, size_str),

        // ---- FFT ----
        "fft" => run_fft(&input_data, size_str),

        other => {
            eprintln!("Unknown function: {other}");
            std::process::exit(1);
        }
    };

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    // Inject elapsed_ns into the JSON output
    let mut result_map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_value(result).expect("Expected JSON object");
    result_map.insert("elapsed_ns".to_string(), serde_json::Value::from(elapsed_ns));

    // Output as JSON
    println!("{}", serde_json::to_string(&result_map).expect("Failed to serialize result"));
}

/// Result structure for JSON output.
#[derive(serde::Serialize)]
struct BenchResult {
    /// The function that was run.
    function: String,
    /// Shape of the output array.
    shape: Vec<usize>,
    /// Output data as flat f64 array.
    data: Vec<f64>,
    /// Data type description.
    dtype: String,
}

/// Result structure for complex FFT output.
#[derive(serde::Serialize)]
struct BenchResultComplex {
    function: String,
    shape: Vec<usize>,
    /// Interleaved real/imag pairs: [re0, im0, re1, im1, ...]
    data_real: Vec<f64>,
    data_imag: Vec<f64>,
    dtype: String,
}

/// Run a unary ufunc (sin, cos, exp, etc.) on a 1D f64 array.
fn run_unary_ufunc<F>(input_data: &[f64], _size_str: &str, func: F) -> serde_json::Value
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, Ix1>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");

    let result = func(&arr).expect("Ufunc failed");

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    serde_json::to_value(BenchResult {
        function: "ufunc".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap()
}

/// Run a stats reduction (mean, var, std, sum) on a 1D f64 array.
fn run_reduction<F>(input_data: &[f64], _size_str: &str, func: F) -> serde_json::Value
where
    F: Fn(&Array<f64, Ix1>) -> ferrum_core::FerrumResult<Array<f64, IxDyn>>,
{
    let dim = Ix1::new([input_data.len()]);
    let arr =
        Array::<f64, Ix1>::from_vec(dim, input_data.to_vec()).expect("Failed to create array");

    let result = func(&arr).expect("Reduction failed");

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    serde_json::to_value(BenchResult {
        function: "reduction".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap()
}

/// Run matmul on a square matrix. The input is a flat array interpreted as an NxN matrix.
/// size_str should be something like "10x10" or "100x100".
fn run_matmul(input_data: &[f64], size_str: &str) -> serde_json::Value {
    let parts: Vec<&str> = size_str.split('x').collect();
    let rows: usize = parts[0].parse().expect("Invalid rows in size");
    let cols: usize = parts[1].parse().expect("Invalid cols in size");

    // Split input data into two matrices: A and B, each rows x cols
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

    let result = ferrum_linalg::matmul(&a, &b).expect("matmul failed");

    let out_data: Vec<f64> = result.iter().copied().collect();
    let shape = result.shape().to_vec();

    serde_json::to_value(BenchResult {
        function: "matmul".to_string(),
        shape,
        data: out_data,
        dtype: "f64".to_string(),
    })
    .unwrap()
}

/// Run FFT on real input data, returns complex output.
fn run_fft(input_data: &[f64], _size_str: &str) -> serde_json::Value {
    // Convert real input to complex
    let complex_data: Vec<Complex<f64>> = input_data
        .iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();

    let dim = Ix1::new([complex_data.len()]);
    let arr = Array::<Complex<f64>, Ix1>::from_vec(dim, complex_data)
        .expect("Failed to create complex array");

    let result =
        ferrum_fft::fft(&arr, None, None, ferrum_fft::FftNorm::Backward).expect("FFT failed");

    let data_real: Vec<f64> = result.iter().map(|c| c.re).collect();
    let data_imag: Vec<f64> = result.iter().map(|c| c.im).collect();
    let shape = result.shape().to_vec();

    serde_json::to_value(BenchResultComplex {
        function: "fft".to_string(),
        shape,
        data_real,
        data_imag,
        dtype: "complex128".to_string(),
    })
    .unwrap()
}
