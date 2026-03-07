//! Verify that SIMD and scalar code paths produce bit-identical results.
//!
//! This integration test exercises two layers:
//!
//! 1. **High-level ufunc functions** (`ferray_ufunc::sin`, `ferray_ufunc::exp`, etc.)
//!    which are generic over `T: Element + Float` and `D: Dimension`.
//!    These currently use `unary_float_op` / `binary_float_op` helpers that
//!    iterate directly — the `FERRAY_FORCE_SCALAR` env var has no effect on them
//!    today, but this test locks in the contract that results are identical
//!    regardless of the env var, future-proofing against SIMD kernels being
//!    wired into the high-level path later.
//!
//! 2. **Low-level dispatch functions** (`dispatch_unary_f64`, `dispatch_binary_f64`)
//!    which **do** branch on `FERRAY_FORCE_SCALAR`. We verify bit-identity at
//!    this layer too.
//!
//! For each operation, we run it twice — once with the env var unset (SIMD path)
//! and once with `FERRAY_FORCE_SCALAR=1` (scalar path) — then compare every
//! output element at the bit level.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::error::FerrayError;

// ---------------------------------------------------------------------------
// Type aliases to keep clippy happy (type_complexity)
// ---------------------------------------------------------------------------

type UnaryFn = fn(&Array<f64, Ix1>) -> Result<Array<f64, Ix1>, FerrayError>;
type BinaryFn = fn(&Array<f64, Ix1>, &Array<f64, Ix1>) -> Result<Array<f64, Ix1>, FerrayError>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assert two f64 slices are bit-identical.
fn assert_bit_identical_f64(a: &[f64], b: &[f64], op_name: &str) {
    assert_eq!(a.len(), b.len(), "{op_name}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{op_name}[{i}]: SIMD bits={:#018x}, scalar bits={:#018x} (SIMD={x}, scalar={y})",
            x.to_bits(),
            y.to_bits(),
        );
    }
}

/// Build a 1-D f64 array from a slice.
fn arr(data: &[f64]) -> Array<f64, Ix1> {
    Array::from_vec(Ix1::new([data.len()]), data.to_vec()).unwrap()
}

/// Run a unary ufunc under both SIMD and scalar env, return both result vecs.
///
/// SAFETY: We manipulate environment variables which is inherently racy in a
/// multi-threaded program. Cargo runs each integration test file in its own
/// process, and we rely on `--test-threads=1` or the fact that env-var checks
/// in the dispatch layer are non-critical (the current high-level ufuncs don't
/// branch on the env var at all). The low-level dispatch tests below are
/// sequential within each test function.
fn run_unary_both(input: &[f64], f: UnaryFn) -> (Vec<f64>, Vec<f64>) {
    let a = arr(input);

    // SIMD path
    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    let simd_result = f(&a).unwrap();
    let simd_vals: Vec<f64> = simd_result.iter().copied().collect();

    // Scalar path
    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    let scalar_result = f(&a).unwrap();
    let scalar_vals: Vec<f64> = scalar_result.iter().copied().collect();

    // Clean up
    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };

    (simd_vals, scalar_vals)
}

/// Run a binary ufunc under both SIMD and scalar env.
fn run_binary_both(a_data: &[f64], b_data: &[f64], f: BinaryFn) -> (Vec<f64>, Vec<f64>) {
    let a = arr(a_data);
    let b = arr(b_data);

    // SIMD path
    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    let simd_result = f(&a, &b).unwrap();
    let simd_vals: Vec<f64> = simd_result.iter().copied().collect();

    // Scalar path
    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    let scalar_result = f(&a, &b).unwrap();
    let scalar_vals: Vec<f64> = scalar_result.iter().copied().collect();

    // Clean up
    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };

    (simd_vals, scalar_vals)
}

// ---------------------------------------------------------------------------
// Standard input vectors used across tests
// ---------------------------------------------------------------------------

/// 100 values in [0.0, 10.0) for general unary tests.
fn general_input() -> Vec<f64> {
    (0..100).map(|i| i as f64 * 0.1).collect()
}

/// 100 values in [-1.0, 1.0) for domain-restricted functions (arcsin, arctanh).
fn unit_input() -> Vec<f64> {
    (0..100).map(|i| (i as f64 - 50.0) * 0.019).collect()
}

/// 100 positive values in (0.5, 50.5) for log-domain functions.
fn positive_input() -> Vec<f64> {
    (1..101).map(|i| i as f64 * 0.5).collect()
}

/// 100 values in [-2.5, 2.5) for exp-domain (avoid overflow).
fn exp_input() -> Vec<f64> {
    (0..100).map(|i| i as f64 * 0.05 - 2.5).collect()
}

/// 100 values >= 1.0 for arccosh domain.
fn arccosh_input() -> Vec<f64> {
    (0..100).map(|i| 1.0 + i as f64 * 0.1).collect()
}

// ===========================================================================
// High-level ufunc identity tests (unary)
// ===========================================================================

#[test]
fn identity_sin() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::sin);
    assert_bit_identical_f64(&s, &c, "sin");
}

#[test]
fn identity_cos() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::cos);
    assert_bit_identical_f64(&s, &c, "cos");
}

#[test]
fn identity_tan() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::tan);
    assert_bit_identical_f64(&s, &c, "tan");
}

#[test]
fn identity_arcsin() {
    let (s, c) = run_unary_both(&unit_input(), ferray_ufunc::arcsin);
    assert_bit_identical_f64(&s, &c, "arcsin");
}

#[test]
fn identity_arccos() {
    let (s, c) = run_unary_both(&unit_input(), ferray_ufunc::arccos);
    assert_bit_identical_f64(&s, &c, "arccos");
}

#[test]
fn identity_arctan() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::arctan);
    assert_bit_identical_f64(&s, &c, "arctan");
}

#[test]
fn identity_sinh() {
    let (s, c) = run_unary_both(&exp_input(), ferray_ufunc::sinh);
    assert_bit_identical_f64(&s, &c, "sinh");
}

#[test]
fn identity_cosh() {
    let (s, c) = run_unary_both(&exp_input(), ferray_ufunc::cosh);
    assert_bit_identical_f64(&s, &c, "cosh");
}

#[test]
fn identity_tanh() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::tanh);
    assert_bit_identical_f64(&s, &c, "tanh");
}

#[test]
fn identity_arcsinh() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::arcsinh);
    assert_bit_identical_f64(&s, &c, "arcsinh");
}

#[test]
fn identity_arccosh() {
    let (s, c) = run_unary_both(&arccosh_input(), ferray_ufunc::arccosh);
    assert_bit_identical_f64(&s, &c, "arccosh");
}

#[test]
fn identity_arctanh() {
    let (s, c) = run_unary_both(&unit_input(), ferray_ufunc::arctanh);
    assert_bit_identical_f64(&s, &c, "arctanh");
}

#[test]
fn identity_exp() {
    let (s, c) = run_unary_both(&exp_input(), ferray_ufunc::exp);
    assert_bit_identical_f64(&s, &c, "exp");
}

#[test]
fn identity_exp2() {
    let (s, c) = run_unary_both(&exp_input(), ferray_ufunc::exp2);
    assert_bit_identical_f64(&s, &c, "exp2");
}

#[test]
fn identity_expm1() {
    let (s, c) = run_unary_both(&exp_input(), ferray_ufunc::expm1);
    assert_bit_identical_f64(&s, &c, "expm1");
}

#[test]
fn identity_log() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::log);
    assert_bit_identical_f64(&s, &c, "log");
}

#[test]
fn identity_log2() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::log2);
    assert_bit_identical_f64(&s, &c, "log2");
}

#[test]
fn identity_log10() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::log10);
    assert_bit_identical_f64(&s, &c, "log10");
}

#[test]
fn identity_log1p() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::log1p);
    assert_bit_identical_f64(&s, &c, "log1p");
}

#[test]
fn identity_sqrt() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::sqrt);
    assert_bit_identical_f64(&s, &c, "sqrt");
}

#[test]
fn identity_cbrt() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::cbrt);
    assert_bit_identical_f64(&s, &c, "cbrt");
}

#[test]
fn identity_square() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::square);
    assert_bit_identical_f64(&s, &c, "square");
}

#[test]
fn identity_absolute() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.3).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::absolute);
    assert_bit_identical_f64(&s, &c, "absolute");
}

#[test]
fn identity_negative() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::negative);
    assert_bit_identical_f64(&s, &c, "negative");
}

#[test]
fn identity_reciprocal() {
    let (s, c) = run_unary_both(&positive_input(), ferray_ufunc::reciprocal);
    assert_bit_identical_f64(&s, &c, "reciprocal");
}

#[test]
fn identity_sign() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::sign);
    assert_bit_identical_f64(&s, &c, "sign");
}

#[test]
fn identity_floor() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.3 + 0.1).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::floor);
    assert_bit_identical_f64(&s, &c, "floor");
}

#[test]
fn identity_ceil() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.3 + 0.1).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::ceil);
    assert_bit_identical_f64(&s, &c, "ceil");
}

#[test]
fn identity_trunc() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.3 + 0.1).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::trunc);
    assert_bit_identical_f64(&s, &c, "trunc");
}

#[test]
fn identity_round() {
    let input: Vec<f64> = (-50..50).map(|i| i as f64 * 0.3 + 0.1).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::round);
    assert_bit_identical_f64(&s, &c, "round");
}

#[test]
fn identity_degrees() {
    let (s, c) = run_unary_both(&general_input(), ferray_ufunc::degrees);
    assert_bit_identical_f64(&s, &c, "degrees");
}

#[test]
fn identity_radians() {
    let input: Vec<f64> = (0..100).map(|i| i as f64 * 3.6).collect();
    let (s, c) = run_unary_both(&input, ferray_ufunc::radians);
    assert_bit_identical_f64(&s, &c, "radians");
}

// ===========================================================================
// High-level ufunc identity tests (binary)
// ===========================================================================

#[test]
fn identity_add() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.3).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::add);
    assert_bit_identical_f64(&s, &c, "add");
}

#[test]
fn identity_subtract() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.3).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::subtract);
    assert_bit_identical_f64(&s, &c, "subtract");
}

#[test]
fn identity_multiply() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.3).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::multiply);
    assert_bit_identical_f64(&s, &c, "multiply");
}

#[test]
fn identity_divide() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.7 + 0.01).collect();
    let b: Vec<f64> = (1..101).map(|i| i as f64 * 0.3).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::divide);
    assert_bit_identical_f64(&s, &c, "divide");
}

#[test]
fn identity_power() {
    let a: Vec<f64> = (1..101).map(|i| i as f64 * 0.1).collect();
    let b: Vec<f64> = (0..100).map(|i| i as f64 * 0.02 + 0.5).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::power);
    assert_bit_identical_f64(&s, &c, "power");
}

#[test]
fn identity_arctan2() {
    let y: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.1).collect();
    let x: Vec<f64> = (0..100).map(|i| (i as f64 - 30.0) * 0.1).collect();
    let (s, c) = run_binary_both(&y, &x, ferray_ufunc::arctan2);
    assert_bit_identical_f64(&s, &c, "arctan2");
}

#[test]
fn identity_hypot() {
    let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.3).collect();
    let b: Vec<f64> = (0..100).map(|i| i as f64 * 0.4).collect();
    let (s, c) = run_binary_both(&a, &b, ferray_ufunc::hypot);
    assert_bit_identical_f64(&s, &c, "hypot");
}

// ===========================================================================
// Low-level dispatch identity tests
// ===========================================================================
//
// These test the `dispatch_unary_f64` and `dispatch_binary_f64` functions
// directly, which ARE the functions that branch on FERRAY_FORCE_SCALAR.

#[test]
fn dispatch_unary_f64_sqrt_identity() {
    let input: Vec<f64> = (1..101).map(|i| i as f64).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    // SIMD path
    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::sqrt);

    // Scalar path
    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::sqrt);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(sqrt)");
}

#[test]
fn dispatch_unary_f64_sin_identity() {
    let input: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::sin);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::sin);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(sin)");
}

#[test]
fn dispatch_unary_f64_exp_identity() {
    let input: Vec<f64> = (0..200).map(|i| i as f64 * 0.03 - 3.0).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::exp);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::exp);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(exp)");
}

#[test]
fn dispatch_unary_f64_ln_identity() {
    let input: Vec<f64> = (1..201).map(|i| i as f64 * 0.5).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::ln);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::ln);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(ln)");
}

#[test]
fn dispatch_unary_f64_cos_identity() {
    let input: Vec<f64> = (0..200).map(|i| i as f64 * 0.05).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::cos);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::cos);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(cos)");
}

#[test]
fn dispatch_unary_f64_floor_identity() {
    let input: Vec<f64> = (-100..100).map(|i| i as f64 * 0.37 + 0.1).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::floor);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::floor);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(floor)");
}

#[test]
fn dispatch_unary_f64_ceil_identity() {
    let input: Vec<f64> = (-100..100).map(|i| i as f64 * 0.37 + 0.1).collect();
    let mut simd_out = vec![0.0f64; input.len()];
    let mut scalar_out = vec![0.0f64; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut simd_out, f64::ceil);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f64(&input, &mut scalar_out, f64::ceil);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_unary_f64(ceil)");
}

#[test]
fn dispatch_binary_f64_add_identity() {
    let a: Vec<f64> = (0..200).map(|i| i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..200).map(|i| (200 - i) as f64 * 0.3).collect();
    let mut simd_out = vec![0.0f64; a.len()];
    let mut scalar_out = vec![0.0f64; a.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut simd_out, |x, y| x + y);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut scalar_out, |x, y| x + y);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_binary_f64(add)");
}

#[test]
fn dispatch_binary_f64_mul_identity() {
    let a: Vec<f64> = (0..200).map(|i| i as f64 * 0.7).collect();
    let b: Vec<f64> = (0..200).map(|i| (200 - i) as f64 * 0.3).collect();
    let mut simd_out = vec![0.0f64; a.len()];
    let mut scalar_out = vec![0.0f64; a.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut simd_out, |x, y| x * y);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut scalar_out, |x, y| x * y);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_binary_f64(mul)");
}

#[test]
fn dispatch_binary_f64_div_identity() {
    let a: Vec<f64> = (0..200).map(|i| i as f64 * 0.7 + 0.01).collect();
    let b: Vec<f64> = (1..201).map(|i| i as f64 * 0.3).collect();
    let mut simd_out = vec![0.0f64; a.len()];
    let mut scalar_out = vec![0.0f64; a.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut simd_out, |x, y| x / y);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_binary_f64(&a, &b, &mut scalar_out, |x, y| x / y);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f64(&simd_out, &scalar_out, "dispatch_binary_f64(div)");
}

// ===========================================================================
// Low-level dispatch f32 identity tests
// ===========================================================================

/// Assert two f32 slices are bit-identical.
fn assert_bit_identical_f32(a: &[f32], b: &[f32], op_name: &str) {
    assert_eq!(a.len(), b.len(), "{op_name}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{op_name}[{i}]: SIMD bits={:#010x}, scalar bits={:#010x} (SIMD={x}, scalar={y})",
            x.to_bits(),
            y.to_bits(),
        );
    }
}

#[test]
fn dispatch_unary_f32_sqrt_identity() {
    let input: Vec<f32> = (1..201).map(|i| i as f32).collect();
    let mut simd_out = vec![0.0f32; input.len()];
    let mut scalar_out = vec![0.0f32; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f32(&input, &mut simd_out, f32::sqrt);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f32(&input, &mut scalar_out, f32::sqrt);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f32(&simd_out, &scalar_out, "dispatch_unary_f32(sqrt)");
}

#[test]
fn dispatch_unary_f32_sin_identity() {
    let input: Vec<f32> = (0..200).map(|i| i as f32 * 0.05).collect();
    let mut simd_out = vec![0.0f32; input.len()];
    let mut scalar_out = vec![0.0f32; input.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_unary_f32(&input, &mut simd_out, f32::sin);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_unary_f32(&input, &mut scalar_out, f32::sin);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f32(&simd_out, &scalar_out, "dispatch_unary_f32(sin)");
}

#[test]
fn dispatch_binary_f32_add_identity() {
    let a: Vec<f32> = (0..200).map(|i| i as f32 * 0.7).collect();
    let b: Vec<f32> = (0..200).map(|i| (200 - i) as f32 * 0.3).collect();
    let mut simd_out = vec![0.0f32; a.len()];
    let mut scalar_out = vec![0.0f32; a.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_binary_f32(&a, &b, &mut simd_out, |x, y| x + y);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_binary_f32(&a, &b, &mut scalar_out, |x, y| x + y);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f32(&simd_out, &scalar_out, "dispatch_binary_f32(add)");
}

#[test]
fn dispatch_binary_f32_mul_identity() {
    let a: Vec<f32> = (0..200).map(|i| i as f32 * 0.7).collect();
    let b: Vec<f32> = (0..200).map(|i| (200 - i) as f32 * 0.3).collect();
    let mut simd_out = vec![0.0f32; a.len()];
    let mut scalar_out = vec![0.0f32; a.len()];

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    ferray_ufunc::dispatch::dispatch_binary_f32(&a, &b, &mut simd_out, |x, y| x * y);

    unsafe { std::env::set_var("FERRAY_FORCE_SCALAR", "1") };
    ferray_ufunc::dispatch::dispatch_binary_f32(&a, &b, &mut scalar_out, |x, y| x * y);

    unsafe { std::env::remove_var("FERRAY_FORCE_SCALAR") };
    assert_bit_identical_f32(&simd_out, &scalar_out, "dispatch_binary_f32(mul)");
}
