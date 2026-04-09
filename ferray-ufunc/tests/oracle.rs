//! Oracle tests: validate ferray-ufunc against NumPy fixture outputs.
//!
//! Each test loads a JSON fixture from `fixtures/ufunc/`, constructs input
//! arrays, calls the corresponding ferray function, and compares the output
//! to NumPy's result within the fixture's ULP tolerance.
//!
//! The macros below take a bare function `path` and pin its concrete type
//! via an internal `let` binding to a `fn` pointer. This monomorphizes
//! the generic ufunc to `(Array<f64, IxDyn>, ...) -> ...` so callers can
//! write `unary_oracle!(oracle_sin, "sin.json", ferray_ufunc::sin)` without
//! a closure or turbofish.

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_core::error::FerrayResult;
use ferray_test_oracle::*;

fn ufunc_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("ufunc").join(name)
}

// ---------------------------------------------------------------------------
// Macros to reduce boilerplate
//
// Each macro takes a bare function path and pins the generic to a concrete
// `fn` pointer with the f64/IxDyn instantiation expected by the runner.
// ---------------------------------------------------------------------------

macro_rules! unary_oracle {
    ($test_name:ident, $file:expr, $func:path) => {
        #[test]
        fn $test_name() {
            let func: fn(&Array<f64, IxDyn>) -> FerrayResult<Array<f64, IxDyn>> = $func;
            run_unary_f64_oracle(&ufunc_path($file), func);
        }
    };
}

macro_rules! binary_oracle {
    ($test_name:ident, $file:expr, $func:path) => {
        #[test]
        fn $test_name() {
            let func: fn(
                &Array<f64, IxDyn>,
                &Array<f64, IxDyn>,
            ) -> FerrayResult<Array<f64, IxDyn>> = $func;
            run_binary_f64_oracle(&ufunc_path($file), func);
        }
    };
}

macro_rules! bool_oracle {
    ($test_name:ident, $file:expr, $func:path) => {
        #[test]
        fn $test_name() {
            let func: fn(&Array<f64, IxDyn>) -> FerrayResult<Array<bool, IxDyn>> = $func;
            run_unary_bool_oracle(&ufunc_path($file), func);
        }
    };
}

// ---------------------------------------------------------------------------
// Trigonometric (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_sin, "sin.json", ferray_ufunc::sin);
unary_oracle!(oracle_cos, "cos.json", ferray_ufunc::cos);
unary_oracle!(oracle_tan, "tan.json", ferray_ufunc::tan);
unary_oracle!(oracle_arcsin, "arcsin.json", ferray_ufunc::arcsin);
unary_oracle!(oracle_arccos, "arccos.json", ferray_ufunc::arccos);
unary_oracle!(oracle_arctan, "arctan.json", ferray_ufunc::arctan);
unary_oracle!(oracle_sinh, "sinh.json", ferray_ufunc::sinh);
unary_oracle!(oracle_cosh, "cosh.json", ferray_ufunc::cosh);
unary_oracle!(oracle_tanh, "tanh.json", ferray_ufunc::tanh);
unary_oracle!(oracle_arcsinh, "arcsinh.json", ferray_ufunc::arcsinh);
unary_oracle!(oracle_arccosh, "arccosh.json", ferray_ufunc::arccosh);
unary_oracle!(oracle_arctanh, "arctanh.json", ferray_ufunc::arctanh);

// ---------------------------------------------------------------------------
// Exponential / logarithmic (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_exp, "exp.json", ferray_ufunc::exp);
unary_oracle!(oracle_exp2, "exp2.json", ferray_ufunc::exp2);
unary_oracle!(oracle_expm1, "expm1.json", ferray_ufunc::expm1);
unary_oracle!(oracle_log, "log.json", ferray_ufunc::log);
unary_oracle!(oracle_log2, "log2.json", ferray_ufunc::log2);
unary_oracle!(oracle_log10, "log10.json", ferray_ufunc::log10);
unary_oracle!(oracle_log1p, "log1p.json", ferray_ufunc::log1p);

// ---------------------------------------------------------------------------
// Arithmetic (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_absolute, "absolute.json", ferray_ufunc::absolute);
unary_oracle!(oracle_negative, "negative.json", ferray_ufunc::negative);
unary_oracle!(oracle_sqrt, "sqrt.json", ferray_ufunc::sqrt);
unary_oracle!(oracle_cbrt, "cbrt.json", ferray_ufunc::cbrt);
unary_oracle!(oracle_square, "square.json", ferray_ufunc::square);
unary_oracle!(
    oracle_reciprocal,
    "reciprocal.json",
    ferray_ufunc::reciprocal
);
unary_oracle!(oracle_sinc, "sinc.json", ferray_ufunc::sinc);

// ---------------------------------------------------------------------------
// Rounding (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_ceil, "ceil.json", ferray_ufunc::ceil);
unary_oracle!(oracle_floor, "floor.json", ferray_ufunc::floor);
unary_oracle!(oracle_trunc, "trunc.json", ferray_ufunc::trunc);
unary_oracle!(oracle_rint, "rint.json", ferray_ufunc::rint);
unary_oracle!(oracle_fix, "fix.json", ferray_ufunc::fix);

// ---------------------------------------------------------------------------
// Arithmetic (binary)
// ---------------------------------------------------------------------------

binary_oracle!(oracle_add, "add.json", ferray_ufunc::add);
binary_oracle!(oracle_subtract, "subtract.json", ferray_ufunc::subtract);
binary_oracle!(oracle_multiply, "multiply.json", ferray_ufunc::multiply);
binary_oracle!(oracle_divide, "divide.json", ferray_ufunc::divide);
binary_oracle!(oracle_power, "power.json", ferray_ufunc::power);
binary_oracle!(oracle_remainder, "remainder.json", ferray_ufunc::remainder);
binary_oracle!(oracle_mod, "mod.json", ferray_ufunc::mod_);
binary_oracle!(oracle_maximum, "maximum.json", ferray_ufunc::maximum);
binary_oracle!(oracle_minimum, "minimum.json", ferray_ufunc::minimum);
binary_oracle!(oracle_fmax, "fmax.json", ferray_ufunc::fmax);
binary_oracle!(oracle_fmin, "fmin.json", ferray_ufunc::fmin);
binary_oracle!(oracle_heaviside, "heaviside.json", ferray_ufunc::heaviside);
binary_oracle!(oracle_arctan2, "arctan2.json", ferray_ufunc::arctan2);

// ---------------------------------------------------------------------------
// Boolean predicates
// ---------------------------------------------------------------------------

bool_oracle!(oracle_isnan, "isnan.json", ferray_ufunc::isnan);
bool_oracle!(oracle_isinf, "isinf.json", ferray_ufunc::isinf);
bool_oracle!(oracle_isfinite, "isfinite.json", ferray_ufunc::isfinite);

// ---------------------------------------------------------------------------
// Special-parameter functions (hand-written)
// ---------------------------------------------------------------------------

#[test]
fn oracle_round() {
    // ferray_ufunc::round takes only the array (banker's rounding, no decimals param).
    // Only test cases with decimals=0 or no decimals param.
    let suite = load_fixture(&ufunc_path("round.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let decimals = case
            .inputs
            .get("decimals")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if decimals != 0 {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_ufunc::round(&arr).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_clip() {
    let suite = load_fixture(&ufunc_path("clip.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let a_min = parse_f64_value(&case.inputs["a_min"]);
        let a_max = parse_f64_value(&case.inputs["a_max"]);
        let result = ferray_ufunc::clip(&arr, a_min, a_max).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_nan_to_num() {
    let suite = load_fixture(&ufunc_path("nan_to_num.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let nan = case
            .inputs
            .get("nan")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let posinf = case
            .inputs
            .get("posinf")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::MAX);
        let neginf = case
            .inputs
            .get("neginf")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::MIN);
        let result = ferray_ufunc::nan_to_num(&arr, Some(nan), Some(posinf), Some(neginf)).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}
