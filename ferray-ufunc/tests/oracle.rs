//! Oracle tests: validate ferray-ufunc against NumPy fixture outputs.
//!
//! Each test loads a JSON fixture from `fixtures/ufunc/`, constructs input
//! arrays, calls the corresponding ferray function, and compares the output
//! to NumPy's result within the fixture's ULP tolerance.
//!
//! The closures in the macro invocations below (`|x| ferray_ufunc::sin(x)`)
//! are intentionally not replaced with bare function paths — they provide
//! the type-inference context needed by the generic ufunc signatures.

#![allow(clippy::redundant_closure)]

use ferray_test_oracle::*;

fn ufunc_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("ufunc").join(name)
}

// ---------------------------------------------------------------------------
// Macros to reduce boilerplate
// ---------------------------------------------------------------------------

macro_rules! unary_oracle {
    ($test_name:ident, $file:expr, $func:expr) => {
        #[test]
        fn $test_name() {
            run_unary_f64_oracle(&ufunc_path($file), $func);
        }
    };
}

macro_rules! binary_oracle {
    ($test_name:ident, $file:expr, $func:expr) => {
        #[test]
        fn $test_name() {
            run_binary_f64_oracle(&ufunc_path($file), $func);
        }
    };
}

macro_rules! bool_oracle {
    ($test_name:ident, $file:expr, $func:expr) => {
        #[test]
        fn $test_name() {
            run_unary_bool_oracle(&ufunc_path($file), $func);
        }
    };
}

// ---------------------------------------------------------------------------
// Trigonometric (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_sin, "sin.json", |x| ferray_ufunc::sin(x));
unary_oracle!(oracle_cos, "cos.json", |x| ferray_ufunc::cos(x));
unary_oracle!(oracle_tan, "tan.json", |x| ferray_ufunc::tan(x));
unary_oracle!(oracle_arcsin, "arcsin.json", |x| ferray_ufunc::arcsin(x));
unary_oracle!(oracle_arccos, "arccos.json", |x| ferray_ufunc::arccos(x));
unary_oracle!(oracle_arctan, "arctan.json", |x| ferray_ufunc::arctan(x));
unary_oracle!(oracle_sinh, "sinh.json", |x| ferray_ufunc::sinh(x));
unary_oracle!(oracle_cosh, "cosh.json", |x| ferray_ufunc::cosh(x));
unary_oracle!(oracle_tanh, "tanh.json", |x| ferray_ufunc::tanh(x));
unary_oracle!(oracle_arcsinh, "arcsinh.json", |x| ferray_ufunc::arcsinh(x));
unary_oracle!(oracle_arccosh, "arccosh.json", |x| ferray_ufunc::arccosh(x));
unary_oracle!(oracle_arctanh, "arctanh.json", |x| ferray_ufunc::arctanh(x));

// ---------------------------------------------------------------------------
// Exponential / logarithmic (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_exp, "exp.json", |x| ferray_ufunc::exp(x));
unary_oracle!(oracle_exp2, "exp2.json", |x| ferray_ufunc::exp2(x));
unary_oracle!(oracle_expm1, "expm1.json", |x| ferray_ufunc::expm1(x));
unary_oracle!(oracle_log, "log.json", |x| ferray_ufunc::log(x));
unary_oracle!(oracle_log2, "log2.json", |x| ferray_ufunc::log2(x));
unary_oracle!(oracle_log10, "log10.json", |x| ferray_ufunc::log10(x));
unary_oracle!(oracle_log1p, "log1p.json", |x| ferray_ufunc::log1p(x));

// ---------------------------------------------------------------------------
// Arithmetic (unary)
// ---------------------------------------------------------------------------

unary_oracle!(
    oracle_absolute,
    "absolute.json",
    |x| ferray_ufunc::absolute(x)
);
unary_oracle!(
    oracle_negative,
    "negative.json",
    |x| ferray_ufunc::negative(x)
);
unary_oracle!(oracle_sqrt, "sqrt.json", |x| ferray_ufunc::sqrt(x));
unary_oracle!(oracle_cbrt, "cbrt.json", |x| ferray_ufunc::cbrt(x));
unary_oracle!(oracle_square, "square.json", |x| ferray_ufunc::square(x));
unary_oracle!(oracle_reciprocal, "reciprocal.json", |x| {
    ferray_ufunc::reciprocal(x)
});
unary_oracle!(oracle_sinc, "sinc.json", |x| ferray_ufunc::sinc(x));

// ---------------------------------------------------------------------------
// Rounding (unary)
// ---------------------------------------------------------------------------

unary_oracle!(oracle_ceil, "ceil.json", |x| ferray_ufunc::ceil(x));
unary_oracle!(oracle_floor, "floor.json", |x| ferray_ufunc::floor(x));
unary_oracle!(oracle_trunc, "trunc.json", |x| ferray_ufunc::trunc(x));
unary_oracle!(oracle_rint, "rint.json", |x| ferray_ufunc::rint(x));
unary_oracle!(oracle_fix, "fix.json", |x| ferray_ufunc::fix(x));

// ---------------------------------------------------------------------------
// Arithmetic (binary)
// ---------------------------------------------------------------------------

binary_oracle!(oracle_add, "add.json", |a, b| ferray_ufunc::add(a, b));
binary_oracle!(oracle_subtract, "subtract.json", |a, b| {
    ferray_ufunc::subtract(a, b)
});
binary_oracle!(oracle_multiply, "multiply.json", |a, b| {
    ferray_ufunc::multiply(a, b)
});
binary_oracle!(oracle_divide, "divide.json", |a, b| ferray_ufunc::divide(
    a, b
));
binary_oracle!(oracle_power, "power.json", |a, b| ferray_ufunc::power(a, b));
binary_oracle!(oracle_remainder, "remainder.json", |a, b| {
    ferray_ufunc::remainder(a, b)
});
binary_oracle!(oracle_mod, "mod.json", |a, b| ferray_ufunc::mod_(a, b));
binary_oracle!(
    oracle_maximum,
    "maximum.json",
    |a, b| ferray_ufunc::maximum(a, b)
);
binary_oracle!(
    oracle_minimum,
    "minimum.json",
    |a, b| ferray_ufunc::minimum(a, b)
);
binary_oracle!(oracle_fmax, "fmax.json", |a, b| ferray_ufunc::fmax(a, b));
binary_oracle!(oracle_fmin, "fmin.json", |a, b| ferray_ufunc::fmin(a, b));
binary_oracle!(oracle_heaviside, "heaviside.json", |a, b| {
    ferray_ufunc::heaviside(a, b)
});
binary_oracle!(
    oracle_arctan2,
    "arctan2.json",
    |a, b| ferray_ufunc::arctan2(a, b)
);

// ---------------------------------------------------------------------------
// Boolean predicates
// ---------------------------------------------------------------------------

bool_oracle!(oracle_isnan, "isnan.json", |x| ferray_ufunc::isnan(x));
bool_oracle!(oracle_isinf, "isinf.json", |x| ferray_ufunc::isinf(x));
bool_oracle!(
    oracle_isfinite,
    "isfinite.json",
    |x| ferray_ufunc::isfinite(x)
);

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
