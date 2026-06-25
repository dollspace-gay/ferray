//! Conformance tests for ferray-ufunc's complex accessors and predicates
//! against NumPy reference outputs.
//!
//! The tests call crate-root re-exports and name the canonical
//! `ferray_ufunc::ops::complex::*` paths so the surface coverage gate can
//! prove both the user-facing and inner items.

use ferray_test_oracle::{
    MIN_ULP_TOLERANCE, assert_complex_slice_ulp, assert_f64_slice_ulp, get_dtype, load_fixture,
    make_complex_array, make_f64_array, parse_bool_data, parse_complex_data, parse_f64_data,
    parse_shape,
};

type ArrBool = ferray_core::Array<bool, ferray_core::dimension::IxDyn>;
type ArrComplex = ferray_core::Array<num_complex::Complex<f64>, ferray_core::dimension::IxDyn>;
type ArrF64 = ferray_core::Array<f64, ferray_core::dimension::IxDyn>;
type ComplexBoolFn = fn(&ArrComplex) -> ferray_core::error::FerrayResult<ArrBool>;
type ComplexToComplexFn = fn(&ArrComplex) -> ferray_core::error::FerrayResult<ArrComplex>;
type ComplexToF64Fn = fn(&ArrComplex) -> ferray_core::error::FerrayResult<ArrF64>;
type F64BoolFn = fn(&ArrF64) -> ferray_core::error::FerrayResult<ArrBool>;
type ObjComplexFn = fn(&ArrComplex) -> bool;
type ObjF64Fn = fn(&ArrF64) -> bool;
type PowerComplexFn =
    fn(&ArrComplex, num_complex::Complex<f64>) -> ferray_core::error::FerrayResult<ArrComplex>;

fn ufunc_fixture(name: &str) -> std::path::PathBuf {
    ferray_test_oracle::fixtures_dir().join("ufunc").join(name)
}

fn assert_shape(name: &str, case_name: &str, actual: &[usize], expected: &[usize]) {
    assert_eq!(actual, expected, "{name} case {case_name} shape mismatch");
}

fn run_complex_to_f64(fixture: &str, name: &str, f: ComplexToF64Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_complex_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_shape(name, &case.name, result.shape(), &expected_shape);
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

fn run_complex_to_complex(fixture: &str, name: &str, f: ComplexToComplexFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_complex_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_shape(name, &case.name, result.shape(), &expected_shape);
        let expected = parse_complex_data(&case.expected["data"]);
        assert_complex_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

fn run_complex_power(fixture: &str, name: &str, f: PowerComplexFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_complex_array(&case.inputs["x"]);
        let exponent = parse_complex_data(&case.inputs["exponent"]["data"])[0];
        let result = f(&arr, exponent).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_shape(name, &case.name, result.shape(), &expected_shape);
        let expected = parse_complex_data(&case.expected["data"]);
        assert_complex_slice_ulp(
            result.as_slice().expect("contiguous"),
            &expected,
            case.tolerance_ulps.max(MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

fn run_complex_to_bool(fixture: &str, name: &str, f: ComplexBoolFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_complex_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_shape(name, &case.name, result.shape(), &expected_shape);
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} data mismatch",
            case.name
        );
    }
}

fn run_f64_to_bool(fixture: &str, name: &str, f: F64BoolFn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let arr = make_f64_array(&case.inputs["x"]);
        let result = f(&arr).unwrap_or_else(|e| panic!("{name} {}: {e}", case.name));
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert_shape(name, &case.name, result.shape(), &expected_shape);
        let expected = parse_bool_data(&case.expected["data"]);
        assert_eq!(
            result.as_slice().expect("contiguous"),
            expected.as_slice(),
            "{name} case {} data mismatch",
            case.name
        );
    }
}

fn run_object_predicate(fixture: &str, name: &str, f_complex: ObjComplexFn, f_f64: ObjF64Fn) {
    let suite = load_fixture(&ufunc_fixture(fixture));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let result = match get_dtype(input) {
            "complex128" => f_complex(&make_complex_array(input)),
            "float64" => f_f64(&make_f64_array(input)),
            dtype => panic!("{name} case {} unsupported dtype {dtype}", case.name),
        };
        let expected_shape = parse_shape(&case.expected["shape"]);
        assert!(expected_shape.is_empty(), "{name} expected scalar bool");
        let expected = parse_bool_data(&case.expected["data"])[0];
        assert_eq!(result, expected, "{name} case {} mismatch", case.name);
    }
}

/// Covers: `ferray_ufunc::real` (re-export of
/// `ferray_ufunc::ops::complex::real`).
#[test]
fn real_matches_numpy() {
    run_complex_to_f64("real.json", "real", ferray_ufunc::real);
}

/// Covers: `ferray_ufunc::imag` (re-export of
/// `ferray_ufunc::ops::complex::imag`).
#[test]
fn imag_matches_numpy() {
    run_complex_to_f64("imag.json", "imag", ferray_ufunc::imag);
}

/// Covers: `ferray_ufunc::angle` (re-export of
/// `ferray_ufunc::ops::complex::angle`).
#[test]
fn angle_matches_numpy() {
    run_complex_to_f64("angle.json", "angle", ferray_ufunc::angle);
}

/// Covers: `ferray_ufunc::abs` (re-export of
/// `ferray_ufunc::ops::complex::abs`).
#[test]
fn abs_complex_matches_numpy() {
    run_complex_to_f64("abs_complex.json", "abs", ferray_ufunc::abs);
}

/// Covers: `ferray_ufunc::conj` (re-export of
/// `ferray_ufunc::ops::complex::conj`).
#[test]
fn conj_matches_numpy() {
    run_complex_to_complex("conj.json", "conj", ferray_ufunc::conj);
}

/// Covers: `ferray_ufunc::conjugate` (re-export of
/// `ferray_ufunc::ops::complex::conjugate`).
#[test]
fn conjugate_matches_numpy() {
    run_complex_to_complex("conjugate.json", "conjugate", ferray_ufunc::conjugate);
}

/// Covers: `ferray_ufunc::iscomplex` (re-export of
/// `ferray_ufunc::ops::complex::iscomplex`).
#[test]
fn iscomplex_matches_numpy() {
    run_complex_to_bool("iscomplex.json", "iscomplex", ferray_ufunc::iscomplex);
}

/// Covers: `ferray_ufunc::isreal` (re-export of
/// `ferray_ufunc::ops::complex::isreal`).
#[test]
fn isreal_matches_numpy() {
    run_complex_to_bool("isreal.json", "isreal", ferray_ufunc::isreal);
}

/// Covers: `ferray_ufunc::iscomplex_real` (re-export of
/// `ferray_ufunc::ops::complex::iscomplex_real`).
#[test]
fn iscomplex_real_matches_numpy() {
    run_f64_to_bool(
        "iscomplex_real.json",
        "iscomplex_real",
        ferray_ufunc::iscomplex_real,
    );
}

/// Covers: `ferray_ufunc::isreal_real` (re-export of
/// `ferray_ufunc::ops::complex::isreal_real`).
#[test]
fn isreal_real_matches_numpy() {
    run_f64_to_bool("isreal_real.json", "isreal_real", ferray_ufunc::isreal_real);
}

/// Covers: `ferray_ufunc::iscomplexobj` (re-export of
/// `ferray_ufunc::ops::complex::iscomplexobj`).
#[test]
fn iscomplexobj_matches_numpy() {
    run_object_predicate(
        "iscomplexobj.json",
        "iscomplexobj",
        ferray_ufunc::iscomplexobj,
        ferray_ufunc::iscomplexobj,
    );
}

/// Covers: `ferray_ufunc::isrealobj` (re-export of
/// `ferray_ufunc::ops::complex::isrealobj`).
#[test]
fn isrealobj_matches_numpy() {
    run_object_predicate(
        "isrealobj.json",
        "isrealobj",
        ferray_ufunc::isrealobj,
        ferray_ufunc::isrealobj,
    );
}

/// Covers: `ferray_ufunc::isscalar` (re-export of
/// `ferray_ufunc::ops::complex::isscalar`).
#[test]
fn isscalar_matches_numpy() {
    run_object_predicate(
        "isscalar.json",
        "isscalar",
        ferray_ufunc::isscalar,
        ferray_ufunc::isscalar,
    );
}

/// Covers: `ferray_ufunc::sin_complex`.
#[test]
fn sin_complex_matches_numpy() {
    run_complex_to_complex("sin_complex.json", "sin_complex", ferray_ufunc::sin_complex);
}

/// Covers: `ferray_ufunc::cos_complex`.
#[test]
fn cos_complex_matches_numpy() {
    run_complex_to_complex("cos_complex.json", "cos_complex", ferray_ufunc::cos_complex);
}

/// Covers: `ferray_ufunc::tan_complex`.
#[test]
fn tan_complex_matches_numpy() {
    run_complex_to_complex("tan_complex.json", "tan_complex", ferray_ufunc::tan_complex);
}

/// Covers: `ferray_ufunc::sinh_complex`.
#[test]
fn sinh_complex_matches_numpy() {
    run_complex_to_complex(
        "sinh_complex.json",
        "sinh_complex",
        ferray_ufunc::sinh_complex,
    );
}

/// Covers: `ferray_ufunc::cosh_complex`.
#[test]
fn cosh_complex_matches_numpy() {
    run_complex_to_complex(
        "cosh_complex.json",
        "cosh_complex",
        ferray_ufunc::cosh_complex,
    );
}

/// Covers: `ferray_ufunc::tanh_complex`.
#[test]
fn tanh_complex_matches_numpy() {
    run_complex_to_complex(
        "tanh_complex.json",
        "tanh_complex",
        ferray_ufunc::tanh_complex,
    );
}

/// Covers: `ferray_ufunc::exp_complex`.
#[test]
fn exp_complex_matches_numpy() {
    run_complex_to_complex("exp_complex.json", "exp_complex", ferray_ufunc::exp_complex);
}

/// Covers: `ferray_ufunc::expm1_complex` (re-export of
/// `ferray_ufunc::ops::complex::expm1_complex`).
#[test]
fn expm1_complex_matches_numpy() {
    run_complex_to_complex(
        "expm1_complex.json",
        "expm1_complex",
        ferray_ufunc::expm1_complex,
    );
}

/// Covers: `ferray_ufunc::ln_complex`.
#[test]
fn ln_complex_matches_numpy() {
    run_complex_to_complex("ln_complex.json", "ln_complex", ferray_ufunc::ln_complex);
}

/// Covers: `ferray_ufunc::log1p_complex` (re-export of
/// `ferray_ufunc::ops::complex::log1p_complex`).
#[test]
fn log1p_complex_matches_numpy() {
    run_complex_to_complex(
        "log1p_complex.json",
        "log1p_complex",
        ferray_ufunc::log1p_complex,
    );
}

/// Covers: `ferray_ufunc::log2_complex` (re-export of
/// `ferray_ufunc::ops::complex::log2_complex`).
#[test]
fn log2_complex_matches_numpy() {
    run_complex_to_complex(
        "log2_complex.json",
        "log2_complex",
        ferray_ufunc::log2_complex,
    );
}

/// Covers: `ferray_ufunc::log10_complex` (re-export of
/// `ferray_ufunc::ops::complex::log10_complex`).
#[test]
fn log10_complex_matches_numpy() {
    run_complex_to_complex(
        "log10_complex.json",
        "log10_complex",
        ferray_ufunc::log10_complex,
    );
}

/// Covers: `ferray_ufunc::sqrt_complex`.
#[test]
fn sqrt_complex_matches_numpy() {
    run_complex_to_complex(
        "sqrt_complex.json",
        "sqrt_complex",
        ferray_ufunc::sqrt_complex,
    );
}

/// Covers: `ferray_ufunc::asin_complex`.
#[test]
fn asin_complex_matches_numpy() {
    run_complex_to_complex(
        "asin_complex.json",
        "asin_complex",
        ferray_ufunc::asin_complex,
    );
}

/// Covers: `ferray_ufunc::acos_complex`.
#[test]
fn acos_complex_matches_numpy() {
    run_complex_to_complex(
        "acos_complex.json",
        "acos_complex",
        ferray_ufunc::acos_complex,
    );
}

/// Covers: `ferray_ufunc::atan_complex`.
#[test]
fn atan_complex_matches_numpy() {
    run_complex_to_complex(
        "atan_complex.json",
        "atan_complex",
        ferray_ufunc::atan_complex,
    );
}

/// Covers: `ferray_ufunc::asinh_complex`.
#[test]
fn asinh_complex_matches_numpy() {
    run_complex_to_complex(
        "asinh_complex.json",
        "asinh_complex",
        ferray_ufunc::asinh_complex,
    );
}

/// Covers: `ferray_ufunc::acosh_complex`.
#[test]
fn acosh_complex_matches_numpy() {
    run_complex_to_complex(
        "acosh_complex.json",
        "acosh_complex",
        ferray_ufunc::acosh_complex,
    );
}

/// Covers: `ferray_ufunc::atanh_complex`.
#[test]
fn atanh_complex_matches_numpy() {
    run_complex_to_complex(
        "atanh_complex.json",
        "atanh_complex",
        ferray_ufunc::atanh_complex,
    );
}

/// Covers: `ferray_ufunc::power_complex` (re-export of
/// `ferray_ufunc::ops::complex::power_complex`).
#[test]
fn power_complex_matches_numpy() {
    run_complex_power(
        "power_complex.json",
        "power_complex",
        ferray_ufunc::power_complex,
    );
}
