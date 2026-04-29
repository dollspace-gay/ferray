//! Oracle tests: validate ferray-fft against `NumPy` fixture outputs.

// Oracle tests cross fixture data through `f64` JSON values, recover bit
// patterns, and ULP-compare against NumPy reference outputs — both the
// casts and the float comparisons are part of the contract.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_fft::FftNorm;
use ferray_test_oracle::*;
use num_complex::Complex;

fn fft_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("fft").join(name)
}

// ---------------------------------------------------------------------------
// Complex FFTs
// ---------------------------------------------------------------------------

#[test]
fn oracle_fft() {
    run_complex_to_complex_oracle(&fft_path("fft.json"), |x| {
        ferray_fft::fft(x, None, None, FftNorm::Backward)
    });
}

#[test]
fn oracle_ifft() {
    run_complex_to_complex_oracle(&fft_path("ifft.json"), |x| {
        ferray_fft::ifft(x, None, None, FftNorm::Backward)
    });
}

// ---------------------------------------------------------------------------
// Real FFTs
// ---------------------------------------------------------------------------

#[test]
fn oracle_rfft() {
    let suite = load_fixture(&fft_path("rfft.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        if dtype != "float64" {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_fft::rfft(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        assert_complex_slice_ulp(
            result_slice,
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    // Guard against silently passing because every fixture case got
    // skipped — see issue #231 (oracle tests skipped non-f64 dtypes
    // without telling anyone). At least one f64 case must run.
    assert!(
        tested > 0,
        "oracle_rfft: no f64 fixture cases were tested — is rfft.json empty or all complex/f32?"
    );
}

#[test]
fn oracle_irfft() {
    let suite = load_fixture(&fft_path("irfft.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        if dtype != "complex128" {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_complex_array(input);
        let n = case
            .inputs
            .get("n")
            .and_then(ferray_test_oracle::serde_json::Value::as_u64)
            .map(|v| v as usize);
        let result = ferray_fft::irfft(&arr, n, None, FftNorm::Backward).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(
        tested > 0,
        "oracle_irfft: no complex128 fixture cases were tested (#231)"
    );
}

// ---------------------------------------------------------------------------
// Frequency utilities
// ---------------------------------------------------------------------------

#[test]
fn oracle_fftfreq() {
    let suite = load_fixture(&fft_path("fftfreq.json"));
    for case in &suite.test_cases {
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let d = case
            .inputs
            .get("d")
            .and_then(ferray_test_oracle::serde_json::Value::as_f64)
            .unwrap_or(1.0);
        let result = ferray_fft::fftfreq(n, d).unwrap();
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
fn oracle_rfftfreq() {
    let suite = load_fixture(&fft_path("rfftfreq.json"));
    for case in &suite.test_cases {
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let d = case
            .inputs
            .get("d")
            .and_then(ferray_test_oracle::serde_json::Value::as_f64)
            .unwrap_or(1.0);
        let result = ferray_fft::rfftfreq(n, d).unwrap();
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

// ---------------------------------------------------------------------------
// Shift utilities
// ---------------------------------------------------------------------------

#[test]
fn oracle_fftshift() {
    let suite = load_fixture(&fft_path("fftshift.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_fft::fftshift(&arr, None).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "oracle_fftshift: no f64 cases tested (#231)");
}

#[test]
fn oracle_ifftshift() {
    let suite = load_fixture(&fft_path("ifftshift.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if should_skip_f64(input) {
            continue;
        }
        let arr = make_f64_array(input);
        let result = ferray_fft::ifftshift(&arr, None).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_f64_slice_ulp(
            result.as_slice().unwrap(),
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
        tested += 1;
    }
    assert!(tested > 0, "oracle_ifftshift: no f64 cases tested (#231)");
}

// ---------------------------------------------------------------------------
// 2D FFTs
// ---------------------------------------------------------------------------

#[test]
fn oracle_fft2() {
    let suite = load_fixture(&fft_path("fft2.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() || shape.len() < 2 {
            continue;
        }
        let arr = if dtype == "complex128" {
            make_complex_array(input)
        } else if dtype == "float64" {
            let data = parse_f64_data(&input["data"]);
            let complex_data: Vec<Complex<f64>> =
                data.into_iter().map(|r| Complex::new(r, 0.0)).collect();
            Array::from_vec(IxDyn::new(&shape), complex_data).unwrap()
        } else {
            continue;
        };
        let result = ferray_fft::fft2(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        assert_complex_slice_ulp(
            result_slice,
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}

#[test]
fn oracle_ifft2() {
    let suite = load_fixture(&fft_path("ifft2.json"));
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() || shape.len() < 2 {
            continue;
        }
        let arr = if dtype == "complex128" {
            make_complex_array(input)
        } else {
            continue;
        };
        let result = ferray_fft::ifft2(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_slice = result.as_slice().unwrap();
        assert_complex_slice_ulp(
            result_slice,
            &expected,
            case.tolerance_ulps
                .max(ferray_test_oracle::MIN_ULP_TOLERANCE),
            &case.name,
        );
    }
}
