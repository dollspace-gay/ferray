//! Conformance tests for the complex-FFT family of ferray-fft.
//!
//! Each test loads the JSON fixture under `fixtures/fft/` and compares
//! the ferray output to the NumPy reference using the Stage 1 FFT
//! tolerance (`TOL_FFT_F64_REL = 1e-10`, see
//! `docs/conformance-suites.md`). A small absolute tolerance is added
//! to absorb near-zero ULP noise on the inverse spectrum bins that
//! analytically equal 0.
//!
//! Tests name BOTH the user-facing crate-root re-export AND the inner
//! canonical module path in doc-comment "path anchor" lines so the
//! surface-coverage gate's textual match picks up both spellings.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    unused_imports
)]

use ferray_core::Array;
use ferray_core::dimension::IxDyn;
use ferray_fft::FftNorm;
use ferray_test_oracle::{
    TOL_FFT_F64_REL, assert_close_f64_slice, fixtures_dir, get_dtype, load_fixture,
    make_complex_array, make_f64_array, parse_complex_data, parse_shape,
};
use num_complex::Complex;

const TOL_ABS: f64 = 1e-12;

fn fft_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("fft").join(name)
}

/// Interleave a `Complex<f64>` slice into a flat `f64` slice of length 2N
/// so it can be fed to `assert_close_f64_slice` together with the
/// reference output (`expected` lives as `Vec<Complex<f64>>`).
fn interleave_complex(z: &[Complex<f64>]) -> Vec<f64> {
    let mut out = Vec::with_capacity(z.len() * 2);
    for c in z {
        out.push(c.re);
        out.push(c.im);
    }
    out
}

// ===========================================================================
// 1-D complex forward / inverse FFT
//
// User-facing re-exports (crate root):
//   - `ferray_fft::fft`
//   - `ferray_fft::ifft`
// Inner canonical paths:
//   - `ferray_fft::complex::fft`
//   - `ferray_fft::complex::ifft`
// ===========================================================================

/// Path anchors for the surface gate:
/// `ferray_fft::fft`, `ferray_fft::complex::fft`.
#[test]
fn conformance_fft_1d() {
    let suite = load_fixture(&fft_path("fft.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = if dtype == "complex128" {
            make_complex_array(input)
        } else if dtype == "float64" {
            let data = ferray_test_oracle::parse_f64_data(&input["data"]);
            let cdata: Vec<Complex<f64>> =
                data.into_iter().map(|r| Complex::new(r, 0.0)).collect();
            Array::from_vec(IxDyn::new(&shape), cdata).unwrap()
        } else {
            continue;
        };
        let result = ferray_fft::fft(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_flat = interleave_complex(result.as_slice().unwrap());
        let expected_flat = interleave_complex(&expected);
        assert_close_f64_slice(&result_flat, &expected_flat, TOL_FFT_F64_REL, TOL_ABS);
        tested += 1;
    }
    assert!(tested > 0, "conformance_fft_1d: no cases tested");
}

/// Path anchors for the surface gate:
/// `ferray_fft::ifft`, `ferray_fft::complex::ifft`.
#[test]
fn conformance_ifft_1d() {
    let suite = load_fixture(&fft_path("ifft.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if get_dtype(input) != "complex128" {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.is_empty() {
            continue;
        }
        let arr = make_complex_array(input);
        let result = ferray_fft::ifft(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_flat = interleave_complex(result.as_slice().unwrap());
        let expected_flat = interleave_complex(&expected);
        assert_close_f64_slice(&result_flat, &expected_flat, TOL_FFT_F64_REL, TOL_ABS);
        tested += 1;
    }
    assert!(tested > 0, "conformance_ifft_1d: no cases tested");
}

// ===========================================================================
// 2-D complex forward / inverse FFT
//
// User-facing re-exports:
//   - `ferray_fft::fft2`
//   - `ferray_fft::ifft2`
// Inner canonical paths:
//   - `ferray_fft::complex::fft2`
//   - `ferray_fft::complex::ifft2`
// ===========================================================================

/// Path anchors: `ferray_fft::fft2`, `ferray_fft::complex::fft2`.
#[test]
fn conformance_fft2() {
    let suite = load_fixture(&fft_path("fft2.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        let dtype = get_dtype(input);
        let shape = parse_shape(&input["shape"]);
        if shape.len() < 2 {
            continue;
        }
        let arr = if dtype == "complex128" {
            make_complex_array(input)
        } else if dtype == "float64" {
            let data = ferray_test_oracle::parse_f64_data(&input["data"]);
            let cdata: Vec<Complex<f64>> =
                data.into_iter().map(|r| Complex::new(r, 0.0)).collect();
            Array::from_vec(IxDyn::new(&shape), cdata).unwrap()
        } else {
            continue;
        };
        let result = ferray_fft::fft2(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_flat = interleave_complex(result.as_slice().unwrap());
        let expected_flat = interleave_complex(&expected);
        assert_close_f64_slice(&result_flat, &expected_flat, TOL_FFT_F64_REL, TOL_ABS);
        tested += 1;
    }
    assert!(tested > 0, "conformance_fft2: no cases tested");
}

/// Path anchors: `ferray_fft::ifft2`, `ferray_fft::complex::ifft2`.
#[test]
fn conformance_ifft2() {
    let suite = load_fixture(&fft_path("ifft2.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let input = &case.inputs["x"];
        if get_dtype(input) != "complex128" {
            continue;
        }
        let shape = parse_shape(&input["shape"]);
        if shape.len() < 2 {
            continue;
        }
        let arr = make_complex_array(input);
        let result = ferray_fft::ifft2(&arr, None, None, FftNorm::Backward).unwrap();
        let expected = parse_complex_data(&case.expected["data"]);
        let result_flat = interleave_complex(result.as_slice().unwrap());
        let expected_flat = interleave_complex(&expected);
        assert_close_f64_slice(&result_flat, &expected_flat, TOL_FFT_F64_REL, TOL_ABS);
        tested += 1;
    }
    assert!(tested > 0, "conformance_ifft2: no cases tested");
}
