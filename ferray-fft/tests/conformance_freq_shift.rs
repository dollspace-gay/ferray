//! Conformance tests for the frequency-grid and shift utilities of
//! ferray-fft.
//!
//! `fftfreq` / `rfftfreq` return analytically exact rational values, so
//! both `tolerance_ulps = 0` in the fixtures; we still use the Stage 1
//! FFT relative tolerance (`TOL_FFT_F64_REL = 1e-10`) plus a small
//! absolute floor to keep the comparator robust against literal-0 bins.
//!
//! `fftshift` / `ifftshift` are pure index permutations — they preserve
//! values bit-for-bit, so the tolerance is generously loose.
//!
//! Tests anchor BOTH the crate-root re-export and the inner canonical
//! module path for the surface-coverage gate.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp,
    unused_imports
)]

use ferray_test_oracle::{
    TOL_FFT_F64_REL, assert_close_f64_slice, fixtures_dir, load_fixture, make_f64_array,
    parse_f64_data, serde_json, should_skip_f64,
};

const TOL_ABS: f64 = 1e-12;

fn fft_path(name: &str) -> std::path::PathBuf {
    fixtures_dir().join("fft").join(name)
}

// ===========================================================================
// fftfreq / rfftfreq
//
// User-facing re-exports: `ferray_fft::fftfreq`, `ferray_fft::rfftfreq`
// Inner canonical paths:   `ferray_fft::freq::fftfreq`, `ferray_fft::freq::rfftfreq`
// ===========================================================================

/// Path anchors: `ferray_fft::fftfreq`, `ferray_fft::freq::fftfreq`.
#[test]
fn conformance_fftfreq() {
    let suite = load_fixture(&fft_path("fftfreq.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let d = case
            .inputs
            .get("d")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0);
        let result = ferray_fft::fftfreq(n, d).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_close_f64_slice(
            result.as_slice().unwrap(),
            &expected,
            TOL_FFT_F64_REL,
            TOL_ABS,
        );
        tested += 1;
    }
    assert!(tested > 0, "conformance_fftfreq: no cases tested");
}

/// Path anchors: `ferray_fft::rfftfreq`, `ferray_fft::freq::rfftfreq`.
#[test]
fn conformance_rfftfreq() {
    let suite = load_fixture(&fft_path("rfftfreq.json"));
    let mut tested = 0usize;
    for case in &suite.test_cases {
        let n = case.inputs["n"].as_u64().unwrap() as usize;
        let d = case
            .inputs
            .get("d")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0);
        let result = ferray_fft::rfftfreq(n, d).unwrap();
        let expected = parse_f64_data(&case.expected["data"]);
        assert_close_f64_slice(
            result.as_slice().unwrap(),
            &expected,
            TOL_FFT_F64_REL,
            TOL_ABS,
        );
        tested += 1;
    }
    assert!(tested > 0, "conformance_rfftfreq: no cases tested");
}

// ===========================================================================
// fftshift / ifftshift — pure index permutations
//
// User-facing re-exports: `ferray_fft::fftshift`, `ferray_fft::ifftshift`
// Inner canonical paths:   `ferray_fft::shift::fftshift`, `ferray_fft::shift::ifftshift`
// ===========================================================================

/// Path anchors: `ferray_fft::fftshift`, `ferray_fft::shift::fftshift`.
#[test]
fn conformance_fftshift() {
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
        assert_close_f64_slice(
            result.as_slice().unwrap(),
            &expected,
            TOL_FFT_F64_REL,
            TOL_ABS,
        );
        tested += 1;
    }
    assert!(tested > 0, "conformance_fftshift: no f64 cases tested");
}

/// Path anchors: `ferray_fft::ifftshift`, `ferray_fft::shift::ifftshift`.
#[test]
fn conformance_ifftshift() {
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
        assert_close_f64_slice(
            result.as_slice().unwrap(),
            &expected,
            TOL_FFT_F64_REL,
            TOL_ABS,
        );
        tested += 1;
    }
    assert!(tested > 0, "conformance_ifftshift: no f64 cases tested");
}
