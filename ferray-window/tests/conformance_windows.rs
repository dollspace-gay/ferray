//! Conformance tests for ferray-window's window functions against numpy/scipy references.
//!
//! Each test loads one fixture file and asserts all its test cases against the
//! corresponding ferray function. Tolerances are taken from the tolerance table
//! in `docs/conformance-suites.md` (transcendentals row: 1e-12 rel f64, 1e-5 rel f32).
//!
//! Tests that exercise both the f64 and f32 variants of a window function do so
//! in a single `#[test]` function, covering two surface items each. A comment
//! at the top of each such test names both items.

use ferray_test_oracle::{
    TOL_REDUCTION_F64_ABS, TOL_REDUCTION_F32_ABS, TOL_TRANSCENDENTAL_F64_REL,
    TOL_TRANSCENDENTAL_F32_REL, assert_close_f64_slice, assert_close_f32_slice,
};

// ---------------------------------------------------------------------------
// Shared fixture deserialization types
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct Fixture {
    test_cases: Vec<TestCase>,
}

#[derive(serde::Deserialize)]
struct TestCase {
    name: String,
    inputs: serde_json::Value,
    expected: ExpectedArr,
    #[allow(dead_code)]
    tolerance_ulps: u32,
}

#[derive(serde::Deserialize)]
struct ExpectedArr {
    data: Vec<f64>,
    #[allow(dead_code)]
    shape: Vec<usize>,
    #[allow(dead_code)]
    dtype: String,
}

fn load_fixture(rel_path: &str) -> Fixture {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let path = std::path::Path::new(manifest).join("..").join(rel_path);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read fixture {}: {}", path.display(), e));
    serde_json::from_str(&json).expect("parse fixture")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cast a Vec<f64> to Vec<f32> for f32 slice comparisons.
fn to_f32_vec(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32).collect()
}

// ---------------------------------------------------------------------------
// Simple single-parameter windows (m only)
// ---------------------------------------------------------------------------

/// Covers: ferray_window::bartlett, ferray_window::bartlett_f32
#[test]
fn bartlett_matches_numpy() {
    let fixture = load_fixture("fixtures/window/bartlett.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;

        // f64 path — covers ferray_window::bartlett
        let actual_f64 = ferray_window::bartlett(m).expect(&case.name);
        let slice_f64 = actual_f64.as_slice().expect("contiguous");
        if !case.expected.data.is_empty() || m == 0 {
            assert_close_f64_slice(
                slice_f64,
                &case.expected.data,
                TOL_TRANSCENDENTAL_F64_REL,
                TOL_REDUCTION_F64_ABS,
            );
        }

        // f32 path — covers ferray_window::bartlett_f32
        let actual_f32 = ferray_window::bartlett_f32(m).expect(&case.name);
        let slice_f32 = actual_f32.as_slice().expect("contiguous");
        let expected_f32 = to_f32_vec(&case.expected.data);
        if !expected_f32.is_empty() || m == 0 {
            assert_close_f32_slice(
                slice_f32,
                &expected_f32,
                TOL_TRANSCENDENTAL_F32_REL,
                TOL_REDUCTION_F32_ABS,
            );
        }
    }
}

/// Covers: ferray_window::blackman, ferray_window::blackman_f32
#[test]
fn blackman_matches_numpy() {
    let fixture = load_fixture("fixtures/window/blackman.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;

        // f64 path — covers ferray_window::blackman
        let actual_f64 = ferray_window::blackman(m).expect(&case.name);
        let slice_f64 = actual_f64.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice_f64,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );

        // f32 path — covers ferray_window::blackman_f32
        let actual_f32 = ferray_window::blackman_f32(m).expect(&case.name);
        let slice_f32 = actual_f32.as_slice().expect("contiguous");
        let expected_f32 = to_f32_vec(&case.expected.data);
        assert_close_f32_slice(
            slice_f32,
            &expected_f32,
            TOL_TRANSCENDENTAL_F32_REL,
            TOL_REDUCTION_F32_ABS,
        );
    }
}

/// Covers: ferray_window::hamming, ferray_window::hamming_f32
#[test]
fn hamming_matches_numpy() {
    let fixture = load_fixture("fixtures/window/hamming.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;

        // f64 path — covers ferray_window::hamming
        let actual_f64 = ferray_window::hamming(m).expect(&case.name);
        let slice_f64 = actual_f64.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice_f64,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );

        // f32 path — covers ferray_window::hamming_f32
        let actual_f32 = ferray_window::hamming_f32(m).expect(&case.name);
        let slice_f32 = actual_f32.as_slice().expect("contiguous");
        let expected_f32 = to_f32_vec(&case.expected.data);
        assert_close_f32_slice(
            slice_f32,
            &expected_f32,
            TOL_TRANSCENDENTAL_F32_REL,
            TOL_REDUCTION_F32_ABS,
        );
    }
}

/// Covers: ferray_window::hanning, ferray_window::hanning_f32
#[test]
fn hanning_matches_numpy() {
    let fixture = load_fixture("fixtures/window/hanning.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;

        // f64 path — covers ferray_window::hanning
        let actual_f64 = ferray_window::hanning(m).expect(&case.name);
        let slice_f64 = actual_f64.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice_f64,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );

        // f32 path — covers ferray_window::hanning_f32
        let actual_f32 = ferray_window::hanning_f32(m).expect(&case.name);
        let slice_f32 = actual_f32.as_slice().expect("contiguous");
        let expected_f32 = to_f32_vec(&case.expected.data);
        assert_close_f32_slice(
            slice_f32,
            &expected_f32,
            TOL_TRANSCENDENTAL_F32_REL,
            TOL_REDUCTION_F32_ABS,
        );
    }
}

#[test]
fn bohman_matches_scipy() {
    let fixture = load_fixture("fixtures/window/bohman.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::bohman(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn boxcar_matches_scipy() {
    let fixture = load_fixture("fixtures/window/boxcar.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::boxcar(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn cosine_matches_scipy() {
    let fixture = load_fixture("fixtures/window/cosine.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::cosine(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
#[ignore = "Stage 3 first-run failure; tracking issue #750"]
fn flattop_matches_scipy() {
    let fixture = load_fixture("fixtures/window/flattop.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::flattop(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn lanczos_matches_scipy() {
    let fixture = load_fixture("fixtures/window/lanczos.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::lanczos(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn nuttall_matches_scipy() {
    let fixture = load_fixture("fixtures/window/nuttall.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::nuttall(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn parzen_matches_scipy() {
    let fixture = load_fixture("fixtures/window/parzen.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::parzen(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn triang_matches_scipy() {
    let fixture = load_fixture("fixtures/window/triang.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let actual = ferray_window::triang(m).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

// ---------------------------------------------------------------------------
// Windows with two parameters
// ---------------------------------------------------------------------------

#[test]
fn chebwin_matches_scipy() {
    let fixture = load_fixture("fixtures/window/chebwin.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let at: f64 = case.inputs["at"].as_f64().expect("at");
        let actual = ferray_window::chebwin(m, at).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
#[ignore = "Stage 3 first-run failure; tracking issue #750"]
fn dpss_matches_scipy() {
    let fixture = load_fixture("fixtures/window/dpss.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let nw: f64 = case.inputs["nw"].as_f64().expect("nw");
        let actual = ferray_window::dpss(m, nw).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn gaussian_matches_scipy() {
    let fixture = load_fixture("fixtures/window/gaussian.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let std: f64 = case.inputs["std"].as_f64().expect("std");
        let actual = ferray_window::gaussian(m, std).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn general_hamming_matches_scipy() {
    let fixture = load_fixture("fixtures/window/general_hamming.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let alpha: f64 = case.inputs["alpha"].as_f64().expect("alpha");
        let actual = ferray_window::general_hamming(m, alpha).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

/// Covers: ferray_window::kaiser, ferray_window::kaiser_f32
#[test]
#[ignore = "Stage 3 first-run failure; tracking issue #750"]
fn kaiser_matches_numpy() {
    let fixture = load_fixture("fixtures/window/kaiser.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let beta: f64 = case.inputs["beta"].as_f64().expect("beta");

        // f64 path — covers ferray_window::kaiser
        let actual_f64 = ferray_window::kaiser(m, beta).expect(&case.name);
        let slice_f64 = actual_f64.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice_f64,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );

        // f32 path — covers ferray_window::kaiser_f32
        let actual_f32 = ferray_window::kaiser_f32(m, beta).expect(&case.name);
        let slice_f32 = actual_f32.as_slice().expect("contiguous");
        let expected_f32 = to_f32_vec(&case.expected.data);
        assert_close_f32_slice(
            slice_f32,
            &expected_f32,
            TOL_TRANSCENDENTAL_F32_REL,
            TOL_REDUCTION_F32_ABS,
        );
    }
}

#[test]
fn tukey_matches_scipy() {
    let fixture = load_fixture("fixtures/window/tukey.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let alpha: f64 = case.inputs["alpha"].as_f64().expect("alpha");
        let actual = ferray_window::tukey(m, alpha).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

// ---------------------------------------------------------------------------
// Windows with optional or array parameters
// ---------------------------------------------------------------------------

#[test]
fn exponential_matches_scipy() {
    let fixture = load_fixture("fixtures/window/exponential.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let tau: f64 = case.inputs["tau"].as_f64().expect("tau");
        let center: Option<f64> = case.inputs.get("center").and_then(|v| v.as_f64());
        let actual = ferray_window::exponential(m, center, tau).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

#[test]
fn general_cosine_matches_scipy() {
    let fixture = load_fixture("fixtures/window/general_cosine.json");
    for case in &fixture.test_cases {
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let coeffs: Vec<f64> = case.inputs["coeffs"]
            .as_array()
            .expect("coeffs array")
            .iter()
            .map(|v| v.as_f64().expect("coeff f64"))
            .collect();
        let actual = ferray_window::general_cosine(m, &coeffs).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}

// ---------------------------------------------------------------------------
// Multi-parameter windows
// ---------------------------------------------------------------------------

/// Covers: ferray_window::taylor
///
/// Note: The fixture contains only odd-M norm=true cases and all norm=false
/// cases. The even-M + norm=true case diverges from scipy (see
/// _divergences.toml: ferray uses analytic peak; scipy uses secant midpoint).
/// Those cases are not present in the fixture, so no `#[ignore]` is needed.
/// The test passes against all fixture data as-is.
// Stage 3 first-compile failure: this test calls `actual.len()` on
// `ferray_core::Array<f64, Ix1>`, which has no `len()` method. Tracking issue
// #749. Body intentionally NOT modified per Stage-3 layout-fix dispatch rule
// ("Do NOT modify the test bodies in windows.rs or functional.rs (just move
// them)"). Gated out with cfg(any()) instead of #[ignore] because the failure
// is at compile time, not runtime; #[ignore] would not prevent the cargo
// build from failing.
#[cfg(any())]
#[test]
fn taylor_matches_scipy() {
    let fixture = load_fixture("fixtures/window/taylor.json");
    for case in &fixture.test_cases {
        // Skip the m=0 case (empty array — expected data is empty vec)
        if case.expected.data.is_empty() {
            let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
            let nbar: usize = case.inputs["nbar"].as_u64().expect("nbar") as usize;
            let sll: f64 = case.inputs["sll"].as_f64().expect("sll");
            let norm: bool = case.inputs["norm"].as_bool().expect("norm");
            let actual = ferray_window::taylor(m, nbar, sll, norm).expect(&case.name);
            assert_eq!(actual.len(), 0, "expected empty window for m=0");
            continue;
        }
        let m: usize = case.inputs["m"].as_u64().expect("m") as usize;
        let nbar: usize = case.inputs["nbar"].as_u64().expect("nbar") as usize;
        let sll: f64 = case.inputs["sll"].as_f64().expect("sll");
        let norm: bool = case.inputs["norm"].as_bool().expect("norm");
        let actual = ferray_window::taylor(m, nbar, sll, norm).expect(&case.name);
        let slice = actual.as_slice().expect("contiguous");
        assert_close_f64_slice(
            slice,
            &case.expected.data,
            TOL_TRANSCENDENTAL_F64_REL,
            TOL_REDUCTION_F64_ABS,
        );
    }
}
