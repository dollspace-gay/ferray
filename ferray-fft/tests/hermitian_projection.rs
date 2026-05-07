// Regression tests for the silent Hermitian projection on the c2r axis.
//
// Before 0.3.8, ferray-fft surfaced realfft's `FftError::InputValues` —
// the contract violation realfft signals when the imaginary parts of the
// DC and/or Nyquist bins on the c2r axis are non-zero — as a hard error
// (1-D fast path) or a per-lane panic (multi-lane path). scipy and
// PyTorch silently project the input onto the Hermitian subspace via
// pocketfft's c2r kernel / `aten::_fft_c2r`, producing a numerically
// correct result.
//
// As of 0.3.8, ferray-fft projects the DC bin and (for even output
// length) the Nyquist bin on the c2r axis to real values before invoking
// realfft, matching scipy/torch semantics. These tests guard:
//
//   * 1-D irfft with non-Hermitian input succeeds (no Err).
//   * Multi-axis irfftn with non-Hermitian input succeeds (no panic
//     in the rayon worker pool).
//   * hfft with non-Hermitian input succeeds.
//   * Already-Hermitian inputs are unchanged (no precision drift).
//   * Odd output length only projects DC (no Nyquist exists).
//
// Reference values were computed against scipy 1.17.1 (see
// /tmp/w4_scipy_probe.py in the W4 investigation). Tolerance is
// 1e-10 on f64.

#![allow(clippy::float_cmp)]

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2};
use ferray_fft::{FftNorm, hfft, irfft, irfftn};
use num_complex::Complex;

const fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

/// 1-D irfft on a deliberately non-Hermitian input — DC and Nyquist
/// imaginary parts are non-zero. scipy reports the result below
/// silently; ferray-fft must do the same.
#[test]
fn irfft_1d_non_hermitian_silent_projection() {
    // Input: 5 complex bins (n/2+1 = 5 → output n = 8).
    // Both DC (index 0) and Nyquist (index 4) carry non-zero imaginary
    // parts; scipy.fft.irfft accepts this without error.
    let spec = vec![
        c(1.0, 1.0),  // DC: imag should be projected to 0
        c(2.0, 3.0),
        c(4.0, -1.0),
        c(-1.0, 0.5),
        c(7.0, 9.0),  // Nyquist (n=8 is even): imag should be projected to 0
    ];
    let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([5]), spec).unwrap();
    let result = irfft(&a, Some(8), None, FftNorm::Backward)
        .expect("irfft on non-Hermitian input must not error");

    // scipy reference: scipy.fft.irfft([1+1j, 2+3j, 4-1j, -1+0.5j, 7+9j], n=8)
    // (scipy 1.17.1, generated 2026-05-06)
    let expected = [
        2.25,
        -0.5883883476483185,
        -0.625,
        -2.149048519428139,
        1.75,
        -0.4116116523516815,
        0.625,
        0.14904851942813968,
    ];
    let got: Vec<f64> = result.iter().copied().collect();
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-10,
            "bin {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

/// 2-D irfftn on the multi-lane parallel path (3 lanes × 5 bins).
/// Pre-fix this would panic in rayon workers; post-fix it must produce
/// the same output as if the caller had manually zeroed DC.im and
/// Nyquist.im on every lane before calling.
#[test]
fn irfftn_2d_multi_lane_non_hermitian_no_panic() {
    // 3×5 spectrum with arbitrary complex values in every cell. With
    // s=(3,8) the c2r runs on axis 1 with output length 8 (even, so
    // Nyquist exists at index 4 of every row).
    let spec = vec![
        c(1.0, 1.0), c(2.0, 3.0), c(4.0, -1.0), c(-1.0, 0.5), c(7.0, 9.0),
        c(0.5, -0.25), c(1.5, 2.0), c(-3.0, 1.5), c(2.5, -0.5), c(-1.0, 4.0),
        c(-2.0, 3.5), c(0.0, 1.0), c(1.5, -2.0), c(-0.5, 0.25), c(3.0, -1.5),
    ];
    let a = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([3, 5]), spec).unwrap();
    let result = irfftn(&a, Some(&[3, 8]), None, FftNorm::Backward)
        .expect("irfftn on non-Hermitian input must not panic");

    // scipy reference: scipy.fft.irfftn(spec_2d, s=(3, 8))
    // (scipy 1.17.1, generated 2026-05-06; spec_2d is the 3×5 array
    // constructed above with arbitrary complex bins on every cell —
    // including non-zero imaginary parts on DC and Nyquist of every
    // row of axis 1, which scipy projects silently).
    let expected: [f64; 24] = [
        0.9375, -0.4918042024541294, -0.3333333333333333, -1.0364320279485242,
        0.1875, -0.04986246421253722, 0.625, -0.0052346387181424845,
        0.32246937562474765, 0.29130410127242934, 0.15186289921756746, -0.6876224976306726,
        0.48355376744909917, 0.7966095336497376, 0.08118988160479113, 0.22649802687618184,
        0.9900306243752521, -0.38788824646661846, -0.4435295658842341, -0.4249939938489427,
        1.0789462325509007, -1.1583587217888818, -0.08118988160479113, -0.07221486872989968,
    ];
    let got: Vec<f64> = result.iter().copied().collect();
    assert_eq!(got.len(), expected.len());
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-10,
            "idx {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

/// hfft delegates to irfft via conjugation; it must inherit the
/// projection. Pre-fix this errored / panicked on the same class of
/// input. Post-fix it must succeed and round-trip the projected form.
#[test]
fn hfft_non_hermitian_silent_projection() {
    // 5 complex bins → hfft output length 8.
    let spec = vec![
        c(1.0, 1.0),  // DC
        c(2.0, 3.0),
        c(4.0, -1.0),
        c(-1.0, 0.5),
        c(7.0, 9.0),  // Nyquist (n=8 is even)
    ];
    let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([5]), spec).unwrap();
    let result = hfft(&a, Some(8), None, FftNorm::Backward)
        .expect("hfft on non-Hermitian input must not error");
    assert_eq!(result.shape(), &[8]);

    // scipy reference: scipy.fft.hfft([1+1j, 2+3j, 4-1j, -1+0.5j, 7+9j], n=8)
    let expected = [
        18.0,
        1.1923881554251174,
        5.0,
        -3.292893218813454,
        14.0,
        -17.192388155425117,
        -5.0,
        -4.707106781186546,
    ];
    let got: Vec<f64> = result.iter().copied().collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-10,
            "bin {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

/// Already-Hermitian inputs must produce the same result as before the
/// fix — projecting zero is zero, and the algorithm must be a strict
/// no-op for valid inputs.
#[test]
fn irfft_already_hermitian_unchanged() {
    // Hermitian: DC.im = 0, Nyquist.im = 0.
    let spec = vec![
        c(2.0, 0.0),  // DC, im = 0
        c(1.0, -0.5),
        c(-0.5, 0.25),
        c(0.75, -1.0),
        c(3.0, 0.0),  // Nyquist, im = 0
    ];
    let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([5]), spec).unwrap();
    let result = irfft(&a, Some(8), None, FftNorm::Backward).unwrap();
    // For backward-norm irfft the sum of all output samples equals the
    // DC bin's real part (which is X[0].re = 2.0 here). This is the
    // identity sum_k irfft(X)[k] = X[0] when X[0].im is already 0.
    let sum: f64 = result.iter().sum();
    assert!((sum - 2.0).abs() < 1e-10, "DC sum mismatch: {sum}");

    // Compare element-wise against scipy reference.
    // scipy.fft.irfft([2+0j, 1-0.5j, -0.5+0.25j, 0.75-1j, 3+0j], n=8)
    let expected = [
        0.9375,
        0.1218592167691146,
        0.625,
        0.15847086912079608,
        0.0625,
        -0.4968592167691146,
        0.875,
        -0.2834708691207961,
    ];
    let got: Vec<f64> = result.iter().copied().collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-10,
            "bin {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}

/// Odd output length: no Nyquist bin exists, so only DC is projected.
/// realfft's `ComplexToRealOdd` only validates DC. This guards the
/// `output_len % 2 == 0` gate in the helper.
#[test]
fn irfft_odd_output_only_projects_dc() {
    // Output length 7 (odd) → input has 7/2 + 1 = 4 bins. DC.im is
    // non-zero; there is no Nyquist bin.
    let spec = vec![
        c(1.0, 2.5),  // DC: imag should be projected
        c(0.5, 1.0),
        c(-1.0, 0.5),
        c(0.25, -0.75),
    ];
    let a = Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([4]), spec).unwrap();
    let result = irfft(&a, Some(7), None, FftNorm::Backward)
        .expect("irfft odd-N on non-Hermitian DC must not error");
    assert_eq!(result.shape(), &[7]);

    // scipy reference: scipy.fft.irfft([1+2.5j, 0.5+1j, -1+0.5j, 0.25-0.75j], n=7)
    // (odd N → no Nyquist; only DC.im is projected)
    let expected = [
        0.07142857142857142,
        -0.038531147517341956,
        0.028920341975679543,
        0.01674956198680084,
        -0.37652354444338454,
        0.7971258588180034,
        0.5008303577516712,
    ];
    let got: Vec<f64> = result.iter().copied().collect();
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-10,
            "bin {i}: got {g}, expected {e}, diff {}",
            (g - e).abs()
        );
    }
}
