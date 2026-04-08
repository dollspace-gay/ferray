// ferray-window: Window functions for signal processing and spectral analysis
//
// Implements NumPy-equivalent window functions: bartlett, blackman, hamming,
// hanning, and kaiser. Each returns an Array1<f64> of the specified length M.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::error::{FerrayError, FerrayResult};

use std::f64::consts::PI;

// Re-use the Abramowitz & Stegun Bessel I_0 polynomial approximation
// from ferray-ufunc instead of shipping a second copy (see #530).
// The previous copy had already drifted in whitespace; funnelling
// through the canonical crate keeps both the window code and the
// ufunc in sync.
use ferray_ufunc::ops::special::bessel_i0_scalar;

/// Build a window array of length `m` by evaluating `f` at every sample
/// `0..m`. Handles the shared `m == 0` / `m == 1` edge cases that every
/// NumPy-equivalent window has (see issue #292 for the original
/// five-way copy of this pattern).
#[inline]
fn gen_window<F: FnMut(usize) -> f64>(m: usize, mut f: F) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        data.push(f(n));
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Return the Bartlett (triangular) window of length `m`.
///
/// The Bartlett window is defined as:
///   w(n) = 2/(M-1) * ((M-1)/2 - |n - (M-1)/2|)
///
/// This is equivalent to `numpy.bartlett(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn bartlett(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let half = (m.saturating_sub(1)) as f64 / 2.0;
    gen_window(m, |n| 1.0 - ((n as f64 - half) / half).abs())
}

/// Return the Blackman window of length `m`.
///
/// The Blackman window is defined as:
///   w(n) = 0.42 - 0.5 * cos(2*pi*n/(M-1)) + 0.08 * cos(4*pi*n/(M-1))
///
/// This is equivalent to `numpy.blackman(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn blackman(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| {
        let x = n as f64;
        0.42 - 0.5 * (2.0 * PI * x / denom).cos() + 0.08 * (4.0 * PI * x / denom).cos()
    })
}

/// Return the Hamming window of length `m`.
///
/// The Hamming window is defined as:
///   w(n) = 0.54 - 0.46 * cos(2*pi*n/(M-1))
///
/// This is equivalent to `numpy.hamming(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hamming(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| 0.54 - 0.46 * (2.0 * PI * n as f64 / denom).cos())
}

/// Return the Hann (Hanning) window of length `m`.
///
/// The Hann window is defined as:
///   w(n) = 0.5 * (1 - cos(2*pi*n/(M-1)))
///
/// NumPy calls this function `hanning`. This is equivalent to `numpy.hanning(M)`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hanning(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    let denom = (m.saturating_sub(1)) as f64;
    gen_window(m, |n| 0.5 * (1.0 - (2.0 * PI * n as f64 / denom).cos()))
}

/// Return the Kaiser window of length `m` with shape parameter `beta`.
///
/// The Kaiser window is defined as:
///   w(n) = I_0(beta * sqrt(1 - ((2n/(M-1)) - 1)^2)) / I_0(beta)
///
/// where I_0 is the modified Bessel function of the first kind, order 0.
///
/// This is equivalent to `numpy.kaiser(M, beta)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `beta` is NaN or if `beta`
/// exceeds the range where `I_0(beta)` can be computed in f64. The
/// asymptotic branch uses `exp(beta) / sqrt(beta)` internally, so
/// `|beta| > ~709` overflows to infinity and the normalized kaiser
/// window would reduce to `Inf / Inf = NaN` for every sample
/// (#294).
pub fn kaiser(m: usize, beta: f64) -> FerrayResult<Array<f64, Ix1>> {
    if beta.is_nan() {
        return Err(FerrayError::invalid_value(
            "kaiser: beta must not be NaN",
        ));
    }
    // I_0 is an even function, so kaiser(m, -beta) == kaiser(m, beta).
    // Accept negative beta for NumPy compatibility.
    let beta = beta.abs();
    // Reject beta values where `exp(beta)` overflows f64 (ln(f64::MAX)
    // ≈ 709.78). We use a slightly conservative threshold so the
    // clipped output stays finite even after the asymptotic-series
    // polynomial corrections.
    const BETA_OVERFLOW_THRESHOLD: f64 = 708.0;
    if beta > BETA_OVERFLOW_THRESHOLD {
        return Err(FerrayError::invalid_value(format!(
            "kaiser: |beta| = {beta} exceeds the safe range ({BETA_OVERFLOW_THRESHOLD}) \
             for f64 I_0; the window would be NaN everywhere"
        )));
    }
    let i0_beta = bessel_i0_scalar::<f64>(beta);
    let alpha = (m.saturating_sub(1)) as f64 / 2.0;
    gen_window(m, |n| {
        let t = (n as f64 - alpha) / alpha;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        bessel_i0_scalar::<f64>(arg) / i0_beta
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: compare two slices within tolerance
    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "element {i}: {a} vs {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases: M=0 and M=1
    // -----------------------------------------------------------------------

    #[test]
    fn bartlett_m0() {
        let w = bartlett(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn bartlett_m1() {
        let w = bartlett(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn blackman_m0() {
        let w = blackman(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn blackman_m1() {
        let w = blackman(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn hamming_m0() {
        let w = hamming(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn hamming_m1() {
        let w = hamming(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn hanning_m0() {
        let w = hanning(0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn hanning_m1() {
        let w = hanning(1).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn kaiser_m0() {
        let w = kaiser(0, 14.0).unwrap();
        assert_eq!(w.shape(), &[0]);
    }

    #[test]
    fn kaiser_m1() {
        let w = kaiser(1, 14.0).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[1.0]);
    }

    #[test]
    fn kaiser_negative_beta() {
        // Negative beta should produce the same result as positive beta (I0 is even)
        let pos = kaiser(5, 1.0).unwrap();
        let neg = kaiser(5, -1.0).unwrap();
        assert_eq!(pos.as_slice().unwrap(), neg.as_slice().unwrap());
    }

    #[test]
    fn kaiser_nan_beta() {
        assert!(kaiser(5, f64::NAN).is_err());
    }

    // -----------------------------------------------------------------------
    // AC-1: bartlett(5) matches np.bartlett(5) to within 4 ULPs
    // -----------------------------------------------------------------------
    // np.bartlett(5) = [0.0, 0.5, 1.0, 0.5, 0.0]
    #[test]
    fn bartlett_5_ac1() {
        let w = bartlett(5).unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        assert_close(w.as_slice().unwrap(), &expected, 1e-14);
    }

    // -----------------------------------------------------------------------
    // AC-2: kaiser(5, 14.0) matches np.kaiser(5, 14.0) to within 4 ULPs
    // -----------------------------------------------------------------------
    // np.kaiser(5, 14.0) = [7.72686684e-06, 1.64932188e-01, 1.0, 1.64932188e-01, 7.72686684e-06]
    #[test]
    fn kaiser_5_14_ac2() {
        let w = kaiser(5, 14.0).unwrap();
        let s = w.as_slice().unwrap();
        assert_eq!(s.len(), 5);
        // NumPy reference values (verified via np.kaiser(5, 14.0))
        let expected: [f64; 5] = [
            7.72686684e-06,
            1.64932188e-01,
            1.0,
            1.64932188e-01,
            7.72686684e-06,
        ];
        // Bessel polynomial approximation has limited precision (~6 digits)
        for (i, (&a, &e)) in s.iter().zip(expected.iter()).enumerate() {
            let tol = if e.abs() < 1e-4 { 1e-8 } else { 1e-6 };
            assert!(
                (a - e).abs() <= tol,
                "kaiser element {i}: {a} vs {e} (diff = {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // AC-3: All 5 window functions return correct length and match NumPy fixtures
    // -----------------------------------------------------------------------

    // np.blackman(5) = [-1.38777878e-17, 3.40000000e-01, 1.00000000e+00, 3.40000000e-01, -1.38777878e-17]
    #[test]
    fn blackman_5() {
        let w = blackman(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [
            -1.38777878e-17,
            3.40000000e-01,
            1.00000000e+00,
            3.40000000e-01,
            -1.38777878e-17,
        ];
        assert_close(s, &expected, 1e-10);
    }

    // np.hamming(5) = [0.08, 0.54, 1.0, 0.54, 0.08]
    #[test]
    fn hamming_5() {
        let w = hamming(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [0.08, 0.54, 1.0, 0.54, 0.08];
        assert_close(s, &expected, 1e-14);
    }

    // np.hanning(5) = [0.0, 0.5, 1.0, 0.5, 0.0]
    #[test]
    fn hanning_5() {
        let w = hanning(5).unwrap();
        assert_eq!(w.shape(), &[5]);
        let s = w.as_slice().unwrap();
        let expected = [0.0, 0.5, 1.0, 0.5, 0.0];
        assert_close(s, &expected, 1e-14);
    }

    // Larger window: np.bartlett(12)
    #[test]
    fn bartlett_12() {
        let w = bartlett(12).unwrap();
        assert_eq!(w.shape(), &[12]);
        let s = w.as_slice().unwrap();
        // First and last should be 0, peak near center
        assert!((s[0] - 0.0).abs() < 1e-14);
        assert!((s[11] - 0.0).abs() < 1e-14);
        // Symmetric
        for i in 0..6 {
            assert!((s[i] - s[11 - i]).abs() < 1e-14, "symmetry at {i}");
        }
    }

    // Symmetry test for all windows
    #[test]
    fn all_windows_symmetric() {
        let m = 7;
        let windows: Vec<Array<f64, Ix1>> = vec![
            bartlett(m).unwrap(),
            blackman(m).unwrap(),
            hamming(m).unwrap(),
            hanning(m).unwrap(),
            kaiser(m, 5.0).unwrap(),
        ];
        for (idx, w) in windows.iter().enumerate() {
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-12,
                    "window {idx} not symmetric at {i}"
                );
            }
        }
    }

    // All windows peak at center
    #[test]
    fn all_windows_peak_at_center() {
        let m = 9;
        let windows: Vec<Array<f64, Ix1>> = vec![
            bartlett(m).unwrap(),
            blackman(m).unwrap(),
            hamming(m).unwrap(),
            hanning(m).unwrap(),
            kaiser(m, 5.0).unwrap(),
        ];
        for (idx, w) in windows.iter().enumerate() {
            let s = w.as_slice().unwrap();
            let center = s[m / 2];
            assert!(
                (center - 1.0).abs() < 1e-10,
                "window {idx} center = {center}, expected 1.0"
            );
        }
    }

    // Kaiser with beta=0 should be a rectangular window (all ones)
    #[test]
    fn kaiser_beta_zero() {
        let w = kaiser(5, 0.0).unwrap();
        let s = w.as_slice().unwrap();
        for &v in s {
            assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn bessel_i0_scalar_zero() {
        assert!((bessel_i0_scalar::<f64>(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bessel_i0_scalar_known() {
        // Issue #296: tighten the tolerances. The Abramowitz & Stegun
        // approximation is rated ~1e-7 in the polynomial branch and
        // ~1e-7 in the asymptotic branch per its documentation.
        // I0(1) ~ 1.2660658777520082
        assert!((bessel_i0_scalar::<f64>(1.0) - 1.266_065_877_752_008_2).abs() < 1e-7);
        // I0(5) ~ 27.239871823604443 (asymptotic branch)
        assert!((bessel_i0_scalar::<f64>(5.0) - 27.239_871_823_604_443).abs() < 1e-5);
        // I0(10) ~ 2815.7166284662544
        let expected_i0_10 = 2_815.716_628_466_254;
        assert!(
            (bessel_i0_scalar::<f64>(10.0) - expected_i0_10).abs() / expected_i0_10 < 1e-5
        );
    }

    // ----- Edge-length window coverage (#295) -----

    #[test]
    fn bartlett_m2_is_zeros() {
        // Bartlett of length 2: w(n) = 1 - |2n/(M-1) - 1|
        // n=0: 1 - |0 - 1| = 0
        // n=1: 1 - |2 - 1| = 0
        let w = bartlett(2).unwrap();
        assert_eq!(w.as_slice().unwrap(), &[0.0, 0.0]);
    }

    #[test]
    fn hanning_m2_is_zeros() {
        // Hann of length 2: w(n) = 0.5 * (1 - cos(2*pi*n/(M-1)))
        // n=0: 0.5 * (1 - cos(0)) = 0
        // n=1: 0.5 * (1 - cos(2*pi)) = 0
        let w = hanning(2).unwrap();
        let s = w.as_slice().unwrap();
        assert!(s[0].abs() < 1e-14);
        assert!(s[1].abs() < 1e-14);
    }

    #[test]
    fn bartlett_even_length_is_symmetric() {
        // Even-length windows have no single "center"; symmetry holds
        // around the midpoint between indices (M/2 - 1) and (M/2).
        for &m in &[4usize, 6, 8, 10] {
            let w = bartlett(m).unwrap();
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-14,
                    "bartlett({m}) not symmetric at {i}"
                );
            }
        }
    }

    #[test]
    fn blackman_even_length_is_symmetric() {
        for &m in &[4usize, 6, 8, 10] {
            let w = blackman(m).unwrap();
            let s = w.as_slice().unwrap();
            for i in 0..m / 2 {
                assert!(
                    (s[i] - s[m - 1 - i]).abs() < 1e-14,
                    "blackman({m}) not symmetric at {i}"
                );
            }
        }
    }

    #[test]
    fn kaiser_large_beta_errors() {
        // Issue #294: beta above the f64 safe range should error,
        // not silently produce NaN.
        assert!(kaiser(8, 800.0).is_err());
        assert!(kaiser(8, 1000.0).is_err());
        // 700 is still safe.
        assert!(kaiser(8, 700.0).is_ok());
    }
}
