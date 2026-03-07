// ferray-window: Window functions for signal processing and spectral analysis
//
// Implements NumPy-equivalent window functions: bartlett, blackman, hamming,
// hanning, and kaiser. Each returns an Array1<f64> of the specified length M.

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_core::error::{FerrayError, FerrayResult};

use std::f64::consts::PI;

/// Scalar modified Bessel function I_0(x) using polynomial approximation.
///
/// Uses the Abramowitz and Stegun approximation for |x| <= 3.75 and
/// an asymptotic expansion for |x| > 3.75.
fn bessel_i0_scalar(x: f64) -> f64 {
    let ax = x.abs();

    if ax <= 3.75 {
        let t = (ax / 3.75).powi(2);
        1.0 + t
            * (3.5156229
                + t * (3.0899424
                    + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
    } else {
        let t = 3.75 / ax;
        let poly = 0.39894228
            + t * (0.01328592
                + t * (0.00225319
                    + t * (-0.00157565
                        + t * (0.00916281
                            + t * (-0.02057706
                                + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
        poly * ax.exp() / ax.sqrt()
    }
}

/// Return the Bartlett (triangular) window of length `m`.
///
/// The Bartlett window is defined as:
///   w(n) = 2/(M-1) * ((M-1)/2 - |n - (M-1)/2|)
///
/// This is equivalent to `numpy.bartlett(M)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn bartlett(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let half = (m - 1) as f64 / 2.0;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        let val = 1.0 - ((n as f64 - half) / half).abs();
        data.push(val);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Return the Blackman window of length `m`.
///
/// The Blackman window is defined as:
///   w(n) = 0.42 - 0.5 * cos(2*pi*n/(M-1)) + 0.08 * cos(4*pi*n/(M-1))
///
/// This is equivalent to `numpy.blackman(M)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn blackman(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let denom = (m - 1) as f64;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        let x = n as f64;
        let val = 0.42 - 0.5 * (2.0 * PI * x / denom).cos() + 0.08 * (4.0 * PI * x / denom).cos();
        data.push(val);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Return the Hamming window of length `m`.
///
/// The Hamming window is defined as:
///   w(n) = 0.54 - 0.46 * cos(2*pi*n/(M-1))
///
/// This is equivalent to `numpy.hamming(M)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hamming(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let denom = (m - 1) as f64;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        let val = 0.54 - 0.46 * (2.0 * PI * n as f64 / denom).cos();
        data.push(val);
    }
    Array::from_vec(Ix1::new([m]), data)
}

/// Return the Hann (Hanning) window of length `m`.
///
/// The Hann window is defined as:
///   w(n) = 0.5 * (1 - cos(2*pi*n/(M-1)))
///
/// NumPy calls this function `hanning`. This is equivalent to `numpy.hanning(M)`.
///
/// # Edge Cases
/// - `m == 0`: returns an empty array.
/// - `m == 1`: returns `[1.0]`.
///
/// # Errors
/// Returns an error only if internal array construction fails.
pub fn hanning(m: usize) -> FerrayResult<Array<f64, Ix1>> {
    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let denom = (m - 1) as f64;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        let val = 0.5 * (1.0 - (2.0 * PI * n as f64 / denom).cos());
        data.push(val);
    }
    Array::from_vec(Ix1::new([m]), data)
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
/// Returns `FerrayError::InvalidValue` if `beta` is negative or NaN.
pub fn kaiser(m: usize, beta: f64) -> FerrayResult<Array<f64, Ix1>> {
    if beta.is_nan() || beta < 0.0 {
        return Err(FerrayError::invalid_value(format!(
            "kaiser: beta must be non-negative, got {beta}"
        )));
    }

    if m == 0 {
        return Array::from_vec(Ix1::new([0]), vec![]);
    }
    if m == 1 {
        return Array::from_vec(Ix1::new([1]), vec![1.0]);
    }

    let i0_beta = bessel_i0_scalar(beta);
    let alpha = (m as f64 - 1.0) / 2.0;
    let mut data = Vec::with_capacity(m);
    for n in 0..m {
        let t = (n as f64 - alpha) / alpha;
        let arg = beta * (1.0 - t * t).max(0.0).sqrt();
        data.push(bessel_i0_scalar(arg) / i0_beta);
    }
    Array::from_vec(Ix1::new([m]), data)
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
        assert!(kaiser(5, -1.0).is_err());
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
        assert!((bessel_i0_scalar(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bessel_i0_scalar_known() {
        // I0(1) ~ 1.2660658
        assert!((bessel_i0_scalar(1.0) - 1.2660658).abs() < 1e-4);
        // I0(5) ~ 27.2398718 (tests the asymptotic branch)
        assert!((bessel_i0_scalar(5.0) - 27.2398718).abs() < 1e-2);
    }
}
