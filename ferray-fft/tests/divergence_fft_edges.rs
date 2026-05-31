//! ACToR critic divergence probes for the ferray-fft LIBRARY kernels.
//!
//! Each expected value below is captured from the numpy 2.4.4 oracle
//! (`ferray-python/.venv/bin/python`, R-CHAR-3) at full f64 precision —
//! never copied from the ferray side. These probe the norm modes
//! (forward/ortho), n truncation/padding, and irfft output length that
//! the conformance suites (`conformance_complex.rs`, `conformance_real.rs`)
//! do NOT exercise (those only test `FftNorm::Backward` with default `n`).
//!
//! VERDICT (critic audit 2026-05-31): all probes below PASS — ferray-fft
//! matches numpy to f64 precision on every audited edge. This file is the
//! positive coverage artifact extending the conformance suites into the
//! norm/n/irfft-length corners; there is NO divergence to pin here.
//!
//! Upstream: `numpy/fft/_pocketfft.py` (fft/ifft/irfft norm + n handling).

#![allow(clippy::float_cmp, clippy::cast_precision_loss)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use ferray_fft::FftNorm;
use num_complex::Complex;

const REL: f64 = 1e-10;
const ABS: f64 = 1e-12;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn assert_close_c(got: &[Complex<f64>], expected: &[Complex<f64>], label: &str) {
    assert_eq!(
        got.len(),
        expected.len(),
        "{label}: length mismatch got {} expected {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let dr = (g.re - e.re).abs();
        let di = (g.im - e.im).abs();
        let tr = ABS + REL * e.re.abs();
        let ti = ABS + REL * e.im.abs();
        assert!(
            dr <= tr && di <= ti,
            "{label}[{i}]: got {g:?} expected {e:?} (dr={dr}, di={di})"
        );
    }
}

fn assert_close_f(got: &[f64], expected: &[f64], label: &str) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let d = (g - e).abs();
        let t = ABS + REL * e.abs();
        assert!(d <= t, "{label}[{i}]: got {g} expected {e} (d={d})");
    }
}

/// Probe: 1-D `fft` with `norm='forward'`.
/// Oracle: `np.fft.fft([1,2,3,4], norm='forward')`
///   = [2.5, -0.5+0.5j, -0.5, -0.5-0.5j]
/// (forward norm scales the forward transform by 1/n).
#[test]
fn divergence_fft_forward_norm() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::fft(&a, None, None, FftNorm::Forward).unwrap();
    let expected = [c(2.5, 0.0), c(-0.5, 0.5), c(-0.5, 0.0), c(-0.5, -0.5)];
    assert_close_c(r.as_slice().unwrap(), &expected, "fft forward");
}

/// Probe: 1-D `fft` with `norm='ortho'`.
/// Oracle: `np.fft.fft([1,2,3,4], norm='ortho')`
///   = [5, -1+1j, -1, -1-1j]  (1/sqrt(4) = 0.5 scaling).
#[test]
fn divergence_fft_ortho_norm() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::fft(&a, None, None, FftNorm::Ortho).unwrap();
    let expected = [c(5.0, 0.0), c(-1.0, 1.0), c(-1.0, 0.0), c(-1.0, -1.0)];
    assert_close_c(r.as_slice().unwrap(), &expected, "fft ortho");
}

/// Probe: 1-D `ifft` with `norm='forward'`.
/// Oracle: `np.fft.ifft([1,2,3,4], norm='forward')`
///   = [10, -2-2j, -2, -2+2j]  (forward norm => no scaling on inverse).
#[test]
fn divergence_ifft_forward_norm() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::ifft(&a, None, None, FftNorm::Forward).unwrap();
    let expected = [c(10.0, 0.0), c(-2.0, -2.0), c(-2.0, 0.0), c(-2.0, 2.0)];
    assert_close_c(r.as_slice().unwrap(), &expected, "ifft forward");
}

/// Probe: 1-D `ifft` with `norm='ortho'`.
/// Oracle: `np.fft.ifft([1,2,3,4], norm='ortho')`
///   = [5, -1-1j, -1, -1+1j].
#[test]
fn divergence_ifft_ortho_norm() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::ifft(&a, None, None, FftNorm::Ortho).unwrap();
    let expected = [c(5.0, 0.0), c(-1.0, -1.0), c(-1.0, 0.0), c(-1.0, 1.0)];
    assert_close_c(r.as_slice().unwrap(), &expected, "ifft ortho");
}

/// Probe: `fft` with `n` < length (truncation).
/// Oracle: `np.fft.fft([1,2,3,4], n=2)` = [3, -1].
#[test]
fn divergence_fft_n_truncate() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::fft(&a, Some(2), None, FftNorm::Backward).unwrap();
    let expected = [c(3.0, 0.0), c(-1.0, 0.0)];
    assert_close_c(r.as_slice().unwrap(), &expected, "fft n=2 truncate");
}

/// Probe: `fft` with `n` > length (zero-padding).
/// Oracle: `np.fft.fft([1,2,3,4], n=6)`
///   = [10, -3.5-4.330127018922193j, 2.5+0.8660254037844386j,
///       -2, 2.5-0.8660254037844386j, -3.5+4.330127018922193j].
#[test]
fn divergence_fft_n_pad() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([4]),
        vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::fft(&a, Some(6), None, FftNorm::Backward).unwrap();
    let expected = [
        c(10.0, 0.0),
        c(-3.5, -4.330_127_018_922_193),
        c(2.5, 0.866_025_403_784_438_6),
        c(-2.0, 0.0),
        c(2.5, -0.866_025_403_784_438_6),
        c(-3.5, 4.330_127_018_922_193),
    ];
    assert_close_c(r.as_slice().unwrap(), &expected, "fft n=6 pad");
}

/// Probe: `irfft` default output length from a 3-bin spectrum.
/// Oracle: `np.fft.irfft([10, -2+2j, -2])` = [1, 2, 3, 4]  (length 2*(3-1)=4).
#[test]
fn divergence_irfft_default_len() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([3]),
        vec![c(10.0, 0.0), c(-2.0, 2.0), c(-2.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::irfft(&a, None, None, FftNorm::Backward).unwrap();
    let expected = [1.0, 2.0, 3.0, 4.0];
    assert_close_f(r.as_slice().unwrap(), &expected, "irfft default len");
}

/// Probe: `irfft` with explicit odd `n`.
/// Oracle (full f64 precision): `np.fft.irfft([10, -2+2j, -2], n=5)`
///   = [0.4, 1.6391547869638772, 1.9297717981660214,
///      2.870228201833979, 3.160845213036123].
#[test]
fn divergence_irfft_n_odd() {
    let a = Array::<Complex<f64>, Ix1>::from_vec(
        Ix1::new([3]),
        vec![c(10.0, 0.0), c(-2.0, 2.0), c(-2.0, 0.0)],
    )
    .unwrap();
    let r = ferray_fft::irfft(&a, Some(5), None, FftNorm::Backward).unwrap();
    let expected = [
        0.4,
        1.639_154_786_963_877_2,
        1.929_771_798_166_021_4,
        2.870_228_201_833_979,
        3.160_845_213_036_123,
    ];
    assert_close_f(r.as_slice().unwrap(), &expected, "irfft n=5");
}
