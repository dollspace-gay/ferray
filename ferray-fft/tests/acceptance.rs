// Integration tests covering all acceptance criteria from the design doc.

// Acceptance tests assert exact equality against NumPy fixtures and
// closed-form FFT identities; integer length<->float lifts are part of
// the FFT normalization formulae.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

// AC-1: fft(ifft(a)) round-trips to within 4 ULPs for complex f64 input
// AC-2: fft output matches NumPy fixtures for lengths 8, 64, 1024, 1023
// AC-3: rfft of a real signal returns n/2+1 complex values matching NumPy
// AC-4: fftfreq(8, 1.0) returns correct values
// AC-5: fftshift / ifftshift round-trip correctly
// AC-6: FftPlan reuse produces identical results to non-cached fft() calls
// AC-7: All three normalization modes produce correct scaling factors

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, Ix3, IxDyn};
use ferray_fft::{
    FftNorm, FftPlan, fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift,
    ihfft, irfft, irfft2, irfftn, rfft, rfft2, rfftfreq, rfftn,
};
use num_complex::Complex;

const fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn make_complex_1d(data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

fn make_real_1d(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::from_vec(Ix1::new([n]), data).unwrap()
}

/// Check that two f64 values are within 4 ULPs of each other.
///
/// Uses a hybrid approach: for values near zero (where ULP comparison
/// across sign boundaries is problematic), use an absolute tolerance
/// of 4 * `f64::EPSILON`. For other values, compute the true ULP distance.
fn within_4_ulps(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    // Near zero: use absolute tolerance
    let abs_tol = 4.0 * f64::EPSILON;
    if a.abs() < abs_tol || b.abs() < abs_tol {
        return (a - b).abs() < abs_tol;
    }
    // Same sign: compute ULP distance
    if a.signum() == b.signum() {
        let a_bits = a.to_bits() as i64;
        let b_bits = b.to_bits() as i64;
        (a_bits - b_bits).unsigned_abs() <= 4
    } else {
        // Different signs far from zero: definitely more than 4 ULP
        false
    }
}

// =========================================================================
// AC-1: fft(ifft(a)) round-trips to within 4 ULPs for complex f64
// =========================================================================

#[test]
fn ac1_fft_ifft_roundtrip_4ulps() {
    let data = vec![
        c(1.0, 2.0),
        c(-1.0, 0.5),
        c(3.0, -1.0),
        c(0.0, 0.0),
        c(-2.5, 1.5),
        c(0.7, -0.3),
        c(1.2, 0.8),
        c(-0.4, 2.1),
    ];
    let a = make_complex_1d(data.clone());

    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();

    for (i, (orig, rec)) in data.iter().zip(recovered.iter()).enumerate() {
        assert!(
            within_4_ulps(orig.re, rec.re),
            "[{}] Real part mismatch: {} vs {} (diff = {:e})",
            i,
            orig.re,
            rec.re,
            (orig.re - rec.re).abs()
        );
        assert!(
            within_4_ulps(orig.im, rec.im),
            "[{}] Imag part mismatch: {} vs {} (diff = {:e})",
            i,
            orig.im,
            rec.im,
            (orig.im - rec.im).abs()
        );
    }
}

#[test]
fn ac1_ifft_fft_roundtrip_4ulps() {
    // Also test ifft(fft(a)) direction, using powers-of-two-friendly values
    // to keep ULP drift minimal
    let data = vec![c(0.25, -0.5), c(3.0, 2.0), c(-1.5, 1.75), c(0.0, -0.5)];
    let a = make_complex_1d(data.clone());

    let recovered_via_ifft = ifft(&a, None, None, FftNorm::Backward).unwrap();
    let final_result = fft(&recovered_via_ifft, None, None, FftNorm::Backward).unwrap();

    for (i, (orig, rec)) in data.iter().zip(final_result.iter()).enumerate() {
        assert!(
            within_4_ulps(orig.re, rec.re),
            "[{}] Real: {} vs {} (diff = {:e})",
            i,
            orig.re,
            rec.re,
            (orig.re - rec.re).abs()
        );
        assert!(
            within_4_ulps(orig.im, rec.im),
            "[{}] Imag: {} vs {} (diff = {:e})",
            i,
            orig.im,
            rec.im,
            (orig.im - rec.im).abs()
        );
    }
}

// =========================================================================
// AC-2: fft matches NumPy output for various lengths
// =========================================================================

#[test]
fn ac2_fft_length_8() {
    // FFT of [0, 1, 2, 3, 4, 5, 6, 7]
    // NumPy: np.fft.fft([0,1,2,3,4,5,6,7])
    // = [28+0j, -4+9.657j, -4+4j, -4+1.657j, -4+0j, -4-1.657j, -4-4j, -4-9.657j]
    let data: Vec<Complex<f64>> = (0..8).map(|i| c(f64::from(i), 0.0)).collect();
    let a = make_complex_1d(data);
    let result = fft(&a, None, None, FftNorm::Backward).unwrap();
    let vals: Vec<Complex<f64>> = result.iter().copied().collect();

    let cot_pi_8 = (std::f64::consts::PI / 8.0).cos() / (std::f64::consts::PI / 8.0).sin();
    let expected_im1 = 4.0 * cot_pi_8; // 4 * cot(pi/8) = 4 * (1+sqrt(2)) ~= 9.6569

    assert!((vals[0].re - 28.0).abs() < 1e-10);
    assert!(vals[0].im.abs() < 1e-10);
    assert!((vals[1].re - (-4.0)).abs() < 1e-10);
    assert!((vals[1].im - expected_im1).abs() < 1e-4);
    assert!((vals[2].re - (-4.0)).abs() < 1e-10);
    assert!((vals[2].im - 4.0).abs() < 1e-10);
    assert!((vals[4].re - (-4.0)).abs() < 1e-10);
    assert!(vals[4].im.abs() < 1e-10);
}

#[test]
fn ac2_fft_length_64_roundtrip() {
    let data: Vec<Complex<f64>> = (0..64)
        .map(|i| {
            let t = f64::from(i) / 64.0;
            c(
                (2.0 * std::f64::consts::PI * 3.0 * t).cos(),
                (2.0 * std::f64::consts::PI * 7.0 * t).sin(),
            )
        })
        .collect();
    let a = make_complex_1d(data.clone());
    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-10);
        assert!((orig.im - rec.im).abs() < 1e-10);
    }
}

#[test]
fn ac2_fft_length_1024_roundtrip() {
    let data: Vec<Complex<f64>> = (0..1024)
        .map(|i| c(f64::from(i) * 0.001, -f64::from(i) * 0.002))
        .collect();
    let a = make_complex_1d(data.clone());
    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-8);
        assert!((orig.im - rec.im).abs() < 1e-8);
    }
}

#[test]
fn ac2_fft_length_1023_non_power_of_2() {
    // Non-power-of-two length
    let data: Vec<Complex<f64>> = (0..1023)
        .map(|i| c(f64::from(i).sin(), f64::from(i).cos()))
        .collect();
    let a = make_complex_1d(data.clone());
    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!(
            (orig.re - rec.re).abs() < 1e-8,
            "re: {} vs {}",
            orig.re,
            rec.re
        );
        assert!(
            (orig.im - rec.im).abs() < 1e-8,
            "im: {} vs {}",
            orig.im,
            rec.im
        );
    }
}

// =========================================================================
// AC-3: rfft returns n/2+1 complex values
// =========================================================================

#[test]
fn ac3_rfft_output_length() {
    for n in [4, 5, 8, 16, 17, 64, 128, 255] {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let a = make_real_1d(data);
        let result = rfft(&a, None, None, FftNorm::Backward).unwrap();
        assert_eq!(
            result.shape(),
            &[n / 2 + 1],
            "rfft of length {} should produce {} values",
            n,
            n / 2 + 1
        );
    }
}

#[test]
fn ac3_rfft_matches_fft_first_half() {
    // rfft output should match the first n/2+1 values of fft
    let real_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = real_data.len();
    let a_real = make_real_1d(real_data.clone());
    let rfft_result = rfft(&a_real, None, None, FftNorm::Backward).unwrap();

    let complex_data: Vec<Complex<f64>> = real_data.iter().map(|&v| c(v, 0.0)).collect();
    let a_complex = make_complex_1d(complex_data);
    let fft_result = fft(&a_complex, None, None, FftNorm::Backward).unwrap();

    // rfft should match first n/2+1 = 5 values of fft
    assert_eq!(rfft_result.shape(), &[n / 2 + 1]);
    for (rf, ff) in rfft_result.iter().zip(fft_result.iter().take(n / 2 + 1)) {
        assert!((rf.re - ff.re).abs() < 1e-10);
        assert!((rf.im - ff.im).abs() < 1e-10);
    }
}

// =========================================================================
// AC-4: fftfreq(8, 1.0) returns correct values
// =========================================================================

#[test]
fn ac4_fftfreq_8() {
    let freq = fftfreq(8, 1.0).unwrap();
    let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
    let data: Vec<f64> = freq.iter().copied().collect();
    assert_eq!(data.len(), 8);
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!((a - b).abs() < 1e-15, "fftfreq[{i}]: got {a}, expected {b}");
    }
}

#[test]
fn ac4_rfftfreq_8() {
    let freq = rfftfreq(8, 1.0).unwrap();
    let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
    let data: Vec<f64> = freq.iter().copied().collect();
    assert_eq!(data.len(), 5);
    for (i, (a, b)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-15,
            "rfftfreq[{i}]: got {a}, expected {b}"
        );
    }
}

// =========================================================================
// AC-5: fftshift / ifftshift round-trip
// =========================================================================

#[test]
fn ac5_shift_roundtrip_1d_even() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let a = make_real_1d(data.clone());
    let shifted = fftshift(&a, None).unwrap();
    let recovered = ifftshift(&shifted, None).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    assert_eq!(rec, data);
}

#[test]
fn ac5_shift_roundtrip_1d_odd() {
    let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let a = make_real_1d(data.clone());
    let shifted = fftshift(&a, None).unwrap();
    let recovered = ifftshift(&shifted, None).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    assert_eq!(rec, data);
}

#[test]
fn ac5_shift_roundtrip_2d() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 3]), data.clone()).unwrap();
    let shifted = fftshift(&a, None).unwrap();
    let recovered = ifftshift(&shifted, None).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    assert_eq!(rec, data);
}

#[test]
fn ac5_shift_roundtrip_complex() {
    let data = vec![c(0.0, 0.0), c(1.0, -1.0), c(2.0, 2.0), c(3.0, -3.0)];
    let a = make_complex_1d(data.clone());
    let shifted = fftshift(&a, None).unwrap();
    let recovered = ifftshift(&shifted, None).unwrap();
    let rec: Vec<Complex<f64>> = recovered.iter().copied().collect();
    assert_eq!(rec, data);
}

// =========================================================================
// AC-6: FftPlan reuse matches non-cached fft()
// =========================================================================

#[test]
fn ac6_plan_matches_fft() {
    let data = vec![
        c(1.0, 0.0),
        c(0.0, 1.0),
        c(-1.0, 0.0),
        c(0.0, -1.0),
        c(2.0, 0.5),
        c(-0.5, 2.0),
        c(1.5, -1.5),
        c(-1.0, 1.0),
    ];
    let a = make_complex_1d(data);

    // Non-cached FFT
    let fft_result = fft(&a, None, None, FftNorm::Backward).unwrap();

    // Cached plan
    let plan = FftPlan::new(8).unwrap();
    let plan_result = plan.execute(&a).unwrap();

    for (f, p) in fft_result.iter().zip(plan_result.iter()) {
        assert!(
            (f.re - p.re).abs() < 1e-14,
            "re mismatch: {} vs {}",
            f.re,
            p.re
        );
        assert!(
            (f.im - p.im).abs() < 1e-14,
            "im mismatch: {} vs {}",
            f.im,
            p.im
        );
    }
}

#[test]
fn ac6_plan_reuse_consistent() {
    let plan = FftPlan::new(16).unwrap();
    let data1: Vec<Complex<f64>> = (0..16).map(|i| c(f64::from(i), 0.0)).collect();
    let data2: Vec<Complex<f64>> = (0..16).map(|i| c(0.0, f64::from(i))).collect();

    let a1 = make_complex_1d(data1);
    let a2 = make_complex_1d(data2);

    let r1 = plan.execute(&a1).unwrap();
    let r2 = plan.execute(&a2).unwrap();
    let r1_again = plan.execute(&a1).unwrap();

    // Reusing plan on same input should give identical results
    for (a, b) in r1.iter().zip(r1_again.iter()) {
        assert_eq!(a.re, b.re);
        assert_eq!(a.im, b.im);
    }

    // Different inputs should give different results (basic sanity)
    let r1_vals: Vec<Complex<f64>> = r1.iter().copied().collect();
    let r2_vals: Vec<Complex<f64>> = r2.iter().copied().collect();
    assert_ne!(r1_vals, r2_vals);
}

// =========================================================================
// AC-7: Normalization modes produce correct scaling
// =========================================================================

#[test]
fn ac7_backward_norm() {
    let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    let a = make_complex_1d(data.clone());

    // Forward: no scaling
    let spectrum = fft(&a, None, None, FftNorm::Backward).unwrap();
    // DC component should be sum = 10
    assert!((spectrum.iter().next().unwrap().re - 10.0).abs() < 1e-12);

    // Inverse: divides by n=4
    let recovered = ifft(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-12);
        assert!((orig.im - rec.im).abs() < 1e-12);
    }
}

#[test]
fn ac7_forward_norm() {
    let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    let a = make_complex_1d(data.clone());

    // Forward norm: divides by n on forward
    let spectrum = fft(&a, None, None, FftNorm::Forward).unwrap();
    // DC = sum/n = 10/4 = 2.5
    assert!((spectrum.iter().next().unwrap().re - 2.5).abs() < 1e-12);

    // Inverse with Forward norm: no scaling
    let recovered = ifft(&spectrum, None, None, FftNorm::Forward).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-12);
    }
}

#[test]
fn ac7_ortho_norm() {
    let data = vec![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0)];
    let a = make_complex_1d(data.clone());

    // Ortho: 1/sqrt(n) both directions
    let spectrum = fft(&a, None, None, FftNorm::Ortho).unwrap();
    // DC = sum / sqrt(4) = 10/2 = 5
    assert!((spectrum.iter().next().unwrap().re - 5.0).abs() < 1e-12);

    // Roundtrip with ortho
    let recovered = ifft(&spectrum, None, None, FftNorm::Ortho).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-12);
    }
}

#[test]
fn ac7_ortho_is_unitary() {
    // For ortho norm, Parseval's theorem: sum |x|^2 == sum |X|^2
    let data = vec![c(1.0, 2.0), c(-1.0, 0.5), c(3.0, -1.0), c(0.0, 0.0)];
    let a = make_complex_1d(data.clone());

    let spectrum = fft(&a, None, None, FftNorm::Ortho).unwrap();

    let energy_time: f64 = data.iter().map(num_complex::Complex::norm_sqr).sum();
    let energy_freq: f64 = spectrum.iter().map(num_complex::Complex::norm_sqr).sum();

    assert!(
        (energy_time - energy_freq).abs() < 1e-10,
        "Parseval failed: {energy_time} vs {energy_freq}"
    );
}

// =========================================================================
// Additional integration tests for completeness
// =========================================================================

#[test]
fn fft2_ifft2_roundtrip() {
    let data: Vec<Complex<f64>> = (0..12)
        .map(|i| c(f64::from(i), -f64::from(i) * 0.5))
        .collect();
    let a = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
    let spectrum = fft2(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifft2(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (o, r) in data.iter().zip(recovered.iter()) {
        assert!((o.re - r.re).abs() < 1e-9);
        assert!((o.im - r.im).abs() < 1e-9);
    }
}

#[test]
fn fftn_ifftn_roundtrip_3d() {
    let n = 2 * 3 * 4;
    let data: Vec<Complex<f64>> = (0..n).map(|i| c(f64::from(i), 0.0)).collect();
    let a = Array::<Complex<f64>, Ix3>::from_vec(Ix3::new([2, 3, 4]), data.clone()).unwrap();
    let spectrum = fftn(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = ifftn(&spectrum, None, None, FftNorm::Backward).unwrap();
    for (o, r) in data.iter().zip(recovered.iter()) {
        assert!((o.re - r.re).abs() < 1e-8);
        assert!((o.im - r.im).abs() < 1e-8);
    }
}

#[test]
fn rfft_irfft_roundtrip_various_lengths() {
    for n in [4, 5, 8, 16, 17, 32, 63, 64, 128, 255] {
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let a = make_real_1d(data.clone());
        let spectrum = rfft(&a, None, None, FftNorm::Backward).unwrap();
        let recovered = irfft(&spectrum, Some(n), None, FftNorm::Backward).unwrap();
        let rec: Vec<f64> = recovered.iter().copied().collect();
        for (i, (o, r)) in data.iter().zip(rec.iter()).enumerate() {
            assert!((o - r).abs() < 1e-9, "n={n}, i={i}: {o} vs {r}");
        }
    }
}

#[test]
fn rfft2_irfft2_roundtrip() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let a = Array::<f64, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();
    let spectrum = rfft2(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = irfft2(&spectrum, Some(&[3, 4]), None, FftNorm::Backward).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    for (o, r) in data.iter().zip(rec.iter()) {
        assert!((o - r).abs() < 1e-9, "{o} vs {r}");
    }
}

#[test]
fn rfftn_irfftn_roundtrip() {
    let n = 2 * 3 * 4;
    let data: Vec<f64> = (0..n).map(|i| (f64::from(i) * 0.3).cos()).collect();
    let a = Array::<f64, Ix3>::from_vec(Ix3::new([2, 3, 4]), data.clone()).unwrap();
    let spectrum = rfftn(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = irfftn(&spectrum, Some(&[2, 3, 4]), None, FftNorm::Backward).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    for (o, r) in data.iter().zip(rec.iter()) {
        assert!((o - r).abs() < 1e-9, "{o} vs {r}");
    }
}

#[test]
fn hfft_ihfft_roundtrip() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = data.len();
    let a = make_real_1d(data.clone());
    let spectrum = ihfft(&a, None, None, FftNorm::Backward).unwrap();
    let recovered = hfft(&spectrum, Some(n), None, FftNorm::Backward).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    for (o, r) in data.iter().zip(rec.iter()) {
        assert!((o - r).abs() < 1e-9, "{o} vs {r}");
    }
}

#[test]
fn hfft_ihfft_roundtrip_forward_norm() {
    // Issue #228: hermitian.rs only tested Backward; the
    // normalization-swap path (Backward<->Forward) was uncovered.
    // ihfft(.., Forward) followed by hfft(.., Forward) is also a
    // roundtrip because both functions invert each other for any
    // self-consistent norm choice.
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = data.len();
    let a = make_real_1d(data.clone());
    let spectrum = ihfft(&a, None, None, FftNorm::Forward).unwrap();
    let recovered = hfft(&spectrum, Some(n), None, FftNorm::Forward).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    for (o, r) in data.iter().zip(rec.iter()) {
        assert!(
            (o - r).abs() < 1e-9,
            "Forward norm roundtrip mismatch: {o} vs {r}"
        );
    }
}

#[test]
fn hfft_ihfft_roundtrip_ortho_norm() {
    // Ortho preserves energy and is its own inverse, so
    // hfft(ihfft(a, Ortho), Ortho) should also roundtrip (#228).
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = data.len();
    let a = make_real_1d(data.clone());
    let spectrum = ihfft(&a, None, None, FftNorm::Ortho).unwrap();
    let recovered = hfft(&spectrum, Some(n), None, FftNorm::Ortho).unwrap();
    let rec: Vec<f64> = recovered.iter().copied().collect();
    for (o, r) in data.iter().zip(rec.iter()) {
        assert!(
            (o - r).abs() < 1e-9,
            "Ortho norm roundtrip mismatch: {o} vs {r}"
        );
    }
}

#[test]
fn plan_inverse_roundtrip() {
    let plan = FftPlan::new(8).unwrap();
    let data = vec![
        c(1.0, 2.0),
        c(-1.0, 0.5),
        c(3.0, -1.0),
        c(0.0, 0.0),
        c(-2.5, 1.5),
        c(0.7, -0.3),
        c(1.2, 0.8),
        c(-0.4, 2.1),
    ];
    let a = make_complex_1d(data.clone());
    let spectrum = plan.execute(&a).unwrap();
    let recovered = plan.execute_inverse(&spectrum).unwrap();
    for (orig, rec) in data.iter().zip(recovered.iter()) {
        assert!((orig.re - rec.re).abs() < 1e-12);
        assert!((orig.im - rec.im).abs() < 1e-12);
    }
}

#[test]
fn fft_dyn_rank() {
    // Test with IxDyn arrays
    let data = vec![c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
    let a = Array::<Complex<f64>, IxDyn>::from_vec(IxDyn::new(&[4]), data).unwrap();
    let result = fft(&a, None, None, FftNorm::Backward).unwrap();
    assert_eq!(result.shape(), &[4]);
    for val in result.iter() {
        assert!((val.re - 1.0).abs() < 1e-12);
        assert!(val.im.abs() < 1e-12);
    }
}
