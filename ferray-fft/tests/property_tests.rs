// Property-based tests for ferray-fft
//
// Tests mathematical invariants of FFT operations using proptest.

// Property tests sample integer sizes and lift them into f64 frequency
// indices; FFT roundtrip identities assert exact float equality on
// known-zero imaginary parts.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::float_cmp
)]

use ferray_core::Array;
use ferray_core::dimension::Ix1;
use num_complex::Complex;

use ferray_fft::norm::FftNorm;
use ferray_fft::{fft, fftfreq, fftshift, ifft, ifftshift};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn complex_arr(data: Vec<Complex<f64>>) -> Array<Complex<f64>, Ix1> {
    let n = data.len();
    Array::<Complex<f64>, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

fn real_to_complex(data: &[f64]) -> Vec<Complex<f64>> {
    data.iter().map(|&x| Complex::new(x, 0.0)).collect()
}

fn complex_approx_eq(a: &[Complex<f64>], b: &[Complex<f64>], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).norm() < tol)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. Round-trip: ifft(fft(x)) ~= x
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fft_ifft_roundtrip(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data.clone());

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();
        let recovered = ifft(&transformed, None, None, FftNorm::Backward).unwrap();

        let recovered_data: Vec<Complex<f64>> = recovered.iter().copied().collect();
        prop_assert!(
            complex_approx_eq(&complex_data, &recovered_data, 1e-8),
            "ifft(fft(x)) != x for n={}",
            n
        );
    }

    // -----------------------------------------------------------------------
    // 2. Linearity: fft(a*x + b*y) ~= a*fft(x) + b*fft(y)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fft_linearity(
        x_data in proptest::collection::vec(-5.0f64..5.0, 8),
        y_data in proptest::collection::vec(-5.0f64..5.0, 8),
        a in -5.0f64..5.0,
        b in -5.0f64..5.0,
    ) {
        let x_c = real_to_complex(&x_data);
        let y_c = real_to_complex(&y_data);

        // a*x + b*y
        let combo: Vec<Complex<f64>> = x_c.iter().zip(y_c.iter())
            .map(|(&xi, &yi)| Complex::new(a, 0.0) * xi + Complex::new(b, 0.0) * yi)
            .collect();

        let x_arr = complex_arr(x_c);
        let y_arr = complex_arr(y_c);
        let combo_arr = complex_arr(combo);

        let fft_combo = fft(&combo_arr, None, None, FftNorm::Backward).unwrap();
        let fft_x = fft(&x_arr, None, None, FftNorm::Backward).unwrap();
        let fft_y = fft(&y_arr, None, None, FftNorm::Backward).unwrap();

        // a*fft(x) + b*fft(y)
        let expected: Vec<Complex<f64>> = fft_x.iter().zip(fft_y.iter())
            .map(|(&fx, &fy)| Complex::new(a, 0.0) * fx + Complex::new(b, 0.0) * fy)
            .collect();

        let actual: Vec<Complex<f64>> = fft_combo.iter().copied().collect();

        prop_assert!(
            complex_approx_eq(&actual, &expected, 1e-6),
            "FFT not linear"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Parseval's theorem: sum(|x|^2) ~= sum(|fft(x)|^2) / N
    // -----------------------------------------------------------------------
    #[test]
    fn prop_parseval(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data.clone());

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();

        let energy_time: f64 = complex_data.iter().map(num_complex::Complex::norm_sqr).sum();
        let energy_freq: f64 = transformed.iter().map(num_complex::Complex::norm_sqr).sum();

        let ratio = energy_freq / n as f64;
        let diff = (energy_time - ratio).abs();
        let scale = energy_time.abs().max(1.0);

        prop_assert!(
            diff / scale < 1e-6,
            "Parseval violated: time_energy={}, freq_energy/N={}, diff={}",
            energy_time, ratio, diff
        );
    }

    // -----------------------------------------------------------------------
    // 4. Real input symmetry: fft of real data has conjugate symmetry
    // -----------------------------------------------------------------------
    #[test]
    fn prop_real_input_conjugate_symmetry(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data);

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();
        let fft_data: Vec<Complex<f64>> = transformed.iter().copied().collect();

        // For real input: X[k] = conj(X[N-k])
        for k in 1..n / 2 {
            let xk = fft_data[k];
            let xnk = fft_data[n - k];
            let diff = (xk - xnk.conj()).norm();
            prop_assert!(
                diff < 1e-8,
                "Conjugate symmetry violated at k={}: X[k]={:?}, conj(X[N-k])={:?}",
                k, xk, xnk.conj()
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. DC component: fft(x)[0] == sum(x)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_dc_component(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data.clone());

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();
        let dc = *transformed.iter().next().unwrap();

        let expected: Complex<f64> = complex_data.iter().sum();
        let diff = (dc - expected).norm();

        prop_assert!(
            diff < 1e-8,
            "DC component: fft[0]={:?}, sum(x)={:?}, diff={}",
            dc, expected, diff
        );
    }

    // -----------------------------------------------------------------------
    // 6. fftshift/ifftshift roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fftshift_ifftshift_roundtrip(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data.clone());

        let shifted = fftshift(&input, None).unwrap();
        let unshifted = ifftshift(&shifted, None).unwrap();

        let result: Vec<Complex<f64>> = unshifted.iter().copied().collect();

        prop_assert!(
            complex_approx_eq(&complex_data, &result, 1e-15),
            "ifftshift(fftshift(x)) != x"
        );
    }

    // -----------------------------------------------------------------------
    // 7. fftfreq has correct length and DC at index 0
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fftfreq_properties(n in 2usize..=64) {
        let freqs = fftfreq(n, 1.0).unwrap();
        prop_assert_eq!(freqs.size(), n);

        let data: Vec<f64> = freqs.iter().copied().collect();
        // DC frequency should be 0
        prop_assert!(
            data[0].abs() < 1e-15,
            "fftfreq[0] = {} should be 0",
            data[0]
        );
    }

    // -----------------------------------------------------------------------
    // 8. FFT of constant signal: only DC component is nonzero
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fft_constant_signal(c in -10.0f64..10.0, n in 2usize..=32) {
        let complex_data: Vec<Complex<f64>> = vec![Complex::new(c, 0.0); n];
        let input = complex_arr(complex_data);

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();
        let fft_data: Vec<Complex<f64>> = transformed.iter().copied().collect();

        // DC component should be n * c
        let dc = fft_data[0];
        prop_assert!(
            (dc - Complex::new(c * n as f64, 0.0)).norm() < 1e-8,
            "DC of constant {} signal with n={}: got {:?}, expected {}",
            c, n, dc, c * n as f64
        );

        // All other components should be ~0
        for (k, &val) in fft_data.iter().enumerate().skip(1) {
            prop_assert!(
                val.norm() < 1e-8,
                "Non-DC component fft[{}] = {:?} should be ~0 for constant signal",
                k, val
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. FFT preserves output length
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fft_preserves_length(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data);

        let transformed = fft(&input, None, None, FftNorm::Backward).unwrap();
        prop_assert_eq!(transformed.size(), n);
    }

    // -----------------------------------------------------------------------
    // 10. Ortho normalization: fft with ortho norm preserves energy exactly
    // -----------------------------------------------------------------------
    #[test]
    fn prop_fft_ortho_preserves_energy(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let complex_data = real_to_complex(&data);
        let input = complex_arr(complex_data.clone());

        let transformed = fft(&input, None, None, FftNorm::Ortho).unwrap();

        let energy_time: f64 = complex_data.iter().map(num_complex::Complex::norm_sqr).sum();
        let energy_freq: f64 = transformed.iter().map(num_complex::Complex::norm_sqr).sum();

        let diff = (energy_time - energy_freq).abs();
        let scale = energy_time.abs().max(1.0);

        prop_assert!(
            diff / scale < 1e-6,
            "Ortho norm should preserve energy: time={}, freq={}, diff={}",
            energy_time, energy_freq, diff
        );
    }

    // --- rfft/irfft property tests ---

    #[test]
    fn prop_rfft_irfft_roundtrip(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let arr = ferray_core::Array::<f64, ferray_core::Ix1>::from_vec(
            ferray_core::Ix1::new([n]),
            data.clone(),
        ).unwrap();

        let spectrum = ferray_fft::rfft(&arr, None, None, FftNorm::Backward).unwrap();
        let recovered = ferray_fft::irfft(&spectrum, Some(n), None, FftNorm::Backward).unwrap();

        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (orig - rec).abs();
            prop_assert!(
                diff < 1e-8,
                "rfft/irfft roundtrip failed at index {}: {} vs {} (diff={})",
                i, orig, rec, diff
            );
        }
    }

    #[test]
    fn prop_rfft_output_length(
        n in 2usize..=64,
    ) {
        let data: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let arr = ferray_core::Array::<f64, ferray_core::Ix1>::from_vec(
            ferray_core::Ix1::new([n]),
            data,
        ).unwrap();

        let spectrum = ferray_fft::rfft(&arr, None, None, FftNorm::Backward).unwrap();
        // rfft output should have n/2+1 elements
        let expected_len = n / 2 + 1;
        prop_assert_eq!(spectrum.shape()[0], expected_len);
    }

    #[test]
    fn prop_rfft_dc_component(
        data in proptest::collection::vec(-10.0f64..10.0, 4..=32),
    ) {
        let n = data.len();
        let arr = ferray_core::Array::<f64, ferray_core::Ix1>::from_vec(
            ferray_core::Ix1::new([n]),
            data.clone(),
        ).unwrap();

        let spectrum = ferray_fft::rfft(&arr, None, None, FftNorm::Backward).unwrap();
        // DC component (index 0) should equal the sum of input
        let expected_dc: f64 = data.iter().sum();
        let dc = spectrum.iter().next().unwrap();
        prop_assert!(
            (dc.re - expected_dc).abs() < 1e-8,
            "DC component: expected {}, got {}",
            expected_dc, dc.re
        );
        prop_assert!(
            dc.im.abs() < 1e-8,
            "DC imaginary should be 0, got {}",
            dc.im
        );
    }
}
