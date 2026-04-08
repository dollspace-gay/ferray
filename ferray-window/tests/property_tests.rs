// Property-based tests for ferray-window
//
// Tests mathematical invariants of window functions and functional utilities
// using proptest.

use ferray_core::Array;
use ferray_core::dimension::Ix1;

use ferray_window::{bartlett, blackman, hamming, hanning, kaiser, vectorize};

use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn arr1(data: Vec<f64>) -> Array<f64, Ix1> {
    let n = data.len();
    Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. bartlett(m) is symmetric: w[i] == w[m-1-i]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_bartlett_symmetric(m in 2usize..=100) {
        let w = bartlett(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            prop_assert!(
                (s[i] - s[m - 1 - i]).abs() < 1e-14,
                "bartlett({}) not symmetric at {}: {} vs {}",
                m, i, s[i], s[m - 1 - i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 2. hanning(m) is symmetric: w[i] == w[m-1-i]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_hanning_symmetric(m in 2usize..=100) {
        let w = hanning(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            prop_assert!(
                (s[i] - s[m - 1 - i]).abs() < 1e-14,
                "hanning({}) not symmetric at {}: {} vs {}",
                m, i, s[i], s[m - 1 - i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 3. hamming(m) is approximately symmetric: w[i] ~= w[m-1-i]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_hamming_symmetric(m in 2usize..=100) {
        let w = hamming(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            prop_assert!(
                (s[i] - s[m - 1 - i]).abs() < 1e-12,
                "hamming({}) not symmetric at {}: {} vs {}",
                m, i, s[i], s[m - 1 - i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. blackman(m) is approximately symmetric: w[i] ~= w[m-1-i]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_blackman_symmetric(m in 2usize..=100) {
        let w = blackman(m).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            prop_assert!(
                (s[i] - s[m - 1 - i]).abs() < 1e-12,
                "blackman({}) not symmetric at {}: {} vs {}",
                m, i, s[i], s[m - 1 - i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. kaiser(m, beta) is approximately symmetric: w[i] ~= w[m-1-i]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_kaiser_symmetric(m in 2usize..=100, beta in 0.0f64..20.0) {
        let w = kaiser(m, beta).unwrap();
        let s = w.as_slice().unwrap();
        for i in 0..m / 2 {
            prop_assert!(
                (s[i] - s[m - 1 - i]).abs() < 1e-10,
                "kaiser({}, {}) not symmetric at {}: {} vs {}",
                m, beta, i, s[i], s[m - 1 - i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. All window values are bounded in [0, 1] for bartlett/hanning/hamming/blackman
    // -----------------------------------------------------------------------
    #[test]
    fn prop_all_windows_bounded(m in 2usize..=100) {
        let windows: Vec<(&str, Vec<f64>)> = vec![
            ("bartlett", bartlett(m).unwrap().iter().copied().collect()),
            ("hanning", hanning(m).unwrap().iter().copied().collect()),
            ("hamming", hamming(m).unwrap().iter().copied().collect()),
            ("blackman", blackman(m).unwrap().iter().copied().collect()),
        ];
        for (name, vals) in &windows {
            for (i, &v) in vals.iter().enumerate() {
                // Blackman can produce tiny negative values near zero (~-1e-17)
                prop_assert!(
                    (-1e-15..=1.0 + 1e-15).contains(&v),
                    "{}({}) out of [0,1] at index {}: {}",
                    name, m, i, v
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // 7. All window functions return array of length m
    // -----------------------------------------------------------------------
    #[test]
    fn prop_all_windows_correct_length(m in 0usize..=100) {
        let w_bartlett = bartlett(m).unwrap();
        let w_hanning = hanning(m).unwrap();
        let w_hamming = hamming(m).unwrap();
        let w_blackman = blackman(m).unwrap();
        let w_kaiser = kaiser(m, 5.0).unwrap();
        prop_assert_eq!(w_bartlett.shape(), &[m]);
        prop_assert_eq!(w_hanning.shape(), &[m]);
        prop_assert_eq!(w_hamming.shape(), &[m]);
        prop_assert_eq!(w_blackman.shape(), &[m]);
        prop_assert_eq!(w_kaiser.shape(), &[m]);
    }

    // -----------------------------------------------------------------------
    // 8. kaiser(m, 0.0) is approximately rectangular (all ones)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_kaiser_beta_zero_is_rectangular(m in 2usize..=100) {
        let w = kaiser(m, 0.0).unwrap();
        let s = w.as_slice().unwrap();
        for (i, &v) in s.iter().enumerate() {
            prop_assert!(
                (v - 1.0).abs() < 1e-10,
                "kaiser({}, 0.0) at index {} = {}, expected 1.0",
                m, i, v
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. All window functions with m=1 return [1.0]
    // -----------------------------------------------------------------------
    #[test]
    fn prop_window_m1_is_one(_ in 0usize..1) {
        let windows: Vec<(&str, Array<f64, Ix1>)> = vec![
            ("bartlett", bartlett(1).unwrap()),
            ("hanning", hanning(1).unwrap()),
            ("hamming", hamming(1).unwrap()),
            ("blackman", blackman(1).unwrap()),
            ("kaiser", kaiser(1, 5.0).unwrap()),
        ];
        for (name, w) in &windows {
            let s = w.as_slice().unwrap();
            prop_assert_eq!(s.len(), 1, "{} m=1 has wrong length", name);
            prop_assert!(
                (s[0] - 1.0).abs() < 1e-15,
                "{} m=1 value = {}, expected 1.0",
                name, s[0]
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10. All window functions with m=0 return empty array
    // -----------------------------------------------------------------------
    #[test]
    fn prop_window_m0_is_empty(_ in 0usize..1) {
        let windows: Vec<(&str, Array<f64, Ix1>)> = vec![
            ("bartlett", bartlett(0).unwrap()),
            ("hanning", hanning(0).unwrap()),
            ("hamming", hamming(0).unwrap()),
            ("blackman", blackman(0).unwrap()),
            ("kaiser", kaiser(0, 5.0).unwrap()),
        ];
        for (name, w) in &windows {
            prop_assert_eq!(
                w.shape(), &[0],
                "{} m=0 should have shape [0], got {:?}",
                name, w.shape()
            );
        }
    }

    // -----------------------------------------------------------------------
    // 11. vectorize(|x| x) applied to array returns same array (identity)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_vectorize_identity(data in proptest::collection::vec(-100.0f64..100.0, 0..=50)) {
        let a = arr1(data.clone());
        let id = vectorize(|x: f64| x);
        let result = id(&a).unwrap();
        let r_data: Vec<f64> = result.iter().copied().collect();
        prop_assert_eq!(
            r_data, data,
            "vectorize identity should return the same data"
        );
    }

    // -----------------------------------------------------------------------
    // 12. vectorize(f . g)(x) == vectorize(f)(vectorize(g)(x)) (composition)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_vectorize_composition(data in proptest::collection::vec(-10.0f64..10.0, 0..=50)) {
        let a = arr1(data);
        // g(x) = x + 1.0, f(x) = x * 2.0
        let g = vectorize(|x: f64| x + 1.0);
        let f = vectorize(|x: f64| x * 2.0);
        let fg = vectorize(|x: f64| (x + 1.0) * 2.0);

        let composed = fg(&a).unwrap();
        let step1 = g(&a).unwrap();
        let chained = f(&step1).unwrap();

        let composed_data: Vec<f64> = composed.iter().copied().collect();
        let chained_data: Vec<f64> = chained.iter().copied().collect();

        for (i, (&c, &ch)) in composed_data.iter().zip(chained_data.iter()).enumerate() {
            prop_assert!(
                (c - ch).abs() < 1e-12,
                "composition mismatch at index {}: {} vs {}",
                i, c, ch
            );
        }
    }
}
