// Property-based tests for ferray-random
//
// Tests mathematical invariants of random number generation using proptest.

use ferray_random::default_rng_seeded;

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // -----------------------------------------------------------------------
    // 1. Determinism: same seed produces same output
    // -----------------------------------------------------------------------
    #[test]
    fn prop_determinism(seed in 0u64..1_000_000) {
        let mut rng1 = default_rng_seeded(seed);
        let mut rng2 = default_rng_seeded(seed);

        let a = rng1.random(100).unwrap();
        let b = rng2.random(100).unwrap();

        let a_data: Vec<f64> = a.iter().copied().collect();
        let b_data: Vec<f64> = b.iter().copied().collect();
        prop_assert_eq!(a_data, b_data, "same seed {} produced different output", seed);
    }

    // -----------------------------------------------------------------------
    // 2. Shape: output shape matches requested shape
    // -----------------------------------------------------------------------
    #[test]
    fn prop_random_shape(n in 1usize..=1000) {
        let mut rng = default_rng_seeded(42);
        let arr = rng.random(n).unwrap();
        prop_assert_eq!(arr.shape(), &[n]);
        prop_assert_eq!(arr.size(), n);
    }

    // -----------------------------------------------------------------------
    // 3. Range: uniform samples in [0, 1)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_random_range(seed in 0u64..100_000) {
        let mut rng = default_rng_seeded(seed);
        let arr = rng.random(500).unwrap();
        for &v in arr.iter() {
            prop_assert!(
                (0.0..1.0).contains(&v),
                "random() value {} outside [0, 1)",
                v
            );
        }
    }

    // -----------------------------------------------------------------------
    // 4. Range: uniform(low, high) samples in [low, high)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_uniform_range(
        seed in 0u64..100_000,
        low in -100.0f64..100.0,
        span in 0.01f64..100.0,
    ) {
        let high = low + span;
        let mut rng = default_rng_seeded(seed);
        let arr = rng.uniform(low, high, 500).unwrap();
        for &v in arr.iter() {
            prop_assert!(
                v >= low && v < high,
                "uniform({}, {}) value {} outside range",
                low, high, v
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. Range: integers(low, high) in [low, high)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_integers_range(
        seed in 0u64..100_000,
        low in -50i64..50,
        span in 1i64..50,
    ) {
        let high = low + span;
        let mut rng = default_rng_seeded(seed);
        let arr = rng.integers(low, high, 500).unwrap();
        for &v in arr.iter() {
            prop_assert!(
                v >= low && v < high,
                "integers({}, {}) value {} outside range",
                low, high, v
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. Shape: uniform output shape matches requested
    // -----------------------------------------------------------------------
    #[test]
    fn prop_uniform_shape(n in 1usize..=1000) {
        let mut rng = default_rng_seeded(42);
        let arr = rng.uniform(0.0, 1.0, n).unwrap();
        prop_assert_eq!(arr.shape(), &[n]);
    }

    // -----------------------------------------------------------------------
    // 7. Shape: integers output shape matches requested
    // -----------------------------------------------------------------------
    #[test]
    fn prop_integers_shape(n in 1usize..=1000) {
        let mut rng = default_rng_seeded(42);
        let arr = rng.integers(0, 100, n).unwrap();
        prop_assert_eq!(arr.shape(), &[n]);
    }

    // -----------------------------------------------------------------------
    // 8. Normal moments: mean ~= loc, var ~= scale^2 for large samples
    // -----------------------------------------------------------------------
    #[test]
    fn prop_normal_moments(seed in 0u64..1000) {
        let loc = 3.0;
        let scale = 2.0;
        let n = 50_000;
        let mut rng = default_rng_seeded(seed);
        let arr = rng.normal(loc, scale, n).unwrap();
        let slice = arr.as_slice().unwrap();

        let sample_mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let sample_var: f64 = slice.iter()
            .map(|&x| (x - sample_mean).powi(2))
            .sum::<f64>() / n as f64;

        // Use generous tolerance for statistical tests (5 sigma)
        let se_mean = (scale * scale / n as f64).sqrt();
        prop_assert!(
            (sample_mean - loc).abs() < 5.0 * se_mean,
            "normal mean {} too far from {} (se={})",
            sample_mean, loc, se_mean
        );

        let expected_var = scale * scale;
        prop_assert!(
            (sample_var - expected_var).abs() < 0.3,
            "normal var {} too far from {}",
            sample_var, expected_var
        );
    }

    // -----------------------------------------------------------------------
    // 9. Standard normal has zero mean approximately
    // -----------------------------------------------------------------------
    #[test]
    fn prop_standard_normal_zero_mean(seed in 0u64..1000) {
        let n = 50_000;
        let mut rng = default_rng_seeded(seed);
        let arr = rng.standard_normal(n).unwrap();
        let slice = arr.as_slice().unwrap();

        let sample_mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let se = (1.0 / n as f64).sqrt();

        prop_assert!(
            sample_mean.abs() < 5.0 * se,
            "standard_normal mean {} too far from 0 (se={})",
            sample_mean, se
        );
    }

    // -----------------------------------------------------------------------
    // 10. Different seeds produce different outputs (with high probability)
    // -----------------------------------------------------------------------
    #[test]
    fn prop_different_seeds_differ(seed in 1u64..1_000_000) {
        let mut rng1 = default_rng_seeded(seed);
        let mut rng2 = default_rng_seeded(seed + 1);

        let a = rng1.random(100).unwrap();
        let b = rng2.random(100).unwrap();

        let a_data: Vec<f64> = a.iter().copied().collect();
        let b_data: Vec<f64> = b.iter().copied().collect();

        // They should differ (extremely unlikely to be equal by chance)
        prop_assert_ne!(a_data, b_data);
    }

    // -----------------------------------------------------------------------
    // 11. Deterministic integers
    // -----------------------------------------------------------------------
    #[test]
    fn prop_integers_deterministic(seed in 0u64..1_000_000) {
        let mut rng1 = default_rng_seeded(seed);
        let mut rng2 = default_rng_seeded(seed);

        let a = rng1.integers(0, 100, 200).unwrap();
        let b = rng2.integers(0, 100, 200).unwrap();

        let a_data: Vec<i64> = a.iter().copied().collect();
        let b_data: Vec<i64> = b.iter().copied().collect();
        prop_assert_eq!(a_data, b_data);
    }

    // -----------------------------------------------------------------------
    // 12. Uniform mean is approximately (low + high) / 2
    // -----------------------------------------------------------------------
    #[test]
    fn prop_uniform_mean(
        seed in 0u64..1000,
        low in -50.0f64..50.0,
        span in 1.0f64..50.0,
    ) {
        let high = low + span;
        let n = 50_000;
        let mut rng = default_rng_seeded(seed);
        let arr = rng.uniform(low, high, n).unwrap();
        let slice = arr.as_slice().unwrap();

        let sample_mean: f64 = slice.iter().sum::<f64>() / n as f64;
        let expected_mean = (low + high) / 2.0;
        let expected_var = (high - low).powi(2) / 12.0;
        let se = (expected_var / n as f64).sqrt();

        prop_assert!(
            (sample_mean - expected_mean).abs() < 5.0 * se,
            "uniform({},{}) mean {} too far from {} (se={})",
            low, high, sample_mean, expected_mean, se
        );
    }
}
