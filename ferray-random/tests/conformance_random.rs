//! Conformance tests for ferray-random. Each test names the canonical
//! inner path of the `Generator` method it exercises (e.g.
//! `ferray_random::distributions::normal::Generator::standard_normal`)
//! so the surface-coverage gate's text match picks it up.
//!
//! RNG conformance is validated against **distributional moments**
//! (sample mean and sample variance), not raw sample bytes. The Stage 1
//! plan explicitly excludes per-sample equality with NumPy because
//! ferray-random's BitGenerator state machine differs from numpy's
//! reference and reproducing the byte-exact stream is out of scope.
//! Sample sizes and tolerances mirror
//! `fixtures/random/distribution_moments.json` (10_000 samples,
//! tolerance 0.02–0.10).

use ferray_core::Array;
use ferray_core::dimension::{Ix1, IxDyn};

use ferray_random::default_rng_seeded;

const N: usize = 10_000;

fn mean_var_f64(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

fn mean_var_i64(samples: &[i64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| (x as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var)
}

// ---------------------------------------------------------------------------
// standard_normal: mean ~ 0, var ~ 1
//
// Covers:
//   - `ferray_random::distributions::normal::Generator::standard_normal`
// ---------------------------------------------------------------------------
#[test]
fn standard_normal_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.standard_normal(N).expect("standard_normal");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert!(
        mean.abs() < 0.05,
        "standard_normal mean={mean}, expected ~0"
    );
    assert!(
        (var - 1.0).abs() < 0.05,
        "standard_normal var={var}, expected ~1"
    );
    assert!(s.iter().any(|&x| x < -1.0) && s.iter().any(|&x| x > 1.0));
}

// ---------------------------------------------------------------------------
// uniform(low, high): mean ~ (low+high)/2, var ~ (high-low)^2 / 12
// integers(low, high): same continuous moment approximation in [low, high)
//
// Covers:
//   - `ferray_random::distributions::uniform::Generator::uniform`
//   - `ferray_random::distributions::uniform::Generator::integers`
// ---------------------------------------------------------------------------
#[test]
fn uniform_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.uniform(0.0, 1.0, N).expect("uniform");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert!(
        (mean - 0.5).abs() < 0.02,
        "uniform mean={mean}, expected ~0.5"
    );
    assert!(
        (var - 1.0 / 12.0).abs() < 0.02,
        "uniform var={var}, expected ~1/12"
    );
    // Range check: every sample in [0, 1).
    assert!(s.iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
fn integers_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.integers(0, 100, N).expect("integers");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_i64(s);
    // U{0,..,99}: mean = 49.5, var = (100^2 - 1)/12 ~= 833.25.
    assert!(
        (mean - 49.5).abs() < 1.0,
        "integers mean={mean}, expected ~49.5"
    );
    assert!(
        (var - 833.25).abs() < 50.0,
        "integers var={var}, expected ~833.25"
    );
    assert!(s.iter().all(|&x| (0..100).contains(&x)));
}

// ---------------------------------------------------------------------------
// exponential(scale): mean ~ scale, var ~ scale^2
//
// Covers:
//   - `ferray_random::distributions::exponential::Generator::exponential`
// ---------------------------------------------------------------------------
#[test]
fn exponential_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.exponential(2.0, N).expect("exponential");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert!(
        (mean - 2.0).abs() < 0.1,
        "exponential mean={mean}, expected ~2"
    );
    assert!(
        (var - 4.0).abs() < 0.4,
        "exponential var={var}, expected ~4"
    );
    assert!(s.iter().all(|&x| x >= 0.0));
}

// ---------------------------------------------------------------------------
// gamma(alpha, scale): mean ~ alpha*scale, var ~ alpha*scale^2
//
// Covers:
//   - `ferray_random::distributions::gamma::Generator::gamma`
// ---------------------------------------------------------------------------
#[test]
fn gamma_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.gamma(2.0, 3.0, N).expect("gamma");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert!((mean - 6.0).abs() < 0.2, "gamma mean={mean}, expected ~6");
    assert!((var - 18.0).abs() < 1.5, "gamma var={var}, expected ~18");
    assert!(s.iter().all(|&x| x >= 0.0));
}

// ---------------------------------------------------------------------------
// beta(a, b): mean ~ a/(a+b), var ~ ab / ((a+b)^2 (a+b+1))
// For a=2, b=5: mean = 2/7, var = 10 / (49 * 8) = 10/392.
//
// Covers:
//   - `ferray_random::distributions::gamma::Generator::beta`
// ---------------------------------------------------------------------------
#[test]
fn beta_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.beta(2.0, 5.0, N).expect("beta");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    let expected_mean = 2.0 / 7.0;
    let expected_var = (2.0 * 5.0) / (7.0_f64.powi(2) * 8.0);
    assert!(
        (mean - expected_mean).abs() < 0.02,
        "beta mean={mean}, expected ~{expected_mean}"
    );
    assert!(
        (var - expected_var).abs() < 0.02,
        "beta var={var}, expected ~{expected_var}"
    );
    assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
}

// ---------------------------------------------------------------------------
// shuffle / permutation / choice: structural properties (length / multiset
// equality / value range), not moments.
//
// Covers:
//   - `ferray_random::permutations::Generator::shuffle`
//   - `ferray_random::permutations::Generator::permutation`
//   - `ferray_random::permutations::Generator::choice`
// ---------------------------------------------------------------------------
#[test]
fn shuffle_preserves_multiset() {
    let mut rng = default_rng_seeded(42);
    let data: Vec<i64> = (0..100).collect();
    let mut arr = Array::<i64, Ix1>::from_vec(Ix1::new([100]), data.clone()).unwrap();
    rng.shuffle(&mut arr).expect("shuffle");
    let mut got: Vec<i64> = arr.as_slice().unwrap().to_vec();
    got.sort_unstable();
    assert_eq!(got, data, "shuffle must preserve the multiset");
    // The probability of identity-permutation under uniform shuffle is 1/100!
    let identity: Vec<i64> = (0..100).collect();
    assert_ne!(
        arr.as_slice().unwrap().to_vec(),
        identity,
        "100! < 1/2^256: identity outcome statistically impossible"
    );
}

#[test]
fn permutation_returns_a_permutation() {
    let mut rng = default_rng_seeded(42);
    let data: Vec<i64> = (0..50).collect();
    let arr = Array::<i64, Ix1>::from_vec(Ix1::new([50]), data.clone()).unwrap();
    let permuted = rng.permutation(&arr).expect("permutation");
    let mut got: Vec<i64> = permuted.as_slice().unwrap().to_vec();
    got.sort_unstable();
    assert_eq!(got, data);
    // Original must be untouched.
    assert_eq!(arr.as_slice().unwrap().to_vec(), data);
}

#[test]
fn choice_samples_lie_in_source_set() {
    let mut rng = default_rng_seeded(42);
    let pool: Vec<i64> = (10..20).collect();
    let arr = Array::<i64, Ix1>::from_vec(Ix1::new([10]), pool.clone()).unwrap();

    // With replacement — sample size > pool size is allowed.
    let sampled = rng.choice(&arr, 1000, true, None).expect("choice replace");
    assert_eq!(sampled.shape(), &[1000]);
    assert!(sampled.as_slice().unwrap().iter().all(|x| pool.contains(x)));

    // Without replacement — all samples distinct.
    let sampled2 = rng
        .choice(&arr, 10, false, None)
        .expect("choice no-replace");
    let mut s: Vec<i64> = sampled2.as_slice().unwrap().to_vec();
    s.sort_unstable();
    s.dedup();
    assert_eq!(s.len(), 10, "without-replacement samples must be unique");
}

// ---------------------------------------------------------------------------
// shape parameter accepts arrays — exercise the `IntoShape` trait too so
// it's not orphaned. The 2-D output flattens to N*K values; the marginals
// of each value-bin are still standard normal.
//
// Covers:
//   - `ferray_random::distributions::normal::Generator::standard_normal`
//     (multi-dim shape path)
// ---------------------------------------------------------------------------
#[test]
fn standard_normal_2d_shape_moments() {
    let mut rng = default_rng_seeded(7);
    let arr: Array<f64, IxDyn> = rng
        .standard_normal([100usize, 100usize])
        .expect("2d normal");
    assert_eq!(arr.shape(), &[100, 100]);
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert!(mean.abs() < 0.05);
    assert!((var - 1.0).abs() < 0.05);
}
