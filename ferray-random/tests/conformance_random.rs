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
//! `fixtures/random/distribution_moments.json` (100_000 samples,
//! tolerance 0.02–0.10).

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use serde::Deserialize;

use ferray_random::{
    BitGenerator, Generator as RandomGenerator, MT19937, Pcg64, Pcg64Dxsm, Philox, SeedSequence,
    Sfc64, Xoshiro256StarStar, default_rng, default_rng_seeded,
};

#[derive(Debug, Deserialize)]
struct MomentFixture {
    test_cases: Vec<MomentCase>,
}

#[derive(Debug, Deserialize)]
struct MomentCase {
    name: String,
    inputs: serde_json::Value,
    expected: MomentExpected,
}

#[derive(Debug, Deserialize)]
struct MomentExpected {
    expected_mean: f64,
    expected_var: f64,
    mean_tolerance: f64,
    var_tolerance: f64,
}

fn moment_case(name: &str) -> MomentCase {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures/random/distribution_moments.json");
    let json = std::fs::read_to_string(path).expect("read random distribution fixture");
    let fixture: MomentFixture = serde_json::from_str(&json).expect("parse random fixture");
    fixture
        .test_cases
        .into_iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("missing random fixture case {name}"))
}

fn fixture_size(case: &MomentCase) -> usize {
    case.inputs["size"]
        .as_u64()
        .unwrap_or_else(|| panic!("fixture case {} missing size", case.name)) as usize
}

fn mean_var_f64(samples: &[f64]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

fn mean_var_f32(samples: &[f32]) -> (f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
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

fn assert_moments(case: &MomentCase, mean: f64, var: f64) {
    assert!(
        (mean - case.expected.expected_mean).abs() < case.expected.mean_tolerance,
        "{} mean={mean}, expected ~{} +/- {}",
        case.name,
        case.expected.expected_mean,
        case.expected.mean_tolerance
    );
    assert!(
        (var - case.expected.expected_var).abs() < case.expected.var_tolerance,
        "{} var={var}, expected ~{} +/- {}",
        case.name,
        case.expected.expected_var,
        case.expected.var_tolerance
    );
}

// ---------------------------------------------------------------------------
// standard_normal: mean ~ 0, var ~ 1
//
// Covers:
//   - `ferray_random::distributions::normal::Generator::standard_normal`
// ---------------------------------------------------------------------------
#[test]
fn standard_normal_distribution_moments() {
    let case = moment_case("standard_normal_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng
        .standard_normal(fixture_size(&case))
        .expect("standard_normal");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert_moments(&case, mean, var);
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
    let case = moment_case("uniform_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng.uniform(0.0, 1.0, fixture_size(&case)).expect("uniform");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert_moments(&case, mean, var);
    // Range check: every sample in [0, 1).
    assert!(s.iter().all(|&x| (0.0..1.0).contains(&x)));
}

#[test]
fn integers_distribution_moments() {
    let mut rng = default_rng_seeded(42);
    let arr = rng.integers(0, 100, 100_000).expect("integers");
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
    let case = moment_case("exponential_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng
        .exponential(2.0, fixture_size(&case))
        .expect("exponential");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert_moments(&case, mean, var);
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
    let case = moment_case("gamma_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng.gamma(2.0, 3.0, fixture_size(&case)).expect("gamma");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert_moments(&case, mean, var);
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
    let case = moment_case("beta_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng.beta(2.0, 5.0, fixture_size(&case)).expect("beta");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_f64(s);
    assert_moments(&case, mean, var);
    assert!(s.iter().all(|&x| (0.0..=1.0).contains(&x)));
}

// ---------------------------------------------------------------------------
// discrete fixture cases: Poisson and binomial moments.
//
// Covers:
//   - `ferray_random::distributions::discrete::Generator::poisson`
//   - `ferray_random::distributions::discrete::Generator::binomial`
// ---------------------------------------------------------------------------
#[test]
fn poisson_distribution_moments() {
    let case = moment_case("poisson_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng.poisson(5.0, fixture_size(&case)).expect("poisson");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_i64(s);
    assert_moments(&case, mean, var);
    assert!(s.iter().all(|&x| x >= 0));
}

#[test]
fn binomial_distribution_moments() {
    let case = moment_case("binomial_moments");
    let mut rng = default_rng_seeded(42);
    let arr = rng
        .binomial(10, 0.3, fixture_size(&case))
        .expect("binomial");
    let s = arr.as_slice().expect("contiguous");
    let (mean, var) = mean_var_i64(s);
    assert_moments(&case, mean, var);
    assert!(s.iter().all(|&x| (0..=10).contains(&x)));
}

// ---------------------------------------------------------------------------
// Uniform family variants: dtype, out, broadcast-array parameters, and shape.
//
// Covers:
//   - `ferray_random::IntoShape`
//   - `ferray_random::shape::IntoShape`
//   - `ferray_random::distributions::uniform::Generator::random`
//   - `ferray_random::distributions::uniform::Generator::random_f32`
//   - `ferray_random::distributions::uniform::Generator::random_into`
//   - `ferray_random::distributions::uniform::Generator::uniform_array`
//   - `ferray_random::distributions::uniform::Generator::uniform_f32`
// ---------------------------------------------------------------------------
#[test]
fn uniform_variants_match_numpy_shape_and_range_contracts() {
    let mut rng = default_rng_seeded(42);

    let random = rng.random([3usize, 4usize]).expect("random");
    assert_eq!(random.shape(), &[3, 4]);
    assert!(
        random
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| (0.0..1.0).contains(&x))
    );

    let random_f32 = rng.random_f32(vec![2usize, 5usize]).expect("random_f32");
    assert_eq!(random_f32.shape(), &[2, 5]);
    assert!(
        random_f32
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| (0.0..1.0).contains(&x))
    );

    let mut allocated_rng = default_rng_seeded(7);
    let mut out_rng = default_rng_seeded(7);
    let allocated = allocated_rng.random([3usize, 4usize]).unwrap();
    let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), vec![0.0; 12]).unwrap();
    out_rng.random_into(&mut out).expect("random_into");
    assert_eq!(allocated.as_slice().unwrap(), out.as_slice().unwrap());

    let uniform_f32 = rng.uniform_f32(-2.0, 3.0, [64usize]).expect("uniform_f32");
    assert!(
        uniform_f32
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| (-2.0..3.0).contains(&x))
    );

    let low = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![0.0]).unwrap();
    let high =
        Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let broadcast = rng.uniform_array(&low, &high).expect("uniform_array");
    assert_eq!(broadcast.shape(), &[2, 3]);
    for (&x, &h) in broadcast.as_slice().unwrap().iter().zip(high.iter()) {
        assert!(x >= 0.0 && x < h);
    }
}

// ---------------------------------------------------------------------------
// Normal family variants: scalar params, dtype, out, broadcast-array params,
// lognormal, ziggurat kernel, and parallel generation.
//
// Covers:
//   - `ferray_random::distributions::ziggurat::standard_normal_ziggurat`
//   - `ferray_random::distributions::ziggurat::standard_normal_ziggurat_f32`
//   - `ferray_random::distributions::normal::Generator::normal`
//   - `ferray_random::distributions::normal::Generator::normal_array`
//   - `ferray_random::distributions::normal::Generator::normal_f32`
//   - `ferray_random::distributions::normal::Generator::lognormal`
//   - `ferray_random::distributions::normal::Generator::lognormal_f32`
//   - `ferray_random::distributions::normal::Generator::standard_normal_f32`
//   - `ferray_random::distributions::normal::Generator::standard_normal_into`
//   - `ferray_random::parallel::Generator::standard_normal_parallel`
// ---------------------------------------------------------------------------
#[test]
fn normal_variants_match_numpy_distribution_contracts() {
    let n = 100_000usize;

    let mut rng = default_rng_seeded(42);
    let normal = rng.normal(5.0, 2.0, n).expect("normal");
    let (mean, var) = mean_var_f64(normal.as_slice().unwrap());
    assert!((mean - 5.0).abs() < 0.08, "normal mean={mean}");
    assert!((var - 4.0).abs() < 0.16, "normal var={var}");

    let mut rng = default_rng_seeded(43);
    let normal_f32 = rng.normal_f32(-1.0, 0.5, n).expect("normal_f32");
    let (mean, var) = mean_var_f32(normal_f32.as_slice().unwrap());
    assert!((mean + 1.0).abs() < 0.08, "normal_f32 mean={mean}");
    assert!((var - 0.25).abs() < 0.06, "normal_f32 var={var}");

    let mut rng = default_rng_seeded(44);
    let std_f32 = rng.standard_normal_f32(n).expect("standard_normal_f32");
    let (mean, var) = mean_var_f32(std_f32.as_slice().unwrap());
    assert!(mean.abs() < 0.04, "standard_normal_f32 mean={mean}");
    assert!((var - 1.0).abs() < 0.06, "standard_normal_f32 var={var}");

    let mut allocated_rng = default_rng_seeded(45);
    let mut out_rng = default_rng_seeded(45);
    let allocated = allocated_rng.standard_normal([32usize]).unwrap();
    let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[32]), vec![0.0; 32]).unwrap();
    out_rng
        .standard_normal_into(&mut out)
        .expect("standard_normal_into");
    assert_eq!(allocated.as_slice().unwrap(), out.as_slice().unwrap());

    let mut rng = default_rng_seeded(46);
    let loc = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![0.0, 10.0]).unwrap();
    let scale = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[1]), vec![1.0]).unwrap();
    let array = rng.normal_array(&loc, &scale).expect("normal_array");
    assert_eq!(array.shape(), &[2]);

    let mut rng = default_rng_seeded(47);
    let lognormal = rng.lognormal(0.0, 0.25, n).expect("lognormal");
    assert!(lognormal.as_slice().unwrap().iter().all(|&x| x > 0.0));
    let log_mean = lognormal.as_slice().unwrap().iter().sum::<f64>() / n as f64;
    let expected_mean = (0.25_f64.powi(2) / 2.0).exp();
    assert!((log_mean - expected_mean).abs() < 0.03);

    let mut rng = default_rng_seeded(48);
    let lognormal_f32 = rng.lognormal_f32(0.0, 0.25, n).expect("lognormal_f32");
    assert!(lognormal_f32.as_slice().unwrap().iter().all(|&x| x > 0.0));

    let mut rng = default_rng_seeded(50);
    let parallel = rng
        .standard_normal_parallel([10_000usize])
        .expect("standard_normal_parallel");
    let (mean, var) = mean_var_f64(parallel.as_slice().unwrap());
    assert!(mean.abs() < 0.08);
    assert!((var - 1.0).abs() < 0.12);
}

// ---------------------------------------------------------------------------
// Exponential family variants: standard distribution, dtype, out, and array
// scale parameter.
//
// Covers:
//   - `ferray_random::distributions::exponential::Generator::standard_exponential`
//   - `ferray_random::distributions::exponential::Generator::standard_exponential_f32`
//   - `ferray_random::distributions::exponential::Generator::standard_exponential_into`
//   - `ferray_random::distributions::exponential::Generator::exponential_array`
//   - `ferray_random::distributions::exponential::Generator::exponential_f32`
// ---------------------------------------------------------------------------
#[test]
fn exponential_variants_match_numpy_distribution_contracts() {
    let n = 100_000usize;

    let mut rng = default_rng_seeded(42);
    let standard = rng.standard_exponential(n).expect("standard_exponential");
    let (mean, var) = mean_var_f64(standard.as_slice().unwrap());
    assert!((mean - 1.0).abs() < 0.04);
    assert!((var - 1.0).abs() < 0.08);
    assert!(standard.as_slice().unwrap().iter().all(|&x| x >= 0.0));

    let mut rng = default_rng_seeded(43);
    let standard_f32 = rng
        .standard_exponential_f32(n)
        .expect("standard_exponential_f32");
    let (mean, var) = mean_var_f32(standard_f32.as_slice().unwrap());
    assert!((mean - 1.0).abs() < 0.05);
    assert!((var - 1.0).abs() < 0.1);

    let mut allocated_rng = default_rng_seeded(44);
    let mut out_rng = default_rng_seeded(44);
    let allocated = allocated_rng
        .standard_exponential([8usize, 4usize])
        .unwrap();
    let mut out = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[8, 4]), vec![0.0; 32]).unwrap();
    out_rng
        .standard_exponential_into(&mut out)
        .expect("standard_exponential_into");
    assert_eq!(allocated.as_slice().unwrap(), out.as_slice().unwrap());

    let mut rng = default_rng_seeded(45);
    let scale = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![0.5, 1.0, 2.0]).unwrap();
    let array = rng.exponential_array(&scale).expect("exponential_array");
    assert_eq!(array.shape(), &[3]);
    assert!(array.as_slice().unwrap().iter().all(|&x| x >= 0.0));

    let mut rng = default_rng_seeded(46);
    let exp_f32 = rng.exponential_f32(2.0, n).expect("exponential_f32");
    let (mean, var) = mean_var_f32(exp_f32.as_slice().unwrap());
    assert!((mean - 2.0).abs() < 0.08);
    assert!((var - 4.0).abs() < 0.3);
}

// ---------------------------------------------------------------------------
// Generator and BitGenerator state contracts: deterministic seed, byte state
// round-trip, bounded integers, and non-deterministic constructor sanity.
//
// Covers:
//   - `ferray_random::BitGenerator`
//   - `ferray_random::Generator`
//   - `ferray_random::MT19937`
//   - `ferray_random::Pcg64`
//   - `ferray_random::Pcg64Dxsm`
//   - `ferray_random::Philox`
//   - `ferray_random::Sfc64`
//   - `ferray_random::Xoshiro256StarStar`
//   - `ferray_random::bitgen::BitGenerator`
//   - `ferray_random::bitgen::MT19937`
//   - `ferray_random::bitgen::Pcg64`
//   - `ferray_random::bitgen::Pcg64Dxsm`
//   - `ferray_random::bitgen::Philox`
//   - `ferray_random::bitgen::Sfc64`
//   - `ferray_random::bitgen::Xoshiro256StarStar`
//   - `ferray_random::bitgen::mt19937::MT19937`
//   - `ferray_random::bitgen::pcg64::Pcg64`
//   - `ferray_random::bitgen::pcg64dxsm::Pcg64Dxsm`
//   - `ferray_random::bitgen::philox::Philox`
//   - `ferray_random::bitgen::sfc64::Sfc64`
//   - `ferray_random::bitgen::xoshiro256::Xoshiro256StarStar`
//   - `ferray_random::default_rng`
//   - `ferray_random::default_rng_seeded`
//   - `ferray_random::generator::Generator`
//   - `ferray_random::generator::Generator::bit_generator`
//   - `ferray_random::generator::Generator::bytes`
//   - `ferray_random::generator::Generator::new`
//   - `ferray_random::generator::Generator::next_f32`
//   - `ferray_random::generator::Generator::next_f64`
//   - `ferray_random::generator::Generator::next_u64`
//   - `ferray_random::generator::Generator::next_u64_bounded`
//   - `ferray_random::generator::Generator::set_state_bytes`
//   - `ferray_random::generator::Generator::state_bytes`
//   - `ferray_random::generator::default_rng`
//   - `ferray_random::generator::default_rng_seeded`
// ---------------------------------------------------------------------------
#[test]
fn generator_and_bitgenerator_state_contracts() {
    fn roundtrip<B: BitGenerator + Clone>(mut a: B, expected_state_len: usize) {
        for _ in 0..7 {
            a.next_u64();
        }
        let snapshot = a.state_bytes().expect("state_bytes");
        assert_eq!(snapshot.len(), expected_state_len);
        let expected: Vec<u64> = (0..16).map(|_| a.next_u64()).collect();
        let mut b = a.clone();
        b.set_state_bytes(&snapshot).expect("set_state_bytes");
        let restored: Vec<u64> = (0..16).map(|_| b.next_u64()).collect();
        assert_eq!(expected, restored);
    }

    roundtrip(Xoshiro256StarStar::seed_from_u64(42), 32);
    roundtrip(Pcg64::seed_from_u64(43), 32);
    roundtrip(Pcg64Dxsm::seed_from_u64(44), 32);
    roundtrip(Philox::seed_from_u64(45), 44);
    roundtrip(Sfc64::seed_from_u64(46), 32);
    roundtrip(MT19937::seed_from_u64(47), 312 * 8 + 8);

    let mut seeded_a = default_rng_seeded(123);
    let mut seeded_b = default_rng_seeded(123);
    assert_eq!(seeded_a.next_u64(), seeded_b.next_u64());

    let mut nondeterministic = default_rng();
    assert!((0.0..1.0).contains(&nondeterministic.next_f64()));

    let mut rng = RandomGenerator::new(Xoshiro256StarStar::seed_from_u64(99));
    let bounded: Vec<u64> = (0..128).map(|_| rng.next_u64_bounded(17)).collect();
    assert!(bounded.iter().all(|&x| x < 17));
    assert!((0.0..1.0).contains(&rng.next_f64()));
    assert!((0.0..1.0).contains(&rng.next_f32()));
    let _ = rng.bit_generator().next_u64();

    let snapshot = rng.state_bytes().expect("generator state_bytes");
    let expected: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    rng.set_state_bytes(&snapshot)
        .expect("generator set_state_bytes");
    let restored: Vec<u64> = (0..8).map(|_| rng.next_u64()).collect();
    assert_eq!(expected, restored);

    assert_eq!(rng.bytes(0).len(), 0);
    assert_eq!(rng.bytes(13).len(), 13);
}

// ---------------------------------------------------------------------------
// SeedSequence parity surface: entropy, spawn keys, deterministic state, and
// seeding concrete bit generators.
//
// Covers:
//   - `ferray_random::SeedSequence`
//   - `ferray_random::bitgen::SeedSequence`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::entropy`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::generate_state`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::generate_u64`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::new`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::seed`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::spawn`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::spawn_key`
//   - `ferray_random::bitgen::seed_sequence::SeedSequence::with_spawn_key`
// ---------------------------------------------------------------------------
#[test]
fn seed_sequence_contracts_match_numpy_seedsequence_model() {
    let seq = SeedSequence::new(2026);
    assert_eq!(seq.entropy(), 2026);
    assert!(seq.spawn_key().is_empty());
    assert_eq!(
        seq.generate_state(8),
        SeedSequence::new(2026).generate_state(8)
    );
    assert_eq!(seq.generate_u64(), SeedSequence::new(2026).generate_u64());

    let keyed = SeedSequence::with_spawn_key(2026, vec![1, 2, 3]);
    assert_eq!(keyed.spawn_key(), &[1, 2, 3]);
    assert_ne!(keyed.generate_state(8), seq.generate_state(8));

    let mut root = SeedSequence::new(2026);
    let children = root.spawn(3);
    assert_eq!(children.len(), 3);
    assert_eq!(children[0].spawn_key(), &[0]);
    assert_eq!(children[1].spawn_key(), &[1]);
    assert_ne!(children[0].generate_state(8), children[1].generate_state(8));

    let mut a: Pcg64 = seq.seed();
    let mut b: Pcg64 = SeedSequence::new(2026).seed();
    for _ in 0..16 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
}

// ---------------------------------------------------------------------------
// Generator spawning mirrors NumPy's independent child-stream model.
//
// Covers:
//   - `ferray_random::generator::spawn_generators`
//   - `ferray_random::parallel::Generator::spawn`
// ---------------------------------------------------------------------------
#[test]
fn generator_spawn_contracts_create_independent_deterministic_children() {
    let mut direct_parent = default_rng_seeded(2026);
    let mut direct_children = ferray_random::generator::spawn_generators(&mut direct_parent, 4)
        .expect("spawn_generators");
    assert_eq!(direct_children.len(), 4);
    let direct_first: Vec<u64> = direct_children
        .iter_mut()
        .map(RandomGenerator::next_u64)
        .collect();
    for i in 0..direct_first.len() {
        for j in (i + 1)..direct_first.len() {
            assert_ne!(direct_first[i], direct_first[j]);
        }
    }

    let mut a = default_rng_seeded(7);
    let mut b = default_rng_seeded(7);
    let mut children_a = a.spawn(3).expect("spawn");
    let mut children_b = b.spawn(3).expect("spawn");
    for (left, right) in children_a.iter_mut().zip(children_b.iter_mut()) {
        for _ in 0..16 {
            assert_eq!(left.next_u64(), right.next_u64());
        }
    }
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

// ---------------------------------------------------------------------------
// Remaining discrete distributions: direct moment/range/shape evidence.
//
// Covers:
//   - `ferray_random::distributions::discrete::Generator::geometric`
//   - `ferray_random::distributions::discrete::Generator::hypergeometric`
//   - `ferray_random::distributions::discrete::Generator::logseries`
//   - `ferray_random::distributions::discrete::Generator::negative_binomial`
//   - `ferray_random::distributions::discrete::Generator::poisson_array`
//   - `ferray_random::distributions::discrete::Generator::zipf`
// ---------------------------------------------------------------------------
#[test]
fn remaining_discrete_distributions_match_numpy_contracts() {
    let n = 100_000usize;

    let mut rng = default_rng_seeded(100);
    let geometric = rng.geometric(0.25, n).expect("geometric");
    let s = geometric.as_slice().unwrap();
    let (mean, var) = mean_var_i64(s);
    assert!((mean - 4.0).abs() < 0.12, "geometric mean={mean}");
    assert!((var - 12.0).abs() < 0.7, "geometric var={var}");
    assert!(s.iter().all(|&x| x >= 1));

    let mut rng = default_rng_seeded(101);
    let hyper = rng.hypergeometric(30, 70, 10, n).expect("hypergeometric");
    let s = hyper.as_slice().unwrap();
    let (mean, var) = mean_var_i64(s);
    assert!((mean - 3.0).abs() < 0.08, "hypergeometric mean={mean}");
    assert!(
        (var - 1.909_090_909).abs() < 0.18,
        "hypergeometric var={var}"
    );
    assert!(s.iter().all(|&x| (0..=10).contains(&x)));

    let mut rng = default_rng_seeded(102);
    let negative = rng
        .negative_binomial(5.0, 0.4, n)
        .expect("negative_binomial");
    let s = negative.as_slice().unwrap();
    let (mean, var) = mean_var_i64(s);
    assert!((mean - 7.5).abs() < 0.2, "negative_binomial mean={mean}");
    assert!((var - 18.75).abs() < 1.2, "negative_binomial var={var}");
    assert!(s.iter().all(|&x| x >= 0));

    let mut rng = default_rng_seeded(103);
    let logseries = rng.logseries(0.6, 10_000).expect("logseries");
    let s = logseries.as_slice().unwrap();
    assert_eq!(logseries.shape(), &[10_000]);
    assert!(s.iter().all(|&x| x >= 1));
    assert!(s.iter().any(|&x| x > 1));

    let mut rng = default_rng_seeded(104);
    let zipf = rng.zipf(2.5, 10_000).expect("zipf");
    let s = zipf.as_slice().unwrap();
    assert_eq!(zipf.shape(), &[10_000]);
    assert!(s.iter().all(|&x| x >= 1));
    assert!(s.iter().any(|&x| x > 1));

    let mut rng = default_rng_seeded(105);
    let lam = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[4]), vec![0.0, 1.0, 5.0, 12.0]).unwrap();
    let poisson = rng.poisson_array(&lam).expect("poisson_array");
    assert_eq!(poisson.shape(), &[4]);
    let s = poisson.as_slice().unwrap();
    assert_eq!(s[0], 0);
    assert!(s.iter().all(|&x| x >= 0));
}

// ---------------------------------------------------------------------------
// Remaining gamma-family distributions.
//
// Covers:
//   - `ferray_random::distributions::gamma::Generator::chisquare`
//   - `ferray_random::distributions::gamma::Generator::f`
//   - `ferray_random::distributions::gamma::Generator::noncentral_chisquare`
//   - `ferray_random::distributions::gamma::Generator::noncentral_f`
//   - `ferray_random::distributions::gamma::Generator::standard_gamma`
//   - `ferray_random::distributions::gamma::Generator::standard_t`
//   - `ferray_random::distributions::gamma::Generator::student_t`
// ---------------------------------------------------------------------------
#[test]
fn remaining_gamma_family_distributions_match_numpy_contracts() {
    let n = 100_000usize;

    let mut rng = default_rng_seeded(120);
    let standard_gamma = rng.standard_gamma(2.0, n).expect("standard_gamma");
    let (mean, var) = mean_var_f64(standard_gamma.as_slice().unwrap());
    assert!((mean - 2.0).abs() < 0.08, "standard_gamma mean={mean}");
    assert!((var - 2.0).abs() < 0.15, "standard_gamma var={var}");

    let mut rng = default_rng_seeded(121);
    let chisquare = rng.chisquare(4.0, n).expect("chisquare");
    let (mean, var) = mean_var_f64(chisquare.as_slice().unwrap());
    assert!((mean - 4.0).abs() < 0.12, "chisquare mean={mean}");
    assert!((var - 8.0).abs() < 0.45, "chisquare var={var}");

    let mut rng = default_rng_seeded(122);
    let f_dist = rng.f(5.0, 10.0, n).expect("f");
    let s = f_dist.as_slice().unwrap();
    let mean = s.iter().sum::<f64>() / n as f64;
    assert!((mean - 1.25).abs() < 0.1, "f mean={mean}");
    assert!(s.iter().all(|&x| x >= 0.0 && x.is_finite()));

    let mut rng = default_rng_seeded(123);
    let student = rng.student_t(8.0, n).expect("student_t");
    let (mean, var) = mean_var_f64(student.as_slice().unwrap());
    assert!(mean.abs() < 0.06, "student_t mean={mean}");
    assert!((var - (8.0 / 6.0)).abs() < 0.18, "student_t var={var}");

    let mut a = default_rng_seeded(124);
    let mut b = default_rng_seeded(124);
    let student = a.student_t(6.0, 256).expect("student_t");
    let standard = b.standard_t(6.0, 256).expect("standard_t");
    assert_eq!(student.as_slice().unwrap(), standard.as_slice().unwrap());

    let mut rng = default_rng_seeded(125);
    let noncentral_chi = rng
        .noncentral_chisquare(3.0, 2.0, n)
        .expect("noncentral_chisquare");
    let s = noncentral_chi.as_slice().unwrap();
    let mean = s.iter().sum::<f64>() / n as f64;
    assert!(
        (mean - 5.0).abs() < 0.16,
        "noncentral_chisquare mean={mean}"
    );
    assert!(s.iter().all(|&x| x >= 0.0));

    let mut rng = default_rng_seeded(126);
    let noncentral_f = rng
        .noncentral_f(5.0, 12.0, 2.0, 10_000)
        .expect("noncentral_f");
    assert_eq!(noncentral_f.shape(), &[10_000]);
    assert!(
        noncentral_f
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| x >= 0.0 && x.is_finite())
    );
}

// ---------------------------------------------------------------------------
// Miscellaneous continuous distributions.
//
// Covers:
//   - `ferray_random::distributions::misc_continuous::Generator::gumbel`
//   - `ferray_random::distributions::misc_continuous::Generator::laplace`
//   - `ferray_random::distributions::misc_continuous::Generator::logistic`
//   - `ferray_random::distributions::misc_continuous::Generator::pareto`
//   - `ferray_random::distributions::misc_continuous::Generator::power`
//   - `ferray_random::distributions::misc_continuous::Generator::rayleigh`
//   - `ferray_random::distributions::misc_continuous::Generator::standard_cauchy`
//   - `ferray_random::distributions::misc_continuous::Generator::triangular`
//   - `ferray_random::distributions::misc_continuous::Generator::vonmises`
//   - `ferray_random::distributions::misc_continuous::Generator::wald`
//   - `ferray_random::distributions::misc_continuous::Generator::weibull`
// ---------------------------------------------------------------------------
#[test]
fn misc_continuous_distributions_match_numpy_contracts() {
    let n = 50_000usize;

    let mut rng = default_rng_seeded(140);
    let laplace = rng.laplace(2.0, 3.0, n).expect("laplace");
    let (mean, var) = mean_var_f64(laplace.as_slice().unwrap());
    assert!((mean - 2.0).abs() < 0.15, "laplace mean={mean}");
    assert!((var - 18.0).abs() < 1.2, "laplace var={var}");

    let mut rng = default_rng_seeded(141);
    let logistic = rng.logistic(1.5, 2.0, n).expect("logistic");
    let (mean, _) = mean_var_f64(logistic.as_slice().unwrap());
    assert!((mean - 1.5).abs() < 0.18, "logistic mean={mean}");

    let mut rng = default_rng_seeded(142);
    let rayleigh = rng.rayleigh(2.0, n).expect("rayleigh");
    let s = rayleigh.as_slice().unwrap();
    assert!(s.iter().all(|&x| x >= 0.0));
    let mean = s.iter().sum::<f64>() / n as f64;
    assert!(
        (mean - (2.0 * (std::f64::consts::PI / 2.0).sqrt())).abs() < 0.08,
        "rayleigh mean={mean}"
    );

    let mut rng = default_rng_seeded(143);
    let weibull = rng.weibull(1.5, 10_000).expect("weibull");
    assert!(weibull.as_slice().unwrap().iter().all(|&x| x >= 0.0));

    let mut rng = default_rng_seeded(144);
    let pareto = rng.pareto(3.0, 10_000).expect("pareto");
    assert!(pareto.as_slice().unwrap().iter().all(|&x| x >= 0.0));

    let mut rng = default_rng_seeded(145);
    let gumbel = rng.gumbel(0.0, 1.0, n).expect("gumbel");
    let mean = gumbel.as_slice().unwrap().iter().sum::<f64>() / n as f64;
    assert!((mean - 0.577_215_664_9).abs() < 0.08, "gumbel mean={mean}");

    let mut rng = default_rng_seeded(146);
    let power = rng.power(2.5, 10_000).expect("power");
    assert!(
        power
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| (0.0..=1.0).contains(&x))
    );

    let mut rng = default_rng_seeded(147);
    let triangular = rng.triangular(-2.0, 0.5, 4.0, 10_000).expect("triangular");
    assert!(
        triangular
            .as_slice()
            .unwrap()
            .iter()
            .all(|&x| (-2.0..=4.0).contains(&x))
    );

    let mut rng = default_rng_seeded(148);
    let vonmises = rng.vonmises(0.25, 2.0, 10_000).expect("vonmises");
    assert!(vonmises.as_slice().unwrap().iter().all(|&x| x.is_finite()));

    let mut rng = default_rng_seeded(149);
    let wald = rng.wald(2.0, 3.0, 10_000).expect("wald");
    assert!(wald.as_slice().unwrap().iter().all(|&x| x > 0.0));

    let mut rng = default_rng_seeded(150);
    let cauchy = rng.standard_cauchy(10_000).expect("standard_cauchy");
    let s = cauchy.as_slice().unwrap();
    assert!(s.iter().all(|&x| x.is_finite()));
    assert!(s.iter().any(|&x| x < 0.0) && s.iter().any(|&x| x > 0.0));
}

// ---------------------------------------------------------------------------
// Multivariate distributions.
//
// Covers:
//   - `ferray_random::distributions::multivariate::Generator::dirichlet`
//   - `ferray_random::distributions::multivariate::Generator::multinomial`
//   - `ferray_random::distributions::multivariate::Generator::multivariate_hypergeometric`
//   - `ferray_random::distributions::multivariate::Generator::multivariate_normal`
//   - `ferray_random::distributions::multivariate::Generator::multivariate_normal_array`
// ---------------------------------------------------------------------------
#[test]
fn multivariate_distributions_match_numpy_shape_and_sum_contracts() {
    let mut rng = default_rng_seeded(160);
    let multinomial = rng
        .multinomial(20, &[0.2, 0.3, 0.5], 64)
        .expect("multinomial");
    assert_eq!(multinomial.shape(), &[64, 3]);
    for row in multinomial.as_slice().unwrap().chunks_exact(3) {
        assert_eq!(row.iter().sum::<i64>(), 20);
        assert!(row.iter().all(|&x| x >= 0));
    }

    let mut rng = default_rng_seeded(161);
    let dirichlet = rng.dirichlet(&[0.5, 1.5, 3.0], 64).expect("dirichlet");
    assert_eq!(dirichlet.shape(), &[64, 3]);
    for row in dirichlet.as_slice().unwrap().chunks_exact(3) {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(row.iter().all(|&x| x >= 0.0));
    }

    let mut rng = default_rng_seeded(162);
    let mvhg = rng
        .multivariate_hypergeometric(&[10, 20, 30], 12, 64)
        .expect("multivariate_hypergeometric");
    assert_eq!(mvhg.shape(), &[64, 3]);
    for row in mvhg.as_slice().unwrap().chunks_exact(3) {
        assert_eq!(row.iter().sum::<i64>(), 12);
        assert!((0..=10).contains(&row[0]));
        assert!((0..=20).contains(&row[1]));
        assert!((0..=30).contains(&row[2]));
    }

    let mut rng = default_rng_seeded(163);
    let mvn = rng
        .multivariate_normal(&[1.0, -2.0], &[1.0, 0.25, 0.25, 2.0], 20_000)
        .expect("multivariate_normal");
    assert_eq!(mvn.shape(), &[20_000, 2]);
    let mut mean0 = 0.0;
    let mut mean1 = 0.0;
    for row in mvn.as_slice().unwrap().chunks_exact(2) {
        mean0 += row[0];
        mean1 += row[1];
    }
    mean0 /= 20_000.0;
    mean1 /= 20_000.0;
    assert!((mean0 - 1.0).abs() < 0.04, "mvn mean0={mean0}");
    assert!((mean1 + 2.0).abs() < 0.06, "mvn mean1={mean1}");

    let mut rng = default_rng_seeded(164);
    let mean = Array::<f64, Ix1>::from_vec(Ix1::new([2]), vec![1.0, -2.0]).unwrap();
    let cov = Array::<f64, Ix2>::from_vec(Ix2::new([2, 2]), vec![1.0, 0.25, 0.25, 2.0]).unwrap();
    let mvn_array = rng
        .multivariate_normal_array(&mean, &cov, 128)
        .expect("multivariate_normal_array");
    assert_eq!(mvn_array.shape(), &[128, 2]);
    assert!(mvn_array.as_slice().unwrap().iter().all(|&x| x.is_finite()));
}

// ---------------------------------------------------------------------------
// Dynamic permutation helpers.
//
// Covers:
//   - `ferray_random::permutations::Generator::choice_dyn`
//   - `ferray_random::permutations::Generator::permutation_range`
//   - `ferray_random::permutations::Generator::permuted`
//   - `ferray_random::permutations::Generator::permuted_dyn`
//   - `ferray_random::permutations::Generator::shuffle_dyn`
// ---------------------------------------------------------------------------
#[test]
fn dynamic_permutation_helpers_match_numpy_shape_and_multiset_contracts() {
    let mut rng = default_rng_seeded(180);
    let range = rng.permutation_range(32).expect("permutation_range");
    let mut got = range.as_slice().unwrap().to_vec();
    got.sort_unstable();
    assert_eq!(got, (0..32).collect::<Vec<i64>>());

    let source = Array::<i64, Ix1>::from_vec(Ix1::new([8]), (0..8).collect()).unwrap();
    let permuted = rng.permuted(&source, 0).expect("permuted");
    let mut got = permuted.as_slice().unwrap().to_vec();
    got.sort_unstable();
    assert_eq!(got, (0..8).collect::<Vec<i64>>());
    assert_eq!(source.as_slice().unwrap(), &(0..8).collect::<Vec<i64>>());

    let mut matrix = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), (0..12).collect()).unwrap();
    rng.shuffle_dyn(&mut matrix, 0).expect("shuffle_dyn");
    let mut rows = matrix
        .as_slice()
        .unwrap()
        .chunks_exact(4)
        .map(<[i64]>::to_vec)
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9, 10, 11]]
    );

    let arr = Array::<i64, IxDyn>::from_vec(IxDyn::new(&[3, 4]), (0..12).collect()).unwrap();
    let chosen = rng
        .choice_dyn(&arr, 2, false, None, 0, false)
        .expect("choice_dyn");
    assert_eq!(chosen.shape(), &[2, 4]);
    let chosen_values = chosen.as_slice().unwrap();
    for row in chosen_values.chunks_exact(4) {
        assert!(row == [0, 1, 2, 3] || row == [4, 5, 6, 7] || row == [8, 9, 10, 11]);
    }

    let independently_permuted = rng.permuted_dyn(&arr, 0).expect("permuted_dyn");
    assert_eq!(independently_permuted.shape(), &[3, 4]);
    for col in 0..4 {
        let mut values = (0..3)
            .map(|row| independently_permuted.as_slice().unwrap()[row * 4 + col])
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, vec![col as i64, (col + 4) as i64, (col + 8) as i64]);
    }
}
