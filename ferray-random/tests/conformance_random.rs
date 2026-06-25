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
use ferray_core::dimension::{Ix1, IxDyn};
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
