// ferray-random: Generator struct — the main user-facing RNG API
//
// Wraps a BitGenerator and provides distribution sampling methods.
// Takes &mut self — stateful, NOT Sync.

use ferray_core::{Array, FerrayError, IxDyn};

use crate::bitgen::{BitGenerator, Xoshiro256StarStar};

/// The main random number generator, wrapping a pluggable [`BitGenerator`].
///
/// `Generator` takes `&mut self` for all sampling methods — it is stateful
/// and NOT `Sync`. Thread-safety is handled by spawning independent generators
/// via [`spawn`](Generator::spawn) or using the parallel generation API.
///
/// # Example
/// ```
/// use ferray_random::{default_rng_seeded, Generator};
///
/// let mut rng = default_rng_seeded(42);
/// let values = rng.random(10).unwrap();
/// assert_eq!(values.shape(), &[10]);
/// ```
pub struct Generator<B: BitGenerator = Xoshiro256StarStar> {
    /// The underlying bit generator.
    pub(crate) bg: B,
    /// The seed used to create this generator (for spawn).
    pub(crate) seed: u64,
}

impl<B: BitGenerator> Generator<B> {
    /// Create a new `Generator` wrapping the given `BitGenerator`.
    pub const fn new(bg: B) -> Self {
        Self { bg, seed: 0 }
    }

    /// Create a new `Generator` with a known seed (stored for spawn).
    pub(crate) const fn new_with_seed(bg: B, seed: u64) -> Self {
        Self { bg, seed }
    }

    /// Access the underlying `BitGenerator` mutably.
    #[inline]
    pub const fn bit_generator(&mut self) -> &mut B {
        &mut self.bg
    }

    /// Generate the next random `u64`.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.bg.next_u64()
    }

    /// Generate the next random `f64` in [0, 1).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        self.bg.next_f64()
    }

    /// Generate the next random `f32` in [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        self.bg.next_f32()
    }

    /// Generate a `u64` in [0, bound).
    #[inline]
    pub fn next_u64_bounded(&mut self, bound: u64) -> u64 {
        self.bg.next_u64_bounded(bound)
    }

    /// Serialize the underlying [`BitGenerator`]'s full internal state
    /// to a byte vector — pair with [`set_state_bytes`](Self::set_state_bytes)
    /// to restore. Used to checkpoint reproducible experiments (#453).
    ///
    /// The format is the LE-byte serialization of the bit generator's
    /// state words; it is stable per-generator-type but **not**
    /// portable across different `BitGenerator` implementations
    /// (Pcg64 state cannot be loaded into Xoshiro256**).
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if the underlying generator does
    /// not implement state serialization.
    pub fn state_bytes(&self) -> Result<Vec<u8>, FerrayError> {
        self.bg.state_bytes()
    }

    /// Restore the underlying [`BitGenerator`]'s state from previously
    /// captured bytes.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if the byte length is wrong for
    /// this generator type or the embedded state is invalid (e.g.
    /// all-zero state for Xoshiro256**, even `inc` for Pcg64).
    pub fn set_state_bytes(&mut self, bytes: &[u8]) -> Result<(), FerrayError> {
        self.bg.set_state_bytes(bytes)
    }

    /// Generate `n` random bytes as a `Vec<u8>`.
    ///
    /// Equivalent to `numpy.random.Generator.bytes(n)`. Each byte is
    /// drawn from the underlying bit generator's `u64` stream and
    /// little-endian-decomposed; calling `bytes(n)` advances the bit
    /// generator by `ceil(n / 8)` `u64` draws (#446).
    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n);
        let full_words = n / 8;
        for _ in 0..full_words {
            out.extend_from_slice(&self.bg.next_u64().to_le_bytes());
        }
        let remainder = n % 8;
        if remainder > 0 {
            let bytes = self.bg.next_u64().to_le_bytes();
            out.extend_from_slice(&bytes[..remainder]);
        }
        out
    }
}

/// Create a `Generator` with the default `BitGenerator` (Xoshiro256**)
/// seeded from a non-deterministic source (using the system time as a
/// simple entropy source).
///
/// # Example
/// ```
/// let mut rng = ferray_random::default_rng();
/// let val = rng.next_f64();
/// assert!((0.0..1.0).contains(&val));
/// ```
#[must_use]
pub fn default_rng() -> Generator<Xoshiro256StarStar> {
    // Use OS entropy via getrandom for proper seeding.
    // Falls back to time-based entropy if getrandom fails.
    let seed = {
        let mut buf = [0u8; 8];
        if getrandom::fill(&mut buf).is_ok() {
            u64::from_ne_bytes(buf)
        } else {
            // Fallback: time + stack address
            use std::time::SystemTime;
            let dur = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or_default();
            let nanos = dur.as_nanos();
            let mut s = nanos as u64;
            s ^= (nanos >> 64) as u64;
            let stack_var: u8 = 0;
            s ^= &raw const stack_var as u64;
            s
        }
    };
    default_rng_seeded(seed)
}

/// Create a `Generator` with the default `BitGenerator` (Xoshiro256**)
/// from a specific seed, ensuring deterministic output.
///
/// # Example
/// ```
/// let mut rng1 = ferray_random::default_rng_seeded(42);
/// let mut rng2 = ferray_random::default_rng_seeded(42);
/// assert_eq!(rng1.next_u64(), rng2.next_u64());
/// ```
#[must_use]
pub fn default_rng_seeded(seed: u64) -> Generator<Xoshiro256StarStar> {
    let bg = Xoshiro256StarStar::seed_from_u64(seed);
    Generator::new_with_seed(bg, seed)
}

/// Spawn `n` independent child generators from this generator.
///
/// Uses `jump()` if available (Xoshiro256**), otherwise uses
/// `stream()` (Philox), otherwise falls back to seeding from
/// the parent generator's output.
///
/// # Errors
/// Returns `FerrayError::InvalidValue` if `n` is zero.
pub fn spawn_generators<B: BitGenerator + Clone>(
    parent: &mut Generator<B>,
    n: usize,
) -> Result<Vec<Generator<B>>, FerrayError> {
    if n == 0 {
        return Err(FerrayError::invalid_value("spawn count must be > 0"));
    }

    let mut children = Vec::with_capacity(n);

    // Try jump-based spawning first
    let mut test_bg = parent.bg.clone();
    if test_bg.jump().is_some() {
        // Jump-based: each child starts at a 2^128 offset
        let mut current = parent.bg.clone();
        for _ in 0..n {
            children.push(Generator::new(current.clone()));
            current.jump();
        }
        // Advance parent past all children
        parent.bg = current;
        return Ok(children);
    }

    // Try stream-based spawning
    if let Some(first) = B::stream(parent.seed, 0) {
        drop(first);
        for i in 0..n {
            if let Some(bg) = B::stream(parent.seed, i as u64) {
                children.push(Generator::new(bg));
            }
        }
        if children.len() == n {
            return Ok(children);
        }
        children.clear();
    }

    // Fallback: seed from parent output (less ideal but works for PCG64)
    for _ in 0..n {
        let child_seed = parent.bg.next_u64();
        let bg = B::seed_from_u64(child_seed);
        children.push(Generator::new(bg));
    }
    Ok(children)
}

// Helper: generate a Vec<f64> of given total size using a closure.
pub(crate) fn generate_vec<B: BitGenerator>(
    rng: &mut Generator<B>,
    size: usize,
    mut f: impl FnMut(&mut B) -> f64,
) -> Vec<f64> {
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(f(&mut rng.bg));
    }
    data
}

// Helper: generate a Vec<f32> of given total size using a closure.
pub(crate) fn generate_vec_f32<B: BitGenerator>(
    rng: &mut Generator<B>,
    size: usize,
    mut f: impl FnMut(&mut B) -> f32,
) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(f(&mut rng.bg));
    }
    data
}

// Helper: generate a Vec<i64> of given total size using a closure.
pub(crate) fn generate_vec_i64<B: BitGenerator>(
    rng: &mut Generator<B>,
    size: usize,
    mut f: impl FnMut(&mut B) -> i64,
) -> Vec<i64> {
    let mut data = Vec::with_capacity(size);
    for _ in 0..size {
        data.push(f(&mut rng.bg));
    }
    data
}

/// Total element count for a shape, returning 0 for an empty shape.
#[inline]
pub(crate) fn shape_size(shape: &[usize]) -> usize {
    if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    }
}

/// Wrap a `Vec<f64>` into an `Array<f64, IxDyn>` with the given shape.
pub(crate) fn vec_to_array_f64(
    data: Vec<f64>,
    shape: &[usize],
) -> Result<Array<f64, IxDyn>, FerrayError> {
    Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), data)
}

/// Wrap a `Vec<f32>` into an `Array<f32, IxDyn>` with the given shape.
pub(crate) fn vec_to_array_f32(
    data: Vec<f32>,
    shape: &[usize],
) -> Result<Array<f32, IxDyn>, FerrayError> {
    Array::<f32, IxDyn>::from_vec(IxDyn::new(shape), data)
}

/// Wrap a `Vec<i64>` into an `Array<i64, IxDyn>` with the given shape.
pub(crate) fn vec_to_array_i64(
    data: Vec<i64>,
    shape: &[usize],
) -> Result<Array<i64, IxDyn>, FerrayError> {
    Array::<i64, IxDyn>::from_vec(IxDyn::new(shape), data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_bytes_roundtrip_via_generator() {
        // #453: Generator::state_bytes / set_state_bytes round-trip
        // — capture state, draw a chunk, restore, draw the same chunk
        // again and verify byte-equality.
        let mut a = default_rng_seeded(2026);
        // Burn a few values so we are not at the seed boundary.
        for _ in 0..11 {
            a.next_u64();
        }
        let snap = a.state_bytes().unwrap();
        let from_a: Vec<u64> = (0..32).map(|_| a.next_u64()).collect();

        let mut b = default_rng_seeded(0); // wrong seed on purpose
        b.set_state_bytes(&snap).unwrap();
        let from_b: Vec<u64> = (0..32).map(|_| b.next_u64()).collect();
        assert_eq!(from_a, from_b);
    }

    #[test]
    fn set_state_bytes_rejects_wrong_size() {
        let mut a = default_rng_seeded(0);
        assert!(a.set_state_bytes(&[0u8; 4]).is_err());
    }

    #[test]
    fn default_rng_seeded_deterministic() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn default_rng_works() {
        let mut rng = default_rng();
        let v = rng.next_f64();
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn spawn_xoshiro() {
        let mut parent = default_rng_seeded(42);
        let children = spawn_generators(&mut parent, 4).unwrap();
        assert_eq!(children.len(), 4);
    }

    #[test]
    fn spawn_zero_is_error() {
        let mut parent = default_rng_seeded(42);
        assert!(spawn_generators(&mut parent, 0).is_err());
    }

    // ----- bytes() coverage (#446) -----

    #[test]
    fn bytes_length_zero() {
        let mut rng = default_rng_seeded(42);
        assert!(rng.bytes(0).is_empty());
    }

    #[test]
    fn bytes_length_full_word() {
        let mut rng = default_rng_seeded(42);
        let b = rng.bytes(8);
        assert_eq!(b.len(), 8);
    }

    #[test]
    fn bytes_length_partial_word() {
        let mut rng = default_rng_seeded(42);
        let b = rng.bytes(13);
        assert_eq!(b.len(), 13);
    }

    #[test]
    fn bytes_deterministic_for_same_seed() {
        let mut rng1 = default_rng_seeded(42);
        let mut rng2 = default_rng_seeded(42);
        assert_eq!(rng1.bytes(64), rng2.bytes(64));
    }
}
