// ferray-random: BitGenerator trait and implementations

mod pcg64;
mod philox;
mod xoshiro256;

/// `SplitMix64`: a fast 64-bit hash-based PRNG used exclusively for
/// seeding other generators (Xoshiro256**, PCG64) from a single u64
/// seed. The function was previously copy-pasted into both
/// `pcg64.rs` and `xoshiro256.rs` (#259).
pub(crate) const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub use pcg64::Pcg64;
pub use philox::Philox;
pub use xoshiro256::Xoshiro256StarStar;

/// Trait for pluggable pseudo-random number generators.
///
/// All `BitGenerators` are `Send` (can be transferred between threads) but NOT `Sync`
/// (they are stateful and require `&mut self`).
///
/// Concrete implementations: [`Pcg64`], [`Philox`], [`Xoshiro256StarStar`].
pub trait BitGenerator: Send {
    /// Generate the next 64-bit unsigned integer.
    fn next_u64(&mut self) -> u64;

    /// Create a new generator seeded from a single `u64`.
    fn seed_from_u64(seed: u64) -> Self
    where
        Self: Sized;

    /// Advance the generator state by a large step (2^128 for Xoshiro256**).
    ///
    /// Returns `Some(())` if jump is supported, `None` otherwise.
    /// After calling `jump`, the generator's state has advanced as if
    /// `2^128` calls to `next_u64` had been made.
    fn jump(&mut self) -> Option<()>;

    /// Create a new generator from a seed and a stream ID.
    ///
    /// Returns `Some(Self)` if the generator supports stream-based parallelism
    /// (e.g., Philox), `None` otherwise.
    fn stream(seed: u64, stream_id: u64) -> Option<Self>
    where
        Self: Sized;

    /// Generate a uniformly distributed `f64` in [0, 1).
    ///
    /// Uses the upper 53 bits of `next_u64()` for full double precision.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Generate a uniformly distributed `f32` in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Fill a byte slice with random bytes.
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        while i + 8 <= dest.len() {
            let val = self.next_u64();
            dest[i..i + 8].copy_from_slice(&val.to_le_bytes());
            i += 8;
        }
        if i < dest.len() {
            let val = self.next_u64();
            let bytes = val.to_le_bytes();
            for (j, byte) in dest[i..].iter_mut().enumerate() {
                *byte = bytes[j];
            }
        }
    }

    /// Generate a `u64` in the range `[0, bound)` using rejection sampling.
    fn next_u64_bounded(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        // Lemire's nearly divisionless method
        let mut x = self.next_u64();
        let mut m = (x as u128) * (bound as u128);
        let mut l = m as u64;
        if l < bound {
            let threshold = bound.wrapping_neg() % bound;
            while l < threshold {
                x = self.next_u64();
                m = (x as u128) * (bound as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
}
