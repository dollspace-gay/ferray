// ferray-random: BitGenerator trait and implementations

use ferray_core::FerrayError;

mod mt19937;
mod pcg64;
mod pcg64dxsm;
mod philox;
mod seed_sequence;
mod sfc64;
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

pub use mt19937::MT19937;
pub use pcg64::Pcg64;
pub use pcg64dxsm::Pcg64Dxsm;
pub use philox::Philox;
pub use seed_sequence::SeedSequence;
pub use sfc64::Sfc64;
pub use xoshiro256::Xoshiro256StarStar;

/// Trait for pluggable pseudo-random number generators.
///
/// All `BitGenerators` are `Send` (can be transferred between threads) but NOT `Sync`
/// (they are stateful and require `&mut self`).
///
/// Concrete implementations: [`Pcg64`], [`Pcg64Dxsm`], [`Philox`],
/// [`Xoshiro256StarStar`], [`MT19937`], [`Sfc64`].
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

    /// Serialize the full internal state of this generator to a
    /// little-endian byte vector. Pair with [`set_state_bytes`] to
    /// restore.
    ///
    /// Default impl returns an error — concrete generators that
    /// support serialization override both methods. Used to checkpoint
    /// reproducible experiments (#453), the rough Rust equivalent of
    /// numpy's `Generator.bit_generator.state` / `__getstate__` /
    /// `__setstate__`.
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if the generator does not implement
    /// state serialization.
    fn state_bytes(&self) -> Result<Vec<u8>, FerrayError> {
        Err(FerrayError::invalid_value(
            "this BitGenerator does not implement state_bytes",
        ))
    }

    /// Restore the generator's state from previously-serialized bytes
    /// produced by [`state_bytes`].
    ///
    /// # Errors
    /// `FerrayError::InvalidValue` if the byte length doesn't match
    /// the expected state size, or if the embedded state is invalid
    /// (e.g. all-zero state for Xoshiro256**).
    fn set_state_bytes(&mut self, bytes: &[u8]) -> Result<(), FerrayError> {
        let _ = bytes;
        Err(FerrayError::invalid_value(
            "this BitGenerator does not implement set_state_bytes",
        ))
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

#[cfg(test)]
mod state_tests {
    //! Round-trip tests for `state_bytes` / `set_state_bytes` (#453).
    //!
    //! For each concrete BitGenerator: capture state, draw N values,
    //! roll back to the captured state, draw N more — outputs must
    //! match exactly.

    use super::*;

    fn roundtrip<B: BitGenerator + Sized>(make: impl Fn() -> B, expected_size: usize) {
        let mut a = make();
        // Burn a few values so we're not at the seeded initial state.
        for _ in 0..7 {
            a.next_u64();
        }
        let snapshot = a.state_bytes().unwrap();
        assert_eq!(
            snapshot.len(),
            expected_size,
            "{} expected state size {expected_size}, got {}",
            std::any::type_name::<B>(),
            snapshot.len()
        );

        let mut from_a: Vec<u64> = Vec::with_capacity(64);
        for _ in 0..64 {
            from_a.push(a.next_u64());
        }

        let mut b = make();
        b.set_state_bytes(&snapshot).unwrap();
        let mut from_b: Vec<u64> = Vec::with_capacity(64);
        for _ in 0..64 {
            from_b.push(b.next_u64());
        }
        assert_eq!(from_a, from_b, "round-trip diverged for {}", std::any::type_name::<B>());
    }

    #[test]
    fn roundtrip_xoshiro256() {
        roundtrip(|| Xoshiro256StarStar::seed_from_u64(42), 32);
    }

    #[test]
    fn roundtrip_sfc64() {
        roundtrip(|| Sfc64::seed_from_u64(0xc0ffee), 32);
    }

    #[test]
    fn roundtrip_pcg64() {
        roundtrip(|| Pcg64::seed_from_u64(7), 32);
    }

    #[test]
    fn roundtrip_pcg64dxsm() {
        roundtrip(|| Pcg64Dxsm::seed_from_u64(7), 32);
    }

    #[test]
    fn roundtrip_philox() {
        roundtrip(|| Philox::seed_from_u64(123), 44);
    }

    #[test]
    fn roundtrip_mt19937() {
        roundtrip(|| MT19937::seed_from_u64(2026), 312 * 8 + 8);
    }

    #[test]
    fn xoshiro_rejects_all_zero_state() {
        let mut g = Xoshiro256StarStar::seed_from_u64(0);
        let zeros = vec![0u8; 32];
        assert!(g.set_state_bytes(&zeros).is_err());
    }

    #[test]
    fn pcg64_rejects_even_inc() {
        let mut g = Pcg64::seed_from_u64(0);
        let mut bad = vec![0u8; 32];
        bad[16] = 2; // inc low byte even
        assert!(g.set_state_bytes(&bad).is_err());
    }

    #[test]
    fn philox_rejects_oversize_buf_idx() {
        let mut g = Philox::seed_from_u64(0);
        let mut bad = vec![0u8; 44];
        bad[40] = 5; // buf_idx > 4
        assert!(g.set_state_bytes(&bad).is_err());
    }

    #[test]
    fn wrong_length_returns_error() {
        let mut g = Xoshiro256StarStar::seed_from_u64(0);
        assert!(g.set_state_bytes(&[0u8; 16]).is_err());
    }
}
