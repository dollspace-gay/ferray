// ferray-random: Xoshiro256** BitGenerator implementation
//
// Reference: David Blackman and Sebastiano Vigna, "Scrambled Linear
// Pseudorandom Number Generators", ACM TOMS, 2021.
// Period: 2^256 - 1. Jump: 2^128.

// Jump-table constants are 64-bit hex magic numbers from the Vigna paper;
// underscore separators would diverge from the canonical published form.
#![allow(clippy::unreadable_literal)]

use super::BitGenerator;

/// Xoshiro256** pseudo-random number generator.
///
/// This is the default `BitGenerator` for ferray-random. It has a period of
/// 2^256 - 1 and supports `jump()` for parallel generation (each jump
/// advances the state by 2^128 steps).
///
/// # Example
/// ```
/// use ferray_random::bitgen::Xoshiro256StarStar;
/// use ferray_random::bitgen::BitGenerator;
///
/// let mut rng = Xoshiro256StarStar::seed_from_u64(42);
/// let val = rng.next_u64();
/// ```
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

impl Xoshiro256StarStar {
    /// Create from an explicit 4-element state. The state must not be all zeros.
    fn from_state(s: [u64; 4]) -> Self {
        debug_assert!(
            s != [0, 0, 0, 0],
            "Xoshiro256** state must not be all zeros"
        );
        Self { s }
    }
}

impl BitGenerator for Xoshiro256StarStar {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        let mut out = Vec::with_capacity(32);
        for &w in &self.s {
            out.extend_from_slice(&w.to_le_bytes());
        }
        Ok(out)
    }

    fn set_state_bytes(&mut self, bytes: &[u8]) -> Result<(), ferray_core::FerrayError> {
        if bytes.len() != 32 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Xoshiro256** state must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        let mut s = [0u64; 4];
        for (i, chunk) in bytes.chunks_exact(8).enumerate() {
            s[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        if s == [0, 0, 0, 0] {
            return Err(ferray_core::FerrayError::invalid_value(
                "Xoshiro256** state must not be all zeros",
            ));
        }
        self.s = s;
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    fn seed_from_u64(seed: u64) -> Self {
        // Use the shared splitmix64 helper (#259).
        let mut sm = seed;
        let s0 = super::splitmix64(&mut sm);
        let s1 = super::splitmix64(&mut sm);
        let s2 = super::splitmix64(&mut sm);
        let s3 = super::splitmix64(&mut sm);
        // Ensure state is not all zeros
        if s0 | s1 | s2 | s3 == 0 {
            Self::from_state([1, 0, 0, 0])
        } else {
            Self::from_state([s0, s1, s2, s3])
        }
    }

    fn jump(&mut self) -> Option<()> {
        // Jump polynomial for 2^128 steps
        const JUMP: [u64; 4] = [
            0x180ec6d33cfd0aba,
            0xd5a61266f0c9392c,
            0xa9582618e03fc9aa,
            0x39abdc4529b1661c,
        ];

        let mut s0: u64 = 0;
        let mut s1: u64 = 0;
        let mut s2: u64 = 0;
        let mut s3: u64 = 0;

        for &jmp in &JUMP {
            for b in 0..64 {
                if (jmp >> b) & 1 != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }

        self.s[0] = s0;
        self.s[1] = s1;
        self.s[2] = s2;
        self.s[3] = s3;

        Some(())
    }

    fn stream(_seed: u64, _stream_id: u64) -> Option<Self> {
        // Xoshiro256** does not support stream IDs; use jump() instead
        None
    }
}

impl Clone for Xoshiro256StarStar {
    fn clone(&self) -> Self {
        Self { s: self.s }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut rng1 = Xoshiro256StarStar::seed_from_u64(42);
        let mut rng2 = Xoshiro256StarStar::seed_from_u64(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut rng1 = Xoshiro256StarStar::seed_from_u64(42);
        let mut rng2 = Xoshiro256StarStar::seed_from_u64(43);
        let mut same = true;
        for _ in 0..100 {
            if rng1.next_u64() != rng2.next_u64() {
                same = false;
                break;
            }
        }
        assert!(!same);
    }

    #[test]
    fn jump_advances_state() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(1234);
        let before = rng.next_u64();
        let mut rng2 = Xoshiro256StarStar::seed_from_u64(1234);
        let _ = rng2.next_u64();
        rng2.jump();
        let after = rng2.next_u64();
        // After jump, output should differ
        assert_ne!(before, after);
    }

    #[test]
    fn stream_not_supported() {
        assert!(Xoshiro256StarStar::stream(42, 0).is_none());
    }

    #[test]
    fn next_f64_range() {
        let mut rng = Xoshiro256StarStar::seed_from_u64(999);
        for _ in 0..10_000 {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
