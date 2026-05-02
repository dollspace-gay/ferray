// ferray-random: PCG64 BitGenerator implementation
//
// PCG-XSL-RR 128/64 (LCG) from Melissa O'Neill's PCG paper.
// Period: 2^128. No jump support.

// Hex magic constants are 64-bit nothing-up-my-sleeve seeds reproduced
// verbatim from the PCG and SplitMix64 references; underscore separators
// would diverge from the canonical published form.
#![allow(clippy::unreadable_literal)]

use super::BitGenerator;

/// PCG64 (PCG-XSL-RR 128/64) pseudo-random number generator.
///
/// Uses a 128-bit linear congruential generator with a permutation-based
/// output function. Period is 2^128. Does not support `jump()` or `stream()`.
///
/// # Example
/// ```
/// use ferray_random::bitgen::Pcg64;
/// use ferray_random::bitgen::BitGenerator;
///
/// let mut rng = Pcg64::seed_from_u64(42);
/// let val = rng.next_u64();
/// ```
pub struct Pcg64 {
    state: u128,
    inc: u128,
}

/// Default multiplier for the LCG (from PCG paper).
const PCG_DEFAULT_MULTIPLIER: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

impl Pcg64 {
    /// Internal step function: advance the LCG state.
    #[inline]
    const fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER)
            .wrapping_add(self.inc);
    }

    /// Output function: XSL-RR permutation of the 128-bit state to 64-bit output.
    #[inline]
    const fn output(state: u128) -> u64 {
        let xsl = ((state >> 64) ^ state) as u64;
        let rot = (state >> 122) as u32;
        xsl.rotate_right(rot)
    }
}

impl BitGenerator for Pcg64 {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&self.state.to_le_bytes());
        out.extend_from_slice(&self.inc.to_le_bytes());
        Ok(out)
    }

    fn set_state_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), ferray_core::FerrayError> {
        if bytes.len() != 32 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Pcg64 state must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        self.state = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
        let inc = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
        if inc & 1 == 0 {
            return Err(ferray_core::FerrayError::invalid_value(
                "Pcg64 inc must be odd",
            ));
        }
        self.inc = inc;
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        let old_state = self.state;
        self.step();
        Self::output(old_state)
    }

    fn seed_from_u64(seed: u64) -> Self {
        // Use SplitMix64 expansion for seeding (#259 — shared helper).
        let seed128 = {
            let mut s = seed;
            let a = super::splitmix64(&mut s);
            let b = super::splitmix64(&mut s);
            ((a as u128) << 64) | (b as u128)
        };
        // inc must be odd
        let inc = {
            let mut s = seed.wrapping_add(0xda3e39cb94b95bdb);
            let a = super::splitmix64(&mut s);
            let b = super::splitmix64(&mut s);
            (((a as u128) << 64) | (b as u128)) | 1
        };

        let mut rng = Self { state: 0, inc };
        rng.step();
        rng.state = rng.state.wrapping_add(seed128);
        rng.step();
        rng
    }

    fn jump(&mut self) -> Option<()> {
        // PCG64 does not support jump-ahead
        None
    }

    fn stream(_seed: u64, _stream_id: u64) -> Option<Self> {
        // PCG64 does not support stream IDs in this implementation
        None
    }
}

impl Clone for Pcg64 {
    fn clone(&self) -> Self {
        Self {
            state: self.state,
            inc: self.inc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut rng1 = Pcg64::seed_from_u64(42);
        let mut rng2 = Pcg64::seed_from_u64(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut rng1 = Pcg64::seed_from_u64(42);
        let mut rng2 = Pcg64::seed_from_u64(43);
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
    fn jump_not_supported() {
        let mut rng = Pcg64::seed_from_u64(42);
        assert!(rng.jump().is_none());
    }

    #[test]
    fn stream_not_supported() {
        assert!(Pcg64::stream(42, 0).is_none());
    }

    #[test]
    fn output_covers_full_range() {
        let mut rng = Pcg64::seed_from_u64(12345);
        let mut seen_high = false;
        let mut seen_low = false;
        for _ in 0..10_000 {
            let v = rng.next_u64();
            if v > (u64::MAX / 2) {
                seen_high = true;
            } else {
                seen_low = true;
            }
            if seen_high && seen_low {
                break;
            }
        }
        assert!(seen_high && seen_low);
    }
}
