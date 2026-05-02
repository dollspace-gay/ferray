// ferray-random: PCG64DXSM BitGenerator implementation
//
// PCG-DXSM 128/64 (cheap-multiply variant of PCG-XSL-RR) — recommended
// upgrade over PCG64 by Melissa O'Neill in 2022. NumPy added it as
// `PCG64DXSM` in 1.21. Period 2^128.

#![allow(clippy::unreadable_literal)]

use super::BitGenerator;

/// PCG64DXSM (DXSM output function) pseudo-random number generator.
///
/// 128-bit linear congruential generator with the DXSM (double-xorshift
/// multiply) output function. Period 2^128. Recommended over the
/// classic XSL-RR `Pcg64` for new applications because the DXSM
/// permutation has stronger empirical bit-level randomness.
pub struct Pcg64Dxsm {
    state: u128,
    inc: u128,
}

const PCG_CHEAP_MULTIPLIER: u64 = 0xda94_2042_e4dd_58b5;

impl Pcg64Dxsm {
    #[inline]
    const fn step(&mut self) {
        // Cheap-multiplier LCG advance: state = state * mult + inc, where
        // mult is a 64-bit constant lifted to 128 bits.
        let mult128 = PCG_CHEAP_MULTIPLIER as u128;
        self.state = self.state.wrapping_mul(mult128).wrapping_add(self.inc);
    }

    #[inline]
    const fn output(state: u128) -> u64 {
        // DXSM permutation:
        //   hi = state >> 64; lo = state | 1
        //   hi ^= hi >> 32
        //   hi *= cheap_mult
        //   hi ^= hi >> 48
        //   hi *= lo
        let hi0 = (state >> 64) as u64;
        let lo = (state as u64) | 1;
        let mut hi = hi0;
        hi ^= hi >> 32;
        hi = hi.wrapping_mul(PCG_CHEAP_MULTIPLIER);
        hi ^= hi >> 48;
        hi = hi.wrapping_mul(lo);
        hi
    }
}

impl BitGenerator for Pcg64Dxsm {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&self.state.to_le_bytes());
        out.extend_from_slice(&self.inc.to_le_bytes());
        Ok(out)
    }

    fn set_state_bytes(&mut self, bytes: &[u8]) -> Result<(), ferray_core::FerrayError> {
        if bytes.len() != 32 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Pcg64Dxsm state must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        self.state = u128::from_le_bytes(bytes[0..16].try_into().unwrap());
        let inc = u128::from_le_bytes(bytes[16..32].try_into().unwrap());
        if inc & 1 == 0 {
            return Err(ferray_core::FerrayError::invalid_value(
                "Pcg64Dxsm inc must be odd",
            ));
        }
        self.inc = inc;
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        let old = self.state;
        self.step();
        Self::output(old)
    }

    fn seed_from_u64(seed: u64) -> Self {
        let seed128 = {
            let mut s = seed;
            let a = super::splitmix64(&mut s);
            let b = super::splitmix64(&mut s);
            ((a as u128) << 64) | (b as u128)
        };
        let inc = {
            let mut s = seed.wrapping_add(0xda3e_39cb_94b9_5bdb);
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
        None
    }

    fn stream(_seed: u64, _stream_id: u64) -> Option<Self> {
        None
    }
}

impl Clone for Pcg64Dxsm {
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
        let mut a = Pcg64Dxsm::seed_from_u64(7);
        let mut b = Pcg64Dxsm::seed_from_u64(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Pcg64Dxsm::seed_from_u64(7);
        let mut b = Pcg64Dxsm::seed_from_u64(8);
        let mut diff = false;
        for _ in 0..100 {
            if a.next_u64() != b.next_u64() {
                diff = true;
                break;
            }
        }
        assert!(diff);
    }

    #[test]
    fn full_range() {
        let mut rng = Pcg64Dxsm::seed_from_u64(0xfeed_face);
        let mut hi = false;
        let mut lo = false;
        for _ in 0..10_000 {
            let v = rng.next_u64();
            if v > u64::MAX / 2 {
                hi = true;
            } else {
                lo = true;
            }
            if hi && lo {
                break;
            }
        }
        assert!(hi && lo);
    }
}
