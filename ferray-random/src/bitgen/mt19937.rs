// ferray-random: MT19937-64 BitGenerator implementation
//
// 64-bit Mersenne Twister (MT19937-64) per Matsumoto & Nishimura (2000).
// Period 2^19937 - 1. Matches NumPy's `MT19937` BitGenerator (which
// historically wraps the 32-bit Mersenne Twister; our implementation
// uses the 64-bit variant since `BitGenerator::next_u64` is the trait
// contract — both are members of the same family with equivalent
// statistical properties).

#![allow(clippy::unreadable_literal)]

use super::BitGenerator;

const NN: usize = 312;
const MM: usize = 156;
const MATRIX_A: u64 = 0xB502_6F5A_A966_19E9;
const UM: u64 = 0xFFFF_FFFF_8000_0000;
const LM: u64 = 0x0000_0000_7FFF_FFFF;

/// MT19937-64 Mersenne Twister BitGenerator.
///
/// Period 2^19937 - 1. The `jump` operation is not implemented for the
/// 64-bit Mersenne Twister in this crate; the standard polynomial-jump
/// table is large and not required for the typical use case.
pub struct MT19937 {
    mt: [u64; NN],
    index: usize,
}

impl MT19937 {
    fn fill_state(&mut self) {
        for i in 0..NN - MM {
            let x = (self.mt[i] & UM) | (self.mt[i + 1] & LM);
            self.mt[i] = self.mt[i + MM] ^ (x >> 1) ^ if x & 1 == 0 { 0 } else { MATRIX_A };
        }
        for i in NN - MM..NN - 1 {
            let x = (self.mt[i] & UM) | (self.mt[i + 1] & LM);
            self.mt[i] = self.mt[i + MM - NN] ^ (x >> 1) ^ if x & 1 == 0 { 0 } else { MATRIX_A };
        }
        let x = (self.mt[NN - 1] & UM) | (self.mt[0] & LM);
        self.mt[NN - 1] = self.mt[MM - 1] ^ (x >> 1) ^ if x & 1 == 0 { 0 } else { MATRIX_A };
        self.index = 0;
    }
}

impl BitGenerator for MT19937 {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        // Layout: NN × u64 followed by index as u64. Total
        // 312 × 8 + 8 = 2504 bytes.
        let mut out = Vec::with_capacity(NN * 8 + 8);
        for &w in &self.mt {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(self.index as u64).to_le_bytes());
        Ok(out)
    }

    fn set_state_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), ferray_core::FerrayError> {
        let expected = NN * 8 + 8;
        if bytes.len() != expected {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "MT19937 state must be {expected} bytes, got {}",
                bytes.len()
            )));
        }
        let mut mt = [0u64; NN];
        for (i, chunk) in bytes[..NN * 8].chunks_exact(8).enumerate() {
            mt[i] = u64::from_le_bytes(chunk.try_into().unwrap());
        }
        let index = u64::from_le_bytes(bytes[NN * 8..].try_into().unwrap());
        if index > NN as u64 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "MT19937 index must be in [0, {NN}], got {index}"
            )));
        }
        self.mt = mt;
        self.index = index as usize;
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        if self.index >= NN {
            self.fill_state();
        }
        let mut x = self.mt[self.index];
        self.index += 1;
        // Tempering.
        x ^= (x >> 29) & 0x5555_5555_5555_5555;
        x ^= (x << 17) & 0x71D6_7FFF_EDA6_0000;
        x ^= (x << 37) & 0xFFF7_EEE0_0000_0000;
        x ^= x >> 43;
        x
    }

    fn seed_from_u64(seed: u64) -> Self {
        let mut mt = [0u64; NN];
        mt[0] = seed;
        for i in 1..NN {
            mt[i] = 6364136223846793005u64
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 62))
                .wrapping_add(i as u64);
        }
        Self { mt, index: NN }
    }

    fn jump(&mut self) -> Option<()> {
        // Polynomial-jump for MT19937-64 isn't included here; users that
        // need stream parallelism should pick `Philox` (which has
        // `stream`) or seed multiple `MT19937`s with disjoint seeds via
        // SeedSequence.
        None
    }

    fn stream(_seed: u64, _stream_id: u64) -> Option<Self> {
        None
    }
}

impl Clone for MT19937 {
    fn clone(&self) -> Self {
        Self {
            mt: self.mt,
            index: self.index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut a = MT19937::seed_from_u64(123);
        let mut b = MT19937::seed_from_u64(123);
        for _ in 0..512 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = MT19937::seed_from_u64(1);
        let mut b = MT19937::seed_from_u64(2);
        let mut differ = false;
        for _ in 0..100 {
            if a.next_u64() != b.next_u64() {
                differ = true;
                break;
            }
        }
        assert!(differ);
    }

    #[test]
    fn output_covers_full_range() {
        let mut rng = MT19937::seed_from_u64(0xdead_beef);
        let mut high = false;
        let mut low = false;
        for _ in 0..10_000 {
            let v = rng.next_u64();
            if v > u64::MAX / 2 {
                high = true;
            } else {
                low = true;
            }
            if high && low {
                break;
            }
        }
        assert!(high && low);
    }

    #[test]
    fn uniform_f64_in_unit_interval() {
        let mut rng = MT19937::seed_from_u64(42);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }
}
