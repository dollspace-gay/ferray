// ferray-random: Philox 4x32 counter-based BitGenerator
//
// Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
// SC '11: Proceedings of the 2011 International Conference for High
// Performance Computing.
//
// Philox 4x32-10: 4 words of 32 bits, 10 rounds.
// Supports stream IDs natively (the key encodes the stream).

use super::BitGenerator;

/// Philox 4x32-10 counter-based pseudo-random number generator.
///
/// This generator natively supports stream IDs, making it ideal for
/// parallel generation. Each (seed, `stream_id`) pair produces an
/// independent, non-overlapping output sequence.
///
/// # Example
/// ```
/// use ferray_random::bitgen::Philox;
/// use ferray_random::bitgen::BitGenerator;
///
/// let mut rng = Philox::seed_from_u64(42);
/// let val = rng.next_u64();
/// ```
pub struct Philox {
    /// 4x32-bit counter
    counter: [u32; 4],
    /// 2x32-bit key (derived from seed and stream)
    key: [u32; 2],
    /// Buffered output (4 x u32 = 2 x u64)
    buffer: [u32; 4],
    /// Index into buffer (0..4)
    buf_idx: usize,
}

// Philox 4x32 round constants
const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// Single Philox round: two multiplications with xor-folding.
///
/// `key` is taken by reference so the caller's stationary key buffer
/// stays put across the unrolled round sequence; copying the 8-byte key
/// every round defeats register allocation in release builds.
#[inline]
#[allow(clippy::trivially_copy_pass_by_ref)]
const fn philox_round(ctr: &mut [u32; 4], key: &[u32; 2]) {
    let lo0 = ctr[0] as u64 * PHILOX_M0 as u64;
    let lo1 = ctr[2] as u64 * PHILOX_M1 as u64;
    let hi0 = (lo0 >> 32) as u32;
    let lo0 = lo0 as u32;
    let hi1 = (lo1 >> 32) as u32;
    let lo1 = lo1 as u32;
    let new0 = hi1 ^ ctr[1] ^ key[0];
    let new1 = lo1;
    let new2 = hi0 ^ ctr[3] ^ key[1];
    let new3 = lo0;
    ctr[0] = new0;
    ctr[1] = new1;
    ctr[2] = new2;
    ctr[3] = new3;
}

/// Bump key between rounds.
#[inline]
const fn philox_bump_key(key: &mut [u32; 2]) {
    key[0] = key[0].wrapping_add(PHILOX_W0);
    key[1] = key[1].wrapping_add(PHILOX_W1);
}

/// Full Philox 4x32-10 bijection: 10 rounds.
const fn philox4x32_10(counter: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut ctr = counter;
    let mut k = key;
    // 10 rounds with key bumps between rounds
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    philox_bump_key(&mut k);
    philox_round(&mut ctr, &k);
    ctr
}

impl Philox {
    /// Increment the 128-bit counter (4 x u32, little-endian).
    fn increment_counter(&mut self) {
        for word in &mut self.counter {
            *word = word.wrapping_add(1);
            if *word != 0 {
                return;
            }
        }
    }

    /// Generate the next block of 4 random u32 values.
    fn generate_block(&mut self) {
        self.buffer = philox4x32_10(self.counter, self.key);
        self.increment_counter();
        self.buf_idx = 0;
    }

    /// Create a Philox generator with explicit key and starting counter.
    fn new_with_key(key: [u32; 2], counter: [u32; 4]) -> Self {
        let mut rng = Self {
            counter,
            key,
            buffer: [0; 4],
            buf_idx: 4, // Force generation on first call
        };
        rng.generate_block();
        rng
    }
}

impl BitGenerator for Philox {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        // Layout (little-endian, 44 bytes total):
        //   counter:  4 × u32 = 16 bytes
        //   key:      2 × u32 =  8 bytes
        //   buffer:   4 × u32 = 16 bytes
        //   buf_idx:      u32 =  4 bytes
        let mut out = Vec::with_capacity(44);
        for &w in &self.counter {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for &w in &self.key {
            out.extend_from_slice(&w.to_le_bytes());
        }
        for &w in &self.buffer {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.extend_from_slice(&(self.buf_idx as u32).to_le_bytes());
        Ok(out)
    }

    fn set_state_bytes(&mut self, bytes: &[u8]) -> Result<(), ferray_core::FerrayError> {
        if bytes.len() != 44 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Philox state must be 44 bytes, got {}",
                bytes.len()
            )));
        }
        let mut counter = [0u32; 4];
        let mut key = [0u32; 2];
        let mut buffer = [0u32; 4];
        for (i, chunk) in bytes[0..16].chunks_exact(4).enumerate() {
            counter[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        for (i, chunk) in bytes[16..24].chunks_exact(4).enumerate() {
            key[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        for (i, chunk) in bytes[24..40].chunks_exact(4).enumerate() {
            buffer[i] = u32::from_le_bytes(chunk.try_into().unwrap());
        }
        let buf_idx = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        if buf_idx > 4 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Philox buf_idx must be in [0, 4], got {buf_idx}"
            )));
        }
        self.counter = counter;
        self.key = key;
        self.buffer = buffer;
        self.buf_idx = buf_idx as usize;
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        if self.buf_idx >= 4 {
            self.generate_block();
        }
        let lo = self.buffer[self.buf_idx] as u64;
        self.buf_idx += 1;
        if self.buf_idx >= 4 {
            self.generate_block();
        }
        let hi = self.buffer[self.buf_idx] as u64;
        self.buf_idx += 1;
        lo | (hi << 32)
    }

    fn seed_from_u64(seed: u64) -> Self {
        let key = [seed as u32, (seed >> 32) as u32];
        Self::new_with_key(key, [0; 4])
    }

    fn jump(&mut self) -> Option<()> {
        // Philox supports jump by advancing the counter by 2^64
        // (each counter value produces 4 u32 = 2 u64, so this is 2^65 u64 outputs)
        self.counter[2] = self.counter[2].wrapping_add(1);
        if self.counter[2] == 0 {
            self.counter[3] = self.counter[3].wrapping_add(1);
        }
        self.buf_idx = 4; // Force re-generation
        Some(())
    }

    fn stream(seed: u64, stream_id: u64) -> Option<Self> {
        // Encode stream_id into the counter's upper 64 bits
        let key = [seed as u32, (seed >> 32) as u32];
        let counter = [0u32, 0u32, stream_id as u32, (stream_id >> 32) as u32];
        Some(Self::new_with_key(key, counter))
    }
}

impl Clone for Philox {
    fn clone(&self) -> Self {
        Self {
            counter: self.counter,
            key: self.key,
            buffer: self.buffer,
            buf_idx: self.buf_idx,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut rng1 = Philox::seed_from_u64(42);
        let mut rng2 = Philox::seed_from_u64(42);
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut rng1 = Philox::seed_from_u64(42);
        let mut rng2 = Philox::seed_from_u64(43);
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
    fn stream_support() {
        let rng0 = Philox::stream(42, 0);
        let rng1 = Philox::stream(42, 1);
        assert!(rng0.is_some());
        assert!(rng1.is_some());

        let mut rng0 = rng0.unwrap();
        let mut rng1 = rng1.unwrap();
        // Different streams should produce different output
        let v0 = rng0.next_u64();
        let v1 = rng1.next_u64();
        assert_ne!(v0, v1);
    }

    #[test]
    fn jump_support() {
        let mut rng = Philox::seed_from_u64(42);
        assert!(rng.jump().is_some());
    }

    #[test]
    fn stream_deterministic() {
        let mut rng1 = Philox::stream(42, 7).unwrap();
        let mut rng2 = Philox::stream(42, 7).unwrap();
        for _ in 0..1000 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
