// ferray-random: SFC64 (Small Fast Counting) BitGenerator implementation
//
// Sebastian Vigna's SFC64 — a 4-state 64-bit chaotic counting generator.
// Period >= 2^64 (typical: 2^255). Used by NumPy as the `SFC64`
// BitGenerator since 1.17.

#![allow(clippy::unreadable_literal)]

use super::BitGenerator;

/// SFC64 (Small Fast Counting) pseudo-random number generator.
///
/// Four-word 64-bit state generator by Sebastian Vigna. Very fast,
/// statistically robust on standard test suites. Period at least
/// 2^64 from any starting state and typically 2^255.
pub struct Sfc64 {
    a: u64,
    b: u64,
    c: u64,
    counter: u64,
}

impl Sfc64 {
    #[inline]
    fn advance(&mut self) -> u64 {
        let tmp = self.a.wrapping_add(self.b).wrapping_add(self.counter);
        self.counter = self.counter.wrapping_add(1);
        self.a = self.b ^ (self.b >> 11);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(24).wrapping_add(tmp);
        tmp
    }
}

impl BitGenerator for Sfc64 {
    fn state_bytes(&self) -> Result<Vec<u8>, ferray_core::FerrayError> {
        let mut out = Vec::with_capacity(32);
        out.extend_from_slice(&self.a.to_le_bytes());
        out.extend_from_slice(&self.b.to_le_bytes());
        out.extend_from_slice(&self.c.to_le_bytes());
        out.extend_from_slice(&self.counter.to_le_bytes());
        Ok(out)
    }

    fn set_state_bytes(
        &mut self,
        bytes: &[u8],
    ) -> Result<(), ferray_core::FerrayError> {
        if bytes.len() != 32 {
            return Err(ferray_core::FerrayError::invalid_value(format!(
                "Sfc64 state must be 32 bytes, got {}",
                bytes.len()
            )));
        }
        self.a = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        self.b = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        self.c = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        self.counter = u64::from_le_bytes(bytes[24..32].try_into().unwrap());
        Ok(())
    }

    fn next_u64(&mut self) -> u64 {
        self.advance()
    }

    fn seed_from_u64(seed: u64) -> Self {
        // NumPy's SFC64 init: set a = b = c = seed, counter = 1, then
        // advance 12 times to "warm up" the state. We follow the
        // upstream reference exactly.
        let mut rng = Self {
            a: seed,
            b: seed,
            c: seed,
            counter: 1,
        };
        for _ in 0..12 {
            let _ = rng.advance();
        }
        rng
    }

    fn jump(&mut self) -> Option<()> {
        None
    }

    fn stream(_seed: u64, _stream_id: u64) -> Option<Self> {
        None
    }
}

impl Clone for Sfc64 {
    fn clone(&self) -> Self {
        Self {
            a: self.a,
            b: self.b,
            c: self.c,
            counter: self.counter,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_output() {
        let mut a = Sfc64::seed_from_u64(0xc0ffee);
        let mut b = Sfc64::seed_from_u64(0xc0ffee);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = Sfc64::seed_from_u64(1);
        let mut b = Sfc64::seed_from_u64(2);
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
        let mut rng = Sfc64::seed_from_u64(0x1234_5678_9abc_def0);
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
