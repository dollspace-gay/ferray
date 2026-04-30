// ferray-random: SeedSequence — entropy mixing and child-spawn API
//
// Implements the NumPy `SeedSequence` interface: mix arbitrary entropy
// (a primary u128 seed plus a spawn-key path) into a deterministic,
// well-distributed sequence of 32-bit words suitable for seeding any
// BitGenerator. Spawning produces children whose entropy paths differ
// from the parent's, so independent streams from a single root seed are
// reproducible.

#![allow(clippy::unreadable_literal)]

use super::BitGenerator;
use super::splitmix64;

/// SeedSequence — deterministic entropy expansion and child spawning.
///
/// Equivalent to `numpy.random.SeedSequence`. Construct from an
/// `entropy: u64`; call [`generate_state`](Self::generate_state) for
/// `n_words` 32-bit seed words, or [`spawn`](Self::spawn) to create
/// `n` independent child sequences whose state is reproducibly derived
/// from this one.
#[derive(Debug, Clone)]
pub struct SeedSequence {
    /// User-supplied root entropy.
    entropy: u64,
    /// Path of indices identifying this sequence within the parent's
    /// spawn tree. Empty for the root.
    spawn_key: Vec<u64>,
    /// How many children have been spawned from this sequence so far.
    /// Used to give the next spawn a fresh key.
    n_children_spawned: u64,
}

impl SeedSequence {
    /// Create a SeedSequence from a single entropy value.
    #[must_use]
    pub fn new(entropy: u64) -> Self {
        Self {
            entropy,
            spawn_key: Vec::new(),
            n_children_spawned: 0,
        }
    }

    /// Create a SeedSequence with an explicit spawn-key path.
    ///
    /// The spawn key disambiguates children of the same parent.
    /// Normally users construct via [`new`](Self::new) and let
    /// [`spawn`](Self::spawn) populate the key automatically.
    #[must_use]
    pub fn with_spawn_key(entropy: u64, spawn_key: Vec<u64>) -> Self {
        Self {
            entropy,
            spawn_key,
            n_children_spawned: 0,
        }
    }

    /// The entropy this sequence was created with.
    #[must_use]
    pub fn entropy(&self) -> u64 {
        self.entropy
    }

    /// The spawn-key path of this sequence (empty for the root).
    #[must_use]
    pub fn spawn_key(&self) -> &[u64] {
        &self.spawn_key
    }

    /// Generate `n_words` 32-bit seed words from this sequence.
    ///
    /// The output is a deterministic function of `(entropy, spawn_key)`
    /// only — calling `generate_state` repeatedly returns the same
    /// words. Different sequences produce statistically independent
    /// outputs.
    #[must_use]
    pub fn generate_state(&self, n_words: usize) -> Vec<u32> {
        // Mix entropy + spawn_key into a SplitMix64 state, then emit
        // `n_words` 32-bit halves.
        let mut state = self.entropy;
        for &k in &self.spawn_key {
            state ^= k.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            // Round-trip through SplitMix64 to avalanche bits.
            let _ = splitmix64(&mut state);
        }
        let mut out = Vec::with_capacity(n_words);
        let mut produced = 0usize;
        while produced < n_words {
            let v = splitmix64(&mut state);
            out.push((v & 0xFFFF_FFFF) as u32);
            produced += 1;
            if produced < n_words {
                out.push((v >> 32) as u32);
                produced += 1;
            }
        }
        out.truncate(n_words);
        out
    }

    /// Generate a single u64 derived from this sequence — convenient for
    /// seeding a [`BitGenerator`] via [`BitGenerator::seed_from_u64`].
    #[must_use]
    pub fn generate_u64(&self) -> u64 {
        let words = self.generate_state(2);
        (u64::from(words[1]) << 32) | u64::from(words[0])
    }

    /// Seed a fresh BitGenerator deterministically from this sequence.
    pub fn seed<B: BitGenerator>(&self) -> B {
        B::seed_from_u64(self.generate_u64())
    }

    /// Create `n` independent child sequences.
    ///
    /// Each child has the same `entropy` as the parent but a different
    /// `spawn_key`, so their `generate_state` outputs do not overlap.
    /// Mutates `self`'s spawn counter so successive calls yield fresh
    /// children.
    #[must_use]
    pub fn spawn(&mut self, n: usize) -> Vec<SeedSequence> {
        let mut children = Vec::with_capacity(n);
        for _ in 0..n {
            let mut key = self.spawn_key.clone();
            key.push(self.n_children_spawned);
            self.n_children_spawned += 1;
            children.push(SeedSequence {
                entropy: self.entropy,
                spawn_key: key,
                n_children_spawned: 0,
            });
        }
        children
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_state_deterministic() {
        let s1 = SeedSequence::new(42);
        let s2 = SeedSequence::new(42);
        assert_eq!(s1.generate_state(8), s2.generate_state(8));
    }

    #[test]
    fn different_entropy_differs() {
        let s1 = SeedSequence::new(42).generate_state(8);
        let s2 = SeedSequence::new(43).generate_state(8);
        assert_ne!(s1, s2);
    }

    #[test]
    fn spawn_produces_independent_children() {
        let mut root = SeedSequence::new(1234);
        let children = root.spawn(4);
        assert_eq!(children.len(), 4);
        let mut all_distinct = true;
        for i in 0..children.len() {
            for j in (i + 1)..children.len() {
                if children[i].generate_state(8) == children[j].generate_state(8) {
                    all_distinct = false;
                }
            }
        }
        assert!(all_distinct);
        // Children also differ from root.
        for c in &children {
            assert_ne!(c.generate_state(8), root.generate_state(8));
        }
    }

    #[test]
    fn spawn_advances_counter() {
        let mut a = SeedSequence::new(1);
        let mut b = SeedSequence::new(1);
        let _first = a.spawn(2); // a.n_children_spawned = 2
        let second = a.spawn(1)[0].clone(); // spawn_key uses index 2
        let zeroth = b.spawn(1)[0].clone(); // spawn_key uses index 0
        assert_ne!(second.generate_state(4), zeroth.generate_state(4));
    }

    #[test]
    fn seed_bitgenerator_roundtrips() {
        use crate::bitgen::Pcg64;
        let s = SeedSequence::new(7);
        let mut a: Pcg64 = s.seed();
        let mut b: Pcg64 = s.seed();
        for _ in 0..32 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
