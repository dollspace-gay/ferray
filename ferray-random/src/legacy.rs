// ferray-random: NumPy legacy `RandomState` (MT19937 + polar Box-Muller)
//
// Bit-identical reimplementation of `numpy.random.RandomState` for the
// `standard_normal` path. This is the *legacy* generator — the 32-bit
// Mersenne Twister (MT19937) seeded via `init_genrand` (numpy's legacy
// single-integer seeding), the `rk_double` 53-bit uniform, and the cached
// polar Box-Muller gaussian (`rk_gauss`).
//
// It is intentionally distinct from the modern [`crate::Generator`] /
// [`crate::MT19937`] (which is the 64-bit MT19937-64 + ziggurat normal).
// scikit-learn's `check_random_state` produces a legacy `RandomState`,
// and `randomized_svd` draws its random projection via
// `random_state.standard_normal(size=...)`. Reproducing that draw
// bit-for-bit requires this exact legacy path, not the modern one.
//
// Algorithm references (numpy `randomkit.c` / `mtrand`, stable since the
// 1.x series):
//   - seeding:   `init_genrand`  (reference MT19937 single-int seeding)
//   - uniform:   `rk_double`   (two uint32 -> 53-bit double)
//   - gaussian:  `rk_gauss`    (polar Box-Muller with one-value cache)

#![allow(clippy::unreadable_literal)]

use ferray_core::{Array, FerrayError, Ix2};

/// Number of state words in the 32-bit Mersenne Twister.
const N: usize = 624;
/// Recurrence offset.
const M: usize = 397;
/// Most-significant-word mask (bit 31).
const MATRIX_A: u32 = 0x9908b0df;
/// Most significant w-r bits.
const UPPER_MASK: u32 = 0x80000000;
/// Least significant r bits.
const LOWER_MASK: u32 = 0x7fffffff;

/// NumPy legacy `RandomState`: a 32-bit MT19937 plus the cached polar
/// Box-Muller gaussian state, matching `numpy.random.RandomState`.
///
/// This reproduces `np.random.RandomState(seed).standard_normal(...)`
/// bit-for-bit. The gaussian cache (`has_gauss` / `gauss`) persists across
/// calls on the same instance, exactly as numpy does — so the polar method
/// draws two normals at a time and hands the second one out on the next
/// call.
///
/// # Example
/// ```
/// use ferray_random::RandomState;
///
/// let mut rs = RandomState::new(42);
/// let z = rs.standard_normal(3);
/// // Bit-identical to np.random.RandomState(42).standard_normal(3).
/// assert_eq!(z[0], 0.4967141530112327);
/// ```
#[derive(Clone)]
pub struct RandomState {
    /// MT19937 state words (32-bit).
    mt: [u32; N],
    /// Read cursor into `mt`; `N` means "exhausted, regenerate".
    mti: usize,
    /// Whether `gauss` holds a cached normal from a prior polar draw.
    has_gauss: bool,
    /// The cached normal value (valid only when `has_gauss`).
    gauss: f64,
}

impl RandomState {
    /// Construct a legacy `RandomState` from an integer seed, matching
    /// `np.random.RandomState(seed)`.
    ///
    /// numpy's legacy `RandomState` seeds a 32-bit MT19937 directly via
    /// `init_genrand(seed)` (the reference single-integer MT19937 seeding),
    /// for any integer seed in `[0, 2^32 - 1]`. numpy itself **rejects**
    /// seeds outside that range (`ValueError: Seed must be between 0 and
    /// 2**32 - 1`). To stay infallible and bit-identical on the supported
    /// range while never panicking, this constructor reduces the seed
    /// modulo `2^32` before seeding; pass a seed in `[0, 2^32 - 1]` to match
    /// numpy exactly. Use [`RandomState::try_new`] when you need numpy's
    /// out-of-range rejection.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        let mut state = Self {
            mt: [0u32; N],
            mti: N + 1,
            has_gauss: false,
            gauss: 0.0,
        };
        state.init_genrand((seed & 0xffff_ffff) as u32);
        state
    }

    /// Construct a legacy `RandomState`, rejecting out-of-range seeds
    /// exactly as numpy does.
    ///
    /// # Errors
    /// Returns [`FerrayError`] if `seed > 2^32 - 1`, matching numpy's
    /// `ValueError: Seed must be between 0 and 2**32 - 1`.
    pub fn try_new(seed: u64) -> Result<Self, FerrayError> {
        if seed > u64::from(u32::MAX) {
            return Err(FerrayError::invalid_value(
                "Seed must be between 0 and 2**32 - 1",
            ));
        }
        Ok(Self::new(seed))
    }

    /// `init_genrand(s)` — the reference MT19937 single-integer seeding
    /// that numpy's legacy `RandomState` applies directly to an integer
    /// seed.
    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            // mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.mti = N;
    }

    /// Draw the next 32-bit word from the MT19937 stream (numpy
    /// `rk_random`). Regenerates the full state block when exhausted.
    fn next_u32(&mut self) -> u32 {
        if self.mti >= N {
            // Generate N words at once.
            for kk in 0..N - M {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            }
            for kk in N - M..N - 1 {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] =
                    self.mt[kk + M - N] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ if y & 1 == 0 { 0 } else { MATRIX_A };
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        // Tempering.
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// `rk_double` — a uniform double in [0, 1) with full 53-bit mantissa,
    /// built from two 32-bit draws. The high word (`a`) is drawn FIRST.
    fn next_double(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64; // top 27 bits
        let b = (self.next_u32() >> 6) as f64; // top 26 bits
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `rk_gauss` — one standard normal via the cached polar Box-Muller
    /// method. Draws two normals per pair of accepted uniforms, caches the
    /// second, and hands it out on the next call.
    fn next_gauss(&mut self) -> f64 {
        if self.has_gauss {
            self.has_gauss = false;
            return self.gauss;
        }
        let (f, x1, x2) = loop {
            let x1 = 2.0 * self.next_double() - 1.0;
            let x2 = 2.0 * self.next_double() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                break ((-2.0 * r2.ln() / r2).sqrt(), x1, x2);
            }
        };
        // Cache f*x1, return f*x2 (numpy order).
        self.gauss = f * x1;
        self.has_gauss = true;
        f * x2
    }

    /// Draw `n` standard-normal samples, bit-identical to
    /// `np.random.RandomState(seed).standard_normal(n)`.
    ///
    /// The gaussian cache persists across calls, so two successive
    /// `standard_normal(1)` calls equal a single `standard_normal(2)`.
    #[must_use]
    pub fn standard_normal(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.next_gauss());
        }
        out
    }

    /// Draw an `m × n` standard-normal array filled in C (row-major)
    /// order, bit-identical to
    /// `np.random.RandomState(seed).standard_normal(size=(m, n))`.
    ///
    /// # Errors
    /// Returns [`FerrayError`] if the `(m, n)` shape and the generated
    /// element count cannot form a valid array (only possible on internal
    /// inconsistency; the element count always equals `m * n`).
    pub fn standard_normal_2d(
        &mut self,
        shape: (usize, usize),
    ) -> Result<Array<f64, Ix2>, FerrayError> {
        let (m, n) = shape;
        let total = m.checked_mul(n).ok_or_else(|| {
            FerrayError::invalid_value(format!("shape ({m}, {n}) overflows usize"))
        })?;
        let data = self.standard_normal(total);
        Array::<f64, Ix2>::from_vec(Ix2::new([m, n]), data)
    }

    /// Draw `n` uniform samples in [0, 1), bit-identical to
    /// `np.random.RandomState(seed).random_sample(n)` (numpy's
    /// `rk_double`).
    #[must_use]
    pub fn random_sample(&mut self, n: usize) -> Vec<f64> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.next_double());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Oracle values captured from a live numpy 2.4.5 interpreter as exact
    // f64 hex floats (R-CHAR-3): these are NOT literal-copied from ferray.
    // Reproduce with, e.g.:
    //   python3 -c "import numpy as np; \
    //     print([x.hex() for x in np.random.RandomState(42).standard_normal(8)])"

    fn h(s: &str) -> f64 {
        // Parse a C99 hex-float (Rust's std cannot parse hex floats directly).
        // We round-trip through the bit pattern numpy produced by storing the
        // decimal forms here; instead we keep hex strings and decode them.
        parse_hex_f64(s)
    }

    /// Minimal C99 hex-float (`0x1.aaap+3`) parser for test oracles.
    fn parse_hex_f64(s: &str) -> f64 {
        let neg = s.starts_with('-');
        let s = s.trim_start_matches('-');
        let s = s.strip_prefix("0x").expect("hex float must start 0x");
        let (mantissa, exp) = {
            let (m, e) = s.split_once('p').expect("hex float must have p exponent");
            (m, e.parse::<i32>().expect("valid exponent"))
        };
        let (int_part, frac_part) = match mantissa.split_once('.') {
            Some((i, f)) => (i, f),
            None => (mantissa, ""),
        };
        let mut value = u64::from_str_radix(int_part, 16).expect("valid int part") as f64;
        let mut scale = 1.0_f64 / 16.0;
        for c in frac_part.chars() {
            let d = c.to_digit(16).expect("valid hex digit") as f64;
            value += d * scale;
            scale /= 16.0;
        }
        value *= 2.0_f64.powi(exp);
        if neg { -value } else { value }
    }

    fn assert_bit_identical(got: &[f64], expected_hex: &[&str]) {
        assert_eq!(got.len(), expected_hex.len(), "length mismatch");
        for (i, (g, e)) in got.iter().zip(expected_hex.iter()).enumerate() {
            let want = h(e);
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "divergence at index {i}: got {g:?} ({:#018x}), want {want:?} ({:#018x})",
                g.to_bits(),
                want.to_bits()
            );
        }
    }

    #[test]
    fn standard_normal_seed0_n8() {
        let got = RandomState::new(0).standard_normal(8);
        assert_bit_identical(
            &got,
            &[
                "0x1.c398ef3e5cfa5p+0",
                "0x1.99c2cfacc8953p-2",
                "0x1.f51d25222c9abp-1",
                "0x1.1ed5969e3314ep+1",
                "0x1.de1847cb13ddap+0",
                "-0x1.f45dc42a58c82p-1",
                "0x1.e671fd33295e9p-1",
                "-0x1.35fac49d38436p-3",
            ],
        );
    }

    #[test]
    fn standard_normal_seed42_n8() {
        let got = RandomState::new(42).standard_normal(8);
        assert_bit_identical(
            &got,
            &[
                "0x1.fca2a28a9307cp-2",
                "-0x1.1b2a505de052ap-3",
                "0x1.4b9dd50245e68p-1",
                "0x1.85e548e01aa2bp+0",
                "-0x1.df8bcdf57a640p-3",
                "-0x1.df8332670db3bp-3",
                "0x1.94474a8407408p+0",
                "0x1.88ed346f0d7f3p-1",
            ],
        );
    }

    #[test]
    fn standard_normal_seed1_n8() {
        let got = RandomState::new(1).standard_normal(8);
        assert_bit_identical(
            &got,
            &[
                "0x1.9fd5190657c50p+0",
                "-0x1.393822fb7d9adp-1",
                "-0x1.0e6c872548fd2p-1",
                "-0x1.12ae1255cb80cp+0",
                "0x1.bb16b5735117dp-1",
                "-0x1.2698d1ecca259p+1",
                "0x1.beabfbd8fc605p+0",
                "-0x1.85bce931aaff6p-1",
            ],
        );
    }

    #[test]
    fn standard_normal_seed12345_n8() {
        let got = RandomState::new(12345).standard_normal(8);
        assert_bit_identical(
            &got,
            &[
                "-0x1.a33dc4f5d203dp-3",
                "0x1.ea701f566080fp-2",
                "-0x1.09f3df0ae3d0ep-1",
                "-0x1.1c88aeb5231f1p-1",
                "0x1.f73d654602e58p+0",
                "0x1.64b63ea2acb4bp+0",
                "0x1.7c8cf8427b26ep-4",
                "0x1.20821040b7d8bp-2",
            ],
        );
    }

    /// Odd count: exercises the polar method's pairwise generation plus the
    /// leftover cache (the 3rd value comes from a fresh pair; the unused 4th
    /// stays cached). This is the most important parity case.
    #[test]
    fn standard_normal_seed42_n3_odd() {
        let got = RandomState::new(42).standard_normal(3);
        assert_bit_identical(
            &got,
            &[
                "0x1.fca2a28a9307cp-2",
                "-0x1.1b2a505de052ap-3",
                "0x1.4b9dd50245e68p-1",
            ],
        );
    }

    #[test]
    fn standard_normal_seed7_n5_odd() {
        let got = RandomState::new(7).standard_normal(5);
        assert_bit_identical(
            &got,
            &[
                "0x1.b0c64ae2deb29p+0",
                "-0x1.dd1eafa1d414dp-2",
                "0x1.0cdcdf34c3aedp-5",
                "0x1.a14bf2d03adc9p-2",
                "-0x1.93edb81e0483fp-1",
            ],
        );
    }

    /// Cross-call gaussian cache: two `standard_normal(1)` calls on the same
    /// instance must equal a single `standard_normal(2)`. This is numpy's
    /// `has_gauss` state persisting across calls.
    #[test]
    fn gauss_cache_persists_across_calls() {
        let mut rs = RandomState::new(99);
        let a = rs.standard_normal(1);
        let b = rs.standard_normal(1);
        let combined = [a[0], b[0]];
        let single = RandomState::new(99).standard_normal(2);
        assert_eq!(combined[0].to_bits(), single[0].to_bits());
        assert_eq!(combined[1].to_bits(), single[1].to_bits());
    }

    /// 2D fill is C-order (row-major), matching
    /// `RandomState(seed).standard_normal((3, 4))`.
    #[test]
    fn standard_normal_2d_c_order_seed42_3x4() {
        let arr = RandomState::new(42)
            .standard_normal_2d((3, 4))
            .expect("3x4 array");
        assert_eq!(arr.shape(), &[3, 4]);
        let flat: Vec<f64> = arr.iter().copied().collect();
        assert_bit_identical(
            &flat,
            &[
                "0x1.fca2a28a9307cp-2",
                "-0x1.1b2a505de052ap-3",
                "0x1.4b9dd50245e68p-1",
                "0x1.85e548e01aa2bp+0",
                "-0x1.df8bcdf57a640p-3",
                "-0x1.df8332670db3bp-3",
                "0x1.94474a8407408p+0",
                "0x1.88ed346f0d7f3p-1",
                "-0x1.e0bde4b799e8bp-2",
                "0x1.15ca6e16a2d7dp-1",
                "-0x1.da8a2aec11b70p-2",
                "-0x1.dce842b16efe7p-2",
            ],
        );
    }

    /// The 2D fill must equal the flat 1D draw of the same total count from
    /// the same seed (size=(3,4) == 12 sequential gaussians, C-order).
    #[test]
    fn standard_normal_2d_equals_flat() {
        let flat = RandomState::new(42).standard_normal(12);
        let arr = RandomState::new(42)
            .standard_normal_2d((3, 4))
            .expect("array");
        let arr_flat: Vec<f64> = arr.iter().copied().collect();
        for (a, b) in flat.iter().zip(arr_flat.iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn try_new_rejects_out_of_range_seed() {
        // numpy: ValueError: Seed must be between 0 and 2**32 - 1
        assert!(RandomState::try_new(u64::from(u32::MAX) + 1).is_err());
        assert!(RandomState::try_new(u64::from(u32::MAX)).is_ok());
        assert!(RandomState::try_new(0).is_ok());
    }

    #[test]
    fn clone_produces_independent_streams() {
        let mut a = RandomState::new(2026);
        let _ = a.standard_normal(3);
        let mut b = a.clone();
        assert_eq!(a.standard_normal(5), b.standard_normal(5));
    }
}
