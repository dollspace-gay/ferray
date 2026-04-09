// ferray-core: 256-bit two's-complement signed integer (REQ-1 extension)
//
// `I256` exists so that `u128 + i128` has a lossless common integer
// type. The union of `u128`'s and `i128`'s ranges needs 129 signed
// bits, and Rust has no native integer wider than 128 bits. ferray
// rounds up to 256 bits so the arithmetic is uniform and every
// operation stays inside the compile-time Element type universe
// (see issue #562, which was opened to track this in place of the
// lossy `result_type(U128, I*)` -> F64 path).
//
// The representation is little-endian `[u64; 4]` in two's complement:
// the top bit of `limbs[3]` is the sign bit. All binary operations
// use wrapping ripple-carry arithmetic, so Add/Sub/Mul/Neg wrap
// modulo 2^256 exactly like native integers.

use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, Mul, Neg, Sub};

/// 256-bit two's-complement signed integer.
///
/// Holds the full union of `u128` and `i128`, which is why ferray
/// uses it as the promoted type for mixed 128-bit integer operands.
///
/// The internal representation is four little-endian 64-bit limbs
/// in two's-complement form. `limbs[0]` is the least significant;
/// the top bit of `limbs[3]` is the sign bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct I256 {
    /// Four little-endian 64-bit limbs in two's-complement form.
    pub limbs: [u64; 4],
}

impl I256 {
    /// The additive identity (`0`).
    pub const ZERO: Self = Self { limbs: [0; 4] };

    /// The multiplicative identity (`1`).
    pub const ONE: Self = Self {
        limbs: [1, 0, 0, 0],
    };

    /// `-1`, represented as all-bits-set in two's complement.
    pub const NEG_ONE: Self = Self {
        limbs: [u64::MAX; 4],
    };

    /// The most negative representable value, `-2^255`.
    pub const MIN: Self = Self {
        limbs: [0, 0, 0, 1u64 << 63],
    };

    /// The most positive representable value, `2^255 - 1`.
    pub const MAX: Self = Self {
        limbs: [u64::MAX, u64::MAX, u64::MAX, (1u64 << 63) - 1],
    };

    /// Construct from explicit limbs (little-endian).
    #[inline]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self { limbs }
    }

    /// Return the raw little-endian limbs.
    #[inline]
    pub const fn to_limbs(self) -> [u64; 4] {
        self.limbs
    }

    /// Is this value negative? (inspects the top bit of the highest limb).
    #[inline]
    pub const fn is_negative(self) -> bool {
        (self.limbs[3] >> 63) & 1 == 1
    }

    /// Two's-complement negation: `-self`.
    #[inline]
    pub const fn wrapping_neg(self) -> Self {
        // `-x = !x + 1` in two's complement.
        let mut out = [0u64; 4];
        let mut i = 0;
        while i < 4 {
            out[i] = !self.limbs[i];
            i += 1;
        }
        // Add 1 with carry propagation.
        let (mut lo, mut carry) = out[0].overflowing_add(1);
        out[0] = lo;
        let mut j = 1;
        while j < 4 && carry {
            (lo, carry) = out[j].overflowing_add(1);
            out[j] = lo;
            j += 1;
        }
        Self { limbs: out }
    }

    /// Wrapping add modulo 2^256.
    #[inline]
    pub const fn wrapping_add(self, rhs: Self) -> Self {
        let mut out = [0u64; 4];
        let mut carry = false;
        let mut i = 0;
        while i < 4 {
            let (s1, c1) = self.limbs[i].overflowing_add(rhs.limbs[i]);
            let (s2, c2) = s1.overflowing_add(carry as u64);
            out[i] = s2;
            carry = c1 || c2;
            i += 1;
        }
        Self { limbs: out }
    }

    /// Wrapping subtract modulo 2^256.
    #[inline]
    pub const fn wrapping_sub(self, rhs: Self) -> Self {
        self.wrapping_add(rhs.wrapping_neg())
    }

    /// Wrapping multiply modulo 2^256.
    ///
    /// Implements schoolbook multiplication by widening each 64-bit
    /// partial product to 128 bits and propagating carries. Sign is
    /// handled automatically by two's complement wrapping — the
    /// low-order limbs of a signed multiply are the same as the
    /// corresponding unsigned multiply.
    pub const fn wrapping_mul(self, rhs: Self) -> Self {
        let mut out = [0u128; 4];
        let mut i = 0;
        while i < 4 {
            let mut carry: u128 = 0;
            let mut j = 0;
            while i + j < 4 {
                let prod = (self.limbs[i] as u128) * (rhs.limbs[j] as u128) + out[i + j] + carry;
                out[i + j] = prod & 0xFFFF_FFFF_FFFF_FFFF;
                carry = prod >> 64;
                j += 1;
            }
            i += 1;
        }
        Self {
            limbs: [out[0] as u64, out[1] as u64, out[2] as u64, out[3] as u64],
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

impl From<i128> for I256 {
    #[inline]
    fn from(v: i128) -> Self {
        let bits = v as u128;
        let lo = bits as u64;
        let mid = (bits >> 64) as u64;
        // Sign-extend the top two limbs.
        let sign_ext = if v < 0 { u64::MAX } else { 0 };
        Self {
            limbs: [lo, mid, sign_ext, sign_ext],
        }
    }
}

impl From<u128> for I256 {
    #[inline]
    fn from(v: u128) -> Self {
        Self {
            limbs: [v as u64, (v >> 64) as u64, 0, 0],
        }
    }
}

impl From<i64> for I256 {
    #[inline]
    fn from(v: i64) -> Self {
        Self::from(v as i128)
    }
}

impl From<u64> for I256 {
    #[inline]
    fn from(v: u64) -> Self {
        Self::from(v as u128)
    }
}

impl From<i32> for I256 {
    #[inline]
    fn from(v: i32) -> Self {
        Self::from(v as i128)
    }
}

impl From<u32> for I256 {
    #[inline]
    fn from(v: u32) -> Self {
        Self::from(v as u128)
    }
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

impl Ord for I256 {
    fn cmp(&self, other: &Self) -> Ordering {
        // Signed comparison: first compare sign bits, then compare
        // the remaining limbs from most significant down.
        let a_neg = self.is_negative();
        let b_neg = other.is_negative();
        match (a_neg, b_neg) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        // Same sign: top limb first, treated as unsigned for ordering
        // (within same sign the bit patterns are monotonic).
        for i in (0..4).rev() {
            match self.limbs[i].cmp(&other.limbs[i]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    }
}

impl PartialOrd for I256 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Operator traits (use wrapping semantics)
// ---------------------------------------------------------------------------

impl Add for I256 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.wrapping_add(rhs)
    }
}

impl Sub for I256 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.wrapping_sub(rhs)
    }
}

impl Mul for I256 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self.wrapping_mul(rhs)
    }
}

impl Neg for I256 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self.wrapping_neg()
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl fmt::Display for I256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // If the value fits in i128, use the native Display. Most
        // values ferray encounters in practice will, so this is the
        // common fast path.
        if let Some(as_i128) = self.try_to_i128() {
            return write!(f, "{as_i128}");
        }
        // Otherwise, emit a hex-prefixed 256-bit literal so the
        // output is at least unambiguous. (Full decimal printing for
        // a 256-bit integer needs a repeated divmod loop; we keep
        // the implementation minimal.)
        write!(
            f,
            "0x{:016x}{:016x}{:016x}{:016x}",
            self.limbs[3], self.limbs[2], self.limbs[1], self.limbs[0]
        )
    }
}

impl I256 {
    /// Try to narrow to `i128`. Returns `None` if the value is
    /// outside `[i128::MIN, i128::MAX]`.
    pub fn try_to_i128(self) -> Option<i128> {
        // For positive values, limbs[2] and limbs[3] must be zero.
        // For negative values, limbs[2] and limbs[3] must be
        // sign-extended (all ones) and the top bit of limbs[1]
        // must match the sign.
        if self.is_negative() {
            if self.limbs[2] == u64::MAX && self.limbs[3] == u64::MAX {
                // Reassemble the lower 128 bits as i128.
                let lo = self.limbs[0] as u128;
                let hi = self.limbs[1] as u128;
                let bits = lo | (hi << 64);
                let as_i128 = bits as i128;
                if as_i128 < 0 {
                    return Some(as_i128);
                }
            }
            None
        } else {
            if self.limbs[2] == 0 && self.limbs[3] == 0 {
                let lo = self.limbs[0] as u128;
                let hi = self.limbs[1] as u128;
                let bits = lo | (hi << 64);
                if bits <= i128::MAX as u128 {
                    return Some(bits as i128);
                }
            }
            None
        }
    }

    /// Try to narrow to `u128`. Returns `None` if negative or
    /// above `u128::MAX`.
    pub fn try_to_u128(self) -> Option<u128> {
        if self.is_negative() {
            return None;
        }
        if self.limbs[2] == 0 && self.limbs[3] == 0 {
            let lo = self.limbs[0] as u128;
            let hi = self.limbs[1] as u128;
            Some(lo | (hi << 64))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_and_one_constants() {
        assert_eq!(I256::ZERO.limbs, [0, 0, 0, 0]);
        assert_eq!(I256::ONE.limbs, [1, 0, 0, 0]);
        assert!(!I256::ZERO.is_negative());
        assert!(!I256::ONE.is_negative());
    }

    #[test]
    fn from_i128_positive_and_negative() {
        let p = I256::from(42i128);
        assert_eq!(p.limbs, [42, 0, 0, 0]);
        assert!(!p.is_negative());

        let n = I256::from(-1i128);
        assert_eq!(n.limbs, [u64::MAX; 4]);
        assert!(n.is_negative());
    }

    #[test]
    fn from_u128_never_negative() {
        let a = I256::from(u128::MAX);
        assert!(!a.is_negative());
        assert_eq!(a.limbs[0], u64::MAX);
        assert_eq!(a.limbs[1], u64::MAX);
        assert_eq!(a.limbs[2], 0);
        assert_eq!(a.limbs[3], 0);
    }

    #[test]
    fn add_within_i128_range() {
        let a = I256::from(1_000_000_000i128);
        let b = I256::from(2_500_000_000i128);
        let s = a + b;
        assert_eq!(s.try_to_i128(), Some(3_500_000_000i128));
    }

    #[test]
    fn add_crosses_i128_boundary() {
        // u128::MAX + 1 = 2^128, which doesn't fit in i128 but does
        // fit in i256.
        let a = I256::from(u128::MAX);
        let s = a + I256::ONE;
        assert_eq!(s.limbs, [0, 0, 1, 0]);
        assert!(!s.is_negative());
        assert_eq!(s.try_to_i128(), None);
    }

    #[test]
    fn sub_negative_result() {
        let a = I256::from(10i128);
        let b = I256::from(25i128);
        let d = a - b;
        assert_eq!(d.try_to_i128(), Some(-15i128));
    }

    #[test]
    fn neg_round_trip() {
        let a = I256::from(123i128);
        assert_eq!((-a).try_to_i128(), Some(-123i128));
        assert_eq!(-(-a), a);
    }

    #[test]
    fn mul_small() {
        let a = I256::from(7i128);
        let b = I256::from(8i128);
        assert_eq!((a * b).try_to_i128(), Some(56i128));
    }

    #[test]
    fn mul_large_crosses_i128_boundary() {
        // i128::MAX * 2 doesn't fit in i128 but does fit in i256.
        let a = I256::from(i128::MAX);
        let two = I256::from(2i128);
        let product = a * two;
        // Manually compute expected: 2 * (2^127 - 1) = 2^128 - 2.
        let expected = I256::from(u128::MAX - 1);
        assert_eq!(product, expected);
    }

    #[test]
    fn cmp_negative_and_positive() {
        let neg = I256::from(-5i128);
        let pos = I256::from(5i128);
        assert!(neg < pos);
        assert!(pos > neg);
        assert_eq!(neg.cmp(&neg), Ordering::Equal);
    }

    #[test]
    fn cmp_large_values() {
        let a = I256::from(u128::MAX);
        let b = a + I256::ONE; // 2^128
        assert!(b > a);
        assert!(a < b);
    }

    #[test]
    fn try_to_u128_rejects_negative() {
        let n = I256::from(-1i128);
        assert_eq!(n.try_to_u128(), None);
    }

    #[test]
    fn try_to_u128_exact_max() {
        let m = I256::from(u128::MAX);
        assert_eq!(m.try_to_u128(), Some(u128::MAX));
    }

    #[test]
    fn display_small_values_uses_decimal() {
        assert_eq!(format!("{}", I256::from(42i128)), "42");
        assert_eq!(format!("{}", I256::from(-42i128)), "-42");
    }

    #[test]
    fn display_huge_value_uses_hex_fallback() {
        let big = I256::from(u128::MAX) + I256::ONE;
        let s = format!("{big}");
        assert!(s.starts_with("0x"));
    }

    #[test]
    fn u128_plus_i128_is_lossless() {
        // The motivating use case: u128::MAX + i128::MIN.
        // = 2^128 - 1 + (-(2^127))
        // = 2^127 - 1
        let u = I256::from(u128::MAX);
        let i = I256::from(i128::MIN);
        let sum = u + i;
        assert_eq!(sum.try_to_i128(), Some(i128::MAX));
    }

    #[test]
    fn i256_is_a_valid_ferray_element_type() {
        // Verify the Element trait impl actually compiles and that
        // `Array<I256, _>` construction works end-to-end.
        use crate::Array;
        use crate::dimension::Ix1;
        use crate::dtype::Element;

        assert_eq!(<I256 as Element>::dtype(), crate::dtype::DType::I256);
        assert_eq!(<I256 as Element>::zero(), I256::ZERO);
        assert_eq!(<I256 as Element>::one(), I256::ONE);

        let arr = Array::<I256, Ix1>::from_vec(
            Ix1::new([3]),
            vec![I256::from(u128::MAX), I256::from(i128::MIN), I256::ZERO],
        )
        .unwrap();
        assert_eq!(arr.shape(), &[3]);
    }

    #[test]
    fn promotion_returns_i256_for_u128_plus_i128() {
        // End-to-end: the public result_type API returns I256 for
        // the motivating mixed pair.
        use crate::dtype::DType;
        use crate::dtype::promotion::result_type;
        assert_eq!(result_type(DType::U128, DType::I128).unwrap(), DType::I256);
    }
}
