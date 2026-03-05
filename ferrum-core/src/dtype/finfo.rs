// ferrum-core: finfo<T> and iinfo<T> type introspection (REQ-34)
//
// Mirrors numpy.finfo and numpy.iinfo for compile-time type metadata.

use std::fmt;

// ---------------------------------------------------------------------------
// FloatInfo — returned by finfo::<T>()
// ---------------------------------------------------------------------------

/// Compile-time metadata about a floating-point type.
///
/// Mirrors `numpy.finfo`. All fields match their NumPy equivalents.
#[derive(Clone, Copy, PartialEq)]
pub struct FloatInfo {
    /// Machine epsilon: smallest representable positive number such that
    /// `1.0 + eps != 1.0`.
    pub eps: f64,
    /// The smallest positive normal number.
    pub smallest_normal: f64,
    /// The smallest positive subnormal number.
    pub smallest_subnormal: f64,
    /// The largest representable finite number.
    pub max: f64,
    /// The most negative finite number.
    pub min: f64,
    /// Number of bits in the type.
    pub bits: u32,
    /// Number of mantissa bits (significand precision - 1).
    pub nmant: u32,
    /// Number of exponent bits.
    pub nexp: u32,
    /// Maximum exponent (base-2).
    pub maxexp: i32,
    /// Minimum exponent (base-2, normal range).
    pub minexp: i32,
}

impl fmt::Debug for FloatInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FloatInfo")
            .field("eps", &self.eps)
            .field("smallest_normal", &self.smallest_normal)
            .field("smallest_subnormal", &self.smallest_subnormal)
            .field("max", &self.max)
            .field("min", &self.min)
            .field("bits", &self.bits)
            .field("nmant", &self.nmant)
            .field("nexp", &self.nexp)
            .field("maxexp", &self.maxexp)
            .field("minexp", &self.minexp)
            .finish()
    }
}

impl fmt::Display for FloatInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FloatInfo(bits={}, eps={:e}, min={:e}, max={:e})",
            self.bits, self.eps, self.min, self.max
        )
    }
}

// ---------------------------------------------------------------------------
// IntInfo — returned by iinfo::<T>()
// ---------------------------------------------------------------------------

/// Compile-time metadata about an integer type.
///
/// Mirrors `numpy.iinfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntInfo {
    /// The smallest representable value.
    pub min: i128,
    /// The largest representable value.
    pub max: i128,
    /// Number of bits in the type.
    pub bits: u32,
}

impl fmt::Display for IntInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "IntInfo(bits={}, min={}, max={})",
            self.bits, self.min, self.max
        )
    }
}

// ---------------------------------------------------------------------------
// FloatType trait — sealed, for types that support finfo
// ---------------------------------------------------------------------------

/// Marker trait for floating-point types that support [`finfo`].
///
/// This trait is sealed and implemented for `f32` and `f64`.
pub trait FloatType: sealed::SealedFloat {
    /// Return the [`FloatInfo`] for this type.
    fn float_info() -> FloatInfo;
}

/// Marker trait for integer types that support [`iinfo`].
///
/// This trait is sealed and implemented for all integer types.
pub trait IntType: sealed::SealedInt {
    /// Return the [`IntInfo`] for this type.
    fn int_info() -> IntInfo;
}

mod sealed {
    pub trait SealedFloat {}
    pub trait SealedInt {}
}

// ---------------------------------------------------------------------------
// FloatType implementations
// ---------------------------------------------------------------------------

impl sealed::SealedFloat for f32 {}
impl FloatType for f32 {
    fn float_info() -> FloatInfo {
        FloatInfo {
            eps: f32::EPSILON as f64,
            smallest_normal: f32::MIN_POSITIVE as f64,
            smallest_subnormal: {
                // f32 smallest subnormal = 2^(-149)
                let mut v: f32 = f32::MIN_POSITIVE;
                while v / 2.0 > 0.0 {
                    v /= 2.0;
                }
                v as f64
            },
            max: f32::MAX as f64,
            min: f32::MIN as f64,
            bits: 32,
            nmant: f32::MANTISSA_DIGITS - 1, // 23
            nexp: 8,
            maxexp: f32::MAX_EXP,   // 128
            minexp: f32::MIN_EXP,   // -125
        }
    }
}

impl sealed::SealedFloat for f64 {}
impl FloatType for f64 {
    fn float_info() -> FloatInfo {
        FloatInfo {
            eps: f64::EPSILON,
            smallest_normal: f64::MIN_POSITIVE,
            smallest_subnormal: {
                // f64 smallest subnormal = 2^(-1074)
                let mut v: f64 = f64::MIN_POSITIVE;
                while v / 2.0 > 0.0 {
                    v /= 2.0;
                }
                v
            },
            max: f64::MAX,
            min: f64::MIN,
            bits: 64,
            nmant: f64::MANTISSA_DIGITS - 1, // 52
            nexp: 11,
            maxexp: f64::MAX_EXP,   // 1024
            minexp: f64::MIN_EXP,   // -1021
        }
    }
}

// ---------------------------------------------------------------------------
// IntType implementations
// ---------------------------------------------------------------------------

macro_rules! impl_int_type {
    ($ty:ty, $bits:expr) => {
        impl sealed::SealedInt for $ty {}
        impl IntType for $ty {
            fn int_info() -> IntInfo {
                IntInfo {
                    min: <$ty>::MIN as i128,
                    max: <$ty>::MAX as i128,
                    bits: $bits,
                }
            }
        }
    };
}

impl_int_type!(i8, 8);
impl_int_type!(i16, 16);
impl_int_type!(i32, 32);
impl_int_type!(i64, 64);
impl_int_type!(i128, 128);
impl_int_type!(u8, 8);
impl_int_type!(u16, 16);
impl_int_type!(u32, 32);
impl_int_type!(u64, 64);
impl_int_type!(u128, 128);

// bool treated as 1-bit unsigned
impl sealed::SealedInt for bool {}
impl IntType for bool {
    fn int_info() -> IntInfo {
        IntInfo {
            min: 0,
            max: 1,
            bits: 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API functions
// ---------------------------------------------------------------------------

/// Return floating-point type metadata for `T`.
///
/// Equivalent to `numpy.finfo(dtype)`.
///
/// # Examples
/// ```
/// use ferrum_core::dtype::finfo::finfo;
/// let info = finfo::<f64>();
/// assert_eq!(info.eps, f64::EPSILON);
/// assert_eq!(info.bits, 64);
/// ```
pub fn finfo<T: FloatType>() -> FloatInfo {
    T::float_info()
}

/// Return integer type metadata for `T`.
///
/// Equivalent to `numpy.iinfo(dtype)`.
///
/// # Examples
/// ```
/// use ferrum_core::dtype::finfo::iinfo;
/// let info = iinfo::<i32>();
/// assert_eq!(info.min, i32::MIN as i128);
/// assert_eq!(info.max, i32::MAX as i128);
/// ```
pub fn iinfo<T: IntType>() -> IntInfo {
    T::int_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finfo_f64() {
        let info = finfo::<f64>();
        assert_eq!(info.eps, f64::EPSILON);
        assert_eq!(info.max, f64::MAX);
        assert_eq!(info.min, f64::MIN);
        assert_eq!(info.smallest_normal, f64::MIN_POSITIVE);
        assert_eq!(info.bits, 64);
        assert_eq!(info.nmant, 52);
        assert_eq!(info.nexp, 11);
        assert_eq!(info.maxexp, 1024);
        assert_eq!(info.minexp, -1021);
        assert!(info.smallest_subnormal > 0.0);
        assert!(info.smallest_subnormal < info.smallest_normal);
    }

    #[test]
    fn finfo_f32() {
        let info = finfo::<f32>();
        assert_eq!(info.eps, f32::EPSILON as f64);
        assert_eq!(info.max, f32::MAX as f64);
        assert_eq!(info.min, f32::MIN as f64);
        assert_eq!(info.bits, 32);
        assert_eq!(info.nmant, 23);
        assert_eq!(info.nexp, 8);
        assert_eq!(info.maxexp, 128);
        assert_eq!(info.minexp, -125);
    }

    #[test]
    fn iinfo_i32() {
        let info = iinfo::<i32>();
        assert_eq!(info.min, i32::MIN as i128);
        assert_eq!(info.max, i32::MAX as i128);
        assert_eq!(info.bits, 32);
    }

    #[test]
    fn iinfo_u8() {
        let info = iinfo::<u8>();
        assert_eq!(info.min, 0);
        assert_eq!(info.max, 255);
        assert_eq!(info.bits, 8);
    }

    #[test]
    fn iinfo_i64() {
        let info = iinfo::<i64>();
        assert_eq!(info.min, i64::MIN as i128);
        assert_eq!(info.max, i64::MAX as i128);
        assert_eq!(info.bits, 64);
    }

    #[test]
    fn iinfo_bool() {
        let info = iinfo::<bool>();
        assert_eq!(info.min, 0);
        assert_eq!(info.max, 1);
        assert_eq!(info.bits, 1);
    }

    #[test]
    fn iinfo_u128() {
        let info = iinfo::<u128>();
        assert_eq!(info.min, 0);
        assert_eq!(info.max, u128::MAX as i128);
        assert_eq!(info.bits, 128);
    }

    #[test]
    fn display_float_info() {
        let info = finfo::<f64>();
        let s = info.to_string();
        assert!(s.contains("FloatInfo"));
        assert!(s.contains("bits=64"));
    }

    #[test]
    fn display_int_info() {
        let info = iinfo::<i32>();
        let s = info.to_string();
        assert!(s.contains("IntInfo"));
        assert!(s.contains("bits=32"));
    }
}
