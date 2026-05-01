// ferray-core: Type promotion rules (REQ-23)
//
// Implements NumPy's type promotion rules: the smallest type that can represent
// both inputs without precision loss.
//
// - `promoted_type!()` is a compile-time proc macro (in ferray-core-macros)
// - `result_type()` is the runtime equivalent using DType

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::format;

use num_complex::Complex;

use crate::dtype::{DType, Element};
use crate::error::{FerrayError, FerrayResult};

// ---------------------------------------------------------------------------
// Promoted trait — compile-time type promotion via trait resolution
// ---------------------------------------------------------------------------

/// Trait that resolves the promoted type of two `Element` types at compile time.
///
/// This is the trait-based counterpart to `promoted_type!()`. The macro is
/// generally more convenient, but this trait can be useful in generic contexts.
pub trait Promoted<Rhs: Element>: Element {
    /// The promoted type that can represent both `Self` and `Rhs`.
    type Output: Element;
}

// ---------------------------------------------------------------------------
// PromoteTo — explicit conversion for promoted operations
// ---------------------------------------------------------------------------

/// Trait for converting a value to a promoted type.
///
/// This is used by `add_promoted()`, `mul_promoted()`, etc. to explicitly
/// convert operands before performing mixed-type operations.
pub trait PromoteTo<Target: Element>: Element {
    /// Convert this value to the target type.
    fn promote(self) -> Target;
}

// Implement PromoteTo for same-type (identity)
macro_rules! impl_promote_identity {
    ($ty:ty) => {
        impl PromoteTo<$ty> for $ty {
            #[inline]
            fn promote(self) -> $ty {
                self
            }
        }
    };
}

impl_promote_identity!(bool);
impl_promote_identity!(u8);
impl_promote_identity!(u16);
impl_promote_identity!(u32);
impl_promote_identity!(u64);
impl_promote_identity!(u128);
impl_promote_identity!(i8);
impl_promote_identity!(i16);
impl_promote_identity!(i32);
impl_promote_identity!(i64);
impl_promote_identity!(i128);
impl_promote_identity!(f32);
impl_promote_identity!(f64);
impl_promote_identity!(Complex<f32>);
impl_promote_identity!(Complex<f64>);

// Implement PromoteTo for numeric widening conversions
macro_rules! impl_promote_as {
    ($from:ty => $to:ty) => {
        impl PromoteTo<$to> for $from {
            #[inline]
            fn promote(self) -> $to {
                self as $to
            }
        }
    };
}

// bool -> everything (can't use `as` cast for bool -> float)
macro_rules! impl_promote_bool_to_int {
    ($to:ty) => {
        impl PromoteTo<$to> for bool {
            #[inline]
            fn promote(self) -> $to {
                self as $to
            }
        }
    };
}

impl_promote_bool_to_int!(u8);
impl_promote_bool_to_int!(u16);
impl_promote_bool_to_int!(u32);
impl_promote_bool_to_int!(u64);
impl_promote_bool_to_int!(u128);
impl_promote_bool_to_int!(i8);
impl_promote_bool_to_int!(i16);
impl_promote_bool_to_int!(i32);
impl_promote_bool_to_int!(i64);
impl_promote_bool_to_int!(i128);

impl PromoteTo<f32> for bool {
    #[inline]
    fn promote(self) -> f32 {
        if self { 1.0 } else { 0.0 }
    }
}

impl PromoteTo<f64> for bool {
    #[inline]
    fn promote(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

impl PromoteTo<Complex<f32>> for bool {
    #[inline]
    fn promote(self) -> Complex<f32> {
        Complex::new(if self { 1.0 } else { 0.0 }, 0.0)
    }
}

impl PromoteTo<Complex<f64>> for bool {
    #[inline]
    fn promote(self) -> Complex<f64> {
        Complex::new(if self { 1.0 } else { 0.0 }, 0.0)
    }
}

// Unsigned integer widening
impl_promote_as!(u8 => u16);
impl_promote_as!(u8 => u32);
impl_promote_as!(u8 => u64);
impl_promote_as!(u8 => u128);
impl_promote_as!(u8 => i16);
impl_promote_as!(u8 => i32);
impl_promote_as!(u8 => i64);
impl_promote_as!(u8 => i128);
impl_promote_as!(u8 => f32);
impl_promote_as!(u8 => f64);
impl_promote_as!(u16 => u32);
impl_promote_as!(u16 => u64);
impl_promote_as!(u16 => u128);
impl_promote_as!(u16 => i32);
impl_promote_as!(u16 => i64);
impl_promote_as!(u16 => i128);
impl_promote_as!(u16 => f32);
impl_promote_as!(u16 => f64);
impl_promote_as!(u32 => u64);
impl_promote_as!(u32 => u128);
impl_promote_as!(u32 => i64);
impl_promote_as!(u32 => i128);
impl_promote_as!(u32 => f64);
impl_promote_as!(u64 => u128);
impl_promote_as!(u64 => i128);
impl_promote_as!(u64 => f64);
impl_promote_as!(u128 => f64);

// Signed integer widening
impl_promote_as!(i8 => i16);
impl_promote_as!(i8 => i32);
impl_promote_as!(i8 => i64);
impl_promote_as!(i8 => i128);
impl_promote_as!(i8 => f32);
impl_promote_as!(i8 => f64);
impl_promote_as!(i16 => i32);
impl_promote_as!(i16 => i64);
impl_promote_as!(i16 => i128);
impl_promote_as!(i16 => f32);
impl_promote_as!(i16 => f64);
impl_promote_as!(i32 => i64);
impl_promote_as!(i32 => i128);
impl_promote_as!(i32 => f64);
impl_promote_as!(i64 => i128);
impl_promote_as!(i64 => f64);
impl_promote_as!(i128 => f64);

// Float widening
impl_promote_as!(f32 => f64);

// Integer -> Complex
macro_rules! impl_promote_int_to_complex {
    ($from:ty) => {
        impl PromoteTo<Complex<f32>> for $from {
            #[inline]
            fn promote(self) -> Complex<f32> {
                Complex::new(self as f32, 0.0)
            }
        }
        impl PromoteTo<Complex<f64>> for $from {
            #[inline]
            fn promote(self) -> Complex<f64> {
                Complex::new(self as f64, 0.0)
            }
        }
    };
}

impl_promote_int_to_complex!(u8);
impl_promote_int_to_complex!(u16);
impl_promote_int_to_complex!(u32);
impl_promote_int_to_complex!(u64);
impl_promote_int_to_complex!(u128);
impl_promote_int_to_complex!(i8);
impl_promote_int_to_complex!(i16);
impl_promote_int_to_complex!(i32);
impl_promote_int_to_complex!(i64);
impl_promote_int_to_complex!(i128);

// Float -> Complex
impl PromoteTo<Complex<Self>> for f32 {
    #[inline]
    fn promote(self) -> Complex<Self> {
        Complex::new(self, 0.0)
    }
}

impl PromoteTo<Complex<f64>> for f32 {
    #[inline]
    fn promote(self) -> Complex<f64> {
        Complex::new(self as f64, 0.0)
    }
}

impl PromoteTo<Complex<Self>> for f64 {
    #[inline]
    fn promote(self) -> Complex<Self> {
        Complex::new(self, 0.0)
    }
}

impl PromoteTo<Complex<f64>> for Complex<f32> {
    #[inline]
    fn promote(self) -> Complex<f64> {
        Complex::new(self.re as f64, self.im as f64)
    }
}

// ---------------------------------------------------------------------------
// Promoted trait implementations (compile-time promotion pairs)
// ---------------------------------------------------------------------------

// Macro for implementing Promoted where both sides promote to the same Output
macro_rules! impl_promoted {
    ($a:ty, $b:ty => $out:ty) => {
        impl Promoted<$b> for $a {
            type Output = $out;
        }
        impl Promoted<$a> for $b {
            type Output = $out;
        }
    };
}

// Same-type promotions
macro_rules! impl_promoted_self {
    ($ty:ty) => {
        impl Promoted<$ty> for $ty {
            type Output = $ty;
        }
    };
}

impl_promoted_self!(bool);
impl_promoted_self!(u8);
impl_promoted_self!(u16);
impl_promoted_self!(u32);
impl_promoted_self!(u64);
impl_promoted_self!(u128);
impl_promoted_self!(i8);
impl_promoted_self!(i16);
impl_promoted_self!(i32);
impl_promoted_self!(i64);
impl_promoted_self!(i128);
impl_promoted_self!(f32);
impl_promoted_self!(f64);
impl_promoted_self!(Complex<f32>);
impl_promoted_self!(Complex<f64>);

// bool + anything -> anything
impl_promoted!(bool, u8 => u8);
impl_promoted!(bool, u16 => u16);
impl_promoted!(bool, u32 => u32);
impl_promoted!(bool, u64 => u64);
impl_promoted!(bool, u128 => u128);
impl_promoted!(bool, i8 => i8);
impl_promoted!(bool, i16 => i16);
impl_promoted!(bool, i32 => i32);
impl_promoted!(bool, i64 => i64);
impl_promoted!(bool, i128 => i128);
impl_promoted!(bool, f32 => f32);
impl_promoted!(bool, f64 => f64);
impl_promoted!(bool, Complex<f32> => Complex<f32>);
impl_promoted!(bool, Complex<f64> => Complex<f64>);

// Unsigned + Unsigned
impl_promoted!(u8, u16 => u16);
impl_promoted!(u8, u32 => u32);
impl_promoted!(u8, u64 => u64);
impl_promoted!(u8, u128 => u128);
impl_promoted!(u16, u32 => u32);
impl_promoted!(u16, u64 => u64);
impl_promoted!(u16, u128 => u128);
impl_promoted!(u32, u64 => u64);
impl_promoted!(u32, u128 => u128);
impl_promoted!(u64, u128 => u128);

// Signed + Signed
impl_promoted!(i8, i16 => i16);
impl_promoted!(i8, i32 => i32);
impl_promoted!(i8, i64 => i64);
impl_promoted!(i8, i128 => i128);
impl_promoted!(i16, i32 => i32);
impl_promoted!(i16, i64 => i64);
impl_promoted!(i16, i128 => i128);
impl_promoted!(i32, i64 => i64);
impl_promoted!(i32, i128 => i128);
impl_promoted!(i64, i128 => i128);

// Unsigned + Signed (need next-size signed to hold both ranges)
impl_promoted!(u8, i8 => i16);
impl_promoted!(u8, i16 => i16);
impl_promoted!(u8, i32 => i32);
impl_promoted!(u8, i64 => i64);
impl_promoted!(u8, i128 => i128);
impl_promoted!(u16, i8 => i32);
impl_promoted!(u16, i16 => i32);
impl_promoted!(u16, i32 => i32);
impl_promoted!(u16, i64 => i64);
impl_promoted!(u16, i128 => i128);
impl_promoted!(u32, i8 => i64);
impl_promoted!(u32, i16 => i64);
impl_promoted!(u32, i32 => i64);
impl_promoted!(u32, i64 => i64);
impl_promoted!(u32, i128 => i128);
impl_promoted!(u64, i8 => i128);
impl_promoted!(u64, i16 => i128);
impl_promoted!(u64, i32 => i128);
impl_promoted!(u64, i64 => i128);
impl_promoted!(u64, i128 => i128);
impl_promoted!(u128, i8 => f64);
impl_promoted!(u128, i16 => f64);
impl_promoted!(u128, i32 => f64);
impl_promoted!(u128, i64 => f64);
impl_promoted!(u128, i128 => f64);

// Integer + Float (ensure enough precision)
// Integers up to 16 bits fit in f32 (24-bit mantissa). Larger need f64.
impl_promoted!(u8, f32 => f32);
impl_promoted!(u8, f64 => f64);
impl_promoted!(u16, f32 => f32);
impl_promoted!(u16, f64 => f64);
impl_promoted!(u32, f32 => f64);
impl_promoted!(u32, f64 => f64);
impl_promoted!(u64, f32 => f64);
impl_promoted!(u64, f64 => f64);
impl_promoted!(u128, f32 => f64);
impl_promoted!(u128, f64 => f64);
impl_promoted!(i8, f32 => f32);
impl_promoted!(i8, f64 => f64);
impl_promoted!(i16, f32 => f32);
impl_promoted!(i16, f64 => f64);
impl_promoted!(i32, f32 => f64);
impl_promoted!(i32, f64 => f64);
impl_promoted!(i64, f32 => f64);
impl_promoted!(i64, f64 => f64);
impl_promoted!(i128, f32 => f64);
impl_promoted!(i128, f64 => f64);

// Float + Float
impl_promoted!(f32, f64 => f64);

// Real + Complex
impl_promoted!(u8, Complex<f32> => Complex<f32>);
impl_promoted!(u8, Complex<f64> => Complex<f64>);
impl_promoted!(u16, Complex<f32> => Complex<f32>);
impl_promoted!(u16, Complex<f64> => Complex<f64>);
impl_promoted!(u32, Complex<f32> => Complex<f64>);
impl_promoted!(u32, Complex<f64> => Complex<f64>);
impl_promoted!(u64, Complex<f32> => Complex<f64>);
impl_promoted!(u64, Complex<f64> => Complex<f64>);
impl_promoted!(u128, Complex<f32> => Complex<f64>);
impl_promoted!(u128, Complex<f64> => Complex<f64>);
impl_promoted!(i8, Complex<f32> => Complex<f32>);
impl_promoted!(i8, Complex<f64> => Complex<f64>);
impl_promoted!(i16, Complex<f32> => Complex<f32>);
impl_promoted!(i16, Complex<f64> => Complex<f64>);
impl_promoted!(i32, Complex<f32> => Complex<f64>);
impl_promoted!(i32, Complex<f64> => Complex<f64>);
impl_promoted!(i64, Complex<f32> => Complex<f64>);
impl_promoted!(i64, Complex<f64> => Complex<f64>);
impl_promoted!(i128, Complex<f32> => Complex<f64>);
impl_promoted!(i128, Complex<f64> => Complex<f64>);
impl_promoted!(f32, Complex<f32> => Complex<f32>);
impl_promoted!(f32, Complex<f64> => Complex<f64>);
impl_promoted!(f64, Complex<f32> => Complex<f64>);
impl_promoted!(f64, Complex<f64> => Complex<f64>);

// Complex + Complex
impl_promoted!(Complex<f32>, Complex<f64> => Complex<f64>);

// ---------------------------------------------------------------------------
// result_type() — runtime type promotion
// ---------------------------------------------------------------------------

/// Determine the result type of a binary operation between two dtypes at runtime.
///
/// Follows `NumPy`'s type promotion rules: returns the smallest type that can
/// represent both inputs without precision loss.
///
/// # 128-bit promotion behaviour
///
/// ferray supports `u128`/`i128` as a NumPy-incompatible extension.
/// No native Rust integer wider than 128 bits exists on stable, so
/// ferray provides [`crate::dtype::I256`] — a 256-bit
/// two's-complement type — and uses it as the promoted dtype for
/// mixed `u128` + any signed int. This closes the former lossy
/// `F64` fallback (issue #375, #562). The result of arithmetic on
/// `(u128, i128)` arrays is therefore fully lossless.
///
/// `U128` or `I128` mixed with a float type still promotes to `F64`
/// because the caller already asked for a float result — the
/// precision-above-`2^53` loss is the standard IEEE-754 contract.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if promotion is not possible (should
/// not happen for valid `DType` values).
pub fn result_type(a: DType, b: DType) -> FerrayResult<DType> {
    if a == b {
        return Ok(a);
    }

    // Use a static lookup table for the promotion result.
    let result = promote_dtypes(a, b);
    result.ok_or_else(|| FerrayError::invalid_dtype(format!("cannot promote {a} and {b}")))
}

/// Stable ordinal for promotion ordering.
///
/// `DType as u32` no longer compiles since `DateTime64(TimeUnit)` and
/// `Timedelta64(TimeUnit)` carry data — Rust only allows the C-style
/// cast for fieldless enums. We hand-build an ordinal that's consistent
/// with the historical numeric ordering of the primitive variants and
/// places the time types at the end (they don't promote with numeric
/// types in NumPy either way).
const fn dtype_ord(dt: DType) -> u32 {
    use DType::{
        Bool, Complex32, Complex64, F32, F64, I8, I16, I32, I64, I128, I256, U8, U16, U32, U64,
        U128,
    };
    match dt {
        Bool => 0,
        U8 => 1,
        U16 => 2,
        U32 => 3,
        U64 => 4,
        U128 => 5,
        I8 => 6,
        I16 => 7,
        I32 => 8,
        I64 => 9,
        I128 => 10,
        I256 => 11,
        F32 => 12,
        F64 => 13,
        Complex32 => 14,
        Complex64 => 15,
        #[cfg(feature = "f16")]
        DType::F16 => 16,
        #[cfg(feature = "bf16")]
        DType::BF16 => 17,
        DType::DateTime64(_) => 100,
        DType::Timedelta64(_) => 101,
        // Struct dtypes don't promote (#342). Place them out of the
        // numeric ordering range; promote_dtypes returns None when
        // either side is structured (atomic dtype).
        DType::Struct(_) => 200,
    }
}

/// Internal promotion function returning Option.
fn promote_dtypes(a: DType, b: DType) -> Option<DType> {
    use DType::{
        Bool, Complex32, Complex64, F32, F64, I8, I16, I32, I64, I128, I256, U8, U16, U32, U64,
        U128,
    };

    if a == b {
        return Some(a);
    }

    // Ensure canonical ordering: if b < a in our ordinal, swap them
    // so we only need to handle (smaller, larger) pairs.
    let (lo, hi) = if dtype_ord(a) <= dtype_ord(b) {
        (a, b)
    } else {
        (b, a)
    };

    // Bool promotes to anything
    if lo == Bool {
        return Some(hi);
    }

    let result = match (lo, hi) {
        // Unsigned + Unsigned
        (U8, U16) => U16,
        (U8, U32) => U32,
        (U8, U64) => U64,
        (U8, U128) => U128,
        (U16, U32) => U32,
        (U16, U64) => U64,
        (U16, U128) => U128,
        (U32, U64) => U64,
        (U32, U128) => U128,
        (U64, U128) => U128,

        // Signed + Signed
        (I8, I16) => I16,
        (I8, I32) => I32,
        (I8, I64) => I64,
        (I8, I128) => I128,
        (I16, I32) => I32,
        (I16, I64) => I64,
        (I16, I128) => I128,
        (I32, I64) => I64,
        (I32, I128) => I128,
        (I64, I128) => I128,

        // Unsigned + Signed
        (U8, I8) => I16,
        (U8, I16) => I16,
        (U8, I32) => I32,
        (U8, I64) => I64,
        (U8, I128) => I128,
        (U16, I8) => I32,
        (U16, I16) => I32,
        (U16, I32) => I32,
        (U16, I64) => I64,
        (U16, I128) => I128,
        (U32, I8) => I64,
        (U32, I16) => I64,
        (U32, I32) => I64,
        (U32, I64) => I64,
        (U32, I128) => I128,
        (U64, I8) => I128,
        (U64, I16) => I128,
        (U64, I32) => I128,
        (U64, I64) => I128,
        (U64, I128) => I128,
        // `U128 + <any signed int>` promotes to the custom
        // `I256` type which can losslessly hold the full union
        // (129 signed bits would suffice; we round up to 256 for
        // alignment and arithmetic simplicity). This closes the
        // former lossy-`F64` fallback (#375, #562).
        (U128, I8) => I256,
        (U128, I16) => I256,
        (U128, I32) => I256,
        (U128, I64) => I256,
        (U128, I128) => I256,

        // Integer + Float
        (U8 | U16 | I8 | I16, F32) => F32,
        (U8 | U16 | U32 | U64 | U128, F64) => F64,
        (I8 | I16 | I32 | I64 | I128, F64) => F64,
        (U32 | U64 | U128, F32) => F64,
        (I32 | I64 | I128, F32) => F64,

        // Float + Float
        (F32, F64) => F64,

        // Real + Complex
        (U8 | U16 | I8 | I16 | F32, Complex32) => Complex32,
        (U32 | U64 | U128, Complex32) => Complex64,
        (I32 | I64 | I128, Complex32) => Complex64,
        (F64, Complex32) => Complex64,

        // Complex + Complex, and anything + Complex64
        (Complex32, Complex64) => Complex64,
        (_, Complex64) => Complex64,

        _ => return None,
    };

    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_type_same() {
        assert_eq!(result_type(DType::F64, DType::F64).unwrap(), DType::F64);
        assert_eq!(result_type(DType::I32, DType::I32).unwrap(), DType::I32);
    }

    #[test]
    fn result_type_float_promotion() {
        assert_eq!(result_type(DType::F32, DType::F64).unwrap(), DType::F64);
    }

    #[test]
    fn result_type_int_float() {
        assert_eq!(result_type(DType::I32, DType::F32).unwrap(), DType::F64);
        assert_eq!(result_type(DType::I16, DType::F32).unwrap(), DType::F32);
        assert_eq!(result_type(DType::U8, DType::F64).unwrap(), DType::F64);
    }

    #[test]
    fn result_type_u128_signed_promotes_to_i256() {
        // Issue #375 / #562: `u128 + any signed int` has no lossless
        // native common type, so ferray promotes to its custom I256
        // (256-bit two's-complement) which can hold the full union.
        assert_eq!(result_type(DType::U128, DType::I8).unwrap(), DType::I256);
        assert_eq!(result_type(DType::U128, DType::I16).unwrap(), DType::I256);
        assert_eq!(result_type(DType::U128, DType::I32).unwrap(), DType::I256);
        assert_eq!(result_type(DType::U128, DType::I64).unwrap(), DType::I256);
        assert_eq!(result_type(DType::U128, DType::I128).unwrap(), DType::I256);
        // Symmetric check.
        assert_eq!(result_type(DType::I64, DType::U128).unwrap(), DType::I256);
    }

    #[test]
    fn result_type_u128_float_still_f64() {
        // `U128 + F*` still promotes to F64 because the caller asked
        // for a float result — precision loss above 2^53 is the
        // standard IEEE-754 contract.
        assert_eq!(result_type(DType::U128, DType::F32).unwrap(), DType::F64);
        assert_eq!(result_type(DType::U128, DType::F64).unwrap(), DType::F64);
        assert_eq!(result_type(DType::I128, DType::F32).unwrap(), DType::F64);
        assert_eq!(result_type(DType::I128, DType::F64).unwrap(), DType::F64);
    }

    #[test]
    fn result_type_complex() {
        assert_eq!(
            result_type(DType::Complex32, DType::F64).unwrap(),
            DType::Complex64
        );
        assert_eq!(
            result_type(DType::F32, DType::Complex32).unwrap(),
            DType::Complex32
        );
        assert_eq!(
            result_type(DType::Complex32, DType::Complex64).unwrap(),
            DType::Complex64
        );
    }

    #[test]
    fn result_type_unsigned_signed() {
        assert_eq!(result_type(DType::U8, DType::I8).unwrap(), DType::I16);
        assert_eq!(result_type(DType::U16, DType::I16).unwrap(), DType::I32);
        assert_eq!(result_type(DType::U32, DType::I32).unwrap(), DType::I64);
        assert_eq!(result_type(DType::U64, DType::I64).unwrap(), DType::I128);
    }

    #[test]
    fn result_type_bool() {
        assert_eq!(result_type(DType::Bool, DType::F64).unwrap(), DType::F64);
        assert_eq!(result_type(DType::Bool, DType::I32).unwrap(), DType::I32);
        assert_eq!(result_type(DType::Bool, DType::Bool).unwrap(), DType::Bool);
    }

    #[test]
    fn promoted_trait_compile_time() {
        // Verify the trait resolves correctly
        fn check_promotion<A, B>() -> DType
        where
            A: Promoted<B>,
            B: Element,
            <A as Promoted<B>>::Output: Element,
        {
            <A as Promoted<B>>::Output::dtype()
        }

        assert_eq!(check_promotion::<f32, f64>(), DType::F64);
        assert_eq!(check_promotion::<i32, f32>(), DType::F64);
        assert_eq!(check_promotion::<Complex<f32>, f64>(), DType::Complex64);
        assert_eq!(check_promotion::<u8, i8>(), DType::I16);
    }

    #[test]
    fn promote_to_identity() {
        let a: i32 = PromoteTo::<i32>::promote(42i32);
        assert_eq!(a, 42i32);
        let b: f64 = PromoteTo::<f64>::promote(2.5f64);
        assert_eq!(b, 2.5f64);
    }

    #[test]
    fn promote_to_widening() {
        let x: f64 = PromoteTo::<f64>::promote(42i32);
        assert_eq!(x, 42.0);

        let y: i16 = PromoteTo::<i16>::promote(255u8);
        assert_eq!(y, 255);

        let z: Complex<f64> = PromoteTo::<Complex<f64>>::promote(2.5f32);
        assert_eq!(z, Complex::new(2.5f32 as f64, 0.0));
    }
}
