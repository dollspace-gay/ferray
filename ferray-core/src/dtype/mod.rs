// ferray-core: Element trait and DType runtime enum (REQ-6, REQ-7)

use core::fmt;

use num_complex::Complex;

pub mod casting;
pub mod finfo;
pub mod promotion;

// ---------------------------------------------------------------------------
// SliceInfoElem — used by the s![] macro
// ---------------------------------------------------------------------------

/// One element of a multi-axis slice specification, produced by the `s![]` macro.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SliceInfoElem {
    /// A single integer index along one axis. Reduces dimensionality by 1.
    Index(isize),
    /// A slice (start..end with step) along one axis.
    Slice {
        /// Start index (inclusive). 0 means the beginning.
        start: isize,
        /// End index (exclusive). `None` means the end of the axis.
        end: Option<isize>,
        /// Step size. Must not be 0.
        step: isize,
    },
}

// ---------------------------------------------------------------------------
// DType runtime enum
// ---------------------------------------------------------------------------

/// Runtime descriptor for the element type stored in an array.
///
/// Mirrors the set of types implementing [`Element`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// `bool`
    Bool,
    /// `u8`
    U8,
    /// `u16`
    U16,
    /// `u32`
    U32,
    /// `u64`
    U64,
    /// `u128`
    U128,
    /// `i8`
    I8,
    /// `i16`
    I16,
    /// `i32`
    I32,
    /// `i64`
    I64,
    /// `i128`
    I128,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `Complex<f32>`
    Complex32,
    /// `Complex<f64>`
    Complex64,
    /// `f16` — only available with the `f16` feature.
    #[cfg(feature = "f16")]
    F16,
}

impl DType {
    /// Size in bytes of one element of this dtype.
    #[inline]
    pub fn size_of(self) -> usize {
        match self {
            Self::Bool => core::mem::size_of::<bool>(),
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::U128 => 16,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::I128 => 16,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Complex32 => 8,
            Self::Complex64 => 16,
            #[cfg(feature = "f16")]
            Self::F16 => 2,
        }
    }

    /// Required alignment in bytes for this dtype.
    #[inline]
    pub fn alignment(self) -> usize {
        match self {
            Self::Bool => core::mem::align_of::<bool>(),
            Self::U8 => 1,
            Self::U16 => 2,
            Self::U32 => 4,
            Self::U64 => 8,
            Self::U128 => 16,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,
            Self::I128 => 16,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Complex32 => core::mem::align_of::<Complex<f32>>(),
            Self::Complex64 => core::mem::align_of::<Complex<f64>>(),
            #[cfg(feature = "f16")]
            Self::F16 => core::mem::align_of::<half::f16>(),
        }
    }

    /// `true` if the dtype is a floating-point type (f16, f32, f64).
    #[inline]
    pub fn is_float(self) -> bool {
        #[cfg(feature = "f16")]
        if matches!(self, Self::F16) {
            return true;
        }
        matches!(self, Self::F32 | Self::F64)
    }

    /// `true` if the dtype is an integer type (signed or unsigned, including bool).
    #[inline]
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Bool
                | Self::U8
                | Self::U16
                | Self::U32
                | Self::U64
                | Self::U128
                | Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::I128
        )
    }

    /// `true` if the dtype is a complex type.
    #[inline]
    pub fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }

    /// `true` if the dtype is a signed numeric type.
    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::I128
                | Self::F32
                | Self::F64
                | Self::Complex32
                | Self::Complex64
        ) || {
            #[cfg(feature = "f16")]
            {
                matches!(self, Self::F16)
            }
            #[cfg(not(feature = "f16"))]
            {
                false
            }
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::U128 => "uint128",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::I128 => "int128",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::Complex32 => "complex64",
            Self::Complex64 => "complex128",
            #[cfg(feature = "f16")]
            Self::F16 => "float16",
        };
        write!(f, "{name}")
    }
}

// ---------------------------------------------------------------------------
// Element trait
// ---------------------------------------------------------------------------

/// Trait bound for types that can be stored in a ferray array.
///
/// This is implemented for all supported numeric types plus `bool`.
/// The trait is sealed: downstream crates cannot implement it for new types
/// (use `DynArray` or `FerrayRecord` for custom element types).
pub trait Element:
    Clone + fmt::Debug + fmt::Display + Send + Sync + PartialEq + 'static + private::Sealed
{
    /// The runtime [`DType`] tag for this element type.
    fn dtype() -> DType;

    /// The zero/additive-identity value.
    fn zero() -> Self;

    /// The one/multiplicative-identity value.
    fn one() -> Self;
}

mod private {
    pub trait Sealed {}
}

// ---------------------------------------------------------------------------
// Element implementations
// ---------------------------------------------------------------------------

macro_rules! impl_element_int {
    ($ty:ty, $variant:ident) => {
        impl private::Sealed for $ty {}

        impl Element for $ty {
            #[inline]
            fn dtype() -> DType {
                DType::$variant
            }

            #[inline]
            fn zero() -> Self {
                0 as Self
            }

            #[inline]
            fn one() -> Self {
                1 as Self
            }
        }
    };
}

macro_rules! impl_element_float {
    ($ty:ty, $variant:ident) => {
        impl private::Sealed for $ty {}

        impl Element for $ty {
            #[inline]
            fn dtype() -> DType {
                DType::$variant
            }

            #[inline]
            fn zero() -> Self {
                0.0 as Self
            }

            #[inline]
            fn one() -> Self {
                1.0 as Self
            }
        }
    };
}

// bool
impl private::Sealed for bool {}

impl Element for bool {
    #[inline]
    fn dtype() -> DType {
        DType::Bool
    }

    #[inline]
    fn zero() -> Self {
        false
    }

    #[inline]
    fn one() -> Self {
        true
    }
}

// Unsigned integers
impl_element_int!(u8, U8);
impl_element_int!(u16, U16);
impl_element_int!(u32, U32);
impl_element_int!(u64, U64);
impl_element_int!(u128, U128);

// Signed integers
impl_element_int!(i8, I8);
impl_element_int!(i16, I16);
impl_element_int!(i32, I32);
impl_element_int!(i64, I64);
impl_element_int!(i128, I128);

// Floats
impl_element_float!(f32, F32);
impl_element_float!(f64, F64);

// Complex
impl private::Sealed for Complex<f32> {}

impl Element for Complex<f32> {
    #[inline]
    fn dtype() -> DType {
        DType::Complex32
    }

    #[inline]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}

impl private::Sealed for Complex<f64> {}

impl Element for Complex<f64> {
    #[inline]
    fn dtype() -> DType {
        DType::Complex64
    }

    #[inline]
    fn zero() -> Self {
        Complex::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Complex::new(1.0, 0.0)
    }
}

// f16 — feature-gated
#[cfg(feature = "f16")]
impl private::Sealed for half::f16 {}

#[cfg(feature = "f16")]
impl Element for half::f16 {
    #[inline]
    fn dtype() -> DType {
        DType::F16
    }

    #[inline]
    fn zero() -> Self {
        half::f16::ZERO
    }

    #[inline]
    fn one() -> Self {
        half::f16::ONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_size_of() {
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::Bool.size_of(), 1);
        assert_eq!(DType::Complex64.size_of(), 16);
        assert_eq!(DType::I128.size_of(), 16);
    }

    #[test]
    fn dtype_introspection() {
        assert!(DType::F64.is_float());
        assert!(DType::F32.is_float());
        assert!(!DType::I32.is_float());

        assert!(DType::I32.is_integer());
        assert!(DType::Bool.is_integer());
        assert!(!DType::F64.is_integer());

        assert!(DType::Complex32.is_complex());
        assert!(DType::Complex64.is_complex());
        assert!(!DType::F64.is_complex());
    }

    #[test]
    fn dtype_display() {
        assert_eq!(DType::F64.to_string(), "float64");
        assert_eq!(DType::I32.to_string(), "int32");
        assert_eq!(DType::Complex64.to_string(), "complex128");
        assert_eq!(DType::Bool.to_string(), "bool");
    }

    #[test]
    fn element_trait() {
        assert_eq!(f64::dtype(), DType::F64);
        assert_eq!(f64::zero(), 0.0);
        assert_eq!(f64::one(), 1.0);
        assert_eq!(i32::dtype(), DType::I32);
        assert_eq!(bool::dtype(), DType::Bool);
        assert!(!bool::zero());
        assert!(bool::one());
    }

    #[test]
    fn complex_element() {
        assert_eq!(Complex::<f64>::dtype(), DType::Complex64);
        assert_eq!(Complex::<f64>::zero(), Complex::new(0.0, 0.0));
        assert_eq!(Complex::<f64>::one(), Complex::new(1.0, 0.0));
    }

    #[test]
    fn dtype_alignment() {
        assert_eq!(DType::F64.alignment(), 8);
        assert_eq!(DType::U8.alignment(), 1);
    }

    #[test]
    fn dtype_signed() {
        assert!(DType::I32.is_signed());
        assert!(DType::F64.is_signed());
        assert!(DType::Complex64.is_signed());
        assert!(!DType::U32.is_signed());
        assert!(!DType::Bool.is_signed());
    }
}
