// ferray-core: Element trait and DType runtime enum (REQ-6, REQ-7)

use core::fmt;

use num_complex::Complex;

pub mod casting;
pub mod datetime;
pub mod finfo;
pub mod i256;
pub mod promotion;
pub mod unsafe_cast;

pub use datetime::{DateTime64, NAT, TimeUnit, Timedelta64};
pub use i256::I256;

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
    /// `I256` — 256-bit two's-complement signed integer. Used as
    /// the promoted type for mixed `u128` + `i128` arithmetic
    /// (no native Rust type can hold that union losslessly).
    I256,
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
    /// `bf16` (bfloat16) — only available with the `bf16` feature.
    #[cfg(feature = "bf16")]
    BF16,
    /// `datetime64[unit]` — calendar instant as i64 ticks since the Unix
    /// epoch, with the [`TimeUnit`] tagging the meaning of one tick.
    /// Backed by the [`DateTime64`] element type.
    DateTime64(TimeUnit),
    /// `timedelta64[unit]` — duration as i64 ticks. Backed by the
    /// [`Timedelta64`] element type.
    Timedelta64(TimeUnit),
    /// Structured (record) dtype holding a static descriptor for each
    /// field (#342). Mirrors numpy's structured dtype:
    ///
    /// ```text
    /// np.dtype([('x', 'f4'), ('y', 'f4')])
    /// ```
    ///
    /// The slice is `&'static` so that `DType` stays `Copy`. The
    /// `FerrayRecord` derive macro emits exactly this shape via
    /// `Type::field_descriptors() -> &'static [FieldDescriptor]`,
    /// so the typical construction site is:
    ///
    /// ```ignore
    /// let dt = DType::Struct(<Point as FerrayRecord>::field_descriptors());
    /// ```
    ///
    /// Dynamically-constructed descriptors must be leaked into a
    /// `&'static [FieldDescriptor]` (e.g. via `Box::leak`); ferray-core
    /// does not allocate field descriptors at runtime.
    Struct(&'static [crate::record::FieldDescriptor]),

    /// Fixed-width ASCII string with `n` bytes per element. Mirrors
    /// numpy's `'S{n}'` dtype family. Backed by `[u8; n]` rows in a
    /// flat byte buffer; trailing zeros pad short strings (#741).
    FixedAscii(usize),

    /// Fixed-width Unicode string with `n` codepoints per element
    /// stored as UCS-4 (`4 * n` bytes per element). Mirrors numpy's
    /// `'U{n}'` dtype family (#741).
    FixedUnicode(usize),

    /// Fixed-width raw byte / void dtype with `n` bytes per
    /// element. Mirrors numpy's `'V{n}'` dtype family — opaque
    /// fixed-size byte payloads with no UTF-8 / ASCII assumption
    /// (#741).
    RawBytes(usize),
}

impl DType {
    /// Size in bytes of one element of this dtype.
    ///
    /// Note: lost const-ness when [`Self::Struct`] was added (#342) —
    /// computing struct size requires iterating field descriptors,
    /// which stable const fn doesn't support yet.
    #[inline]
    #[must_use]
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
            Self::I256 => 32,
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Complex32 => 8,
            Self::Complex64 => 16,
            #[cfg(feature = "f16")]
            Self::F16 => 2,
            #[cfg(feature = "bf16")]
            Self::BF16 => 2,
            // datetime64 / timedelta64 are i64 counts; the unit changes
            // the interpretation but not the storage size.
            Self::DateTime64(_) | Self::Timedelta64(_) => 8,
            // #342: struct size = max(field.offset + field.size). This
            // matches `size_of::<T>()` for a #[repr(C)] FerrayRecord
            // because the derive emits offsets via `offset_of!` and
            // sizes via `size_of::<field_ty>()`.
            Self::Struct(fields) => {
                let mut max_end = 0usize;
                for f in fields {
                    let end = f.offset + f.size;
                    if end > max_end {
                        max_end = end;
                    }
                }
                max_end
            }
            // String / void dtypes: fixed bytes per element.
            // FixedUnicode is UCS-4 (4 bytes per codepoint).
            Self::FixedAscii(n) | Self::RawBytes(n) => n,
            Self::FixedUnicode(n) => 4 * n,
        }
    }

    /// Required alignment in bytes for this dtype.
    ///
    /// Note: lost const-ness with [`Self::Struct`] (#342) — struct
    /// alignment is the max of field alignments, which requires
    /// iterating field descriptors.
    #[inline]
    #[must_use]
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
            // I256 uses `[u64; 4]` so its natural alignment matches u64.
            Self::I256 => core::mem::align_of::<I256>(),
            Self::F32 => 4,
            Self::F64 => 8,
            Self::Complex32 => core::mem::align_of::<Complex<f32>>(),
            Self::Complex64 => core::mem::align_of::<Complex<f64>>(),
            #[cfg(feature = "f16")]
            Self::F16 => core::mem::align_of::<half::f16>(),
            #[cfg(feature = "bf16")]
            Self::BF16 => core::mem::align_of::<half::bf16>(),
            Self::DateTime64(_) | Self::Timedelta64(_) => core::mem::align_of::<i64>(),
            // #342: struct alignment = max of field alignments.
            // Falls back to 1 for an empty struct.
            Self::Struct(fields) => {
                let mut max_align = 1usize;
                for f in fields {
                    let a = f.dtype.alignment();
                    if a > max_align {
                        max_align = a;
                    }
                }
                max_align
            }
            // FixedAscii / RawBytes are byte-aligned arrays;
            // FixedUnicode is naturally u32-aligned (UCS-4).
            Self::FixedAscii(_) | Self::RawBytes(_) => 1,
            Self::FixedUnicode(_) => core::mem::align_of::<u32>(),
        }
    }

    /// `true` if the dtype is a floating-point type (f16, bf16, f32, f64).
    #[inline]
    #[must_use]
    pub const fn is_float(self) -> bool {
        #[cfg(feature = "f16")]
        if matches!(self, Self::F16) {
            return true;
        }
        #[cfg(feature = "bf16")]
        if matches!(self, Self::BF16) {
            return true;
        }
        matches!(self, Self::F32 | Self::F64)
    }

    /// `true` if the dtype is an integer type (signed or unsigned, including bool).
    #[inline]
    #[must_use]
    pub const fn is_integer(self) -> bool {
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
                | Self::I256
        )
    }

    /// `true` if the dtype is a complex type.
    #[inline]
    #[must_use]
    pub const fn is_complex(self) -> bool {
        matches!(self, Self::Complex32 | Self::Complex64)
    }

    /// `true` if the dtype is a signed numeric type.
    #[inline]
    #[must_use]
    pub const fn is_signed(self) -> bool {
        matches!(
            self,
            Self::I8
                | Self::I16
                | Self::I32
                | Self::I64
                | Self::I128
                | Self::I256
                | Self::F32
                | Self::F64
                | Self::Complex32
                | Self::Complex64
        ) || {
            #[cfg(feature = "f16")]
            if matches!(self, Self::F16) {
                return true;
            }
            #[cfg(feature = "bf16")]
            if matches!(self, Self::BF16) {
                return true;
            }
            false
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => f.write_str("bool"),
            Self::U8 => f.write_str("uint8"),
            Self::U16 => f.write_str("uint16"),
            Self::U32 => f.write_str("uint32"),
            Self::U64 => f.write_str("uint64"),
            Self::U128 => f.write_str("uint128"),
            Self::I8 => f.write_str("int8"),
            Self::I16 => f.write_str("int16"),
            Self::I32 => f.write_str("int32"),
            Self::I64 => f.write_str("int64"),
            Self::I128 => f.write_str("int128"),
            Self::I256 => f.write_str("int256"),
            Self::F32 => f.write_str("float32"),
            Self::F64 => f.write_str("float64"),
            Self::Complex32 => f.write_str("complex64"),
            Self::Complex64 => f.write_str("complex128"),
            #[cfg(feature = "f16")]
            Self::F16 => f.write_str("float16"),
            #[cfg(feature = "bf16")]
            Self::BF16 => f.write_str("bfloat16"),
            Self::DateTime64(u) => write!(f, "datetime64[{u}]"),
            Self::Timedelta64(u) => write!(f, "timedelta64[{u}]"),
            Self::Struct(fields) => {
                // numpy's structured-dtype repr: [('name', dtype), ...]
                f.write_str("struct[")?;
                for (i, fld) in fields.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "('{}', {})", fld.name, fld.dtype)?;
                }
                f.write_str("]")
            }
            // Match numpy's '|S{n}' / '<U{n}' / '|V{n}' descriptors
            // with the canonical type tag.
            Self::FixedAscii(n) => write!(f, "S{n}"),
            Self::FixedUnicode(n) => write!(f, "U{n}"),
            Self::RawBytes(n) => write!(f, "V{n}"),
        }
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

// Blanket impl: any FerrayRecord type seals into Element so npy
// load_record / save_record can produce Array<T, D> for structured
// dtypes (#742). This is sound because FerrayRecord is a separate
// `unsafe` trait that no primitive type implements; the orphan +
// coherence checks treat this as non-overlapping with the concrete
// primitive impls below.
impl<T: crate::record::FerrayRecord> private::Sealed for T {}

impl<T: crate::record::FerrayRecord + Clone + fmt::Debug + fmt::Display + Send + Sync + PartialEq>
    Element for T
{
    fn dtype() -> DType {
        DType::Struct(<T as crate::record::FerrayRecord>::field_descriptors())
    }
    fn zero() -> Self {
        // Records have no canonical zero — callers that need
        // zero-initialised storage should construct it explicitly.
        // This panic is unreachable in practice because the only
        // codepath that calls Element::zero on a record is
        // Array::zeros, which we already reject for FerrayRecord
        // types higher up the stack.
        panic!(
            "FerrayRecord type {} has no canonical Element::zero; \
             construct records explicitly",
            std::any::type_name::<T>()
        )
    }
    fn one() -> Self {
        panic!(
            "FerrayRecord type {} has no canonical Element::one; \
             construct records explicitly",
            std::any::type_name::<T>()
        )
    }
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

// I256 — hand-written because the macro wants `0 as Self` / `1 as Self`.
impl private::Sealed for I256 {}

impl Element for I256 {
    #[inline]
    fn dtype() -> DType {
        DType::I256
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

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
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Self::new(1.0, 0.0)
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
        Self::new(0.0, 0.0)
    }

    #[inline]
    fn one() -> Self {
        Self::new(1.0, 0.0)
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
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
    }
}

// bf16 — feature-gated
#[cfg(feature = "bf16")]
impl private::Sealed for half::bf16 {}

#[cfg(feature = "bf16")]
impl Element for half::bf16 {
    #[inline]
    fn dtype() -> DType {
        DType::BF16
    }

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn one() -> Self {
        Self::ONE
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

    // ----- DType::Struct (#342) ----------------------------------------

    #[test]
    fn struct_dtype_size_of_sums_field_extents() {
        // Two f64 fields at offsets 0 and 8 → size 16.
        static FIELDS: &[crate::record::FieldDescriptor] = &[
            crate::record::FieldDescriptor {
                name: "x",
                dtype: DType::F64,
                offset: 0,
                size: 8,
            },
            crate::record::FieldDescriptor {
                name: "y",
                dtype: DType::F64,
                offset: 8,
                size: 8,
            },
        ];
        assert_eq!(DType::Struct(FIELDS).size_of(), 16);
    }

    #[test]
    fn struct_dtype_alignment_is_max_field_alignment() {
        // f64 field (align 8) + i8 field (align 1) → max alignment 8.
        static FIELDS: &[crate::record::FieldDescriptor] = &[
            crate::record::FieldDescriptor {
                name: "big",
                dtype: DType::F64,
                offset: 0,
                size: 8,
            },
            crate::record::FieldDescriptor {
                name: "tag",
                dtype: DType::I8,
                offset: 8,
                size: 1,
            },
        ];
        assert_eq!(DType::Struct(FIELDS).alignment(), 8);
    }

    #[test]
    fn struct_dtype_display_format() {
        static FIELDS: &[crate::record::FieldDescriptor] = &[
            crate::record::FieldDescriptor {
                name: "x",
                dtype: DType::F32,
                offset: 0,
                size: 4,
            },
            crate::record::FieldDescriptor {
                name: "y",
                dtype: DType::I32,
                offset: 4,
                size: 4,
            },
        ];
        let s = format!("{}", DType::Struct(FIELDS));
        assert_eq!(s, "struct[('x', float32), ('y', int32)]");
    }

    #[test]
    fn struct_dtype_equality_and_hash_via_field_slice() {
        static A: &[crate::record::FieldDescriptor] = &[crate::record::FieldDescriptor {
            name: "x",
            dtype: DType::F64,
            offset: 0,
            size: 8,
        }];
        static B: &[crate::record::FieldDescriptor] = &[crate::record::FieldDescriptor {
            name: "x",
            dtype: DType::F64,
            offset: 0,
            size: 8,
        }];
        // Different static addresses but identical contents → equal.
        assert_eq!(DType::Struct(A), DType::Struct(B));
        // Hash consistency: equal values must produce equal hashes.
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        DType::Struct(A).hash(&mut ha);
        DType::Struct(B).hash(&mut hb);
        assert_eq!(ha.finish(), hb.finish());
    }

    #[test]
    fn struct_dtype_is_copy() {
        // The whole point of using &'static [FieldDescriptor] (rather
        // than Arc<[...]>) is to keep DType: Copy. Pin that contract.
        static FIELDS: &[crate::record::FieldDescriptor] = &[];
        let dt = DType::Struct(FIELDS);
        let copied = dt; // Implicit Copy.
        assert_eq!(dt, copied);
    }
}
