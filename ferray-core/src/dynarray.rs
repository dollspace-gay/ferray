// ferray-core: DynArray — runtime-typed array enum (REQ-30)

use num_complex::Complex;

use crate::array::owned::Array;
use crate::dimension::IxDyn;
use crate::dtype::casting::CastKind;
use crate::dtype::{DType, DateTime64, I256, TimeUnit, Timedelta64};
use crate::error::{FerrayError, FerrayResult};

/// A runtime-typed array whose element type is determined at runtime.
///
/// This is analogous to a Python `numpy.ndarray` where the dtype is a
/// runtime property. Each variant wraps an `Array<T, IxDyn>` for the
/// corresponding element type.
///
/// Use this when the element type is not known at compile time (e.g.,
/// loading from a file, receiving from Python/FFI).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DynArray {
    /// `bool` elements
    Bool(Array<bool, IxDyn>),
    /// `u8` elements
    U8(Array<u8, IxDyn>),
    /// `u16` elements
    U16(Array<u16, IxDyn>),
    /// `u32` elements
    U32(Array<u32, IxDyn>),
    /// `u64` elements
    U64(Array<u64, IxDyn>),
    /// `u128` elements
    U128(Array<u128, IxDyn>),
    /// `i8` elements
    I8(Array<i8, IxDyn>),
    /// `i16` elements
    I16(Array<i16, IxDyn>),
    /// `i32` elements
    I32(Array<i32, IxDyn>),
    /// `i64` elements
    I64(Array<i64, IxDyn>),
    /// `i128` elements
    I128(Array<i128, IxDyn>),
    /// `I256` elements — 256-bit two's-complement signed integer,
    /// used as the promoted type for mixed `u128` + signed-int
    /// arithmetic (#375, #562).
    I256(Array<I256, IxDyn>),
    /// `f32` elements
    F32(Array<f32, IxDyn>),
    /// `f64` elements
    F64(Array<f64, IxDyn>),
    /// `Complex<f32>` elements
    Complex32(Array<Complex<f32>, IxDyn>),
    /// `Complex<f64>` elements
    Complex64(Array<Complex<f64>, IxDyn>),
    /// `f16` elements (feature-gated)
    #[cfg(feature = "f16")]
    F16(Array<half::f16, IxDyn>),
    /// `bf16` (bfloat16) elements (feature-gated)
    #[cfg(feature = "bf16")]
    BF16(Array<half::bf16, IxDyn>),
    /// `datetime64[unit]` elements — the `TimeUnit` rides alongside the
    /// data because [`DType::DateTime64`] is parameterized but the
    /// element-level [`Element::dtype`](crate::dtype::Element::dtype)
    /// can only return one fixed unit.
    DateTime64(Array<DateTime64, IxDyn>, TimeUnit),
    /// `timedelta64[unit]` elements.
    Timedelta64(Array<Timedelta64, IxDyn>, TimeUnit),
}

/// Dispatch a single expression across every `DynArray` variant, binding
/// the inner `Array<T, IxDyn>` to `$binding`. This turns repeated 17-way
/// match arms into one-line methods (see issue #125); the f16/bf16
/// variants are conditionally compiled in the same way as the enum.
macro_rules! dispatch {
    ($value:expr, $binding:ident => $expr:expr) => {
        match $value {
            Self::Bool($binding) => $expr,
            Self::U8($binding) => $expr,
            Self::U16($binding) => $expr,
            Self::U32($binding) => $expr,
            Self::U64($binding) => $expr,
            Self::U128($binding) => $expr,
            Self::I8($binding) => $expr,
            Self::I16($binding) => $expr,
            Self::I32($binding) => $expr,
            Self::I64($binding) => $expr,
            Self::I128($binding) => $expr,
            Self::I256($binding) => $expr,
            Self::F32($binding) => $expr,
            Self::F64($binding) => $expr,
            Self::Complex32($binding) => $expr,
            Self::Complex64($binding) => $expr,
            #[cfg(feature = "f16")]
            Self::F16($binding) => $expr,
            #[cfg(feature = "bf16")]
            Self::BF16($binding) => $expr,
            // datetime64 / timedelta64 carry an extra TimeUnit slot;
            // the dispatch only needs to bind the inner Array.
            Self::DateTime64($binding, _) => $expr,
            Self::Timedelta64($binding, _) => $expr,
        }
    };
}

impl DynArray {
    /// The runtime dtype of the elements in this array.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        match self {
            Self::Bool(_) => DType::Bool,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::U128(_) => DType::U128,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::I128(_) => DType::I128,
            Self::I256(_) => DType::I256,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::Complex32(_) => DType::Complex32,
            Self::Complex64(_) => DType::Complex64,
            #[cfg(feature = "f16")]
            Self::F16(_) => DType::F16,
            #[cfg(feature = "bf16")]
            Self::BF16(_) => DType::BF16,
            Self::DateTime64(_, u) => DType::DateTime64(*u),
            Self::Timedelta64(_, u) => DType::Timedelta64(*u),
        }
    }

    /// Shape as a slice.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        dispatch!(self, a => a.shape())
    }

    /// Number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Total number of elements.
    #[must_use]
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the array has zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Size in bytes of one element.
    #[must_use]
    pub fn itemsize(&self) -> usize {
        self.dtype().size_of()
    }

    /// Total size in bytes.
    #[must_use]
    pub fn nbytes(&self) -> usize {
        self.size() * self.itemsize()
    }

    /// Try to extract the inner `Array<f64, IxDyn>`.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if the dtype is not `f64`.
    pub fn try_into_f64(self) -> FerrayResult<Array<f64, IxDyn>> {
        match self {
            Self::F64(a) => Ok(a),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected float64, got {}",
                other.dtype()
            ))),
        }
    }

    /// Try to extract the inner `Array<f32, IxDyn>`.
    pub fn try_into_f32(self) -> FerrayResult<Array<f32, IxDyn>> {
        match self {
            Self::F32(a) => Ok(a),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected float32, got {}",
                other.dtype()
            ))),
        }
    }

    /// Try to extract the inner `Array<i64, IxDyn>`.
    pub fn try_into_i64(self) -> FerrayResult<Array<i64, IxDyn>> {
        match self {
            Self::I64(a) => Ok(a),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected int64, got {}",
                other.dtype()
            ))),
        }
    }

    /// Try to extract the inner `Array<i32, IxDyn>`.
    pub fn try_into_i32(self) -> FerrayResult<Array<i32, IxDyn>> {
        match self {
            Self::I32(a) => Ok(a),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected int32, got {}",
                other.dtype()
            ))),
        }
    }

    /// Try to extract the inner `Array<bool, IxDyn>`.
    pub fn try_into_bool(self) -> FerrayResult<Array<bool, IxDyn>> {
        match self {
            Self::Bool(a) => Ok(a),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected bool, got {}",
                other.dtype()
            ))),
        }
    }

    /// Cast this array to a different element dtype at the requested safety level.
    ///
    /// Mirrors `NumPy`'s `arr.astype(dtype, casting=...)`. The conversion routes
    /// through [`crate::dtype::unsafe_cast::CastTo`] for the underlying typed
    /// arrays, so it supports the same set of element pairs (every primitive
    /// numeric, bool, and Complex<f32>/Complex<f64> in any combination).
    ///
    /// `f16` / `bf16` are not yet supported in this dispatch and will return
    /// `FerrayError::InvalidDtype` — track via the umbrella casting issue.
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if:
    /// - The cast is not permitted at the chosen `casting` level, or
    /// - either source or target dtype is `f16`/`bf16` (not yet wired).
    pub fn astype(&self, target: DType, casting: CastKind) -> FerrayResult<Self> {
        // Reject f16/bf16 — see method docs.
        #[cfg(feature = "f16")]
        if matches!(self, Self::F16(_)) || target == DType::F16 {
            return Err(FerrayError::invalid_dtype(
                "DynArray::astype does not yet support f16",
            ));
        }
        #[cfg(feature = "bf16")]
        if matches!(self, Self::BF16(_)) || target == DType::BF16 {
            return Err(FerrayError::invalid_dtype(
                "DynArray::astype does not yet support bf16",
            ));
        }
        // I256 is accepted as a storage type (from promotion) but
        // generic cast-through-`CastTo` is not yet wired for it.
        // Reject both source and target explicitly — users who want
        // I256 arrays should construct them directly rather than via
        // astype (#562).
        if matches!(self, Self::I256(_)) || target == DType::I256 {
            return Err(FerrayError::invalid_dtype(
                "DynArray::astype does not yet support I256 — construct I256 arrays directly",
            ));
        }
        // datetime64 / timedelta64 don't go through the generic CastTo
        // machinery (they're not in numeric promotion's hierarchy). Casts
        // between time and numeric dtypes are intentionally not supported
        // by NumPy either — datetime arithmetic uses dedicated kernels.
        if matches!(self, Self::DateTime64(_, _) | Self::Timedelta64(_, _))
            || matches!(target, DType::DateTime64(_) | DType::Timedelta64(_))
        {
            return Err(FerrayError::invalid_dtype(format!(
                "DynArray::astype: cast involving {target} not supported \
                 — datetime/timedelta dtypes use dedicated arithmetic, not generic casts"
            )));
        }

        // Inner macro: dispatch on the *source* variant. The target type `$U`
        // is fixed by the outer match below.
        macro_rules! cast_into {
            ($U:ty) => {
                match self {
                    Self::Bool(a) => a.cast::<$U>(casting),
                    Self::U8(a) => a.cast::<$U>(casting),
                    Self::U16(a) => a.cast::<$U>(casting),
                    Self::U32(a) => a.cast::<$U>(casting),
                    Self::U64(a) => a.cast::<$U>(casting),
                    Self::U128(a) => a.cast::<$U>(casting),
                    Self::I8(a) => a.cast::<$U>(casting),
                    Self::I16(a) => a.cast::<$U>(casting),
                    Self::I32(a) => a.cast::<$U>(casting),
                    Self::I64(a) => a.cast::<$U>(casting),
                    Self::I128(a) => a.cast::<$U>(casting),
                    Self::F32(a) => a.cast::<$U>(casting),
                    Self::F64(a) => a.cast::<$U>(casting),
                    Self::Complex32(a) => a.cast::<$U>(casting),
                    Self::Complex64(a) => a.cast::<$U>(casting),
                    Self::I256(_) => unreachable!("I256 source rejected above"),
                    #[cfg(feature = "f16")]
                    Self::F16(_) => unreachable!("f16 source rejected above"),
                    #[cfg(feature = "bf16")]
                    Self::BF16(_) => unreachable!("bf16 source rejected above"),
                    Self::DateTime64(_, _) | Self::Timedelta64(_, _) => {
                        unreachable!("time-dtype source rejected above")
                    }
                }
            };
        }

        Ok(match target {
            DType::Bool => Self::Bool(cast_into!(bool)?),
            DType::U8 => Self::U8(cast_into!(u8)?),
            DType::U16 => Self::U16(cast_into!(u16)?),
            DType::U32 => Self::U32(cast_into!(u32)?),
            DType::U64 => Self::U64(cast_into!(u64)?),
            DType::U128 => Self::U128(cast_into!(u128)?),
            DType::I8 => Self::I8(cast_into!(i8)?),
            DType::I16 => Self::I16(cast_into!(i16)?),
            DType::I32 => Self::I32(cast_into!(i32)?),
            DType::I64 => Self::I64(cast_into!(i64)?),
            DType::I128 => Self::I128(cast_into!(i128)?),
            DType::F32 => Self::F32(cast_into!(f32)?),
            DType::F64 => Self::F64(cast_into!(f64)?),
            DType::Complex32 => Self::Complex32(cast_into!(Complex<f32>)?),
            DType::Complex64 => Self::Complex64(cast_into!(Complex<f64>)?),
            DType::I256 => unreachable!("I256 target rejected above"),
            #[cfg(feature = "f16")]
            DType::F16 => unreachable!("f16 target rejected above"),
            #[cfg(feature = "bf16")]
            DType::BF16 => unreachable!("bf16 target rejected above"),
            DType::DateTime64(_) | DType::Timedelta64(_) => {
                unreachable!("time-dtype target rejected above")
            }
            // #342: DynArray currently has no Struct variant; casting
            // a numeric array to a structured dtype is rejected.
            // Future work: extend DynArray with a Struct variant
            // backed by a raw byte buffer.
            DType::Struct(_) => {
                return Err(FerrayError::invalid_dtype(
                    "DynArray cannot represent structured dtype targets yet",
                ));
            }
        })
    }

    /// Create a `DynArray` of zeros with the given dtype and shape.
    pub fn zeros(dtype: DType, shape: &[usize]) -> FerrayResult<Self> {
        let dim = IxDyn::new(shape);
        Ok(match dtype {
            DType::Bool => Self::Bool(Array::zeros(dim)?),
            DType::U8 => Self::U8(Array::zeros(dim)?),
            DType::U16 => Self::U16(Array::zeros(dim)?),
            DType::U32 => Self::U32(Array::zeros(dim)?),
            DType::U64 => Self::U64(Array::zeros(dim)?),
            DType::U128 => Self::U128(Array::zeros(dim)?),
            DType::I8 => Self::I8(Array::zeros(dim)?),
            DType::I16 => Self::I16(Array::zeros(dim)?),
            DType::I32 => Self::I32(Array::zeros(dim)?),
            DType::I64 => Self::I64(Array::zeros(dim)?),
            DType::I128 => Self::I128(Array::zeros(dim)?),
            DType::I256 => Self::I256(Array::zeros(dim)?),
            DType::F32 => Self::F32(Array::zeros(dim)?),
            DType::F64 => Self::F64(Array::zeros(dim)?),
            DType::Complex32 => Self::Complex32(Array::zeros(dim)?),
            DType::Complex64 => Self::Complex64(Array::zeros(dim)?),
            #[cfg(feature = "f16")]
            DType::F16 => Self::F16(Array::zeros(dim)?),
            #[cfg(feature = "bf16")]
            DType::BF16 => Self::BF16(Array::zeros(dim)?),
            // datetime64 / timedelta64 carry a TimeUnit alongside the
            // typed Array. The element value is the i64 zero (the Unix
            // epoch for datetime, a no-op duration for timedelta).
            DType::DateTime64(unit) => Self::DateTime64(Array::zeros(dim)?, unit),
            DType::Timedelta64(unit) => Self::Timedelta64(Array::zeros(dim)?, unit),
            // #342: structured dtype zero-construction not yet wired
            // into DynArray (would need a Struct variant + per-field
            // zero initialization).
            DType::Struct(_) => {
                return Err(FerrayError::invalid_dtype(
                    "DynArray::zeros doesn't support structured dtypes yet",
                ));
            }
        })
    }

    /// Construct a [`DynArray::DateTime64`] from a typed array plus its unit.
    #[must_use]
    pub fn from_datetime64(arr: Array<DateTime64, IxDyn>, unit: TimeUnit) -> Self {
        Self::DateTime64(arr, unit)
    }

    /// Construct a [`DynArray::Timedelta64`] from a typed array plus its unit.
    #[must_use]
    pub fn from_timedelta64(arr: Array<Timedelta64, IxDyn>, unit: TimeUnit) -> Self {
        Self::Timedelta64(arr, unit)
    }

    /// Try to extract the inner `Array<DateTime64, IxDyn>` along with the
    /// stored [`TimeUnit`].
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if the dtype is not `datetime64`.
    pub fn try_into_datetime64(self) -> FerrayResult<(Array<DateTime64, IxDyn>, TimeUnit)> {
        match self {
            Self::DateTime64(a, u) => Ok((a, u)),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected datetime64, got {}",
                other.dtype()
            ))),
        }
    }

    /// Try to extract the inner `Array<Timedelta64, IxDyn>` along with the
    /// stored [`TimeUnit`].
    ///
    /// # Errors
    /// Returns `FerrayError::InvalidDtype` if the dtype is not `timedelta64`.
    pub fn try_into_timedelta64(self) -> FerrayResult<(Array<Timedelta64, IxDyn>, TimeUnit)> {
        match self {
            Self::Timedelta64(a, u) => Ok((a, u)),
            other => Err(FerrayError::invalid_dtype(format!(
                "expected timedelta64, got {}",
                other.dtype()
            ))),
        }
    }
}

impl std::fmt::Display for DynArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        dispatch!(self, a => write!(f, "{a}"))
    }
}

// Conversion from typed arrays to DynArray
macro_rules! impl_from_array_dyn {
    ($ty:ty, $variant:ident) => {
        impl From<Array<$ty, IxDyn>> for DynArray {
            fn from(a: Array<$ty, IxDyn>) -> Self {
                Self::$variant(a)
            }
        }
    };
}

impl_from_array_dyn!(bool, Bool);
impl_from_array_dyn!(u8, U8);
impl_from_array_dyn!(u16, U16);
impl_from_array_dyn!(u32, U32);
impl_from_array_dyn!(u64, U64);
impl_from_array_dyn!(u128, U128);
impl_from_array_dyn!(i8, I8);
impl_from_array_dyn!(i16, I16);
impl_from_array_dyn!(i32, I32);
impl_from_array_dyn!(i64, I64);
impl_from_array_dyn!(i128, I128);
impl_from_array_dyn!(I256, I256);
impl_from_array_dyn!(f32, F32);
impl_from_array_dyn!(f64, F64);
impl_from_array_dyn!(Complex<f32>, Complex32);
impl_from_array_dyn!(Complex<f64>, Complex64);
#[cfg(feature = "f16")]
impl_from_array_dyn!(half::f16, F16);
#[cfg(feature = "bf16")]
impl_from_array_dyn!(half::bf16, BF16);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynarray_zeros_f64() {
        let da = DynArray::zeros(DType::F64, &[2, 3]).unwrap();
        assert_eq!(da.dtype(), DType::F64);
        assert_eq!(da.shape(), &[2, 3]);
        assert_eq!(da.ndim(), 2);
        assert_eq!(da.size(), 6);
        assert_eq!(da.itemsize(), 8);
        assert_eq!(da.nbytes(), 48);
    }

    #[test]
    fn dynarray_zeros_i32() {
        let da = DynArray::zeros(DType::I32, &[4]).unwrap();
        assert_eq!(da.dtype(), DType::I32);
        assert_eq!(da.shape(), &[4]);
    }

    #[test]
    fn dynarray_try_into_f64() {
        let da = DynArray::zeros(DType::F64, &[3]).unwrap();
        let arr = da.try_into_f64().unwrap();
        assert_eq!(arr.shape(), &[3]);
    }

    #[test]
    fn dynarray_try_into_wrong_type() {
        let da = DynArray::zeros(DType::I32, &[3]).unwrap();
        assert!(da.try_into_f64().is_err());
    }

    // ----- DynArray::astype tests (issue #361) -----

    #[test]
    fn dynarray_astype_f64_to_i32_unsafe() {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[3]), vec![1.5, 2.7, -3.9]).unwrap();
        let dy = DynArray::F64(arr);
        let casted = dy.astype(DType::I32, CastKind::Unsafe).unwrap();
        assert_eq!(casted.dtype(), DType::I32);
        match casted {
            DynArray::I32(a) => assert_eq!(a.as_slice().unwrap(), &[1, 2, -3]),
            _ => panic!("expected I32"),
        }
    }

    #[test]
    fn dynarray_astype_safe_widening() {
        let arr = Array::<i32, IxDyn>::from_vec(IxDyn::new(&[3]), vec![10, 20, 30]).unwrap();
        let dy = DynArray::I32(arr);
        let casted = dy.astype(DType::I64, CastKind::Safe).unwrap();
        assert_eq!(casted.dtype(), DType::I64);
        match casted {
            DynArray::I64(a) => assert_eq!(a.as_slice().unwrap(), &[10i64, 20, 30]),
            _ => panic!("expected I64"),
        }
    }

    #[test]
    fn dynarray_astype_safe_narrowing_errors() {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let dy = DynArray::F64(arr);
        assert!(dy.astype(DType::F32, CastKind::Safe).is_err());
    }

    #[test]
    fn dynarray_astype_complex_to_real_unsafe() {
        let arr = Array::<Complex<f64>, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![Complex::new(1.5, 9.0), Complex::new(2.5, -1.0)],
        )
        .unwrap();
        let dy = DynArray::Complex64(arr);
        let casted = dy.astype(DType::F64, CastKind::Unsafe).unwrap();
        match casted {
            DynArray::F64(a) => assert_eq!(a.as_slice().unwrap(), &[1.5, 2.5]),
            _ => panic!("expected F64"),
        }
    }

    #[test]
    fn dynarray_astype_bool_to_u8_safe() {
        let arr =
            Array::<bool, IxDyn>::from_vec(IxDyn::new(&[3]), vec![true, false, true]).unwrap();
        let dy = DynArray::Bool(arr);
        let casted = dy.astype(DType::U8, CastKind::Safe).unwrap();
        match casted {
            DynArray::U8(a) => assert_eq!(a.as_slice().unwrap(), &[1u8, 0, 1]),
            _ => panic!("expected U8"),
        }
    }

    #[test]
    fn dynarray_astype_no_kind_requires_identity() {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(&[2]), vec![1.0, 2.0]).unwrap();
        let dy = DynArray::F64(arr);
        assert!(dy.astype(DType::F64, CastKind::No).is_ok());
        assert!(dy.astype(DType::F32, CastKind::No).is_err());
    }

    #[test]
    fn dynarray_from_typed() {
        let arr = Array::<f64, IxDyn>::zeros(IxDyn::new(&[2, 2])).unwrap();
        let da: DynArray = arr.into();
        assert_eq!(da.dtype(), DType::F64);
    }

    #[test]
    fn dynarray_display() {
        let da = DynArray::zeros(DType::I32, &[3]).unwrap();
        let s = format!("{da}");
        assert!(s.contains("[0, 0, 0]"));
    }

    #[test]
    fn dynarray_is_empty() {
        let da = DynArray::zeros(DType::F32, &[0]).unwrap();
        assert!(da.is_empty());
    }

    // ----- f16 / bf16 DynArray coverage (#139) -----

    #[cfg(feature = "f16")]
    #[test]
    fn dynarray_f16_zeros_shape_and_dtype() {
        let da = DynArray::zeros(DType::F16, &[2, 3]).unwrap();
        assert_eq!(da.dtype(), DType::F16);
        assert_eq!(da.shape(), &[2, 3]);
        assert_eq!(da.size(), 6);
        assert_eq!(da.itemsize(), 2);
        assert_eq!(da.nbytes(), 12);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn dynarray_f16_from_typed_roundtrips() {
        use half::f16;
        let raw = [f16::from_f32(1.0), f16::from_f32(2.5), f16::from_f32(-3.0)];
        let arr = Array::<f16, IxDyn>::from_vec(IxDyn::new(&[3]), raw.to_vec()).unwrap();
        let da: DynArray = arr.into();
        assert_eq!(da.dtype(), DType::F16);
        assert_eq!(da.shape(), &[3]);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn dynarray_bf16_zeros_shape_and_dtype() {
        let da = DynArray::zeros(DType::BF16, &[4]).unwrap();
        assert_eq!(da.dtype(), DType::BF16);
        assert_eq!(da.shape(), &[4]);
        assert_eq!(da.itemsize(), 2);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn dynarray_bf16_from_typed_roundtrips() {
        use half::bf16;
        let raw = [bf16::from_f32(1.0), bf16::from_f32(2.0)];
        let arr = Array::<bf16, IxDyn>::from_vec(IxDyn::new(&[2]), raw.to_vec()).unwrap();
        let da: DynArray = arr.into();
        assert_eq!(da.dtype(), DType::BF16);
    }
}
