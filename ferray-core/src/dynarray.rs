// ferray-core: DynArray — runtime-typed array enum (REQ-30)

use num_complex::Complex;

use crate::array::owned::Array;
use crate::dimension::IxDyn;
use crate::dtype::DType;
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
}

impl DynArray {
    /// The runtime dtype of the elements in this array.
    pub fn dtype(&self) -> DType {
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
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::Complex32(_) => DType::Complex32,
            Self::Complex64(_) => DType::Complex64,
            #[cfg(feature = "f16")]
            Self::F16(_) => DType::F16,
        }
    }

    /// Shape as a slice.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Bool(a) => a.shape(),
            Self::U8(a) => a.shape(),
            Self::U16(a) => a.shape(),
            Self::U32(a) => a.shape(),
            Self::U64(a) => a.shape(),
            Self::U128(a) => a.shape(),
            Self::I8(a) => a.shape(),
            Self::I16(a) => a.shape(),
            Self::I32(a) => a.shape(),
            Self::I64(a) => a.shape(),
            Self::I128(a) => a.shape(),
            Self::F32(a) => a.shape(),
            Self::F64(a) => a.shape(),
            Self::Complex32(a) => a.shape(),
            Self::Complex64(a) => a.shape(),
            #[cfg(feature = "f16")]
            Self::F16(a) => a.shape(),
        }
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Whether the array has zero elements.
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Size in bytes of one element.
    pub fn itemsize(&self) -> usize {
        self.dtype().size_of()
    }

    /// Total size in bytes.
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
            DType::F32 => Self::F32(Array::zeros(dim)?),
            DType::F64 => Self::F64(Array::zeros(dim)?),
            DType::Complex32 => Self::Complex32(Array::zeros(dim)?),
            DType::Complex64 => Self::Complex64(Array::zeros(dim)?),
            #[cfg(feature = "f16")]
            DType::F16 => Self::F16(Array::zeros(dim)?),
        })
    }
}

impl std::fmt::Display for DynArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool(a) => write!(f, "{a}"),
            Self::U8(a) => write!(f, "{a}"),
            Self::U16(a) => write!(f, "{a}"),
            Self::U32(a) => write!(f, "{a}"),
            Self::U64(a) => write!(f, "{a}"),
            Self::U128(a) => write!(f, "{a}"),
            Self::I8(a) => write!(f, "{a}"),
            Self::I16(a) => write!(f, "{a}"),
            Self::I32(a) => write!(f, "{a}"),
            Self::I64(a) => write!(f, "{a}"),
            Self::I128(a) => write!(f, "{a}"),
            Self::F32(a) => write!(f, "{a}"),
            Self::F64(a) => write!(f, "{a}"),
            Self::Complex32(a) => write!(f, "{a}"),
            Self::Complex64(a) => write!(f, "{a}"),
            #[cfg(feature = "f16")]
            Self::F16(a) => write!(f, "{a}"),
        }
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
impl_from_array_dyn!(f32, F32);
impl_from_array_dyn!(f64, F64);
impl_from_array_dyn!(Complex<f32>, Complex32);
impl_from_array_dyn!(Complex<f64>, Complex64);
#[cfg(feature = "f16")]
impl_from_array_dyn!(half::f16, F16);

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
}
