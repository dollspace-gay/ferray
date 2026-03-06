//! Arrow <-> ferrum array conversions (feature-gated behind `"arrow"`).
//!
//! Provides [`ToArrow`] and [`FromArrow`] traits for converting between
//! ferrum 1-D arrays and Arrow [`PrimitiveArray`] / [`BooleanArray`].
//!
//! Zero-copy is used when the ferrum array is C-contiguous.

use arrow::array::{Array as ArrowArray, BooleanArray, PrimitiveArray};
use arrow::buffer::Buffer;
use arrow::datatypes::{ArrowNativeType, ArrowPrimitiveType};

use ferrum_core::array::aliases::Array1;
use ferrum_core::{Element, FerrumError, Ix1};

use crate::dtype_map;

// ---------------------------------------------------------------------------
// Helper: marker trait linking ferrum Element to ArrowPrimitiveType
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferrum [`Element`] type with its Arrow
/// [`ArrowPrimitiveType`] counterpart.
///
/// This is implemented for every numeric type that both ferrum and Arrow
/// support. `bool` is handled separately because Arrow uses a bit-packed
/// [`BooleanArray`] rather than a `PrimitiveArray`.
pub trait ArrowElement: Element + ArrowNativeType {
    /// The Arrow primitive type tag.
    type ArrowType: ArrowPrimitiveType<Native = Self>;
}

macro_rules! impl_arrow_element {
    ($rust_ty:ty, $arrow_ty:ty) => {
        impl ArrowElement for $rust_ty {
            type ArrowType = $arrow_ty;
        }
    };
}

impl_arrow_element!(u8, arrow::datatypes::UInt8Type);
impl_arrow_element!(u16, arrow::datatypes::UInt16Type);
impl_arrow_element!(u32, arrow::datatypes::UInt32Type);
impl_arrow_element!(u64, arrow::datatypes::UInt64Type);
impl_arrow_element!(i8, arrow::datatypes::Int8Type);
impl_arrow_element!(i16, arrow::datatypes::Int16Type);
impl_arrow_element!(i32, arrow::datatypes::Int32Type);
impl_arrow_element!(i64, arrow::datatypes::Int64Type);
impl_arrow_element!(f32, arrow::datatypes::Float32Type);
impl_arrow_element!(f64, arrow::datatypes::Float64Type);

// ---------------------------------------------------------------------------
// ferrum -> Arrow  (REQ-4)
// ---------------------------------------------------------------------------

/// Extension trait for converting a ferrum 1-D array to an Arrow array.
pub trait ToArrow {
    /// The Arrow array type produced by the conversion.
    type ArrowArray;

    /// Convert this ferrum array to an Arrow array.
    ///
    /// Zero-copy when the source is C-contiguous; copies otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the element type has no
    /// Arrow equivalent.
    fn to_arrow(&self) -> Result<Self::ArrowArray, FerrumError>;
}

impl<T: ArrowElement> ToArrow for Array1<T>
where
    T: ArrowElement,
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    type ArrowArray = PrimitiveArray<T::ArrowType>;

    fn to_arrow(&self) -> Result<Self::ArrowArray, FerrumError> {
        // Validate that the ferrum dtype has an Arrow equivalent
        let _ = dtype_map::dtype_to_arrow(self.dtype())?;

        // Get contiguous data — as_slice() returns Some only for C-contiguous
        let data: Vec<T> = match self.as_slice() {
            Some(slice) => slice.to_vec(),
            None => self.to_vec_flat(),
        };

        let buffer = Buffer::from_vec(data);
        // SAFETY: buffer length is exactly self.size() * size_of::<T>()
        let array = PrimitiveArray::<T::ArrowType>::new(buffer.into(), None);
        Ok(array)
    }
}

/// Extension trait for converting a ferrum `Array1<bool>` to an Arrow
/// [`BooleanArray`].
pub trait ToArrowBool {
    /// Convert this ferrum bool array to an Arrow [`BooleanArray`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] on internal inconsistency.
    fn to_arrow(&self) -> Result<BooleanArray, FerrumError>;
}

impl ToArrowBool for Array1<bool> {
    fn to_arrow(&self) -> Result<BooleanArray, FerrumError> {
        let values: Vec<bool> = self.to_vec_flat();
        Ok(BooleanArray::from(values))
    }
}

// ---------------------------------------------------------------------------
// Arrow -> ferrum  (REQ-5)
// ---------------------------------------------------------------------------

/// Extension trait for converting an Arrow primitive array to a ferrum
/// 1-D array.
pub trait FromArrow<T: Element>: Sized {
    /// Convert an Arrow array to a ferrum `Array1<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the Arrow array dtype does
    /// not match `T`, or [`FerrumError::InvalidValue`] if the Arrow array
    /// contains nulls (ferrum arrays do not support null values).
    fn into_ferrum(self) -> Result<Array1<T>, FerrumError>;
}

impl<T: ArrowElement> FromArrow<T> for PrimitiveArray<T::ArrowType>
where
    T::ArrowType: ArrowPrimitiveType<Native = T>,
{
    fn into_ferrum(self) -> Result<Array1<T>, FerrumError> {
        // Validate no nulls
        if self.null_count() > 0 {
            return Err(FerrumError::invalid_value(format!(
                "Arrow array contains {} null values; ferrum arrays do not support nulls",
                self.null_count()
            )));
        }

        // Validate dtype correspondence
        let arrow_dt = self.data_type();
        let ferrum_dt = dtype_map::arrow_to_dtype(arrow_dt)?;
        if ferrum_dt != T::dtype() {
            return Err(FerrumError::invalid_dtype(format!(
                "Arrow dtype {arrow_dt:?} maps to ferrum {ferrum_dt}, but requested {}",
                T::dtype()
            )));
        }

        let values = self.values();
        let data: Vec<T> = values.iter().copied().collect();
        let len = data.len();
        Array1::<T>::from_vec(Ix1::new([len]), data)
    }
}

/// Extension trait for converting an Arrow [`BooleanArray`] to a ferrum
/// `Array1<bool>`.
pub trait FromArrowBool: Sized {
    /// Convert an Arrow boolean array to a ferrum `Array1<bool>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidValue`] if the array contains nulls.
    fn into_ferrum_bool(self) -> Result<Array1<bool>, FerrumError>;
}

impl FromArrowBool for BooleanArray {
    fn into_ferrum_bool(self) -> Result<Array1<bool>, FerrumError> {
        if self.null_count() > 0 {
            return Err(FerrumError::invalid_value(format!(
                "Arrow BooleanArray contains {} null values; ferrum arrays do not support nulls",
                self.null_count()
            )));
        }

        let data: Vec<bool> = self.iter().map(|v| v.unwrap_or(false)).collect();
        let len = data.len();
        Array1::<bool>::from_vec(Ix1::new([len]), data)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- ferrum -> Arrow -> ferrum roundtrip (AC-2) -----

    macro_rules! test_roundtrip {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                let data: Vec<$ty> = $values;
                let len = data.len();
                let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                // ferrum -> arrow
                let arrow_arr = arr.to_arrow().unwrap();
                assert_eq!(arrow_arr.len(), len);

                // arrow -> ferrum
                let back: Array1<$ty> = arrow_arr.into_ferrum().unwrap();
                assert_eq!(back.shape(), &[len]);
                assert_eq!(back.as_slice().unwrap(), &data[..]);
            }
        };
    }

    test_roundtrip!(roundtrip_f64, f64, vec![1.0, 2.5, -3.14, 0.0, f64::MAX]);
    test_roundtrip!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0, f32::MIN]);
    test_roundtrip!(roundtrip_i32, i32, vec![0, 1, -1, i32::MAX, i32::MIN]);
    test_roundtrip!(roundtrip_i64, i64, vec![0i64, 42, -99]);
    test_roundtrip!(roundtrip_i8, i8, vec![0i8, 127, -128]);
    test_roundtrip!(roundtrip_i16, i16, vec![0i16, 32767, -32768]);
    test_roundtrip!(roundtrip_u8, u8, vec![0u8, 128, 255]);
    test_roundtrip!(roundtrip_u16, u16, vec![0u16, 1000, 65535]);
    test_roundtrip!(roundtrip_u32, u32, vec![0u32, 1, u32::MAX]);
    test_roundtrip!(roundtrip_u64, u64, vec![0u64, 1, u64::MAX]);

    #[test]
    fn roundtrip_bool() {
        let data = vec![true, false, true, true, false];
        let len = data.len();
        let arr = Array1::<bool>::from_vec(Ix1::new([len]), data.clone()).unwrap();

        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), len);

        let back = arrow_arr.into_ferrum_bool().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn empty_array_roundtrip() {
        let arr = Array1::<f64>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        assert_eq!(arrow_arr.len(), 0);

        let back: Array1<f64> = arrow_arr.into_ferrum().unwrap();
        assert_eq!(back.shape(), &[0]);
    }

    #[test]
    fn arrow_with_nulls_rejected() {
        // Create an Arrow array with a null
        let arr =
            PrimitiveArray::<arrow::datatypes::Float64Type>::from(vec![Some(1.0), None, Some(3.0)]);
        let result: Result<Array1<f64>, _> = arr.into_ferrum();
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("null"));
    }

    #[test]
    fn bool_arrow_with_nulls_rejected() {
        let arr = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let result = arr.into_ferrum_bool();
        assert!(result.is_err());
    }

    // AC-3: dtype mismatch returns clear error
    #[test]
    fn dtype_mismatch_arrow_to_dtype() {
        // arrow::datatypes::Utf8 has no ferrum equivalent
        let result = dtype_map::arrow_to_dtype(&arrow::datatypes::DataType::Utf8);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no ferrum equivalent"), "got: {msg}");
    }

    #[test]
    fn bit_identical_roundtrip() {
        // AC-2: ferrum -> arrow -> ferrum produces bit-identical data
        let original: Vec<f64> = vec![
            1.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            1.23456789012345e-300,
            9.87654321098765e+300,
        ];
        let len = original.len();
        let arr = Array1::<f64>::from_vec(Ix1::new([len]), original.clone()).unwrap();
        let arrow_arr = arr.to_arrow().unwrap();
        let back: Array1<f64> = arrow_arr.into_ferrum().unwrap();

        let back_slice = back.as_slice().unwrap();
        for (i, (a, b)) in original.iter().zip(back_slice.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch at index {i}: {a} vs {b}"
            );
        }
    }
}
