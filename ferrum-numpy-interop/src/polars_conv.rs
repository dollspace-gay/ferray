//! Polars <-> ferrum array conversions (feature-gated behind `"polars"`).
//!
//! Provides [`ToPolars`] and [`FromPolars`] traits for converting between
//! ferrum 1-D arrays and Polars [`Series`].

use polars::prelude::*;

use ferrum_core::array::aliases::Array1;
use ferrum_core::{Element, FerrumError, Ix1};

use crate::dtype_map;

// ---------------------------------------------------------------------------
// Sealed marker: ferrum Element <-> Polars PolarsNumericType
// ---------------------------------------------------------------------------

/// Sealed marker associating a ferrum [`Element`] type with Polars
/// chunked-array extraction.
///
/// Implemented for all numeric types that both ferrum and Polars support.
/// `bool` is handled separately.
pub trait PolarsElement: Element {
    /// The Polars numeric type tag, whose `Native` associated type is `Self`.
    type PolarsType: PolarsNumericType<Native = Self>;

    /// Extract a reference to the typed [`ChunkedArray`] from a [`Series`].
    ///
    /// # Errors
    ///
    /// Returns an error if the Series dtype does not match.
    fn extract_ca(series: &Series) -> Result<&ChunkedArray<Self::PolarsType>, FerrumError>;
}

macro_rules! impl_polars_element {
    ($rust_ty:ty, $polars_ty:ty, $extractor:ident) => {
        impl PolarsElement for $rust_ty {
            type PolarsType = $polars_ty;

            fn extract_ca(series: &Series) -> Result<&ChunkedArray<Self::PolarsType>, FerrumError> {
                series.$extractor().map_err(|e| {
                    FerrumError::invalid_dtype(format!("Polars Series dtype mismatch: {e}"))
                })
            }
        }
    };
}

impl_polars_element!(u8, UInt8Type, u8);
impl_polars_element!(u16, UInt16Type, u16);
impl_polars_element!(u32, UInt32Type, u32);
impl_polars_element!(u64, UInt64Type, u64);
impl_polars_element!(i8, Int8Type, i8);
impl_polars_element!(i16, Int16Type, i16);
impl_polars_element!(i32, Int32Type, i32);
impl_polars_element!(i64, Int64Type, i64);
impl_polars_element!(f32, Float32Type, f32);
impl_polars_element!(f64, Float64Type, f64);

// ---------------------------------------------------------------------------
// ferrum -> Polars  (REQ-6)
// ---------------------------------------------------------------------------

/// Extension trait for converting a ferrum 1-D array to a Polars [`Series`].
pub trait ToPolars {
    /// Convert this ferrum array to a Polars [`Series`] with the given name.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the element type has no
    /// Polars equivalent.
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrumError>;
}

impl<T: PolarsElement> ToPolars for Array1<T>
where
    ChunkedArray<T::PolarsType>: IntoSeries,
{
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrumError> {
        // Validate dtype mapping exists
        let _ = dtype_map::dtype_to_polars(self.dtype())?;

        let data: Vec<T> = self.to_vec_flat();
        let ca = ChunkedArray::<T::PolarsType>::from_slice(name.into(), &data);
        Ok(ca.into_series())
    }
}

/// Extension trait for converting a ferrum `Array1<bool>` to a Polars [`Series`].
pub trait ToPolarsBool {
    /// Convert this ferrum bool array to a Polars [`Series`].
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] on internal inconsistency.
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrumError>;
}

impl ToPolarsBool for Array1<bool> {
    fn to_polars_series(&self, name: &str) -> Result<Series, FerrumError> {
        let data: Vec<bool> = self.to_vec_flat();
        let ca = BooleanChunked::new(name.into(), &data);
        Ok(ca.into_series())
    }
}

// ---------------------------------------------------------------------------
// Polars -> ferrum  (REQ-7)
// ---------------------------------------------------------------------------

/// Extension trait for converting a Polars [`Series`] to a ferrum 1-D array.
pub trait FromPolars<T: Element>: Sized {
    /// Convert a Polars Series to a ferrum `Array1<T>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the Series dtype does not
    /// match `T`, or [`FerrumError::InvalidValue`] if the Series contains
    /// null values.
    fn into_ferrum(self) -> Result<Array1<T>, FerrumError>;
}

impl<T: PolarsElement> FromPolars<T> for Series {
    fn into_ferrum(self) -> Result<Array1<T>, FerrumError> {
        // Validate dtype
        let polars_dt = self.dtype().clone();
        let ferrum_dt = dtype_map::polars_to_dtype(&polars_dt)?;
        if ferrum_dt != T::dtype() {
            return Err(FerrumError::invalid_dtype(format!(
                "Polars Series has dtype {polars_dt:?} (ferrum {ferrum_dt}), but requested {}",
                T::dtype()
            )));
        }

        // Check for nulls
        if self.null_count() > 0 {
            return Err(FerrumError::invalid_value(format!(
                "Polars Series contains {} null values; ferrum arrays do not support nulls",
                self.null_count()
            )));
        }

        let ca = T::extract_ca(&self)?;
        let data: Vec<T> = ca.into_no_null_iter().collect();
        let len = data.len();
        Array1::<T>::from_vec(Ix1::new([len]), data)
    }
}

/// Extension trait for converting a Polars [`Series`] with boolean dtype
/// to a ferrum `Array1<bool>`.
pub trait FromPolarsBool: Sized {
    /// Convert a Polars boolean Series to a ferrum `Array1<bool>`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrumError::InvalidDtype`] if the Series is not boolean,
    /// or [`FerrumError::InvalidValue`] if it contains nulls.
    fn into_ferrum_bool(self) -> Result<Array1<bool>, FerrumError>;
}

impl FromPolarsBool for Series {
    fn into_ferrum_bool(self) -> Result<Array1<bool>, FerrumError> {
        if *self.dtype() != DataType::Boolean {
            return Err(FerrumError::invalid_dtype(format!(
                "expected Boolean Series, got {:?}",
                self.dtype()
            )));
        }

        if self.null_count() > 0 {
            return Err(FerrumError::invalid_value(format!(
                "Polars Series contains {} null values; ferrum arrays do not support nulls",
                self.null_count()
            )));
        }

        let ca = self.bool().map_err(|e| {
            FerrumError::invalid_dtype(format!("failed to extract BooleanChunked: {e}"))
        })?;
        let data: Vec<bool> = ca.into_no_null_iter().collect();
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

    macro_rules! test_roundtrip {
        ($name:ident, $ty:ty, $values:expr) => {
            #[test]
            fn $name() {
                let data: Vec<$ty> = $values;
                let len = data.len();
                let arr = Array1::<$ty>::from_vec(Ix1::new([len]), data.clone()).unwrap();

                let series = arr.to_polars_series("test").unwrap();
                assert_eq!(series.len(), len);

                let back: Array1<$ty> = series.into_ferrum().unwrap();
                assert_eq!(back.shape(), &[len]);
                assert_eq!(back.as_slice().unwrap(), &data[..]);
            }
        };
    }

    test_roundtrip!(roundtrip_f64, f64, vec![1.0, 2.5, -3.14, 0.0]);
    test_roundtrip!(roundtrip_f32, f32, vec![1.0f32, -2.5, 0.0]);
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

        let series = arr.to_polars_series("flags").unwrap();
        assert_eq!(series.len(), len);
        assert_eq!(*series.dtype(), DataType::Boolean);

        let back = series.into_ferrum_bool().unwrap();
        assert_eq!(back.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn empty_series_roundtrip() {
        let arr = Array1::<f64>::from_vec(Ix1::new([0]), vec![]).unwrap();
        let series = arr.to_polars_series("empty").unwrap();
        assert_eq!(series.len(), 0);

        let back: Array1<f64> = series.into_ferrum().unwrap();
        assert_eq!(back.shape(), &[0]);
    }

    #[test]
    fn series_name_preserved() {
        let arr = Array1::<i32>::from_vec(Ix1::new([3]), vec![1, 2, 3]).unwrap();
        let series = arr.to_polars_series("my_column").unwrap();
        assert_eq!(series.name().as_str(), "my_column");
    }

    #[test]
    fn dtype_mismatch_rejected() {
        // Create an i32 series and try to extract as f64
        let series = Series::new("test".into(), &[1i32, 2, 3]);
        let result: Result<Array1<f64>, _> = series.into_ferrum();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("dtype") || msg.contains("mismatch"),
            "expected dtype error, got: {msg}"
        );
    }

    #[test]
    fn series_with_nulls_rejected() {
        let series = Series::new("test".into(), &[Some(1.0f64), None, Some(3.0)]);
        let result: Result<Array1<f64>, _> = series.into_ferrum();
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("null"), "expected null error, got: {msg}");
    }

    #[test]
    fn bool_series_with_nulls_rejected() {
        let series = Series::new("test".into(), &[Some(true), None, Some(false)]);
        let result = series.into_ferrum_bool();
        assert!(result.is_err());
    }

    #[test]
    fn non_bool_series_into_ferrum_bool_rejected() {
        let series = Series::new("test".into(), &[1i32, 2, 3]);
        let result = series.into_ferrum_bool();
        assert!(result.is_err());
    }

    #[test]
    fn bit_identical_f64_roundtrip() {
        let original: Vec<f64> = vec![
            1.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            1.23456789012345e-300,
            9.87654321098765e+300,
        ];
        let len = original.len();
        let arr = Array1::<f64>::from_vec(Ix1::new([len]), original.clone()).unwrap();
        let series = arr.to_polars_series("precise").unwrap();
        let back: Array1<f64> = series.into_ferrum().unwrap();

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
