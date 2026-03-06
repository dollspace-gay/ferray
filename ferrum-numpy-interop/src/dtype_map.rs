//! Mapping between ferrum [`DType`], Arrow [`DataType`], and NumPy dtype codes.
//!
//! This module provides bidirectional conversion functions so that every
//! interop path (NumPy, Arrow, Polars) shares a single source of truth for
//! type correspondence.

#[cfg(any(feature = "arrow", feature = "polars"))]
use ferrum_core::DType;
#[cfg(any(feature = "arrow", feature = "polars"))]
use ferrum_core::FerrumError;

// ---------------------------------------------------------------------------
// Arrow DataType <-> DType
// ---------------------------------------------------------------------------

/// Convert a ferrum [`DType`] to the corresponding Arrow [`DataType`].
///
/// # Errors
///
/// Returns [`FerrumError::InvalidDtype`] if the ferrum dtype has no Arrow
/// equivalent (e.g. `Complex32`, `Complex64`, `U128`, `I128`).
#[cfg(feature = "arrow")]
pub fn dtype_to_arrow(dt: DType) -> Result<arrow::datatypes::DataType, FerrumError> {
    use arrow::datatypes::DataType as AD;
    match dt {
        DType::Bool => Ok(AD::Boolean),
        DType::U8 => Ok(AD::UInt8),
        DType::U16 => Ok(AD::UInt16),
        DType::U32 => Ok(AD::UInt32),
        DType::U64 => Ok(AD::UInt64),
        DType::I8 => Ok(AD::Int8),
        DType::I16 => Ok(AD::Int16),
        DType::I32 => Ok(AD::Int32),
        DType::I64 => Ok(AD::Int64),
        DType::F32 => Ok(AD::Float32),
        DType::F64 => Ok(AD::Float64),
        other => Err(FerrumError::invalid_dtype(format!(
            "ferrum dtype {other} has no Arrow equivalent"
        ))),
    }
}

/// Convert an Arrow [`DataType`] to the corresponding ferrum [`DType`].
///
/// # Errors
///
/// Returns [`FerrumError::InvalidDtype`] for Arrow types that ferrum does
/// not support (e.g. `Utf8`, `Timestamp`, `Struct`, etc.).
#[cfg(feature = "arrow")]
pub fn arrow_to_dtype(ad: &arrow::datatypes::DataType) -> Result<DType, FerrumError> {
    use arrow::datatypes::DataType as AD;
    match ad {
        AD::Boolean => Ok(DType::Bool),
        AD::UInt8 => Ok(DType::U8),
        AD::UInt16 => Ok(DType::U16),
        AD::UInt32 => Ok(DType::U32),
        AD::UInt64 => Ok(DType::U64),
        AD::Int8 => Ok(DType::I8),
        AD::Int16 => Ok(DType::I16),
        AD::Int32 => Ok(DType::I32),
        AD::Int64 => Ok(DType::I64),
        AD::Float32 => Ok(DType::F32),
        AD::Float64 => Ok(DType::F64),
        other => Err(FerrumError::invalid_dtype(format!(
            "Arrow DataType {other:?} has no ferrum equivalent"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Polars DataType <-> DType
// ---------------------------------------------------------------------------

/// Convert a ferrum [`DType`] to the corresponding Polars [`DataType`].
///
/// # Errors
///
/// Returns [`FerrumError::InvalidDtype`] if the ferrum dtype has no Polars
/// equivalent (e.g. `Complex32`, `Complex64`, `U128`, `I128`, `Bool`-as-bitfield).
#[cfg(feature = "polars")]
pub fn dtype_to_polars(dt: DType) -> Result<polars::prelude::DataType, FerrumError> {
    use polars::prelude::DataType as PD;
    match dt {
        DType::Bool => Ok(PD::Boolean),
        DType::U8 => Ok(PD::UInt8),
        DType::U16 => Ok(PD::UInt16),
        DType::U32 => Ok(PD::UInt32),
        DType::U64 => Ok(PD::UInt64),
        DType::I8 => Ok(PD::Int8),
        DType::I16 => Ok(PD::Int16),
        DType::I32 => Ok(PD::Int32),
        DType::I64 => Ok(PD::Int64),
        DType::F32 => Ok(PD::Float32),
        DType::F64 => Ok(PD::Float64),
        other => Err(FerrumError::invalid_dtype(format!(
            "ferrum dtype {other} has no Polars equivalent"
        ))),
    }
}

/// Convert a Polars [`DataType`] to the corresponding ferrum [`DType`].
///
/// # Errors
///
/// Returns [`FerrumError::InvalidDtype`] for Polars types that ferrum does
/// not support (e.g. `String`, `Date`, `Datetime`, etc.).
#[cfg(feature = "polars")]
pub fn polars_to_dtype(pd: &polars::prelude::DataType) -> Result<DType, FerrumError> {
    use polars::prelude::DataType as PD;
    match pd {
        PD::Boolean => Ok(DType::Bool),
        PD::UInt8 => Ok(DType::U8),
        PD::UInt16 => Ok(DType::U16),
        PD::UInt32 => Ok(DType::U32),
        PD::UInt64 => Ok(DType::U64),
        PD::Int8 => Ok(DType::I8),
        PD::Int16 => Ok(DType::I16),
        PD::Int32 => Ok(DType::I32),
        PD::Int64 => Ok(DType::I64),
        PD::Float32 => Ok(DType::F32),
        PD::Float64 => Ok(DType::F64),
        other => Err(FerrumError::invalid_dtype(format!(
            "Polars DataType {other:?} has no ferrum equivalent"
        ))),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "arrow")]
    mod arrow_tests {
        use crate::dtype_map::{arrow_to_dtype, dtype_to_arrow};
        use arrow::datatypes::DataType as AD;
        use ferrum_core::DType;

        #[test]
        fn roundtrip_all_supported_dtypes() {
            let dtypes = [
                (DType::Bool, AD::Boolean),
                (DType::U8, AD::UInt8),
                (DType::U16, AD::UInt16),
                (DType::U32, AD::UInt32),
                (DType::U64, AD::UInt64),
                (DType::I8, AD::Int8),
                (DType::I16, AD::Int16),
                (DType::I32, AD::Int32),
                (DType::I64, AD::Int64),
                (DType::F32, AD::Float32),
                (DType::F64, AD::Float64),
            ];

            for (ferrum_dt, arrow_dt) in &dtypes {
                let converted = dtype_to_arrow(*ferrum_dt).unwrap();
                assert_eq!(&converted, arrow_dt);
                let back = arrow_to_dtype(&converted).unwrap();
                assert_eq!(back, *ferrum_dt);
            }
        }

        #[test]
        fn complex_has_no_arrow_equiv() {
            assert!(dtype_to_arrow(DType::Complex32).is_err());
            assert!(dtype_to_arrow(DType::Complex64).is_err());
        }

        #[test]
        fn unsupported_arrow_type() {
            assert!(arrow_to_dtype(&AD::Utf8).is_err());
        }
    }

    #[cfg(feature = "polars")]
    mod polars_tests {
        use crate::dtype_map::{dtype_to_polars, polars_to_dtype};
        use ferrum_core::DType;
        use polars::prelude::DataType as PD;

        #[test]
        fn roundtrip_all_supported_dtypes() {
            let dtypes = [
                (DType::Bool, PD::Boolean),
                (DType::U8, PD::UInt8),
                (DType::U16, PD::UInt16),
                (DType::U32, PD::UInt32),
                (DType::U64, PD::UInt64),
                (DType::I8, PD::Int8),
                (DType::I16, PD::Int16),
                (DType::I32, PD::Int32),
                (DType::I64, PD::Int64),
                (DType::F32, PD::Float32),
                (DType::F64, PD::Float64),
            ];

            for (ferrum_dt, polars_dt) in &dtypes {
                let converted = dtype_to_polars(*ferrum_dt).unwrap();
                assert_eq!(&converted, polars_dt);
                let back = polars_to_dtype(&converted).unwrap();
                assert_eq!(back, *ferrum_dt);
            }
        }

        #[test]
        fn complex_has_no_polars_equiv() {
            assert!(dtype_to_polars(DType::Complex32).is_err());
            assert!(dtype_to_polars(DType::Complex64).is_err());
        }

        #[test]
        fn unsupported_polars_type() {
            assert!(polars_to_dtype(&PD::String).is_err());
        }
    }
}
