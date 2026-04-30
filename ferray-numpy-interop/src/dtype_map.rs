//! Mapping between ferray [`DType`], Arrow [`DataType`], and `NumPy` dtype codes.
//!
//! This module provides bidirectional conversion functions so that every
//! interop path (`NumPy`, Arrow, Polars) shares a single source of truth for
//! type correspondence.

#[cfg(any(feature = "arrow", feature = "polars"))]
use ferray_core::DType;
#[cfg(any(feature = "arrow", feature = "polars"))]
use ferray_core::FerrayError;

// ---------------------------------------------------------------------------
// Arrow DataType <-> DType
// ---------------------------------------------------------------------------

/// Convert a ferray [`DType`] to the corresponding Arrow [`DataType`].
///
/// # Errors
///
/// Returns [`FerrayError::InvalidDtype`] if the ferray dtype has no Arrow
/// equivalent (e.g. `Complex32`, `Complex64`, `U128`, `I128`).
#[cfg(feature = "arrow")]
pub fn dtype_to_arrow(dt: DType) -> Result<arrow::datatypes::DataType, FerrayError> {
    use arrow::datatypes::{DataType as AD, TimeUnit as ATU};
    use ferray_core::dtype::TimeUnit;

    fn to_arrow_time_unit(u: TimeUnit) -> Result<ATU, FerrayError> {
        match u {
            TimeUnit::Ns => Ok(ATU::Nanosecond),
            TimeUnit::Us => Ok(ATU::Microsecond),
            TimeUnit::Ms => Ok(ATU::Millisecond),
            TimeUnit::S => Ok(ATU::Second),
            // Arrow's TimeUnit only has Ns/Us/Ms/S — minute/hour/day
            // datetime64 units have no direct Arrow correspondence.
            other => Err(FerrayError::invalid_dtype(format!(
                "Arrow has no time unit equivalent for ferray TimeUnit::{other:?}"
            ))),
        }
    }

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
        // Arrow's Float16 covers IEEE 754 binary16 — same encoding as
        // ferray's `DType::F16`. bfloat16 has no Arrow primitive (Arrow
        // doesn't define bf16 in the canonical schema), so we surface
        // it as InvalidDtype with a hint.
        #[cfg(feature = "f16")]
        DType::F16 => Ok(AD::Float16),
        #[cfg(feature = "bf16")]
        DType::BF16 => Err(FerrayError::invalid_dtype(
            "Arrow has no native bfloat16 type — pass through as f32 or use a struct(real, imag)-style workaround",
        )),
        // datetime64 → Arrow Timestamp(unit, None). Arrow Timestamp's
        // Option<Tz> is set to None since ferray doesn't track timezones
        // (NumPy datetime64 is also TZ-naive).
        DType::DateTime64(u) => Ok(AD::Timestamp(to_arrow_time_unit(u)?, None)),
        // timedelta64 → Arrow Duration(unit).
        DType::Timedelta64(u) => Ok(AD::Duration(to_arrow_time_unit(u)?)),
        other => Err(FerrayError::invalid_dtype(format!(
            "ferray dtype {other} has no Arrow equivalent"
        ))),
    }
}

/// Convert an Arrow [`DataType`] to the corresponding ferray [`DType`].
///
/// # Errors
///
/// Returns [`FerrayError::InvalidDtype`] for Arrow types that ferray does
/// not support (e.g. `Utf8`, `Timestamp`, `Struct`, etc.).
#[cfg(feature = "arrow")]
pub fn arrow_to_dtype(ad: &arrow::datatypes::DataType) -> Result<DType, FerrayError> {
    use arrow::datatypes::{DataType as AD, TimeUnit as ATU};
    use ferray_core::dtype::TimeUnit;

    fn from_arrow_time_unit(u: &ATU) -> TimeUnit {
        match u {
            ATU::Nanosecond => TimeUnit::Ns,
            ATU::Microsecond => TimeUnit::Us,
            ATU::Millisecond => TimeUnit::Ms,
            ATU::Second => TimeUnit::S,
        }
    }

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
        #[cfg(feature = "f16")]
        AD::Float16 => Ok(DType::F16),
        // Arrow Timestamp -> datetime64. Timezone-tagged timestamps are
        // mapped to the same TZ-naive ferray dtype (the TZ is dropped);
        // round-tripping a TZ-tagged timestamp through ferray loses TZ.
        AD::Timestamp(u, _tz) => Ok(DType::DateTime64(from_arrow_time_unit(u))),
        AD::Duration(u) => Ok(DType::Timedelta64(from_arrow_time_unit(u))),
        other => Err(FerrayError::invalid_dtype(format!(
            "Arrow DataType {other:?} has no ferray equivalent"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Polars DataType <-> DType
// ---------------------------------------------------------------------------

/// Convert a ferray [`DType`] to the corresponding Polars [`DataType`].
///
/// # Errors
///
/// Returns [`FerrayError::InvalidDtype`] if the ferray dtype has no Polars
/// equivalent (e.g. `Complex32`, `Complex64`, `U128`, `I128`, `Bool`-as-bitfield).
#[cfg(feature = "polars")]
pub fn dtype_to_polars(dt: DType) -> Result<polars::prelude::DataType, FerrayError> {
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
        other => Err(FerrayError::invalid_dtype(format!(
            "ferray dtype {other} has no Polars equivalent"
        ))),
    }
}

/// Convert a Polars [`DataType`] to the corresponding ferray [`DType`].
///
/// # Errors
///
/// Returns [`FerrayError::InvalidDtype`] for Polars types that ferray does
/// not support (e.g. `String`, `Date`, `Datetime`, etc.).
#[cfg(feature = "polars")]
pub fn polars_to_dtype(pd: &polars::prelude::DataType) -> Result<DType, FerrayError> {
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
        other => Err(FerrayError::invalid_dtype(format!(
            "Polars DataType {other:?} has no ferray equivalent"
        ))),
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "arrow")]
    mod arrow_tests {
        use crate::dtype_map::{arrow_to_dtype, dtype_to_arrow};
        use arrow::datatypes::DataType as AD;
        use ferray_core::DType;

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

            for (ferray_dt, arrow_dt) in &dtypes {
                let converted = dtype_to_arrow(*ferray_dt).unwrap();
                assert_eq!(&converted, arrow_dt);
                let back = arrow_to_dtype(&converted).unwrap();
                assert_eq!(back, *ferray_dt);
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

        #[test]
        fn datetime64_to_arrow_timestamp() {
            use arrow::datatypes::TimeUnit as ATU;
            use ferray_core::dtype::TimeUnit;
            assert_eq!(
                dtype_to_arrow(DType::DateTime64(TimeUnit::Ns)).unwrap(),
                AD::Timestamp(ATU::Nanosecond, None)
            );
            assert_eq!(
                dtype_to_arrow(DType::DateTime64(TimeUnit::Ms)).unwrap(),
                AD::Timestamp(ATU::Millisecond, None)
            );
        }

        #[test]
        fn timedelta64_to_arrow_duration() {
            use arrow::datatypes::TimeUnit as ATU;
            use ferray_core::dtype::TimeUnit;
            assert_eq!(
                dtype_to_arrow(DType::Timedelta64(TimeUnit::Us)).unwrap(),
                AD::Duration(ATU::Microsecond)
            );
        }

        #[test]
        fn arrow_timestamp_to_datetime64() {
            use arrow::datatypes::TimeUnit as ATU;
            use ferray_core::dtype::TimeUnit;
            let arrow_dt = AD::Timestamp(ATU::Nanosecond, None);
            assert_eq!(
                arrow_to_dtype(&arrow_dt).unwrap(),
                DType::DateTime64(TimeUnit::Ns)
            );
            // Timezone is dropped on the way back.
            let arrow_tz = AD::Timestamp(ATU::Microsecond, Some("UTC".into()));
            assert_eq!(
                arrow_to_dtype(&arrow_tz).unwrap(),
                DType::DateTime64(TimeUnit::Us)
            );
        }

        #[test]
        fn datetime64_minute_unit_arrow_unsupported() {
            use ferray_core::dtype::TimeUnit;
            // Arrow has no minute/hour/day TimeUnit — surface InvalidDtype.
            assert!(dtype_to_arrow(DType::DateTime64(TimeUnit::M)).is_err());
            assert!(dtype_to_arrow(DType::DateTime64(TimeUnit::H)).is_err());
            assert!(dtype_to_arrow(DType::DateTime64(TimeUnit::D)).is_err());
        }
    }

    #[cfg(feature = "polars")]
    mod polars_tests {
        use crate::dtype_map::{dtype_to_polars, polars_to_dtype};
        use ferray_core::DType;
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

            for (ferray_dt, polars_dt) in &dtypes {
                let converted = dtype_to_polars(*ferray_dt).unwrap();
                assert_eq!(&converted, polars_dt);
                let back = polars_to_dtype(&converted).unwrap();
                assert_eq!(back, *ferray_dt);
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
