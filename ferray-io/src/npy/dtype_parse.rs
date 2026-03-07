// ferray-io: NumPy dtype string parsing
//
// Parses dtype descriptor strings like "<f8", "|b1", ">i4" into
// (DType, Endianness) pairs.

use ferray_core::dtype::DType;
use ferray_core::error::{FerrayError, FerrayResult};

/// Byte order / endianness extracted from a NumPy dtype string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    /// Little-endian (`<` prefix).
    Little,
    /// Big-endian (`>` prefix).
    Big,
    /// Not applicable / single byte (`|` prefix, or `=` for native).
    Native,
}

impl Endianness {
    /// Whether byte-swapping is needed on the current platform.
    #[inline]
    pub fn needs_swap(self) -> bool {
        match self {
            Self::Little => cfg!(target_endian = "big"),
            Self::Big => cfg!(target_endian = "little"),
            Self::Native => false,
        }
    }
}

/// Parse a NumPy dtype descriptor string into a `(DType, Endianness)` pair.
///
/// Examples:
/// - `"<f8"` -> `(DType::F64, Endianness::Little)`
/// - `"|b1"` -> `(DType::Bool, Endianness::Native)`
/// - `">i4"` -> `(DType::I32, Endianness::Big)`
pub fn parse_dtype_str(s: &str) -> FerrayResult<(DType, Endianness)> {
    if s.len() < 2 {
        return Err(FerrayError::invalid_dtype(format!(
            "dtype string too short: '{s}'"
        )));
    }

    let (endian, type_str) = match s.as_bytes()[0] {
        b'<' => (Endianness::Little, &s[1..]),
        b'>' => (Endianness::Big, &s[1..]),
        b'|' => (Endianness::Native, &s[1..]),
        b'=' => (Endianness::Native, &s[1..]),
        // If no endian prefix, assume native
        _ => (Endianness::Native, s),
    };

    let dtype = match type_str {
        "b1" => DType::Bool,
        "u1" => DType::U8,
        "u2" => DType::U16,
        "u4" => DType::U32,
        "u8" => DType::U64,
        "u16" => DType::U128,
        "i1" => DType::I8,
        "i2" => DType::I16,
        "i4" => DType::I32,
        "i8" => DType::I64,
        "i16" => DType::I128,
        "f4" => DType::F32,
        "f8" => DType::F64,
        "c8" => DType::Complex32,
        "c16" => DType::Complex64,
        _ => {
            return Err(FerrayError::invalid_dtype(format!(
                "unsupported dtype descriptor: '{s}'"
            )));
        }
    };

    Ok((dtype, endian))
}

/// Convert a `DType` to its NumPy dtype descriptor string with the given endianness.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` for unsupported dtype variants.
pub fn dtype_to_descr(dtype: DType, endian: Endianness) -> FerrayResult<String> {
    let prefix = match endian {
        Endianness::Little => '<',
        Endianness::Big => '>',
        Endianness::Native => '|',
    };

    let type_str = match dtype {
        DType::Bool => "b1",
        DType::U8 => "u1",
        DType::U16 => "u2",
        DType::U32 => "u4",
        DType::U64 => "u8",
        DType::U128 => "u16",
        DType::I8 => "i1",
        DType::I16 => "i2",
        DType::I32 => "i4",
        DType::I64 => "i8",
        DType::I128 => "i16",
        DType::F32 => "f4",
        DType::F64 => "f8",
        DType::Complex32 => "c8",
        DType::Complex64 => "c16",
        _ => {
            return Err(FerrayError::invalid_dtype(format!(
                "unsupported dtype for descriptor: {dtype:?}"
            )));
        }
    };

    // Bool and single-byte types use '|' regardless
    let actual_prefix = match dtype {
        DType::Bool | DType::U8 | DType::I8 => '|',
        _ => prefix,
    };

    Ok(format!("{actual_prefix}{type_str}"))
}

/// Return the native-endian dtype descriptor for a given DType.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` for unsupported dtype variants.
pub fn dtype_to_native_descr(dtype: DType) -> FerrayResult<String> {
    let endian = if cfg!(target_endian = "little") {
        Endianness::Little
    } else {
        Endianness::Big
    };
    dtype_to_descr(dtype, endian)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_common_dtypes() {
        assert_eq!(
            parse_dtype_str("<f8").unwrap(),
            (DType::F64, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<f4").unwrap(),
            (DType::F32, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str(">i4").unwrap(),
            (DType::I32, Endianness::Big)
        );
        assert_eq!(
            parse_dtype_str("<i8").unwrap(),
            (DType::I64, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("|b1").unwrap(),
            (DType::Bool, Endianness::Native)
        );
        assert_eq!(
            parse_dtype_str("<u1").unwrap(),
            (DType::U8, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<c8").unwrap(),
            (DType::Complex32, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<c16").unwrap(),
            (DType::Complex64, Endianness::Little)
        );
    }

    #[test]
    fn parse_unsigned_types() {
        assert_eq!(
            parse_dtype_str("<u2").unwrap(),
            (DType::U16, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<u4").unwrap(),
            (DType::U32, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<u8").unwrap(),
            (DType::U64, Endianness::Little)
        );
    }

    #[test]
    fn parse_128bit_types() {
        assert_eq!(
            parse_dtype_str("<i16").unwrap(),
            (DType::I128, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str("<u16").unwrap(),
            (DType::U128, Endianness::Little)
        );
    }

    #[test]
    fn parse_invalid() {
        assert!(parse_dtype_str("x").is_err());
        assert!(parse_dtype_str("<z4").is_err());
        assert!(parse_dtype_str("").is_err());
    }

    #[test]
    fn roundtrip_descr() {
        let dtypes = [
            DType::Bool,
            DType::U8,
            DType::U16,
            DType::U32,
            DType::U64,
            DType::I8,
            DType::I16,
            DType::I32,
            DType::I64,
            DType::F32,
            DType::F64,
            DType::Complex32,
            DType::Complex64,
        ];
        for dt in dtypes {
            let descr = dtype_to_native_descr(dt).unwrap();
            let (parsed_dt, _) = parse_dtype_str(&descr).unwrap();
            assert_eq!(
                parsed_dt, dt,
                "roundtrip failed for {dt:?}: descr='{descr}'"
            );
        }
    }

    #[test]
    fn endianness_swap() {
        if cfg!(target_endian = "little") {
            assert!(!Endianness::Little.needs_swap());
            assert!(Endianness::Big.needs_swap());
        } else {
            assert!(Endianness::Little.needs_swap());
            assert!(!Endianness::Big.needs_swap());
        }
        assert!(!Endianness::Native.needs_swap());
    }
}
