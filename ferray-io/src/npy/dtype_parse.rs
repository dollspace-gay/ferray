// ferray-io: NumPy dtype string parsing
//
// Parses dtype descriptor strings like "<f8", "|b1", ">i4" into
// (DType, Endianness) pairs.

use ferray_core::dtype::DType;
use ferray_core::error::{FerrayError, FerrayResult};

/// Byte order / endianness extracted from a `NumPy` dtype string.
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
    #[must_use]
    pub const fn needs_swap(self) -> bool {
        match self {
            Self::Little => cfg!(target_endian = "big"),
            Self::Big => cfg!(target_endian = "little"),
            Self::Native => false,
        }
    }
}

/// Parse a `NumPy` dtype descriptor string into a `(DType, Endianness)` pair.
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

    // Structured (record) dtype descriptor (#742). numpy serialises
    // these as a Python list literal,
    // e.g. "[('x', '<f4'), ('y', '<f4')]". We parse the field list,
    // resolve each (name, dtype) into a FieldDescriptor with byte
    // offsets and sizes, then leak the resulting slice into 'static
    // memory so the `DType::Struct(&'static [FieldDescriptor])`
    // contract holds. Leaks are bounded — one leak per distinct
    // structured dtype encountered, and only at load time.
    if s.starts_with('[') && s.ends_with(']') {
        let fields = parse_structured_descr(s)?;
        let mut descriptors: Vec<ferray_core::record::FieldDescriptor> =
            Vec::with_capacity(fields.len());
        let mut offset = 0_usize;
        for (name, dt_str) in &fields {
            let (field_dtype, _field_endian) = parse_dtype_str(dt_str)?;
            let size = field_dtype.size_of();
            descriptors.push(ferray_core::record::FieldDescriptor {
                name: Box::leak(name.clone().into_boxed_str()),
                dtype: field_dtype,
                offset,
                size,
            });
            offset += size;
        }
        let leaked: &'static [ferray_core::record::FieldDescriptor] =
            Box::leak(descriptors.into_boxed_slice());
        return Ok((DType::Struct(leaked), Endianness::Native));
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
        // IEEE 754 binary16 — standard NumPy `np.float16` descriptor.
        #[cfg(feature = "f16")]
        "f2" => DType::F16,
        "f4" => DType::F32,
        "f8" => DType::F64,
        // bfloat16 — non-standard NumPy descriptor. We use the ferray-specific
        // tag `"bf16"` so ferray round-trip is lossless; files saved this way
        // are NOT loadable by vanilla NumPy (which has no native bf16 dtype).
        #[cfg(feature = "bf16")]
        "bf16" => DType::BF16,
        "c8" => DType::Complex32,
        "c16" => DType::Complex64,
        // datetime64 / timedelta64 with explicit unit:
        //   "M8[ns]" / "m8[ns]" etc. The leading 'M'/'m' is the type tag
        //   (M = datetime, m = timedelta), '8' is the byte size, and the
        //   bracketed suffix is the TimeUnit.
        other if other.starts_with("M8[") && other.ends_with(']') => {
            let unit_str = &other[3..other.len() - 1];
            let unit =
                ferray_core::dtype::TimeUnit::from_descr_suffix(unit_str).ok_or_else(|| {
                    FerrayError::invalid_dtype(format!("unknown datetime64 unit: '{unit_str}'"))
                })?;
            DType::DateTime64(unit)
        }
        other if other.starts_with("m8[") && other.ends_with(']') => {
            let unit_str = &other[3..other.len() - 1];
            let unit =
                ferray_core::dtype::TimeUnit::from_descr_suffix(unit_str).ok_or_else(|| {
                    FerrayError::invalid_dtype(format!("unknown timedelta64 unit: '{unit_str}'"))
                })?;
            DType::Timedelta64(unit)
        }
        // Fixed-width string / unicode / void descriptors (numpy
        // S20 / U10 / V8 family). DType variants land via #741.
        other
            if (other.starts_with('S') || other.starts_with('U') || other.starts_with('V'))
                && other.len() > 1
                && other[1..].chars().all(|c| c.is_ascii_digit()) =>
        {
            let width: usize = other[1..].parse().map_err(|_| {
                FerrayError::invalid_dtype(format!(
                    "fixed-width dtype width is not a valid integer: '{}'",
                    &other[1..]
                ))
            })?;
            match other.as_bytes()[0] {
                b'S' => DType::FixedAscii(width),
                b'U' => DType::FixedUnicode(width),
                _ => DType::RawBytes(width),
            }
        }
        _ => {
            return Err(FerrayError::invalid_dtype(format!(
                "unsupported dtype descriptor: '{s}'"
            )));
        }
    };

    Ok((dtype, endian))
}

/// Convert a `DType` to its `NumPy` dtype descriptor string with the given endianness.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` for unsupported dtype variants.
pub fn dtype_to_descr(dtype: DType, endian: Endianness) -> FerrayResult<String> {
    let prefix = match endian {
        Endianness::Little => '<',
        Endianness::Big => '>',
        Endianness::Native => '|',
    };

    // datetime64 / timedelta64 carry a unit and so don't fit the simple
    // `<type_str>` prefix path; format them directly here.
    if let DType::DateTime64(u) = dtype {
        let actual_prefix = match endian {
            Endianness::Native => '|',
            _ => prefix,
        };
        return Ok(format!("{actual_prefix}M8[{}]", u.descr_suffix()));
    }
    if let DType::Timedelta64(u) = dtype {
        let actual_prefix = match endian {
            Endianness::Native => '|',
            _ => prefix,
        };
        return Ok(format!("{actual_prefix}m8[{}]", u.descr_suffix()));
    }
    // Structured dtype (#742): emit numpy's list-of-tuples form,
    // recursing per-field through dtype_to_descr with the same
    // endianness selection.
    if let DType::Struct(fields) = dtype {
        let mut parts = Vec::with_capacity(fields.len());
        for fld in fields {
            let inner = dtype_to_descr(fld.dtype, endian)?;
            parts.push(format!("('{}', '{inner}')", fld.name));
        }
        return Ok(format!("[{}]", parts.join(", ")));
    }
    // Fixed-width string / void dtypes (#741): emit the canonical
    // numpy descriptor with byte-aligned prefix.
    if let DType::FixedAscii(n) = dtype {
        return Ok(format!("|S{n}"));
    }
    if let DType::FixedUnicode(n) = dtype {
        let actual_prefix = match endian {
            Endianness::Native => '<',
            _ => prefix,
        };
        return Ok(format!("{actual_prefix}U{n}"));
    }
    if let DType::RawBytes(n) = dtype {
        return Ok(format!("|V{n}"));
    }

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
        #[cfg(feature = "f16")]
        DType::F16 => "f2",
        DType::F32 => "f4",
        DType::F64 => "f8",
        #[cfg(feature = "bf16")]
        DType::BF16 => "bf16",
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

/// Parse a numpy structured-dtype descriptor like
/// `[('x', '<f4'), ('y', '<f4'), ('label', '|S10')]` into a list of
/// `(field_name, field_dtype_descr)` pairs. Used for diagnostic
/// messaging only — actual loading into FerrayRecord arrays is
/// tracked in #735.
fn parse_structured_descr(s: &str) -> FerrayResult<Vec<(String, String)>> {
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| {
            FerrayError::invalid_dtype(format!("malformed structured descriptor: '{s}'"))
        })?;
    let mut out = Vec::new();
    // Split top-level on `),` boundaries — each field is a tuple.
    let mut depth = 0_i32;
    let mut cur = String::new();
    for c in inner.chars() {
        match c {
            '(' => {
                depth += 1;
                cur.push(c);
            }
            ')' => {
                depth -= 1;
                cur.push(c);
                if depth == 0 {
                    let trimmed = cur.trim().trim_start_matches(',').trim();
                    if !trimmed.is_empty() {
                        let pair = parse_structured_field(trimmed)?;
                        out.push(pair);
                    }
                    cur.clear();
                }
            }
            _ => cur.push(c),
        }
    }
    if depth != 0 {
        return Err(FerrayError::invalid_dtype(format!(
            "structured descriptor has unbalanced parentheses: '{s}'"
        )));
    }
    Ok(out)
}

/// Parse one `('name', 'dtype')` tuple into the unquoted strings.
fn parse_structured_field(field: &str) -> FerrayResult<(String, String)> {
    let inside = field
        .trim()
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or_else(|| {
            FerrayError::invalid_dtype(format!("structured field must be a tuple: '{field}'"))
        })?;
    let parts: Vec<&str> = inside.splitn(2, ',').map(str::trim).collect();
    if parts.len() != 2 {
        return Err(FerrayError::invalid_dtype(format!(
            "structured field '{field}' must have exactly two parts (name, dtype)"
        )));
    }
    let name = parts[0].trim_matches(|c| c == '\'' || c == '"').to_string();
    let dtype = parts[1].trim_matches(|c| c == '\'' || c == '"').to_string();
    Ok((name, dtype))
}

/// Return the native-endian dtype descriptor for a given `DType`.
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

    #[cfg(feature = "f16")]
    #[test]
    fn parse_f16_descriptor() {
        assert_eq!(
            parse_dtype_str("<f2").unwrap(),
            (DType::F16, Endianness::Little)
        );
        assert_eq!(
            parse_dtype_str(">f2").unwrap(),
            (DType::F16, Endianness::Big)
        );
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_roundtrip_descr() {
        let d = dtype_to_native_descr(DType::F16).unwrap();
        let (parsed, _) = parse_dtype_str(&d).unwrap();
        assert_eq!(parsed, DType::F16);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn parse_bf16_descriptor() {
        assert_eq!(
            parse_dtype_str("<bf16").unwrap(),
            (DType::BF16, Endianness::Little)
        );
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn bf16_roundtrip_descr() {
        let d = dtype_to_native_descr(DType::BF16).unwrap();
        let (parsed, _) = parse_dtype_str(&d).unwrap();
        assert_eq!(parsed, DType::BF16);
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

    // -- datetime64 / timedelta64 descriptor parsing --

    #[test]
    fn parse_datetime64_ns() {
        use ferray_core::dtype::TimeUnit;
        let (dt, e) = parse_dtype_str("<M8[ns]").unwrap();
        assert_eq!(dt, DType::DateTime64(TimeUnit::Ns));
        assert_eq!(e, Endianness::Little);
    }

    #[test]
    fn parse_datetime64_us() {
        use ferray_core::dtype::TimeUnit;
        let (dt, _) = parse_dtype_str("<M8[us]").unwrap();
        assert_eq!(dt, DType::DateTime64(TimeUnit::Us));
    }

    #[test]
    fn parse_timedelta64_ms() {
        use ferray_core::dtype::TimeUnit;
        let (dt, _) = parse_dtype_str("<m8[ms]").unwrap();
        assert_eq!(dt, DType::Timedelta64(TimeUnit::Ms));
    }

    #[test]
    fn datetime64_roundtrip_descr() {
        use ferray_core::dtype::TimeUnit;
        for u in [
            TimeUnit::Ns,
            TimeUnit::Us,
            TimeUnit::Ms,
            TimeUnit::S,
            TimeUnit::D,
        ] {
            let dt = DType::DateTime64(u);
            let s = dtype_to_descr(dt, Endianness::Little).unwrap();
            let (parsed, _) = parse_dtype_str(&s).unwrap();
            assert_eq!(parsed, dt, "roundtrip failed for {dt}");
        }
    }

    #[test]
    fn timedelta64_roundtrip_descr() {
        use ferray_core::dtype::TimeUnit;
        for u in [
            TimeUnit::Ns,
            TimeUnit::Us,
            TimeUnit::Ms,
            TimeUnit::S,
            TimeUnit::D,
        ] {
            let dt = DType::Timedelta64(u);
            let s = dtype_to_descr(dt, Endianness::Little).unwrap();
            let (parsed, _) = parse_dtype_str(&s).unwrap();
            assert_eq!(parsed, dt, "roundtrip failed for {dt}");
        }
    }

    #[test]
    fn unknown_datetime_unit_errors() {
        assert!(parse_dtype_str("<M8[foobar]").is_err());
    }

    // ---- structured dtype descriptors (#735, #742) ---------------------

    #[test]
    fn parse_structured_two_fields_returns_struct() {
        let (dt, _) = parse_dtype_str("[('x', '<f4'), ('y', '<f4')]").unwrap();
        match dt {
            DType::Struct(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "x");
                assert_eq!(fields[0].dtype, DType::F32);
                assert_eq!(fields[0].offset, 0);
                assert_eq!(fields[0].size, 4);
                assert_eq!(fields[1].name, "y");
                assert_eq!(fields[1].dtype, DType::F32);
                assert_eq!(fields[1].offset, 4);
                assert_eq!(fields[1].size, 4);
            }
            _ => panic!("expected Struct, got {dt:?}"),
        }
    }

    #[test]
    fn parse_structured_three_mixed_fields_returns_struct() {
        let (dt, _) = parse_dtype_str("[('a', '<i4'), ('b', '<f8'), ('c', '|S10')]").unwrap();
        match dt {
            DType::Struct(fields) => {
                assert_eq!(fields.len(), 3);
                assert_eq!(fields[0].dtype, DType::I32);
                assert_eq!(fields[1].dtype, DType::F64);
                assert_eq!(fields[2].dtype, DType::FixedAscii(10));
                // Offsets accumulate: 0, 4, 12.
                assert_eq!(fields[0].offset, 0);
                assert_eq!(fields[1].offset, 4);
                assert_eq!(fields[2].offset, 12);
            }
            _ => panic!("expected Struct, got {dt:?}"),
        }
    }

    #[test]
    fn parse_structured_single_field_returns_struct() {
        let (dt, _) = parse_dtype_str("[('only', '<f8')]").unwrap();
        match dt {
            DType::Struct(fields) => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name, "only");
                assert_eq!(fields[0].dtype, DType::F64);
            }
            _ => panic!("expected Struct, got {dt:?}"),
        }
    }

    #[test]
    fn structured_descr_helper_extracts_pairs() {
        let pairs = parse_structured_descr("[('x', '<f4'), ('y', '<i8')]").unwrap();
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("x".into(), "<f4".into()));
        assert_eq!(pairs[1], ("y".into(), "<i8".into()));
    }

    // ---- fixed-width string / unicode / void parser (#734, #741) ------

    #[test]
    fn parse_fixed_ascii_string_descriptor() {
        let (dt, e) = parse_dtype_str("|S20").unwrap();
        assert_eq!(dt, DType::FixedAscii(20));
        assert_eq!(e, Endianness::Native);
    }

    #[test]
    fn parse_fixed_unicode_descriptor() {
        let (dt, e) = parse_dtype_str("<U10").unwrap();
        assert_eq!(dt, DType::FixedUnicode(10));
        assert_eq!(e, Endianness::Little);
    }

    #[test]
    fn parse_void_descriptor() {
        let (dt, e) = parse_dtype_str("|V8").unwrap();
        assert_eq!(dt, DType::RawBytes(8));
        assert_eq!(e, Endianness::Native);
    }

    #[test]
    fn parse_string_dtype_with_zero_width() {
        // S0 / U0 / V0 are valid descriptors per numpy (degenerate
        // but parseable).
        assert_eq!(parse_dtype_str("|S0").unwrap().0, DType::FixedAscii(0));
        assert_eq!(parse_dtype_str("<U0").unwrap().0, DType::FixedUnicode(0));
        assert_eq!(parse_dtype_str("|V0").unwrap().0, DType::RawBytes(0));
    }

    #[test]
    fn fixed_string_dtype_size_and_alignment() {
        // S20 = 20 ASCII bytes; U10 = 10 codepoints * 4 = 40 bytes;
        // V8 = 8 raw bytes.
        assert_eq!(DType::FixedAscii(20).size_of(), 20);
        assert_eq!(DType::FixedUnicode(10).size_of(), 40);
        assert_eq!(DType::RawBytes(8).size_of(), 8);
        assert_eq!(DType::FixedAscii(20).alignment(), 1);
        assert_eq!(DType::FixedUnicode(10).alignment(), 4);
        assert_eq!(DType::RawBytes(8).alignment(), 1);
    }

    #[test]
    fn fixed_string_dtype_display_matches_numpy_descriptor() {
        assert_eq!(format!("{}", DType::FixedAscii(20)), "S20");
        assert_eq!(format!("{}", DType::FixedUnicode(10)), "U10");
        assert_eq!(format!("{}", DType::RawBytes(8)), "V8");
    }

    #[test]
    fn non_string_descriptors_unaffected() {
        // Make sure the new branch doesn't false-match common dtypes
        // that happen to start with characters near 'S/U/V'.
        assert!(parse_dtype_str("<u4").is_ok());
        assert!(parse_dtype_str("<u8").is_ok());
        assert!(parse_dtype_str("<i4").is_ok());
    }
}
