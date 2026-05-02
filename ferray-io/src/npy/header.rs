// ferray-io: .npy header parsing and writing
//
// The .npy header is a Python dict literal with keys 'descr', 'fortran_order', 'shape'.
// Example: "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }"

use ferray_core::dtype::DType;
use ferray_core::error::{FerrayError, FerrayResult};

use super::dtype_parse::{self, Endianness};
use crate::format;

/// Parsed .npy file header.
#[derive(Debug, Clone)]
pub struct NpyHeader {
    /// The dtype descriptor string (e.g., "<f8").
    pub descr: String,
    /// Parsed dtype.
    pub dtype: DType,
    /// Parsed endianness.
    pub endianness: Endianness,
    /// Whether the data is stored in Fortran (column-major) order.
    pub fortran_order: bool,
    /// Shape of the array.
    pub shape: Vec<usize>,
    /// Format version (major, minor).
    pub version: (u8, u8),
}

/// Read and parse a .npy header from a reader.
///
/// After this function returns, the reader is positioned at the start of the data.
pub fn read_header<R: std::io::Read>(reader: &mut R) -> FerrayResult<NpyHeader> {
    // Read magic
    let mut magic = [0u8; format::NPY_MAGIC_LEN];
    reader
        .read_exact(&mut magic)
        .map_err(|e| FerrayError::io_error(format!("failed to read .npy magic: {e}")))?;

    if magic != *format::NPY_MAGIC {
        return Err(FerrayError::io_error(
            "not a valid .npy file: bad magic number",
        ));
    }

    // Read version
    let mut version = [0u8; 2];
    reader
        .read_exact(&mut version)
        .map_err(|e| FerrayError::io_error(format!("failed to read .npy version: {e}")))?;

    let major = version[0];
    let minor = version[1];

    if !matches!((major, minor), (1..=3, 0)) {
        return Err(FerrayError::io_error(format!(
            "unsupported .npy format version {major}.{minor}"
        )));
    }

    // Read header length
    let header_len = if major == 1 {
        let mut buf = [0u8; 2];
        reader
            .read_exact(&mut buf)
            .map_err(|e| FerrayError::io_error(format!("failed to read header length: {e}")))?;
        u16::from_le_bytes(buf) as usize
    } else {
        let mut buf = [0u8; 4];
        reader
            .read_exact(&mut buf)
            .map_err(|e| FerrayError::io_error(format!("failed to read header length: {e}")))?;
        let raw_len = u32::from_le_bytes(buf) as usize;
        // Cap header length at 1 MB to prevent unbounded allocation from
        // untrusted files. Legitimate .npy headers are typically < 1 KB.
        const MAX_HEADER_LEN: usize = 1_048_576;
        if raw_len > MAX_HEADER_LEN {
            return Err(FerrayError::io_error(format!(
                "header length {raw_len} exceeds maximum allowed size ({MAX_HEADER_LEN} bytes)"
            )));
        }
        raw_len
    };

    // Read header string
    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|e| FerrayError::io_error(format!("failed to read header: {e}")))?;

    let header_str = std::str::from_utf8(&header_bytes)
        .map_err(|e| FerrayError::io_error(format!("header is not valid UTF-8: {e}")))?;

    // Parse the header dict
    let (descr, fortran_order, shape) = parse_header_dict(header_str)?;
    let (dtype, endianness) = dtype_parse::parse_dtype_str(&descr)?;

    Ok(NpyHeader {
        descr,
        dtype,
        endianness,
        fortran_order,
        shape,
        version: (major, minor),
    })
}

/// Write a .npy header to a writer. Returns the total preamble+header size.
pub fn write_header<W: std::io::Write>(
    writer: &mut W,
    dtype: DType,
    shape: &[usize],
    fortran_order: bool,
) -> FerrayResult<()> {
    let descr = dtype_parse::dtype_to_native_descr(dtype)?;
    let fortran_str = if fortran_order { "True" } else { "False" };

    let shape_str = format_shape(shape);

    // Structured-dtype descriptors are Python list literals
    // (e.g. "[('x', '<f4'), ('y', '<f4')]") and contain
    // embedded apostrophes — emit them without the outer quotes
    // so the parser can read them back. Numeric scalars use the
    // canonical quoted form (#742).
    let descr_field = if descr.starts_with('[') {
        format!("'descr': {descr}")
    } else {
        format!("'descr': '{descr}'")
    };
    let dict = format!("{{{descr_field}, 'fortran_order': {fortran_str}, 'shape': {shape_str}, }}");

    // Try version 1.0 first (header length fits in u16)
    // Preamble: magic(6) + version(2) + header_len(2) = 10 for v1
    // Preamble: magic(6) + version(2) + header_len(4) = 12 for v2
    let preamble_v1 = format::NPY_MAGIC_LEN + 2 + 2; // 10
    let padding_needed_v1 = compute_padding(preamble_v1 + dict.len() + 1); // +1 for newline
    let total_header_v1 = dict.len() + padding_needed_v1 + 1;

    if total_header_v1 <= format::MAX_HEADER_LEN_V1 {
        // Version 1.0
        writer.write_all(format::NPY_MAGIC)?;
        writer.write_all(&[1, 0])?;
        writer.write_all(&(total_header_v1 as u16).to_le_bytes())?;
        writer.write_all(dict.as_bytes())?;
        write_padding(writer, padding_needed_v1)?;
    } else {
        // Version 2.0
        let preamble_v2 = format::NPY_MAGIC_LEN + 2 + 4; // 12
        let padding_needed_v2 = compute_padding(preamble_v2 + dict.len() + 1);
        let total_header_v2 = dict.len() + padding_needed_v2 + 1;

        writer.write_all(format::NPY_MAGIC)?;
        writer.write_all(&[2, 0])?;
        writer.write_all(&(total_header_v2 as u32).to_le_bytes())?;
        writer.write_all(dict.as_bytes())?;
        write_padding(writer, padding_needed_v2)?;
    }
    writer.write_all(b"\n")?;

    Ok(())
}

/// Compute the header size (preamble + header dict + padding + newline) for reading purposes.
/// Returns the byte offset where data begins.
#[must_use]
pub const fn compute_data_offset(version: (u8, u8), header_len: usize) -> usize {
    let preamble = format::NPY_MAGIC_LEN + 2 + if version.0 == 1 { 2 } else { 4 };
    preamble + header_len
}

const fn compute_padding(current_total: usize) -> usize {
    let remainder = current_total % format::HEADER_ALIGNMENT;
    if remainder == 0 {
        0
    } else {
        format::HEADER_ALIGNMENT - remainder
    }
}

/// Write `count` space bytes as header padding. Uses a single
/// `write_all` call instead of one per byte (#237).
fn write_padding<W: std::io::Write>(writer: &mut W, count: usize) -> FerrayResult<()> {
    // Stack buffer covers all realistic header padding; the .npy
    // spec aligns to 64 bytes, so count is always < 64.
    const MAX_STACK: usize = 128;
    if count <= MAX_STACK {
        let buf = [b' '; MAX_STACK];
        writer.write_all(&buf[..count])?;
    } else {
        let buf = vec![b' '; count];
        writer.write_all(&buf)?;
    }
    Ok(())
}

fn format_shape(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let parts: Vec<String> = shape.iter().map(std::string::ToString::to_string).collect();
            format!("({})", parts.join(", "))
        }
    }
}

/// Parse the Python dict-like header string.
///
/// We do simple string parsing here rather than pulling in a full Python parser.
/// The format is well-defined: `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`
fn parse_header_dict(header: &str) -> FerrayResult<(String, bool, Vec<usize>)> {
    let header = header.trim();

    // Strip outer braces
    let inner = header
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| FerrayError::io_error("header dict missing braces"))?
        .trim();

    let descr = extract_string_value(inner, "descr")?;
    let fortran_order = extract_bool_value(inner, "fortran_order")?;
    let shape = extract_shape_value(inner, "shape")?;

    Ok((descr, fortran_order, shape))
}

/// Extract a string value for a given key from the dict body.
///
/// Handles both numpy's quoted scalar descrs (`'<f8'`) and the
/// unquoted Python list-literal form used for structured dtypes
/// (`[('x', '<f4'), ('y', '<f4')]`) (#742).
fn extract_string_value(dict_body: &str, key: &str) -> FerrayResult<String> {
    let pattern = format!("'{key}':");
    let pos = dict_body
        .find(&pattern)
        .ok_or_else(|| FerrayError::io_error(format!("header missing key '{key}'")))?;

    let after_key = dict_body[pos + pattern.len()..].trim_start();

    let first = after_key
        .as_bytes()
        .first()
        .ok_or_else(|| FerrayError::io_error(format!("missing value for key '{key}'")))?;

    // Structured-dtype descriptor — Python list literal. Walk by
    // bracket depth, accepting any apostrophes inside the list.
    if *first == b'[' {
        let mut depth = 0_i32;
        let mut end = None;
        for (i, c) in after_key.char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end.ok_or_else(|| {
            FerrayError::io_error(format!("unterminated structured descr for key '{key}'"))
        })?;
        return Ok(after_key[..end].to_string());
    }

    if *first != b'\'' && *first != b'"' {
        return Err(FerrayError::io_error(format!(
            "expected string value for key '{key}'"
        )));
    }

    let qc = *first as char;
    let value_start = &after_key[1..];
    let end = value_start
        .find(qc)
        .ok_or_else(|| FerrayError::io_error(format!("unterminated string for key '{key}'")))?;

    Ok(value_start[..end].to_string())
}

/// Extract a boolean value for a given key from the dict body.
fn extract_bool_value(dict_body: &str, key: &str) -> FerrayResult<bool> {
    let pattern = format!("'{key}':");
    let pos = dict_body
        .find(&pattern)
        .ok_or_else(|| FerrayError::io_error(format!("header missing key '{key}'")))?;

    let after_key = dict_body[pos + pattern.len()..].trim_start();

    if after_key.starts_with("True") {
        Ok(true)
    } else if after_key.starts_with("False") {
        Ok(false)
    } else {
        Err(FerrayError::io_error(format!(
            "expected True/False for key '{key}'"
        )))
    }
}

/// Extract a tuple shape value for a given key from the dict body.
fn extract_shape_value(dict_body: &str, key: &str) -> FerrayResult<Vec<usize>> {
    let pattern = format!("'{key}':");
    let pos = dict_body
        .find(&pattern)
        .ok_or_else(|| FerrayError::io_error(format!("header missing key '{key}'")))?;

    let after_key = dict_body[pos + pattern.len()..].trim_start();

    // Find the opening paren
    if !after_key.starts_with('(') {
        return Err(FerrayError::io_error(format!(
            "expected tuple for key '{key}'"
        )));
    }

    let close = after_key
        .find(')')
        .ok_or_else(|| FerrayError::io_error(format!("unterminated tuple for key '{key}'")))?;

    let tuple_inner = &after_key[1..close];
    let tuple_inner = tuple_inner.trim();

    if tuple_inner.is_empty() {
        return Ok(vec![]);
    }

    let parts: FerrayResult<Vec<usize>> = tuple_inner
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| FerrayError::io_error(format!("invalid shape dimension '{s}': {e}")))
        })
        .collect();

    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_header() {
        let header = "{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }";
        let (descr, fortran, shape) = parse_header_dict(header).unwrap();
        assert_eq!(descr, "<f8");
        assert!(!fortran);
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn parse_1d_header() {
        let header = "{'descr': '<i4', 'fortran_order': False, 'shape': (10,), }";
        let (descr, fortran, shape) = parse_header_dict(header).unwrap();
        assert_eq!(descr, "<i4");
        assert!(!fortran);
        assert_eq!(shape, vec![10]);
    }

    #[test]
    fn parse_scalar_header() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
        let (descr, fortran, shape) = parse_header_dict(header).unwrap();
        assert_eq!(descr, "<f4");
        assert!(!fortran);
        assert!(shape.is_empty());
    }

    #[test]
    fn parse_fortran_order() {
        let header = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let (_, fortran, _) = parse_header_dict(header).unwrap();
        assert!(fortran);
    }

    #[test]
    fn format_shape_empty() {
        assert_eq!(format_shape(&[]), "()");
    }

    #[test]
    fn format_shape_1d() {
        assert_eq!(format_shape(&[5]), "(5,)");
    }

    #[test]
    fn format_shape_2d() {
        assert_eq!(format_shape(&[3, 4]), "(3, 4)");
    }

    #[test]
    fn write_read_roundtrip() {
        let mut buf = Vec::new();
        write_header(&mut buf, DType::F64, &[3, 4], false).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let header = read_header(&mut cursor).unwrap();

        assert_eq!(header.dtype, DType::F64);
        assert_eq!(header.shape, vec![3, 4]);
        assert!(!header.fortran_order);
    }

    #[test]
    fn header_alignment() {
        for shape in [&[3, 4][..], &[100, 200, 300], &[1]] {
            let mut buf = Vec::new();
            write_header(&mut buf, DType::F64, shape, false).unwrap();
            assert_eq!(
                buf.len() % format::HEADER_ALIGNMENT,
                0,
                "header not aligned for shape {shape:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // v2.0 header coverage (#240).
    //
    // The v2.0 write path triggers when the dict exceeds u16::MAX bytes;
    // organically that requires a shape with tens of thousands of axes.
    // We exercise the writer by forcing a mega-shape, then exercise the
    // reader against a hand-crafted v2.0 byte blob to catch any future
    // skew between the two.
    // -----------------------------------------------------------------------

    /// Force the v2.0 write path with a synthetic 22000-dim shape and
    /// confirm the round-trip produces a v2.0 header that parses back
    /// to the same shape.
    #[test]
    fn v20_writer_roundtrip_huge_shape() {
        // 22000 axes, each "1," contributes 2 chars beyond the prefix
        // — comfortably above the 65535-byte v1 threshold.
        let shape: Vec<usize> = vec![1usize; 22000];
        let mut buf = Vec::new();
        write_header(&mut buf, DType::F64, &shape, false).unwrap();

        // Bytes [6..8] hold the version major/minor; v2.0 should kick in.
        assert_eq!(buf[6], 2, "expected major version 2, got {}", buf[6]);
        assert_eq!(buf[7], 0, "expected minor version 0, got {}", buf[7]);

        let mut cursor = std::io::Cursor::new(buf);
        let header = read_header(&mut cursor).unwrap();
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.dtype, DType::F64);
        assert_eq!(header.shape.len(), 22000);
        assert!(header.shape.iter().all(|&d| d == 1));
    }

    /// Hand-craft a minimal v2.0 header and verify the reader walks
    /// through the 4-byte length branch correctly. This guards against
    /// the case where every v2.0 header happens to come from our own
    /// writer (so a bug in the reader's u32 length branch would only
    /// surface against numpy-produced files).
    #[test]
    fn v20_reader_handcrafted_minimal() {
        // Build a tiny dict, then pad to the 64-byte boundary so the
        // header is well-formed even though the v1 path could have
        // covered it. The point is to assert the reader's u32-length
        // branch parses correctly.
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (5,), }";
        let preamble_v2 = format::NPY_MAGIC_LEN + 2 + 4; // 12
        let body_with_newline = dict.len() + 1;
        let pad = {
            let total = preamble_v2 + body_with_newline;
            let r = total % format::HEADER_ALIGNMENT;
            if r == 0 {
                0
            } else {
                format::HEADER_ALIGNMENT - r
            }
        };
        let header_len = dict.len() + pad + 1;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(format::NPY_MAGIC);
        bytes.extend_from_slice(&[2, 0]);
        bytes.extend_from_slice(&(header_len as u32).to_le_bytes());
        bytes.extend_from_slice(dict.as_bytes());
        bytes.extend(std::iter::repeat_n(b' ', pad));
        bytes.push(b'\n');

        let mut cursor = std::io::Cursor::new(bytes);
        let header = read_header(&mut cursor).unwrap();
        assert_eq!(header.version, (2, 0));
        assert_eq!(header.dtype, DType::F64);
        assert_eq!(header.shape, vec![5]);
        assert!(!header.fortran_order);
    }

    /// The reader caps v2.0 header length at 1 MB; a header claiming
    /// more should be rejected before the buffer is allocated.
    #[test]
    fn v20_reader_rejects_oversized_header() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(format::NPY_MAGIC);
        bytes.extend_from_slice(&[2, 0]);
        // 2 MB header length — must be rejected.
        bytes.extend_from_slice(&(2u32 * 1024 * 1024).to_le_bytes());
        let mut cursor = std::io::Cursor::new(bytes);
        let err = read_header(&mut cursor).unwrap_err();
        assert!(
            err.to_string().contains("exceeds maximum"),
            "expected oversized-header error, got: {err}"
        );
    }

    /// v3.0 is also accepted by the writer/reader pair (numpy added it
    /// for UTF-8 field names). Make sure a v3.0 header round-trips
    /// through the same u32-length branch as v2.0.
    #[test]
    fn v30_reader_handcrafted_minimal() {
        let dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (3,), }";
        let preamble = format::NPY_MAGIC_LEN + 2 + 4;
        let body_with_newline = dict.len() + 1;
        let pad = {
            let total = preamble + body_with_newline;
            let r = total % format::HEADER_ALIGNMENT;
            if r == 0 {
                0
            } else {
                format::HEADER_ALIGNMENT - r
            }
        };
        let header_len = dict.len() + pad + 1;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(format::NPY_MAGIC);
        bytes.extend_from_slice(&[3, 0]);
        bytes.extend_from_slice(&(header_len as u32).to_le_bytes());
        bytes.extend_from_slice(dict.as_bytes());
        bytes.extend(std::iter::repeat_n(b' ', pad));
        bytes.push(b'\n');

        let mut cursor = std::io::Cursor::new(bytes);
        let header = read_header(&mut cursor).unwrap();
        assert_eq!(header.version, (3, 0));
        assert_eq!(header.shape, vec![3]);
    }
}
