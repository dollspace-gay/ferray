// ferray-io: Format constants, shared types, and `numpy.lib.format`-equivalent helpers.

use ferray_core::DynArray;
use ferray_core::dtype::DType;
use ferray_core::error::FerrayResult;

use crate::npy::dtype_parse::{Endianness, dtype_to_descr, parse_dtype_str};

/// The magic string at the start of every `.npy` file.
pub const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Length of the magic string.
pub const NPY_MAGIC_LEN: usize = 6;

/// Supported `.npy` format version 1.0 (header length stored in 2 bytes).
pub const VERSION_1_0: (u8, u8) = (1, 0);

/// Supported `.npy` format version 2.0 (header length stored in 4 bytes).
pub const VERSION_2_0: (u8, u8) = (2, 0);

/// Supported `.npy` format version 3.0 (header length stored in 4 bytes, UTF-8 header).
pub const VERSION_3_0: (u8, u8) = (3, 0);

/// Alignment of the header + preamble in bytes.
pub const HEADER_ALIGNMENT: usize = 64;

/// Maximum header length for version 1.0 (`u16::MAX`).
pub const MAX_HEADER_LEN_V1: usize = u16::MAX as usize;

/// Mode for memory-mapped file access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemmapMode {
    /// Read-only memory mapping. The file is opened for reading and the
    /// resulting slice is immutable.
    ReadOnly,
    /// Read-write memory mapping. Modifications are written back to the
    /// underlying file.
    ReadWrite,
    /// Copy-on-write memory mapping. Modifications are kept in memory
    /// and are not written back to the file.
    CopyOnWrite,
}

// ---------------------------------------------------------------------------
// numpy.lib.format-equivalent helpers
// ---------------------------------------------------------------------------

/// Header dictionary parsed from an `.npy` v1.0 stream.
///
/// Mirrors the dict NumPy's `numpy.lib.format.header_data_from_array_1_0`
/// returns: `descr` (the dtype descriptor string), `fortran_order`, and
/// `shape`. Stored as a small struct so callers don't need an untyped map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderData {
    /// dtype descriptor string (e.g. `"<f8"`, `"|b1"`).
    pub descr: String,
    /// `true` if the underlying data is laid out in Fortran (column-major) order.
    pub fortran_order: bool,
    /// Array shape.
    pub shape: Vec<usize>,
}

/// Parse a NumPy dtype descriptor string into a [`DType`].
///
/// Equivalent to `numpy.lib.format.descr_to_dtype` for the basic primitive
/// dtype family. Endianness is dropped — call
/// [`crate::npy::dtype_parse::parse_dtype_str`] directly to retain it.
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` for an unsupported or malformed
/// descriptor.
pub fn descr_to_dtype(descr: &str) -> FerrayResult<DType> {
    let (dt, _) = parse_dtype_str(descr)?;
    Ok(dt)
}

/// Build a [`HeaderData`] dictionary from a [`DynArray`].
///
/// Equivalent to `numpy.lib.format.header_data_from_array_1_0`. Always
/// emits a little-endian descriptor (matching NumPy's convention for
/// freshly-saved `.npy` files) and `fortran_order = false` (ferray
/// arrays are always C-contiguous at the public boundary).
///
/// # Errors
/// Returns `FerrayError::InvalidDtype` if the array's dtype cannot be
/// rendered as a NumPy descriptor.
pub fn header_data_from_array_1_0(array: &DynArray) -> FerrayResult<HeaderData> {
    let descr = dtype_to_descr(array.dtype(), Endianness::Little)?;
    Ok(HeaderData {
        descr,
        fortran_order: false,
        shape: array.shape().to_vec(),
    })
}

/// Read an `.npy`-formatted array from `reader` and return it as a
/// [`DynArray`].
///
/// Standalone equivalent of `numpy.lib.format.read_array`. Internally
/// delegates to [`crate::npy::load_dynamic_from_reader`].
///
/// # Errors
/// Returns errors from the underlying NPY parser (bad magic, dtype
/// mismatch, truncated stream, etc.).
pub fn read_array<R: std::io::Read>(reader: &mut R) -> FerrayResult<DynArray> {
    crate::npy::load_dynamic_from_reader(reader)
}

/// Write a [`DynArray`] to `writer` as an `.npy` v1.0 stream.
///
/// Standalone equivalent of `numpy.lib.format.write_array`. Internally
/// delegates to [`crate::npy::save_dynamic_to_writer`].
///
/// # Errors
/// Returns errors from the underlying NPY writer (I/O failure, dtype
/// not encodable as a NumPy descriptor, etc.).
pub fn write_array<W: std::io::Write>(writer: &mut W, array: &DynArray) -> FerrayResult<()> {
    crate::npy::save_dynamic_to_writer(writer, array)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::{Array, IxDyn};

    #[test]
    fn magic_bytes() {
        assert_eq!(NPY_MAGIC.len(), NPY_MAGIC_LEN);
        assert_eq!(NPY_MAGIC[0], 0x93);
        assert_eq!(&NPY_MAGIC[1..], b"NUMPY");
    }

    #[test]
    fn memmap_mode_variants() {
        let modes = [
            MemmapMode::ReadOnly,
            MemmapMode::ReadWrite,
            MemmapMode::CopyOnWrite,
        ];
        assert_eq!(modes.len(), 3);
        assert_ne!(MemmapMode::ReadOnly, MemmapMode::ReadWrite);
    }

    #[test]
    fn descr_to_dtype_basic() {
        assert_eq!(descr_to_dtype("<f8").unwrap(), DType::F64);
        assert_eq!(descr_to_dtype("<f4").unwrap(), DType::F32);
        assert_eq!(descr_to_dtype("<i4").unwrap(), DType::I32);
        assert_eq!(descr_to_dtype("|b1").unwrap(), DType::Bool);
    }

    #[test]
    fn descr_to_dtype_invalid_errs() {
        assert!(descr_to_dtype("<garbage").is_err());
    }

    #[test]
    fn header_data_from_array_basic() {
        let arr: Array<f64, IxDyn> = Array::from_vec(IxDyn::new(&[2, 3]), vec![1.0; 6]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let h = header_data_from_array_1_0(&dyn_arr).unwrap();
        assert_eq!(h.descr, "<f8");
        assert!(!h.fortran_order);
        assert_eq!(h.shape, vec![2, 3]);
    }

    #[test]
    fn read_array_write_array_roundtrip() {
        let arr: Array<i32, IxDyn> = Array::from_vec(IxDyn::new(&[3]), vec![10, 20, 30]).unwrap();
        let dyn_arr: DynArray = arr.into();
        let mut buf = Vec::<u8>::new();
        write_array(&mut buf, &dyn_arr).unwrap();
        // Re-read via read_array.
        let mut cursor = std::io::Cursor::new(buf);
        let restored = read_array(&mut cursor).unwrap();
        assert_eq!(restored.shape(), &[3]);
        assert_eq!(restored.dtype(), DType::I32);
    }
}
