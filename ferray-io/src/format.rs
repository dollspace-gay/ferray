// ferray-io: Format constants and shared types

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
