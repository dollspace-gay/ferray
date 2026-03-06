// ferrum-io: Memory-mapped array I/O
//
// REQ-10: memmap::<T>(path, mode) with MemmapMode::ReadOnly, ReadWrite, CopyOnWrite
// REQ-11: Memory-mapped arrays are views into file memory, not owned copies

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read};
use std::marker::PhantomData;
use std::path::Path;

use memmap2::{Mmap, MmapMut, MmapOptions};

use ferrum_core::dimension::IxDyn;
use ferrum_core::dtype::Element;
use ferrum_core::error::{FerrumError, FerrumResult};
use ferrum_core::Array;

use crate::format::MemmapMode;
use crate::npy::header::{self, NpyHeader};
use crate::npy::NpyElement;

/// A read-only memory-mapped array backed by a `.npy` file.
///
/// The array data is mapped directly from the file. No copy is made.
/// The data remains valid as long as this struct is alive.
pub struct MemmapArray<T: Element> {
    /// The underlying memory map.
    _mmap: Mmap,
    /// Pointer to the start of element data.
    data_ptr: *const T,
    /// Shape of the array.
    shape: Vec<usize>,
    /// Number of elements.
    len: usize,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

// SAFETY: The underlying Mmap is Send + Sync and the data pointer
// is derived from it. We only provide read access to the data.
unsafe impl<T: Element> Send for MemmapArray<T> {}
unsafe impl<T: Element> Sync for MemmapArray<T> {}

impl<T: Element> MemmapArray<T> {
    /// Return the shape of the mapped array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the mapped data as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: data_ptr points to properly aligned, initialized data
        // within the mmap region, and self.len is validated during construction.
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }

    /// Copy the memory-mapped data into an owned `Array`.
    pub fn to_array(&self) -> FerrumResult<Array<T, IxDyn>> {
        let data = self.as_slice().to_vec();
        Array::from_vec(IxDyn::new(&self.shape), data)
    }
}

/// A read-write memory-mapped array backed by a `.npy` file.
///
/// Modifications to the array data are written back to the underlying file.
pub struct MemmapArrayMut<T: Element> {
    /// The underlying mutable memory map.
    _mmap: MmapMut,
    /// Pointer to the start of element data.
    data_ptr: *mut T,
    /// Shape of the array.
    shape: Vec<usize>,
    /// Number of elements.
    len: usize,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

unsafe impl<T: Element> Send for MemmapArrayMut<T> {}
unsafe impl<T: Element> Sync for MemmapArrayMut<T> {}

impl<T: Element> MemmapArrayMut<T> {
    /// Return the shape of the mapped array.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the mapped data as a slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr, self.len) }
    }

    /// Return the mapped data as a mutable slice.
    ///
    /// Modifications will be persisted to the file (for ReadWrite mode)
    /// or kept in memory only (for CopyOnWrite mode).
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr, self.len) }
    }

    /// Copy the memory-mapped data into an owned `Array`.
    pub fn to_array(&self) -> FerrumResult<Array<T, IxDyn>> {
        let data = self.as_slice().to_vec();
        Array::from_vec(IxDyn::new(&self.shape), data)
    }

    /// Flush changes to disk (only meaningful for ReadWrite mode).
    pub fn flush(&self) -> FerrumResult<()> {
        self._mmap
            .flush()
            .map_err(|e| FerrumError::io_error(format!("failed to flush mmap: {e}")))
    }
}

/// Open a `.npy` file as a read-only memory-mapped array.
///
/// The file must contain data in native byte order and C-contiguous layout.
///
/// # Errors
/// - `FerrumError::InvalidDtype` if the file dtype doesn't match `T`.
/// - `FerrumError::IoError` on file or mapping failures.
pub fn memmap_readonly<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
) -> FerrumResult<MemmapArray<T>> {
    let (header, data_offset) = read_npy_header_with_offset(path.as_ref())?;
    validate_dtype::<T>(&header)?;
    validate_native_endian(&header)?;

    let file = File::open(path.as_ref())?;
    let mmap = unsafe {
        MmapOptions::new()
            .offset(data_offset as u64)
            .len(header.shape.iter().product::<usize>() * std::mem::size_of::<T>())
            .map(&file)
            .map_err(|e| FerrumError::io_error(format!("mmap failed: {e}")))?
    };

    let len: usize = header.shape.iter().product();
    let data_ptr = mmap.as_ptr() as *const T;

    // Validate alignment
    if (data_ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(FerrumError::io_error(
            "memory-mapped data is not properly aligned for the element type",
        ));
    }

    Ok(MemmapArray {
        _mmap: mmap,
        data_ptr,
        shape: header.shape,
        len,
        _marker: PhantomData,
    })
}

/// Open a `.npy` file as a mutable memory-mapped array.
///
/// # Arguments
/// - `mode`: `MemmapMode::ReadWrite` persists changes to disk.
///   `MemmapMode::CopyOnWrite` keeps changes in memory only.
///
/// # Errors
/// - `FerrumError::InvalidDtype` if the file dtype doesn't match `T`.
/// - `FerrumError::IoError` on file or mapping failures.
/// - `FerrumError::InvalidValue` if `mode` is `ReadOnly` (use `memmap_readonly` instead).
pub fn memmap_mut<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
    mode: MemmapMode,
) -> FerrumResult<MemmapArrayMut<T>> {
    if mode == MemmapMode::ReadOnly {
        return Err(FerrumError::invalid_value(
            "use memmap_readonly for read-only access",
        ));
    }

    let (header, data_offset) = read_npy_header_with_offset(path.as_ref())?;
    validate_dtype::<T>(&header)?;
    validate_native_endian(&header)?;

    let len: usize = header.shape.iter().product();
    let data_bytes = len * std::mem::size_of::<T>();

    let mmap = match mode {
        MemmapMode::ReadWrite => {
            let file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(path.as_ref())?;
            unsafe {
                MmapOptions::new()
                    .offset(data_offset as u64)
                    .len(data_bytes)
                    .map_mut(&file)
                    .map_err(|e| FerrumError::io_error(format!("mmap_mut failed: {e}")))?
            }
        }
        MemmapMode::CopyOnWrite => {
            let file = File::open(path.as_ref())?;
            unsafe {
                MmapOptions::new()
                    .offset(data_offset as u64)
                    .len(data_bytes)
                    .map_copy(&file)
                    .map_err(|e| {
                        FerrumError::io_error(format!("mmap copy-on-write failed: {e}"))
                    })?
            }
        }
        MemmapMode::ReadOnly => unreachable!(),
    };

    let data_ptr = mmap.as_ptr() as *mut T;

    if (data_ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(FerrumError::io_error(
            "memory-mapped data is not properly aligned for the element type",
        ));
    }

    Ok(MemmapArrayMut {
        _mmap: mmap,
        data_ptr,
        shape: header.shape,
        len,
        _marker: PhantomData,
    })
}

/// Combined entry point matching NumPy's `memmap` function signature.
///
/// Dispatches to `memmap_readonly` or `memmap_mut` based on `mode`.
/// For ReadOnly mode, copies the data to an owned array (since the return type
/// must be uniform). For mutable modes, returns the data copied into an owned array
/// after applying the mapping.
///
/// For zero-copy access, use `memmap_readonly` or `memmap_mut` directly.
pub fn open_memmap<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
    mode: MemmapMode,
) -> FerrumResult<Array<T, IxDyn>> {
    match mode {
        MemmapMode::ReadOnly => {
            let mapped = memmap_readonly::<T, _>(path)?;
            mapped.to_array()
        }
        _ => {
            let mapped = memmap_mut::<T, _>(path, mode)?;
            mapped.to_array()
        }
    }
}

/// Read the npy header and compute the data byte offset.
fn read_npy_header_with_offset(path: &Path) -> FerrumResult<(NpyHeader, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let hdr = header::read_header(&mut reader)?;

    // Compute data offset: the reader has consumed the header, so we compute
    // the offset from the version and header_len.
    let preamble_len = crate::format::NPY_MAGIC_LEN + 2; // magic + version bytes
    let header_len_field_size = if hdr.version.0 == 1 { 2 } else { 4 };

    // We need to figure out the total bytes consumed.
    // Re-read the file to get the header length from the raw bytes.
    let file2 = File::open(path)?;
    let mut reader2 = BufReader::new(file2);
    let mut skip = vec![0u8; preamble_len + header_len_field_size];
    reader2.read_exact(&mut skip)?;

    let header_len = if hdr.version.0 == 1 {
        u16::from_le_bytes([skip[preamble_len], skip[preamble_len + 1]]) as usize
    } else {
        u32::from_le_bytes([
            skip[preamble_len],
            skip[preamble_len + 1],
            skip[preamble_len + 2],
            skip[preamble_len + 3],
        ]) as usize
    };

    let data_offset = preamble_len + header_len_field_size + header_len;

    Ok((hdr, data_offset))
}

fn validate_dtype<T: Element>(header: &NpyHeader) -> FerrumResult<()> {
    if header.dtype != T::dtype() {
        return Err(FerrumError::invalid_dtype(format!(
            "expected dtype {:?} for type {}, but file has {:?}",
            T::dtype(),
            std::any::type_name::<T>(),
            header.dtype,
        )));
    }
    Ok(())
}

fn validate_native_endian(header: &NpyHeader) -> FerrumResult<()> {
    if header.endianness.needs_swap() {
        return Err(FerrumError::io_error(
            "memory-mapped arrays require native byte order; file has non-native endianness",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy;
    use ferrum_core::dimension::Ix1;

    fn test_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ferrum_io_mmap_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn test_file(name: &str) -> std::path::PathBuf {
        test_dir().join(name)
    }

    #[test]
    fn memmap_readonly_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let arr =
            Array::<f64, Ix1>::from_vec(Ix1::new([5]), data.clone()).unwrap();

        let path = test_file("mm_ro_f64.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<f64, _>(&path).unwrap();
        assert_eq!(mapped.shape(), &[5]);
        assert_eq!(mapped.as_slice(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_to_array() {
        let data = vec![10i32, 20, 30];
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([3]), data.clone()).unwrap();

        let path = test_file("mm_to_arr.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<i32, _>(&path).unwrap();
        let owned = mapped.to_array().unwrap();
        assert_eq!(owned.shape(), &[3]);
        assert_eq!(owned.as_slice().unwrap(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_readwrite_persist() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();

        let path = test_file("mm_rw.npy");
        npy::save(&path, &arr).unwrap();

        // Modify via mmap
        {
            let mut mapped = memmap_mut::<f64, _>(&path, MemmapMode::ReadWrite).unwrap();
            mapped.as_slice_mut()[0] = 999.0;
            mapped.flush().unwrap();
        }

        // Read back and verify the change persisted
        let loaded: Array<f64, Ix1> = npy::load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap()[0], 999.0);
        assert_eq!(loaded.as_slice().unwrap()[1], 2.0);
        assert_eq!(loaded.as_slice().unwrap()[2], 3.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_copy_on_write() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();

        let path = test_file("mm_cow.npy");
        npy::save(&path, &arr).unwrap();

        // Modify via copy-on-write mmap
        {
            let mut mapped =
                memmap_mut::<f64, _>(&path, MemmapMode::CopyOnWrite).unwrap();
            mapped.as_slice_mut()[0] = 999.0;
            assert_eq!(mapped.as_slice()[0], 999.0);
        }

        // Original file should be unmodified
        let loaded: Array<f64, Ix1> = npy::load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap()[0], 1.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_wrong_dtype_error() {
        let data = vec![1.0_f64, 2.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([2]), data).unwrap();

        let path = test_file("mm_wrong_dt.npy");
        npy::save(&path, &arr).unwrap();

        let result = memmap_readonly::<f32, _>(&path);
        assert!(result.is_err());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_memmap_readonly() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data.clone()).unwrap();

        let path = test_file("mm_open_ro.npy");
        npy::save(&path, &arr).unwrap();

        let loaded = open_memmap::<f64, _>(&path, MemmapMode::ReadOnly).unwrap();
        assert_eq!(loaded.shape(), &[3]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }
}
