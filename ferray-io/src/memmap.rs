// ferray-io: Memory-mapped array I/O
//
// REQ-10: memmap::<T>(path, mode) with MemmapMode::ReadOnly, ReadWrite, CopyOnWrite
// REQ-11: Memory-mapped arrays are views into file memory, not owned copies

use std::fs::{File, OpenOptions};
use std::io::{BufReader, Seek};
use std::marker::PhantomData;
use std::path::Path;

use memmap2::{Mmap, MmapMut, MmapOptions};

use ferray_core::Array;
use ferray_core::array::view::ArrayView;
use ferray_core::dimension::IxDyn;
use ferray_core::dtype::Element;
use ferray_core::error::{FerrayError, FerrayResult};

use crate::format::MemmapMode;
use crate::npy::NpyElement;
use crate::npy::checked_total_elements;
use crate::npy::header::{self, NpyHeader};

/// A read-only memory-mapped array backed by a `.npy` file.
///
/// The array data is mapped directly from the file. No copy is made.
/// The data remains valid as long as this struct is alive.
pub struct MemmapArray<T: Element> {
    /// The underlying memory map. The data pointer is always derived
    /// from this on demand (#238) so it can never dangle.
    mmap: Mmap,
    /// Shape of the array.
    shape: Vec<usize>,
    /// Number of elements.
    len: usize,
    /// Marker for the element type.
    _marker: PhantomData<T>,
}

// SAFETY: Mmap is Send + Sync, and we only expose &T views derived
// from the live mmap. PhantomData<T> stays well-defined (T: Element
// requires Send + Sync via the trait bounds).
unsafe impl<T: Element> Send for MemmapArray<T> {}
unsafe impl<T: Element> Sync for MemmapArray<T> {}

impl<T: Element> MemmapArray<T> {
    /// Return the shape of the mapped array.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Pointer to the start of element data, always derived from the
    /// live mmap (#238). The previous design stored a `*const T`
    /// alongside the mmap; refactors that took ownership of `_mmap`
    /// or reordered fields could leave the pointer dangling. Deriving
    /// on demand makes the invariant structural.
    fn data_ptr(&self) -> *const T {
        self.mmap.as_ptr().cast::<T>()
    }

    /// Return the mapped data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: data_ptr is derived from a live Mmap that outlives
        // this borrow; alignment and length are validated at construction.
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len) }
    }

    /// Copy the memory-mapped data into an owned `Array`.
    pub fn to_array(&self) -> FerrayResult<Array<T, IxDyn>> {
        let data = self.as_slice().to_vec();
        Array::from_vec(IxDyn::new(&self.shape), data)
    }

    /// Borrow the memory-mapped data as an `ArrayView<T, IxDyn>` so it
    /// can be passed directly to ferray functions that expect an
    /// `&Array<_>` or `ArrayView<_>` (#496). The view is C-contiguous
    /// row-major and lives as long as the underlying mmap.
    #[must_use]
    pub fn view(&self) -> ArrayView<'_, T, IxDyn> {
        // Row-major strides for the shape: stride[i] = product(shape[i+1..]).
        let ndim = self.shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        // SAFETY: data_ptr() is valid for reads of `len * size_of::<T>()`
        // bytes for the lifetime of `self.mmap`, which transitively
        // outlives the borrow `&self`. Strides describe the same
        // C-contiguous layout the data was written in.
        unsafe { ArrayView::from_shape_ptr(self.data_ptr(), &self.shape, &strides) }
    }
}

/// A read-write memory-mapped array backed by a `.npy` file.
///
/// Modifications to the array data are written back to the underlying file.
pub struct MemmapArrayMut<T: Element> {
    /// The underlying mutable memory map. The data pointer is always
    /// derived from this on demand (#238) so it can never dangle.
    mmap: MmapMut,
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
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data_ptr(&self) -> *const T {
        self.mmap.as_ptr().cast::<T>()
    }

    fn data_ptr_mut(&mut self) -> *mut T {
        self.mmap.as_mut_ptr().cast::<T>()
    }

    /// Return the mapped data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.data_ptr(), self.len) }
    }

    /// Return the mapped data as a mutable slice.
    ///
    /// Modifications will be persisted to the file (for `ReadWrite` mode)
    /// or kept in memory only (for `CopyOnWrite` mode).
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        let len = self.len;
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr_mut(), len) }
    }

    /// Copy the memory-mapped data into an owned `Array`.
    pub fn to_array(&self) -> FerrayResult<Array<T, IxDyn>> {
        let data = self.as_slice().to_vec();
        Array::from_vec(IxDyn::new(&self.shape), data)
    }

    /// Borrow the memory-mapped data as an immutable
    /// `ArrayView<T, IxDyn>`. See [`MemmapArray::view`] for the
    /// rationale (#496).
    #[must_use]
    pub fn view(&self) -> ArrayView<'_, T, IxDyn> {
        let ndim = self.shape.len();
        let mut strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        // SAFETY: same invariants as MemmapArray::view.
        unsafe { ArrayView::from_shape_ptr(self.data_ptr(), &self.shape, &strides) }
    }

    /// Flush changes to disk (only meaningful for `ReadWrite` mode).
    pub fn flush(&self) -> FerrayResult<()> {
        self.mmap
            .flush()
            .map_err(|e| FerrayError::io_error(format!("failed to flush mmap: {e}")))
    }
}

/// Open a `.npy` file as a read-only memory-mapped array.
///
/// The file must contain data in native byte order and C-contiguous layout.
///
/// # Errors
/// - `FerrayError::InvalidDtype` if the file dtype doesn't match `T`.
/// - `FerrayError::IoError` on file or mapping failures.
pub fn memmap_readonly<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
) -> FerrayResult<MemmapArray<T>> {
    let (header, data_offset) = read_npy_header_with_offset(path.as_ref())?;
    validate_dtype::<T>(&header)?;
    validate_native_endian(&header)?;

    let len = checked_total_elements(&header.shape)?;
    let file = File::open(path.as_ref())?;
    let mmap = unsafe {
        MmapOptions::new()
            .offset(data_offset as u64)
            .len(len * std::mem::size_of::<T>())
            .map(&file)
            .map_err(|e| FerrayError::io_error(format!("mmap failed: {e}")))?
    };
    // Validate alignment against the live mmap pointer; we don't store
    // it (#238) so re-derive once here for the alignment check.
    let probe_ptr = mmap.as_ptr().cast::<T>();
    if (probe_ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(FerrayError::io_error(
            "memory-mapped data is not properly aligned for the element type",
        ));
    }

    Ok(MemmapArray {
        mmap,
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
/// - `FerrayError::InvalidDtype` if the file dtype doesn't match `T`.
/// - `FerrayError::IoError` on file or mapping failures.
/// - `FerrayError::InvalidValue` if `mode` is `ReadOnly` (use `memmap_readonly` instead).
pub fn memmap_mut<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
    mode: MemmapMode,
) -> FerrayResult<MemmapArrayMut<T>> {
    if mode == MemmapMode::ReadOnly {
        return Err(FerrayError::invalid_value(
            "use memmap_readonly for read-only access",
        ));
    }

    let (header, data_offset) = read_npy_header_with_offset(path.as_ref())?;
    validate_dtype::<T>(&header)?;
    validate_native_endian(&header)?;

    let len = checked_total_elements(&header.shape)?;
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
                    .map_err(|e| FerrayError::io_error(format!("mmap_mut failed: {e}")))?
            }
        }
        MemmapMode::CopyOnWrite => {
            let file = File::open(path.as_ref())?;
            unsafe {
                MmapOptions::new()
                    .offset(data_offset as u64)
                    .len(data_bytes)
                    .map_copy(&file)
                    .map_err(|e| FerrayError::io_error(format!("mmap copy-on-write failed: {e}")))?
            }
        }
        MemmapMode::ReadOnly => unreachable!(),
    };

    let probe_ptr = mmap.as_ptr().cast::<T>();
    if (probe_ptr as usize) % std::mem::align_of::<T>() != 0 {
        return Err(FerrayError::io_error(
            "memory-mapped data is not properly aligned for the element type",
        ));
    }

    Ok(MemmapArrayMut {
        mmap,
        shape: header.shape,
        len,
        _marker: PhantomData,
    })
}

/// Combined entry point matching `NumPy`'s `memmap` function signature.
///
/// Dispatches to `memmap_readonly` or `memmap_mut` based on `mode`,
/// then copies the mapped data into an owned `Array<T, IxDyn>`.
///
/// **This always copies** because `Array<T, IxDyn>` owns its buffer.
/// For zero-copy access, use [`memmap_readonly`] or [`memmap_mut`]
/// directly and call `.view()` on the result to get an `ArrayView`
/// backed by the mmap (#239, #496).
pub fn open_memmap<T: Element + NpyElement, P: AsRef<Path>>(
    path: P,
    mode: MemmapMode,
) -> FerrayResult<Array<T, IxDyn>> {
    if mode == MemmapMode::ReadOnly {
        let mapped = memmap_readonly::<T, _>(path)?;
        mapped.to_array()
    } else {
        let mapped = memmap_mut::<T, _>(path, mode)?;
        mapped.to_array()
    }
}

/// Read the npy header and compute the data byte offset.
///
/// Uses `stream_position()` after parsing the header to determine the data
/// offset in a single open, avoiding a TOCTOU race from re-opening the file.
fn read_npy_header_with_offset(path: &Path) -> FerrayResult<(NpyHeader, usize)> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let hdr = header::read_header(&mut reader)?;

    // read_header consumes exactly the header bytes; the reader is now
    // positioned at the start of the data section.
    let data_offset = reader
        .stream_position()
        .map_err(|e| FerrayError::io_error(format!("failed to get stream position: {e}")))?
        as usize;

    Ok((hdr, data_offset))
}

fn validate_dtype<T: Element>(header: &NpyHeader) -> FerrayResult<()> {
    if header.dtype != T::dtype() {
        return Err(FerrayError::invalid_dtype(format!(
            "expected dtype {:?} for type {}, but file has {:?}",
            T::dtype(),
            std::any::type_name::<T>(),
            header.dtype,
        )));
    }
    Ok(())
}

fn validate_native_endian(header: &NpyHeader) -> FerrayResult<()> {
    if header.endianness.needs_swap() {
        return Err(FerrayError::io_error(
            "memory-mapped arrays require native byte order; file has non-native endianness",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // Tests assert exact roundtrip equality on hand-picked memmap values.
mod tests {
    use super::*;
    use crate::npy;
    use ferray_core::dimension::Ix1;

    fn test_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ferray_io_mmap_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    fn test_file(name: &str) -> std::path::PathBuf {
        test_dir().join(name)
    }

    #[test]
    fn memmap_readonly_f64() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([5]), data.clone()).unwrap();

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
            let mut mapped = memmap_mut::<f64, _>(&path, MemmapMode::CopyOnWrite).unwrap();
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

    #[test]
    fn memmap_view_borrows_underlying_data() {
        // Issue #496: memmap arrays should expose an ArrayView so they
        // can be passed to ferray functions that take `&Array` /
        // `ArrayView` without an intermediate copy.
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, ferray_core::dimension::Ix2>::from_vec(
            ferray_core::dimension::Ix2::new([2, 3]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_view.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<f64, _>(&path).unwrap();
        let view = mapped.view();
        assert_eq!(view.shape(), &[2, 3]);
        let collected: Vec<f64> = view.iter().copied().collect();
        assert_eq!(collected, data);
        let _ = std::fs::remove_file(&path);
    }

    // ----------------------------------------------------------------------
    // Multidimensional memmap coverage (#243). Existing tests are nearly
    // all 1-D; add 2-D and 3-D paths for readonly, ReadWrite, and view.
    // ----------------------------------------------------------------------

    #[test]
    fn memmap_readonly_2d_shape_and_data() {
        let data: Vec<f64> = (0..12).map(|i| i as f64 + 0.5).collect();
        let arr = Array::<f64, ferray_core::dimension::Ix2>::from_vec(
            ferray_core::dimension::Ix2::new([3, 4]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_ro_2d.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<f64, _>(&path).unwrap();
        assert_eq!(mapped.shape(), &[3, 4]);
        assert_eq!(mapped.as_slice(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_readonly_3d_shape_and_data() {
        let data: Vec<i32> = (0..24).collect();
        let arr = Array::<i32, ferray_core::dimension::Ix3>::from_vec(
            ferray_core::dimension::Ix3::new([2, 3, 4]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_ro_3d.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<i32, _>(&path).unwrap();
        assert_eq!(mapped.shape(), &[2, 3, 4]);
        assert_eq!(mapped.as_slice(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_readwrite_2d_persists() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, ferray_core::dimension::Ix2>::from_vec(
            ferray_core::dimension::Ix2::new([2, 3]),
            data,
        )
        .unwrap();

        let path = test_file("mm_rw_2d.npy");
        npy::save(&path, &arr).unwrap();

        // Modify the (1, 2) element via mmap (row-major linear index 5).
        {
            let mut mapped = memmap_mut::<f64, _>(&path, MemmapMode::ReadWrite).unwrap();
            assert_eq!(mapped.shape(), &[2, 3]);
            mapped.as_slice_mut()[5] = -42.0;
            mapped.flush().unwrap();
        }

        let loaded: Array<f64, ferray_core::dimension::Ix2> = npy::load(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        let row1: Vec<f64> = loaded.iter().copied().collect();
        assert_eq!(row1[5], -42.0);
        // Other elements unchanged.
        assert_eq!(row1[..5], [1.0, 2.0, 3.0, 4.0, 5.0]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_view_3d_strides_match_row_major() {
        // The view's stride pattern must reconstruct the original logical
        // layout for a 3-D array. Stride[k] = product(shape[k+1..]).
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let arr = Array::<f64, ferray_core::dimension::Ix3>::from_vec(
            ferray_core::dimension::Ix3::new([2, 3, 4]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_view_3d.npy");
        npy::save(&path, &arr).unwrap();

        let mapped = memmap_readonly::<f64, _>(&path).unwrap();
        let view = mapped.view();
        assert_eq!(view.shape(), &[2, 3, 4]);
        let collected: Vec<f64> = view.iter().copied().collect();
        assert_eq!(collected, data);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_open_memmap_2d_readonly() {
        // open_memmap (the numpy-shaped entry point) must preserve 2-D
        // shape on the readonly path.
        let data: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
        let arr = Array::<f32, ferray_core::dimension::Ix2>::from_vec(
            ferray_core::dimension::Ix2::new([3, 5]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_open_ro_2d.npy");
        npy::save(&path, &arr).unwrap();

        let loaded = open_memmap::<f32, _>(&path, MemmapMode::ReadOnly).unwrap();
        assert_eq!(loaded.shape(), &[3, 5]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn memmap_copy_on_write_2d_isolates_changes() {
        // CoW changes must stay in memory; the on-disk file is unchanged.
        let data = vec![10.0_f64, 20.0, 30.0, 40.0];
        let arr = Array::<f64, ferray_core::dimension::Ix2>::from_vec(
            ferray_core::dimension::Ix2::new([2, 2]),
            data.clone(),
        )
        .unwrap();

        let path = test_file("mm_cow_2d.npy");
        npy::save(&path, &arr).unwrap();

        {
            let mut mapped = memmap_mut::<f64, _>(&path, MemmapMode::CopyOnWrite).unwrap();
            assert_eq!(mapped.shape(), &[2, 2]);
            mapped.as_slice_mut()[3] = 999.0;
            assert_eq!(mapped.as_slice()[3], 999.0);
        }

        let loaded: Array<f64, ferray_core::dimension::Ix2> = npy::load(&path).unwrap();
        let flat: Vec<f64> = loaded.iter().copied().collect();
        assert_eq!(flat, data);
        let _ = std::fs::remove_file(&path);
    }
}
