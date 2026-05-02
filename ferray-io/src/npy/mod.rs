// ferray-io: .npy file I/O
//
// REQ-1: save(path, &array) writes .npy format
// REQ-2: load::<T, D>(path) reads .npy and returns Result<Array<T, D>, FerrayError>
// REQ-3: load_dynamic(path) reads .npy and returns Result<DynArray, FerrayError>
// REQ-6: Support format versions 1.0, 2.0, 3.0
// REQ-12: Support reading/writing both little-endian and big-endian

pub mod dtype_parse;
pub mod header;

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use ferray_core::Array;
use ferray_core::dimension::{Dimension, IxDyn};
use ferray_core::dtype::{DType, Element};
use ferray_core::dynarray::DynArray;
use ferray_core::error::{FerrayError, FerrayResult};
use ferray_core::record::FerrayRecord;

use self::dtype_parse::Endianness;

/// Compute the total number of elements from a shape, using checked
/// multiplication to guard against overflow from untrusted `.npy` files.
pub(crate) fn checked_total_elements(shape: &[usize]) -> FerrayResult<usize> {
    shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            FerrayError::io_error("shape overflow: total elements exceed usize::MAX")
        })
    })
}

/// Save an array to a `.npy` file.
///
/// The file is written in native byte order with C (row-major) layout.
///
/// # Errors
/// Returns `FerrayError::IoError` if the file cannot be created or written.
pub fn save<T: Element + NpyElement, D: Dimension, P: AsRef<Path>>(
    path: P,
    array: &Array<T, D>,
) -> FerrayResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut writer = BufWriter::new(file);
    save_to_writer(&mut writer, array)
}

/// Save an array to a writer in `.npy` format.
///
/// If the array is C-contiguous, its data buffer is written directly.
/// Otherwise, elements are iterated in logical (row-major) order and
/// written individually — this handles transposed, sliced, or
/// otherwise non-contiguous arrays transparently.
pub fn save_to_writer<T: Element + NpyElement, D: Dimension, W: Write>(
    writer: &mut W,
    array: &Array<T, D>,
) -> FerrayResult<()> {
    let fortran_order = false;
    header::write_header(writer, T::dtype(), array.shape(), fortran_order)?;

    // Write data — fast path for contiguous, fallback for strided
    if let Some(slice) = array.as_slice() {
        T::write_slice(slice, writer)?;
    } else {
        // Non-contiguous: collect into logical order and write
        let data: Vec<T> = array.iter().cloned().collect();
        T::write_slice(&data, writer)?;
    }

    writer.flush()?;
    Ok(())
}

/// Load an array from a `.npy` file with compile-time type and dimension.
///
/// # Security note (#499)
/// Unlike numpy's `np.load`, ferray-io does **not** support pickle.
/// The `.npy` v1/v2/v3 binary format with primitive dtypes is the
/// only payload type understood; pickled object arrays
/// (`dtype=object`) are rejected at the dtype-parse stage with an
/// `InvalidDtype` error. This rules out the arbitrary-code-execution
/// vector that motivates numpy's `allow_pickle=False` default —
/// loading an untrusted `.npy` file with ferray-io can fail or
/// surface non-finite numbers, but cannot execute attacker-supplied
/// code.
///
/// # Errors
/// - Returns `FerrayError::InvalidDtype` if the file's dtype doesn't match `T`,
///   or if the file's dtype is unsupported (including `object`/pickle).
/// - Returns `FerrayError::ShapeMismatch` if the file's shape doesn't match `D`.
/// - Returns `FerrayError::IoError` on file read failures.
pub fn load<T: Element + NpyElement, D: Dimension, P: AsRef<Path>>(
    path: P,
) -> FerrayResult<Array<T, D>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to open file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    load_from_reader(&mut reader)
}

/// Load an array from a reader in `.npy` format with compile-time type.
pub fn load_from_reader<T: Element + NpyElement, D: Dimension, R: Read>(
    reader: &mut R,
) -> FerrayResult<Array<T, D>> {
    let hdr = header::read_header(reader)?;

    // Check dtype matches T
    if hdr.dtype != T::dtype() {
        return Err(FerrayError::invalid_dtype(format!(
            "expected dtype {:?} for type {}, but file has {:?}",
            T::dtype(),
            std::any::type_name::<T>(),
            hdr.dtype,
        )));
    }

    // Check dimension compatibility
    if let Some(ndim) = D::NDIM {
        if ndim != hdr.shape.len() {
            return Err(FerrayError::shape_mismatch(format!(
                "expected {} dimensions, but file has {} (shape {:?})",
                ndim,
                hdr.shape.len(),
                hdr.shape,
            )));
        }
    }

    let total_elements = checked_total_elements(&hdr.shape)?;
    let data = T::read_vec(reader, total_elements, hdr.endianness)?;

    let dim = build_dimension::<D>(&hdr.shape)?;

    if hdr.fortran_order {
        Array::from_vec_f(dim, data)
    } else {
        Array::from_vec(dim, data)
    }
}

/// Load a structured-dtype `.npy` file into an `Array<T, D>` where
/// `T: FerrayRecord` (#742).
///
/// The file's structured dtype must match `T`'s field descriptors
/// exactly: same number of fields, same names, same per-field
/// dtype, same offsets, same record size. Mismatches produce a
/// `FerrayError::InvalidDtype` describing the conflict.
///
/// On match, the raw byte block is read into a `Vec<u8>` and
/// reinterpreted as `Vec<T>` via the record's known size and
/// `#[repr(C)]` layout. The resulting array borrows the same
/// row-major / fortran-order convention as numeric loads.
///
/// # Safety
/// `T: FerrayRecord` carries an unsafe-trait contract that the
/// type is `#[repr(C)]` with no padding outside the field
/// descriptors. The reinterpretation only proceeds after
/// validating the file's descriptors match.
///
/// # Errors
/// - `FerrayError::InvalidDtype` if the file doesn't carry a
///   structured dtype, or if its field layout disagrees with
///   `T::field_descriptors()`.
/// - `FerrayError::IoError` on read failure or short data.
/// - `FerrayError::ShapeMismatch` if `D` is fixed-rank and the
///   file's shape rank disagrees.
pub fn load_record<T: FerrayRecord + Element, D: Dimension, P: AsRef<Path>>(
    path: P,
) -> FerrayResult<Array<T, D>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to open file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    load_record_from_reader(&mut reader)
}

/// Load a structured-dtype array from a reader into a
/// `FerrayRecord`-typed array.
///
/// # Errors
/// Same as [`load_record`].
pub fn load_record_from_reader<T: FerrayRecord + Element, D: Dimension, R: Read>(
    reader: &mut R,
) -> FerrayResult<Array<T, D>> {
    let hdr = header::read_header(reader)?;
    let DType::Struct(file_fields) = hdr.dtype else {
        return Err(FerrayError::invalid_dtype(format!(
            "load_record: file has dtype {:?}, expected a structured dtype",
            hdr.dtype
        )));
    };
    let target_fields = T::field_descriptors();
    if file_fields.len() != target_fields.len() {
        return Err(FerrayError::invalid_dtype(format!(
            "load_record: file has {} fields, target type {} has {}",
            file_fields.len(),
            std::any::type_name::<T>(),
            target_fields.len()
        )));
    }
    for (i, (a, b)) in file_fields.iter().zip(target_fields.iter()).enumerate() {
        if a.name != b.name || a.dtype != b.dtype || a.offset != b.offset || a.size != b.size {
            return Err(FerrayError::invalid_dtype(format!(
                "load_record: field {i} mismatch — file {:?} vs target {:?}",
                a, b
            )));
        }
    }
    let record_size = T::record_size();
    if let Some(ndim) = D::NDIM
        && ndim != hdr.shape.len()
    {
        return Err(FerrayError::shape_mismatch(format!(
            "expected {} dimensions, but file has {} (shape {:?})",
            ndim,
            hdr.shape.len(),
            hdr.shape,
        )));
    }
    let total = checked_total_elements(&hdr.shape)?;
    let byte_len = total
        .checked_mul(record_size)
        .ok_or_else(|| FerrayError::io_error("structured payload size overflows usize"))?;
    let mut bytes = vec![0u8; byte_len];
    reader
        .read_exact(&mut bytes)
        .map_err(|e| FerrayError::io_error(format!("failed to read structured data: {e}")))?;
    // Reinterpret the byte buffer as Vec<T>. SAFETY: validated
    // above that the file's structured layout matches T's field
    // descriptors; T: FerrayRecord asserts #[repr(C)] with the
    // exact field offsets / sizes / dtypes that produced the bytes.
    debug_assert_eq!(bytes.len(), total * record_size);
    let mut bytes = std::mem::ManuallyDrop::new(bytes);
    let data = unsafe { Vec::<T>::from_raw_parts(bytes.as_mut_ptr().cast::<T>(), total, total) };
    let dim = build_dimension::<D>(&hdr.shape)?;
    if hdr.fortran_order {
        Array::from_vec_f(dim, data)
    } else {
        Array::from_vec(dim, data)
    }
}

/// Save a `FerrayRecord`-typed array to a structured-dtype `.npy`
/// file (#742). The structured descriptor is built from
/// `T::field_descriptors()`; the raw record bytes are written
/// straight from the array's contiguous slice.
///
/// # Errors
/// `FerrayError::IoError` on file create / write failure.
pub fn save_record<T: FerrayRecord + Element, D: Dimension, P: AsRef<Path>>(
    path: P,
    array: &Array<T, D>,
) -> FerrayResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut writer = BufWriter::new(file);
    save_record_to_writer(&mut writer, array)?;
    writer
        .flush()
        .map_err(|e| FerrayError::io_error(e.to_string()))?;
    Ok(())
}

/// Save a `FerrayRecord` array to a writer. See [`save_record`].
///
/// # Errors
/// Same as [`save_record`].
pub fn save_record_to_writer<T: FerrayRecord + Element, D: Dimension, W: Write>(
    writer: &mut W,
    array: &Array<T, D>,
) -> FerrayResult<()> {
    let dtype = DType::Struct(T::field_descriptors());
    header::write_header(writer, dtype, array.shape(), false)?;
    let slice = array
        .as_slice()
        .ok_or_else(|| FerrayError::io_error("save_record requires a contiguous array"))?;
    let byte_len = slice.len() * T::record_size();
    let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), byte_len) };
    writer
        .write_all(bytes)
        .map_err(|e| FerrayError::io_error(e.to_string()))?;
    Ok(())
}

/// Load a `.npy` file with runtime type dispatch.
///
/// Returns a `DynArray` whose variant corresponds to the file's dtype.
///
/// # Errors
/// Returns errors on I/O failures or unsupported dtypes.
pub fn load_dynamic<P: AsRef<Path>>(path: P) -> FerrayResult<DynArray> {
    let file = File::open(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to open file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut reader = BufReader::new(file);
    load_dynamic_from_reader(&mut reader)
}

/// Load a `.npy` from a reader with runtime type dispatch.
pub fn load_dynamic_from_reader<R: Read>(reader: &mut R) -> FerrayResult<DynArray> {
    let hdr = header::read_header(reader)?;
    let total = checked_total_elements(&hdr.shape)?;
    let dim = IxDyn::new(&hdr.shape);

    macro_rules! load_typed {
        ($ty:ty, $variant:ident) => {{
            let data = <$ty as NpyElement>::read_vec(reader, total, hdr.endianness)?;
            let arr = if hdr.fortran_order {
                Array::<$ty, IxDyn>::from_vec_f(dim, data)?
            } else {
                Array::<$ty, IxDyn>::from_vec(dim, data)?
            };
            Ok(DynArray::$variant(arr))
        }};
    }

    match hdr.dtype {
        DType::Bool => load_typed!(bool, Bool),
        DType::U8 => load_typed!(u8, U8),
        DType::U16 => load_typed!(u16, U16),
        DType::U32 => load_typed!(u32, U32),
        DType::U64 => load_typed!(u64, U64),
        DType::U128 => load_typed!(u128, U128),
        DType::I8 => load_typed!(i8, I8),
        DType::I16 => load_typed!(i16, I16),
        DType::I32 => load_typed!(i32, I32),
        DType::I64 => load_typed!(i64, I64),
        DType::I128 => load_typed!(i128, I128),
        #[cfg(feature = "f16")]
        DType::F16 => load_typed!(half::f16, F16),
        DType::F32 => load_typed!(f32, F32),
        DType::F64 => load_typed!(f64, F64),
        #[cfg(feature = "bf16")]
        DType::BF16 => load_typed!(half::bf16, BF16),
        DType::Complex32 => {
            load_complex32_dynamic(reader, total, dim, hdr.fortran_order, hdr.endianness)
        }
        DType::Complex64 => {
            load_complex64_dynamic(reader, total, dim, hdr.fortran_order, hdr.endianness)
        }
        DType::DateTime64(unit) => {
            let data = <ferray_core::dtype::DateTime64 as NpyElement>::read_vec(
                reader,
                total,
                hdr.endianness,
            )?;
            let arr = if hdr.fortran_order {
                Array::<ferray_core::dtype::DateTime64, IxDyn>::from_vec_f(dim, data)?
            } else {
                Array::<ferray_core::dtype::DateTime64, IxDyn>::from_vec(dim, data)?
            };
            Ok(DynArray::DateTime64(arr, unit))
        }
        DType::Timedelta64(unit) => {
            let data = <ferray_core::dtype::Timedelta64 as NpyElement>::read_vec(
                reader,
                total,
                hdr.endianness,
            )?;
            let arr = if hdr.fortran_order {
                Array::<ferray_core::dtype::Timedelta64, IxDyn>::from_vec_f(dim, data)?
            } else {
                Array::<ferray_core::dtype::Timedelta64, IxDyn>::from_vec(dim, data)?
            };
            Ok(DynArray::Timedelta64(arr, unit))
        }
        _ => Err(FerrayError::invalid_dtype(format!(
            "unsupported dtype {:?} for .npy loading",
            hdr.dtype
        ))),
    }
}

/// Read complex64 (Complex<f32>) data via raw bytes.
fn load_complex32_dynamic<R: Read>(
    reader: &mut R,
    total: usize,
    dim: IxDyn,
    fortran_order: bool,
    endian: Endianness,
) -> FerrayResult<DynArray> {
    let byte_count = total * 8;
    let mut raw = vec![0u8; byte_count];
    reader.read_exact(&mut raw)?;

    if endian.needs_swap() {
        for chunk in raw.chunks_exact_mut(4) {
            chunk.reverse();
        }
    }

    load_complex32_from_bytes_copy(&raw, total, dim, fortran_order)
}

/// Build a Complex32 `DynArray` from raw bytes, respecting Fortran order.
fn load_complex32_from_bytes_copy(
    bytes: &[u8],
    total: usize,
    dim: IxDyn,
    fortran_order: bool,
) -> FerrayResult<DynArray> {
    use num_complex::Complex;

    // Parse raw bytes into Vec<Complex<f32>> by reading f32 pairs
    let mut data = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(8) {
        let re = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let im = f32::from_ne_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
        data.push(Complex::new(re, im));
    }

    let arr = if fortran_order {
        Array::<Complex<f32>, IxDyn>::from_vec_f(dim, data)?
    } else {
        Array::<Complex<f32>, IxDyn>::from_vec(dim, data)?
    };
    Ok(DynArray::Complex32(arr))
}

/// Read complex128 (Complex<f64>) data via raw bytes.
fn load_complex64_dynamic<R: Read>(
    reader: &mut R,
    total: usize,
    dim: IxDyn,
    fortran_order: bool,
    endian: Endianness,
) -> FerrayResult<DynArray> {
    let byte_count = total * 16;
    let mut raw = vec![0u8; byte_count];
    reader.read_exact(&mut raw)?;

    if endian.needs_swap() {
        for chunk in raw.chunks_exact_mut(8) {
            chunk.reverse();
        }
    }

    load_complex64_from_bytes_copy(&raw, total, dim, fortran_order)
}

fn load_complex64_from_bytes_copy(
    bytes: &[u8],
    total: usize,
    dim: IxDyn,
    fortran_order: bool,
) -> FerrayResult<DynArray> {
    use num_complex::Complex;

    // Parse raw bytes into Vec<Complex<f64>> by reading f64 pairs
    let mut data = Vec::with_capacity(total);
    for chunk in bytes.chunks_exact(16) {
        let re = f64::from_ne_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]);
        let im = f64::from_ne_bytes([
            chunk[8], chunk[9], chunk[10], chunk[11], chunk[12], chunk[13], chunk[14], chunk[15],
        ]);
        data.push(Complex::new(re, im));
    }

    let arr = if fortran_order {
        Array::<Complex<f64>, IxDyn>::from_vec_f(dim, data)?
    } else {
        Array::<Complex<f64>, IxDyn>::from_vec(dim, data)?
    };
    Ok(DynArray::Complex64(arr))
}

/// Save a `DynArray` to a `.npy` file.
pub fn save_dynamic<P: AsRef<Path>>(path: P, array: &DynArray) -> FerrayResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create file '{}': {e}",
            path.as_ref().display()
        ))
    })?;
    let mut writer = BufWriter::new(file);
    save_dynamic_to_writer(&mut writer, array)
}

/// Save a `DynArray` to a writer in `.npy` format.
pub fn save_dynamic_to_writer<W: Write>(writer: &mut W, array: &DynArray) -> FerrayResult<()> {
    macro_rules! save_typed {
        ($arr:expr, $dtype:expr, $ty:ty) => {{
            header::write_header(writer, $dtype, $arr.shape(), false)?;
            if let Some(s) = $arr.as_slice() {
                <$ty as NpyElement>::write_slice(s, writer)?;
            } else {
                let data: Vec<$ty> = $arr.iter().cloned().collect();
                <$ty as NpyElement>::write_slice(&data, writer)?;
            }
        }};
    }

    match array {
        DynArray::Bool(a) => save_typed!(a, DType::Bool, bool),
        DynArray::U8(a) => save_typed!(a, DType::U8, u8),
        DynArray::U16(a) => save_typed!(a, DType::U16, u16),
        DynArray::U32(a) => save_typed!(a, DType::U32, u32),
        DynArray::U64(a) => save_typed!(a, DType::U64, u64),
        DynArray::U128(a) => save_typed!(a, DType::U128, u128),
        DynArray::I8(a) => save_typed!(a, DType::I8, i8),
        DynArray::I16(a) => save_typed!(a, DType::I16, i16),
        DynArray::I32(a) => save_typed!(a, DType::I32, i32),
        DynArray::I64(a) => save_typed!(a, DType::I64, i64),
        DynArray::I128(a) => save_typed!(a, DType::I128, i128),
        #[cfg(feature = "f16")]
        DynArray::F16(a) => save_typed!(a, DType::F16, half::f16),
        DynArray::F32(a) => save_typed!(a, DType::F32, f32),
        DynArray::F64(a) => save_typed!(a, DType::F64, f64),
        #[cfg(feature = "bf16")]
        DynArray::BF16(a) => save_typed!(a, DType::BF16, half::bf16),
        DynArray::Complex32(a) => {
            header::write_header(writer, DType::Complex32, a.shape(), false)?;
            save_complex_raw(a.as_slice(), 8, writer)?;
        }
        DynArray::Complex64(a) => {
            header::write_header(writer, DType::Complex64, a.shape(), false)?;
            save_complex_raw(a.as_slice(), 16, writer)?;
        }
        _ => {
            return Err(FerrayError::invalid_dtype(
                "unsupported DynArray variant for .npy saving",
            ));
        }
    }

    writer.flush()?;
    Ok(())
}

/// Write complex array data as raw bytes without naming the Complex type.
/// `elem_size` is the total size per element (8 for Complex<f32>, 16 for Complex<f64>).
fn save_complex_raw<T, W: Write>(
    slice_opt: Option<&[T]>,
    elem_size: usize,
    writer: &mut W,
) -> FerrayResult<()> {
    let slice = slice_opt
        .ok_or_else(|| FerrayError::io_error("cannot save non-contiguous complex array"))?;
    let byte_len = slice.len() * elem_size;
    let bytes = unsafe { std::slice::from_raw_parts(slice.as_ptr().cast::<u8>(), byte_len) };
    writer.write_all(bytes)?;
    Ok(())
}

/// Build a dimension value of type `D` from a shape slice.
///
/// Delegates to `Dimension::from_dim_slice` (#236), which is the
/// trait's canonical inverse of `as_slice` and already supports
/// every fixed rank Ix0..Ix6 plus IxDyn. The previous implementation
/// boxed each candidate into `Box<dyn Any>` and downcast against
/// `TypeId::of::<D>()` — that path required a heap allocation per
/// load and broke when the dimension trait grew new variants.
fn build_dimension<D: Dimension>(shape: &[usize]) -> FerrayResult<D> {
    if let Some(ndim) = D::NDIM {
        if shape.len() != ndim {
            return Err(FerrayError::shape_mismatch(format!(
                "expected {ndim} dimensions, got {}",
                shape.len()
            )));
        }
    }
    D::from_dim_slice(shape).ok_or_else(|| {
        FerrayError::shape_mismatch(format!("shape {shape:?} does not match dimension type"))
    })
}

// ---------------------------------------------------------------------------
// NpyElement trait -- sealed, provides binary read/write for each element type
// ---------------------------------------------------------------------------

/// Trait for element types that support .npy binary serialization.
///
/// This is sealed and implemented for all primitive `Element` types
/// (excluding Complex, which is handled via raw byte I/O in the dynamic path).
pub trait NpyElement: Element + private::NpySealed {
    /// Write a contiguous slice of elements to a writer in native byte order.
    fn write_slice<W: Write>(data: &[Self], writer: &mut W) -> FerrayResult<()>;

    /// Read `count` elements from a reader, applying byte-swapping if needed.
    fn read_vec<R: Read>(
        reader: &mut R,
        count: usize,
        endian: Endianness,
    ) -> FerrayResult<Vec<Self>>;
}

mod private {
    pub trait NpySealed {}
}

// ---------------------------------------------------------------------------
// Macro for implementing NpyElement for primitive numeric types
// ---------------------------------------------------------------------------

macro_rules! impl_npy_element {
    ($ty:ty, $size:expr) => {
        impl private::NpySealed for $ty {}

        impl NpyElement for $ty {
            fn write_slice<W: Write>(data: &[$ty], writer: &mut W) -> FerrayResult<()> {
                // Bulk write: reinterpret the typed slice as raw bytes and write
                // in a single call. This is safe because the data is contiguous
                // and we're writing in native byte order.
                let byte_len = data.len() * $size;
                // SAFETY: &[T] is contiguous and properly aligned; reinterpreting
                // as &[u8] of length len*size_of::<T> is valid for any Copy type.
                let bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
                writer.write_all(bytes)?;
                Ok(())
            }

            fn read_vec<R: Read>(
                reader: &mut R,
                count: usize,
                endian: Endianness,
            ) -> FerrayResult<Vec<$ty>> {
                if !endian.needs_swap() {
                    // Fast path: native endianness — bulk read and reinterpret.
                    let byte_len = count * $size;
                    let mut raw = vec![0u8; byte_len];
                    reader.read_exact(&mut raw)?;
                    let mut result = Vec::with_capacity(count);
                    for chunk in raw.chunks_exact($size) {
                        let arr: [u8; $size] = chunk.try_into().unwrap();
                        result.push(<$ty>::from_ne_bytes(arr));
                    }
                    Ok(result)
                } else {
                    // Byte-swap path: read all bytes, then swap+convert per element.
                    let byte_len = count * $size;
                    let mut raw = vec![0u8; byte_len];
                    reader.read_exact(&mut raw)?;
                    let mut result = Vec::with_capacity(count);
                    for chunk in raw.chunks_exact_mut($size) {
                        chunk.reverse();
                        let arr: [u8; $size] = chunk.try_into().unwrap();
                        result.push(<$ty>::from_ne_bytes(arr));
                    }
                    Ok(result)
                }
            }
        }
    };
}

// Bool -- special case
impl private::NpySealed for bool {}

impl NpyElement for bool {
    fn write_slice<W: Write>(data: &[Self], writer: &mut W) -> FerrayResult<()> {
        // Stage one u8 per bool into a single contiguous buffer, then
        // emit one bulk write_all instead of one syscall per element
        // (#234). For very large bool arrays the staging cost is a
        // memory copy that's far cheaper than per-element I/O.
        let bytes: Vec<u8> = data.iter().map(|&v| u8::from(v)).collect();
        writer.write_all(&bytes)?;
        Ok(())
    }

    fn read_vec<R: Read>(
        reader: &mut R,
        count: usize,
        _endian: Endianness,
    ) -> FerrayResult<Vec<Self>> {
        // Bulk read all bytes in a single read_exact, then map to bool
        // (#234) — eliminates the per-element read syscall.
        let mut raw = vec![0u8; count];
        reader.read_exact(&mut raw)?;
        Ok(raw.into_iter().map(|b| b != 0).collect())
    }
}

impl_npy_element!(u8, 1);
impl_npy_element!(u16, 2);
impl_npy_element!(u32, 4);
impl_npy_element!(u64, 8);
impl_npy_element!(u128, 16);
impl_npy_element!(i8, 1);
impl_npy_element!(i16, 2);
impl_npy_element!(i32, 4);
impl_npy_element!(i64, 8);
impl_npy_element!(i128, 16);
impl_npy_element!(f32, 4);
impl_npy_element!(f64, 8);

// Complex<f32> / Complex<f64> — typed npy I/O (#498). The default
// macro can't be reused because num_complex::Complex<T> doesn't expose
// `from_ne_bytes`; we read/write the two halves explicitly.
macro_rules! impl_npy_element_complex {
    ($float:ty, $size:expr) => {
        impl private::NpySealed for num_complex::Complex<$float> {}

        impl NpyElement for num_complex::Complex<$float> {
            fn write_slice<W: Write>(
                data: &[num_complex::Complex<$float>],
                writer: &mut W,
            ) -> FerrayResult<()> {
                // Re/Im share the same alignment as a Complex<T>, so the
                // raw byte view round-trips on every supported target.
                let byte_len = data.len() * $size * 2;
                let bytes =
                    unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), byte_len) };
                writer.write_all(bytes)?;
                Ok(())
            }

            fn read_vec<R: Read>(
                reader: &mut R,
                count: usize,
                endian: Endianness,
            ) -> FerrayResult<Vec<num_complex::Complex<$float>>> {
                let byte_len = count * $size * 2;
                let mut raw = vec![0u8; byte_len];
                reader.read_exact(&mut raw)?;
                let mut result = Vec::with_capacity(count);
                for chunk in raw.chunks_exact_mut($size * 2) {
                    let (re_bytes, im_bytes) = chunk.split_at_mut($size);
                    if endian.needs_swap() {
                        re_bytes.reverse();
                        im_bytes.reverse();
                    }
                    let re_arr: [u8; $size] = re_bytes.try_into().unwrap();
                    let im_arr: [u8; $size] = im_bytes.try_into().unwrap();
                    result.push(num_complex::Complex::new(
                        <$float>::from_ne_bytes(re_arr),
                        <$float>::from_ne_bytes(im_arr),
                    ));
                }
                Ok(result)
            }
        }
    };
}

impl_npy_element_complex!(f32, 4);
impl_npy_element_complex!(f64, 8);

#[cfg(feature = "f16")]
impl_npy_element!(half::f16, 2);
#[cfg(feature = "bf16")]
impl_npy_element!(half::bf16, 2);

// datetime64 / timedelta64: i64-backed newtypes. Both delegate to i64's
// bulk-bytes read/write path. We can't use `impl_npy_element!` because
// the macro calls `<$ty>::from_ne_bytes`, which only exists on primitive
// integers; the newtype wrapper has to translate by hand.

impl private::NpySealed for ferray_core::dtype::DateTime64 {}
impl private::NpySealed for ferray_core::dtype::Timedelta64 {}

macro_rules! impl_npy_time_element {
    ($ty:path) => {
        impl NpyElement for $ty {
            fn write_slice<W: Write>(data: &[Self], writer: &mut W) -> FerrayResult<()> {
                // Bulk write: the newtype is `#[repr(transparent)]` over
                // i64, so a `&[Self]` is bit-identical to a `&[i64]`,
                // and reinterpreting as `&[u8]` of length `len * 8`
                // gives us a single write_all (#234).
                // SAFETY: Self is repr(transparent) over i64, contiguous
                // and properly aligned for u8 reinterpretation.
                let bytes = unsafe {
                    std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 8)
                };
                writer.write_all(bytes)?;
                Ok(())
            }

            fn read_vec<R: Read>(
                reader: &mut R,
                count: usize,
                endian: Endianness,
            ) -> FerrayResult<Vec<Self>> {
                let mut raw = vec![0u8; count * 8];
                reader.read_exact(&mut raw)?;
                let mut out = Vec::with_capacity(count);
                if endian.needs_swap() {
                    for chunk in raw.chunks_exact_mut(8) {
                        chunk.reverse();
                        let arr: [u8; 8] = chunk.try_into().unwrap();
                        out.push(Self(i64::from_ne_bytes(arr)));
                    }
                } else {
                    for chunk in raw.chunks_exact(8) {
                        let arr: [u8; 8] = chunk.try_into().unwrap();
                        out.push(Self(i64::from_ne_bytes(arr)));
                    }
                }
                Ok(out)
            }
        }
    };
}

impl_npy_time_element!(ferray_core::dtype::DateTime64);
impl_npy_time_element!(ferray_core::dtype::Timedelta64);

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::dimension::{Ix1, Ix2};
    use std::io::Cursor;

    /// Construct a temporary file path inside a fresh `TempDir`.
    /// Caller binds the `TempDir` to keep it alive for the test's
    /// scope; on test completion or panic the directory and its
    /// contents are removed automatically (#244).
    fn temp_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().expect("failed to create test TempDir");
        let path = dir.path().join(name);
        (dir, path)
    }

    // ---- structured-dtype round-trips (#742) ---------------------------

    #[repr(C)]
    #[derive(Clone, Debug, PartialEq)]
    struct Point {
        x: f64,
        y: f64,
        label: i32,
    }

    unsafe impl ferray_core::record::FerrayRecord for Point {
        fn field_descriptors() -> &'static [ferray_core::record::FieldDescriptor] {
            static FIELDS: [ferray_core::record::FieldDescriptor; 3] = [
                ferray_core::record::FieldDescriptor {
                    name: "x",
                    dtype: DType::F64,
                    offset: 0,
                    size: 8,
                },
                ferray_core::record::FieldDescriptor {
                    name: "y",
                    dtype: DType::F64,
                    offset: 8,
                    size: 8,
                },
                ferray_core::record::FieldDescriptor {
                    name: "label",
                    dtype: DType::I32,
                    offset: 16,
                    size: 4,
                },
            ];
            &FIELDS
        }

        fn record_size() -> usize {
            std::mem::size_of::<Self>()
        }
    }

    impl std::fmt::Display for Point {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "Point({}, {}, {})", self.x, self.y, self.label)
        }
    }
    // Element is now provided automatically via the blanket impl
    // for `T: FerrayRecord` in ferray-core (#742).

    #[test]
    fn save_record_then_load_record_roundtrip_1d() {
        let pts = vec![
            Point {
                x: 1.0,
                y: 2.0,
                label: 10,
            },
            Point {
                x: -3.5,
                y: 4.25,
                label: -1,
            },
            Point {
                x: 0.0,
                y: 0.0,
                label: 0,
            },
        ];
        let arr = Array::<Point, Ix1>::from_vec(Ix1::new([3]), pts.clone()).unwrap();
        let (_dir, path) = temp_path("rt_record_1d.npy");
        save_record(&path, &arr).unwrap();
        let loaded: Array<Point, Ix1> = load_record(&path).unwrap();
        assert_eq!(loaded.shape(), &[3]);
        let s = loaded.as_slice().unwrap();
        for (a, b) in s.iter().zip(pts.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn load_record_rejects_wrong_dtype_file() {
        // Save a regular f64 array, then try to load it as Point.
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), vec![1.0, 2.0, 3.0]).unwrap();
        let (_dir, path) = temp_path("wrong_dtype_file.npy");
        save(&path, &arr).unwrap();
        let res: Result<Array<Point, Ix1>, _> = load_record(&path);
        let err = res.unwrap_err();
        assert!(
            err.to_string().contains("expected a structured dtype"),
            "got: {err}"
        );
    }

    // ---- typed Complex round-trips (#498) ------------------------------

    #[test]
    fn roundtrip_complex64_1d() {
        use num_complex::Complex;
        let data: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.0, 0.5),
            Complex::new(0.0, -7.25),
            Complex::new(1e6, -1e6),
        ];
        let arr = Array::<Complex<f32>, Ix1>::from_vec(Ix1::new([4]), data.clone()).unwrap();
        let (_dir, path) = temp_path("rt_c64_1d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<Complex<f32>, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[4]);
        assert_eq!(loaded.as_slice().unwrap(), data.as_slice());
    }

    #[test]
    fn roundtrip_complex128_2d() {
        use num_complex::Complex;
        let data: Vec<Complex<f64>> = (0..6)
            .map(|i| Complex::new(i as f64, -(i as f64)))
            .collect();
        let arr = Array::<Complex<f64>, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();
        let (_dir, path) = temp_path("rt_c128_2d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<Complex<f64>, Ix2> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(loaded.as_slice().unwrap(), data.as_slice());
    }

    #[test]
    fn roundtrip_f64_1d() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([5]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_f64_1d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<f64, Ix1> = load(&path).unwrap();

        assert_eq!(loaded.shape(), &[5]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_f32_2d() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f32, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_f32_2d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<f32, Ix2> = load(&path).unwrap();

        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_i32() {
        let data = vec![10i32, 20, 30, 40];
        let arr = Array::<i32, Ix1>::from_vec(Ix1::new([4]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_i32.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<i32, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_i64() {
        let data = vec![100i64, 200, 300];
        let arr = Array::<i64, Ix1>::from_vec(Ix1::new([3]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_i64.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<i64, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_u8() {
        let data = vec![0u8, 128, 255];
        let arr = Array::<u8, Ix1>::from_vec(Ix1::new([3]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_u8.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<u8, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_bool() {
        let data = vec![true, false, true, true, false];
        let arr = Array::<bool, Ix1>::from_vec(Ix1::new([5]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_bool.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<bool, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn roundtrip_in_memory() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data.clone()).unwrap();

        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded: Array<f64, Ix1> = load_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn load_dynamic_f64() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();

        let (_dir, path) = temp_path("dyn_f64.npy");
        save(&path, &arr).unwrap();
        let dyn_arr = load_dynamic(&path).unwrap();

        assert_eq!(dyn_arr.dtype(), DType::F64);
        assert_eq!(dyn_arr.shape(), &[3]);
    }

    #[test]
    fn load_wrong_dtype_error() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();

        let (_dir, path) = temp_path("wrong_dtype.npy");
        save(&path, &arr).unwrap();

        let result = load::<f32, Ix1, _>(&path);
        assert!(result.is_err());
    }

    #[test]
    fn load_wrong_ndim_error() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let arr = Array::<f64, Ix1>::from_vec(Ix1::new([3]), data).unwrap();

        let (_dir, path) = temp_path("wrong_ndim.npy");
        save(&path, &arr).unwrap();

        let result = load::<f64, Ix2, _>(&path);
        assert!(result.is_err());
    }

    #[test]
    fn roundtrip_dynamic() {
        let data = vec![10i32, 20, 30];
        let arr = Array::<i32, IxDyn>::from_vec(IxDyn::new(&[3]), data.clone()).unwrap();
        let dyn_arr = DynArray::I32(arr);

        let (_dir, path) = temp_path("rt_dynamic.npy");
        save_dynamic(&path, &dyn_arr).unwrap();

        let loaded = load_dynamic(&path).unwrap();
        assert_eq!(loaded.dtype(), DType::I32);
        assert_eq!(loaded.shape(), &[3]);

        let loaded_arr = loaded.try_into_i32().unwrap();
        assert_eq!(loaded_arr.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn load_dynamic_ixdyn() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec(Ix2::new([2, 3]), data.clone()).unwrap();

        let (_dir, path) = temp_path("dyn_ixdyn.npy");
        save(&path, &arr).unwrap();

        // Load as IxDyn
        let loaded: Array<f64, IxDyn> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[test]
    fn load_fortran_order_npy() {
        // Manually construct a .npy file with fortran_order=True.
        // Data for a 2x3 f64 array in column-major order:
        //   logical layout: [[1, 2, 3], [4, 5, 6]]
        //   Fortran storage: [1, 4, 2, 5, 3, 6] (columns first)
        let mut buf = Vec::new();
        // Write header manually
        let header_str = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
        let header_len = header_str.len();
        // Pad to 64-byte alignment (magic=6 + version=2 + hdr_len=2 + header)
        let total_before_pad = 6 + 2 + 2 + header_len;
        let padding = 64 - (total_before_pad % 64);
        let padded_header_len = header_len + padding;

        // Magic
        buf.extend_from_slice(b"\x93NUMPY");
        // Version 1.0
        buf.push(1);
        buf.push(0);
        // Header length (little-endian u16)
        buf.extend_from_slice(&(padded_header_len as u16).to_le_bytes());
        // Header string
        buf.extend_from_slice(header_str.as_bytes());
        // Padding (spaces + newline)
        buf.extend(std::iter::repeat_n(b' ', padding - 1));
        buf.push(b'\n');

        // Data: 6 f64 values in Fortran (column-major) order
        // Logical: [[1, 2, 3], [4, 5, 6]]
        // Fortran storage: col0=[1,4], col1=[2,5], col2=[3,6]
        for &v in &[1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let mut cursor = Cursor::new(buf);
        let loaded: Array<f64, Ix2> = load_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        // The logical data should be [[1,2,3],[4,5,6]] regardless of storage order
        let flat: Vec<f64> = loaded.iter().copied().collect();
        assert_eq!(flat, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn roundtrip_from_vec_f() {
        // Create a Fortran-order array, save, and reload
        let data = vec![1.0_f64, 4.0, 2.0, 5.0, 3.0, 6.0];
        let arr = Array::<f64, Ix2>::from_vec_f(Ix2::new([2, 3]), data).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);

        // Save (will iterate in logical order for non-contiguous)
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded: Array<f64, Ix2> = load_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.shape(), &[2, 3]);
        // Compare logical element order
        let orig: Vec<f64> = arr.iter().copied().collect();
        let back: Vec<f64> = loaded.iter().copied().collect();
        assert_eq!(orig, back);
    }

    // --- Malformed .npy file tests ---

    #[test]
    fn malformed_bad_magic() {
        let data = b"NOT_NPY_FILE_DATA_HERE";
        let mut cursor = Cursor::new(data.to_vec());
        let result = load_from_reader::<f64, Ix1, _>(&mut cursor);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("magic") || msg.contains("not a valid"),
            "got: {msg}"
        );
    }

    #[test]
    fn malformed_truncated_header() {
        // Valid magic + version but truncated before header length
        let mut data = Vec::new();
        data.extend_from_slice(b"\x93NUMPY");
        data.push(1); // version 1
        data.push(0);
        // Missing header length bytes
        let mut cursor = Cursor::new(data);
        let result = load_from_reader::<f64, Ix1, _>(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn malformed_truncated_data() {
        // Valid header but not enough data bytes
        let mut buf = Vec::new();
        let header_str = "{'descr': '<f8', 'fortran_order': False, 'shape': (100,), }";
        let header_len = header_str.len();
        let total = 6 + 2 + 2 + header_len;
        let padding = 64 - (total % 64);
        let padded_len = header_len + padding;

        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(padded_len as u16).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend(std::iter::repeat_n(b' ', padding - 1));
        buf.push(b'\n');
        // Only write 3 f64 values instead of 100
        for &v in &[1.0_f64, 2.0, 3.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        let mut cursor = Cursor::new(buf);
        let result = load_from_reader::<f64, Ix1, _>(&mut cursor);
        assert!(result.is_err(), "should fail with truncated data");
    }

    #[test]
    fn malformed_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(b"\x93NUMPY");
        data.push(9); // version 9.0 — unsupported
        data.push(0);
        data.extend_from_slice(&[10, 0]); // header length
        data.extend_from_slice(b"0123456789"); // dummy header
        let mut cursor = Cursor::new(data);
        let result = load_from_reader::<f64, Ix1, _>(&mut cursor);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("version"), "got: {msg}");
    }

    #[test]
    fn malformed_empty_file() {
        let mut cursor = Cursor::new(Vec::<u8>::new());
        let result = load_from_reader::<f64, Ix1, _>(&mut cursor);
        assert!(result.is_err());
    }

    #[test]
    fn load_big_endian_f64() {
        // Construct a .npy file with big-endian f64 data ('>f8')
        let mut buf = Vec::new();
        let header_str = "{'descr': '>f8', 'fortran_order': False, 'shape': (3,), }";
        let header_len = header_str.len();
        let total = 6 + 2 + 2 + header_len;
        let padding = 64 - (total % 64);
        let padded_len = header_len + padding;

        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(padded_len as u16).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend(std::iter::repeat_n(b' ', padding - 1));
        buf.push(b'\n');

        // Write 3 f64 values in BIG-endian byte order
        for &v in &[1.0_f64, 2.5, -4.75] {
            buf.extend_from_slice(&v.to_be_bytes());
        }

        let mut cursor = Cursor::new(buf);
        let loaded: Array<f64, Ix1> = load_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.shape(), &[3]);
        let data = loaded.as_slice().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-15);
        assert!((data[1] - 2.5).abs() < 1e-15);
        assert!((data[2] - (-4.75)).abs() < 1e-15);
    }

    #[test]
    fn load_big_endian_i32() {
        // Big-endian i32 ('>i4')
        let mut buf = Vec::new();
        let header_str = "{'descr': '>i4', 'fortran_order': False, 'shape': (4,), }";
        let header_len = header_str.len();
        let total = 6 + 2 + 2 + header_len;
        let padding = 64 - (total % 64);
        let padded_len = header_len + padding;

        buf.extend_from_slice(b"\x93NUMPY");
        buf.push(1);
        buf.push(0);
        buf.extend_from_slice(&(padded_len as u16).to_le_bytes());
        buf.extend_from_slice(header_str.as_bytes());
        buf.extend(std::iter::repeat_n(b' ', padding - 1));
        buf.push(b'\n');

        for &v in &[1_i32, -2, 1000, i32::MAX] {
            buf.extend_from_slice(&v.to_be_bytes());
        }

        let mut cursor = Cursor::new(buf);
        let loaded: Array<i32, Ix1> = load_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.shape(), &[4]);
        let data = loaded.as_slice().unwrap();
        assert_eq!(data, &[1, -2, 1000, i32::MAX]);
    }

    // -----------------------------------------------------------------------
    // f16 / bf16 round-trip tests (issue #118)
    // -----------------------------------------------------------------------

    #[cfg(feature = "f16")]
    #[test]
    fn roundtrip_f16_1d() {
        use half::f16;
        let data: Vec<f16> = [0.0, 1.0, -1.5, 2.25, 3.5, -0.125]
            .iter()
            .map(|&v: &f32| f16::from_f32(v))
            .collect();
        let arr = Array::<f16, Ix1>::from_vec(Ix1::new([6]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_f16_1d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<f16, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[6]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn roundtrip_f16_2d() {
        use half::f16;
        let data: Vec<f16> = (0..12)
            .map(|i| f16::from_f32((i as f32).mul_add(0.25, -1.0)))
            .collect();
        let arr = Array::<f16, Ix2>::from_vec(Ix2::new([3, 4]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_f16_2d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<f16, Ix2> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[3, 4]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn roundtrip_f16_dynamic() {
        use half::f16;
        let data: Vec<f16> = (0..8).map(|i| f16::from_f32(i as f32)).collect();
        let arr = Array::<f16, IxDyn>::from_vec(IxDyn::new(&[2, 4]), data.clone()).unwrap();
        let dyn_in = DynArray::F16(arr);

        let (_dir, path) = temp_path("rt_f16_dyn.npy");
        save_dynamic(&path, &dyn_in).unwrap();
        let loaded = load_dynamic(&path).unwrap();
        assert_eq!(loaded.dtype(), DType::F16);
        assert_eq!(loaded.shape(), &[2, 4]);
        match loaded {
            DynArray::F16(a) => assert_eq!(a.as_slice().unwrap(), &data[..]),
            _ => panic!("expected F16 variant"),
        }
    }

    #[cfg(feature = "f16")]
    #[test]
    fn f16_descriptor_is_f2() {
        use half::f16;
        let arr = Array::<f16, Ix1>::from_vec(Ix1::new([2]), vec![f16::ZERO, f16::ONE]).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        // The header must contain the standard NumPy `f2` dtype descriptor
        // so files are interoperable with vanilla NumPy. Inspect only the
        // header region — trailing bytes are raw data, not text.
        let header_len = buf.len().saturating_sub(4); // strip the 4 data bytes
        let header = String::from_utf8_lossy(&buf[..header_len]);
        assert!(
            header.contains("f2"),
            "expected 'f2' in header, got: {header}"
        );
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn roundtrip_bf16_1d() {
        use half::bf16;
        let data: Vec<bf16> = [0.0, 1.0, -1.5, 2.25, 3.5, -0.125]
            .iter()
            .map(|&v: &f32| bf16::from_f32(v))
            .collect();
        let arr = Array::<bf16, Ix1>::from_vec(Ix1::new([6]), data.clone()).unwrap();

        let (_dir, path) = temp_path("rt_bf16_1d.npy");
        save(&path, &arr).unwrap();
        let loaded: Array<bf16, Ix1> = load(&path).unwrap();
        assert_eq!(loaded.shape(), &[6]);
        assert_eq!(loaded.as_slice().unwrap(), &data[..]);
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn roundtrip_bf16_dynamic() {
        use half::bf16;
        let data: Vec<bf16> = (0..6).map(|i| bf16::from_f32(i as f32 * 0.5)).collect();
        let arr = Array::<bf16, IxDyn>::from_vec(IxDyn::new(&[2, 3]), data.clone()).unwrap();
        let dyn_in = DynArray::BF16(arr);

        let (_dir, path) = temp_path("rt_bf16_dyn.npy");
        save_dynamic(&path, &dyn_in).unwrap();
        let loaded = load_dynamic(&path).unwrap();
        assert_eq!(loaded.dtype(), DType::BF16);
        assert_eq!(loaded.shape(), &[2, 3]);
        match loaded {
            DynArray::BF16(a) => assert_eq!(a.as_slice().unwrap(), &data[..]),
            _ => panic!("expected BF16 variant"),
        }
    }

    #[cfg(feature = "bf16")]
    #[test]
    fn bf16_descriptor_is_bf16_tag() {
        use half::bf16;
        let arr = Array::<bf16, Ix1>::from_vec(Ix1::new([2]), vec![bf16::ZERO, bf16::ONE]).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        // ferray-specific non-NumPy tag for bfloat16.
        let header_len = buf.len().saturating_sub(4); // strip the 4 data bytes
        let header = String::from_utf8_lossy(&buf[..header_len]);
        assert!(
            header.contains("bf16"),
            "expected 'bf16' in header, got: {header}"
        );
    }

    #[test]
    fn datetime64_npy_roundtrip() {
        use ferray_core::dtype::DateTime64;
        let original = vec![
            DateTime64(0),
            DateTime64::nat(),
            DateTime64(1_700_000_000_000_000_000),
            DateTime64(-1),
        ];
        let arr = Array::<DateTime64, Ix1>::from_vec(Ix1::new([4]), original.clone()).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        // Header should advertise the M8[ns] descriptor.
        let header = String::from_utf8_lossy(&buf);
        assert!(
            header.contains("M8[ns]"),
            "expected 'M8[ns]' in header, got: {header}"
        );

        let mut cursor = std::io::Cursor::new(buf);
        let back: Array<DateTime64, Ix1> = load_from_reader(&mut cursor).unwrap();
        let v: Vec<DateTime64> = back.iter().copied().collect();
        // Direct bit-level equality (the underlying i64s must match).
        for (a, b) in v.iter().zip(original.iter()) {
            assert_eq!(a.0, b.0, "DateTime64 i64 round-trip mismatch");
        }
    }

    #[test]
    fn timedelta64_npy_roundtrip() {
        use ferray_core::dtype::Timedelta64;
        let original = vec![Timedelta64(1000), Timedelta64::nat(), Timedelta64(0)];
        let arr = Array::<Timedelta64, Ix1>::from_vec(Ix1::new([3]), original.clone()).unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();
        let header = String::from_utf8_lossy(&buf);
        assert!(
            header.contains("m8[ns]"),
            "expected 'm8[ns]' in header, got: {header}"
        );

        let mut cursor = std::io::Cursor::new(buf);
        let back: Array<Timedelta64, Ix1> = load_from_reader(&mut cursor).unwrap();
        let v: Vec<Timedelta64> = back.iter().copied().collect();
        for (a, b) in v.iter().zip(original.iter()) {
            assert_eq!(a.0, b.0);
        }
    }

    #[test]
    fn datetime64_dynarray_load_preserves_unit() {
        use ferray_core::dtype::{DateTime64, TimeUnit};
        let arr = Array::<DateTime64, Ix1>::from_vec(
            Ix1::new([3]),
            vec![DateTime64(0), DateTime64(1), DateTime64::nat()],
        )
        .unwrap();
        let mut buf = Vec::new();
        save_to_writer(&mut buf, &arr).unwrap();

        let mut cursor = std::io::Cursor::new(buf);
        let dyn_arr = load_dynamic_from_reader(&mut cursor).unwrap();
        // Unit is preserved (Ns from the Element::dtype default).
        assert_eq!(
            dyn_arr.dtype(),
            ferray_core::DType::DateTime64(TimeUnit::Ns)
        );
        let (typed, unit) = dyn_arr.try_into_datetime64().unwrap();
        assert_eq!(unit, TimeUnit::Ns);
        let vals: Vec<i64> = typed.iter().map(|v| v.0).collect();
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 1);
        assert_eq!(vals[2], i64::MIN); // NaT
    }

    // ----------------------------------------------------------------------
    // Complex roundtrips through save_dynamic / load_dynamic (#241).
    //
    // Complex32/64 take the dedicated raw-bytes path (save_complex_raw +
    // load_complex{32,64}_dynamic) rather than the NpyElement macro path.
    // These tests pin that special handling: header dtype tag, buffer
    // layout (interleaved re/im), and round-trip exactness.
    // ----------------------------------------------------------------------

    #[test]
    fn complex64_dynamic_roundtrip() {
        use num_complex::Complex;
        let data: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 2.0),
            Complex::new(-3.5, 4.25),
            Complex::new(0.0, -7.0),
            Complex::new(f32::INFINITY, f32::NEG_INFINITY),
        ];
        let arr = Array::<Complex<f32>, IxDyn>::from_vec(IxDyn::new(&[4]), data.clone()).unwrap();
        let dyn_in = DynArray::Complex32(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded = load_dynamic_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.dtype(), DType::Complex32);
        assert_eq!(loaded.shape(), &[4]);
        match loaded {
            DynArray::Complex32(a) => {
                let got: Vec<Complex<f32>> = a.iter().copied().collect();
                assert_eq!(got, data);
            }
            _ => panic!("expected Complex32 variant"),
        }
    }

    #[test]
    fn complex128_dynamic_roundtrip() {
        use num_complex::Complex;
        let data: Vec<Complex<f64>> = (0..6)
            .map(|i| Complex::new(i as f64 * 0.5, -i as f64 * 0.25))
            .collect();
        let arr =
            Array::<Complex<f64>, IxDyn>::from_vec(IxDyn::new(&[2, 3]), data.clone()).unwrap();
        let dyn_in = DynArray::Complex64(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded = load_dynamic_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.dtype(), DType::Complex64);
        assert_eq!(loaded.shape(), &[2, 3]);
        match loaded {
            DynArray::Complex64(a) => {
                let got: Vec<Complex<f64>> = a.iter().copied().collect();
                assert_eq!(got, data);
            }
            _ => panic!("expected Complex64 variant"),
        }
    }

    #[test]
    fn complex64_dynamic_header_tag() {
        // The on-disk descr for Complex<f32> must be '<c8' (numpy tag).
        use num_complex::Complex;
        let arr = Array::<Complex<f32>, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        )
        .unwrap();
        let dyn_in = DynArray::Complex32(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();
        // Inspect the header without touching the data: parse via read_header.
        let mut cursor = Cursor::new(buf);
        let hdr = header::read_header(&mut cursor).unwrap();
        assert_eq!(hdr.dtype, DType::Complex32);
        assert_eq!(hdr.shape, vec![2]);
        assert_eq!(hdr.descr, "<c8");
    }

    #[test]
    fn complex128_dynamic_header_tag() {
        use num_complex::Complex;
        let arr = Array::<Complex<f64>, IxDyn>::from_vec(
            IxDyn::new(&[2]),
            vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)],
        )
        .unwrap();
        let dyn_in = DynArray::Complex64(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();
        let mut cursor = Cursor::new(buf);
        let hdr = header::read_header(&mut cursor).unwrap();
        assert_eq!(hdr.dtype, DType::Complex64);
        assert_eq!(hdr.descr, "<c16");
    }

    #[test]
    fn complex64_dynamic_2d_shape() {
        // Multidim shape is preserved through the dynamic path.
        use num_complex::Complex;
        let data: Vec<Complex<f32>> = (0..12)
            .map(|i| Complex::new(i as f32, (i + 100) as f32))
            .collect();
        let arr =
            Array::<Complex<f32>, IxDyn>::from_vec(IxDyn::new(&[3, 4]), data.clone()).unwrap();
        let dyn_in = DynArray::Complex32(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();

        let mut cursor = Cursor::new(buf);
        let loaded = load_dynamic_from_reader(&mut cursor).unwrap();
        assert_eq!(loaded.shape(), &[3, 4]);
        match loaded {
            DynArray::Complex32(a) => {
                let got: Vec<Complex<f32>> = a.iter().copied().collect();
                assert_eq!(got, data);
            }
            _ => panic!("expected Complex32 variant"),
        }
    }

    #[test]
    fn complex128_dynamic_special_values() {
        // NaN + inf + signed zero round-trip bit-exactly through the
        // raw-bytes path.
        use num_complex::Complex;
        let data: Vec<Complex<f64>> = vec![
            Complex::new(f64::NAN, 0.0),
            Complex::new(0.0, f64::NAN),
            Complex::new(f64::INFINITY, f64::NEG_INFINITY),
            Complex::new(0.0, -0.0),
        ];
        let arr = Array::<Complex<f64>, IxDyn>::from_vec(IxDyn::new(&[4]), data.clone()).unwrap();
        let dyn_in = DynArray::Complex64(arr);

        let mut buf = Vec::new();
        save_dynamic_to_writer(&mut buf, &dyn_in).unwrap();
        let mut cursor = Cursor::new(buf);
        let loaded = load_dynamic_from_reader(&mut cursor).unwrap();
        match loaded {
            DynArray::Complex64(a) => {
                let got: Vec<Complex<f64>> = a.iter().copied().collect();
                // NaN != NaN — compare bit patterns instead.
                for (g, e) in got.iter().zip(data.iter()) {
                    assert_eq!(g.re.to_bits(), e.re.to_bits());
                    assert_eq!(g.im.to_bits(), e.im.to_bits());
                }
            }
            _ => panic!("expected Complex64 variant"),
        }
    }
}
