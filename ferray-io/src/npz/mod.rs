// ferray-io: .npz file I/O
//
// REQ-4: savez(path, &[("name", DynArray)]) writes multiple arrays to .npz (zip archive)
// REQ-5: savez_compressed(path, ...) writes gzip-compressed .npz

use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use ferray_core::dynarray::DynArray;
use ferray_core::error::{FerrayError, FerrayResult};
use zip::ZipArchive;

use crate::npy;

/// Type-erased reader source backing an [`NpzFile`]. We don't expose
/// the concrete underlying reader to keep the type signature simple.
type NpzReader = Box<dyn ReadSeek + Send>;

/// Convenience trait combining `Read + Seek + Send` so the boxed
/// reader can be stored in a single `Box<dyn ...>`.
trait ReadSeek: Read + std::io::Seek {}
impl<T: Read + std::io::Seek + ?Sized> ReadSeek for T {}

/// Save multiple arrays to an uncompressed `.npz` file (zip archive of `.npy` files).
///
/// Each entry in `arrays` is a `(name, array)` pair. The name is used as the
/// filename inside the archive (`.npy` extension is appended automatically).
///
/// # Errors
/// Returns `FerrayError::IoError` on file creation or write failures.
pub fn savez<P: AsRef<Path>>(path: P, arrays: &[(&str, &DynArray)]) -> FerrayResult<()> {
    savez_impl(path, arrays, zip::CompressionMethod::Stored)
}

/// Save multiple arrays to a gzip-compressed `.npz` file.
///
/// Same as [`savez`] but each entry is individually compressed with DEFLATE.
///
/// # Errors
/// Returns `FerrayError::IoError` on file creation or write failures.
pub fn savez_compressed<P: AsRef<Path>>(path: P, arrays: &[(&str, &DynArray)]) -> FerrayResult<()> {
    savez_impl(path, arrays, zip::CompressionMethod::Deflated)
}

/// Shared implementation for [`savez`] and [`savez_compressed`]; the
/// only difference between the two is the compression method (#233).
fn savez_impl<P: AsRef<Path>>(
    path: P,
    arrays: &[(&str, &DynArray)],
    method: zip::CompressionMethod,
) -> FerrayResult<()> {
    let file = File::create(path.as_ref()).map_err(|e| {
        FerrayError::io_error(format!(
            "failed to create .npz file '{}': {e}",
            path.as_ref().display()
        ))
    })?;

    let mut zip_writer = zip::ZipWriter::new(file);

    for (name, array) in arrays {
        let entry_name = if std::path::Path::new(name)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("npy"))
        {
            name.to_string()
        } else {
            format!("{name}.npy")
        };

        let options = zip::write::SimpleFileOptions::default().compression_method(method);

        zip_writer.start_file(&entry_name, options).map_err(|e| {
            FerrayError::io_error(format!("failed to create zip entry '{entry_name}': {e}"))
        })?;

        npy::save_dynamic_to_writer(&mut zip_writer, array)?;
    }

    zip_writer
        .finish()
        .map_err(|e| FerrayError::io_error(format!("failed to finalize .npz file: {e}")))?;

    Ok(())
}

/// A `.npz` archive opened lazily — entries are decompressed and
/// parsed only when [`get`](Self::get) is called.
///
/// The previous implementation read every entry into a `Vec<Vec<u8>>`
/// up front (#232, #492), which negated the named-access API for
/// large archives where callers only need a few arrays. The reader is
/// kept alive inside the struct so each `get` call can seek back to
/// the requested entry on demand.
pub struct NpzFile {
    archive: ZipArchive<NpzReader>,
    /// Pre-extracted name list (without `.npy` extension) so the
    /// `names`/`len`/`is_empty` queries don't require the lock-step
    /// `&mut self` that `ZipArchive::by_index` requires.
    names: Vec<String>,
}

impl NpzFile {
    /// Open a `.npz` file for lazy access.
    ///
    /// The file's directory is parsed and entry names are cached, but
    /// no array data is decompressed until [`get`](Self::get) is
    /// called for a specific name.
    pub fn open<P: AsRef<Path>>(path: P) -> FerrayResult<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            FerrayError::io_error(format!(
                "failed to open .npz file '{}': {e}",
                path.as_ref().display()
            ))
        })?;
        let reader: NpzReader = Box::new(BufReader::new(file));
        Self::from_boxed_reader(reader)
    }

    /// Read a `.npz` from any `Read + Seek` source.
    pub fn from_reader<R: Read + std::io::Seek + Send + 'static>(reader: R) -> FerrayResult<Self> {
        Self::from_boxed_reader(Box::new(reader))
    }

    fn from_boxed_reader(reader: NpzReader) -> FerrayResult<Self> {
        let archive = ZipArchive::new(reader)
            .map_err(|e| FerrayError::io_error(format!("failed to read .npz archive: {e}")))?;

        // Pre-extract the entry names so we can answer `names()`
        // queries without grabbing `&mut self`.
        let names: Vec<String> = archive
            .file_names()
            .map(|n| n.strip_suffix(".npy").unwrap_or(n).to_string())
            .collect();

        Ok(Self { archive, names })
    }

    /// List the names of arrays stored in the archive.
    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(String::as_str).collect()
    }

    /// Retrieve a named array as a `DynArray`.
    ///
    /// Decompresses and parses the entry on demand — large archives
    /// pay no cost for entries that are never requested.
    ///
    /// # Errors
    /// Returns `FerrayError::IoError` if the name is not found or the
    /// data is invalid.
    pub fn get(&mut self, name: &str) -> FerrayResult<DynArray> {
        // Resolve the requested logical name to the on-disk
        // `<name>.npy` entry. NumPy always writes the .npy suffix, but
        // we tolerate either form.
        let entry_name = if self.names.iter().any(|n| n == name) {
            format!("{name}.npy")
        } else {
            return Err(FerrayError::io_error(format!(
                "array '{name}' not found in .npz archive"
            )));
        };
        let mut entry = self.archive.by_name(&entry_name).map_err(|e| {
            FerrayError::io_error(format!("failed to read .npz entry '{entry_name}': {e}"))
        })?;
        // Materialize the entry's bytes into a small Vec — most .npy
        // payloads aren't huge and decompression is streaming anyway.
        // We can't pass `entry` directly to `load_dynamic_from_reader`
        // because the npy parser uses Seek and `ZipFile` does not.
        let mut data = Vec::new();
        entry
            .read_to_end(&mut data)
            .map_err(|e| FerrayError::io_error(format!("failed to read .npz entry data: {e}")))?;
        let mut cursor = Cursor::new(data);
        npy::load_dynamic_from_reader(&mut cursor)
    }

    /// Number of arrays in the archive.
    #[must_use]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Whether the archive is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferray_core::Array;
    use ferray_core::dimension::IxDyn;
    use ferray_core::dtype::DType;

    /// Construct a temporary file path inside a fresh `TempDir` (#244).
    /// Caller binds the `TempDir` for the test's scope so on completion
    /// or panic the directory is removed automatically.
    fn temp_path(name: &str) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::TempDir::new().expect("failed to create test TempDir");
        let path = dir.path().join(name);
        (dir, path)
    }

    fn make_f64_dyn(data: Vec<f64>, shape: &[usize]) -> DynArray {
        let arr = Array::<f64, IxDyn>::from_vec(IxDyn::new(shape), data).unwrap();
        DynArray::F64(arr)
    }

    fn make_i32_dyn(data: Vec<i32>, shape: &[usize]) -> DynArray {
        let arr = Array::<i32, IxDyn>::from_vec(IxDyn::new(shape), data).unwrap();
        DynArray::I32(arr)
    }

    #[test]
    fn savez_and_load() {
        let a = make_f64_dyn(vec![1.0, 2.0, 3.0], &[3]);
        let b = make_i32_dyn(vec![10, 20, 30, 40], &[2, 2]);

        let (_dir, path) = temp_path("test.npz");
        savez(&path, &[("a", &a), ("b", &b)]).unwrap();

        let mut npz = NpzFile::open(&path).unwrap();
        assert_eq!(npz.len(), 2);

        let mut names = npz.names();
        names.sort_unstable();
        assert_eq!(names, vec!["a", "b"]);

        let loaded_a = npz.get("a").unwrap();
        assert_eq!(loaded_a.dtype(), DType::F64);
        assert_eq!(loaded_a.shape(), &[3]);

        let loaded_b = npz.get("b").unwrap();
        assert_eq!(loaded_b.dtype(), DType::I32);
        assert_eq!(loaded_b.shape(), &[2, 2]);
    }

    #[test]
    fn savez_compressed_and_load() {
        let a = make_f64_dyn(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let (_dir, path) = temp_path("test_compressed.npz");
        savez_compressed(&path, &[("data", &a)]).unwrap();

        let mut npz = NpzFile::open(&path).unwrap();
        assert_eq!(npz.len(), 1);

        let loaded = npz.get("data").unwrap();
        assert_eq!(loaded.dtype(), DType::F64);
        assert_eq!(loaded.shape(), &[2, 3]);
    }

    #[test]
    fn npz_missing_key() {
        let a = make_f64_dyn(vec![1.0], &[1]);

        let (_dir, path) = temp_path("npz_missing.npz");
        savez(&path, &[("a", &a)]).unwrap();

        let mut npz = NpzFile::open(&path).unwrap();
        assert!(npz.get("nonexistent").is_err());
    }

    #[test]
    fn npz_lazy_get_then_get() {
        // Issue #232/#492: open should not eagerly load every entry,
        // but `get` calls should still work for repeated lookups in
        // any order.
        let a = make_f64_dyn(vec![1.0, 2.0, 3.0], &[3]);
        let b = make_i32_dyn(vec![10, 20, 30], &[3]);

        let (_dir, path) = temp_path("test_lazy.npz");
        savez(&path, &[("a", &a), ("b", &b)]).unwrap();

        let mut npz = NpzFile::open(&path).unwrap();
        // Lazy: read b first, then a, then b again.
        let _b1 = npz.get("b").unwrap();
        let _a = npz.get("a").unwrap();
        let _b2 = npz.get("b").unwrap();
    }

    #[test]
    fn npz_empty() {
        let (_dir, path) = temp_path("npz_empty.npz");
        savez(&path, &[]).unwrap();

        let npz = NpzFile::open(&path).unwrap();
        assert!(npz.is_empty());
        assert_eq!(npz.len(), 0);
    }
}
