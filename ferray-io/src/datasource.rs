// ferray-io: DataSource — local-file abstraction with transparent decompression.
//
// Analogue of `numpy.lib.npyio.DataSource`. NumPy's class is mostly a
// path-resolution + URL-caching layer that returns file-like objects;
// ferray's version covers the local-file half (the majority use case)
// and transparently decompresses .gz inputs via flate2 (already in
// workspace deps). URL fetching is intentionally not included — for
// remote files the caller should download first via the project's
// safe-fetch path and then point DataSource at the local file.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use ferray_core::error::{FerrayError, FerrayResult};
use flate2::read::GzDecoder;

/// Reader handle returned by [`DataSource::open`]. Implements [`Read`].
pub enum DataSourceReader {
    /// Plain (uncompressed) file.
    Plain(BufReader<File>),
    /// Gzip-compressed file.
    Gzip(Box<GzDecoder<BufReader<File>>>),
}

impl Read for DataSourceReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Self::Plain(r) => r.read(buf),
            Self::Gzip(r) => r.read(buf),
        }
    }
}

/// Local-file abstraction with transparent gzip decompression.
///
/// Analogue of `numpy.lib.npyio.DataSource`. Wraps an optional base
/// directory; `open(name)` resolves `name` relative to the base (if any)
/// and opens the file, transparently decompressing it when the path ends
/// in `.gz`.
///
/// URL fetching, the `bz2`/`zstd` formats, and the auto-cache behavior
/// of NumPy's class are not implemented — for remote sources, fetch via
/// the project's safe-fetch path first, then point DataSource at the
/// local copy.
#[derive(Debug, Clone, Default)]
pub struct DataSource {
    base: Option<PathBuf>,
}

impl DataSource {
    /// Create a new DataSource. If `base` is `Some`, paths passed to
    /// [`open`](Self::open) and [`exists`](Self::exists) are resolved
    /// relative to it.
    #[must_use]
    pub fn new(base: Option<PathBuf>) -> Self {
        Self { base }
    }

    /// Resolve a name (relative to `base`, if set) to an absolute path.
    ///
    /// # Errors
    /// Returns `FerrayError::IoError` if `std::path::absolute` fails to
    /// canonicalize the path (e.g. permission errors on the parent).
    pub fn abspath(&self, name: &str) -> FerrayResult<PathBuf> {
        let p: PathBuf = self
            .base
            .as_ref()
            .map_or_else(|| PathBuf::from(name), |b| b.join(name));
        std::path::absolute(&p)
            .map_err(|e| FerrayError::io_error(format!("DataSource::abspath: {e}")))
    }

    /// Whether the named file exists relative to the base directory.
    #[must_use]
    pub fn exists(&self, name: &str) -> bool {
        let p = self
            .base
            .as_ref()
            .map_or_else(|| PathBuf::from(name), |b| b.join(name));
        p.exists()
    }

    /// Open a file for reading.
    ///
    /// If `name` ends in `.gz`, the returned reader transparently
    /// decompresses on the fly.
    ///
    /// # Errors
    /// Returns `FerrayError::IoError` if the file cannot be opened.
    pub fn open(&self, name: &str) -> FerrayResult<DataSourceReader> {
        let p = self
            .base
            .as_ref()
            .map_or_else(|| PathBuf::from(name), |b| b.join(name));
        Self::open_path(&p)
    }

    /// Open an absolute path directly, bypassing the configured base.
    ///
    /// # Errors
    /// Returns `FerrayError::IoError` if the file cannot be opened.
    pub fn open_path<P: AsRef<Path>>(path: P) -> FerrayResult<DataSourceReader> {
        let path = path.as_ref();
        let f = File::open(path).map_err(|e| {
            FerrayError::io_error(format!(
                "DataSource::open: failed to open '{}': {e}",
                path.display()
            ))
        })?;
        let buf = BufReader::new(f);
        if matches!(path.extension().and_then(|e| e.to_str()), Some("gz")) {
            Ok(DataSourceReader::Gzip(Box::new(GzDecoder::new(buf))))
        } else {
            Ok(DataSourceReader::Plain(buf))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    fn temp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "ferray_io_datasource_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn datasource_open_plain() {
        let dir = temp_dir();
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello world").unwrap();

        let ds = DataSource::new(Some(dir.clone()));
        let mut r = ds.open("hello.txt").unwrap();
        let mut buf = String::new();
        r.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "hello world");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn datasource_open_gzip() {
        let dir = temp_dir();
        let path = dir.join("greet.txt.gz");
        let f = File::create(&path).unwrap();
        let mut enc = GzEncoder::new(f, Compression::default());
        enc.write_all(b"compressed payload").unwrap();
        enc.finish().unwrap();

        let ds = DataSource::new(Some(dir.clone()));
        let mut r = ds.open("greet.txt.gz").unwrap();
        let mut buf = String::new();
        r.read_to_string(&mut buf).unwrap();
        assert_eq!(buf, "compressed payload");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn datasource_exists_and_abspath() {
        let dir = temp_dir();
        std::fs::write(dir.join("present.txt"), b"x").unwrap();

        let ds = DataSource::new(Some(dir.clone()));
        assert!(ds.exists("present.txt"));
        assert!(!ds.exists("missing.txt"));
        let abs = ds.abspath("present.txt").unwrap();
        assert!(abs.is_absolute());

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn datasource_open_missing_file_errs() {
        let ds = DataSource::new(None);
        assert!(ds.open("/nonexistent/path/file.txt").is_err());
    }
}
