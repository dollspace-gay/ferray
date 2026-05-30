//! Bindings for the `numpy` file-I/O surface — `.npy` / `.npz` binary
//! serialization and delimited-text I/O — wired over the already-shipped
//! `ferray-io` crate.
//!
//! `ferray-io` implements the NumPy-compatible formats (`numpy/lib/
//! _npyio_impl.py`, `numpy/lib/format.py`) on the Rust side, typed in
//! `ferray_core::DynArray`. This module is the thin marshalling shim that
//! exposes them at `ferray.save` / `ferray.load` / `ferray.savez` /
//! `ferray.savez_compressed` / `ferray.savetxt` / `ferray.loadtxt` /
//! `ferray.genfromtxt` so `import ferray as np` round-trips file I/O
//! against numpy.
//!
//! The boundary contract is `numpy.ndarray` in both directions: the write
//! path converts the incoming numpy array to a `DynArray`
//! ([`conv::pyany_to_dynarray`]) and the read path converts the loaded
//! `DynArray` back to a `numpy.ndarray` ([`conv::dynarray_to_pyarray`]),
//! preserving dtype + shape (R-CODE-4 / R-DEV-3).
//!
//! ## REQ status
//!
//! Binding-level marshalling conventions for the file-IO surface. Two
//! states only (goal.md R-DEFER-2). numpy cites use `file:line` against the
//! read-only tree at `/home/doll/numpy-ref` (oracle: numpy 2.4.5); each
//! convention's non-test production consumer is the `#[pyfunction]` that is
//! the public API boundary, registered in `lib.rs`.
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | IO-NPY (save/load `.npy`) | SHIPPED | `save`/`load` here convert numpy↔`DynArray` and call `ferray::io::save_dynamic` / `load_dynamic` (`numpy/lib/_npyio_impl.py def save` :505, `def load` :312). Consumer: the `save`/`load` `#[pyfunction]`s registered in `lib.rs`. Verified: `tests/test_expansion_io.py::test_save_load_roundtrip*`. |
//! | IO-NPYEXT (numpy appends `.npy`) | SHIPPED | `save` appends `.npy` when the path has no extension, matching `numpy/lib/_npyio_impl.py:566-567` (`if not file.endswith('.npy'): file = file + '.npy'`). Consumer: `save`. Verified: `tests/test_expansion_io.py::test_save_appends_npy_extension`. |
//! | IO-NPZ (savez / savez_compressed / load `.npz`) | SHIPPED | `savez`/`savez_compressed` name positional args `arr_0,arr_1,…` and keyword args by key (`numpy/lib/_npyio_impl.py def savez` :581, `_savez` :756-772, `.npz` append :763-764), calling `ferray::io::savez` / `savez_compressed`. `load` on a `.npz` returns a name→ndarray `dict` via `ferray::io::NpzFile`. Consumers: `savez`/`savez_compressed`/`load`. Verified: `tests/test_expansion_io.py::test_savez_*`, `::test_load_npz_*`. |
//! | IO-TXT (savetxt / loadtxt) | SHIPPED | `savetxt` writes via `ferray::io::savetxt`/`savetxt_1d` with numpy defaults (space delimiter, `%.18e` fmt — `numpy/lib/_npyio_impl.py def savetxt` :1399, `fmt='%.18e', delimiter=' '`); `loadtxt` parses float64 via `ferray::io::loadtxt` and collapses single-row/single-column to 1-D (`def loadtxt` :1120, `dtype=float`). Consumers: `savetxt`/`loadtxt`. Verified: `tests/test_expansion_io.py::test_savetxt_loadtxt_*`. |
//! | IO-GENFROMTXT (genfromtxt missing→nan) | SHIPPED | `genfromtxt` parses float64 with empty/unparseable cells filled by `nan` via `ferray::io::genfromtxt` (`numpy/lib/_npyio_impl.py def genfromtxt` :1735, `filling_values` default nan for a float dtype). Consumer: `genfromtxt`. Verified: `tests/test_expansion_io.py::test_genfromtxt_*`. |
//! | IO-COMPLEX (complex / datetime `.npy` dtypes) | NOT-STARTED | The numpy-interop `NpElement` set excludes complex/datetime/128-bit, so `pyany_to_dynarray`/`dynarray_to_pyarray` raise `TypeError` for those rather than a lossy cast (R-CODE-4). Root cause is the interop crate's `NpElement` coverage (tracked in numpy-interop #739), not this binding — open follow-up. |
//! | IO-FILEOBJ (open file-object input) | NOT-STARTED | numpy accepts an open file object as well as a path str (`numpy/lib/_npyio_impl.py:562` `if hasattr(file, 'write')`). These bindings accept a path str only; file-object support routes through `*_to_writer`/`*_from_reader` and is an open follow-up. |

use std::path::PathBuf;

use ferray_core::Array;
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

use crate::conv::{dynarray_to_pyarray, ferr_to_pyerr, pyany_to_dynarray};

// ---------------------------------------------------------------------------
// .npy single-array I/O
// ---------------------------------------------------------------------------

/// `numpy.save(file, arr, allow_pickle=True)` — write `arr` to a `.npy`
/// file.
///
/// numpy appends `.npy` when the filename has no extension
/// (`numpy/lib/_npyio_impl.py:566-567`). `file` is accepted as a path
/// string; an open file-object is an open follow-up (see IO-FILEOBJ).
#[pyfunction]
#[pyo3(signature = (file, arr, allow_pickle=true))]
pub fn save(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    arr: &Bound<'_, PyAny>,
    allow_pickle: bool,
) -> PyResult<()> {
    // `allow_pickle` exists in numpy's signature (`save(file, arr,
    // allow_pickle=True)` :505) but only affects object arrays, which
    // ferray's binary format doesn't carry; accept and ignore so the kwarg
    // surface matches numpy (R-DEV-2).
    let _ = allow_pickle;
    let path = resolve_save_path(file)?;
    let dyn_arr = pyany_to_dynarray(py, arr)?;
    ferray::io::npy::save_dynamic(&path, &dyn_arr).map_err(ferr_to_pyerr)
}

/// Resolve a `numpy.save` `file` argument to a `.npy` path, appending the
/// `.npy` extension when absent (numpy `_npyio_impl.py:566-567`).
fn resolve_save_path(file: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    let s: String = file
        .extract()
        .map_err(|_| PyTypeError::new_err("file must be a path string"))?;
    let needs_ext = !std::path::Path::new(&s)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("npy"));
    Ok(if needs_ext {
        PathBuf::from(format!("{s}.npy"))
    } else {
        PathBuf::from(s)
    })
}

/// `numpy.load(file)` — read a `.npy` (single array) or `.npz` (archive).
///
/// A `.npy` returns a `numpy.ndarray`; a `.npz` returns a name→ndarray
/// `dict` (numpy returns an `NpzFile`; `z['x']` access works on a dict the
/// same way). Dispatch is by the zip magic, mirroring numpy's
/// `_npyio_impl.py:458-467` (`_ZIP_PREFIX = b'PK\x03\x04'` → `NpzFile`).
#[pyfunction]
#[pyo3(signature = (file, mmap_mode=None, allow_pickle=false, fix_imports=true, encoding="ASCII"))]
pub fn load<'py>(
    py: Python<'py>,
    file: &Bound<'py, PyAny>,
    mmap_mode: Option<&Bound<'py, PyAny>>,
    allow_pickle: bool,
    fix_imports: bool,
    encoding: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = (mmap_mode, allow_pickle, fix_imports, encoding);
    let s: String = file
        .extract()
        .map_err(|_| PyTypeError::new_err("file must be a path string"))?;
    let path = PathBuf::from(&s);

    if is_zip(&path)? {
        return load_npz(py, &path);
    }
    let dyn_arr = ferray::io::load_dynamic(&path).map_err(ferr_to_pyerr)?;
    dynarray_to_pyarray(py, dyn_arr)
}

/// Return `true` if `path` begins with the PK zip magic (`.npz` archives
/// are zip files; `.npy` files begin with `\x93NUMPY`).
fn is_zip(path: &std::path::Path) -> PyResult<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)
        .map_err(|e| PyValueError::new_err(format!("failed to open '{}': {e}", path.display())))?;
    let mut magic = [0u8; 2];
    match f.read_exact(&mut magic) {
        Ok(()) => Ok(&magic == b"PK"),
        // A file shorter than two bytes can't be a zip archive.
        Err(_) => Ok(false),
    }
}

/// Open a `.npz` archive and return a `{name: ndarray}` dict.
fn load_npz<'py>(py: Python<'py>, path: &std::path::Path) -> PyResult<Bound<'py, PyAny>> {
    let mut npz = ferray::io::NpzFile::open(path).map_err(ferr_to_pyerr)?;
    let names: Vec<String> = npz.names().into_iter().map(str::to_string).collect();
    let out = PyDict::new(py);
    for name in names {
        let dyn_arr = npz.get(&name).map_err(ferr_to_pyerr)?;
        let arr = dynarray_to_pyarray(py, dyn_arr)?;
        out.set_item(name, arr)?;
    }
    Ok(out.into_any())
}

// ---------------------------------------------------------------------------
// .npz multi-array archives
// ---------------------------------------------------------------------------

/// `numpy.savez(file, *args, **kwds)` — write multiple arrays to a `.npz`.
///
/// Positional args are named `arr_0, arr_1, …`; keyword args use the key
/// (`numpy/lib/_npyio_impl.py def savez` :581, `_savez` :767-772).
#[pyfunction]
#[pyo3(signature = (file, *args, **kwds))]
pub fn savez(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    savez_dispatch(py, file, args, kwds, false)
}

/// `numpy.savez_compressed(file, *args, **kwds)` — DEFLATE-compressed `.npz`
/// (`numpy/lib/_npyio_impl.py def savez_compressed` :682).
#[pyfunction]
#[pyo3(signature = (file, *args, **kwds))]
pub fn savez_compressed(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwds: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    savez_dispatch(py, file, args, kwds, true)
}

/// Shared body for [`savez`] / [`savez_compressed`]: build the
/// `(name, DynArray)` list (numpy's `arr_N` positional naming + keyword
/// keys) and call the matching `ferray-io` writer. The `.npz` extension is
/// appended when absent (numpy `_savez` :763-764).
fn savez_dispatch(
    py: Python<'_>,
    file: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwds: Option<&Bound<'_, PyDict>>,
    compressed: bool,
) -> PyResult<()> {
    let s: String = file
        .extract()
        .map_err(|_| PyTypeError::new_err("file must be a path string"))?;
    let needs_ext = !std::path::Path::new(&s)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("npz"));
    let path = if needs_ext {
        PathBuf::from(format!("{s}.npz"))
    } else {
        PathBuf::from(s)
    };

    // Materialize every array as a DynArray, owning the names so the
    // `&[(&str, &DynArray)]` slice ferray-io wants can borrow them.
    let mut named: Vec<(String, ferray_core::DynArray)> = Vec::new();
    for (i, item) in args.iter().enumerate() {
        named.push((format!("arr_{i}"), pyany_to_dynarray(py, &item)?));
    }
    if let Some(kwds) = kwds {
        for (k, v) in kwds.iter() {
            let key: String = k.extract()?;
            named.push((key, pyany_to_dynarray(py, &v)?));
        }
    }
    let refs: Vec<(&str, &ferray_core::DynArray)> =
        named.iter().map(|(n, a)| (n.as_str(), a)).collect();

    if compressed {
        ferray::io::savez_compressed(&path, &refs).map_err(ferr_to_pyerr)
    } else {
        ferray::io::savez(&path, &refs).map_err(ferr_to_pyerr)
    }
}

// ---------------------------------------------------------------------------
// delimited-text I/O
// ---------------------------------------------------------------------------

/// `numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', ...)` — write a
/// 1-D or 2-D array as delimited text.
///
/// Defaults mirror numpy (`numpy/lib/_npyio_impl.py def savetxt` :1399):
/// `%.18e` scientific format, single-space delimiter, `\n` newline. The
/// array is coerced to `float64` first — numpy's default `%.18e` is a float
/// format and savetxt formats each value as a float.
#[pyfunction]
#[pyo3(signature = (fname, x, fmt="%.18e", delimiter=" ", newline="\n", header="", footer="", comments="# "))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors numpy.savetxt's kwarg surface (R-DEV-2)"
)]
pub fn savetxt(
    py: Python<'_>,
    fname: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    fmt: &str,
    delimiter: &str,
    newline: &str,
    header: &str,
    footer: &str,
    comments: &str,
) -> PyResult<()> {
    // `comments` prefixes header/footer in numpy; for the round-trip surface
    // these bindings write header/footer verbatim. Accept it for ABI parity.
    let _ = comments;
    let path: String = fname
        .extract()
        .map_err(|_| PyTypeError::new_err("fname must be a path string"))?;

    // numpy's default fmt is a float format; coerce X to a contiguous float64
    // ndarray so 1-D and 2-D inputs are both handled in C order.
    let np = py.import("numpy")?;
    let arr = np.call_method1("ascontiguousarray", (x,))?;
    let arr = np.call_method1("asarray", (arr, "float64"))?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;

    let delim_char = single_char(delimiter, ' ');
    let opts = ferray::io::SaveTxtOptions {
        delimiter: delim_char,
        fmt: Some(fmt.to_string()),
        header: opt_str(header),
        footer: opt_str(footer),
        newline: newline.to_string(),
    };

    use ferray_numpy_interop::AsFerray;
    use numpy::PyReadonlyArrayDyn;
    let view = arr.extract::<PyReadonlyArrayDyn<f64>>()?;
    let fa: Array<f64, IxDyn> = view.as_ferray().map_err(ferr_to_pyerr)?;

    match ndim {
        0 | 1 => {
            let n = fa.shape().iter().product::<usize>();
            let data: Vec<f64> = fa.iter().copied().collect();
            let a1 = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).map_err(ferr_to_pyerr)?;
            ferray::io::savetxt_1d(&path, &a1, &opts).map_err(ferr_to_pyerr)
        }
        2 => {
            let shape = fa.shape();
            let (r, c) = (shape[0], shape[1]);
            let data: Vec<f64> = fa.iter().copied().collect();
            let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([r, c]), data).map_err(ferr_to_pyerr)?;
            ferray::io::savetxt(&path, &a2, &opts).map_err(ferr_to_pyerr)
        }
        n => Err(PyValueError::new_err(format!(
            "savetxt: expected a 1-D or 2-D array, got {n}-D"
        ))),
    }
}

/// `numpy.loadtxt(fname, dtype=float, delimiter=None, skiprows=0, ...)` —
/// parse a delimited text file into an array.
///
/// Parses `float64` by default (numpy's default `dtype=float`,
/// `numpy/lib/_npyio_impl.py def loadtxt` :1120). A single-row or
/// single-column file collapses to 1-D, otherwise the result is 2-D —
/// matching numpy's `loadtxt` squeeze. A non-float `dtype` is cast at the
/// boundary via numpy `astype`.
#[pyfunction]
#[pyo3(signature = (fname, dtype=None, delimiter=None, skiprows=0))]
pub fn loadtxt<'py>(
    py: Python<'py>,
    fname: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
    delimiter: Option<&str>,
    skiprows: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let path: String = fname
        .extract()
        .map_err(|_| PyTypeError::new_err("fname must be a path string"))?;
    // numpy's default delimiter is "any whitespace"; ferray-io's parser
    // splits on whitespace when given a space delimiter.
    let delim_char = delimiter.map_or(' ', |d| single_char(d, ' '));

    let arr2 = ferray::io::loadtxt::<f64, _>(&path, delim_char, skiprows).map_err(ferr_to_pyerr)?;
    let shape = arr2.shape();
    let (r, c) = (shape[0], shape[1]);
    let data: Vec<f64> = arr2.iter().copied().collect();

    let np_arr = build_loadtxt_array(py, data, r, c)?;
    // Honor an explicit `dtype=` by casting at the boundary (numpy parses
    // straight into the requested dtype; we parse float then cast).
    if let Some(dt) = dtype {
        return np_arr.call_method1("astype", (dt,));
    }
    Ok(np_arr)
}

/// `numpy.genfromtxt(fname, ...)` — like `loadtxt`, but empty / unparseable
/// cells become `nan` (numpy default `filling_values` for a float dtype,
/// `numpy/lib/_npyio_impl.py def genfromtxt` :1735).
#[pyfunction]
#[pyo3(signature = (fname, dtype=None, delimiter=None, skip_header=0))]
pub fn genfromtxt<'py>(
    py: Python<'py>,
    fname: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
    delimiter: Option<&str>,
    skip_header: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let path: String = fname
        .extract()
        .map_err(|_| PyTypeError::new_err("fname must be a path string"))?;
    let delim_char = delimiter.map_or(',', |d| single_char(d, ','));

    // numpy's float-dtype genfromtxt fills missing/unparseable cells with
    // nan; pass NAN as the fill and an empty extra-marker list (ferray-io
    // already recognizes "", "NA", "nan", … as missing).
    let arr2 = ferray::io::genfromtxt(&path, delim_char, f64::NAN, skip_header, &[])
        .map_err(ferr_to_pyerr)?;
    let shape = arr2.shape();
    let (r, c) = (shape[0], shape[1]);
    let data: Vec<f64> = arr2.iter().copied().collect();

    let np_arr = build_loadtxt_array(py, data, r, c)?;
    if let Some(dt) = dtype {
        return np_arr.call_method1("astype", (dt,));
    }
    Ok(np_arr)
}

/// Build a numpy `float64` array from a flat row-major buffer, collapsing a
/// single-row or single-column grid to 1-D (numpy `loadtxt` squeeze).
fn build_loadtxt_array<'py>(
    py: Python<'py>,
    data: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    if rows == 1 || cols == 1 {
        // 1-D collapse: a single row OR single column is a flat vector.
        let n = rows * cols;
        let a1 = Array::<f64, Ix1>::from_vec(Ix1::new([n]), data).map_err(ferr_to_pyerr)?;
        return Ok(a1.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    let a2 = Array::<f64, Ix2>::from_vec(Ix2::new([rows, cols]), data).map_err(ferr_to_pyerr)?;
    Ok(a2.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// First char of a delimiter string, falling back to `default` for an empty
/// string. ferray-io's text parser is single-char-delimited.
fn single_char(s: &str, default: char) -> char {
    s.chars().next().unwrap_or(default)
}

/// Map an empty header/footer string to `None` (numpy's default header is
/// `''`, meaning "no header line").
fn opt_str(s: &str) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}
