//! Bindings for `numpy.char` — vectorized string operations.
//!
//! NumPy's `np.char.lower(arr)` accepts a `numpy.ndarray` of unicode
//! strings (dtype `<U`) and returns one of the same shape. ferray-Rust
//! uses its own `StringArray<D>` type internally; this module
//! marshals between numpy unicode arrays and `StringArray` so callers
//! see only standard NumPy values.
//!
//! Functions returning numeric output (e.g. `count` → `u64`,
//! `equal` → `bool`, `str_len` → `u64`) round-trip through the
//! existing `IntoNumPy` pipeline. String-output functions go through
//! a small helper that builds a Python list and hands it to
//! `numpy.asarray`.
//!
//! The full ferray-strings surface includes regex, encoding,
//! splitting, partitioning, alignment, and translate operations —
//! those are deferred to phase-4 follow-ups inside this issue's scope.

use ferray_core::array::aliases::ArrayD;
use ferray_core::dimension::IxDyn;
use ferray_numpy_interop::IntoNumPy;
use ferray_strings as fs;
use ferray_strings::StringArray;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::conv::ferr_to_pyerr;

// ---------------------------------------------------------------------------
// Marshalling helpers
// ---------------------------------------------------------------------------

/// Coerce any string-array-like Python object (numpy unicode array,
/// list of str, list of list of str, …) into a ferray `StringArray<IxDyn>`.
///
/// We route through `numpy.asarray` to preserve shape, then flatten
/// and read back as a `Vec<String>` plus the original shape.
fn py_to_string_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<StringArray<IxDyn>> {
    let np = py.import("numpy")?;
    // Force unicode dtype so non-string inputs (numbers, bools) are
    // stringified on the way in — same behaviour numpy uses when
    // np.char.* is given a non-string array.
    let arr = np.call_method1("asarray", (obj, "U"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let flat = arr.call_method0("flatten")?;
    let data: Vec<String> = flat.extract()?;
    StringArray::<IxDyn>::from_vec_dyn(&shape, data).map_err(ferr_to_pyerr)
}

/// Push a ferray `StringArray<IxDyn>` back into a `numpy.ndarray` of
/// unicode strings. Builds a flat Python list and lets `numpy.asarray`
/// pick the correct `<U<n>` dtype, then reshapes.
fn string_array_to_pyarray<'py>(
    py: Python<'py>,
    sa: StringArray<IxDyn>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = sa.shape().to_vec();
    let data: Vec<String> = sa.into_vec();
    let np = py.import("numpy")?;
    let flat = np.call_method1("asarray", (PyList::new(py, &data)?,))?;
    if shape.len() <= 1 {
        return Ok(flat);
    }
    flat.call_method1("reshape", (shape,))
}

// ---------------------------------------------------------------------------
// Case operations (string → string)
// ---------------------------------------------------------------------------

macro_rules! bind_unary_string_op {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let sa = py_to_string_array(py, a)?;
            let r = $ferr_path(&sa).map_err(ferr_to_pyerr)?;
            string_array_to_pyarray(py, r)
        }
    };
}

bind_unary_string_op!(lower, fs::lower);
bind_unary_string_op!(upper, fs::upper);
bind_unary_string_op!(capitalize, fs::capitalize);
bind_unary_string_op!(title, fs::title);
bind_unary_string_op!(swapcase, fs::swapcase);

// ---------------------------------------------------------------------------
// Strip family (optional `chars` parameter)
// ---------------------------------------------------------------------------

macro_rules! bind_strip {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, chars = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            chars: Option<&str>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let sa = py_to_string_array(py, a)?;
            let r = $ferr_path(&sa, chars).map_err(ferr_to_pyerr)?;
            string_array_to_pyarray(py, r)
        }
    };
}

bind_strip!(strip, fs::strip);
bind_strip!(lstrip, fs::lstrip);
bind_strip!(rstrip, fs::rstrip);

// ---------------------------------------------------------------------------
// Search / query (string → numeric or bool)
// ---------------------------------------------------------------------------

/// `numpy.char.count(a, sub)`.
#[pyfunction]
pub fn count<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::count(&sa, sub).map_err(ferr_to_pyerr)?;
    // numpy strings.count returns signed int64 (generate_umath.py:1281).
    let r_dyn =
        ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.find(a, sub)`.
#[pyfunction]
pub fn find<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, sub: &str) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::find(&sa, sub).map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.startswith(a, prefix)`.
#[pyfunction]
pub fn startswith<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    prefix: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::startswith(&sa, prefix).map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.endswith(a, suffix)`.
#[pyfunction]
pub fn endswith<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    suffix: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::endswith(&sa, suffix).map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.str_len(a)` — character length of each element.
#[pyfunction]
pub fn str_len<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::str_len(&sa).map_err(ferr_to_pyerr)?;
    // numpy strings.str_len returns signed int64 of code-point counts
    // (string_ufuncs.cpp:118 num_codepoints, :1537 NPY_DEFAULT_INT).
    let r_dyn =
        ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Replace / concat / multiply
// ---------------------------------------------------------------------------

/// `numpy.char.replace(a, old, new, count=-1)`.
#[pyfunction]
#[pyo3(signature = (a, old, new, count = None))]
pub fn replace<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    old: &str,
    new: &str,
    count: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    // numpy's `count=-1` means "replace all". Map to None for ferray.
    let max_count: Option<usize> = match count {
        None | Some(-1) => None,
        Some(n) if n < 0 => None,
        Some(n) => Some(n as usize),
    };
    let r = fs::replace(&sa, old, new, max_count).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r)
}

/// `numpy.char.add(a, b)` — elementwise concatenation.
#[pyfunction]
pub fn add<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let sb = py_to_string_array(py, b)?;
    let r = fs::add_same(&sa, &sb).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r)
}

/// `numpy.char.multiply(a, n)` — repeat each element `n` times.
#[pyfunction]
pub fn multiply<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = py_to_string_array(py, a)?;
    let r = fs::multiply(&sa, n).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Pairwise comparison (string × string → bool)
// ---------------------------------------------------------------------------

macro_rules! bind_string_compare {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let sa = py_to_string_array(py, a)?;
            let sb = py_to_string_array(py, b)?;
            let r = $ferr_path(&sa, &sb).map_err(ferr_to_pyerr)?;
            let r_dyn = ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat())
                .map_err(ferr_to_pyerr)?;
            Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    };
}

bind_string_compare!(equal, fs::equal);
bind_string_compare!(not_equal, fs::not_equal);
bind_string_compare!(less, fs::less);
bind_string_compare!(less_equal, fs::less_equal);
bind_string_compare!(greater, fs::greater);
bind_string_compare!(greater_equal, fs::greater_equal);
