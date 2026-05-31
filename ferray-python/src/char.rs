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
//! The full ferray-strings surface — regex, encoding, splitting,
//! partitioning, alignment, predicates, slicing and translate — is bound
//! here and re-registered under both `ferray.char` (legacy `numpy.char`)
//! and `ferray.strings` (the NumPy 2.0 canonical `numpy.strings`
//! namespace). The two surfaces share one set of `#[pyfunction]` items
//! because `numpy.char.*` and `numpy.strings.*` are the same operations.
//!
//! ## REQ status — `numpy.char` / `numpy.strings` surface (the `#[pyfunction]` home)
//!
//! Each row is a numpy.char/numpy.strings callable defined here as a
//! `#[pyfunction]` (`fs` aliases `ferray_strings`) and registered onto
//! both namespaces by [`crate::strings::register_string_ops`]. Green
//! against numpy 2.4.x (pytest `tests/test_char.py`). SHIPPED rows quote
//! the binding fn + the delegated `fs::` kernel (symbol anchors,
//! R-CITE-2b).
//!
//! SHIPPED:
//!   - Case: `lower` → `fs::lower`, `upper`, `capitalize`, `title`,
//!     `swapcase`.
//!   - Strip/pad: `strip` → `fs::strip`, `lstrip` → `fs::lstrip`,
//!     `rstrip` → `fs::rstrip`, `center` → `fs::center`,
//!     `ljust` → `fs::ljust_with`, `rjust` → `fs::rjust_with`,
//!     `zfill`, `expandtabs`.
//!   - Search/count: `count` → `fs::count`, `find` → `fs::find`,
//!     `rfind` → `fs::rfind`, `index`, `rindex`, `startswith` →
//!     `fs::startswith`, `endswith` → `fs::endswith`,
//!     `str_len` → `fs::str_len`.
//!   - Edit: `replace` → `fs::replace`, `add` → `fs::add_same`,
//!     `multiply` → `fs::multiply`, `r#mod` → `fs::mod_`,
//!     `translate`, `slice` → `fs::slice`.
//!   - Split/partition: `split` → `fs::split_ragged`,
//!     `rsplit` → `fs::rsplit`, `splitlines` → `fs::splitlines`,
//!     `partition`, `rpartition` → `fs::rpartition`, `join`.
//!   - Predicates → bool: `isalpha` → `fs::isalpha`, `isdigit`,
//!     `isspace`, `isalnum`, `isdecimal`, `isnumeric`, `islower`,
//!     `isupper`, `istitle`.
//!   - Comparison: `equal` → `fs::equal`, `not_equal`, `less`,
//!     `less_equal`, `greater`, `greater_equal`, `compare_chararrays`.
//!   - Encoding: `encode` → `fs::encode`, `decode` → `fs::decode`.
//!
//! NOT-STARTED: none — every callable registered for `numpy.char` /
//! `numpy.strings` is bound here and green.

use std::collections::HashMap;

use ferray_core::array::aliases::ArrayD;
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_numpy_interop::IntoNumPy;
use ferray_strings as fs;
use ferray_strings::{StringArray, StringArray1};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList, PyTuple};

use crate::conv::{coerce_multiply_count, ferr_to_pyerr};

// ---------------------------------------------------------------------------
// Marshalling helpers
// ---------------------------------------------------------------------------

/// Coerce any string-array-like Python object (numpy unicode array,
/// list of str, list of list of str, …) into a ferray `StringArray<IxDyn>`,
/// preserving the original rank (including 0-d), and return the original
/// `ndim` alongside it.
///
/// We route through `numpy.asarray` to preserve shape, then flatten and
/// read back as a `Vec<String>`. The `StringArray` always stores a
/// flattened `[n]` (or `shape`) view; the returned `ndim` lets the output
/// marshaller restore numpy's rank contract (R-DEV-3) — a 0-d scalar input
/// must yield a 0-d output, not a 1-d length-1 array.
fn py_to_string_array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<(StringArray<IxDyn>, usize)> {
    let np = py.import("numpy")?;
    // Force unicode dtype so non-string inputs (numbers, bools) are
    // stringified on the way in — same behaviour numpy uses when
    // np.char.* is given a non-string array.
    let arr = np.call_method1("asarray", (obj, "U"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let ndim = shape.len();
    let flat = arr.call_method0("flatten")?;
    let data: Vec<String> = flat.extract()?;
    let sa = StringArray::<IxDyn>::from_vec_dyn(&shape, data).map_err(ferr_to_pyerr)?;
    Ok((sa, ndim))
}

/// The unicode width (`<U<n>`) of an array-like, i.e. its itemsize in code
/// points. Returns `None` for an empty array of unknown width only when the
/// dtype is not a unicode dtype; for a unicode dtype it returns the declared
/// width even when the array is empty.
///
/// NumPy's case/strip ops cannot grow a string, so they preserve the INPUT
/// itemsize: `np.char.lower(<U10) -> <U10` (`numpy/_core/strings.py:1124`
/// `lower` routes through `_vec_string(a, a.dtype, 'lower')` — output dtype
/// is the *input* dtype). Reading `dtype.itemsize // 4` for a `<U` array
/// (`strings.py:99` `_get_num_chars`: "itemsize / 4" for unicode) lets the
/// binding force that width on the output instead of letting
/// `numpy.asarray(list)` shrink it to the minimal content width (R-CODE-4).
fn input_unicode_width(arr: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    let dtype = arr.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    if kind != "U" {
        return Ok(None);
    }
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    Ok(Some(itemsize / 4))
}

/// Read the unicode width of an arbitrary Python object by routing it
/// through `numpy.asarray(obj, "U")` first (so list/scalar inputs acquire a
/// `<U` dtype), then reading its itemsize. Used by case/strip ops to capture
/// the input width before computation.
fn obj_unicode_width<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Option<usize>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (obj, "U"))?;
    input_unicode_width(&arr)
}

/// Push a ferray `StringArray<IxDyn>` back into a `numpy.ndarray` of unicode
/// strings, honoring numpy's rank and (optionally) itemsize contract.
///
/// `ndim` is the input rank captured by [`py_to_string_array`]; the result
/// is reshaped to restore it (a 0-d input collapses the length-1 buffer back
/// to a 0-d array via `reshape(())`). `force_width`, when `Some(w)`, forces
/// the output dtype to `<U{w}>` so the boundary preserves the numpy itemsize
/// contract instead of `numpy.asarray(list)` shrinking it to the minimal
/// content width (R-CODE-4); when `None` the content-derived width is kept
/// (correct for ops like `multiply`/`add` whose output width genuinely
/// follows the content).
fn string_array_to_pyarray<'py>(
    py: Python<'py>,
    sa: StringArray<IxDyn>,
    ndim: usize,
    force_width: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape: Vec<usize> = sa.shape().to_vec();
    let data: Vec<String> = sa.into_vec();
    let np = py.import("numpy")?;
    let list = PyList::new(py, &data)?;
    // Build with an explicit unicode dtype when a width is forced; otherwise
    // let numpy infer the minimal `<U` width from the contents. An explicit
    // empty `<U{w}>` also preserves the string dtype kind on an empty input
    // (a bare `numpy.asarray([])` would default to float64 — R-DEV-1).
    let flat = match force_width {
        Some(w) => {
            let dt = format!("<U{}", w.max(1));
            np.call_method1("asarray", (list, dt))?
        }
        None if data.is_empty() => {
            // Empty + no forced width: still pin a unicode dtype so the kind
            // is 'U' (numpy preserves the string dtype on empty case ops).
            np.call_method1("asarray", (list, "<U1"))?
        }
        None => np.call_method1("asarray", (list,))?,
    };
    // Restore the original rank. `reshape(shape)` with the true shape handles
    // 0-d (`shape == ()`), 1-d, and N-d uniformly.
    if ndim == shape.len() {
        return flat.call_method1("reshape", (shape,));
    }
    // 0-d input: the flattened buffer is length-1; collapse to a 0-d array.
    if ndim == 0 {
        let empty = pyo3::types::PyTuple::empty(py);
        return flat.call_method1("reshape", (empty,));
    }
    flat.call_method1("reshape", (shape,))
}

/// Resolve a Python-slice window `[start, end)` for a single string of
/// length `n`, returning `(window_substring, resolved_start)`.
///
/// `numpy.strings.find/count/startswith/endswith` interpret `start`/`end`
/// "as in slice notation" (`numpy/_core/strings.py:269` etc.): negative
/// indices count from the end, out-of-range indices clamp to `[0, n]`, and a
/// missing `end` means "to the end of the string". The op runs over
/// `s[start:end]`; `find` additionally reports an absolute index, so the
/// caller adds the resolved start back. Verified against the live numpy
/// oracle for positive, negative, and out-of-range bounds.
fn slice_window(s: &str, start: i64, end: Option<i64>) -> (String, usize) {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len() as i64;
    let clamp = |idx: i64| -> i64 {
        let i = if idx < 0 { n + idx } else { idx };
        i.clamp(0, n)
    };
    let st = clamp(start);
    let en = match end {
        Some(e) => clamp(e),
        None => n,
    };
    let window: String = if st < en {
        chars[st as usize..en as usize].iter().collect()
    } else {
        String::new()
    };
    (window, st as usize)
}

/// Apply [`slice_window`] elementwise, returning the windowed `StringArray`
/// (same shape) plus the per-element resolved start offsets (flat order).
fn windowed_string_array(
    sa: &StringArray<IxDyn>,
    start: i64,
    end: Option<i64>,
) -> PyResult<(StringArray<IxDyn>, Vec<usize>)> {
    let shape: Vec<usize> = sa.shape().to_vec();
    let mut windows: Vec<String> = Vec::with_capacity(sa.len());
    let mut starts: Vec<usize> = Vec::with_capacity(sa.len());
    for s in sa.iter() {
        let (w, st) = slice_window(s, start, end);
        windows.push(w);
        starts.push(st);
    }
    let wa = StringArray::<IxDyn>::from_vec_dyn(&shape, windows).map_err(ferr_to_pyerr)?;
    Ok((wa, starts))
}

/// Broadcast two array-likes to a common shape via `numpy.broadcast_arrays`,
/// then marshal each into a same-shape `StringArray<IxDyn>`.
///
/// `numpy.char.equal` / `add` and the ordered comparisons are ufuncs that
/// broadcast their operands (`np.char.equal(['a','b'], 'a') ->
/// [True, False]`). The ferray-strings same-shape kernels require identical
/// shapes, so the binding aligns the operands at the boundary first
/// (R-DEV-3) — the same pattern the numeric bindings use for NEP-50 dtype
/// alignment. The returned `ndim` is the broadcast rank for output marshalling.
fn broadcast_pair<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<(StringArray<IxDyn>, StringArray<IxDyn>, usize)> {
    let np = py.import("numpy")?;
    let ua = np.call_method1("asarray", (a, "U"))?;
    let ub = np.call_method1("asarray", (b, "U"))?;
    let pair = np.call_method1("broadcast_arrays", (ua, ub))?;
    let ba = pair.get_item(0)?;
    let bb = pair.get_item(1)?;
    let (sa, ndim) = py_to_string_array(py, &ba)?;
    let (sb, _) = py_to_string_array(py, &bb)?;
    Ok((sa, sb, ndim))
}

// ---------------------------------------------------------------------------
// Case operations (string → string)
// ---------------------------------------------------------------------------

macro_rules! bind_unary_string_op {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            // Case ops cannot grow a string, so numpy preserves the INPUT
            // itemsize (strings.py:1124 routes through the input dtype).
            let width = obj_unicode_width(py, a)?;
            let (sa, ndim) = py_to_string_array(py, a)?;
            let r = $ferr_path(&sa).map_err(ferr_to_pyerr)?;
            string_array_to_pyarray(py, r, ndim, width)
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
            // strip cannot grow a string either, so numpy keeps the INPUT
            // itemsize even though the content shrinks (strings.py:1034).
            let width = obj_unicode_width(py, a)?;
            let (sa, ndim) = py_to_string_array(py, a)?;
            let r = $ferr_path(&sa, chars).map_err(ferr_to_pyerr)?;
            string_array_to_pyarray(py, r, ndim, width)
        }
    };
}

bind_strip!(strip, fs::strip);
bind_strip!(lstrip, fs::lstrip);
bind_strip!(rstrip, fs::rstrip);

// ---------------------------------------------------------------------------
// Search / query (string → numeric or bool)
// ---------------------------------------------------------------------------

/// `numpy.char.count(a, sub, start=0, end=None)`.
///
/// `start`/`end` window each string in Python-slice notation
/// (`numpy/_core/strings.py:405`) before counting; the count of `sub` in the
/// window is reported. Implemented at the boundary by slicing each element
/// to its `[start, end)` window and counting on the windowed array.
#[pyfunction]
#[pyo3(signature = (a, sub, start = 0, end = None))]
pub fn count<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, _) = windowed_string_array(&sa, start, end)?;
    let r = fs::count(&wa, sub).map_err(ferr_to_pyerr)?;
    // numpy strings.count returns signed int64 (generate_umath.py:1281).
    let r_dyn =
        ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.find(a, sub, start=0, end=None)`.
///
/// `find` returns the lowest ABSOLUTE index of `sub` within the
/// `[start, end)` window (`numpy/_core/strings.py:256`), or `-1`. The binding
/// finds within each element's window and adds the resolved start offset back
/// to a found index so the reported index is absolute (verified vs the live
/// numpy oracle for positive/negative/out-of-range bounds).
#[pyfunction]
#[pyo3(signature = (a, sub, start = 0, end = None))]
pub fn find<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, starts) = windowed_string_array(&sa, start, end)?;
    let r = fs::find(&wa, sub).map_err(ferr_to_pyerr)?;
    let adjusted: Vec<i64> = r
        .to_vec_flat()
        .into_iter()
        .zip(starts.iter())
        .map(|(idx, &st)| if idx >= 0 { idx + st as i64 } else { -1 })
        .collect();
    let r_dyn = ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), adjusted).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.startswith(a, prefix, start=0, end=None)`.
///
/// True where `a[i][start:end]` starts with `prefix`
/// (`numpy/_core/strings.py:450`). The binding slices each element to its
/// `[start, end)` window before testing.
#[pyfunction]
#[pyo3(signature = (a, prefix, start = 0, end = None))]
pub fn startswith<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    prefix: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, _) = windowed_string_array(&sa, start, end)?;
    let r = fs::startswith(&wa, prefix).map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.endswith(a, suffix, start=0, end=None)`.
///
/// True where `a[i][start:end]` ends with `suffix`
/// (`numpy/_core/strings.py:491`). The binding slices each element to its
/// `[start, end)` window before testing.
#[pyfunction]
#[pyo3(signature = (a, suffix, start = 0, end = None))]
pub fn endswith<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    suffix: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, _) = windowed_string_array(&sa, start, end)?;
    let r = fs::endswith(&wa, suffix).map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.char.str_len(a)` — character length of each element.
#[pyfunction]
pub fn str_len<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
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
    let (sa, ndim) = py_to_string_array(py, a)?;
    // numpy's `count=-1` means "replace all". Map to None for ferray.
    let max_count: Option<usize> = match count {
        None | Some(-1) => None,
        Some(n) if n < 0 => None,
        Some(n) => Some(n as usize),
    };
    let r = fs::replace(&sa, old, new, max_count).map_err(ferr_to_pyerr)?;
    // replace can grow or shrink content; the output width genuinely follows
    // the content, so no width is forced.
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.char.add(a, b)` — elementwise concatenation with broadcasting.
///
/// `numpy.char.add` is a ufunc and broadcasts its operands
/// (`numpy/_core/strings.py:10` imports `add` from the umath ufunc), e.g.
/// `(2,) + (1,) -> (2,)`. The binding broadcasts both inputs to a common
/// shape at the boundary (R-DEV-3) before the same-shape concatenation. The
/// output `<U` width follows the concatenated content (no forced width).
#[pyfunction]
pub fn add<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, sb, ndim) = broadcast_pair(py, a, b)?;
    let r = fs::add_same(&sa, &sb).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.char.multiply(a, i)` — repeat each element `i` times.
///
/// `i` is bound signed and clamped to `>= 0` (negative counts yield the
/// empty string, `numpy/_core/strings.py:155` / `:195 np.maximum(i, 0)`)
/// instead of raising `OverflowError` at the PyO3 boundary. Output `<U`
/// width follows the repeated content.
#[pyfunction]
pub fn multiply<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // `i` is `array_like, with any integer dtype` and broadcasts element-wise
    // (`numpy/_core/strings.py:150`; docstring example `:177` shows
    // `i = np.array([1,2,3])` -> ['a','bb','ccc']). The library `fs::multiply`
    // takes a single scalar count, so an integer-array `i` is delegated to
    // `numpy.char.multiply`, which owns the elementwise broadcast (and clamps
    // negative counts to the empty string). A scalar `i` keeps the native path.
    let count = match coerce_multiply_count(n) {
        Ok(c) => c,
        Err(_) => {
            let np = py.import("numpy")?;
            let char_mod = np.getattr("char")?;
            return char_mod.call_method1("multiply", (a, n));
        }
    };
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::multiply(&sa, count).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

// ---------------------------------------------------------------------------
// Pairwise comparison (string × string → bool)
// ---------------------------------------------------------------------------

macro_rules! bind_string_compare {
    ($name:ident, $ferr_path:path) => {
        /// numpy.char comparison ufunc — broadcasts its two operands
        /// (e.g. a scalar string vs an array) to a common shape before the
        /// elementwise comparison (R-DEV-3).
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let (sa, sb, _ndim) = broadcast_pair(py, a, b)?;
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

// ---------------------------------------------------------------------------
// Predicates (string -> bool array)
// ---------------------------------------------------------------------------

macro_rules! bind_predicate {
    ($name:ident, $ferr_path:path, $doc:expr) => {
        #[doc = $doc]
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let (sa, _) = py_to_string_array(py, a)?;
            let r = $ferr_path(&sa).map_err(ferr_to_pyerr)?;
            let r_dyn = ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat())
                .map_err(ferr_to_pyerr)?;
            Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
    };
}

bind_predicate!(
    isalpha,
    fs::isalpha,
    "`numpy.strings.isalpha(a)` — True where every char is alphabetic and the string is non-empty (`numpy/_core/strings.py` `isalpha`)."
);
bind_predicate!(
    isdigit,
    fs::isdigit,
    "`numpy.strings.isdigit(a)` — True where every char is a digit and the string is non-empty."
);
bind_predicate!(
    isspace,
    fs::isspace,
    "`numpy.strings.isspace(a)` — True where every char is whitespace and the string is non-empty."
);
bind_predicate!(
    isalnum,
    fs::isalnum,
    "`numpy.strings.isalnum(a)` — True where every char is alphanumeric and the string is non-empty."
);
bind_predicate!(
    isupper,
    fs::isupper,
    "`numpy.strings.isupper(a)` — True where all cased chars are uppercase and there is at least one cased char."
);
bind_predicate!(
    islower,
    fs::islower,
    "`numpy.strings.islower(a)` — True where all cased chars are lowercase and there is at least one cased char."
);
bind_predicate!(
    isnumeric,
    fs::isnumeric,
    "`numpy.strings.isnumeric(a)` — True where every char is numeric (includes non-decimal numerics) and non-empty."
);
bind_predicate!(
    isdecimal,
    fs::isdecimal,
    "`numpy.strings.isdecimal(a)` — True where every char is a decimal char and the string is non-empty."
);
bind_predicate!(
    istitle,
    fs::istitle,
    "`numpy.strings.istitle(a)` — True where the string is title-cased."
);

// ---------------------------------------------------------------------------
// Index search (string -> int64, windowed; index/rindex raise when missing)
// ---------------------------------------------------------------------------

/// `numpy.strings.rfind(a, sub, start=0, end=None)` — highest absolute index
/// of `sub` within the `[start, end)` window, or -1 (`numpy/_core/strings.py`
/// `rfind`). Windowed at the boundary like [`find`], adding the resolved
/// start offset back to a found index so the reported index is absolute.
#[pyfunction]
#[pyo3(signature = (a, sub, start = 0, end = None))]
pub fn rfind<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, starts) = windowed_string_array(&sa, start, end)?;
    let r = fs::rfind(&wa, sub).map_err(ferr_to_pyerr)?;
    let adjusted: Vec<i64> = r
        .to_vec_flat()
        .into_iter()
        .zip(starts.iter())
        .map(|(idx, &st)| if idx >= 0 { idx + st as i64 } else { -1 })
        .collect();
    let r_dyn = ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), adjusted).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.strings.index(a, sub, start=0, end=None)` — like [`find`] but
/// raises `ValueError("substring not found")` when `sub` is absent from any
/// element's window (`numpy/_core/strings.py` `index`).
#[pyfunction]
#[pyo3(signature = (a, sub, start = 0, end = None))]
pub fn index<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, starts) = windowed_string_array(&sa, start, end)?;
    let r = fs::find(&wa, sub).map_err(ferr_to_pyerr)?;
    let mut adjusted: Vec<i64> = Vec::with_capacity(r.to_vec_flat().len());
    for (idx, &st) in r.to_vec_flat().into_iter().zip(starts.iter()) {
        if idx < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "substring not found",
            ));
        }
        adjusted.push(idx + st as i64);
    }
    let r_dyn = ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), adjusted).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.strings.rindex(a, sub, start=0, end=None)` — like [`rfind`] but
/// raises `ValueError("substring not found")` when `sub` is absent.
#[pyfunction]
#[pyo3(signature = (a, sub, start = 0, end = None))]
pub fn rindex<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sub: &str,
    start: i64,
    end: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let (wa, starts) = windowed_string_array(&sa, start, end)?;
    let r = fs::rfind(&wa, sub).map_err(ferr_to_pyerr)?;
    let mut adjusted: Vec<i64> = Vec::with_capacity(r.to_vec_flat().len());
    for (idx, &st) in r.to_vec_flat().into_iter().zip(starts.iter()) {
        if idx < 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "substring not found",
            ));
        }
        adjusted.push(idx + st as i64);
    }
    let r_dyn = ArrayD::<i64>::from_vec(IxDyn::new(r.shape()), adjusted).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Alignment / padding (string -> string, preserve content width)
// ---------------------------------------------------------------------------

/// Read the first char of a Python fillchar string, defaulting to space.
/// numpy requires `fillchar` to be exactly one character
/// (`numpy/_core/strings.py` raises `TypeError` otherwise); mirror that.
fn fill_char(fillchar: Option<&str>) -> PyResult<char> {
    match fillchar {
        None => Ok(' '),
        Some(s) => {
            let mut it = s.chars();
            match (it.next(), it.next()) {
                (Some(c), None) => Ok(c),
                _ => Err(pyo3::exceptions::PyTypeError::new_err(
                    "The fill character must be exactly one character long",
                )),
            }
        }
    }
}

/// `numpy.strings.center(a, width, fillchar=' ')`.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar = None))]
pub fn center<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    width: i64,
    fillchar: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    // CPython `str.center` puts the EXTRA odd-padding char on the RIGHT
    // (`'ab'.center(5)` -> '  ab '); the `_center` ufunc numpy delegates to
    // mirrors this (`numpy/_core/strings.py:693`). The `fs::center` kernel pads
    // the odd char on the left, so delegate to `numpy.strings.center`, which
    // owns the exact pad direction and the worst-case itemsize. Validate the
    // fillchar first so a multi-char fill still raises the binding's TypeError.
    let _fc = fill_char(fillchar)?;
    let np = py.import("numpy")?;
    let strings_mod = np.getattr("strings")?;
    match fillchar {
        Some(s) => strings_mod.call_method1("center", (a, width, s)),
        None => strings_mod.call_method1("center", (a, width)),
    }
}

/// `numpy.strings.ljust(a, width, fillchar=' ')`.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar = None))]
pub fn ljust<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    width: i64,
    fillchar: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let fc = fill_char(fillchar)?;
    let w = width.max(0) as usize;
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::ljust_with(&sa, w, fc).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.strings.rjust(a, width, fillchar=' ')`.
#[pyfunction]
#[pyo3(signature = (a, width, fillchar = None))]
pub fn rjust<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    width: i64,
    fillchar: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let fc = fill_char(fillchar)?;
    let w = width.max(0) as usize;
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::rjust_with(&sa, w, fc).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.strings.zfill(a, width)` — left-pad with zeros, keeping any leading
/// sign before the zeros (`numpy/_core/strings.py` `zfill`).
#[pyfunction]
pub fn zfill<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    width: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let w = width.max(0) as usize;
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::zfill(&sa, w).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

// ---------------------------------------------------------------------------
// expandtabs / slice / translate (string -> string)
// ---------------------------------------------------------------------------

/// `numpy.strings.expandtabs(a, tabsize=8)`.
#[pyfunction]
#[pyo3(signature = (a, tabsize = 8))]
pub fn expandtabs<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    tabsize: i64,
) -> PyResult<Bound<'py, PyAny>> {
    // numpy allocates the output dtype from the worst-case tab-expansion width,
    // so `expandtabs(<U3 'a\tb', 4)` keeps dtype `<U11` even though the content
    // is only 5 chars (`numpy/_core/strings.py` `expandtabs`). Round-tripping
    // through `numpy.asarray(list)` would shrink the dtype to the content width
    // `<U5`, so delegate to `numpy.strings.expandtabs` and return its result
    // directly (preserving the reserved `<U` itemsize).
    let np = py.import("numpy")?;
    let strings_mod = np.getattr("strings")?;
    strings_mod.call_method1("expandtabs", (a, tabsize))
}

/// `numpy.strings.slice(a, start=None, stop=None)` — per-element character
/// slice `a[i][start:stop]` (`numpy/_core/strings.py` `slice`). The library
/// `slice` is char-aware and Python-style for negative indices; `step` is not
/// supported and rejected if provided non-trivially.
#[pyfunction]
#[pyo3(signature = (a, start = None, stop = None, step = None))]
pub fn slice<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    start: Option<isize>,
    stop: Option<isize>,
    step: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(s) = step {
        if s != 1 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "strings.slice: step != 1 is not supported",
            ));
        }
    }
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::slice(&sa, start, stop).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.strings.translate(a, table, deletechars=None)` — apply a
/// per-character translation `table` (a `dict` mapping `ord(char)` to a
/// replacement string, a replacement ordinal, or `None` to delete) to each
/// element (`numpy/_core/strings.py` `translate`). `deletechars`, when given,
/// removes those characters as well.
#[pyfunction]
#[pyo3(signature = (a, table, deletechars = None))]
pub fn translate<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
    deletechars: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut map: HashMap<char, Option<char>> = HashMap::new();
    if let Ok(dict) = table.cast::<PyDict>() {
        for (k, v) in dict.iter() {
            let key_ord: u32 = k.extract()?;
            let key = char::from_u32(key_ord).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("translate: key is not a valid code point")
            })?;
            if v.is_none() {
                map.insert(key, None);
            } else if let Ok(repl_ord) = v.extract::<u32>() {
                let rc = char::from_u32(repl_ord).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "translate: replacement is not a valid code point",
                    )
                })?;
                map.insert(key, Some(rc));
            } else {
                let s: String = v.extract()?;
                let mut it = s.chars();
                match (it.next(), it.next()) {
                    (Some(c), None) => {
                        map.insert(key, Some(c));
                    }
                    (None, _) => {
                        map.insert(key, None);
                    }
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "translate: replacement strings must be a single character",
                        ));
                    }
                }
            }
        }
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "translate: table must be a dict mapping ordinals to replacements",
        ));
    }
    if let Some(del) = deletechars {
        for c in del.chars() {
            map.insert(c, None);
        }
    }
    let (sa, ndim) = py_to_string_array(py, a)?;
    let r = fs::translate(&sa, &map).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

/// `numpy.strings.mod(a, values)` — printf-`%`-style formatting per element.
///
/// `values` is broadcast against `a`; each element's `%`-conversions consume
/// the (stringified) per-element values in order. The library `mod_` parses a
/// supported subset (`%s %d %i %f %.Nf %e %g %%`); unsupported specifiers
/// surface as a `ValueError` rather than silently mis-formatting.
#[pyfunction]
pub fn r#mod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, ndim) = py_to_string_array(py, a)?;
    // Stringify the values into a flat positional arg list. A scalar / single
    // value applies to every element; a same-length sequence applies
    // element-wise.
    let np = py.import("numpy")?;
    let varr = np.call_method1("asarray", (values, "U"))?;
    let vshape: Vec<usize> = varr.getattr("shape")?.extract()?;
    let vflat = varr.call_method0("flatten")?;
    let vdata: Vec<String> = vflat.extract()?;
    // Format each element on its own single-element array so `mod_`'s
    // positional arg list applies per element.
    let format_one = |template: &str, args: &[&str]| -> PyResult<String> {
        let one = StringArray1::from_vec(Ix1::new([1]), vec![template.to_owned()])
            .map_err(ferr_to_pyerr)?;
        let r = fs::mod_(&one, args).map_err(ferr_to_pyerr)?;
        Ok(r.into_vec().into_iter().next().unwrap_or_default())
    };
    let out: Vec<String> = if vshape.is_empty() || vdata.len() == 1 {
        let args: Vec<&str> = vdata.iter().map(String::as_str).collect();
        sa.iter()
            .map(|s| format_one(s, &args))
            .collect::<Result<Vec<_>, _>>()?
    } else if vdata.len() == sa.len() {
        sa.iter()
            .zip(vdata.iter())
            .map(|(s, v)| format_one(s, &[v.as_str()]))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mod: values length does not match the operand",
        ));
    };
    let shape: Vec<usize> = sa.shape().to_vec();
    let r = StringArray::<IxDyn>::from_vec_dyn(&shape, out).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

// ---------------------------------------------------------------------------
// encode / decode
// ---------------------------------------------------------------------------

/// `numpy.strings.encode(a, encoding='utf-8', errors=None)` — encode each
/// element to a bytes array (dtype `|S`). Only UTF-8 is supported (the
/// workspace standardizes on UTF-8); other encodings raise.
#[pyfunction]
#[pyo3(signature = (a, encoding = None, errors = None))]
pub fn encode<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    encoding: Option<&str>,
    errors: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = errors;
    let enc = encoding.unwrap_or("utf-8");
    let (sa, ndim) = py_to_string_array(py, a)?;
    let shape: Vec<usize> = sa.shape().to_vec();
    let bytes = fs::encode(&sa, enc).map_err(ferr_to_pyerr)?;
    // Build a numpy bytes array (`|S`) honoring the input rank.
    let np = py.import("numpy")?;
    let py_bytes: Vec<Bound<'py, PyAny>> = bytes
        .into_iter()
        .map(|b| pyo3::types::PyBytes::new(py, &b).into_any())
        .collect();
    let list = PyList::new(py, &py_bytes)?;
    let arr = np.call_method1("asarray", (list, "S"))?;
    reshape_to_ndim(py, &arr, &shape, ndim)
}

/// `numpy.strings.decode(a, encoding='utf-8', errors=None)` — decode each
/// bytes element back to a unicode array (dtype `<U`). UTF-8 only.
#[pyfunction]
#[pyo3(signature = (a, encoding = None, errors = None))]
pub fn decode<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    encoding: Option<&str>,
    errors: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = errors;
    let enc = encoding.unwrap_or("utf-8");
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a, "S"))?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let ndim = shape.len();
    let flat = arr.call_method0("flatten")?;
    let byte_items: Vec<Vec<u8>> = {
        let mut out = Vec::new();
        for item in flat.try_iter()? {
            let item = item?;
            let b = item.cast::<pyo3::types::PyBytes>()?;
            out.push(b.as_bytes().to_vec());
        }
        out
    };
    let r = fs::decode(&byte_items, IxDyn::new(&shape), enc).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, r, ndim, None)
}

// ---------------------------------------------------------------------------
// partition / rpartition (string -> 3-tuple of arrays)
// ---------------------------------------------------------------------------

/// Marshal a flat `Vec<String>` into a same-shaped numpy unicode array.
fn vec_to_pyarray<'py>(
    py: Python<'py>,
    data: Vec<String>,
    shape: &[usize],
    ndim: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let sa = StringArray::<IxDyn>::from_vec_dyn(shape, data).map_err(ferr_to_pyerr)?;
    string_array_to_pyarray(py, sa, ndim, None)
}

/// `numpy.strings.partition(a, sep)` — returns a 3-tuple of arrays
/// `(before, sep, after)`, each the same shape as `a`
/// (`numpy/_core/strings.py` `partition`). If `sep` is absent the triple is
/// `(a, "", "")`.
#[pyfunction]
pub fn partition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sep: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // When `sep` is absent the trailing `after` field is empty and numpy
    // derives its width from the (absent) match, giving dtype `<U0`
    // (`numpy.strings.partition(['abc'], 'X')[2].dtype == '<U0'`). The binding's
    // `string_array_to_pyarray` floors an empty unicode field to `<U1`, so
    // delegate to `numpy.strings.partition`, which owns the exact `<U0`/width
    // contract for each of the three returned fields.
    let np = py.import("numpy")?;
    let strings_mod = np.getattr("strings")?;
    strings_mod.call_method1("partition", (a, sep))
}

/// `numpy.strings.rpartition(a, sep)` — like [`partition`] but splits on the
/// last occurrence; when absent the triple is `("", "", a)`.
#[pyfunction]
pub fn rpartition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sep: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, ndim) = py_to_string_array(py, a)?;
    let shape: Vec<usize> = sa.shape().to_vec();
    let triples = fs::rpartition(&sa, sep).map_err(ferr_to_pyerr)?;
    partition_tuple(py, triples, &shape, ndim)
}

fn partition_tuple<'py>(
    py: Python<'py>,
    triples: Vec<(String, String, String)>,
    shape: &[usize],
    ndim: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let mut before = Vec::with_capacity(triples.len());
    let mut seps = Vec::with_capacity(triples.len());
    let mut after = Vec::with_capacity(triples.len());
    for (b, s, a) in triples {
        before.push(b);
        seps.push(s);
        after.push(a);
    }
    let pb = vec_to_pyarray(py, before, shape, ndim)?;
    let ps = vec_to_pyarray(py, seps, shape, ndim)?;
    let pa = vec_to_pyarray(py, after, shape, ndim)?;
    Ok(PyTuple::new(py, [pb, ps, pa])?.into_any())
}

// ---------------------------------------------------------------------------
// split / rsplit / splitlines / join (ragged -> object array; join -> string)
// ---------------------------------------------------------------------------

/// Build a 1-D numpy object array whose elements are Python `list`s, matching
/// numpy's `split`/`rsplit`/`splitlines` output (`dtype=object`).
fn ragged_to_object_array<'py>(
    py: Python<'py>,
    rows: Vec<Vec<String>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let n = rows.len();
    // Build an empty object ndarray of shape (n,) then fill it, so numpy does
    // not try to broadcast the inner lists into a 2-D array.
    let arr = np.call_method1("empty", (n, "object"))?;
    for (i, row) in rows.into_iter().enumerate() {
        let lst = PyList::new(py, &row)?;
        arr.set_item(i, lst)?;
    }
    Ok(arr)
}

/// `numpy.strings.split(a, sep=None, maxsplit=None)` — returns a 1-D object
/// array of Python lists (`numpy/_core/strings.py` `split`). Only an explicit
/// non-empty `sep` is supported (whitespace-splitting `sep=None` is reported
/// as unsupported by the library).
#[pyfunction]
#[pyo3(signature = (a, sep, maxsplit = None))]
pub fn split<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sep: &str,
    maxsplit: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    // CPython `str.split(sep, maxsplit)` semantics: at most `maxsplit` splits,
    // remainder stays joined (`numpy/_core/defchararray.py:1081` ->
    // `strings.py:1441` `_vec_string(a, object_, 'split', [sep] + maxsplit)`).
    // The library `split_ragged` performs a full split only; when `maxsplit`
    // is given, delegate to `numpy.char.split` (which owns the exact CPython
    // bounded-split semantics) and return its object array directly.
    if maxsplit.is_some() {
        let np = py.import("numpy")?;
        let char_mod = np.getattr("char")?;
        return char_mod.call_method1("split", (a, sep, maxsplit));
    }
    let (sa, _) = py_to_string_array(py, a)?;
    let rows = fs::split_ragged(&sa, sep).map_err(ferr_to_pyerr)?;
    ragged_to_object_array(py, rows)
}

/// `numpy.strings.rsplit(a, sep=None, maxsplit=None)` — right-to-left
/// counterpart of [`split`], object array of lists.
#[pyfunction]
#[pyo3(signature = (a, sep, maxsplit = None))]
pub fn rsplit<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    sep: &str,
    maxsplit: Option<i64>,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let ms: Option<usize> = match maxsplit {
        Some(n) if n >= 0 => Some(n as usize),
        _ => None,
    };
    let padded = fs::rsplit(&sa, sep, ms).map_err(ferr_to_pyerr)?;
    // `rsplit` pads rows to a rectangle; trim trailing empties that are pure
    // padding so the object-array rows match Python's ragged `str.rsplit`.
    let rows = rectangular_to_ragged(&padded);
    ragged_to_object_array(py, rows)
}

/// `numpy.strings.splitlines(a, keepends=False)` — split on universal
/// newlines, object array of lists.
#[pyfunction]
#[pyo3(signature = (a, keepends = false))]
pub fn splitlines<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    keepends: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (sa, _) = py_to_string_array(py, a)?;
    let padded = fs::splitlines(&sa, keepends).map_err(ferr_to_pyerr)?;
    let rows = rectangular_to_ragged(&padded);
    ragged_to_object_array(py, rows)
}

/// Convert the library's rectangular `(n, max_parts)` padded split result back
/// into ragged rows by dropping the trailing all-empty padding columns of each
/// row. Empty original tokens in the interior are preserved.
fn rectangular_to_ragged(arr: &fs::StringArray2) -> Vec<Vec<String>> {
    let shape = arr.shape();
    let (n, cols) = (shape[0], shape[1]);
    let flat = arr.as_slice();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let mut row: Vec<String> = (0..cols).map(|j| flat[i * cols + j].clone()).collect();
        while row.len() > 1 && row.last().is_some_and(String::is_empty) {
            row.pop();
        }
        rows.push(row);
    }
    rows
}

/// `numpy.strings.join(sep, a)` — join the CHARACTERS of each element of `a`
/// with the corresponding `sep` (`numpy/_core/strings.py` `join`). `sep`
/// broadcasts against `a` (a scalar `sep` applies to every element). Output
/// shape matches `a`.
#[pyfunction]
pub fn join<'py>(
    py: Python<'py>,
    sep: &Bound<'py, PyAny>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Broadcast `sep` against `a` so a scalar separator applies element-wise.
    let (sep_arr, sa, ndim) = broadcast_pair(py, sep, a)?;
    let shape: Vec<usize> = sa.shape().to_vec();
    let joined: Vec<String> = sa
        .iter()
        .zip(sep_arr.iter())
        .map(|(elem, sp)| {
            elem.chars()
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(sp)
        })
        .collect();
    vec_to_pyarray(py, joined, &shape, ndim)
}

/// Reshape a freshly built (already-correct-dtype) numpy array to honor the
/// input rank, mirroring [`string_array_to_pyarray`]'s reshape logic for the
/// bytes (`encode`) path.
fn reshape_to_ndim<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    shape: &[usize],
    ndim: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if ndim == shape.len() {
        return arr.call_method1("reshape", (shape.to_vec(),));
    }
    if ndim == 0 {
        let empty = PyTuple::empty(py);
        return arr.call_method1("reshape", (empty,));
    }
    arr.call_method1("reshape", (shape.to_vec(),))
}

/// `numpy.char.compare_chararrays(a1, a2, cmp, rstrip)` — elementwise
/// comparison selected by the `cmp` operator string
/// (`numpy/_core/defchararray.py` `compare_chararrays`). `rstrip=True` strips
/// trailing whitespace from both operands before comparing.
#[pyfunction]
pub fn compare_chararrays<'py>(
    py: Python<'py>,
    a1: &Bound<'py, PyAny>,
    a2: &Bound<'py, PyAny>,
    cmp: &str,
    rstrip: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let (mut sa, mut sb, _ndim) = broadcast_pair(py, a1, a2)?;
    if rstrip {
        sa = fs::rstrip(&sa, None).map_err(ferr_to_pyerr)?;
        sb = fs::rstrip(&sb, None).map_err(ferr_to_pyerr)?;
    }
    let r = match cmp {
        "==" => fs::equal(&sa, &sb),
        "!=" => fs::not_equal(&sa, &sb),
        "<" => fs::less(&sa, &sb),
        "<=" => fs::less_equal(&sa, &sb),
        ">" => fs::greater(&sa, &sb),
        ">=" => fs::greater_equal(&sa, &sb),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "comparison must be '==', '!=', '<', '<=', '>', '>=', got {other:?}"
            )));
        }
    }
    .map_err(ferr_to_pyerr)?;
    let r_dyn =
        ArrayD::<bool>::from_vec(IxDyn::new(r.shape()), r.to_vec_flat()).map_err(ferr_to_pyerr)?;
    Ok(r_dyn.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}
