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
    let count = coerce_multiply_count(n)?;
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
