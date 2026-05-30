//! Shared helpers for the ferray Python bindings.
//!
//! - [`ferr_to_pyerr`] maps `FerrayError` variants onto CPython
//!   exception types.
//! - [`extract_shape`] coerces a Python int-or-sequence into
//!   `Vec<usize>`, matching `numpy.zeros(5)` and `numpy.zeros((3, 4))`.
//! - [`as_ndarray`] normalises any array-like Python object to a
//!   `numpy.ndarray` so dispatch can read its `.dtype`.
//! - [`dtype_name`] reads the canonical NumPy dtype name (`"float64"`,
//!   `"int32"`, …) from an `ndarray`.
//!
//! The crate-public macros [`match_dtype_all`] and [`match_dtype_float`]
//! handle the 11-way and 2-way dtype dispatch used by every binding
//! module. They expand into a `match` over a dtype string, binding a
//! local type alias `type T = …` in each arm so the body is written
//! once and Rust monomorphises it per dtype.

use ferray_core::FerrayError;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

/// Build a `numpy.exceptions.AxisError` for an out-of-bounds axis.
///
/// NumPy raises `numpy.exceptions.AxisError(axis, ndim)` (which
/// subclasses both `ValueError` and `IndexError`) whenever an `axis`
/// argument is outside `[-ndim, ndim)`. The binding layer must mirror
/// that exception *type* — a plain `ValueError` would let
/// `except numpy.exceptions.AxisError` clauses silently miss the error.
/// See numpy/exceptions.py `class AxisError(ValueError, IndexError)` and
/// numpy/_core/numeric.py:1427 `normalize_axis_tuple`.
pub fn axis_error(py: Python<'_>, axis: isize, ndim: usize) -> PyErr {
    match py
        .import("numpy.exceptions")
        .and_then(|m| m.getattr("AxisError"))
        .and_then(|cls| cls.call1((axis, ndim)))
    {
        Ok(exc) => PyErr::from_value(exc),
        // If numpy is somehow unavailable, fall back to a ValueError with
        // the same message numpy would produce. AxisError subclasses
        // ValueError, so `except ValueError` still catches both forms.
        Err(_) => PyValueError::new_err(format!(
            "axis {axis} is out of bounds for array of dimension {ndim}"
        )),
    }
}

/// Normalize a single (possibly negative) axis against `ndim`, mirroring
/// numpy's `normalize_axis_index`.
///
/// Maps `axis` in `[-ndim, ndim)` to `[0, ndim)` (negative axes count
/// from the end). An axis outside that range raises
/// `numpy.exceptions.AxisError` — *not* `OverflowError` (which a bare
/// `usize` extraction would give for a negative int) and *not* a plain
/// `ValueError`. See numpy/_core/numeric.py:1427.
pub fn normalize_axis(py: Python<'_>, axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    if axis < -n || axis >= n {
        return Err(axis_error(py, axis, ndim));
    }
    Ok(if axis < 0 {
        (axis + n) as usize
    } else {
        axis as usize
    })
}

/// Normalize a tuple/sequence of (possibly negative) axes against
/// `ndim`, mirroring numpy's `normalize_axis_tuple`
/// (numpy/_core/numeric.py:1427).
///
/// Each axis is normalized through [`normalize_axis`]. When
/// `allow_duplicate` is false, a repeated axis raises `ValueError`
/// (numpy: "repeated axis"), matching numpy's behaviour for
/// `expand_dims` / `flip` tuple axes.
pub fn normalize_axis_tuple(
    py: Python<'_>,
    axes: &[isize],
    ndim: usize,
    allow_duplicate: bool,
) -> PyResult<Vec<usize>> {
    let mut out: Vec<usize> = Vec::with_capacity(axes.len());
    for &ax in axes {
        let norm = normalize_axis(py, ax, ndim)?;
        if !allow_duplicate && out.contains(&norm) {
            return Err(PyValueError::new_err("repeated axis"));
        }
        out.push(norm);
    }
    Ok(out)
}

/// Extract an `axis`-like Python object into a vector of raw (un-normalized)
/// signed axes. Accepts a single int or a sequence of ints; the caller
/// normalizes against the relevant `ndim` via [`normalize_axis_tuple`].
/// Mirrors numpy's `if not isinstance(axis, (tuple, list)): axis = (axis,)`.
pub fn extract_axis_tuple(obj: &Bound<'_, PyAny>) -> PyResult<Vec<isize>> {
    if let Ok(n) = obj.extract::<isize>() {
        return Ok(vec![n]);
    }
    obj.extract::<Vec<isize>>()
        .map_err(|_| PyTypeError::new_err("axis must be an int or a tuple/list of ints"))
}

/// Parse a `percentile`/`quantile` `q` argument, which numpy documents as
/// `q : array_like of float` (numpy/lib/_function_base_impl.py:4083 for
/// `percentile`, :4284 for `quantile`).
///
/// Returns `Ok(Err(scalar))` for a single float `q` (numpy collapses the
/// new leading q-axis, returning the bare reduction) and `Ok(Ok(vec))` for
/// a sequence of floats (numpy stacks one reduction per q along a new
/// leading axis). The binding loops the scalar-`q` library call for each
/// entry and stacks the results, mirroring numpy's `q array_like` contract
/// (R-DEV-2). A non-numeric / non-sequence object raises `TypeError`,
/// matching numpy's `asarray(q)` failure.
#[allow(
    clippy::type_complexity,
    reason = "scalar-vs-sequence sum type is clearer inline than a named enum"
)]
pub fn extract_q(obj: &Bound<'_, PyAny>) -> PyResult<Result<Vec<f64>, f64>> {
    // A scalar must be matched first: a 0-d numpy array / Python float both
    // extract as f64, and numpy treats those as the scalar-q (collapsed) form.
    if let Ok(scalar) = obj.extract::<f64>() {
        return Ok(Err(scalar));
    }
    obj.extract::<Vec<f64>>()
        .map(Ok)
        .map_err(|_| PyTypeError::new_err("q must be a float or a sequence of floats"))
}

/// Normalize a single (possibly negative) `axis` for a `numpy.fft`
/// transform, raising a plain `IndexError` (NOT `numpy.exceptions.AxisError`)
/// when it is out of bounds or the input is 0-d.
///
/// `numpy.fft` does NOT route axis validation through `AxisError`. The
/// front-end (`numpy/fft/_pocketfft.py:213` `n = a.shape[axis]`) indexes the
/// shape tuple directly to default `n`, and `_raw_fft` later calls
/// `normalize_axis_index(axis, a.ndim)` (`_pocketfft.py:88`). Both surface a
/// plain `IndexError` ("tuple index out of range") for an out-of-range axis,
/// and the shape-index path raises `IndexError` for a 0-d input (`a.shape[-1]`
/// on an empty shape tuple). Confirmed live against numpy 2.4:
/// `type(np.fft.fft(np.float64(3.0)).exception) is IndexError`. ferray's
/// `resolve_axis` returns a `FerrayError::InvalidValue` that
/// [`ferr_to_pyerr`] would map to `ValueError`, so the binding must normalize
/// at the boundary to mirror numpy's exception *type* (R-DEV-2).
pub fn fft_normalize_axis(py: Python<'_>, axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    if ndim == 0 || axis < -n || axis >= n {
        let _ = py; // keep signature uniform with the other axis helpers
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "tuple index out of range",
        ));
    }
    Ok(if axis < 0 {
        (axis + n) as usize
    } else {
        axis as usize
    })
}

/// Validate a `numpy.fft` transform-length `n`, raising numpy's exact
/// `ValueError` when `n < 1`.
///
/// `numpy/fft/_pocketfft.py:59` — `if n < 1: raise ValueError(f"Invalid
/// number of FFT data points ({n}) specified.")`. ferray binds `n` as a bare
/// `usize`, so a negative Python int raises `OverflowError` at the PyO3
/// boundary before any length check runs. Binding `n` signed and validating
/// here mirrors numpy's exception *type* and message. `None` (the default)
/// passes through unchanged so the transform uses the input length.
pub fn validate_fft_n(n: Option<isize>) -> PyResult<Option<usize>> {
    match n {
        None => Ok(None),
        Some(v) => {
            if v < 1 {
                return Err(PyValueError::new_err(format!(
                    "Invalid number of FFT data points ({v}) specified."
                )));
            }
            Ok(Some(v as usize))
        }
    }
}

/// Resolve a `numpy.fft` N-D `s` (shape) sequence against the input shape,
/// honoring the numpy 2.0 `s[i] == -1` "use the whole input axis" sentinel.
///
/// `numpy/fft/_pocketfft.py:736-737` (`_cook_nd_args`) — `# use the whole
/// input array along axis 'i' if s[i] == -1` / `s = [a.shape[_a] if _s == -1
/// else _s for _s, _a in zip(s, axes)]`. ferray binds `s` as
/// `Option<Vec<usize>>`, so a `-1` entry raises `OverflowError` at the PyO3
/// boundary instead of being honored. Binding `s` signed lets the boundary
/// substitute `shape[axes[i]]` for each `-1` and validate every other entry
/// (`s[i] >= 1`, else numpy's `ValueError`). `axes` is the *already
/// normalized* axis list these `s` entries pair with; `None` `s` passes
/// through unchanged.
pub fn resolve_fft_s(
    s: Option<Vec<isize>>,
    axes: &[usize],
    shape: &[usize],
) -> PyResult<Option<Vec<usize>>> {
    let Some(s) = s else { return Ok(None) };
    if s.len() != axes.len() {
        return Err(PyValueError::new_err(
            "Shape and axes have different lengths.",
        ));
    }
    let mut out = Vec::with_capacity(s.len());
    for (&si, &ax) in s.iter().zip(axes.iter()) {
        if si == -1 {
            out.push(shape[ax]);
        } else if si < 1 {
            return Err(PyValueError::new_err(format!(
                "Invalid number of FFT data points ({si}) specified."
            )));
        } else {
            out.push(si as usize);
        }
    }
    Ok(Some(out))
}

/// Reject a complex-dtype input to a real-input FFT transform with the
/// numpy `TypeError`.
///
/// numpy's real transforms (`rfft`, `rfft2`, `rfftn`, `ihfft`) dispatch to
/// real-only ufuncs with no complex loop (`numpy/fft/_pocketfft.py:81`
/// `pfu.rfft_n_even` / `pfu.rfft_n_odd`), so a complex input raises
/// `TypeError: ufunc 'rfft_n_even' not supported for the input types ...`.
/// ferray instead silently coerces complex → float (discarding the imaginary
/// part) — a silent lossy round-trip across the boundary (R-CODE-4). This
/// guard detects a complex dtype (`name` starting `"complex"`) and raises the
/// matching `TypeError` before any cast (R-DEV-2). `dt` is the input's numpy
/// dtype name (from [`dtype_name`]).
pub fn reject_complex_for_real_fft(dt: &str, func: &str) -> PyResult<()> {
    if dt.starts_with("complex") {
        return Err(PyTypeError::new_err(format!(
            "ufunc '{func}' not supported for the input types, and the inputs \
             could not be safely coerced to any supported types"
        )));
    }
    Ok(())
}

/// Convert a [`FerrayError`] into the closest CPython exception.
///
/// Most ferray failure modes correspond to `ValueError` (bad shape,
/// invalid argument); dtype mismatches surface as `TypeError` to match
/// NumPy's behaviour where `np.zeros((3,), dtype="banana")` raises
/// `TypeError`.
pub fn ferr_to_pyerr(e: FerrayError) -> PyErr {
    let msg = e.to_string();
    match e {
        FerrayError::InvalidDtype { .. } => PyTypeError::new_err(msg),
        _ => PyValueError::new_err(msg),
    }
}

/// Convert a [`FerrayError`] from a `numpy.linalg.*` operation into the
/// closest CPython exception, raising `numpy.linalg.LinAlgError` where
/// NumPy does.
///
/// NumPy's linalg functions raise `numpy.linalg.LinAlgError` (a subclass
/// of `ValueError`) for the Linear-Algebra failure modes — singular
/// matrix (_linalg.py:145 `raise LinAlgError("Singular matrix")`),
/// non-positive-definite cholesky (_linalg.py:148
/// `raise LinAlgError("Matrix is not positive definite")`), and the
/// square-matrix-required guard (_linalg.py:259
/// `raise LinAlgError('Last 2 dimensions of the array must be square')`).
/// A bare `ValueError` would let `except numpy.linalg.LinAlgError`
/// clauses silently miss those errors, so the binding must mirror the
/// exact exception *type* (R-DEV-2).
///
/// ferray-linalg surfaces singular / non-convergence / numerical-
/// instability failures via [`FerrayError::is_linalg_error`], and the
/// non-square guard via a `ShapeMismatch` whose message names "square"
/// (e.g. `"inv requires a square matrix, got 2x3"`). Both map onto
/// `LinAlgError`. Everything else falls through to [`ferr_to_pyerr`].
pub fn linalg_err_to_pyerr(py: Python<'_>, e: FerrayError) -> PyErr {
    let is_linalg = e.is_linalg_error();
    let is_nonsquare =
        matches!(&e, FerrayError::ShapeMismatch { message } if message.contains("square"));
    if is_linalg || is_nonsquare {
        let msg = e.to_string();
        return match py
            .import("numpy.linalg")
            .and_then(|m| m.getattr("LinAlgError"))
            .and_then(|cls| cls.call1((msg.clone(),)))
        {
            Ok(exc) => PyErr::from_value(exc),
            // If numpy is somehow unavailable, fall back to a ValueError
            // with the same message. LinAlgError subclasses ValueError, so
            // `except ValueError` still catches both forms.
            Err(_) => PyValueError::new_err(msg),
        };
    }
    ferr_to_pyerr(e)
}

/// Coerce an integer/bool array-like to `float64` before a `numpy.linalg`
/// call, mirroring numpy's `_commonType` (`numpy/linalg/_linalg.py`),
/// which promotes any non-inexact (integer / bool) input to `float64`
/// before dispatching to LAPACK. det/inv/norm/solve of an int array must
/// therefore return `float64`, not raise. Inexact inputs (float / complex)
/// are returned unchanged so their dtype is preserved (R-CODE-4).
pub fn promote_linalg_input<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, obj)?;
    let dt = dtype_name(&arr)?;
    if matches!(
        dt.as_str(),
        "float64" | "f64" | "float32" | "f32" | "float16" | "f16" | "complex64" | "complex128"
    ) {
        return Ok(arr);
    }
    // Integer / bool / anything else -> float64 (numpy `_commonType`).
    coerce_dtype(py, &arr, "float64")
}

/// Coerce a window-length argument `M` to a signed integer count, mirroring
/// NumPy's window functions.
///
/// NumPy types `M` as `_FloatLike_co` and casts it to `float64` internally —
/// every window does `values = np.array([0.0, M]); M = values[1]`
/// (numpy/lib/_function_base_impl.py:3346 `hanning`, :3244 `bartlett`,
/// :3137 `blackman`, :3445 `hamming`, :3727 `kaiser`). So both an integer
/// (`np.hanning(8)`) and an integral float (`np.hanning(8.0)`) are accepted,
/// and a value `< 1` (e.g. `-1`, `-1.0`, `0`) routes through each window's
/// `if M < 1: return array([], dtype=float64)` guard
/// (:3349-3350 `hanning`, etc.; `kaiser` reaches the empty array via
/// `n = arange(0, M)` over a non-positive `M`, :3733).
///
/// The ferray-window library typed `M` as a `usize` count, so a negative or
/// float `M` was rejected at the PyO3 boundary (`OverflowError` /
/// `TypeError`). This helper restores numpy's contract on the marshalling
/// side: accept any int- or float-coercible object, truncate toward zero to
/// an integer count (numpy's window math is defined on integer sample counts
/// for the integral-`M` inputs these bindings target), and let the caller
/// short-circuit `M < 1 -> empty float64 array`. A non-numeric object raises
/// `TypeError`, matching numpy (`np.hanning('x')` -> TypeError).
pub fn coerce_window_m(obj: &Bound<'_, PyAny>) -> PyResult<isize> {
    if let Ok(n) = obj.extract::<isize>() {
        return Ok(n);
    }
    if let Ok(f) = obj.extract::<f64>() {
        if !f.is_finite() {
            return Err(PyValueError::new_err(
                "M must be a finite number (got a non-finite float)",
            ));
        }
        // Truncate toward zero, matching `int(M)` for the integral-float
        // inputs these bindings target.
        return Ok(f.trunc() as isize);
    }
    Err(PyTypeError::new_err(
        "M must be an integer or a float (got a non-numeric object)",
    ))
}

/// Clear the `WRITEABLE` flag on a numpy `ndarray`, making it read-only,
/// then return it.
///
/// NumPy's `broadcast_to` (numpy/lib/_stride_tricks_impl.py:517
/// `_broadcast_to(array, shape, subok=subok, readonly=True)`) and
/// `sliding_window_view` (`:181 writeable=False` default) return read-only
/// views whose `flags.writeable` is `False`. ferray materialises an owned
/// copy at the boundary, which numpy-pyarray creates writeable; to honor
/// numpy's output-object contract (R-DEV-3) the binding sets
/// `result.flags.writeable = False` — the same mechanism numpy itself uses
/// internally (`result.flags.writeable = True/False`). Assigning the flag
/// is numpy's public, supported API for toggling writeability on an array
/// the caller owns.
pub fn set_readonly<'py>(arr: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let flags = arr.getattr("flags")?;
    flags.setattr("writeable", false)?;
    Ok(arr)
}

/// Return the `numpy.ma.masked` singleton.
///
/// `numpy.ma`'s all-masked full reductions don't return a finite/NaN scalar —
/// they return the module-level `masked` constant (`numpy/ma/core.py:5250`
/// `result = masked`, mirrored for mean/min/max/var/std), so user code can
/// test `am.sum() is np.ma.masked`. The ferray-ma library has no Python
/// `masked` object and surfaces the all-masked case as its `NaN` analog;
/// the binding maps that back to the genuine singleton (R-DEV-3) so the
/// `is`-identity contract holds across the boundary.
pub fn ma_masked_singleton<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy.ma")?.getattr("masked")
}

/// Return the `numpy.ma.nomask` singleton.
///
/// `numpy.ma.getmask` of an array with no mask returns the module-level
/// `nomask` constant (`numpy/ma/core.py:1468` `return getattr(a, '_mask',
/// nomask)`), not a full `array([False, ...])`. The `.mask` property mirrors
/// this. The binding returns the genuine singleton so `getmask(a) is
/// np.ma.nomask` holds (R-DEV-3).
pub fn ma_nomask<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy.ma")?.getattr("nomask")
}

/// Accept either a Python integer (1-D) or a sequence of integers
/// (N-D). Mirrors `numpy.zeros(5)` and `numpy.zeros((3, 4))`.
pub fn extract_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if let Ok(n) = obj.extract::<usize>() {
        return Ok(vec![n]);
    }
    obj.extract::<Vec<usize>>().map_err(|_| {
        PyTypeError::new_err(
            "shape must be an int or a sequence of ints (got a non-coercible object)",
        )
    })
}

/// Accept a Python int (1-D) or a sequence of ints (N-D) as a *signed*
/// shape, then validate every dimension is non-negative — raising
/// `ValueError("negative dimensions are not allowed")` for a negative one.
///
/// NumPy rejects a negative dimension with a `ValueError`, not a
/// `TypeError`/`OverflowError`: the shape is parsed into a signed
/// `npy_intp` array and then checked (`PyArray_IntpFromSequence` ->
/// the dimension-validation loop in `numpy/_core/src/multiarray/ctors.c`),
/// e.g. `np.zeros(-1)` -> `ValueError: negative dimensions are not
/// allowed`. Binding `shape` as `usize` would instead surface a
/// `TypeError`/`OverflowError` at PyO3 extraction, before ferray's own
/// check runs — so we extract signed and validate here to mirror numpy's
/// exception *type* (R-DEV-2).
pub fn extract_signed_shape(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let signed: Vec<isize> = if let Ok(n) = obj.extract::<isize>() {
        vec![n]
    } else {
        obj.extract::<Vec<isize>>().map_err(|_| {
            PyTypeError::new_err(
                "shape must be an int or a sequence of ints (got a non-coercible object)",
            )
        })?
    };
    let mut out = Vec::with_capacity(signed.len());
    for d in signed {
        if d < 0 {
            return Err(PyValueError::new_err("negative dimensions are not allowed"));
        }
        out.push(d as usize);
    }
    Ok(out)
}

/// Validate a single (possibly negative) dimension count `n` and return it
/// as `usize`, raising `ValueError("negative dimensions are not allowed")`
/// when negative.
///
/// Used by `eye`/`identity` whose `n`/`m` are scalar counts. `np.eye(-1)`
/// and `np.identity(-2)` raise `ValueError` (they route through
/// `zeros((N, M))` / `eye` -> `zeros`), not `OverflowError`.
pub fn validate_dim(n: isize) -> PyResult<usize> {
    if n < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    Ok(n as usize)
}

/// Validate a `num`/`num_samples` argument for `linspace`/`logspace`/
/// `geomspace`, raising numpy's exact message on a negative value.
///
/// `numpy/_core/function_base.py:122-126` linspace:
/// `num = operator.index(num); if num < 0: raise ValueError(
/// f"Number of samples, {num}, must be non-negative.")`. Binding `num` as
/// `usize` would raise `OverflowError` at PyO3 extraction; we bind it
/// signed and validate here to mirror numpy's `ValueError` type + message.
pub fn validate_num_samples(num: isize) -> PyResult<usize> {
    if num < 0 {
        return Err(PyValueError::new_err(format!(
            "Number of samples, {num}, must be non-negative."
        )));
    }
    Ok(num as usize)
}

/// Infer the default numpy dtype name for a Python scalar passed to a
/// creation function (`arange`, `full`), keying off the operand's *type*
/// rather than its numeric value.
///
/// NumPy's `np.full` defaults `dtype` to `np.array(fill_value).dtype`
/// (`numpy/_core/numeric.py:382-384`), and `np.arange` likewise dispatches
/// on the operand type — so the dtype follows the *Python type*, not
/// whether the value happens to be integral:
///   - a Python `bool` (`True`/`False`)            -> `"bool"`
///   - a Python `int`                              -> `"int64"`
///   - a Python `float` (even an integral `1.0`)   -> `"float64"`
///
/// `bool` is checked before `int` because `bool` is a subclass of `int`
/// in CPython (`isinstance(True, int)` is `True`). A non-numeric object
/// returns `None` so the caller can fall back / error. Returning `&'static
/// str` keeps the dtype-string contract the dispatch macros consume.
pub fn pyscalar_default_dtype(obj: &Bound<'_, PyAny>) -> Option<&'static str> {
    // `bool` first: it subclasses `int`, so the `int` check would shadow it.
    if obj.is_instance_of::<pyo3::types::PyBool>() {
        return Some("bool");
    }
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return Some("int64");
    }
    if obj.is_instance_of::<pyo3::types::PyFloat>() {
        return Some("float64");
    }
    None
}

/// Normalize any dtype-like Python object — a `str` (`"float64"`), a numpy
/// scalar *type* object (`numpy.float64`), or a `numpy.dtype` instance
/// (`numpy.dtype("int8")`) — to its canonical numpy dtype name string
/// (`"float64"`, `"int8"`, `"complex128"`, …).
///
/// numpy's creation functions accept all three forms in `dtype=` because the
/// front-ends funnel the argument through `numpy.dtype(obj)`
/// (`numpy/_core/numeric.py def full` -> `empty(shape, dtype, …)`; the C
/// `PyArray_DescrConverter` accepts a type object, a `dtype` instance, or a
/// string). ferray's bindings bound `dtype` as a PyO3 `&str`, so passing the
/// type object ferray itself now exposes (`fr.float64`) raised
/// `TypeError: argument 'dtype': 'type' object is not an instance of 'str'`.
/// Routing the incoming object through `numpy.dtype(obj).name` reproduces
/// numpy's exact acceptance set and yields the canonical string the dispatch
/// macros consume (R-DEV-2). An object numpy can't interpret as a dtype
/// surfaces numpy's own `TypeError` here, matching `np.zeros(3, dtype=object())`.
pub fn normalize_dtype(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let np = py.import("numpy")?;
    let dt = np.getattr("dtype")?.call1((obj,))?;
    dt.getattr("name")?.extract()
}

/// Normalize an optional `dtype=` argument: `None` (the default) passes
/// through as `None`, otherwise the object is normalized through
/// [`normalize_dtype`]. Used by every creation binding whose `dtype` defaults
/// to "inherit / infer" (`full`, `arange`, `linspace`, `*_like`, …).
pub fn normalize_opt_dtype(
    py: Python<'_>,
    obj: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<String>> {
    match obj {
        None => Ok(None),
        Some(o) => Ok(Some(normalize_dtype(py, o)?)),
    }
}

/// Normalise any array-like Python object (numpy.ndarray, list, tuple,
/// nested sequence) to a `numpy.ndarray` by routing through
/// `numpy.asarray`. The returned object's `.dtype` is then queryable
/// by the dispatch macros.
pub fn as_ndarray<'py>(py: Python<'py>, obj: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    np.call_method1("asarray", (obj,))
}

/// Read the canonical NumPy dtype name (e.g. `"float64"`) from a
/// `numpy.ndarray`. The caller must ensure `arr` is already an
/// `ndarray`; pass it through [`as_ndarray`] first if not.
pub fn dtype_name(arr: &Bound<'_, PyAny>) -> PyResult<String> {
    arr.getattr("dtype")?.getattr("name")?.extract()
}

// ---------------------------------------------------------------------------
// DynArray <-> numpy.ndarray marshalling (file-IO boundary, refs #832)
//
// ferray-io's binary/text I/O is typed in `ferray_core::DynArray` (a
// runtime-typed array enum) on the Rust side, but the Python boundary is a
// `numpy.ndarray`. These two helpers bridge that gap so the io.rs bindings
// can stay thin:
//
//   numpy.ndarray --pyany_to_dynarray--> DynArray  (write path: save/savez/...)
//   DynArray --dynarray_to_pyarray-----> numpy.ndarray (read path: load/...)
//
// The dtype set is the `NpElement` set the numpy-interop crate already
// supports (bool, int8/16/32/64, uint8/16/32/64, float32/64) — exactly the
// dtypes that round-trip through `PyReadonlyArrayDyn<T>` / `into_pyarray`.
// numpy's `.npy` format also stores complex / datetime arrays; those dtypes
// are not in the interop `NpElement` set, so they surface a `TypeError` here
// rather than a silent lossy cast (R-CODE-4) and are tracked as a follow-up.
// ---------------------------------------------------------------------------

/// Convert any array-like Python object into a `ferray_core::DynArray`,
/// dispatching on the input's numpy dtype.
///
/// The object is first normalized to a C-contiguous `numpy.ndarray` via
/// `numpy.ascontiguousarray` (the interop `as_ferray` reads elements in
/// C order), then a `PyReadonlyArrayDyn<T>` view is taken for the matched
/// dtype and copied into the corresponding `DynArray` variant. This is the
/// inverse of [`dynarray_to_pyarray`] and preserves numpy's dtype + shape
/// across the boundary (R-CODE-4).
pub fn pyany_to_dynarray(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<ferray_core::DynArray> {
    use ferray_core::DynArray;
    use ferray_numpy_interop::AsFerray;
    use numpy::PyReadonlyArrayDyn;

    // Normalize to a contiguous ndarray so the IxDyn view reads in C order.
    let arr = py
        .import("numpy")?
        .call_method1("ascontiguousarray", (obj,))?;
    let dt = dtype_name(&arr)?;

    macro_rules! to_dyn {
        ($ty:ty, $variant:ident) => {{
            let view = arr.extract::<PyReadonlyArrayDyn<$ty>>()?;
            let fa = view.as_ferray().map_err(ferr_to_pyerr)?;
            Ok(DynArray::$variant(fa))
        }};
    }

    match dt.as_str() {
        "float64" | "f64" => to_dyn!(f64, F64),
        "float32" | "f32" => to_dyn!(f32, F32),
        "int64" | "i64" => to_dyn!(i64, I64),
        "int32" | "i32" => to_dyn!(i32, I32),
        "int16" | "i16" => to_dyn!(i16, I16),
        "int8" | "i8" => to_dyn!(i8, I8),
        "uint64" | "u64" => to_dyn!(u64, U64),
        "uint32" | "u32" => to_dyn!(u32, U32),
        "uint16" | "u16" => to_dyn!(u16, U16),
        "uint8" | "u8" => to_dyn!(u8, U8),
        "bool" => to_dyn!(bool, Bool),
        other => Err(PyTypeError::new_err(format!(
            "ferray file-IO does not yet support dtype {other:?} (supported: \
             bool, int8/16/32/64, uint8/16/32/64, float32/64)"
        ))),
    }
}

/// Convert a `ferray_core::DynArray` into a Python-owned `numpy.ndarray`,
/// dispatching on the array's runtime dtype.
///
/// This is the inverse of [`pyany_to_dynarray`]: the inner
/// `Array<T, IxDyn>` is pushed across the boundary via the interop
/// `IntoNumPy` impl, preserving the loaded dtype + shape so a `load` round
/// trips a `save` exactly (R-DEV-3). A `DynArray` carrying a dtype outside
/// the interop `NpElement` set (complex / datetime / 128-bit) raises a
/// `TypeError` rather than a lossy cast.
pub fn dynarray_to_pyarray<'py>(
    py: Python<'py>,
    arr: ferray_core::DynArray,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::DynArray;
    use ferray_numpy_interop::IntoNumPy;

    macro_rules! from_dyn {
        ($inner:expr) => {{ Ok($inner.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()) }};
    }

    match arr {
        DynArray::F64(a) => from_dyn!(a),
        DynArray::F32(a) => from_dyn!(a),
        DynArray::I64(a) => from_dyn!(a),
        DynArray::I32(a) => from_dyn!(a),
        DynArray::I16(a) => from_dyn!(a),
        DynArray::I8(a) => from_dyn!(a),
        DynArray::U64(a) => from_dyn!(a),
        DynArray::U32(a) => from_dyn!(a),
        DynArray::U16(a) => from_dyn!(a),
        DynArray::U8(a) => from_dyn!(a),
        DynArray::Bool(a) => from_dyn!(a),
        other => Err(PyTypeError::new_err(format!(
            "ferray file-IO cannot marshal dtype {} to numpy yet (supported: \
             bool, int8/16/32/64, uint8/16/32/64, float32/64)",
            other.dtype()
        ))),
    }
}

/// Dispatch over all 11 supported element types.
///
/// Inside the body, the type alias `T` is bound to the concrete
/// element type, so a single body expression like
///
/// ```ignore
/// match_dtype_all!(name, T => {
///     let view = arr.extract::<PyReadonlyArrayDyn<T>>()?;
///     let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
///     // ... call typed ferray fn returning ArrayD<T> or Array1<T> ...
///     result.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
/// })
/// ```
///
/// expands into 11 monomorphic arms plus a `TypeError` fallback. The
/// body must end in an expression of `Bound<'py, PyAny>`; the macro is
/// itself an expression so it can sit inside `Ok(...)`.
#[macro_export]
macro_rules! match_dtype_all {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "float64" | "f64" => { #[allow(non_camel_case_types)] type $T = f64; $body }
            "float32" | "f32" => { #[allow(non_camel_case_types)] type $T = f32; $body }
            "int64"   | "i64" => { #[allow(non_camel_case_types)] type $T = i64; $body }
            "int32"   | "i32" => { #[allow(non_camel_case_types)] type $T = i32; $body }
            "int16"   | "i16" => { #[allow(non_camel_case_types)] type $T = i16; $body }
            "int8"    | "i8"  => { #[allow(non_camel_case_types)] type $T = i8;  $body }
            "uint64"  | "u64" => { #[allow(non_camel_case_types)] type $T = u64; $body }
            "uint32"  | "u32" => { #[allow(non_camel_case_types)] type $T = u32; $body }
            "uint16"  | "u16" => { #[allow(non_camel_case_types)] type $T = u16; $body }
            "uint8"   | "u8"  => { #[allow(non_camel_case_types)] type $T = u8;  $body }
            "bool"             => { #[allow(non_camel_case_types)] type $T = bool; $body }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype: {other:?} (supported: bool, int8/16/32/64, uint8/16/32/64, float32/64)"
                )));
            }
        }
    };
}

/// Dispatch restricted to floating-point element types. Use for
/// functions whose ferray-side bound is `LinspaceNum` or `Float`
/// (linspace, logspace, geomspace, special functions).
#[macro_export]
macro_rules! match_dtype_float {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "float64" | "f64" => { #[allow(non_camel_case_types)] type $T = f64; $body }
            "float32" | "f32" => { #[allow(non_camel_case_types)] type $T = f32; $body }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype for floating-point op: {other:?} (supported: float32, float64)"
                )));
            }
        }
    };
}

/// Dispatch over numeric element types (integers + floats, no bool).
/// Use for arithmetic ops whose ferray-side bound is
/// `Add/Sub/Mul/Div<Output = T> + Copy` — bool doesn't implement
/// these and trying to monomorphise yields a confusing error.
#[macro_export]
macro_rules! match_dtype_numeric {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "float64" | "f64" => { #[allow(non_camel_case_types)] type $T = f64; $body }
            "float32" | "f32" => { #[allow(non_camel_case_types)] type $T = f32; $body }
            "int64"   | "i64" => { #[allow(non_camel_case_types)] type $T = i64; $body }
            "int32"   | "i32" => { #[allow(non_camel_case_types)] type $T = i32; $body }
            "int16"   | "i16" => { #[allow(non_camel_case_types)] type $T = i16; $body }
            "int8"    | "i8"  => { #[allow(non_camel_case_types)] type $T = i8;  $body }
            "uint64"  | "u64" => { #[allow(non_camel_case_types)] type $T = u64; $body }
            "uint32"  | "u32" => { #[allow(non_camel_case_types)] type $T = u32; $body }
            "uint16"  | "u16" => { #[allow(non_camel_case_types)] type $T = u16; $body }
            "uint8"   | "u8"  => { #[allow(non_camel_case_types)] type $T = u8;  $body }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype for numeric op: {other:?} (supported: int8/16/32/64, uint8/16/32/64, float32/64)"
                )));
            }
        }
    };
}

/// Dispatch over types that have a meaningful comparison (PartialOrd
/// is not required at the macro level, but the ferray-side bound
/// always is). Same arms as `match_dtype_numeric!` plus bool — equality
/// comparisons over bool make sense; ordering comparisons may not but
/// the ferray-side bound enforces that per-function.
#[macro_export]
macro_rules! match_dtype_orderable {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "float64" | "f64" => {
                #[allow(non_camel_case_types)]
                type $T = f64;
                $body
            }
            "float32" | "f32" => {
                #[allow(non_camel_case_types)]
                type $T = f32;
                $body
            }
            "int64" | "i64" => {
                #[allow(non_camel_case_types)]
                type $T = i64;
                $body
            }
            "int32" | "i32" => {
                #[allow(non_camel_case_types)]
                type $T = i32;
                $body
            }
            "int16" | "i16" => {
                #[allow(non_camel_case_types)]
                type $T = i16;
                $body
            }
            "int8" | "i8" => {
                #[allow(non_camel_case_types)]
                type $T = i8;
                $body
            }
            "uint64" | "u64" => {
                #[allow(non_camel_case_types)]
                type $T = u64;
                $body
            }
            "uint32" | "u32" => {
                #[allow(non_camel_case_types)]
                type $T = u32;
                $body
            }
            "uint16" | "u16" => {
                #[allow(non_camel_case_types)]
                type $T = u16;
                $body
            }
            "uint8" | "u8" => {
                #[allow(non_camel_case_types)]
                type $T = u8;
                $body
            }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype for ordering op: {other:?}"
                )));
            }
        }
    };
}

/// Dispatch over types that have meaningful bitwise operations:
/// integer types (signed + unsigned) plus bool. Floats are excluded.
#[macro_export]
macro_rules! match_dtype_bitwise {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "int64" | "i64" => {
                #[allow(non_camel_case_types)]
                type $T = i64;
                $body
            }
            "int32" | "i32" => {
                #[allow(non_camel_case_types)]
                type $T = i32;
                $body
            }
            "int16" | "i16" => {
                #[allow(non_camel_case_types)]
                type $T = i16;
                $body
            }
            "int8" | "i8" => {
                #[allow(non_camel_case_types)]
                type $T = i8;
                $body
            }
            "uint64" | "u64" => {
                #[allow(non_camel_case_types)]
                type $T = u64;
                $body
            }
            "uint32" | "u32" => {
                #[allow(non_camel_case_types)]
                type $T = u32;
                $body
            }
            "uint16" | "u16" => {
                #[allow(non_camel_case_types)]
                type $T = u16;
                $body
            }
            "uint8" | "u8" => {
                #[allow(non_camel_case_types)]
                type $T = u8;
                $body
            }
            "bool" => {
                #[allow(non_camel_case_types)]
                type $T = bool;
                $body
            }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype for bitwise op: {other:?} (supported: bool + integer types)"
                )));
            }
        }
    };
}

/// Dispatch restricted to integer types only (no bool). Used for shifts
/// where the bit-shift amount must be a numeric integer type.
#[macro_export]
macro_rules! match_dtype_int_only {
    ($dtype:expr, $T:ident => $body:block) => {
        match $dtype {
            "int64" | "i64" => {
                #[allow(non_camel_case_types)]
                type $T = i64;
                $body
            }
            "int32" | "i32" => {
                #[allow(non_camel_case_types)]
                type $T = i32;
                $body
            }
            "int16" | "i16" => {
                #[allow(non_camel_case_types)]
                type $T = i16;
                $body
            }
            "int8" | "i8" => {
                #[allow(non_camel_case_types)]
                type $T = i8;
                $body
            }
            "uint64" | "u64" => {
                #[allow(non_camel_case_types)]
                type $T = u64;
                $body
            }
            "uint32" | "u32" => {
                #[allow(non_camel_case_types)]
                type $T = u32;
                $body
            }
            "uint16" | "u16" => {
                #[allow(non_camel_case_types)]
                type $T = u16;
                $body
            }
            "uint8" | "u8" => {
                #[allow(non_camel_case_types)]
                type $T = u8;
                $body
            }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype: {other:?} (integer types required)"
                )));
            }
        }
    };
}

/// Dispatch a binary (or unary) op that has SEPARATE float and integer
/// implementations: floating dtypes route to `$float_fn`, integer dtypes
/// to `$int_fn`. Used where numpy registers both a float loop and an
/// integer loop with different semantics — e.g. `maximum` (NaN-propagating
/// float vs. ordered int), `power` (`powf` vs. wrapping int pow),
/// `floor_divide` / `remainder` (true float floor-div vs. integer
/// floor-div). bool is rejected (these ops have no bool loop in numpy).
#[macro_export]
macro_rules! match_dtype_float_or_int {
    ($dtype:expr, $T:ident, $float_fn:path, $int_fn:path => $body:block) => {
        match $dtype {
            "float64" | "f64" => {
                #[allow(non_camel_case_types)]
                type $T = f64;
                macro_rules! __op {
                    () => {
                        $float_fn
                    };
                }
                $body
            }
            "float32" | "f32" => {
                #[allow(non_camel_case_types)]
                type $T = f32;
                macro_rules! __op {
                    () => {
                        $float_fn
                    };
                }
                $body
            }
            "int64" | "i64" => {
                #[allow(non_camel_case_types)]
                type $T = i64;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "int32" | "i32" => {
                #[allow(non_camel_case_types)]
                type $T = i32;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "int16" | "i16" => {
                #[allow(non_camel_case_types)]
                type $T = i16;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "int8" | "i8" => {
                #[allow(non_camel_case_types)]
                type $T = i8;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "uint64" | "u64" => {
                #[allow(non_camel_case_types)]
                type $T = u64;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "uint32" | "u32" => {
                #[allow(non_camel_case_types)]
                type $T = u32;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "uint16" | "u16" => {
                #[allow(non_camel_case_types)]
                type $T = u16;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            "uint8" | "u8" => {
                #[allow(non_camel_case_types)]
                type $T = u8;
                macro_rules! __op {
                    () => {
                        $int_fn
                    };
                }
                $body
            }
            other => {
                return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                    "unsupported dtype for numeric op: {other:?}"
                )));
            }
        }
    };
}

/// Coerce a `numpy.char.multiply` / `numpy.strings.multiply` repeat count
/// to a non-negative `usize`, clamping a negative count to `0`.
///
/// `numpy/_core/strings.py:155` documents "Values in ``i`` of less than 0
/// are treated as 0 (which yields an empty string)", implemented at
/// `:195` as `i = np.maximum(i, 0)`. So `np.char.multiply(['ab'], -2)`
/// returns `['']`, it does NOT raise. The binding previously typed the
/// count as a bare PyO3 `usize`, which raises `OverflowError` at extraction
/// for any negative Python int — surfacing the wrong exception and never
/// reaching numpy's clamp. Binding the count signed and clamping here
/// mirrors numpy's contract (R-DEV-2). A non-integer raises `TypeError`,
/// matching numpy's `raise TypeError(... for operand 'i')` (`:194`).
pub fn coerce_multiply_count(obj: &Bound<'_, PyAny>) -> PyResult<usize> {
    let n: i64 = obj
        .extract::<i64>()
        .map_err(|_| PyTypeError::new_err("unsupported type for operand 'i' (expected integer)"))?;
    Ok(if n < 0 { 0 } else { n as usize })
}

/// Coerce a Python array-like to a `numpy.ndarray` of the requested
/// dtype. Used by binary ufuncs to align two inputs to a common dtype
/// before extracting typed views — the alternative (extract both then
/// reject mismatches) would surprise users who pass e.g. an `int`
/// alongside a `float` array.
pub fn coerce_dtype<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("asarray", (obj, dt))
}

/// Compute the NumPy NEP-50 promoted dtype name for a binary ufunc over
/// two array-likes, mirroring `numpy.result_type(a, b)`
/// (numpy/_core/src/multiarray/convert_datatype.c — the type-promotion
/// table that every binary ufunc consults before selecting a loop).
///
/// `add([1,2], [1.5,2.5])` promotes `int64 + float64 -> float64`, so the
/// float operand's fractional part survives instead of being truncated
/// to the first operand's integer dtype. `maximum([1,5],[3,2])` promotes
/// `int64 + int64 -> int64`, keeping the integer loop.
pub fn binary_result_dtype<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let np = py.import("numpy")?;
    let dt = np.getattr("result_type")?.call1((a, b))?;
    dt.getattr("name")?.extract()
}

/// For a unary "promote-to-float" ufunc (`exp`, `log`, `sqrt`, `sin`,
/// `fabs`, …), return `(compute_dtype, output_dtype)` for an input whose
/// dtype is `in_dt`.
///
/// NumPy promotes integer/bool input to the smallest inexact float that
/// safely holds it (`result_type(x, float16)`): bool/int8/uint8 ->
/// float16, int16/uint16 -> float32, the wider ints -> float64
/// (generate_umath.py:1003 `fabs` `TD(flts ...)` / the inexact common
/// type). Float input keeps its dtype. The binding computes in float32 for
/// any float16/float32 output (numpy's f16 ufuncs upcast to f32 internally)
/// and in float64 otherwise, then narrows the result back to `output_dtype`
/// at the boundary — so no `half::f16` plumbing is needed in Rust while the
/// returned array still carries numpy's exact promoted dtype.
pub fn unary_promote_dtypes<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
    in_dt: &str,
) -> PyResult<(String, String)> {
    if matches!(
        in_dt,
        "float64" | "f64" | "float32" | "f32" | "float16" | "f16"
    ) {
        let compute = if matches!(in_dt, "float64" | "f64") {
            "float64"
        } else {
            "float32"
        };
        return Ok((compute.to_string(), in_dt.to_string()));
    }
    let np = py.import("numpy")?;
    let f16 = np.getattr("float16")?;
    let out = np.getattr("result_type")?.call1((sample, f16))?;
    let out_name: String = out.getattr("name")?.extract()?;
    let compute = if out_name == "float64" {
        "float64"
    } else {
        "float32"
    };
    Ok((compute.to_string(), out_name))
}

/// Return `true` if every argument is a 0-dimensional array-like (a Python
/// scalar, a numpy scalar, or a 0-d `ndarray`).
///
/// NumPy's `$OUT_SCALAR` contract: a ufunc whose inputs are all scalar /
/// 0-d returns a numpy *scalar* (e.g. `numpy.float64`), not a 0-d
/// `ndarray` (numpy/_core/code_generators/ufunc_docstrings.py
/// `$OUT_SCALAR_2`). The binding produces a 0-d ndarray, then collapses it
/// to a scalar via [`scalarize`] when this predicate holds.
pub fn all_scalar_inputs(py: Python<'_>, inputs: &[&Bound<'_, PyAny>]) -> PyResult<bool> {
    for obj in inputs {
        let arr = as_ndarray(py, obj)?;
        let ndim: usize = arr.getattr("ndim")?.extract()?;
        if ndim != 0 {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Collapse a 0-d ndarray result into a numpy scalar via `ndarray[()]`
/// (numpy's own scalar-extraction), matching the `$OUT_SCALAR` contract.
/// If `result` is not 0-d it is returned unchanged.
pub fn scalarize<'py>(result: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ndim: usize = result.getattr("ndim")?.extract()?;
    if ndim == 0 {
        let empty = pyo3::types::PyTuple::empty(result.py());
        return result.get_item(empty);
    }
    Ok(result)
}

/// Sort `roots` then build a `complex128` ndarray, down-converting it to a
/// REAL (`float64`) array when every imaginary part is exactly zero —
/// mirroring numpy's `polyroots` sort + real-cast.
///
/// `numpy.polynomial.polynomial.polyroots` sorts the companion-matrix
/// eigenvalues (`numpy/polynomial/polynomial.py:1603` `r.sort()`) and then
/// routes them through `numpy.linalg._linalg._to_real_if_imag_zero`
/// (`polynomial.py:1606-1607`). `r.sort()` on a complex array is
/// lexicographic by (real, then imaginary) part — so `x^2 - 5x + 6` yields
/// `[2., 3.]`, not the eigen-/quadratic-solver emit order. That post-sort is
/// a polyroots-level (Python-API) step, so the binding — which IS the analog
/// of `polyroots` for the class `roots()` method — applies it here at the
/// boundary. `_to_real_if_imag_zero` (`numpy/linalg/_linalg.py:190`) then
/// returns `w.real` when the input's expected result type is non-complex and
/// `all(w.imag == 0.0)` — so the roots of a real polynomial whose roots are
/// all real come back as a `float64` array, only staying `complex128` when
/// some root has a non-zero imaginary part. The coefficient input here is
/// always real (`f64`), so `t.dtype` is non-complex and the real-cast hinges
/// solely on `all(w.imag == 0.0)`. `total_cmp` gives a panic-free total order
/// on the `f64` real/imag parts (R-CODE-2). The ferray binding previously
/// hard-coded an unsorted `complex128`; this restores numpy's ordering +
/// dtype contract across the boundary (R-DEV-3).
pub fn complex_roots_to_pyarray<'py>(
    py: Python<'py>,
    mut roots: Vec<num_complex::Complex<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    roots.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    let all_real = roots.iter().all(|c| c.im == 0.0);
    if all_real {
        let reals: Vec<f64> = roots.iter().map(|c| c.re).collect();
        let arr = numpy::PyArray1::<f64>::from_vec(py, reals);
        return Ok(arr.into_any());
    }
    let arr = numpy::PyArray1::<num_complex::Complex<f64>>::from_vec(py, roots);
    Ok(arr.into_any())
}
