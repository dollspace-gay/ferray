//! Bindings for the `numpy` array-manipulation surface.
//!
//! These ops are pure data moves / views — reshape, transpose,
//! concatenate, stack, where, flip, roll, repeat, tile and siblings — so
//! they carry **no element arithmetic** and are generic over
//! `T: Element` in ferray-core. They therefore dispatch through
//! [`match_dtype_all_complex!`], which covers all 11 real dtypes **plus**
//! `complex64`/`complex128` and marshals each element type uniformly via
//! the [`DynMarshal`](crate::conv::DynMarshal) trait: real dtypes route
//! through `AsFerray`/`IntoNumPy`, complex dtypes through the
//! `complex_*` helpers in [`crate::fft`] (numpy's complex types
//! deliberately do not implement `NpElement`). NumPy preserves complex
//! dtype + values through every one of these ops, so ferray does too
//! (#933).
//!
//! `pad` (#938) now also dispatches through [`match_dtype_all_complex!`]: its
//! `constant_values` `f64` maps to `(v + 0j)` for a complex array (matching
//! numpy's promotion of a real `constant_values` against complex data), so the
//! default `constant_values=0` fill becomes `0+0j` and the data move preserves
//! the imaginary part (`np.pad(complex, (1,1))`, live). Consumer: top-level
//! `ferray.pad` in `lib.rs` `_ferray`. Pinned green:
//! `tests/test_divergence_complex_converge_audit.py::test_D_pad`;
//! `tests/test_expansion_complex_dclass.py` (pad cases).
//!
//! The 1-D-only `r_` / `c_` / `trim_zeros` helpers stay on the real-only
//! [`match_dtype_all!`] (they marshal through the `Array1` `NpElement` path).
//! Supporting a new *real* dtype is still one new arm in the macro and zero
//! changes here.
//!
//! ## REQ status
//!
//! Every numpy array-manipulation callable this module registers is SHIPPED:
//! each `#[pyfunction]` dispatches a dtype-generic data move through the
//! `ferray_core::manipulation` (`fm` / `fme`) kernel for that op. (Evidence =
//! the registered `#[pyfunction]` + the `fm::*` / `fme::*` library fn it
//! delegates to; pytest GREEN.)
//!
//! SHIPPED — reshape / ravel / view family:
//!   - `reshape`, `ravel`, `flatten`, `squeeze`, `expand_dims`,
//!     `broadcast_to` → `fm`/`fme` reshape & view kernels.
//!   - `transpose`, `swapaxes`, `moveaxis`, `rollaxis` → axis-permutation
//!     kernels.
//!   - `flip`, `fliplr`, `flipud`, `rot90`, `roll` → reversal/roll kernels.
//!   - `atleast_1d` / `atleast_2d` / `atleast_3d` → dimension-lift kernels.
//!
//! SHIPPED — triangular / diagonal family:
//!   - `tril`, `triu`, `diag`, `diagflat` → `fm`/`fme` triangular & diagonal
//!     kernels.
//!
//! SHIPPED — join / split family:
//!   - `concatenate`, `stack`, `vstack`, `hstack`, `dstack`, `column_stack`,
//!     `row_stack`, `block` → join kernels.
//!   - `split`, `array_split`, and the `bind_axis_split!`-generated `vsplit` /
//!     `hsplit` / `dsplit` → `fm::array_split` / `fm::vsplit` / `fm::hsplit` /
//!     `fm::dsplit`.
//!
//! SHIPPED — repeat / pad / edit family:
//!   - `tile`, `repeat`, `pad` (dispatches through `match_dtype_all_complex!`,
//!     #938), `resize`, `trim_zeros` → tiling/pad/resize kernels.
//!   - `delete`, `insert`, `append` → element-edit kernels.
//!
//! SHIPPED — 1-D index-expression helpers:
//!   - `r_`, `c_` → `Array1`-path concatenation helpers (real-only
//!     `match_dtype_all!`).
//!
//! Boundary dispatch is dtype-uniform: real dtypes route via
//! `AsFerray`/`IntoNumPy`, complex via `crate::fft::complex_*`, with float16 /
//! datetime / flexible (string) inputs delegated to their specialized paths
//! (`f16_delegate_*` / `datetime::delegate_manip*` / `string_delegate*`).
//!
//! NOT-STARTED: none — the full registered manipulation surface is shipped.

use ferray_core::array::aliases::ArrayD;
use ferray_core::dimension::IxDyn;
use ferray_core::manipulation as fm;
use ferray_core::manipulation::extended as fme;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::conv::{
    DynMarshal, as_ndarray, dtype_name, extract_axis_tuple, ferr_to_pyerr, normalize_axis,
    normalize_axis_tuple,
};
use crate::{match_dtype_all, match_dtype_all_complex};

/// Read an array-like's `ndim` (after `as_ndarray` coercion).
fn obj_ndim(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    arr.getattr("ndim")?.extract()
}

/// True if the first element of a Python list of array-likes is a
/// datetime64 / timedelta64 array, so the list-consuming ops
/// (`concatenate`/`stack`/`vstack`/`hstack`/`column_stack`/`dstack`) can route
/// to the numpy-delegation datetime path (#948). The list-stack ops require a
/// uniform element kind, so probing the first input determines the whole op.
fn list_first_is_time<'py>(py: Python<'py>, list: &Bound<'py, PyList>) -> PyResult<bool> {
    if list.is_empty() {
        return Ok(false);
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    crate::datetime::is_time_array(&first)
}

// ---------------------------------------------------------------------------
// Non-real flexible-dtype (`<U`/`<S` string, `V` structured/void, `O` object)
// detect-and-delegate seam (REQ-2 #960 string; #964 structured/object)
// ---------------------------------------------------------------------------
//
// Fixed-width string arrays (numpy `<U` unicode, kind `'U'`; `<S`/`|S` bytes,
// kind `'S'`), structured/void arrays (kind `'V'`, e.g.
// `dtype=[('a','i4'),('b','f4')]`), and object arrays (kind `'O'`,
// `dtype=object`) are all *non-real flexible* dtypes with no ferray `Element`
// and no `DynArray` variant (`.design/ferray-core-string.md`, `dynarray.rs`
// rejects `FixedUnicode`/`FixedAscii`/`RawBytes`, #741/#964), so they never
// enter the Rust library — they ride the boundary as numpy ndarrays. Every
// data-move / select / order op detects the `'U'`/`'S'`/`'V'`/`'O'` kind
// ([`is_flexible_array`]) ahead of its real-only `match_dtype_all*!` / ordering
// macro and delegates to numpy's own op, returning the numpy result **directly**
// (no transport). This is the string/datetime analogue of the `is_time_array` +
// `delegate_manip` seam (#948) and the float16 `is_float16_dtype` +
// `f16_delegate` seam (#955), minus the int64-view round-trip: numpy owns the
// dtype passthrough, the structured field order, the concatenate/stack
// width-promotion (`np.concatenate([['ab'],['cde']])` -> `<U3`), the object
// passthrough, and the lexicographic / object-by-comparison sort
// (`np.sort(['c','a','b']) == ['a','b','c']`).

/// `true` if a numpy ndarray is a fixed-width string array — unicode (`<U`,
/// dtype kind `'U'`) or bytes (`<S`/`|S`, kind `'S'`).
///
/// Keys off `dtype.kind`, NOT `dtype.name`: `np.dtype('U2').name` is `'str64'`,
/// `np.dtype('S4').name` is `'bytes32'` (the `.name` encodes the bit width and is
/// unstable across widths), while `.kind` is stably `'U'`/`'S'` (verified live,
/// numpy 2.4.4). Mirrors the `is_string_array` detection convention shipped for
/// construction in `creation.rs` (#959) and the datetime `time_kind` seam.
pub(crate) fn is_string_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(kind == "U" || kind == "S")
}

/// `true` if a numpy ndarray is a *non-real flexible* dtype that has no ferray
/// `Element` / `DynArray` variant and therefore rides the boundary as a numpy
/// ndarray: fixed-width string (`<U` unicode, kind `'U'`; `<S`/`|S` bytes, kind
/// `'S'`), **structured/void** (`kind == 'V'`, e.g.
/// `dtype=[('a','i4'),('b','f4')]`), or **object** (`kind == 'O'`,
/// `dtype=object`). The data-move / select / order ops in this module gate on
/// this superset to delegate the op to numpy (which owns the dtype passthrough,
/// field-order semantics, and object-by-comparison ordering), returning numpy's
/// result directly with no transport (R-CODE-4, #964 — the structured/object
/// extension of the string `is_string_array` seam, #960).
///
/// Keys off the stable `dtype.kind`, NOT `dtype.name` (`np.dtype('U2').name` is
/// `'str64'`, a structured dtype's `.name` is `'void64'` — both width-encoded and
/// unstable; `.kind` is stably one of `'U'/'S'/'V'/'O'`, verified live numpy
/// 2.4.5). Strict superset of [`is_string_array`].
pub(crate) fn is_flexible_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(kind == "U" || kind == "S" || kind == "V" || kind == "O")
}

/// `true` if the first element of a list of array-likes is a non-real flexible
/// (string / structured-void / object) array, so the list-consuming ops
/// (`concatenate`/`stack`/`vstack`/`hstack`/`column_stack`/`dstack`/`block`)
/// route to the numpy-delegation path. The list-stack ops require a uniform
/// element kind, so the first input determines the whole op (mirrors
/// [`list_first_is_time`], covering string/structured-void/object per #964).
pub(crate) fn list_first_is_flexible<'py>(
    py: Python<'py>,
    list: &Bound<'py, PyList>,
) -> PyResult<bool> {
    if list.is_empty() {
        return Ok(false);
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    is_flexible_array(&first)
}

/// Delegate a data-move / select / order op over a fixed-width string (`<U`/`<S`)
/// array to numpy's own op, returning numpy's string ndarray **directly**.
///
/// Strings never enter the Rust library (no `Element`, no `DynArray` variant —
/// `.design/ferray-core-string.md`), so there is **no transport**: numpy owns the
/// dtype passthrough (and concatenate/stack width-promotion, lexicographic sort),
/// and the result rides the boundary as a numpy ndarray with no lossy cast
/// (R-CODE-4). This is the string analogue of [`crate::datetime::delegate_manip`]
/// minus the int64-view round-trip. `func` is resolved on the `numpy` module;
/// `args`/`kwargs` are the positional/keyword args numpy's op expects.
pub(crate) fn string_delegate<'py>(
    py: Python<'py>,
    func: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    np.getattr(func)?.call(args, kwargs)
}

/// Delegate a splitting op (`split`/`array_split`/`vsplit`/`hsplit`/`dsplit`)
/// over a string array to numpy, returning numpy's `list` of string-array views
/// **directly** as a Python list (no transport). Mirrors
/// [`crate::datetime::delegate_manip_list`] minus the int64-view round-trip.
pub(crate) fn string_delegate_list<'py>(
    py: Python<'py>,
    func: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let result = np.getattr(func)?.call(args, kwargs)?;
    let parts: Vec<Bound<'py, PyAny>> = result.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, parts)?.into_any())
}

// ---------------------------------------------------------------------------
// Shape transforms
// ---------------------------------------------------------------------------

/// Resolve a numpy-style shape spec that may contain a single `-1`
/// (inferred dimension) against the total element count `size`.
///
/// Mirrors numpy/_core/fromnumeric.py:208 `reshape(a, shape, ...)` — at
/// most one `-1` is allowed and is inferred so the product matches
/// `size`. A `-1` arrives here as a signed int because the binding now
/// extracts `Vec<isize>` instead of `Vec<usize>` (so a bare `-1` no
/// longer raises `OverflowError`).
fn resolve_shape(raw: &[isize], size: usize) -> PyResult<Vec<usize>> {
    let mut neg_pos: Option<usize> = None;
    let mut known: usize = 1;
    for (i, &d) in raw.iter().enumerate() {
        if d == -1 {
            if neg_pos.is_some() {
                return Err(PyValueError::new_err(
                    "can only specify one unknown dimension",
                ));
            }
            neg_pos = Some(i);
        } else if d < 0 {
            return Err(PyValueError::new_err(format!(
                "negative dimensions not allowed (got {d})"
            )));
        } else {
            known = known.saturating_mul(d as usize);
        }
    }
    let mut out: Vec<usize> = raw.iter().map(|&d| d.max(0) as usize).collect();
    if let Some(i) = neg_pos {
        if known == 0 || size % known != 0 {
            return Err(PyValueError::new_err(format!(
                "cannot reshape array of size {size} into shape {raw:?}"
            )));
        }
        out[i] = size / known;
    }
    Ok(out)
}

/// `numpy.reshape(a, shape, order='C')` — return an array with a new
/// shape. A single `-1` in `shape` is inferred; `order='F'` reshapes in
/// column-major (Fortran) order.
#[pyfunction]
#[pyo3(signature = (a, shape, order = "C"))]
pub fn reshape<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    order: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // datetime64/timedelta64 (#948): a pure data-move, so numpy preserves the
    // dtype+unit (`np.reshape(M8[D],(2,2)).dtype == datetime64[D]`, live). Delegate
    // to numpy and marshal the result back through the int64-view transport (the
    // real-only `match_dtype_all_complex!` below would raise TypeError).
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", order)?;
        return crate::datetime::delegate_manip(py, "reshape", (&arr, shape), Some(&kwargs));
    }
    // string `<U`/`<S` (REQ-2, #960): a pure data-move preserves the string dtype
    // (`np.reshape(<U,(2,2)).dtype == <U`, live); delegate to numpy and return its
    // string ndarray directly (no transport). Same detect-and-delegate seam as the
    // datetime/f16 branches, applied to every data-move op in this module.
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", order)?;
        return string_delegate(py, "reshape", (&arr, shape), Some(&kwargs));
    }
    // float16 (REQ-5, #955): a pure data-move preserves the dtype
    // (`np.reshape(f16,(2,2)).dtype == float16`, live); delegate to numpy as the
    // real-only `match_dtype_all_complex!` below has no float16 arm. The SAME
    // detect-and-delegate seam is applied to every data-move op in this module.
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", order)?;
        return crate::conv::f16_delegate(py, "reshape", (&arr, shape), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let size: usize = arr.getattr("size")?.extract()?;
    let raw: Vec<isize> = if let Ok(n) = shape.extract::<isize>() {
        vec![n]
    } else {
        shape.extract()?
    };
    let new_shape = resolve_shape(&raw, size)?;
    let fortran = match order {
        "C" | "A" | "K" => false,
        "F" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "order must be one of 'C', 'F', 'A', 'K' (got {other:?})"
            )));
        }
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = if fortran {
            // Fortran-order reshape via the standard identity
            //   reshape(a, s, 'F') == transpose(reshape(transpose(a), rev(s), 'C'))
            // The C-contiguous ravel of `transpose(a)` equals the F-order
            // ravel of `a`; reshaping into the reversed target and
            // transposing back recovers numpy's column-major fill order.
            let at: ArrayD<T> = fm::transpose(&fa, None).map_err(ferr_to_pyerr)?;
            let mut rev: Vec<usize> = new_shape.clone();
            rev.reverse();
            let rt: ArrayD<T> = fm::reshape(&at, &rev).map_err(ferr_to_pyerr)?;
            fm::transpose(&rt, None).map_err(ferr_to_pyerr)?
        } else {
            fm::reshape(&fa, &new_shape).map_err(ferr_to_pyerr)?
        };
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.ravel(a, order='C')` — flatten to 1-D. The native kernel flattens in
/// C (row-major) order; a non-default `order` (`'F'`/`'A'`/`'K'`) delegates to
/// numpy, which owns the column-major / memory-layout traversal.
#[pyfunction]
#[pyo3(signature = (a, order = "C"))]
pub fn ravel<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    order: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if order != "C" {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", order)?;
        return np.call_method("ravel", (&arr,), Some(&kwargs));
    }
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "ravel", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "ravel", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "ravel", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r = fm::ravel(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r.into_dyn())?
    }))
}

/// `numpy.ndarray.flatten` as a free function.
#[pyfunction]
pub fn flatten<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        // `flatten` is an ndarray method, not a numpy free function — call it on
        // the coerced datetime64 array, then marshal back (#948).
        let flat = arr.call_method0("flatten")?;
        return crate::datetime::datetime_roundtrip(py, &flat);
    }
    if is_flexible_array(&arr)? {
        // `flatten` is an ndarray method; call it on the string array and return
        // numpy's string result directly (no transport, REQ-2, #960).
        return arr.call_method0("flatten");
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r = fm::flatten(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r.into_dyn())?
    }))
}

/// `numpy.squeeze(a, axis=None)` — remove length-1 axes. `axis` may be
/// a single (possibly negative) int or a tuple of ints
/// (numpy/_core/fromnumeric.py:1597).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn squeeze<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::datetime::delegate_manip(py, "squeeze", (&arr,), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return string_delegate(py, "squeeze", (&arr,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::conv::f16_delegate(py, "squeeze", (&arr,), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    // Normalize axis to a descending-sorted list of axes to squeeze (so
    // removing one does not shift the indices of those not yet removed).
    let axes: Option<Vec<usize>> = match axis {
        None => None,
        Some(obj) => {
            let raw = extract_axis_tuple(obj)?;
            let mut norm = normalize_axis_tuple(py, &raw, ndim, false)?;
            norm.sort_unstable();
            norm.reverse();
            Some(norm)
        }
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let mut cur: ArrayD<T> = T::extract_dyn(&arr)?;
        match &axes {
            None => cur = fm::squeeze(&cur, None).map_err(ferr_to_pyerr)?,
            Some(list) => {
                for &ax in list {
                    cur = fm::squeeze(&cur, Some(ax)).map_err(ferr_to_pyerr)?;
                }
            }
        }
        T::emit_dyn(py, cur)?
    }))
}

/// `numpy.expand_dims(a, axis)` — insert one or more length-1 axes.
/// `axis` may be a single (possibly negative) int or a tuple of ints
/// (numpy/lib/_shape_base_impl.py:514). Each axis is normalized against
/// `out_ndim = len(axis) + a.ndim`; an out-of-range axis raises
/// `numpy.exceptions.AxisError`.
#[pyfunction]
pub fn expand_dims<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::datetime::delegate_manip(py, "expand_dims", (&arr,), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return string_delegate(py, "expand_dims", (&arr,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::conv::f16_delegate(py, "expand_dims", (&arr,), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let raw = extract_axis_tuple(axis)?;
    let out_ndim = raw.len() + ndim;
    let mut axes = normalize_axis_tuple(py, &raw, out_ndim, false)?;
    // Insert in ascending order so earlier insertions don't shift the
    // positions of later (already-normalized) target axes.
    axes.sort_unstable();
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let mut cur: ArrayD<T> = T::extract_dyn(&arr)?;
        for &ax in &axes {
            cur = fm::expand_dims(&cur, ax).map_err(ferr_to_pyerr)?;
        }
        T::emit_dyn(py, cur)?
    }))
}

/// `numpy.broadcast_to(a, shape)` — broadcast to a target shape.
#[pyfunction]
pub fn broadcast_to<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // NOTE: the top-level `numpy.broadcast_to` is served by
    // `stride_tricks::broadcast_to` (lib.rs overrides this binding), so the
    // datetime passthrough for `broadcast_to` belongs in that module (filed as a
    // #948 follow-up blocker, outside this manifest). This manipulation-module
    // `broadcast_to` keeps its real/complex behavior unchanged.
    let dt = dtype_name(&arr)?;
    let target: Vec<usize> = if let Ok(n) = shape.extract::<usize>() {
        vec![n]
    } else {
        shape.extract()?
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::broadcast_to(&fa, &target).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// Axis transforms
// ---------------------------------------------------------------------------

/// `numpy.transpose(a, axes=None)` — permute axes.
#[pyfunction]
#[pyo3(signature = (a, axes = None))]
pub fn transpose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axes: Option<Vec<isize>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        // numpy.transpose accepts axes=None or a sequence of ints.
        kwargs.set_item("axes", axes.clone())?;
        return crate::datetime::delegate_manip(py, "transpose", (&arr,), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axes", axes.clone())?;
        return string_delegate(py, "transpose", (&arr,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        // numpy.transpose accepts axes=None or a sequence of ints.
        kwargs.set_item("axes", axes.clone())?;
        return crate::conv::f16_delegate(py, "transpose", (&arr,), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    // numpy/_core/fromnumeric.py:605 transpose — explicit axes may be
    // negative and are normalized against ndim.
    let norm_axes: Option<Vec<usize>> = match axes {
        Some(ax) => Some(normalize_axis_tuple(py, &ax, ndim, false)?),
        None => None,
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::transpose(&fa, norm_axes.as_deref()).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.swapaxes(a, axis1, axis2)`.
#[pyfunction]
pub fn swapaxes<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis1: isize,
    axis2: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "swapaxes", (&arr, axis1, axis2), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "swapaxes", (&arr, axis1, axis2), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "swapaxes", (&arr, axis1, axis2), None);
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax1 = normalize_axis(py, axis1, ndim)?;
    let ax2 = normalize_axis(py, axis2, ndim)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::swapaxes(&fa, ax1, ax2).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.moveaxis(a, source, destination)` — move axes to new
/// positions. `source`/`destination` may be a single (possibly
/// negative) int or sequences of ints (numpy/_core/numeric.py:1489).
/// Implemented by building the full destination permutation and
/// delegating to `transpose`, mirroring numpy's algorithm.
#[pyfunction]
pub fn moveaxis<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    destination: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "moveaxis", (&arr, source, destination), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "moveaxis", (&arr, source, destination), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "moveaxis", (&arr, source, destination), None);
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let src = normalize_axis_tuple(py, &extract_axis_tuple(source)?, ndim, false)?;
    let dst = normalize_axis_tuple(py, &extract_axis_tuple(destination)?, ndim, false)?;
    if src.len() != dst.len() {
        return Err(PyValueError::new_err(
            "`source` and `destination` arguments must have the same number of elements",
        ));
    }
    // numpy/_core/numeric.py:1489 — order = [n for n in range(ndim)
    // if n not in source]; then insert each source axis at its dest.
    let mut order: Vec<usize> = (0..ndim).filter(|n| !src.contains(n)).collect();
    let mut pairs: Vec<(usize, usize)> = dst.iter().copied().zip(src.iter().copied()).collect();
    pairs.sort_by_key(|&(d, _)| d);
    for (d, s) in pairs {
        order.insert(d, s);
    }
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::transpose(&fa, Some(&order)).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.rollaxis(a, axis, start=0)` — legacy, prefer moveaxis.
#[pyfunction]
#[pyo3(signature = (a, axis, start = 0))]
pub fn rollaxis<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: isize,
    start: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "rollaxis", (&arr, axis, start), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "rollaxis", (&arr, axis, start), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "rollaxis", (&arr, axis, start), None);
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    // numpy/_core/numeric.py rollaxis: `axis` is normalized into [0, ndim) via
    // normalize_axis_index; `start` accepts negatives (start += ndim) and is
    // valid in [0, ndim] (rollaxis uniquely allows start == ndim — a roll to the
    // very end). A bare usize binding raised OverflowError on either negative.
    let norm_axis = normalize_axis(py, axis, ndim)?;
    let n = ndim as isize;
    let adj_start = if start < 0 { start + n } else { start };
    if adj_start < 0 || adj_start > n {
        return Err(crate::conv::axis_error(py, start, ndim + 1));
    }
    let norm_start = adj_start as usize;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::rollaxis(&fa, norm_axis, norm_start).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// Flips and rotations
// ---------------------------------------------------------------------------

/// `numpy.flip(m, axis=None)` — reverse element order along the given
/// axis or axes. `axis=None` (default) reverses every axis; `axis` may
/// also be a single (possibly negative) int or a tuple of ints
/// (numpy/lib/_function_base_impl.py:284). An out-of-range axis raises
/// `numpy.exceptions.AxisError`.
#[pyfunction]
#[pyo3(signature = (m, axis = None))]
pub fn flip<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::datetime::delegate_manip(py, "flip", (&arr,), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return string_delegate(py, "flip", (&arr,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::conv::f16_delegate(py, "flip", (&arr,), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let axes: Vec<usize> = match axis {
        None => (0..ndim).collect(),
        Some(obj) => normalize_axis_tuple(py, &extract_axis_tuple(obj)?, ndim, false)?,
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let mut cur: ArrayD<T> = T::extract_dyn(&arr)?;
        for &ax in &axes {
            cur = fm::flip(&cur, ax).map_err(ferr_to_pyerr)?;
        }
        T::emit_dyn(py, cur)?
    }))
}

/// `numpy.fliplr(m)` — flip along axis 1.
#[pyfunction]
pub fn fliplr<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "fliplr", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "fliplr", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "fliplr", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::fliplr(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.flipud(m)` — flip along axis 0.
#[pyfunction]
pub fn flipud<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "flipud", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "flipud", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "flipud", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::flipud(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.rot90(m, k=1, axes=(0, 1))` — rotate 90° `k` times in the
/// plane defined by `axes` (numpy/lib/_function_base_impl.py:180). The
/// default `(0, 1)` plane is delegated to ferray-core's fast path; an
/// arbitrary plane replays numpy's flip/transpose algorithm.
#[pyfunction]
#[pyo3(signature = (m, k = 1, axes = (0, 1)))]
pub fn rot90<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    k: i32,
    axes: (isize, isize),
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("k", k)?;
        kwargs.set_item("axes", (axes.0, axes.1))?;
        return crate::datetime::delegate_manip(py, "rot90", (&arr,), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("k", k)?;
        kwargs.set_item("axes", (axes.0, axes.1))?;
        return string_delegate(py, "rot90", (&arr,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("k", k)?;
        kwargs.set_item("axes", (axes.0, axes.1))?;
        return crate::conv::f16_delegate(py, "rot90", (&arr,), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax0 = normalize_axis(py, axes.0, ndim)?;
    let ax1 = normalize_axis(py, axes.1, ndim)?;
    if ax0 == ax1 {
        return Err(PyValueError::new_err("axes must be different"));
    }
    let use_core_fast_path = ax0 == 0 && ax1 == 1;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = if use_core_fast_path {
            fm::rot90(&fa, k).map_err(ferr_to_pyerr)?
        } else {
            // numpy/lib/_function_base_impl.py:180 — generic plane.
            let kk = k.rem_euclid(4);
            match kk {
                0 => {
                    // No rotation: return a same-shape owned copy.
                    let shp: Vec<usize> = fa.shape().to_vec();
                    fm::reshape(&fa, &shp).map_err(ferr_to_pyerr)?
                }
                2 => {
                    let f0 = fm::flip(&fa, ax0).map_err(ferr_to_pyerr)?;
                    fm::flip(&f0, ax1).map_err(ferr_to_pyerr)?
                }
                _ => {
                    // axes_list with ax0/ax1 swapped, used as a transpose perm.
                    let mut axes_list: Vec<usize> = (0..ndim).collect();
                    axes_list.swap(ax0, ax1);
                    if kk == 1 {
                        let f = fm::flip(&fa, ax1).map_err(ferr_to_pyerr)?;
                        fm::transpose(&f, Some(&axes_list)).map_err(ferr_to_pyerr)?
                    } else {
                        // kk == 3
                        let t = fm::transpose(&fa, Some(&axes_list)).map_err(ferr_to_pyerr)?;
                        fm::flip(&t, ax1).map_err(ferr_to_pyerr)?
                    }
                }
            }
        };
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.roll(a, shift, axis=None)` — `shift` and `axis` may each be an int
/// or a tuple; when both are tuples they zip (roll `shift[i]` along `axis[i]`),
/// an int `shift` with a tuple `axis` applies the same shift to each axis, and
/// `axis=None` rolls the flattened array (numpy/_core/numeric.py:1226,
/// `np.roll(x2, (1, 1), axis=(1, 0))` → multi-axis roll).
///
/// `shift`/`axis` arrive as `PyAny`: the scalar shift + (None | int axis) case
/// takes the in-crate `fm::roll` fast path; any tuple shift OR tuple axis is
/// delegated to `numpy.roll`, which owns the shift/axis zip + broadcast
/// semantics. The real array is forwarded unchanged (no lossy round-trip,
/// R-CODE-4).
#[pyfunction]
#[pyo3(signature = (a, shift, axis = None))]
pub fn roll<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    shift: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Multi-axis roll (tuple shift and/or tuple axis): numpy owns the
    // shift/axis zip + broadcast semantics. Delegate the whole op (works for
    // every dtype, including the datetime/string/f16/complex arms below).
    let shift_is_scalar = shift.extract::<isize>().is_ok();
    let axis_is_scalar = match axis {
        None => true,
        Some(ax) => ax.extract::<isize>().is_ok(),
    };
    if !shift_is_scalar || !axis_is_scalar {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        match axis {
            Some(ax) => kwargs.set_item("axis", ax)?,
            None => kwargs.set_item("axis", py.None())?,
        }
        return np.getattr("roll")?.call((&arr, shift), Some(&kwargs));
    }
    // Scalar shift (+ None | int axis) fast path.
    let shift: isize = shift.extract()?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::datetime::delegate_manip(py, "roll", (&arr, shift), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return string_delegate(py, "roll", (&arr, shift), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::conv::f16_delegate(py, "roll", (&arr, shift), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let norm_axis: Option<usize> = match axis {
        Some(ax) => Some(normalize_axis(py, ax.extract::<isize>()?, ndim)?),
        None => None,
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fm::roll(&fa, shift, norm_axis).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// atleast_*d
// ---------------------------------------------------------------------------

/// `numpy.atleast_1d(a)`.
#[pyfunction]
pub fn atleast_1d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "atleast_1d", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "atleast_1d", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "atleast_1d", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::atleast_1d(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.atleast_2d(a)`.
#[pyfunction]
pub fn atleast_2d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "atleast_2d", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "atleast_2d", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "atleast_2d", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::atleast_2d(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.atleast_3d(a)`.
#[pyfunction]
pub fn atleast_3d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "atleast_3d", (&arr,), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "atleast_3d", (&arr,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "atleast_3d", (&arr,), None);
    }
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::atleast_3d(&fa).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// Triangular and diagonal extraction
// ---------------------------------------------------------------------------

/// `numpy.tril(m, k=0)` — lower triangle (zero-out above-k diagonal).
#[pyfunction]
#[pyo3(signature = (m, k = 0))]
pub fn tril<'py>(py: Python<'py>, m: &Bound<'py, PyAny>, k: isize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = ferray_core::creation::tril(&fa, k).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.triu(m, k=0)` — upper triangle (zero-out below-k diagonal).
#[pyfunction]
#[pyo3(signature = (m, k = 0))]
pub fn triu<'py>(py: Python<'py>, m: &Bound<'py, PyAny>, k: isize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = ferray_core::creation::triu(&fa, k).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.diag(v, k=0)` — extract a diagonal or build a 2-D array
/// from a 1-D diagonal.
#[pyfunction]
#[pyo3(signature = (v, k = 0))]
pub fn diag<'py>(py: Python<'py>, v: &Bound<'py, PyAny>, k: isize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, v)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = ferray_core::creation::diag(&fa, k).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.diagflat(v, k=0)` — flatten then build diagonal matrix.
#[pyfunction]
#[pyo3(signature = (v, k = 0))]
pub fn diagflat<'py>(
    py: Python<'py>,
    v: &Bound<'py, PyAny>,
    k: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, v)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = ferray_core::creation::diagflat(&fa, k).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// Concatenation / stacking
// ---------------------------------------------------------------------------

/// Coerce a Python sequence-of-array-likes into a `Vec<ArrayD<T>>`.
///
/// Every input goes through `numpy.asarray(item, dt_name)` so the caller
/// can pass mixed-dtype lists; ferray's stack/concat APIs require a
/// uniform element type. `dt_name` must be the **promoted** common dtype
/// of the whole list — see [`common_dtype`]. Passing the first array's
/// dtype instead would TRUNCATE later inputs
/// (numpy/_core/multiarray.py:198 promotes via `result_type`).
fn collect_typed<'py, T: DynMarshal>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    dt_name: &str,
) -> PyResult<Vec<ArrayD<T>>> {
    let np = py.import("numpy")?;
    let list = arrays.cast::<PyList>()?;
    let mut out: Vec<ArrayD<T>> = Vec::with_capacity(list.len());
    for item in list.iter() {
        let coerced = np.call_method1("asarray", (&item, dt_name))?;
        out.push(T::extract_dyn(&coerced)?);
    }
    Ok(out)
}

/// Compute the common (promoted) NumPy dtype name across every element
/// of an array-like list, mirroring numpy's `result_type` rule used by
/// `concatenate`/`stack` (numpy/_core/multiarray.py:198). Taking the
/// first array's dtype instead silently truncates wider later inputs.
fn common_dtype<'py>(py: Python<'py>, list: &Bound<'py, PyList>) -> PyResult<String> {
    let np = py.import("numpy")?;
    // Coerce every entry to an ndarray, then fold result_type across them.
    // `result_type` takes the arrays as *positional* args, so build the
    // args tuple from the items (which CPython unpacks positionally).
    let items: Vec<Bound<'py, PyAny>> = list
        .iter()
        .map(|it| as_ndarray(py, &it))
        .collect::<PyResult<Vec<_>>>()?;
    let args = pyo3::types::PyTuple::new(py, &items)?;
    let dt = np.getattr("result_type")?.call1(args)?;
    dt.getattr("name")?.extract()
}

/// Compute the common (promoted) NumPy dtype name across a tuple of
/// array-likes via `result_type` (NEP-50), the tuple analogue of
/// [`common_dtype`] used by the `*args` ops `r_` / `c_`.
fn common_dtype_tuple<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<String> {
    let np = py.import("numpy")?;
    let items: Vec<Bound<'py, PyAny>> = arrays
        .iter()
        .map(|it| as_ndarray(py, &it))
        .collect::<PyResult<Vec<_>>>()?;
    let args = pyo3::types::PyTuple::new(py, &items)?;
    let dt = np.getattr("result_type")?.call1(args)?;
    dt.getattr("name")?.extract()
}

/// `true` if a promoted dtype name is a complex type (`complex64`/
/// `complex128`), which the real-only `match_dtype_all!` macro cannot
/// represent (so the op must delegate to numpy to preserve the imaginary
/// part — R-CODE-4).
fn is_complex_dtype(name: &str) -> bool {
    name.starts_with("complex")
}

/// `numpy.concatenate(arrays, axis=0)` — join along an existing axis.
///
/// Inputs are promoted to a common dtype (numpy/_core/multiarray.py:198
/// `result_type`, not the first array's dtype). `axis` may be negative,
/// and `axis=None` flattens every input before concatenating. An
/// out-of-range axis raises `numpy.exceptions.AxisError`.
#[pyfunction]
#[pyo3(signature = (arrays, axis = Some(0)))]
pub fn concatenate<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err(
            "concatenate: need at least one array",
        ));
    }
    // datetime64/timedelta64 (#948): pure data-move; numpy preserves the dtype and
    // promotes cross-unit inputs to the finer unit (matching `result_type`).
    // Delegate to numpy and marshal back (the real-only macro raises TypeError).
    if list_first_is_time(py, list)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::datetime::delegate_manip(py, "concatenate", (arrays,), Some(&kwargs));
    }
    // string `<U`/`<S` (REQ-2, #960): numpy owns the dtype passthrough AND the
    // width-promotion to the max input width (`np.concatenate([['ab'],['cde']])`
    // -> `<U3`, live); delegate to numpy and return its string ndarray directly.
    if list_first_is_flexible(py, list)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return string_delegate(py, "concatenate", (arrays,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::conv::f16_delegate(py, "concatenate", (arrays,), Some(&kwargs));
    }
    let dt = common_dtype(py, list)?;
    // For axis=None numpy ravels each input and concatenates along axis 0.
    let (flatten, norm_axis): (bool, usize) = match axis {
        None => (true, 0),
        Some(ax) => {
            let ndim = obj_ndim(&as_ndarray(py, &list.get_item(0)?)?)?;
            (false, normalize_axis(py, ax, ndim)?)
        }
    };
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let mut typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        if flatten {
            typed = typed
                .iter()
                .map(|a| {
                    let n = a.size();
                    fm::reshape(a, &[n]).map_err(ferr_to_pyerr)
                })
                .collect::<PyResult<Vec<_>>>()?;
        }
        let r: ArrayD<T> = fm::concatenate(&typed, norm_axis).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.stack(arrays, axis=0)` — stack along a new axis.
///
/// Inputs are promoted to a common dtype (numpy/_core/shape_base.py:379
/// promotes via `result_type`). `axis` is normalized against the result
/// ndim (`input_ndim + 1`) and may be negative; an out-of-range axis
/// raises `numpy.exceptions.AxisError`.
#[pyfunction]
#[pyo3(signature = (arrays, axis = 0))]
pub fn stack<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("stack: need at least one array"));
    }
    if list_first_is_time(py, list)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::datetime::delegate_manip(py, "stack", (arrays,), Some(&kwargs));
    }
    if list_first_is_flexible(py, list)? {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return string_delegate(py, "stack", (arrays,), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return crate::conv::f16_delegate(py, "stack", (arrays,), Some(&kwargs));
    }
    let dt = common_dtype(py, list)?;
    let result_ndim = obj_ndim(&as_ndarray(py, &list.get_item(0)?)?)? + 1;
    let norm_axis = normalize_axis(py, axis, result_ndim)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::stack(&typed, norm_axis).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.vstack(arrays)` — stack along axis 0.
#[pyfunction]
pub fn vstack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("vstack: need at least one array"));
    }
    if list_first_is_time(py, list)? {
        return crate::datetime::delegate_manip(py, "vstack", (arrays,), None);
    }
    if list_first_is_flexible(py, list)? {
        return string_delegate(py, "vstack", (arrays,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        return crate::conv::f16_delegate(py, "vstack", (arrays,), None);
    }
    // numpy/_core/shape_base.py vstack promotes all inputs to result_type
    // (NEP-50), NOT the first array's dtype. Use the shared common_dtype helper
    // (matching concatenate/stack) so wider later inputs are not truncated.
    let dt = common_dtype(py, list)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::vstack(&typed).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.hstack(arrays)` — stack along axis 1 (or 0 for 1-D).
#[pyfunction]
pub fn hstack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("hstack: need at least one array"));
    }
    if list_first_is_time(py, list)? {
        return crate::datetime::delegate_manip(py, "hstack", (arrays,), None);
    }
    if list_first_is_flexible(py, list)? {
        return string_delegate(py, "hstack", (arrays,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        return crate::conv::f16_delegate(py, "hstack", (arrays,), None);
    }
    // numpy/_core/shape_base.py hstack promotes all inputs to result_type
    // (NEP-50), NOT the first array's dtype (matching concatenate/stack).
    let dt = common_dtype(py, list)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::hstack(&typed).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

// ---------------------------------------------------------------------------
// Padding / tiling / repeating / element insertion
// ---------------------------------------------------------------------------

/// `numpy.pad(array, pad_width, mode='constant', **kwargs)`.
///
/// Delegates every mode to `numpy.pad` (#973): numpy owns the correct
/// `reflect` algorithm (mirror WITHOUT the edge element — the prior
/// ferray path mirrored WITH the edge, i.e. did `symmetric`, giving
/// `[2,1,1,2,…]` where numpy gives `[3,2,1,2,…]`), plus `symmetric`,
/// `edge`, `wrap`, `constant`, and the previously-missing
/// `linear_ramp`/`maximum`/`mean`/`minimum`/`median`/`empty`. All
/// mode-specific kwargs (`constant_values`, `reflect_type`,
/// `stat_length`, `end_values`) flow through `**kwargs`. The numpy
/// result is re-marshalled through the `match_dtype_all_complex!`
/// extract/emit seam (#933) so the dtype/shape contract is preserved and
/// a fresh ferray-owned buffer is returned (R-CODE-4), not a raw numpy
/// view. datetime64/timedelta64 (#948), string `<U`/`<S` (#960) and
/// float16 (#955) keep their dedicated marshalling arms.
#[pyfunction]
#[pyo3(signature = (array, pad_width, mode = "constant", **kwargs))]
pub fn pad<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    pad_width: &Bound<'py, PyAny>,
    mode: &str,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, array)?;

    // Build the kwargs dict forwarded to numpy.pad: `mode` plus every
    // mode-specific kwarg the caller supplied (`constant_values`,
    // `reflect_type`, `stat_length`, `end_values`).
    let np_kwargs = pyo3::types::PyDict::new(py);
    np_kwargs.set_item("mode", mode)?;
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            np_kwargs.set_item(k, v)?;
        }
    }

    // datetime64/timedelta64 (#948): numpy pads with the epoch tick (0) for the
    // default `constant_values=0` and preserves the dtype+unit
    // (`np.pad(M8[D],(1,1))` -> '1970-01-01' fill, live). Delegate and marshal.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "pad", (&arr, pad_width), Some(&np_kwargs));
    }
    // string `<U`/`<S` (REQ-2, #960): numpy pads with the empty string `''` for
    // the default `constant_values=0` and preserves the dtype+width
    // (`np.pad(<U2,(1,1))` -> `''` fill, live); delegate to numpy.
    if is_flexible_array(&arr)? {
        return string_delegate(py, "pad", (&arr, pad_width), Some(&np_kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "pad", (&arr, pad_width), Some(&np_kwargs));
    }
    let dt = dtype_name(&arr)?;

    // Real/complex path (#973): delegate to numpy.pad so reflect is correct
    // and all modes/kwargs are supported, then re-marshal the result through
    // the DynMarshal seam (#933) to return a fresh ferray-owned buffer with
    // the dtype/shape contract preserved (the complex arm keeps the imaginary
    // part). `np.pad` keeps the input dtype, so the result dtype == `dt`.
    let np = py.import("numpy")?;
    let result = np
        .getattr("pad")?
        .call((&arr, pad_width), Some(&np_kwargs))?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let r: ArrayD<T> = T::extract_dyn(&result)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.tile(A, reps)` — construct an array by repeating A `reps` times.
#[pyfunction]
pub fn tile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    reps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "tile", (&arr, reps), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "tile", (&arr, reps), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "tile", (&arr, reps), None);
    }
    let reps_vec: Vec<usize> = if let Ok(n) = reps.extract::<usize>() {
        vec![n]
    } else {
        reps.extract()?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::tile(&fa, &reps_vec).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.repeat(a, repeats, axis=None)`
/// (numpy/_core/fromnumeric.py:438). `repeats` may be a scalar (every
/// element repeated the same number of times) or an array_like of
/// per-element counts; `axis` may be negative.
#[pyfunction]
#[pyo3(signature = (a, repeats, axis = None))]
pub fn repeat<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    repeats: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::datetime::delegate_manip(py, "repeat", (&arr, repeats), Some(&kwargs));
    }
    if is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return string_delegate(py, "repeat", (&arr, repeats), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::conv::f16_delegate(py, "repeat", (&arr, repeats), Some(&kwargs));
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let norm_axis: Option<usize> = match axis {
        Some(ax) => Some(normalize_axis(py, ax, ndim)?),
        None => None,
    };
    // Scalar vs per-element repeat counts.
    let scalar_reps: Option<usize> = repeats.extract::<usize>().ok();
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = match scalar_reps {
            Some(n) => fme::repeat(&fa, n, norm_axis).map_err(ferr_to_pyerr)?,
            None => {
                let counts: Vec<usize> = repeats.extract()?;
                repeat_per_element::<T>(&fa, &counts, norm_axis)?
            }
        };
        T::emit_dyn(py, r)?
    }))
}

/// Per-element `numpy.repeat`: element/sub-array `i` along the chosen
/// axis is emitted `counts[i]` times (numpy/_core/fromnumeric.py:438).
/// `axis=None` flattens first, matching numpy. `counts` must match the
/// length of the chosen axis (a length-1 `counts` broadcasts).
fn repeat_per_element<T: ferray_core::Element>(
    a: &ArrayD<T>,
    counts: &[usize],
    axis: Option<usize>,
) -> PyResult<ArrayD<T>> {
    match axis {
        None => {
            let flat: Vec<T> = a.iter().cloned().collect();
            let counts = broadcast_counts(counts, flat.len())?;
            let mut data: Vec<T> = Vec::new();
            for (val, &c) in flat.iter().zip(counts.iter()) {
                for _ in 0..c {
                    data.push(val.clone());
                }
            }
            let n = data.len();
            ArrayD::<T>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)
        }
        Some(ax) => {
            let shape = a.shape().to_vec();
            let ndim = shape.len();
            if ax >= ndim {
                return Err(ferr_to_pyerr(ferray_core::FerrayError::axis_out_of_bounds(
                    ax, ndim,
                )));
            }
            let axis_len = shape[ax];
            let counts = broadcast_counts(counts, axis_len)?;
            let new_axis_len: usize = counts.iter().sum();
            let mut new_shape = shape.clone();
            new_shape[ax] = new_axis_len;
            let total: usize = new_shape.iter().product();
            let src: Vec<T> = a.iter().cloned().collect();

            // Map each output index along `ax` back to its source index.
            let mut src_along: Vec<usize> = Vec::with_capacity(new_axis_len);
            for (i, &c) in counts.iter().enumerate() {
                for _ in 0..c {
                    src_along.push(i);
                }
            }

            // C-order strides for source and output.
            let mut src_strides = vec![1usize; ndim];
            let mut out_strides = vec![1usize; ndim];
            for i in (0..ndim.saturating_sub(1)).rev() {
                src_strides[i] = src_strides[i + 1] * shape[i + 1];
                out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
            }

            let mut data: Vec<T> = Vec::with_capacity(total);
            for flat in 0..total {
                let mut rem = flat;
                let mut src_flat = 0usize;
                for i in 0..ndim {
                    let idx = rem / out_strides[i];
                    rem %= out_strides[i];
                    let src_idx = if i == ax { src_along[idx] } else { idx };
                    src_flat += src_idx * src_strides[i];
                }
                data.push(src[src_flat].clone());
            }
            ArrayD::<T>::from_vec(IxDyn::new(&new_shape), data).map_err(ferr_to_pyerr)
        }
    }
}

/// Broadcast a `repeats` count list to `len`, matching numpy's rule that
/// a length-1 `repeats` applies to every element.
fn broadcast_counts(counts: &[usize], len: usize) -> PyResult<Vec<usize>> {
    if counts.len() == len {
        Ok(counts.to_vec())
    } else if counts.len() == 1 {
        Ok(vec![counts[0]; len])
    } else {
        Err(PyValueError::new_err(format!(
            "operands could not be broadcast together with shape ({},) ({},)",
            counts.len(),
            len,
        )))
    }
}

/// `numpy.delete(arr, obj, axis=None)` — remove sub-arrays along `axis`
/// (numpy/lib/_function_base_impl.py:5221). `obj` may be an int, a list
/// of ints (negatives normalized), a `slice`, or a boolean mask the same
/// length as the axis. `axis` may be negative.
///
/// `axis=None` (the default) flattens `arr` first, then deletes — numpy owns
/// that flatten-and-delete path (plus the full `obj` resolution against the flat
/// length), so delegate to `numpy.delete`. The native per-axis kernel runs only
/// when `axis` is given. The prior signature made `axis` REQUIRED, so the
/// common `np.delete(a, obj)` call (axis defaulted to None) raised TypeError.
#[pyfunction]
#[pyo3(signature = (arr, obj, axis = None))]
pub fn delete<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    obj: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, arr)?;
    let axis = match axis {
        Some(ax) => ax,
        None => {
            let np = py.import("numpy")?;
            return np.call_method1("delete", (&arr, obj));
        }
    };
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "delete", (&arr, obj, axis), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "delete", (&arr, obj, axis), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "delete", (&arr, obj, axis), None);
    }
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax = normalize_axis(py, axis, ndim)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let axis_len = shape[ax];
    let indices = resolve_delete_obj(py, obj, axis_len, ax)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::delete(&fa, &indices, ax).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// Resolve the `obj` argument of `numpy.delete` to a list of usize
/// indices along an axis of length `axis_len`. Handles `slice`, boolean
/// mask, and int / list-of-ints (negatives counted from the end).
fn resolve_delete_obj(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    axis_len: usize,
    axis: usize,
) -> PyResult<Vec<usize>> {
    // Python `slice` → slice.indices(axis_len) gives (start, stop, step).
    if let Ok(sl) = obj.cast::<pyo3::types::PySlice>() {
        let indices = sl.indices(axis_len as isize)?;
        let (start, stop, step) = (indices.start, indices.stop, indices.step);
        let mut out = Vec::new();
        if step > 0 {
            let mut i = start;
            while i < stop {
                out.push(i as usize);
                i += step;
            }
        } else if step < 0 {
            let mut i = start;
            while i > stop {
                out.push(i as usize);
                i += step;
            }
        }
        return Ok(out);
    }
    // Boolean mask the same length as the axis → indices where true.
    let arr = as_ndarray(py, obj)?;
    let is_bool = dtype_name(&arr).map(|n| n == "bool").unwrap_or(false);
    if is_bool {
        let mask: Vec<bool> = arr.call_method0("tolist")?.extract()?;
        if mask.len() != axis_len {
            return Err(PyValueError::new_err(format!(
                "boolean array argument obj to delete must be one dimensional and \
                 match the axis length of {axis_len}"
            )));
        }
        return Ok(mask
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i) } else { None })
            .collect());
    }
    // Non-integer, non-boolean obj (e.g. a float scalar/array) is not a valid
    // index. numpy raises a plain `IndexError` from its fancy-index path
    // (`keep[obj,] = False`): "arrays used as indices must be of integer (or
    // boolean) type" (numpy/lib/_function_base_impl.py:5221 delete). Sniff the
    // coerced dtype.kind: anything outside {signed int 'i', unsigned 'u',
    // bool 'b'} is rejected here, before the int extraction below would raise a
    // (non-IndexError) TypeError.
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if !matches!(kind.as_str(), "i" | "u" | "b") {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "arrays used as indices must be of integer (or boolean) type",
        ));
    }
    // Int or list of ints (negatives count from the end).
    let raw: Vec<isize> = if let Ok(n) = obj.extract::<isize>() {
        vec![n]
    } else {
        obj.extract()?
    };
    raw.iter()
        .map(|&i| {
            let norm = if i < 0 { i + axis_len as isize } else { i };
            if norm < 0 || norm >= axis_len as isize {
                // numpy/lib/_function_base_impl.py:5353 raises a *plain*
                // `IndexError` formatted with the ORIGINAL (pre-normalization)
                // index value and the axis number, e.g.
                // "index 20 is out of bounds for axis 0 with size 10" (and
                // "index -20 ..." for a negative oob). This is NOT an AxisError.
                Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {i} is out of bounds for axis {axis} with size {axis_len}"
                )))
            } else {
                Ok(norm as usize)
            }
        })
        .collect()
}

/// `numpy.insert(arr, obj, values, axis=None)` — insert `values` before the
/// indices in `obj` along `axis`.
///
/// numpy owns the full `obj` surface (int, `slice`, or a sequence of indices),
/// the `axis=None` flatten path (the default), value broadcasting, and dtype
/// promotion against the inserted `values` (e.g. inserting a float into an int
/// array promotes to float64). The prior native binding accepted only a SINGLE
/// `int` `obj` and a REQUIRED `int` `axis`, so the common `np.insert(a, i, v)`
/// call (axis defaulted to None) raised TypeError, and slice/sequence `obj`
/// were unsupported. Delegate the whole op to `numpy.insert` for exact parity.
#[pyfunction]
#[pyo3(signature = (arr, obj, values, axis = None))]
pub fn insert<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    obj: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, arr)?;
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(ax) = axis {
        kwargs.set_item("axis", ax)?;
    }
    np.call_method("insert", (&arr_a, obj, values), Some(&kwargs))
}

/// `numpy.append(arr, values, axis=None)`.
#[pyfunction]
#[pyo3(signature = (arr, values, axis = None))]
pub fn append<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, arr)?;
    if crate::datetime::is_time_array(&arr_a)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::datetime::delegate_manip(py, "append", (&arr_a, values), Some(&kwargs));
    }
    if is_flexible_array(&arr_a)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return string_delegate(py, "append", (&arr_a, values), Some(&kwargs));
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr_a)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        return crate::conv::f16_delegate(py, "append", (&arr_a, values), Some(&kwargs));
    }
    let dt = dtype_name(&arr_a)?;
    let arr_v = crate::conv::coerce_dtype(py, values, dt.as_str())?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr_a)?;
        let fv: ArrayD<T> = T::extract_dyn(&arr_v)?;
        let r: ArrayD<T> = fme::append(&fa, &fv, axis).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.resize(a, new_shape)` — like reshape but cycles values when
/// the new shape is larger than the input.
#[pyfunction]
pub fn resize<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    new_shape: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip(py, "resize", (&arr, new_shape), None);
    }
    if is_flexible_array(&arr)? {
        return string_delegate(py, "resize", (&arr, new_shape), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate(py, "resize", (&arr, new_shape), None);
    }
    let target: Vec<usize> = if let Ok(n) = new_shape.extract::<usize>() {
        vec![n]
    } else {
        new_shape.extract()?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fme::resize(&fa, &target).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.trim_zeros(filt, trim='fb')` — trim leading/trailing zeros.
#[pyfunction]
#[pyo3(signature = (filt, trim = "fb"))]
pub fn trim_zeros<'py>(
    py: Python<'py>,
    filt: &Bound<'py, PyAny>,
    trim: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, filt)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: numpy::PyReadonlyArray1<T> = arr.extract()?;
        let fa: ferray_core::array::aliases::Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ferray_core::array::aliases::Array1<T> =
            fme::trim_zeros(&fa, trim).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Splitting (split / array_split / vsplit / hsplit / dsplit)
// ---------------------------------------------------------------------------

/// Convert a `Vec<ArrayD<T>>` into a Python list of `numpy.ndarray`.
/// Routes each element through [`DynMarshal::emit_dyn`] so the helper
/// works for both real and complex dtypes.
fn dyn_arrays_to_pylist<'py, T: DynMarshal>(
    py: Python<'py>,
    arrays: Vec<ArrayD<T>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py_arrays: Vec<Bound<'py, PyAny>> = arrays
        .into_iter()
        .map(|a| T::emit_dyn(py, a))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, py_arrays)?.into_any())
}

/// `numpy.split(ary, indices_or_sections, axis=0)`.
///
/// `indices_or_sections` may be an int (split into N equal sections,
/// errors if axis isn't divisible by N) or a sequence of split points.
#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis = 0))]
pub fn split<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    indices_or_sections: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, ary)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip_list(
            py,
            "split",
            (&arr, indices_or_sections, axis),
            None,
        );
    }
    if is_flexible_array(&arr)? {
        return string_delegate_list(py, "split", (&arr, indices_or_sections, axis), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate_list(
            py,
            "split",
            (&arr, indices_or_sections, axis),
            None,
        );
    }
    let dt = dtype_name(&arr)?;

    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let parts: Vec<ArrayD<T>> = if let Ok(n) = indices_or_sections.extract::<usize>() {
            fm::split(&fa, n, axis).map_err(ferr_to_pyerr)?
        } else {
            let indices: Vec<usize> = indices_or_sections.extract()?;
            fm::array_split(&fa, &indices, axis).map_err(ferr_to_pyerr)?
        };
        dyn_arrays_to_pylist(py, parts)?
    }))
}

/// `numpy.array_split(ary, indices_or_sections, axis=0)` — like split,
/// but allows uneven sections (no error if axis size isn't divisible).
#[pyfunction]
#[pyo3(signature = (ary, indices_or_sections, axis = 0))]
pub fn array_split<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    indices_or_sections: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, ary)?;
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::delegate_manip_list(
            py,
            "array_split",
            (&arr, indices_or_sections, axis),
            None,
        );
    }
    if is_flexible_array(&arr)? {
        return string_delegate_list(py, "array_split", (&arr, indices_or_sections, axis), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        return crate::conv::f16_delegate_list(
            py,
            "array_split",
            (&arr, indices_or_sections, axis),
            None,
        );
    }
    let dt = dtype_name(&arr)?;

    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let parts: Vec<ArrayD<T>> = if let Ok(n) = indices_or_sections.extract::<usize>() {
            fm::array_split_n(&fa, n, axis).map_err(ferr_to_pyerr)?
        } else {
            let indices: Vec<usize> = indices_or_sections.extract()?;
            fm::array_split(&fa, &indices, axis).map_err(ferr_to_pyerr)?
        };
        dyn_arrays_to_pylist(py, parts)?
    }))
}

macro_rules! bind_axis_split {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            ary: &Bound<'py, PyAny>,
            n_sections: usize,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, ary)?;
            if crate::datetime::is_time_array(&arr)? {
                return crate::datetime::delegate_manip_list(
                    py,
                    stringify!($name),
                    (&arr, n_sections),
                    None,
                );
            }
            if is_flexible_array(&arr)? {
                return string_delegate_list(py, stringify!($name), (&arr, n_sections), None);
            }
            if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
                return crate::conv::f16_delegate_list(
                    py,
                    stringify!($name),
                    (&arr, n_sections),
                    None,
                );
            }
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_all_complex!(dt.as_str(), T => {
                let fa: ArrayD<T> = T::extract_dyn(&arr)?;
                let parts: Vec<ArrayD<T>> = $ferr_path(&fa, n_sections).map_err(ferr_to_pyerr)?;
                dyn_arrays_to_pylist(py, parts)?
            }))
        }
    };
}

bind_axis_split!(vsplit, fm::vsplit);
bind_axis_split!(hsplit, fm::hsplit);
bind_axis_split!(dsplit, fm::dsplit);

/// `numpy.column_stack(arrays)` — stack 1-D arrays as columns into 2-D.
#[pyfunction]
pub fn column_stack<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err(
            "column_stack: need at least one array",
        ));
    }
    if list_first_is_time(py, list)? {
        return crate::datetime::delegate_manip(py, "column_stack", (arrays,), None);
    }
    if list_first_is_flexible(py, list)? {
        return string_delegate(py, "column_stack", (arrays,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        return crate::conv::f16_delegate(py, "column_stack", (arrays,), None);
    }
    // numpy/_core/shape_base.py column_stack promotes all inputs to result_type
    // (NEP-50), NOT the first array's dtype (matching concatenate/stack).
    let dt = common_dtype(py, list)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::column_stack(&typed).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}

/// `numpy.row_stack(arrays)` — alias for `vstack`.
#[pyfunction]
pub fn row_stack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    vstack(py, arrays)
}

/// `numpy.block(arrays)` — assemble an array from nested lists of blocks.
///
/// numpy.block's output dimensionality is `max(nesting_depth, max_atom_ndim)`
/// and it concatenates recursively at every nesting level: `block([a, b])`
/// (depth 1, 1-D atoms) is 1-D, while `block([[a, b]])` (depth 2) is 2-D, and a
/// list of scalars `block([1, 2, 3])` is 1-D. The prior binding required a
/// strict 2-D list-of-lists grid — it raised TypeError on a depth-1 list or a
/// list of scalars, and flattened a single-row `[[a, b]]` to 1-D instead of 2-D.
/// numpy owns the recursive depth/concatenation/promotion algorithm (and every
/// dtype: complex, datetime, string, f16, mixed), so delegate the whole op.
#[pyfunction]
pub fn block<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    np.call_method1("block", (arrays,))
}

/// Functional form of `numpy.r_[a, b, c, ...]` for the common
/// concat-1d-arrays case. NumPy's `np.r_` slice-syntax extras
/// (e.g. `np.r_[1:5, 8:12]`) need to be built up with arange/linspace
/// first and then passed through this function.
#[pyfunction]
#[pyo3(signature = (*arrays))]
pub fn r_<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("r_: need at least one array"));
    }
    // numpy/lib/_index_tricks promotes via result_type (NEP-50): r_[int8, int64]
    // -> int64, r_[int64, complex128] -> complex128 (no imaginary discard). The
    // real-only `match_dtype_all!` below cannot represent complex, so when the
    // promoted dtype is complex, delegate the whole op to numpy's r_ (which owns
    // the concatenate + promotion and preserves the imaginary part).
    let dt = common_dtype_tuple(py, arrays)?;
    if is_complex_dtype(&dt) {
        // `np.r_` is indexed via `__getitem__` with the (a, b, ...) tuple key:
        // `np.r_[a, b]` == `np.r_.__getitem__((a, b))`. Forward the *args tuple.
        let np = py.import("numpy")?;
        return np.getattr("r_")?.get_item(arrays);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let mut owned: Vec<ferray_core::array::aliases::Array1<T>> =
            Vec::with_capacity(arrays.len());
        for item in arrays.iter() {
            let coerced = crate::conv::coerce_dtype(py, &item, dt.as_str())?;
            // Flatten to 1-D before extracting — numpy.r_ works on 1-D inputs.
            let flat = coerced.call_method0("ravel")?;
            let view: numpy::PyReadonlyArray1<T> = flat.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let r: ferray_core::array::aliases::Array1<T> =
            fm::r_(&owned).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Functional form of `numpy.c_[a, b, c, ...]` for stacking 1-D arrays
/// as columns of a 2-D array.
#[pyfunction]
#[pyo3(signature = (*arrays))]
pub fn c_<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if arrays.is_empty() {
        return Err(PyValueError::new_err("c_: need at least one array"));
    }
    // numpy/lib/_index_tricks promotes via result_type (NEP-50): c_[int8, int64]
    // -> int64. As with r_, a complex promoted dtype can't ride the real-only
    // `match_dtype_all!`, so delegate to numpy's c_ (preserving the imaginary
    // part).
    let dt = common_dtype_tuple(py, arrays)?;
    if is_complex_dtype(&dt) {
        // `np.c_[a, b]` == `np.c_.__getitem__((a, b))`; forward the *args tuple.
        let np = py.import("numpy")?;
        return np.getattr("c_")?.get_item(arrays);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let mut owned: Vec<ferray_core::array::aliases::Array1<T>> =
            Vec::with_capacity(arrays.len());
        for item in arrays.iter() {
            let coerced = crate::conv::coerce_dtype(py, &item, dt.as_str())?;
            let flat = coerced.call_method0("ravel")?;
            let view: numpy::PyReadonlyArray1<T> = flat.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let r: ArrayD<T> = fm::c_(&owned).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.dstack(arrays)` — stack along axis 2.
#[pyfunction]
pub fn dstack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("dstack: need at least one array"));
    }
    if list_first_is_time(py, list)? {
        return crate::datetime::delegate_manip(py, "dstack", (arrays,), None);
    }
    if list_first_is_flexible(py, list)? {
        return string_delegate(py, "dstack", (arrays,), None);
    }
    if crate::conv::is_float16_dtype(dtype_name(&as_ndarray(py, &list.get_item(0)?)?)?.as_str()) {
        return crate::conv::f16_delegate(py, "dstack", (arrays,), None);
    }
    // numpy/_core/shape_base.py dstack promotes all inputs to result_type
    // (NEP-50), NOT the first array's dtype (matching concatenate/stack).
    let dt = common_dtype(py, list)?;
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::dstack(&typed).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, r)?
    }))
}
