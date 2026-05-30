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
