//! Top-level numpy alias / introspection bindings.
//!
//! This module hosts the array-introspection and dtype-introspection
//! functions numpy exposes at the package root — `numpy.ndim`,
//! `numpy.shape`, `numpy.size`, `numpy.isscalar`, `numpy.isfortran`,
//! `numpy.result_type`, `numpy.promote_types`, `numpy.can_cast`,
//! `numpy.min_scalar_type`, `numpy.issubdtype`, `numpy.isdtype`,
//! `numpy.common_type`, and `numpy.astype`.
//!
//! These functions operate purely on the *boundary* objects that the
//! ferray bindings already produce: every ferray array surfaced through
//! PyO3 is a genuine `numpy.ndarray` carrying a genuine `numpy.dtype`
//! (the binding layer round-trips through `numpy.asarray` /
//! `IntoNumPy`). The dtype algebra (`promote_types`, `result_type`,
//! `can_cast`, …) is therefore evaluated against those same numpy dtype
//! objects so the result is bit-for-bit consistent with what a caller
//! would observe by reading `arr.dtype` off a ferray array. This mirrors
//! how numpy itself defines these as thin wrappers over the dtype
//! machinery (`numpy/_core/numerictypes.py:412` `def issubdtype`,
//! `numpy/_core/numerictypes.py:322` `def isdtype`).
//!
//! The pure 1:1 ufunc/manipulation aliases (`acos`→`arccos`,
//! `concat`→`concatenate`, `pow`→`power`, `amax`→`max`, …) are *not*
//! here: numpy itself defines them as bare Python assignments
//! (`numpy/_core/__init__.py:138` `acos = numeric.arccos`), so the
//! ferray shim mirrors them as re-exports in
//! `python/ferray/__init__.py` rather than re-wrapping the already-bound
//! `#[pyfunction]`.
//!
//! ## REQ status — `numpy` introspection / dtype-algebra surface (INFRA)
//!
//! This module is the dtype-alias + introspection surface: each callable
//! is bound here as a `#[pyfunction]` operating on the *boundary* numpy
//! objects the rest of the shim already produces (every ferray array is a
//! genuine `numpy.ndarray` with a genuine `numpy.dtype`), so the dtype
//! algebra is evaluated against numpy's own dtype machinery rather than a
//! ferray kernel — this is INFRA, not a kernel delegation. Green against
//! numpy 2.4.x. Symbol anchors per R-CITE-2b.
//!
//! SHIPPED:
//!   - Array introspection: `ndim` / `shape` / `size` / `isscalar` /
//!     `isfortran` (`numpy/_core/fromnumeric.py` +
//!     `numpy/_core/numeric.py`).
//!   - dtype algebra: `result_type` / `promote_types` / `can_cast` /
//!     `min_scalar_type` / `common_type` / `astype` — thin wrappers over
//!     numpy's dtype objects so results are bit-for-bit consistent with
//!     `arr.dtype` read off a ferray array.
//!   - dtype predicates: `issubdtype` (`numpy/_core/numerictypes.py:412`
//!     `def issubdtype`), `isdtype` (`numpy/_core/numerictypes.py:322`
//!     `def isdtype`).
//!   - `divmod` alias surfaced at the package root.
//!
//! NOT-STARTED: none — every introspection/dtype callable registered here
//! is bound and green. (The pure 1:1 `acos`→`arccos` style aliases are by
//! design re-exports in `python/ferray/__init__.py`, not rows here.)

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

use crate::conv::as_ndarray;

// ---------------------------------------------------------------------------
// Array introspection — numpy/_core/fromnumeric.py + numpy/_core/numeric.py
// ---------------------------------------------------------------------------

/// `numpy.ndim(a)` — number of array dimensions.
///
/// numpy/_core/fromnumeric.py:3483 `def ndim(a)` returns `a.ndim` when
/// `a` has the attribute, else `asarray(a).ndim`. A Python scalar
/// therefore yields `0`. Mirrors that contract exactly.
#[pyfunction]
pub fn ndim(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<usize> {
    if let Ok(nd) = a.getattr("ndim") {
        if let Ok(n) = nd.extract::<usize>() {
            return Ok(n);
        }
    }
    let arr = as_ndarray(py, a)?;
    arr.getattr("ndim")?.extract()
}

/// `numpy.shape(a)` — the shape of an array.
///
/// numpy/_core/fromnumeric.py:2085 `def shape(a)` returns `a.shape`
/// when present, else `asarray(a).shape`. A Python scalar yields `()`.
#[pyfunction]
pub fn shape<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    if let Ok(sh) = a.getattr("shape") {
        if let Ok(t) = sh.cast_into::<PyTuple>() {
            return Ok(t);
        }
    }
    let arr = as_ndarray(py, a)?;
    arr.getattr("shape")?
        .cast_into::<PyTuple>()
        .map_err(Into::into)
}

/// `numpy.size(a, axis=None)` — number of elements (along an axis).
///
/// numpy/_core/fromnumeric.py:3526 `def size(a, axis=None)` returns
/// `a.size` when `axis is None`, else `a.shape[axis]`. A Python scalar
/// yields `1`. An out-of-range axis raises `numpy.exceptions.AxisError`
/// (matching numpy's `a.shape[axis]` IndexError surface upgraded to the
/// numpy axis-error type).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn size(py: Python<'_>, a: &Bound<'_, PyAny>, axis: Option<isize>) -> PyResult<usize> {
    let arr = if a.hasattr("shape")? {
        a.clone()
    } else {
        as_ndarray(py, a)?
    };
    match axis {
        None => arr.getattr("size")?.extract(),
        Some(ax) => {
            let shape: Vec<isize> = arr.getattr("shape")?.extract()?;
            let ndim = shape.len() as isize;
            let idx = if ax < 0 { ax + ndim } else { ax };
            if idx < 0 || idx >= ndim {
                return Err(crate::conv::axis_error(py, ax, ndim.max(0) as usize));
            }
            Ok(shape[idx as usize] as usize)
        }
    }
}

/// `numpy.isscalar(element)` — `True` for a scalar element.
///
/// numpy/_core/numeric.py:1911 `def isscalar(element)` is `True` only
/// for genuine Python/numpy scalars (numpy generic, `numbers.Number`,
/// `str`, `bytes`), and `False` for ndarrays and sequences — note
/// `np.isscalar([3])` is `False`. The binding routes to numpy's own
/// `isscalar` so the (deliberately quirky) contract matches exactly
/// across every type.
#[pyfunction]
pub fn isscalar(py: Python<'_>, element: &Bound<'_, PyAny>) -> PyResult<bool> {
    py.import("numpy")?
        .call_method1("isscalar", (element,))?
        .extract()
}

/// `numpy.isfortran(a)` — `True` if the array is Fortran-contiguous and
/// at least 2-D.
///
/// numpy/_core/numeric.py:549 `def isfortran(a)` returns `a.flags.fnc`
/// (`f_contiguous and not c_contiguous`).
#[pyfunction]
pub fn isfortran(py: Python<'_>, a: &Bound<'_, PyAny>) -> PyResult<bool> {
    let arr = as_ndarray(py, a)?;
    py.import("numpy")?
        .call_method1("isfortran", (arr,))?
        .extract()
}

// ---------------------------------------------------------------------------
// dtype introspection — numpy/_core/numerictypes.py + multiarray
// ---------------------------------------------------------------------------

/// `numpy.astype(x, dtype, /, *, copy=True)` — cast an array to a dtype
/// (array-API spec function). `x` is normalised through `asarray` so an
/// array-like is accepted (strictly more permissive than numpy's
/// ndarray-only `astype`, never changing the result for a real ndarray).
#[pyfunction]
#[pyo3(signature = (x, dtype, copy = true))]
pub fn astype<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    dtype: &Bound<'py, PyAny>,
    copy: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("copy", copy)?;
    arr.call_method("astype", (dtype,), Some(&kwargs))
}

/// `numpy.can_cast(from_, to, casting='safe')` — whether a cast is
/// allowed under a casting rule.
#[pyfunction]
#[pyo3(signature = (from_, to, casting = "safe"))]
pub fn can_cast(
    py: Python<'_>,
    from_: &Bound<'_, PyAny>,
    to: &Bound<'_, PyAny>,
    casting: &str,
) -> PyResult<bool> {
    let kwargs = PyDict::new(py);
    kwargs.set_item("casting", casting)?;
    py.import("numpy")?
        .call_method("can_cast", (from_, to), Some(&kwargs))?
        .extract()
}

/// `numpy.promote_types(type1, type2)` — the smallest dtype to which
/// both inputs can be safely cast.
#[pyfunction]
pub fn promote_types<'py>(
    py: Python<'py>,
    type1: &Bound<'py, PyAny>,
    type2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?
        .call_method1("promote_types", (type1, type2))
}

/// `numpy.result_type(*arrays_and_dtypes)` — the dtype that results from
/// applying numpy's promotion rules (NEP-50) to all inputs.
#[pyfunction]
#[pyo3(signature = (*args))]
pub fn result_type<'py>(
    py: Python<'py>,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("result_type", args)
}

/// `numpy.min_scalar_type(a)` — the minimal dtype that can hold a scalar.
#[pyfunction]
pub fn min_scalar_type<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("min_scalar_type", (a,))
}

/// `numpy.issubdtype(arg1, arg2)` — whether the first dtype is a
/// sub-dtype of the second in numpy's scalar-type hierarchy
/// (numpy/_core/numerictypes.py:412 `def issubdtype`).
#[pyfunction]
pub fn issubdtype(
    py: Python<'_>,
    arg1: &Bound<'_, PyAny>,
    arg2: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    py.import("numpy")?
        .call_method1("issubdtype", (arg1, arg2))?
        .extract()
}

/// `numpy.isdtype(dtype, kind)` — array-API dtype-kind predicate
/// (numpy/_core/numerictypes.py:322 `def isdtype`). `kind` may be a
/// single dtype, a kind string (`'bool'`, `'signed integer'`,
/// `'integral'`, `'real floating'`, `'complex floating'`, `'numeric'`),
/// or a tuple of either.
#[pyfunction]
pub fn isdtype(
    py: Python<'_>,
    dtype: &Bound<'_, PyAny>,
    kind: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    py.import("numpy")?
        .call_method1("isdtype", (dtype, kind))?
        .extract()
}

/// `numpy.common_type(*arrays)` — a common scalar *type* (always
/// inexact: float or complex) for the inputs.
#[pyfunction]
#[pyo3(signature = (*arrays))]
pub fn common_type<'py>(
    py: Python<'py>,
    arrays: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1("common_type", arrays)
}

// ---------------------------------------------------------------------------
// divmod ufunc — generate_umath.py
// ---------------------------------------------------------------------------

/// `numpy.divmod(x1, x2)` — simultaneous floor-division and remainder,
/// returning the `(x1 // x2, x1 % x2)` tuple.
///
/// numpy's `divmod` ufunc works on integer dtypes (where the dedicated
/// float-only `modf`-style binding diverges), so the binding composes
/// the already-bound `floor_divide` and `mod` ufuncs — both of which
/// carry numpy-correct integer + float kernels — into the 2-tuple. This
/// keeps the integer-dtype contract (`np.divmod(7, 3) == (2, 1)`) that a
/// float-only path cannot express.
///
/// numpy's `divmod` ufunc registers integer and float loops only — no
/// bool loop (numpy/_core/code_generators/generate_umath.py:1048
/// `'divmod': Ufunc(2, 2, None, …, TD(ints, …), TD(flts), …)`). A bool
/// operand therefore promotes `bool -> int8` under ufunc type resolution,
/// so `np.divmod(True, True) == (np.int8(1), np.int8(0))`. ferray's
/// `floor_divide`/`mod` dispatch (`match_dtype_numeric!`, conv.rs:967)
/// has no bool arm, so the promotion is applied here: if either operand
/// is bool dtype, both are coerced to `int8` before the integer divmod
/// runs, yielding an int8 tuple that matches numpy.
#[pyfunction]
pub fn divmod<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let x1_bool = crate::conv::dtype_name(&as_ndarray(py, x1)?)? == "bool";
    let x2_bool = crate::conv::dtype_name(&as_ndarray(py, x2)?)? == "bool";
    if x1_bool || x2_bool {
        let a = crate::conv::coerce_dtype(py, x1, "int8")?;
        let b = crate::conv::coerce_dtype(py, x2, "int8")?;
        let q = crate::ufunc::floor_divide(py, &a, &b)?;
        let r = crate::ufunc::mod_(py, &a, &b)?;
        return PyTuple::new(py, [q, r]);
    }
    let q = crate::ufunc::floor_divide(py, x1, x2)?;
    let r = crate::ufunc::mod_(py, x1, x2)?;
    PyTuple::new(py, [q, r])
}
