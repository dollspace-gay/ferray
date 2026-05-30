//! Bindings for the `numpy` statistics surface — reductions,
//! nan-aware reductions, sort/search, set operations, cumulative
//! reductions, and `where`.
//!
//! Reductions return a `numpy.ndarray` (0-D when `axis=None`). NumPy
//! itself returns 0-D scalars (e.g. `numpy.int64`) for the same case;
//! a 0-D ndarray and a NumPy scalar are arithmetically interchangeable
//! and the deviation is invisible to almost all callers. If exact
//! parity is needed for a specific call, do `result.item()` to drop to
//! a Python scalar.
//!
//! Index-returning functions (`argmin`, `argmax`, `argsort`,
//! `count_nonzero`, `searchsorted`) are cast from ferray's internal
//! `u64` to `int64` so they match NumPy's `intp` default on 64-bit
//! platforms.

use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::dimension::{Ix1, IxDyn};
use ferray_linalg as fl;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    as_ndarray, coerce_dtype, dtype_name, extract_q, ferr_to_pyerr, normalize_axis,
    promote_linalg_input,
};
use crate::{match_dtype_all, match_dtype_float, match_dtype_numeric, match_dtype_orderable};

// ---------------------------------------------------------------------------
// Boolean reductions: all / any
// ---------------------------------------------------------------------------

/// `numpy.all(a, axis=None)` — test whether all elements are truthy.
///
/// `axis=None` returns a numpy `bool_` scalar (0-D bool array); an integer
/// `axis` collapses that axis, returning a bool ndarray
/// (numpy/_core/fromnumeric.py:2585 `all`, dispatching to
/// `numpy/_core/_methods.py:67` `_all` which `reduce`s `logical_and`). The
/// library `ferray_ufunc::all` is the whole-array fold; `all_axis` collapses
/// one axis. The whole-array `bool` is returned as a 0-D bool array (numpy's
/// `bool_` scalar and a 0-D bool array are interchangeable; `.item()` drops
/// to a Python bool if needed).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn all<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match axis {
            None => {
                let r = ferray_ufunc::all(&fa);
                ArrayD::<bool>::from_vec(IxDyn::new(&[]), vec![r])
                    .map_err(ferr_to_pyerr)?
                    .into_pyarray(py)
                    .map_err(ferr_to_pyerr)?
                    .into_any()
            }
            Some(ax) => {
                let r = ferray_ufunc::all_axis(&fa, ax).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }
    }))
}

/// `numpy.any(a, axis=None)` — test whether any element is truthy.
///
/// Mirrors [`all`]: `axis=None` returns a 0-D bool, an integer `axis`
/// returns a bool ndarray (numpy/_core/fromnumeric.py:2479 `any` ->
/// `_methods.py:62` `_any` reducing `logical_or`).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn any<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match axis {
            None => {
                let r = ferray_ufunc::any(&fa);
                ArrayD::<bool>::from_vec(IxDyn::new(&[]), vec![r])
                    .map_err(ferr_to_pyerr)?
                    .into_pyarray(py)
                    .map_err(ferr_to_pyerr)?
                    .into_any()
            }
            Some(ax) => {
                let r = ferray_ufunc::any_axis(&fa, ax).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }
    }))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Cast a `u64` ndarray to `i64` to match NumPy's default `intp`. The
/// values are always indices so the cast is non-lossy on 64-bit.
fn u64_arrd_to_i64_pyarray<'py>(py: Python<'py>, arr: ArrayD<u64>) -> PyResult<Bound<'py, PyAny>> {
    let shape = arr.shape().to_vec();
    let data: Vec<i64> = arr.iter().map(|&v| v as i64).collect();
    let cast = ArrayD::<i64>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
    Ok(cast.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

fn u64_arr1_to_i64_pyarray<'py>(py: Python<'py>, arr: Array1<u64>) -> PyResult<Bound<'py, PyAny>> {
    let n = arr.shape()[0];
    let data: Vec<i64> = arr.iter().map(|&v| v as i64).collect();
    let cast = Array1::<i64>::from_vec(Ix1::new([n]), data).map_err(ferr_to_pyerr)?;
    Ok(cast.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// True when the numpy array has zero elements (any axis of length 0).
/// Empty `mean`/`var`/`std`/`median` reductions over `axis=None` return
/// `nan` in NumPy (with a `RuntimeWarning`) rather than raising — see
/// numpy `_core/_methods.py:122` (`_mean`) and `:154-156` (`_var`).
fn is_empty_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let size: usize = arr.getattr("size")?.extract()?;
    Ok(size == 0)
}

/// A 0-D `float64` array holding `nan`. This is the binding's scalar
/// return for an empty `mean`/`var`/`std`/`median` reduction, matching
/// NumPy's `nan` (float64, 0-D) result for an empty slice.
fn nan_scalar_f64<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let r = ArrayD::<f64>::from_vec(IxDyn::new(&[]), vec![f64::NAN]).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// Reductions: numeric (sum, prod) — accept any numeric dtype
// ---------------------------------------------------------------------------

/// Normalize a binding `axis` argument against an array's `ndim`, raising
/// `numpy.exceptions.AxisError` (subclass of ValueError AND IndexError) for
/// an out-of-bounds axis — matching numpy/exceptions.py:108
/// `class AxisError(ValueError, IndexError)`. ferray's library
/// `validate_axis` surfaces a `ValueError`, which `except IndexError` /
/// `except np.exceptions.AxisError` would silently miss, so the binding
/// validates at the boundary (R-DEV-2). A negative axis (valid in numpy) is
/// folded into `[0, ndim)`; `None` passes through.
fn norm_axis(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
    axis: Option<isize>,
) -> PyResult<Option<usize>> {
    match axis {
        None => Ok(None),
        Some(ax) => {
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            Ok(Some(normalize_axis(py, ax, ndim)?))
        }
    }
}

/// `numpy.sum(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn sum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // bool sums in the platform integer (int64) — numpy/_core/fromnumeric.py:2325
    // ("`a` is signed then the platform integer is used"). ferray's library
    // ReduceAcc maps bool -> i64, so dispatch bool to the same promoting path.
    if dt.as_str() == "bool" {
        let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
        let fa: ArrayD<bool> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?;
        return Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sum(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.prod(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn prod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    // bool products promote to int64 like sum — numpy/_core/fromnumeric.py:2692.
    if dt.as_str() == "bool" {
        let view: PyReadonlyArrayDyn<bool> = arr.extract()?;
        let fa: ArrayD<bool> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?;
        return Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::prod(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Reductions: orderable (min, max, ptp) — any orderable dtype
// ---------------------------------------------------------------------------

/// `numpy.min(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn min<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::min(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.max(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn max<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::max(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.ptp(a, axis=None)` — peak-to-peak (max - min).
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn ptp<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::ptp(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Reductions: float-only (mean, var, std)
// ---------------------------------------------------------------------------

/// `numpy.mean(a, axis=None)` — float-only.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn mean<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Empty reduction: NumPy returns nan (float64, 0-D) + RuntimeWarning,
    // it does NOT raise (numpy/_core/_methods.py:122).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    // Promote integer inputs to float64 to match NumPy's mean semantics.
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::mean(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.var(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn var<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Empty reduction: NumPy warns "Degrees of freedom <= 0" and returns
    // nan (float64, 0-D); it does NOT raise (numpy/_core/_methods.py:154).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::var(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.std(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn std<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Empty reduction: `_std` delegates to `_var`, which returns nan
    // (float64, 0-D) for an empty slice; it does NOT raise
    // (numpy/_core/_methods.py:217 -> :154).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::std_(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Argmin / argmax (return int64 index arrays)
// ---------------------------------------------------------------------------

/// `numpy.argmin(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argmin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argmin(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.argmax(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argmax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argmax(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// NaN-aware reductions (float-only)
// ---------------------------------------------------------------------------

macro_rules! bind_nan_reduction {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            axis: Option<usize>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, a)?;
            let dt = dtype_name(&arr)?;
            let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
                arr
            } else {
                coerce_dtype(py, &arr, "float64")?
            };
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r = $ferr_path(&fa, axis).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

bind_nan_reduction!(nansum, ferray_stats::nansum);
bind_nan_reduction!(nanmean, ferray_stats::nanmean);
bind_nan_reduction!(nanmin, ferray_stats::nanmin);
bind_nan_reduction!(nanmax, ferray_stats::nanmax);
bind_nan_reduction!(nanprod, ferray_stats::nanprod);

/// `numpy.nanvar(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn nanvar<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanvar(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.nanstd(a, axis=None, ddof=0)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None, ddof = 0))]
pub fn nanstd<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanstd(&fa, axis, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.nanmedian(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanmedian<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::nanmedian(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.nanargmin(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanargmin<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::nanargmin(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.nanargmax(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn nanargmax<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
        arr
    } else {
        coerce_dtype(py, &arr, "float64")?
    };
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::nanargmax(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Cumulative reductions (cumsum, cumprod, diff)
// ---------------------------------------------------------------------------

/// `numpy.cumsum(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumsum<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // numpy cumsum promotes narrow-int accumulators to the platform int
        // (fromnumeric.py:2853-2855); ferray_stats::cumsum now returns the
        // promoted dtype, so let the bound type follow it.
        let r = ferray_stats::cumsum(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.cumprod(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn cumprod<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // numpy cumprod promotes narrow-int accumulators to the platform int;
        // ferray_stats::cumprod now returns the promoted dtype.
        let r = ferray_stats::cumprod(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.diff(a, n=1)` — discrete difference along the only axis.
/// ferray's diff is 1-D-only; multi-axis diff is deferred.
#[pyfunction]
#[pyo3(signature = (a, n = 1))]
pub fn diff<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, n: usize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_numeric!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::diff(&fa, n).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Sort / argsort
// ---------------------------------------------------------------------------

/// `numpy.sort(a, axis=-1)` — sorted copy.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn sort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_stats::SortKind;
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_stats::sort(&fa, axis, SortKind::Quick)
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.argsort(a, axis=-1)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn argsort<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argsort(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

/// `numpy.searchsorted(a, v, side="left")`.
#[pyfunction]
#[pyo3(signature = (a, v, side = "left"))]
pub fn searchsorted<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    side: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_stats::Side;
    let s = match side {
        "left" => Side::Left,
        "right" => Side::Right,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "side must be 'left' or 'right', got {other:?}"
            )));
        }
    };
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_v = coerce_dtype(py, v, dt.as_str())?;
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vv: PyReadonlyArray1<T> = arr_v.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: Array1<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::searchsorted(&fa, &fv, s).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Unique / count_nonzero
// ---------------------------------------------------------------------------

/// `numpy.unique(a)` — sorted unique values.
#[pyfunction]
pub fn unique<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::unique_values(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.count_nonzero(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn count_nonzero<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let r: ArrayD<u64> = match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::count_nonzero(&fa, axis).map_err(ferr_to_pyerr)?
    });
    u64_arrd_to_i64_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Set operations (1-D)
// ---------------------------------------------------------------------------

macro_rules! bind_set_op {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, b, assume_unique = false))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
            assume_unique: bool,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr_a = as_ndarray(py, a)?;
            let dt = dtype_name(&arr_a)?;
            let arr_b = coerce_dtype(py, b, dt.as_str())?;
            Ok(match_dtype_orderable!(dt.as_str(), T => {
                let va: PyReadonlyArray1<T> = arr_a.extract()?;
                let vb: PyReadonlyArray1<T> = arr_b.extract()?;
                let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
                let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
                let r: Array1<T> = $ferr_path(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

bind_set_op!(union1d, ferray_stats::union1d);
bind_set_op!(intersect1d, ferray_stats::intersect1d);
bind_set_op!(setdiff1d, ferray_stats::setdiff1d);
bind_set_op!(setxor1d, ferray_stats::setxor1d);

/// `numpy.in1d(ar1, ar2)` — bool array of presence.
#[pyfunction]
#[pyo3(signature = (a, b, assume_unique = false))]
pub fn in1d<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    assume_unique: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vb: PyReadonlyArray1<T> = arr_b.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<bool> = ferray_stats::in1d(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.isin(element, test_elements)`.
#[pyfunction]
#[pyo3(signature = (element, test_elements, assume_unique = false))]
pub fn isin<'py>(
    py: Python<'py>,
    element: &Bound<'py, PyAny>,
    test_elements: &Bound<'py, PyAny>,
    assume_unique: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, element)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, test_elements, dt.as_str())?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let va: PyReadonlyArray1<T> = arr_a.extract()?;
        let vb: PyReadonlyArray1<T> = arr_b.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: Array1<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<bool> = ferray_stats::isin(&fa, &fb, assume_unique).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// where_
// ---------------------------------------------------------------------------

/// `numpy.partition(a, kth)` — 1-D partial sort: kth element in its
/// final sorted position, smaller before, larger after.
#[pyfunction]
pub fn partition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    kth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::partition(&fa, kth).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.argpartition(a, kth)`.
#[pyfunction]
pub fn argpartition<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    kth: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_stats::argpartition(&fa, kth).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

/// `numpy.lexsort(keys)` — indirect lexicographic sort. `keys` is a
/// sequence of 1-D arrays; the LAST key is the primary sort key.
#[pyfunction]
pub fn lexsort<'py>(py: Python<'py>, keys: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = keys.cast::<pyo3::types::PyList>()?;
    if list.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lexsort: need at least one key array",
        ));
    }
    // Sniff dtype from the first key; coerce all to match.
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    let r: Array1<u64> = match_dtype_orderable!(dt.as_str(), T => {
        let mut owned: Vec<Array1<T>> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let coerced = coerce_dtype(py, &item, dt.as_str())?;
            let view: PyReadonlyArray1<T> = coerced.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let refs: Vec<&Array1<T>> = owned.iter().collect();
        ferray_stats::lexsort(&refs).map_err(ferr_to_pyerr)?
    });
    u64_arr1_to_i64_pyarray(py, r)
}

/// `numpy.unique(a, return_index=False, return_inverse=False, return_counts=False)`.
///
/// When extra return flags are requested, NumPy returns a tuple. We
/// match that here — passing all three flags returns a 4-tuple
/// `(values, indices, inverse, counts)`.
#[pyfunction]
#[pyo3(signature = (a, return_index = false, return_inverse = false, return_counts = false))]
pub fn unique_extended<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    return_index: bool,
    return_inverse: bool,
    return_counts: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_orderable!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let result = ferray_stats::unique(&fa, return_index, return_inverse, return_counts)
            .map_err(ferr_to_pyerr)?;
        let values_py = result.values.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let mut tuple_items: Vec<Bound<'py, pyo3::types::PyAny>> = vec![values_py];
        if let Some(idx) = result.indices {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, idx)?);
        }
        if let Some(inv) = result.inverse {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, inv)?);
        }
        if let Some(counts) = result.counts {
            tuple_items.push(u64_arr1_to_i64_pyarray(py, counts)?);
        }
        if tuple_items.len() == 1 {
            tuple_items.into_iter().next().unwrap()
        } else {
            pyo3::types::PyTuple::new(py, tuple_items)?.into_any()
        }
    }))
}

// ---------------------------------------------------------------------------
// histogram family (#704)
// ---------------------------------------------------------------------------

/// Parse the `bins` argument into a ferray `Bins<T>`. Accepts an int
/// (number of equal-width bins) or a sequence of float edges.
fn parse_bins<'py, T>(bins: &Bound<'py, PyAny>) -> PyResult<ferray_stats::Bins<T>>
where
    T: 'py,
    f64: Into<T>,
{
    if let Ok(n) = bins.extract::<usize>() {
        return Ok(ferray_stats::Bins::Count(n));
    }
    let edges_f64: Vec<f64> = bins.extract()?;
    let edges: Vec<T> = edges_f64.into_iter().map(Into::into).collect();
    Ok(ferray_stats::Bins::Edges(edges))
}

/// `numpy.histogram(a, bins=10, range=None, density=False)` →
/// `(hist, bin_edges)`. Histograms float input only (integer auto-promoted).
#[pyfunction]
#[pyo3(signature = (a, bins = 10usize, range = None, density = false))]
pub fn histogram<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    bins: usize,
    range: Option<(f64, f64)>,
    density: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, a)?;
    let arr_f64 = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArray1<f64> = arr_f64.extract()?;
    let fa: Array1<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let bins_e = ferray_stats::Bins::Count(bins);
    let (h, e) = ferray_stats::histogram(&fa, bins_e, range, density).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let e_py = e.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, e_py])?.into_any())
}

/// `numpy.histogram_bin_edges(a, bins=10, range=None)`.
#[pyfunction]
#[pyo3(signature = (a, bins = 10usize, range = None))]
pub fn histogram_bin_edges<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, a)?;
    let arr_f64 = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArray1<f64> = arr_f64.extract()?;
    let fa: Array1<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let bins_e = ferray_stats::Bins::Count(bins);
    let edges: Array1<f64> =
        ferray_stats::histogram_bin_edges(&fa, bins_e, range).map_err(ferr_to_pyerr)?;
    Ok(edges.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.histogram2d(x, y, bins=10)` → `(hist, x_edges, y_edges)`.
#[pyfunction]
#[pyo3(signature = (x, y, bins = (10usize, 10usize)))]
pub fn histogram2d<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
    bins: (usize, usize),
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = coerce_dtype(py, x, "float64")?;
    let arr_y = coerce_dtype(py, y, "float64")?;
    let vx: PyReadonlyArray1<f64> = arr_x.extract()?;
    let vy: PyReadonlyArray1<f64> = arr_y.extract()?;
    let fx: Array1<f64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    let fy: Array1<f64> = vy.as_ferray().map_err(ferr_to_pyerr)?;
    let (h, ex, ey) = ferray_stats::histogram2d(&fx, &fy, bins).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let ex_py = ex.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let ey_py = ey.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, ex_py, ey_py])?.into_any())
}

/// `numpy.histogramdd(sample, bins)` → `(hist, edges)`.
#[pyfunction]
pub fn histogramdd<'py>(
    py: Python<'py>,
    sample: &Bound<'py, PyAny>,
    bins: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = coerce_dtype(py, sample, "float64")?;
    let view: numpy::PyReadonlyArray2<f64> = arr.extract()?;
    let fa: Array2<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let (h, edges) = ferray_stats::histogramdd(&fa, &bins).map_err(ferr_to_pyerr)?;
    let h_py = h.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    let edges_py: Vec<Bound<'py, PyAny>> = edges
        .into_iter()
        .map(|e| Ok(e.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()))
        .collect::<PyResult<Vec<_>>>()?;
    let edges_list = pyo3::types::PyList::new(py, edges_py)?.into_any();
    Ok(pyo3::types::PyTuple::new(py, [h_py, edges_list])?.into_any())
}

/// `numpy.bincount(x, weights=None, minlength=0)`.
#[pyfunction]
#[pyo3(signature = (x, weights = None, minlength = 0))]
pub fn bincount<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    weights: Option<&Bound<'py, PyAny>>,
    minlength: usize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = coerce_dtype(py, x, "uint64")?;
    let vx: PyReadonlyArray1<u64> = arr_x.extract()?;
    let fx: Array1<u64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    if let Some(w) = weights {
        let arr_w = coerce_dtype(py, w, "float64")?;
        let vw: PyReadonlyArray1<f64> = arr_w.extract()?;
        let fw: Array1<f64> = vw.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<f64> =
            ferray_stats::bincount_weighted(&fx, &fw, minlength).map_err(ferr_to_pyerr)?;
        Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
    } else {
        let r: Array1<u64> = ferray_stats::bincount_u64(&fx, minlength).map_err(ferr_to_pyerr)?;
        u64_arr1_to_i64_pyarray(py, r)
    }
}

/// `numpy.digitize(x, bins, right=False)`.
#[pyfunction]
#[pyo3(signature = (x, bins, right = false))]
pub fn digitize<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    bins: &Bound<'py, PyAny>,
    right: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = coerce_dtype(py, x, "float64")?;
    let arr_b = coerce_dtype(py, bins, "float64")?;
    let vx: PyReadonlyArray1<f64> = arr_x.extract()?;
    let vb: PyReadonlyArray1<f64> = arr_b.extract()?;
    let fx: Array1<f64> = vx.as_ferray().map_err(ferr_to_pyerr)?;
    let fb: Array1<f64> = vb.as_ferray().map_err(ferr_to_pyerr)?;
    let r: Array1<u64> = ferray_stats::digitize(&fx, &fb, right).map_err(ferr_to_pyerr)?;
    u64_arr1_to_i64_pyarray(py, r)
}

// Force the parse_bins helper to count as used (will be wired into a
// future histogram(bins=array_of_edges) variant; phase-2.1 work).
#[allow(dead_code)]
fn _force_parse_bins_used<T>(b: &Bound<'_, PyAny>) -> PyResult<ferray_stats::Bins<T>>
where
    T: 'static,
    f64: Into<T>,
{
    parse_bins::<T>(b)
}

// ---------------------------------------------------------------------------
// percentile / quantile / median / cov / corrcoef (#703)
// ---------------------------------------------------------------------------

/// Whether a `percentile`/`quantile` library call computes percentages
/// (`true`) or fractions (`false`). Picks which scalar-q library fn to call.
enum QuantileKind {
    Percentile,
    Quantile,
}

/// Shared implementation for `percentile`/`quantile`, honoring numpy's
/// `q : array_like of float` contract (numpy/lib/_function_base_impl.py:4083
/// percentile, :4284 quantile). A scalar `q` returns the bare reduction; a
/// sequence `q` computes one reduction per entry and stacks them along a new
/// leading axis via `numpy.stack` (`np.percentile(x, [25,50,75])` ->
/// shape `(3, ...)`), matching numpy exactly (R-DEV-3). The library
/// percentile/quantile are scalar-q, so the binding loops them.
fn quantile_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
    kind: QuantileKind,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;

    // Compute one scalar-q reduction, returning a `Bound` pyarray.
    let single = |q_val: f64| -> PyResult<Bound<'py, PyAny>> {
        Ok(match_dtype_float!(real_dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r = match kind {
                QuantileKind::Percentile => ferray_stats::percentile(&fa, q_val as T, axis),
                QuantileKind::Quantile => ferray_stats::quantile(&fa, q_val as T, axis),
            }
            .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }))
    };

    match extract_q(q)? {
        Err(scalar) => single(scalar),
        Ok(seq) => {
            let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
            for q_val in seq {
                results.push(single(q_val)?);
            }
            // Stack per-q results along a new leading axis (numpy's q-axis).
            let np = py.import("numpy")?;
            let list = pyo3::types::PyList::new(py, results)?;
            np.call_method1("stack", (list,))
        }
    }
}

/// `numpy.percentile(a, q, axis=None)` — Linear interpolation method.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn percentile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    quantile_dispatch(py, a, q, axis, QuantileKind::Percentile)
}

/// `numpy.quantile(a, q, axis=None)` — Linear interpolation method.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn quantile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    quantile_dispatch(py, a, q, axis, QuantileKind::Quantile)
}

/// `numpy.median(a, axis=None)`.
#[pyfunction]
#[pyo3(signature = (a, axis = None))]
pub fn median<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // Empty reduction: NumPy's `_median` warns and returns nan
    // (float64, 0-D); it does NOT raise
    // (numpy/lib/_function_base_impl.py:4003).
    if axis.is_none() && is_empty_array(&arr)? {
        return nan_scalar_f64(py);
    }
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_stats::median(&fa, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.cov(m, rowvar=True, ddof=None)`.
#[pyfunction]
#[pyo3(signature = (m, rowvar = true, ddof = None))]
pub fn cov<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    rowvar: bool,
    ddof: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = ferray_stats::cov(&fa, rowvar, ddof).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.corrcoef(x, rowvar=True)`.
#[pyfunction]
#[pyo3(signature = (x, rowvar = true))]
pub fn corrcoef<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    rowvar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array2;
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = ferray_stats::corrcoef(&fa, rowvar).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.where(condition, x, y)` — three-argument form (broadcasting).
#[pyfunction]
#[pyo3(name = "where_")]
pub fn where_fn<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond = as_ndarray(py, condition)?;
    // Broadcast condition + x + y to a common shape via numpy.
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&cond, x, y))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let cond_b = coerce_dtype(py, &bcast[0], "bool")?;
    let arr_x = as_ndarray(py, &bcast[1])?;
    let dt = dtype_name(&arr_x)?;
    let arr_y = coerce_dtype(py, &bcast[2], dt.as_str())?;
    let cond_view: PyReadonlyArrayDyn<bool> = cond_b.extract()?;
    let cond_fa: ArrayD<bool> = cond_view.as_ferray().map_err(ferr_to_pyerr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let vx: PyReadonlyArrayDyn<T> = arr_x.extract()?;
        let vy: PyReadonlyArrayDyn<T> = arr_y.extract()?;
        let fx: ArrayD<T> = vx.as_ferray().map_err(ferr_to_pyerr)?;
        let fy: ArrayD<T> = vy.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_stats::where_(&cond_fa, &fx, &fy).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// nan-aware cumulative reductions
// ---------------------------------------------------------------------------

macro_rules! bind_nan_cumulative {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (a, axis = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            axis: Option<usize>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, a)?;
            let dt = dtype_name(&arr)?;
            let arr = if matches!(dt.as_str(), "float64" | "float32" | "f64" | "f32") {
                arr
            } else {
                coerce_dtype(py, &arr, "float64")?
            };
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                let r = $ferr_path(&fa, axis).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

// `numpy.nancumsum` treats NaN as 0, `nancumprod` treats NaN as 1
// (numpy/lib/_nanfunctions_impl.py:825 nancumsum, :915 nancumprod). The
// library re-exports the ferray-ufunc cumulative-with-nan-skip kernels.
bind_nan_cumulative!(nancumsum, ferray_stats::nancumsum);
bind_nan_cumulative!(nancumprod, ferray_stats::nancumprod);

// ---------------------------------------------------------------------------
// nan-aware percentile / quantile
// ---------------------------------------------------------------------------

/// Shared scalar-or-sequence dispatch for `nanpercentile`/`nanquantile`,
/// mirroring the `q : array_like of float` contract numpy documents for
/// `numpy.nanpercentile` (numpy/lib/_nanfunctions_impl.py:1156) and
/// `numpy.nanquantile` (:1382). A scalar `q` returns the bare reduction; a
/// sequence stacks one reduction per `q` along a new leading axis (matching
/// the non-nan `percentile`/`quantile` dispatch in [`quantile_dispatch`]).
/// The library `nanpercentile`/`nanquantile` are scalar-q, so the binding
/// loops them.
fn nan_quantile_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
    kind: QuantileKind,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let axis = norm_axis(py, &arr, axis)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;

    let single = |q_val: f64| -> PyResult<Bound<'py, PyAny>> {
        Ok(match_dtype_float!(real_dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r = match kind {
                QuantileKind::Percentile => ferray_stats::nanpercentile(&fa, q_val as T, axis),
                QuantileKind::Quantile => ferray_stats::nanquantile(&fa, q_val as T, axis),
            }
            .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }))
    };

    match extract_q(q)? {
        Err(scalar) => single(scalar),
        Ok(seq) => {
            let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(seq.len());
            for q_val in seq {
                results.push(single(q_val)?);
            }
            let np = py.import("numpy")?;
            let list = pyo3::types::PyList::new(py, results)?;
            np.call_method1("stack", (list,))
        }
    }
}

/// `numpy.nanpercentile(a, q, axis=None)` — percentile ignoring NaNs.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn nanpercentile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    nan_quantile_dispatch(py, a, q, axis, QuantileKind::Percentile)
}

/// `numpy.nanquantile(a, q, axis=None)` — quantile ignoring NaNs.
#[pyfunction]
#[pyo3(signature = (a, q, axis = None))]
pub fn nanquantile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    q: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    nan_quantile_dispatch(py, a, q, axis, QuantileKind::Quantile)
}

// ---------------------------------------------------------------------------
// average (weighted mean)
// ---------------------------------------------------------------------------

/// `numpy.average(a, axis=None, weights=None)`.
///
/// numpy/lib/_function_base_impl.py:560 `average` — with no `weights` this
/// is the plain `mean`; with `weights` it is `sum(a*w)/sum(w)` along `axis`.
/// Integer/bool input promotes to `float64` (numpy computes the weighted
/// average in inexact arithmetic). The library `ferray_stats::average`
/// requires `weights.shape() == a.shape()`, so the binding broadcasts a
/// 1-D `weights` against `a` via numpy before dispatch when an `axis` is
/// given, matching numpy's `weights` along a single axis.
#[pyfunction]
#[pyo3(signature = (a, axis = None, weights = None))]
pub fn average<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
    weights: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let arr = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
    let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let r = match weights {
        None => ferray_stats::average(&fa, None, axis).map_err(ferr_to_pyerr)?,
        Some(w) => {
            let warr = coerce_dtype(py, w, "float64")?;
            // numpy broadcasts a 1-D `weights` against the reduced axis.
            // When `weights` already matches `a`'s shape, broadcast is a
            // no-op; otherwise reshape it onto `axis` then broadcast.
            let np = py.import("numpy")?;
            let target_shape = arr.getattr("shape")?;
            let wshape: Vec<usize> = warr.getattr("shape")?.extract()?;
            let ashape: Vec<usize> = arr.getattr("shape")?.extract()?;
            let warr_b = if wshape == ashape {
                warr
            } else if let Some(ax) = axis {
                // Reshape 1-D weights to broadcast along `axis`, then expand.
                let ndim = ashape.len();
                let mut newshape = vec![1usize; ndim];
                if ax < ndim {
                    newshape[ax] = wshape.iter().product();
                }
                let reshaped = np.call_method1("reshape", (&warr, newshape))?;
                np.call_method1("broadcast_to", (reshaped, target_shape))?
            } else {
                np.call_method1("broadcast_to", (&warr, target_shape))?
            };
            let wview: PyReadonlyArrayDyn<f64> = warr_b.extract()?;
            let fw: ArrayD<f64> = wview.as_ferray().map_err(ferr_to_pyerr)?;
            ferray_stats::average(&fa, Some(&fw), axis).map_err(ferr_to_pyerr)?
        }
    };
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// sort_complex
// ---------------------------------------------------------------------------

/// `numpy.sort_complex(a)` — sort a complex array lexicographically by real
/// part then imaginary part (numpy/lib/_function_base_impl.py:638). A real
/// input is upcast to complex first (numpy: `b = array(a, copy=True);
/// b.sort(); ... if not issubclass(b.dtype.type, complexfloating): ... return
/// b.astype(complex)` — a real array's result is the complex-typed sorted
/// values). The library `ferray_stats::sort_complex` operates on
/// `Complex<f64>` / `Complex<f32>`; the binding coerces the input to
/// `complex128` (or keeps `complex64`) before dispatch.
#[pyfunction]
pub fn sort_complex<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let target = if dt.as_str() == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_dtype(py, &arr, target)?;
    // The library `sort_complex` is Ix1 over the flattened input. Extract the
    // complex data into a 1-D ferray array, sort, and return as a fresh numpy
    // complex array via the same materialiser the fft bindings use.
    if target == "complex64" {
        let fa = complex_pyarray_to_ferray::<f32>(&arr)?;
        let n = fa.size();
        let flat: Vec<num_complex::Complex<f32>> = fa.iter().copied().collect();
        let a1 = Array1::<num_complex::Complex<f32>>::from_vec(Ix1::new([n]), flat)
            .map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sort_complex(&a1).map_err(ferr_to_pyerr)?;
        let data: Vec<num_complex::Complex<f32>> = r.iter().copied().collect();
        let rd = ArrayD::<num_complex::Complex<f32>>::from_vec(IxDyn::new(&[n]), data)
            .map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, rd)
    } else {
        let fa = complex_pyarray_to_ferray::<f64>(&arr)?;
        let n = fa.size();
        let flat: Vec<num_complex::Complex<f64>> = fa.iter().copied().collect();
        let a1 = Array1::<num_complex::Complex<f64>>::from_vec(Ix1::new([n]), flat)
            .map_err(ferr_to_pyerr)?;
        let r = ferray_stats::sort_complex(&a1).map_err(ferr_to_pyerr)?;
        let data: Vec<num_complex::Complex<f64>> = r.iter().copied().collect();
        let rd = ArrayD::<num_complex::Complex<f64>>::from_vec(IxDyn::new(&[n]), data)
            .map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, rd)
    }
}

// ---------------------------------------------------------------------------
// cross (1-D 3-vector)
// ---------------------------------------------------------------------------

/// `numpy.cross(a, b)` — the legacy top-level cross product
/// (`numpy/_core/numeric.py:cross`).
///
/// Routes through `ferray_linalg::cross`, which computes the cross product
/// along the last axis and supports:
/// - **3-component** vectors `(...,3) × (...,3) → (...,3)`, including stacked
///   inputs that broadcast over their leading batch axes (#836);
/// - **2-component** vectors `(...,2) × (...,2)`, returning the scalar
///   z-component (the legacy form numpy 2.4.5 still allows with a
///   `DeprecationWarning`).
///
/// Integer inputs are promoted to `float64` via `promote_linalg_input`
/// (the library cross is float-typed); numpy's documented contract is the
/// numerical cross-product value, which is preserved.
#[pyfunction]
pub fn cross<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::cross(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
