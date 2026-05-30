//! Bindings for the `numpy` array-manipulation surface.
//!
//! Functions here all dispatch on the input array's NumPy dtype via
//! [`match_dtype_all!`], extract a `PyReadonlyArrayDyn<T>` view,
//! materialise an `ArrayD<T>` through `AsFerray`, run the typed ferray
//! function, and push the result back through `IntoNumPy`.
//!
//! The split between the per-function bindings here and the
//! per-dtype dispatch in the macro means the cost of supporting a new
//! dtype is one new arm in `match_dtype_all!` and zero changes in
//! this file.

use ferray_core::array::aliases::ArrayD;
use ferray_core::dimension::IxDyn;
use ferray_core::manipulation as fm;
use ferray_core::manipulation::extended as fme;
use ferray_numpy_interop::numpy_conv::NpElement;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};

use crate::conv::{
    as_ndarray, axis_error, dtype_name, extract_axis_tuple, ferr_to_pyerr, normalize_axis,
    normalize_axis_tuple,
};
use crate::match_dtype_all;

/// Read an array-like's `ndim` (after `as_ndarray` coercion).
fn obj_ndim(arr: &Bound<'_, PyAny>) -> PyResult<usize> {
    arr.getattr("ndim")?.extract()
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
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
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
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.ravel(a)` — flatten to 1-D.
#[pyfunction]
pub fn ravel<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = fm::ravel(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.ndarray.flatten` as a free function.
#[pyfunction]
pub fn flatten<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r = fm::flatten(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let mut cur: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match &axes {
            None => cur = fm::squeeze(&cur, None).map_err(ferr_to_pyerr)?,
            Some(list) => {
                for &ax in list {
                    cur = fm::squeeze(&cur, Some(ax)).map_err(ferr_to_pyerr)?;
                }
            }
        }
        cur.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let raw = extract_axis_tuple(axis)?;
    let out_ndim = raw.len() + ndim;
    let mut axes = normalize_axis_tuple(py, &raw, out_ndim, false)?;
    // Insert in ascending order so earlier insertions don't shift the
    // positions of later (already-normalized) target axes.
    axes.sort_unstable();
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let mut cur: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        for &ax in &axes {
            cur = fm::expand_dims(&cur, ax).map_err(ferr_to_pyerr)?;
        }
        cur.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let target: Vec<usize> = if let Ok(n) = shape.extract::<usize>() {
        vec![n]
    } else {
        shape.extract()?
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::broadcast_to(&fa, &target).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    // numpy/_core/fromnumeric.py:605 transpose — explicit axes may be
    // negative and are normalized against ndim.
    let norm_axes: Option<Vec<usize>> = match axes {
        Some(ax) => Some(normalize_axis_tuple(py, &ax, ndim, false)?),
        None => None,
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::transpose(&fa, norm_axes.as_deref()).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax1 = normalize_axis(py, axis1, ndim)?;
    let ax2 = normalize_axis(py, axis2, ndim)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::swapaxes(&fa, ax1, ax2).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::transpose(&fa, Some(&order)).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.rollaxis(a, axis, start=0)` — legacy, prefer moveaxis.
#[pyfunction]
#[pyo3(signature = (a, axis, start = 0))]
pub fn rollaxis<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: usize,
    start: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::rollaxis(&fa, axis, start).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let axes: Vec<usize> = match axis {
        None => (0..ndim).collect(),
        Some(obj) => normalize_axis_tuple(py, &extract_axis_tuple(obj)?, ndim, false)?,
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let mut cur: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        for &ax in &axes {
            cur = fm::flip(&cur, ax).map_err(ferr_to_pyerr)?;
        }
        cur.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.fliplr(m)` — flip along axis 1.
#[pyfunction]
pub fn fliplr<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::fliplr(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.flipud(m)` — flip along axis 0.
#[pyfunction]
pub fn flipud<'py>(py: Python<'py>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::flipud(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax0 = normalize_axis(py, axes.0, ndim)?;
    let ax1 = normalize_axis(py, axes.1, ndim)?;
    if ax0 == ax1 {
        return Err(PyValueError::new_err("axes must be different"));
    }
    let use_core_fast_path = ax0 == 0 && ax1 == 1;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
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
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.roll(a, shift, axis=None)` — `axis` may be negative
/// (numpy/_core/numeric.py:1226).
#[pyfunction]
#[pyo3(signature = (a, shift, axis = None))]
pub fn roll<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    shift: isize,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let norm_axis: Option<usize> = match axis {
        Some(ax) => Some(normalize_axis(py, ax, ndim)?),
        None => None,
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fm::roll(&fa, shift, norm_axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// atleast_*d
// ---------------------------------------------------------------------------

/// `numpy.atleast_1d(a)`.
#[pyfunction]
pub fn atleast_1d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::atleast_1d(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.atleast_2d(a)`.
#[pyfunction]
pub fn atleast_2d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::atleast_2d(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.atleast_3d(a)`.
#[pyfunction]
pub fn atleast_3d<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::atleast_3d(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_core::creation::tril(&fa, k).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.triu(m, k=0)` — upper triangle (zero-out below-k diagonal).
#[pyfunction]
#[pyo3(signature = (m, k = 0))]
pub fn triu<'py>(py: Python<'py>, m: &Bound<'py, PyAny>, k: isize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, m)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_core::creation::triu(&fa, k).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.diag(v, k=0)` — extract a diagonal or build a 2-D array
/// from a 1-D diagonal.
#[pyfunction]
#[pyo3(signature = (v, k = 0))]
pub fn diag<'py>(py: Python<'py>, v: &Bound<'py, PyAny>, k: isize) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, v)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_core::creation::diag(&fa, k).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_core::creation::diagflat(&fa, k).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
fn collect_typed<'py, T: NpElement>(
    py: Python<'py>,
    arrays: &Bound<'py, PyAny>,
    dt_name: &str,
) -> PyResult<Vec<ArrayD<T>>> {
    let np = py.import("numpy")?;
    let list = arrays.cast::<PyList>()?;
    let mut out: Vec<ArrayD<T>> = Vec::with_capacity(list.len());
    for item in list.iter() {
        let coerced = np.call_method1("asarray", (&item, dt_name))?;
        let view: PyReadonlyArrayDyn<T> = coerced.extract()?;
        out.push(view.as_ferray().map_err(ferr_to_pyerr)?);
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
    let dt = common_dtype(py, list)?;
    // For axis=None numpy ravels each input and concatenates along axis 0.
    let (flatten, norm_axis): (bool, usize) = match axis {
        None => (true, 0),
        Some(ax) => {
            let ndim = obj_ndim(&as_ndarray(py, &list.get_item(0)?)?)?;
            (false, normalize_axis(py, ax, ndim)?)
        }
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
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
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = common_dtype(py, list)?;
    let result_ndim = obj_ndim(&as_ndarray(py, &list.get_item(0)?)?)? + 1;
    let norm_axis = normalize_axis(py, axis, result_ndim)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::stack(&typed, norm_axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.vstack(arrays)` — stack along axis 0.
#[pyfunction]
pub fn vstack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("vstack: need at least one array"));
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::vstack(&typed).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.hstack(arrays)` — stack along axis 1 (or 0 for 1-D).
#[pyfunction]
pub fn hstack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<PyList>()?;
    if list.is_empty() {
        return Err(PyValueError::new_err("hstack: need at least one array"));
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::hstack(&typed).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Padding / tiling / repeating / element insertion
// ---------------------------------------------------------------------------

/// `numpy.pad(array, pad_width, mode='constant', constant_values=0)`.
///
/// Supports the most common modes: `constant`, `edge`, `reflect`,
/// `symmetric`, `wrap`. The `linear_ramp`/`maximum`/`minimum`/`mean`/
/// `median` modes are deferred to a smaller follow-up.
#[pyfunction]
#[pyo3(signature = (array, pad_width, mode = "constant", constant_values = 0.0))]
pub fn pad<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    pad_width: &Bound<'py, PyAny>,
    mode: &str,
    constant_values: f64,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::manipulation::extended::PadMode;
    let arr = as_ndarray(py, array)?;
    let dt = dtype_name(&arr)?;

    // pad_width can be:
    //   int n        -> apply (n, n) on every axis
    //   (before, after) tuple -> apply same to every axis
    //   sequence of (before, after) tuples -> per-axis
    let arr_ndim: usize = arr.getattr("ndim")?.extract()?;
    let widths: Vec<(usize, usize)> = if let Ok(n) = pad_width.extract::<usize>() {
        vec![(n, n); arr_ndim]
    } else if let Ok((b, a)) = pad_width.extract::<(usize, usize)>() {
        vec![(b, a); arr_ndim]
    } else {
        pad_width.extract::<Vec<(usize, usize)>>()?
    };

    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let pad_mode: PadMode<T> = match mode {
            "constant" => {
                // Coerce f64 → T via the FromF64Lossy trait used in `full`,
                // but it lives in creation.rs. Inline the conversion for
                // the small number of supported types.
                let cv: T = ferray_python_constant_cast::<T>(constant_values);
                PadMode::Constant(cv)
            }
            "edge" => PadMode::Edge,
            "reflect" => PadMode::Reflect,
            "symmetric" => PadMode::Symmetric,
            "wrap" => PadMode::Wrap,
            other => {
                return Err(PyValueError::new_err(format!(
                    "pad mode {other:?} not supported (try constant/edge/reflect/symmetric/wrap)"
                )));
            }
        };
        let r: ArrayD<T> = fme::pad(&fa, &widths, &pad_mode).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Helper for `pad` — coerce an `f64` constant to the dispatched element
/// type. Local to this module so we don't have to expose the
/// `FromF64Lossy` trait from `creation`.
trait PadConstantCast {
    fn from_f64(v: f64) -> Self;
}
macro_rules! impl_pad_cast {
    ($($t:ty),*) => {
        $(impl PadConstantCast for $t { fn from_f64(v: f64) -> Self { v as Self } })*
    };
}
impl_pad_cast!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl PadConstantCast for bool {
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}
fn ferray_python_constant_cast<T: PadConstantCast>(v: f64) -> T {
    T::from_f64(v)
}

/// `numpy.tile(A, reps)` — construct an array by repeating A `reps` times.
#[pyfunction]
pub fn tile<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    reps: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let reps_vec: Vec<usize> = if let Ok(n) = reps.extract::<usize>() {
        vec![n]
    } else {
        reps.extract()?
    };
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::tile(&fa, &reps_vec).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let norm_axis: Option<usize> = match axis {
        Some(ax) => Some(normalize_axis(py, ax, ndim)?),
        None => None,
    };
    // Scalar vs per-element repeat counts.
    let scalar_reps: Option<usize> = repeats.extract::<usize>().ok();
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = match scalar_reps {
            Some(n) => fme::repeat(&fa, n, norm_axis).map_err(ferr_to_pyerr)?,
            None => {
                let counts: Vec<usize> = repeats.extract()?;
                repeat_per_element::<T>(&fa, &counts, norm_axis)?
            }
        };
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
                return Err(ferr_to_pyerr(
                    ferray_core::FerrayError::axis_out_of_bounds(ax, ndim),
                ));
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

/// `numpy.delete(arr, obj, axis)` — remove sub-arrays along `axis`
/// (numpy/lib/_function_base_impl.py:5221). `obj` may be an int, a list
/// of ints (negatives normalized), a `slice`, or a boolean mask the same
/// length as the axis. `axis` may be negative.
#[pyfunction]
pub fn delete<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    obj: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, arr)?;
    let dt = dtype_name(&arr)?;
    let ndim = obj_ndim(&arr)?;
    let ax = normalize_axis(py, axis, ndim)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let axis_len = shape[ax];
    let indices = resolve_delete_obj(py, obj, axis_len)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::delete(&fa, &indices, ax).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Resolve the `obj` argument of `numpy.delete` to a list of usize
/// indices along an axis of length `axis_len`. Handles `slice`, boolean
/// mask, and int / list-of-ints (negatives counted from the end).
fn resolve_delete_obj(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    axis_len: usize,
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
                Err(axis_error(py, i, axis_len))
            } else {
                Ok(norm as usize)
            }
        })
        .collect()
}

/// `numpy.insert(arr, obj, values, axis)`.
#[pyfunction]
pub fn insert<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    obj: usize,
    values: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, arr)?;
    let dt = dtype_name(&arr_a)?;
    let arr_v = crate::conv::coerce_dtype(py, values, dt.as_str())?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vv: PyReadonlyArrayDyn<T> = arr_v.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: ArrayD<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::insert(&fa, obj, &fv, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
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
    let dt = dtype_name(&arr_a)?;
    let arr_v = crate::conv::coerce_dtype(py, values, dt.as_str())?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vv: PyReadonlyArrayDyn<T> = arr_v.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: ArrayD<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::append(&fa, &fv, axis).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
    let target: Vec<usize> = if let Ok(n) = new_shape.extract::<usize>() {
        vec![n]
    } else {
        new_shape.extract()?
    };
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fme::resize(&fa, &target).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
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
fn dyn_arrays_to_pylist<'py, T: ferray_core::Element + numpy::Element + Clone>(
    py: Python<'py>,
    arrays: Vec<ArrayD<T>>,
) -> PyResult<Bound<'py, PyAny>>
where
    ArrayD<T>: ferray_numpy_interop::IntoNumPy<T, ferray_core::dimension::IxDyn>,
{
    let py_arrays: Vec<Bound<'py, PyAny>> = arrays
        .into_iter()
        .map(|a| Ok(a.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()))
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
    let dt = dtype_name(&arr)?;

    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
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
    let dt = dtype_name(&arr)?;

    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
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
            let dt = dtype_name(&arr)?;
            Ok(match_dtype_all!(dt.as_str(), T => {
                let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
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
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::column_stack(&typed).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.row_stack(arrays)` — alias for `vstack`.
#[pyfunction]
pub fn row_stack<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    vstack(py, arrays)
}

/// `numpy.block(arrays)` — assemble an array from a 2-D nested-list grid.
#[pyfunction]
pub fn block<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let outer = arrays.cast::<PyList>()?;
    if outer.is_empty() {
        return Err(PyValueError::new_err("block: empty input"));
    }
    // Sniff dtype from the first inner element of the first row.
    let first_row = outer.get_item(0)?;
    let first_row_list = first_row.cast::<PyList>()?;
    if first_row_list.is_empty() {
        return Err(PyValueError::new_err("block: empty row"));
    }
    let first = as_ndarray(py, &first_row_list.get_item(0)?)?;
    let dt = dtype_name(&first)?;

    Ok(match_dtype_all!(dt.as_str(), T => {
        let mut grid: Vec<Vec<ArrayD<T>>> = Vec::with_capacity(outer.len());
        for row_obj in outer.iter() {
            let row_list = row_obj.cast::<PyList>()?;
            grid.push(collect_typed::<T>(py, row_list.as_any(), dt.as_str())?);
        }
        let r: ArrayD<T> = fm::block(&grid).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
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
    let first = as_ndarray(py, &arrays.get_item(0)?)?;
    let dt = dtype_name(&first)?;
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
    let first = as_ndarray(py, &arrays.get_item(0)?)?;
    let dt = dtype_name(&first)?;
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
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let typed: Vec<ArrayD<T>> = collect_typed::<T>(py, arrays, dt.as_str())?;
        let r: ArrayD<T> = fm::dstack(&typed).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
