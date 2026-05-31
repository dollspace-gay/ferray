//! Bindings for the `numpy` index-construction and gather surface.
//!
//! These functions return integer index arrays (or tuples of them) for
//! use in fancy indexing, plus the gather/scatter family (`take`,
//! `take_along_axis`, `compress`, `put`, `putmask`, …). Output dtypes are
//! pinned to match NumPy's conventions: `int64` for `argwhere`, `int64`
//! for `indices`, and `int64` for the tuple components returned by
//! `tril_indices` / `triu_indices` (NumPy uses platform-int — int64 on
//! 64-bit platforms, which is the only target we care about today).
//!
//! ## REQ status
//!
//! This module is the PyO3 marshalling shim over `ferray-core`'s
//! `indexing::extended` (design `.design/ferray-core.md`, REQ-15a
//! "Extended indexing functions"). The library REQ is SHIPPED; this file
//! owns only the boundary contract (R-DEV-2/R-DEV-3):
//!
//! - REQ-15a SHIPPED (binding) — `take` / `take_along_axis` / `compress` /
//!   `indices` / `nonzero` / `put` / `putmask` / `choose` / `select` /
//!   `place` / `extract` are bound here and registered on `_ferray`
//!   (`crate::lib::_ferray`) + re-exported at `ferray.*`
//!   (`python/ferray/__init__.py`). The boundary marshalling enforced
//!   here: `take`/`compress` default `axis=None` ⇒ flattened source
//!   (numpy/_core/fromnumeric.py:135-137, :2155-2156); `take` `mode`
//!   ∈ {raise,wrap,clip} index arithmetic (:142-151); N-d index-shape
//!   preservation (:194-198) and scalar-index ⇒ 0-d result (:132-134);
//!   `take_along_axis` element-wise N-d gather
//!   (numpy/lib/_shape_base_impl.py); IndexError vs AxisError vs ValueError
//!   typing; `indices` `dtype=`/`sparse=` kwargs
//!   (numpy/_core/numeric.py:1726); `nonzero` 0-d ⇒ ValueError (:1994);
//!   `putmask` global-flat value cycling; top-level `put`
//!   (fromnumeric.py:489).
//! - REQ-15a-CPLX SHIPPED (#938) — `take` and `nonzero` now accept complex
//!   input via the `match_dtype_all_complex!` + `DynMarshal` seam (#933):
//!   `take` is a pure gather (`fi::take` over `Complex<T>`, dtype preserved),
//!   `nonzero` treats `z != 0` as `re != 0 || im != 0` (`fi::nonzero` uses
//!   `*v != T::zero()`, which compares both parts for `Complex<T>`). Consumer:
//!   the `take`/`nonzero` `#[pyfunction]`s registered on `_ferray`. numpy:
//!   `np.take([3+4j,...],[0,2])`, `np.nonzero([0j,1+1j,2+0j])` (live 2.4.5).
//!   Pinned green: `tests/test_divergence_complex_converge_audit.py::test_D_take`
//!   / `::test_D_nonzero`; `tests/test_expansion_complex_dclass.py` (take/nonzero).

use ferray_core::FerrayError;
use ferray_core::array::aliases::{Array1, Array2, ArrayD};
use ferray_core::dimension::{Ix1, Ix2, IxDyn};
use ferray_core::indexing::extended as fi;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::conv::{DynMarshal, as_ndarray, dtype_name, ferr_to_pyerr, normalize_axis};
use crate::{match_dtype_all, match_dtype_all_complex};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

/// Map a [`FerrayError`] from a `take`/`put`-style gather onto the CPython
/// exception numpy raises.
///
/// numpy's `take` (mode='raise') surfaces an **`IndexError`** for an
/// out-of-bounds index — `numpy/_core/fromnumeric.py:145` ("'raise' --
/// raise an error") routes through `PyArray_TakeFrom`, which raises
/// `IndexError: index N is out of bounds for size M`. A plain `ValueError`
/// (which [`ferr_to_pyerr`] would produce for `FerrayError::IndexOutOfBounds`)
/// is NOT an `IndexError` in numpy's hierarchy, so `except IndexError`
/// clauses would silently miss it (R-DEV-2). Axis errors keep their
/// `AxisError` typing via the boundary [`normalize_axis`]; everything else
/// falls through to [`ferr_to_pyerr`].
fn gather_err_to_pyerr(e: FerrayError) -> PyErr {
    match e {
        FerrayError::IndexOutOfBounds { .. } => {
            pyo3::exceptions::PyIndexError::new_err(e.to_string())
        }
        other => ferr_to_pyerr(other),
    }
}

/// Apply numpy's `take`/`put` `mode` to a raw (possibly out-of-bounds or
/// negative) index against an axis of length `n`, returning the resolved
/// non-negative index — or `None` when `mode='raise'` should defer the
/// bounds check to the library so it raises numpy's `IndexError`.
///
/// `numpy/_core/fromnumeric.py:142-151`:
///   * 'raise' (default) — out-of-bounds raises; negatives index from the end.
///   * 'wrap'            — `idx % n` (Python-style wrap, handles negatives).
///   * 'clip'            — clamp into `[0, n-1]`; "disables indexing with
///     negative numbers", so a negative clips to `0`.
///
/// For 'raise' we leave negative-from-end normalization and the bounds check
/// to ferray-core (`normalize_index`), only signalling an error via
/// [`gather_err_to_pyerr`]; this helper rewrites the index only for the
/// 'wrap'/'clip' modes the library has no notion of.
fn apply_take_mode(idx: isize, n: usize, mode: &str) -> PyResult<isize> {
    if n == 0 {
        // Mirror numpy: take on a zero-length axis with any index raises.
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "cannot do a non-empty take from an empty axes.",
        ));
    }
    let n_i = n as isize;
    match mode {
        "raise" => Ok(idx),
        "wrap" => Ok(idx.rem_euclid(n_i)),
        "clip" => Ok(idx.clamp(0, n_i - 1)),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "clipmode not understood: {other:?} (expected 'raise', 'wrap' or 'clip')"
        ))),
    }
}

/// Extract a `take`/`put` index argument that may be a Python scalar int, a
/// flat sequence, or an N-d nested sequence, returning the flattened
/// row-major `Vec<isize>` plus the index *shape* (`None` ⇒ a scalar int,
/// whose result numpy collapses to a 0-d scalar).
///
/// numpy (`fromnumeric.py:132-134` "Also allow scalars for indices";
/// `:194-198` "If `indices` is not one dimensional, the output also has
/// these dimensions.") preserves the index array's shape in the output. The
/// ferray library `take` only accepts a flat `&[isize]`, so the binding
/// flattens here and re-applies the shape to the result at the boundary.
fn extract_index_arg(
    py: Python<'_>,
    indices: &Bound<'_, PyAny>,
) -> PyResult<(Vec<isize>, Option<Vec<usize>>)> {
    // Scalar int (but NOT a 0-d/Nd ndarray, which extracts via asarray below).
    if indices.is_instance_of::<pyo3::types::PyInt>() {
        let v: isize = indices.extract()?;
        return Ok((vec![v], None));
    }
    // Route everything else (list, tuple, ndarray, numpy scalar) through
    // numpy.asarray so a nested/N-d index keeps its shape.
    let arr = as_ndarray(py, indices)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    if shape.is_empty() {
        // 0-d index array — numpy treats it like a scalar (0-d result).
        let v: isize = arr.call_method0("item")?.extract()?;
        return Ok((vec![v], None));
    }
    let flat = arr.call_method0("ravel")?;
    let data: Vec<isize> = flat.call_method0("tolist")?.extract()?;
    Ok((data, Some(shape)))
}

/// Reshape a 1-D numpy result so the gathered axis carries the original
/// index shape, via numpy's own `reshape` (the result already lives as a
/// `numpy.ndarray` at the boundary). A `None` index shape means the index
/// was a scalar ⇒ collapse the axis (numpy returns a 0-d scalar / reduced
/// array). `pre`/`post` are the source dimensions on either side of the
/// gathered `axis`.
fn reshape_take_result<'py>(
    result: Bound<'py, PyAny>,
    pre: &[usize],
    index_shape: &Option<Vec<usize>>,
    post: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    let mut target: Vec<usize> = Vec::new();
    target.extend_from_slice(pre);
    if let Some(ish) = index_shape {
        target.extend_from_slice(ish);
    }
    target.extend_from_slice(post);
    let reshaped = result.call_method1("reshape", (target,))?;
    if index_shape.is_none() && pre.is_empty() && post.is_empty() {
        // Scalar index on a 1-D source ⇒ numpy returns a 0-d scalar.
        let empty = PyTuple::empty(reshaped.py());
        return reshaped.get_item(empty);
    }
    Ok(reshaped)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn usize_vec_to_numpy<'py>(py: Python<'py>, v: Vec<usize>) -> PyResult<Bound<'py, PyAny>> {
    // Cast through i64 (NumPy's default index dtype on 64-bit platforms).
    let data: Vec<i64> = v.iter().map(|&x| x as i64).collect();
    let arr = Array1::<i64>::from_vec(Ix1::new([data.len()]), data).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// indices / diag_indices / tril_indices / triu_indices / mask_indices
// ---------------------------------------------------------------------------

/// `numpy.indices(dimensions, dtype=int, sparse=False)` — a grid of
/// coordinate index arrays.
///
/// With `sparse=False` (default) returns a single ndarray of shape
/// `(ndim, *dimensions)` whose `[d, ...]` slice is the coordinate array
/// for axis `d`. With `sparse=True` returns a *tuple* of `ndim` open grids,
/// `grid[i]` shaped with `1`s on every axis but its own
/// (`numpy/_core/numeric.py:1726` `def indices(dimensions, dtype=int,
/// sparse=False)`; doc :1749-1752). `dtype` controls the output element
/// type (default platform int = `int64`), mirroring numpy's `dtype=int`.
#[pyfunction]
#[pyo3(signature = (dimensions, dtype = None, sparse = false))]
pub fn indices<'py>(
    py: Python<'py>,
    dimensions: Vec<usize>,
    dtype: Option<&Bound<'py, PyAny>>,
    sparse: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let per_axis = fi::indices(&dimensions).map_err(ferr_to_pyerr)?;
    let ndim = per_axis.len();
    let total: usize = dimensions.iter().product();

    if sparse {
        // numpy: each grid[i] has dimensions[i] along axis i and 1 elsewhere.
        // Build axis i's 1-D ramp [0, 1, ..., dim_i-1] then reshape to the
        // open-grid shape; coerce to `dtype` if requested.
        let mut grids: Vec<Bound<'py, PyAny>> = Vec::with_capacity(ndim);
        for (i, dim_i) in dimensions.iter().copied().enumerate() {
            let ramp: Vec<i64> = (0..dim_i as i64).collect();
            let arr = Array1::<i64>::from_vec(Ix1::new([dim_i]), ramp).map_err(ferr_to_pyerr)?;
            let obj = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            let mut shape = vec![1usize; ndim];
            shape[i] = dim_i;
            let mut g = obj.call_method1("reshape", (shape,))?;
            if let Some(dt) = dtype {
                g = g.call_method1("astype", (dt,))?;
            }
            grids.push(g);
        }
        return Ok(PyTuple::new(py, grids)?.into_any());
    }

    // Dense: stack into shape (ndim, *dimensions).
    let mut data: Vec<i64> = Vec::with_capacity(ndim * total);
    for ax in &per_axis {
        for v in ax.iter() {
            data.push(*v as i64);
        }
    }
    let mut full_shape: Vec<usize> = vec![ndim];
    full_shape.extend_from_slice(&dimensions);
    let arr = ArrayD::<i64>::from_vec(IxDyn::new(&full_shape), data).map_err(ferr_to_pyerr)?;
    let obj = arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    match dtype {
        Some(dt) => Ok(obj.call_method1("astype", (dt,))?),
        None => Ok(obj),
    }
}

/// `numpy.diag_indices(n, ndim=2)` — tuple of `ndim` index arrays, each
/// `[0, 1, …, n-1]`, that select the main diagonal of an n^ndim cube.
#[pyfunction]
#[pyo3(signature = (n, ndim = 2))]
pub fn diag_indices<'py>(py: Python<'py>, n: usize, ndim: usize) -> PyResult<Bound<'py, PyAny>> {
    let groups = fi::diag_indices(n, ndim);
    let arrays: Vec<Bound<'py, PyAny>> = groups
        .into_iter()
        .map(|g| usize_vec_to_numpy(py, g))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, arrays)?.into_any())
}

/// `numpy.tril_indices(n, k=0, m=None)` — `(rows, cols)` index pair
/// selecting the lower triangle of an `(n, m)` matrix.
#[pyfunction]
#[pyo3(signature = (n, k = 0, m = None))]
pub fn tril_indices<'py>(
    py: Python<'py>,
    n: usize,
    k: isize,
    m: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = fi::tril_indices(n, k, m);
    let r = usize_vec_to_numpy(py, rows)?;
    let c = usize_vec_to_numpy(py, cols)?;
    Ok(PyTuple::new(py, [r, c])?.into_any())
}

/// `numpy.triu_indices(n, k=0, m=None)` — `(rows, cols)` index pair
/// selecting the upper triangle of an `(n, m)` matrix.
#[pyfunction]
#[pyo3(signature = (n, k = 0, m = None))]
pub fn triu_indices<'py>(
    py: Python<'py>,
    n: usize,
    k: isize,
    m: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let (rows, cols) = fi::triu_indices(n, k, m);
    let r = usize_vec_to_numpy(py, rows)?;
    let c = usize_vec_to_numpy(py, cols)?;
    Ok(PyTuple::new(py, [r, c])?.into_any())
}

/// `numpy.mask_indices(n, mask_func, k=0)` — flattened indices of the
/// elements selected by a triangular mask. ferray exposes the three
/// canonical mask kinds (`tril`, `triu`, `mask_indices`-style with
/// explicit kind).
#[pyfunction]
#[pyo3(signature = (n, kind, k = 0))]
pub fn mask_indices<'py>(
    py: Python<'py>,
    n: usize,
    kind: &str,
    k: isize,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::indexing::extended::MaskKind;
    let mk = match kind {
        "tril" => MaskKind::Tril,
        "triu" => MaskKind::Triu,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "mask kind must be 'tril' or 'triu', got {other:?}"
            )));
        }
    };
    let v = fi::mask_indices(n, mk, k);
    usize_vec_to_numpy(py, v)
}

// ---------------------------------------------------------------------------
// ravel_multi_index / unravel_index
// ---------------------------------------------------------------------------

/// `numpy.ravel_multi_index(multi_index, dims, mode='raise', order='C')` —
/// convert per-axis coordinate arrays into flat indices.
///
/// `order` ∈ {'C','F'} selects the raveling order: 'C' (row-major, default)
/// vs 'F' (column-major), which changes the per-axis stride weighting.
/// ferray-core's `ravel_multi_index` is row-major only, so the binding
/// delegates the 'F' (column-major) case to numpy (which owns C/F raveling)
/// and keeps the native path for the default 'C' order.
#[pyfunction]
#[pyo3(signature = (multi_index, dims, order = "C"))]
pub fn ravel_multi_index<'py>(
    py: Python<'py>,
    multi_index: &Bound<'py, PyAny>,
    dims: Vec<usize>,
    order: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // 'F' (column-major) order changes the stride weighting; numpy owns the
    // C/F raveling arithmetic. The native path below is row-major ('C') only.
    if order != "C" {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", order)?;
        return np.call_method("ravel_multi_index", (multi_index, dims), Some(&kwargs));
    }
    let multi_index: Vec<Vec<usize>> = multi_index.extract()?;
    let refs: Vec<&[usize]> = multi_index.iter().map(|v| v.as_slice()).collect();
    let flat = fi::ravel_multi_index(&refs, &dims).map_err(ferr_to_pyerr)?;
    usize_vec_to_numpy(py, flat)
}

/// `numpy.unravel_index(indices, shape)` — convert flat indices into a
/// tuple of per-axis coordinate arrays.
///
/// numpy accepts a SCALAR int `indices` and returns a tuple of scalar
/// coordinates (`np.unravel_index(3, (2,2)) -> (1, 1)`), not a tuple of
/// 1-element arrays. ferray-core's `unravel_index` takes a slice and yields
/// per-axis arrays, so the binding routes a scalar int through numpy (which
/// owns the scalar-tuple return form) and keeps the native path for a
/// sequence of indices.
#[pyfunction]
pub fn unravel_index<'py>(
    py: Python<'py>,
    indices: &Bound<'py, PyAny>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    // Scalar int ⇒ numpy returns a tuple of scalar coordinates (not arrays);
    // delegate so the scalar-tuple shape/types match numpy exactly.
    if indices.is_instance_of::<pyo3::types::PyInt>() {
        let np = py.import("numpy")?;
        return np.call_method1("unravel_index", (indices, shape));
    }
    let indices: Vec<usize> = indices.extract()?;
    let groups = fi::unravel_index(&indices, &shape).map_err(ferr_to_pyerr)?;
    let arrays: Vec<Bound<'py, PyAny>> = groups
        .into_iter()
        .map(|g| usize_vec_to_numpy(py, g))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, arrays)?.into_any())
}

// ---------------------------------------------------------------------------
// flatnonzero / nonzero / argwhere
// ---------------------------------------------------------------------------

/// `numpy.flatnonzero(a)` — flat indices of non-zero elements.
#[pyfunction]
pub fn flatnonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // string `<U`/`<S` (REQ-2, #960): int index output (`''` is falsy); delegate
    // to numpy directly (no transport — strings never enter the Rust library).
    if crate::manipulation::is_flexible_array(&arr)? {
        return crate::manipulation::string_delegate(py, "flatnonzero", (&arr,), None);
    }
    // float16 (REQ-5, #955): int index output; delegate to numpy as
    // `match_dtype_all!` has no float16 arm.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "flatnonzero", (&arr,), None);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let v = fi::flatnonzero(&fa);
        usize_vec_to_numpy(py, v)?
    }))
}

/// `numpy.nonzero(a)` — tuple of per-axis index arrays.
#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    // numpy/_core/fromnumeric.py:1994 nonzero — "Calling nonzero on 0d arrays
    // is not allowed. Use np.atleast_1d(scalar).nonzero() instead." numpy
    // raises ValueError for a 0-d input; ferray-core's `nonzero` would instead
    // return an empty tuple, so guard at the boundary (R-DEV-2).
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if ndim == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead. If the context of this error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.",
        ));
    }
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 nonzero (#947): coordinates of ticks != 0 (NaT is
    // nonzero). Routed through the #947 time dispatch ahead of the real-only
    // `match_dtype_all_complex!`, which would otherwise raise `TypeError`.
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::nonzero_time(py, &arr);
    }
    // string `<U`/`<S` (REQ-2, #960): tuple of int index arrays of non-empty
    // strings (`''` is falsy); delegate to numpy directly. numpy returns a tuple,
    // which rides the boundary unchanged.
    if crate::manipulation::is_flexible_array(&arr)? {
        return crate::manipulation::string_delegate(py, "nonzero", (&arr,), None);
    }
    // float16 (REQ-5, #955): tuple of int index arrays; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "nonzero", (&arr,), None);
    }
    // Complex `nonzero`: numpy treats `z != 0` as `re != 0 || im != 0`
    // (numpy/_core/fromnumeric.py:1994 dispatches to the elementwise truth of
    // the complex value). ferray-core `nonzero` uses `*val != T::zero()`, which
    // for `Complex<T>` compares both parts — matching numpy. Route complex
    // through the DynMarshal seam (#933) so the imaginary part is preserved on
    // extract.
    let groups = match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        fi::nonzero(&fa)
    });
    let arrays: Vec<Bound<'py, PyAny>> = groups
        .into_iter()
        .map(|g| usize_vec_to_numpy(py, g))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, arrays)?.into_any())
}

// ---------------------------------------------------------------------------
// take / take_along_axis / choose / compress / select / place / putmask / extract
// ---------------------------------------------------------------------------

/// `numpy.take(a, indices, axis=None, out=None, mode='raise')` — gather
/// along an axis (or the flattened array when `axis is None`).
///
/// Mirrors `numpy/_core/fromnumeric.py:107` `def take(a, indices, axis=None,
/// out=None, mode='raise')`:
///   * `axis=None` (default) operates on the **flattened** array (:135-137).
///   * `indices` may be a scalar (0-d result, :132-134) or N-d (the index
///     shape is preserved in the output, :194-198).
///   * `mode` ∈ {'raise','wrap','clip'} resolves out-of-bounds indices
///     (:142-151); 'raise' surfaces an `IndexError`.
/// An out-of-range `axis` raises `numpy.exceptions.AxisError` (R-DEV-2).
#[pyfunction]
#[pyo3(signature = (a, indices, axis = None, mode = "raise"))]
pub fn take<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    axis: Option<isize>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::Axis;
    let mut arr = as_ndarray(py, a)?;
    // string `<U`/`<S` (REQ-2, #960): `take` is a pure gather preserving the
    // string dtype (`np.take(<U,...).dtype == <U`, live); delegate to numpy
    // directly (no transport — strings never enter the Rust library).
    if crate::manipulation::is_flexible_array(&arr)? {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        kwargs.set_item("mode", mode)?;
        return crate::manipulation::string_delegate(py, "take", (&arr, indices), Some(&kwargs));
    }
    // float16 (REQ-5, #955): `take` is a pure gather preserving the dtype
    // (`np.take(f16,...).dtype == float16`, live); delegate to numpy as
    // `match_dtype_all_complex!` has no float16 arm.
    if crate::conv::is_float16_dtype(dtype_name(&arr)?.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        kwargs.set_item("mode", mode)?;
        return crate::conv::f16_delegate(py, "take", (&arr, indices), Some(&kwargs));
    }
    let (flat_idx, index_shape) = extract_index_arg(py, indices)?;

    // axis=None ⇒ flatten the source and gather along axis 0.
    let ax: usize = match axis {
        None => {
            arr = arr.call_method0("ravel")?;
            0
        }
        Some(raw) => {
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            normalize_axis(py, raw, ndim)?
        }
    };

    let src_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let axis_len = src_shape[ax];
    let resolved: Vec<isize> = flat_idx
        .iter()
        .map(|&i| apply_take_mode(i, axis_len, mode))
        .collect::<PyResult<Vec<_>>>()?;

    let pre: Vec<usize> = src_shape[..ax].to_vec();
    let post: Vec<usize> = src_shape[ax + 1..].to_vec();

    let dt = dtype_name(&arr)?;
    // `take` is a pure gather (no arithmetic) — generic over `T: Element` in
    // ferray-core, so it works unchanged for `Complex<T>`. Route through the
    // DynMarshal seam (#933) so complex input preserves its imaginary part
    // (numpy `np.take` keeps complex dtype, verified live).
    let result = match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r: ArrayD<T> = fi::take(&fa, &resolved, Axis(ax)).map_err(gather_err_to_pyerr)?;
        T::emit_dyn(py, r)?
    });
    reshape_take_result(result, &pre, &index_shape, &post)
}

/// Row-major strides for `shape` (number of flat elements to skip per
/// unit step along each axis), used by the element-wise gather below.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// `numpy.take_along_axis(a, indices, axis)` — element-wise gather where
/// `indices` has the same ndim as `a` and is matched coordinate-by-coordinate
/// along `axis`.
///
/// `numpy/lib/_shape_base_impl.py` `take_along_axis`: for an output position
/// `(.., j, ..)` the value is `a[.., indices[.., j, ..], ..]` — i.e. the
/// index array selects per-element along `axis`, NOT a flat slice. ferray's
/// library `take_along_axis` only does a flat `index_select`, so the
/// element-wise gather is performed here at the boundary (the index array's
/// shape equals `a`'s shape except along `axis`, where it gives the output
/// length). An out-of-range `axis` raises `AxisError`; an out-of-bounds
/// index raises `IndexError` (R-DEV-2).
#[pyfunction]
pub fn take_along_axis<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let idx_arr = as_ndarray(py, indices)?;

    let a_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    let i_shape: Vec<usize> = idx_arr.getattr("shape")?.extract()?;
    let ndim = a_shape.len();
    if i_shape.len() != ndim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "`indices` and `a` must have the same number of dimensions ({} vs {})",
            i_shape.len(),
            ndim
        )));
    }
    let ax = normalize_axis(py, axis, ndim)?;

    // Output shape: a_shape with the gathered axis replaced by i_shape[ax].
    let mut out_shape = a_shape.clone();
    out_shape[ax] = i_shape[ax];
    let out_total: usize = out_shape.iter().product();

    let a_strides = row_major_strides(&a_shape);
    let i_strides = row_major_strides(&i_shape);

    // Flat (row-major) index values.
    let idx_flat: Vec<isize> = idx_arr
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;

    let axis_len = a_shape[ax] as isize;

    let dt = dtype_name(&arr)?;
    // string `<U`/`<S` (REQ-2, #960): pure gather preserving the string dtype;
    // delegate to numpy directly.
    if crate::manipulation::is_flexible_array(&arr)? {
        return crate::manipulation::string_delegate(
            py,
            "take_along_axis",
            (&arr, &idx_arr, axis),
            None,
        );
    }
    // float16 (REQ-5, #955): pure gather preserving the dtype; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "take_along_axis", (&arr, &idx_arr, axis), None);
    }
    let result = match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let src: Vec<T> = fa.iter().cloned().collect();
        let mut out: Vec<T> = Vec::with_capacity(out_total);
        // Walk every output multi-index in row-major order.
        let mut coord = vec![0usize; ndim];
        for _ in 0..out_total {
            // Flat offset into the (same-shape-as-out) index array.
            let mut i_off = 0usize;
            for d in 0..ndim {
                // index array broadcasts size-1 axes; here it matches out_shape
                // except potentially size-1, so clamp the coord.
                let c = if i_shape[d] == 1 { 0 } else { coord[d] };
                i_off += c * i_strides[d];
            }
            let raw = idx_flat[i_off];
            let norm = if raw < 0 { raw + axis_len } else { raw };
            if norm < 0 || norm >= axis_len {
                return Err(gather_err_to_pyerr(FerrayError::index_out_of_bounds(
                    raw, ax, a_shape[ax],
                )));
            }
            // Source flat offset: out coords, but axis coord = gathered index.
            let mut a_off = 0usize;
            for d in 0..ndim {
                let c = if d == ax { norm as usize } else { coord[d] };
                a_off += c * a_strides[d];
            }
            out.push(src[a_off]);
            // Increment row-major counter.
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < out_shape[d] { break; }
                coord[d] = 0;
            }
        }
        let r = ArrayD::<T>::from_vec(IxDyn::new(&out_shape), out).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    });
    Ok(result)
}

/// `numpy.choose(a, choices, out=None, mode='raise')` — for each index in
/// `a`, pick from `choices`. `a` must be u64; choices is a list of
/// same-shape arrays.
///
/// `numpy/_core/fromnumeric.py` `def choose(a, choices, out=None,
/// mode='raise')`: `mode` ∈ {'raise','wrap','clip'} resolves out-of-bounds
/// index entries ('raise' raises, 'wrap' folds `idx % n`, 'clip' clamps).
/// ferray-core's `choose` only implements the default 'raise' semantics, so
/// the binding delegates the 'wrap'/'clip' index-folding to numpy (which owns
/// it) and keeps the native path for the default 'raise' mode.
#[pyfunction]
#[pyo3(signature = (a, choices, mode = "raise"))]
pub fn choose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // 'wrap'/'clip' fold out-of-bounds index entries into range; numpy owns
    // that index arithmetic. The native path below implements only 'raise'.
    if mode != "raise" {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("mode", mode)?;
        return np.call_method("choose", (a, choices), Some(&kwargs));
    }
    let idx_arr = crate::conv::coerce_dtype(py, a, "uint64")?;
    let idx_view: PyReadonlyArrayDyn<u64> = idx_arr.extract()?;
    let idx_fa: ArrayD<u64> = idx_view.as_ferray().map_err(ferr_to_pyerr)?;

    let list = choices.cast::<pyo3::types::PyList>()?;
    if list.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "choose: empty choices",
        ));
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;

    // float16 (REQ-5, #955): choices stay float16; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "choose", (a, choices), None);
    }
    // complex (#972): `choose` is a pure gather (`numpy/_core/fromnumeric.py:choose`
    // → `_wrapfunc(a, 'choose', ...)`) and numpy preserves the complex dtype of the
    // choices. The `match_dtype_all!` real path is sealed to real dtypes and would
    // raise `TypeError` on complex, while coercing the choices to float64 would drop
    // every imaginary part (R-CODE-4). Delegate the complex case to numpy, which
    // owns the complex result.
    let first_kind: String = first.getattr("dtype")?.getattr("kind")?.extract()?;
    if first_kind == "c" {
        let np = py.import("numpy")?;
        return np.call_method1("choose", (a, choices));
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let mut owned: Vec<ArrayD<T>> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let coerced = crate::conv::coerce_dtype(py, &item, dt.as_str())?;
            let view: PyReadonlyArrayDyn<T> = coerced.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let r: ArrayD<T> = fi::choose(&idx_fa, &owned).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.compress(condition, a, axis=None)` — select slices where
/// `condition` is true along `axis`, or elements of the flattened array
/// when `axis is None`.
///
/// `numpy/_core/fromnumeric.py:2138` `def compress(condition, a, axis=None,
/// out=None)`; :2155-2156 "axis : int, optional ... If None (default), work
/// on the flattened array." ferray-core's `compress` has no flatten path, so
/// the binding ravels the source when `axis is None` (R-DEV-2/R-DEV-3). An
/// out-of-range `axis` raises `AxisError`.
#[pyfunction]
#[pyo3(signature = (condition, a, axis = None))]
pub fn compress<'py>(
    py: Python<'py>,
    condition: Vec<bool>,
    a: &Bound<'py, PyAny>,
    axis: Option<isize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::Axis;
    let mut arr = as_ndarray(py, a)?;
    let ax: usize = match axis {
        None => {
            arr = arr.call_method0("ravel")?;
            0
        }
        Some(raw) => {
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            normalize_axis(py, raw, ndim)?
        }
    };
    let dt = dtype_name(&arr)?;
    // string `<U`/`<S` (REQ-2, #960): compress preserves the string dtype;
    // delegate to numpy directly. (`arr` may have been raveled above; pass it so
    // numpy compresses the same flattened/axis form.)
    if crate::manipulation::is_flexible_array(&arr)? {
        return crate::manipulation::string_delegate(py, "compress", (condition, &arr), None);
    }
    // float16 (REQ-5, #955): compress preserves the dtype; delegate to numpy.
    // (`arr` may have been raveled above; pass it directly so numpy compresses
    // the same flattened/axis form.)
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "compress", (condition, &arr), None);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fi::compress(&condition, &fa, Axis(ax)).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.select(condlist, choicelist, default=0)` — multi-condition
/// elementwise selection.
///
/// `numpy/lib/_function_base_impl.py:813` `def select(condlist, choicelist,
/// default=0)`: where no condition matches, the `default` value fills the
/// output using the **choice** dtype. A previous binding typed `default: f64`
/// and round-tripped it through f64, so an int64 default above 2**53 lost its
/// low bits (#1020). Accepting `default` as a raw object and delegating to
/// numpy lets the default cast to the choice dtype exactly (R-CODE-4).
#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default = None))]
pub fn select<'py>(
    py: Python<'py>,
    condlist: &Bound<'py, PyAny>,
    choicelist: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond_list = condlist.cast::<pyo3::types::PyList>()?;
    let choice_list = choicelist.cast::<pyo3::types::PyList>()?;
    if cond_list.is_empty() || choice_list.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "select: empty condlist/choicelist",
        ));
    }
    if cond_list.len() != choice_list.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "select: condlist and choicelist must have same length",
        ));
    }

    // Delegate to numpy so the `default` casts to the choice dtype exactly
    // (int64 default above 2**53 stays lossless; complex defaults preserved).
    // numpy's `select` default is the integer `0` when omitted.
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(def) = default {
        kwargs.set_item("default", def)?;
    } else {
        kwargs.set_item("default", 0i64)?;
    }
    np.call_method("select", (condlist, choicelist), Some(&kwargs))
}

/// `numpy.place(arr, mask, vals)` — non-mutating: returns a NEW array
/// with `vals` placed at positions where `mask` is true. (NumPy mutates
/// in place; PyReadonlyArray prevents that, so we return a copy. Use
/// `arr = ferray.place(arr, mask, vals)` for in-place semantics.)
#[pyfunction]
pub fn place<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    vals: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Delegate to numpy on a fresh copy so `vals` is placed with the target
    // array's OWN dtype (int64 above 2**53 stays lossless, #1019; complex
    // preserved). numpy `place` mutates in place and returns None; we return
    // the mutated copy, preserving ferray's copy-returning contract.
    let arr_a = as_ndarray(py, arr)?;
    let copy = arr_a.call_method0("copy")?;
    let np = py.import("numpy")?;
    np.call_method1("place", (&copy, mask, vals))?;
    Ok(copy)
}

/// `numpy.putmask(a, mask, values)` — set `a.flat[n] = values[n %
/// len(values)]` for each global flat position `n` where `mask.flat[n]` is
/// true; non-mutating (returns a copy, same caveat as `place`).
///
/// numpy cycles `values` by the **global** flat index `n` (not by the
/// mask-relative count): `np.putmask(arange(5), x>1, [-33,-44])` ⇒
/// `[0,1,-33,-44,-33]` (positions 2,3,4 take `values[2%2],values[3%2],
/// values[4%2]`). ferray-core's `putmask` only accepts a length-1 or
/// length-`size` value slice, so the binding pre-expands `values` to the
/// full array size by global flat position — yielding numpy's cycling — and
/// then defers to the library (R-DEV-1: the library `putmask` is the
/// production consumer of the expanded vector).
#[pyfunction]
pub fn putmask<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    mask: &Bound<'py, PyAny>,
    values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Delegate to numpy on a fresh copy so `values` are written with the target
    // array's OWN dtype (int64 above 2**53 stays lossless, #1018; complex
    // preserved) and cycled by global flat index per numpy's contract. numpy
    // `putmask` mutates in place and returns None; we return the mutated copy,
    // preserving ferray's copy-returning contract.
    let arr_a = as_ndarray(py, a)?;
    let copy = arr_a.call_method0("copy")?;
    let np = py.import("numpy")?;
    np.call_method1("putmask", (&copy, mask, values))?;
    Ok(copy)
}

/// `numpy.put(a, ind, v, mode='raise')` — set `a.flat[ind] = v` (cycling
/// `v` when shorter than `ind`); non-mutating in this binding (returns the
/// modified copy, mirroring the `place`/`putmask` caveat — `PyReadonlyArray`
/// forbids in-place mutation, so use `a = ferray.put(a, ind, v)`).
///
/// `numpy/_core/fromnumeric.py:489` `def put(a, ind, v, mode='raise')`,
/// doc :492 "a.flat[ind] = v"; `v` "repeated as necessary" when shorter than
/// `ind` (:508-510). ferray-core's `Array::put` already cycles values by the
/// i-th index position and normalizes negative indices, raising
/// `IndexOutOfBounds` for an out-of-range index ⇒ numpy's `IndexError`
/// (R-DEV-2). `mode` resolves out-of-bounds indices per :511-520.
#[pyfunction]
#[pyo3(signature = (a, ind, v, mode = "raise"))]
pub fn put<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    ind: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // Delegate to numpy on a fresh copy so `v` is written with the target
    // array's OWN dtype: an int64 value above 2**53 stays lossless (#1017) and
    // complex values are accepted (#1021) instead of being rejected by an f64
    // value param. numpy `put` (`numpy/_core/fromnumeric.py:489`,
    // `a.flat[ind] = v`) mutates in place and returns None; we return the
    // mutated copy, preserving ferray's copy-returning contract. `mode`
    // resolves out-of-bounds indices (raise/wrap/clip) and is threaded through.
    let arr_a = as_ndarray(py, a)?;
    let copy = arr_a.call_method0("copy")?;
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("mode", mode)?;
    np.call_method("put", (&copy, ind, v), Some(&kwargs))?;
    Ok(copy)
}

/// `numpy.extract(condition, arr)` — return a 1-D array of elements
/// from `arr` where `condition` is true.
#[pyfunction]
pub fn extract<'py>(
    py: Python<'py>,
    condition: &Bound<'py, PyAny>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let cond_arr = crate::conv::coerce_dtype(py, condition, "bool")?;
    let cond_view: PyReadonlyArrayDyn<bool> = cond_arr.extract()?;
    let cond_fa: ArrayD<bool> = cond_view.as_ferray().map_err(ferr_to_pyerr)?;
    let arr_a = as_ndarray(py, arr)?;
    let dt = dtype_name(&arr_a)?;
    // string `<U`/`<S` (REQ-2, #960): extract returns a 1-D string array of the
    // selected elements, dtype preserved; delegate to numpy directly.
    if crate::manipulation::is_flexible_array(&arr_a)? {
        return crate::manipulation::string_delegate(py, "extract", (condition, arr), None);
    }
    // float16 (REQ-5, #955): extract preserves the dtype; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "extract", (condition, arr), None);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = fi::extract(&cond_fa, &fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.argwhere(a)` — `(N, ndim)` int64 array of multi-indices of
/// non-zero elements.
#[pyfunction]
pub fn argwhere<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // string `<U`/`<S` (REQ-2, #960): int64 multi-index output of non-empty
    // strings; delegate to numpy directly.
    if crate::manipulation::is_flexible_array(&arr)? {
        return crate::manipulation::string_delegate(py, "argwhere", (&arr,), None);
    }
    // float16 (REQ-5, #955): int64 multi-index output; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        return crate::conv::f16_delegate(py, "argwhere", (&arr,), None);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<i64> = fi::argwhere(&fa).map_err(ferr_to_pyerr)?;
        // Convert to ArrayD so we can use the same IntoNumPy path
        // everything else takes; Array2<i64> already has IntoNumPy<.., Ix2>
        // but the macro arm returns Bound<'py, PyAny>, so use that.
        let _ = (Ix1::new([0]), Ix2::new([0, 0]), IxDyn::new(&[]));
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
