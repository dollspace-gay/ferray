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

/// `numpy.ravel_multi_index(multi_index, dims)` — convert per-axis
/// coordinate arrays into flat indices.
#[pyfunction]
pub fn ravel_multi_index<'py>(
    py: Python<'py>,
    multi_index: Vec<Vec<usize>>,
    dims: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let refs: Vec<&[usize]> = multi_index.iter().map(|v| v.as_slice()).collect();
    let flat = fi::ravel_multi_index(&refs, &dims).map_err(ferr_to_pyerr)?;
    usize_vec_to_numpy(py, flat)
}

/// `numpy.unravel_index(indices, shape)` — convert flat indices into a
/// tuple of per-axis coordinate arrays.
#[pyfunction]
pub fn unravel_index<'py>(
    py: Python<'py>,
    indices: Vec<usize>,
    shape: Vec<usize>,
) -> PyResult<Bound<'py, PyAny>> {
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

/// `numpy.choose(a, choices)` — for each index in `a`, pick from
/// `choices`. `a` must be u64; choices is a list of same-shape arrays.
#[pyfunction]
pub fn choose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    choices: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
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
#[pyfunction]
#[pyo3(signature = (condlist, choicelist, default = 0.0))]
pub fn select<'py>(
    py: Python<'py>,
    condlist: &Bound<'py, PyAny>,
    choicelist: &Bound<'py, PyAny>,
    default: f64,
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

    // Collect the bool conditions.
    let mut conds: Vec<ArrayD<bool>> = Vec::with_capacity(cond_list.len());
    for c in cond_list.iter() {
        let coerced = crate::conv::coerce_dtype(py, &c, "bool")?;
        let view: PyReadonlyArrayDyn<bool> = coerced.extract()?;
        conds.push(view.as_ferray().map_err(ferr_to_pyerr)?);
    }

    // Sniff dtype from the first choice.
    let first = as_ndarray(py, &choice_list.get_item(0)?)?;
    let dt = dtype_name(&first)?;

    // float16 (REQ-5, #955): choices stay float16; delegate to numpy.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("default", default)?;
        return crate::conv::f16_delegate(py, "select", (condlist, choicelist), Some(&kwargs));
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let mut choices: Vec<ArrayD<T>> = Vec::with_capacity(choice_list.len());
        for ch in choice_list.iter() {
            let coerced = crate::conv::coerce_dtype(py, &ch, dt.as_str())?;
            let view: PyReadonlyArrayDyn<T> = coerced.extract()?;
            choices.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let def: T = default_cast::<T>(default);
        let r: ArrayD<T> = fi::select(&conds, &choices, def).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

trait DefaultCast {
    fn from_f64(v: f64) -> Self;
}
macro_rules! impl_default_cast {
    ($($t:ty),*) => {
        $(impl DefaultCast for $t { fn from_f64(v: f64) -> Self { v as Self } })*
    };
}
impl_default_cast!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl DefaultCast for bool {
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}
fn default_cast<T: DefaultCast>(v: f64) -> T {
    T::from_f64(v)
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
    vals: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, arr)?;
    let dt = dtype_name(&arr_a)?;
    // float16 (REQ-5, #955): ferray's `place` returns the MODIFIED COPY (a
    // documented deviation from numpy's in-place void contract). Preserve that
    // copy-returning contract for float16 by mutating a numpy copy and returning
    // it — `match_dtype_all!` has no float16 arm.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        let copy = arr_a.call_method0("copy")?;
        crate::conv::f16_delegate(py, "place", (&copy, mask, vals), None)?;
        return Ok(copy);
    }
    let mask_arr = crate::conv::coerce_dtype(py, mask, "bool")?;
    let mask_view: PyReadonlyArrayDyn<bool> = mask_arr.extract()?;
    let mask_fa: ArrayD<bool> = mask_view.as_ferray().map_err(ferr_to_pyerr)?;

    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let mut fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let typed_vals: Vec<T> = vals.iter().map(|&v| default_cast::<T>(v)).collect();
        fi::place(&mut fa, &mask_fa, &typed_vals).map_err(ferr_to_pyerr)?;
        fa.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
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
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "putmask: values array is empty",
        ));
    }
    // float16 (REQ-5, #955): mutate a numpy copy and return it (ferray's
    // copy-returning deviation), as `match_dtype_all!` has no float16 arm.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        let copy = arr_a.call_method0("copy")?;
        crate::conv::f16_delegate(py, "putmask", (&copy, mask, values), None)?;
        return Ok(copy);
    }
    let mask_arr = crate::conv::coerce_dtype(py, mask, "bool")?;
    let mask_view: PyReadonlyArrayDyn<bool> = mask_arr.extract()?;
    let mask_fa: ArrayD<bool> = mask_view.as_ferray().map_err(ferr_to_pyerr)?;

    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let mut fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let size = fa.size();
        // Expand to one value per global flat position, cycling by n % len.
        let expanded: Vec<T> = (0..size)
            .map(|n| default_cast::<T>(values[n % values.len()]))
            .collect();
        fi::putmask(&mut fa, &mask_fa, &expanded).map_err(ferr_to_pyerr)?;
        fa.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
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
    ind: Vec<isize>,
    v: Vec<f64>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    if v.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "put: values array is empty",
        ));
    }
    // float16 (REQ-5, #955): mutate a numpy copy and return it (ferray's
    // copy-returning deviation), as `match_dtype_all!` has no float16 arm.
    if crate::conv::is_float16_dtype(dt.as_str()) {
        let copy = arr_a.call_method0("copy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("mode", mode)?;
        crate::conv::f16_delegate(py, "put", (&copy, ind, v), Some(&kwargs))?;
        return Ok(copy);
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let mut fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let size = fa.size();
        // Resolve `mode` for wrap/clip against the flattened length; 'raise'
        // defers OOB to the library so it raises numpy's IndexError.
        let resolved: Vec<isize> = ind
            .iter()
            .map(|&i| apply_take_mode(i, size, mode))
            .collect::<PyResult<Vec<_>>>()?;
        let typed_vals: Vec<T> = v.iter().map(|&x| default_cast::<T>(x)).collect();
        fa.put(&resolved, &typed_vals).map_err(gather_err_to_pyerr)?;
        fa.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
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
