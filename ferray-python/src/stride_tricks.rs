//! Bindings for `numpy.lib.stride_tricks` (and the top-level
//! `numpy.broadcast_arrays` / `numpy.broadcast_shapes` / `numpy.broadcast_to`
//! aliases).
//!
//! Upstream: `numpy/lib/_stride_tricks_impl.py`. This shim mirrors numpy's
//! observable output-object contract (R-DEV-3) and kwarg surface (R-DEV-2):
//!
//! - `broadcast_arrays` returns a **tuple** (`:656 return tuple(result)`),
//!   preserves **each input's own dtype** (`:649` marshals every input with
//!   `np.array(_m, copy=None, subok=subok)` — broadcasting changes only the
//!   shape, `:653-655`), and accepts the `subok` kwarg (`:589`).
//! - `broadcast_to` returns a **read-only** view (`:517 readonly=True`) and
//!   accepts the `subok` kwarg (`:475`).
//! - `sliding_window_view` accepts `axis` (`:180`), positional/duplicate axes
//!   (`:427 normalize_axis_tuple(..., allow_duplicate=True)`), the keyword-only
//!   `subok` / `writeable` (`:181`), defaults to **read-only** (`writeable=False`),
//!   and raises `ValueError` (not `OverflowError`) for a negative window
//!   (`:416-417`).
//! - `as_strided` (`:39`) is exposed on `ferray.lib.stride_tricks.as_strided`.
//!   ferray-core uses element strides while numpy uses byte strides, so the
//!   binding delegates to numpy's already-correct byte-stride implementation
//!   (the output is a `numpy.ndarray`, which is what callers consume) rather
//!   than risk a unit mismatch at the boundary.
//!
//! ## REQ status
//! (REQs from `.design/ferray-stride-tricks.md`; library REQ-5/6/7 live in
//! `ferray-stride-tricks`. This is the PyO3 binding surface.)
//!
//! - REQ-1 `sliding_window_view` — SHIPPED. `fn sliding_window_view` routes to
//!   `ferray_stride_tricks::{sliding_window_view, sliding_window_view_axis}`;
//!   read-only by default, `axis`/`writeable`/`subok` kwargs honored. Consumer:
//!   registered on `ferray.lib.stride_tricks` in `lib.rs`
//!   (`register_stride_tricks_module`).
//! - REQ-2 `broadcast_to` — SHIPPED. `fn broadcast_to` returns a read-only
//!   view + `subok` kwarg. Consumer: top-level `ferray.broadcast_to` in
//!   `lib.rs` (`_ferray`). Complex dtype (#938): `broadcast_to` is a pure
//!   stride/view op (`fst::broadcast_to` over `Complex<T>`), routed through the
//!   `match_dtype_all_complex!` + `DynMarshal` seam (#933) so the read-only
//!   complex view keeps its imaginary part. numpy: `np.broadcast_to(complex,
//!   (2,3))` (live). Pinned green:
//!   `tests/test_divergence_complex_converge_audit.py::test_D_broadcast_to`;
//!   `tests/test_expansion_complex_dclass.py::test_broadcast_to_complex`.
//! - REQ-3 `broadcast_arrays` — SHIPPED. `fn broadcast_arrays` returns a tuple,
//!   per-array dtype preserved. Consumer: top-level `ferray.broadcast_arrays`
//!   and `ferray.lib.stride_tricks.broadcast_arrays` in `lib.rs`.
//! - REQ-4 `broadcast_shapes` — SHIPPED. `fn broadcast_shapes`. Consumer:
//!   top-level `ferray.broadcast_shapes` in `lib.rs`.
//! - REQ-5 `as_strided` (binding) — SHIPPED. `fn as_strided` delegates to numpy.
//!   Consumer: `ferray.lib.stride_tricks.as_strided` in `lib.rs`.

use ferray_core::array::aliases::ArrayD;
use ferray_core::array::view::ArrayView;
use ferray_core::dimension::IxDyn;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use ferray_stride_tricks as fst;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::conv::{
    DynMarshal, as_ndarray, dtype_name, extract_axis_tuple, ferr_to_pyerr, normalize_axis_tuple,
    set_readonly,
};
use crate::{match_dtype_all, match_dtype_all_complex};

/// Materialise an `ArrayView<T, IxDyn>` into an owned `ArrayD<T>` so
/// the buffer can cross the FFI boundary safely. Strided-view callers
/// in NumPy expect a writable copy when the result is consumed by
/// generic code.
fn view_to_owned<T: ferray_core::Element + Clone>(
    view: ArrayView<'_, T, IxDyn>,
) -> Result<ArrayD<T>, ferray_core::FerrayError> {
    let shape = view.shape().to_vec();
    let data: Vec<T> = view.iter().cloned().collect();
    ArrayD::<T>::from_vec(IxDyn::new(&shape), data)
}

/// `numpy.broadcast_shapes(*shapes)` — compute the broadcast shape
/// from a sequence of input shapes.
#[pyfunction]
#[pyo3(signature = (*shapes))]
pub fn broadcast_shapes<'py>(
    py: Python<'py>,
    shapes: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vecs: Vec<Vec<usize>> = shapes
        .iter()
        .map(|s| {
            if let Ok(n) = s.extract::<usize>() {
                Ok(vec![n])
            } else {
                s.extract::<Vec<usize>>()
            }
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&[usize]> = shape_vecs.iter().map(|v| v.as_slice()).collect();
    let result = fst::broadcast_shapes(&refs).map_err(ferr_to_pyerr)?;
    Ok(PyTuple::new(py, result)?.into_any())
}

/// `numpy.broadcast_arrays(*args, subok=False)` — broadcast a sequence of
/// arrays to a common shape and return them as a **tuple** of ndarrays,
/// each keeping its **own dtype**.
///
/// numpy (`_stride_tricks_impl.py:649-656`) marshals every input
/// independently (`np.array(_m, copy=None, subok=subok)`), computes one
/// common shape, broadcasts each input to it, and returns `tuple(result)`.
/// The previous binding sniffed the FIRST input's dtype and coerced ALL
/// inputs to it — silently truncating e.g. `float64 1.5 -> int64 1`
/// (R-CODE-4). Here each input is marshalled at its own dtype, the broadcast
/// target shape is computed dtype-independently from the input shapes, and
/// each input is broadcast through the per-dtype path. `subok` is accepted
/// to mirror numpy's signature (ferray has no ndarray subclasses, so it is a
/// no-op beyond acceptance).
#[pyfunction]
#[pyo3(signature = (*args, subok = false))]
pub fn broadcast_arrays<'py>(
    py: Python<'py>,
    args: &Bound<'py, PyTuple>,
    subok: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = subok; // accepted for numpy parity; ferray has no subclasses.
    if args.is_empty() {
        // numpy `:656 return tuple(result)` with empty `result` -> ().
        return Ok(PyTuple::empty(py).into_any());
    }

    // Normalise every input to an ndarray and compute the common broadcast
    // shape from the SHAPES alone (dtype-independent).
    let mut arrs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(args.len());
    let mut shapes: Vec<Vec<usize>> = Vec::with_capacity(args.len());
    for item in args.iter() {
        let arr = as_ndarray(py, &item)?;
        let shp: Vec<usize> = arr.getattr("shape")?.extract()?;
        arrs.push(arr);
        shapes.push(shp);
    }
    let refs: Vec<&[usize]> = shapes.iter().map(|v| v.as_slice()).collect();
    let target = fst::broadcast_shapes(&refs).map_err(ferr_to_pyerr)?;

    // Broadcast each input at its OWN dtype, preserving it.
    let mut outputs: Vec<Bound<'py, PyAny>> = Vec::with_capacity(arrs.len());
    for arr in &arrs {
        let dt = dtype_name(arr)?;
        let out = match_dtype_all!(dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let bcast = fst::broadcast_to(&fa, &target).map_err(ferr_to_pyerr)?;
            let owned = view_to_owned(bcast).map_err(ferr_to_pyerr)?;
            owned.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        });
        outputs.push(out);
    }
    Ok(PyTuple::new(py, outputs)?.into_any())
}

/// `numpy.broadcast_to(array, shape, subok=False)` — broadcast to a target
/// shape, returning a **read-only** view.
///
/// numpy `_stride_tricks_impl.py:517 _broadcast_to(array, shape,
/// subok=subok, readonly=True)`; the docstring (`:491-494`) promises "A
/// readonly view on the original array". ferray materialises an owned copy
/// at the boundary, so the binding clears the `WRITEABLE` flag to honor the
/// contract (R-DEV-3). `subok` mirrors numpy's signature (`:475`).
#[pyfunction]
#[pyo3(signature = (array, shape, subok = false))]
pub fn broadcast_to<'py>(
    py: Python<'py>,
    array: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    subok: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = subok; // accepted for numpy parity; ferray has no subclasses.
    let arr = as_ndarray(py, array)?;
    let dt = dtype_name(&arr)?;
    let target: Vec<usize> = if let Ok(n) = shape.extract::<usize>() {
        vec![n]
    } else {
        shape.extract()?
    };
    // `broadcast_to` is a pure stride/view op (generic over `T: Element`), so
    // it works unchanged for `Complex<T>`: numpy returns a read-only complex
    // view (`np.broadcast_to(complex, (2,3))`, verified live). Route through the
    // DynMarshal seam (#933) so the imaginary part survives extract/emit.
    let out = match_dtype_all_complex!(dt.as_str(), T => {
        let fa: ArrayD<T> = T::extract_dyn(&arr)?;
        let r = fst::broadcast_to(&fa, &target).map_err(ferr_to_pyerr)?;
        let owned = view_to_owned(r).map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, owned)?
    });
    set_readonly(out)
}

/// `numpy.lib.stride_tricks.sliding_window_view(x, window_shape, axis=None,
/// *, subok=False, writeable=False)` — return a sliding window over `x`.
///
/// Mirrors numpy `_stride_tricks_impl.py:180-444`:
/// - `axis=None` (default) windows every axis (`window_shape` must have one
///   entry per dimension).
/// - an explicit `axis` (int or tuple, possibly with repeats) routes through
///   `normalize_axis_tuple(axis, ndim, allow_duplicate=True)` (`:427`) and
///   `sliding_window_view_axis`.
/// - `writeable=False` (default) returns a **read-only** view (`:181`,
///   docstring `:387-391`); `writeable=True` keeps it writable.
/// - a negative `window_shape` entry raises `ValueError` (`:416-417`), NOT the
///   `OverflowError` a bare `usize` extraction would give.
#[pyfunction]
#[pyo3(signature = (x, window_shape, axis = None, *, subok = false, writeable = false))]
pub fn sliding_window_view<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    window_shape: &Bound<'py, PyAny>,
    axis: Option<&Bound<'py, PyAny>>,
    subok: bool,
    writeable: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = subok; // accepted for numpy parity; ferray has no subclasses.

    // Bind window_shape SIGNED so a negative entry surfaces numpy's
    // ValueError (`:416-417`) rather than an OverflowError at extraction.
    let win_signed: Vec<isize> = if let Ok(n) = window_shape.extract::<isize>() {
        vec![n]
    } else {
        window_shape.extract()?
    };
    if win_signed.iter().any(|&w| w < 0) {
        return Err(PyValueError::new_err(
            "`window_shape` cannot contain negative values",
        ));
    }
    let win: Vec<usize> = win_signed.iter().map(|&w| w as usize).collect();

    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let ndim: usize = arr.getattr("ndim")?.extract()?;

    // Resolve the axis list. None -> all axes (window per dimension); an
    // explicit axis -> normalize (allow_duplicate=True, `:427`).
    let axes: Option<Vec<usize>> = match axis {
        None => None,
        Some(ax) => {
            let raw = extract_axis_tuple(ax)?;
            Some(normalize_axis_tuple(py, &raw, ndim, true)?)
        }
    };

    let out = match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let v = match &axes {
            None => fst::sliding_window_view(&fa, &win).map_err(ferr_to_pyerr)?,
            Some(axs) => {
                fst::sliding_window_view_axis(&fa, &win, axs).map_err(ferr_to_pyerr)?
            }
        };
        let owned = view_to_owned(v).map_err(ferr_to_pyerr)?;
        owned.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    });

    // numpy defaults writeable=False -> read-only view (`:181`).
    if writeable {
        Ok(out)
    } else {
        set_readonly(out)
    }
}

/// `numpy.lib.stride_tricks.as_strided(x, shape=None, strides=None,
/// subok=False, writeable=True, *, check_bounds=None)`
/// (`_stride_tricks_impl.py:39`).
///
/// ferray-core models strides in element units while numpy's `strides`
/// argument is in **bytes**; exposing ferray's element-stride primitive
/// directly would silently reinterpret the caller's byte strides. To honor
/// numpy's exact contract (R-DEV-7 — preserve the observable behaviour,
/// implementation may differ) the binding delegates to numpy's byte-stride
/// implementation. The result is a `numpy.ndarray` view, which is what every
/// caller of `as_strided` consumes.
#[pyfunction]
#[pyo3(signature = (x, shape = None, strides = None, subok = false, writeable = true, *, check_bounds = None))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors numpy as_strided signature"
)]
pub fn as_strided<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    shape: Option<&Bound<'py, PyAny>>,
    strides: Option<&Bound<'py, PyAny>>,
    subok: bool,
    writeable: bool,
    check_bounds: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let st = np.getattr("lib")?.getattr("stride_tricks")?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("shape", shape)?;
    kwargs.set_item("strides", strides)?;
    kwargs.set_item("subok", subok)?;
    kwargs.set_item("writeable", writeable)?;
    // `check_bounds` is keyword-only and only present on newer numpy
    // (_stride_tricks_impl.py:40). Forward it only when the caller passed it
    // so older installed numpy (no `check_bounds` param) still accepts the
    // call.
    if let Some(cb) = check_bounds {
        kwargs.set_item("check_bounds", cb)?;
    }
    st.call_method("as_strided", (x,), Some(&kwargs))
}
