//! Bindings for the `numpy.fft` submodule.
//!
//! ferray's FFT is float-only (`T: FftFloat` → `f32` / `f64`). The
//! complex variants take `Array<Complex<T>>` directly; the real
//! variants (`rfft`/`irfft`) take `Array<T>` real and return
//! `Array<Complex<T>>` complex. The `ferray-numpy-interop` crate's
//! `AsFerray`/`IntoNumPy` don't cover `Complex<T>`, so we do the
//! conversion manually via the `numpy` crate's `PyArrayDyn<Complex<T>>`
//! support — `numpy::Element` is implemented for both `Complex<f32>`
//! and `Complex<f64>`.
//!
//! Default dtype handling matches NumPy: integer or real inputs to
//! `fft` are promoted to `complex128`; real inputs to `rfft` are
//! promoted to `float64`.
//!
//! ## REQ status
//!
//! This shim marshals `numpy.fft.*` onto the (numpy-correct) `ferray-fft`
//! library; the boundary/ABI requirements it owns map onto numpy's public
//! `numpy/fft/_pocketfft.py` + `numpy/fft/_helper.py` contract:
//!
//! - REQ-AXES-INT SHIPPED — `fftshift`/`ifftshift` accept a scalar int *or*
//!   sequence `axes` via `extract_shift_axes` (→ `conv::extract_axis_tuple`);
//!   consumer: `pub fn fftshift` / `pub fn ifftshift`. numpy `_helper.py:69`.
//! - REQ-SIGNED-N SHIPPED — `fft`/`ifft`/`rfft`/`irfft`/`hfft`/`ihfft` bind
//!   `n: Option<isize>` and validate via `conv::validate_fft_n` (`n < 1` →
//!   `ValueError`); consumer: every 1-D transform `pub fn`. numpy
//!   `_pocketfft.py:59`.
//! - REQ-S-SENTINEL SHIPPED — `fftn`/`ifftn`/`fft2`/`ifft2`/`rfft2`/`rfftn`/
//!   `irfft2`/`irfftn` bind `s: Option<Vec<isize>>` and route through
//!   `conv::resolve_fft_s` (numpy 2.0 `s[i] == -1` whole-axis sentinel);
//!   consumer: `resolve_nd_args` / `resolve_nd_args_with_default`. numpy
//!   `_pocketfft.py:737`.
//! - REQ-COMPLEX-REJECT SHIPPED — real transforms (`rfft`/`ihfft`/`rfft2`/
//!   `rfftn`) reject complex input with `TypeError` via
//!   `conv::reject_complex_for_real_fft` instead of the prior silent lossy
//!   cast; consumer: those four `pub fn`s. numpy `_pocketfft.py:81`.
//! - REQ-EXC-TYPES SHIPPED — axis-OOB / 0-d input → `IndexError`
//!   (`conv::fft_normalize_axis`), `fftfreq`/`rfftfreq` negative `n` →
//!   `ValueError` and `d == 0` → `ZeroDivisionError` (`validate_freq_args`);
//!   consumer: every transform `pub fn` + `pub fn fftfreq`/`rfftfreq`. numpy
//!   `_pocketfft.py:88`, `_helper.py:170-171`.
//! - REQ-OUT-KWARG SHIPPED — every transform (`fft`/`ifft`/`rfft`/`irfft`/
//!   `hfft`/`ihfft`/`fftn`/`ifftn`/`fft2`/`ifft2`/`rfft2`/`irfft2`/`rfftn`/
//!   `irfftn`) accepts the numpy 2.0 `out=` kwarg via `apply_out` (writes in
//!   place, returns `out`); consumer: every transform `pub fn`. numpy
//!   `_pocketfft.py` carries `out=None` on every transform (rfft:325,
//!   fftn:756, …).

use ferray_core::array::aliases::{Array1, ArrayD};
use ferray_core::dimension::IxDyn;
use ferray_fft as ff;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use num_complex::Complex;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    as_ndarray, coerce_dtype, dtype_name, ferr_to_pyerr, fft_normalize_axis,
    reject_complex_for_real_fft, resolve_fft_s, validate_fft_n,
};

/// Read the shape of a normalized `numpy.ndarray` as a `Vec<usize>`.
fn ndarray_shape(arr: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    arr.getattr("shape")?.extract()
}

/// Normalize an N-D `axes` argument for an `numpy.fft` N-D transform,
/// raising a plain `IndexError` for an out-of-bounds axis (mirroring
/// `numpy.fft`'s shape-indexing path — see [`fft_normalize_axis`]). `None`
/// defaults to every axis (`0..ndim`).
fn fft_normalize_axes(py: Python<'_>, axes: Option<&[isize]>, ndim: usize) -> PyResult<Vec<usize>> {
    match axes {
        Some(ax) => ax
            .iter()
            .map(|&a| fft_normalize_axis(py, a, ndim))
            .collect(),
        None => Ok((0..ndim).collect()),
    }
}

// ---------------------------------------------------------------------------
// Manual Complex<T> ↔ numpy.ndarray helpers
// ---------------------------------------------------------------------------

/// Extract a `numpy.ndarray` of complex values into a ferray `ArrayD<Complex<T>>`.
pub(crate) fn complex_pyarray_to_ferray<'py, T>(
    arr: &Bound<'py, PyAny>,
) -> PyResult<ArrayD<Complex<T>>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let view: PyReadonlyArrayDyn<Complex<T>> = arr.extract()?;
    let nd = view.as_array();
    let shape: Vec<usize> = nd.shape().to_vec();
    let data: Vec<Complex<T>> = nd.iter().cloned().collect();
    ArrayD::<Complex<T>>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)
}

/// Push a ferray `ArrayD<Complex<T>>` into a fresh Python-owned numpy
/// complex ndarray. Mirrors `IntoNumPy::into_pyarray` but for the
/// non-`NpElement` complex types.
pub(crate) fn complex_ferray_to_pyarray<'py, T>(
    py: Python<'py>,
    arr: ArrayD<Complex<T>>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    Complex<T>: ferray_core::Element + numpy::Element,
{
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<Complex<T>> = arr.iter().cloned().collect();
    // Build a 1-D PyArray then reshape to the target shape — same
    // pattern ferray-numpy-interop uses for IntoNumPy on IxDyn.
    let flat = numpy::PyArray1::<Complex<T>>::from_vec(py, data);
    let reshaped: Bound<'py, PyArrayDyn<Complex<T>>> = flat
        .reshape(shape.as_slice())
        .map_err(|e| PyValueError::new_err(format!("complex reshape failed: {e}")))?;
    Ok(reshaped.into_any())
}

/// Cast any input to the target complex dtype before extracting.
/// Matches NumPy's `fft(int_array)` → complex128 promotion.
fn coerce_to_complex<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    complex_dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?
        .call_method1("asarray", (obj, complex_dtype))
}

/// Pick the complex dtype that pairs with a real input dtype.
fn complex_pair_for(dtype: &str) -> &'static str {
    match dtype {
        "float32" | "f32" => "complex64",
        _ => "complex128",
    }
}

// ---------------------------------------------------------------------------
// Norm parsing
// ---------------------------------------------------------------------------

fn parse_norm(s: Option<&str>) -> PyResult<ff::FftNorm> {
    Ok(match s {
        None | Some("backward") => ff::FftNorm::Backward,
        Some("forward") => ff::FftNorm::Forward,
        Some("ortho") => ff::FftNorm::Ortho,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "norm must be 'backward', 'forward', or 'ortho', got {other:?}"
            )));
        }
    })
}

// ---------------------------------------------------------------------------
// Complex FFT (fft / ifft)
// ---------------------------------------------------------------------------

/// Write the binding's freshly-built result into a caller-supplied `out=`
/// numpy array (numpy 2.0 fft `out` kwarg), returning `out` itself. When
/// `out` is `None`, the result is returned unchanged.
///
/// numpy's `out` writes the transform in place and returns the same object
/// (`np.fft.fft(a, out=out) is out`). We mirror the observable contract:
/// copy `result`'s values into `out` via `numpy.copyto` (which validates the
/// shape) and hand `out` back.
fn apply_out<'py>(
    py: Python<'py>,
    result: Bound<'py, PyAny>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        None => Ok(result),
        Some(out) => {
            py.import("numpy")?
                .call_method1("copyto", (&out, &result))?;
            Ok(out)
        }
    }
}

/// `numpy.fft.fft(a, n=None, axis=-1, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn fft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n_param = validate_fft_n(n)?;
    let cdt = match dt.as_str() {
        "complex64" => "complex64",
        _ => complex_pair_for(dt.as_str()),
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> =
            ff::fft(&fa, n_param, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> =
            ff::fft(&fa, n_param, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.ifft(a, n=None, axis=-1, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn ifft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n = validate_fft_n(n)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        complex_pair_for(dt.as_str())
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> = ff::ifft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> = ff::ifft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// N-D Complex FFT (fftn / ifftn)
// ---------------------------------------------------------------------------

/// Resolve the `(s, axes)` pair for an N-D complex transform at the
/// binding boundary: normalize `axes` (IndexError on OOB), honor the
/// `s[i] == -1` whole-axis sentinel, and validate `s[i] >= 1`. Returns the
/// `usize` shape vector (or `None`) and the normalized `usize` axes the
/// library will transform over.
fn resolve_nd_args(
    py: Python<'_>,
    shape: &[usize],
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
) -> PyResult<(Option<Vec<usize>>, Vec<isize>)> {
    resolve_nd_args_with_default(py, shape, s, axes, |ndim| (0..ndim).collect())
}

/// Like [`resolve_nd_args`] but with a caller-supplied default-axes rule
/// (used by `fft2`/`ifft2`, whose default axes are the last two — not all).
fn resolve_nd_args_with_default(
    py: Python<'_>,
    shape: &[usize],
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    default_axes: impl Fn(usize) -> Vec<usize>,
) -> PyResult<(Option<Vec<usize>>, Vec<isize>)> {
    let norm_axes = match axes.as_deref() {
        Some(ax) => fft_normalize_axes(py, Some(ax), shape.len())?,
        None => default_axes(shape.len()),
    };
    // Defer the "too few dimensions" error to the library, which owns the
    // exact message; just resolve `s` against whatever axes we settled on.
    let resolved_s = resolve_fft_s(s, &norm_axes, shape)?;
    let axes_signed: Vec<isize> = norm_axes.iter().map(|&a| a as isize).collect();
    Ok((resolved_s, axes_signed))
}

/// `numpy.fft.fftn(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn fftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) = resolve_nd_args(py, &ndarray_shape(&arr)?, s, axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        complex_pair_for(dt.as_str())
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> =
            ff::fftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> =
            ff::fftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.ifftn(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn ifftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) = resolve_nd_args(py, &ndarray_shape(&arr)?, s, axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        complex_pair_for(dt.as_str())
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> =
            ff::ifftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> =
            ff::ifftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// 2-D Complex FFT (fft2 / ifft2)
// ---------------------------------------------------------------------------

/// Default-axes rule for the 2-D transforms: the last two axes
/// (`numpy.fft.fft2` defaults `axes=(-2, -1)`). Saturates for `ndim < 2`
/// so the library can raise its own "requires at least 2 dimensions" error.
fn last_two_axes(ndim: usize) -> Vec<usize> {
    if ndim >= 2 {
        vec![ndim - 2, ndim - 1]
    } else {
        (0..ndim).collect()
    }
}

/// `numpy.fft.fft2(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn fft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) =
        resolve_nd_args_with_default(py, &ndarray_shape(&arr)?, s, axes, last_two_axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        complex_pair_for(dt.as_str())
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> =
            ff::fft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> =
            ff::fft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.ifft2(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn ifft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) =
        resolve_nd_args_with_default(py, &ndarray_shape(&arr)?, s, axes, last_two_axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        complex_pair_for(dt.as_str())
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<Complex<f32>> =
            ff::ifft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<Complex<f64>> =
            ff::ifft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// Real FFT (rfft / irfft)
// ---------------------------------------------------------------------------

/// `numpy.fft.rfft(a, n=None, axis=-1, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn rfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    reject_complex_for_real_fft(dt.as_str(), "rfft_n_even")?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n = validate_fft_n(n)?;
    // Real input: promote integer to float64, leave float32/float64 alone.
    let real_dt = match dt.as_str() {
        "float32" | "f32" => "float32",
        "float64" | "f64" => "float64",
        _ => "float64",
    };
    let arr = coerce_dtype(py, &arr, real_dt)?;
    let n_norm = parse_norm(norm)?;
    let r = if real_dt == "float32" {
        let view: PyReadonlyArrayDyn<f32> = arr.extract()?;
        let fa: ArrayD<f32> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f32>> = ff::rfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f64>> = ff::rfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.irfft(a, n=None, axis=-1, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn irfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n = validate_fft_n(n)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ff::irfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ff::irfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// 2-D and N-D Real FFT (rfft2 / irfft2 / rfftn / irfftn)
// ---------------------------------------------------------------------------

/// `numpy.fft.rfft2(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn rfft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    reject_complex_for_real_fft(dt.as_str(), "rfft_n_even")?;
    let (s_res, axes_res) =
        resolve_nd_args_with_default(py, &ndarray_shape(&arr)?, s, axes, last_two_axes)?;
    let real_dt = match dt.as_str() {
        "float32" | "f32" => "float32",
        "float64" | "f64" => "float64",
        _ => "float64",
    };
    let arr = coerce_dtype(py, &arr, real_dt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if real_dt == "float32" {
        let view: PyReadonlyArrayDyn<f32> = arr.extract()?;
        let fa: ArrayD<f32> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f32>> =
            ff::rfft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f64>> =
            ff::rfft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.irfft2(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn irfft2<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) =
        resolve_nd_args_with_default(py, &ndarray_shape(&arr)?, s, axes, last_two_axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ff::irfft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ff::irfft2(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    apply_out(py, r, out)
}

/// `numpy.fft.rfftn(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn rfftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    reject_complex_for_real_fft(dt.as_str(), "rfft_n_even")?;
    let (s_res, axes_res) = resolve_nd_args(py, &ndarray_shape(&arr)?, s, axes)?;
    let real_dt = match dt.as_str() {
        "float32" | "f32" => "float32",
        "float64" | "f64" => "float64",
        _ => "float64",
    };
    let arr = coerce_dtype(py, &arr, real_dt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if real_dt == "float32" {
        let view: PyReadonlyArrayDyn<f32> = arr.extract()?;
        let fa: ArrayD<f32> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f32>> =
            ff::rfftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f64>> =
            ff::rfftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

/// `numpy.fft.irfftn(a, s=None, axes=None, norm=None, out=None)`.
#[pyfunction]
#[pyo3(signature = (a, s = None, axes = None, norm = None, out = None))]
pub fn irfftn<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    s: Option<Vec<isize>>,
    axes: Option<Vec<isize>>,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let (s_res, axes_res) = resolve_nd_args(py, &ndarray_shape(&arr)?, s, axes)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let s_ref = s_res.as_deref();
    let axes_ref = Some(axes_res.as_slice());
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ff::irfftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ff::irfftn(&fa, s_ref, axes_ref, n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// Hermitian FFT (#707)
// ---------------------------------------------------------------------------

/// `numpy.fft.hfft(a, n=None, axis=-1, norm=None, out=None)` — Hermitian FFT
/// (input has Hermitian symmetry, output is real).
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn hfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n = validate_fft_n(n)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_to_complex(py, &arr, cdt)?;
    let n_norm = parse_norm(norm)?;
    let r = if cdt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ff::hfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ff::hfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    apply_out(py, r, out)
}

/// `numpy.fft.ihfft(a, n=None, axis=-1, norm=None, out=None)` — inverse Hermitian
/// FFT (real input → complex output with Hermitian symmetry).
#[pyfunction]
#[pyo3(signature = (a, n = None, axis = -1, norm = None, out = None))]
pub fn ihfft<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: Option<isize>,
    axis: isize,
    norm: Option<&str>,
    out: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    reject_complex_for_real_fft(dt.as_str(), "rfft_n_even")?;
    let ax = fft_normalize_axis(py, axis, ndarray_shape(&arr)?.len())? as isize;
    let n = validate_fft_n(n)?;
    let real_dt = match dt.as_str() {
        "float32" | "f32" => "float32",
        "float64" | "f64" => "float64",
        _ => "float64",
    };
    let arr = coerce_dtype(py, &arr, real_dt)?;
    let n_norm = parse_norm(norm)?;
    let r = if real_dt == "float32" {
        let view: PyReadonlyArrayDyn<f32> = arr.extract()?;
        let fa: ArrayD<f32> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f32>> = ff::ihfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    } else {
        let view: PyReadonlyArrayDyn<f64> = arr.extract()?;
        let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<Complex<f64>> = ff::ihfft(&fa, n, Some(ax), n_norm).map_err(ferr_to_pyerr)?;
        complex_ferray_to_pyarray(py, r)?
    };
    apply_out(py, r, out)
}

// ---------------------------------------------------------------------------
// Frequency / shift helpers
// ---------------------------------------------------------------------------

/// Validate a `numpy.fft.fftfreq`/`rfftfreq` `(n, d)` pair, mirroring
/// numpy's exact exception *types*:
///
/// - negative `n` -> `ValueError("negative dimensions are not allowed")`
///   (numpy/fft/_helper.py:171 `results = empty(n, int, ...)` rejects a
///   negative length), NOT the `OverflowError` a bare `usize` extraction
///   would raise.
/// - `d == 0` -> `ZeroDivisionError("float division by zero")`
///   (numpy/fft/_helper.py:170 `val = 1.0 / (n * d)` divides a Python float
///   by zero), NOT a pre-emptive `ValueError`.
fn validate_freq_args(n: isize, d: f64) -> PyResult<usize> {
    if n < 0 {
        return Err(PyValueError::new_err("negative dimensions are not allowed"));
    }
    if d == 0.0 {
        return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
            "float division by zero",
        ));
    }
    Ok(n as usize)
}

/// `numpy.fft.fftfreq(n, d=1.0)`.
#[pyfunction]
#[pyo3(signature = (n, d = 1.0))]
pub fn fftfreq<'py>(py: Python<'py>, n: isize, d: f64) -> PyResult<Bound<'py, PyAny>> {
    let n = validate_freq_args(n, d)?;
    let r: Array1<f64> = ff::fftfreq(n, d).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.fft.rfftfreq(n, d=1.0)`.
#[pyfunction]
#[pyo3(signature = (n, d = 1.0))]
pub fn rfftfreq<'py>(py: Python<'py>, n: isize, d: f64) -> PyResult<Bound<'py, PyAny>> {
    let n = validate_freq_args(n, d)?;
    let r: Array1<f64> = ff::rfftfreq(n, d).map_err(ferr_to_pyerr)?;
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Extract the `axes` argument shared by `fftshift`/`ifftshift`, accepting
/// either a scalar int or a sequence of ints (numpy/fft/_helper.py:69 /
/// :117 `elif isinstance(axes, integer_types): shift = x.shape[axes] // 2`).
/// `None` shifts every axis. Reuses [`extract_axis_tuple`] so a bare Python
/// int is honored instead of rejected at the PyO3 `Vec<isize>` boundary.
fn extract_shift_axes(axes: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<isize>>> {
    match axes {
        None => Ok(None),
        Some(obj) if obj.is_none() => Ok(None),
        Some(obj) => Ok(Some(crate::conv::extract_axis_tuple(obj)?)),
    }
}

/// `numpy.fft.fftshift(x, axes=None)`.
#[pyfunction]
#[pyo3(signature = (x, axes = None))]
pub fn fftshift<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axes: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let axes_vec = extract_shift_axes(axes.as_ref())?;
    let axes_ref = axes_vec.as_deref();
    Ok(crate::match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ff::fftshift(&fa, axes_ref).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.fft.ifftshift(x, axes=None)`.
#[pyfunction]
#[pyo3(signature = (x, axes = None))]
pub fn ifftshift<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axes: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let axes_vec = extract_shift_axes(axes.as_ref())?;
    let axes_ref = axes_vec.as_deref();
    Ok(crate::match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ff::ifftshift(&fa, axes_ref).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
