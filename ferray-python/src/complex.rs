//! Bindings for complex-array operations: `real`, `imag`, `conj`,
//! `conjugate`, `angle`, `real_if_close`, `iscomplex`, `isreal`,
//! `iscomplexobj`, `isrealobj`.
//!
//! ferray's complex *compute* lives in `ferray-ufunc` and requires
//! `Array<Complex<T>>`, so complex inputs are routed through the same
//! `complex_pyarray_to_ferray` helpers used by the FFT module. The
//! marshalling-side contract this module owns is numpy's output-object
//! behaviour, where ferray previously diverged (R-DEV-3 / R-CODE-4):
//!
//!   * **real-input dtype PRESERVATION.** `np.real`/`np.imag`/`np.conj` of a
//!     *real* (non-complex) array return `val.real` / `val.imag` /
//!     `val.conjugate()` with the input's dtype unchanged
//!     (numpy/lib/_type_check_impl.py:86 `real` -> `val.real`, :128 `imag`
//!     -> `val.imag`; `np.conjugate` on a real array is the identity ufunc
//!     keeping the real dtype). The previous binding coerced every input
//!     through `complex128`, so `np.real(int32)` came back `float64` and
//!     `np.conj(float64)` came back `complex128` — a lossy dtype round-trip
//!     across the boundary. We now route a real input through numpy's own
//!     `.real` / `.imag` / `.conjugate()` so the dtype is preserved; only a
//!     genuinely *complex* input reaches the ferray-ufunc compute path.
//!
//!   * **`angle(z, deg=False)`** — the `deg` kwarg
//!     (numpy/lib/_function_base_impl.py:1694 `def angle(z, deg=False)`,
//!     :1742-1743 `if deg: a *= 180 / pi`). A real input takes
//!     `arctan2(0, z)`; a complex input uses `arctan2(imag, real)`.
//!
//!   * **scalar-vs-0d collapse.** numpy's `iscomplex`/`isreal` end in
//!     `res[()]` (numpy/lib/_type_check_impl.py:210-211) and `real`/`imag`/
//!     `angle` return `val.real`/`val.imag`/`arctan2(...)` which collapse a
//!     scalar / 0-d input to a numpy *scalar*, not a 0-d ndarray. We reuse
//!     [`crate::conv::scalarize`] / [`crate::conv::all_scalar_inputs`].
//!
//! ## REQ status
//!
//! Every numpy complex-surface callable this module registers is SHIPPED;
//! the complex compute paths delegate to `ferray-ufunc`, while real-dtype and
//! dtype-collapse contracts are honored at the boundary. (Evidence = the
//! registered `#[pyfunction]` + the library fn it delegates to.)
//!
//! SHIPPED:
//!   - `real` / `imag` — `#[pyfunction] real` / `imag` delegate the complex
//!     path to `ferray_ufunc::real` / `ferray_ufunc::imag`; a real input is
//!     routed through numpy's own `.real` / `.imag` so the dtype is preserved.
//!   - `conj` / `conjugate` — `complex_unary_conj!(conj, ferray_ufunc::conj)`
//!     / `complex_unary_conj!(conjugate, ferray_ufunc::conjugate)`; a real
//!     input takes numpy's identity `.conjugate()` keeping its real dtype.
//!   - `angle` — `#[pyfunction] angle` (with the `deg` kwarg) delegates the
//!     complex path to `ferray_ufunc::angle`; a real input takes
//!     `arctan2(0, z)` at the boundary.
//!   - `real_if_close` — `#[pyfunction] real_if_close` (with the `tol` kwarg);
//!     numpy's reference dtype-collapse algorithm is reproduced at the
//!     boundary (no separate ferray-ufunc op exists for it).
//!   - `iscomplex` / `isreal` — `#[pyfunction] iscomplex` / `isreal` delegate
//!     the complex path to `ferray_ufunc::iscomplex` / `ferray_ufunc::isreal`;
//!     real-numeric/string kinds short-circuit to all-False / all-True / numpy.
//!   - `iscomplexobj` / `isrealobj` — `#[pyfunction] iscomplexobj` /
//!     `isrealobj`; pure dtype-kind predicates evaluated at the boundary.
//!
//! NOT-STARTED: none — the full registered complex surface is shipped.

use ferray_core::array::aliases::ArrayD;
use ferray_numpy_interop::IntoNumPy;
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{all_scalar_inputs, as_ndarray, dtype_name, ferr_to_pyerr, scalarize};
use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};

/// `true` if the numpy dtype name denotes a complex dtype.
fn is_complex_dtype(dt: &str) -> bool {
    matches!(dt, "complex64" | "complex128")
}

/// Collapse the result to a numpy scalar when the original input was a
/// scalar / 0-d array, mirroring numpy's `val.real` / `res[()]` contract.
fn scalarize_if_scalar_input<'py>(
    py: Python<'py>,
    original: &Bound<'py, PyAny>,
    result: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if all_scalar_inputs(py, &[original])? {
        return scalarize(result);
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// real / imag (complex -> underlying float; real input -> dtype preserved)
// ---------------------------------------------------------------------------

/// `numpy.real(val)` — the real part. For a real input the input's dtype is
/// preserved (numpy/lib/_type_check_impl.py:86 `return val.real`); for a
/// complex input the result is the underlying float (complex64 -> float32,
/// complex128 -> float64).
#[pyfunction]
pub fn real<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;

    let out = if !is_complex_dtype(dt.as_str()) {
        // numpy `val.real` on a real array is the array itself, dtype intact.
        arr.getattr("real")?
    } else if dt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ferray_ufunc::real(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ferray_ufunc::real(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    scalarize_if_scalar_input(py, x, out)
}

/// `numpy.imag(val)` — the imaginary part. For a real input this is zeros of
/// the input's dtype (numpy/lib/_type_check_impl.py:128 `return val.imag`);
/// for a complex input the underlying float.
#[pyfunction]
pub fn imag<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;

    let out = if !is_complex_dtype(dt.as_str()) {
        // numpy `val.imag` on a real array is zeros of the same dtype.
        arr.getattr("imag")?
    } else if dt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ferray_ufunc::imag(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ferray_ufunc::imag(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    scalarize_if_scalar_input(py, x, out)
}

// ---------------------------------------------------------------------------
// conj / conjugate (complex -> complex; real input -> dtype preserved)
// ---------------------------------------------------------------------------

macro_rules! complex_unary_conj {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;

            let out = if !is_complex_dtype(dt.as_str()) {
                // numpy.conjugate on a real array is the identity ufunc; the
                // real dtype (int64/float64/...) is preserved.
                arr.call_method0("conjugate")?
            } else if dt == "complex64" {
                let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
                let r: ArrayD<Complex<f32>> = $ferr_path(&fa).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray(py, r)?
            } else {
                let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
                let r: ArrayD<Complex<f64>> = $ferr_path(&fa).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray(py, r)?
            };
            scalarize_if_scalar_input(py, x, out)
        }
    };
}

complex_unary_conj!(conj, ferray_ufunc::conj);
complex_unary_conj!(conjugate, ferray_ufunc::conjugate);

// ---------------------------------------------------------------------------
// angle (complex -> float; supports the `deg` kwarg)
// ---------------------------------------------------------------------------

/// `numpy.angle(z, deg=False)` — counter-clockwise angle from the positive
/// real axis (numpy/lib/_function_base_impl.py:1694). For a complex input we
/// compute via ferray-ufunc; for a real input numpy takes `arctan2(0, z)`
/// (:1738-1741). When `deg` is true the result is scaled by `180 / pi`
/// (:1742-1743 `if deg: a *= 180 / pi`).
#[pyfunction]
#[pyo3(signature = (z, deg = false))]
pub fn angle<'py>(
    py: Python<'py>,
    z: &Bound<'py, PyAny>,
    deg: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, z)?;
    let dt = dtype_name(&arr)?;

    let out = if !is_complex_dtype(dt.as_str()) {
        // numpy real path: arctan2(0, zreal) -> float64 (or float32 for a
        // float32 input, matching numpy's arctan2 promotion).
        let np = py.import("numpy")?;
        let zimag = np.call_method1("zeros_like", (&arr,))?;
        np.call_method1("arctan2", (zimag, &arr))?
    } else if dt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<f32> = ferray_ufunc::angle(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<f64> = ferray_ufunc::angle(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };

    let out = if deg {
        // a *= 180 / pi (numpy/lib/_function_base_impl.py:1742-1743).
        let factor = 180.0_f64 / std::f64::consts::PI;
        out.call_method1("__mul__", (factor,))?
    } else {
        out
    };
    scalarize_if_scalar_input(py, z, out)
}

// ---------------------------------------------------------------------------
// real_if_close (collapse complex -> real when all imag parts within tol*eps)
// ---------------------------------------------------------------------------

/// `numpy.real_if_close(a, tol=100)` — if `a` is complex with every imaginary
/// part within `tol` machine-epsilons of zero, return the real part; else `a`
/// unchanged (numpy/lib/_type_check_impl.py:496-550). A real input is returned
/// unchanged with its dtype intact (:543-544
/// `if not issubclass(type_, complexfloating): return a`). The tolerance is
/// `tol * finfo(dtype).eps` when `tol > 1`, else `tol` itself (:545-547).
///
/// We mirror numpy's reference algorithm directly at the boundary using the
/// numpy primitives (`finfo`, `absolute`, `all`) so the dtype-collapse
/// contract is exact; there is no separate ferray-ufunc op for it.
#[pyfunction]
#[pyo3(signature = (a, tol = 100.0))]
pub fn real_if_close<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    tol: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;

    // Real input -> returned unchanged (dtype preserved).
    if !is_complex_dtype(dt.as_str()) {
        return Ok(arr);
    }

    // tol -> tol * eps when tol > 1 (numpy/lib/_type_check_impl.py:545-547).
    let eff_tol = if tol > 1.0 {
        let dtype_obj = arr.getattr("dtype")?;
        let finfo = np.call_method1("finfo", (dtype_obj,))?;
        let eps: f64 = finfo.getattr("eps")?.extract()?;
        eps * tol
    } else {
        tol
    };

    let imag = arr.getattr("imag")?;
    let abs_imag = np.call_method1("absolute", (imag,))?;
    let within = np.call_method1("less", (abs_imag, eff_tol))?;
    let all_within: bool = np.call_method1("all", (within,))?.extract()?;

    if all_within {
        Ok(arr.getattr("real")?)
    } else {
        Ok(arr)
    }
}

// ---------------------------------------------------------------------------
// iscomplex / isreal (complex -> bool array; scalar input -> scalar bool)
// ---------------------------------------------------------------------------

/// `numpy.iscomplex(x)` — True where the imaginary part is nonzero. A
/// scalar / 0-d input collapses to a numpy bool scalar via `res[()]`
/// (numpy/lib/_type_check_impl.py:210-211).
#[pyfunction]
pub fn iscomplex<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;

    let out = if !is_complex_dtype(dt.as_str()) {
        // For non-complex input, NumPy returns an all-False bool array.
        let np = py.import("numpy")?;
        let shape = arr.getattr("shape")?;
        np.call_method1("zeros", (shape, "bool"))?.into_any()
    } else if dt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<bool> = ferray_ufunc::iscomplex(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<bool> = ferray_ufunc::iscomplex(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    scalarize_if_scalar_input(py, x, out)
}

/// `numpy.isreal(x)` — True where the imaginary part is zero. A scalar / 0-d
/// input collapses to a numpy bool scalar (numpy/lib/_type_check_impl.py
/// `isreal` -> `imag(x) == 0`).
#[pyfunction]
pub fn isreal<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;

    // numpy `isreal(x)` is literally `imag(x) == 0`
    // (numpy/lib/_type_check_impl.py:262, imag :166). For a *real numeric*
    // dtype (kind in {i,u,f,b}) `imag(x)` is all-zero so the result is
    // all-True (the fast path). For string/bytes/object/etc. (kind not in
    // {i,u,f,b,c}), `imag(x)` returns the elements themselves and `"..." == 0`
    // is False elementwise — so we delegate to `numpy.isreal(x)` to mirror
    // numpy exactly (e.g. all-False for str_/bytes_, incl. the empty string).
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let out = if matches!(kind.as_str(), "i" | "u" | "f" | "b") {
        // Real numeric input: imag is all zeros -> all True.
        let np = py.import("numpy")?;
        let shape = arr.getattr("shape")?;
        np.call_method1("ones", (shape, "bool"))?.into_any()
    } else if !is_complex_dtype(dt.as_str()) {
        // Non-numeric, non-complex (string/bytes/object/...): defer to numpy's
        // own `imag(x) == 0`, which is all-False for these kinds.
        let np = py.import("numpy")?;
        np.call_method1("isreal", (&arr,))?.into_any()
    } else if dt == "complex64" {
        let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr)?;
        let r: ArrayD<bool> = ferray_ufunc::isreal(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    } else {
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr)?;
        let r: ArrayD<bool> = ferray_ufunc::isreal(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    };
    scalarize_if_scalar_input(py, x, out)
}

/// `numpy.iscomplexobj(x)` — True if the array's dtype is complex.
#[pyfunction]
pub fn iscomplexobj<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<bool> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(is_complex_dtype(dt.as_str()))
}

/// `numpy.isrealobj(x)` — True if the array's dtype is real (not complex).
#[pyfunction]
pub fn isrealobj<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<bool> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(!is_complex_dtype(dt.as_str()))
}
