//! Bindings for the `numpy.linalg` submodule.
//!
//! All functions live under `ferray.linalg.*` and accept `numpy.ndarray`
//! inputs. ferray's linalg is float-only (`T: LinalgFloat`, sealed to
//! `f32`/`f64`), so we use [`match_dtype_float`] throughout. Decomposition
//! functions return tuples; we marshal them as Python tuples so callers
//! can unpack with `Q, R = linalg.qr(a)` exactly as in NumPy.
//!
//! The complex variants (`matmul_complex`, `eig`, `eigvals`,
//! `inv_complex`, `det_complex`, `solve_complex`) round-trip
//! `Array<Complex<T>>` through the `complex_*_ferray_to_pyarray` helpers.
//!
//! ## Boundary conventions (numpy.linalg marshalling)
//!
//! Two boundary concerns are mirrored from `numpy/linalg/_linalg.py`:
//!
//! - **`numpy.linalg.LinAlgError`** (`_linalg.py:115 class
//!   LinAlgError(ValueError)`): singular / non-positive-definite / square-
//!   required failures raise this exception *type*, not a bare
//!   `ValueError`. The binding routes the owning crate's errors through
//!   [`crate::conv::linalg_err_to_pyerr`].
//! - **integer→float64 promotion** (`_commonType`): det/inv/norm/solve/qr/
//!   svd/cholesky of an integer/bool array return float64 rather than
//!   rejecting the dtype. The binding promotes via
//!   [`crate::conv::promote_linalg_input`].
//!
//! ## REQ status
//!
//! Two states only (SHIPPED / NOT-STARTED), per goal.md. Tracks the
//! ferray-python `numpy.linalg` marshalling surface. Verified by
//! `ferray-python/tests/divergence_linalg.py` against the live numpy 2.4.5
//! oracle.
//!
//! | REQ | Status | Evidence |
//! | --- | --- | --- |
//! | RC3/RC7 (LinAlgError exception type) | SHIPPED | Impl: `linalg_err_to_pyerr` in `conv.rs`. Consumer: `det`/`inv`/`solve`/`cholesky`/`qr`/`svd`/`eigvals` `.map_err(\|e\| linalg_err_to_pyerr(py, e))` in `linalg.rs`; `LinAlgError` registered in `lib.rs` `register_linalg_module`. |
//! | RC2/RC4 (int→float64 `_commonType`) | SHIPPED | Impl: `promote_linalg_input` in `conv.rs`. Consumer: `det`/`inv`/`norm`/`solve`/`qr`/`svd`/`cholesky` in `linalg.rs`. |
//! | RC8 (top-level products) | SHIPPED | Impl: `dot`/`vdot`/`matmul`/`inner`/`outer` `#[pyfunction]`s in `linalg.rs`. Consumer: registered on the root module in `lib.rs` `_ferray` + re-exported in `python/ferray/__init__.py`. |
//! | qr(mode='r') / svd(compute_uv=False) | SHIPPED | Impl: `r_only` branch of `qr` (routes to `fl::qr` Reduced, returns bare R) + `compute_uv` branch of `svd` (routes to `fl::svdvals`) in `linalg.rs`. Consumer: registered in `lib.rs`. |
//! | matmul 1d×1d → scalar | SHIPPED | Impl: `one_d` branch of `matmul` routes to `fl::dot` (0-d result) in `linalg.rs`. Consumer: registered top-level + under `linalg` in `lib.rs`. |
//! | norm ord=-1/-2 on 2-D matrix | SHIPPED | Impl: `parse_norm_order` maps `-1.0`→`NormOrder::NegL1`, `-2.0`→`NormOrder::NegL2` in `linalg.rs`. Consumer: `norm`/`norm_axis`/`cond` call `parse_norm_order`. |
//! | eigvals (general eigenvalues) | SHIPPED | Impl: `eigvals` `#[pyfunction]` in `linalg.rs` (routes to `fl::eigvals`, complex result). Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | matrix_norm / vector_norm (NumPy 2.0 array-API) | SHIPPED | Impl: `matrix_norm` (default `ord='fro'`, routes to `fl::norm` matrix path) + `vector_norm` (axis=None flattens, axis routes to `fl::norm_axis`) `#[pyfunction]`s in `linalg.rs`. Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | svdvals (NumPy 2.0 array-API) | SHIPPED | Impl: `svdvals` `#[pyfunction]` in `linalg.rs` (routes to `fl::svdvals`). Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | linalg.cross (NumPy 2.0 array-API) | SHIPPED | Impl: `cross` `#[pyfunction]` in `linalg.rs` (1-D 3-component, routes to `fl::cross`; stacked input raises ValueError — library lacks the batched path). Consumer: registered in `lib.rs` `register_linalg_module`. |

use ferray_core::array::aliases::{Array1, Array2, ArrayD};
use ferray_core::dimension::IxDyn;
use ferray_linalg as fl;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::{PyArrayMethods, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

use crate::conv::{
    as_ndarray, dtype_name, ferr_to_pyerr, linalg_err_to_pyerr, promote_linalg_input,
};
use crate::match_dtype_float;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_norm_order(ord: &Bound<'_, PyAny>) -> PyResult<fl::NormOrder> {
    use fl::NormOrder;
    if ord.is_none() {
        return Ok(NormOrder::L2);
    }
    if let Ok(s) = ord.extract::<String>() {
        return Ok(match s.to_lowercase().as_str() {
            "fro" => NormOrder::Fro,
            "nuc" => NormOrder::Nuc,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown norm order string: {other:?} (expected 'fro' or 'nuc')"
                )));
            }
        });
    }
    if let Ok(f) = ord.extract::<f64>() {
        if f.is_infinite() {
            return Ok(if f > 0.0 {
                NormOrder::Inf
            } else {
                NormOrder::NegInf
            });
        }
        if f == 1.0 {
            return Ok(NormOrder::L1);
        }
        if f == 2.0 {
            return Ok(NormOrder::L2);
        }
        // For a 2-D matrix, numpy maps ord=-1 to the MIN column abs-sum and
        // ord=-2 to the smallest singular value (_linalg.py norm()'s
        // `_multi_svd_norm` / `ord=-1` row). ferray-linalg models these as
        // the NegL1 / NegL2 matrix-norm variants (#798); the scalar `norm`
        // dispatcher picks the matrix interpretation for 2-D input. (For a
        // 1-D vector, NegL1/NegL2 reduce to the same `min`/p-norm numpy
        // gives, so this routing is safe for both ranks.)
        if f == -1.0 {
            return Ok(NormOrder::NegL1);
        }
        if f == -2.0 {
            return Ok(NormOrder::NegL2);
        }
        return Ok(NormOrder::P(f));
    }
    Err(PyTypeError::new_err(
        "ord must be None, a number, or one of 'fro' / 'nuc'",
    ))
}

fn parse_qr_mode(mode: &str) -> PyResult<fl::QrMode> {
    match mode {
        "reduced" | "r" => Ok(fl::QrMode::Reduced),
        "complete" => Ok(fl::QrMode::Complete),
        other => Err(PyValueError::new_err(format!(
            "qr mode must be 'reduced' or 'complete', got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Norms / measures
// ---------------------------------------------------------------------------

/// `numpy.linalg.norm(a, ord=None)` — full-array reduction.
///
/// Axis-wise norms (`numpy.linalg.norm(a, ord, axis=k)`) are deferred to
/// a follow-up since they require an `Option<usize>` parameter through
/// the `norm_axis` codepath.
#[pyfunction]
#[pyo3(signature = (a, ord = None))]
pub fn norm<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    ord: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    let ord_value = match ord {
        Some(o) => parse_norm_order(o)?,
        // numpy: `ord=None` is the Frobenius norm for a 2-D matrix and the
        // 2-norm for a vector (`_linalg.py:2650` norm table: "None →
        // Frobenius norm / 2-norm"). The default therefore depends on the
        // input rank, which only the binding knows from the live `ndim`.
        // For 2-D inputs route to `Fro`; everything else (1-D, and N-D which
        // the library flattens for `ord=None`) keeps the 2-norm.
        None => {
            let ndim: usize = arr.getattr("ndim")?.extract()?;
            if ndim == 2 {
                fl::NormOrder::Fro
            } else {
                fl::NormOrder::L2
            }
        }
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::norm(&fa, ord_value).map_err(|e| linalg_err_to_pyerr(py, e))?;
        // Wrap in a 0-D numpy array — NumPy returns a 0-D scalar here.
        let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
            .map_err(ferr_to_pyerr)?;
        arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.matrix_norm(x, ord='fro', keepdims=False)` (NumPy 2.0
/// array-API). Computes the matrix norm of the trailing 2 axes. ferray's
/// linalg is single-matrix, so we mirror `numpy/linalg/_linalg.py
/// matrix_norm` for the 2-D case: route to the library `norm` with the
/// matrix-norm interpretation. The default `ord='fro'` is the Frobenius
/// norm (`_linalg.py def matrix_norm(x, /, *, keepdims=False, ord="fro")`).
#[pyfunction]
#[pyo3(signature = (x, *, keepdims = false, ord = None))]
pub fn matrix_norm<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    keepdims: bool,
    ord: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, x)?;
    let dt = dtype_name(&arr)?;
    // Default ord is 'fro' (Frobenius), unlike the scalar `norm` whose
    // default is the 2-norm. matrix_norm always treats input as a matrix.
    let ord_value = match ord {
        Some(o) => parse_norm_order(o)?,
        None => fl::NormOrder::Fro,
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::norm(&fa.into_dyn(), ord_value).map_err(|e| linalg_err_to_pyerr(py, e))?;
        if keepdims {
            let arr2 = ArrayD::<T>::from_vec(IxDyn::new(&[1, 1]), vec![scalar])
                .map_err(ferr_to_pyerr)?;
            arr2.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        } else {
            let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
                .map_err(ferr_to_pyerr)?;
            arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }
    }))
}

/// `numpy.linalg.vector_norm(x, axis=None, keepdims=False, ord=2)` (NumPy
/// 2.0 array-API). With the default `axis=None` numpy flattens `x` to a
/// vector and computes the `ord`-norm (`_linalg.py def vector_norm(x, /, *,
/// axis=None, keepdims=False, ord=2)`); ferray's scalar `norm` over a
/// flattened array gives the identical reduction. An explicit `axis` routes
/// to the library `norm_axis` reducer.
#[pyfunction]
#[pyo3(signature = (x, *, axis = None, keepdims = false, ord = None))]
pub fn vector_norm<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: Option<usize>,
    keepdims: bool,
    ord: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, x)?;
    let dt = dtype_name(&arr)?;
    // vector_norm's default ord is 2 (the L2 vector norm).
    let ord_value = match ord {
        Some(o) => parse_norm_order(o)?,
        None => fl::NormOrder::L2,
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        match axis {
            None => {
                // Flatten to a 1-D vector and reduce. numpy flattens when
                // axis is None (_linalg.py vector_norm: `x = x.ravel()`).
                let flat: Vec<T> = fa.iter().copied().collect();
                let n = flat.len();
                let vec1 = ArrayD::<T>::from_vec(IxDyn::new(&[n]), flat)
                    .map_err(ferr_to_pyerr)?;
                let scalar: T = fl::norm(&vec1, ord_value).map_err(|e| linalg_err_to_pyerr(py, e))?;
                if keepdims {
                    let shape: Vec<usize> = fa.shape().iter().map(|_| 1usize).collect();
                    let arrk = ArrayD::<T>::from_vec(IxDyn::new(&shape), vec![scalar])
                        .map_err(ferr_to_pyerr)?;
                    arrk.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
                } else {
                    let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
                        .map_err(ferr_to_pyerr)?;
                    arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
                }
            }
            Some(ax) => {
                let r: ArrayD<T> = fl::norm_axis(&fa, ord_value, ax, keepdims).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }
        }
    }))
}

/// `numpy.linalg.svdvals(x)` (NumPy 2.0 array-API) — the 1-D singular
/// values of `x`, sorted in descending order. Equivalent to
/// `svd(x, compute_uv=False)` (`_linalg.py def svdvals(x): return
/// svd(x, compute_uv=False, hermitian=False)`); routes to the library
/// `svdvals` path.
#[pyfunction]
pub fn svdvals<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let s: Array1<T> = fl::svdvals(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.cross(a, b, axis=-1)` (NumPy 2.0 array-API) — the cross
/// product of two 3-component vectors along `axis`. NumPy 2.0's
/// `linalg.cross` REQUIRES exactly 3 components (unlike the legacy
/// top-level `numpy.cross` which also did 2-vectors). ferray-linalg's
/// `cross` is the 1-D 3-element form; this binding handles the 1-D case.
/// Stacked (>1-D) inputs are NOT yet supported by the library and raise
/// `ValueError`.
#[pyfunction]
#[pyo3(signature = (a, b, axis = -1))]
pub fn cross<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    // The library cross only supports 1-D 3-element vectors; reject stacked
    // input rather than silently producing a wrong result.
    let ndim_a: usize = arr_a.getattr("ndim")?.extract()?;
    if ndim_a != 1 || (axis != -1 && axis != 0) {
        return Err(PyValueError::new_err(
            "linalg.cross: only 1-D 3-component vectors are supported (stacked cross is deferred)",
        ));
    }
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::cross(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.det(a)` — determinant of a square matrix.
#[pyfunction]
pub fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::det(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
            .map_err(ferr_to_pyerr)?;
        arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.slogdet(a)` — `(sign, log_abs_det)` tuple.
#[pyfunction]
pub fn slogdet<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (sign, logabs): (T, T) = fl::slogdet(&fa).map_err(ferr_to_pyerr)?;
        let s = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![sign]).map_err(ferr_to_pyerr)?;
        let l = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![logabs]).map_err(ferr_to_pyerr)?;
        let s_py = s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let l_py = l.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [s_py, l_py])?.into_any()
    }))
}

/// `numpy.linalg.matrix_rank(a, tol=None)`.
#[pyfunction]
#[pyo3(signature = (a, tol = None))]
pub fn matrix_rank<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    tol: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let rank: usize = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let t: Option<T> = tol.map(|v| v as T);
        fl::matrix_rank(&fa, t).map_err(ferr_to_pyerr)?
    });
    Ok((rank as i64).into_pyobject(py)?.into_any())
}

/// `numpy.trace(a)` — sum of the main diagonal.
#[pyfunction]
pub fn trace<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::trace(&fa).map_err(ferr_to_pyerr)?;
        let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
            .map_err(ferr_to_pyerr)?;
        arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Decompositions
// ---------------------------------------------------------------------------

/// `numpy.linalg.cholesky(a)`.
#[pyfunction]
pub fn cholesky<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = fl::cholesky(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.qr(a, mode="reduced")`.
///
/// `mode='r'` returns just the R factor as a *bare* ndarray, not a
/// `(Q, R)` tuple — `_linalg.py:1142 if mode == 'r': ... return wrap(r)`.
/// We compute the reduced decomposition (R is identical between numpy's
/// 'reduced' and 'r' modes) and discard Q. Other modes return `(Q, R)`.
#[pyfunction]
#[pyo3(signature = (a, mode = "reduced"))]
pub fn qr<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, mode: &str) -> PyResult<Bound<'py, PyAny>> {
    let r_only = mode == "r";
    // 'r' shares R with the reduced decomposition; map it to Reduced here.
    let qmode = if r_only {
        fl::QrMode::Reduced
    } else {
        parse_qr_mode(mode)?
    };
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (q, r): (Array2<T>, Array2<T>) = fl::qr(&fa, qmode).map_err(|e| linalg_err_to_pyerr(py, e))?;
        let r_py = r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        if r_only {
            r_py
        } else {
            let q_py = q.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            PyTuple::new(py, [q_py, r_py])?.into_any()
        }
    }))
}

/// `scipy.linalg.lu(a)` — `(P, L, U)` tuple. NumPy itself doesn't expose
/// this directly (it's in scipy), but ferray-Rust does, so we surface it.
#[pyfunction]
pub fn lu<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (p, l, u) = fl::lu(&fa).map_err(ferr_to_pyerr)?;
        let p_py = p.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let l_py = l.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let u_py = u.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [p_py, l_py, u_py])?.into_any()
    }))
}

/// `numpy.linalg.svd(a, full_matrices=True, compute_uv=True)`.
///
/// `compute_uv=False` returns ONLY the 1-D singular values (routing to the
/// library's `svdvals`), matching `_linalg.py:1668 def svd(a,
/// full_matrices=True, compute_uv=True, ...)` whose `if compute_uv:` branch
/// (line 1810) returns `(u, s, vh)` and otherwise returns the bare `s`.
#[pyfunction]
#[pyo3(signature = (a, full_matrices = true, compute_uv = true))]
pub fn svd<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    full_matrices: bool,
    compute_uv: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        if compute_uv {
            let (u, s, vt) = fl::svd(&fa, full_matrices).map_err(|e| linalg_err_to_pyerr(py, e))?;
            let u_py = u.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            let s_py = s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            let vt_py = vt.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
            PyTuple::new(py, [u_py, s_py, vt_py])?.into_any()
        } else {
            let s: Array1<T> = fl::svdvals(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
            s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }
    }))
}

/// `numpy.linalg.eigh(a)` — eigendecomposition of a symmetric matrix.
#[pyfunction]
pub fn eigh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (w, v) = fl::eigh(&fa).map_err(ferr_to_pyerr)?;
        let w_py = w.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let v_py = v.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [w_py, v_py])?.into_any()
    }))
}

/// `numpy.linalg.eigvalsh(a)` — eigenvalues of a symmetric matrix only.
#[pyfunction]
pub fn eigvalsh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = fl::eigvalsh(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.eigvals(a)` — eigenvalues of a *general* (non-symmetric)
/// matrix (`_linalg.py:1183 def eigvals(a)`). Unlike `eigvalsh`, a general
/// matrix can have complex eigenvalues, so the result is always complex,
/// mirroring the `eig` binding's complex round-tripping (integer/bool input
/// is promoted to float64 first, via `_commonType`).
#[pyfunction]
pub fn eigvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = crate::conv::coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let w = fl::eigvals(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        complex_1d_ferray_to_pyarray(py, w)?
    }))
}

// ---------------------------------------------------------------------------
// Solvers
// ---------------------------------------------------------------------------

/// `numpy.linalg.solve(a, b)` — solve `Ax = b`.
#[pyfunction]
pub fn solve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // numpy `_commonType` promotes integer/bool A (and B) to float64 before
    // the LAPACK solve; align both operands to A's promoted dtype.
    let arr_a = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArray2<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: Array2<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::solve(&fa, &fb).map_err(|e| linalg_err_to_pyerr(py, e))?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.inv(a)` — matrix inverse.
#[pyfunction]
pub fn inv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = fl::inv(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.pinv(a, rcond=None)` — Moore-Penrose pseudoinverse.
#[pyfunction]
#[pyo3(signature = (a, rcond = None))]
pub fn pinv<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rcond: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let rc: Option<T> = rcond.map(|v| v as T);
        let r: Array2<T> = fl::pinv(&fa, rc).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.matrix_power(a, n)`.
#[pyfunction]
pub fn matrix_power<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array2<T> = fl::matrix_power(&fa, n).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Products
// ---------------------------------------------------------------------------

macro_rules! bind_dyn_product {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: &Bound<'py, PyAny>,
            b: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let arr_a = as_ndarray(py, a)?;
            let dt = dtype_name(&arr_a)?;
            let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
            Ok(match_dtype_float!(dt.as_str(), T => {
                let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
                let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
                let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
                let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
                let r: ArrayD<T> = $ferr_path(&fa, &fb).map_err(ferr_to_pyerr)?;
                r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
            }))
        }
    };
}

bind_dyn_product!(dot, fl::dot);
bind_dyn_product!(inner, fl::inner);
bind_dyn_product!(outer, fl::outer);
bind_dyn_product!(kron, fl::kron);

/// `numpy.matmul(a, b)` / the `@` operator.
///
/// For 1-D x 1-D, numpy collapses to the inner product as a 0-D scalar
/// (`np.matmul([1,2,3],[4,5,6])` == `32.0`, ndim 0) — `_linalg.py:3318
/// def matmul(x1, x2, /)` follows the gufunc `(n?,k),(k,m?)->(n?,m?)`
/// signature where both `n?` and `m?` collapse for vector inputs. The
/// ferray-linalg `matmul` rejects 1-D x 1-D (it would need a dot fall-back),
/// so the binding routes that one case to `dot`, whose `ArrayD` result is
/// the 0-D scalar numpy returns. All other rank combinations dispatch to
/// `fl::matmul` unchanged.
#[pyfunction]
pub fn matmul<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    let a_ndim: usize = arr_a.getattr("ndim")?.extract()?;
    let b_ndim: usize = arr_b.getattr("ndim")?.extract()?;
    let one_d = a_ndim == 1 && b_ndim == 1;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = if one_d {
            fl::dot(&fa, &fb).map_err(|e| linalg_err_to_pyerr(py, e))?
        } else {
            fl::matmul(&fa, &fb).map_err(|e| linalg_err_to_pyerr(py, e))?
        };
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.vdot(a, b)` — flattened dot product, returns scalar.
#[pyfunction]
pub fn vdot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::vdot(&fa, &fb).map_err(ferr_to_pyerr)?;
        let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
            .map_err(ferr_to_pyerr)?;
        arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// #709: batched linalg (cholesky_batched, eigh_batched, eigvalsh_batched,
// svd_batched, qr_batched, solve_batched, inv_batched, pinv_batched,
// det_batched, slogdet_batched, matrix_rank_batched)
// ---------------------------------------------------------------------------

/// `numpy.linalg.cholesky` batched — accepts (..., N, N).
#[pyfunction]
pub fn cholesky_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::cholesky_batched(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched QR.
#[pyfunction]
#[pyo3(signature = (a, mode = "reduced"))]
pub fn qr_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let qmode = parse_qr_mode(mode)?;
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (q, r) = fl::qr_batched(&fa, qmode).map_err(ferr_to_pyerr)?;
        let q_py = q.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let r_py = r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [q_py, r_py])?.into_any()
    }))
}

/// Batched SVD.
#[pyfunction]
#[pyo3(signature = (a, full_matrices = true))]
pub fn svd_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    full_matrices: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (u, s, vt) = fl::svd_batched(&fa, full_matrices).map_err(ferr_to_pyerr)?;
        let u_py = u.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let s_py = s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let vt_py = vt.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [u_py, s_py, vt_py])?.into_any()
    }))
}

/// Batched symmetric eigendecomposition.
#[pyfunction]
pub fn eigh_batched<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (w, v) = fl::eigh_batched(&fa).map_err(ferr_to_pyerr)?;
        let w_py = w.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let v_py = v.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [w_py, v_py])?.into_any()
    }))
}

/// Batched eigenvalues-only of symmetric matrices.
#[pyfunction]
pub fn eigvalsh_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::eigvalsh_batched(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched solve(a, b).
#[pyfunction]
pub fn solve_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::solve_batched(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched matrix inverse.
#[pyfunction]
pub fn inv_batched<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::inv_batched(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched Moore-Penrose pseudoinverse.
#[pyfunction]
#[pyo3(signature = (a, rcond = None))]
pub fn pinv_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    rcond: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let rc: Option<T> = rcond.map(|v| v as T);
        let r: ArrayD<T> = fl::pinv_batched(&fa, rc).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched determinant — returns 1-D array of dets, one per batch.
#[pyfunction]
pub fn det_batched<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = fl::det_batched(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Batched signed-log-det. Returns (sign, logabs) tuple of 1-D arrays.
#[pyfunction]
pub fn slogdet_batched<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (s, l): (Array1<T>, Array1<T>) =
            fl::slogdet_batched(&fa).map_err(ferr_to_pyerr)?;
        let s_py = s.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let l_py = l.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [s_py, l_py])?.into_any()
    }))
}

/// Batched matrix rank.
#[pyfunction]
#[pyo3(signature = (a, tol = None))]
pub fn matrix_rank_batched<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    tol: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let t: Option<T> = tol.map(|v| v as T);
        let r: Array1<i64> = fl::matrix_rank_batched(&fa, t).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// #708: complex linalg (matmul_complex, solve_complex, det_complex,
// inv_complex, eig)
// ---------------------------------------------------------------------------

/// Extract a `numpy.ndarray` of complex values into a ferray
/// `ferray_core::Array<Complex<T>, Ix2>`. Mirrors the pattern in
/// fft.rs but pinned to 2-D.
fn complex_pyarray_to_ferray_2d<'py, T>(
    arr: &Bound<'py, PyAny>,
) -> PyResult<ferray_core::Array<num_complex::Complex<T>, ferray_core::dimension::Ix2>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let view: numpy::PyReadonlyArray2<num_complex::Complex<T>> = arr.extract()?;
    let nd = view.as_array();
    let shape = nd.shape();
    let dim = ferray_core::dimension::Ix2::new([shape[0], shape[1]]);
    let data: Vec<num_complex::Complex<T>> = nd.iter().cloned().collect();
    ferray_core::Array::<num_complex::Complex<T>, ferray_core::dimension::Ix2>::from_vec(dim, data)
        .map_err(ferr_to_pyerr)
}

fn complex_pyarray_to_ferray_1d<'py, T>(
    arr: &Bound<'py, PyAny>,
) -> PyResult<ferray_core::Array<num_complex::Complex<T>, ferray_core::dimension::Ix1>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let view: numpy::PyReadonlyArray1<num_complex::Complex<T>> = arr.extract()?;
    let nd = view.as_array();
    let shape = nd.shape();
    let dim = ferray_core::dimension::Ix1::new([shape[0]]);
    let data: Vec<num_complex::Complex<T>> = nd.iter().cloned().collect();
    ferray_core::Array::<num_complex::Complex<T>, ferray_core::dimension::Ix1>::from_vec(dim, data)
        .map_err(ferr_to_pyerr)
}

fn complex_2d_ferray_to_pyarray<'py, T>(
    py: Python<'py>,
    arr: ferray_core::Array<num_complex::Complex<T>, ferray_core::dimension::Ix2>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let shape = [arr.shape()[0], arr.shape()[1]];
    let data: Vec<num_complex::Complex<T>> = arr.iter().cloned().collect();
    let flat = numpy::PyArray1::<num_complex::Complex<T>>::from_vec(py, data);
    let reshaped = flat
        .reshape(shape)
        .map_err(|e| PyValueError::new_err(format!("complex 2d reshape: {e}")))?;
    Ok(reshaped.into_any())
}

fn complex_1d_ferray_to_pyarray<'py, T>(
    py: Python<'py>,
    arr: ferray_core::Array<num_complex::Complex<T>, ferray_core::dimension::Ix1>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let data: Vec<num_complex::Complex<T>> = arr.iter().cloned().collect();
    Ok(numpy::PyArray1::<num_complex::Complex<T>>::from_vec(py, data).into_any())
}

fn coerce_complex_dtype<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    py.import("numpy")?.call_method1("asarray", (obj, cdt))
}

/// `numpy.matmul` for complex arrays. Real arrays should use `linalg.matmul`.
#[pyfunction]
pub fn matmul_complex<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr_a = coerce_complex_dtype(py, &arr_a, cdt)?;
    let arr_b = coerce_complex_dtype(py, b, cdt)?;
    Ok(if cdt == "complex64" {
        let fa = complex_pyarray_to_ferray_2d::<f32>(&arr_a)?;
        let fb = complex_pyarray_to_ferray_2d::<f32>(&arr_b)?;
        let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
        complex_2d_ferray_to_pyarray(py, r)?
    } else {
        let fa = complex_pyarray_to_ferray_2d::<f64>(&arr_a)?;
        let fb = complex_pyarray_to_ferray_2d::<f64>(&arr_b)?;
        let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
        complex_2d_ferray_to_pyarray(py, r)?
    })
}

/// `numpy.linalg.inv` for complex matrices.
#[pyfunction]
pub fn inv_complex<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_complex_dtype(py, &arr, cdt)?;
    Ok(if cdt == "complex64" {
        let fa = complex_pyarray_to_ferray_2d::<f32>(&arr)?;
        let r = fl::inv_complex(&fa).map_err(ferr_to_pyerr)?;
        complex_2d_ferray_to_pyarray(py, r)?
    } else {
        let fa = complex_pyarray_to_ferray_2d::<f64>(&arr)?;
        let r = fl::inv_complex(&fa).map_err(ferr_to_pyerr)?;
        complex_2d_ferray_to_pyarray(py, r)?
    })
}

/// `numpy.linalg.det` for complex matrices.
#[pyfunction]
pub fn det_complex<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr = coerce_complex_dtype(py, &arr, cdt)?;
    Ok(if cdt == "complex64" {
        let fa = complex_pyarray_to_ferray_2d::<f32>(&arr)?;
        let scalar: num_complex::Complex<f32> = fl::det_complex(&fa).map_err(ferr_to_pyerr)?;
        let arr0 = numpy::PyArray1::<num_complex::Complex<f32>>::from_vec(py, vec![scalar]);
        let r = arr0
            .reshape::<[usize; 0]>([])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        r.into_any()
    } else {
        let fa = complex_pyarray_to_ferray_2d::<f64>(&arr)?;
        let scalar: num_complex::Complex<f64> = fl::det_complex(&fa).map_err(ferr_to_pyerr)?;
        let arr0 = numpy::PyArray1::<num_complex::Complex<f64>>::from_vec(py, vec![scalar]);
        let r = arr0
            .reshape::<[usize; 0]>([])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        r.into_any()
    })
}

/// `numpy.linalg.solve` for complex matrices. Accepts a 1-D or 2-D b.
#[pyfunction]
pub fn solve_complex<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr_a = coerce_complex_dtype(py, &arr_a, cdt)?;
    let arr_b = coerce_complex_dtype(py, b, cdt)?;
    let b_ndim: usize = arr_b.getattr("ndim")?.extract()?;
    Ok(if cdt == "complex64" {
        let fa = complex_pyarray_to_ferray_2d::<f32>(&arr_a)?;
        if b_ndim == 1 {
            let fb = complex_pyarray_to_ferray_1d::<f32>(&arr_b)?;
            let r = fl::solve_complex_vec(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_1d_ferray_to_pyarray(py, r)?
        } else {
            let fb = complex_pyarray_to_ferray_2d::<f32>(&arr_b)?;
            let r = fl::solve_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)?
        }
    } else {
        let fa = complex_pyarray_to_ferray_2d::<f64>(&arr_a)?;
        if b_ndim == 1 {
            let fb = complex_pyarray_to_ferray_1d::<f64>(&arr_b)?;
            let r = fl::solve_complex_vec(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_1d_ferray_to_pyarray(py, r)?
        } else {
            let fb = complex_pyarray_to_ferray_2d::<f64>(&arr_b)?;
            let r = fl::solve_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)?
        }
    })
}

/// `numpy.linalg.eig(a)` — eigenvalues and right eigenvectors of a
/// general (non-symmetric) matrix. Always returns complex results.
#[pyfunction]
pub fn eig<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = crate::conv::coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let (w, v) = fl::eig(&fa).map_err(ferr_to_pyerr)?;
        let w_py = complex_1d_ferray_to_pyarray(py, w)?;
        let v_py = complex_2d_ferray_to_pyarray(py, v)?;
        PyTuple::new(py, [w_py, v_py])?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// #711: lstsq, tensorinv, tensorsolve, matrix_transpose, diagonal
// ---------------------------------------------------------------------------

/// `numpy.linalg.lstsq(a, b, rcond=None)` → `(x, residuals, rank, singular_values)`.
#[pyfunction]
#[pyo3(signature = (a, b, rcond = None))]
pub fn lstsq<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rcond: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let arr_b = as_ndarray(py, b)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, &arr_b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArray2<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: Array2<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let rc: Option<T> = rcond.map(|v| v as T);
        let (x, residuals, rank, sv): (ArrayD<T>, Array1<T>, usize, Array1<T>) =
            fl::lstsq(&fa, &fb, rc).map_err(ferr_to_pyerr)?;
        let x_py = x.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let r_py = residuals.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let rank_py = (rank as i64).into_pyobject(py)?.into_any();
        let sv_py = sv.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        PyTuple::new(py, [x_py, r_py, rank_py, sv_py])?.into_any()
    }))
}

/// `numpy.linalg.tensorinv(a, ind=2)`.
#[pyfunction]
#[pyo3(signature = (a, ind = 2))]
pub fn tensorinv<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    ind: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::tensorinv(&fa, ind).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.tensorsolve(a, b, axes=None)`.
#[pyfunction]
#[pyo3(signature = (a, b, axes = None))]
pub fn tensorsolve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axes: Option<Vec<usize>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::tensorsolve(&fa, &fb, axes.as_deref()).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.matrix_transpose(x)` (NumPy 2.0+) — transpose the last two axes.
/// For a 2-D matrix this is the standard transpose.
#[pyfunction]
pub fn matrix_transpose<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(crate::match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // matrix_transpose returns ArrayView<T, Ix2>; materialise to owned.
        let view2 = fl::matrix_transpose(&fa);
        let shape = view2.shape().to_vec();
        let data: Vec<T> = view2.iter().cloned().collect();
        let owned = Array2::<T>::from_vec(
            ferray_core::dimension::Ix2::new([shape[0], shape[1]]),
            data,
        )
        .map_err(ferr_to_pyerr)?;
        owned.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.diagonal(a, offset=0)` — extract a diagonal of a 2-D array.
#[pyfunction]
#[pyo3(signature = (a, offset = 0))]
pub fn diagonal<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    offset: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(crate::match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = fl::diagonal(&fa, offset).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// #710: einsum, multi_dot, norm_axis, cond, top-level vector products
// ---------------------------------------------------------------------------

/// `numpy.einsum(subscripts, *operands)` — Einstein summation.
#[pyfunction]
#[pyo3(signature = (subscripts, *operands))]
pub fn einsum<'py>(
    py: Python<'py>,
    subscripts: &str,
    operands: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    if operands.is_empty() {
        return Err(PyValueError::new_err("einsum: need at least one operand"));
    }
    let first = as_ndarray(py, &operands.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let mut owned: Vec<ArrayD<T>> = Vec::with_capacity(operands.len());
        for op in operands.iter() {
            let coerced = crate::conv::coerce_dtype(py, &op, &real_dt)?;
            let view: PyReadonlyArrayDyn<T> = coerced.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let refs: Vec<&ArrayD<T>> = owned.iter().collect();
        let r: ArrayD<T> = fl::einsum(subscripts, &refs).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.multi_dot(arrays)` — chain matrix multiply with optimal
/// parenthesisation.
#[pyfunction]
pub fn multi_dot<'py>(py: Python<'py>, arrays: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let list = arrays.cast::<pyo3::types::PyList>()?;
    if list.len() < 2 {
        return Err(PyValueError::new_err("multi_dot: need at least 2 matrices"));
    }
    let first = as_ndarray(py, &list.get_item(0)?)?;
    let dt = dtype_name(&first)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let mut owned: Vec<ArrayD<T>> = Vec::with_capacity(list.len());
        for item in list.iter() {
            let coerced = crate::conv::coerce_dtype(py, &item, dt.as_str())?;
            let view: PyReadonlyArrayDyn<T> = coerced.extract()?;
            owned.push(view.as_ferray().map_err(ferr_to_pyerr)?);
        }
        let refs: Vec<&ArrayD<T>> = owned.iter().collect();
        let r: ArrayD<T> = fl::multi_dot(&refs).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.norm(a, ord, axis, keepdims=False)` — axis-aware variant.
#[pyfunction]
#[pyo3(signature = (a, ord = None, axis = None, keepdims = false))]
pub fn norm_axis<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    ord: Option<&Bound<'py, PyAny>>,
    axis: Option<usize>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // If no axis, fall through to the scalar `norm`.
    if axis.is_none() {
        return norm(py, a, ord);
    }
    let ax = axis.unwrap();
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ord_value = match ord {
        Some(o) => parse_norm_order(o)?,
        None => fl::NormOrder::L2,
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::norm_axis(&fa, ord_value, ax, keepdims).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.linalg.cond(a, p=None)` — condition number.
#[pyfunction]
#[pyo3(signature = (a, p = None))]
pub fn cond<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    p: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let ord_value = match p {
        Some(o) => parse_norm_order(o)?,
        None => fl::NormOrder::L2,
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: ferray_core::array::aliases::Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = fl::cond(&fa, ord_value).map_err(ferr_to_pyerr)?;
        let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar])
            .map_err(ferr_to_pyerr)?;
        arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.matvec(m, v)` (NumPy 2.0+).
#[pyfunction]
pub fn matvec<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_m = as_ndarray(py, m)?;
    let dt = dtype_name(&arr_m)?;
    let arr_v = crate::conv::coerce_dtype(py, v, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let vm: PyReadonlyArrayDyn<T> = arr_m.extract()?;
        let vv: PyReadonlyArrayDyn<T> = arr_v.extract()?;
        let fm: ArrayD<T> = vm.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: ArrayD<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::matvec(&fm, &fv).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.vecmat(v, m)` (NumPy 2.0+).
#[pyfunction]
pub fn vecmat<'py>(
    py: Python<'py>,
    v: &Bound<'py, PyAny>,
    m: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_v = as_ndarray(py, v)?;
    let dt = dtype_name(&arr_v)?;
    let arr_m = crate::conv::coerce_dtype(py, m, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let vv: PyReadonlyArrayDyn<T> = arr_v.extract()?;
        let vm: PyReadonlyArrayDyn<T> = arr_m.extract()?;
        let fv: ArrayD<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let fm: ArrayD<T> = vm.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::vecmat(&fv, &fm).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.vecdot(a, b, axis=-1)` (NumPy 2.0+).
#[pyfunction]
#[pyo3(signature = (a, b, axis = -1))]
pub fn vecdot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::vecdot(&fa, &fb, Some(axis)).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.tensordot(a, b, axes=2)`.
#[pyfunction]
#[pyo3(signature = (a, b, axes = 2))]
pub fn tensordot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axes: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    let axes_u: usize = if axes < 0 {
        return Err(PyValueError::new_err(
            "axes must be a non-negative integer (sequence-of-axes form is deferred)",
        ));
    } else {
        axes as usize
    };
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::tensordot(&fa, &fb, fl::TensordotAxes::Scalar(axes_u))
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
