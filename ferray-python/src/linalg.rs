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
//! | RC9 (#931 complex products / norm / decomp dispatch) | SHIPPED | The numpy-named entrypoints `dot`/`vdot`/`inner`/`outer`/`matmul`/`norm`/`det`/`inv`/`solve` now dispatch complex input instead of raising `TypeError` via `match_dtype_float!`. Impl: `is_complex_linalg_dtype` arm in each fn — `dot`/`inner` compose `sum(a*b)` (`complex_sum_product`, no conjugation) / route 2-D x 2-D to `fl::matmul_complex`; `vdot` conjugates the first arg (`complex_sum_product(..,true)`); `outer` builds `a[i]*b[j]` (`complex_outer`); `matmul` collapses 1-D x 1-D to `complex_sum_product` else `fl::matmul_complex`; `norm` reduces `abs(x)` through real `fl::norm` (`complex_norm_dispatch`, real-width result); `det`/`inv`/`solve` route to `det_complex`/`inv_complex`/`solve_complex` (faer complex LU). Consumer: the same `#[pyfunction]`s registered top-level + under `linalg` in `lib.rs` `_ferray`/`register_linalg_module`. Verified by `tests/test_expansion_complex_linalg.py` + the #931 pins in `tests/test_divergence_complex_sweep_audit.py` (dot/vdot/matmul/norm/solve/inv) against live numpy 2.4.5. |
//! | RC10 (#938 complex trace / kron / tensordot compute) | SHIPPED | These products composed only over `match_dtype_float!`, so complex raised `TypeError`. Impl: `is_complex_linalg_dtype` gate in `trace`/`kron`/`tensordot` → `complex_trace_dispatch` (complex diagonal sum), `complex_kron_dispatch` (2-D block product `a[i,j]*b`), `complex_tensordot_dispatch` (scalar-axes contraction `sum a*b`), all composed from `cplx_mul` over flattened complex buffers. Consumer: the same `trace`/`kron`/`tensordot` `#[pyfunction]`s registered top-level in `lib.rs` `_ferray` (and `trace`/`kron` under `register_linalg_module`). Verified by `tests/test_expansion_complex_dclass.py` + the #938 D-class pins in `tests/test_divergence_complex_converge_audit.py` against live numpy 2.4.5. |
//! | qr(mode='r') / svd(compute_uv=False) | SHIPPED | Impl: `r_only` branch of `qr` (routes to `fl::qr` Reduced, returns bare R) + `compute_uv` branch of `svd` (routes to `fl::svdvals`) in `linalg.rs`. Consumer: registered in `lib.rs`. |
//! | matmul 1d×1d → scalar | SHIPPED | Impl: `one_d` branch of `matmul` routes to `fl::dot` (0-d result) in `linalg.rs`. Consumer: registered top-level + under `linalg` in `lib.rs`. |
//! | norm ord=-1/-2 on 2-D matrix | SHIPPED | Impl: `parse_norm_order` maps `-1.0`→`NormOrder::NegL1`, `-2.0`→`NormOrder::NegL2` in `linalg.rs`. Consumer: `norm`/`norm_axis`/`cond` call `parse_norm_order`. |
//! | eigvals (general eigenvalues) | SHIPPED | Impl: `eigvals` `#[pyfunction]` in `linalg.rs` (routes to `fl::eigvals`, complex result). Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | matrix_norm / vector_norm (NumPy 2.0 array-API) | SHIPPED | Impl: `matrix_norm` (default `ord='fro'`, routes to `fl::norm` matrix path) + `vector_norm` (axis=None flattens, axis routes to `fl::norm_axis`) `#[pyfunction]`s in `linalg.rs`. Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | svdvals (NumPy 2.0 array-API) | SHIPPED | Impl: `svdvals` `#[pyfunction]` in `linalg.rs` (routes to `fl::svdvals`). Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | linalg.cross (NumPy 2.0 array-API) | SHIPPED | Impl: `cross` `#[pyfunction]` in `linalg.rs` (3-component, stacked `(...,3)` along `axis`, routes to `fl::cross` after moving the component axis last; #836). Consumer: registered in `lib.rs` `register_linalg_module`. |
//! | RC11 (#971 integer/bool products compute) | SHIPPED | `kron`/`tensordot`/`vdot`/`inner`/`dot`/`matmul`/`trace`/`matrix_power` composed only over the `LinalgFloat`-sealed `match_dtype_float!`, so integer/bool input raised `TypeError: unsupported dtype for floating-point op`; numpy COMPUTES (preserving the input integer dtype for the products, and upcasting narrow ints via the sum accumulator for `trace`: int8->int64, uint8->uint64, bool->int64). Impl: an `is_int_or_bool_dtype` gate (`dtype.kind` in `i`/`u`/`b`) branches AHEAD of each op's real `match_dtype_float!` path and DELEGATES the int/bool case to `numpy.<fn>` / `numpy.linalg.matrix_power` on the original operands (via `numpy_delegate2`), so numpy owns the contraction + exact integer/bool dtype (incl bool logical-AND/OR products). float/complex paths unchanged. Consumer: the same `#[pyfunction]`s registered top-level in `lib.rs` `_ferray` (+ `trace`/`kron`/`matrix_power` under `register_linalg_module`). Verified by `tests/test_expansion_linalg_int.py` (int8/16/32/64, uint8/16/32/64, bool) + the #971 Class-E pins in `tests/test_divergence_dtype_gap_sweep.py` against live numpy 2.4.5. |

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

/// True when `arr`'s dtype is one the real-only `match_dtype_float!` dispatch
/// cannot represent without loss: float16, datetime64/timedelta64, string
/// (`<U`/`<S`), or object/structured (`'O'`/`'V'`). Complex (`kind == 'c'`) is
/// detected separately by the einsum caller. Mirrors the
/// `stride_tricks::is_non_real_dtype` detect seam (#962).
fn is_non_real_dtype(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    if crate::datetime::is_time_array(arr)? {
        return Ok(true);
    }
    if crate::manipulation::is_string_array(arr)? {
        return Ok(true);
    }
    if crate::conv::is_float16_dtype(dtype_name(arr)?.as_str()) {
        return Ok(true);
    }
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(kind == "O" || kind == "V")
}

/// True when `arr`'s dtype is an integer (`'i'`/`'u'`) or boolean (`'b'`).
/// These are the dtypes the `LinalgFloat`-sealed `match_dtype_float!` real
/// path rejects but numpy COMPUTES on (preserving the integer/bool dtype for
/// the products; `trace` upcasts via its sum accumulator). The int/bool case
/// is delegated to numpy so the exact dtype contract is owned by numpy. (#971)
fn is_int_or_bool_dtype(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(kind == "i" || kind == "u" || kind == "b")
}

/// Delegate a two-operand product to the top-level `numpy.<fn>` on the
/// ORIGINAL operands, returning numpy's result unchanged. Used for the
/// integer/bool arm of `kron`/`dot`/`inner`/`vdot`/`matmul`/`tensordot`,
/// where numpy owns both the contraction and the exact integer/bool output
/// dtype (no float promotion). `extra` carries any trailing positional
/// argument numpy's signature needs (e.g. `tensordot`'s `axes`). (#971)
fn numpy_delegate2<'py>(
    py: Python<'py>,
    func: &str,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    extra: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    match extra {
        Some(e) => np.call_method1(func, (a, b, e)),
        None => np.call_method1(func, (a, b)),
    }
}

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

/// `numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)` —
/// matrix-or-vector norm (`numpy/linalg/_linalg.py:2599`).
///
/// With `axis=None` (default) this reduces the whole array: a 2-D input is a
/// matrix norm (default Frobenius), otherwise the flattened vector 2-norm —
/// the existing real + complex (#931) reduction. When `axis` is given (an int
/// for per-row/column vector norms, or a 2-tuple selecting the matrix plane),
/// numpy owns the full vector × matrix `ord` × `axis` matrix, so delegate to
/// `numpy.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)`.
#[pyfunction]
#[pyo3(signature = (a, ord = None, axis = None, keepdims = false))]
pub fn norm<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    ord: Option<&Bound<'py, PyAny>>,
    axis: Option<&Bound<'py, PyAny>>,
    keepdims: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Axis-wise norms (int axis → per-row/column vector norm, 2-tuple axis →
    // matrix norm over the selected plane) cover numpy's full `ord` × `axis`
    // surface incl. SVD-based singular-value norms; delegate the whole call to
    // numpy so every combination matches exactly (`_linalg.py:2599`).
    if let Some(axis) = axis {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(o) = ord {
            kwargs.set_item("ord", o)?;
        }
        kwargs.set_item("axis", axis)?;
        kwargs.set_item("keepdims", keepdims)?;
        return py
            .import("numpy")?
            .getattr("linalg")?
            .getattr("norm")?
            .call((a,), Some(&kwargs));
    }
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
    // Complex norm: numpy expresses every `ord` over `abs(x)` (the real
    // magnitude `|z|`), so reduce the magnitude array through the real `fl::norm`
    // and return the REAL component width (c64->float32, c128->float64 — live
    // numpy 2.4.5). The 2-/nuclear matrix singular-value norms are rejected (need
    // a complex SVD). (#931)
    if is_complex_linalg_dtype(dt.as_str()) {
        let ndim: usize = arr.getattr("ndim")?.extract()?;
        return match dt.as_str() {
            "complex64" | "c8" => complex_norm_dispatch::<f32>(py, &arr, ord_value, ndim),
            _ => complex_norm_dispatch::<f64>(py, &arr, ord_value, ndim),
        };
    }
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
/// top-level `numpy.cross` which also did 2-vectors). Stacked `(...,3)`
/// inputs are supported: the component axis is moved to the end, `fl::cross`
/// computes the batched cross along the last axis, and the result is moved
/// back to `axis` (#836).
#[pyfunction]
#[pyo3(signature = (a, b, axis = -1))]
pub fn cross<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    // numpy preserves the promoted input dtype (int->int64, f16->f16, mixed
    // floats to the wider, complex stays complex); the float-typed library
    // kernel runs only when both operands already share one float width.
    // Everything else delegates to numpy.linalg.cross for exact dtype parity.
    let a_dt = dtype_name(&as_ndarray(py, a)?)?;
    let b_dt = dtype_name(&as_ndarray(py, b)?)?;
    let is_f32 = |d: &str| matches!(d, "float32" | "f32");
    let is_f64 = |d: &str| matches!(d, "float64" | "f64");
    let native_ok = (is_f32(a_dt.as_str()) && is_f32(b_dt.as_str()))
        || (is_f64(a_dt.as_str()) && is_f64(b_dt.as_str()));
    if !native_ok {
        let np = py.import("numpy")?;
        let linalg = np.getattr("linalg")?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("axis", axis)?;
        return linalg.call_method("cross", (a, b), Some(&kwargs));
    }
    let arr_a = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    // numpy 2.0's `linalg.cross` REQUIRES exactly 3 components along `axis`
    // (numpy/linalg/_linalg.py:cross). The library `fl::cross` operates on
    // the LAST axis, so move the component axis to the end, compute, then move
    // it back via metadata-only transposes (no compute delegation).
    let ndim_a: usize = arr_a.getattr("ndim")?.extract()?;
    let axisn = normalize_cross_axis(axis, ndim_a)?;
    let arr_a = move_axis_to_last(py, &arr_a, axisn)?;
    let arr_b = move_axis_to_last(py, &arr_b, axisn)?;
    // linalg.cross is 3-component only.
    let comp: usize = arr_a
        .getattr("shape")?
        .get_item(ndim_a - 1)?
        .extract()
        .unwrap_or(0);
    if comp != 3 {
        return Err(PyValueError::new_err(
            "linalg.cross: input arrays must have exactly 3 components along the cross axis",
        ));
    }
    let out = match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::cross(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    });
    // Move the component axis (currently last) back to `axisn`.
    move_last_to_axis(py, &out, axisn)
}

/// Normalize a (possibly negative) cross `axis` to `[0, ndim)`, raising the
/// numpy `AxisError`-equivalent (ValueError) when out of bounds.
fn normalize_cross_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    let resolved = if axis < 0 { axis + n } else { axis };
    if resolved < 0 || resolved >= n {
        return Err(PyValueError::new_err(format!(
            "cross: axis {axis} is out of bounds for array of dimension {ndim}"
        )));
    }
    Ok(resolved as usize)
}

/// Build the permutation that moves `axis` to the last position and apply it
/// via the ndarray `.transpose(perm)` method (a metadata-only view).
fn move_axis_to_last<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if axis + 1 == ndim {
        return Ok(arr.clone());
    }
    let mut perm: Vec<usize> = (0..ndim).filter(|&i| i != axis).collect();
    perm.push(axis);
    let perm = pyo3::types::PyTuple::new(py, perm)?;
    arr.call_method1("transpose", (perm,))
}

/// Inverse of [`move_axis_to_last`]: the input has its working axis last; move
/// it back to position `axis`.
fn move_last_to_axis<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if axis + 1 == ndim {
        return Ok(arr.clone());
    }
    // Current layout: [non-axis axes in order..., working]. Insert `working`
    // (the last axis) at position `axis`.
    let mut perm: Vec<usize> = Vec::with_capacity(ndim);
    let last = ndim - 1;
    let others: Vec<usize> = (0..last).collect();
    let mut oi = 0;
    for pos in 0..ndim {
        if pos == axis {
            perm.push(last);
        } else {
            perm.push(others[oi]);
            oi += 1;
        }
    }
    let perm = pyo3::types::PyTuple::new(py, perm)?;
    arr.call_method1("transpose", (perm,))
}

/// `numpy.linalg.det(a)` — determinant of a square matrix.
#[pyfunction]
pub fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = promote_linalg_input(py, a)?;
    let dt = dtype_name(&arr)?;
    // Complex determinant via the library complex LU (`fl::det_complex`,
    // faer-backed). `np.linalg.det([[1+1j,2],[1j,1]]) == (1-1j)`, live numpy
    // 2.4.5. Width-preserving (c64->c64, c128->c128). (#931)
    if is_complex_linalg_dtype(dt.as_str()) {
        return det_complex(py, &arr);
    }
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
    // Complex slogdet (#939): numpy `np.linalg.slogdet(complex)` returns a
    // COMPLEX `sign` (`det/|det|`, a unit complex or 0) and a REAL `logabsdet`
    // (`ln|det|`). Compose from the faer-backed complex determinant
    // (`fl::det_complex`) — `sign = det/abs(det)`, `logabsdet = ln(abs(det))`.
    // (numpy/linalg/_linalg.py:2244 slogdet: `sign`, `logabsdet = log(abs(det))`.)
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_slogdet_dispatch::<f32>(py, &arr),
            _ => complex_slogdet_dispatch::<f64>(py, &arr),
        };
    }
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
    // Complex matrix_rank (#939): numpy counts the singular values of a complex
    // SVD that exceed a tolerance (`tol`, or the default `max(s) * max(m,n) *
    // eps`). Route through `fl::svd_complex_*` and count `s > tol`.
    // (numpy/linalg/_linalg.py:2059 matrix_rank.)
    if is_complex_linalg_dtype(dt.as_str()) {
        let rank = match dt.as_str() {
            "complex64" | "c8" => complex_matrix_rank::<f32>(&arr, tol)?,
            _ => complex_matrix_rank::<f64>(&arr, tol)?,
        };
        return matrix_rank_scalar(py, rank as i64);
    }
    let rank: usize = match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let t: Option<T> = tol.map(|v| v as T);
        fl::matrix_rank(&fa, t).map_err(ferr_to_pyerr)?
    });
    matrix_rank_scalar(py, rank as i64)
}

/// numpy.linalg.matrix_rank of a single matrix returns a numpy `intp` scalar
/// (`np.int64` on a 64-bit platform), not a Python `int` (verified live:
/// `type(np.linalg.matrix_rank(M)) is np.int64`). Wrap the count in `numpy.intp`
/// so downstream `dtype`/`type` checks match.
fn matrix_rank_scalar(py: Python<'_>, rank: i64) -> PyResult<Bound<'_, PyAny>> {
    let np = py.import("numpy")?;
    np.call_method1("intp", (rank,))
}

/// `numpy.trace(a)` — sum of the main diagonal.
#[pyfunction]
pub fn trace<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    // Complex trace = complex sum of the main diagonal (numpy computes
    // `(5+3j)` for the 2x2 fixture, verified live). `fl::trace` is sealed to
    // `LinalgFloat` (f32/f64), so compose the diagonal sum from complex add.
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_trace_dispatch::<f32>(py, &arr),
            _ => complex_trace_dispatch::<f64>(py, &arr),
        };
    }
    // Integer/bool input: numpy computes the diagonal sum, UPCASTING narrow
    // integers to the platform-wide accumulator (int8->int64, uint8->uint64,
    // bool->int64) like `add.reduce`. `fl::trace` is `LinalgFloat`-sealed, so
    // delegate to `numpy.trace` which owns that exact dtype contract. (#971)
    if is_int_or_bool_dtype(&arr)? {
        return py.import("numpy")?.call_method1("trace", (a,));
    }
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
    // Complex QR via the library faer-backed complex `Qr` (#939): numpy
    // `np.linalg.qr(complex)` returns a complex unitary Q and complex R with
    // `Q @ R == A`. The real `match_dtype_float!` rejects complex; route the
    // complex arm to `qr_complex_*` (reduced; `mode='r'` returns bare R).
    if is_complex_linalg_dtype(dt.as_str()) {
        let _ = qmode; // complex path is reduced QR (numpy reduced/'r' share R)
        return match dt.as_str() {
            "complex64" | "c8" => complex_qr_dispatch::<f32>(py, &arr, r_only),
            _ => complex_qr_dispatch::<f64>(py, &arr, r_only),
        };
    }
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
    // Complex SVD via the library faer-backed complex `Svd` (#939): numpy
    // `np.linalg.svd(complex)` returns complex U / Vh and REAL singular values
    // `s`, satisfying `U @ diag(s) @ Vh == A`. The real `match_dtype_float!`
    // below would reject complex with `TypeError`. Width-preserving (c64->c8
    // U/Vh + float32 s; c128->c16 + float64 s). full_matrices controls the
    // U/Vh shapes the same way as the real path; compute_uv=False returns only
    // the real `s`.
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_svd_dispatch::<f32>(py, &arr, full_matrices, compute_uv),
            _ => complex_svd_dispatch::<f64>(py, &arr, full_matrices, compute_uv),
        };
    }
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
    // Complex-input eigenvalues (#937): numpy.linalg.eigvals accepts "a complex-
    // or real-valued matrix" (numpy/linalg/_linalg.py:1193). Routing complex
    // input through the real `match_dtype_float!` below coerced it to float64,
    // DROPPING the imaginary part and returning eigenvalues of `M.real` (the
    // R-CODE-4 corruption). Dispatch the faer complex eigensolver first, keeping
    // the input width (c128 -> complex128, c64 -> complex64).
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex128" | "c16" => {
                let fa = complex_pyarray_to_ferray_2d::<f64>(&arr)?;
                let w = fl::complex::eigvals_complex_f64(&fa).map_err(ferr_to_pyerr)?;
                complex_1d_ferray_to_pyarray(py, w)
            }
            _ => {
                let fa = complex_pyarray_to_ferray_2d::<f32>(&arr)?;
                let w = fl::complex::eigvals_complex_f32(&fa).map_err(ferr_to_pyerr)?;
                complex_1d_ferray_to_pyarray(py, w)
            }
        };
    }
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = crate::conv::coerce_dtype(py, &arr, &real_dt)?;
    let result = match_dtype_float!(real_dt.as_str(), T => {
        let view: PyReadonlyArray2<T> = arr.extract()?;
        let fa: Array2<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let w = fl::eigvals(&fa).map_err(|e| linalg_err_to_pyerr(py, e))?;
        complex_1d_ferray_to_pyarray(py, w)?
    });
    // numpy casts a real-input eigenvalue array back to REAL when no genuinely
    // complex (conjugate-pair) eigenvalue is present; only a real matrix with a
    // complex spectrum stays complex (verified live: `np.linalg.eigvals([[4,2],
    // [1,3]]).dtype == float64`, `[[0,-1],[1,0]] -> complex128`).
    real_if_all_imag_zero(py, result)
}

/// numpy casts a real-input eigen result back to a REAL array when every
/// imaginary part is exactly zero (`numpy/linalg/_linalg.py` `_realType`): a real
/// matrix yields real eigenvalues/-vectors unless a genuinely complex
/// conjugate pair appears. faer returns exactly-`0.0` imaginary parts for real
/// eigenvalues, so the exact `imag == 0` test matches numpy's dtype choice.
fn real_if_all_imag_zero<'py>(
    py: Python<'py>,
    arr: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if all_imag_zero(py, &arr)? {
        let np = py.import("numpy")?;
        np.call_method1("ascontiguousarray", (arr.getattr("real")?,))
    } else {
        Ok(arr)
    }
}

/// `true` if every imaginary part of `arr` is exactly zero.
fn all_imag_zero(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let imag = arr.getattr("imag")?;
    let cmp = np.call_method1("equal", (&imag, 0.0))?;
    np.call_method1("all", (cmp,))?.extract()
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
    // Complex solve via the library complex LU (`fl::solve_complex` /
    // `solve_complex_vec`, faer-backed). numpy promotes both operands to the
    // common complex type; if EITHER A or B is complex, take the complex path.
    let b_arr = as_ndarray(py, b)?;
    let b_dt = dtype_name(&b_arr)?;
    if is_complex_linalg_dtype(dt.as_str()) || is_complex_linalg_dtype(b_dt.as_str()) {
        return solve_complex(py, &arr_a, b);
    }
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
    // Complex inverse via the library complex LU (`fl::inv_complex`, faer-backed),
    // width-preserving. `np.linalg.inv` of a complex matrix is complex. (#931)
    if is_complex_linalg_dtype(dt.as_str()) {
        return inv_complex(py, &arr);
    }
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
    // Complex pinv (#939): numpy computes the Moore-Penrose pseudoinverse via a
    // complex SVD: `pinv(A) = V diag(1/s) U^H` for `s > rcond * max(s)`. Route
    // through `fl::svd_complex_*` and compose. (numpy/linalg/_linalg.py:2169
    // pinv: `cutoff = rcond * largest s; reciprocal where s > cutoff`.)
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_pinv_dispatch::<f32>(py, &arr, rcond),
            _ => complex_pinv_dispatch::<f64>(py, &arr, rcond),
        };
    }
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
    // Complex matrix_power (#939): numpy is repeated complex matmul; `n==0` is
    // the identity, `n<0` inverts first then powers. Compose from the existing
    // `fl::matmul_complex` / `fl::inv_complex` complex primitives — no new
    // library fn. (numpy/linalg/_linalg.py:686 matrix_power.)
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_matrix_power_dispatch::<f32>(py, &arr, n),
            _ => complex_matrix_power_dispatch::<f64>(py, &arr, n),
        };
    }
    // Integer/bool input: numpy's `matrix_power` is repeated integer matmul and
    // keeps the input integer dtype (`np.linalg.matrix_power(int_matrix, 2)` is
    // an int matrix). `fl::matrix_power` is `LinalgFloat`-sealed, so delegate to
    // `numpy.linalg.matrix_power`. (#971)
    if is_int_or_bool_dtype(&arr)? {
        return py
            .import("numpy")?
            .getattr("linalg")?
            .call_method1("matrix_power", (a, n));
    }
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

/// `numpy.kron(a, b)`. The real path routes to `fl::kron`; complex input
/// composes the Kronecker product `a[i]*b[j]` from complex multiply (`fl::kron`
/// is sealed to `LinalgFloat`), matching numpy's complex kron (verified live).
#[pyfunction]
pub fn kron<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_kron_dispatch::<f32>(py, &arr_a, &arr_b),
            _ => complex_kron_dispatch::<f64>(py, &arr_a, &arr_b),
        };
    }
    // Integer/bool input: numpy's `kron` keeps the input integer dtype exactly
    // (`np.kron([1,2,3],[1,2,3])` is int64; bool uses logical-AND products).
    // `fl::kron` is `LinalgFloat`-sealed, so delegate to `numpy.kron`. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        return numpy_delegate2(py, "kron", a, b, None);
    }
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::kron(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.dot(a, b)`. For complex input numpy computes `sum(a*b)` for 1-D x 1-D
/// (NO conjugation, `np.dot([1+2j,3+4j],[5+6j,7+8j]) == (-18+68j)`, live numpy
/// 2.4.5) and a complex matrix product for 2-D x 2-D; the real path keeps
/// routing to `fl::dot`. (#931)
#[pyfunction]
pub fn dot<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    if is_complex_linalg_dtype(dt.as_str()) {
        return complex_dot_dispatch(py, &arr_a, b, dt.as_str());
    }
    // Integer/bool input: `np.dot([1,2,3],[1,2,3])` is `14` int64 — numpy keeps
    // the input integer dtype (no float64 accumulator upcast). `fl::dot` is
    // `LinalgFloat`-sealed, so delegate to `numpy.dot`. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        return numpy_delegate2(py, "dot", a, b, None);
    }
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::dot(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.inner(a, b)`. For complex 1-D inputs `np.inner == np.dot`
/// (`sum(a*b)`, no conjugation); the real path routes to `fl::inner`. (#931)
#[pyfunction]
pub fn inner<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    if is_complex_linalg_dtype(dt.as_str()) {
        return complex_dot_dispatch(py, &arr_a, b, dt.as_str());
    }
    // Integer/bool input: numpy's `inner` keeps the input integer dtype
    // (`np.inner([1,2,3],[1,2,3])` is `14` int64). `fl::inner` is
    // `LinalgFloat`-sealed, so delegate to `numpy.inner`. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        return numpy_delegate2(py, "inner", a, b, None);
    }
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fl::inner(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.outer(a, b)`. For complex inputs the `a.size x b.size` matrix
/// `a[i]*b[j]` (no conjugation, both operands flattened); the real path routes
/// to `fl::outer`. Integer inputs preserve the integer dtype — numpy's
/// `np.outer([1,2],[3,4])` is `int64 [[3,4],[6,8]]` (live numpy 2.4.5), so the
/// `LinalgFloat`-sealed `fl::outer` is bypassed for integers and the
/// `a[i]*b[j]` matrix is composed directly over the flattened operands. (#931, #969)
#[pyfunction]
pub fn outer<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    if is_complex_linalg_dtype(dt.as_str()) {
        let cdt = if dt == "complex64" {
            "complex64"
        } else {
            "complex128"
        };
        let arr_a = coerce_complex_dtype(py, &arr_a, cdt)?;
        let arr_b = coerce_complex_dtype(py, b, cdt)?;
        return if cdt == "complex64" {
            complex_outer::<f32>(py, &arr_a, &arr_b)
        } else {
            complex_outer::<f64>(py, &arr_a, &arr_b)
        };
    }
    let arr_b = crate::conv::coerce_dtype(py, b, dt.as_str())?;
    // Float dtypes route to the optimized `fl::outer`; integer dtypes — which
    // `LinalgFloat` does not cover — compose `a[i]*b[j]` directly (numpy keeps
    // the integer dtype rather than promoting to float).
    match dt.as_str() {
        "float64" | "f64" | "float32" | "f32" => Ok(match_dtype_float!(dt.as_str(), T => {
            let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = fl::outer(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        })),
        _ => Ok(crate::match_dtype_numeric!(dt.as_str(), T => {
            let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
            let fa: Vec<T> = va.as_array().iter().copied().collect();
            let fb: Vec<T> = vb.as_array().iter().copied().collect();
            let (m, n) = (fa.len(), fb.len());
            let mut data: Vec<T> = Vec::with_capacity(m * n);
            for x in &fa {
                for y in &fb {
                    data.push(*x * *y);
                }
            }
            let r = ferray_core::Array::<T, ferray_core::dimension::Ix2>::from_vec(
                ferray_core::dimension::Ix2::new([m, n]),
                data,
            )
            .map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        })),
    }
}

/// Complex `dot`/`inner`: 1-D x 1-D → `sum(a*b)` (0-D complex scalar); 2-D x 2-D
/// → complex matrix product via `fl::matmul_complex`. Higher ranks fall back to
/// the flattened sum-product (numpy's 1-D contract), which covers the pinned
/// vector cases; richer N-D complex contractions remain a follow-up.
fn complex_dot_dispatch<'py>(
    py: Python<'py>,
    arr_a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let cdt = if dt == "complex64" {
        "complex64"
    } else {
        "complex128"
    };
    let arr_a = coerce_complex_dtype(py, arr_a, cdt)?;
    let arr_b = coerce_complex_dtype(py, b, cdt)?;
    let a_ndim: usize = arr_a.getattr("ndim")?.extract()?;
    let b_ndim: usize = arr_b.getattr("ndim")?.extract()?;
    if a_ndim == 2 && b_ndim == 2 {
        return if cdt == "complex64" {
            let fa = complex_pyarray_to_ferray_2d::<f32>(&arr_a)?;
            let fb = complex_pyarray_to_ferray_2d::<f32>(&arr_b)?;
            let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)
        } else {
            let fa = complex_pyarray_to_ferray_2d::<f64>(&arr_a)?;
            let fb = complex_pyarray_to_ferray_2d::<f64>(&arr_b)?;
            let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)
        };
    }
    if cdt == "complex64" {
        complex_sum_product::<f32>(py, &arr_a, &arr_b, false)
    } else {
        complex_sum_product::<f64>(py, &arr_a, &arr_b, false)
    }
}

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
    // Complex `@` / matmul: 1-D x 1-D collapses to the 0-D complex inner product
    // (`sum(a*b)`); 2-D x 2-D routes to the library `matmul_complex` (#931).
    if is_complex_linalg_dtype(dt.as_str()) {
        let cdt = if dt == "complex64" {
            "complex64"
        } else {
            "complex128"
        };
        let arr_a = coerce_complex_dtype(py, &arr_a, cdt)?;
        let arr_b = coerce_complex_dtype(py, b, cdt)?;
        let a_ndim: usize = arr_a.getattr("ndim")?.extract()?;
        let b_ndim: usize = arr_b.getattr("ndim")?.extract()?;
        if a_ndim == 1 && b_ndim == 1 {
            return if cdt == "complex64" {
                complex_sum_product::<f32>(py, &arr_a, &arr_b, false)
            } else {
                complex_sum_product::<f64>(py, &arr_a, &arr_b, false)
            };
        }
        return if cdt == "complex64" {
            let fa = complex_pyarray_to_ferray_2d::<f32>(&arr_a)?;
            let fb = complex_pyarray_to_ferray_2d::<f32>(&arr_b)?;
            let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)
        } else {
            let fa = complex_pyarray_to_ferray_2d::<f64>(&arr_a)?;
            let fb = complex_pyarray_to_ferray_2d::<f64>(&arr_b)?;
            let r = fl::matmul_complex(&fa, &fb).map_err(ferr_to_pyerr)?;
            complex_2d_ferray_to_pyarray(py, r)
        };
    }
    // Integer/bool input: numpy's `matmul` keeps the input integer dtype
    // (`np.matmul(int,int)` is an int matrix; bool uses logical-AND/OR).
    // `fl::matmul`/`fl::dot` are `LinalgFloat`-sealed, so delegate to
    // `numpy.matmul` which owns the contraction and exact dtype. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        return numpy_delegate2(py, "matmul", a, b, None);
    }
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
    // Complex `vdot` CONJUGATES the FIRST argument: `sum(conj(a) * b)`
    // (numpy `_core/multiarray` vdot; `np.vdot([1+2j,3+4j],[5+6j,7+8j]) ==
    // (70-8j)`, live numpy 2.4.5). Both operands are flattened. (#931)
    if is_complex_linalg_dtype(dt.as_str()) {
        let cdt = if dt == "complex64" {
            "complex64"
        } else {
            "complex128"
        };
        let arr_a = coerce_complex_dtype(py, &arr_a, cdt)?;
        let arr_b = coerce_complex_dtype(py, b, cdt)?;
        return if cdt == "complex64" {
            complex_sum_product::<f32>(py, &arr_a, &arr_b, true)
        } else {
            complex_sum_product::<f64>(py, &arr_a, &arr_b, true)
        };
    }
    // Integer/bool input: numpy's `vdot` keeps the input integer dtype
    // (`np.vdot([1,2,3],[1,2,3])` is `14` int64; conjugation is a no-op for
    // reals). `fl::vdot` is `LinalgFloat`-sealed, so delegate to
    // `numpy.vdot`. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        return numpy_delegate2(py, "vdot", a, b, None);
    }
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

// ---------------------------------------------------------------------------
// #931: complex dispatch for the numpy-named products / norm / decompositions
// (dot / vdot / inner / outer / matmul / linalg.norm / det / inv / solve).
//
// The real entrypoints route through `match_dtype_float!`, which has NO complex
// arm — a complex input raised `TypeError: unsupported dtype for floating-point
// op: complex128`, while numpy COMPUTES. The complex MATRIX primitives already
// live in `ferray-linalg` (`matmul_complex` / `inv_complex` / `det_complex` /
// `solve_complex` / `solve_complex_vec`); the arithmetic products (dot / vdot /
// inner / outer) and `norm` are COMPOSED here from complex multiply/add over the
// flattened buffers (there is no `LinalgFloat`-generic complex `dot` in the
// library — `LinalgFloat` is sealed to f32/f64). Each helper keeps the input
// complex width (c64->c64, c128->c128; `norm` returns the REAL component width,
// c64->float32 / c128->float64), matching numpy's live 2.4.5 dtype contract.
// ---------------------------------------------------------------------------

/// `true` for the two complex dtype names ferray marshals.
fn is_complex_linalg_dtype(dt: &str) -> bool {
    matches!(dt, "complex128" | "c16" | "complex64" | "c8")
}

/// Flatten a complex numpy array (any rank) to a row-major `Vec<Complex<T>>`.
fn complex_flat<'py, T>(arr: &Bound<'py, PyAny>) -> PyResult<Vec<num_complex::Complex<T>>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let view: numpy::PyReadonlyArrayDyn<num_complex::Complex<T>> = arr.extract()?;
    Ok(view.as_array().iter().cloned().collect())
}

/// 0-D complex scalar result (`np.dot` / `np.vdot` / `np.inner` of two 1-D
/// vectors returns a 0-D complex array). Width-preserving.
fn complex_scalar_to_pyarray<'py, T>(
    py: Python<'py>,
    z: num_complex::Complex<T>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element,
    num_complex::Complex<T>: numpy::Element,
{
    let flat = numpy::PyArray1::<num_complex::Complex<T>>::from_vec(py, vec![z]);
    let r = flat
        .reshape::<[usize; 0]>([])
        .map_err(|e| PyValueError::new_err(format!("complex scalar reshape: {e}")))?;
    Ok(r.into_any())
}

/// `numpy.dot` / `numpy.inner` for two complex 1-D vectors: `sum(a*b)` with NO
/// conjugation (`np.dot([1+2j,3+4j],[5+6j,7+8j]) == (-18+68j)`, live numpy
/// 2.4.5). Width-preserving. Mismatched lengths raise the numpy ValueError.
fn complex_sum_product<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    conjugate_a: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionRealNorm,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(a)?;
    let fb: Vec<num_complex::Complex<T>> = complex_flat::<T>(b)?;
    if fa.len() != fb.len() {
        return Err(PyValueError::new_err(format!(
            "shapes ({},) and ({},) not aligned",
            fa.len(),
            fb.len()
        )));
    }
    let mut acc = num_complex::Complex::<T>::new(T::ZERO, T::ZERO);
    for (x, y) in fa.iter().zip(fb.iter()) {
        // `vdot` conjugates the FIRST argument (numpy/_core/multiarray vdot:
        // `sum(conj(a) * b)`); `dot`/`inner` do not.
        let xa = if conjugate_a {
            num_complex::Complex::new(x.re, -x.im)
        } else {
            *x
        };
        let p = cplx_mul(xa, *y);
        acc = num_complex::Complex::new(acc.re + p.re, acc.im + p.im);
    }
    complex_scalar_to_pyarray(py, acc)
}

/// `numpy.outer(a, b)` for complex inputs: the `a.size x b.size` matrix
/// `a[i]*b[j]` (NO conjugation, `np.outer` flattens both operands). Width-
/// preserving.
fn complex_outer<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionRealNorm,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(a)?;
    let fb: Vec<num_complex::Complex<T>> = complex_flat::<T>(b)?;
    let (m, n) = (fa.len(), fb.len());
    let mut data: Vec<num_complex::Complex<T>> = Vec::with_capacity(m * n);
    for x in &fa {
        for y in &fb {
            data.push(cplx_mul(*x, *y));
        }
    }
    let arr = ferray_core::Array::<num_complex::Complex<T>, ferray_core::dimension::Ix2>::from_vec(
        ferray_core::dimension::Ix2::new([m, n]),
        data,
    )
    .map_err(ferr_to_pyerr)?;
    complex_2d_ferray_to_pyarray(py, arr)
}

/// Row-major shape of a complex numpy array.
fn complex_shape<'py, T>(arr: &Bound<'py, PyAny>) -> PyResult<Vec<usize>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let view: numpy::PyReadonlyArrayDyn<num_complex::Complex<T>> = arr.extract()?;
    Ok(view.as_array().shape().to_vec())
}

/// Complex `trace`: complex sum of the main diagonal of a 2-D array. Mirrors
/// numpy `np.trace` which sums `a[i, i]` (verified live `np.trace(M) == (5+3j)`).
fn complex_trace_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionRealNorm,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let shape = complex_shape::<T>(arr)?;
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "trace: complex input must be 2-D (offset/3-D trace not yet supported)",
        ));
    }
    let (rows, cols) = (shape[0], shape[1]);
    let flat: Vec<num_complex::Complex<T>> = complex_flat::<T>(arr)?;
    let mut acc = num_complex::Complex::<T>::new(T::ZERO, T::ZERO);
    for i in 0..rows.min(cols) {
        let z = flat[i * cols + i];
        acc = num_complex::Complex::new(acc.re + z.re, acc.im + z.im);
    }
    complex_scalar_to_pyarray(py, acc)
}

/// Complex `kron` for 2-D operands: the `(m*p, n*q)` block matrix
/// `result[i*p+k, j*q+l] = a[i,j] * b[k,l]`, composed from complex multiply.
/// Matches numpy `np.kron` of two complex matrices (verified live).
fn complex_kron_dispatch<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionRealNorm,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let sa = complex_shape::<T>(a)?;
    let sb = complex_shape::<T>(b)?;
    // numpy.kron of two 1-D vectors is the 1-D flat Kronecker product
    // `a[i]*b[j]` (`np.kron([1+1j,2],[1,1j]) == [1+1j,-1+1j,2,2j]`, complex128,
    // live numpy 2.4.5). The 2-D block path below requires rank-2 operands, so
    // route the 1-D x 1-D case through the flattened outer product. (#969)
    if sa.len() == 1 && sb.len() == 1 {
        let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(a)?;
        let fb: Vec<num_complex::Complex<T>> = complex_flat::<T>(b)?;
        let mut data: Vec<num_complex::Complex<T>> = Vec::with_capacity(fa.len() * fb.len());
        for x in &fa {
            for y in &fb {
                data.push(cplx_mul(*x, *y));
            }
        }
        let arr =
            ferray_core::Array::<num_complex::Complex<T>, ferray_core::dimension::Ix1>::from_vec(
                ferray_core::dimension::Ix1::new([data.len()]),
                data,
            )
            .map_err(ferr_to_pyerr)?;
        return complex_1d_ferray_to_pyarray(py, arr);
    }
    if sa.len() != 2 || sb.len() != 2 {
        return Err(PyValueError::new_err(
            "kron: complex input must be 1-D or 2-D (higher-rank kron not yet supported)",
        ));
    }
    let (m, n) = (sa[0], sa[1]);
    let (p, q) = (sb[0], sb[1]);
    let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(a)?;
    let fb: Vec<num_complex::Complex<T>> = complex_flat::<T>(b)?;
    let (rows, cols) = (m * p, n * q);
    let mut data: Vec<num_complex::Complex<T>> =
        vec![num_complex::Complex::<T>::new(T::ZERO, T::ZERO); rows * cols];
    for i in 0..m {
        for j in 0..n {
            let aij = fa[i * n + j];
            for k in 0..p {
                for l in 0..q {
                    let r = i * p + k;
                    let c = j * q + l;
                    data[r * cols + c] = cplx_mul(aij, fb[k * q + l]);
                }
            }
        }
    }
    let arr = ferray_core::Array::<num_complex::Complex<T>, ferray_core::dimension::Ix2>::from_vec(
        ferray_core::dimension::Ix2::new([rows, cols]),
        data,
    )
    .map_err(ferr_to_pyerr)?;
    complex_2d_ferray_to_pyarray(py, arr)
}

/// Complex `tensordot(a, b, axes=N)`: contract the LAST `n` axes of `a` with the
/// FIRST `n` axes of `b` (numpy `tensordot` scalar-axes form), summing complex
/// products. The result shape is `a.shape[:-n] + b.shape[n:]`. numpy
/// `np.tensordot(M, M)` of the 2x2 fixture is the scalar `(24+14j)` (live).
fn complex_tensordot_dispatch<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axes: usize,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default + ReductionRealNorm,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let sa = complex_shape::<T>(a)?;
    let sb = complex_shape::<T>(b)?;
    if axes > sa.len() || axes > sb.len() {
        return Err(PyValueError::new_err(
            "tensordot: axes exceeds operand dimensionality",
        ));
    }
    // Contracted axes: last `axes` of a, first `axes` of b — shapes must match.
    let a_con = &sa[sa.len() - axes..];
    let b_con = &sb[..axes];
    if a_con != b_con {
        return Err(PyValueError::new_err(
            "tensordot: shape mismatch on contracted axes",
        ));
    }
    let free_a = &sa[..sa.len() - axes];
    let free_b = &sb[axes..];
    let k: usize = a_con.iter().product::<usize>().max(1);
    let outer_a: usize = free_a.iter().product::<usize>().max(1);
    let outer_b: usize = free_b.iter().product::<usize>().max(1);
    let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(a)?;
    let fb: Vec<num_complex::Complex<T>> = complex_flat::<T>(b)?;
    // `a` is row-major (outer_a, k); `b` is row-major (k, outer_b).
    let mut data: Vec<num_complex::Complex<T>> = Vec::with_capacity(outer_a * outer_b);
    for ia in 0..outer_a {
        for ib in 0..outer_b {
            let mut acc = num_complex::Complex::<T>::new(T::ZERO, T::ZERO);
            for kk in 0..k {
                let x = fa[ia * k + kk];
                let y = fb[kk * outer_b + ib];
                let p = cplx_mul(x, y);
                acc = num_complex::Complex::new(acc.re + p.re, acc.im + p.im);
            }
            data.push(acc);
        }
    }
    let mut out_shape: Vec<usize> = Vec::with_capacity(free_a.len() + free_b.len());
    out_shape.extend_from_slice(free_a);
    out_shape.extend_from_slice(free_b);
    let arr = ArrayD::<num_complex::Complex<T>>::from_vec(IxDyn::new(&out_shape), data)
        .map_err(ferr_to_pyerr)?;
    crate::fft::complex_ferray_to_pyarray(py, arr)
}

/// `numpy.linalg.norm(complex, ord)` — the magnitude reduction. For every
/// `ord` numpy expresses the complex norm over `abs(x)` (the real magnitude
/// `|z| = hypot(re,im)`), EXCEPT the 2-/Frobenius cases which numpy computes as
/// `sqrt(sum (conj(x)*x).real) == sqrt(sum |x|^2)` (numpy/linalg/_linalg.py:2803
/// `s = (x.conj() * x).real; sqrt(add.reduce(s))` and :2842 for the matrix
/// Frobenius). The real L2 norm of the magnitude array equals exactly that, and
/// the real L1/Inf/-Inf/P norms of the magnitude array equal numpy's complex
/// `add.reduce(abs(x))` / `abs(x).max()` rows (:2800/:2828/:2832). So: build the
/// REAL magnitude array, then route through the existing real `fl::norm`. The
/// result is the REAL component width (numpy returns float32 for c64, float64
/// for c128 — verified live). `ord` strings other than 'fro' and the matrix
/// singular-value norms ('nuc', 2, -2) need a complex SVD and are NOT composable
/// from `abs(x)`; they return the numpy ValueError / a not-yet-supported error.
fn complex_norm_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    ord_value: fl::NormOrder,
    ndim: usize,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element
        + Copy
        + numpy::Element
        + Default
        + ReductionRealNorm
        + ferray_linalg::LinalgFloat,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
    ArrayD<T>: IntoNumPy<T, IxDyn>,
{
    use fl::NormOrder;
    // The singular-value matrix norms need a complex SVD (not composable from
    // `abs(x)`); reject with the numpy-equivalent message rather than a wrong
    // magnitude-norm answer (no silent divergence).
    if ndim == 2 && matches!(ord_value, NormOrder::Nuc | NormOrder::NegL2 | NormOrder::L2) {
        return Err(PyValueError::new_err(
            "linalg.norm: complex matrix 2-norm / nuclear norm (singular-value \
             based) is not yet supported; use a vector ord or 'fro'",
        ));
    }
    let fa: Vec<num_complex::Complex<T>> = complex_flat::<T>(arr)?;
    let shape: Vec<usize> = {
        let view: numpy::PyReadonlyArrayDyn<num_complex::Complex<T>> = arr.extract()?;
        view.as_array().shape().to_vec()
    };
    // `|z| = hypot(re, im)` per element — numpy's `abs(x)` over a complex array.
    let mags: Vec<T> = fa
        .iter()
        .map(|z| ReductionRealNorm::hypot(z.re, z.im))
        .collect();
    let mag_arr = ArrayD::<T>::from_vec(IxDyn::new(&shape), mags).map_err(ferr_to_pyerr)?;
    let scalar: T = fl::norm(&mag_arr, ord_value).map_err(|e| linalg_err_to_pyerr(py, e))?;
    let arr0 = ArrayD::<T>::from_vec(IxDyn::new(&[]), vec![scalar]).map_err(ferr_to_pyerr)?;
    Ok(arr0.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// The real-component scalar ops the complex products / norm need, sealed to the
/// two real widths ferray marshals. `num-traits` is not a direct dependency of
/// `ferray-python`, so this provides `hypot` plus the four real ops used to
/// compose complex multiply/add/conj WITHOUT pulling a new crate, and avoids any
/// `num_traits::Num`/`Neg` bound on the generic helpers.
trait ReductionRealNorm:
    Copy
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    /// Machine epsilon for this real width (numpy's `finfo(dtype).eps`).
    const EPS: Self;
    fn hypot(re: Self, im: Self) -> Self;
    fn ln(self) -> Self;
    fn recip(self) -> Self;
    fn from_f64(x: f64) -> Self;
}
impl ReductionRealNorm for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPS: Self = f32::EPSILON;
    fn hypot(re: Self, im: Self) -> Self {
        f32::hypot(re, im)
    }
    fn ln(self) -> Self {
        f32::ln(self)
    }
    fn recip(self) -> Self {
        1.0f32 / self
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
}
impl ReductionRealNorm for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const EPS: Self = f64::EPSILON;
    fn hypot(re: Self, im: Self) -> Self {
        f64::hypot(re, im)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
    fn recip(self) -> Self {
        1.0f64 / self
    }
    fn from_f64(x: f64) -> Self {
        x
    }
}

/// `(a.re, a.im) * (b.re, b.im)` as a raw complex multiply over the real
/// components — avoids a `num_traits::Num` bound on `Complex<T>`.
fn cplx_mul<T: ReductionRealNorm>(
    a: num_complex::Complex<T>,
    b: num_complex::Complex<T>,
) -> num_complex::Complex<T> {
    num_complex::Complex::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

// ---------------------------------------------------------------------------
// #939: complex svd / qr / pinv / matrix_rank / slogdet / matrix_power.
//
// SVD/QR route to the faer-backed library complex decompositions
// (`fl::complex::svd_complex_*` / `qr_complex_*`). pinv/matrix_rank are
// composed from the complex SVD; slogdet from `fl::det_complex`;
// matrix_power from repeated `fl::matmul_complex` (+ `fl::inv_complex` for
// n<0). The dispatch is monomorphised over the two real widths ferray
// marshals, bridged through `ComplexLinalgWidth` so the concrete library
// fns can be named without re-stating faer's `ComplexField` bound here.
// ---------------------------------------------------------------------------

type CplxArr2<T> = ferray_core::Array<num_complex::Complex<T>, ferray_core::dimension::Ix2>;
type RealArr1<T> = ferray_core::Array<T, ferray_core::dimension::Ix1>;

/// Bridge each real width to its concrete `fl::complex::*` entry points.
trait ComplexLinalgWidth: Sized + Copy + ferray_core::Element + numpy::Element
where
    num_complex::Complex<Self>: ferray_core::Element + numpy::Element,
{
    fn svd_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, RealArr1<Self>, CplxArr2<Self>)>;
    fn qr_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, CplxArr2<Self>)>;
    fn matmul_complex(
        a: &CplxArr2<Self>,
        b: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<CplxArr2<Self>>;
    fn inv_complex(a: &CplxArr2<Self>) -> ferray_core::error::FerrayResult<CplxArr2<Self>>;
    fn det_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<num_complex::Complex<Self>>;
}
impl ComplexLinalgWidth for f32 {
    fn svd_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, RealArr1<Self>, CplxArr2<Self>)> {
        fl::complex::svd_complex_f32(a)
    }
    fn qr_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, CplxArr2<Self>)> {
        fl::complex::qr_complex_f32(a)
    }
    fn matmul_complex(
        a: &CplxArr2<Self>,
        b: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<CplxArr2<Self>> {
        fl::matmul_complex(a, b)
    }
    fn inv_complex(a: &CplxArr2<Self>) -> ferray_core::error::FerrayResult<CplxArr2<Self>> {
        fl::inv_complex(a)
    }
    fn det_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<num_complex::Complex<Self>> {
        fl::det_complex(a)
    }
}
impl ComplexLinalgWidth for f64 {
    fn svd_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, RealArr1<Self>, CplxArr2<Self>)> {
        fl::complex::svd_complex_f64(a)
    }
    fn qr_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<(CplxArr2<Self>, CplxArr2<Self>)> {
        fl::complex::qr_complex_f64(a)
    }
    fn matmul_complex(
        a: &CplxArr2<Self>,
        b: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<CplxArr2<Self>> {
        fl::matmul_complex(a, b)
    }
    fn inv_complex(a: &CplxArr2<Self>) -> ferray_core::error::FerrayResult<CplxArr2<Self>> {
        fl::inv_complex(a)
    }
    fn det_complex(
        a: &CplxArr2<Self>,
    ) -> ferray_core::error::FerrayResult<num_complex::Complex<Self>> {
        fl::det_complex(a)
    }
}

/// Trait alias bundling every bound the complex-decomp dispatchers need.
trait ComplexDecompScalar:
    ReductionRealNorm
    + ComplexLinalgWidth
    + ferray_core::Element
    + numpy::Element
    + Default
    + PartialOrd
where
    num_complex::Complex<Self>: ferray_core::Element + numpy::Element,
{
}
impl<T> ComplexDecompScalar for T
where
    T: ReductionRealNorm
        + ComplexLinalgWidth
        + ferray_core::Element
        + numpy::Element
        + Default
        + PartialOrd,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
}

/// `conj(z)`.
fn cplx_conj<T: ReductionRealNorm>(z: num_complex::Complex<T>) -> num_complex::Complex<T> {
    num_complex::Complex::new(z.re, -z.im)
}

/// Build a real 1-D numpy array from a `Vec<T>`.
fn real_1d_to_pyarray<'py, T>(py: Python<'py>, data: Vec<T>) -> Bound<'py, PyAny>
where
    T: numpy::Element,
{
    numpy::PyArray1::<T>::from_vec(py, data).into_any()
}

/// `numpy.linalg.svd` complex arm. Returns `(U, s, Vh)` (or bare real `s` when
/// `compute_uv=False`). `full_matrices` is honoured the same way numpy does;
/// the library returns the full U (m×m) / Vh (n×n), so for `full_matrices=False`
/// we slice to the thin shapes U[:, :k] / Vh[:k, :].
fn complex_svd_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    full_matrices: bool,
    compute_uv: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let (m, n) = (fa.shape()[0], fa.shape()[1]);
    let k = m.min(n);
    let (u, s, vh) = T::svd_complex(&fa).map_err(ferr_to_pyerr)?;
    let sd: Vec<T> = s.iter().copied().collect();
    if !compute_uv {
        return Ok(real_1d_to_pyarray(py, sd));
    }
    let ud: Vec<num_complex::Complex<T>> = u.iter().copied().collect();
    let vhd: Vec<num_complex::Complex<T>> = vh.iter().copied().collect();
    let (u_out, vh_out) = if full_matrices {
        (
            build_cplx_2d::<T>(m, m, ud)?,
            build_cplx_2d::<T>(n, n, vhd)?,
        )
    } else {
        // Thin: U[:, :k] (m×k) and Vh[:k, :] (k×n).
        let mut u_thin = Vec::with_capacity(m * k);
        for i in 0..m {
            for j in 0..k {
                u_thin.push(ud[i * m + j]);
            }
        }
        let mut vh_thin = Vec::with_capacity(k * n);
        for i in 0..k {
            for j in 0..n {
                vh_thin.push(vhd[i * n + j]);
            }
        }
        (
            build_cplx_2d::<T>(m, k, u_thin)?,
            build_cplx_2d::<T>(k, n, vh_thin)?,
        )
    };
    let u_py = complex_2d_ferray_to_pyarray(py, u_out)?;
    let s_py = real_1d_to_pyarray(py, sd);
    let vh_py = complex_2d_ferray_to_pyarray(py, vh_out)?;
    Ok(PyTuple::new(py, [u_py, s_py, vh_py])?.into_any())
}

/// Assemble a row-major complex `Array<Complex<T>, Ix2>` of shape `(rows, cols)`.
fn build_cplx_2d<T>(
    rows: usize,
    cols: usize,
    data: Vec<num_complex::Complex<T>>,
) -> PyResult<CplxArr2<T>>
where
    T: ferray_core::Element + Copy,
    num_complex::Complex<T>: ferray_core::Element,
{
    CplxArr2::<T>::from_vec(ferray_core::dimension::Ix2::new([rows, cols]), data)
        .map_err(ferr_to_pyerr)
}

/// `numpy.linalg.qr` complex arm (reduced; `r_only` returns bare R).
fn complex_qr_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    r_only: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let (q, r) = T::qr_complex(&fa).map_err(ferr_to_pyerr)?;
    let r_py = complex_2d_ferray_to_pyarray(py, r)?;
    if r_only {
        Ok(r_py)
    } else {
        let q_py = complex_2d_ferray_to_pyarray(py, q)?;
        Ok(PyTuple::new(py, [q_py, r_py])?.into_any())
    }
}

/// `numpy.linalg.pinv` complex arm: `pinv(A) = V diag(1/s) U^H` for the
/// singular values above `cutoff = rcond * max(s)`. From the complex SVD
/// `A = U diag(s) Vh`, `V = Vh^H`, so `pinv = Vh^H diag(sinv) U^H`.
fn complex_pinv_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    rcond: Option<f64>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let (m, n) = (fa.shape()[0], fa.shape()[1]);
    let k = m.min(n);
    let (u, s, vh) = T::svd_complex(&fa).map_err(ferr_to_pyerr)?;
    let ud: Vec<num_complex::Complex<T>> = u.iter().copied().collect();
    let sd: Vec<T> = s.iter().copied().collect();
    let vhd: Vec<num_complex::Complex<T>> = vh.iter().copied().collect();
    // numpy default rcond is 1e-15 (`pinv(a, rcond=1e-15)`); cutoff scales the
    // largest singular value (numpy/linalg/_linalg.py:2169).
    let rcond_t = T::from_f64(rcond.unwrap_or(1e-15));
    let smax = sd
        .iter()
        .copied()
        .fold(T::ZERO, |acc, x| if x > acc { x } else { acc });
    let cutoff = rcond_t * smax;
    let sinv: Vec<T> = sd
        .iter()
        .map(|&sv| if sv > cutoff { sv.recip() } else { T::ZERO })
        .collect();
    // pinv[i,j] = sum_p conj(Vh[p,i]) * sinv[p] * conj(U[j,p]).  (V = Vh^H,
    // U^H[p,j] = conj(U[j,p]).) Result shape is (n, m).
    let mut data: Vec<num_complex::Complex<T>> =
        vec![num_complex::Complex::new(T::ZERO, T::ZERO); n * m];
    for i in 0..n {
        for j in 0..m {
            let mut acc = num_complex::Complex::new(T::ZERO, T::ZERO);
            for p in 0..k {
                let v = cplx_conj(vhd[p * n + i]); // V[i,p] = conj(Vh[p,i])
                let uh = cplx_conj(ud[j * m + p]); // U^H[p,j] = conj(U[j,p])
                let term = cplx_mul(cplx_mul(v, num_complex::Complex::new(sinv[p], T::ZERO)), uh);
                acc = num_complex::Complex::new(acc.re + term.re, acc.im + term.im);
            }
            data[i * m + j] = acc;
        }
    }
    let out = build_cplx_2d::<T>(n, m, data)?;
    complex_2d_ferray_to_pyarray(py, out)
}

/// `numpy.linalg.matrix_rank` complex arm: count singular values above the
/// tolerance. Default tol = `max(s) * max(m, n) * eps`
/// (numpy/linalg/_linalg.py:2059).
fn complex_matrix_rank<'py, T>(arr: &Bound<'py, PyAny>, tol: Option<f64>) -> PyResult<usize>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let (m, n) = (fa.shape()[0], fa.shape()[1]);
    let (_u, s, _vh) = T::svd_complex(&fa).map_err(ferr_to_pyerr)?;
    let sd: Vec<T> = s.iter().copied().collect();
    let smax = sd
        .iter()
        .copied()
        .fold(T::ZERO, |acc, x| if x > acc { x } else { acc });
    let tol_t = match tol {
        Some(t) => T::from_f64(t),
        None => smax * T::from_f64(m.max(n) as f64) * T::EPS,
    };
    Ok(sd.iter().filter(|&&sv| sv > tol_t).count())
}

/// `numpy.linalg.slogdet` complex arm: `(sign, logabsdet)` where
/// `sign = det / |det|` (complex unit, or 0) and `logabsdet = ln|det|` (real).
fn complex_slogdet_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let det = T::det_complex(&fa).map_err(ferr_to_pyerr)?;
    let absdet = T::hypot(det.re, det.im);
    let (sign, logabs) = if absdet == T::ZERO {
        // Singular: numpy returns sign 0+0j and logabsdet -inf.
        (
            num_complex::Complex::new(T::ZERO, T::ZERO),
            T::from_f64(f64::NEG_INFINITY),
        )
    } else {
        let inv = absdet.recip();
        (
            num_complex::Complex::new(det.re * inv, det.im * inv),
            absdet.ln(),
        )
    };
    let sign_py = complex_scalar_to_pyarray(py, sign)?;
    // logabsdet is a REAL 0-D scalar (build via PyArray1 + reshape to 0-d so we
    // only need `numpy::Element`, not the `IntoNumPy`/`NpElement` bound).
    let l_flat = numpy::PyArray1::<T>::from_vec(py, vec![logabs]);
    let l_py = l_flat
        .reshape::<[usize; 0]>([])
        .map_err(|e| PyValueError::new_err(format!("slogdet logabsdet reshape: {e}")))?
        .into_any();
    Ok(PyTuple::new(py, [sign_py, l_py])?.into_any())
}

/// `numpy.linalg.matrix_power` complex arm: repeated complex matmul. `n==0` is
/// the identity, `n<0` inverts first. Composed from `matmul_complex` /
/// `inv_complex` (no new library fn).
fn complex_matrix_power_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    n: i64,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ComplexDecompScalar,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    let fa = complex_pyarray_to_ferray_2d::<T>(arr)?;
    let dim = fa.shape()[0];
    if fa.shape()[0] != fa.shape()[1] {
        return Err(PyValueError::new_err(
            "matrix_power: input must be a square 2-D matrix",
        ));
    }
    // Base matrix: A for n>=0, inv(A) for n<0.
    let (base, p) = if n < 0 {
        (
            T::inv_complex(&fa).map_err(ferr_to_pyerr)?,
            (-(n as i128)) as u128,
        )
    } else {
        (fa, n as u128)
    };
    // Identity (n==0 short-circuits to it).
    let mut acc = identity_cplx::<T>(dim)?;
    if p == 0 {
        return complex_2d_ferray_to_pyarray(py, acc);
    }
    // Exponentiation by squaring over complex matmul.
    let mut b = base;
    let mut e = p;
    while e > 0 {
        if e & 1 == 1 {
            acc = T::matmul_complex(&acc, &b).map_err(ferr_to_pyerr)?;
        }
        e >>= 1;
        if e > 0 {
            b = T::matmul_complex(&b, &b).map_err(ferr_to_pyerr)?;
        }
    }
    complex_2d_ferray_to_pyarray(py, acc)
}

/// `n×n` complex identity.
fn identity_cplx<T>(n: usize) -> PyResult<CplxArr2<T>>
where
    T: ReductionRealNorm + ferray_core::Element + Copy,
    num_complex::Complex<T>: ferray_core::Element,
{
    let mut data = vec![num_complex::Complex::new(T::ZERO, T::ZERO); n * n];
    for i in 0..n {
        data[i * n + i] = num_complex::Complex::new(T::ONE, T::ZERO);
    }
    build_cplx_2d::<T>(n, n, data)
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
    // Complex-input eigendecomposition (#937): numpy.linalg.eig accepts "a
    // complex- or real-valued matrix" (numpy/linalg/_linalg.py:1372) and returns
    // complex eigenpairs. The real path below coerced complex to float64 and
    // returned the eigenpairs of `M.real` (R-CODE-4 imaginary discard). Dispatch
    // faer's complex eigensolver, keeping the input width. Eigenvalue order is
    // not canonical (numpy/linalg/_linalg.py:1199), so set/sort-compare downstream.
    if is_complex_linalg_dtype(dt.as_str()) {
        let (w_py, v_py) = match dt.as_str() {
            "complex128" | "c16" => {
                let fa = complex_pyarray_to_ferray_2d::<f64>(&arr)?;
                let (w, v) = fl::complex::eig_complex_f64(&fa).map_err(ferr_to_pyerr)?;
                (
                    complex_1d_ferray_to_pyarray(py, w)?,
                    complex_2d_ferray_to_pyarray(py, v)?,
                )
            }
            _ => {
                let fa = complex_pyarray_to_ferray_2d::<f32>(&arr)?;
                let (w, v) = fl::complex::eig_complex_f32(&fa).map_err(ferr_to_pyerr)?;
                (
                    complex_1d_ferray_to_pyarray(py, w)?,
                    complex_2d_ferray_to_pyarray(py, v)?,
                )
            }
        };
        return Ok(PyTuple::new(py, [w_py, v_py])?.into_any());
    }
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
        // numpy returns a real (w, v) pair for a real matrix whose entire
        // spectrum is real; only a genuinely complex eigenvalue keeps the pair
        // complex (numpy/linalg/_linalg.py `_realType`). Gate BOTH on the
        // eigenvalues' imaginary parts so eigenvectors track the eigenvalues.
        if all_imag_zero(py, &w_py)? {
            let np = py.import("numpy")?;
            let w_r = np.call_method1("ascontiguousarray", (w_py.getattr("real")?,))?;
            let v_r = np.call_method1("ascontiguousarray", (v_py.getattr("real")?,))?;
            PyTuple::new(py, [w_r, v_r])?.into_any()
        } else {
            PyTuple::new(py, [w_py, v_py])?.into_any()
        }
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
    // complex (#972): `diagonal` is a pure view extraction (no arithmetic) and
    // numpy returns the diagonal preserving the complex dtype
    // (`numpy/_core/fromnumeric.py:diagonal` → `asanyarray(a).diagonal`). The
    // `match_dtype_all!` real path is sealed to real dtypes and would raise
    // `TypeError` on complex, while a float64 coercion would drop the imaginary
    // part (R-CODE-4). Delegate the complex case to numpy, which owns the complex
    // diagonal.
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    if kind == "c" {
        let np = py.import("numpy")?;
        return np.call_method1("diagonal", (&arr, offset));
    }
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

    // The real-only path below coerces EVERY operand to `float32`/`float64`
    // (`coerce_dtype`), which silently DROPS the imaginary part of a complex
    // operand (R-CODE-4 data corruption #965) and would lose float16 precision /
    // misroute datetime etc. numpy's `einsum` performs the full complex
    // contraction (no conjugation) and applies NEP-50 promotion for mixed
    // real+complex inputs (live numpy 2.4.4: `einsum('ij,jk->ik', [[1+1j,0],[0,1]],
    // [[1+1j,0],[0,1]])` -> `[[2j,0],[0,1]]` complex128). So if ANY operand carries
    // a non-real dtype (complex64/128, float16, datetime64/timedelta64, string,
    // object/structured), delegate the WHOLE einsum to `numpy.einsum(subscripts,
    // *operands)` ahead of the real path — numpy owns the complex contraction +
    // NEP-50 + dtype preservation. The real einsum path stays byte-identical.
    let mut materialised: Vec<Bound<'py, PyAny>> = Vec::with_capacity(operands.len());
    let mut any_non_real = false;
    for op in operands.iter() {
        let arr = as_ndarray(py, &op)?;
        let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        any_non_real |= kind == "c" || is_non_real_dtype(&arr)?;
        materialised.push(arr);
    }
    if any_non_real {
        let np = py.import("numpy")?;
        let mut args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(operands.len() + 1);
        args.push(subscripts.into_pyobject(py)?.into_any());
        args.extend(materialised);
        return np.call_method1("einsum", PyTuple::new(py, args)?);
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
        return norm(py, a, ord, None, keepdims);
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
    // Complex tensordot composes the contraction `sum over last `axes` dims of
    // a * last `axes` dims of b` from complex multiply/add (`fl::tensordot` is
    // sealed to `LinalgFloat`). numpy `np.tensordot(M, M)` of the 2x2 complex
    // fixture is the scalar `(24+14j)` (verified live).
    if is_complex_linalg_dtype(dt.as_str()) {
        return match dt.as_str() {
            "complex64" | "c8" => complex_tensordot_dispatch::<f32>(py, &arr_a, &arr_b, axes_u),
            _ => complex_tensordot_dispatch::<f64>(py, &arr_a, &arr_b, axes_u),
        };
    }
    // Integer/bool input: numpy's `tensordot` keeps the input integer dtype
    // (`np.tensordot(int_matrix, int_matrix)` is int64). `fl::tensordot` is
    // `LinalgFloat`-sealed, so delegate to `numpy.tensordot` with the scalar
    // `axes` argument. (#971)
    if is_int_or_bool_dtype(&arr_a)? {
        let axes_arg = axes.into_pyobject(py)?.into_any();
        return numpy_delegate2(py, "tensordot", a, b, Some(axes_arg));
    }
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
