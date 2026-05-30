//! Bindings for the `numpy.emath` (a.k.a. `numpy.lib.scimath`) submodule.
//!
//! `numpy.emath` provides the "smart", complex-aware versions of a handful
//! of math functions whose mathematically-valid output leaves the real
//! domain for certain inputs. Unlike `numpy.sqrt` (which returns `nan` for a
//! negative real input) or `numpy.arccos` (which returns `nan` for `|x| > 1`),
//! the `emath.*` functions return the complex principal value instead — but
//! ONLY when the input is actually out of the real domain. An in-domain real
//! input still yields a real result with the same dtype `numpy.<op>` would.
//!
//! The mechanism is numpy's own (`numpy/lib/_scimath_impl.py`):
//!
//!   * `_fix_real_lt_zero(x)` (`:96`) — if `any(isreal(x) & (x < 0))`, cast
//!     `x` to complex BEFORE applying the op. Used by `sqrt`/`log`/`log2`/
//!     `log10`/`logn` (on both `n` and `x`) and `power` (on the base `x`).
//!   * `_fix_int_lt_zero(x)` (`:125`) — if `any(isreal(x) & (x < 0))`,
//!     promote `x` to double (`x * 1.0`), NOT complex. Used by `power` on the
//!     exponent `p`, so `emath.power([2, 4], -2)` is float `[0.25, 0.0625]`,
//!     not the integer-pow `[0, 0]`.
//!   * `_fix_real_abs_gt_1(x)` (`:153`) — if `any(isreal(x) & (abs(x) > 1))`,
//!     cast `x` to complex. Used by `arccos`/`arcsin`/`arctanh`.
//!
//! ferray-ufunc already ships both the real ufuncs (re-exported as the
//! top-level `ferray.sqrt`/`log`/… bindings) and the complex transcendental
//! kernels (`sqrt_complex`/`ln_complex`/`log2_complex`/`log10_complex`/
//! `acos_complex`/`asin_complex`/`atanh_complex`, `numpy` parity for
//! `np.sqrt(complex_array)` etc.). This module is the thin boundary shim that
//! runs numpy's domain check, then routes the in-domain case to the real
//! ferray binding (preserving numpy's real-dtype contract) and the
//! out-of-domain case to the complex ferray-ufunc kernel (returning
//! `complex128`). The actual transcendental compute always happens in ferray.
//!
//! The domain predicates are evaluated with numpy boolean ops at the boundary
//! (`isreal(x) & (x < 0)`, `.any()`), mirroring `_fix_*` literally — this is
//! marshalling glue, not compute (R-DEV-7: numpy's observable contract is
//! preserved; the implementation may differ).

use ferray_core::array::aliases::ArrayD;
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    all_scalar_inputs, as_ndarray, coerce_dtype, dtype_name, ferr_to_pyerr, scalarize,
};
use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};

/// `true` if the numpy dtype name denotes a complex dtype. A genuinely
/// complex input always takes the complex compute path (numpy's `_fix_*`
/// helpers are no-ops on complex input — `isreal(complex) == False` — so the
/// underlying complex ufunc runs unchanged).
fn is_complex_dtype(dt: &str) -> bool {
    matches!(dt, "complex64" | "complex128")
}

/// Evaluate numpy's `any(isreal(x) & (x < 0))` predicate on `arr`
/// (`_fix_real_lt_zero`, `numpy/lib/_scimath_impl.py:120`). Returns `true`
/// when the real-valued op would leave the real domain for at least one
/// element, so the input must be cast to complex first.
fn any_real_lt_zero(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let isreal = np.call_method1("isreal", (arr,))?;
    let lt = np.call_method1("less", (arr, 0))?;
    let masked = np.call_method1("logical_and", (isreal, lt))?;
    np.call_method1("any", (masked,))?.extract()
}

/// Evaluate numpy's `any(isreal(x) & (abs(x) > 1))` predicate on `arr`
/// (`_fix_real_abs_gt_1`, `numpy/lib/_scimath_impl.py:176`). Used by the
/// inverse-trig family (`arccos`/`arcsin`/`arctanh`).
fn any_real_abs_gt_1(py: Python<'_>, arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let isreal = np.call_method1("isreal", (arr,))?;
    let absx = np.call_method1("abs", (arr,))?;
    let gt = np.call_method1("greater", (absx, 1))?;
    let masked = np.call_method1("logical_and", (isreal, gt))?;
    np.call_method1("any", (masked,))?.extract()
}

/// Collapse the result to a numpy scalar when the original input was a
/// scalar / 0-d array, mirroring numpy's `out : ndarray or scalar`
/// contract (`_scimath_impl.py` docstrings: "If `x` was a scalar, so is
/// `out`"). numpy's emath functions inherit the scalar-vs-array decision
/// from the underlying ufunc, which returns a numpy scalar for a 0-d input.
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

/// Apply a complex-`f64` unary kernel to `arr_c`, which has already been cast
/// to `complex128`. The compute happens in the ferray-ufunc complex kernel;
/// the result is returned as a `complex128` ndarray.
fn complex_unary<'py>(
    py: Python<'py>,
    arr_c: &Bound<'py, PyAny>,
    kernel: impl Fn(&ArrayD<Complex<f64>>) -> ferray_core::error::FerrayResult<ArrayD<Complex<f64>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr_c)?;
    let r = kernel(&fa).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray::<f64>(py, r)
}

/// Shared implementation for the `_fix_real_lt_zero` family (`sqrt`/`log`/
/// `log2`/`log10`): run the domain check; on an out-of-domain real input (or
/// any complex input) cast to `complex128` and run `complex_kernel`; on an
/// in-domain real input delegate to the real ferray binding `real_fn` so the
/// numpy real-dtype contract is preserved.
fn dispatch_lt_zero<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    complex_kernel: impl Fn(
        &ArrayD<Complex<f64>>,
    ) -> ferray_core::error::FerrayResult<ArrayD<Complex<f64>>>,
    real_fn: impl Fn(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let go_complex = is_complex_dtype(dt.as_str()) || any_real_lt_zero(py, &arr)?;
    if go_complex {
        let arr_c = if is_complex_dtype(dt.as_str()) {
            arr
        } else {
            coerce_dtype(py, &arr, "complex128")?
        };
        let out = complex_unary(py, &arr_c, complex_kernel)?;
        return scalarize_if_scalar_input(py, x, out);
    }
    real_fn(py, x)
}

/// Shared implementation for the `_fix_real_abs_gt_1` family (`arccos`/
/// `arcsin`/`arctanh`): identical to [`dispatch_lt_zero`] but keyed on the
/// `abs(x) > 1` predicate.
///
/// BRANCH-CUT CORRECTION. numpy's `_fix_real_abs_gt_1` casts a real
/// out-of-domain `x` to `x + 0.0j` and applies the complex ufunc, whose
/// branch cut places the result on the `+0.0`-imaginary side (e.g.
/// `np.emath.arccos(2) == np.arccos(2 + 0j) == -1.3169j`). ferray-ufunc's
/// complex inverse-trig kernels delegate to `num_complex`'s `acos`/`asin`/
/// `atanh`, which on the EXACT real axis (`imag == +0.0`) pick a branch that
/// matches numpy for `x < -1` but is the CONJUGATE of numpy for `x > 1`
/// (verified live across arccos/arcsin/arctanh: `nc(x) == np(x+0j)` for
/// `x < 0`, `nc(x) == conj(np(x+0j))` for `x > 0`). The discriminant is the
/// sign of the input's real part. On the REAL-cast path we therefore
/// conjugate ONLY the elements whose input `real(x) > 0`. In-domain elements
/// (`|x| <= 1`) yield a real result, so conjugating them is a no-op — the
/// `real(x) > 0` selector is safe to apply element-wise across the whole
/// array. For a GENUINELY complex input (non-zero imaginary part)
/// `num_complex` already matches numpy exactly (verified for `2+1j`,
/// `-2+0.3j`, …), so that path is NOT corrected.
fn dispatch_abs_gt_1<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    complex_kernel: impl Fn(
        &ArrayD<Complex<f64>>,
    ) -> ferray_core::error::FerrayResult<ArrayD<Complex<f64>>>,
    real_fn: impl Fn(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let input_complex = is_complex_dtype(dt.as_str());
    let go_complex = input_complex || any_real_abs_gt_1(py, &arr)?;
    if go_complex {
        let arr_c = if input_complex {
            arr.clone()
        } else {
            coerce_dtype(py, &arr, "complex128")?
        };
        let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_c)?;
        let r = complex_kernel(&fa).map_err(ferr_to_pyerr)?;
        let mut out = complex_ferray_to_pyarray::<f64>(py, r)?;
        if !input_complex {
            // Real-cast path: select numpy's `x + 0.0j` branch element-wise —
            // conjugate where the input real part is positive (`x > 1`),
            // keep where it is non-positive (`x < -1`). `np.where` on the
            // boolean mask `real(x) > 0`.
            let np = py.import("numpy")?;
            let mask = np.call_method1("greater", (&arr, 0))?;
            let conj = np.call_method1("conj", (&out,))?;
            out = np.call_method1("where", (mask, conj, &out))?;
        }
        return scalarize_if_scalar_input(py, x, out);
    }
    real_fn(py, x)
}

// ---------------------------------------------------------------------------
// _fix_real_lt_zero family: sqrt, log, log2, log10, logn
// ---------------------------------------------------------------------------

/// `numpy.emath.sqrt(x)` — complex for `x < 0`, real otherwise
/// (`numpy/lib/_scimath_impl.py:187`, `x = _fix_real_lt_zero(x);
/// return nx.sqrt(x)`).
#[pyfunction]
pub fn sqrt<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_lt_zero(py, x, ferray_ufunc::sqrt_complex, crate::ufunc::sqrt)
}

/// `numpy.emath.log(x)` — natural log; complex for `x <= 0`, real otherwise
/// (`_scimath_impl.py:243`).
#[pyfunction]
pub fn log<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_lt_zero(py, x, ferray_ufunc::ln_complex, crate::ufunc::log)
}

/// `numpy.emath.log2(x)` — base-2 log; complex for `x <= 0`
/// (`_scimath_impl.py:387`).
#[pyfunction]
pub fn log2<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_lt_zero(py, x, ferray_ufunc::log2_complex, crate::ufunc::log2)
}

/// `numpy.emath.log10(x)` — base-10 log; complex for `x <= 0`
/// (`_scimath_impl.py:293`).
#[pyfunction]
pub fn log10<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_lt_zero(py, x, ferray_ufunc::log10_complex, crate::ufunc::log10)
}

/// `numpy.emath.logn(n, x)` — log base `n` of `x`. Both `n` and `x` go
/// through `_fix_real_lt_zero` independently, then the result is
/// `log(x) / log(n)` (`_scimath_impl.py:380-382`). Each `log` is computed in
/// ferray (real or complex per its own domain); the final division is
/// NEP-50 promotion glue done with `numpy.divide` at the boundary so a
/// mixed real/complex pair promotes exactly as numpy's `nx.log(x) /
/// nx.log(n)` would (R-DEV-7).
#[pyfunction]
pub fn logn<'py>(
    py: Python<'py>,
    n: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    // Compute log(x) and log(n) as ndarrays (real or complex per domain),
    // WITHOUT collapsing scalars yet — the division decides the final form.
    let log_x = emath_log_array(py, x)?;
    let log_n = emath_log_array(py, n)?;
    let np = py.import("numpy")?;
    let out = np.call_method1("divide", (log_x, log_n))?;
    // numpy emath collapses a scalar (n, x) pair to a numpy scalar.
    if all_scalar_inputs(py, &[n, x])? {
        return scalarize(out);
    }
    Ok(out)
}

/// Helper for [`logn`]: apply `_fix_real_lt_zero` then `log`, returning the
/// ferray-computed result as a numpy ndarray (never scalar-collapsed, so the
/// caller controls the final scalarization).
fn emath_log_array<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    if is_complex_dtype(dt.as_str()) || any_real_lt_zero(py, &arr)? {
        let arr_c = if is_complex_dtype(dt.as_str()) {
            arr
        } else {
            coerce_dtype(py, &arr, "complex128")?
        };
        return complex_unary(py, &arr_c, ferray_ufunc::ln_complex);
    }
    // In-domain real: ferray real log, then ensure ndarray form (the binding
    // may return a numpy scalar for a 0-d input; re-wrap via asarray so the
    // division sees an array and we control scalarization).
    let r = crate::ufunc::log(py, x)?;
    as_ndarray(py, &r)
}

// ---------------------------------------------------------------------------
// _fix_real_abs_gt_1 family: arccos, arcsin, arctanh
// ---------------------------------------------------------------------------

/// `numpy.emath.arccos(x)` — complex for `abs(x) > 1`, real in `[0, pi]`
/// otherwise (`_scimath_impl.py:496`, `x = _fix_real_abs_gt_1(x)`).
#[pyfunction]
pub fn arccos<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_abs_gt_1(py, x, ferray_ufunc::acos_complex, crate::ufunc::arccos)
}

/// `numpy.emath.arcsin(x)` — complex for `abs(x) > 1` (`_scimath_impl.py:543`).
#[pyfunction]
pub fn arcsin<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_abs_gt_1(py, x, ferray_ufunc::asin_complex, crate::ufunc::arcsin)
}

/// `numpy.emath.arctanh(x)` — complex for `abs(x) > 1` (`_scimath_impl.py:591`).
#[pyfunction]
pub fn arctanh<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_abs_gt_1(py, x, ferray_ufunc::atanh_complex, crate::ufunc::arctanh)
}

// ---------------------------------------------------------------------------
// power: _fix_real_lt_zero on base, _fix_int_lt_zero on exponent
// ---------------------------------------------------------------------------

/// `numpy.emath.power(x, p)` — `x ** p`, complex when the base `x` has any
/// real negative component (`_scimath_impl.py:441`, `x = _fix_real_lt_zero(x);
/// p = _fix_int_lt_zero(p); return nx.power(x, p)`).
///
/// Two independent fix-ups:
///   * base `x` → complex if `any(isreal(x) & (x < 0))` (`_fix_real_lt_zero`).
///   * exponent `p` → double (`p * 1.0`) if `any(isreal(p) & (p < 0))`
///     (`_fix_int_lt_zero`, `:148-149`), so `power([2,4], -2)` is float, NOT
///     integer-pow. `p` never becomes complex on its own.
///
/// On the real path the in-domain compute delegates to the real ferray
/// `power` binding (after the int→float exponent fix). On the complex path
/// both operands are cast to `complex128`, broadcast via numpy, and raised
/// element-wise with `Complex::powc` — the faithful analog of
/// `nx.power(complex_x, p)` (R-DEV-7).
#[pyfunction]
pub fn power<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    p: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_x = as_ndarray(py, x)?;
    let dt_x = dtype_name(&arr_x)?;
    let base_complex = is_complex_dtype(dt_x.as_str()) || any_real_lt_zero(py, &arr_x)?;

    if base_complex {
        // _fix_real_lt_zero(x): cast base to complex128.
        let xc = if is_complex_dtype(dt_x.as_str()) {
            arr_x
        } else {
            coerce_dtype(py, &arr_x, "complex128")?
        };
        // numpy then does nx.power(complex_x, p): p is broadcast against x and
        // promoted to complex by the ufunc. Cast p to complex128 and broadcast
        // both via numpy so the element-wise powc sees aligned buffers.
        let pc = coerce_dtype(py, p, "complex128")?;
        let np = py.import("numpy")?;
        let bcast = np.call_method1("broadcast_arrays", (xc, pc))?;
        let xb = bcast.get_item(0)?;
        let pb = bcast.get_item(1)?;
        // Materialise a C-contiguous owned copy so the typed view extraction
        // sees a dense buffer. `np.array(..., order="C")` PRESERVES a 0-d shape
        // (unlike `ascontiguousarray`, which promotes 0-d to shape `(1,)` and
        // would block the scalar collapse below).
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("order", "C")?;
        let xb = np.call_method("array", (xb,), Some(&kwargs))?;
        let pb = np.call_method("array", (pb,), Some(&kwargs))?;
        let base: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&xb)?;
        let expo: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&pb)?;
        let data: Vec<Complex<f64>> = base
            .iter()
            .zip(expo.iter())
            .map(|(z, w)| z.powc(*w))
            .collect();
        let r =
            ArrayD::<Complex<f64>>::from_vec(base.dim().clone(), data).map_err(ferr_to_pyerr)?;
        let out = complex_ferray_to_pyarray::<f64>(py, r)?;
        // Scalar collapse when both inputs were scalar / 0-d.
        if all_scalar_inputs(py, &[x, p])? {
            return scalarize(out);
        }
        return Ok(out);
    }

    // Real path. _fix_int_lt_zero(p): if any p < 0, promote p to float64 so
    // the negative-exponent case is true-division, not integer pow.
    let arr_p = as_ndarray(py, p)?;
    let p_fixed: Bound<'py, PyAny> = if any_real_lt_zero(py, &arr_p)? {
        coerce_dtype(py, &arr_p, "float64")?
    } else {
        arr_p
    };
    crate::ufunc::power(py, x, &p_fixed)
}
