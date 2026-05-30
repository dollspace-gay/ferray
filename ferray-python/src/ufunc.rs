//! Bindings for the `numpy` ufunc surface — elementwise math,
//! comparison, logical, and predicate functions.
//!
//! The function families fall into a small number of dispatch shapes:
//!
//! - **unary float**: `T: Float`. Body: extract `PyReadonlyArrayDyn<T>`
//!   → `ArrayD<T>` → call ferray fn → return ndarray. Macro:
//!   [`match_dtype_float`].
//! - **unary numeric**: `T: Numeric`. Same pipe; macro:
//!   [`match_dtype_numeric`].
//! - **binary numeric (broadcasting)**: ferray exposes
//!   `add_broadcast` / `subtract_broadcast` etc that accept different
//!   `D1`/`D2` dimensionalities, so we coerce both inputs to ArrayD<T>
//!   on the same dtype (sniffed from the first input, the second is
//!   coerced via `numpy.asarray(b, dtype)`).
//! - **comparison (broadcasting → bool)**: ferray returns
//!   `Array<bool, IxDyn>`, which we ship back as `bool` ndarray.
//! - **predicate float → bool**: e.g. `isnan`, returns `Array<bool, D>`.
//! - **logical**: any input dtype implementing the `Logical` trait;
//!   returns bool array.
//!
//! Adding a new function is one binding plus one registration in
//! `lib.rs`. Adding a new dtype is one new arm in the relevant
//! `match_dtype_*` macro and zero changes here.

use ferray_core::array::aliases::ArrayD;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{as_ndarray, coerce_dtype, dtype_name, ferr_to_pyerr};
use crate::{match_dtype_all, match_dtype_float, match_dtype_numeric};

// ---------------------------------------------------------------------------
// Helper: per-dispatch-shape inline macros that capture the full
// extract → ferray-call → return-ndarray pipeline. Defined locally so
// each binding is one expression.
// ---------------------------------------------------------------------------

/// Unary float ufunc body: in `Array<T, IxDyn>` (T: Float) → out same.
macro_rules! unary_float_body {
    ($py:expr, $arr:expr, $func:path) => {{
        let dt = dtype_name(&$arr)?;
        match_dtype_float!(dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = $arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = $func(&fa).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Unary float predicate body: in float → out `Array<bool, IxDyn>`.
macro_rules! unary_float_predicate_body {
    ($py:expr, $arr:expr, $func:path) => {{
        let dt = dtype_name(&$arr)?;
        match_dtype_float!(dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = $arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<bool> = $func(&fa).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Binary numeric broadcast body: dispatch on first arg's dtype, coerce
/// the second to match, call `func(a, b)`.
macro_rules! binary_numeric_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr_a)?;
        match_dtype_numeric!(dt.as_str(), T => {
            let arr_b = coerce_dtype($py, $b, dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            // Let the result dtype follow the op: same-type ops (add/sub/mul)
            // return `ArrayD<T>`, but `divide` is true-division and returns the
            // promoted float output (`ferray_ufunc::TrueDivide::Output`), so the
            // bound type must not be pinned to `T` (numpy divide int->float64).
            let r = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Binary float same-shape body (e.g. power, maximum). ferray's
/// non-broadcast variants require matching shapes, so we coerce both
/// to the first's shape via `numpy.broadcast_arrays` first.
macro_rules! binary_float_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr_a)?;
        match_dtype_float!(dt.as_str(), T => {
            let np = $py.import("numpy")?;
            // numpy.broadcast_arrays returns a list of views with a
            // common shape; we then coerce dtype to align both inputs.
            let pair = np.call_method1("broadcast_arrays", (&arr_a, $b))?;
            let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
            let arr_a2 = coerce_dtype($py, &pair_list[0], dt.as_str())?;
            let arr_b2 = coerce_dtype($py, &pair_list[1], dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b2.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Binary comparison body: numeric inputs → bool output, broadcasting.
macro_rules! comparison_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr_a)?;
        match_dtype_numeric!(dt.as_str(), T => {
            let arr_b = coerce_dtype($py, $b, dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<bool> = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Logical-binary body: any dtype implementing Logical → bool output.
macro_rules! logical_binary_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr_a)?;
        match_dtype_all!(dt.as_str(), T => {
            let arr_b = coerce_dtype($py, $b, dt.as_str())?;
            let np = $py.import("numpy")?;
            let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
            let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
            let va: PyReadonlyArrayDyn<T> = pair_list[0].extract()?;
            let vb: PyReadonlyArrayDyn<T> = pair_list[1].extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<bool> = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Logical-unary body: any dtype implementing Logical → bool output.
macro_rules! logical_unary_body {
    ($py:expr, $a:expr, $func:path) => {{
        let arr = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr)?;
        match_dtype_all!(dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<bool> = $func(&fa).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

// ---------------------------------------------------------------------------
// Trigonometric (unary float)
// ---------------------------------------------------------------------------

macro_rules! bind_unary_float {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            Ok(unary_float_body!(py, arr, $ferr_path))
        }
    };
}

bind_unary_float!(sin, ferray_ufunc::sin);
bind_unary_float!(cos, ferray_ufunc::cos);
bind_unary_float!(tan, ferray_ufunc::tan);
bind_unary_float!(arcsin, ferray_ufunc::arcsin);
bind_unary_float!(arccos, ferray_ufunc::arccos);
bind_unary_float!(arctan, ferray_ufunc::arctan);
bind_unary_float!(sinh, ferray_ufunc::sinh);
bind_unary_float!(cosh, ferray_ufunc::cosh);
bind_unary_float!(tanh, ferray_ufunc::tanh);
bind_unary_float!(arcsinh, ferray_ufunc::arcsinh);
bind_unary_float!(arccosh, ferray_ufunc::arccosh);
bind_unary_float!(arctanh, ferray_ufunc::arctanh);
bind_unary_float!(degrees, ferray_ufunc::degrees);
bind_unary_float!(radians, ferray_ufunc::radians);
bind_unary_float!(deg2rad, ferray_ufunc::deg2rad);
bind_unary_float!(rad2deg, ferray_ufunc::rad2deg);

// Exponential / logarithmic
bind_unary_float!(exp, ferray_ufunc::exp);
bind_unary_float!(exp2, ferray_ufunc::exp2);
bind_unary_float!(expm1, ferray_ufunc::expm1);
bind_unary_float!(log, ferray_ufunc::log);
bind_unary_float!(log1p, ferray_ufunc::log1p);
bind_unary_float!(log2, ferray_ufunc::log2);
bind_unary_float!(log10, ferray_ufunc::log10);

// Roots / squares / reciprocal — all float-only in ferray
bind_unary_float!(sqrt, ferray_ufunc::sqrt);
bind_unary_float!(cbrt, ferray_ufunc::cbrt);
bind_unary_float!(square, ferray_ufunc::square);
bind_unary_float!(reciprocal, ferray_ufunc::reciprocal);

// Sign / absolute / negative — float-only in ferray
bind_unary_float!(negative, ferray_ufunc::negative);
bind_unary_float!(positive, ferray_ufunc::positive);
bind_unary_float!(absolute, ferray_ufunc::absolute);
bind_unary_float!(fabs, ferray_ufunc::fabs);
bind_unary_float!(sign, ferray_ufunc::sign);

// Rounding — all float-only
bind_unary_float!(floor, ferray_ufunc::floor);
bind_unary_float!(ceil, ferray_ufunc::ceil);
bind_unary_float!(round, ferray_ufunc::round);
bind_unary_float!(trunc, ferray_ufunc::trunc);
bind_unary_float!(rint, ferray_ufunc::rint);
bind_unary_float!(fix, ferray_ufunc::fix);

// `np.abs` is just an alias for `np.absolute`. ferray-Rust's `abs`
// is the complex-absolute (takes `Array<Complex<T>>`), so we don't
// bind it directly — `abs` is exported as a Python-level alias of
// `absolute` from `python/ferray/__init__.py`.

// ---------------------------------------------------------------------------
// Predicates: float → bool
// ---------------------------------------------------------------------------

macro_rules! bind_predicate_float {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            Ok(unary_float_predicate_body!(py, arr, $ferr_path))
        }
    };
}

bind_predicate_float!(isnan, ferray_ufunc::isnan);
bind_predicate_float!(isinf, ferray_ufunc::isinf);
bind_predicate_float!(isfinite, ferray_ufunc::isfinite);
bind_predicate_float!(isneginf, ferray_ufunc::isneginf);
bind_predicate_float!(isposinf, ferray_ufunc::isposinf);
bind_predicate_float!(signbit, ferray_ufunc::signbit);

// ---------------------------------------------------------------------------
// Binary arithmetic (broadcasting, numeric inputs)
// ---------------------------------------------------------------------------

macro_rules! bind_binary_numeric_broadcast {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            Ok(binary_numeric_body!(py, x1, x2, $ferr_path))
        }
    };
}

bind_binary_numeric_broadcast!(add, ferray_ufunc::add_broadcast);
bind_binary_numeric_broadcast!(subtract, ferray_ufunc::subtract_broadcast);
bind_binary_numeric_broadcast!(multiply, ferray_ufunc::multiply_broadcast);
bind_binary_numeric_broadcast!(divide, ferray_ufunc::divide_broadcast);

// ---------------------------------------------------------------------------
// Binary float (broadcasting via numpy.broadcast_arrays + same-shape
// ferray fn) — power, maximum, minimum, fmax, fmin, copysign, hypot, arctan2
// ---------------------------------------------------------------------------

macro_rules! bind_binary_float {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            Ok(binary_float_body!(py, x1, x2, $ferr_path))
        }
    };
}

bind_binary_float!(power, ferray_ufunc::power);
bind_binary_float!(maximum, ferray_ufunc::maximum);
bind_binary_float!(minimum, ferray_ufunc::minimum);
bind_binary_float!(fmax, ferray_ufunc::fmax);
bind_binary_float!(fmin, ferray_ufunc::fmin);
bind_binary_float!(copysign, ferray_ufunc::copysign);
bind_binary_float!(hypot, ferray_ufunc::hypot);
bind_binary_float!(arctan2, ferray_ufunc::arctan2);
bind_binary_float!(logaddexp, ferray_ufunc::logaddexp);
bind_binary_float!(logaddexp2, ferray_ufunc::logaddexp2);
bind_binary_float!(heaviside, ferray_ufunc::heaviside);

// ---------------------------------------------------------------------------
// Comparisons (broadcasting → bool)
// ---------------------------------------------------------------------------

macro_rules! bind_comparison {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            Ok(comparison_body!(py, x1, x2, $ferr_path))
        }
    };
}

bind_comparison!(equal, ferray_ufunc::equal_broadcast);
bind_comparison!(not_equal, ferray_ufunc::not_equal_broadcast);
bind_comparison!(less, ferray_ufunc::less_broadcast);
bind_comparison!(less_equal, ferray_ufunc::less_equal_broadcast);
bind_comparison!(greater, ferray_ufunc::greater_broadcast);
bind_comparison!(greater_equal, ferray_ufunc::greater_equal_broadcast);

// ---------------------------------------------------------------------------
// Logical (any Logical-implementing dtype → bool)
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn logical_and<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_and))
}

#[pyfunction]
pub fn logical_or<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_or))
}

#[pyfunction]
pub fn logical_xor<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_xor))
}

#[pyfunction]
pub fn logical_not<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    Ok(logical_unary_body!(py, x, ferray_ufunc::logical_not))
}

// ---------------------------------------------------------------------------
// Other (clip, where_)
// ---------------------------------------------------------------------------

/// `numpy.array_equal(a, b)` — true iff same shape and all elements equal.
#[pyfunction]
pub fn array_equal<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_b = coerce_dtype(py, b, dt.as_str())?;
    let dt_b = dtype_name(&arr_b)?;
    if dt != dt_b {
        return Ok(false);
    }
    let result = match_dtype_all!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        if fa.shape() != fb.shape() {
            false
        } else {
            ferray_ufunc::array_equal(&fa, &fb)
        }
    });
    Ok(result)
}

/// `numpy.array_equiv(a, b)` — like `array_equal` but allows broadcasting.
#[pyfunction]
pub fn array_equiv<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let np = py.import("numpy")?;
    // Broadcast both inputs first; if shapes are incompatible, return False.
    let pair = match np.call_method1("broadcast_arrays", (&arr_a, b)) {
        Ok(p) => p,
        Err(_) => return Ok(false),
    };
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let coerced_b = coerce_dtype(py, &bcast[1], dt.as_str())?;
    let result = match_dtype_all!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vb: PyReadonlyArrayDyn<T> = coerced_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_ufunc::array_equiv(&fa, &fb)
    });
    Ok(result)
}

/// `numpy.allclose(a, b, rtol=1e-5, atol=1e-8)`.
#[pyfunction]
#[pyo3(signature = (a, b, rtol = 1e-5, atol = 1e-8))]
pub fn allclose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rtol: f64,
    atol: f64,
) -> PyResult<bool> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    // Promote integer inputs to float64 for the close-comparison.
    let real_dt = if matches!(dt.as_str(), "float64" | "f64" | "float32" | "f32") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_a = coerce_dtype(py, &arr_a, &real_dt)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, b))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_b = coerce_dtype(py, &bcast[1], &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        ferray_ufunc::allclose(&fa, &fb, rtol as T, atol as T).map_err(ferr_to_pyerr)?
    }))
}

/// `numpy.isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False)` — elementwise.
#[pyfunction]
#[pyo3(signature = (a, b, rtol = 1e-5, atol = 1e-8, equal_nan = false))]
pub fn isclose<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    rtol: f64,
    atol: f64,
    equal_nan: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let real_dt = if matches!(dt.as_str(), "float64" | "f64" | "float32" | "f32") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_a = coerce_dtype(py, &arr_a, &real_dt)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, b))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_b = coerce_dtype(py, &bcast[1], &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r = ferray_ufunc::isclose(&fa, &fb, rtol as T, atol as T, equal_nan)
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Bitwise (#699)
// ---------------------------------------------------------------------------

/// Bitwise binary body: integer or bool dtype, both inputs broadcast.
macro_rules! bitwise_binary_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let dt = dtype_name(&arr_a)?;
        let arr_b = coerce_dtype($py, $b, dt.as_str())?;
        let np = $py.import("numpy")?;
        let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
        let bcast: Vec<Bound<PyAny>> = pair.extract()?;
        crate::match_dtype_bitwise!(dt.as_str(), T => {
            let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
            let vb: PyReadonlyArrayDyn<T> = bcast[1].extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

#[pyfunction]
pub fn bitwise_and<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(bitwise_binary_body!(py, x1, x2, ferray_ufunc::bitwise_and))
}

#[pyfunction]
pub fn bitwise_or<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(bitwise_binary_body!(py, x1, x2, ferray_ufunc::bitwise_or))
}

#[pyfunction]
pub fn bitwise_xor<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(bitwise_binary_body!(py, x1, x2, ferray_ufunc::bitwise_xor))
}

/// `numpy.invert(x)` — bitwise NOT (also exposed as `bitwise_not`).
#[pyfunction]
pub fn invert<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(crate::match_dtype_bitwise!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::invert(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

#[pyfunction]
pub fn bitwise_not<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    invert(py, x)
}

/// `numpy.left_shift(x1, x2)` — `x1 << x2`. Shift amount coerced to uint32.
#[pyfunction]
pub fn left_shift<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, x1)?;
    let dt = dtype_name(&arr_a)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, x2))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_b = coerce_dtype(py, &bcast[1], "uint32")?;
    Ok(crate::match_dtype_int_only!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vb: PyReadonlyArrayDyn<u32> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<u32> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::left_shift(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.right_shift(x1, x2)`.
#[pyfunction]
pub fn right_shift<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, x1)?;
    let dt = dtype_name(&arr_a)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, x2))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_b = coerce_dtype(py, &bcast[1], "uint32")?;
    Ok(crate::match_dtype_int_only!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vb: PyReadonlyArrayDyn<u32> = arr_b.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<u32> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::right_shift(&fa, &fb).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Special / signal-processing (#702) — gradient, trapezoid, ediff1d,
// sinc, i0, convolve, correlate, interp.
// ---------------------------------------------------------------------------

bind_unary_float!(sinc, ferray_ufunc::sinc);
bind_unary_float!(i0, ferray_ufunc::i0);

/// `numpy.gradient(f, *, dx=1.0)` — central-difference gradient of a 1-D
/// array. Multi-axis gradient is deferred (numpy's full API takes a list
/// of varargs spacings; ferray-ufunc's gradient is 1-D-only).
#[pyfunction]
#[pyo3(signature = (f, dx = 1.0))]
pub fn gradient<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    dx: f64,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, f)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr = coerce_dtype(py, &arr, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let view: numpy::PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::gradient(&fa, Some(dx as T))
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.trapezoid(y, x=None, dx=1.0)` (formerly `numpy.trapz`).
#[pyfunction]
#[pyo3(signature = (y, x = None, dx = 1.0))]
pub fn trapezoid<'py>(
    py: Python<'py>,
    y: &Bound<'py, PyAny>,
    x: Option<&Bound<'py, PyAny>>,
    dx: f64,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_y = as_ndarray(py, y)?;
    let dt = dtype_name(&arr_y)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_y = coerce_dtype(py, &arr_y, &real_dt)?;
    let arr_x = match x {
        Some(xa) => Some(coerce_dtype(py, xa, &real_dt)?),
        None => None,
    };
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let vy: numpy::PyReadonlyArray1<T> = arr_y.extract()?;
        let fy: Array1<T> = vy.as_ferray().map_err(ferr_to_pyerr)?;
        let scalar: T = match arr_x.as_ref() {
            Some(xa) => {
                let vx: numpy::PyReadonlyArray1<T> = xa.extract()?;
                let fx: Array1<T> = vx.as_ferray().map_err(ferr_to_pyerr)?;
                ferray_ufunc::trapezoid(&fy, Some(&fx), None).map_err(ferr_to_pyerr)?
            }
            None => ferray_ufunc::trapezoid(&fy, None, Some(dx as T))
                .map_err(ferr_to_pyerr)?,
        };
        scalar.into_pyobject(py)?.into_any()
    }))
}

/// `numpy.ediff1d(ary, to_end=None, to_begin=None)` — first differences.
#[pyfunction]
#[pyo3(signature = (ary, to_end = None, to_begin = None))]
pub fn ediff1d<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    to_end: Option<Vec<f64>>,
    to_begin: Option<Vec<f64>>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, ary)?;
    let dt = dtype_name(&arr)?;
    Ok(crate::match_dtype_numeric!(dt.as_str(), T => {
        let view: numpy::PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let end: Option<Vec<T>> = to_end.as_ref().map(|v| v.iter().map(|&x| x as T).collect());
        let begin: Option<Vec<T>> = to_begin.as_ref().map(|v| v.iter().map(|&x| x as T).collect());
        let r: Array1<T> = ferray_ufunc::ediff1d(&fa, end.as_deref(), begin.as_deref())
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

fn parse_convolve_mode(mode: &str) -> PyResult<ferray_ufunc::ConvolveMode> {
    Ok(match mode {
        "full" => ferray_ufunc::ConvolveMode::Full,
        "same" => ferray_ufunc::ConvolveMode::Same,
        "valid" => ferray_ufunc::ConvolveMode::Valid,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "convolve mode must be 'full'|'same'|'valid', got {other:?}"
            )));
        }
    })
}

fn parse_correlate_mode(mode: &str) -> PyResult<ferray_stats::CorrelateMode> {
    Ok(match mode {
        "full" => ferray_stats::CorrelateMode::Full,
        "same" => ferray_stats::CorrelateMode::Same,
        "valid" => ferray_stats::CorrelateMode::Valid,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "correlate mode must be 'full'|'same'|'valid', got {other:?}"
            )));
        }
    })
}

/// `numpy.convolve(a, v, mode='full')` — discrete linear convolution.
#[pyfunction]
#[pyo3(signature = (a, v, mode = "full"))]
pub fn convolve<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let m = parse_convolve_mode(mode)?;
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let arr_v = coerce_dtype(py, v, dt.as_str())?;
    Ok(crate::match_dtype_numeric!(dt.as_str(), T => {
        let va: numpy::PyReadonlyArray1<T> = arr_a.extract()?;
        let vv: numpy::PyReadonlyArray1<T> = arr_v.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: Array1<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::convolve(&fa, &fv, m).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.correlate(a, v, mode='valid')` — cross-correlation.
#[pyfunction]
#[pyo3(signature = (a, v, mode = "valid"))]
pub fn correlate<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    mode: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let m = parse_correlate_mode(mode)?;
    let arr_a = as_ndarray(py, a)?;
    let dt = dtype_name(&arr_a)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_a = coerce_dtype(py, &arr_a, &real_dt)?;
    let arr_v = coerce_dtype(py, v, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let va: numpy::PyReadonlyArray1<T> = arr_a.extract()?;
        let vv: numpy::PyReadonlyArray1<T> = arr_v.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: Array1<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::correlate(&fa, &fv, m).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.interp(x, xp, fp)` — 1-D linear interpolation.
#[pyfunction]
pub fn interp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    xp: &Bound<'py, PyAny>,
    fp: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = as_ndarray(py, x)?;
    let dt = dtype_name(&arr_x)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_x = coerce_dtype(py, &arr_x, &real_dt)?;
    let arr_xp = coerce_dtype(py, xp, &real_dt)?;
    let arr_fp = coerce_dtype(py, fp, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let vx: numpy::PyReadonlyArray1<T> = arr_x.extract()?;
        let vxp: numpy::PyReadonlyArray1<T> = arr_xp.extract()?;
        let vfp: numpy::PyReadonlyArray1<T> = arr_fp.extract()?;
        let fx: Array1<T> = vx.as_ferray().map_err(ferr_to_pyerr)?;
        let fxp: Array1<T> = vxp.as_ferray().map_err(ferr_to_pyerr)?;
        let ffp: Array1<T> = vfp.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::interp(&fx, &fxp, &ffp).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// Float intrinsics (#701) — gcd, lcm, fmod, divmod, frexp, ldexp, modf,
// spacing, nextafter, exp_fast.
// ---------------------------------------------------------------------------

bind_unary_float!(exp_fast, ferray_ufunc::exp_fast);
bind_unary_float!(spacing, ferray_ufunc::spacing);
bind_binary_float!(gcd, ferray_ufunc::gcd);
bind_binary_float!(lcm, ferray_ufunc::lcm);
bind_binary_float!(fmod, ferray_ufunc::fmod);
bind_binary_float!(nextafter, ferray_ufunc::nextafter);

/// `numpy.divmod(x1, x2)` → tuple `(quotient, remainder)`.
#[pyfunction]
pub fn divmod<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, x1)?;
    let dt = dtype_name(&arr_a)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, x2))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_a2 = coerce_dtype(py, &bcast[0], dt.as_str())?;
    let arr_b2 = coerce_dtype(py, &bcast[1], dt.as_str())?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
        let vb: PyReadonlyArrayDyn<T> = arr_b2.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
        let (q, r): (ArrayD<T>, ArrayD<T>) =
            ferray_ufunc::divmod(&fa, &fb).map_err(ferr_to_pyerr)?;
        let q_py = q.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let r_py = r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        pyo3::types::PyTuple::new(py, [q_py, r_py])?.into_any()
    }))
}

/// `numpy.frexp(x)` → tuple `(mantissa, exponent)`.
#[pyfunction]
pub fn frexp<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
        let (mant, exp): (ArrayD<T>, ArrayD<i32>) =
            ferray_ufunc::frexp(&fa).map_err(ferr_to_pyerr)?;
        let m_py = mant.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let e_py = exp.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        pyo3::types::PyTuple::new(py, [m_py, e_py])?.into_any()
    }))
}

/// `numpy.ldexp(x, n)` → x * 2**n.
#[pyfunction]
pub fn ldexp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr_a = as_ndarray(py, x)?;
    let dt = dtype_name(&arr_a)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_a = coerce_dtype(py, &arr_a, &real_dt)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, n))?;
    let bcast: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_n = coerce_dtype(py, &bcast[1], "int32")?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let va: PyReadonlyArrayDyn<T> = bcast[0].extract()?;
        let vn: PyReadonlyArrayDyn<i32> = arr_n.extract()?;
        let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fn_arr: ArrayD<i32> = vn.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::ldexp(&fa, &fn_arr).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.modf(x)` → tuple `(fractional, integer)` parts.
#[pyfunction]
pub fn modf<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
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
        let (frac, int_part): (ArrayD<T>, ArrayD<T>) =
            ferray_ufunc::modf(&fa).map_err(ferr_to_pyerr)?;
        let f_py = frac.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        let i_py = int_part.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
        pyo3::types::PyTuple::new(py, [f_py, i_py])?.into_any()
    }))
}

/// `numpy.bitwise_count(x)` — popcount (number of 1-bits).
#[pyfunction]
pub fn bitwise_count<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    Ok(crate::match_dtype_int_only!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<u32> = ferray_ufunc::bitwise_count(&fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.clip(a, a_min, a_max)` — limit values to `[a_min, a_max]`.
#[pyfunction]
pub fn clip<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    a_min: f64,
    a_max: f64,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let lo = a_min as T;
        let hi = a_max as T;
        let r: ArrayD<T> = ferray_ufunc::clip(&fa, lo, hi).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}
