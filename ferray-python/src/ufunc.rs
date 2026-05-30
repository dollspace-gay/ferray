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

use crate::conv::{
    all_scalar_inputs, as_ndarray, binary_result_dtype, coerce_dtype, dtype_name, ferr_to_pyerr,
    scalarize, unary_promote_dtypes,
};
use crate::{
    match_dtype_all, match_dtype_float, match_dtype_float_or_int, match_dtype_int_only,
    match_dtype_numeric,
};

// ---------------------------------------------------------------------------
// Helper: per-dispatch-shape inline macros that capture the full
// extract → ferray-call → return-ndarray pipeline. Defined locally so
// each binding is one expression.
// ---------------------------------------------------------------------------

/// Unary "promote-to-float" ufunc body (REQ-23): float input keeps its
/// dtype; integer/bool input promotes to the NumPy-promoted float dtype
/// (`result_type(x, float16)`). The promotion happens at the Python
/// boundary — the input is coerced to the *compute* float (f32/f64), the
/// existing `T: Float` kernel runs, and the result is narrowed to the
/// *output* float dtype (which may be float16) by numpy. This keeps the
/// returned array's dtype byte-for-byte numpy-correct without needing
/// `half::f16` plumbing inside Rust.
macro_rules! unary_float_body {
    ($py:expr, $arr:expr, $func:path) => {{
        let in_dt = dtype_name(&$arr)?;
        let (compute_dt, out_dt) = unary_promote_dtypes($py, &$arr, in_dt.as_str())?;
        let arr_c = coerce_dtype($py, &$arr, compute_dt.as_str())?;
        let result = match_dtype_float!(compute_dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr_c.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = $func(&fa).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        });
        // Narrow to numpy's exact output dtype (identity when compute == out).
        if out_dt != compute_dt {
            coerce_dtype($py, &result, out_dt.as_str())?
        } else {
            result
        }
    }};
}

/// Unary algebraic body that PRESERVES the input numeric dtype (REQ-24 /
/// int-identity family: `negative`, `absolute`, `sign`, `floor`, `ceil`,
/// `trunc`, `fix`). Float dtypes route to `$float_fn`, integer dtypes to
/// `$int_fn` (which keeps the integer dtype).
macro_rules! unary_numeric_split_body {
    ($py:expr, $arr:expr, $float_fn:path, $int_fn:path) => {{
        let dt = dtype_name(&$arr)?;
        match_dtype_float_or_int!(dt.as_str(), T, $float_fn, $int_fn => {
            let view: PyReadonlyArrayDyn<T> = $arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = __op!()(&fa).map_err(ferr_to_pyerr)?;
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

/// Binary numeric broadcast body (`add`/`subtract`/`multiply`/`divide`):
/// promote BOTH inputs to the NEP-50 common dtype (`result_type(a, b)`),
/// not the first operand's dtype, then call `func(a, b)`. Promoting to
/// the first operand's dtype would truncate the wider operand —
/// `add(int[1,2], float[1.5,2.5])` must yield `float64 [2.5,4.5]`, not
/// `int64 [2,4]`. The result dtype follows the op: `add`/`subtract`/
/// `multiply` return the common dtype; `divide` is true-division and
/// returns its promoted float output (numpy `int/int -> float64`).
macro_rules! binary_numeric_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let arr_b0 = as_ndarray($py, $b)?;
        let dt = binary_result_dtype($py, &arr_a, &arr_b0)?;
        match_dtype_numeric!(dt.as_str(), T => {
            let arr_a2 = coerce_dtype($py, &arr_a, dt.as_str())?;
            let arr_b = coerce_dtype($py, &arr_b0, dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Binary "promote-to-float" body (`hypot`, `arctan2`, `copysign`,
/// `logaddexp`, …): numpy registers ONLY float loops, so any integer
/// input promotes to float (`result_type(a, b, float16)`). Computed in
/// f32/f64 at the boundary and narrowed to numpy's exact output dtype.
macro_rules! binary_float_promote_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let arr_b = as_ndarray($py, $b)?;
        // Promote inputs together (covers int+int -> f64, int8+int8 -> f16).
        let common = binary_result_dtype($py, &arr_a, &arr_b)?;
        let (compute_dt, out_dt) = unary_promote_dtypes($py, &arr_a, common.as_str())?;
        let np = $py.import("numpy")?;
        let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
        let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
        let arr_a2 = coerce_dtype($py, &pair_list[0], compute_dt.as_str())?;
        let arr_b2 = coerce_dtype($py, &pair_list[1], compute_dt.as_str())?;
        let result = match_dtype_float!(compute_dt.as_str(), T => {
            let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b2.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = $func(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        });
        if out_dt != compute_dt {
            coerce_dtype($py, &result, out_dt.as_str())?
        } else {
            result
        }
    }};
}

/// Binary body for ops that have SEPARATE float and integer loops with
/// the SAME output kind (`maximum`/`minimum`, `power`, `floor_divide`,
/// `remainder`/`mod`): promote both inputs to the common numeric dtype
/// (`result_type(a, b)`), then route float dtypes to `$float_fn` and
/// integer dtypes to `$int_fn`.
macro_rules! binary_numeric_split_body {
    ($py:expr, $a:expr, $b:expr, $float_fn:path, $int_fn:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let arr_b = as_ndarray($py, $b)?;
        let dt = binary_result_dtype($py, &arr_a, &arr_b)?;
        let np = $py.import("numpy")?;
        let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
        let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
        match_dtype_float_or_int!(dt.as_str(), T, $float_fn, $int_fn => {
            let arr_a2 = coerce_dtype($py, &pair_list[0], dt.as_str())?;
            let arr_b2 = coerce_dtype($py, &pair_list[1], dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
            let vb: PyReadonlyArrayDyn<T> = arr_b2.extract()?;
            let fa: ArrayD<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
            let fb: ArrayD<T> = vb.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = __op!()(&fa, &fb).map_err(ferr_to_pyerr)?;
            r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
        })
    }};
}

/// Binary comparison body: numeric inputs → bool output, broadcasting.
/// Both inputs promote to the NEP-50 common dtype (`result_type(a, b)`),
/// so `equal(int, float)` compares the float-promoted values rather than
/// truncating the float operand to the first's integer dtype.
macro_rules! comparison_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let arr_b0 = as_ndarray($py, $b)?;
        let dt = binary_result_dtype($py, &arr_a, &arr_b0)?;
        match_dtype_numeric!(dt.as_str(), T => {
            let arr_a2 = coerce_dtype($py, &arr_a, dt.as_str())?;
            let arr_b = coerce_dtype($py, &arr_b0, dt.as_str())?;
            let va: PyReadonlyArrayDyn<T> = arr_a2.extract()?;
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

/// Bind a unary "promote-to-float" ufunc (REQ-23): integer/bool input
/// promotes to float, float input keeps its dtype. Scalar / 0-d input
/// returns a numpy scalar (`$OUT_SCALAR`).
macro_rules! bind_unary_float {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            let out = unary_float_body!(py, arr, $ferr_path);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

/// Bind a unary algebraic ufunc that PRESERVES the input numeric dtype
/// (REQ-24 int-identity family): float input → `$float_fn`, integer input
/// → `$int_fn` (keeping the integer dtype).
macro_rules! bind_unary_numeric_split {
    ($name:ident, $float_fn:path, $int_fn:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            let out = unary_numeric_split_body!(py, arr, $float_fn, $int_fn);
            if scalar { scalarize(out) } else { Ok(out) }
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

// Exponential / logarithmic (REQ-23: int -> float)
bind_unary_float!(exp, ferray_ufunc::exp);
bind_unary_float!(exp2, ferray_ufunc::exp2);
bind_unary_float!(expm1, ferray_ufunc::expm1);
bind_unary_float!(log, ferray_ufunc::log);
bind_unary_float!(log1p, ferray_ufunc::log1p);
bind_unary_float!(log2, ferray_ufunc::log2);
bind_unary_float!(log10, ferray_ufunc::log10);

// Roots — REQ-23: int -> float.
bind_unary_float!(sqrt, ferray_ufunc::sqrt);
bind_unary_float!(cbrt, ferray_ufunc::cbrt);
// `rint` has NO integer loop in numpy (generate_umath.py:1021) — int -> float.
bind_unary_float!(rint, ferray_ufunc::rint);
// `fabs` registers only float loops — int -> float (generate_umath.py:1003).
bind_unary_float!(fabs, ferray_ufunc::fabs);

// Sign / absolute / negative / positive — int-identity (int -> int).
bind_unary_numeric_split!(negative, ferray_ufunc::negative, ferray_ufunc::negative_int);
bind_unary_numeric_split!(absolute, ferray_ufunc::absolute, ferray_ufunc::absolute_int);
bind_unary_numeric_split!(sign, ferray_ufunc::sign, ferray_ufunc::sign_int);

// Rounding floor/ceil/trunc/fix — int-identity (REQ-24, int -> int).
bind_unary_numeric_split!(floor, ferray_ufunc::floor, ferray_ufunc::floor_int);
bind_unary_numeric_split!(ceil, ferray_ufunc::ceil, ferray_ufunc::ceil_int);
bind_unary_numeric_split!(trunc, ferray_ufunc::trunc, ferray_ufunc::trunc_int);
bind_unary_numeric_split!(fix, ferray_ufunc::fix, ferray_ufunc::fix_int);

// `round` keeps int dtype (generate_umath.py `TD(bints)` on `rint`/`around`'s
// int loops); float route is `ferray_ufunc::round`.
bind_unary_numeric_split!(round, ferray_ufunc::round, ferray_ufunc::round_int);

/// `numpy.around(a, decimals=0)` / `numpy.round(a, decimals=0)` — round to
/// `decimals` places with half-to-even (banker's) rounding.
///
/// numpy/_core/fromnumeric.py:3343 `around` documents "For values exactly
/// halfway between rounded decimal values, NumPy rounds to the nearest even
/// value", implemented as `multiply(a, 10**decimals)`, round-half-even, then
/// `divide` back (numpy/_core/src/multiarray/calculation.c `_round`). The
/// `decimals == 0` case is the existing `round`/`rint` half-to-even kernel;
/// for `decimals != 0` the binding scales by `10**decimals`, applies the
/// half-to-even `ferray_ufunc::rint`, and unscales — matching numpy's
/// `np.rint(a * 10**d) / 10**d`. Integer input with `decimals >= 0` is
/// returned unchanged (numpy: rounding an int to >= 0 places is the
/// identity, dtype preserved). The scaled-round path computes in float64.
#[pyfunction]
#[pyo3(signature = (a, decimals = 0))]
pub fn around<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    decimals: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    let is_int = !matches!(
        dt.as_str(),
        "float64" | "f64" | "float32" | "f32" | "float16" | "f16"
    );
    // Integer input rounded to >= 0 decimal places is the identity.
    if is_int && decimals >= 0 {
        return Ok(arr);
    }
    // decimals == 0 on a float keeps the float dtype and is exactly rint.
    if decimals == 0 && !is_int {
        return Ok(match_dtype_float!(dt.as_str(), T => {
            let view: PyReadonlyArrayDyn<T> = arr.extract()?;
            let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
            let r: ArrayD<T> = ferray_ufunc::rint(&fa).map_err(ferr_to_pyerr)?;
            r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        }));
    }
    // General path: scale by 10**decimals, half-to-even round, unscale.
    // Integer input with negative decimals also flows here; numpy returns the
    // input dtype, so we narrow the float64 result back at the end.
    let arr_f = coerce_dtype(py, &arr, "float64")?;
    let view: PyReadonlyArrayDyn<f64> = arr_f.extract()?;
    let fa: ArrayD<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let scale = 10f64.powi(decimals);
    let scaled: Vec<f64> = fa.iter().map(|&x| x * scale).collect();
    let shape = fa.shape().to_vec();
    let scaled_arr = ArrayD::<f64>::from_vec(ferray_core::dimension::IxDyn::new(&shape), scaled)
        .map_err(ferr_to_pyerr)?;
    let rounded: ArrayD<f64> = ferray_ufunc::rint(&scaled_arr).map_err(ferr_to_pyerr)?;
    let unscaled: Vec<f64> = rounded.iter().map(|&x| x / scale).collect();
    let out = ArrayD::<f64>::from_vec(ferray_core::dimension::IxDyn::new(&shape), unscaled)
        .map_err(ferr_to_pyerr)?;
    let result = out.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any();
    // Integer input keeps its integer dtype (numpy narrows back).
    if is_int {
        coerce_dtype(py, &result, dt.as_str())
    } else if !matches!(dt.as_str(), "float64" | "f64") {
        // float32 input keeps float32.
        coerce_dtype(py, &result, dt.as_str())
    } else {
        Ok(result)
    }
}

/// `numpy.nan_to_num(x, nan=0.0, posinf=None, neginf=None)` — replace NaN
/// with `nan`, +Inf with `posinf` (default the dtype's largest finite), and
/// -Inf with `neginf` (default the most negative finite)
/// (numpy/lib/_type_check_impl.py:382). The library
/// `ferray_ufunc::nan_to_num` takes the same three optional replacements.
/// Integer input has no NaN/Inf, so it is returned unchanged (numpy returns
/// the input dtype). Float input keeps its float dtype.
#[pyfunction]
#[pyo3(signature = (x, nan = 0.0, posinf = None, neginf = None))]
pub fn nan_to_num<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    nan: f64,
    posinf: Option<f64>,
    neginf: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    // Integer / bool input has no NaN or Inf — numpy returns it unchanged.
    if !matches!(
        dt.as_str(),
        "float64" | "f64" | "float32" | "f32" | "float16" | "f16"
    ) {
        return Ok(arr);
    }
    Ok(match_dtype_float!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        // `T` is a concrete `f32`/`f64` alias in each arm, so the `as` cast
        // is the exact narrowing numpy applies when the replacement is stored
        // in the output dtype.
        let nan_t = nan as T;
        let posinf_t = posinf.map(|v| v as T);
        let neginf_t = neginf.map(|v| v as T);
        let r: ArrayD<T> =
            ferray_ufunc::nan_to_num(&fa, Some(nan_t), posinf_t, neginf_t).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.unwrap(p, discont=None, axis=-1, period=2*pi)` — unwrap a phase
/// angle by changing deltas greater than `discont` to their `2*pi`
/// complement (numpy/lib/_function_base_impl.py:1734). The library
/// `ferray_ufunc::unwrap` is the 1-D unwrap with `discont` defaulting to pi.
/// Integer/bool input promotes to `float64` (numpy computes in inexact).
#[pyfunction]
#[pyo3(signature = (p, discont = None))]
pub fn unwrap<'py>(
    py: Python<'py>,
    p: &Bound<'py, PyAny>,
    discont: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, p)?;
    let dt = dtype_name(&arr)?;
    let real_dt = if matches!(dt.as_str(), "float32" | "f32") {
        "float32"
    } else {
        "float64"
    };
    let arr = coerce_dtype(py, &arr, real_dt)?;
    Ok(match_dtype_float!(real_dt, T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let d: Option<T> = discont.map(|v| v as T);
        let r: ArrayD<T> = ferray_ufunc::unwrap(&fa, d).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `positive` / `square` / `reciprocal` keep the integer dtype but have no
/// int kernel in ferray-ufunc; numpy's integer semantics are: `positive` =
/// identity, `square` = `x*x` (wrapping), `reciprocal` = `1 // x`
/// (generate_umath.py:516/:524/:540 `TD(ints + flts)`). Float input routes
/// to the float kernel. These are handled in dedicated `#[pyfunction]`s
/// below rather than the split macro because the int branch reuses other
/// ferray-ufunc integer ops instead of a single `_int` entry point.
macro_rules! bind_unary_float_only_int_fallback {
    ($name:ident, $ferr_path:path, $int_body:expr) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            let dt = dtype_name(&arr)?;
            let out = if matches!(
                dt.as_str(),
                "float64" | "f64" | "float32" | "f32"
            ) {
                match_dtype_float!(dt.as_str(), T => {
                    let view: PyReadonlyArrayDyn<T> = arr.extract()?;
                    let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
                    let r: ArrayD<T> = $ferr_path(&fa).map_err(ferr_to_pyerr)?;
                    r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
                })
            } else {
                let int_fn: fn(Python<'py>, &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> =
                    $int_body;
                int_fn(py, &arr)?
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

/// `positive(int)` = identity: return the input array unchanged.
fn positive_int_array<'py>(
    _py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(arr.clone())
}

/// `square(int)` = `x * x` with wrapping (numpy's fixed-width int square),
/// computed via ferray's integer multiply.
fn square_int_array<'py>(py: Python<'py>, arr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let dt = dtype_name(arr)?;
    Ok(match_dtype_int_only!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::multiply_broadcast(&fa, &fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `reciprocal(int)` = `1 // x` (numpy truncating integer reciprocal),
/// computed via ferray's integer floor-divide.
fn reciprocal_int_array<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dt = dtype_name(arr)?;
    Ok(match_dtype_int_only!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let ones: ArrayD<T> = ArrayD::<T>::from_elem(fa.dim().clone(), <T as ferray_core::Element>::one())
            .map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_ufunc::floor_divide_int(&ones, &fa).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

bind_unary_float_only_int_fallback!(positive, ferray_ufunc::positive, positive_int_array);
bind_unary_float_only_int_fallback!(square, ferray_ufunc::square, square_int_array);
bind_unary_float_only_int_fallback!(reciprocal, ferray_ufunc::reciprocal, reciprocal_int_array);

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

/// Write a computed ufunc result into a caller-supplied `out=` ndarray
/// (numpy's `$OUT` kwarg contract — every binary ufunc accepts
/// `out : ndarray, None, or tuple`), then return `out`. The assignment
/// goes through numpy's `ndarray.__setitem__` so dtype casting matches
/// numpy's `out=` semantics. When `out` is absent the freshly-built
/// `result` is returned (scalarized for all-scalar inputs).
fn finish_with_out<'py>(
    out: Option<&Bound<'py, PyAny>>,
    result: Bound<'py, PyAny>,
    scalar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    match out {
        Some(target) if !target.is_none() => {
            let py = result.py();
            // `numpy.copyto(dst, src, casting="unsafe")` writes the result in
            // place with numpy's `out=` casting rules and broadcasting.
            let np = py.import("numpy")?;
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("casting", "unsafe")?;
            np.call_method("copyto", (target, &result), Some(&kwargs))?;
            Ok(target.clone())
        }
        _ if scalar => scalarize(result),
        _ => Ok(result),
    }
}

macro_rules! bind_binary_numeric_broadcast {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        #[pyo3(signature = (x1, x2, out = None))]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
            out: Option<&Bound<'py, PyAny>>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let result = binary_numeric_body!(py, x1, x2, $ferr_path);
            finish_with_out(out, result, scalar)
        }
    };
}

bind_binary_numeric_broadcast!(add, ferray_ufunc::add_broadcast);
bind_binary_numeric_broadcast!(subtract, ferray_ufunc::subtract_broadcast);
bind_binary_numeric_broadcast!(multiply, ferray_ufunc::multiply_broadcast);
bind_binary_numeric_broadcast!(divide, ferray_ufunc::divide_broadcast);

// ---------------------------------------------------------------------------
// Binary float-promote (numpy registers ONLY float loops; int input
// promotes to float) — fmax, fmin, copysign, hypot, arctan2, logaddexp,
// logaddexp2, heaviside.
// ---------------------------------------------------------------------------

macro_rules! bind_binary_float_promote {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = binary_float_promote_body!(py, x1, x2, $ferr_path);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_float_promote!(fmax, ferray_ufunc::fmax);
bind_binary_float_promote!(fmin, ferray_ufunc::fmin);
bind_binary_float_promote!(copysign, ferray_ufunc::copysign);
bind_binary_float_promote!(hypot, ferray_ufunc::hypot);
bind_binary_float_promote!(arctan2, ferray_ufunc::arctan2);
bind_binary_float_promote!(logaddexp, ferray_ufunc::logaddexp);
bind_binary_float_promote!(logaddexp2, ferray_ufunc::logaddexp2);
bind_binary_float_promote!(heaviside, ferray_ufunc::heaviside);

// ---------------------------------------------------------------------------
// Binary numeric-split (separate float + integer loops, same output kind) —
// power, maximum, minimum, floor_divide, remainder/mod.
// ---------------------------------------------------------------------------

macro_rules! bind_binary_numeric_split {
    ($name:ident, $float_fn:path, $int_fn:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = binary_numeric_split_body!(py, x1, x2, $float_fn, $int_fn);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_numeric_split!(power, ferray_ufunc::power, ferray_ufunc::power_int);
bind_binary_numeric_split!(maximum, ferray_ufunc::maximum, ferray_ufunc::maximum_ord);
bind_binary_numeric_split!(minimum, ferray_ufunc::minimum, ferray_ufunc::minimum_ord);
bind_binary_numeric_split!(
    floor_divide,
    ferray_ufunc::floor_divide,
    ferray_ufunc::floor_divide_int
);
bind_binary_numeric_split!(
    remainder,
    ferray_ufunc::remainder,
    ferray_ufunc::remainder_int
);
bind_binary_numeric_split!(mod_, ferray_ufunc::mod_, ferray_ufunc::mod_int);

/// `numpy.true_divide(x1, x2)` — alias of `divide` (always true-division,
/// int -> float64). generate_umath.py:404 "'true_divide' : aliased to
/// divide".
#[pyfunction]
#[pyo3(signature = (x1, x2, out = None))]
pub fn true_divide<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    divide(py, x1, x2, out)
}

/// `numpy.float_power(x1, x2)` — power that ALWAYS promotes to float
/// (int -> float64), unlike `power` which keeps the int dtype
/// (generate_umath.py:490 `float_power` `TD(flts...)`).
#[pyfunction]
pub fn float_power<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let out = binary_float_promote_body!(py, x1, x2, ferray_ufunc::float_power);
    if scalar { scalarize(out) } else { Ok(out) }
}

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
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = comparison_body!(py, x1, x2, $ferr_path);
            if scalar { scalarize(out) } else { Ok(out) }
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
bind_binary_float_promote!(fmod, ferray_ufunc::fmod);
bind_binary_float_promote!(nextafter, ferray_ufunc::nextafter);

/// `gcd` / `lcm` are INTEGER-ONLY in numpy — they register only `TD(ints)`
/// loops (generate_umath.py:1156 `gcd`, :1163 `lcm`), so a float input
/// raises `TypeError` (a `UFuncTypeError`). The binding promotes both
/// inputs to the common integer dtype and routes to ferray's integer
/// `gcd_int`/`lcm_int`; a float dtype falls through to the `TypeError` arm.
///
/// ferray-ufunc's `gcd_int`/`lcm_int` are bounded `num_traits::Signed`, so
/// only the SIGNED integer dtypes are dispatched here (unsigned gcd/lcm is
/// a ferray-ufunc library gap — see spillover note in the dispatch issue).
/// numpy promotes the common dtype, so `gcd(int32, int64) -> int64`.
macro_rules! bind_binary_signed_int_only {
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let arr_a = as_ndarray(py, x1)?;
            let arr_b = as_ndarray(py, x2)?;
            let dt = binary_result_dtype(py, &arr_a, &arr_b)?;
            let np = py.import("numpy")?;
            let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
            let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
            macro_rules! __gcd_arm {
                ($Tn:ty) => {{
                    let arr_a2 = coerce_dtype(py, &pair_list[0], dt.as_str())?;
                    let arr_b2 = coerce_dtype(py, &pair_list[1], dt.as_str())?;
                    let va: PyReadonlyArrayDyn<$Tn> = arr_a2.extract()?;
                    let vb: PyReadonlyArrayDyn<$Tn> = arr_b2.extract()?;
                    let fa: ArrayD<$Tn> = va.as_ferray().map_err(ferr_to_pyerr)?;
                    let fb: ArrayD<$Tn> = vb.as_ferray().map_err(ferr_to_pyerr)?;
                    let r: ArrayD<$Tn> = $ferr_path(&fa, &fb).map_err(ferr_to_pyerr)?;
                    r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
                }};
            }
            let out = match dt.as_str() {
                "int64" | "i64" => __gcd_arm!(i64),
                "int32" | "i32" => __gcd_arm!(i32),
                "int16" | "i16" => __gcd_arm!(i16),
                "int8" | "i8" => __gcd_arm!(i8),
                other => {
                    return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                        "ufunc {:?} not supported for the input types (signed integer required): {other:?}",
                        stringify!($name)
                    )));
                }
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_signed_int_only!(gcd, ferray_ufunc::gcd_int);
bind_binary_signed_int_only!(lcm, ferray_ufunc::lcm_int);

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
///
/// NumPy defines `clip(a, lo, hi) == minimum(maximum(a, lo), hi)`
/// (fromnumeric.py `clip`), where `a_min` / `a_max` are `array_like or
/// None` — a `None` bound means "do not clip that side" (one-sided clip).
/// Because it is built from `maximum` / `minimum`, integer arrays keep
/// their integer dtype and array-valued bounds broadcast against `a`. The
/// binding therefore delegates to the [`maximum`] / [`minimum`] bindings,
/// which already carry the int loops, mixed-dtype promotion, and
/// broadcasting — instead of the old scalar-`f64`, float-only path.
#[pyfunction]
#[pyo3(signature = (a, a_min, a_max))]
pub fn clip<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    a_min: Option<&Bound<'py, PyAny>>,
    a_max: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let lo = a_min.filter(|v| !v.is_none());
    let hi = a_max.filter(|v| !v.is_none());
    if lo.is_none() && hi.is_none() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "One of max or min must be given",
        ));
    }
    // maximum(a, a_min) clamps the lower bound; minimum(., a_max) the upper.
    let lower = match lo {
        Some(lo_v) => maximum(py, a, lo_v)?,
        None => as_ndarray(py, a)?,
    };
    match hi {
        Some(hi_v) => minimum(py, &lower, hi_v),
        None => Ok(lower),
    }
}
