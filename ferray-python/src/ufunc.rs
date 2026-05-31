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
//!
//! ## REQ status — `numpy` ufunc surface registered by this shim
//!
//! This is a PyO3 *shim*: each row is a numpy callable bound here as a
//! `#[pyfunction]` that delegates to a `ferray-ufunc` kernel. The whole
//! surface is green against the numpy 2.4.x oracle (pytest
//! `tests/test_ufunc.py`). SHIPPED rows quote the binding fn + the
//! library fn it delegates to (symbol anchors, R-CITE-2b).
//!
//! SHIPPED:
//!   - Binary arithmetic (broadcasting): `add` / `subtract` / `multiply`
//!     / `divide` / `true_divide` delegate to
//!     `ferray_ufunc::{add,subtract,multiply,divide}_broadcast` via the
//!     `binary_numeric_body!` macro, with a `complex_binary_arith_dispatch!`
//!     arm for complex dtypes. NEP-50 promotion + integer-wrap +
//!     true-division semantics live in the kernel (see
//!     `ferray-ufunc/src/ops/arithmetic.rs` REQ table).
//!   - Binary numeric (split int/float): `floor_divide` / `remainder` /
//!     `mod_` / `fmod` / `float_power` via `bind_binary_numeric_split!` /
//!     `bind_binary_float_promote!`.
//!   - Bitwise + shifts: `bitwise_and` / `bitwise_or` / `bitwise_xor` /
//!     `bitwise_not` / `invert` / `left_shift` / `right_shift` /
//!     `bitwise_count` via `bind_binary_int_only!`.
//!   - Unary float transcendentals: `sin` / `cos` / `tan` / `exp` /
//!     `log` / `cbrt` / `fabs` / `ceil` / `floor` / `trunc` / `fix` /
//!     `degrees` / `radians` / `deg2rad` / `rad2deg` / `i0` / `spacing`
//!     bound by `bind_unary_float!`, each with a complex-loop guard
//!     (`bind_unary_float!(sin, ferray_ufunc::sin, complex =
//!     ferray_ufunc::sin_complex)`) so a complex input either COMPUTES
//!     complex or RAISES `TypeError`, matching numpy's loop registration.
//!   - Unary numeric split: `rint` / `sinc` / `exp2` and the
//!     `bind_unary_numeric_split!` family.
//!   - Comparisons → bool: `bind_comparison!`; logical:
//!     `logical_and` / `logical_or` / `logical_xor` / `logical_not`;
//!     predicates → bool: `bind_predicate_float!`.
//!   - Multi-output: `frexp` / `modf` / `divmod` (return tuples of arrays).
//!   - Reductions/utilities re-homed here: `clip`, `around`, `nan_to_num`,
//!     `interp`, `unwrap`, `gradient`, `ediff1d`, `trapezoid`, `ldexp`,
//!     `allclose` / `isclose` / `array_equal` / `array_equiv`.
//!   - Convolution: `convolve` / `correlate` (`pub fn convolve`,
//!     `pub fn correlate`, with `complex_convolve_dispatch` /
//!     `complex_correlate_dispatch` complex arms).
//!
//! NOT-STARTED: none — every callable registered in this module is bound
//! and green.

use ferray_core::array::aliases::ArrayD;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use num_complex::Complex;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    all_scalar_inputs, as_ndarray, binary_result_dtype, coerce_dtype, dtype_name, ferr_to_pyerr,
    scalarize, unary_promote_dtypes,
};
use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};
use crate::{
    match_dtype_all, match_dtype_float, match_dtype_float_or_int, match_dtype_int_only,
    match_dtype_numeric,
};

// ---------------------------------------------------------------------------
// Complex-input dispatch for unary transcendentals (#920/#921/#922).
//
// numpy registers a complex loop for sqrt/exp/log/sin/cos/... (a complex
// input computes a complex result), but registers NO complex loop for
// cbrt/fabs/degrees/... (a complex input raises TypeError). The prior binding
// funnelled EVERY unary-float op through `coerce_dtype(arr, "float32")`, which
// numpy casts complex -> float32 DROPPING the imaginary part (a silent lossy
// boundary round-trip — R-CODE-4) and then re-tagged the result complex. Both
// outcomes are wrong: ops with a complex loop must COMPUTE complex; ops without
// one must RAISE. The helpers below give each unary-float op the correct
// complex behavior BEFORE the f32 coercion can corrupt the data.
// ---------------------------------------------------------------------------

/// `true` for the two complex dtype names ferray marshals.
fn is_complex_dtype(dt: &str) -> bool {
    matches!(dt, "complex128" | "c16" | "complex64" | "c8")
}

/// Raise numpy's exact `TypeError` for a complex input to a ufunc that has NO
/// complex loop (`cbrt`, `fabs`, `degrees`, `radians`, `deg2rad`, `rad2deg`,
/// `i0`, `spacing`, …). numpy: `np.cbrt([1+2j])` -> `TypeError: ufunc 'cbrt'
/// not supported for the input types`. This replaces the prior silent
/// imag-discard (R-CODE-4) with the numpy contract (R-DEV-2). (`absolute`,
/// `negative`, `sinc`, and `rint` DO register a complex loop in numpy and are
/// handled by their dedicated complex arms, not this rejecter.)
fn reject_complex_unary(func: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "ufunc '{func}' not supported for the input types, and the inputs \
         could not be safely coerced to any supported types according to the \
         casting rule ''safe''"
    ))
}

/// Dispatch a unary complex transcendental over both complex widths: extract
/// the complex array via the manual marshaller, call the matching
/// `$cfn::<f32|f64>` complex op, and push the complex result back across the
/// boundary. `$cfn` is a `ferray_ufunc::*_complex` path generic over
/// `Complex<T>`. The result keeps the input's complex width (c64 -> c64,
/// c128 -> c128), matching numpy.
macro_rules! complex_unary_dispatch {
    ($py:expr, $arr:expr, $dt:expr, $cfn:path) => {{
        match $dt {
            "complex128" | "c16" => {
                let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&$arr)?;
                let r: ArrayD<Complex<f64>> = $cfn(&fa).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, r)?
            }
            "complex64" | "c8" => {
                let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&$arr)?;
                let r: ArrayD<Complex<f32>> = $cfn(&fa).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, r)?
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "complex_unary_dispatch: expected a complex dtype, got {other:?}"
                )));
            }
        }
    }};
}

/// `exp2` has no `ferray_ufunc::exp2_complex` library op, but numpy computes it
/// complex (`np.exp2([1+2j])` -> `2**z`). Compose it inline via num_complex's
/// `Complex::exp2`, mirroring the `*_complex` family's element-wise shape so the
/// binding stays correct without a net-new library op (the brief's "compose via
/// num_complex when the library op is missing").
fn exp2_complex_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f64>> = fa.iter().map(|z| z.exp2()).collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f32>> = fa.iter().map(|z| z.exp2()).collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "exp2_complex_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Complex arms for abs / negative / sinc / rint (#923/#927/#928).
//
// numpy registers a complex loop for `absolute` (→ REAL magnitude),
// `negative` (complex negate), `sinc` (`sin(pi*z)/(pi*z)`, z==0→1), and `rint`
// (per-component round-half-to-even). The prior binding routed all four to
// `reject_complex_unary`, raising `TypeError` where numpy COMPUTES — and for
// `absolute`/`negative` the `match_dtype_float_or_int!` body had no complex arm
// at all. These helpers compose the numpy result from the existing library
// (`ferray_ufunc::abs`) / `num_complex`, preserving the input complex width
// (c64→c64, c128→c128; `abs` narrows c64→float32, c128→float64 per numpy).
// ---------------------------------------------------------------------------

/// `abs`/`absolute` over complex → REAL magnitude array (`ferray_ufunc::abs`,
/// `Array<Complex<T>,D> -> Array<T,D>`). numpy: `np.abs([3+4j]) == [5.0]`,
/// dtype `float64` for c128 / `float32` for c64. The real result ships back via
/// `IntoNumPy::into_pyarray` (f32/f64 are `NpElement`), NOT the complex helper —
/// so no imaginary part is fabricated (R-CODE-4).
fn complex_abs_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let r: ArrayD<f64> = ferray_ufunc::abs(&fa).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let r: ArrayD<f32> = ferray_ufunc::abs(&fa).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_abs_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `negative` over complex → `-z` per element (`num_complex::Complex`'s `Neg`).
/// `ferray_ufunc::negative` is `T: Float`-bounded (no Complex impl), so the
/// negate is composed inline. numpy: `np.negative([1+2j,3-1j]) == [-1-2j,-3+1j]`,
/// dtype preserved (c64→c64, c128→c128).
fn complex_negative_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f64>> = fa.iter().map(|z| -z).collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f32>> = fa.iter().map(|z| -z).collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_negative_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `sinc` over complex → `sin(pi*z)/(pi*z)`, with `z == 0 → 1+0j`
/// (`num_complex::Complex::sin`). numpy registers a complex `sinc` loop
/// (`numpy/lib/_function_base_impl.py` `sinc`: `y = pi*where(x==0, 1.0e-20, x)`
/// → `sin(y)/y`); we use the exact `x==0 → 1` limit instead of numpy's
/// `1e-20` fudge, which agrees to full precision at the interior points and is
/// exactly `1+0j` at the origin. numpy: `np.sinc([1+2j]) == [-34.09-17.05j]`,
/// `np.sinc([0j]) == [1+0j]`. Width preserved.
fn complex_sinc_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let pi = std::f64::consts::PI;
            let one = Complex::<f64>::new(1.0, 0.0);
            let data: Vec<Complex<f64>> = fa
                .iter()
                .map(|z| {
                    if z.re == 0.0 && z.im == 0.0 {
                        one
                    } else {
                        let y = z * pi;
                        y.sin() / y
                    }
                })
                .collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let pi = std::f32::consts::PI;
            let one = Complex::<f32>::new(1.0, 0.0);
            let data: Vec<Complex<f32>> = fa
                .iter()
                .map(|z| {
                    if z.re == 0.0 && z.im == 0.0 {
                        one
                    } else {
                        let y = z * pi;
                        y.sin() / y
                    }
                })
                .collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_sinc_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `rint` over complex → round real AND imaginary parts independently to the
/// nearest even integer (`f64::round_ties_even`). numpy registers a complex
/// `rint` loop that rounds each component half-to-even
/// (`numpy/_core/src/umath/loops_unary_complex.dispatch.c.src`): `np.rint(
/// [1.4+2.6j]) == [1+3j]`, `np.rint([2.5+3.5j]) == [2+4j]` (ties to even).
/// Width preserved.
fn complex_rint_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f64>> = fa
                .iter()
                .map(|z| Complex::<f64>::new(z.re.round_ties_even(), z.im.round_ties_even()))
                .collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f32>> = fa
                .iter()
                .map(|z| Complex::<f32>::new(z.re.round_ties_even(), z.im.round_ties_even()))
                .collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_rint_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `around`/`round` over complex with a `decimals` argument → round the real
/// AND imaginary parts independently to `decimals` places, half-to-even. numpy
/// registers a complex `around` loop: `np.round([1.4+2.6j]) == [1+3j]`,
/// `np.round([1.45+2.65j], 1) == [1.4+2.6j]` (live 2.4.5). Mirrors
/// [`complex_rint_dispatch`] (decimals == 0) plus the `10**decimals` scale /
/// `round_ties_even` / unscale of the real `around` path, applied per component.
/// Width preserved (c64->c64, c128->c128); no imag discard (R-CODE-4).
fn complex_around_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
    decimals: i32,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let scale = 10f64.powi(decimals);
            let round1 = |x: f64| (x * scale).round_ties_even() / scale;
            let data: Vec<Complex<f64>> = fa
                .iter()
                .map(|z| Complex::<f64>::new(round1(z.re), round1(z.im)))
                .collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            // numpy computes the scaled round in the element width; for c64 the
            // f32 components are scaled/rounded/unscaled in f32.
            let scale = 10f32.powi(decimals);
            let round1 = |x: f32| (x * scale).round_ties_even() / scale;
            let data: Vec<Complex<f32>> = fa
                .iter()
                .map(|z| Complex::<f32>::new(round1(z.re), round1(z.im)))
                .collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_around_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Complex arms for the common unary ufuncs square / reciprocal / sign /
// isnan / isinf / isfinite (#934).
//
// numpy registers a complex loop for each: `square` (z*z), `reciprocal`
// (1/z, with 1/0 -> nan+nanj), `sign` (z/|z| for z!=0, 0+0j for z==0), and the
// three complex predicates isnan/isinf/isfinite (per-part rule -> bool). The
// prior bindings rejected complex: `square`/`reciprocal` funnel a complex input
// (neither float64/f32 nor an integer dtype) into their `match_dtype_int_only!`
// int fallback (which raises), `sign` routes through the real
// `match_dtype_float_or_int!` split (raises), and the predicates route through
// `match_dtype_float!` (raises). These helpers compute the numpy result from
// `num_complex` directly, preserving the complex width (c64->c64, c128->c128)
// for the value ops and returning a `bool` array for the predicates — no
// imaginary-part discard (R-CODE-4). Verified live (numpy 2.4.4):
//   np.square([1+2j]) == [-3+4j]            (c64 in -> c64 out, c128 -> c128)
//   np.reciprocal([1+2j]) == [0.2-0.4j]; np.reciprocal([0j]) == [nan+nanj]
//   np.sign([3+4j]) == [0.6+0.8j]; np.sign([0j]) == [0+0j]; |sign| == 1
//   np.isnan([1+2j, nan+0j, 1+nanj]) == [F,T,T]   (NaN in EITHER part)
//   np.isinf([1+2j, inf+0j, 1+infj]) == [F,T,T]   (Inf in EITHER part)
//   np.isfinite([1+2j, inf+0j, nan+0j]) == [T,F,F] (finite iff BOTH parts)
// ---------------------------------------------------------------------------

/// `square` over complex → `z*z` per element (`num_complex`'s `Mul`). numpy
/// registers a complex `square` loop; `np.square([1+2j]) == [-3+4j]`. Width
/// preserved (c64→c64, c128→c128) — no imag discard (R-CODE-4).
fn complex_square_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f64>> = fa.iter().map(|z| z * z).collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f32>> = fa.iter().map(|z| z * z).collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_square_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `reciprocal` over complex → `1/z` per element (`num_complex`'s `Div`).
/// `num_complex` yields `nan+nanj` for `1/(0+0j)`, matching numpy's complex
/// `reciprocal` loop (verified live: `np.reciprocal([0j]) == [nan+nanj]`,
/// emitting the same `invalid value encountered` RuntimeWarning numpy does — we
/// compute the same value without raising, R-DEV-1). Width preserved.
fn complex_reciprocal_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let one = Complex::<f64>::new(1.0, 0.0);
            let data: Vec<Complex<f64>> = fa.iter().map(|z| one / z).collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let one = Complex::<f32>::new(1.0, 0.0);
            let data: Vec<Complex<f32>> = fa.iter().map(|z| one / z).collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_reciprocal_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `sign` over complex → unit phasor `z/|z|` for `z != 0`, `0+0j` for `z == 0`
/// (numpy 2.x complex-sign convention: `numpy/_core/src/umath/loops.c.src`
/// `npy_csign`; verified live: `np.sign([3+4j]) == [0.6+0.8j]`, `np.sign([0j])
/// == [0+0j]`, `abs(sign) == 1`). The `z == 0 → 0` special case is required
/// because `z/|z|` is `0/0 = nan+nanj` at the origin in `num_complex` — numpy
/// returns exactly `0+0j` there. Width preserved.
fn complex_sign_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let zero = Complex::<f64>::new(0.0, 0.0);
            let data: Vec<Complex<f64>> = fa
                .iter()
                .map(|z| {
                    if z.re == 0.0 && z.im == 0.0 {
                        zero
                    } else {
                        z / z.norm()
                    }
                })
                .collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let zero = Complex::<f32>::new(0.0, 0.0);
            let data: Vec<Complex<f32>> = fa
                .iter()
                .map(|z| {
                    if z.re == 0.0 && z.im == 0.0 {
                        zero
                    } else {
                        z / z.norm()
                    }
                })
                .collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_sign_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// Per-element predicate kind for the complex isnan/isinf/isfinite loops.
#[derive(Clone, Copy)]
enum ComplexPredicate {
    /// `isnan`: NaN in EITHER part.
    Nan,
    /// `isinf`: ±Inf in EITHER part.
    Inf,
    /// `isfinite`: finite (neither NaN nor Inf) in BOTH parts.
    Finite,
}

/// `isnan`/`isinf`/`isfinite` over complex → `bool` array, per numpy's complex
/// predicate loops (`numpy/_core/src/umath/loops_unary_complex.dispatch.c.src`):
/// `isnan(z) = isnan(re) || isnan(im)`, `isinf(z) = isinf(re) || isinf(im)`,
/// `isfinite(z) = isfinite(re) && isfinite(im)`. Verified live (numpy 2.4.4):
/// `np.isnan([1+2j, nan+0j, 1+nanj]) == [F,T,T]`,
/// `np.isinf([1+2j, inf+0j, 1+infj]) == [F,T,T]`,
/// `np.isfinite([1+2j, inf+0j, nan+0j]) == [T,F,F]`. The result is a real `bool`
/// array (`into_pyarray`), NOT a complex one — so no imaginary part is
/// fabricated (R-CODE-4).
fn complex_predicate_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
    kind: ComplexPredicate,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<bool> = fa
                .iter()
                .map(|z| match kind {
                    ComplexPredicate::Nan => z.re.is_nan() || z.im.is_nan(),
                    ComplexPredicate::Inf => z.re.is_infinite() || z.im.is_infinite(),
                    ComplexPredicate::Finite => z.re.is_finite() && z.im.is_finite(),
                })
                .collect();
            let r = ArrayD::<bool>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let shape = fa.shape().to_vec();
            let data: Vec<bool> = fa
                .iter()
                .map(|z| match kind {
                    ComplexPredicate::Nan => z.re.is_nan() || z.im.is_nan(),
                    ComplexPredicate::Inf => z.re.is_infinite() || z.im.is_infinite(),
                    ComplexPredicate::Finite => z.re.is_finite() && z.im.is_finite(),
                })
                .collect();
            let r = ArrayD::<bool>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_predicate_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Complex arms for BINARY arithmetic + comparison (#924).
//
// numpy registers a complex loop for add/subtract/multiply/divide/power
// (complex compute), equal/not_equal (bool), and less/greater/<=/>=
// (LEXICOGRAPHIC `(real, then imag)`, does NOT raise — verified live numpy
// 2.4.5: `np.less([1+2j,3+2j],[1+5j,3+1j]) == [True False]`). It registers NO
// complex loop for floor_divide/remainder/mod/bitwise_*/left_shift/right_shift
// (those RAISE `TypeError`). The top-level binding's `binary_numeric_body!` /
// `comparison_body!` funnel through `match_dtype_numeric!` (real-only) and so
// reject every complex input with `TypeError` even though the library ops
// already accept complex (`add_broadcast`/… `T: WrappingArith`/`TrueDivide`,
// `equal_broadcast` `T: PartialEq`; #869). These helpers give each binary op
// the correct complex behavior BEFORE the real funnel: arithmetic computes via
// the existing complex-capable broadcast ops, comparison computes lexicographic
// (mirroring ma #874's `cmp_complex_arm!`), and the unsupported ops raise.
//
// Both operands are broadcast + coerced to the common complex dtype in numpy
// (`result_type` + `broadcast_arrays`), so `complex+real` / `complex+scalar`
// promotion (numpy `complex+float -> complex`) and reflected operands all reuse
// numpy's own promotion table — and the two ferray arrays are the SAME shape,
// so even the same-`D` `power_complex` op applies.
// ---------------------------------------------------------------------------

/// `true` if numpy's `result_type(a, b)` for the two inputs is a complex dtype
/// — the signal that the binary op must take its complex arm rather than the
/// real `match_dtype_numeric!` funnel. Reuses numpy's promotion table, so
/// `complex+real`, `complex+python-scalar`, and reflected operands all classify
/// correctly (numpy: `result_type([1+2j], 1) == complex128`).
fn binary_is_complex(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    // Normalize both operands to ndarrays first: `result_type` on a raw Python
    // list (`[2,4]`) misparses as a structured-dtype field spec, so the inputs
    // must be `as_ndarray`'d exactly as the real `binary_numeric_body!` path
    // does before sniffing the promoted dtype.
    let arr_a = as_ndarray(py, a)?;
    let arr_b = as_ndarray(py, b)?;
    Ok(is_complex_dtype(
        binary_result_dtype(py, &arr_a, &arr_b)?.as_str(),
    ))
}

/// `true` if BOTH operands have an integer / unsigned / bool dtype
/// (`dtype.kind ∈ {i, u, b}`). Used by `fmax`/`fmin` to keep numpy's integer
/// result (their float-promote body would upcast int -> float64).
fn binary_both_integer(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let kind = |o: &Bound<'_, PyAny>| -> PyResult<bool> {
        let arr = as_ndarray(py, o)?;
        let k: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
        Ok(matches!(k.as_str(), "i" | "u" | "b"))
    };
    Ok(kind(a)? && kind(b)?)
}

/// `true` if BOTH operands are integer/unsigned/bool dtype AND the exponent
/// (`x2`) contains ANY negative element — the case numpy's integer `power` loop
/// REJECTS with `ValueError: Integers to negative integer powers are not
/// allowed.` (`numpy/_core/src/umath/loops.c.src:519`, scalarmath `:1553`).
/// ferray's integer `power_int` kernel would instead silently return `0` (a
/// wrong VALUE, R-CODE-4), so the `power` binding must raise here BEFORE the
/// kernel runs. A FLOAT base or FLOAT exponent is NOT both-integer and is left
/// untouched (`np.power(2.0, -1) == 0.5`). The negativity test uses the numpy
/// oracle (`np.any(x2 < 0)`) so it covers every signed-integer width and array
/// shape without re-extracting the exponent into Rust.
fn power_is_negative_int_exponent(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    if !binary_both_integer(py, a, b)? {
        return Ok(false);
    }
    let exp = as_ndarray(py, b)?;
    let np = py.import("numpy")?;
    let zero = 0i64;
    let neg = np.call_method1("less", (&exp, zero))?;
    let any: bool = np.call_method1("any", (neg,))?.extract()?;
    Ok(any)
}

/// Broadcast both operands to a common shape and coerce both to the common
/// complex dtype (`result_type`, then `broadcast_arrays`), returning the two
/// numpy ndarrays plus the resolved complex dtype name. Centralizes the
/// promotion + broadcast every complex binary arm shares.
fn complex_binary_operands<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, String)> {
    let arr_a = as_ndarray(py, a)?;
    let arr_b = as_ndarray(py, b)?;
    let dt = binary_result_dtype(py, &arr_a, &arr_b)?;
    let np = py.import("numpy")?;
    let pair = np.call_method1("broadcast_arrays", (&arr_a, &arr_b))?;
    let pair_list: Vec<Bound<PyAny>> = pair.extract()?;
    let arr_a2 = coerce_dtype(py, &pair_list[0], dt.as_str())?;
    let arr_b2 = coerce_dtype(py, &pair_list[1], dt.as_str())?;
    Ok((arr_a2, arr_b2, dt))
}

/// Complex arm for `add`/`subtract`/`multiply`/`divide`/`power`: compute via the
/// matching complex-capable `ferray_ufunc` op at the resolved complex width
/// (c64→c64, c128→c128) and ship the complex result back. `$cfn` is generic over
/// `Complex<T>`. numpy: `np.add([1+2j],[3+4j]) == [4+6j]`; division is true
/// complex division; `power` is the principal complex power.
macro_rules! complex_binary_arith_dispatch {
    ($py:expr, $a:expr, $b:expr, $cfn:path) => {{
        let (arr_a, arr_b, dt) = complex_binary_operands($py, $a, $b)?;
        match dt.as_str() {
            "complex128" | "c16" => {
                let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_a)?;
                let fb: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_b)?;
                let r: ArrayD<Complex<f64>> = $cfn(&fa, &fb).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, r)?
            }
            "complex64" | "c8" => {
                let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_a)?;
                let fb: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_b)?;
                let r: ArrayD<Complex<f32>> = $cfn(&fa, &fb).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, r)?
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "complex_binary_arith_dispatch: expected a complex dtype, got {other:?}"
                )));
            }
        }
    }};
}

/// Lexicographic complex compare (`<`, `>`, `<=`, `>=`) over a same-shape pair —
/// numpy compares `(real, then imag)` and does NOT raise (verified live, numpy
/// 2.4.5). `Complex` is NOT `PartialOrd`, so the order is computed directly from
/// the `re`/`im` parts, mirroring ma #874's `cmp_complex_arm!`. A NaN part makes
/// every ordering compare `false` (numpy's `invalid value encountered` → False,
/// matching f32/f64 NaN semantics). `op` is `"lt"|"le"|"gt"|"ge"`.
fn lexicographic_cmp<T: PartialOrd + Copy>(a_re: T, a_im: T, b_re: T, b_im: T, op: &str) -> bool {
    match op {
        "lt" => a_re < b_re || (a_re == b_re && a_im < b_im),
        "le" => a_re < b_re || (a_re == b_re && a_im <= b_im),
        "gt" => a_re > b_re || (a_re == b_re && a_im > b_im),
        "ge" => a_re > b_re || (a_re == b_re && a_im >= b_im),
        _ => false,
    }
}

/// Whole-operand lexicographic min/max select for the complex `maximum`/
/// `minimum`/`fmax`/`fmin` loops. Returns the SELECTED `Complex` (width
/// preserved — no imag discard, R-CODE-4). numpy registers these as complex
/// ufunc loops comparing `(real, then imag)`; verified live (numpy 2.4.5):
///
/// - `nan_propagate = true` (`maximum`/`minimum`): if `a` has a NaN part return
///   `a`; else if `b` has a NaN part return `b`; else the lexicographic
///   `max`/`min`. (`np.maximum([nan+nanj],[1+5j]) == [nan+nanj]`;
///   `np.maximum([1+5j],[nan+nanj]) == [nan+nanj]` — `a` takes precedence.)
/// - `nan_propagate = false` (`fmax`/`fmin`): if exactly one operand has a NaN
///   part return the OTHER (non-NaN) operand; if both NaN return `a`; else the
///   lexicographic `max`/`min`. (`np.fmax([nan+nanj],[1+5j]) == [1+5j]`;
///   `np.fmax([nan+nanj],[nan+nanj]) == [nan+nanj]`.)
///
/// `want_max` picks the larger (`"gt"`) vs smaller (`"lt"`) on the finite path.
/// NaN of a part is detected by `x != x` (no `Float` bound needed); ties (equal
/// real, equal imag) keep `b` for max and `a` for min via the strict `gt`/`lt`,
/// matching numpy (equal operands are interchangeable).
fn complex_minmax_select<T: PartialOrd + Copy>(
    a: Complex<T>,
    b: Complex<T>,
    want_max: bool,
    nan_propagate: bool,
) -> Complex<T> {
    // A NaN part is unorderable against itself: `partial_cmp` yields `None`
    // (no `Float` bound needed — `num_traits` is not a direct dependency).
    let is_nan = |v: T| v.partial_cmp(&v).is_none();
    let a_nan = is_nan(a.re) || is_nan(a.im);
    let b_nan = is_nan(b.re) || is_nan(b.im);
    if a_nan || b_nan {
        if nan_propagate {
            // maximum/minimum: propagate; `a` takes precedence when NaN.
            return if a_nan { a } else { b };
        }
        // fmax/fmin: suppress — return the non-NaN operand, or `a` if both NaN.
        return if a_nan && b_nan {
            a
        } else if a_nan {
            b
        } else {
            a
        };
    }
    let op = if want_max { "gt" } else { "lt" };
    if lexicographic_cmp(a.re, a.im, b.re, b.im, op) {
        a
    } else {
        b
    }
}

/// Complex arm for `maximum`/`minimum`/`fmax`/`fmin` → complex array (width
/// preserved). Both operands are broadcast + coerced to the common complex
/// dtype, then [`complex_minmax_select`] picks the whole `Complex` per element.
fn complex_minmax_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    want_max: bool,
    nan_propagate: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    let (arr_a, arr_b, dt) = complex_binary_operands(py, a, b)?;
    match dt.as_str() {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_a)?;
            let fb: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_b)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f64>> = fa
                .iter()
                .zip(fb.iter())
                .map(|(x, y)| complex_minmax_select(*x, *y, want_max, nan_propagate))
                .collect();
            let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_a)?;
            let fb: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_b)?;
            let shape = fa.shape().to_vec();
            let data: Vec<Complex<f32>> = fa
                .iter()
                .zip(fb.iter())
                .map(|(x, y)| complex_minmax_select(*x, *y, want_max, nan_propagate))
                .collect();
            let r = ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&shape), data)
                .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_minmax_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// Complex arm for the four ordering comparisons → bool array, lexicographic
/// `(real, then imag)`. Both operands are already broadcast + coerced to the
/// common complex dtype.
fn complex_order_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    op: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    let (arr_a, arr_b, dt) = complex_binary_operands(py, a, b)?;
    match dt.as_str() {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_a)?;
            let fb: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_b)?;
            let shape = fa.shape().to_vec();
            let data: Vec<bool> = fa
                .iter()
                .zip(fb.iter())
                .map(|(x, y)| lexicographic_cmp(x.re, x.im, y.re, y.im, op))
                .collect();
            let r = ArrayD::<bool>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_a)?;
            let fb: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_b)?;
            let shape = fa.shape().to_vec();
            let data: Vec<bool> = fa
                .iter()
                .zip(fb.iter())
                .map(|(x, y)| lexicographic_cmp(x.re, x.im, y.re, y.im, op))
                .collect();
            let r = ArrayD::<bool>::from_vec(IxDyn::new(&shape), data).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_order_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// Complex arm for `equal`/`not_equal` → bool array, computed via the
/// `PartialEq`-bounded `equal_broadcast`/`not_equal_broadcast` (Complex
/// satisfies `PartialEq`). numpy: `np.equal([1+2j],[1+2j]) == [True]`.
macro_rules! complex_eq_dispatch {
    ($py:expr, $a:expr, $b:expr, $cfn:path) => {{
        let (arr_a, arr_b, dt) = complex_binary_operands($py, $a, $b)?;
        match dt.as_str() {
            "complex128" | "c16" => {
                let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_a)?;
                let fb: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(&arr_b)?;
                let r: ArrayD<bool> = $cfn(&fa, &fb).map_err(ferr_to_pyerr)?;
                r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "complex64" | "c8" => {
                let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_a)?;
                let fb: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(&arr_b)?;
                let r: ArrayD<bool> = $cfn(&fa, &fb).map_err(ferr_to_pyerr)?;
                r.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "complex_eq_dispatch: expected a complex dtype, got {other:?}"
                )));
            }
        }
    }};
}

// ---------------------------------------------------------------------------
// float16 arm for BINARY arithmetic (#953, REQ-3).
//
// The real-only `match_dtype_numeric!` / `match_dtype_float_or_int!` macros
// have NO float16 arm, so every binary arithmetic op rejects a float16 operand
// with `TypeError: unsupported dtype for numeric op: "float16"` — even though
// numpy computes them all (`np.add(f16,f16).dtype == float16`, `f16+f32 ->
// float32`, `f16+int32 -> float64`, verified live numpy 2.4.5). The installed
// pyo3-numpy build has no `NumpyElement`/`PyReadonlyArrayDyn` for `half::f16`
// (the `numpy/half` feature is off — see `.design/ferray-core-float16.md`), so
// a typed f16 view can't be taken at the boundary.
//
// The arm below DETECTS a float16 operand and DELEGATES the whole op to numpy's
// own top-level function on the ORIGINAL operands. numpy then owns (a) NEP-50
// promotion (`result_type`: f16+f16->f16, f16+f32->f32, f16+pyfloat->f16,
// f16+int-array->f64) and (b) the f16 compute contract (numpy computes float16
// arithmetic at float32 width and rounds the result back to float16, overflow
// to inf). This is bit-for-bit numpy-correct and keeps `half::f16` entirely out
// of the Rust boundary — the SAME detect-and-delegate pattern the float16
// reductions (`crate::stats::f16_reduce`, #954) and creation coercion (#952)
// already use. The guard fires ONLY when an operand is float16, so the real
// f32/f64/int arithmetic paths are completely unchanged.
// ---------------------------------------------------------------------------

/// `true` for the two float16 dtype names numpy reports.
fn is_float16_dtype(dt: &str) -> bool {
    matches!(dt, "float16" | "f16")
}

/// `true` if EITHER binary operand is a float16 array/scalar — the signal that
/// the op must take the numpy-delegate float16 arm rather than the real-only
/// `match_dtype_numeric!` / `match_dtype_float_or_int!` funnel. Both operands
/// are normalized to ndarrays first (matching the real path's `as_ndarray`),
/// then their dtype names are sniffed. A float16-on-either-side pair covers
/// every NEP-50 case (f16+f16, f16+f32, f16+int-array, reflected), all of which
/// numpy resolves via its own `result_type` inside the delegated call.
fn binary_involves_float16(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let arr_a = as_ndarray(py, a)?;
    let arr_b = as_ndarray(py, b)?;
    Ok(is_float16_dtype(dtype_name(&arr_a)?.as_str())
        || is_float16_dtype(dtype_name(&arr_b)?.as_str()))
}

/// Delegate a binary arithmetic op with a float16 operand to numpy's own
/// top-level function (`func` = `"add"`, `"subtract"`, `"multiply"`, `"divide"`,
/// `"power"`, `"floor_divide"`, `"remainder"`, `"mod"`, `"maximum"`,
/// `"minimum"`) on the ORIGINAL operands. numpy owns NEP-50 promotion + the
/// float32-compute / float16-narrow contract, so the returned object carries
/// numpy's exact result dtype (f16/f32/f64) and value (overflow -> inf). For
/// all-scalar operands numpy returns a 0-d numpy scalar, which the caller's
/// `scalarize`/`finish_with_out` path passes through unchanged (R-DEV-3).
fn f16_binary_delegate<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    func: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.call_method1(func, (a, b))
}

/// Raise numpy's exact `TypeError` for a complex input to a BINARY ufunc with
/// NO complex loop (`floor_divide`, `remainder`, `mod`, `bitwise_and`/`or`/
/// `xor`, `left_shift`, `right_shift`). numpy: `np.floor_divide(z, z)` →
/// `TypeError: ufunc 'floor_divide' not supported for the input types`. Mirrors
/// the unary `reject_complex_unary` (R-DEV-2).
fn reject_complex_binary(func: &str) -> PyErr {
    PyTypeError::new_err(format!(
        "ufunc '{func}' not supported for the input types, and the inputs \
         could not be safely coerced to any supported types according to the \
         casting rule ''safe''"
    ))
}

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
    // No-complex-loop form: a complex input RAISES `TypeError` (numpy has no
    // complex loop for this op — cbrt/fabs/degrees/radians/deg2rad/rad2deg/
    // sinc/i0). The guard runs BEFORE `unary_float_body!`'s f32 coercion, so
    // the imaginary part is never silently dropped (R-CODE-4).
    ($name:ident, $ferr_path:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;
            if is_complex_dtype(dt.as_str()) {
                return Err(reject_complex_unary(stringify!($name)));
            }
            let scalar = all_scalar_inputs(py, &[x])?;
            let out = unary_float_body!(py, arr, $ferr_path);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Complex-loop form: a complex input dispatches to the matching
    // `ferray_ufunc::*_complex` op and returns a complex array (numpy computes
    // complex), BEFORE the real f32 path. Real input keeps the existing
    // promote-to-float behavior unchanged.
    ($name:ident, $ferr_path:path, complex = $cfn:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            let out = if is_complex_dtype(dt.as_str()) {
                complex_unary_dispatch!(py, arr, dt.as_str(), $cfn)
            } else {
                unary_float_body!(py, arr, $ferr_path)
            };
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
            // float16 (REQ-5, #955): delegate to numpy (f32-compute, f16-narrow);
            // the real-only `match_dtype_float_or_int!` body has no float16 arm.
            if is_float16_dtype(dtype_name(&arr)?.as_str()) {
                let scalar = all_scalar_inputs(py, &[x])?;
                let out = crate::conv::f16_delegate(py, stringify!($name), (&arr,), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let scalar = all_scalar_inputs(py, &[x])?;
            let out = unary_numeric_split_body!(py, arr, $float_fn, $int_fn);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Complex-loop form: a complex input dispatches to `$cfn` (numpy registers a
    // complex loop for this op — `absolute` → real magnitude, `negative` →
    // complex negate). Real/integer input keeps the existing float/int split.
    ($name:ident, $float_fn:path, $int_fn:path, complex = $cfn:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            // float16 (REQ-5, #955): delegate to numpy ahead of the real-only body.
            if is_float16_dtype(dt.as_str()) {
                let out = crate::conv::f16_delegate(py, stringify!($name), (&arr,), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let out = if is_complex_dtype(dt.as_str()) {
                $cfn(py, &arr, dt.as_str())?
            } else {
                unary_numeric_split_body!(py, arr, $float_fn, $int_fn)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Time-aware complex-loop form (`absolute`/`negative`/`sign`, #947): a
    // datetime64/timedelta64 input routes through `$tu`
    // (`crate::datetime::TimeUnary`) — `abs(td)`/`negative(td)` -> timedelta,
    // `sign(td)` -> timedelta `{-1,0,1}`; datetime input RAISES numpy's exact
    // `UFuncTypeError`. Complex input keeps the `$cfn` loop; real/int the split.
    ($name:ident, $float_fn:path, $int_fn:path, complex = $cfn:path, time = $tu:expr) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;
            if crate::datetime::is_time_array(&arr)? {
                let out = crate::datetime::unary_time(py, &arr, $tu)?;
                let scalar = all_scalar_inputs(py, &[x])?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let scalar = all_scalar_inputs(py, &[x])?;
            // float16 (REQ-5, #955): delegate to numpy ahead of the real-only body.
            if is_float16_dtype(dt.as_str()) {
                let out = crate::conv::f16_delegate(py, stringify!($name), (&arr,), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let out = if is_complex_dtype(dt.as_str()) {
                $cfn(py, &arr, dt.as_str())?
            } else {
                unary_numeric_split_body!(py, arr, $float_fn, $int_fn)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

// Trigonometric + hyperbolic + inverse — numpy registers a complex loop for
// each (generate_umath.py `'fdg' + cmplx`), so complex input computes complex
// via the matching `ferray_ufunc::*_complex` op.
bind_unary_float!(sin, ferray_ufunc::sin, complex = ferray_ufunc::sin_complex);
bind_unary_float!(cos, ferray_ufunc::cos, complex = ferray_ufunc::cos_complex);
bind_unary_float!(tan, ferray_ufunc::tan, complex = ferray_ufunc::tan_complex);
bind_unary_float!(
    arcsin,
    ferray_ufunc::arcsin,
    complex = ferray_ufunc::asin_complex
);
bind_unary_float!(
    arccos,
    ferray_ufunc::arccos,
    complex = ferray_ufunc::acos_complex
);
bind_unary_float!(
    arctan,
    ferray_ufunc::arctan,
    complex = ferray_ufunc::atan_complex
);
bind_unary_float!(
    sinh,
    ferray_ufunc::sinh,
    complex = ferray_ufunc::sinh_complex
);
bind_unary_float!(
    cosh,
    ferray_ufunc::cosh,
    complex = ferray_ufunc::cosh_complex
);
bind_unary_float!(
    tanh,
    ferray_ufunc::tanh,
    complex = ferray_ufunc::tanh_complex
);
bind_unary_float!(
    arcsinh,
    ferray_ufunc::arcsinh,
    complex = ferray_ufunc::asinh_complex
);
bind_unary_float!(
    arccosh,
    ferray_ufunc::arccosh,
    complex = ferray_ufunc::acosh_complex
);
bind_unary_float!(
    arctanh,
    ferray_ufunc::arctanh,
    complex = ferray_ufunc::atanh_complex
);
// degrees/radians/deg2rad/rad2deg register NO complex loop — complex RAISES.
bind_unary_float!(degrees, ferray_ufunc::degrees);
bind_unary_float!(radians, ferray_ufunc::radians);
bind_unary_float!(deg2rad, ferray_ufunc::deg2rad);
bind_unary_float!(rad2deg, ferray_ufunc::rad2deg);

// Exponential / logarithmic (REQ-23: int -> float). All register a complex
// loop in numpy; `exp2` has no `exp2_complex` lib op so it is composed inline.
bind_unary_float!(exp, ferray_ufunc::exp, complex = ferray_ufunc::exp_complex);
bind_unary_float!(
    expm1,
    ferray_ufunc::expm1,
    complex = ferray_ufunc::expm1_complex
);
bind_unary_float!(log, ferray_ufunc::log, complex = ferray_ufunc::ln_complex);
bind_unary_float!(
    log1p,
    ferray_ufunc::log1p,
    complex = ferray_ufunc::log1p_complex
);
bind_unary_float!(
    log2,
    ferray_ufunc::log2,
    complex = ferray_ufunc::log2_complex
);
bind_unary_float!(
    log10,
    ferray_ufunc::log10,
    complex = ferray_ufunc::log10_complex
);

/// `numpy.exp2(x)` — `2**x`. Real/int input promotes to float; complex input
/// computes the complex `2**z` (numpy registers a complex loop, but ferray-ufunc
/// has no `exp2_complex`, so the binding composes it via num_complex — see
/// [`exp2_complex_dispatch`]).
#[pyfunction]
pub fn exp2<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let scalar = all_scalar_inputs(py, &[x])?;
    let out = if is_complex_dtype(dt.as_str()) {
        exp2_complex_dispatch(py, &arr, dt.as_str())?
    } else {
        unary_float_body!(py, arr, ferray_ufunc::exp2)
    };
    if scalar { scalarize(out) } else { Ok(out) }
}

// Roots — REQ-23: int -> float. `sqrt` registers a complex loop (complex sqrt,
// principal branch); `cbrt` does NOT (complex RAISES).
bind_unary_float!(
    sqrt,
    ferray_ufunc::sqrt,
    complex = ferray_ufunc::sqrt_complex
);
bind_unary_float!(cbrt, ferray_ufunc::cbrt);
/// `numpy.rint(x)` — round to the nearest integer, ties to even. `rint` has NO
/// integer loop in numpy (generate_umath.py:1021), so int/bool input promotes
/// to float. numpy ALSO registers a complex `rint` loop that rounds the real
/// AND imaginary parts independently to nearest-even
/// (`np.rint([1.4+2.6j]) == [1+3j]`, `np.rint([2.5+3.5j]) == [2+4j]`); the
/// complex arm composes that per-component (see [`complex_rint_dispatch`]).
#[pyfunction]
pub fn rint<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let scalar = all_scalar_inputs(py, &[x])?;
    let out = if is_complex_dtype(dt.as_str()) {
        complex_rint_dispatch(py, &arr, dt.as_str())?
    } else {
        unary_float_body!(py, arr, ferray_ufunc::rint)
    };
    if scalar { scalarize(out) } else { Ok(out) }
}
// `fabs` registers only float loops — int -> float (generate_umath.py:1003);
// no complex loop -> complex RAISES.
bind_unary_float!(fabs, ferray_ufunc::fabs);

// Sign / absolute / negative / positive — int-identity (int -> int).
// `negative` registers a complex loop (complex negate); `absolute` registers a
// complex loop returning the REAL magnitude (`fr.abs` is a Python alias of
// `absolute`, so this complex arm covers both). numpy: `np.negative([1+2j]) ==
// [-1-2j]`; `np.abs([3+4j]) == [5.0]` (float64 for c128, float32 for c64).
bind_unary_numeric_split!(
    negative,
    ferray_ufunc::negative,
    ferray_ufunc::negative_int,
    complex = complex_negative_dispatch,
    time = crate::datetime::TimeUnary::Negative
);
bind_unary_numeric_split!(
    absolute,
    ferray_ufunc::absolute,
    ferray_ufunc::absolute_int,
    complex = complex_abs_dispatch,
    time = crate::datetime::TimeUnary::Abs
);
// `sign` registers a complex loop in numpy returning the unit phasor `z/|z|`
// (0+0j for z==0) — the prior split macro had no complex arm so `fr.sign(complex)`
// raised TypeError where numpy computes (#934). Real/integer input keeps the
// existing float/int split.
bind_unary_numeric_split!(
    sign,
    ferray_ufunc::sign,
    ferray_ufunc::sign_int,
    complex = complex_sign_dispatch,
    time = crate::datetime::TimeUnary::Sign
);

// Rounding floor/ceil/trunc/fix — int-identity (REQ-24, int -> int).
bind_unary_numeric_split!(floor, ferray_ufunc::floor, ferray_ufunc::floor_int);
bind_unary_numeric_split!(ceil, ferray_ufunc::ceil, ferray_ufunc::ceil_int);
bind_unary_numeric_split!(trunc, ferray_ufunc::trunc, ferray_ufunc::trunc_int);
bind_unary_numeric_split!(fix, ferray_ufunc::fix, ferray_ufunc::fix_int);

// `round` keeps int dtype (generate_umath.py `TD(bints)` on `rint`/`around`'s
// int loops); float route is `ferray_ufunc::round`. A complex input rounds the
// real AND imaginary parts independently to nearest-even (decimals=0), reusing
// the #928 complex `rint` arm — numpy: `np.round([1.4+2.6j]) == [1+3j]`. The
// prior split macro had no complex arm so `fr.round(complex)` returned the input
// UNROUNDED (a no-op, wrong value — R-CODE-4).
bind_unary_numeric_split!(
    round,
    ferray_ufunc::round,
    ferray_ufunc::round_int,
    complex = complex_rint_dispatch
);

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
    // Complex input: round the real AND imaginary parts independently to
    // `decimals` places, half-to-even — numpy registers a complex `around` loop
    // (`np.round([1.4+2.6j]) == [1+3j]`, `np.round([1.45+2.65j],1) == [1.4+2.6j]`,
    // live 2.4.5). MUST branch BEFORE the float64 coercion below, which numpy
    // casts complex->float64 DROPPING the imaginary part (R-CODE-4).
    if is_complex_dtype(dt.as_str()) {
        return complex_around_dispatch(py, &arr, dt.as_str(), decimals);
    }
    // float16 (REQ-5, #955): `around` preserves the dtype (`np.round(f16).dtype ==
    // float16`, live); delegate to numpy as the float bodies below have no f16
    // arm (and the float64 scale path would widen).
    if is_float16_dtype(dt.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("decimals", decimals)?;
        return crate::conv::f16_delegate(py, "around", (&arr,), Some(&kwargs));
    }
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
    // Complex input: numpy replaces NaN/±Inf in BOTH the real and imaginary
    // parts independently, keeping the result complex
    // (numpy/lib/_type_check_impl.py:382 nan_to_num operates on `x.real` and
    // `x.imag`). The float-only path below has no complex arm and would
    // mis-handle (or drop the imaginary part of) the input (R-CODE-4).
    // Delegate the complex case to numpy.nan_to_num, forwarding the
    // nan/posinf/neginf replacements, and return its complex result.
    if is_complex_dtype(dt.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("nan", nan)?;
        kwargs.set_item("posinf", posinf)?;
        kwargs.set_item("neginf", neginf)?;
        let np = py.import("numpy")?;
        return np.call_method("nan_to_num", (&arr,), Some(&kwargs));
    }
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
/// angle by changing deltas greater than `discont` to their `period`
/// complement (numpy/lib/_function_base_impl.py:1734). The library
/// `ferray_ufunc::unwrap` is the 1-D unwrap with `discont` defaulting to pi over
/// the default `2*pi` period. A non-default `period`, a non-default `axis`, or an
/// N-D input delegates to numpy, which owns the period-based wrapping and the
/// N-D/axis reduction. Integer/bool input promotes to `float64`.
#[pyfunction]
#[pyo3(signature = (p, discont = None, axis = -1, period = None))]
pub fn unwrap<'py>(
    py: Python<'py>,
    p: &Bound<'py, PyAny>,
    discont: Option<f64>,
    axis: isize,
    // `period` is forwarded to numpy as the ORIGINAL object (not coerced to
    // f64): numpy keeps an integer result when both the input and the period are
    // integer (`np.unwrap([0,5,10], period=6).dtype == int64`), so passing a
    // float `6.0` here would wrongly upcast the result to float64.
    period: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, p)?;
    let dt = dtype_name(&arr)?;
    // numpy.unwrap raises TypeError on complex input (its internal `mod` / `remainder`
    // ufunc has no complex loop): "ufunc 'remainder' not supported for the input
    // types" (verified live numpy 2.4.4). Reject complex before the float coercion
    // would silently unwrap the real parts (R-CODE-4 / R-DEV-2: preserve the
    // exception type).
    if matches!(dt.as_str(), "complex128" | "c16" | "complex64" | "c8") {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "ufunc 'remainder' not supported for the input types: numpy.unwrap does not accept complex input",
        ));
    }
    // The native 1-D kernel only covers the default 2*pi period along a 1-D
    // array's only axis; a non-default period, a non-default axis, or an N-D
    // input delegates to numpy.unwrap.
    let ndim: usize = arr.getattr("ndim")?.extract()?;
    if period.is_some() || ndim != 1 || !(axis == -1 || axis == 0) {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(d) = discont {
            kwargs.set_item("discont", d)?;
        }
        kwargs.set_item("axis", axis)?;
        if let Some(pp) = period {
            kwargs.set_item("period", pp)?;
        }
        return np.call_method("unwrap", (&arr,), Some(&kwargs));
    }
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
            // float16 (REQ-5, #955): delegate to numpy (f32-compute, f16-narrow);
            // neither the real-only `match_dtype_float!` nor the int fallback has
            // a float16 arm.
            if is_float16_dtype(dt.as_str()) {
                let out = crate::conv::f16_delegate(py, stringify!($name), (&arr,), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
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
    // Complex-loop form (#934): a complex input dispatches to `$cfn` (numpy
    // registers a complex loop — `square` → z*z, `reciprocal` → 1/z), computing
    // BEFORE the float/int fallback (which would otherwise raise on complex).
    // Real float input → float kernel; integer input → the int fallback body.
    ($name:ident, $ferr_path:path, $int_body:expr, complex = $cfn:path) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let scalar = all_scalar_inputs(py, &[x])?;
            let dt = dtype_name(&arr)?;
            // float16 (REQ-5, #955): delegate to numpy (f32-compute, f16-narrow).
            if is_float16_dtype(dt.as_str()) {
                let out = crate::conv::f16_delegate(py, stringify!($name), (&arr,), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let out = if is_complex_dtype(dt.as_str()) {
                $cfn(py, &arr, dt.as_str())?
            } else if matches!(dt.as_str(), "float64" | "f64" | "float32" | "f32") {
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
bind_unary_float_only_int_fallback!(
    square,
    ferray_ufunc::square,
    square_int_array,
    complex = complex_square_dispatch
);
bind_unary_float_only_int_fallback!(
    reciprocal,
    ferray_ufunc::reciprocal,
    reciprocal_int_array,
    complex = complex_reciprocal_dispatch
);

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
            // float16 (REQ-5, #955): predicate output is bool (width-independent);
            // delegate to numpy as the real-only `match_dtype_float!` has no f16 arm.
            if is_float16_dtype(dtype_name(&arr)?.as_str()) {
                return crate::conv::f16_delegate(py, stringify!($name), (&arr,), None);
            }
            Ok(unary_float_predicate_body!(py, arr, $ferr_path))
        }
    };
    // Complex-loop form (#934): numpy registers a complex isnan/isinf/isfinite
    // loop applying the per-part rule (`$kind`) → bool. A complex input computes
    // BEFORE the real `match_dtype_float!` body, which would otherwise raise.
    ($name:ident, $ferr_path:path, complex = $kind:expr) => {
        #[pyfunction]
        pub fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
            let arr = as_ndarray(py, x)?;
            let dt = dtype_name(&arr)?;
            if is_complex_dtype(dt.as_str()) {
                return complex_predicate_dispatch(py, &arr, dt.as_str(), $kind);
            }
            // float16 (REQ-5, #955): delegate to numpy (bool output).
            if is_float16_dtype(dt.as_str()) {
                return crate::conv::f16_delegate(py, stringify!($name), (&arr,), None);
            }
            Ok(unary_float_predicate_body!(py, arr, $ferr_path))
        }
    };
}

bind_predicate_float!(isnan, ferray_ufunc::isnan, complex = ComplexPredicate::Nan);
bind_predicate_float!(isinf, ferray_ufunc::isinf, complex = ComplexPredicate::Inf);
bind_predicate_float!(
    isfinite,
    ferray_ufunc::isfinite,
    complex = ComplexPredicate::Finite
);
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

/// `numpy.add(x1, x2)` — element-wise sum. Datetime64 / timedelta64 operands
/// route through the ferray-ufunc datetime kernels
/// (`crate::datetime::add_time`): `datetime + timedelta -> datetime`,
/// `timedelta + timedelta -> timedelta`. Numeric operands take the
/// NEP-50-promoted broadcast path.
#[pyfunction]
#[pyo3(signature = (x1, x2, out = None))]
pub fn add<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        let result = crate::datetime::add_time(py, x1, x2)?;
        return finish_with_out(out, result, false);
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let result = if binary_involves_float16(py, x1, x2)? {
        f16_binary_delegate(py, x1, x2, "add")?
    } else if binary_is_complex(py, x1, x2)? {
        complex_binary_arith_dispatch!(py, x1, x2, ferray_ufunc::add_broadcast)
    } else {
        binary_numeric_body!(py, x1, x2, ferray_ufunc::add_broadcast)
    };
    finish_with_out(out, result, scalar)
}

/// `numpy.subtract(x1, x2)` — element-wise difference. Datetime64 /
/// timedelta64 operands route through the ferray-ufunc datetime kernels
/// (`crate::datetime::subtract_time`): `datetime - datetime -> timedelta`,
/// `datetime - timedelta -> datetime`, `timedelta - timedelta -> timedelta`.
#[pyfunction]
#[pyo3(signature = (x1, x2, out = None))]
pub fn subtract<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        let result = crate::datetime::subtract_time(py, x1, x2)?;
        return finish_with_out(out, result, false);
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let result = if binary_involves_float16(py, x1, x2)? {
        f16_binary_delegate(py, x1, x2, "subtract")?
    } else if binary_is_complex(py, x1, x2)? {
        complex_binary_arith_dispatch!(py, x1, x2, ferray_ufunc::subtract_broadcast)
    } else {
        binary_numeric_body!(py, x1, x2, ferray_ufunc::subtract_broadcast)
    };
    finish_with_out(out, result, scalar)
}

/// `numpy.multiply(x1, x2)` — element-wise product. Datetime64 / timedelta64
/// operands route through the ferray-ufunc timedelta kernels
/// (`crate::datetime::multiply_time`): `timedelta * int/float -> timedelta`
/// (and the reflected `int/float * timedelta`); `timedelta * timedelta` and
/// `datetime * anything` RAISE numpy's exact `UFuncTypeError` (REQ-2, #942).
/// Numeric operands take the NEP-50-promoted broadcast path.
#[pyfunction]
#[pyo3(signature = (x1, x2, out = None))]
pub fn multiply<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        let result = crate::datetime::multiply_time(py, x1, x2)?;
        return finish_with_out(out, result, false);
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let result = if binary_involves_float16(py, x1, x2)? {
        f16_binary_delegate(py, x1, x2, "multiply")?
    } else if binary_is_complex(py, x1, x2)? {
        complex_binary_arith_dispatch!(py, x1, x2, ferray_ufunc::multiply_broadcast)
    } else {
        binary_numeric_body!(py, x1, x2, ferray_ufunc::multiply_broadcast)
    };
    finish_with_out(out, result, scalar)
}

/// `numpy.divide(x1, x2)` — element-wise true-division. Datetime64 /
/// timedelta64 operands route through `crate::datetime::divide_time`:
/// `timedelta / int/float -> timedelta` (trunc toward zero), `timedelta /
/// timedelta -> float64` (ratio); `int / timedelta` and `datetime / x` RAISE
/// numpy's exact `UFuncTypeError` (REQ-2, #942).
#[pyfunction]
#[pyo3(signature = (x1, x2, out = None))]
pub fn divide<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
    out: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        let result = crate::datetime::divide_time(py, x1, x2)?;
        return finish_with_out(out, result, false);
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let result = if binary_involves_float16(py, x1, x2)? {
        f16_binary_delegate(py, x1, x2, "divide")?
    } else if binary_is_complex(py, x1, x2)? {
        complex_binary_arith_dispatch!(py, x1, x2, ferray_ufunc::divide_broadcast)
    } else {
        binary_numeric_body!(py, x1, x2, ferray_ufunc::divide_broadcast)
    };
    finish_with_out(out, result, scalar)
}

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
    // No-complex-loop form (`heaviside`/`copysign`/`nextafter`/`hypot`/
    // `logaddexp`/`logaddexp2`/`fmod`/`arctan2`): numpy registers ONLY float
    // loops for these — a complex input on EITHER operand RAISES `TypeError`
    // (verified live, numpy 2.4: `np.hypot([1+2j],[1]) -> TypeError`). The prior
    // plain arm funnelled through `binary_float_promote_body!`, whose
    // `coerce_dtype(..,"float32"/"float64")` casts complex -> real DROPPING the
    // imaginary part (a silent lossy boundary round-trip — R-CODE-4) and computed
    // a wrong real result. The guard runs before the real funnel so the error
    // matches numpy's `ufunc 'X' not supported for the input types` (R-DEV-2).
    ($name:ident, $ferr_path:path, reject_complex = $func:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            if binary_is_complex(py, x1, x2)? {
                return Err(reject_complex_binary($func));
            }
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = binary_float_promote_body!(py, x1, x2, $ferr_path);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Complex-loop form (`fmax`/`fmin`): numpy registers a complex loop that
    // selects the whole operand LEXICOGRAPHICALLY `(real, then imag)` with
    // NaN-SUPPRESS semantics (return the non-NaN operand; both-NaN -> `a`). The
    // imaginary part MUST be preserved — the prior real-only funnel silently
    // discarded it (R-CODE-4 corruption). `want_max` picks fmax vs fmin.
    ($name:ident, $ferr_path:path, complex_minmax = $want_max:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            // Integer/bool operands: numpy's `fmax`/`fmin` have integer loops and
            // KEEP the promoted integer dtype (for non-NaN-capable ints they equal
            // `maximum`/`minimum`). The float-promote body below would upcast to
            // float64, so delegate the all-integer case to numpy, which returns
            // the correct int result (and the right scalar-vs-array shape).
            if binary_both_integer(py, x1, x2)? {
                let np = py.import("numpy")?;
                return np.call_method1(stringify!($name), (x1, x2));
            }
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = if binary_is_complex(py, x1, x2)? {
                complex_minmax_dispatch(py, x1, x2, $want_max, false)?
            } else {
                binary_float_promote_body!(py, x1, x2, $ferr_path)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_float_promote!(fmax, ferray_ufunc::fmax, complex_minmax = true);
bind_binary_float_promote!(fmin, ferray_ufunc::fmin, complex_minmax = false);
bind_binary_float_promote!(
    copysign,
    ferray_ufunc::copysign,
    reject_complex = "copysign"
);
bind_binary_float_promote!(hypot, ferray_ufunc::hypot, reject_complex = "hypot");
bind_binary_float_promote!(arctan2, ferray_ufunc::arctan2, reject_complex = "arctan2");
bind_binary_float_promote!(
    logaddexp,
    ferray_ufunc::logaddexp,
    reject_complex = "logaddexp"
);
bind_binary_float_promote!(
    logaddexp2,
    ferray_ufunc::logaddexp2,
    reject_complex = "logaddexp2"
);
bind_binary_float_promote!(
    heaviside,
    ferray_ufunc::heaviside,
    reject_complex = "heaviside"
);

// ---------------------------------------------------------------------------
// Binary numeric-split (separate float + integer loops, same output kind) —
// power, maximum, minimum, floor_divide, remainder/mod.
// ---------------------------------------------------------------------------

macro_rules! bind_binary_numeric_split {
    // Plain form: complex input falls through to the real funnel
    // (`match_dtype_float_or_int!` rejects it with `TypeError`). `maximum`/
    // `minimum` now use the `complex_minmax` arm below (numpy registers a
    // complex-ordering loop, #929); this plain arm remains for any op whose
    // complex input numpy genuinely rejects.
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
    // Complex-loop form (`power`): a complex input computes via `$cfn`
    // (`ferray_ufunc::power_complex`) before the real float/int split. numpy
    // registers a complex `power` loop (`np.power([1+2j],[2+0j]) == [-3+4j]`).
    // A float16 operand delegates to numpy's own `$npfunc` (REQ-3, #953): numpy
    // owns the f16 NEP-50 promotion + float32-compute/float16-narrow contract
    // (`np.power(f16,f16).dtype == float16`, `f16**f32 -> float32`), keeping
    // `half::f16` out of the real funnel.
    ($name:ident, $float_fn:path, $int_fn:path, complex = $cfn:path, f16 = $npfunc:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            // Integer base ** NEGATIVE integer exponent: numpy's integer power
            // loop RAISES `ValueError` (loops.c.src:519 / scalarmath:1553); the
            // ferray `power_int` kernel would silently return 0 (R-CODE-4). Raise
            // numpy's exact message BEFORE the kernel runs. Float base/exponent
            // and non-negative integer exponents are unaffected.
            if power_is_negative_int_exponent(py, x1, x2)? {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Integers to negative integer powers are not allowed.",
                ));
            }
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = if binary_involves_float16(py, x1, x2)? {
                f16_binary_delegate(py, x1, x2, $npfunc)?
            } else if binary_is_complex(py, x1, x2)? {
                complex_binary_arith_dispatch!(py, x1, x2, $cfn)
            } else {
                binary_numeric_split_body!(py, x1, x2, $float_fn, $int_fn)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // No-complex-loop form (`floor_divide`/`remainder`/`mod`): a complex input
    // RAISES `TypeError` (numpy registers NO complex loop for these). The guard
    // runs before the real split so the error matches numpy's
    // `ufunc 'floor_divide' not supported for the input types`.
    ($name:ident, $float_fn:path, $int_fn:path, reject_complex = $func:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            if binary_is_complex(py, x1, x2)? {
                return Err(reject_complex_binary($func));
            }
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            let out = binary_numeric_split_body!(py, x1, x2, $float_fn, $int_fn);
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Complex-minmax form (`maximum`/`minimum`): numpy DOES register a complex
    // loop that selects the whole operand LEXICOGRAPHICALLY `(real, then imag)`
    // with NaN-PROPAGATE semantics (a NaN-part operand propagates; `a` first).
    // The imaginary part is preserved (width c64->c64, c128->c128). `want_max`
    // picks maximum vs minimum.
    ($name:ident, $float_fn:path, $int_fn:path, complex_minmax = $want_max:expr, f16 = $npfunc:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            // datetime64/timedelta64 elementwise extremum (#947): same-kind time
            // operands -> dt|td by int64 tick (NaT propagates; cross-unit promotes
            // to the finer unit). Routed ahead of the real/complex split, which
            // would otherwise raise `TypeError` on the time dtype. A mixed or
            // time-vs-numeric pair falls through (numpy raises there too).
            if crate::datetime::is_time_pair_same_kind(py, x1, x2)? {
                let scalar = all_scalar_inputs(py, &[x1, x2])?;
                let out = crate::datetime::minmax_time(py, x1, x2, $want_max)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            // float16 operand -> delegate to numpy's `$npfunc` (REQ-3, #953):
            // numpy's complex-free `maximum`/`minimum` f16 loop owns the
            // float32-compute / float16-narrow + NEP-50 promotion.
            let out = if binary_involves_float16(py, x1, x2)? {
                f16_binary_delegate(py, x1, x2, $npfunc)?
            } else if binary_is_complex(py, x1, x2)? {
                complex_minmax_dispatch(py, x1, x2, $want_max, true)?
            } else {
                binary_numeric_split_body!(py, x1, x2, $float_fn, $int_fn)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_numeric_split!(
    power,
    ferray_ufunc::power,
    ferray_ufunc::power_int,
    complex = ferray_ufunc::ops::arithmetic::power_complex,
    f16 = "power"
);
bind_binary_numeric_split!(
    maximum,
    ferray_ufunc::maximum,
    ferray_ufunc::maximum_ord,
    complex_minmax = true,
    f16 = "maximum"
);
bind_binary_numeric_split!(
    minimum,
    ferray_ufunc::minimum,
    ferray_ufunc::minimum_ord,
    complex_minmax = false,
    f16 = "minimum"
);
/// `numpy.floor_divide(x1, x2)` — element-wise floor division. Datetime64 /
/// timedelta64 operands route through `crate::datetime::floordiv_time`:
/// `timedelta // int/float -> timedelta` (numpy `#define`s its floor_divide to
/// the divide loop -> trunc toward zero), `timedelta // timedelta -> int64`
/// (true floor); `int // timedelta` and `datetime // x` RAISE numpy's exact
/// `UFuncTypeError` (REQ-2, #942). A complex operand RAISES (no complex loop).
#[pyfunction]
pub fn floor_divide<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        return crate::datetime::floordiv_time(py, x1, x2);
    }
    if binary_involves_float16(py, x1, x2)? {
        let scalar = all_scalar_inputs(py, &[x1, x2])?;
        let out = f16_binary_delegate(py, x1, x2, "floor_divide")?;
        return if scalar { scalarize(out) } else { Ok(out) };
    }
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("floor_divide"));
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let out = binary_numeric_split_body!(
        py,
        x1,
        x2,
        ferray_ufunc::floor_divide,
        ferray_ufunc::floor_divide_int
    );
    if scalar { scalarize(out) } else { Ok(out) }
}

/// `numpy.remainder(x1, x2)` — element-wise Python-style modulo. Datetime64 /
/// timedelta64 operands route through `crate::datetime::mod_time`: only
/// `timedelta % timedelta -> timedelta` (Python floor-mod, sign follows the
/// divisor) is defined; `timedelta % int`, `datetime % x` RAISE numpy's exact
/// `UFuncTypeError` (REQ-2, #942). A complex operand RAISES (no complex loop).
#[pyfunction]
pub fn remainder<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        return crate::datetime::mod_time(py, x1, x2);
    }
    if binary_involves_float16(py, x1, x2)? {
        let scalar = all_scalar_inputs(py, &[x1, x2])?;
        let out = f16_binary_delegate(py, x1, x2, "remainder")?;
        return if scalar { scalarize(out) } else { Ok(out) };
    }
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("remainder"));
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let out = binary_numeric_split_body!(
        py,
        x1,
        x2,
        ferray_ufunc::remainder,
        ferray_ufunc::remainder_int
    );
    if scalar { scalarize(out) } else { Ok(out) }
}

/// `numpy.mod(x1, x2)` — alias of `remainder`. Same datetime/timedelta routing
/// (`timedelta % timedelta -> timedelta`; `timedelta % int`/`datetime % x`
/// RAISE) (REQ-2, #942). A complex operand RAISES (no complex loop).
#[pyfunction]
pub fn mod_<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if crate::datetime::is_time_op(py, x1, x2)? {
        return crate::datetime::mod_time(py, x1, x2);
    }
    if binary_involves_float16(py, x1, x2)? {
        let scalar = all_scalar_inputs(py, &[x1, x2])?;
        // numpy's `mod` is an alias of `remainder`; delegate to either (same loop).
        let out = f16_binary_delegate(py, x1, x2, "mod")?;
        return if scalar { scalarize(out) } else { Ok(out) };
    }
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("remainder"));
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let out = binary_numeric_split_body!(py, x1, x2, ferray_ufunc::mod_, ferray_ufunc::mod_int);
    if scalar { scalarize(out) } else { Ok(out) }
}

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

/// `true` if EITHER comparison operand, viewed as a numpy array, is a *non-real
/// flexible* dtype — kind ∈ {`'U'`,`'S'`,`'V'`,`'O'`} (fixed-width string,
/// structured/void, or object; #964's [`crate::manipulation::is_flexible_array`]).
/// These kinds have no ferray `Element`/`DynArray` variant, so the real-only
/// `comparison_body!` / `complex_*_dispatch!` reject them with `unsupported dtype
/// for numeric op`. numpy, by contrast, defines comparison contracts for each:
///
/// - **object** (`'O'`): COMPUTES element-wise via the Python `==`/`<`/… on the
///   stored objects → a `bool` array (`np.equal(obj,obj2) == [T,F,T]`,
///   `np.less(obj,obj2)` likewise; verified live numpy 2.4.5).
/// - **string** (`'U'`/`'S'`): codepoint-lexicographic compare → `bool` (the
///   #961 contract, formerly gated by a stricter both-operands string check);
///   a string-vs-non-string pair raises numpy's own
///   `UFuncTypeError` (a `TypeError` subclass) — which is *more* numpy-faithful
///   than ferray's prior `unsupported dtype` text.
/// - **structured/void** (`'V'`): numpy registers NO comparison loop —
///   `np.equal(struct,struct)` (and ordering) raise `_UFuncNoLoopError` (a
///   `TypeError` subclass), live.
///
/// Gating `true` when EITHER operand is flexible and DELEGATING the whole op to
/// `np.<op>(x1, x2)` on the ORIGINAL operands lets numpy own each of these: it
/// computes where a loop exists (object/string → `bool`) and raises its own exact
/// exception where none does (structured → `TypeError` subclass). Either-operand
/// (not both) matches numpy's promotion: `object`-vs-`int`/`str` still takes the
/// object loop (`np.equal(obj, int_arr) == [T,F,T]`, live). Keys off the stable
/// `dtype.kind`, NOT `dtype.name` (R-CHAR-3-derived contract). The
/// real/numeric/complex/datetime/float16 compare paths are untouched (their kinds
/// are never in {U,S,V,O}).
fn is_flexible_compare(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    Ok(
        crate::manipulation::is_flexible_array(&aa)?
            || crate::manipulation::is_flexible_array(&ba)?,
    )
}

/// Element-wise comparison of two flexible-kind (string `<U`/`<S`, object, or
/// structured/void) operands by delegating to numpy's own `np.<op>(x1, x2)` on
/// the ORIGINAL operands. numpy owns the per-kind contract: the
/// codepoint-lexicographic string compare with broadcasting and
/// width-independence (`<U2` vs `<U3`), the object Python-`==`/`<` element-wise
/// compare, and the structured/void NO-loop raise. Where a loop exists the result
/// rides the boundary as a numpy `bool` ndarray with no transport (the flexible
/// kinds never enter the Rust library — no `Element`, no `DynArray` variant,
/// R-CODE-4); where none exists numpy's own exception (`UFuncTypeError` /
/// `_UFuncNoLoopError`, both `TypeError` subclasses) propagates. The caller gates
/// on [`is_flexible_compare`] first. `op` is the numpy ufunc name
/// (`less`/`greater`/`less_equal`/`greater_equal`/`equal`/`not_equal`). Mirrors
/// [`crate::datetime::compare_time`] minus the int64-view round-trip (REQ-3,
/// `.design/ferray-core-string.md`; #967 object/structured extension).
fn compare_string<'py>(
    py: Python<'py>,
    op: &str,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    np.getattr(op)?.call1((x1, x2))
}

macro_rules! bind_comparison {
    // Equality form (`equal`/`not_equal`): a complex input computes via the
    // `PartialEq`-bounded `$cfn` (`equal_broadcast`/`not_equal_broadcast`,
    // Complex OK). numpy: `np.equal([1+2j],[1+2j]) == [True]`.
    ($name:ident, $ferr_path:path, eq_complex = $cfn:path, time = $tcmp:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            // datetime64/timedelta64 (same kind) compare by int64 tick value with
            // NaT-unordered semantics (REQ-3, #943), ahead of the real-only path.
            if crate::datetime::is_time_compare(py, x1, x2)? {
                let out = crate::datetime::compare_time(py, x1, x2, $tcmp)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            // flexible kinds U/S/V/O (REQ-3, #961 string + #967 object/structured):
            // numpy defines a comparison contract for each flexible kind that the
            // real-only body below cannot express — string `<U`/`<S` compares
            // codepoint-lexicographically → bool, OBJECT computes element-wise via
            // Python `==` → bool, and structured/void registers NO loop → raises.
            // Detect either operand as flexible and delegate `np.<op>` on the
            // ORIGINAL operands: numpy computes where a loop exists (object/string →
            // bool) and raises its own exact exception where none does (structured →
            // TypeError subclass; string-vs-non-string → UFuncTypeError), matching
            // numpy rather than ferray's `unsupported dtype` text. Mirrors the
            // datetime arm above.
            if is_flexible_compare(py, x1, x2)? {
                let out = compare_string(py, stringify!($name), x1, x2)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            // float16 (REQ-5, #955): numpy registers a float16 compare loop
            // (generate_umath.py:591 `equal` `TD(inexact + times, out='?')`) but
            // the real-only `match_dtype_numeric!` body has no float16 arm; detect
            // a float16 operand and delegate the compare to numpy on the ORIGINAL
            // operands (bool output is width-independent), keeping `half::f16` out
            // of the Rust boundary.
            if binary_involves_float16(py, x1, x2)? {
                let out = crate::conv::f16_delegate(py, stringify!($name), (x1, x2), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let out = if binary_is_complex(py, x1, x2)? {
                complex_eq_dispatch!(py, x1, x2, $cfn)
            } else {
                comparison_body!(py, x1, x2, $ferr_path)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
    // Ordering form (`less`/`greater`/`less_equal`/`greater_equal`): a complex
    // input computes LEXICOGRAPHICALLY `(real, then imag)` — numpy does NOT
    // raise (`np.less([1+2j,3+2j],[1+5j,3+1j]) == [True False]`, live). `$op` is
    // the lexicographic op tag `"lt"|"le"|"gt"|"ge"` (Complex is not
    // `PartialOrd`, so the real `less_broadcast` cannot take it).
    ($name:ident, $ferr_path:path, order_complex = $op:expr, time = $tcmp:expr) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            x1: &Bound<'py, PyAny>,
            x2: &Bound<'py, PyAny>,
        ) -> PyResult<Bound<'py, PyAny>> {
            let scalar = all_scalar_inputs(py, &[x1, x2])?;
            // datetime64/timedelta64 (same kind) order by int64 tick value with
            // NaT-unordered semantics (REQ-3, #943), ahead of the real-only path.
            if crate::datetime::is_time_compare(py, x1, x2)? {
                let out = crate::datetime::compare_time(py, x1, x2, $tcmp)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            // flexible kinds U/S/V/O (REQ-3, #961 string + #967 object/structured):
            // numpy orders strings codepoint-lexicographically → bool and orders
            // OBJECT element-wise via Python `<` → bool, but registers NO ordering
            // loop for structured/void → raises. Detect either operand as flexible
            // and delegate `np.<op>` on the ORIGINAL operands: numpy computes where a
            // loop exists (object/string → bool) and raises its own exact exception
            // where none does (structured → TypeError subclass; string-vs-non-string
            // → UFuncTypeError), matching numpy. Mirrors the datetime arm above.
            if is_flexible_compare(py, x1, x2)? {
                let out = compare_string(py, stringify!($name), x1, x2)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            // float16 (REQ-5, #955): delegate the ordering compare to numpy on the
            // ORIGINAL operands (numpy registers the float16 compare loop —
            // generate_umath.py:567 `less` `TD(inexact + times, out='?')`).
            if binary_involves_float16(py, x1, x2)? {
                let out = crate::conv::f16_delegate(py, stringify!($name), (x1, x2), None)?;
                return if scalar { scalarize(out) } else { Ok(out) };
            }
            let out = if binary_is_complex(py, x1, x2)? {
                complex_order_dispatch(py, x1, x2, $op)?
            } else {
                comparison_body!(py, x1, x2, $ferr_path)
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

use crate::datetime::TimeCompare;

bind_comparison!(
    equal,
    ferray_ufunc::equal_broadcast,
    eq_complex = ferray_ufunc::equal_broadcast,
    time = TimeCompare::Equal
);
bind_comparison!(
    not_equal,
    ferray_ufunc::not_equal_broadcast,
    eq_complex = ferray_ufunc::not_equal_broadcast,
    time = TimeCompare::NotEqual
);
bind_comparison!(
    less,
    ferray_ufunc::less_broadcast,
    order_complex = "lt",
    time = TimeCompare::Less
);
bind_comparison!(
    less_equal,
    ferray_ufunc::less_equal_broadcast,
    order_complex = "le",
    time = TimeCompare::LessEqual
);
bind_comparison!(
    greater,
    ferray_ufunc::greater_broadcast,
    order_complex = "gt",
    time = TimeCompare::Greater
);
bind_comparison!(
    greater_equal,
    ferray_ufunc::greater_equal_broadcast,
    order_complex = "ge",
    time = TimeCompare::GreaterEqual
);

// ---------------------------------------------------------------------------
// Logical (any Logical-implementing dtype → bool)
// ---------------------------------------------------------------------------

/// Which boolean logical reduction a complex binary dispatch should apply.
#[derive(Clone, Copy)]
enum LogicalBinOp {
    And,
    Or,
    Xor,
}

/// Complex arm for `logical_and`/`logical_or`/`logical_xor` (#935): both operands
/// are broadcast + coerced to the common complex width, then the
/// `Logical`-bounded `ferray_ufunc::logical_*` reduces each `Complex` via
/// nonzero-truthiness (`re != 0 || im != 0`) to a bool result. numpy:
/// `np.logical_and([1+2j,0j,..],[2+0j,1-1j,..]) == [True,False,..]` (live). bool
/// output, no imag discard (R-CODE-4). `op` selects the reduction so the three
/// ops share one body; the `Complex<f32>`/`Complex<f64>` arms monomorphise it.
fn complex_logical_binary_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    op: LogicalBinOp,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! logical_arm {
        ($Tc:ty, $arr_a:expr, $arr_b:expr) => {{
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&$arr_a)?;
            let fb: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&$arr_b)?;
            let r: ArrayD<bool> = match op {
                LogicalBinOp::And => ferray_ufunc::logical_and(&fa, &fb),
                LogicalBinOp::Or => ferray_ufunc::logical_or(&fa, &fb),
                LogicalBinOp::Xor => ferray_ufunc::logical_xor(&fa, &fb),
            }
            .map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }};
    }
    let (arr_a, arr_b, dt) = complex_binary_operands(py, a, b)?;
    match dt.as_str() {
        "complex128" | "c16" => logical_arm!(f64, arr_a, arr_b),
        "complex64" | "c8" => logical_arm!(f32, arr_a, arr_b),
        other => Err(PyTypeError::new_err(format!(
            "complex_logical_binary_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// Complex arm for `logical_not` (#935): each `Complex` reduces via
/// nonzero-truthiness to its boolean NOT. numpy: `np.logical_not([0j,1+0j,0+2j])
/// == [True,False,False]` (live). bool output, no imag discard (R-CODE-4).
fn complex_logical_not_dispatch<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    macro_rules! not_arm {
        ($Tc:ty) => {{
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(arr)?;
            let r: ArrayD<bool> = ferray_ufunc::logical_not(&fa).map_err(ferr_to_pyerr)?;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }};
    }
    match dt {
        "complex128" | "c16" => not_arm!(f64),
        "complex64" | "c8" => not_arm!(f32),
        other => Err(PyTypeError::new_err(format!(
            "complex_logical_not_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
}

#[pyfunction]
pub fn logical_and<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return complex_logical_binary_dispatch(py, x1, x2, LogicalBinOp::And);
    }
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_and))
}

#[pyfunction]
pub fn logical_or<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return complex_logical_binary_dispatch(py, x1, x2, LogicalBinOp::Or);
    }
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_or))
}

#[pyfunction]
pub fn logical_xor<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return complex_logical_binary_dispatch(py, x1, x2, LogicalBinOp::Xor);
    }
    Ok(logical_binary_body!(py, x1, x2, ferray_ufunc::logical_xor))
}

#[pyfunction]
pub fn logical_not<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    // Complex logical_not: numpy maps each element to its boolean NOT-truthiness
    // (a complex is truthy iff EITHER part is nonzero — `np.logical_not([0j,
    // 1+0j,0+2j]) == [True,False,False]`, live). The library `Logical for
    // Complex<f32/f64>` (`is_truthy = re!=0 || im!=0`) already encodes this, so
    // the generic `logical_not` folds complex directly — but `match_dtype_all!`
    // (inside `logical_unary_body!`) has no complex arm, so a complex input would
    // otherwise raise `TypeError`. No imag discard (R-CODE-4).
    if is_complex_dtype(dt.as_str()) {
        return complex_logical_not_dispatch(py, &arr, dt.as_str());
    }
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
    // Complex array_equal: same-shape element-wise `==`-all over the complex
    // pair (`numpy/_core/numeric.py:2545` `asarray(a1 == a2).all()`). The library
    // `array_equal<T>` is `PartialEq`-bounded, which `Complex<f32/f64>` satisfy,
    // so it compares complex directly — but `match_dtype_all!` has no complex arm
    // and would raise `TypeError`. numpy: `np.array_equal([1+2j],[1+2j]) == True`
    // (live). No imag discard (R-CODE-4).
    if is_complex_dtype(dt.as_str()) {
        return complex_array_equal(&arr_a, &arr_b, dt.as_str());
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

/// Complex arm for `array_equal` (#935): `True` iff same shape and every complex
/// element compares equal, via the `PartialEq`-bounded `ferray_ufunc::array_equal`.
fn complex_array_equal<'py>(
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<bool> {
    macro_rules! eq_arm {
        ($Tc:ty) => {{
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(a)?;
            let fb: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(b)?;
            Ok(if fa.shape() != fb.shape() {
                false
            } else {
                ferray_ufunc::array_equal(&fa, &fb)
            })
        }};
    }
    match dt {
        "complex128" | "c16" => eq_arm!(f64),
        "complex64" | "c8" => eq_arm!(f32),
        other => Err(PyTypeError::new_err(format!(
            "complex_array_equal: expected a complex dtype, got {other:?}"
        ))),
    }
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
///
/// Both operands promote to the NEP-50 common dtype (`result_type(a, b)`),
/// matching numpy's bitwise type resolver (generate_umath.py:722/733/744
/// register `TD(ints)` and the resolver promotes both operands): a mixed
/// width pair widens to the larger dtype (`int8 & int64 -> int64`) rather
/// than silently keeping operand-1's narrower dtype.
macro_rules! bitwise_binary_body {
    ($py:expr, $a:expr, $b:expr, $func:path) => {{
        let arr_a = as_ndarray($py, $a)?;
        let arr_b0 = as_ndarray($py, $b)?;
        let dt = binary_result_dtype($py, &arr_a, &arr_b0)?;
        let arr_a = coerce_dtype($py, &arr_a, dt.as_str())?;
        let arr_b = coerce_dtype($py, &arr_b0, dt.as_str())?;
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
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("bitwise_and"));
    }
    Ok(bitwise_binary_body!(py, x1, x2, ferray_ufunc::bitwise_and))
}

#[pyfunction]
pub fn bitwise_or<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("bitwise_or"));
    }
    Ok(bitwise_binary_body!(py, x1, x2, ferray_ufunc::bitwise_or))
}

#[pyfunction]
pub fn bitwise_xor<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("bitwise_xor"));
    }
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
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("left_shift"));
    }
    let arr_a0 = as_ndarray(py, x1)?;
    let arr_b0 = as_ndarray(py, x2)?;
    // numpy's shift type resolver (generate_umath.py:761) promotes BOTH
    // operands to `result_type(x1, x2)` for the output dtype; the shift count
    // is then read as an unsigned amount. Coerce the value operand to the
    // promoted dtype rather than keeping operand-1's narrower dtype.
    let dt = binary_result_dtype(py, &arr_a0, &arr_b0)?;
    let arr_a = coerce_dtype(py, &arr_a0, dt.as_str())?;
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
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("right_shift"));
    }
    let arr_a0 = as_ndarray(py, x1)?;
    let arr_b0 = as_ndarray(py, x2)?;
    // numpy's shift type resolver (generate_umath.py:769) promotes BOTH
    // operands to `result_type(x1, x2)` for the output dtype; the shift count
    // is then read as an unsigned amount. Coerce the value operand to the
    // promoted dtype rather than keeping operand-1's narrower dtype.
    let dt = binary_result_dtype(py, &arr_a0, &arr_b0)?;
    let arr_a = coerce_dtype(py, &arr_a0, dt.as_str())?;
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

/// `numpy.sinc(x)` — the normalized sinc, `sin(pi*x)/(pi*x)` with `sinc(0) = 1`
/// (numpy/lib/_function_base_impl.py `sinc`). Real/int input promotes to float.
/// numpy registers a complex loop computing the same `sin(pi*z)/(pi*z)`
/// (`np.sinc([1+2j]) == [-34.09-17.05j]`, `np.sinc([0j]) == [1+0j]`); the
/// complex arm composes it via num_complex (see [`complex_sinc_dispatch`]).
#[pyfunction]
pub fn sinc<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    let scalar = all_scalar_inputs(py, &[x])?;
    let out = if is_complex_dtype(dt.as_str()) {
        complex_sinc_dispatch(py, &arr, dt.as_str())?
    } else {
        unary_float_body!(py, arr, ferray_ufunc::sinc)
    };
    if scalar { scalarize(out) } else { Ok(out) }
}

bind_unary_float!(i0, ferray_ufunc::i0);

/// `numpy.gradient(f, *varargs, axis=None, edge_order=1)` — central-difference
/// gradient. A 1-D input returns a single array; an N-D input returns a tuple of
/// per-axis gradients (one central-difference array per axis, edges one-sided at
/// `edge_order`). `*varargs` is the spacing (scalar `dx` per axis or coordinate
/// arrays); `axis` (int or tuple) restricts the axes computed.
///
/// numpy owns the full N-D central-difference kernel plus the `*varargs`
/// spacing, `axis` (int/tuple/`None`-flatten) and `edge_order` (1|2) contract
/// (`numpy/lib/_function_base_impl.py` `gradient`), and the 1-D-array-vs-N-D-tuple
/// return shape. The prior binding accepted only a 1-D `PyReadonlyArray1` + a
/// single `dx: f64`, so `fr.gradient([[1.,2],[3,4]])` raised
/// `TypeError: 'ndarray' object is not an instance of 'ndarray'` while numpy
/// returns the per-axis tuple (verified live, numpy 2.4:
/// `np.gradient([[1.,2,4],[3,5,9]])` → `(grad_axis0, grad_axis1)`).
///
/// The whole op is delegated to numpy. The argument array crosses the boundary
/// as a numpy ndarray (no lossy f64 coercion or list round-trip — R-CODE-4), so
/// numpy's dtype/shape contract is preserved exactly for every dtype:
/// float32/float64 stay put, integer/bool promote to float64, complex computes
/// the complex gradient, and a datetime64/timedelta64 input yields a
/// timedelta64 result (#946 — datetime differences are timedeltas, no
/// silent-float corruption). The "array too small" / `axis` `AxisError` ride
/// back unchanged.
#[pyfunction]
#[pyo3(signature = (f, *varargs, axis = None, edge_order = 1))]
pub fn gradient<'py>(
    py: Python<'py>,
    f: &Bound<'py, PyAny>,
    varargs: &Bound<'py, pyo3::types::PyTuple>,
    axis: Option<&Bound<'py, PyAny>>,
    edge_order: i64,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, f)?;
    let kwargs = pyo3::types::PyDict::new(py);
    // numpy.gradient's `axis` default is `None` (all axes). Pass the value
    // through verbatim so numpy's own int/tuple/None-flatten handling applies.
    match axis {
        Some(ax) => kwargs.set_item("axis", ax)?,
        None => kwargs.set_item("axis", py.None())?,
    }
    kwargs.set_item("edge_order", edge_order)?;
    // The boundary array becomes the leading positional, followed by the
    // `*varargs` spacing (scalar `dx` per axis or coordinate arrays) verbatim.
    let mut args: Vec<Bound<'py, PyAny>> = Vec::with_capacity(1 + varargs.len());
    args.push(arr);
    for v in varargs.iter() {
        args.push(v);
    }
    py.import("numpy")?
        .getattr("gradient")?
        .call(pyo3::types::PyTuple::new(py, args)?, Some(&kwargs))
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
    // Complex sample values make the trapezoidal integral genuinely complex
    // (`numpy.trapezoid` sums `diff(x)/2 * (y[1:]+y[:-1])` over `asarray(y)`).
    // The real path below coerces `y` (and `x`) to f64, silently discarding the
    // imaginary part (R-CODE-4). Delegate the complex case to numpy.trapezoid,
    // forwarding `x` / `dx`, and return its complex result unchanged. (A complex
    // `x` spacing also forces a complex result, so sniff both operands.)
    let x_complex = match x {
        Some(xa) => is_complex_dtype(dtype_name(&as_ndarray(py, xa)?)?.as_str()),
        None => false,
    };
    if is_complex_dtype(dt.as_str()) || x_complex {
        let kwargs = pyo3::types::PyDict::new(py);
        match x {
            Some(xa) => kwargs.set_item("x", xa)?,
            None => kwargs.set_item("dx", dx)?,
        }
        let np = py.import("numpy")?;
        return np.call_method("trapezoid", (y,), Some(&kwargs));
    }
    // When `x` is provided, the result dtype is `result_type(y, x, float)` per
    // NEP-50: numpy computes `d = diff(x)` (keeping x's dtype,
    // `numpy/lib/_function_base_impl.py:5035`) then `d * (y[1:]+y[:-1]) / 2.0`
    // (`:5048`), promoting across BOTH operands. The native real path below
    // derives the compute dtype from `y` alone and coerces `x` DOWN to it
    // (lossy, R-CODE-4). Delegate the real `x`-provided path to numpy.trapezoid,
    // which owns the exact NEP-50 promotion and returns the right numpy scalar.
    // (The `dx`-only and no-`x` real paths keep the native kernel, whose
    // y-derived compute dtype is already correct.)
    if x.is_some() {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(xa) = x {
            kwargs.set_item("x", xa)?;
        }
        let np = py.import("numpy")?;
        return np.call_method("trapezoid", (y,), Some(&kwargs));
    }
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
        // numpy.trapezoid returns a numpy float SCALAR of the computation dtype
        // (float64 for int/float64 input, float32 for float32), not a Python
        // float. Wrap the value via `numpy.<real_dt>` so `type()` matches numpy.
        let np = py.import("numpy")?;
        let pyf = scalar.into_pyobject(py)?.into_any();
        np.call_method1(real_dt.as_str(), (pyf,))?
    }))
}

/// `numpy.ediff1d(ary, to_end=None, to_begin=None)` — first differences.
/// `to_end`/`to_begin` accept a scalar or any array-like (numpy ravels them).
#[pyfunction]
#[pyo3(signature = (ary, to_end = None, to_begin = None))]
pub fn ediff1d<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    to_end: Option<&Bound<'py, PyAny>>,
    to_begin: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, ary)?;
    let dt = dtype_name(&arr)?;
    // datetime64/timedelta64 ediff1d -> timedelta64 (consecutive differences).
    // Routed through the #947 time dispatch ahead of the real-only
    // `match_dtype_numeric!`. The `to_end`/`to_begin` kwargs are not part of the
    // time divergence pin; the time arm forwards them only when present (numpy
    // validates the time append).
    if crate::datetime::is_time_array(&arr)? {
        return crate::datetime::ediff1d_time(py, &arr, None, None);
    }
    // complex (#972): ediff1d on complex computes genuinely-complex consecutive
    // differences (`numpy/lib/_arraysetops_impl.py` `ediff1d` operates over the
    // original array via `ary[1:] - ary[:-1]`), but `match_dtype_numeric!` is
    // sealed to real numeric dtypes and would reject the complex array. Delegate
    // the complex case to numpy, which owns the complex result (R-CODE-4). The
    // `to_end`/`to_begin` objects (scalar or array-like) are forwarded as-is.
    if is_complex_dtype(dt.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(e) = to_end {
            kwargs.set_item("to_end", e)?;
        }
        if let Some(b) = to_begin {
            kwargs.set_item("to_begin", b)?;
        }
        let np = py.import("numpy")?;
        return np.call_method("ediff1d", (&arr,), Some(&kwargs));
    }
    // float16 (REQ-5, #955): ediff1d preserves the dtype (`np.ediff1d(f16).dtype
    // == float16`, live); delegate to numpy as `match_dtype_numeric!` has no
    // float16 arm. `to_end`/`to_begin` forwarded as-is when present.
    if is_float16_dtype(dt.as_str()) {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(e) = to_end {
            kwargs.set_item("to_end", e)?;
        }
        if let Some(b) = to_begin {
            kwargs.set_item("to_begin", b)?;
        }
        return crate::conv::f16_delegate(py, "ediff1d", (&arr,), Some(&kwargs));
    }
    // Appendage present (#1002/#1003): numpy owns the appendage semantics —
    // `np.asanyarray(x).ravel()` (so an N-D appendage is flattened, #1002) plus
    // the `np.can_cast(x, ary.dtype, casting="same_kind")` gate that RAISES
    // TypeError on an incompatible appendage (e.g. a float on an int array, #1003)
    // (`numpy/lib/_arraysetops_impl.py:98-115`). The prior native path funnelled
    // every appendage through `Vec<f64>` then cast `as T`, which both rejected
    // N-D array-likes and silently truncated 0.5 -> 0 (R-CODE-4). Delegate the
    // whole real-numeric case to numpy when an appendage is supplied, forwarding
    // the raw `to_end`/`to_begin` objects; keep the native kernel only when both
    // appendages are absent.
    if to_end.is_some() || to_begin.is_some() {
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(e) = to_end {
            kwargs.set_item("to_end", e)?;
        }
        if let Some(b) = to_begin {
            kwargs.set_item("to_begin", b)?;
        }
        let np = py.import("numpy")?;
        return np.call_method("ediff1d", (&arr,), Some(&kwargs));
    }
    Ok(crate::match_dtype_numeric!(dt.as_str(), T => {
        let view: numpy::PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_ufunc::ediff1d(&fa, None, None)
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

/// Complex arm for `convolve` (#935): the discrete linear convolution
/// `(a * v)[n] = sum_m a[m] v[n-m]` over complex inputs, computed via the
/// `Add + Mul`-bounded `ferray_ufunc::convolve` at the resolved complex width
/// (c64 only when BOTH operands are c64, else c128 — numpy `result_type`).
/// Unlike `correlate`, convolve does NOT conjugate `v`
/// (`numpy/_core/numeric.py:897`). No imag discard (R-CODE-4). numpy:
/// `np.convolve([1+2j,-3+4j,0j,2-1j],[2+0j,1-1j,3+3j,1+0j])` -> a length-7
/// complex array (live, full mode).
fn complex_convolve_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    dt: &str,
    mode: ferray_ufunc::ConvolveMode,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    macro_rules! convolve_arm {
        ($Tc:ty, $cdt:expr) => {{
            let arr_a = coerce_dtype(py, a, $cdt)?;
            let arr_v = coerce_dtype(py, v, $cdt)?;
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&arr_a)?;
            let fv: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&arr_v)?;
            // Rebuild each side as `Array1` from its flat data (the library
            // `convolve` is 1-D; no `into_dimensionality` on ferray's `Array`).
            let a_data: Vec<Complex<$Tc>> = fa.iter().copied().collect();
            let v_data: Vec<Complex<$Tc>> = fv.iter().copied().collect();
            let fa1: Array1<Complex<$Tc>> =
                Array1::from_vec(Ix1::new([a_data.len()]), a_data).map_err(ferr_to_pyerr)?;
            let fv1: Array1<Complex<$Tc>> =
                Array1::from_vec(Ix1::new([v_data.len()]), v_data).map_err(ferr_to_pyerr)?;
            let r: Array1<Complex<$Tc>> =
                ferray_ufunc::convolve(&fa1, &fv1, mode).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r.into_dyn())
        }};
    }
    match dt {
        "complex128" | "c16" => convolve_arm!(f64, "complex128"),
        "complex64" | "c8" => convolve_arm!(f32, "complex64"),
        other => Err(PyTypeError::new_err(format!(
            "complex_convolve_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
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
    let arr_v0 = as_ndarray(py, v)?;
    let dt = dtype_name(&arr_a)?;
    // Complex convolve: a sliding sum-of-products like correlate, but with NO
    // conjugation of the second operand (`numpy/_core/numeric.py:897` `convolve`
    // = `multiarray.correlate(a, v[::-1], mode)` — it reverses `v` WITHOUT
    // conjugating, unlike `numpy.correlate` which conjugates). The library
    // `ferray_ufunc::convolve` is generic over `T: Add + Mul + Copy`, satisfied
    // by `Complex<f32/f64>`, so it computes complex directly. Branch BEFORE the
    // real coercion below — `match_dtype_numeric!` has no complex arm and would
    // raise `TypeError`. The common width follows numpy's `result_type` (c64 only
    // when BOTH operands are c64). numpy: `np.convolve([1+2j,..],[2+0j,..])`
    // -> `[2+4j, -3+9j, ...]` (live, no imag discard — R-CODE-4).
    {
        let dv = dtype_name(&arr_v0)?;
        if is_complex_dtype(dt.as_str()) || is_complex_dtype(dv.as_str()) {
            let cdt = binary_result_dtype(py, &arr_a, &arr_v0)?;
            return complex_convolve_dispatch(py, &arr_a, &arr_v0, cdt.as_str(), m);
        }
    }
    // numpy promotes BOTH operands to `result_type(a, v)` (NEP-50): convolving an
    // integer sequence with a float kernel yields float64, NOT a truncated int.
    // The prior code coerced `v` to A's dtype (`dt`), so an int `a` truncated the
    // fractional part of a float `v` — `convolve([1,2,3],[0,1,0.5])` returned the
    // int kernel `[0,1,0]`'s result instead of the float one (R-CODE-4).
    let dt = binary_result_dtype(py, &arr_a, &arr_v0)?;
    let arr_a = coerce_dtype(py, a, dt.as_str())?;
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

/// Complex cross-correlation `c[k] = sum_n a[n+k] * conj(v[n])` — numpy
/// CONJUGATES the second operand for complex inputs (numpy/_core/numeric.py:721
/// `correlate`). Mirrors `ferray_stats::correlate`'s convolution-style sliding
/// loop (reverse `v`, accumulate `a[i-j] * v_rev[j]`) but with each `v` element
/// conjugated, keeping the common complex width (c64 when both operands are c64,
/// else c128 — numpy `result_type`). Composed inline because the library
/// `correlate` is `T: Float`-bounded (no `Complex` impl); no imag discard
/// (R-CODE-4).
fn complex_correlate_dispatch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    dt: &str,
    mode: ferray_stats::CorrelateMode,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;

    // Per-width concrete sliding loop. `Complex<f32>`/`Complex<f64>` satisfy the
    // standard `Mul`/`Add`/`Neg` (`num_traits::Num`) used here, so the body stays
    // free of a num-traits dependency at the binding level by being concrete.
    macro_rules! correlate_arm {
        ($Tc:ty, $cdt:expr) => {{
            let arr_a = coerce_dtype(py, a, $cdt)?;
            let arr_v = coerce_dtype(py, v, $cdt)?;
            let fa: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&arr_a)?;
            let fv: ArrayD<Complex<$Tc>> = complex_pyarray_to_ferray::<$Tc>(&arr_v)?;
            let a_data: Vec<Complex<$Tc>> = fa.iter().copied().collect();
            let v_data: Vec<Complex<$Tc>> = fv.iter().copied().collect();
            let la = a_data.len();
            let lv = v_data.len();
            if la == 0 || lv == 0 {
                return Err(PyTypeError::new_err("correlate requires non-empty arrays"));
            }
            // numpy.correlate(a, v) = convolve(a, reverse(conj(v))). Reverse v
            // and conjugate each element up front (matching the real library's
            // reversed sliding loop, with the complex conjugation numpy applies).
            let zero = Complex::<$Tc>::new(0.0, 0.0);
            let v_rev_conj: Vec<Complex<$Tc>> = v_data.iter().rev().map(|z| z.conj()).collect();
            let full_len = la + lv - 1;
            let mut full = vec![zero; full_len];
            for (i, out) in full.iter_mut().enumerate() {
                let mut s = zero;
                for (j, vj) in v_rev_conj.iter().enumerate() {
                    let ai = i as isize - j as isize;
                    if ai >= 0 && (ai as usize) < la {
                        s += a_data[ai as usize] * *vj;
                    }
                }
                *out = s;
            }
            let result: Vec<Complex<$Tc>> = match mode {
                ferray_stats::CorrelateMode::Full => full,
                ferray_stats::CorrelateMode::Same => {
                    let out_len = la.max(lv);
                    let start = (full_len - out_len) / 2;
                    full[start..start + out_len].to_vec()
                }
                ferray_stats::CorrelateMode::Valid => {
                    let out_len = la.max(lv) - la.min(lv) + 1;
                    let start = la.min(lv) - 1;
                    full[start..start + out_len].to_vec()
                }
            };
            let n = result.len();
            let r: Array1<Complex<$Tc>> =
                Array1::from_vec(Ix1::new([n]), result).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, r.into_dyn())
        }};
    }

    match dt {
        "complex128" | "c16" => correlate_arm!(f64, "complex128"),
        "complex64" | "c8" => correlate_arm!(f32, "complex64"),
        other => Err(PyTypeError::new_err(format!(
            "complex_correlate_dispatch: expected a complex dtype, got {other:?}"
        ))),
    }
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
    let arr_v0 = as_ndarray(py, v)?;
    let dt = dtype_name(&arr_a)?;
    // Complex correlate: numpy CONJUGATES the second operand
    // (`np.correlate(a,v) = sum(a[n+k]*conj(v[n]))`, numpy/_core/numeric.py:721
    // `correlate` docstring), computing a complex result. MUST branch BEFORE the
    // float64 coercion below, which numpy casts complex->float64 DROPPING the
    // imaginary part with a `ComplexWarning` — the R-CODE-4 corruption this REQ
    // eliminates (`fr.correlate([1+2j,...],[2+0j,...])` returned the real `[1.0]`;
    // numpy returns the complex `[-3+4j]`). The common complex width follows
    // numpy's `result_type` (c64 only when BOTH operands are c64).
    {
        let dv = dtype_name(&arr_v0)?;
        if is_complex_dtype(dt.as_str()) || is_complex_dtype(dv.as_str()) {
            let cdt = binary_result_dtype(py, &arr_a, &arr_v0)?;
            return complex_correlate_dispatch(py, &arr_a, &arr_v0, cdt.as_str(), m);
        }
    }
    let real_dt = if matches!(dt.as_str(), "float32" | "f32" | "float64" | "f64") {
        dt.as_str().to_string()
    } else {
        "float64".to_string()
    };
    let arr_a = coerce_dtype(py, &arr_a, &real_dt)?;
    let arr_v = coerce_dtype(py, &arr_v0, &real_dt)?;
    Ok(match_dtype_float!(real_dt.as_str(), T => {
        let va: numpy::PyReadonlyArray1<T> = arr_a.extract()?;
        let vv: numpy::PyReadonlyArray1<T> = arr_v.extract()?;
        let fa: Array1<T> = va.as_ferray().map_err(ferr_to_pyerr)?;
        let fv: Array1<T> = vv.as_ferray().map_err(ferr_to_pyerr)?;
        let r: Array1<T> = ferray_stats::correlate(&fa, &fv, m).map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.interp(x, xp, fp, left=None, right=None, period=None)` — 1-D linear
/// interpolation. The native kernel covers the default (no `left`/`right`
/// fill-value overrides, no `period` wraparound); any of those, or a complex
/// `fp`, delegates to numpy, which owns the fill/periodic semantics.
#[pyfunction]
#[pyo3(signature = (x, xp, fp, left = None, right = None, period = None))]
pub fn interp<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    xp: &Bound<'py, PyAny>,
    fp: &Bound<'py, PyAny>,
    left: Option<&Bound<'py, PyAny>>,
    right: Option<&Bound<'py, PyAny>>,
    period: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr_x = as_ndarray(py, x)?;
    let dt = dtype_name(&arr_x)?;
    // numpy.interp supports COMPLEX `fp` values (interpolated re/im independently,
    // `numpy/lib/_function_base_impl.py` `compiled_interp_complex`) and the
    // `left`/`right` fill-value and `period` wraparound kwargs. The real native
    // path coerces `fp` to f64 (dropping a complex imaginary part, R-CODE-4) and
    // ignores those kwargs, so delegate whenever any is in play.
    if is_complex_dtype(dtype_name(&as_ndarray(py, fp)?)?.as_str())
        || left.is_some()
        || right.is_some()
        || period.is_some()
    {
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(l) = left {
            kwargs.set_item("left", l)?;
        }
        if let Some(r) = right {
            kwargs.set_item("right", r)?;
        }
        if let Some(p) = period {
            kwargs.set_item("period", p)?;
        }
        return np.call_method("interp", (x, xp, fp), Some(&kwargs));
    }
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

/// `numpy.fmod(x1, x2)` — C `fmod`, the remainder with the SIGN OF THE
/// DIVIDEND (`fmod(5,-3) == 2`, unlike `remainder`'s sign-of-divisor).
///
/// numpy registers `TD(ints, ...)` FIRST for `fmod` (generate_umath.py:446),
/// so an all-integer operand pair KEEPS the integer dtype (no float
/// promotion): `np.fmod(int64 [5,-5], int64 [3,3]) -> int64 [2,-2]`. Rust's
/// `%` on signed integers truncates toward zero / takes the sign of the
/// dividend — identical to C `fmod` — but ferray's float-only body upcast
/// every integer pair to float64. The all-integer case is delegated to numpy
/// (which owns the exact int result, the int-div-by-zero -> 0 + warning
/// contract instead of a Rust panic, and the scalar-vs-array shape); the
/// float / int+float / complex contracts are unchanged.
#[pyfunction]
pub fn fmod<'py>(
    py: Python<'py>,
    x1: &Bound<'py, PyAny>,
    x2: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if binary_is_complex(py, x1, x2)? {
        return Err(reject_complex_binary("fmod"));
    }
    if binary_both_integer(py, x1, x2)? {
        // numpy keeps the promoted integer dtype (result_type) and returns 0
        // for a zero divisor (with a RuntimeWarning) rather than panicking.
        return py.import("numpy")?.call_method1("fmod", (x1, x2));
    }
    let scalar = all_scalar_inputs(py, &[x1, x2])?;
    let out = binary_float_promote_body!(py, x1, x2, ferray_ufunc::fmod);
    if scalar { scalarize(out) } else { Ok(out) }
}
bind_binary_float_promote!(
    nextafter,
    ferray_ufunc::nextafter,
    reject_complex = "nextafter"
);

/// `gcd` / `lcm` are INTEGER-ONLY in numpy — they register only `TD(ints)`
/// loops (generate_umath.py:1156 `gcd`, :1163 `lcm`), so a float input
/// raises `TypeError` (a `UFuncTypeError`). The binding promotes both
/// inputs to the common integer dtype and routes to ferray's integer
/// `gcd_int`/`lcm_int`; a float dtype falls through to the `TypeError` arm.
///
/// `TD(ints)` covers SIGNED `int8/16/32/64` AND UNSIGNED `uint8/16/32/64`, and
/// ferray-ufunc's `gcd_int`/`lcm_int` now serve every one of those widths via
/// the `GcdAbs` abstraction (the kernels dropped their old `num_traits::Signed`
/// bound — see `.design/ferray-ufunc.md` REQ-27). Both operands are promoted to
/// numpy's COMMON dtype first (`result_type`), so mixed-width / mixed-sign cases
/// dispatch the promoted width: `gcd(int32,int64) -> int64`,
/// `gcd(uint8,uint32) -> uint32`, `gcd(int8,uint8) -> int16` (NEP-50 signed +
/// unsigned promotion). A pair whose common dtype is NOT integer — e.g.
/// `gcd(uint64,int64)`, which numpy promotes to FLOAT64, a dtype gcd has no loop
/// for — falls through to the `TypeError` arm (no float gcd is added).
macro_rules! bind_binary_int_only {
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
                // Unsigned integer dtypes are covered by numpy's `TD(ints)`
                // registration for gcd/lcm (generate_umath.py:1156/:1163) and
                // are now served DOWN in ferray-ufunc's `gcd_int`/`lcm_int` via
                // the `GcdAbs` abstraction (the kernels are no longer
                // `Signed`-bounded). The promoted common dtype is dispatched, so
                // `gcd(uint8,uint32) -> uint32`. A pair that numpy promotes to a
                // non-integer common dtype (e.g. `gcd(uint64,int64) -> float64`)
                // never reaches a uint arm — it hits the `other` TypeError arm.
                "uint64" | "u64" => __gcd_arm!(u64),
                "uint32" | "u32" => __gcd_arm!(u32),
                "uint16" | "u16" => __gcd_arm!(u16),
                "uint8" | "u8" => __gcd_arm!(u8),
                other => {
                    return Err(::pyo3::exceptions::PyTypeError::new_err(format!(
                        "ufunc {:?} not supported for the input types (integer required): {other:?}",
                        stringify!($name)
                    )));
                }
            };
            if scalar { scalarize(out) } else { Ok(out) }
        }
    };
}

bind_binary_int_only!(gcd, ferray_ufunc::gcd_int);
bind_binary_int_only!(lcm, ferray_ufunc::lcm_int);

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
    // numpy registers ONLY float `ldexp` loops — a complex input on EITHER
    // operand RAISES `TypeError` (verified live, numpy 2.4). The prior path
    // funnelled through `coerce_dtype(..,"float64")`, casting complex -> real and
    // DROPPING the imaginary part (a silent lossy boundary round-trip, R-CODE-4).
    if binary_is_complex(py, x, n)? {
        return Err(reject_complex_binary("ldexp"));
    }
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
/// `clip` (numpy/_core/fromnumeric.py:2208) preserves the INPUT array's
/// dtype against Python-scalar bounds under NEP-50 (the array's dtype wins
/// over a weak python int/float): `np.clip(int8 [1,5,9], 2, 7).dtype ==
/// int8`, `float32 -> float32`. The result only promotes when a bound is
/// itself an array of a wider dtype. The prior binding routed clip through
/// the [`maximum`] / [`minimum`] bindings, whose `binary_result_dtype(
/// result_type(a, scalar))` upcast a python-scalar bound to its default
/// int64 / float64, so an int8 array clipped to int64 — wrong dtype.
///
/// numpy owns the exact NEP-50 weak-scalar clip dtype, plus the `None`
/// (one-sided), array-valued, and 0-d/scalar bound contracts, so the
/// binding delegates the whole op to `numpy.clip` on the ORIGINAL operands
/// (the brief's "delegate when native clip can't express weak-scalar
/// promotion"). A `None` bound is forwarded as Python `None`.
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
    let none = py.None();
    let lo_arg = lo.map_or_else(|| none.bind(py).clone(), |v| v.clone());
    let hi_arg = hi.map_or_else(|| none.bind(py).clone(), |v| v.clone());
    py.import("numpy")?
        .call_method1("clip", (a, lo_arg, hi_arg))
}
