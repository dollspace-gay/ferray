//! Bindings for the `numpy` array-creation surface.
//!
//! Three categories of binding live here:
//!
//! 1. **Shape + dtype** (zeros, ones, empty, full, eye, identity, tri,
//!    arange) — driven by [`creation_dispatch`] which monomorphises on
//!    the requested dtype string.
//! 2. **Array → array** (zeros_like, ones_like, full_like, copy,
//!    asarray, ascontiguousarray, asfortranarray, asanyarray, array) —
//!    extract the numpy input via `AsFerray`, run the typed ferray fn,
//!    push back through `IntoNumPy`. The `match_dtype_all!` macro
//!    handles the 11-way dispatch.
//! 3. **Range** (arange, linspace, logspace, geomspace) — float by
//!    default; an integer `dtype` is accepted (numpy floors/casts the
//!    float sequence) and `linspace` carries the `retstep` kwarg.
//!
//! Every binding returns a real `numpy.ndarray` (not a ferray-specific
//! type), so callers using `import ferray as np` see standard NumPy
//! values throughout.
//!
//! ## REQ status
//!
//! Binding-level marshalling conventions for the creation surface. Two
//! states only (goal.md R-DEFER-2). numpy cites use `file:line` against the
//! read-only tree at `/home/doll/numpy-ref` (oracle: numpy 2.4.5); each
//! convention has a non-test production consumer (the `#[pyfunction]` that
//! is the public API boundary, registered in `lib.rs`).
//!
//! | REQ | Status | Evidence |
//! |---|---|---|
//! | RC-INFER (Python-scalar dtype inference) | SHIPPED | `pyscalar_default_dtype` in `conv.rs` keys the default dtype off the operand's *Python type* (bool/int/float), consumed by `arange` and `full` here (`np.full` defaults `dtype` to `np.array(fill_value).dtype`, `numpy/_core/numeric.py:382-384`). Pinned green: `test_arange_single_float_arg_yields_float_dtype`, `test_arange_float_step_int_endpoints_yields_float_dtype`, `test_full_float_fill_yields_float_dtype`, `test_full_bool_fill_yields_bool_dtype`. |
//! | RC-NEGDIM (negative dimension -> ValueError) | SHIPPED | `extract_signed_shape`/`validate_dim` in `conv.rs` extract signed then raise `ValueError("negative dimensions are not allowed")`, consumed by `zeros`/`ones`/`empty`/`full` (shape) and `eye`/`identity` (`n`/`m`). numpy: `np.zeros(-1)`/`np.eye(-1)` -> `ValueError`. Pinned green: `test_zeros_negative_dim_raises_valueerror`, `test_eye_negative_n_raises_valueerror`, `test_identity_negative_n_raises_valueerror`. |
//! | RC-ARANGE0 (zero step -> ZeroDivisionError) | SHIPPED | `arange` raises `PyZeroDivisionError` for `step == 0` (numpy sizes the result by dividing by step). Pinned green: `test_arange_zero_step_raises_zerodivision`. |
//! | RC-ARANGEINT (integer dtype, fractional step) | SHIPPED | `arange_int` counts elements from the float division then fills with integer arithmetic (`T(start) + i*T(step)`), so `np.arange(0,5,0.5,dtype='int64')` is ten zeros. Pinned green: `test_arange_float_step_with_int_dtype_does_not_zero_step_error`. |
//! | RC-LINSPACE (retstep / int dtype / negative num) | SHIPPED | `linspace` binds `num` signed + validates via `validate_num_samples` (`function_base.py:122-126`), returns `(samples, step)` on `retstep=True` (`function_base.py:185-186`), and floors+casts for an integer `dtype` (`function_base.py:181-182`); `logspace`/`geomspace` accept an integer `dtype` via `float_range_to_int_array` (numpy `astype`, `function_base.py:453`). Pinned green: `test_linspace_negative_num_raises_valueerror`, `test_linspace_retstep_returns_tuple`, `test_linspace_integer_dtype_floors_toward_neg_inf`, `test_logspace_integer_dtype_supported`, `test_geomspace_integer_dtype_supported`. |
//! | RC-F16 (float16 dtype) | SHIPPED | The `creation_dispatch` `float16` arm builds the values in f32 via ferray-core then narrows to `float16` with numpy's `astype` (the installed pyo3-numpy has no f16 `NumpyElement`); consumed by `zeros`/`ones`/`empty`/`zeros_like`/`ones_like`. Pinned green: `test_zeros_float16_dtype_supported`, `test_zeros_like_preserves_float16_dtype`. |
//! | RC-DTYPEOBJ (accept dtype type-objects / `np.dtype` instances / str) | SHIPPED | `normalize_dtype`/`normalize_opt_dtype` in `conv.rs` route any `dtype=` object through `numpy.dtype(obj).name` (numpy's `PyArray_DescrConverter` accepts all three forms), consumed by `zeros`/`ones`/`empty`/`full`/`eye`/`identity`/`tri`/`arange`/`linspace`/`logspace`/`geomspace`/`zeros_like`/`ones_like`/`empty_like`/`full_like` here. numpy: `np.zeros(3, dtype=np.float64)` / `np.dtype('int8')` both accepted; `fr.float64 is np.float64`. Pinned green: `tests/test_expansion_dtype_accept.py::test_zeros_accepts_all_dtype_spellings`, `::test_eye_accepts_all_dtype_spellings`, `::test_ferray_reexposed_type_object_is_numpy_type`. |
//! | RC-COMPLEX (complex64/complex128 creation kernel) | SHIPPED | `creation_dispatch!`'s `complex64`/`complex128` arms build `ArrayD<Complex<f32|f64>>` via ferray-core's `Element`-generic `zeros`/`ones` and push them across the boundary with `complex_ferray_to_pyarray` (the shared fft helper); `full_impl` handles complex fills via `extract_complex`. Consumed by `zeros`/`ones`/`empty`/`full`/`full_like`. numpy: `np.zeros(3, dtype='complex128')`, `np.full((2,), 1+2j).dtype == complex128` (numpy/_core/numeric.py:382-384). Pinned green: `tests/test_expansion_dtype_accept.py::test_zeros_complex_dtype`, `::test_full_complex_fill_default_dtype`, `::test_full_complex_explicit_dtype`. |
//! | RC-CARANGE (complex-dtype arange, real args) | SHIPPED | `complex_arange` (consumed by the `arange` `#[pyfunction]` here for `dtype=complex64\|complex128`) builds the real `(start,stop,step)` ramp via `fc::arange::<f64>` then lifts each value to `Complex` with a zero imaginary part, marshalled out via `complex_ferray_to_pyarray`. numpy: `np.arange(0,3,1,dtype='complex128') == [0j,1+0j,2+0j]` (live, numpy 2.4.5). Pinned green: `tests/test_divergence_complex_sweep_audit.py::test_divergence_arange_complex_dtype_raises`; `tests/test_expansion_complex_misc.py::test_arange_complex128_dtype_matches_numpy`, `::test_arange_complex64_dtype_matches_numpy`, `::test_arange_complex_single_arg`. |
//! | RC-CLINSPACE (complex-endpoint linspace) | SHIPPED | `complex_linspace` (consumed by the `linspace` `#[pyfunction]` here when an endpoint is complex or `dtype` is complex) computes `y = arange(0,num)*step + start` with `step = (stop-start)/div`, pinning `y[-1]=stop` on `endpoint`, marshalled via `complex_ferray_to_pyarray`; `retstep` returns `(samples, complex step)` (`numpy/_core/function_base.py:130-176,185-186`). numpy: `np.linspace(0j,1+1j,5) == [0j,0.25+0.25j,0.5+0.5j,0.75+0.75j,1+1j]` (live). Pinned green: `tests/test_divergence_complex_sweep_audit.py::test_divergence_linspace_complex_endpoints_raises`; `tests/test_expansion_complex_misc.py::test_linspace_complex_endpoints_matches_numpy`, `::test_linspace_complex_endpoint_pinned_exact`, `::test_linspace_complex_dtype_real_endpoints`. |
//! | RC-CDCLASS (#938 complex eye/identity/vander/geomspace) | SHIPPED | `eye`/`identity` route their fill (`fc::eye`/`fc::identity`, `Element`-generic) through `match_dtype_all_complex!` + `T::emit_dyn` so `dtype=complex` yields `1+0j`/`0j`; `vander` dispatches complex to `complex_vander_dispatch` (`fc::vander` over `Complex<T>`, complex powers `x^j`); `geomspace` accepts complex endpoints (`extract_complex_scalar`) and computes `complex_geomspace` (`start*(stop/start)**linspace(0,1,num)`). Consumer: the `eye`/`identity`/`vander`/`geomspace` `#[pyfunction]`s registered top-level in `lib.rs` `_ferray`. numpy: `np.eye(2,dtype=complex)`, `np.vander([3+4j,...])`, `np.geomspace(1+1j,8+8j,3)` (live 2.4.5). Pinned green: `tests/test_divergence_complex_converge_audit.py::test_D_eye_complex`/`::test_D_identity_complex`/`::test_D_vander`/`::test_D_geomspace`; `tests/test_expansion_complex_dclass.py` (eye/identity/vander/geomspace cases). |
//! | RC-DATETIME (#941 datetime64/timedelta64 input coercion) | SHIPPED | `array`/`asarray` branch on `crate::datetime::is_time_dtype_name` ahead of `match_dtype_all!`, and `zeros`/`ones`/`empty`/`full`/`zeros_like`/`ones_like`/`empty_like`/`full_like` route the `"M"`/`"m"` case through `time_shape_create` (numpy builds the buffer; the int64-view transport `crate::datetime::datetime_roundtrip` marshals it, preserving unit+shape+NaT, R-CODE-4). Consumer: the `array`/`asarray`/`zeros`/`ones`/`empty`/`full`/`*_like` `#[pyfunction]`s registered top-level in `lib.rs`. numpy: `np.array(['2020-01-01'],dtype='datetime64[D]')`, `np.zeros(3,dtype='datetime64[s]')`, `np.full(2,'2021-06-15',dtype='datetime64[D]')` (live 2.4.5). Pinned green: `tests/test_expansion_datetime_construct.py` (57 cases). |
//! | RC-STRING (#959 fixed-width string `<U`/`<S` input coercion) | SHIPPED | `array`/`asarray` branch on `is_string_array` (sniffs `dtype.kind ∈ {'U','S'}`, NOT the width-encoding `.name` `str64`/`bytes32`) ahead of `match_dtype_all!` and return numpy's string ndarray via `arr.copy()`; `zeros`/`ones`/`empty`/`full`/`zeros_like`/`ones_like`/`empty_like`/`full_like` detect a string `dtype=` off the ORIGINAL object's `.kind` (`dtype_obj_is_string`; the normalised `.name` `str160` does not round-trip through `np.dtype`) — or the source array's string dtype for `*_like` (`string_like_dtype`) — and delegate to numpy via `string_shape_create` (`np.zeros`/`np.full`(shape[,fill],dtype)), returning the `<U`/`<S` ndarray unchanged. No transport: strings have no ferray `Element`/`DynArray` variant (`.design/ferray-core-string.md`, #741), so they ride the boundary as numpy ndarrays (R-CODE-4). `full` also infers `<U`/`<S` width from a string/bytes `fill_value` when no `dtype=` is given (`fill_is_string`). Consumer: the `array`/`asarray`/`zeros`/`ones`/`empty`/`full`/`*_like` `#[pyfunction]`s registered top-level in `lib.rs`. numpy (live 2.4.4): `np.array(['ab','cd'])`→`<U2`, `np.zeros(3,'U5')`→empty strings, `np.full(2,'hi','U4')`, `np.full_like(<U2,'longfill')`→`'lo'`. Pinned green: `tests/test_expansion_string_construct.py` (38 cases). |
//! | RC-ARANGE-TIME (#945 datetime64/timedelta64 arange) | SHIPPED | `arange` branches to `arange_time` here when `dtype` is datetime64/timedelta64 (`crate::datetime::is_time_dtype_name`) or a start/stop/step operand is a datetime64/timedelta64 scalar/array (`crate::datetime::any_time_operand`), AHEAD of the f64-coercion path that raised `TypeError: must be real number, not str`. `arange_time` delegates to `numpy.arange(start[,stop[,step]],dtype=dtype)` on the original operands (numpy owns the string->datetime64 parse, calendar month/year stepping, int/timedelta64 step, half-open `[start,stop)` semantics) then marshals the result back through the int64-view transport (`crate::datetime::datetime_roundtrip`, #941), preserving unit+shape+NaT (R-CODE-4). Consumer: the `arange` `#[pyfunction]` registered top-level in `lib.rs`. numpy 2.4.5: `np.arange('2020-01','2020-04',dtype='datetime64[M]')==['2020-01','2020-02','2020-03']`, `np.arange(5,dtype='timedelta64[D]')==[0..4]`, `np.arange(d0,d1,np.timedelta64(2,'D'))`. Pinned green: `tests/test_expansion_datetime_arange.py`. |
//! | RC-ARANGE-FLOAT (per-element float construction) | NOT-STARTED | `np.arange(1.0,2.0,0.3)` last element is `1.9000000000000001`; ferray-core's float `arange` diverges. Root cause is in ferray-core (`creation/mod.rs`), not this binding — open ferray-core follow-up. Pin: `test_arange_float_last_element_matches_numpy`. |
//! | RC-GEOM-EXACT (geomspace endpoint pinning) | NOT-STARTED | `np.geomspace(-1,-1000,4)` is exactly `[-1,-10,-100,-1000]` via sign rotation + endpoint pinning; ferray-core's `geomspace` lacks both. Root cause in ferray-core, not this binding — open ferray-core follow-up. Pin: `test_geomspace_negative_endpoints_exact`. |

use ferray_core::array::aliases::ArrayD;
use ferray_core::creation as fc;
use ferray_core::dimension::IxDyn;
use ferray_numpy_interop::{AsFerray, IntoNumPy};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyZeroDivisionError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    DynMarshal, as_ndarray, dtype_name, extract_shape, extract_signed_shape, ferr_to_pyerr,
    normalize_dtype, normalize_opt_dtype, pyscalar_default_dtype, validate_dim,
    validate_num_samples,
};
use crate::fft::{complex_ferray_to_pyarray, complex_pyarray_to_ferray};
use crate::{match_dtype_all, match_dtype_all_complex, match_dtype_float};
use num_complex::Complex;

/// Round-trip a complex `numpy.ndarray` (dtype `"complex64"`/`"complex128"`)
/// through ferray-core's `Element`-generic `copy`, returning a fresh
/// Python-owned complex ndarray.
///
/// The 11-real-dtype `match_dtype_all!` macro funnels every arm through the
/// interop `into_pyarray`, which has no `Complex<T>` `NumpyElement` impl, so a
/// complex input previously hit the macro's `TypeError` fallthrough
/// (`fr.array([1+2j])` -> "unsupported dtype: complex128"). numpy's `array`/
/// `asarray`/`copy` accept complex; this branch routes complex through the
/// shared `complex_pyarray_to_ferray` / `complex_ferray_to_pyarray` marshallers
/// (the same pair the fft + creation-dispatch complex arms use), preserving the
/// `complex64`/`complex128` width and shape across the boundary (R-CODE-4). A
/// non-complex `dt` is rejected so the caller only invokes this for the complex
/// arms.
fn complex_roundtrip_copy<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match dt {
        "complex128" | "c16" => {
            let fa: ArrayD<Complex<f64>> = complex_pyarray_to_ferray::<f64>(arr)?;
            let r: ArrayD<Complex<f64>> = fc::copy(&fa);
            complex_ferray_to_pyarray(py, r)
        }
        "complex64" | "c8" => {
            let fa: ArrayD<Complex<f32>> = complex_pyarray_to_ferray::<f32>(arr)?;
            let r: ArrayD<Complex<f32>> = fc::copy(&fa);
            complex_ferray_to_pyarray(py, r)
        }
        other => Err(PyTypeError::new_err(format!(
            "complex_roundtrip_copy: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `true` for the two complex dtype names ferray marshals
/// (`complex64`/`complex128`, plus their `c8`/`c16` aliases).
fn is_complex_dtype(dt: &str) -> bool {
    matches!(dt, "complex128" | "c16" | "complex64" | "c8")
}

/// `true` if a numpy ndarray is a fixed-width string array — unicode (`<U`,
/// dtype kind `'U'`) or bytes (`<S`/`|S`, kind `'S'`).
///
/// String arrays are a *flexible* (itemsize-parameterised) dtype with no ferray
/// `Element` and no `DynArray` variant (`.design/ferray-core-string.md`,
/// `dynarray.rs` rejects `FixedUnicode`/`FixedAscii`/`RawBytes`, #741), so they
/// never enter the Rust library — they ride the boundary as numpy ndarrays. The
/// detection keys off `dtype.kind`, NOT `dtype.name`: `np.dtype('U2').name` is
/// `'str64'`, `np.dtype('S4').name` is `'bytes32'` (the `.name` encodes the bit
/// width and is unstable across widths), while `.kind` is stably `'U'`/`'S'`
/// (verified live, numpy 2.4.5). Mirrors the datetime `time_kind` seam
/// (`datetime.rs`, kind `'M'`/`'m'`), minus the int64-view transport.
fn is_string_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(kind == "U" || kind == "S")
}

/// `true` if a `dtype=` argument names a fixed-width string dtype (unicode `'U'`
/// or bytes `'S'`). Routes the object through `numpy.dtype(obj).kind` (numpy's
/// `PyArray_DescrConverter` accepts a str like `'U5'`/`'S4'`, a `numpy.dtype`
/// instance, or a type object), then sniffs the stable `.kind` — NOT the
/// width-encoding `.name` the real-only `creation_dispatch!` macro matches on
/// (`np.dtype('U5').name == 'str160'`, which does not even round-trip back
/// through `np.dtype(...)`; only `.kind` is reliable). Used by `zeros`/`ones`/
/// `empty`/`full`/`*_like` to detect a string `dtype=` BEFORE the name is
/// normalised, so the original dtype object can be forwarded to numpy intact.
fn dtype_obj_is_string(py: Python<'_>, dtype: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let dt = np.call_method1("dtype", (dtype,))?;
    let kind: String = dt.getattr("kind")?.extract()?;
    Ok(kind == "U" || kind == "S")
}

/// Delegate a shape-based creation call (`zeros`/`ones`/`empty`/`full`) for a
/// fixed-width string `dtype` straight to numpy, returning numpy's string
/// ndarray unchanged.
///
/// String arrays never enter the Rust library (no `Element`, no `DynArray`
/// variant — `.design/ferray-core-string.md`), so there is **no transport**:
/// numpy owns the empty-string / fill-value semantics and the `<U`/`<S` width,
/// and the resulting ndarray rides the boundary directly. The ORIGINAL `dtype`
/// object is forwarded (numpy understands `'U5'`/`'S4'`/a `numpy.dtype`; the
/// normalised `.name` like `'str160'` does NOT round-trip), so width + kind are
/// preserved with no lossy cast (R-CODE-4). `empty` maps to numpy's `zeros`
/// (the binding's defined-buffer contract, matching the real-dtype + datetime
/// arms). This is the string analogue of [`time_shape_create`], minus the
/// int64-view round-trip.
fn string_shape_create<'py>(
    py: Python<'py>,
    np_func: &str,
    shape: &[usize],
    dtype: &Bound<'py, PyAny>,
    fill: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let shape_tuple = pyo3::types::PyTuple::new(py, shape)?;
    // `empty` -> numpy `zeros` (defined buffer), matching the real-dtype arm.
    let func = if np_func == "empty" { "zeros" } else { np_func };
    match fill {
        Some(fv) => np.call_method1(func, (shape_tuple, fv, dtype)),
        None => np.call_method1(func, (shape_tuple, dtype)),
    }
}

/// `true` for the float16 dtype name (plus its `f16` alias).
///
/// float16 cannot ride the 11-real-dtype `match_dtype_all!` macro: the
/// installed pyo3-numpy build has no `NumpyElement`/`PyReadonlyArrayDyn`
/// for `half::f16` (the `numpy/half` feature is off, see
/// `.design/ferray-core-float16.md`), so a `PyReadonlyArrayDyn<f16>` view
/// cannot be taken at the boundary. The `array`/`asarray`/`full` coercion
/// paths therefore detect float16 here and let numpy own the buffer + the
/// f32->f16 narrow, mirroring the SHIPPED `creation_dispatch!` float16 arm.
fn is_float16_dtype(dt: &str) -> bool {
    matches!(dt, "float16" | "f16")
}

// ---------------------------------------------------------------------------
// Shape + dtype creation
// ---------------------------------------------------------------------------

/// Dispatch a shape-only creation function (`zeros`, `ones`, …) over
/// the supported dtypes. Adding a dtype here extends every shape-only
/// binding at once.
///
/// `$func:path` cannot be followed by a turbofish in macro expansion,
/// so we let type inference drive monomorphisation via the explicit
/// `ArrayD<T>` annotation on the binding.
macro_rules! creation_dispatch {
    ($func:path, $py:expr, $shape:expr, $dtype:expr) => {{
        let dim = IxDyn::new(&$shape);
        match $dtype {
            "float64" | "f64" => {
                let arr: ArrayD<f64> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "float32" | "f32" => {
                let arr: ArrayD<f32> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "float16" | "f16" => {
                // The installed pyo3-numpy build has no `NumpyElement` for
                // `half::f16`, so `into_pyarray` can't carry a 16-bit float
                // across the boundary. NumPy *does* support float16, so we
                // construct the values in f32 via ferray-core (f32->f16 is
                // exact for the 0.0/1.0 fills these creation fns produce) and
                // narrow to float16 with numpy's own `astype`, preserving the
                // exact float16 dtype contract (R-CODE-4).
                let arr: ArrayD<f32> = $func(dim).map_err(ferr_to_pyerr)?;
                let f32_any = arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any();
                f32_any
                    .call_method1("astype", ("float16",))?
            }
            "int64" | "i64" => {
                let arr: ArrayD<i64> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "int32" | "i32" => {
                let arr: ArrayD<i32> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "int16" | "i16" => {
                let arr: ArrayD<i16> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "int8" | "i8" => {
                let arr: ArrayD<i8> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "uint64" | "u64" => {
                let arr: ArrayD<u64> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "uint32" | "u32" => {
                let arr: ArrayD<u32> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "uint16" | "u16" => {
                let arr: ArrayD<u16> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "uint8" | "u8" => {
                let arr: ArrayD<u8> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "bool" => {
                let arr: ArrayD<bool> = $func(dim).map_err(ferr_to_pyerr)?;
                arr.into_pyarray($py).map_err(ferr_to_pyerr)?.into_any()
            }
            "complex128" | "c16" => {
                // ferray-core creation is generic over `Element`, which
                // includes `Complex<f64>`; the installed pyo3-numpy has no
                // `NumpyElement`/`into_pyarray` for complex, so we push the
                // result across the boundary via the manual complex helper
                // shared with the fft bindings. numpy's `zeros`/`ones`/`full`
                // support complex dtypes, so this restores parity.
                let arr: ArrayD<Complex<f64>> = $func(dim).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, arr)?
            }
            "complex64" | "c8" => {
                let arr: ArrayD<Complex<f32>> = $func(dim).map_err(ferr_to_pyerr)?;
                complex_ferray_to_pyarray($py, arr)?
            }
            other => {
                return Err(PyTypeError::new_err(format!(
                    "unsupported dtype: {other:?} (supported: bool, int8/16/32/64, uint8/16/32/64, float32/64, complex64/128)"
                )));
            }
        }
    }};
}

/// `numpy.zeros(shape, dtype="float64")` equivalent.
///
/// `dtype` accepts a string, a numpy scalar *type* object (`fr.float64`), or
/// a `numpy.dtype` instance — all normalized through [`normalize_dtype`]
/// (numpy's `PyArray_DescrConverter` accepts the same three forms).
#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
pub fn zeros<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = extract_signed_shape(shape)?;
    // Fixed-width string dtype (`<U`/`<S`): numpy owns the zero-fill (empty
    // strings) + the width, and strings never enter the Rust library, so detect
    // the string `dtype=` off the ORIGINAL object's `.kind` (not the normalised
    // name) and delegate straight to numpy (REQ-1, R-CODE-4).
    if let Some(d) = dtype
        && dtype_obj_is_string(py, d)?
    {
        return string_shape_create(py, "zeros", &shape_vec, d, None);
    }
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "zeros", &shape_vec, &dt, None);
    }
    Ok(creation_dispatch!(fc::zeros, py, shape_vec, dt.as_str()))
}

/// Build a datetime64 / timedelta64 array of a given `shape` for a shape-based
/// creation function (`zeros`/`ones`/`empty`/`full`), routed through the
/// int64-view transport.
///
/// numpy owns the fill semantics for the time dtypes — `np.zeros(n,
/// dtype='datetime64[s]')` is the epoch (ticks `0`), `np.ones(n,
/// dtype='datetime64[D]')` is ticks `1` (NOT the epoch), and
/// `np.full(n, '2021-06-15', dtype='datetime64[D]')` parses the fill string —
/// so this delegates the construction to numpy's `np_func`(shape[, fill],
/// dtype) and then round-trips the resulting datetime64 / timedelta64 buffer
/// through the ferray int64-view transport ([`crate::datetime::datetime_roundtrip`]),
/// preserving unit + shape + NaT with no lossy cast (R-1, R-CODE-4). `empty`
/// maps to numpy's `zeros` (the binding's defined-buffer contract, matching the
/// real-dtype `empty` arm).
fn time_shape_create<'py>(
    py: Python<'py>,
    np_func: &str,
    shape: &[usize],
    dt: &str,
    fill: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let shape_tuple = pyo3::types::PyTuple::new(py, shape)?;
    // `empty` -> numpy `zeros` (defined buffer), matching the real-dtype arm.
    let func = if np_func == "empty" { "zeros" } else { np_func };
    let built = match fill {
        Some(fv) => np.call_method1(func, (shape_tuple, fv, dt))?,
        None => np.call_method1(func, (shape_tuple, dt))?,
    };
    crate::datetime::datetime_roundtrip(py, &built)
}

/// `numpy.ones(shape, dtype="float64")` equivalent. `dtype` accepts a string,
/// a numpy type object, or a `numpy.dtype` instance (see [`zeros`]).
#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
pub fn ones<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = extract_signed_shape(shape)?;
    // String dtype: numpy owns the fill (`np.ones(n, 'U5')` is `'1'` strings)
    // + width; delegate on the original object (REQ-1, R-CODE-4).
    if let Some(d) = dtype
        && dtype_obj_is_string(py, d)?
    {
        return string_shape_create(py, "ones", &shape_vec, d, None);
    }
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "ones", &shape_vec, &dt, None);
    }
    Ok(creation_dispatch!(fc::ones, py, shape_vec, dt.as_str()))
}

/// `numpy.empty(shape, dtype="float64")` equivalent.
///
/// NumPy's `empty` returns uninitialised memory (faster). ferray's
/// `UninitArray` requires unsafe assume_init plumbing that doesn't
/// translate cleanly to Python, so this binding currently allocates a
/// zeroed buffer — the externally observable behaviour (uninitialised
/// values are technically allowed, but most callers immediately
/// overwrite) is a strict superset.
#[pyfunction]
#[pyo3(signature = (shape, dtype = None))]
pub fn empty<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = extract_signed_shape(shape)?;
    // String dtype: `empty` maps to numpy `zeros` (empty strings) per the
    // binding's defined-buffer contract; delegate on the original object
    // (REQ-1, R-CODE-4).
    if let Some(d) = dtype
        && dtype_obj_is_string(py, d)?
    {
        return string_shape_create(py, "empty", &shape_vec, d, None);
    }
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "empty", &shape_vec, &dt, None);
    }
    Ok(creation_dispatch!(fc::zeros, py, shape_vec, dt.as_str()))
}

/// `numpy.identity(n, dtype="float64")` equivalent — n×n identity matrix.
///
/// `n` is bound signed and validated (`np.identity(-2)` -> `ValueError`,
/// not `OverflowError`).
#[pyfunction]
#[pyo3(signature = (n, dtype = None))]
pub fn identity<'py>(
    py: Python<'py>,
    n: isize,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = validate_dim(n)?;
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    // `identity` fills `T::one()` on the diagonal, `T::zero()` elsewhere —
    // generic over `T: Element`, so `dtype=complex` yields `1+0j`/`0j` exactly
    // as numpy (`np.identity(2, dtype=complex)`, verified live). Route the
    // emit through the DynMarshal seam (#933) so complex output keeps its
    // imaginary part.
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let arr = fc::identity::<T>(n).map_err(ferr_to_pyerr)?;
        let dynd = ArrayD::<T>::from_vec(IxDyn::new(arr.shape()), arr.iter().cloned().collect())
            .map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, dynd)?
    }))
}

/// `numpy.eye(n, m=None, k=0, dtype="float64")` — 2-D array with ones
/// on the k-th diagonal.
///
/// `n`/`m` are bound signed and validated (`np.eye(-1)` -> `ValueError`,
/// not `OverflowError`; `eye` routes through `zeros((N, M))` in numpy,
/// `numpy/_core/twodim_base.py`).
#[pyfunction]
#[pyo3(signature = (n, m = None, k = 0, dtype = None))]
pub fn eye<'py>(
    py: Python<'py>,
    n: isize,
    m: Option<isize>,
    k: isize,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = validate_dim(n)?;
    let m = match m {
        Some(m) => validate_dim(m)?,
        None => n,
    };
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    // `eye` fills `T::one()` on the k-th diagonal, `T::zero()` elsewhere —
    // generic over `T: Element`, so `dtype=complex` yields `1+0j`/`0j` exactly
    // as numpy (`np.eye(2, dtype=complex)`, verified live). Route the emit
    // through the DynMarshal seam (#933) so complex output keeps its imaginary
    // part.
    Ok(match_dtype_all_complex!(dt.as_str(), T => {
        let arr = fc::eye::<T>(n, m, k).map_err(ferr_to_pyerr)?;
        let dynd = ArrayD::<T>::from_vec(IxDyn::new(arr.shape()), arr.iter().cloned().collect())
            .map_err(ferr_to_pyerr)?;
        T::emit_dyn(py, dynd)?
    }))
}

/// `numpy.tri(n, m=None, k=0, dtype="float64")` — 2-D array with ones
/// at and below the k-th diagonal.
#[pyfunction]
#[pyo3(signature = (n, m = None, k = 0, dtype = None))]
pub fn tri<'py>(
    py: Python<'py>,
    n: usize,
    m: Option<usize>,
    k: isize,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = m.unwrap_or(n);
    let dt = match dtype {
        Some(d) => normalize_dtype(py, d)?,
        None => "float64".to_string(),
    };
    Ok(match_dtype_all!(dt.as_str(), T => {
        let arr = fc::tri::<T>(n, m, k).map_err(ferr_to_pyerr)?;
        let dynd = ArrayD::<T>::from_vec(IxDyn::new(arr.shape()), arr.iter().cloned().collect())
            .map_err(ferr_to_pyerr)?;
        dynd.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// arange / linspace / logspace / geomspace
// ---------------------------------------------------------------------------

/// `numpy.arange([start,] stop[, step], dtype=None)` equivalent.
///
/// **dtype inference (R-DEV-3).** When `dtype` is omitted numpy keys the
/// output dtype off the *Python type* of the operands, not their numeric
/// value: any Python-`float` operand forces `float64`, otherwise `int64`
/// (`np.arange(3.0).dtype == float64`, `np.arange(0, 3, 1.0).dtype ==
/// float64`). The operands are bound as `&Bound<PyAny>` so [`pyscalar_default_dtype`]
/// can inspect the source type — a value-based `.fract()` heuristic would
/// misclassify an integral-valued float step as integer.
///
/// **Zero step (R-DEV-2).** numpy 2.4 computes the length by dividing by
/// `step`, so a zero step raises `ZeroDivisionError`, not `ValueError`.
///
/// **Integer dtype with a fractional step.** numpy computes the element
/// *count* from the float division (`ceil((stop-start)/step)`) but fills
/// each element with integer arithmetic (`start + i*step` with `step` cast
/// to the integer dtype), so `np.arange(0, 5, 0.5, dtype='int64')` is ten
/// zeros (step `0.5` truncates to `0`). We mirror that exactly.
#[pyfunction]
#[pyo3(signature = (start, stop = None, step = None, dtype = None))]
pub fn arange<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: Option<&Bound<'py, PyAny>>,
    step: Option<&Bound<'py, PyAny>>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dtype: Option<String> = normalize_opt_dtype(py, dtype)?;
    let dtype: Option<&str> = dtype.as_deref();

    // datetime64 / timedelta64 arange (REQ-5, #945). numpy owns the datetime
    // range algebra — string->datetime64 parse, calendar-month/year stepping, an
    // int or timedelta64 step, and the half-open `[start, stop)` semantics — so
    // the real-only path below, which coerces start/stop to f64
    // (`start.extract::<f64>()`) and raises `TypeError: must be real number, not
    // str` on a datetime64 string bound, cannot serve it. When the request is a
    // time arange (an explicit `dtype='datetime64[u]'`/`'timedelta64[u]'`, OR a
    // datetime64/timedelta64 start/stop/step operand) delegate the whole arange
    // to numpy and marshal the resulting datetime64/timedelta64 buffer back
    // through the int64-view transport (#831), preserving numpy's unit + shape +
    // NaT with no lossy cast (R-CODE-4). numpy (live, 2.4.5):
    // `np.arange('2020-01','2020-04',dtype='datetime64[M]')` ==
    // `['2020-01','2020-02','2020-03']`; `np.arange(d0,d1,np.timedelta64(2,'D'))`
    // steps by the timedelta; `np.arange(5,dtype='timedelta64[D]')` == `[0..4]`.
    let dtype_is_time = dtype.is_some_and(crate::datetime::is_time_dtype_name);
    if dtype_is_time || crate::datetime::any_time_operand(py, [Some(start), stop, step])? {
        return arange_time(py, start, stop, step, dtype);
    }

    // Extract numeric values for the length/fill math.
    let start_f: f64 = start.extract()?;
    let (s, e, start_obj, stop_obj) = match stop {
        Some(v) => (start_f, v.extract::<f64>()?, start, Some(v)),
        None => (0.0, start_f, start, None),
    };
    let st: f64 = match step {
        Some(v) => v.extract()?,
        None => 1.0,
    };

    // dtype inference keys off the Python operand *type* (not value).
    // A float anywhere -> float64; else int64. Operands present are
    // start, stop (if given) and step (if given); the implicit start=0
    // and step=1 are Python ints and never force float.
    let inferred = {
        let mut any_float = false;
        for obj in [Some(start_obj), stop_obj, step].into_iter().flatten() {
            match pyscalar_default_dtype(obj) {
                Some("float64") => any_float = true,
                Some(_) => {}
                None => {
                    return Err(PyTypeError::new_err(
                        "arange: start/stop/step must be int, float, or bool",
                    ));
                }
            }
        }
        if any_float { "float64" } else { "int64" }
    };
    let dt = dtype.unwrap_or(inferred);

    // numpy divides by step to size the result -> zero step is a
    // ZeroDivisionError, regardless of dtype.
    if st == 0.0 {
        return Err(PyZeroDivisionError::new_err(
            "division by zero in arange (step cannot be zero)",
        ));
    }

    // Complex-dtype arange: numpy builds the SAME real ramp the float path
    // produces, then casts each element to complex with a zero imaginary part
    // (`np.arange(0,3,1,dtype='complex128') == [0j,1+0j,2+0j]`, live, numpy
    // 2.4.5). The length/fill math is the real `(start, stop, step)` arithmetic;
    // only the output dtype is complex. (numpy's behaviour for genuinely
    // *complex-valued* start/stop/step args is degenerate/undocumented — it
    // collapses to empty or a real-part-driven ramp depending on the operands —
    // and is NOT in scope here; real numeric args with `dtype=complex*` is the
    // pinned, well-defined case.) No imag fabrication beyond numpy's own 0j.
    if matches!(dt, "complex128" | "c16" | "complex64" | "c8") {
        return complex_arange(py, s, e, st, dt);
    }

    let arr_any = match dt {
        "float64" | "f64" => fc::arange::<f64>(s, e, st)
            .map_err(ferr_to_pyerr)?
            .into_pyarray(py)
            .map_err(ferr_to_pyerr)?
            .into_any(),
        "float32" | "f32" => fc::arange::<f32>(s as f32, e as f32, st as f32)
            .map_err(ferr_to_pyerr)?
            .into_pyarray(py)
            .map_err(ferr_to_pyerr)?
            .into_any(),
        "int64" | "i64" => arange_int::<i64>(py, s, e, st)?,
        "int32" | "i32" => arange_int::<i32>(py, s, e, st)?,
        "int16" | "i16" => arange_int::<i16>(py, s, e, st)?,
        "int8" | "i8" => arange_int::<i8>(py, s, e, st)?,
        "uint64" | "u64" => arange_int::<u64>(py, s, e, st)?,
        "uint32" | "u32" => arange_int::<u32>(py, s, e, st)?,
        "uint16" | "u16" => arange_int::<u16>(py, s, e, st)?,
        "uint8" | "u8" => arange_int::<u8>(py, s, e, st)?,
        other => {
            return Err(PyTypeError::new_err(format!(
                "unsupported dtype for arange: {other:?}"
            )));
        }
    };

    Ok(arr_any)
}

/// Build a datetime64 / timedelta64 `arange` (REQ-5, #945) by delegating the
/// range construction to numpy and marshalling the result back through the
/// ferray int64-view transport.
///
/// numpy owns the datetime range algebra (string->datetime64 parse, calendar
/// month/year stepping, an int-or-timedelta64 step, the half-open
/// `[start, stop)` semantics), so this binding builds numpy's exact
/// `numpy.arange(start[, stop[, step]], dtype=dtype)` on the ORIGINAL Python
/// operands — preserving numpy's parse and stepping — then round-trips the
/// resulting datetime64 / timedelta64 buffer through
/// [`crate::datetime::datetime_roundtrip`] (zero-copy `.view('int64')` ->
/// `ArrayD<i64>` -> `int64_to_datetime64`/`int64_to_timedelta64`, #831). This
/// preserves numpy's dtype + unit + shape + NaT with no lossy cast (R-CODE-4)
/// and yields a fresh ferray-marshalled `numpy.ndarray`. The arg list mirrors
/// numpy's positional surface: `stop`/`step` are only forwarded when present, so
/// `fr.arange(5, dtype='timedelta64[D]')` (a single positional treated as the
/// stop, start=0) and `fr.arange(d0, d1, td)` both delegate exactly as numpy.
fn arange_time<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: Option<&Bound<'py, PyAny>>,
    step: Option<&Bound<'py, PyAny>>,
    dtype: Option<&str>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(dt) = dtype {
        kwargs.set_item("dtype", dt)?;
    }
    // Forward the positional operands numpy received, in order. `step` requires
    // `stop` (numpy's positional contract), so a present `step` always pairs with
    // a present `stop`.
    let built = match (stop, step) {
        (Some(stop), Some(step)) => np.call_method("arange", (start, stop, step), Some(&kwargs))?,
        (Some(stop), None) => np.call_method("arange", (start, stop), Some(&kwargs))?,
        (None, _) => np.call_method("arange", (start,), Some(&kwargs))?,
    };
    crate::datetime::datetime_roundtrip(py, &built)
}

/// Build an integer-dtype `arange` matching numpy's count-from-float /
/// fill-with-integer-step semantics.
///
/// The element count is `ceil((stop - start) / step)` computed in float
/// (numpy `_calc_length`); each element is then `T(start) + i*T(step)`,
/// where `T(..)` truncates toward zero — so a fractional step that
/// truncates to zero yields a run of `T(start)` (e.g. ten zeros for
/// `arange(0, 5, 0.5, dtype='int64')`).
fn arange_int<'py, T>(py: Python<'py>, s: f64, e: f64, st: f64) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + FromF64Lossy + std::ops::Add<Output = T> + Copy,
    ArrayD<T>: IntoNumPy<T, IxDyn, PyArrayType = numpy::PyArrayDyn<T>>,
{
    let raw = (e - s) / st;
    let n = if raw <= 0.0 {
        0usize
    } else {
        raw.ceil() as usize
    };
    let s_t = T::from_f64_lossy(s);
    let st_t = T::from_f64_lossy(st);
    let mut data: Vec<T> = Vec::with_capacity(n);
    let mut acc = s_t;
    for _ in 0..n {
        data.push(acc);
        acc = acc + st_t;
    }
    let arr: ArrayD<T> = ArrayD::<T>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Build a complex-dtype `arange` for the in-scope case (real numeric
/// `start`/`stop`/`step`, `dtype=complex64|complex128`): the real ramp numpy's
/// float path produces, with each element cast to complex with a zero imaginary
/// part. numpy: `np.arange(0,3,1,dtype='complex128') == [0j,1+0j,2+0j]` (live).
/// No imag fabrication beyond numpy's `+0j`.
fn complex_arange<'py>(
    py: Python<'py>,
    s: f64,
    e: f64,
    st: f64,
    dt: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // The real ramp via ferray-core's float `arange` (same length/value math the
    // real float path uses); lift each value to `Complex` with `im = 0`.
    let real = fc::arange::<f64>(s, e, st).map_err(ferr_to_pyerr)?;
    let real_flat: Vec<f64> = real.iter().copied().collect();
    let n = real_flat.len();
    macro_rules! ramp_arm {
        ($Tc:ty) => {{
            let zero = 0.0 as $Tc;
            let data: Vec<Complex<$Tc>> = real_flat
                .iter()
                .map(|&v| Complex::new(v as $Tc, zero))
                .collect();
            let arr: ArrayD<Complex<$Tc>> =
                ArrayD::<Complex<$Tc>>::from_vec(IxDyn::new(&[n]), data).map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, arr)
        }};
    }
    match dt {
        "complex128" | "c16" => ramp_arm!(f64),
        "complex64" | "c8" => ramp_arm!(f32),
        other => Err(PyTypeError::new_err(format!(
            "complex_arange: expected a complex dtype, got {other:?}"
        ))),
    }
}

/// `numpy.linspace(start, stop, num=50, endpoint=True, dtype="float64")` equivalent.
/// True if `dtype` names an integer element type. linspace/logspace/
/// geomspace accept an integer `dtype`: numpy computes the sequence in
/// float and then casts (`linspace` floors toward -inf first,
/// `function_base.py:181-182`; logspace/geomspace plain `astype`).
fn is_integer_dtype(dtype: &str) -> bool {
    matches!(
        dtype,
        "int64"
            | "i64"
            | "int32"
            | "i32"
            | "int16"
            | "i16"
            | "int8"
            | "i8"
            | "uint64"
            | "u64"
            | "uint32"
            | "u32"
            | "uint16"
            | "u16"
            | "uint8"
            | "u8"
    )
}

/// Cast a computed `float64` range to an integer-dtype numpy array.
///
/// `floor` mirrors numpy's linspace integer path (`floor(y, out=y)` toward
/// -inf before the cast); logspace/geomspace pass `floor = false` (a plain
/// truncating `astype`). The per-element `as` cast in [`FromF64Lossy`]
/// truncates toward zero, so we apply `f64::floor` first when requested.
fn float_range_to_int_array<'py>(
    py: Python<'py>,
    values: Vec<f64>,
    dtype: &str,
    floor: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let n = values.len();
    let prepared: Vec<f64> = if floor {
        values.into_iter().map(f64::floor).collect()
    } else {
        values
    };
    Ok(match_dtype_all!(dtype, T => {
        let data: Vec<T> = prepared.iter().map(|&v| T::from_f64_lossy(v)).collect();
        let arr: ArrayD<T> = ArrayD::<T>::from_vec(IxDyn::new(&[n]), data)
            .map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `true` if a Python object is a genuine complex value (a Python `complex` or a
/// numpy complex scalar). A real number returns `false` (it extracts as `f64`),
/// so the real linspace/path is unaffected. Mirrors the `extract_complex`
/// component-reading convention (pyo3's optional num-complex `FromPyObject` is
/// not enabled in this crate).
fn is_complex_obj(obj: &Bound<'_, PyAny>) -> bool {
    if obj.extract::<f64>().is_ok() {
        return false;
    }
    // A `complex`/numpy complex scalar exposes a nonzero-capable `.imag`; treat
    // any object with both `.real` and `.imag` numeric attributes as complex.
    obj.getattr("real").and_then(|v| v.extract::<f64>()).is_ok()
        && obj.getattr("imag").and_then(|v| v.extract::<f64>()).is_ok()
}

/// The two complex output widths a complex `linspace` can produce.
#[derive(Clone, Copy)]
enum ComplexWidth {
    /// complex128 (f64 components) — the default for complex endpoints.
    C128,
    /// complex64 (f32 components) — requested via `dtype='complex64'`.
    C64,
}

/// Complex arm for `linspace` (#935): interpolate between complex `start`/`stop`.
/// numpy computes `dt = result_type(start, stop, ensure_inexact=True)` (complex),
/// `step = (stop-start)/div` with `div = num-1` (endpoint) else `num`, then
/// `y = arange(0,num)*step + start`, pinning `y[-1] = stop` when `endpoint` and
/// `num > 1` (`function_base.py:130-176`). `step` is `nan+nanj` when `div <= 0`.
/// numpy: `np.linspace(0j,1+1j,5) == [0j,0.25+0.25j,0.5+0.5j,0.75+0.75j,1+1j]`
/// (live). On `retstep`, returns `(samples, complex step)` matching numpy's tuple
/// (`function_base.py:185-186`). No imag discard (R-CODE-4).
fn complex_linspace<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: &Bound<'py, PyAny>,
    num: usize,
    endpoint: bool,
    retstep: bool,
    width: ComplexWidth,
) -> PyResult<Bound<'py, PyAny>> {
    // Read both endpoints as `Complex<f64>` (a real number yields `im = 0`),
    // reusing the same `.real`/`.imag` component convention as `extract_complex`.
    let s = extract_complex(start)?;
    let e = extract_complex(stop)?;

    let div = if endpoint {
        num as isize - 1
    } else {
        num as isize
    };
    let step = if div > 0 {
        (e - s) / (div as f64)
    } else {
        Complex::new(f64::NAN, f64::NAN)
    };

    // y[i] = start + i*step, with the last point pinned exactly to `stop`.
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(num);
    for i in 0..num {
        data.push(s + step * (i as f64));
    }
    if endpoint && num > 1 {
        data[num - 1] = e;
    }

    let samples = match width {
        ComplexWidth::C128 => {
            let arr: ArrayD<Complex<f64>> =
                ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&[num]), data)
                    .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, arr)?
        }
        ComplexWidth::C64 => {
            let narrowed: Vec<Complex<f32>> = data
                .iter()
                .map(|z| Complex::new(z.re as f32, z.im as f32))
                .collect();
            let arr: ArrayD<Complex<f32>> =
                ArrayD::<Complex<f32>>::from_vec(IxDyn::new(&[num]), narrowed)
                    .map_err(ferr_to_pyerr)?;
            complex_ferray_to_pyarray(py, arr)?
        }
    };

    if retstep {
        // numpy returns the complex `step` scalar; build a Python `complex`.
        let step_obj = pyo3::types::PyComplex::from_doubles(py, step.re, step.im);
        let tuple = pyo3::types::PyTuple::new(py, [samples, step_obj.into_any()])?;
        Ok(tuple.into_any())
    } else {
        Ok(samples)
    }
}

/// `numpy.linspace(start, stop, num=50, endpoint=True, retstep=False,
/// dtype=None)` equivalent.
///
/// - **Negative `num` (R-DEV-2).** `num` is bound signed and validated:
///   `np.linspace(0, 1, -1)` raises `ValueError("Number of samples, -1,
///   must be non-negative.")` (`function_base.py:122-126`).
/// - **`retstep=True`.** Returns the `(samples, step)` tuple
///   (`function_base.py:185-186`); `step` is `(stop-start)/div` with
///   `div = num-1` (endpoint) or `num`, and `nan` when `div <= 0`.
/// - **Integer `dtype`.** Accepted; the float sequence is floored toward
///   -inf then cast (`function_base.py:181-182`), so
///   `np.linspace(-0.5, -5.5, 6, dtype='int64')` is `[-1,-2,-3,-4,-5,-6]`.
/// - **Complex endpoints (#935).** When `start`/`stop` are complex (or the
///   resolved `dtype` is complex), the interpolation is complex: numpy resolves
///   `dt = result_type(start, stop, ensure_inexact=True)` (complex) and computes
///   `y = arange(0,num) * step + start` with `step = (stop-start)/div`, pinning
///   `y[-1] = stop` on `endpoint` (`function_base.py:130-176`). numpy:
///   `np.linspace(0j,1+1j,5) == [0j,0.25+0.25j,0.5+0.5j,0.75+0.75j,1+1j]` (live).
#[pyfunction]
#[pyo3(signature = (start, stop, num = 50, endpoint = true, retstep = false, dtype = None))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors numpy.linspace's full keyword surface"
)]
pub fn linspace<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: &Bound<'py, PyAny>,
    num: isize,
    endpoint: bool,
    retstep: bool,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let num = validate_num_samples(num)?;
    let dt_owned = normalize_opt_dtype(py, dtype)?;

    // Complex path: either endpoint is a genuine complex, OR an explicit complex
    // `dtype` was requested (numpy casts a real-endpoint complex linspace to a
    // complex array with zero imaginary part). Detected before the f64 extraction
    // below, which would otherwise raise `TypeError` on a Python `complex`.
    let endpoints_complex = is_complex_obj(start) || is_complex_obj(stop);
    let dtype_complex = matches!(
        dt_owned.as_deref(),
        Some("complex128") | Some("c16") | Some("complex64") | Some("c8")
    );
    if endpoints_complex || dtype_complex {
        let width = match dt_owned.as_deref() {
            Some("complex64") | Some("c8") => ComplexWidth::C64,
            _ => ComplexWidth::C128, // explicit c128, or inferred from endpoints
        };
        return complex_linspace(py, start, stop, num, endpoint, retstep, width);
    }

    let start: f64 = start.extract()?;
    let stop: f64 = stop.extract()?;
    let dt = dt_owned.as_deref().unwrap_or("float64");

    // numpy's step: delta / div, with div = (num-1) if endpoint else num;
    // undefined (nan) when div <= 0 (function_base.py:153-169).
    let div = if endpoint {
        num as isize - 1
    } else {
        num as isize
    };
    let step = if div > 0 {
        (stop - start) / div as f64
    } else {
        f64::NAN
    };

    let samples = if is_integer_dtype(dt) {
        let values = fc::linspace::<f64>(start, stop, num, endpoint).map_err(ferr_to_pyerr)?;
        let flat: Vec<f64> = values.iter().copied().collect();
        float_range_to_int_array(py, flat, dt, /* floor = */ true)?
    } else {
        match_dtype_float!(dt, T => {
            let arr = fc::linspace::<T>(start as T, stop as T, num, endpoint)
                .map_err(ferr_to_pyerr)?;
            arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
        })
    };

    if retstep {
        let tuple = pyo3::types::PyTuple::new(py, [samples, step.into_pyobject(py)?.into_any()])?;
        Ok(tuple.into_any())
    } else {
        Ok(samples)
    }
}

/// `numpy.logspace(start, stop, num=50, endpoint=True, base=10.0,
/// dtype=None)`. An integer `dtype` is accepted (numpy `astype` at the
/// end): `np.logspace(0, 2, 3, dtype='int64')` is `[1, 10, 100]`.
#[pyfunction]
#[pyo3(signature = (start, stop, num = 50, endpoint = true, base = 10.0, dtype = None))]
#[allow(
    clippy::too_many_arguments,
    reason = "mirrors numpy.logspace's full keyword surface"
)]
pub fn logspace<'py>(
    py: Python<'py>,
    start: f64,
    stop: f64,
    num: isize,
    endpoint: bool,
    base: f64,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let num = validate_num_samples(num)?;
    let dt_owned = normalize_opt_dtype(py, dtype)?;
    let dt = dt_owned.as_deref().unwrap_or("float64");
    if is_integer_dtype(dt) {
        let values =
            fc::logspace::<f64>(start, stop, num, endpoint, base).map_err(ferr_to_pyerr)?;
        let flat: Vec<f64> = values.iter().copied().collect();
        return float_range_to_int_array(py, flat, dt, /* floor = */ false);
    }
    Ok(match_dtype_float!(dt, T => {
        let arr = fc::logspace::<T>(start as T, stop as T, num, endpoint, base)
            .map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None)`.
/// An integer `dtype` is accepted (numpy `result.astype(dtype)`,
/// `function_base.py:453`): `np.geomspace(1, 100, 3, dtype='int64')` is
/// `[1, 10, 100]`.
#[pyfunction]
#[pyo3(signature = (start, stop, num = 50, endpoint = true, dtype = None))]
pub fn geomspace<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
    stop: &Bound<'py, PyAny>,
    num: isize,
    endpoint: bool,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let num = validate_num_samples(num)?;
    let dt_owned = normalize_opt_dtype(py, dtype)?;
    let dt = dt_owned.as_deref();
    // Complex endpoints: numpy computes a complex geometric progression
    // `start * (stop/start) ** linspace(0, 1, num)` (numpy/_core/function_base.py
    // `geomspace`), staying in complex128. Detected when no real dtype is
    // requested and either endpoint is a Python complex (`extract::<f64>` fails).
    let want_complex = matches!(dt, Some("complex128") | Some("c16"))
        || (dt.is_none() && (start.extract::<f64>().is_err() || stop.extract::<f64>().is_err()));
    if want_complex {
        let s = extract_complex_scalar(start)?;
        let e = extract_complex_scalar(stop)?;
        return complex_geomspace(py, s, e, num, endpoint);
    }
    let start_f: f64 = start.extract()?;
    let stop_f: f64 = stop.extract()?;
    let dt = dt.unwrap_or("float64");
    if is_integer_dtype(dt) {
        let values = fc::geomspace::<f64>(start_f, stop_f, num, endpoint).map_err(ferr_to_pyerr)?;
        let flat: Vec<f64> = values.iter().copied().collect();
        return float_range_to_int_array(py, flat, dt, /* floor = */ false);
    }
    Ok(match_dtype_float!(dt, T => {
        let arr = fc::geomspace::<T>(start_f as T, stop_f as T, num, endpoint)
            .map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Extract a Python scalar (`complex`, `float`, or `int`) into a `Complex<f64>`
/// via its `.real`/`.imag` attributes — Python's `complex`/`float`/`int` all
/// expose them, so the same path covers a complex endpoint and a real one
/// promoted to complex.
fn extract_complex_scalar(obj: &Bound<'_, PyAny>) -> PyResult<Complex<f64>> {
    let re: f64 = obj.getattr("real")?.extract()?;
    let im: f64 = obj.getattr("imag")?.extract()?;
    Ok(Complex::new(re, im))
}

/// Complex `geomspace`: `start * (stop/start) ** linspace(0, 1, num)`, computed
/// in complex128 exactly as numpy (numpy/_core/function_base.py `geomspace`,
/// verified live against `np.geomspace(1+1j, 8+8j, 3)`). The `**` for a complex
/// base and real exponent uses `Complex::powf`, matching numpy's complex power.
fn complex_geomspace<'py>(
    py: Python<'py>,
    start: Complex<f64>,
    stop: Complex<f64>,
    num: usize,
    endpoint: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::dimension::IxDyn;
    if start == Complex::new(0.0, 0.0) || stop == Complex::new(0.0, 0.0) {
        return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
            "geomspace: start and stop must be non-zero",
        )));
    }
    // Real-valued exponents linspace(0, 1, num) — endpoint-aware, matching the
    // real geomspace path.
    let exps = fc::linspace::<f64>(0.0, 1.0, num, endpoint).map_err(ferr_to_pyerr)?;
    let ratio = stop / start;
    let mut data: Vec<Complex<f64>> = Vec::with_capacity(num);
    for (i, t) in exps.iter().enumerate() {
        // Pin the endpoints to the exact requested values (numpy assigns
        // `result[0] = start` / `result[-1] = stop`), avoiding last-ULP drift.
        if i == 0 {
            data.push(start);
        } else if endpoint && i == num - 1 {
            data.push(stop);
        } else {
            data.push(start * ratio.powf(*t));
        }
    }
    let r = ArrayD::<Complex<f64>>::from_vec(IxDyn::new(&[num]), data).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray(py, r)
}

// ---------------------------------------------------------------------------
// Array → array (dispatch on input dtype)
// ---------------------------------------------------------------------------

/// `numpy.zeros_like(a, dtype=None)` equivalent.
#[pyfunction]
#[pyo3(signature = (a, dtype = None))]
pub fn zeros_like<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    // String dtype: either an explicit string `dtype=`, or (no dtype) the source
    // array is `<U`/`<S` — preserve the string dtype + width via numpy
    // (REQ-1, R-CODE-4).
    if let Some(d) = string_like_dtype(py, &arr, dtype)? {
        return string_shape_create(py, "zeros", &shape, &d, None);
    }
    let dt = match normalize_opt_dtype(py, dtype)? {
        Some(d) => d,
        None => dtype_name(&arr)?,
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "zeros", &shape, &dt, None);
    }
    Ok(creation_dispatch!(fc::zeros, py, shape, dt.as_str()))
}

/// Resolve the string `dtype` object for a `*_like` call, if the result should
/// be a fixed-width string array — returns `Some(dtype_obj)` when an explicit
/// `dtype=` names a string dtype, or (no explicit dtype) the source array `arr`
/// is itself `<U`/`<S`; otherwise `None`. The returned object is forwarded
/// verbatim to numpy (an explicit `'U4'`/`numpy.dtype`, or `arr.dtype`), so the
/// width is preserved without going through the non-round-tripping `.name`.
fn string_like_dtype<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match dtype {
        Some(d) => {
            if dtype_obj_is_string(py, d)? {
                Ok(Some(d.clone()))
            } else {
                Ok(None)
            }
        }
        None => {
            if is_string_array(arr)? {
                Ok(Some(arr.getattr("dtype")?))
            } else {
                Ok(None)
            }
        }
    }
}

/// `numpy.ones_like(a, dtype=None)` equivalent.
#[pyfunction]
#[pyo3(signature = (a, dtype = None))]
pub fn ones_like<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    // String dtype: `np.ones_like(<U)` is `'1'` strings; preserve via numpy.
    if let Some(d) = string_like_dtype(py, &arr, dtype)? {
        return string_shape_create(py, "ones", &shape, &d, None);
    }
    let dt = match normalize_opt_dtype(py, dtype)? {
        Some(d) => d,
        None => dtype_name(&arr)?,
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "ones", &shape, &dt, None);
    }
    Ok(creation_dispatch!(fc::ones, py, shape, dt.as_str()))
}

/// `numpy.empty_like(prototype, dtype=None)` equivalent. Returns an array
/// of the same shape and dtype as `prototype` (`numpy/_core/numeric.py
/// def empty_like`). NumPy makes no guarantee about the *contents* of an
/// `empty`/`empty_like` buffer; ferray-core's `empty_like` yields an
/// uninitialised buffer that the library always overwrites before use, so
/// at the binding boundary we return a defined (zero-filled) buffer of the
/// matching shape/dtype — the only observable contract numpy specifies.
#[pyfunction]
#[pyo3(signature = (a, dtype = None))]
pub fn empty_like<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    // String dtype: `empty_like` maps to numpy `zeros` (empty strings) per the
    // binding's defined-buffer contract; preserve the string dtype + width.
    if let Some(d) = string_like_dtype(py, &arr, dtype)? {
        return string_shape_create(py, "empty", &shape, &d, None);
    }
    let dt = match normalize_opt_dtype(py, dtype)? {
        Some(d) => d,
        None => dtype_name(&arr)?,
    };
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return time_shape_create(py, "empty", &shape, &dt, None);
    }
    Ok(creation_dispatch!(fc::zeros, py, shape, dt.as_str()))
}

/// `numpy.full_like(a, fill_value, dtype=None)` equivalent. When `dtype`
/// is omitted the result inherits the *source array's* dtype (numpy
/// `full_like` defaults `dtype` to `a.dtype`).
#[pyfunction]
#[pyo3(signature = (a, fill_value, dtype = None))]
pub fn full_like<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    fill_value: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    // String dtype: numpy truncates the fill to the source width
    // (`np.full_like(<U2, 'longfill')` -> `'lo'`); delegate to numpy with the
    // resolved string dtype object (explicit `dtype=`, or the source array's)
    // so the width contract is preserved exactly (REQ-1, R-CODE-4).
    if let Some(d) = string_like_dtype(py, &arr, dtype)? {
        return string_shape_create(py, "full", &shape, &d, Some(fill_value));
    }
    let dt = match normalize_opt_dtype(py, dtype)? {
        Some(d) => d,
        None => dtype_name(&arr)?,
    };
    full_impl(py, &shape, fill_value, dt.as_str())
}

/// `numpy.full(shape, fill_value, dtype=None)` equivalent.
///
/// When `dtype` is omitted numpy defaults it to `np.array(fill_value).dtype`
/// (`numpy/_core/numeric.py:382-384`) — i.e. it keys off the *Python type*
/// of `fill_value`, not its numeric value: a Python `bool` -> `bool`, a
/// Python `int` -> `int64`, a Python `float` (even integral `1.0`) ->
/// `float64`. We bind `fill_value` as `&Bound<PyAny>` and read its type via
/// [`pyscalar_default_dtype`]; a value-based `.fract()` heuristic would mis-
/// classify `1.0` as integer and erase the bool case entirely.
#[pyfunction]
#[pyo3(signature = (shape, fill_value, dtype = None))]
pub fn full<'py>(
    py: Python<'py>,
    shape: &Bound<'py, PyAny>,
    fill_value: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = extract_signed_shape(shape)?;
    // String dtype, explicit (`dtype='U4'`) or inferred from a string/bytes fill
    // (`fr.full(2,'hi')` -> `<U2`). numpy owns the parse + width inference + the
    // `<U`/`<S` fill, and strings never enter the Rust library — detect the
    // string case and delegate straight to numpy, returning its string ndarray
    // unchanged (REQ-1, R-CODE-4). The explicit-dtype branch keys off the
    // original `dtype` object's `.kind`; the no-dtype branch detects a string /
    // bytes `fill_value` (numpy infers `<U{len}`/`<S{len}`).
    match dtype {
        Some(d) if dtype_obj_is_string(py, d)? => {
            return string_shape_create(py, "full", &shape_vec, d, Some(fill_value));
        }
        None if fill_is_string(fill_value) => {
            // No dtype: let numpy infer `<U`/`<S` width from the fill.
            let np = py.import("numpy")?;
            let shape_tuple = pyo3::types::PyTuple::new(py, &shape_vec)?;
            return np.call_method1("full", (shape_tuple, fill_value));
        }
        _ => {}
    }
    let dt = match normalize_opt_dtype(py, dtype)? {
        Some(d) => d,
        // numpy defaults `dtype` to `np.array(fill_value).dtype`
        // (numpy/_core/numeric.py:382-384). For a Python int/float/bool the
        // type-keyed inference applies; for a Python `complex` (and any other
        // array-like fill) we route through numpy's own `np.asarray` so the
        // inferred dtype (e.g. `complex128` for `1+2j`) exactly matches numpy.
        None => match pyscalar_default_dtype(fill_value) {
            Some(d) => d.to_string(),
            None => dtype_name(&as_ndarray(py, fill_value)?)?,
        },
    };
    full_impl(py, &shape_vec, fill_value, dt.as_str())
}

/// `true` if a `full`/`full_like` `fill_value` is a Python `str` or `bytes`
/// (numpy infers a `<U{len}`/`<S{len}` dtype from it when no `dtype=` is given).
fn fill_is_string(fill_value: &Bound<'_, PyAny>) -> bool {
    fill_value.is_instance_of::<pyo3::types::PyString>()
        || fill_value.is_instance_of::<pyo3::types::PyBytes>()
}

fn full_impl<'py>(
    py: Python<'py>,
    shape: &[usize],
    fill_value: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    // datetime64 / timedelta64 fill: numpy parses the fill (e.g. the string
    // `'2021-06-15'`) and owns the unit math, so delegate `np.full(shape,
    // fill, dtype)` and round-trip the resulting time buffer through the
    // int64-view transport (R-1, R-CODE-4). Routed before the f64 extraction
    // below, which would raise on a datetime string.
    if crate::datetime::is_time_dtype_name(dtype) {
        return time_shape_create(py, "full", shape, dtype, Some(fill_value));
    }
    let dim = IxDyn::new(shape);
    // Complex fills carry both parts; extract a `Complex<f64>` from the Python
    // object (a Python `complex`, a numpy complex scalar, or a real number all
    // extract) and build the complex array via the shared complex helper.
    if matches!(dtype, "complex128" | "c16") {
        let fv: Complex<f64> = extract_complex(fill_value)?;
        let arr: ArrayD<Complex<f64>> = fc::full(dim, fv).map_err(ferr_to_pyerr)?;
        return complex_ferray_to_pyarray(py, arr);
    }
    if matches!(dtype, "complex64" | "c8") {
        let fv: Complex<f64> = extract_complex(fill_value)?;
        let fv = Complex::new(fv.re as f32, fv.im as f32);
        let arr: ArrayD<Complex<f32>> = fc::full(dim, fv).map_err(ferr_to_pyerr)?;
        return complex_ferray_to_pyarray(py, arr);
    }
    // float16 fill: the macro below has no `half::f16` arm (no `NumpyElement`),
    // so delegate `np.full(shape, fill, 'float16')` to numpy, which owns the
    // f32->f16 round-to-nearest-even narrow of the fill value and builds the
    // float16 buffer directly. Preserves numpy's exact float16 dtype + shape
    // contract instead of raising (REQ-1, R-CODE-4); mirrors the SHIPPED
    // `creation_dispatch!` float16 arm.
    if is_float16_dtype(dtype) {
        let np = py.import("numpy")?;
        let shape_tuple = pyo3::types::PyTuple::new(py, shape)?;
        return np.call_method1("full", (shape_tuple, fill_value, "float16"));
    }
    let fill_f: f64 = fill_value.extract()?;
    Ok(match_dtype_all!(dtype, T => {
        let fv = T::from_f64_lossy(fill_f);
        let arr: ArrayD<T> = fc::full(dim, fv).map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Extract a `Complex<f64>` fill value from a Python object — a Python
/// `complex`, a numpy complex scalar, or any real number (real numbers
/// extract with a zero imaginary part, matching `np.full((2,), 3,
/// dtype=complex)`).
fn extract_complex(obj: &Bound<'_, PyAny>) -> PyResult<Complex<f64>> {
    // A real number extracts directly with a zero imaginary part. Otherwise
    // read `.real`/`.imag`, which both a Python `complex` and a numpy complex
    // scalar expose (pyo3's optional num-complex `FromPyObject` is not
    // enabled in this crate, so we read the components by attribute).
    if let Ok(re) = obj.extract::<f64>() {
        return Ok(Complex::new(re, 0.0));
    }
    let re: f64 = obj.getattr("real").and_then(|v| v.extract()).map_err(|_| {
        PyTypeError::new_err("full: complex fill_value must be a number or complex")
    })?;
    let im: f64 = obj.getattr("imag").and_then(|v| v.extract()).map_err(|_| {
        PyTypeError::new_err("full: complex fill_value must be a number or complex")
    })?;
    Ok(Complex::new(re, im))
}

/// Local trait for fill-value coercion. Avoids pulling in `num-traits`
/// just for this one path; the conversion is intentionally lossy
/// (e.g. `full((3,), 1.5, dtype="int64")` produces `[1, 1, 1]`,
/// matching NumPy).
trait FromF64Lossy {
    fn from_f64_lossy(v: f64) -> Self;
}

macro_rules! impl_from_f64_lossy {
    ($($t:ty),*) => {
        $(impl FromF64Lossy for $t { fn from_f64_lossy(v: f64) -> Self { v as Self } })*
    };
}
impl_from_f64_lossy!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);

impl FromF64Lossy for bool {
    fn from_f64_lossy(v: f64) -> Self {
        v != 0.0
    }
}

/// `numpy.copy(a)` — return a shallow copy. Supports any dtype.
#[pyfunction]
pub fn copy<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    if is_complex_dtype(dt.as_str()) {
        return complex_roundtrip_copy(py, &arr, dt.as_str());
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fc::copy(&fa);
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.ascontiguousarray(a)` — return a C-contiguous copy.
#[pyfunction]
pub fn ascontiguousarray<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fc::ascontiguousarray(&fa);
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.asfortranarray(a)` — return an F-contiguous copy.
///
/// **Known gap**: ferray-numpy-interop's `IntoNumPy` pipeline
/// allocates a `PyArray1` and reshapes, producing a C-contiguous
/// output regardless of ferray's internal layout. The data is
/// numerically correct but `.flags["F_CONTIGUOUS"]` will be `False`
/// until the interop layer learns to preserve layout. Tracked
/// against `ferray-numpy-interop`.
#[pyfunction]
pub fn asfortranarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fc::asfortranarray(&fa);
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.asanyarray(a)` — return the input as an array (no copy when
/// already an ndarray of the right dtype). ferray's implementation
/// always materialises a fresh buffer; semantically still correct.
#[pyfunction]
pub fn asanyarray<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let arr = as_ndarray(py, a)?;
    let dt = dtype_name(&arr)?;
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fc::asanyarray(&fa);
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.array(object, dtype=None)` — construct a new array from an
/// array-like. The implementation is a thin wrapper that routes
/// through `numpy.asarray` for the input normalisation and then takes
/// a fresh copy through ferray to validate the buffer.
#[pyfunction]
#[pyo3(signature = (obj, dtype = None))]
pub fn array<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    // numpy.asarray accepts a str / type object / dtype instance natively;
    // pass the object through unchanged so `dtype=fr.float64` works.
    let arr = match dtype {
        Some(d) => np.call_method1("asarray", (obj, d))?,
        None => np.call_method1("asarray", (obj,))?,
    };
    // Round-trip through ferray to exercise the marshalling path
    // and produce a fresh owned buffer (matches NumPy `array` semantics
    // which always copies unless `copy=False` is passed).
    let dt = dtype_name(&arr)?;
    // Complex input cannot ride the 11-real-dtype macro (no complex
    // `into_pyarray`); route it through the complex marshaller so
    // `fr.array([1+2j])` builds a complex ndarray instead of raising.
    if is_complex_dtype(dt.as_str()) {
        return complex_roundtrip_copy(py, &arr, dt.as_str());
    }
    // datetime64 / timedelta64 input (a string-list + `dtype='datetime64[D]'`,
    // or a numpy datetime/timedelta array) likewise can't ride the real-only
    // macro (no `M`/`m` arm → `TypeError`). numpy has already parsed the strings
    // / preserved the unit in `arr`; round-trip it through the int64-view
    // transport so `fr.array(['2020-01-01'], dtype='datetime64[D]')` builds a
    // datetime64 ndarray preserving unit + shape + NaT instead of raising
    // (R-1, R-CODE-4).
    if crate::datetime::is_time_dtype_name(dt.as_str()) {
        return crate::datetime::datetime_roundtrip(py, &arr);
    }
    // Fixed-width string input (a str/bytes list with no dtype -> numpy infers
    // `<U{maxwidth}`/`<S{maxwidth}`, or an explicit `dtype='U3'`/`'S4'`, or a
    // numpy `<U`/`<S` array) cannot ride the 11-real-dtype macro (no string
    // `Element` -> the macro's `TypeError: unsupported dtype: "str64"`). numpy
    // has already parsed the input into a `<U`/`<S` buffer in `arr` (via the
    // `np.asarray(obj, dtype)` above, which owns the U-width inference + the
    // explicit width parse), so return a fresh owned copy of that string
    // ndarray — matching numpy's `array` always-copies contract and preserving
    // the string dtype + width + shape across the boundary instead of raising
    // (REQ-1, R-CODE-4). Detection keys off `dtype.kind` ∈ {U,S}, NOT the
    // width-encoding `.name` (`str64`/`bytes32`). Strings never enter the Rust
    // library; mirrors the datetime/float16 detect-and-delegate arms.
    if is_string_array(&arr)? {
        return arr.call_method0("copy");
    }
    // float16 input (a number-list + `dtype='float16'`, or a numpy float16
    // array) cannot ride the 11-real-dtype macro (no `NumpyElement` for
    // `half::f16` -> the macro's `TypeError`). numpy has already parsed the
    // values into a float16 buffer in `arr` (via `np.asarray(obj, 'float16')`
    // above, applying its own round-to-nearest-even f32->f16 narrow), so return
    // a fresh owned copy of that float16 ndarray — matching numpy's `array`
    // always-copies contract and preserving the float16 dtype + shape across the
    // boundary instead of raising (REQ-1, R-CODE-4). Mirrors the SHIPPED
    // `creation_dispatch!` float16 arm.
    if is_float16_dtype(dt.as_str()) {
        return arr.call_method0("copy");
    }
    Ok(match_dtype_all!(dt.as_str(), T => {
        let view: PyReadonlyArrayDyn<T> = arr.extract()?;
        let fa: ArrayD<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = fc::copy(&fa);
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// ---------------------------------------------------------------------------
// from* family (#692)
// ---------------------------------------------------------------------------

/// `numpy.frombuffer(buffer, dtype='float64', count=-1, offset=0)`.
///
/// Accepts Python `bytes` or any object exposing the buffer protocol
/// that PyO3 can extract as `Vec<u8>`. The buffer length must be a
/// multiple of `sizeof(dtype)`; the resulting array is 1-D with the
/// inferred length.
#[pyfunction]
#[pyo3(signature = (buffer, dtype = "float64", count = -1, offset = 0))]
pub fn frombuffer<'py>(
    py: Python<'py>,
    buffer: Vec<u8>,
    dtype: &str,
    count: i64,
    offset: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if offset > buffer.len() {
        return Err(PyTypeError::new_err("frombuffer: offset > buffer length"));
    }
    let buf = &buffer[offset..];

    Ok(match_dtype_all!(dtype, T => {
        let elem = std::mem::size_of::<T>().max(1);
        let n_elements = if count < 0 {
            buf.len() / elem
        } else {
            count as usize
        };
        let needed = n_elements * elem;
        if buf.len() < needed {
            return Err(PyTypeError::new_err(format!(
                "frombuffer: buffer too short ({} bytes) for {} elements of {} bytes",
                buf.len(), n_elements, elem,
            )));
        }
        let dim = ferray_core::dimension::Ix1::new([n_elements]);
        let arr: ferray_core::array::aliases::Array1<T> =
            fc::frombuffer::<T, _>(dim, &buf[..needed]).map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.fromiter(iterable, dtype, count=-1)`.
#[pyfunction]
#[pyo3(signature = (iterable, dtype = "float64", count = -1))]
pub fn fromiter<'py>(
    py: Python<'py>,
    iterable: &Bound<'py, PyAny>,
    dtype: &str,
    count: i64,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    let iter = iterable.try_iter()?;
    let mut take_n: Option<usize> = if count >= 0 {
        Some(count as usize)
    } else {
        None
    };

    Ok(match_dtype_all!(dtype, T => {
        let mut data: Vec<T> = Vec::new();
        for item in iter {
            if let Some(remaining) = take_n {
                if remaining == 0 {
                    break;
                }
                take_n = Some(remaining - 1);
            }
            let py_item = item?;
            let val: T = match py_item.extract::<T>() {
                Ok(v) => v,
                Err(_) => {
                    // Fallback: extract as f64 then cast lossily.
                    let f: f64 = py_item.extract()?;
                    <T as PadConstantCast>::from_f64(f)
                }
            };
            data.push(val);
        }
        let n = data.len();
        let arr: Array1<T> = Array1::<T>::from_vec(Ix1::new([n]), data)
            .map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

// Local f64-cast trait for fromiter's fallback. Mirrors the helper in
// manipulation.rs but kept private to creation.rs to avoid a circular
// re-export.
trait PadConstantCast {
    fn from_f64(v: f64) -> Self;
}
macro_rules! impl_constant_cast {
    ($($t:ty),*) => {
        $(impl PadConstantCast for $t { fn from_f64(v: f64) -> Self { v as Self } })*
    };
}
impl_constant_cast!(f64, f32, i64, i32, i16, i8, u64, u32, u16, u8);
impl PadConstantCast for bool {
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }
}

/// `numpy.fromstring(string, dtype=float, count=-1, sep=' ')`.
///
/// Parses a string of separated numbers into a 1-D array. Empty `sep`
/// is rejected (NumPy treats that as binary mode, which we don't support;
/// use `frombuffer(s.encode(), dtype=...)` for binary parsing).
#[pyfunction]
#[pyo3(signature = (s, dtype = "float64", count = -1, sep = " "))]
pub fn fromstring<'py>(
    py: Python<'py>,
    s: &str,
    dtype: &str,
    count: i64,
    sep: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if sep.is_empty() {
        return Err(PyTypeError::new_err(
            "fromstring: empty sep (binary mode) is not supported; use frombuffer",
        ));
    }
    let _ = count; // ferray's fromstring parses everything; truncation happens after.
    Ok(match_dtype_all!(dtype, T => {
        let arr: ferray_core::array::aliases::Array1<T> =
            fromstring_dispatch::<T>(s, sep).map_err(ferr_to_pyerr)?;
        let arr = if count >= 0 && (count as usize) < arr.shape()[0] {
            // Truncate to count elements via slicing into a new Array1.
            let n = count as usize;
            let data: Vec<T> = arr.iter().take(n).cloned().collect();
            ferray_core::array::aliases::Array1::<T>::from_vec(
                ferray_core::dimension::Ix1::new([n]),
                data,
            )
            .map_err(ferr_to_pyerr)?
        } else {
            arr
        };
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Helper: dispatch `fromstring` over the supported element types via
/// the underlying `FromStr` bound.
fn fromstring_dispatch<T>(
    s: &str,
    sep: &str,
) -> Result<ferray_core::array::aliases::Array1<T>, ferray_core::FerrayError>
where
    T: ferray_core::Element + std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    fc::fromstring::<T>(s, sep)
}

/// `numpy.fromfile(file, dtype=float, count=-1, sep='')`.
///
/// Currently supports text mode (non-empty `sep` only). Binary mode
/// requires platform-specific buffer handling and is deferred — use
/// `frombuffer(open(path, "rb").read(), dtype=...)` for that path.
#[pyfunction]
#[pyo3(signature = (path, dtype = "float64", count = -1, sep = " "))]
pub fn fromfile<'py>(
    py: Python<'py>,
    path: &str,
    dtype: &str,
    count: i64,
    sep: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if sep.is_empty() {
        return Err(PyTypeError::new_err(
            "fromfile: empty sep (binary mode) is not supported; read the bytes and use frombuffer",
        ));
    }
    let s = std::fs::read_to_string(path).map_err(|e| PyTypeError::new_err(e.to_string()))?;
    fromstring(py, &s, dtype, count, sep)
}

/// `numpy.fromfunction(function, shape, dtype=float)`.
///
/// Calls `function(*indices)` for each index in the shape's grid and
/// collects the results into a fresh array. NumPy passes `meshgrid`
/// arrays of indices for vectorised evaluation; we call per-index and
/// rely on the user's function to be cheap. Vectorised mode is a
/// separate follow-up.
#[pyfunction]
#[pyo3(signature = (function, shape, dtype = "float64"))]
pub fn fromfunction<'py>(
    py: Python<'py>,
    function: &Bound<'py, PyAny>,
    shape: &Bound<'py, PyAny>,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let shape_vec = extract_shape(shape)?;
    let total: usize = shape_vec.iter().product();

    Ok(match_dtype_all!(dtype, T => {
        let mut data: Vec<T> = Vec::with_capacity(total);
        // Iterate row-major over the shape, calling `function` with the
        // unpacked index tuple.
        let ndim = shape_vec.len();
        let mut idx = vec![0usize; ndim];
        for _ in 0..total {
            let py_idx = pyo3::types::PyTuple::new(py, idx.iter().copied())?;
            let result = function.call1(py_idx.into_pyobject(py)?)?;
            let val: T = match result.extract::<T>() {
                Ok(v) => v,
                Err(_) => {
                    let f: f64 = result.extract()?;
                    <T as PadConstantCast>::from_f64(f)
                }
            };
            data.push(val);
            // Increment row-major.
            for i in (0..ndim).rev() {
                idx[i] += 1;
                if idx[i] < shape_vec[i] {
                    break;
                }
                idx[i] = 0;
            }
        }
        let dim = ferray_core::dimension::IxDyn::new(&shape_vec);
        let arr: ArrayD<T> = ArrayD::<T>::from_vec(dim, data).map_err(ferr_to_pyerr)?;
        arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// `numpy.meshgrid(*xi, indexing='xy')` — coordinate grids for evaluating
/// a function over a regular grid. All inputs are coerced to `float64`.
#[pyfunction]
#[pyo3(signature = (*xi, indexing = "xy"))]
pub fn meshgrid<'py>(
    py: Python<'py>,
    xi: &Bound<'py, pyo3::types::PyTuple>,
    indexing: &str,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    // Complex passthrough (#937): numpy.meshgrid is a pure coordinate-tiling op
    // with NO arithmetic — it preserves the input dtype, so a complex input
    // yields a complex128/complex64 grid (verified live numpy 2.4.4). The prior
    // path coerced every input to float64, DROPPING the imaginary part (R-CODE-4).
    // If ANY input is complex, build the grids over the common complex width.
    if meshgrid_any_complex(py, xi)? {
        return complex_meshgrid(py, xi, indexing);
    }
    let np = py.import("numpy")?;
    let mut owned: Vec<Array1<f64>> = Vec::with_capacity(xi.len());
    for item in xi.iter() {
        let coerced = np.call_method1("asarray", (&item, "float64"))?;
        let view: numpy::PyReadonlyArray1<f64> = coerced.extract()?;
        let arr: Array1<f64> = view.as_ferray().map_err(ferr_to_pyerr)?;
        owned.push(arr);
    }
    let _ = Ix1::new([0]); // suppress unused-import lint when binding only meshgrid
    let grids: Vec<ArrayD<f64>> =
        ferray_core::creation::meshgrid(&owned, indexing).map_err(ferr_to_pyerr)?;
    let py_arrays: Vec<Bound<'py, PyAny>> = grids
        .into_iter()
        .map(|g| {
            g.into_pyarray(py)
                .map_err(ferr_to_pyerr)
                .map(|p| p.into_any())
        })
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, py_arrays)?.into_any())
}

/// `true` if any `meshgrid` input has a complex dtype (numpy promotes a mixed
/// real/complex meshgrid to the common complex width — checked via `numpy.asarray`
/// + `numpy.iscomplexobj`).
fn meshgrid_any_complex<'py>(
    py: Python<'py>,
    xi: &Bound<'py, pyo3::types::PyTuple>,
) -> PyResult<bool> {
    let np = py.import("numpy")?;
    for item in xi.iter() {
        let a = np.call_method1("asarray", (&item,))?;
        if np.call_method1("iscomplexobj", (&a,))?.extract::<bool>()? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Complex `meshgrid`: the same coordinate-tiling as [`ferray_core::creation::meshgrid`]
/// but over `Complex<T>`, with a dtype-passthrough (no arithmetic). All inputs are
/// promoted to the common complex width (`complex64` only when EVERY complex input
/// is `complex64` and no `float64`/`complex128` widens it — numpy's promotion;
/// here the simplest matching rule is `complex128` unless all inputs are at most
/// `complex64`/`float32`/integer widths). To stay faithful to numpy's observable
/// width without re-deriving the full promotion table, the common dtype is taken
/// from `numpy.result_type` over the inputs (cast to a complex type if needed).
fn complex_meshgrid<'py>(
    py: Python<'py>,
    xi: &Bound<'py, pyo3::types::PyTuple>,
    indexing: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    // numpy's own result_type gives the exact promoted dtype (e.g. complex64 + complex64
    // -> complex64; complex64 + float64 -> complex128). Coerce every input to it.
    let arrs: Vec<Bound<'py, PyAny>> = xi
        .iter()
        .map(|it| np.call_method1("asarray", (&it,)))
        .collect::<PyResult<_>>()?;
    let args = pyo3::types::PyTuple::new(py, &arrs)?;
    let common = np.getattr("result_type")?.call1(args)?;
    let common_name: String = common.getattr("name")?.extract()?;
    // result_type over complex inputs is always a complex dtype here; pin to the
    // marshalled widths.
    let (cdt, is_c64) = match common_name.as_str() {
        "complex64" => ("complex64", true),
        _ => ("complex128", false),
    };
    if is_c64 {
        complex_meshgrid_width::<f32>(py, &arrs, indexing, cdt)
    } else {
        complex_meshgrid_width::<f64>(py, &arrs, indexing, cdt)
    }
}

/// Tile `meshgrid` coordinate grids over one complex width `T`, mirroring the
/// `ferray_core::creation::meshgrid` index math (no arithmetic — pure gather).
fn complex_meshgrid_width<'py, T>(
    py: Python<'py>,
    arrs: &[Bound<'py, PyAny>],
    indexing: &str,
    cdt: &str,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    num_complex::Complex<T>: ferray_core::Element + numpy::Element,
{
    use num_complex::Complex;
    if indexing != "xy" && indexing != "ij" {
        return Err(ferr_to_pyerr(ferray_core::FerrayError::invalid_value(
            "meshgrid: indexing must be 'xy' or 'ij'",
        )));
    }
    let np = py.import("numpy")?;
    // Extract each 1-D complex coordinate vector at the common width.
    let mut coords: Vec<Vec<Complex<T>>> = Vec::with_capacity(arrs.len());
    for a in arrs {
        let coerced = np.call_method1("asarray", (a, cdt))?;
        let flat = coerced.call_method0("ravel")?;
        let fa = complex_pyarray_to_ferray::<T>(&flat)?;
        coords.push(fa.iter().copied().collect());
    }
    let ndim = coords.len();
    if ndim == 0 {
        return Ok(pyo3::types::PyList::new(py, Vec::<Bound<'py, PyAny>>::new())?.into_any());
    }
    let mut shapes: Vec<usize> = coords.iter().map(|c| c.len()).collect();
    if indexing == "xy" && ndim >= 2 {
        shapes.swap(0, 1);
    }
    let total: usize = shapes.iter().product();
    let mut results: Vec<Bound<'py, PyAny>> = Vec::with_capacity(ndim);
    for (k, src) in coords.iter().enumerate() {
        let effective_k = if indexing == "xy" && ndim >= 2 {
            match k {
                0 => 1,
                1 => 0,
                other => other,
            }
        } else {
            k
        };
        let mut data: Vec<Complex<T>> = Vec::with_capacity(total);
        for flat in 0..total {
            let mut rem = flat;
            let mut idx_k = 0usize;
            for (d, &s) in shapes.iter().enumerate().rev() {
                if d == effective_k {
                    idx_k = rem % s;
                }
                rem /= s;
            }
            data.push(src[idx_k]);
        }
        let grid = ferray_core::array::aliases::ArrayD::<Complex<T>>::from_vec(
            ferray_core::dimension::IxDyn::new(&shapes),
            data,
        )
        .map_err(ferr_to_pyerr)?;
        results.push(complex_ferray_to_pyarray::<T>(py, grid)?);
    }
    Ok(pyo3::types::PyList::new(py, results)?.into_any())
}

/// `numpy.mgrid` access via call form: `mgrid([(start, stop, step), ...])`.
///
/// Python's `numpy.mgrid[0:5, 0:3]` slice syntax doesn't translate to
/// a function call cleanly, so we expose this as `ferray.mgrid([(0, 5, 1), (0, 3, 1)])`.
/// For the slice-syntax convenience, see the pure-Python `MGridIndexer`
/// helper that may be added in a follow-up.
#[pyfunction]
pub fn mgrid<'py>(py: Python<'py>, ranges: Vec<(f64, f64, f64)>) -> PyResult<Bound<'py, PyAny>> {
    let grids = ferray_core::creation::mgrid(&ranges).map_err(ferr_to_pyerr)?;
    let py_arrays: Vec<Bound<'py, PyAny>> = grids
        .into_iter()
        .map(|g| {
            g.into_pyarray(py)
                .map_err(ferr_to_pyerr)
                .map(|p| p.into_any())
        })
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, py_arrays)?.into_any())
}

/// `numpy.ogrid` — sparse open mesh.
#[pyfunction]
pub fn ogrid<'py>(py: Python<'py>, ranges: Vec<(f64, f64, f64)>) -> PyResult<Bound<'py, PyAny>> {
    let grids = ferray_core::creation::ogrid(&ranges).map_err(ferr_to_pyerr)?;
    let py_arrays: Vec<Bound<'py, PyAny>> = grids
        .into_iter()
        .map(|g| {
            g.into_pyarray(py)
                .map_err(ferr_to_pyerr)
                .map(|p| p.into_any())
        })
        .collect::<PyResult<_>>()?;
    Ok(pyo3::types::PyList::new(py, py_arrays)?.into_any())
}

/// `numpy.vander(x, N=None, increasing=False)` — Vandermonde matrix.
#[pyfunction]
#[pyo3(signature = (x, n = None, increasing = false))]
pub fn vander<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    n: Option<usize>,
    increasing: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_core::array::aliases::Array1;
    let arr = as_ndarray(py, x)?;
    let dt = dtype_name(&arr)?;
    // Complex Vandermonde: `ferray_core::creation::vander` only needs
    // `T: Element + Mul + Copy`, which `Complex<T>` satisfies, so the complex
    // powers `x^j` compute exactly as numpy (`np.vander` of complex, verified
    // live). The macro path can't bind a complex `T` (no `numpy::Element` arm),
    // so dispatch complex through the fft marshalling helpers (#920).
    match dt.as_str() {
        "complex128" | "c16" => return complex_vander_dispatch::<f64>(py, &arr, n, increasing),
        "complex64" | "c8" => return complex_vander_dispatch::<f32>(py, &arr, n, increasing),
        _ => {}
    }
    // vander needs Mul + Copy — covered by float and integer types but
    // not bool. Use match_dtype_numeric (no bool).
    Ok(crate::match_dtype_numeric!(dt.as_str(), T => {
        let view: numpy::PyReadonlyArray1<T> = arr.extract()?;
        let fa: Array1<T> = view.as_ferray().map_err(ferr_to_pyerr)?;
        let r: ArrayD<T> = ferray_core::creation::vander(&fa, n, increasing)
            .map_err(ferr_to_pyerr)?;
        r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any()
    }))
}

/// Complex Vandermonde: builds powers `x^j` of the complex input via
/// `ferray_core::creation::vander` (generic over `T: Element + Mul + Copy`,
/// which `Complex<T>` satisfies). The result matches numpy `np.vander` of a
/// complex array (verified live). Input is flattened to 1-D as numpy requires.
fn complex_vander_dispatch<'py, T>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    n: Option<usize>,
    increasing: bool,
) -> PyResult<Bound<'py, PyAny>>
where
    T: ferray_core::Element + Copy + numpy::Element + Default,
    Complex<T>: ferray_core::Element + numpy::Element + std::ops::Mul<Output = Complex<T>>,
{
    use ferray_core::array::aliases::Array1;
    use ferray_core::dimension::Ix1;
    let fa_dyn: ArrayD<Complex<T>> = complex_pyarray_to_ferray::<T>(arr)?;
    let flat: Vec<Complex<T>> = fa_dyn.iter().copied().collect();
    let len = flat.len();
    let x: Array1<Complex<T>> =
        Array1::<Complex<T>>::from_vec(Ix1::new([len]), flat).map_err(ferr_to_pyerr)?;
    let r: ArrayD<Complex<T>> =
        ferray_core::creation::vander(&x, n, increasing).map_err(ferr_to_pyerr)?;
    complex_ferray_to_pyarray(py, r)
}

/// `numpy.asarray(a, dtype=None)` — like `array` but does not copy if
/// the input is already an array of the requested dtype. ferray
/// currently always copies (see `ferray-numpy-interop` docs).
#[pyfunction]
#[pyo3(signature = (obj, dtype = None))]
pub fn asarray<'py>(
    py: Python<'py>,
    obj: &Bound<'py, PyAny>,
    dtype: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    array(py, obj, dtype)
}
