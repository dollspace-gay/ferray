//! Bindings for numpy's top-level datetime64 / timedelta64 surface:
//! `datetime_as_string`, `is_busday`, `busday_count`, `busday_offset`, plus
//! the datetime64-aware `add` / `subtract` arithmetic and `isnat`.
//!
//! ## datetime64 marshalling
//!
//! numpy's datetime64 / timedelta64 arrays are i64-backed (a count of ticks
//! of the dtype's `TimeUnit`). pyo3-numpy has no native datetime64 element
//! type, so a datetime64 ndarray is bridged to Rust through numpy's own
//! zero-copy `.view('int64')` ([`as_int64_view`]); the typed kernels in
//! `ferray-ufunc::ops::datetime` run on the i64 ticks, and the result is
//! reconstructed via `int64_arr.view('datetime64[<unit>]')`
//! ([`int64_to_datetime64`] / [`int64_to_timedelta64`]). The dtype's unit is
//! read with `numpy.datetime_data` ([`datetime64_unit`]). This routes the
//! genuine ferray-ufunc datetime arithmetic (NaT propagation, finer-unit
//! promotion) rather than delegating to numpy's operators.
//!
//! ## busday calendar (implemented in Rust)
//!
//! `is_busday` / `busday_count` / `busday_offset` are pure weekday-counting
//! algorithms over the int64 *day* count since the 1970-01-01 epoch. The
//! day-of-week convention follows numpy
//! (numpy/_core/src/multiarray/datetime_busday.c:32 `get_day_of_week`:
//! "1970-01-05 is Monday", `day_of_week = (date - 4) % 7`). `weekmask` is a
//! 7-char Mon..Sun string (or 7-element int sequence); `holidays` are excluded
//! days. The roll modes (`forward`/`following`/`backward`/`preceding`/`raise`/
//! `nat`) mirror `apply_business_day_roll`
//! (numpy/_core/src/multiarray/datetime_busday.c:159).

use ferray_core::array::aliases::ArrayD;
use ferray_core::dimension::IxDyn;
use ferray_core::dtype::{DateTime64, NAT, TimeUnit, Timedelta64};
use ferray_numpy_interop::AsFerray;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::conv::{
    as_int64_view, datetime64_unit, ferr_to_pyerr, int64_to_datetime64, int64_to_timedelta64,
};

// ---------------------------------------------------------------------------
// dtype-kind sniffing
// ---------------------------------------------------------------------------

/// What kind of time array an object's dtype is, if any.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimeKind {
    Datetime,
    Timedelta,
}

/// Sniff whether `arr` (already a numpy ndarray) is a datetime64 or
/// timedelta64 array. Returns `None` for any other dtype.
fn time_kind(arr: &Bound<'_, PyAny>) -> PyResult<Option<TimeKind>> {
    let kind: String = arr.getattr("dtype")?.getattr("kind")?.extract()?;
    Ok(match kind.as_str() {
        "M" => Some(TimeKind::Datetime),
        "m" => Some(TimeKind::Timedelta),
        _ => None,
    })
}

/// Parse a datetime64 unit suffix string (e.g. `"D"`, `"s"`, `"ns"`) into a
/// [`TimeUnit`]. A suffix ferray-core's `TimeUnit` does not model (`Y`/`M`
/// calendar units, `W`, `ps`/`fs`/`as`) surfaces a `TypeError` rather than a
/// silent mis-scale.
fn parse_unit(suffix: &str) -> PyResult<TimeUnit> {
    TimeUnit::from_descr_suffix(suffix).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "ferray datetime arithmetic does not yet support the time unit {suffix:?} \
             (supported: D, h, m, s, ms, us, ns)"
        ))
    })
}

/// Extract a datetime64 / timedelta64 ndarray into `(ArrayD<i64> ticks,
/// TimeUnit, unit_str)` via the int64-view bridge.
fn extract_ticks(
    py: Python<'_>,
    arr: &Bound<'_, PyAny>,
) -> PyResult<(ArrayD<i64>, TimeUnit, String)> {
    let (unit_str, _count) = datetime64_unit(py, arr)?;
    let unit = parse_unit(&unit_str)?;
    let i64_view = as_int64_view(py, arr)?;
    let view: PyReadonlyArrayDyn<i64> = i64_view.extract()?;
    let ticks: ArrayD<i64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    Ok((ticks, unit, unit_str))
}

/// Map an `ArrayD<i64>` into an `ArrayD<DateTime64>` (same shape).
fn ticks_to_datetime(ticks: &ArrayD<i64>) -> PyResult<ArrayD<DateTime64>> {
    let data: Vec<DateTime64> = ticks.iter().map(|&t| DateTime64(t)).collect();
    ArrayD::from_vec(IxDyn::new(ticks.shape()), data).map_err(ferr_to_pyerr)
}

/// Map an `ArrayD<i64>` into an `ArrayD<Timedelta64>` (same shape).
fn ticks_to_timedelta(ticks: &ArrayD<i64>) -> PyResult<ArrayD<Timedelta64>> {
    let data: Vec<Timedelta64> = ticks.iter().map(|&t| Timedelta64(t)).collect();
    ArrayD::from_vec(IxDyn::new(ticks.shape()), data).map_err(ferr_to_pyerr)
}

/// Lower a datetime64-result array (shape from `like`) back to numpy via the
/// int64-view bridge.
fn datetime_result<'py>(
    py: Python<'py>,
    out: &ArrayD<DateTime64>,
    unit_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let ticks: Vec<i64> = out.iter().map(|v| v.0).collect();
    let arr = ArrayD::from_vec(IxDyn::new(out.shape()), ticks).map_err(ferr_to_pyerr)?;
    int64_to_datetime64(py, arr, unit_str)
}

fn timedelta_result<'py>(
    py: Python<'py>,
    out: &ArrayD<Timedelta64>,
    unit_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let ticks: Vec<i64> = out.iter().map(|v| v.0).collect();
    let arr = ArrayD::from_vec(IxDyn::new(out.shape()), ticks).map_err(ferr_to_pyerr)?;
    int64_to_timedelta64(py, arr, unit_str)
}

// ---------------------------------------------------------------------------
// datetime-aware arithmetic dispatch (used by ufunc::add / ufunc::subtract)
// ---------------------------------------------------------------------------

/// Returns `true` if a numpy dtype name string denotes a datetime64 /
/// timedelta64 dtype (i.e. starts with `"datetime64"` / `"timedelta64"`).
///
/// Used by the creation/coercion bindings (`array`/`asarray`/`zeros`/`ones`/
/// `empty`/`full` and the `*_like` siblings) to branch on the time dtypes
/// *ahead* of the real-only `match_dtype_all!` / `creation_dispatch!` macros,
/// which would otherwise hit their `TypeError` fallthrough (R-1 input
/// coercion). The name is read with `numpy.dtype(...).name`, e.g.
/// `"datetime64[D]"` / `"timedelta64[ns]"`.
pub fn is_time_dtype_name(name: &str) -> bool {
    name.starts_with("datetime64") || name.starts_with("timedelta64")
}

/// Round-trip an already-constructed numpy datetime64 / timedelta64 ndarray
/// through the ferray int64-view transport, returning a fresh numpy
/// datetime64 / timedelta64 ndarray of the SAME unit + shape + values.
///
/// This is the R-1 input-coercion seam: `array`/`asarray`/`zeros`/`ones`/
/// `empty`/`full` (and the `*_like` siblings) delegate the parse / fill /
/// shape construction to numpy (which owns the string→datetime64 parse and
/// the zeros/ones/full fill semantics), then push the resulting buffer across
/// the ferray boundary via the zero-copy `.view('int64')` bridge
/// ([`as_int64_view`]) and reconstruct the typed array with
/// [`int64_to_datetime64`] / [`int64_to_timedelta64`]. This produces a fresh
/// ferray-marshalled buffer while preserving numpy's dtype + unit + shape with
/// no lossy cast (R-CODE-4); NaT (`i64::MIN`) is carried through unchanged as a
/// plain int64 tick. The unit string is read directly from the dtype
/// ([`datetime64_unit`]), so calendar units (`Y`/`M`/`W`) and the full
/// sub-second range round-trip even though ferray-core's arithmetic `TimeUnit`
/// only models `D..ns` — this path performs no arithmetic, only marshalling.
///
/// `arr` must already be a numpy datetime64 / timedelta64 ndarray (the caller
/// constructs it via `np.asarray(obj, dtype)` / `np.zeros(shape, dtype)` /
/// etc.); a non-time dtype is rejected so the caller only invokes this for the
/// `"M"`/`"m"` arms.
pub fn datetime_roundtrip<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kind = time_kind(arr)?.ok_or_else(|| {
        PyTypeError::new_err("datetime_roundtrip requires a datetime64 / timedelta64 array")
    })?;
    // Read the unit string straight from the dtype (no TimeUnit parse, so
    // Y/M/W and the full sub-second range pass through — this is pure
    // marshalling, not arithmetic).
    let (unit_str, _count) = datetime64_unit(py, arr)?;
    // Zero-copy reinterpret as int64, then materialise an owned ferray ArrayD
    // (the round-trip across the boundary that this binding exists to perform).
    let i64_view = as_int64_view(py, arr)?;
    let view: PyReadonlyArrayDyn<i64> = i64_view.extract()?;
    let ticks: ArrayD<i64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    match kind {
        TimeKind::Datetime => int64_to_datetime64(py, ticks, &unit_str),
        TimeKind::Timedelta => int64_to_timedelta64(py, ticks, &unit_str),
    }
}

/// Returns `true` if either operand is a datetime64 / timedelta64 array, so
/// the ufunc `add` / `subtract` bindings can route to [`add_time`] /
/// [`subtract_time`] instead of the numeric path.
pub fn is_time_op(py: Python<'_>, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    Ok(time_kind(&aa)?.is_some() || time_kind(&ba)?.is_some())
}

/// `add` over datetime/timedelta operands, routed through ferray-ufunc:
///   datetime + timedelta -> datetime   (and the symmetric timedelta + datetime)
///   timedelta + timedelta -> timedelta
/// using the finer-unit promoted kernels. The two operands are broadcast to a
/// common shape first (numpy ufunc broadcasting contract).
pub fn add_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    // Broadcast the two arrays to a common shape (numpy ufunc contract).
    let pair = np.call_method1("broadcast_arrays", (&aa, &ba))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;

    match (ka, kb) {
        (Some(TimeKind::Datetime), Some(TimeKind::Timedelta)) => time_add_dt_td(py, &pl[0], &pl[1]),
        (Some(TimeKind::Timedelta), Some(TimeKind::Datetime)) => {
            // commutative: datetime + timedelta
            time_add_dt_td(py, &pl[1], &pl[0])
        }
        (Some(TimeKind::Timedelta), Some(TimeKind::Timedelta)) => {
            let (ta, ua, _) = extract_ticks(py, &pl[0])?;
            let (tb, ub, _) = extract_ticks(py, &pl[1])?;
            let fa = ticks_to_timedelta(&ta)?;
            let fb = ticks_to_timedelta(&tb)?;
            let (r, unit) =
                ferray_ufunc::add_timedelta_promoted(&fa, ua, &fb, ub).map_err(ferr_to_pyerr)?;
            timedelta_result(py, &r, unit.descr_suffix())
        }
        // datetime + datetime (and any other time-operand pair numpy does not
        // define) -> numpy RAISES its exact `UFuncTypeError`
        // ("ufunc 'add' cannot use operands with types ...", generate_umath.py
        // 'add' registers only Mm/mm/mM, NOT MM). Delegate to numpy on the
        // ORIGINAL operands so its precise exception family surfaces (R-DEV-2)
        // instead of a ferray-side `TypeError`.
        _ => time_numeric_raise(py, "add", a, b),
    }
}

fn time_add_dt_td<'py>(
    py: Python<'py>,
    dt: &Bound<'py, PyAny>,
    td: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let (tdt, udt, _) = extract_ticks(py, dt)?;
    let (ttd, utd, _) = extract_ticks(py, td)?;
    let fa = ticks_to_datetime(&tdt)?;
    let fb = ticks_to_timedelta(&ttd)?;
    let (r, unit) =
        ferray_ufunc::add_datetime_timedelta_promoted(&fa, udt, &fb, utd).map_err(ferr_to_pyerr)?;
    datetime_result(py, &r, unit.descr_suffix())
}

/// `subtract` over datetime/timedelta operands, routed through ferray-ufunc:
///   datetime - datetime  -> timedelta
///   datetime - timedelta -> datetime
///   timedelta - timedelta -> timedelta
pub fn subtract_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    let pair = np.call_method1("broadcast_arrays", (&aa, &ba))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;

    match (ka, kb) {
        (Some(TimeKind::Datetime), Some(TimeKind::Datetime)) => {
            let (ta, ua, _) = extract_ticks(py, &pl[0])?;
            let (tb, ub, _) = extract_ticks(py, &pl[1])?;
            let fa = ticks_to_datetime(&ta)?;
            let fb = ticks_to_datetime(&tb)?;
            let (r, unit) =
                ferray_ufunc::sub_datetime_promoted(&fa, ua, &fb, ub).map_err(ferr_to_pyerr)?;
            timedelta_result(py, &r, unit.descr_suffix())
        }
        (Some(TimeKind::Datetime), Some(TimeKind::Timedelta)) => {
            let (ta, ua, _) = extract_ticks(py, &pl[0])?;
            let (tb, ub, _) = extract_ticks(py, &pl[1])?;
            // datetime - timedelta == datetime + (-timedelta), via the add kernel
            // on the negated timedelta to reuse the finer-unit promotion.
            let neg: Vec<Timedelta64> = tb
                .iter()
                .map(|&t| {
                    if t == NAT {
                        Timedelta64(NAT)
                    } else {
                        Timedelta64(t.wrapping_neg())
                    }
                })
                .collect();
            let fb = ArrayD::from_vec(IxDyn::new(tb.shape()), neg).map_err(ferr_to_pyerr)?;
            let fa = ticks_to_datetime(&ta)?;
            let (r, unit) = ferray_ufunc::add_datetime_timedelta_promoted(&fa, ua, &fb, ub)
                .map_err(ferr_to_pyerr)?;
            datetime_result(py, &r, unit.descr_suffix())
        }
        (Some(TimeKind::Timedelta), Some(TimeKind::Timedelta)) => {
            let (ta, ua, _) = extract_ticks(py, &pl[0])?;
            let (tb, ub, _) = extract_ticks(py, &pl[1])?;
            let neg: Vec<Timedelta64> = tb
                .iter()
                .map(|&t| {
                    if t == NAT {
                        Timedelta64(NAT)
                    } else {
                        Timedelta64(t.wrapping_neg())
                    }
                })
                .collect();
            let fb = ArrayD::from_vec(IxDyn::new(tb.shape()), neg).map_err(ferr_to_pyerr)?;
            let fa = ticks_to_timedelta(&ta)?;
            let (r, unit) =
                ferray_ufunc::add_timedelta_promoted(&fa, ua, &fb, ub).map_err(ferr_to_pyerr)?;
            timedelta_result(py, &r, unit.descr_suffix())
        }
        // Any time/non-time pair numpy does not define for subtract (e.g.
        // datetime - int, timedelta - datetime) -> numpy raises its exact
        // `UFuncTypeError`; delegate so it surfaces (R-DEV-2).
        _ => time_numeric_raise(py, "subtract", a, b),
    }
}

// ---------------------------------------------------------------------------
// timedelta numeric arithmetic dispatch (REQ-2, #942)
//
// numpy registers (generate_umath.py:386-1046, verified live numpy 2.4.5):
//   multiply:     td*int -> td, int*td -> td, td*float -> td, float*td -> td
//                 (td*td and dt*anything RAISE UFuncTypeError).
//   divide:       td/int -> td, td/float -> td, td/td -> float64
//                 (int/td and dt/x RAISE).
//   floor_divide: td//int -> td (= divide loop, trunc), td//float -> td (trunc),
//                 td//td -> int64 (true FLOOR). (`TIMEDELTA_floor_divide` is a
//                 `#define` to `TIMEDELTA_divide`, loops.c.src:428 — so the
//                 SCALAR floor-divide cases truncate, only td//td floors.)
//   remainder:    td%td -> td (Python floor-mod). (td%int and dt%x RAISE — only
//                 the `mm` loop is registered.)
// Every numpy-RAISE pair is surfaced by delegating to numpy on the ORIGINAL
// operands, so numpy's exact `UFuncTypeError` text appears (R-DEV-2). The
// computable pairs run the ferray-ufunc kernels over the int64-view transport.
// ---------------------------------------------------------------------------

/// A timedelta operand broadcast against a numeric scalar: its int64 ticks,
/// arithmetic [`TimeUnit`], the numpy unit-suffix string, and the broadcast
/// numeric multiplier/divisor.
type TdScalarBroadcast = (ArrayD<i64>, TimeUnit, String, ScalarKind);

/// Two timedelta operands broadcast to a common shape: `(a_ticks, a_unit,
/// b_ticks, b_unit, a_unit_str)`.
type TdPairBroadcast = (ArrayD<i64>, TimeUnit, ArrayD<i64>, TimeUnit, String);

/// Numeric (non-time) operand kind extracted from a multiply/divide operand.
enum ScalarKind {
    /// Integer-dtype numeric operand, broadcast to the timedelta's shape.
    Int(Vec<i64>),
    /// Float-dtype numeric operand, broadcast to the timedelta's shape.
    Float(Vec<f64>),
}

/// Raise numpy's EXACT `UFuncTypeError`/`TypeError` for a (op, dtype) pair
/// numpy does not define, by delegating the op to numpy on the original
/// operands (R-DEV-2). Used for `dt * x`, `dt / x`, `td * td`, `int / td`,
/// `td % int`, etc. If numpy unexpectedly does NOT raise, its (computed)
/// result is returned — but every call site is a pair verified live to raise.
fn time_numeric_raise<'py>(
    py: Python<'py>,
    np_func: &str,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    np.call_method1(np_func, (a, b))
}

/// Broadcast a timedelta operand against a numeric (int/float) operand and
/// return `(td_ticks, td_unit, td_unit_str, scalar)` where `scalar` carries the
/// numeric operand broadcast to the same flat (row-major) order as `td_ticks`.
/// The numeric operand's dtype-kind (`"i"`/`"u"` vs `"f"`) selects
/// [`ScalarKind`]; a complex / object / time numeric operand returns `None`
/// (the caller delegates to numpy to raise).
fn broadcast_td_scalar(
    py: Python<'_>,
    td: &Bound<'_, PyAny>,
    num: &Bound<'_, PyAny>,
) -> PyResult<Option<TdScalarBroadcast>> {
    let np = py.import("numpy")?;
    let num_arr = np.call_method1("asarray", (num,))?;
    let num_kind: String = num_arr.getattr("dtype")?.getattr("kind")?.extract()?;
    let td_arr = np.call_method1("asarray", (td,))?;
    // Broadcast the two to a common shape (numpy ufunc contract).
    let pair = np.call_method1("broadcast_arrays", (&td_arr, &num_arr))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;
    let (ticks, unit, unit_str) = extract_ticks(py, &pl[0])?;
    let scalar = match num_kind.as_str() {
        // integer / unsigned / bool numeric operands -> exact int64 path.
        "i" | "u" | "b" => {
            let as_i64 = pl[1].call_method1("astype", ("int64",))?;
            let flat = as_i64.call_method1("ravel", ())?;
            let v: Vec<i64> = flat.extract()?;
            ScalarKind::Int(v)
        }
        "f" => {
            let as_f64 = pl[1].call_method1("astype", ("float64",))?;
            let flat = as_f64.call_method1("ravel", ())?;
            let v: Vec<f64> = flat.extract()?;
            ScalarKind::Float(v)
        }
        // complex / datetime / timedelta / object -> not a valid scalar
        // multiplier; the caller delegates to numpy so it raises.
        _ => return Ok(None),
    };
    Ok(Some((ticks, unit, unit_str, scalar)))
}

/// `td * scalar` / `scalar * td` element-wise, mirroring numpy's `mq`/`qm`/
/// `md`/`dm` multiply loops via [`ferray_ufunc::ops::datetime::mul_timedelta_scalar_i64`]
/// / `mul_timedelta_scalar_f64`. The timedelta keeps its own unit.
fn mul_td_scalar<'py>(
    py: Python<'py>,
    ticks: &ArrayD<i64>,
    unit_str: &str,
    scalar: &ScalarKind,
) -> PyResult<Bound<'py, PyAny>> {
    let flat: Vec<i64> = ticks.iter().copied().collect();
    let out: Vec<i64> = match scalar {
        ScalarKind::Int(ks) => flat
            .iter()
            .zip(ks.iter())
            .map(|(&t, &k)| if t == NAT { NAT } else { t.wrapping_mul(k) })
            .collect(),
        ScalarKind::Float(ks) => flat
            .iter()
            .zip(ks.iter())
            .map(|(&t, &k)| {
                if t == NAT {
                    NAT
                } else {
                    let r = t as f64 * k;
                    if r.is_finite() { r as i64 } else { NAT }
                }
            })
            .collect(),
    };
    let arr = ArrayD::from_vec(IxDyn::new(ticks.shape()), out).map_err(ferr_to_pyerr)?;
    int64_to_timedelta64(py, arr, unit_str)
}

/// `td / scalar` (and the scalar-cased `td // scalar`, which numpy routes
/// through the SAME divide loop) — `mq`/`md` divide semantics: integer
/// division / double-quotient truncation toward zero, zero divisor -> NaT.
fn div_td_scalar<'py>(
    py: Python<'py>,
    ticks: &ArrayD<i64>,
    unit_str: &str,
    scalar: &ScalarKind,
) -> PyResult<Bound<'py, PyAny>> {
    let flat: Vec<i64> = ticks.iter().copied().collect();
    let out: Vec<i64> = match scalar {
        ScalarKind::Int(ks) => flat
            .iter()
            .zip(ks.iter())
            .map(|(&t, &k)| {
                if t == NAT || k == 0 {
                    NAT
                } else {
                    t.wrapping_div(k)
                }
            })
            .collect(),
        ScalarKind::Float(ks) => flat
            .iter()
            .zip(ks.iter())
            .map(|(&t, &k)| {
                if t == NAT {
                    NAT
                } else {
                    let r = t as f64 / k;
                    if r.is_finite() { r as i64 } else { NAT }
                }
            })
            .collect(),
    };
    let arr = ArrayD::from_vec(IxDyn::new(ticks.shape()), out).map_err(ferr_to_pyerr)?;
    int64_to_timedelta64(py, arr, unit_str)
}

/// Broadcast two timedelta operands to a common shape and return their int64
/// tick buffers + units (for the `td op td` kernels).
fn broadcast_td_pair<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<TdPairBroadcast> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let pair = np.call_method1("broadcast_arrays", (&aa, &ba))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;
    let (ta, ua, ua_str) = extract_ticks(py, &pl[0])?;
    let (tb, ub, _ub_str) = extract_ticks(py, &pl[1])?;
    Ok((ta, ua, tb, ub, ua_str))
}

/// `numpy.multiply` over a timedelta operand (REQ-2). `td * int/float` and the
/// reflected `int/float * td` -> timedelta. `td * td` and `datetime *
/// anything` RAISE numpy's exact `UFuncTypeError`.
pub fn multiply_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    match (ka, kb) {
        // timedelta * numeric-scalar  /  numeric-scalar * timedelta.
        (Some(TimeKind::Timedelta), None) => {
            match broadcast_td_scalar(py, &aa, &ba)? {
                Some((ticks, _u, unit_str, scalar)) => {
                    mul_td_scalar(py, &ticks, &unit_str, &scalar)
                }
                // non-numeric (complex/object) operand -> numpy raises.
                None => time_numeric_raise(py, "multiply", a, b),
            }
        }
        (None, Some(TimeKind::Timedelta)) => match broadcast_td_scalar(py, &ba, &aa)? {
            Some((ticks, _u, unit_str, scalar)) => mul_td_scalar(py, &ticks, &unit_str, &scalar),
            None => time_numeric_raise(py, "multiply", a, b),
        },
        // td*td, dt*anything, anything*dt -> numpy raises UFuncTypeError.
        _ => time_numeric_raise(py, "multiply", a, b),
    }
}

/// `numpy.divide` / `true_divide` over a timedelta operand (REQ-2). `td /
/// int/float` -> timedelta (trunc toward zero); `td / td` -> float64 ratio.
/// `int / td`, `datetime / x` RAISE numpy's exact `UFuncTypeError`.
pub fn divide_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    match (ka, kb) {
        // td / td -> float64 ratio (finer common unit).
        (Some(TimeKind::Timedelta), Some(TimeKind::Timedelta)) => {
            let (ta, ua, tb, ub, _) = broadcast_td_pair(py, &aa, &ba)?;
            let fa = ticks_to_timedelta(&ta)?;
            let fb = ticks_to_timedelta(&tb)?;
            let r = ferray_ufunc::ops::datetime::truediv_timedelta(&fa, ua, &fb, ub)
                .map_err(ferr_to_pyerr)?;
            use ferray_numpy_interop::IntoNumPy;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        // td / numeric-scalar -> timedelta (the numeric operand is the divisor;
        // the reflected numeric/td is undefined and raises).
        (Some(TimeKind::Timedelta), None) => match broadcast_td_scalar(py, &aa, &ba)? {
            Some((ticks, _u, unit_str, scalar)) => div_td_scalar(py, &ticks, &unit_str, &scalar),
            None => time_numeric_raise(py, "divide", a, b),
        },
        // numeric / td, dt / x, td_complex etc. -> numpy raises.
        _ => time_numeric_raise(py, "divide", a, b),
    }
}

/// `numpy.floor_divide` over a timedelta operand (REQ-2). `td // int/float` ->
/// timedelta (numpy `#define`s floor_divide to the divide loop for these, so it
/// TRUNCATES toward zero, loops.c.src:428); `td // td` -> int64 (true FLOOR).
/// `int // td`, `datetime // x` RAISE.
pub fn floordiv_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    match (ka, kb) {
        // td // td -> int64 floor.
        (Some(TimeKind::Timedelta), Some(TimeKind::Timedelta)) => {
            let (ta, ua, tb, ub, _) = broadcast_td_pair(py, &aa, &ba)?;
            let fa = ticks_to_timedelta(&ta)?;
            let fb = ticks_to_timedelta(&tb)?;
            let r = ferray_ufunc::ops::datetime::floordiv_timedelta(&fa, ua, &fb, ub)
                .map_err(ferr_to_pyerr)?;
            use ferray_numpy_interop::IntoNumPy;
            Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        }
        // td // numeric-scalar -> timedelta (divide loop = trunc toward zero).
        (Some(TimeKind::Timedelta), None) => match broadcast_td_scalar(py, &aa, &ba)? {
            Some((ticks, _u, unit_str, scalar)) => div_td_scalar(py, &ticks, &unit_str, &scalar),
            None => time_numeric_raise(py, "floor_divide", a, b),
        },
        _ => time_numeric_raise(py, "floor_divide", a, b),
    }
}

/// `numpy.remainder` / `mod` over a timedelta operand (REQ-2). Only `td % td`
/// is defined -> timedelta (Python floor-mod, sign follows divisor); `td %
/// int`, `datetime % x` RAISE numpy's exact `UFuncTypeError`.
pub fn mod_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let ka = time_kind(&aa)?;
    let kb = time_kind(&ba)?;
    match (ka, kb) {
        (Some(TimeKind::Timedelta), Some(TimeKind::Timedelta)) => {
            let (ta, ua, tb, ub, _) = broadcast_td_pair(py, &aa, &ba)?;
            let fa = ticks_to_timedelta(&ta)?;
            let fb = ticks_to_timedelta(&tb)?;
            let (r, unit) = ferray_ufunc::ops::datetime::mod_timedelta(&fa, ua, &fb, ub)
                .map_err(ferr_to_pyerr)?;
            timedelta_result(py, &r, unit.descr_suffix())
        }
        // td % int, int % td, dt % x -> numpy raises (only the mm loop exists).
        _ => time_numeric_raise(py, "remainder", a, b),
    }
}

// ---------------------------------------------------------------------------
// datetime64 / timedelta64 reductions (REQ-4, #944)
//
// Prior to this build the `stats.rs` reduction dispatch was real-only and a
// datetime/timedelta input took one of two divergent paths, both verified live
// (numpy 2.4.5):
//
//   * `mean`/`std`/`var` coerced the input to `float64` ahead of the float
//     dispatch, SILENTLY dropping the datetime/timedelta contract and returning
//     a bare float — `fr.mean(datetime64) -> array(18268.5)`, `fr.std(td) ->
//     0.5`. numpy RAISES for both (`mean(dt)`/`std(td)` are undefined). This is
//     the R-CODE-4 boundary corruption REQ-4 eliminates.
//   * `sum`/`min`/`max`/`ptp`/`argmin`/`argmax`/`cumsum` hit the numeric/
//     orderable macros with no `M`/`m` arm and raised a ferray-side `TypeError`
//     where numpy COMPUTES (e.g. `sum(td)`, `min(dt)`, `ptp(dt)`).
//
// The compute-vs-raise contract, derived LIVE from numpy 2.4.5 (R-CHAR-3):
//
//   timedelta64 (m):  sum -> timedelta, mean -> timedelta (int64 trunc-toward-
//                     zero of sum/n), min/max -> timedelta, ptp -> timedelta,
//                     cumsum -> timedelta, argmin/argmax -> int64;
//                     std/var -> RAISE TypeError ("ufunc 'square' not
//                     supported"); prod -> RAISE UFuncTypeError ('multiply').
//   datetime64  (M):  min/max -> datetime, ptp -> timedelta (max-min),
//                     argmin/argmax -> int64;
//                     sum/mean/std/var/cumsum -> RAISE UFuncTypeError
//                     ('add'/'multiply' cannot use two datetimes); prod -> RAISE.
//
// All compute paths run over the int64 ticks (the existing `.view('int64')`
// transport, #831), NaT (`i64::MIN`) propagating: any NaT operand makes
// sum/mean/min/max/ptp/cumsum produce NaT (verified live), and makes
// argmin/argmax return the FIRST NaT index (numpy treats NaT as both the min
// AND the max winner — `argmin`/`argmax` of a NaT-containing array both return
// the first NaT index, live). The RAISE cases delegate the reduction to numpy
// on the original array so numpy's EXACT exception type + message surface
// (R-DEV-2), instead of fabricating a value.
// ---------------------------------------------------------------------------

/// Which reduction a `time_reduce` call performs. Mirrors the `stats.rs`
/// `#[pyfunction]` surface that carries an `M`/`m` arm.
#[derive(Clone, Copy)]
pub enum TimeReduce {
    Sum,
    Mean,
    Min,
    Max,
    Ptp,
    Cumsum,
    Argmin,
    Argmax,
    Std,
    Var,
    Prod,
}

impl TimeReduce {
    /// The numpy top-level function name this reduction delegates to when the
    /// (kind, op) pair is one numpy RAISES on — so numpy raises its own exact
    /// `UFuncTypeError`/`TypeError` (R-DEV-2).
    fn numpy_name(self) -> &'static str {
        match self {
            TimeReduce::Sum => "sum",
            TimeReduce::Mean => "mean",
            TimeReduce::Min => "min",
            TimeReduce::Max => "max",
            TimeReduce::Ptp => "ptp",
            TimeReduce::Cumsum => "cumsum",
            TimeReduce::Argmin => "argmin",
            TimeReduce::Argmax => "argmax",
            TimeReduce::Std => "std",
            TimeReduce::Var => "var",
            TimeReduce::Prod => "prod",
        }
    }
}

/// Returns `true` if `arr` is a datetime64 / timedelta64 ndarray, so the
/// `stats.rs` reduction `#[pyfunction]`s can branch to [`time_reduce`] ahead of
/// their real-only dispatch macros.
pub fn is_time_array(arr: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(time_kind(arr)?.is_some())
}

/// Delegate a reduction to numpy on the original array so numpy raises its own
/// exact exception (`UFuncTypeError`/`TypeError`) for an op it does not define
/// over the dtype. If numpy unexpectedly does NOT raise, the (computed) result
/// is returned — but every `delegate_to_numpy` call site below is a (kind, op)
/// pair verified live to raise, so this is the exact-exception path (R-DEV-2).
fn delegate_to_numpy<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    kind: TimeReduce,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(ax) = axis {
        kwargs.set_item("axis", ax)?;
    }
    np.call_method(kind.numpy_name(), (arr,), Some(&kwargs))
}

/// Reduce the flat int64 tick buffer of one lane, NaT (`i64::MIN`)
/// propagating, into a single int64 tick (NaT-result for the propagating
/// reductions). `which` selects sum / min / max. Empty min/max return `None`
/// (the caller raises ValueError, matching numpy's empty-min/max). Empty sum
/// returns `Some(0)` (numpy: `sum([],timedelta)==0`).
fn lane_reduce_i64(vals: &[i64], which: TimeReduce) -> Option<i64> {
    let any_nat = vals.contains(&NAT);
    match which {
        TimeReduce::Sum => {
            if any_nat {
                return Some(NAT);
            }
            // Wrapping add mirrors numpy's int64 tick accumulation (overflow
            // wraps, as numpy's datetime64 arithmetic does).
            Some(vals.iter().fold(0i64, |acc, &v| acc.wrapping_add(v)))
        }
        TimeReduce::Mean => {
            if any_nat {
                return Some(NAT);
            }
            let n = vals.len();
            if n == 0 {
                // numpy mean of an empty timedelta raises (handled by caller).
                return None;
            }
            let s = vals.iter().fold(0i64, |acc, &v| acc.wrapping_add(v));
            // numpy timedelta mean truncates toward zero (`-7/3 -> -2`,
            // `5/3 -> 1`), which is exactly Rust's integer `/` (verified live).
            Some(s / n as i64)
        }
        TimeReduce::Min => {
            if vals.is_empty() {
                return None;
            }
            if any_nat {
                return Some(NAT);
            }
            Some(*vals.iter().min().unwrap_or(&NAT))
        }
        TimeReduce::Max => {
            if vals.is_empty() {
                return None;
            }
            if any_nat {
                return Some(NAT);
            }
            Some(*vals.iter().max().unwrap_or(&NAT))
        }
        TimeReduce::Ptp => {
            if vals.is_empty() {
                return None;
            }
            if any_nat {
                return Some(NAT);
            }
            let lo = *vals.iter().min().unwrap_or(&NAT);
            let hi = *vals.iter().max().unwrap_or(&NAT);
            Some(hi.wrapping_sub(lo))
        }
        _ => None,
    }
}

/// Index of the first NaT (`i64::MIN`) in a lane, if any — numpy's
/// argmin/argmax both return the first NaT index when a NaT is present (live).
fn first_nat(vals: &[i64]) -> Option<usize> {
    vals.iter().position(|&v| v == NAT)
}

/// argmin / argmax of a non-empty int64 lane, NaT first-occurrence winning
/// (numpy: a NaT-containing lane returns the FIRST NaT index for BOTH argmin
/// and argmax, live). With no NaT, first-occurrence min/max index.
fn lane_arg(vals: &[i64], want_max: bool) -> usize {
    if let Some(i) = first_nat(vals) {
        return i;
    }
    let mut best = 0usize;
    for (i, &v) in vals.iter().enumerate().skip(1) {
        let better = if want_max {
            v > vals[best]
        } else {
            v < vals[best]
        };
        if better {
            best = i;
        }
    }
    best
}

/// Row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let ndim = shape.len();
    let mut strides = vec![1usize; ndim];
    for d in (0..ndim.saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

/// Walk every output coordinate of an axis reduction over a flat row-major i64
/// buffer, returning `(out_shape, per_lane_results)`.
fn axis_lanes_i64<R>(
    flat: &[i64],
    shape: &[usize],
    ax: usize,
    mut f: impl FnMut(&[i64]) -> R,
) -> PyResult<(Vec<usize>, Vec<R>)> {
    let ndim = shape.len();
    if ax >= ndim {
        return Err(PyValueError::new_err(
            "axis out of bounds for time reduction",
        ));
    }
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter_map(|(d, &s)| if d == ax { None } else { Some(s) })
        .collect();
    let axis_len = shape[ax];
    let strides = row_major_strides(shape);
    let out_dims: Vec<usize> = (0..ndim).filter(|&d| d != ax).collect();
    let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
    let out_n: usize = out_extents.iter().product::<usize>().max(1);
    let mut results: Vec<R> = Vec::with_capacity(out_n);
    for lin in 0..out_n {
        let mut rem = lin;
        let mut base = 0usize;
        for (i, &d) in out_dims.iter().enumerate() {
            let ext = out_extents[i];
            let c = rem % ext;
            rem /= ext;
            base += c * strides[d];
        }
        let lane: Vec<i64> = (0..axis_len)
            .map(|k| flat[base + k * strides[ax]])
            .collect();
        results.push(f(&lane));
    }
    Ok((out_shape, results))
}

/// Build a 0-D-or-axis-reduced int64 ndarray of `ticks` with `out_shape`, then
/// re-tag it as the requested time dtype (`datetime64`/`timedelta64` of
/// `unit_str`) via the int64-view transport.
fn emit_time<'py>(
    py: Python<'py>,
    ticks: Vec<i64>,
    out_shape: &[usize],
    as_datetime: bool,
    unit_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = ArrayD::from_vec(IxDyn::new(out_shape), ticks).map_err(ferr_to_pyerr)?;
    if as_datetime {
        int64_to_datetime64(py, arr, unit_str)
    } else {
        int64_to_timedelta64(py, arr, unit_str)
    }
}

/// Emit an int64 index ndarray (argmin/argmax result) of `out_shape`.
fn emit_index<'py>(
    py: Python<'py>,
    idx: Vec<i64>,
    out_shape: &[usize],
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let arr = ArrayD::from_vec(IxDyn::new(out_shape), idx).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.<reduction>(a, axis=...)` for a datetime64 / timedelta64 input —
/// the REQ-4 (#944) dispatch the `stats.rs` reduction `#[pyfunction]`s route
/// to when their input dtype kind is `"M"`/`"m"`.
///
/// Compute-or-raise per the live numpy contract (module comment): the
/// computable reductions fold over the int64 ticks with NaT propagation and
/// re-tag the result; the undefined (kind, op) pairs delegate to numpy so its
/// exact `UFuncTypeError`/`TypeError` surfaces. This eliminates the
/// silent-float corruption (`fr.mean(dt)->18268.5`, `fr.std(td)->0.5`).
pub fn time_reduce<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    kind: TimeReduce,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let tkind = time_kind(arr)?.ok_or_else(|| {
        PyTypeError::new_err("time_reduce requires a datetime64 / timedelta64 array")
    })?;
    let is_datetime = tkind == TimeKind::Datetime;

    // The (kind, op) pairs numpy RAISES on -> delegate so numpy's exact
    // exception surfaces (verified live, R-DEV-2):
    //   timedelta: std / var / prod
    //   datetime:  sum / mean / std / var / cumsum / prod
    let raises = match kind {
        TimeReduce::Std | TimeReduce::Var | TimeReduce::Prod => true,
        TimeReduce::Sum | TimeReduce::Mean | TimeReduce::Cumsum => is_datetime,
        _ => false,
    };
    if raises {
        return delegate_to_numpy(py, arr, kind, axis);
    }

    // Marshal the int64 ticks + unit via the existing transport.
    let (ticks, _unit, unit_str) = extract_ticks(py, arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();

    // The result dtype of a COMPUTED reduction:
    //   sum/mean/min/max/cumsum -> same kind as input (datetime stays datetime
    //     for min/max; only timedelta reaches sum/mean/cumsum here since the
    //     datetime cases were delegated above).
    //   ptp -> ALWAYS timedelta (datetime ptp = max-min; timedelta ptp stays
    //     timedelta).
    let result_datetime = match kind {
        TimeReduce::Ptp => false,
        _ => is_datetime,
    };

    match kind {
        TimeReduce::Sum
        | TimeReduce::Mean
        | TimeReduce::Min
        | TimeReduce::Max
        | TimeReduce::Ptp => {
            match axis {
                None => {
                    let r = lane_reduce_i64(&flat, kind).ok_or_else(|| {
                        // Only reachable for empty min/max/ptp/mean (numpy raises
                        // ValueError on empty min/max; mean of empty timedelta
                        // raises too).
                        PyValueError::new_err(
                            "zero-size array to reduction operation (no identity)",
                        )
                    })?;
                    emit_time(py, vec![r], &[], result_datetime, &unit_str)
                }
                Some(ax) => {
                    let (out_shape, opt) =
                        axis_lanes_i64(&flat, &shape, ax, |lane| lane_reduce_i64(lane, kind))?;
                    let mut out = Vec::with_capacity(opt.len());
                    for v in opt {
                        out.push(v.ok_or_else(|| {
                            PyValueError::new_err(
                                "zero-size array to reduction operation (no identity)",
                            )
                        })?);
                    }
                    emit_time(py, out, &out_shape, result_datetime, &unit_str)
                }
            }
        }
        TimeReduce::Cumsum => {
            // Only timedelta reaches here (datetime cumsum delegated/raised).
            // numpy cumsum keeps the input shape; `axis=None` flattens row-major.
            // NaT propagates forward: once a lane hits NaT, every later prefix
            // is NaT (verified live).
            let cum = |lane: &[i64]| -> Vec<i64> {
                let mut acc = 0i64;
                let mut nat_seen = false;
                let mut out = Vec::with_capacity(lane.len());
                for &v in lane {
                    if nat_seen || v == NAT {
                        nat_seen = true;
                        out.push(NAT);
                    } else {
                        acc = acc.wrapping_add(v);
                        out.push(acc);
                    }
                }
                out
            };
            match axis {
                None => {
                    let out = cum(&flat);
                    let n = out.len();
                    emit_time(py, out, &[n], false, &unit_str)
                }
                Some(ax) => {
                    if ax >= shape.len() {
                        return Err(PyValueError::new_err("axis out of bounds for cumsum"));
                    }
                    // Scatter each lane's prefix sums back into the input shape.
                    let axis_len = shape[ax];
                    let strides = row_major_strides(&shape);
                    let out_dims: Vec<usize> = (0..shape.len()).filter(|&d| d != ax).collect();
                    let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
                    let out_n: usize = out_extents.iter().product::<usize>().max(1);
                    let mut out_flat = vec![0i64; flat.len()];
                    for lin in 0..out_n {
                        let mut rem = lin;
                        let mut base = 0usize;
                        for (i, &d) in out_dims.iter().enumerate() {
                            let ext = out_extents[i];
                            let c = rem % ext;
                            rem /= ext;
                            base += c * strides[d];
                        }
                        let lane: Vec<i64> = (0..axis_len)
                            .map(|k| flat[base + k * strides[ax]])
                            .collect();
                        for (k, v) in cum(&lane).into_iter().enumerate() {
                            out_flat[base + k * strides[ax]] = v;
                        }
                    }
                    emit_time(py, out_flat, &shape, false, &unit_str)
                }
            }
        }
        TimeReduce::Argmin | TimeReduce::Argmax => {
            let want_max = matches!(kind, TimeReduce::Argmax);
            match axis {
                None => {
                    if flat.is_empty() {
                        return Err(PyValueError::new_err(
                            "attempt to get argmin/argmax of an empty sequence",
                        ));
                    }
                    let i = lane_arg(&flat, want_max) as i64;
                    emit_index(py, vec![i], &[])
                }
                Some(ax) => {
                    let (out_shape, idx) =
                        axis_lanes_i64(&flat, &shape, ax, |lane| lane_arg(lane, want_max) as i64)?;
                    emit_index(py, idx, &out_shape)
                }
            }
        }
        // Delegated above.
        TimeReduce::Std | TimeReduce::Var | TimeReduce::Prod => {
            delegate_to_numpy(py, arr, kind, axis)
        }
    }
}

// ---------------------------------------------------------------------------
// isnat
// ---------------------------------------------------------------------------

/// `numpy.isnat(x)` — element-wise "is Not a Time", routed through
/// `ferray_ufunc::isnat_datetime` / `isnat_timedelta`. numpy raises
/// `TypeError` for any non-datetime/timedelta input
/// (numpy/_core/code_generators/generate_umath.py `isnat`).
#[pyfunction]
pub fn isnat<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x,))?;
    let kind = time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("ufunc 'isnat' is only defined for np.datetime64 and np.timedelta64")
    })?;
    let (ticks, _unit, _) = extract_ticks(py, &arr)?;
    let r = match kind {
        TimeKind::Datetime => {
            let fa = ticks_to_datetime(&ticks)?;
            ferray_ufunc::isnat_datetime(&fa).map_err(ferr_to_pyerr)?
        }
        TimeKind::Timedelta => {
            let fa = ticks_to_timedelta(&ticks)?;
            ferray_ufunc::isnat_timedelta(&fa).map_err(ferr_to_pyerr)?
        }
    };
    Ok(r.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// datetime_as_string
// ---------------------------------------------------------------------------

/// `numpy.datetime_as_string(arr, unit=None, timezone='naive')` — format a
/// datetime64 array as ISO-8601 strings
/// (numpy/_core/src/multiarray/datetime_strings.c `datetime_as_string`).
///
/// numpy's full formatter (timezone conversion, casting `unit`, `'%'`-style)
/// is intricate; this binding produces the ISO strings for the common path
/// (the input's own unit, `timezone='naive'`) by delegating the final
/// rendering to numpy on a *ferray-reconstructed* datetime64 array. The input
/// is first marshalled int64 ticks -> ferray DateTime64 (validating the unit
/// via [`TimeUnit::from_descr_suffix`]) and rebuilt, so an unsupported unit is
/// rejected here rather than silently mis-rendered.
#[pyfunction]
#[pyo3(signature = (arr, unit = None, timezone = "naive"))]
pub fn datetime_as_string<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    unit: Option<&str>,
    timezone: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let a = np.call_method1("asarray", (arr,))?;
    if time_kind(&a)? != Some(TimeKind::Datetime) {
        return Err(PyTypeError::new_err(
            "datetime_as_string requires a datetime64 array",
        ));
    }
    // Validate the input unit through ferray's TimeUnit, marshalling the ticks
    // (this is the ferray-side round-trip that gates the supported unit set).
    let (ticks, _u, unit_str) = extract_ticks(py, &a)?;
    let rebuilt = int64_to_datetime64(py, ticks, &unit_str)?;
    // Render with numpy's formatter on the ferray-reconstructed array.
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(u) = unit {
        kwargs.set_item("unit", u)?;
    }
    kwargs.set_item("timezone", timezone)?;
    np.call_method("datetime_as_string", (rebuilt,), Some(&kwargs))
}

// ---------------------------------------------------------------------------
// busday calendar — implemented in Rust
// ---------------------------------------------------------------------------

/// Day-of-week (Mon=0 .. Sun=6) for a day count since the 1970-01-01 epoch.
///
/// numpy/_core/src/multiarray/datetime_busday.c:32 — "1970-01-05 is Monday",
/// `day_of_week = (date - 4) % 7` (corrected into `[0,7)`). Day 0
/// (1970-01-01) is therefore Thursday (index 3).
#[inline]
fn day_of_week(date: i64) -> usize {
    (date - 4).rem_euclid(7) as usize
}

/// Parse a `weekmask` Python object into 7 booleans (Mon..Sun). Accepts the
/// 7-char string form (`"1111100"`) or a 7-element int/bool sequence
/// (`[1,1,1,1,1,0,0]`), matching numpy's `_busdaycalendar` weekmask parser
/// (numpy/_core/src/multiarray/datetime_busdaycal.c `PyArray_WeekMaskConverter`).
fn parse_weekmask(weekmask: &Bound<'_, PyAny>) -> PyResult<[bool; 7]> {
    if let Ok(s) = weekmask.extract::<String>() {
        // Day-abbreviation form ("Mon Tue ...") is also accepted by numpy, but
        // the 7-char "1111100" binary form is the common one; reject other
        // strings rather than mis-parse.
        let bytes = s.as_bytes();
        if bytes.len() == 7 && bytes.iter().all(|&c| c == b'0' || c == b'1') {
            let mut mask = [false; 7];
            for (i, &c) in bytes.iter().enumerate() {
                mask[i] = c == b'1';
            }
            return Ok(mask);
        }
        return Err(PyValueError::new_err(format!(
            "invalid weekmask string {s:?} (expected 7 chars of '0'/'1', Mon..Sun)"
        )));
    }
    let seq: Vec<i64> = weekmask.extract().map_err(|_| {
        PyValueError::new_err(
            "weekmask must be a 7-char '0'/'1' string or a 7-element int sequence",
        )
    })?;
    if seq.len() != 7 {
        return Err(PyValueError::new_err(
            "weekmask sequence must have exactly 7 elements (Mon..Sun)",
        ));
    }
    let mut mask = [false; 7];
    for (i, &v) in seq.iter().enumerate() {
        mask[i] = v != 0;
    }
    Ok(mask)
}

/// Marshal a date-array-like object to a `Vec<i64>` of day counts (and the
/// original shape, so an output array can be reshaped to match). Scalars
/// produce a length-1 vector with an empty shape.
fn dates_to_days(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<(Vec<i64>, Vec<usize>)> {
    let np = py.import("numpy")?;
    let a = np.call_method1("asarray", (obj, "datetime64[D]"))?;
    let shape: Vec<usize> = a.getattr("shape")?.extract()?;
    let i64a = a.call_method1("astype", ("int64",))?;
    let flat = i64a.call_method1("ravel", ())?;
    let days: Vec<i64> = flat.extract()?;
    Ok((days, shape))
}

/// Parse the `holidays` argument into a sorted, de-duplicated `Vec<i64>` of
/// day counts, dropping any holiday that is not itself a business day
/// (numpy/_core/src/multiarray/datetime_busdaycal.c
/// `PyArray_HolidaysConverter` normalizes + sorts + filters weekend/excluded
/// holidays out). Weekend holidays don't affect the count, matching numpy.
fn parse_holidays(
    py: Python<'_>,
    holidays: Option<&Bound<'_, PyAny>>,
    weekmask: &[bool; 7],
) -> PyResult<Vec<i64>> {
    let Some(h) = holidays else {
        return Ok(Vec::new());
    };
    if h.is_none() {
        return Ok(Vec::new());
    }
    let (days, _shape) = dates_to_days(py, h)?;
    let mut filtered: Vec<i64> = days
        .into_iter()
        .filter(|&d| weekmask[day_of_week(d)])
        .collect();
    filtered.sort_unstable();
    filtered.dedup();
    Ok(filtered)
}

#[inline]
fn is_busday_scalar(day: i64, weekmask: &[bool; 7], holidays: &[i64]) -> bool {
    weekmask[day_of_week(day)] && holidays.binary_search(&day).is_err()
}

/// Count business days in the half-open interval `[begin, end)`. Sign-aware:
/// if `begin > end` the count is negated (numpy/_core/src/multiarray/
/// datetime_busday.c:380 `apply_business_day_count`, which swaps the bounds
/// and negates).
fn busday_count_scalar(begin: i64, end: i64, weekmask: &[bool; 7], holidays: &[i64]) -> i64 {
    if begin == end {
        return 0;
    }
    let (lo, hi, sign) = if begin < end {
        (begin, end, 1i64)
    } else {
        // Swapped range excludes the original `end` and includes the original
        // `begin` (gh-23197): [end+1, begin+1).
        (end + 1, begin + 1, -1i64)
    };
    let mut count = 0i64;
    let mut d = lo;
    while d < hi {
        if is_busday_scalar(d, weekmask, holidays) {
            count += 1;
        }
        d += 1;
    }
    sign * count
}

/// Roll a date to a valid business day per `roll`. Returns `Some(NAT)` for the
/// `nat` mode on a non-busday, `None` (→ caller raises) for `raise`.
fn roll_date(
    date: i64,
    roll: &str,
    weekmask: &[bool; 7],
    holidays: &[i64],
) -> PyResult<Option<i64>> {
    if date == NAT {
        return Ok(Some(NAT));
    }
    if is_busday_scalar(date, weekmask, holidays) {
        return Ok(Some(date));
    }
    match roll {
        "forward" | "following" => {
            let mut d = date;
            while !is_busday_scalar(d, weekmask, holidays) {
                d += 1;
            }
            Ok(Some(d))
        }
        "backward" | "preceding" => {
            let mut d = date;
            while !is_busday_scalar(d, weekmask, holidays) {
                d -= 1;
            }
            Ok(Some(d))
        }
        "modifiedfollowing" => {
            let mut d = date;
            while !is_busday_scalar(d, weekmask, holidays) {
                d += 1;
            }
            // If we crossed into the next month, go backward instead.
            if month_of(d) != month_of(date) {
                let mut b = date;
                while !is_busday_scalar(b, weekmask, holidays) {
                    b -= 1;
                }
                Ok(Some(b))
            } else {
                Ok(Some(d))
            }
        }
        "modifiedpreceding" => {
            let mut d = date;
            while !is_busday_scalar(d, weekmask, holidays) {
                d -= 1;
            }
            if month_of(d) != month_of(date) {
                let mut f = date;
                while !is_busday_scalar(f, weekmask, holidays) {
                    f += 1;
                }
                Ok(Some(f))
            } else {
                Ok(Some(d))
            }
        }
        "nat" => Ok(Some(NAT)),
        "raise" => Ok(None),
        other => Err(PyValueError::new_err(format!(
            "Invalid business day roll {other:?}"
        ))),
    }
}

/// (year, month) of a day count since epoch, for the modified-roll month-edge
/// check. Computed via a civil-calendar algorithm (Howard Hinnant's
/// days_from_civil inverse), avoiding any external dependency.
fn month_of(day: i64) -> (i64, u32) {
    // days since 1970-01-01 -> civil (y, m, d). Algorithm from
    // http://howardhinnant.github.io/date_algorithms.html#civil_from_days
    let z = day + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32)
}

/// Apply a business-day offset: roll `date` to a valid business day, then step
/// `offset` business days in the offset's direction
/// (numpy/_core/src/multiarray/datetime_busday.c:272
/// `apply_business_day_offset`).
fn busday_offset_scalar(
    date: i64,
    mut offset: i64,
    roll: &str,
    weekmask: &[bool; 7],
    holidays: &[i64],
) -> PyResult<i64> {
    let rolled = match roll_date(date, roll, weekmask, holidays)? {
        Some(d) => d,
        None => {
            return Err(PyValueError::new_err(
                "Non-business day date in busday_offset (roll='raise')",
            ));
        }
    };
    if rolled == NAT {
        return Ok(NAT);
    }
    let mut d = rolled;
    while offset > 0 {
        d += 1;
        if is_busday_scalar(d, weekmask, holidays) {
            offset -= 1;
        }
    }
    while offset < 0 {
        d -= 1;
        if is_busday_scalar(d, weekmask, holidays) {
            offset += 1;
        }
    }
    Ok(d)
}

/// `numpy.is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None)`.
#[pyfunction]
#[pyo3(signature = (dates, weekmask = None, holidays = None))]
pub fn is_busday<'py>(
    py: Python<'py>,
    dates: &Bound<'py, PyAny>,
    weekmask: Option<&Bound<'py, PyAny>>,
    holidays: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mask = parse_weekmask_or_default(weekmask)?;
    let hol = parse_holidays(py, holidays, &mask)?;
    let (days, shape) = dates_to_days(py, dates)?;
    let out: Vec<bool> = days
        .iter()
        .map(|&d| is_busday_scalar(d, &mask, &hol))
        .collect();
    build_bool_output(py, out, &shape, dates)
}

/// `numpy.busday_count(begindates, enddates, weekmask='1111100',
/// holidays=None)`.
#[pyfunction]
#[pyo3(signature = (begindates, enddates, weekmask = None, holidays = None))]
pub fn busday_count<'py>(
    py: Python<'py>,
    begindates: &Bound<'py, PyAny>,
    enddates: &Bound<'py, PyAny>,
    weekmask: Option<&Bound<'py, PyAny>>,
    holidays: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mask = parse_weekmask_or_default(weekmask)?;
    let hol = parse_holidays(py, holidays, &mask)?;
    let np = py.import("numpy")?;
    // Broadcast begin/end to a common shape (numpy broadcasts the two).
    let ba = np.call_method1("asarray", (begindates, "datetime64[D]"))?;
    let ea = np.call_method1("asarray", (enddates, "datetime64[D]"))?;
    let pair = np.call_method1("broadcast_arrays", (&ba, &ea))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;
    let scalar = pl[0].getattr("ndim")?.extract::<usize>()? == 0;
    let (begin, shape) = dates_to_days(py, &pl[0])?;
    let (end, _) = dates_to_days(py, &pl[1])?;
    let out: Vec<i64> = begin
        .iter()
        .zip(end.iter())
        .map(|(&b, &e)| busday_count_scalar(b, e, &mask, &hol))
        .collect();
    build_i64_output(py, out, &shape, scalar)
}

/// `numpy.busday_offset(dates, offsets, roll='raise', weekmask='1111100',
/// holidays=None)`.
#[pyfunction]
#[pyo3(signature = (dates, offsets, roll = "raise", weekmask = None, holidays = None))]
pub fn busday_offset<'py>(
    py: Python<'py>,
    dates: &Bound<'py, PyAny>,
    offsets: &Bound<'py, PyAny>,
    roll: &str,
    weekmask: Option<&Bound<'py, PyAny>>,
    holidays: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mask = parse_weekmask_or_default(weekmask)?;
    let hol = parse_holidays(py, holidays, &mask)?;
    let np = py.import("numpy")?;
    let da = np.call_method1("asarray", (dates, "datetime64[D]"))?;
    let oa = np.call_method1("asarray", (offsets, "int64"))?;
    let pair = np.call_method1("broadcast_arrays", (&da, &oa))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;
    let scalar = pl[0].getattr("ndim")?.extract::<usize>()? == 0;
    let (days, shape) = dates_to_days(py, &pl[0])?;
    let off_flat = pl[1].call_method1("ravel", ())?;
    let offs: Vec<i64> = off_flat.extract()?;
    let mut out: Vec<i64> = Vec::with_capacity(days.len());
    for (&d, &o) in days.iter().zip(offs.iter()) {
        out.push(busday_offset_scalar(d, o, roll, &mask, &hol)?);
    }
    // Build an int64 day array, reshape, and view as datetime64[D].
    build_date_output(py, out, &shape, scalar)
}

// ----- output builders -----

fn parse_weekmask_or_default(weekmask: Option<&Bound<'_, PyAny>>) -> PyResult<[bool; 7]> {
    match weekmask {
        None => Ok([true, true, true, true, true, false, false]),
        Some(w) if w.is_none() => Ok([true, true, true, true, true, false, false]),
        Some(w) => parse_weekmask(w),
    }
}

/// Build the bool output for `is_busday`, scalarizing to a numpy `bool_` when
/// the input was a scalar (numpy returns a scalar for a scalar input).
fn build_bool_output<'py>(
    py: Python<'py>,
    data: Vec<bool>,
    shape: &[usize],
    orig: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let scalar = np
        .call_method1("asarray", (orig,))?
        .getattr("ndim")?
        .extract::<usize>()?
        == 0;
    let arr = numpy::PyArray1::<bool>::from_vec(py, data).into_any();
    let reshaped = arr.call_method1("reshape", (shape.to_vec(),))?;
    if scalar {
        let empty = pyo3::types::PyTuple::empty(py);
        reshaped.get_item(empty)
    } else {
        Ok(reshaped)
    }
}

fn build_i64_output<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    shape: &[usize],
    scalar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = numpy::PyArray1::<i64>::from_vec(py, data).into_any();
    let reshaped = arr.call_method1("reshape", (shape.to_vec(),))?;
    if scalar {
        let empty = pyo3::types::PyTuple::empty(py);
        reshaped.get_item(empty)
    } else {
        Ok(reshaped)
    }
}

/// Build a datetime64[D] output (for `busday_offset`) from day counts,
/// scalarizing to a numpy datetime64 scalar for a scalar input.
fn build_date_output<'py>(
    py: Python<'py>,
    data: Vec<i64>,
    shape: &[usize],
    scalar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = numpy::PyArray1::<i64>::from_vec(py, data).into_any();
    let reshaped = arr.call_method1("reshape", (shape.to_vec(),))?;
    let dated = reshaped.call_method1("view", ("datetime64[D]",))?;
    if scalar {
        let empty = pyo3::types::PyTuple::empty(py);
        dated.get_item(empty)
    } else {
        Ok(dated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_is_thursday() {
        // 1970-01-01 (day 0) is Thursday => index 3 (Mon=0).
        assert_eq!(day_of_week(0), 3);
        // 1970-01-05 is Monday => index 0 (numpy get_day_of_week comment).
        assert_eq!(day_of_week(4), 0);
    }

    #[test]
    fn busday_count_one_week() {
        // 2020-01-01 .. 2020-01-08 == 5 business days (Mon-Fri default mask).
        let begin = days_from("2020-01-01");
        let end = days_from("2020-01-08");
        let mask = [true, true, true, true, true, false, false];
        assert_eq!(busday_count_scalar(begin, end, &mask, &[]), 5);
        // Reversed range negates.
        assert_eq!(busday_count_scalar(end, begin, &mask, &[]), -5);
    }

    #[test]
    fn month_of_known_dates() {
        // 2020-01-01 -> (2020, 1); 2020-12-31 -> (2020, 12).
        assert_eq!(month_of(days_from("2020-01-01")), (2020, 1));
        assert_eq!(month_of(days_from("2020-12-31")), (2020, 12));
        assert_eq!(month_of(0), (1970, 1));
    }

    /// Days since epoch for an ISO date string, computed via the civil
    /// algorithm (test-only helper, mirrors numpy's int64 day count).
    fn days_from(s: &str) -> i64 {
        let parts: Vec<i64> = s.split('-').map(|p| p.parse().unwrap()).collect();
        let (y, m, d) = (parts[0], parts[1], parts[2]);
        let yy = if m <= 2 { y - 1 } else { y };
        let era = if yy >= 0 { yy } else { yy - 399 } / 400;
        let yoe = yy - era * 400;
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        era * 146_097 + doe - 719_468
    }
}
