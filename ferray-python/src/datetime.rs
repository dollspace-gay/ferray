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

/// Returns `true` if any of the supplied (optional) operands is a
/// datetime64 / timedelta64 scalar or array (numpy `dtype.kind` ∈ `{"M","m"}`).
///
/// Used by `arange` (`creation.rs`, REQ-5) to detect a datetime/timedelta range
/// from its operands — `fr.arange(d0, d1, td)` carries no `dtype=` kwarg, so the
/// time case must be recognised from a datetime64/timedelta64 `start`/`stop`/
/// `step`. A plain Python string operand is NOT treated as time here (numpy only
/// parses string bounds as datetime when an explicit `dtype='datetime64[u]'` is
/// given, which `arange` detects separately); each operand is sniffed via
/// `numpy.asarray(op).dtype.kind` so a numpy datetime64/timedelta64 scalar
/// (kind `M`/`m`) is recognised while a real number / int (kind `i`/`f`/`u`/`b`)
/// is not.
pub fn any_time_operand<'a, 'py>(
    py: Python<'py>,
    operands: impl IntoIterator<Item = Option<&'a Bound<'py, PyAny>>>,
) -> PyResult<bool>
where
    'py: 'a,
{
    let np = py.import("numpy")?;
    for op in operands.into_iter().flatten() {
        let arr = np.call_method1("asarray", (op,))?;
        if time_kind(&arr)?.is_some() {
            return Ok(true);
        }
    }
    Ok(false)
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
    // Capture the input's TRUE shape before the int64-view round-trip. numpy's
    // own `.view('int64')` promotes a 0-d array to shape `(1,)`
    // (`np.array(np.datetime64('2024-01-01')).view('int64').shape == (1,)`,
    // live numpy 2.4.x), and the reconstruction below inherits that leading
    // axis — so a 0-d datetime64 scalar / 0-d ndarray would emerge as `(1,)`.
    // numpy keeps `np.array`/`np.asarray` of a datetime64 *scalar* 0-d (shape
    // `()`, numpy/_core/numeric.py `asarray`), so we reshape the result back to
    // the input's shape. For ndim>=1 inputs (1-D/N-D/timedelta) the view does
    // not add an axis, so this reshape is a no-op; only the 0-d case is
    // corrected. dtype + unit are untouched (reshape never retypes).
    let orig_shape: Vec<usize> = arr.getattr("shape")?.extract()?;
    // Read the unit string straight from the dtype (no TimeUnit parse, so
    // Y/M/W and the full sub-second range pass through — this is pure
    // marshalling, not arithmetic).
    let (unit_str, _count) = datetime64_unit(py, arr)?;
    // Zero-copy reinterpret as int64, then materialise an owned ferray ArrayD
    // (the round-trip across the boundary that this binding exists to perform).
    let i64_view = as_int64_view(py, arr)?;
    let view: PyReadonlyArrayDyn<i64> = i64_view.extract()?;
    let ticks: ArrayD<i64> = view.as_ferray().map_err(ferr_to_pyerr)?;
    let rebuilt = match kind {
        TimeKind::Datetime => int64_to_datetime64(py, ticks, &unit_str)?,
        TimeKind::Timedelta => int64_to_timedelta64(py, ticks, &unit_str)?,
    };
    // Restore the input's shape (no-op for ndim>=1; restores `()` for 0-d).
    let shape_tuple = pyo3::types::PyTuple::new(py, &orig_shape)?;
    rebuilt.call_method1("reshape", (shape_tuple,))
}

/// Delegate a pure data-movement manipulation op (reshape / transpose / flip /
/// roll / repeat / tile / stack / concatenate and siblings) to numpy on the
/// already-coerced datetime64 / timedelta64 operand(s), then marshal the result
/// back across the ferray boundary via the int64-view transport
/// ([`datetime_roundtrip`]).
///
/// These ops carry **no element arithmetic** — they only move/copy/reorder
/// existing ticks — so numpy preserves the datetime64/timedelta64 dtype + unit +
/// NaT exactly (`np.reshape(M8[D], (2,2)).dtype == 'datetime64[D]'`, live numpy
/// 2.4.5). Running numpy's own op reuses numpy's exact dtype-passthrough
/// semantics (including `concatenate`'s cross-unit promotion to the finer unit,
/// matching `result_type`), and [`datetime_roundtrip`] reconstructs a fresh
/// ferray-marshalled buffer with no lossy cast (R-CODE-4). This mirrors the
/// delegation seam already used by `arange_time` (#945) and the `stats.rs`
/// time-reductions (#944).
///
/// `func` is a callable resolved on the `numpy` module; `args`/`kwargs` are the
/// positional/keyword args numpy's op expects (datetime64 array(s) plus shape /
/// axis / reps). The numpy result must be a datetime64 / timedelta64 array (the
/// caller only invokes this when the input is a time array and the op preserves
/// the dtype); a non-time result surfaces a `TypeError` from
/// [`datetime_roundtrip`].
pub fn delegate_manip<'py>(
    py: Python<'py>,
    func: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let result = np.getattr(func)?.call(args, kwargs)?;
    datetime_roundtrip(py, &result)
}

/// Delegate a splitting op (`split`/`array_split`/`vsplit`/`hsplit`/`dsplit`)
/// to numpy on the datetime64 / timedelta64 array, then marshal **each** part of
/// the returned list back through the int64-view transport (#948). numpy's split
/// ops return a `list` of views that all keep the input's datetime dtype+unit;
/// this rebuilds the list with fresh ferray-marshalled buffers (R-CODE-4).
pub fn delegate_manip_list<'py>(
    py: Python<'py>,
    func: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let result = np.getattr(func)?.call(args, kwargs)?;
    let parts: Vec<Bound<'py, PyAny>> = result.try_iter()?.collect::<PyResult<Vec<_>>>()?;
    let marshalled: Vec<Bound<'py, PyAny>> = parts
        .iter()
        .map(|p| datetime_roundtrip(py, p))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(pyo3::types::PyList::new(py, marshalled)?.into_any())
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
// datetime64 / timedelta64 order-statistic reductions (#946)
//
// median / nanmedian / nanmax / nanmin / nansum / percentile / quantile /
// gradient over datetime64 / timedelta64 inputs. The PRIOR `stats.rs` /
// `ufunc.rs` arms coerced the array to float64 (`coerce_dtype(.., "float64")`)
// and returned a bare float — the R-CODE-4 silent-float corruption this fixes
// (`fr.median(td) -> 4.0` instead of `timedelta64(4,'D')`).
//
// Compute-or-raise per the LIVE numpy 2.4 contract (each derived live, R-CHAR-3):
//   median(td)        -> timedelta   ; median(dt)        RAISES UFuncTypeError
//   nanmedian(td)     -> timedelta   ; nanmedian(dt)     RAISES UFuncTypeError
//   nanmax/nanmin(td) -> timedelta   ; nanmax/nanmin(dt) -> datetime
//   nansum(td)        -> timedelta   ; nansum(dt)        RAISES UFuncTypeError
//   percentile/quantile(td) -> timedelta ; (dt) -> datetime
//   gradient(td)      -> timedelta   ; gradient(dt)      -> timedelta
//   average(td/dt), histogram(td/dt)  RAISE (delegated by the binding)
//
// numpy's `median` / `_quantile` route through `add` (median: `mean` of the two
// middle elements; percentile: `_lerp`). For two datetime64 operands numpy's
// `add` ufunc is undefined (`ufunc 'add' cannot use operands with types
// '<M8[D]' and '<M8[D]'`, numpy/_core/src/umath/loops.c.src) so even-length
// datetime median RAISES — the binding delegates the datetime median/nanmedian
// to numpy so its EXACT exception surfaces (R-DEV-2). datetime
// percentile/min/max stay datetime because numpy never `add`s two datetimes
// there (percentile interpolates via `subtract` + `add` of a timedelta, which
// IS defined). NaT (`i64::MIN`) handling:
//   * median / percentile / quantile (non-nan): any NaT in a lane -> NaT result
//     (numpy's `slices_having_nans` overwrites the lane with NaT, live).
//   * nanmedian / nanmax / nanmin: NaT skipped; an all-NaT lane -> NaT.
//   * nansum: numpy does NOT skip NaT (it is plain `sum`) -> NaT propagates.
// ---------------------------------------------------------------------------

/// Sort a lane's int64 ticks ascending, separating real ticks from NaT.
/// Returns `(sorted_real_ticks, any_nat)`.
fn sorted_real_ticks(lane: &[i64]) -> (Vec<i64>, bool) {
    let any_nat = lane.contains(&NAT);
    let mut real: Vec<i64> = lane.iter().copied().filter(|&v| v != NAT).collect();
    real.sort_unstable();
    (real, any_nat)
}

/// Median of a sorted, NaT-free int64 tick lane. Odd length -> middle tick;
/// even length -> `(lo + hi) / 2` truncated toward zero — numpy computes the
/// even-case median as `mean([lo, hi])`, and timedelta `mean` truncates toward
/// zero (`(2+5)/2 -> 3`, live numpy 2.4). Empty -> `None`.
fn median_of_sorted(sorted: &[i64]) -> Option<i64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        let lo = sorted[n / 2 - 1];
        let hi = sorted[n / 2];
        // Sum in i128 then truncate toward zero on the /2 to mirror numpy's
        // timedelta `mean` (avoids i64 overflow on extreme ticks).
        Some(((lo as i128 + hi as i128) / 2) as i64)
    }
}

/// median (`skip_nat = false`) / nanmedian (`skip_nat = true`) of a datetime64 /
/// timedelta64 array. Datetime medians delegate to numpy (its `add` of two
/// datetimes RAISES); only timedelta reaches the compute path here.
pub fn time_median<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    axis: Option<usize>,
    skip_nat: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let tkind = time_kind(arr)?.ok_or_else(|| {
        PyTypeError::new_err("time_median requires a datetime64 / timedelta64 array")
    })?;
    if tkind == TimeKind::Datetime {
        // numpy `median`/`nanmedian` average the two middle elements via `add`,
        // which is undefined for two datetime64 -> RAISES UFuncTypeError. Even
        // an odd-length datetime median calls `mean` on the single middle slice,
        // which still hits `add.reduce` and raises (verified live). Delegate so
        // numpy's exact exception surfaces (R-DEV-2).
        let np = py.import("numpy")?;
        let kwargs = pyo3::types::PyDict::new(py);
        if let Some(ax) = axis {
            kwargs.set_item("axis", ax)?;
        }
        let name = if skip_nat { "nanmedian" } else { "median" };
        return np.call_method(name, (arr,), Some(&kwargs));
    }

    let (ticks, _unit, unit_str) = extract_ticks(py, arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();

    let lane_median = |lane: &[i64]| -> Option<i64> {
        let (sorted, any_nat) = sorted_real_ticks(lane);
        if !skip_nat && any_nat {
            // Plain median: a NaT anywhere in the lane yields a NaT result.
            return Some(NAT);
        }
        // nanmedian: NaT skipped; an all-NaT (now empty) lane -> NaT.
        match median_of_sorted(&sorted) {
            Some(v) => Some(v),
            None => {
                if skip_nat {
                    Some(NAT)
                } else {
                    // Empty non-nan lane: numpy warns + returns nan (float). Not
                    // reachable for timedelta with axis=None empty (the binding
                    // handles empty earlier); axis lanes are non-empty here.
                    None
                }
            }
        }
    };

    match axis {
        None => {
            let r = lane_median(&flat)
                .ok_or_else(|| PyValueError::new_err("zero-size array to reduction operation"))?;
            emit_time(py, vec![r], &[], false, &unit_str)
        }
        Some(ax) => {
            let (out_shape, opt) = axis_lanes_i64(&flat, &shape, ax, lane_median)?;
            let mut out = Vec::with_capacity(opt.len());
            for v in opt {
                out.push(v.ok_or_else(|| {
                    PyValueError::new_err("zero-size array to reduction operation")
                })?);
            }
            emit_time(py, out, &out_shape, false, &unit_str)
        }
    }
}

/// nansum / nanmax / nanmin over a datetime64 / timedelta64 array.
///
/// * `nansum`: numpy's `nansum` does NOT skip NaT for timedelta (it is plain
///   `sum`) so NaT propagates; datetime `nansum` RAISES (delegated).
/// * `nanmax` / `nanmin`: NaT skipped; an all-NaT lane -> NaT. Datetime stays
///   datetime; timedelta stays timedelta.
pub fn time_nan_reduce<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    which: TimeReduce,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let tkind = time_kind(arr)?.ok_or_else(|| {
        PyTypeError::new_err("time_nan_reduce requires a datetime64 / timedelta64 array")
    })?;
    let is_datetime = tkind == TimeKind::Datetime;

    if matches!(which, TimeReduce::Sum) {
        // nansum == plain sum for time arrays (no NaT skip). datetime sum RAISES
        // -> the existing `time_reduce` Sum arm already delegates to numpy.
        return time_reduce(py, arr, TimeReduce::Sum, axis);
    }

    let (ticks, _unit, unit_str) = extract_ticks(py, arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();
    let want_max = matches!(which, TimeReduce::Max);

    let lane_nan_extremum = |lane: &[i64]| -> Option<i64> {
        if lane.is_empty() {
            return None;
        }
        let real: Vec<i64> = lane.iter().copied().filter(|&v| v != NAT).collect();
        if real.is_empty() {
            // All-NaT lane -> NaT (numpy: RuntimeWarning + NaT, live).
            return Some(NAT);
        }
        if want_max {
            real.iter().copied().max()
        } else {
            real.iter().copied().min()
        }
    };

    match axis {
        None => {
            let r = lane_nan_extremum(&flat).ok_or_else(|| {
                PyValueError::new_err("zero-size array to reduction operation (no identity)")
            })?;
            emit_time(py, vec![r], &[], is_datetime, &unit_str)
        }
        Some(ax) => {
            let (out_shape, opt) = axis_lanes_i64(&flat, &shape, ax, lane_nan_extremum)?;
            let mut out = Vec::with_capacity(opt.len());
            for v in opt {
                out.push(v.ok_or_else(|| {
                    PyValueError::new_err("zero-size array to reduction operation (no identity)")
                })?);
            }
            emit_time(py, out, &out_shape, is_datetime, &unit_str)
        }
    }
}

/// numpy's `_lerp` over two int64 ticks at fraction `t in [0, 1]`, exactly
/// mirroring `numpy/lib/_function_base_impl.py:4594` _lerp:
///   `diff = b - a; lerp = a + (i64)(diff * t)`, but for `t >= 0.5`
///   `lerp = b - (i64)(diff * (1 - t))`.
/// The `(i64)(diff * t)` is numpy's timedelta×double multiply
/// (`TIMEDELTA_md_m_multiply`, loops.c.src:933): a C `(npy_timedelta)double`
/// cast = truncation toward zero. This reproduces `percentile([2,5,8],25) -> 4`
/// (= `5 - trunc(3*0.5) = 5 - 1`) exactly.
fn lerp_ticks(a: i64, b: i64, t: f64) -> i64 {
    let diff = (b as f64) - (a as f64);
    if t >= 0.5 {
        // b - trunc(diff * (1 - t))
        (b as f64 - (diff * (1.0 - t)).trunc()) as i64
    } else {
        // a + trunc(diff * t)
        (a as f64 + (diff * t).trunc()) as i64
    }
}

/// Linear-interpolation (Hyndman&Fan method 7) quantile of a sorted, NaT-free
/// int64 tick lane at fraction `q in [0, 1]`. `virtual_index = q*(n-1)`; the
/// result interpolates between the floor/ceil neighbours via [`lerp_ticks`].
fn quantile_of_sorted(sorted: &[i64], q: f64) -> Option<i64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(sorted[0]);
    }
    let vi = q * ((n - 1) as f64);
    let prev = vi.floor() as usize;
    let next = vi.ceil() as usize;
    if prev == next {
        return Some(sorted[prev]);
    }
    let gamma = vi - prev as f64;
    Some(lerp_ticks(sorted[prev], sorted[next], gamma))
}

/// percentile / quantile over a datetime64 / timedelta64 array at a single
/// `q` fraction (`q in [0, 1]`; the binding rescales percentile's `0..100`).
/// Both kinds COMPUTE (datetime stays datetime — numpy never `add`s two
/// datetimes here, it interpolates a timedelta offset). A lane containing any
/// NaT yields NaT (numpy's non-nan percentile, live).
pub fn time_quantile<'py>(
    py: Python<'py>,
    arr: &Bound<'py, PyAny>,
    q: f64,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let tkind = time_kind(arr)?.ok_or_else(|| {
        PyTypeError::new_err("time_quantile requires a datetime64 / timedelta64 array")
    })?;
    let is_datetime = tkind == TimeKind::Datetime;

    let (ticks, _unit, unit_str) = extract_ticks(py, arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();

    let lane_q = |lane: &[i64]| -> Option<i64> {
        let (sorted, any_nat) = sorted_real_ticks(lane);
        if any_nat {
            return Some(NAT);
        }
        quantile_of_sorted(&sorted, q)
    };

    match axis {
        None => {
            let r = lane_q(&flat)
                .ok_or_else(|| PyValueError::new_err("zero-size array to reduction operation"))?;
            emit_time(py, vec![r], &[], is_datetime, &unit_str)
        }
        Some(ax) => {
            let (out_shape, opt) = axis_lanes_i64(&flat, &shape, ax, lane_q)?;
            let mut out = Vec::with_capacity(opt.len());
            for v in opt {
                out.push(v.ok_or_else(|| {
                    PyValueError::new_err("zero-size array to reduction operation")
                })?);
            }
            emit_time(py, out, &out_shape, is_datetime, &unit_str)
        }
    }
}

// ---------------------------------------------------------------------------
// datetime64 / timedelta64 comparison + ordering (REQ-3, #943)
//
// numpy orders / compares datetime64 and timedelta64 by their int64 TICK value,
// after promoting two operands of the same kind to a COMMON (finer) unit
// (`np.promote_types(M8[D], M8[h]) == M8[h]`, verified live). The bridge is the
// existing `.view('int64')` transport (#831): the operands are first cast to the
// common dtype with numpy's own `astype` (which owns the unit rescale), then
// reinterpreted as int64 ticks and compared in Rust.
//
// NaT (`i64::MIN`) semantics, verified LIVE (numpy 2.4.5, R-CHAR-3):
//   * COMPARISON: any operand NaT makes `<`,`<=`,`>`,`>=`,`==` all FALSE and
//     `!=` TRUE (so `NaT != NaT` is True, every other compare with a NaT
//     operand is False — mirrors IEEE-NaN-style unordered comparison).
//   * ORDERING (sort/argsort/unique/searchsorted/partition): NaT sorts LAST,
//     like NaN. Modelled by mapping NaT -> `i64::MAX` for the order key
//     (`order_key`), so an ascending int64 sort places every NaT after every
//     real tick while preserving the real ticks' relative order.
// ---------------------------------------------------------------------------

/// Which of the six comparison ufuncs a `compare_time` call performs.
#[derive(Clone, Copy)]
pub enum TimeCompare {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}

/// Apply one comparison to two int64 ticks with numpy's NaT-unordered rule:
/// any NaT operand -> every comparison False except `!=` which is True.
#[inline]
fn cmp_ticks(op: TimeCompare, a: i64, b: i64) -> bool {
    if a == NAT || b == NAT {
        return matches!(op, TimeCompare::NotEqual);
    }
    match op {
        TimeCompare::Less => a < b,
        TimeCompare::LessEqual => a <= b,
        TimeCompare::Greater => a > b,
        TimeCompare::GreaterEqual => a >= b,
        TimeCompare::Equal => a == b,
        TimeCompare::NotEqual => a != b,
    }
}

/// Promote two same-kind time operands to a common (finer) unit via numpy's own
/// `promote_types` + `astype`, broadcast them, and return both int64 tick
/// buffers (row-major flat) + the broadcast output shape. The unit rescale is
/// numpy's (it owns e.g. `M8[D] -> M8[h]` = ×24); ferray only reads the int64
/// ticks afterwards, so cross-unit comparison is exact.
fn promote_pair_int64(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<usize>)> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    let common = np.call_method1(
        "promote_types",
        (aa.getattr("dtype")?, ba.getattr("dtype")?),
    )?;
    let ac = aa.call_method1("astype", (&common,))?;
    let bc = ba.call_method1("astype", (&common,))?;
    let pair = np.call_method1("broadcast_arrays", (&ac, &bc))?;
    let pl: Vec<Bound<PyAny>> = pair.extract()?;
    let shape: Vec<usize> = pl[0].getattr("shape")?.extract()?;
    let (ta, _ua, _) = extract_ticks(py, &pl[0])?;
    let (tb, _ub, _) = extract_ticks(py, &pl[1])?;
    let fa: Vec<i64> = ta.iter().copied().collect();
    let fb: Vec<i64> = tb.iter().copied().collect();
    Ok((fa, fb, shape))
}

/// `true` if BOTH operands are time arrays of the same kind (datetime/datetime
/// or timedelta/timedelta) — the only operand pairs numpy's comparison ufuncs
/// define over the time dtypes. A datetime-vs-timedelta or time-vs-numeric pair
/// is NOT comparable (numpy raises / returns NotImplemented), so the caller must
/// fall through to its real / delegate path.
pub fn is_time_compare(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    b: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let ba = np.call_method1("asarray", (b,))?;
    match (time_kind(&aa)?, time_kind(&ba)?) {
        (Some(ka), Some(kb)) => Ok(ka == kb),
        _ => Ok(false),
    }
}

/// Element-wise comparison of two same-kind datetime64 / timedelta64 operands
/// by int64 tick (common finer unit), with numpy's NaT-unordered rule. Returns
/// a bool ndarray of the broadcast shape (REQ-3). The caller (the `ufunc.rs`
/// comparison `#[pyfunction]`s) gates on [`is_time_compare`] first.
pub fn compare_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    op: TimeCompare,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let (fa, fb, shape) = promote_pair_int64(py, a, b)?;
    let out: Vec<bool> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| cmp_ticks(op, x, y))
        .collect();
    let arr = ArrayD::from_vec(IxDyn::new(&shape), out).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

// ---------------------------------------------------------------------------
// ordering: sort / argsort / unique / searchsorted / partition (REQ-3)
// ---------------------------------------------------------------------------

/// Order key for an int64 tick: NaT (`i64::MIN`) maps to `i64::MAX` so an
/// ascending sort places it LAST (numpy sorts NaT last, like NaN). Real ticks
/// keep their value, preserving their relative order.
#[inline]
fn order_key(t: i64) -> i64 {
    if t == NAT { i64::MAX } else { t }
}

/// Lower a `Vec<i64>` of ticks (of `out_shape`) back to numpy as the requested
/// time dtype.
fn emit_ticks<'py>(
    py: Python<'py>,
    ticks: Vec<i64>,
    out_shape: &[usize],
    kind: TimeKind,
    unit_str: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let arr = ArrayD::from_vec(IxDyn::new(out_shape), ticks).map_err(ferr_to_pyerr)?;
    match kind {
        TimeKind::Datetime => int64_to_datetime64(py, arr, unit_str),
        TimeKind::Timedelta => int64_to_timedelta64(py, arr, unit_str),
    }
}

/// `numpy.sort(a, axis=-1)` over a datetime64 / timedelta64 array: sort each
/// 1-D lane along `axis` (default = last, like numpy) by int64 tick with NaT
/// last; re-tag with the input's own unit (REQ-3).
pub fn sort_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    let kind = time_kind(&arr)?
        .ok_or_else(|| PyTypeError::new_err("sort_time requires a datetime64/timedelta64 array"))?;
    let (ticks, _unit, unit_str) = extract_ticks(py, &arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();

    // numpy default axis for `sort` is the LAST axis; `axis=None` flattens.
    if shape.is_empty() {
        return emit_ticks(py, flat, &shape, kind, &unit_str);
    }
    let (work_shape, ax) = match axis {
        None => {
            let n = flat.len();
            (vec![n], 0usize)
        }
        Some(ax) => (shape.clone(), ax),
    };
    let mut buf = flat;
    sort_axis_inplace(&mut buf, &work_shape, ax, false)?;
    emit_ticks(py, buf, &work_shape, kind, &unit_str)
}

/// In-place ascending sort of every lane along `axis` of a row-major i64 buffer,
/// NaT mapped last via [`order_key`]. When `stable` the sort is stable (used by
/// argsort to match numpy's order on tie / NaT runs).
fn sort_axis_inplace(buf: &mut [i64], shape: &[usize], ax: usize, _stable: bool) -> PyResult<()> {
    if ax >= shape.len() {
        return Err(PyValueError::new_err("axis out of bounds for time sort"));
    }
    let axis_len = shape[ax];
    if axis_len <= 1 {
        return Ok(());
    }
    let strides = row_major_strides(shape);
    let out_dims: Vec<usize> = (0..shape.len()).filter(|&d| d != ax).collect();
    let out_extents: Vec<usize> = out_dims.iter().map(|&d| shape[d]).collect();
    let out_n: usize = out_extents.iter().product::<usize>().max(1);
    for lin in 0..out_n {
        let mut rem = lin;
        let mut base = 0usize;
        for (i, &d) in out_dims.iter().enumerate() {
            let ext = out_extents[i];
            let c = rem % ext;
            rem /= ext;
            base += c * strides[d];
        }
        let mut lane: Vec<i64> = (0..axis_len).map(|k| buf[base + k * strides[ax]]).collect();
        lane.sort_by_key(|&t| order_key(t));
        for (k, v) in lane.into_iter().enumerate() {
            buf[base + k * strides[ax]] = v;
        }
    }
    Ok(())
}

/// `numpy.argsort(a, axis=-1)` over a datetime64 / timedelta64 array: a STABLE
/// argsort of each lane by int64 tick (NaT last), returning int64 indices
/// (REQ-3). Stable, so equal / NaT runs keep first-occurrence order (numpy's
/// `argsort` default `kind='quicksort'` is not stable, but a stable order is a
/// valid argsort and matches numpy on the verified live cases).
pub fn argsort_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    let _ = time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("argsort_time requires a datetime64/timedelta64 array")
    })?;
    let (ticks, _unit, _) = extract_ticks(py, &arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();

    let (work_shape, ax) = match axis {
        None => (vec![flat.len()], 0usize),
        Some(ax) => (shape.clone(), ax),
    };
    if ax >= work_shape.len() {
        return Err(PyValueError::new_err("axis out of bounds for argsort"));
    }
    let axis_len = work_shape[ax];
    let strides = row_major_strides(&work_shape);
    let out_dims: Vec<usize> = (0..work_shape.len()).filter(|&d| d != ax).collect();
    let out_extents: Vec<usize> = out_dims.iter().map(|&d| work_shape[d]).collect();
    let out_n: usize = out_extents.iter().product::<usize>().max(1);
    let mut idx_flat = vec![0i64; flat.len()];
    for lin in 0..out_n {
        let mut rem = lin;
        let mut base = 0usize;
        for (i, &d) in out_dims.iter().enumerate() {
            let ext = out_extents[i];
            let c = rem % ext;
            rem /= ext;
            base += c * strides[d];
        }
        let mut order: Vec<usize> = (0..axis_len).collect();
        // Stable sort by order key; ties (incl. NaT runs) keep input order.
        order.sort_by_key(|&k| order_key(flat[base + k * strides[ax]]));
        for (pos, k) in order.into_iter().enumerate() {
            idx_flat[base + pos * strides[ax]] = k as i64;
        }
    }
    let arr = ArrayD::from_vec(IxDyn::new(&work_shape), idx_flat).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// `numpy.unique(a)` over a datetime64 / timedelta64 array: sorted unique ticks
/// (NaT last), with ALL NaT collapsed to a SINGLE trailing NaT (numpy keeps one
/// NaT even though `NaT != NaT`, verified live). Returns a 1-D time array of the
/// input's unit (REQ-3).
pub fn unique_time<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    let kind = time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("unique_time requires a datetime64/timedelta64 array")
    })?;
    let (ticks, _unit, unit_str) = extract_ticks(py, &arr)?;
    let mut flat: Vec<i64> = ticks.iter().copied().collect();
    flat.sort_by_key(|&t| order_key(t));
    let mut out: Vec<i64> = Vec::with_capacity(flat.len());
    let mut nat_emitted = false;
    for t in flat {
        if t == NAT {
            // Collapse every NaT to a single trailing NaT (numpy keeps one).
            if !nat_emitted {
                out.push(NAT);
                nat_emitted = true;
            }
            continue;
        }
        if out.last().copied() != Some(t) {
            out.push(t);
        }
    }
    let n = out.len();
    emit_ticks(py, out, &[n], kind, &unit_str)
}

/// `numpy.searchsorted(a, v, side)` over a sorted datetime64 / timedelta64 `a`
/// and time query `v` (same kind): insertion points by int64 tick (NaT ordered
/// last, via [`order_key`]), returning int64 indices (REQ-3). `a` is assumed
/// already sorted (numpy's contract). The two operands are promoted to a common
/// finer unit first so a cross-unit query is exact.
pub fn searchsorted_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    side: &str,
    v_scalar: bool,
) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let right = match side {
        "left" => false,
        "right" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "side must be 'left' or 'right', got {other:?}"
            )));
        }
    };
    // Promote both to a common unit, in int64 ticks (a is 1-D, v is N-D).
    let (haystack, needles, v_shape) = promote_pair_for_search(py, a, v)?;
    let keys: Vec<i64> = haystack.iter().map(|&t| order_key(t)).collect();
    let out: Vec<i64> = needles
        .iter()
        .map(|&q| {
            let qk = order_key(q);
            // left -> first index i with keys[i] >= qk; right -> > qk.
            let mut lo = 0usize;
            let mut hi = keys.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                let go_right = if right {
                    keys[mid] <= qk
                } else {
                    keys[mid] < qk
                };
                if go_right {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo as i64
        })
        .collect();
    if v_scalar {
        let arr = ArrayD::from_vec(IxDyn::new(&[]), out).map_err(ferr_to_pyerr)?;
        return Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any());
    }
    let arr = ArrayD::from_vec(IxDyn::new(&v_shape), out).map_err(ferr_to_pyerr)?;
    Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
}

/// Promote a sorted haystack `a` and query `v` (same time kind) to a common
/// finer unit and return `(a_ticks, v_ticks, v_shape)` in int64.
fn promote_pair_for_search(
    py: Python<'_>,
    a: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
) -> PyResult<(Vec<i64>, Vec<i64>, Vec<usize>)> {
    let np = py.import("numpy")?;
    let aa = np.call_method1("asarray", (a,))?;
    let va = np.call_method1("asarray", (v,))?;
    let common = np.call_method1(
        "promote_types",
        (aa.getattr("dtype")?, va.getattr("dtype")?),
    )?;
    let ac = aa.call_method1("astype", (&common,))?;
    let vc = va.call_method1("astype", (&common,))?;
    let v_shape: Vec<usize> = vc.getattr("shape")?.extract()?;
    let (ta, _ua, _) = extract_ticks(py, &ac)?;
    let (tv, _uv, _) = extract_ticks(py, &vc)?;
    Ok((
        ta.iter().copied().collect(),
        tv.iter().copied().collect(),
        v_shape,
    ))
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
// datetime64 / timedelta64 elementwise + selection ops (#947)
//
// The LAST datetime blocker: `diff` / `ediff1d` / `abs` / `negative` / `sign`
// / `maximum` / `minimum` / `where` / `clip` / `count_nonzero` / `nonzero`
// over datetime64 / timedelta64 inputs. The PRIOR binding paths rejected the
// `M`/`m` dtype (the real-only `match_dtype_*` macros raised a ferray-side
// `TypeError`) where numpy COMPUTES a datetime/timedelta (or int) result.
//
// Compute-or-raise per the LIVE numpy 2.4.5 contract (each derived live,
// R-CHAR-3):
//   diff(datetime)    -> timedelta   ; diff(timedelta)    -> timedelta
//   ediff1d(datetime) -> timedelta   ; ediff1d(timedelta) -> timedelta
//   abs(td)/negative(td) -> timedelta (|ticks| / -ticks; NaT propagates)
//   sign(td)          -> timedelta   ({-1,0,1} ticks AS a timedelta — verified
//                        live: `np.sign(td).dtype == timedelta64`, NOT int)
//   abs/negative/sign(datetime) -> RAISE UFuncTypeError (undefined)
//   maximum/minimum(dt|td, same-kind) -> dt|td (elementwise by tick; ANY NaT
//                        operand -> NaT, like NaN-propagate; cross-unit ->
//                        finer common unit)
//   where(cond, dt|td, dt|td)  -> dt|td (select by tick)
//   clip(dt|td, lo, hi)        -> dt|td (composes maximum + minimum; NaT bound
//                        -> NaT, handled by the propagation above)
//   count_nonzero(td) -> int (count of ticks != 0; NaT (i64::MIN) IS nonzero)
//   nonzero(td)       -> tuple of int index arrays (ticks != 0; NaT nonzero)
//
// The heavy / edge cases (unit promotion in maximum/minimum/where, numpy's
// exact diff/ediff1d/sign semantics + its precise RAISE exceptions) are
// delegated to numpy on the time array and marshalled back across the ferray
// boundary via the int64-view transport ([`datetime_roundtrip`] / the
// `int64_to_*` helpers), so numpy owns the semantics and ferray owns the
// boundary contract (a fresh ferray-marshalled buffer, no lossy cast,
// R-CODE-4). The count_nonzero / nonzero arms compute directly on the int64
// ticks (NaT = `i64::MIN` is naturally `!= 0`).
// ---------------------------------------------------------------------------

/// `numpy.diff(a, n)` over a datetime64 / timedelta64 array -> timedelta64.
/// `diff(datetime)` and `diff(timedelta)` BOTH yield a timedelta of the input's
/// unit (consecutive differences; `datetime - datetime = timedelta`). Delegated
/// to numpy (which owns the `n`-th-difference fold + NaT propagation) and
/// marshalled back through the int64-view transport.
pub fn diff_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    time_kind(&arr)?
        .ok_or_else(|| PyTypeError::new_err("diff_time requires a datetime64/timedelta64 array"))?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("n", n)?;
    let result = np.call_method("diff", (&arr,), Some(&kwargs))?;
    datetime_roundtrip(py, &result)
}

/// `numpy.ediff1d(ary, to_end=None, to_begin=None)` over a datetime64 /
/// timedelta64 array -> timedelta64 (consecutive differences). Delegated to
/// numpy and marshalled back. `to_end` / `to_begin` are passed through verbatim
/// (numpy validates / coerces them); the binding's typed signature only forwards
/// the bare array case for the time dtypes (the divergence pin).
pub fn ediff1d_time<'py>(
    py: Python<'py>,
    ary: &Bound<'py, PyAny>,
    to_end: Option<&Bound<'py, PyAny>>,
    to_begin: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (ary,))?;
    time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("ediff1d_time requires a datetime64/timedelta64 array")
    })?;
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(e) = to_end {
        kwargs.set_item("to_end", e)?;
    }
    if let Some(b) = to_begin {
        kwargs.set_item("to_begin", b)?;
    }
    let result = np.call_method("ediff1d", (&arr,), Some(&kwargs))?;
    datetime_roundtrip(py, &result)
}

/// Which unary elementwise op a [`unary_time`] call performs.
#[derive(Clone, Copy)]
pub enum TimeUnary {
    Abs,
    Negative,
    Sign,
}

impl TimeUnary {
    fn numpy_name(self) -> &'static str {
        match self {
            TimeUnary::Abs => "absolute",
            TimeUnary::Negative => "negative",
            TimeUnary::Sign => "sign",
        }
    }
}

/// `abs` / `negative` / `sign` over a datetime64 / timedelta64 array (#947).
///
/// * timedelta: `abs(td)` -> magnitude timedelta, `negative(td)` -> `-ticks`
///   timedelta, `sign(td)` -> `{-1,0,1}` AS a timedelta (verified live:
///   `np.sign(td).dtype == timedelta64`). NaT (`i64::MIN`) propagates for
///   abs/negative; for `sign`, numpy treats NaT's negative ticks as `-1` (live).
/// * datetime: numpy RAISES `UFuncTypeError` (abs/negative/sign undefined over
///   datetime64) -> delegate to numpy so its EXACT exception surfaces (R-DEV-2).
///
/// The timedelta arms are computed on the int64 ticks (`abs`/`-`) for the
/// magnitude/negate, and delegated to numpy for `sign` (so numpy's exact
/// NaT-sign + result dtype apply) then marshalled back through the int64-view
/// transport.
pub fn unary_time<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    op: TimeUnary,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (x,))?;
    let kind = time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("unary_time requires a datetime64/timedelta64 array")
    })?;
    if kind == TimeKind::Datetime {
        // abs/negative/sign of datetime64 are undefined in numpy -> delegate so
        // numpy raises its exact UFuncTypeError (R-DEV-2).
        return np.call_method1(op.numpy_name(), (&arr,));
    }
    match op {
        TimeUnary::Abs | TimeUnary::Negative => {
            let (ticks, _u, unit_str) = extract_ticks(py, &arr)?;
            let out: Vec<i64> = ticks
                .iter()
                .map(|&t| {
                    if t == NAT {
                        NAT
                    } else if matches!(op, TimeUnary::Abs) {
                        t.wrapping_abs()
                    } else {
                        t.wrapping_neg()
                    }
                })
                .collect();
            let r = ArrayD::from_vec(IxDyn::new(ticks.shape()), out).map_err(ferr_to_pyerr)?;
            int64_to_timedelta64(py, r, &unit_str)
        }
        TimeUnary::Sign => {
            // `sign(td)` keeps the timedelta dtype with `{-1,0,1}` ticks; numpy
            // treats NaT's `i64::MIN` ticks as a negative -> sign `-1` (verified
            // live). Delegate the exact semantics + marshal back.
            let result = np.call_method1("sign", (&arr,))?;
            datetime_roundtrip(py, &result)
        }
    }
}

/// `numpy.maximum` / `numpy.minimum` over two SAME-KIND datetime64 /
/// timedelta64 operands (#947) -> datetime64 / timedelta64. Element-wise by
/// int64 tick after promoting both operands to a common (finer) unit; ANY NaT
/// operand makes the result NaT (numpy NaN-propagate semantics, verified live:
/// `np.maximum(dt, NaT) -> NaT`). Delegated to numpy (which owns the unit
/// promotion + NaT propagation) and marshalled back through the int64-view
/// transport. The caller gates on [`is_time_op`] / same-kind first; a
/// datetime-vs-timedelta or time-vs-numeric pair falls through to the numeric
/// path / numpy's own raise.
pub fn minmax_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    want_max: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let func = if want_max { "maximum" } else { "minimum" };
    let result = np.call_method1(func, (a, b))?;
    datetime_roundtrip(py, &result)
}

/// `numpy.where(cond, x, y)` over datetime64 / timedelta64 `x`/`y` (#947) ->
/// datetime64 / timedelta64. Delegated to numpy (which owns the broadcast +
/// `result_type` unit promotion + NaT carry) and marshalled back through the
/// int64-view transport. The caller gates on both `x` and `y` being time arrays
/// of the same kind first.
pub fn where_time<'py>(
    py: Python<'py>,
    cond: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    y: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let result = np.call_method1("where", (cond, x, y))?;
    datetime_roundtrip(py, &result)
}

/// `true` if BOTH `x` and `y` are time arrays of the SAME kind (datetime/
/// datetime or timedelta/timedelta) — the pairs `where` / `maximum` / `minimum`
/// / `clip` define over the time dtypes. A mixed or time-vs-numeric pair falls
/// through to the real path.
pub fn is_time_pair_same_kind(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let np = py.import("numpy")?;
    let xa = np.call_method1("asarray", (x,))?;
    let ya = np.call_method1("asarray", (y,))?;
    match (time_kind(&xa)?, time_kind(&ya)?) {
        (Some(kx), Some(ky)) => Ok(kx == ky),
        _ => Ok(false),
    }
}

/// `numpy.count_nonzero(a, axis=None)` over a datetime64 / timedelta64 array
/// (#947) -> int (or int array for `axis`). Counts ticks `!= 0`; NaT
/// (`i64::MIN`) is nonzero, so it IS counted (verified live:
/// `np.count_nonzero([0, NaT]) == 1`). Computed on the int64 ticks.
pub fn count_nonzero_time<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    axis: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("count_nonzero_time requires a datetime64/timedelta64 array")
    })?;
    let (ticks, _u, _unit_str) = extract_ticks(py, &arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let flat: Vec<i64> = ticks.iter().copied().collect();
    match axis {
        None => {
            let n = flat.iter().filter(|&&t| t != 0).count() as i64;
            // numpy returns a Python int for axis=None.
            Ok(n.into_pyobject(py)?.into_any())
        }
        Some(ax) => {
            let (out_shape, counts) = axis_lanes_i64(&flat, &shape, ax, |lane| {
                lane.iter().filter(|&&t| t != 0).count() as i64
            })?;
            emit_index(py, counts, &out_shape)
        }
    }
}

/// `numpy.nonzero(a)` over a datetime64 / timedelta64 array (#947) -> tuple of
/// int64 index arrays (one per dimension), the coordinates of ticks `!= 0`
/// (NaT is nonzero). Computed on the int64 ticks; row-major flat order, matching
/// numpy's C-order `nonzero`.
pub fn nonzero_time<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use ferray_numpy_interop::IntoNumPy;
    let np = py.import("numpy")?;
    let arr = np.call_method1("asarray", (a,))?;
    time_kind(&arr)?.ok_or_else(|| {
        PyTypeError::new_err("nonzero_time requires a datetime64/timedelta64 array")
    })?;
    let (ticks, _u, _unit_str) = extract_ticks(py, &arr)?;
    let shape: Vec<usize> = ticks.shape().to_vec();
    let ndim = shape.len();
    let flat: Vec<i64> = ticks.iter().copied().collect();
    let strides = row_major_strides(&shape);
    // One index vector per dimension; the flat positions of nonzero ticks
    // expanded to per-axis coordinates (row-major, like numpy's nonzero).
    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); ndim];
    for (lin, &t) in flat.iter().enumerate() {
        if t != 0 {
            for d in 0..ndim {
                coords[d].push(((lin / strides[d]) % shape[d]) as i64);
            }
        }
    }
    let arrays: Vec<Bound<'py, PyAny>> = coords
        .into_iter()
        .map(|c| {
            let n = c.len();
            let arr = ArrayD::from_vec(IxDyn::new(&[n]), c).map_err(ferr_to_pyerr)?;
            Ok(arr.into_pyarray(py).map_err(ferr_to_pyerr)?.into_any())
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(pyo3::types::PyTuple::new(py, arrays)?.into_any())
}

// ---------------------------------------------------------------------------
// datetime_as_string
// ---------------------------------------------------------------------------

/// `numpy.datetime_as_string(arr, unit=None, timezone='naive')` — format a
/// datetime64 array as ISO-8601 strings
/// (numpy/_core/src/multiarray/datetime_strings.c `datetime_as_string`).
///
/// numpy's full formatter (timezone conversion, casting `unit`, `'%'`-style)
/// is intricate and shape/unit-sensitive: a 0-d scalar input renders to a 0-d
/// `<U10` string (`array('2024-01-01', dtype='<U10')`), and the output string
/// width follows the array's own unit. This binding validates that the input is
/// datetime64, then delegates the rendering to `numpy.datetime_as_string` on the
/// ORIGINAL array, so shape (0-d stays 0-d), unit, and string width match numpy
/// exactly. (A prior version reconstructed the array through an int64-tick
/// round-trip, which promoted a 0-d scalar to a 1-element 1-D array and widened
/// the unit, yielding `<U28` instead of `<U10`.)
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
    let kwargs = pyo3::types::PyDict::new(py);
    if let Some(u) = unit {
        kwargs.set_item("unit", u)?;
    }
    kwargs.set_item("timezone", timezone)?;
    np.call_method("datetime_as_string", (a,), Some(&kwargs))
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
