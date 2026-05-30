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
        _ => Err(PyTypeError::new_err(
            "unsupported operand dtypes for datetime add (expected datetime64 + \
             timedelta64 or timedelta64 + timedelta64)",
        )),
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
        _ => Err(PyTypeError::new_err(
            "unsupported operand dtypes for datetime subtract",
        )),
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
