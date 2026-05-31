"""Expansion tests (#945) — datetime64 / timedelta64 ``arange`` (REQ-5).

``fr.arange('2020-01','2020-04',dtype='datetime64[M]')`` previously raised
``TypeError: must be real number, not str``: ``arange`` coerced its bounds to
``f64`` (``start.extract::<f64>()``) before any dtype dispatch, rejecting
datetime64 string bounds. numpy CONSTRUCTS a datetime64 / timedelta64 range for
all of these.

The builder added an ``arange_time`` branch (``creation.rs``) ahead of the
f64-coercion path: it fires when ``dtype`` is datetime64/timedelta64
(``is_time_dtype_name``) or a start/stop/step operand is a datetime64/timedelta64
scalar/array (``any_time_operand``), delegates the range construction to
``numpy.arange`` on the ORIGINAL operands (numpy owns the string->datetime64
parse, calendar month/year stepping, the int-or-timedelta64 step, and the
half-open ``[start, stop)`` semantics), then marshals the resulting buffer back
through the ferray int64-view transport (#941,
``crate::datetime::datetime_roundtrip``) — preserving numpy's dtype + unit +
shape + NaT with no lossy cast (R-CODE-4).

Every ``expected`` is derived from a LIVE numpy call (R-CHAR-3), never copied
from the ferray side. ``fr.ndarray`` IS numpy's ndarray, so ``np.asarray(fr_arr)``
is the egress for value/dtype comparison.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """A ferray result -> numpy ndarray for value/dtype comparison."""
    return np.asarray(x)


# ---------------------------------------------------------------------------
# string bounds + dtype=datetime64[unit]  (the headline divergence)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "start,stop,unit",
    [
        ("2020-01", "2020-04", "M"),
        ("2020-01-01", "2020-01-10", "D"),
        ("2020", "2025", "Y"),
        ("2020-01", "2020-12", "M"),
        ("2020-01-01", "2020-02-01", "D"),
    ],
)
def test_arange_string_bounds_datetime_dtype(start, stop, unit):
    dt = f"datetime64[{unit}]"
    expected = np.arange(start, stop, dtype=dt)
    got = _np(fr.arange(start, stop, dtype=dt))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_headline_exact():
    """The exact divergence from the dispatch (#945)."""
    expected = np.arange("2020-01", "2020-04", dtype="datetime64[M]")
    got = _np(fr.arange("2020-01", "2020-04", dtype="datetime64[M]"))
    assert np.array_equal(got, expected)
    assert list(got.astype("datetime64[M]").astype(str)) == [
        "2020-01",
        "2020-02",
        "2020-03",
    ]


# ---------------------------------------------------------------------------
# string bounds + integer step + dtype=datetime64[unit]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("step", [1, 2, 3])
def test_arange_string_bounds_int_step(step):
    dt = "datetime64[M]"
    expected = np.arange("2020-01", "2020-12", step, dtype=dt)
    got = _np(fr.arange("2020-01", "2020-12", step, dtype=dt))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# datetime64 scalar bounds + integer step  (no dtype kwarg -> inferred)
# ---------------------------------------------------------------------------


def test_arange_datetime_bounds_int_step_dtype():
    d0 = np.datetime64("2020-01-01")
    d1 = np.datetime64("2020-01-10")
    expected = np.arange(d0, d1, 2, dtype="datetime64[D]")
    got = _np(fr.arange(d0, d1, 2, dtype="datetime64[D]"))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_datetime_bounds_no_dtype():
    d0 = np.datetime64("2020-01-01")
    d1 = np.datetime64("2020-01-05")
    expected = np.arange(d0, d1)
    got = _np(fr.arange(d0, d1))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# datetime64 bounds + timedelta64 step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ndays", [1, 2, 3])
def test_arange_datetime_bounds_timedelta_step(ndays):
    d0 = np.datetime64("2020-01-01")
    d1 = np.datetime64("2020-01-10")
    step = np.timedelta64(ndays, "D")
    expected = np.arange(d0, d1, step)
    got = _np(fr.arange(d0, d1, step))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# N + dtype=timedelta64  (single positional treated as stop, start=0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["D", "h", "s", "M", "Y"])
def test_arange_n_timedelta_dtype(unit):
    dt = f"timedelta64[{unit}]"
    expected = np.arange(5, dtype=dt)
    got = _np(fr.arange(5, dtype=dt))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_timedelta_int_bounds_step():
    dt = "timedelta64[D]"
    expected = np.arange(2, 10, 2, dtype=dt)
    got = _np(fr.arange(2, 10, 2, dtype=dt))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# empty range (stop <= start)  half-open [start, stop) semantics
# ---------------------------------------------------------------------------


def test_arange_datetime_empty_range():
    expected = np.arange("2020-04", "2020-01", dtype="datetime64[M]")
    got = _np(fr.arange("2020-04", "2020-01", dtype="datetime64[M]"))
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape == (0,)
    assert np.array_equal(got, expected)


def test_arange_datetime_equal_bounds_empty():
    expected = np.arange("2020-01", "2020-01", dtype="datetime64[M]")
    got = _np(fr.arange("2020-01", "2020-01", dtype="datetime64[M]"))
    assert got.shape == expected.shape == (0,)
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# different units round-trip exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"])
def test_arange_units_preserved(unit):
    dt = f"datetime64[{unit}]"
    start = np.datetime64("2020-01-01", unit)
    stop = start + np.timedelta64(5, unit)
    expected = np.arange(start, stop, dtype=dt)
    got = _np(fr.arange(start, stop, dtype=dt))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# regression: real / numeric arange is UNCHANGED by the time branch
# ---------------------------------------------------------------------------


def test_arange_real_int_unchanged():
    expected = np.arange(0, 10, 2)
    got = _np(fr.arange(0, 10, 2))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_real_float_unchanged():
    expected = np.arange(0.0, 5.0, 0.5)
    got = _np(fr.arange(0.0, 5.0, 0.5))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_real_single_arg_unchanged():
    expected = np.arange(5)
    got = _np(fr.arange(5))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_real_int_dtype_unchanged():
    expected = np.arange(0, 5, 0.5, dtype="int64")
    got = _np(fr.arange(0, 5, 0.5, dtype="int64"))
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


def test_arange_real_zero_step_still_raises():
    with pytest.raises(ZeroDivisionError):
        fr.arange(0, 5, 0)


# ---------------------------------------------------------------------------
# 2-element and longer ranges value parity
# ---------------------------------------------------------------------------


def test_arange_datetime_day_long_range():
    expected = np.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]")
    got = _np(fr.arange("2020-01-01", "2020-03-01", dtype="datetime64[D]"))
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)
