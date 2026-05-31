"""REQ-4 (#944): datetime64 / timedelta64 reductions — compute-vs-raise parity.

Every expected value is derived LIVE from numpy 2.4 inside the test (R-CHAR-3):
where numpy COMPUTES, ferray must compute the identical datetime/timedelta
result via the int64-view transport (NO silent-float corruption); where numpy
RAISES, ferray must raise the matching exception (UFuncTypeError/TypeError —
message family acceptable per R-DEV-2), NOT return a bare float.

The corruption this REQ eliminates: prior `fr.mean(datetime64)` returned
`array(18268.5)` and `fr.std(timedelta64)` returned `0.5`, silently dropping the
time dtype where numpy raises.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Fixtures: live numpy datetime64 / timedelta64 arrays.
# ---------------------------------------------------------------------------

def _dt():
    return np.array(["2020-01-01", "2020-01-05", "2020-01-03"], dtype="datetime64[D]")


def _td():
    return np.array([5, 2, 8], dtype="timedelta64[D]")


# ---------------------------------------------------------------------------
# timedelta64 — the COMPUTE cases (sum/mean/min/max/ptp/cumsum/argmin/argmax).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "ptp", "cumsum"])
def test_timedelta_compute_matches_numpy(op):
    td = _td()
    expected = getattr(np, op)(td)
    got = getattr(fr, op)(td)
    # dtype must stay timedelta64 (NOT a float) — the corruption guard.
    assert np.asarray(got).dtype == np.asarray(expected).dtype == np.dtype("timedelta64[D]")
    assert np.array_equal(np.asarray(got), np.asarray(expected))


def test_timedelta_mean_trunc_toward_zero():
    # numpy timedelta mean truncates toward zero, NOT floor: -7/3 -> -2, 5/3 -> 1.
    for vals in ([1, 2, 2], [1, 2, 4], [-1, -2, -2], [-3, -2, -2]):
        td = np.array(vals, dtype="timedelta64[D]")
        expected = np.mean(td)
        got = fr.mean(td)
        assert np.asarray(got).dtype == np.dtype("timedelta64[D]")
        assert np.asarray(got) == np.asarray(expected), vals


@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_timedelta_argextremum_matches_numpy(op):
    td = _td()
    expected = getattr(np, op)(td)
    got = getattr(fr, op)(td)
    assert np.asarray(got).dtype == np.dtype("int64")
    assert int(np.asarray(got)) == int(np.asarray(expected))


def test_timedelta_empty_sum_is_zero():
    # numpy: sum of empty timedelta == timedelta64(0); does NOT raise.
    td = np.array([], dtype="timedelta64[D]")
    expected = np.sum(td)
    got = fr.sum(td)
    assert np.asarray(got).dtype == np.dtype("timedelta64[D]")
    assert np.asarray(got) == np.asarray(expected)


# ---------------------------------------------------------------------------
# datetime64 — the COMPUTE cases (min/max/ptp/argmin/argmax).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op,dtype", [("min", "datetime64[D]"), ("max", "datetime64[D]")])
def test_datetime_minmax_matches_numpy(op, dtype):
    dt = _dt()
    expected = getattr(np, op)(dt)
    got = getattr(fr, op)(dt)
    assert np.asarray(got).dtype == np.dtype(dtype)
    assert np.asarray(got) == np.asarray(expected)


def test_datetime_ptp_is_timedelta():
    # numpy: ptp(datetime) -> timedelta64 (max - min).
    dt = _dt()
    expected = np.ptp(dt)
    got = fr.ptp(dt)
    assert np.asarray(got).dtype == np.dtype("timedelta64[D]") == np.asarray(expected).dtype
    assert np.asarray(got) == np.asarray(expected)


@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_datetime_argextremum_matches_numpy(op):
    dt = _dt()
    expected = getattr(np, op)(dt)
    got = getattr(fr, op)(dt)
    assert np.asarray(got).dtype == np.dtype("int64")
    assert int(np.asarray(got)) == int(np.asarray(expected))


# ---------------------------------------------------------------------------
# The RAISE cases — numpy raises, ferray must NOT return a float (corruption).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op", ["mean", "sum", "std", "var", "cumsum", "prod"])
def test_datetime_undefined_raises(op):
    dt = _dt()
    # Confirm numpy raises for this (kind, op).
    with pytest.raises(Exception):
        getattr(np, op)(dt)
    # ferray must ALSO raise — never a bare float (the corruption guard).
    with pytest.raises(Exception):
        getattr(fr, op)(dt)


@pytest.mark.parametrize("op", ["std", "var", "prod"])
def test_timedelta_undefined_raises(op):
    td = _td()
    with pytest.raises(Exception):
        getattr(np, op)(td)
    with pytest.raises(Exception):
        getattr(fr, op)(td)


def test_corruption_eliminated_mean_datetime():
    # The headline corruption: fr.mean(datetime) MUST NOT be a bare float.
    dt = _dt()
    raised = False
    try:
        got = fr.mean(dt)
    except Exception:
        raised = True
    if not raised:
        # If it didn't raise it must at least preserve the datetime dtype —
        # never a bare float (18268.5).
        assert np.asarray(got).dtype.kind in ("M", "m"), repr(got)
    # numpy raises, so ferray must raise too.
    assert raised


def test_corruption_eliminated_std_timedelta():
    td = _td()
    with pytest.raises(Exception):
        fr.std(td)
    with pytest.raises(Exception):
        fr.var(td)


# ---------------------------------------------------------------------------
# NaT propagation (i64::MIN) — verified live.
# ---------------------------------------------------------------------------

def test_timedelta_nat_propagates():
    tn = np.array([5, 2, "NaT", 8], dtype="timedelta64[D]")
    for op in ["sum", "mean", "min", "max"]:
        expected = getattr(np, op)(tn)
        got = getattr(fr, op)(tn)
        assert np.asarray(got).dtype == np.dtype("timedelta64[D]")
        # NaT != NaT, so compare via isnat.
        assert np.isnat(np.asarray(got)) == np.isnat(np.asarray(expected))


def test_timedelta_cumsum_nat_propagates():
    tn = np.array([5, 2, "NaT", 8], dtype="timedelta64[D]")
    expected = np.cumsum(tn)
    got = np.asarray(fr.cumsum(tn))
    assert got.dtype == np.dtype("timedelta64[D]")
    assert np.array_equal(np.isnat(got), np.isnat(expected))
    # Non-NaT prefix values match.
    mask = ~np.isnat(expected)
    assert np.array_equal(got[mask].astype("int64"), expected[mask].astype("int64"))


def test_datetime_nat_minmax_ptp_propagate():
    dn = np.array(["2020-01-01", "NaT", "2020-01-05"], dtype="datetime64[D]")
    for op, k in [("min", "M"), ("max", "M"), ("ptp", "m")]:
        expected = getattr(np, op)(dn)
        got = np.asarray(getattr(fr, op)(dn))
        assert got.dtype.kind == k
        assert np.isnat(got) == np.isnat(expected)


def test_datetime_nat_argextremum_first_nat():
    # numpy: argmin AND argmax of a NaT-containing array both return the FIRST
    # NaT index (live).
    dn = np.array(["2020-01-05", "NaT", "NaT", "2020-01-01"], dtype="datetime64[D]")
    for op in ["argmin", "argmax"]:
        expected = int(np.asarray(getattr(np, op)(dn)))
        got = int(np.asarray(getattr(fr, op)(dn)))
        assert got == expected, op


# ---------------------------------------------------------------------------
# axis / keepdims-shape support.
# ---------------------------------------------------------------------------

def test_timedelta_axis():
    tm = np.array([[5, 2], [8, 1]], dtype="timedelta64[D]")
    for op in ["sum", "mean", "min", "max", "ptp", "cumsum"]:
        for ax in (0, 1):
            expected = getattr(np, op)(tm, axis=ax)
            got = np.asarray(getattr(fr, op)(tm, axis=ax))
            assert got.dtype == np.asarray(expected).dtype
            assert np.array_equal(got.astype("int64"), np.asarray(expected).astype("int64")), (op, ax)


def test_datetime_axis_minmax_ptp_arg():
    m = np.array(
        [["2020-01-01", "2020-01-05"], ["2020-01-03", "2020-01-02"]],
        dtype="datetime64[D]",
    )
    for op in ["min", "max", "ptp"]:
        for ax in (0, 1):
            expected = np.asarray(getattr(np, op)(m, axis=ax))
            got = np.asarray(getattr(fr, op)(m, axis=ax))
            assert got.dtype == expected.dtype
            assert np.array_equal(got.astype("int64"), expected.astype("int64")), (op, ax)
    for op in ["argmin", "argmax"]:
        for ax in (0, 1):
            expected = np.asarray(getattr(np, op)(m, axis=ax))
            got = np.asarray(getattr(fr, op)(m, axis=ax))
            assert np.array_equal(got, expected), (op, ax)


# ---------------------------------------------------------------------------
# Real / float / int paths UNCHANGED (regression guard).
# ---------------------------------------------------------------------------

def test_real_paths_unchanged():
    a = np.array([3.0, 1.0, 2.0, 4.0])
    assert fr.sum(a) == np.sum(a)
    assert fr.mean(a) == np.mean(a)
    assert fr.min(a) == np.min(a)
    assert fr.max(a) == np.max(a)
    assert fr.std(a) == pytest.approx(float(np.std(a)))
    assert fr.var(a) == pytest.approx(float(np.var(a)))
    assert fr.ptp(a) == np.ptp(a)
    assert int(fr.argmin(a)) == int(np.argmin(a))
    assert int(fr.argmax(a)) == int(np.argmax(a))
    assert np.array_equal(np.asarray(fr.cumsum(a)), np.cumsum(a))
    i = np.array([5, 2, 8, 1], dtype="int64")
    assert fr.sum(i) == np.sum(i)
    assert fr.mean(i) == pytest.approx(float(np.mean(i)))
