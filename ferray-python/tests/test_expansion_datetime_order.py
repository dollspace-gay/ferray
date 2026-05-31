"""REQ-3 (#943, epic #940): datetime64 / timedelta64 comparison + ordering.

`fr.less`/`greater`/`less_equal`/`greater_equal`/`equal`/`not_equal` and
`fr.sort`/`argsort`/`unique`/`searchsorted`/`partition`/`argpartition` over
datetime64 / timedelta64 arrays compare / order by the int64 TICK value (after
promoting two operands of the same kind to a common finer unit), with NaT
(`i64::MIN`) sorting LAST and comparing UNORDERED (every compare with a NaT
operand is False except `!=`).

Every expected value is derived LIVE from numpy 2.4 (R-CHAR-3): each assertion
calls the numpy oracle and the ferray binding on the SAME input and compares.
Real / complex paths are checked to be unregressed.
"""

import numpy as np
import ferray as fr
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

NAT = np.datetime64("NaT")
TNAT = np.timedelta64("NaT")


def dt(values, unit="D"):
    return np.array(values, dtype=f"datetime64[{unit}]")


def td(values, unit="D"):
    return np.array(values, dtype=f"timedelta64[{unit}]")


def assert_same(fr_out, np_out):
    """Same dtype + shape + values (NaT-aware via byte/int64 view)."""
    fr_arr = np.asarray(fr_out)
    np_arr = np.asarray(np_out)
    assert fr_arr.dtype == np_arr.dtype, (fr_arr.dtype, np_arr.dtype)
    assert fr_arr.shape == np_arr.shape, (fr_arr.shape, np_arr.shape)
    if fr_arr.dtype.kind in ("M", "m"):
        # int64 view compares NaT exactly (NaT == NaT under the bit pattern).
        assert np.array_equal(fr_arr.view("int64"), np_arr.view("int64"))
    else:
        assert np.array_equal(fr_arr, np_arr)


# ---------------------------------------------------------------------------
# comparison — datetime
# ---------------------------------------------------------------------------

CMP = [
    ("less", fr.less, np.less),
    ("less_equal", fr.less_equal, np.less_equal),
    ("greater", fr.greater, np.greater),
    ("greater_equal", fr.greater_equal, np.greater_equal),
    ("equal", fr.equal, np.equal),
    ("not_equal", fr.not_equal, np.not_equal),
]


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_datetime(name, frf, npf):
    a = dt(["2020-01-03", "2020-01-01", "2020-01-05"])
    pivot = dt(["2020-01-03"])
    assert_same(frf(a, pivot), npf(a, pivot))
    assert_same(frf(a, a), npf(a, a))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_timedelta(name, frf, npf):
    t = td([5, 2, 9])
    pivot = td([5])
    assert_same(frf(t, pivot), npf(t, pivot))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_cross_unit_datetime(name, frf, npf):
    a = dt(["2020-01-01", "2020-01-02"], "D")
    b = np.array(["2020-01-01T05", "2020-01-02T00"], dtype="datetime64[h]")
    assert_same(frf(a, b), npf(a, b))
    assert_same(frf(b, a), npf(b, a))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_cross_unit_timedelta(name, frf, npf):
    a = td([1, 2], "D")
    b = td([24, 47], "h")
    assert_same(frf(a, b), npf(a, b))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_nat_datetime(name, frf, npf):
    a = dt(["2020-01-01", "NaT", "2020-01-03"])
    pivot = dt(["2020-01-01"])
    assert_same(frf(a, pivot), npf(a, pivot))
    # NaT vs NaT, NaT vs real, real vs NaT
    nat_arr = dt(["NaT", "NaT"])
    real_arr = dt(["2020-01-01", "NaT"])
    assert_same(frf(nat_arr, real_arr), npf(nat_arr, real_arr))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_nat_timedelta(name, frf, npf):
    a = td([5, 2, 9]).astype("timedelta64[D]")
    a2 = np.array([5, "NaT", 9], dtype="timedelta64[D]")
    pivot = td([5])
    assert_same(frf(a2, pivot), npf(a2, pivot))


@pytest.mark.parametrize("name,frf,npf", CMP)
def test_compare_broadcast(name, frf, npf):
    a = dt([["2020-01-01", "2020-01-05"], ["2020-01-03", "2020-01-02"]])
    pivot = dt(["2020-01-03"])
    assert_same(frf(a, pivot), npf(a, pivot))


# ---------------------------------------------------------------------------
# sort / argsort — datetime + timedelta
# ---------------------------------------------------------------------------


def test_sort_datetime():
    a = dt(["2020-01-03", "2020-01-01", "2020-01-05"])
    assert_same(fr.sort(a), np.sort(a))


def test_sort_timedelta():
    t = td([5, 2, 9, 2], "s")
    assert_same(fr.sort(t), np.sort(t))


def test_sort_datetime_nat_last():
    b = dt(["2020-01-01", "NaT", "2020-01-03"])
    assert_same(fr.sort(b), np.sort(b))


def test_sort_2d_axis_last():
    # numpy's sort default axis is -1 (last); pass it explicitly. (The ferray
    # `axis=None` default flattens — a PRE-EXISTING real-path convention shared
    # by the datetime arm, see test_sort_2d_axis_none_matches_real_convention.)
    m = dt([["2020-01-03", "2020-01-01"], ["2020-01-05", "2020-01-02"]])
    assert_same(fr.sort(m, axis=1), np.sort(m, axis=1))


def test_sort_2d_axis_none_matches_real_convention():
    # ferray's `axis=None` default flattens (diverges from numpy's last-axis
    # default) for BOTH the real and the datetime path — the datetime arm mirrors
    # the existing real convention exactly (no NEW divergence introduced).
    m_dt = dt([["2020-01-03", "2020-01-01"], ["2020-01-05", "2020-01-02"]])
    m_int = m_dt.view("int64")  # same int64 ticks, int dtype
    assert_same(
        np.asarray(fr.sort(m_dt)).view("int64"),
        np.asarray(fr.sort(m_int)),
    )


def test_sort_2d_axis0():
    m = dt([["2020-01-03", "2020-01-01"], ["2020-01-05", "2020-01-02"]])
    assert_same(fr.sort(m, axis=0), np.sort(m, axis=0))


def test_argsort_datetime():
    a = dt(["2020-01-03", "2020-01-01", "2020-01-05"])
    assert_same(fr.argsort(a), np.argsort(a))


def test_argsort_datetime_nat():
    b = dt(["2020-01-01", "NaT", "2020-01-03"])
    assert_same(fr.argsort(b), np.argsort(b))


def test_argsort_timedelta():
    t = td([5, 2, 9, 2], "s")
    assert_same(fr.argsort(t), np.argsort(t))


def test_argsort_2d_axis_last():
    m = dt([["2020-01-03", "2020-01-01"], ["2020-01-05", "2020-01-02"]])
    assert_same(fr.argsort(m, axis=1), np.argsort(m, axis=1))


# ---------------------------------------------------------------------------
# unique — datetime + timedelta (NaT collapsed to one, last)
# ---------------------------------------------------------------------------


def test_unique_datetime():
    a = dt(["2020-01-05", "2020-01-01", "2020-01-05", "2020-01-03"])
    assert_same(fr.unique(a), np.unique(a))


def test_unique_datetime_nat():
    b = dt(["2020-01-03", "NaT", "2020-01-01", "NaT", "2020-01-03"])
    assert_same(fr.unique(b), np.unique(b))


def test_unique_timedelta():
    t = td([5, 2, 9, 2], "s")
    assert_same(fr.unique(t), np.unique(t))


def test_unique_timedelta_nat():
    t = np.array([5, TNAT, 2, TNAT], dtype="timedelta64[s]")
    assert_same(fr.unique(t), np.unique(t))


# ---------------------------------------------------------------------------
# searchsorted — datetime + timedelta
# ---------------------------------------------------------------------------


def test_searchsorted_datetime():
    s = np.sort(dt(["2020-01-01", "2020-01-03", "2020-01-05"]))
    q = dt(["2020-01-03"])
    assert_same(fr.searchsorted(s, q), np.searchsorted(s, q))


def test_searchsorted_side_left_right():
    s = np.sort(dt(["2020-01-01", "2020-01-03", "2020-01-05"]))
    q = dt(["2020-01-03"])
    assert_same(
        fr.searchsorted(s, q, side="left"), np.searchsorted(s, q, side="left")
    )
    assert_same(
        fr.searchsorted(s, q, side="right"), np.searchsorted(s, q, side="right")
    )


def test_searchsorted_timedelta():
    s = np.sort(td([5, 2, 9, 2], "s"))
    q = td([3], "s")
    assert_same(fr.searchsorted(s, q), np.searchsorted(s, q))


def test_searchsorted_cross_unit():
    s = np.sort(dt(["2020-01-01", "2020-01-03", "2020-01-05"], "D"))
    q = np.array(["2020-01-03T00"], dtype="datetime64[h]")
    assert_same(fr.searchsorted(s, q), np.searchsorted(s, q))


# ---------------------------------------------------------------------------
# partition / argpartition — datetime + timedelta
# ---------------------------------------------------------------------------


def test_partition_datetime():
    a = dt(["2020-01-03", "2020-01-01", "2020-01-05"])
    # kth element in final sorted position; full sort satisfies the postcondition
    assert_same(fr.partition(a, 1), np.partition(a, 1))


def test_partition_datetime_nat():
    b = dt(["2020-01-05", "2020-01-01", "NaT", "2020-01-03"])
    assert_same(fr.partition(b, 1), np.partition(b, 1))


def test_partition_timedelta():
    t = td([5, 2, 9, 2], "s")
    assert_same(fr.partition(t, 1), np.partition(t, 1))


def test_argpartition_datetime_nat():
    b = dt(["2020-01-05", "2020-01-01", "NaT", "2020-01-03"])
    assert_same(fr.argpartition(b, 1), np.argpartition(b, 1))


# ---------------------------------------------------------------------------
# real / complex paths unregressed
# ---------------------------------------------------------------------------


def test_real_compare_unregressed():
    a = np.array([3.0, 1.0, 5.0])
    assert_same(fr.less(a, 2.0), np.less(a, 2.0))
    assert_same(fr.equal(np.array([1, 2, 3]), 2), np.equal(np.array([1, 2, 3]), 2))


def test_real_sort_unregressed():
    a = np.array([3, 1, 2, 5, 4])
    assert_same(fr.sort(a), np.sort(a))
    assert_same(fr.argsort(a), np.argsort(a))
    assert_same(fr.unique(np.array([3, 1, 3, 2])), np.unique(np.array([3, 1, 3, 2])))


def test_complex_sort_unregressed():
    c = np.array([3 + 1j, 1 + 2j, 1 + 1j])
    assert_same(fr.sort(c), np.sort(c))
