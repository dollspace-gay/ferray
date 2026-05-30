"""datetime64 / timedelta64 top-level surface — ferray vs numpy (refs #831).

Each test constructs the expected value from a live ``numpy`` call (the
oracle), so no value is literal-copied from the ferray side (R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# datetime_data — pure-Python metadata on the shared numpy dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
def test_datetime_data_units(unit):
    dt = np.dtype(f"datetime64[{unit}]")
    assert fr.datetime_data(dt) == np.datetime_data(dt)


def test_datetime_data_timedelta():
    dt = np.dtype("timedelta64[s]")
    assert fr.datetime_data(dt) == np.datetime_data(dt)


# ---------------------------------------------------------------------------
# datetime64 arithmetic through ferray.add / ferray.subtract
# ---------------------------------------------------------------------------

def test_add_datetime_timedelta():
    a = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    t = np.array([1, 2], dtype="timedelta64[D]")
    got = fr.add(a, t)
    exp = np.add(a, t)
    np.testing.assert_array_equal(got, exp)
    assert got.dtype == exp.dtype


def test_add_timedelta_datetime_commutes():
    a = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    t = np.array([1, 2], dtype="timedelta64[D]")
    np.testing.assert_array_equal(fr.add(t, a), np.add(t, a))


def test_subtract_two_datetimes_gives_timedelta():
    a = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    got = fr.subtract(a[1], a[0])
    exp = np.subtract(a[1], a[0])
    assert got == exp
    assert got.dtype == exp.dtype


def test_subtract_datetime_array():
    a = np.array(["2020-01-10", "2020-02-01"], dtype="datetime64[D]")
    b = np.array(["2020-01-01", "2020-01-15"], dtype="datetime64[D]")
    np.testing.assert_array_equal(fr.subtract(a, b), np.subtract(a, b))


def test_subtract_datetime_timedelta_gives_datetime():
    a = np.array(["2020-01-10", "2020-02-01"], dtype="datetime64[D]")
    t = np.array([3, 5], dtype="timedelta64[D]")
    got = fr.subtract(a, t)
    exp = np.subtract(a, t)
    np.testing.assert_array_equal(got, exp)
    assert got.dtype == exp.dtype


def test_timedelta_plus_timedelta():
    s = np.array([10, 20], dtype="timedelta64[s]")
    t = np.array([5, 7], dtype="timedelta64[s]")
    np.testing.assert_array_equal(fr.add(s, t), np.add(s, t))
    np.testing.assert_array_equal(fr.subtract(s, t), np.subtract(s, t))


def test_datetime_arith_seconds_unit():
    a = np.array(["2020-01-01T00:00:00", "2020-01-01T01:00:00"], dtype="datetime64[s]")
    t = np.array([30, 90], dtype="timedelta64[s]")
    np.testing.assert_array_equal(fr.add(a, t), np.add(a, t))


def test_datetime_arith_nat_propagation():
    a = np.array(["2020-01-01", "NaT"], dtype="datetime64[D]")
    t = np.array([1, 2], dtype="timedelta64[D]")
    got = fr.add(a, t)
    exp = np.add(a, t)
    # NaT compares unequal to itself, so compare the int64 view.
    np.testing.assert_array_equal(got.view("int64"), exp.view("int64"))


def test_add_datetime_helper_alias():
    a = np.array(["2020-01-01"], dtype="datetime64[D]")
    t = np.array([4], dtype="timedelta64[D]")
    np.testing.assert_array_equal(fr.add_datetime(a, t), np.add(a, t))
    np.testing.assert_array_equal(fr.sub_datetime(a, t), np.subtract(a, t))


# ---------------------------------------------------------------------------
# isnat
# ---------------------------------------------------------------------------

def test_isnat_datetime():
    a = np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[D]")
    np.testing.assert_array_equal(fr.isnat(a), np.isnat(a))


def test_isnat_timedelta():
    t = np.array([1, np.timedelta64("NaT"), 3], dtype="timedelta64[D]")
    np.testing.assert_array_equal(fr.isnat(t), np.isnat(t))


def test_isnat_rejects_float():
    with pytest.raises(TypeError):
        fr.isnat(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# datetime_as_string
# ---------------------------------------------------------------------------

def test_datetime_as_string_days():
    a = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    np.testing.assert_array_equal(fr.datetime_as_string(a), np.datetime_as_string(a))


def test_datetime_as_string_seconds():
    a = np.array(["2020-01-01T12:30:45"], dtype="datetime64[s]")
    np.testing.assert_array_equal(fr.datetime_as_string(a), np.datetime_as_string(a))


def test_datetime_as_string_ms():
    a = np.array(["2020-01-01T12:00:00.500"], dtype="datetime64[ms]")
    np.testing.assert_array_equal(fr.datetime_as_string(a), np.datetime_as_string(a))


# ---------------------------------------------------------------------------
# is_busday
# ---------------------------------------------------------------------------

def test_is_busday_scalar():
    # 2020-01-04 is a Saturday.
    assert bool(fr.is_busday("2020-01-04")) == bool(np.is_busday("2020-01-04"))
    assert bool(fr.is_busday("2020-01-06")) == bool(np.is_busday("2020-01-06"))


def test_is_busday_array():
    dates = ["2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"]
    np.testing.assert_array_equal(fr.is_busday(dates), np.is_busday(dates))


def test_is_busday_weekmask():
    dates = ["2020-01-04", "2020-01-05", "2020-01-06"]
    wm = "1111110"  # Saturday is now a business day
    np.testing.assert_array_equal(
        fr.is_busday(dates, weekmask=wm), np.is_busday(dates, weekmask=wm)
    )


def test_is_busday_holidays():
    dates = ["2020-01-01", "2020-01-02"]
    hol = ["2020-01-01"]
    np.testing.assert_array_equal(
        fr.is_busday(dates, holidays=hol), np.is_busday(dates, holidays=hol)
    )


# ---------------------------------------------------------------------------
# busday_count
# ---------------------------------------------------------------------------

def test_busday_count_one_week():
    assert int(fr.busday_count("2020-01-01", "2020-01-08")) == int(
        np.busday_count("2020-01-01", "2020-01-08")
    )


def test_busday_count_reversed():
    assert int(fr.busday_count("2020-01-08", "2020-01-01")) == int(
        np.busday_count("2020-01-08", "2020-01-01")
    )


def test_busday_count_weekmask():
    got = fr.busday_count("2020-01-01", "2020-01-08", weekmask="1111110")
    exp = np.busday_count("2020-01-01", "2020-01-08", weekmask="1111110")
    assert int(got) == int(exp)


def test_busday_count_holidays():
    got = fr.busday_count("2020-01-01", "2020-01-08", holidays=["2020-01-02"])
    exp = np.busday_count("2020-01-01", "2020-01-08", holidays=["2020-01-02"])
    assert int(got) == int(exp)


def test_busday_count_seq_weekmask():
    got = fr.busday_count("2020-01-01", "2020-01-08", weekmask=[1, 1, 1, 1, 1, 0, 0])
    exp = np.busday_count("2020-01-01", "2020-01-08", weekmask=[1, 1, 1, 1, 1, 0, 0])
    assert int(got) == int(exp)


def test_busday_count_array():
    begin = ["2020-01-01", "2020-02-01"]
    end = ["2020-01-08", "2020-02-08"]
    np.testing.assert_array_equal(
        fr.busday_count(begin, end), np.busday_count(begin, end)
    )


# ---------------------------------------------------------------------------
# busday_offset
# ---------------------------------------------------------------------------

def test_busday_offset_scalar():
    assert fr.busday_offset("2020-01-01", 2) == np.busday_offset("2020-01-01", 2)


def test_busday_offset_negative():
    assert fr.busday_offset("2020-01-08", -2) == np.busday_offset("2020-01-08", -2)


@pytest.mark.parametrize("roll", ["forward", "following", "backward", "preceding"])
def test_busday_offset_roll(roll):
    # 2020-01-04 is a Saturday — roll modes diverge here.
    assert fr.busday_offset("2020-01-04", 0, roll=roll) == np.busday_offset(
        "2020-01-04", 0, roll=roll
    )


def test_busday_offset_roll_forward_then_step():
    assert fr.busday_offset("2020-01-04", 1, roll="forward") == np.busday_offset(
        "2020-01-04", 1, roll="forward"
    )


def test_busday_offset_raise_on_nonbusday():
    with pytest.raises(ValueError):
        fr.busday_offset("2020-01-04", 0, roll="raise")


def test_busday_offset_array():
    dates = ["2020-01-01", "2020-01-06"]
    offsets = [1, 2]
    np.testing.assert_array_equal(
        fr.busday_offset(dates, offsets), np.busday_offset(dates, offsets)
    )


def test_busday_offset_holidays():
    got = fr.busday_offset("2020-01-01", 1, holidays=["2020-01-02"])
    exp = np.busday_offset("2020-01-01", 1, holidays=["2020-01-02"])
    assert got == exp
