"""Expansion tests (#941) — datetime64 / timedelta64 input coercion.

`fr.array` / `fr.asarray` / `fr.zeros` / `fr.ones` / `fr.empty` / `fr.full`
(and the `*_like` siblings) previously raised
`TypeError: unsupported dtype: datetime64[D]` on a `dtype='datetime64[unit]'` /
`'timedelta64[unit]'` request: the real-only `match_dtype_all!` /
`creation_dispatch!` dispatch macros had no `"M"`/`"m"` arms. numpy CONSTRUCTS
datetime64 / timedelta64 arrays for all of these.

The builder added an `"M"`/`"m"` branch ahead of each real-dtype macro that
routes construction through the existing int64-view transport (#831):
`crate::datetime::datetime_roundtrip` reads the unit straight from the dtype,
zero-copy `.view('int64')`s the buffer into a ferray `ArrayD<i64>`, then
reconstructs the typed array via `int64_to_datetime64` / `int64_to_timedelta64`
— preserving numpy's dtype + unit + shape + NaT with no lossy cast (R-CODE-4).

Every `expected` is derived from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. `fr.ndarray` IS numpy's ndarray, so `np.asarray(fr_arr)`
is the egress for value/dtype comparison.
"""

import numpy as np
import pytest

import ferray as fr

UNITS = ["Y", "M", "D", "h", "m", "s", "ms", "us", "ns"]


def _np(x):
    """A ferray result -> numpy ndarray for value/dtype comparison."""
    return np.asarray(x)


# ---------------------------------------------------------------------------
# fr.array from a string list + datetime64 dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", UNITS)
def test_array_string_list_datetime_units(unit):
    dt = f"datetime64[{unit}]"
    expected = np.array(["2020-01-01", "2020-01-05"], dtype=dt)  # live oracle
    got = _np(fr.array(["2020-01-01", "2020-01-05"], dtype=dt))
    assert str(got.dtype) == str(expected.dtype) == dt
    # Compare on the int64 ticks (exact, NaT-safe).
    assert np.array_equal(got.view("int64"), expected.view("int64"))


def test_array_datetime_default_example_values():
    # The pinned confirm case: values match numpy's parsed ticks.
    expected = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    got = _np(fr.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]"))
    assert np.array_equal(got, expected)
    assert str(got.dtype) == "datetime64[D]"


# ---------------------------------------------------------------------------
# fr.array / fr.asarray from a numpy datetime/timedelta array (unit preserved)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", UNITS)
def test_asarray_from_numpy_datetime_preserves_unit(unit):
    src = np.array(["2021-06-15", "1999-12-31"], dtype=f"datetime64[{unit}]")
    got = _np(fr.asarray(src))
    assert got.dtype == src.dtype  # unit preserved
    assert np.array_equal(got.view("int64"), src.view("int64"))


def test_array_from_numpy_datetime_array():
    src = np.array(["2020-01-01", "2020-01-05"], dtype="datetime64[D]")
    got = _np(fr.array(src))
    assert got.dtype == src.dtype
    assert np.array_equal(got, src)


def test_asarray_roundtrip_dtype_identity():
    # `fr.asarray(np_dt).dtype == np_dt.dtype` across units.
    for unit in UNITS:
        src = np.array(["2020-02-29"], dtype=f"datetime64[{unit}]")
        assert _np(fr.asarray(src)).dtype == src.dtype


# ---------------------------------------------------------------------------
# timedelta64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", UNITS)
def test_array_timedelta_units(unit):
    dt = f"timedelta64[{unit}]"
    expected = np.array([5, 7, 0], dtype=dt)  # live oracle
    got = _np(fr.array([5, 7, 0], dtype=dt))
    assert str(got.dtype) == dt
    assert np.array_equal(got, expected)


def test_array_timedelta_roundtrip_from_numpy():
    src = np.array([5], dtype="timedelta64[D]")
    got = _np(fr.array(src))
    assert got.dtype == src.dtype
    assert np.array_equal(got, src)


# ---------------------------------------------------------------------------
# NaT preserved through construction
# ---------------------------------------------------------------------------


def test_array_datetime_nat_preserved():
    expected = np.array(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]")
    got = _np(fr.array(["NaT", "2020-01-01", "NaT"], dtype="datetime64[D]"))
    assert str(got.dtype) == "datetime64[D]"
    # NaT == i64::MIN tick; compare ticks (NaT != NaT under value equality).
    assert np.array_equal(got.view("int64"), expected.view("int64"))
    assert bool(np.isnat(got[0])) and bool(np.isnat(got[2]))


def test_asarray_timedelta_nat_preserved():
    src = np.array(["NaT", 5], dtype="timedelta64[D]")
    got = _np(fr.asarray(src))
    assert np.array_equal(got.view("int64"), src.view("int64"))


# ---------------------------------------------------------------------------
# fr.zeros / ones / empty / full  (shape, dtype='datetime64[unit]')
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("unit", ["D", "s", "ns", "M"])
def test_zeros_datetime_is_epoch(unit):
    dt = f"datetime64[{unit}]"
    expected = np.zeros(3, dtype=dt)  # epoch / 0 ticks
    got = _np(fr.zeros(3, dtype=dt))
    assert str(got.dtype) == dt
    assert np.array_equal(got, expected)
    assert np.array_equal(got.view("int64"), np.zeros(3, dtype="int64"))


def test_zeros_datetime_seconds_matches_numpy_example():
    got = _np(fr.zeros(3, dtype="datetime64[s]"))
    assert np.array_equal(got, np.zeros(3, dtype="datetime64[s]"))


def test_ones_datetime_is_one_tick():
    # numpy `ones` fills tick 1 (NOT the epoch) for datetime64.
    expected = np.ones(2, dtype="datetime64[D]")
    got = _np(fr.ones(2, dtype="datetime64[D]"))
    assert str(got.dtype) == "datetime64[D]"
    assert np.array_equal(got.view("int64"), expected.view("int64"))


def test_empty_datetime_shape_and_dtype():
    got = _np(fr.empty(4, dtype="datetime64[h]"))
    assert got.shape == (4,)
    assert str(got.dtype) == "datetime64[h]"


@pytest.mark.parametrize("unit", ["D", "M", "s"])
def test_full_datetime_fill_string(unit):
    dt = f"datetime64[{unit}]"
    expected = np.full(2, "2021-06-15", dtype=dt)  # numpy parses the fill
    got = _np(fr.full(2, "2021-06-15", dtype=dt))
    assert str(got.dtype) == dt
    assert np.array_equal(got, expected)


def test_full_timedelta_fill():
    expected = np.full(3, 7, dtype="timedelta64[D]")
    got = _np(fr.full(3, 7, dtype="timedelta64[D]"))
    assert str(got.dtype) == "timedelta64[D]"
    assert np.array_equal(got, expected)


def test_zeros_timedelta():
    expected = np.zeros(3, dtype="timedelta64[s]")
    got = _np(fr.zeros(3, dtype="timedelta64[s]"))
    assert np.array_equal(got, expected)
    assert str(got.dtype) == "timedelta64[s]"


# ---------------------------------------------------------------------------
# *_like siblings preserve a datetime prototype's dtype
# ---------------------------------------------------------------------------


def test_zeros_like_datetime_prototype():
    proto = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    got = _np(fr.zeros_like(proto))
    assert str(got.dtype) == "datetime64[D]"
    assert np.array_equal(got, np.zeros_like(proto))


def test_full_like_datetime_prototype():
    proto = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    got = _np(fr.full_like(proto, "2022-03-03"))
    assert str(got.dtype) == "datetime64[D]"
    assert np.array_equal(got, np.full_like(proto, "2022-03-03"))


# ---------------------------------------------------------------------------
# Multi-dimensional shape preserved
# ---------------------------------------------------------------------------


def test_array_datetime_2d_shape():
    src = np.array(
        [["2020-01-01", "2020-01-02"], ["2020-01-03", "NaT"]],
        dtype="datetime64[D]",
    )
    got = _np(fr.asarray(src))
    assert got.shape == (2, 2)
    assert got.dtype == src.dtype
    assert np.array_equal(got.view("int64"), src.view("int64"))


def test_zeros_datetime_2d_shape():
    got = _np(fr.zeros((2, 3), dtype="datetime64[D]"))
    assert got.shape == (2, 3)
    assert np.array_equal(got, np.zeros((2, 3), dtype="datetime64[D]"))


# ---------------------------------------------------------------------------
# Real / other dtypes UNCHANGED (no regression from the new branch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dt", ["float64", "float32", "int64", "int32", "uint8", "bool"]
)
def test_real_dtypes_unchanged(dt):
    expected = np.array([1, 0, 1], dtype=dt)
    got = _np(fr.array([1, 0, 1], dtype=dt))
    assert str(got.dtype) == dt
    assert np.array_equal(got, expected)


def test_real_zeros_ones_full_unchanged():
    assert np.array_equal(_np(fr.zeros(3, dtype="int32")), np.zeros(3, dtype="int32"))
    assert np.array_equal(_np(fr.ones(2, dtype="float64")), np.ones(2, dtype="float64"))
    assert np.array_equal(_np(fr.full(2, 5, dtype="int64")), np.full(2, 5, dtype="int64"))


def test_complex_array_unchanged():
    z = np.array([1 + 2j, 3 - 4j], dtype="complex128")
    got = _np(fr.array(z.tolist()))
    assert np.iscomplexobj(got)
    assert np.array_equal(got, z)
