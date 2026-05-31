"""REQ-2 (#942, epic #940): timedelta64 numeric arithmetic + datetime raise.

Verifies ferray's `fr.multiply` / `fr.divide` / `fr.true_divide` /
`fr.floor_divide` / `fr.remainder` / `fr.mod` over datetime64 / timedelta64
operands match numpy 2.4.5 EXACTLY (R-CHAR-3: every expected value is produced
by the live numpy oracle in this file, never literal-copied from ferray):

  timedelta * int/float -> timedelta      (and the reflected int/float * td)
  timedelta / int/float -> timedelta      (trunc toward zero)
  timedelta / timedelta -> float64        (ratio, finer common unit)
  timedelta // int/float -> timedelta     (numpy floor_divide == divide loop: trunc)
  timedelta // timedelta -> int64         (true floor)
  timedelta % timedelta -> timedelta      (Python floor-mod, sign follows divisor)
  datetime + datetime / datetime * x / datetime / x / td * td / int / td / td % int
                                          -> RAISE numpy.exceptions UFuncTypeError

NaT (i64::MIN) and zero-divisor edges match numpy's tick-level behaviour.
The real/complex numeric paths are unchanged.
"""

import numpy as np
import pytest

import ferray as fr

# numpy's UFuncTypeError lives in numpy._core._exceptions and subclasses
# TypeError. The exact type is obtained from a live raise so the test pins the
# real numpy exception family (R-DEV-2), not a hand-written class reference.
def _numpy_ufunc_type_error():
    d = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    try:
        np.add(d, d)
    except TypeError as exc:  # UFuncTypeError IS a TypeError subclass
        return type(exc)
    return TypeError  # pragma: no cover - numpy must raise here


_UFUNC_TYPE_ERROR = _numpy_ufunc_type_error()


def _td(vals, unit="D"):
    return np.array(vals, dtype=f"timedelta64[{unit}]")


def _dt(vals, unit="D"):
    return np.array(vals, dtype=f"datetime64[{unit}]")


def _assert_same(fr_out, np_out):
    """ferray output equals numpy: identical dtype + bit-identical values,
    treating NaT and NaN as matching (numpy != on NaT/NaN is False)."""
    fr_out = np.asarray(fr_out)
    np_out = np.asarray(np_out)
    assert fr_out.dtype == np_out.dtype, f"{fr_out.dtype} != {np_out.dtype}"
    assert fr_out.shape == np_out.shape
    # Compare on the raw bits so NaT/NaN positions line up exactly.
    if np_out.dtype.kind in ("M", "m"):
        assert (fr_out.view("int64") == np_out.view("int64")).all()
    elif np_out.dtype.kind == "f":
        a = fr_out.astype("float64")
        b = np_out.astype("float64")
        assert np.array_equal(a, b, equal_nan=True)
    else:
        assert (fr_out == np_out).all()


# ---------------------------------------------------------------------------
# multiply: td * int/float, reflected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [3, 2, 1, 0, -2])
def test_timedelta_times_int(k):
    td = _td([6, 5, 4])
    _assert_same(fr.multiply(td, k), np.multiply(td, k))
    # reflected
    _assert_same(fr.multiply(k, td), np.multiply(k, td))


@pytest.mark.parametrize("k", [1.5, 2.5, 0.5, -2.9, 2.4, 2.6])
def test_timedelta_times_float_truncates(k):
    # numpy casts the double product to int64 = TRUNC toward zero.
    td = _td([5, 3, 1])
    _assert_same(fr.multiply(td, k), np.multiply(td, k))
    _assert_same(fr.multiply(k, td), np.multiply(k, td))


def test_timedelta_times_int_array():
    td = _td([6, 5, 4])
    ks = np.array([1, 2, 3], dtype="int64")
    _assert_same(fr.multiply(td, ks), np.multiply(td, ks))


def test_timedelta_times_int_scalar_value():
    # A SCALAR timedelta input: the VALUE + dtype match numpy. (The scalar-vs-0d
    # RETURN shape is a marshalling concern shared with the already-SHIPPED ±
    # time path — `fr.add`/`fr.subtract` on scalars also return shape `(1,)`
    # where numpy returns `()`; that cross-op divergence is filed separately and
    # is out of REQ-2's arithmetic-algebra scope.)
    td = np.timedelta64(6, "D")
    out = np.asarray(fr.multiply(td, 3))
    expected = np.asarray(np.multiply(td, 3))
    assert out.dtype == expected.dtype
    assert out.view("int64").ravel().tolist() == expected.view("int64").ravel().tolist()


# ---------------------------------------------------------------------------
# divide: td / int/float -> td; td / td -> float64
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 1, 4])
def test_timedelta_div_int(k):
    td = _td([6, 7, -7])
    _assert_same(fr.divide(td, k), np.divide(td, k))
    _assert_same(fr.true_divide(td, k), np.true_divide(td, k))


@pytest.mark.parametrize("k", [2.0, 2.5, 1.5])
def test_timedelta_div_float(k):
    td = _td([5, 7, 6])
    _assert_same(fr.divide(td, k), np.divide(td, k))


def test_timedelta_div_zero_int_is_nat():
    td = _td([6, 5])
    _assert_same(fr.divide(td, 0), np.divide(td, 0))


def test_timedelta_div_timedelta_is_float():
    a = _td([6, 8, 3])
    b = _td([2, 2, 2])
    out = fr.divide(a, b)
    _assert_same(out, np.divide(a, b))
    assert np.asarray(out).dtype == np.dtype("float64")


def test_timedelta_div_timedelta_cross_unit():
    # td[D] / td[h]: numpy promotes to the finer unit (h) -> float ratio.
    a = _td([1, 2], "D")
    b = _td([12, 6], "h")
    _assert_same(fr.divide(a, b), np.divide(a, b))


def test_timedelta_div_timedelta_zero_is_inf():
    a = _td([5])
    b = _td([0])
    _assert_same(fr.divide(a, b), np.divide(a, b))


# ---------------------------------------------------------------------------
# floor_divide: td // int/float -> td (trunc); td // td -> int64 (floor)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, -2])
def test_timedelta_floordiv_int(k):
    td = _td([7, -7, 8])
    _assert_same(fr.floor_divide(td, k), np.floor_divide(td, k))


@pytest.mark.parametrize("k", [2.0, -2.0, 3.0])
def test_timedelta_floordiv_float_truncates(k):
    # numpy floor_divide is #define'd to the divide loop for td // scalar -> trunc.
    td = _td([7, -7, 8])
    _assert_same(fr.floor_divide(td, k), np.floor_divide(td, k))


def test_timedelta_floordiv_timedelta_is_int_floor():
    a = _td([7, -7, 8])
    b = _td([2, 2, 3])
    out = fr.floor_divide(a, b)
    _assert_same(out, np.floor_divide(a, b))
    assert np.asarray(out).dtype == np.dtype("int64")


def test_timedelta_floordiv_timedelta_cross_unit():
    a = _td([1], "D")
    b = _td([5], "h")
    _assert_same(fr.floor_divide(a, b), np.floor_divide(a, b))


# ---------------------------------------------------------------------------
# remainder / mod: td % td -> td (Python floor-mod)
# ---------------------------------------------------------------------------

def test_timedelta_mod_timedelta():
    a = _td([7, 6, 5])
    b = _td([4, 4, 2])
    _assert_same(fr.remainder(a, b), np.remainder(a, b))
    _assert_same(fr.mod(a, b), np.mod(a, b))


def test_timedelta_mod_negative_sign_follows_divisor():
    # Python floor-mod: -7 % 2 == 1, 7 % -2 == -1 (sign follows divisor).
    a = _td([-7, 7])
    b = _td([2, -2])
    _assert_same(fr.remainder(a, b), np.remainder(a, b))


def test_timedelta_mod_cross_unit():
    a = _td([1], "D")
    b = _td([5], "h")
    _assert_same(fr.remainder(a, b), np.remainder(a, b))


def test_timedelta_mod_zero_is_nat():
    a = _td([5])
    b = _td([0])
    _assert_same(fr.remainder(a, b), np.remainder(a, b))


# ---------------------------------------------------------------------------
# NaT propagation across the numeric ops
# ---------------------------------------------------------------------------

def test_nat_propagation():
    nat = _td(["NaT", 6])
    _assert_same(fr.multiply(nat, 3), np.multiply(nat, 3))
    _assert_same(fr.divide(nat, 2), np.divide(nat, 2))
    other = _td([2, 2])
    _assert_same(fr.divide(nat, other), np.divide(nat, other))  # -> nan / value
    _assert_same(fr.floor_divide(nat, other), np.floor_divide(nat, other))  # -> 0
    _assert_same(fr.remainder(nat, other), np.remainder(nat, other))  # -> NaT


# ---------------------------------------------------------------------------
# datetime / undefined-pair RAISES numpy's exact UFuncTypeError family
# ---------------------------------------------------------------------------

def test_datetime_plus_datetime_raises():
    dt = _dt(["2020-01-01", "2020-01-05"])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.add(dt, dt)


def test_datetime_times_scalar_raises():
    dt = _dt(["2020-01-01", "2020-01-05"])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.multiply(dt, 2)


def test_datetime_times_timedelta_raises():
    dt = _dt(["2020-01-01"])
    td = _td([2])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.multiply(dt, td)


def test_datetime_div_scalar_raises():
    dt = _dt(["2020-01-01"])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.divide(dt, 2)


def test_timedelta_times_timedelta_raises():
    a = _td([2])
    b = _td([3])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.multiply(a, b)


def test_int_div_timedelta_raises():
    td = _td([2])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.divide(2, td)


def test_timedelta_mod_int_raises():
    td = _td([7])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.remainder(td, 2)


def test_datetime_floordiv_scalar_raises():
    dt = _dt(["2020-01-01"])
    with pytest.raises(_UFUNC_TYPE_ERROR):
        fr.floor_divide(dt, 2)


# ---------------------------------------------------------------------------
# real path unchanged
# ---------------------------------------------------------------------------

def test_real_multiply_divide_unchanged():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    _assert_same(fr.multiply(a, b), np.multiply(a, b))
    _assert_same(fr.divide(a, b), np.divide(a, b))
    ia = np.array([7, -7, 8], dtype="int64")
    ib = np.array([2, 2, 3], dtype="int64")
    _assert_same(fr.floor_divide(ia, ib), np.floor_divide(ia, ib))
    _assert_same(fr.remainder(ia, ib), np.remainder(ia, ib))


def test_complex_multiply_divide_unchanged():
    a = np.array([1 + 2j, 3 - 1j])
    b = np.array([2 + 0j, 1 + 1j])
    _assert_same(fr.multiply(a, b), np.multiply(a, b))
    _assert_same(fr.divide(a, b), np.divide(a, b))
