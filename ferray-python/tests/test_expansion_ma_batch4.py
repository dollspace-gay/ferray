"""numpy.ma structured-mask + integer-bitwise batch 4 (refs #835 #818).

Verifies fr.ma's structured-dtype mask vocabulary (make_mask_descr,
flatten_mask, flatten_structured_array, fromflex, mvoid), the buffer-protocol
constructor (frombuffer), and the dtype-preserving integer masked bitwise
shifts (left_shift, right_shift) against numpy.ma directly.

Every expected value is derived from a LIVE numpy.ma call (R-CHAR-3) — no
literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# make_mask_descr — all-bool mask dtype mirroring a (structured) dtype
# ---------------------------------------------------------------------------

def test_make_mask_descr_plain():
    dt = np.dtype(np.float64)
    assert fr.ma.make_mask_descr(dt) == np.ma.make_mask_descr(dt)


def test_make_mask_descr_structured():
    dt = np.dtype([("a", int), ("b", float)])
    got = fr.ma.make_mask_descr(dt)
    exp = np.ma.make_mask_descr(dt)
    assert got == exp
    # the mirrored dtype is all-bool fields
    assert all(got[name] == np.dtype(bool) for name in got.names)


def test_make_mask_descr_nested():
    dt = np.dtype([("a", bool), ("b", [("ba", int), ("bb", float)])])
    assert fr.ma.make_mask_descr(dt) == np.ma.make_mask_descr(dt)


# ---------------------------------------------------------------------------
# flatten_mask — flatten a (possibly nested/structured) mask to flat bool
# ---------------------------------------------------------------------------

def test_flatten_mask_flat():
    m = np.array([False, True, False, True])
    got = fr.ma.flatten_mask(m)
    exp = np.ma.flatten_mask(m)
    np.testing.assert_array_equal(got, exp)
    assert got.dtype == exp.dtype == np.dtype(bool)


def test_flatten_mask_structured():
    m = np.array(
        [(0, (0, 0)), (0, (0, 1)), (1, (1, 0))],
        dtype=[("a", bool), ("b", [("ba", bool), ("bb", bool)])],
    )
    got = fr.ma.flatten_mask(m)
    exp = np.ma.flatten_mask(m)
    np.testing.assert_array_equal(got, exp)
    assert got.shape == exp.shape


# ---------------------------------------------------------------------------
# flatten_structured_array — flatten a structured array to 2-D / 1-D
# ---------------------------------------------------------------------------

def test_flatten_structured_array():
    a = np.array(
        [(1, (2, 3.0)), (4, (5, 6.0))],
        dtype=[("x", int), ("y", [("a", int), ("b", float)])],
    )
    got = fr.ma.flatten_structured_array(a)
    exp = np.ma.flatten_structured_array(a)
    np.testing.assert_array_equal(got, exp)
    assert got.shape == exp.shape
    assert got.dtype == exp.dtype


# ---------------------------------------------------------------------------
# fromflex — rebuild a MaskedArray from a flexible _data/_mask structured array
# ---------------------------------------------------------------------------

def test_fromflex_roundtrips_toflex():
    src = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 1])
    flex = src.toflex()  # structured array with _data / _mask fields
    got = fr.ma.fromflex(flex)
    exp = np.ma.fromflex(flex)
    np.testing.assert_array_equal(got.data, exp.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(exp))
    assert got.dtype == exp.dtype
    # round-trips the original masked data + mask
    np.testing.assert_array_equal(got.data, src.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(src))


def test_fromflex_int_dtype_preserved():
    src = np.ma.array([10, 20, 30], mask=[0, 0, 1], dtype=np.int32)
    flex = src.toflex()
    got = fr.ma.fromflex(flex)
    exp = np.ma.fromflex(flex)
    assert got.dtype == exp.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(got.data, exp.data)


# ---------------------------------------------------------------------------
# mvoid — scalar type for a single masked structured/record element
# ---------------------------------------------------------------------------

def test_mvoid_is_numpy_ma_mvoid():
    assert fr.ma.mvoid is np.ma.mvoid


def test_mvoid_constructs_masked_record_element():
    # A single masked structured element is an mvoid instance in numpy.ma.
    a = np.ma.array(
        [(1, 2.0)], mask=[(0, 1)], dtype=[("x", int), ("y", float)]
    )
    elem = a[0]
    assert isinstance(elem, fr.ma.mvoid)


# ---------------------------------------------------------------------------
# frombuffer — MaskedArray over a buffer-protocol object (no mask)
# ---------------------------------------------------------------------------

def test_frombuffer_default_float():
    buf = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64).tobytes()
    got = fr.ma.frombuffer(buf)
    exp = np.ma.frombuffer(buf)
    np.testing.assert_array_equal(got.data, exp.data)
    assert got.dtype == exp.dtype == np.dtype(np.float64)
    # no mask
    assert np.ma.getmaskarray(got).any() == np.ma.getmaskarray(exp).any() == False


def test_frombuffer_dtype_count_offset():
    buf = np.arange(8, dtype=np.int32).tobytes()
    got = fr.ma.frombuffer(buf, dtype=np.int32, count=4, offset=4)
    exp = np.ma.frombuffer(buf, dtype=np.int32, count=4, offset=4)
    np.testing.assert_array_equal(got.data, exp.data)
    assert got.dtype == exp.dtype == np.dtype(np.int32)


# ---------------------------------------------------------------------------
# left_shift / right_shift — masked integer bitwise shifts (dtype-preserving)
# ---------------------------------------------------------------------------

def test_left_shift_masked_int():
    a = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 0])
    got = fr.ma.left_shift(a, 2)
    exp = np.ma.left_shift(a, 2)
    np.testing.assert_array_equal(got.data, exp.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(exp))
    assert got.dtype == exp.dtype
    assert np.issubdtype(got.dtype, np.integer)


def test_right_shift_masked_int():
    a = np.ma.array([4, 8, 16, 7], mask=[0, 1, 0, 0])
    got = fr.ma.right_shift(a, 1)
    exp = np.ma.right_shift(a, 1)
    np.testing.assert_array_equal(got.data, exp.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(exp))
    assert got.dtype == exp.dtype


def test_left_shift_plain_int_ndarray():
    a = np.array([1, 2, 3, 4], dtype=np.int64)
    got = fr.ma.left_shift(a, 3)
    exp = np.ma.left_shift(a, 3)
    np.testing.assert_array_equal(got.data, exp.data)
    assert got.dtype == exp.dtype == np.dtype(np.int64)


def test_right_shift_dtype_int32_preserved():
    a = np.ma.array([8, 16, 32], mask=[0, 0, 1], dtype=np.int32)
    got = fr.ma.right_shift(a, 2)
    exp = np.ma.right_shift(a, 2)
    np.testing.assert_array_equal(got.data, exp.data)
    np.testing.assert_array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(exp))
    assert got.dtype == exp.dtype == np.dtype(np.int32)


# ---------------------------------------------------------------------------
# Registration smoke: all 8 names are present on fr.ma.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "make_mask_descr",
        "flatten_mask",
        "flatten_structured_array",
        "fromflex",
        "mvoid",
        "frombuffer",
        "left_shift",
        "right_shift",
    ],
)
def test_name_registered(name):
    assert hasattr(fr.ma, name)
