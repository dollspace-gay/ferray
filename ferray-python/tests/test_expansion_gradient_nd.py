"""N-D / axis / varargs / edge_order gradient parity vs numpy (#966).

The prior `fr.gradient` binding accepted only a 1-D array (PyReadonlyArray1) +
a single `dx`, so `fr.gradient([[1.,2],[3,4]])` raised
`TypeError: 'ndarray' object is not an instance of 'ndarray'`. numpy supports
N-D input returning a tuple of per-axis central-difference gradients, plus
`*varargs` spacing, `axis` (int/tuple/None) and `edge_order` (1|2).

Every expected value is derived LIVE from `numpy.gradient` on the equivalent
numpy array (R-CHAR-3) — never literal-copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


def _eq(got, exp):
    """Compare a ferray gradient result to numpy's, array or tuple/list."""
    if isinstance(exp, (tuple, list)):
        assert isinstance(got, (tuple, list)), f"expected sequence, got {type(got)}"
        assert len(got) == len(exp)
        for g, e in zip(got, exp):
            np.testing.assert_allclose(np.asarray(g), np.asarray(e))
    else:
        assert not isinstance(got, (tuple, list)), f"expected array, got {type(got)}"
        np.testing.assert_allclose(np.asarray(got), np.asarray(exp))


# ---------------------------------------------------------------------------
# 1-D returns a single array (unchanged from the legacy binding)
# ---------------------------------------------------------------------------


def test_gradient_1d_array_not_tuple():
    a = [1.0, 2.0, 4.0]
    got = fr.gradient(fr.array(a))
    assert not isinstance(got, (tuple, list))
    _eq(got, np.gradient(np.array(a)))


def test_gradient_1d_scalar_dx():
    a = [1.0, 2.0, 4.0, 7.0]
    _eq(fr.gradient(fr.array(a), 2.0), np.gradient(np.array(a), 2.0))


def test_gradient_1d_coordinate_spacing():
    a = [1.0, 2.0, 4.0]
    x = [0.0, 1.0, 3.0]
    _eq(fr.gradient(fr.array(a), fr.array(x)), np.gradient(np.array(a), np.array(x)))


def test_gradient_1d_edge_order_2():
    a = [1.0, 2.0, 4.0, 7.0]
    _eq(fr.gradient(fr.array(a), edge_order=2), np.gradient(np.array(a), edge_order=2))


# ---------------------------------------------------------------------------
# N-D returns a tuple of per-axis gradients
# ---------------------------------------------------------------------------


def test_gradient_2d_returns_tuple_of_two():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    got = fr.gradient(fr.array(a))
    exp = np.gradient(np.array(a))
    assert isinstance(got, (tuple, list))
    assert len(got) == 2
    _eq(got, exp)


def test_gradient_3d_returns_tuple_of_three():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    got = fr.gradient(fr.array(a.tolist()))
    exp = np.gradient(a)
    assert len(got) == 3
    _eq(got, exp)


def test_gradient_2d_axis0():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    _eq(fr.gradient(fr.array(a), axis=0), np.gradient(np.array(a), axis=0))


def test_gradient_2d_axis1():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    _eq(fr.gradient(fr.array(a), axis=1), np.gradient(np.array(a), axis=1))


def test_gradient_2d_axis_tuple():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    _eq(fr.gradient(fr.array(a), axis=(0, 1)), np.gradient(np.array(a), axis=(0, 1)))


def test_gradient_2d_neg_axis():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    _eq(fr.gradient(fr.array(a), axis=-1), np.gradient(np.array(a), axis=-1))


def test_gradient_2d_scalar_dx():
    a = [[1.0, 2.0, 4.0], [3.0, 5.0, 9.0]]
    _eq(fr.gradient(fr.array(a), 2.0), np.gradient(np.array(a), 2.0))


def test_gradient_2d_edge_order_2():
    a = np.arange(12, dtype=float).reshape(3, 4) ** 2
    _eq(fr.gradient(fr.array(a.tolist()), edge_order=2), np.gradient(a, edge_order=2))


# ---------------------------------------------------------------------------
# dtype preservation
# ---------------------------------------------------------------------------


def test_gradient_float32_dtype_preserved():
    a = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    got = fr.gradient(fr.array(a.tolist(), dtype="float32"))
    exp = np.gradient(a)
    assert np.asarray(got).dtype == exp.dtype == np.float32
    _eq(got, exp)


def test_gradient_int_promotes_to_float64():
    a = [1, 2, 4, 8]
    got = fr.gradient(fr.array(a))
    exp = np.gradient(np.array(a))
    assert np.asarray(got).dtype == exp.dtype == np.float64
    _eq(got, exp)


def test_gradient_complex():
    a = [1 + 1j, 2 + 3j, 4 - 1j]
    got = fr.gradient(fr.array(a))
    exp = np.gradient(np.array(a))
    assert np.asarray(got).dtype == exp.dtype
    _eq(got, exp)


# ---------------------------------------------------------------------------
# datetime64 / timedelta64 -> timedelta64 (#946 stays green)
# ---------------------------------------------------------------------------


def test_gradient_timedelta_is_timedelta():
    td = fr.array([5, 2, 8, 1], dtype="timedelta64[s]")
    ntd = np.array([5, 2, 8, 1], dtype="timedelta64[s]")
    got = fr.gradient(td)
    exp = np.gradient(ntd)
    assert np.asarray(got).dtype.kind == "m"
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_gradient_datetime_is_timedelta():
    dt = fr.array(["2020-01-01", "2020-01-03", "2020-01-08"], dtype="datetime64[D]")
    ndt = np.array(["2020-01-01", "2020-01-03", "2020-01-08"], dtype="datetime64[D]")
    got = fr.gradient(dt)
    exp = np.gradient(ndt)
    assert np.asarray(got).dtype.kind == "m"
    np.testing.assert_array_equal(np.asarray(got), exp)


# ---------------------------------------------------------------------------
# error parity: array too small
# ---------------------------------------------------------------------------


def test_gradient_too_small_raises_like_numpy():
    with pytest.raises(ValueError):
        np.gradient(np.array([1.0]))
    with pytest.raises(ValueError):
        fr.gradient(fr.array([1.0]))
