"""Tests for #719 — ferray.lib functional utilities (vectorize,
apply_along_axis, apply_over_axes, piecewise).

These delegate to numpy because they exist to apply user-provided
Python callables — there's no Rust kernel that would be faster.
The tests verify each one behaves identically to the numpy original.
"""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# vectorize
# ---------------------------------------------------------------------------


def test_vectorize_simple():
    f = ferray.lib.vectorize(lambda x: x * 2)
    np.testing.assert_array_equal(f(np.array([1, 2, 3])), np.array([2, 4, 6]))


def test_vectorize_two_arg():
    f = ferray.lib.vectorize(lambda x, y: x + y)
    out = f(np.array([1.0, 2.0]), np.array([10.0, 20.0]))
    np.testing.assert_array_equal(out, np.array([11.0, 22.0]))


def test_vectorize_with_explicit_otypes():
    f = ferray.lib.vectorize(lambda x: x ** 2, otypes=[np.float64])
    out = f(np.array([1, 2, 3]))
    assert out.dtype == np.float64


def test_vectorize_matches_numpy():
    fn = lambda x: x ** 3 - x
    src = np.linspace(-2.0, 2.0, 11)
    fr_out = ferray.lib.vectorize(fn)(src)
    np_out = np.vectorize(fn)(src)
    np.testing.assert_array_equal(fr_out, np_out)


# ---------------------------------------------------------------------------
# apply_along_axis
# ---------------------------------------------------------------------------


def test_apply_along_axis_sum_axis_0():
    arr = np.arange(12).reshape(3, 4)
    out = ferray.lib.apply_along_axis(np.sum, 0, arr)
    np.testing.assert_array_equal(out, np.array([12, 15, 18, 21]))


def test_apply_along_axis_sum_axis_1():
    arr = np.arange(12).reshape(3, 4)
    out = ferray.lib.apply_along_axis(np.sum, 1, arr)
    np.testing.assert_array_equal(out, np.array([6, 22, 38]))


def test_apply_along_axis_extra_args():
    """Extra args/kwargs are forwarded to the callable."""

    def add_const(arr, c):
        return arr + c

    src = np.arange(6).reshape(2, 3)
    out = ferray.lib.apply_along_axis(add_const, 0, src, 100)
    np_out = np.apply_along_axis(add_const, 0, src, 100)
    np.testing.assert_array_equal(out, np_out)


def test_apply_along_axis_returns_array():
    arr = np.arange(12).reshape(3, 4).astype(np.float64)

    def normalise(slice_):
        return slice_ / slice_.sum()

    out = ferray.lib.apply_along_axis(normalise, 1, arr)
    np_out = np.apply_along_axis(normalise, 1, arr)
    np.testing.assert_allclose(out, np_out)


# ---------------------------------------------------------------------------
# apply_over_axes
# ---------------------------------------------------------------------------


def test_apply_over_axes_two_axes():
    arr = np.arange(24).reshape(2, 3, 4)

    def mean_keepdims(a, axis):
        return np.mean(a, axis=axis, keepdims=True)

    out = ferray.lib.apply_over_axes(mean_keepdims, arr, [0, 2])
    np_out = np.apply_over_axes(mean_keepdims, arr, [0, 2])
    np.testing.assert_array_equal(out, np_out)


# ---------------------------------------------------------------------------
# piecewise
# ---------------------------------------------------------------------------


def test_piecewise_simple():
    x = np.linspace(-2.0, 2.0, 9)
    # Negative → -x; non-negative → x.
    out = ferray.lib.piecewise(x, [x < 0], [lambda v: -v, lambda v: v])
    np.testing.assert_allclose(out, np.abs(x))


def test_piecewise_three_pieces():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    out = ferray.lib.piecewise(
        x,
        [x < 0, x == 0, x > 0],
        [lambda v: -v, 0, lambda v: v ** 2],
    )
    np.testing.assert_allclose(out, [1.0, 0.0, 1.0, 4.0])


def test_piecewise_matches_numpy():
    x = np.linspace(-3.0, 3.0, 21)
    fr_out = ferray.lib.piecewise(x, [x < -1, x > 1], [-1.0, 1.0, 0.0])
    np_out = np.piecewise(x, [x < -1, x > 1], [-1.0, 1.0, 0.0])
    np.testing.assert_array_equal(fr_out, np_out)


# ---------------------------------------------------------------------------
# stride_tricks still accessible through the new Python lib package
# ---------------------------------------------------------------------------


def test_stride_tricks_still_accessible():
    out = ferray.lib.stride_tricks.broadcast_shapes((3, 1), (1, 4))
    assert out == (3, 4)


def test_lib_dunder_all_includes_stride_tricks():
    assert "stride_tricks" in ferray.lib.__all__
