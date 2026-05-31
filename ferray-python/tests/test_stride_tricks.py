"""Phase-4 parity tests for ferray.lib.stride_tricks."""

import numpy as np

import ferray


def test_broadcast_shapes_simple():
    assert ferray.broadcast_shapes((3, 1), (1, 4)) == (3, 4)


def test_broadcast_shapes_three_args():
    assert ferray.broadcast_shapes((3, 1), (1, 4), (3, 4)) == (3, 4)


def test_broadcast_shapes_with_int_arg():
    # NumPy accepts a bare int as a 1-D shape.
    assert ferray.broadcast_shapes(5, (1,)) == (5,)


def test_broadcast_arrays_basic():
    a = np.array([1, 2, 3])
    b = np.array([[10], [20], [30]])
    out = ferray.broadcast_arrays(a, b)
    expected = np.broadcast_arrays(a, b)
    assert len(out) == len(expected)
    for got, exp in zip(out, expected):
        assert np.array_equal(got, exp)


def test_broadcast_arrays_three_inputs():
    a = np.array([[1.0, 2.0]])
    b = np.array([[3.0], [4.0]])
    c = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = ferray.broadcast_arrays(a, b, c)
    expected = np.broadcast_arrays(a, b, c)
    for got, exp in zip(out, expected):
        np.testing.assert_array_equal(got, exp)


def test_sliding_window_view_1d():
    src = np.arange(10)
    out = ferray.lib.stride_tricks.sliding_window_view(src, 3)
    expected = np.lib.stride_tricks.sliding_window_view(src, 3)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_sliding_window_view_2d():
    src = np.arange(20).reshape(4, 5)
    out = ferray.lib.stride_tricks.sliding_window_view(src, (2, 2))
    expected = np.lib.stride_tricks.sliding_window_view(src, (2, 2))
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_sliding_window_view_dtype_preserved():
    src = np.arange(10, dtype=np.int32)
    out = ferray.lib.stride_tricks.sliding_window_view(src, 3)
    assert out.dtype == np.int32
