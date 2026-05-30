"""Adversarial divergence tests: ferray vs numpy for stride_tricks.

Audit target: ferray-python/src/stride_tricks.rs
Oracle: numpy 2.4.x (live). Each test asserts the numpy-oracle contract
(value, dtype, shape, AND exception type) that ferray currently fails.

Upstream cites reference /home/doll/numpy-ref/numpy/lib/_stride_tricks_impl.py
"""

import numpy as np
import pytest

import ferray as fr

swv_np = np.lib.stride_tricks.sliding_window_view
swv_fr = fr.lib.stride_tricks.sliding_window_view


# ---------------------------------------------------------------------------
# broadcast_arrays
# ---------------------------------------------------------------------------

def test_broadcast_arrays_returns_tuple():
    """numpy broadcast_arrays returns a tuple, not a list.

    _stride_tricks_impl.py:656  `return tuple(result)`
    The docstring (line 604) states "Returns: broadcasted : tuple of arrays".
    """
    a = np.array([1, 2, 3])
    b = np.array([[10], [20], [30]])
    out = fr.broadcast_arrays(a, b)
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"


def test_broadcast_arrays_empty_returns_tuple():
    """numpy broadcast_arrays() with no args returns an empty tuple ().

    _stride_tricks_impl.py:656  `return tuple(result)` — for an empty
    arg list `result` is `[]`, so the return is `()`.
    """
    out = fr.broadcast_arrays()
    expected = np.broadcast_arrays()
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"
    assert out == expected == ()


def test_broadcast_arrays_preserves_per_array_dtype():
    """numpy broadcast_arrays preserves each input's own dtype/values.

    _stride_tricks_impl.py:649  `args = [np.array(_m, copy=None, ...) ...]`
    — each array keeps its original dtype; broadcasting only changes shape
    (line 653-655 broadcasts shape, never dtype). ferray coerces every
    array to the FIRST input's dtype, corrupting both dtype and values.
    """
    a = np.array([1], dtype=np.int64)
    b = np.array([1.5, 2.5], dtype=np.float64)
    got = fr.broadcast_arrays(a, b)
    exp = np.broadcast_arrays(a, b)
    # dtype must be preserved per-array
    assert got[0].dtype == exp[0].dtype == np.int64
    assert got[1].dtype == exp[1].dtype == np.float64
    # values must not be truncated (1.5 -> 1 is corruption)
    np.testing.assert_array_equal(got[1], exp[1])


# ---------------------------------------------------------------------------
# broadcast_to
# ---------------------------------------------------------------------------

def test_broadcast_to_is_readonly():
    """numpy broadcast_to returns a READONLY view.

    _stride_tricks_impl.py:517  `return _broadcast_to(array, shape,
    subok=subok, readonly=True)`; docstring (line 491-494): "A readonly
    view on the original array". ferray returns a writeable array.
    """
    x = np.array([1, 2, 3])
    out = fr.broadcast_to(x, (3, 3))
    exp = np.broadcast_to(x, (3, 3))
    assert out.flags.writeable == exp.flags.writeable == False  # noqa: E712


# ---------------------------------------------------------------------------
# sliding_window_view
# ---------------------------------------------------------------------------

def test_sliding_window_view_axis_kwarg():
    """numpy sliding_window_view accepts an `axis` argument.

    _stride_tricks_impl.py:180  `def sliding_window_view(x, window_shape,
    axis=None, *, subok=False, writeable=False)`. With axis=0 on a (3,4)
    array and window 3, output shape is (1, 4, 3) (line 433/442).
    ferray's binding has no `axis` parameter -> raises TypeError.
    """
    x = np.arange(12).reshape(3, 4)
    out = swv_fr(x, 3, axis=0)
    exp = swv_np(x, 3, axis=0)
    assert out.shape == exp.shape == (1, 4, 3)
    np.testing.assert_array_equal(out, exp)


def test_sliding_window_view_duplicate_axis():
    """numpy allows the same axis windowed repeatedly via positional axis.

    _stride_tricks_impl.py:427 `normalize_axis_tuple(axis, x.ndim,
    allow_duplicate=True)`; line 437-441 reduces the dim once per use.
    swv(x,(2,3),(1,1)) on a (3,4) array -> shape (3,1,2,3).
    ferray's binding takes only 2 positional args.
    """
    x = np.arange(12).reshape(3, 4)
    out = swv_fr(x, (2, 3), (1, 1))
    exp = swv_np(x, (2, 3), (1, 1))
    assert out.shape == exp.shape == (3, 1, 2, 3)
    np.testing.assert_array_equal(out, exp)


def test_sliding_window_view_is_readonly_by_default():
    """numpy sliding_window_view default writeable=False.

    _stride_tricks_impl.py:181 `writeable=False` default; the result is a
    read-only view (docstring lines 210-214, 387-391). ferray returns a
    writeable array.
    """
    x = np.arange(6)
    out = swv_fr(x, 3)
    exp = swv_np(x, 3)
    assert out.flags.writeable == exp.flags.writeable == False  # noqa: E712


def test_sliding_window_view_negative_window_valueerror():
    """numpy raises ValueError for negative window sizes.

    _stride_tricks_impl.py:416-417  `if np.any(window_shape_array < 0):
    raise ValueError('`window_shape` cannot contain negative values')`.
    ferray raises OverflowError (cannot convert negative int to unsigned).
    """
    x = np.arange(5)
    with pytest.raises(ValueError):
        swv_fr(x, (-1,))


# ---------------------------------------------------------------------------
# as_strided
# ---------------------------------------------------------------------------

def test_as_strided_exists_and_works():
    """numpy exposes lib.stride_tricks.as_strided.

    _stride_tricks_impl.py:39 `def as_strided(x, shape=None, strides=None,
    ...)`. For x=arange(10), as_strided(x, shape=(5,), strides=(8,)) yields
    array([0,1,2,3,4]). ferray does not expose as_strided at all.
    """
    x = np.arange(10)
    assert hasattr(fr.lib.stride_tricks, "as_strided"), "as_strided not bound"
    out = fr.lib.stride_tricks.as_strided(x, shape=(5,), strides=(8,))
    exp = np.lib.stride_tricks.as_strided(x, shape=(5,), strides=(8,))
    assert out.shape == exp.shape == (5,)
    np.testing.assert_array_equal(out, exp)
