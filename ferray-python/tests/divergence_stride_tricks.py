"""Adversarial divergence tests: ferray vs numpy for stride_tricks.

Audit target: ferray-python/src/stride_tricks.rs
Oracle: numpy 2.4.x (live). Every expected value is taken from a LIVE numpy
call (R-CHAR-3 — never literal-copied from the ferray side). Each test asserts
the numpy-oracle contract (value, dtype, shape, return-type, writeable flag,
kwarg surface, exception type) that ferray currently fails.

Upstream cites reference
/home/doll/numpy-ref/numpy/lib/_stride_tricks_impl.py.
"""

import numpy as np
import pytest

import ferray as fr

swv_np = np.lib.stride_tricks.sliding_window_view
swv_fr = fr.lib.stride_tricks.sliding_window_view


# ---------------------------------------------------------------------------
# broadcast_arrays — return type / dtype / kwargs
# ---------------------------------------------------------------------------

def test_broadcast_arrays_returns_tuple():
    """numpy broadcast_arrays returns a tuple, not a list.

    _stride_tricks_impl.py:656  `return tuple(result)`
    docstring (line 604): "broadcasted : tuple of arrays".
    ferray returns a Python list.
    """
    a = np.array([1, 2, 3])
    b = np.array([[10], [20], [30]])
    exp = np.broadcast_arrays(a, b)
    out = fr.broadcast_arrays(a, b)
    assert isinstance(exp, tuple)  # oracle: numpy contract
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"


def test_broadcast_arrays_single_returns_tuple():
    """numpy broadcast_arrays(a) (single arg) still returns a tuple.

    _stride_tricks_impl.py:656 `return tuple(result)`.
    ferray returns a list.
    """
    a = np.array([1, 2, 3])
    exp = np.broadcast_arrays(a)
    out = fr.broadcast_arrays(a)
    assert isinstance(exp, tuple)
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"


def test_broadcast_arrays_empty_returns_tuple():
    """numpy broadcast_arrays() with no args returns the empty tuple ().

    _stride_tricks_impl.py:656 `return tuple(result)` — for an empty arg
    list `result` is `[]`, so the return is `()`. ferray returns `[]`.
    """
    exp = np.broadcast_arrays()
    out = fr.broadcast_arrays()
    assert exp == ()  # oracle: numpy contract
    assert isinstance(out, tuple), f"expected tuple, got {type(out).__name__}"
    assert out == exp


def test_broadcast_arrays_preserves_per_array_dtype():
    """numpy broadcast_arrays preserves each input's own dtype/values.

    _stride_tricks_impl.py:649 `args = [np.array(_m, copy=None, ...) ...]`
    — each array keeps its original dtype; broadcasting changes only shape
    (lines 653-655). ferray coerces every array to the FIRST input's dtype,
    corrupting dtype AND values (1.5 -> 1).
    """
    a = np.array([1], dtype=np.int64)
    b = np.array([1.5, 2.5], dtype=np.float64)
    exp = np.broadcast_arrays(a, b)
    got = fr.broadcast_arrays(a, b)
    assert got[0].dtype == exp[0].dtype  # int64, from oracle
    assert got[1].dtype == exp[1].dtype  # float64, from oracle
    np.testing.assert_array_equal(got[1], exp[1])  # values un-truncated


def test_broadcast_arrays_accepts_subok_kwarg():
    """numpy broadcast_arrays has a `subok` keyword.

    _stride_tricks_impl.py:589 `def broadcast_arrays(*args, subok=False)`.
    Calling with subok=False must succeed (return matches the no-kwarg form).
    ferray's binding has no `subok` parameter -> raises TypeError.
    """
    a = np.array([1, 2, 3])
    b = np.array([[10], [20], [30]])
    exp = np.broadcast_arrays(a, b, subok=False)
    out = fr.broadcast_arrays(a, b, subok=False)
    for g, e in zip(out, exp):
        np.testing.assert_array_equal(g, e)


# ---------------------------------------------------------------------------
# broadcast_to — readonly + subok kwarg
# ---------------------------------------------------------------------------

def test_broadcast_to_is_readonly():
    """numpy broadcast_to returns a READONLY view.

    _stride_tricks_impl.py:517 `return _broadcast_to(array, shape,
    subok=subok, readonly=True)`; docstring (lines 491-494): "A readonly
    view on the original array". ferray returns a writeable array.
    """
    x = np.array([1, 2, 3])
    exp = np.broadcast_to(x, (3, 3))
    out = fr.broadcast_to(x, (3, 3))
    assert exp.flags.writeable is False  # oracle: numpy contract
    assert out.flags.writeable == exp.flags.writeable


def test_broadcast_to_accepts_subok_kwarg():
    """numpy broadcast_to has a `subok` keyword.

    _stride_tricks_impl.py:475 `def broadcast_to(array, shape, subok=False)`.
    ferray's binding has no `subok` parameter -> raises TypeError.
    """
    x = np.array([1, 2, 3])
    exp = np.broadcast_to(x, (3, 3), subok=False)
    out = fr.broadcast_to(x, (3, 3), subok=False)
    np.testing.assert_array_equal(out, exp)


# ---------------------------------------------------------------------------
# sliding_window_view — axis / writeable / subok / readonly / negative window
# ---------------------------------------------------------------------------

def test_sliding_window_view_axis_kwarg():
    """numpy sliding_window_view accepts an `axis` argument.

    _stride_tricks_impl.py:180 `def sliding_window_view(x, window_shape,
    axis=None, *, subok=False, writeable=False)`. With axis=0 on a (3,4)
    array and window 3, the output shape is (1, 4, 3) (lines 433/442).
    ferray's binding has no `axis` parameter -> raises TypeError.
    """
    x = np.arange(12).reshape(3, 4)
    exp = swv_np(x, 3, axis=0)
    out = swv_fr(x, 3, axis=0)
    assert exp.shape == (1, 4, 3)  # oracle
    assert out.shape == exp.shape
    np.testing.assert_array_equal(out, exp)


def test_sliding_window_view_duplicate_axis():
    """numpy allows the same axis windowed repeatedly (positional axis).

    _stride_tricks_impl.py:427 `normalize_axis_tuple(axis, x.ndim,
    allow_duplicate=True)`; lines 437-441 reduce the dim once per use.
    swv(x,(2,3),(1,1)) on a (3,4) array -> shape (3,1,2,3).
    ferray's binding takes only 2 positional args -> TypeError.
    """
    x = np.arange(12).reshape(3, 4)
    exp = swv_np(x, (2, 3), (1, 1))
    out = swv_fr(x, (2, 3), (1, 1))
    assert exp.shape == (3, 1, 2, 3)  # oracle
    assert out.shape == exp.shape
    np.testing.assert_array_equal(out, exp)


def test_sliding_window_view_is_readonly_by_default():
    """numpy sliding_window_view defaults writeable=False.

    _stride_tricks_impl.py:181 `writeable=False` default; the result is a
    read-only view (docstring lines 210-214, 387-391). ferray returns a
    writeable array.
    """
    x = np.arange(6)
    exp = swv_np(x, 3)
    out = swv_fr(x, 3)
    assert exp.flags.writeable is False  # oracle
    assert out.flags.writeable == exp.flags.writeable


def test_sliding_window_view_writeable_kwarg():
    """numpy sliding_window_view accepts a keyword-only `writeable`.

    _stride_tricks_impl.py:181 `*, subok=False, writeable=False`. With
    writeable=True the returned view is writeable. ferray's binding has no
    `writeable` parameter -> raises TypeError.
    """
    x = np.arange(6)
    exp = swv_np(x, 3, writeable=True)
    out = swv_fr(x, 3, writeable=True)
    assert exp.flags.writeable is True  # oracle
    assert out.flags.writeable == exp.flags.writeable


def test_sliding_window_view_subok_kwarg():
    """numpy sliding_window_view accepts a keyword-only `subok`.

    _stride_tricks_impl.py:181 `*, subok=False, writeable=False`.
    Calling with subok=False must succeed. ferray's binding has no `subok`
    parameter -> raises TypeError.
    """
    x = np.arange(6)
    exp = swv_np(x, 3, subok=False)
    out = swv_fr(x, 3, subok=False)
    np.testing.assert_array_equal(out, exp)


def test_sliding_window_view_negative_window_valueerror():
    """numpy raises ValueError for negative window sizes.

    _stride_tricks_impl.py:416-417 `if np.any(window_shape_array < 0):
    raise ValueError('`window_shape` cannot contain negative values')`.
    ferray raises OverflowError (negative int -> unsigned conversion) at the
    binding before the library validation can run.
    """
    x = np.arange(5)
    with pytest.raises(ValueError):
        swv_np(x, (-1,))  # oracle: confirms numpy's exception type
    with pytest.raises(ValueError):
        swv_fr(x, (-1,))


# ---------------------------------------------------------------------------
# as_strided — exposure
# ---------------------------------------------------------------------------

def test_as_strided_exists_and_works():
    """numpy exposes lib.stride_tricks.as_strided.

    _stride_tricks_impl.py:39 `def as_strided(x, shape=None, strides=None,
    ...)`. For x=arange(10), as_strided(x, shape=(5,), strides=(8,)) yields
    array([0,1,2,3,4]). ferray does not expose as_strided at all.
    """
    x = np.arange(10)
    exp = np.lib.stride_tricks.as_strided(x, shape=(5,), strides=(8,))
    assert hasattr(fr.lib.stride_tricks, "as_strided"), "as_strided not bound"
    out = fr.lib.stride_tricks.as_strided(x, shape=(5,), strides=(8,))
    assert out.shape == exp.shape
    np.testing.assert_array_equal(out, exp)
