"""Adversarial divergence tests: ferray manipulation surface vs NumPy 2.4 oracle.

Every test computes the numpy-oracle value/exception LIVE and asserts ferray
matches (value + dtype + shape, or matching exception TYPE). These pin
behaviours where ferray's `manipulation.rs` bindings diverge from numpy.

Cites are to the numpy working tree at /home/doll/numpy-ref.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Negative-axis / negative-shape support
#
# numpy normalizes negative axes/dims via `normalize_axis_index`. ferray's
# bindings extract `usize`/`Vec<usize>` (manipulation.rs: e.g. `reshape`,
# `transpose`, `swapaxes`, `moveaxis`, `roll`, `squeeze`, `flip`), so any
# negative value raises a bare Python OverflowError instead of working.
# ---------------------------------------------------------------------------


def test_reshape_minus_one_inference():
    """numpy/_core/fromnumeric.py:208 `def reshape(a, /, shape, ...)` — a -1
    entry is inferred from the array size. ferray raises OverflowError."""
    src = np.arange(12)
    expected = np.reshape(src, (-1, 4))
    out = fr.reshape(src, (-1, 4))
    assert out.shape == expected.shape == (3, 4)
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected)


def test_transpose_negative_axes():
    """numpy/_core/fromnumeric.py:605 `def transpose(a, axes=None)` — explicit
    axes may be negative and are normalized. ferray raises OverflowError."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.transpose(src, (-1, 0, 1))
    out = fr.transpose(src, axes=(-1, 0, 1))
    assert out.shape == expected.shape == (4, 2, 3)
    assert np.array_equal(out, expected)


def test_swapaxes_negative_axis():
    """numpy swapaxes normalizes negative axis indices. ferray raises
    OverflowError. Oracle: np.swapaxes(x, -1, 0)."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.swapaxes(src, -1, 0)
    out = fr.swapaxes(src, -1, 0)
    assert out.shape == expected.shape == (4, 3, 2)
    assert np.array_equal(out, expected)


def test_moveaxis_negative_source():
    """numpy/lib/_function_base_impl moveaxis normalizes negative source/dest.
    ferray raises OverflowError. Oracle: np.moveaxis(x, -1, 0)."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.moveaxis(src, -1, 0)
    out = fr.moveaxis(src, -1, 0)
    assert out.shape == expected.shape == (4, 2, 3)
    assert np.array_equal(out, expected)


def test_roll_negative_axis():
    """numpy roll accepts negative axis. ferray raises OverflowError.
    Oracle: np.roll(x, 1, axis=-1)."""
    src = np.arange(12).reshape(3, 4)
    expected = np.roll(src, 1, axis=-1)
    out = fr.roll(src, 1, axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_squeeze_negative_axis():
    """numpy/_core/fromnumeric.py:1597 `def squeeze(a, axis=None)` — axis may be
    negative. ferray raises OverflowError. Oracle: np.squeeze(zeros((3,1)),-1)."""
    src = np.zeros((3, 1))
    expected = np.squeeze(src, axis=-1)
    out = fr.squeeze(src, axis=-1)
    assert out.shape == expected.shape == (3,)


def test_repeat_negative_axis():
    """numpy/_core/fromnumeric.py:438 `def repeat(a, repeats, axis=None)` —
    negative axis is normalized. ferray raises OverflowError."""
    src = np.array([[1, 2], [3, 4]])
    expected = np.repeat(src, 2, axis=-1)
    out = fr.repeat(src, 2, axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_concatenate_negative_axis():
    """numpy/_core/multiarray.py:198 concatenate accepts negative axis. ferray
    raises OverflowError. Oracle: np.concatenate([a,b], axis=-1)."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])
    expected = np.concatenate([a, b], axis=-1)
    out = fr.concatenate([a, b], axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# expand_dims contract
# ---------------------------------------------------------------------------


def test_expand_dims_negative_axis():
    """numpy/lib/_shape_base_impl.py:514 `def expand_dims(a, axis)` — negative
    axis inserts at the position counted from the end. ferray raises
    OverflowError. Oracle: np.expand_dims([1,2,3], -1).shape == (3, 1)."""
    src = np.array([1, 2, 3])
    expected = np.expand_dims(src, -1)
    out = fr.expand_dims(src, -1)
    assert out.shape == expected.shape == (3, 1)


def test_expand_dims_tuple_axis():
    """numpy/lib/_shape_base_impl.py:514 expand_dims docstring: "``axis`` may
    also be a tuple". ferray rejects tuple axis with TypeError.
    Oracle: np.expand_dims([1,2], (0,2)).shape == (1, 2, 1)."""
    src = np.array([1, 2])
    expected = np.expand_dims(src, (0, 2))
    out = fr.expand_dims(src, (0, 2))
    assert out.shape == expected.shape == (1, 2, 1)


def test_expand_dims_out_of_bounds_is_axiserror():
    """expand_dims with axis beyond ndim raises numpy.exceptions.AxisError,
    not a plain ValueError. numpy/lib/_shape_base_impl.py:514."""
    src = np.array([1, 2, 3])
    with pytest.raises(np.exceptions.AxisError):
        fr.expand_dims(src, 5)


# ---------------------------------------------------------------------------
# Exception-TYPE contract: out-of-bounds axis must be AxisError
# ---------------------------------------------------------------------------


def test_concatenate_axis_oob_is_axiserror():
    """numpy/_core/multiarray.py:198 concatenate — an out-of-bounds axis raises
    numpy.exceptions.AxisError. ferray raises plain ValueError."""
    a = np.array([1, 2])
    b = np.array([3, 4])
    with pytest.raises(np.exceptions.AxisError):
        fr.concatenate([a, b], axis=5)


def test_flip_axis_oob_is_axiserror():
    """numpy/lib/_function_base_impl.py:284 `def flip(m, axis=None)` — an
    out-of-bounds axis raises numpy.exceptions.AxisError. ferray raises
    plain ValueError."""
    src = np.arange(6).reshape(2, 3)
    with pytest.raises(np.exceptions.AxisError):
        fr.flip(src, 5)


# ---------------------------------------------------------------------------
# flip default axis
# ---------------------------------------------------------------------------


def test_flip_default_axis_reverses_all():
    """numpy/lib/_function_base_impl.py:284 `def flip(m, axis=None)` — with no
    axis, flip reverses along every axis. ferray makes axis a required
    positional argument and raises TypeError."""
    src = np.arange(6).reshape(2, 3)
    expected = np.flip(src)
    out = fr.flip(src)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# repeat with per-element counts
# ---------------------------------------------------------------------------


def test_repeat_per_element_counts():
    """numpy/_core/fromnumeric.py:438 `def repeat(a, repeats, axis=None)` —
    `repeats` may be an array_like of per-element counts. ferray binds
    `repeats: usize` and raises TypeError for a list.
    Oracle: np.repeat([1,2,3], [1,2,3]) == [1,2,2,3,3,3]."""
    src = np.array([1, 2, 3])
    expected = np.repeat(src, [1, 2, 3])
    out = fr.repeat(src, [1, 2, 3])
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# reshape order='F'
# ---------------------------------------------------------------------------


def test_reshape_order_f():
    """numpy/_core/fromnumeric.py:208,221 `def reshape(a, /, shape, order='C',
    ...)` with `order : {'C', 'F', 'A'}`. ferray's binding has no `order`
    kwarg and raises TypeError. Oracle: column-major reshape."""
    src = np.arange(6)
    expected = np.reshape(src, (2, 3), order="F")
    out = fr.reshape(src, (2, 3), order="F")
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)
