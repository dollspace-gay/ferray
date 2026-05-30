"""Adversarial divergence tests: ferray manipulation surface vs NumPy 2.4.5 oracle.

Re-authored from scratch by the acto-critic (supersedes the prior stand-in).
Every test computes the numpy-oracle value/exception LIVE (R-CHAR-3 — no
literal-copy of ferray's side) and asserts ferray matches value + dtype + shape,
or the matching exception TYPE. These pin behaviours where
`ferray-python/src/manipulation.rs` diverges from numpy.

Upstream cites are to the numpy working tree at /home/doll/numpy-ref.

Binding-layer divergences (fix in manipulation.rs / conv.rs):
  - negative-axis / negative-shape: bindings extract usize/Vec<usize>, so a
    negative int raises OverflowError instead of being normalized.
  - tuple/sequence axis params bound as a single usize.
  - missing kwargs (order=, axes=).
  - out-of-bounds axis mapped to plain ValueError, not numpy.exceptions.AxisError.
Library-layer divergence (fix down in ferray-core promotion):
  - concatenate/stack take the first array's dtype (collect_typed) instead of
    promoting across the whole list — silent truncation.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Negative-axis / negative-shape support (binding extracts usize → OverflowError)
# ---------------------------------------------------------------------------


def test_reshape_minus_one_inference_tuple():
    """numpy/_core/fromnumeric.py:208 `def reshape(a, /, shape, order='C', ...)`
    — a -1 entry in the shape tuple is inferred from the array size.
    ferray's `reshape` (manipulation.rs) extracts Vec<usize>, so -1 raises
    OverflowError."""
    src = np.arange(12)
    expected = np.reshape(src, (-1, 4))
    out = fr.reshape(src, (-1, 4))
    assert out.shape == expected.shape == (3, 4)
    assert out.dtype == expected.dtype
    assert np.array_equal(out, expected)


def test_reshape_minus_one_scalar():
    """numpy/_core/fromnumeric.py:208 reshape — a bare `-1` flattens.
    ferray's scalar branch extracts usize, so -1 raises TypeError/OverflowError.
    (New: prior pin only covered the tuple form.)"""
    src = np.arange(6).reshape(2, 3)
    expected = np.reshape(src, -1)
    out = fr.reshape(src, -1)
    assert out.shape == expected.shape == (6,)
    assert np.array_equal(out, expected)


def test_transpose_negative_axes():
    """numpy/_core/fromnumeric.py:605 `def transpose(a, axes=None)` — explicit
    axes may be negative and are normalized. ferray extracts Vec<usize> →
    OverflowError."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.transpose(src, (-1, 0, 1))
    out = fr.transpose(src, axes=(-1, 0, 1))
    assert out.shape == expected.shape == (4, 2, 3)
    assert np.array_equal(out, expected)


def test_swapaxes_negative_axis():
    """numpy/_core/fromnumeric.py:553 `def swapaxes(a, axis1, axis2)` normalizes
    negative axis indices. ferray extracts usize → OverflowError."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.swapaxes(src, -1, 0)
    out = fr.swapaxes(src, -1, 0)
    assert out.shape == expected.shape == (4, 3, 2)
    assert np.array_equal(out, expected)


def test_moveaxis_negative_source():
    """numpy/_core/numeric.py:1489 `def moveaxis(a, source, destination)`
    normalizes negative source/dest. ferray extracts usize → OverflowError."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.moveaxis(src, -1, 0)
    out = fr.moveaxis(src, -1, 0)
    assert out.shape == expected.shape == (4, 2, 3)
    assert np.array_equal(out, expected)


def test_moveaxis_sequence_source():
    """numpy/_core/numeric.py:1489 moveaxis — `source`/`destination` may be
    sequences of ints. ferray binds `source: usize` → TypeError for a list.
    (New.) Oracle: np.moveaxis(x, [0,1], [1,0]).shape == (3, 2, 4)."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.moveaxis(src, [0, 1], [1, 0])
    out = fr.moveaxis(src, [0, 1], [1, 0])
    assert out.shape == expected.shape == (3, 2, 4)
    assert np.array_equal(out, expected)


def test_roll_negative_axis():
    """numpy/_core/numeric.py:1226 `def roll(a, shift, axis=None)` accepts a
    negative axis. ferray extracts usize → OverflowError."""
    src = np.arange(12).reshape(3, 4)
    expected = np.roll(src, 1, axis=-1)
    out = fr.roll(src, 1, axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_squeeze_negative_axis():
    """numpy/_core/fromnumeric.py:1597 `def squeeze(a, axis=None)` — axis may be
    negative. ferray extracts Option<usize> → OverflowError."""
    src = np.zeros((3, 1))
    expected = np.squeeze(src, axis=-1)
    out = fr.squeeze(src, axis=-1)
    assert out.shape == expected.shape == (3,)


def test_squeeze_tuple_axis():
    """numpy/_core/fromnumeric.py:1597 squeeze — `axis` may be a tuple of axes.
    ferray binds `axis: Option<usize>` → TypeError for a tuple. (New.)
    Oracle: np.squeeze(zeros((1,3,1)), axis=(0,2)).shape == (3,)."""
    src = np.zeros((1, 3, 1))
    expected = np.squeeze(src, axis=(0, 2))
    out = fr.squeeze(src, axis=(0, 2))
    assert out.shape == expected.shape == (3,)


def test_repeat_negative_axis():
    """numpy/_core/fromnumeric.py:438 `def repeat(a, repeats, axis=None)` —
    negative axis is normalized. ferray extracts Option<usize> → OverflowError."""
    src = np.array([[1, 2], [3, 4]])
    expected = np.repeat(src, 2, axis=-1)
    out = fr.repeat(src, 2, axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_concatenate_negative_axis():
    """numpy/_core/multiarray.py:198 `def concatenate(arrays, axis=0, ...)`
    accepts a negative axis. ferray extracts usize → OverflowError."""
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])
    expected = np.concatenate([a, b], axis=-1)
    out = fr.concatenate([a, b], axis=-1)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_stack_negative_axis():
    """numpy/_core/shape_base.py:379 `def stack(arrays, axis=0, ...)` accepts a
    negative axis (normalized against result ndim). ferray extracts usize →
    OverflowError. (New.) Oracle: np.stack([a,b], axis=-1).shape == (3, 2)."""
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    expected = np.stack([a, b], axis=-1)
    out = fr.stack([a, b], axis=-1)
    assert out.shape == expected.shape == (3, 2)
    assert np.array_equal(out, expected)


def test_concatenate_axis_none_flattens():
    """numpy/_core/multiarray.py:198 concatenate — `axis=None` flattens all
    inputs before concatenating. ferray binds `axis: usize` → TypeError for
    None. (New.) Oracle: np.concatenate([[[1,2]],[[3,4]]], axis=None) == [1,2,3,4]."""
    a = np.array([[1, 2]])
    b = np.array([[3, 4]])
    expected = np.concatenate([a, b], axis=None)
    out = fr.concatenate([a, b], axis=None)
    assert out.shape == expected.shape == (4,)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# expand_dims contract
# ---------------------------------------------------------------------------


def test_expand_dims_negative_axis():
    """numpy/lib/_shape_base_impl.py:514 `def expand_dims(a, axis)` — negative
    axis inserts a length-1 axis counted from the end. ferray extracts usize →
    OverflowError. Oracle: np.expand_dims([1,2,3], -1).shape == (3, 1)."""
    src = np.array([1, 2, 3])
    expected = np.expand_dims(src, -1)
    out = fr.expand_dims(src, -1)
    assert out.shape == expected.shape == (3, 1)


def test_expand_dims_tuple_axis():
    """numpy/lib/_shape_base_impl.py:514 expand_dims — `axis` may be a tuple,
    inserting multiple length-1 axes. ferray binds `axis: usize` → TypeError.
    Oracle: np.expand_dims([1,2], (0,2)).shape == (1, 2, 1)."""
    src = np.array([1, 2])
    expected = np.expand_dims(src, (0, 2))
    out = fr.expand_dims(src, (0, 2))
    assert out.shape == expected.shape == (1, 2, 1)


def test_expand_dims_out_of_bounds_is_axiserror():
    """numpy/lib/_shape_base_impl.py:514 expand_dims with axis beyond ndim raises
    numpy.exceptions.AxisError. ferray raises plain builtins.ValueError, which is
    NOT an instance of AxisError."""
    src = np.array([1, 2, 3])
    with pytest.raises(np.exceptions.AxisError):
        fr.expand_dims(src, 5)


# ---------------------------------------------------------------------------
# Exception-TYPE contract: out-of-bounds axis must be numpy.exceptions.AxisError
# (AxisError subclasses ValueError, so asserting the subclass pins ferray's
#  use of a plain ValueError.)
# ---------------------------------------------------------------------------


def test_concatenate_axis_oob_is_axiserror():
    """numpy/_core/multiarray.py:198 concatenate — out-of-bounds axis raises
    numpy.exceptions.AxisError. ferray raises plain builtins.ValueError."""
    a = np.array([1, 2])
    b = np.array([3, 4])
    with pytest.raises(np.exceptions.AxisError):
        fr.concatenate([a, b], axis=5)


def test_stack_axis_oob_is_axiserror():
    """numpy/_core/shape_base.py:379 stack — out-of-bounds axis raises
    numpy.exceptions.AxisError. ferray raises plain builtins.ValueError. (New.)"""
    a = np.array([1, 2])
    b = np.array([3, 4])
    with pytest.raises(np.exceptions.AxisError):
        fr.stack([a, b], axis=5)


def test_flip_axis_oob_is_axiserror():
    """numpy/lib/_function_base_impl.py:284 `def flip(m, axis=None)` —
    out-of-bounds axis raises numpy.exceptions.AxisError. ferray raises plain
    builtins.ValueError."""
    src = np.arange(6).reshape(2, 3)
    with pytest.raises(np.exceptions.AxisError):
        fr.flip(src, 5)


# ---------------------------------------------------------------------------
# flip default axis
# ---------------------------------------------------------------------------


def test_flip_default_axis_reverses_all():
    """numpy/lib/_function_base_impl.py:284 `def flip(m, axis=None)` — with no
    axis, flip reverses along every axis. ferray makes axis a required
    positional argument → TypeError. Oracle: np.flip(arange(6).reshape(2,3))."""
    src = np.arange(6).reshape(2, 3)
    expected = np.flip(src)
    out = fr.flip(src)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# rot90 axes kwarg
# ---------------------------------------------------------------------------


def test_rot90_axes_kwarg():
    """numpy/lib/_function_base_impl.py:180 `def rot90(m, k=1, axes=(0, 1))` —
    `axes` selects the rotation plane. ferray's binding (signature (m, k=1))
    has no `axes` kwarg → TypeError. (New.)
    Oracle: np.rot90(arange(24).reshape(2,3,4), axes=(1,2)).shape == (2, 4, 3)."""
    src = np.arange(24).reshape(2, 3, 4)
    expected = np.rot90(src, axes=(1, 2))
    out = fr.rot90(src, axes=(1, 2))
    assert out.shape == expected.shape == (2, 4, 3)
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# repeat with per-element counts
# ---------------------------------------------------------------------------


def test_repeat_per_element_counts():
    """numpy/_core/fromnumeric.py:438 `def repeat(a, repeats, axis=None)` —
    `repeats` may be an array_like of per-element counts. ferray binds
    `repeats: usize` → TypeError for a list.
    Oracle: np.repeat([1,2,3], [1,2,3]) == [1,2,2,3,3,3]."""
    src = np.array([1, 2, 3])
    expected = np.repeat(src, [1, 2, 3])
    out = fr.repeat(src, [1, 2, 3])
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# delete obj: slice / boolean-mask forms
# ---------------------------------------------------------------------------


def test_delete_slice_obj():
    """numpy/lib/_function_base_impl.py:5221 `def delete(arr, obj, axis=None)` —
    `obj` may be a slice. ferray binds obj as int/Vec<usize> → TypeError for a
    slice. (New.) Oracle: np.delete(arange(10), slice(0,5,2), 0)."""
    src = np.arange(10)
    expected = np.delete(src, slice(0, 5, 2), 0)
    out = fr.delete(src, slice(0, 5, 2), 0)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


def test_delete_boolean_mask_obj():
    """numpy/lib/_function_base_impl.py:5221 delete — `obj` may be a boolean
    mask the same length as the axis. ferray coerces to usize → TypeError.
    (New.) Oracle: np.delete(arange(5), [T,F,T,F,T], 0) == [1, 3]."""
    src = np.arange(5)
    mask = np.array([True, False, True, False, True])
    expected = np.delete(src, mask, 0)
    out = fr.delete(src, mask, 0)
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# reshape order='F'
# ---------------------------------------------------------------------------


def test_reshape_order_f():
    """numpy/_core/fromnumeric.py:208 `def reshape(a, /, shape, order='C', ...)`
    with `order : {'C', 'F', 'A'}`. ferray's binding has no `order` kwarg →
    TypeError. Oracle: column-major reshape of arange(6) to (2,3)."""
    src = np.arange(6)
    expected = np.reshape(src, (2, 3), order="F")
    out = fr.reshape(src, (2, 3), order="F")
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


# ---------------------------------------------------------------------------
# dtype promotion across the input list (LIBRARY-layer; collect_typed in
# manipulation.rs takes the FIRST array's dtype instead of promoting).
# This is silent data corruption, not just a missing feature.
# ---------------------------------------------------------------------------


def test_concatenate_promotes_to_common_dtype():
    """numpy/_core/multiarray.py:198 concatenate promotes all inputs to a common
    result dtype (result_type). ferray's `collect_typed` casts every input to
    the FIRST array's dtype → wrong result dtype. (New.)
    Oracle: np.concatenate([int32, int64]).dtype == int64."""
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([3, 4], dtype=np.int64)
    expected = np.concatenate([a, b])
    out = fr.concatenate([a, b])
    assert out.dtype == expected.dtype == np.dtype(np.int64)
    assert np.array_equal(out, expected)


def test_concatenate_promotion_no_truncation():
    """numpy/_core/multiarray.py:198 concatenate — taking the first array's
    dtype TRUNCATES later values. ferray casts int64(1000) into the first
    array's int8 → -24. numpy promotes to int64 and preserves 1000. (New.)
    Oracle: np.concatenate([int8([1]), int64([1000])]) == [1, 1000]."""
    a = np.array([1], dtype=np.int8)
    b = np.array([1000], dtype=np.int64)
    expected = np.concatenate([a, b])
    out = fr.concatenate([a, b])
    assert out.dtype == expected.dtype == np.dtype(np.int64)
    assert np.array_equal(out, expected)


def test_stack_promotes_to_common_dtype():
    """numpy/_core/shape_base.py:379 stack promotes inputs to a common dtype.
    ferray's `collect_typed` takes the first array's dtype. (New.)
    Oracle: np.stack([int32, int64]).dtype == int64."""
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([3, 4], dtype=np.int64)
    expected = np.stack([a, b])
    out = fr.stack([a, b])
    assert out.dtype == expected.dtype == np.dtype(np.int64)
    assert np.array_equal(out, expected)
