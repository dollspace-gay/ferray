"""Divergence #976 — roll multi-axis (tuple shift + tuple axis) and
unique return_index/return_inverse/return_counts/axis/equal_nan kwargs.

Every expected value is derived LIVE from numpy 2.4 (R-CHAR-3): each ferray
result is compared against the matching numpy call, never a literal copied
from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


def _eq(a, b):
    """array_equal that treats NaNs in the same slots as equal."""
    a = np.asarray(a)
    b = np.asarray(b)
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)


def _tuple_eq(got, exp):
    assert isinstance(got, tuple)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert _eq(g, e), f"{g!r} != {e!r}"


# ---------------------------------------------------------------------------
# roll — multi-axis (the pinned #976 TypeError) + the single-axis/flat matrix
# ---------------------------------------------------------------------------

ROLL_CASES = [
    # (array, shift, axis)
    ([[1, 2], [3, 4]], (1, 1), (0, 1)),          # the exact pin
    (np.arange(10).reshape(2, 5), (1, 1), (1, 0)),  # numpy docstring example
    (np.arange(10).reshape(2, 5), (2, 1), (1, 0)),  # numpy docstring example
    (np.arange(10).reshape(2, 5), 2, (0, 1)),       # int shift + tuple axis
    (np.arange(24).reshape(2, 3, 4), (1, 2, 1), (0, 1, 2)),  # 3-D
    (np.arange(24).reshape(2, 3, 4), (1, -2), (0, 2)),       # 3-D partial axes
    ([1, 2, 3, 4, 5], 2, None),                  # flat roll, no axis
    ([1, 2, 3, 4, 5], -2, None),                 # flat roll negative
    (np.arange(10).reshape(2, 5), 1, 0),         # single int axis
    (np.arange(10).reshape(2, 5), -1, 1),        # single negative axis
    (np.arange(12).reshape(3, 4), (0, 0), (0, 1)),  # zero shift tuple
]


@pytest.mark.parametrize("arr,shift,axis", ROLL_CASES)
def test_roll_matches_numpy(arr, shift, axis):
    npa = np.asarray(arr)
    if axis is None:
        got = fr.roll(npa, shift)
        exp = np.roll(npa, shift)
    else:
        got = fr.roll(npa, shift, axis=axis)
        exp = np.roll(npa, shift, axis=axis)
    assert _eq(got, exp)
    assert np.asarray(got).shape == exp.shape


def test_roll_multi_axis_pin_value():
    """The exact divergence pinned in #976."""
    got = fr.roll([[1, 2], [3, 4]], (1, 1), axis=(0, 1))
    assert _eq(got, np.roll([[1, 2], [3, 4]], (1, 1), axis=(0, 1)))


def test_roll_preserves_dtype_float_complex():
    for dt in (np.float32, np.float64, np.complex64, np.complex128):
        a = np.arange(6, dtype=dt).reshape(2, 3)
        got = fr.roll(a, (1, 1), axis=(0, 1))
        exp = np.roll(a, (1, 1), axis=(0, 1))
        assert _eq(got, exp)
        assert np.asarray(got).dtype == exp.dtype


# ---------------------------------------------------------------------------
# unique — return_* / axis / equal_nan kwargs (the pinned #976 TypeError)
# ---------------------------------------------------------------------------

UNIQUE_1D = [1, 1, 2, 3, 3, 3]


def test_unique_plain():
    assert _eq(fr.unique(UNIQUE_1D), np.unique(UNIQUE_1D))


def test_unique_return_counts():
    _tuple_eq(
        fr.unique(UNIQUE_1D, return_counts=True),
        np.unique(UNIQUE_1D, return_counts=True),
    )


def test_unique_return_index():
    _tuple_eq(
        fr.unique(UNIQUE_1D, return_index=True),
        np.unique(UNIQUE_1D, return_index=True),
    )


def test_unique_return_inverse():
    _tuple_eq(
        fr.unique(UNIQUE_1D, return_inverse=True),
        np.unique(UNIQUE_1D, return_inverse=True),
    )


def test_unique_all_flags_tuple_order():
    """numpy tuple order is (values, indices, inverse, counts)."""
    _tuple_eq(
        fr.unique(
            UNIQUE_1D,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        ),
        np.unique(
            UNIQUE_1D,
            return_index=True,
            return_inverse=True,
            return_counts=True,
        ),
    )


def test_unique_axis0_row_unique():
    a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    assert _eq(fr.unique(a, axis=0), np.unique(a, axis=0))


def test_unique_axis1():
    a = np.array([[1, 0, 1], [2, 3, 2]])
    assert _eq(fr.unique(a, axis=1), np.unique(a, axis=1))


def test_unique_axis_return_counts():
    a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
    _tuple_eq(
        fr.unique(a, axis=0, return_counts=True),
        np.unique(a, axis=0, return_counts=True),
    )


def test_unique_equal_nan_false():
    a = [np.nan, np.nan, 1.0]
    assert _eq(fr.unique(a, equal_nan=False), np.unique(a, equal_nan=False))


def test_unique_counts_floats():
    a = [1.5, 1.5, 2.5, 2.5, 2.5]
    _tuple_eq(
        fr.unique(a, return_counts=True),
        np.unique(a, return_counts=True),
    )
