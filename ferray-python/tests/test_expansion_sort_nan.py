"""Parity tests for fr.sort / fr.argsort NaN handling vs numpy (#918).

numpy places NaN LAST in sort order regardless of sign; ferray must match.
Every expected value is computed live from numpy, never hand-copied.
"""

import numpy as np
import pytest

import ferray as fr


def _nan_eq(a, b):
    """Elementwise equality treating NaN==NaN as True (for sort comparison)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b) if np.issubdtype(a.dtype, np.floating) else False
    return bool(np.all((a == b) | both_nan))


SORT_CASES = [
    np.array([3.0, np.nan, 1.0]),
    np.array([np.nan, np.nan, 1.0, 2.0]),
    np.array([np.inf, -np.inf, 0.0, np.nan, 1.0]),
    np.array([1.0, np.nan, -np.nan, 2.0]),
    np.array([0.0, -0.0, 0.0]),
    np.array([5, 2, 8, 1]),  # integers — unaffected
]


@pytest.mark.parametrize("arr", SORT_CASES)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sort_matches_numpy(arr, dtype):
    if np.issubdtype(arr.dtype, np.integer):
        a = arr.astype(np.int64)
    else:
        a = arr.astype(dtype)
    expected = np.sort(a)
    got = fr.sort(a)
    assert _nan_eq(got, expected), f"fr.sort({a}) = {got}, numpy = {expected}"


@pytest.mark.parametrize(
    "arr",
    [
        np.array([3.0, np.nan, 1.0]),
        np.array([np.nan, 2.0, np.nan, 0.0, 1.0]),
        np.array([np.inf, -np.inf, 0.0, np.nan, 1.0]),
    ],
)
def test_argsort_matches_numpy(arr):
    expected = np.argsort(arr)
    got = np.asarray(fr.argsort(arr))
    # numpy nan-last; positions of nan among equal-rank are stable, but for
    # distinct values the permutation is unique.
    assert np.array_equal(got, expected), f"fr.argsort({arr}) = {got}, numpy = {expected}"


def test_sort_2d_axis1_nan_last():
    a = np.array([[3.0, np.nan, 1.0], [np.nan, 2.0, 0.0]])
    expected = np.sort(a, axis=1)
    got = fr.sort(a, axis=1)
    assert _nan_eq(got, expected), f"fr.sort 2d = {got}, numpy = {expected}"


def test_sort_2d_axis0_nan_last():
    a = np.array([[3.0, np.nan, 1.0], [np.nan, 2.0, 0.0]])
    expected = np.sort(a, axis=0)
    got = fr.sort(a, axis=0)
    assert _nan_eq(got, expected), f"fr.sort 2d axis0 = {got}, numpy = {expected}"


def test_integer_sort_unregressed():
    a = np.array([5, 2, 8, 1, 9, 3], dtype=np.int64)
    assert np.array_equal(np.asarray(fr.sort(a)), np.sort(a))
