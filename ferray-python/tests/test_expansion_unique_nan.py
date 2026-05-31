"""Regression coverage for #977 — `fr.unique` on a float array containing NaN.

numpy's `np.unique` (default ``equal_nan=True``) sorts ascending, places NaN
LAST in the total order, and collapses ALL NaN into a single NaN. ferray used
a ``partial_cmp(...).unwrap_or(Equal)`` comparator that treated NaN as equal to
everything, returning the array unsorted with NaN un-deduplicated. The library
``ferray_stats::unique`` now uses the NaN-last total-order comparator (the same
one the #918 sort fix introduced), matching numpy.

Every expected value is taken from the live numpy 2.4 oracle (np.unique),
never literal-copied from ferray (R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


def _arrays_equal_with_nan(got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    g_nan = np.isnan(got)
    e_nan = np.isnan(expected)
    assert np.array_equal(g_nan, e_nan)
    assert np.array_equal(got[~g_nan], expected[~e_nan])


@pytest.mark.parametrize(
    "data",
    [
        [3.0, np.nan, 1.0, np.nan, 1.0],
        [1.0, np.nan, 1.0, np.nan],
        [np.inf, np.nan, 1.0, np.nan, -np.inf],
        [1.0, np.nan, -np.nan, 2.0],
        [np.nan, np.nan, np.nan],
        [5.0, 2.0, 8.0, 1.0, 2.0],  # no NaN
        [7.0, 3.0, 7.0, 3.0, 1.0],  # no NaN dupes
    ],
)
def test_unique_matches_numpy_f64(data):
    a = np.array(data, dtype=np.float64)
    _arrays_equal_with_nan(fr.unique(a), np.unique(a))


def test_unique_f32_nan():
    a = np.array([3.0, np.nan, 1.0, np.nan, 1.0], dtype=np.float32)
    _arrays_equal_with_nan(fr.unique(a), np.unique(a))


def test_unique_return_counts_nan():
    a = np.array([3.0, np.nan, 1.0, np.nan, 1.0], dtype=np.float64)
    fv, fc = fr.unique(a, return_counts=True)
    nv, nc = np.unique(a, return_counts=True)
    _arrays_equal_with_nan(fv, nv)
    assert np.array_equal(np.asarray(fc), nc)


def test_unique_return_inverse_nan():
    a = np.array([3.0, np.nan, 1.0, np.nan, 1.0], dtype=np.float64)
    fv, finv = fr.unique(a, return_inverse=True)
    nv, ninv = np.unique(a, return_inverse=True)
    _arrays_equal_with_nan(fv, nv)
    assert np.array_equal(np.asarray(finv).reshape(-1), np.asarray(ninv).reshape(-1))


def test_unique_int_unregressed():
    a = np.array([5, 2, 8, 1, 2], dtype=np.int64)
    got = np.asarray(fr.unique(a))
    exp = np.unique(a)
    assert got.dtype == exp.dtype
    assert np.array_equal(got, exp)
