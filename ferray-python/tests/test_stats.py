"""Phase-2 parity tests for the ferray stats surface."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_sum_full():
    src = np.array([1, 2, 3, 4, 5])
    assert int(ferray.sum(src)) == int(np.sum(src))


def test_sum_axis():
    src = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(ferray.sum(src, axis=0), np.sum(src, axis=0))
    np.testing.assert_array_equal(ferray.sum(src, axis=1), np.sum(src, axis=1))


def test_prod_full():
    src = np.array([1, 2, 3, 4])
    assert int(ferray.prod(src)) == int(np.prod(src))


def test_prod_axis():
    src = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(ferray.prod(src, axis=0), np.prod(src, axis=0))


def test_min_max_full():
    src = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert int(ferray.min(src)) == int(np.min(src))
    assert int(ferray.max(src)) == int(np.max(src))


def test_min_max_axis():
    src = np.array([[1, 5, 3], [4, 2, 6]])
    np.testing.assert_array_equal(ferray.min(src, axis=0), np.min(src, axis=0))
    np.testing.assert_array_equal(ferray.max(src, axis=1), np.max(src, axis=1))


def test_ptp_matches_numpy():
    src = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert int(ferray.ptp(src)) == int(np.ptp(src))


def test_mean_full():
    src = np.array([1.0, 2.0, 3.0, 4.0])
    assert float(ferray.mean(src)) == pytest.approx(float(np.mean(src)))


def test_mean_promotes_int_to_float64():
    src = np.array([1, 2, 3, 4], dtype=np.int64)
    out = ferray.mean(src)
    assert out.dtype == np.float64
    assert float(out) == pytest.approx(2.5)


def test_var_default_ddof():
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(float(ferray.var(src)), float(np.var(src)))


def test_var_with_ddof():
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(float(ferray.var(src, ddof=1)), float(np.var(src, ddof=1)))


def test_std_default_ddof():
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(float(ferray.std(src)), float(np.std(src)))


# ---------------------------------------------------------------------------
# Argmin / argmax
# ---------------------------------------------------------------------------


def test_argmin_argmax_full():
    src = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert int(ferray.argmin(src)) == int(np.argmin(src))
    assert int(ferray.argmax(src)) == int(np.argmax(src))


def test_argmin_argmax_axis():
    src = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(ferray.argmax(src, axis=1), np.argmax(src, axis=1))


def test_argmax_returns_int64():
    src = np.array([1, 2, 3])
    assert ferray.argmax(src).dtype == np.int64


# ---------------------------------------------------------------------------
# NaN-aware
# ---------------------------------------------------------------------------


def test_nansum_ignores_nan():
    src = np.array([1.0, 2.0, np.nan, 4.0])
    assert float(ferray.nansum(src)) == pytest.approx(float(np.nansum(src)))


def test_nanmean_ignores_nan():
    src = np.array([1.0, np.nan, 3.0, 5.0])
    assert float(ferray.nanmean(src)) == pytest.approx(float(np.nanmean(src)))


def test_nanmin_nanmax_ignore_nan():
    src = np.array([3.0, np.nan, 1.0, 5.0])
    assert float(ferray.nanmin(src)) == 1.0
    assert float(ferray.nanmax(src)) == 5.0


def test_nanvar_nanstd_match_numpy():
    src = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    np.testing.assert_allclose(float(ferray.nanvar(src)), float(np.nanvar(src)))
    np.testing.assert_allclose(float(ferray.nanstd(src)), float(np.nanstd(src)))


def test_nanmedian_matches_numpy():
    src = np.array([np.nan, 1.0, 5.0, 3.0, np.nan, 2.0])
    np.testing.assert_allclose(float(ferray.nanmedian(src)), float(np.nanmedian(src)))


def test_nanargmin_nanargmax():
    src = np.array([3.0, np.nan, 1.0, 5.0])
    assert int(ferray.nanargmin(src)) == int(np.nanargmin(src))
    assert int(ferray.nanargmax(src)) == int(np.nanargmax(src))


# ---------------------------------------------------------------------------
# Cumulative
# ---------------------------------------------------------------------------


def test_cumsum_1d():
    src = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(ferray.cumsum(src), np.cumsum(src))


def test_cumsum_axis():
    src = np.arange(12).reshape(3, 4)
    np.testing.assert_array_equal(ferray.cumsum(src, axis=0), np.cumsum(src, axis=0))
    np.testing.assert_array_equal(ferray.cumsum(src, axis=1), np.cumsum(src, axis=1))


def test_cumprod_1d():
    src = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(ferray.cumprod(src), np.cumprod(src))


def test_diff_default_n():
    src = np.array([1, 2, 4, 7, 0])
    np.testing.assert_array_equal(ferray.diff(src), np.diff(src))


def test_diff_n2():
    src = np.array([1, 2, 4, 7, 0])
    np.testing.assert_array_equal(ferray.diff(src, n=2), np.diff(src, n=2))


# ---------------------------------------------------------------------------
# Sort / argsort / searchsorted
# ---------------------------------------------------------------------------


def test_sort_1d():
    src = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    np.testing.assert_array_equal(ferray.sort(src), np.sort(src))


def test_sort_axis_1():
    src = np.array([[3, 1, 4], [1, 5, 9]])
    np.testing.assert_array_equal(ferray.sort(src, axis=1), np.sort(src, axis=1))


def test_argsort_1d_matches_numpy():
    src = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    # Distinct values would give a unique answer; with ties, just check
    # that the permutation produces a sorted array.
    perm = ferray.argsort(src)
    assert np.array_equal(np.sort(src), src[perm])


def test_searchsorted_left():
    a = np.array([1, 3, 5, 7, 9])
    v = np.array([0, 2, 5, 8, 10])
    np.testing.assert_array_equal(
        ferray.searchsorted(a, v), np.searchsorted(a, v, side="left")
    )


def test_searchsorted_right():
    a = np.array([1, 3, 5, 7, 9])
    v = np.array([0, 2, 5, 8, 10])
    np.testing.assert_array_equal(
        ferray.searchsorted(a, v, side="right"),
        np.searchsorted(a, v, side="right"),
    )


# ---------------------------------------------------------------------------
# Unique / count_nonzero
# ---------------------------------------------------------------------------


def test_unique_matches_numpy():
    src = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
    np.testing.assert_array_equal(ferray.unique(src), np.unique(src))


def test_count_nonzero_full():
    src = np.array([0, 1, 0, 2, 0, 3])
    assert int(ferray.count_nonzero(src)) == int(np.count_nonzero(src))


def test_count_nonzero_axis():
    src = np.array([[0, 1, 0], [2, 0, 3]])
    np.testing.assert_array_equal(
        ferray.count_nonzero(src, axis=0), np.count_nonzero(src, axis=0)
    )


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------


def test_union1d_matches_numpy():
    a = np.array([1, 2, 3, 4])
    b = np.array([3, 4, 5, 6])
    np.testing.assert_array_equal(ferray.union1d(a, b), np.union1d(a, b))


def test_intersect1d_matches_numpy():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5, 6, 7])
    np.testing.assert_array_equal(ferray.intersect1d(a, b), np.intersect1d(a, b))


def test_setdiff1d_matches_numpy():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([3, 4, 5])
    np.testing.assert_array_equal(ferray.setdiff1d(a, b), np.setdiff1d(a, b))


def test_setxor1d_matches_numpy():
    a = np.array([1, 2, 3, 4])
    b = np.array([3, 4, 5, 6])
    np.testing.assert_array_equal(ferray.setxor1d(a, b), np.setxor1d(a, b))


def test_in1d_matches_isin():
    # `np.in1d` was removed in NumPy 2.0; the canonical replacement is
    # `np.isin`. ferray keeps the legacy name as a 1-D-only convenience
    # and we compare against the new function.
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6])
    np.testing.assert_array_equal(ferray.in1d(a, b), np.isin(a, b))


def test_isin_matches_numpy():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([2, 4, 6])
    np.testing.assert_array_equal(ferray.isin(a, b), np.isin(a, b))


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


def test_where_three_arg():
    cond = np.array([True, False, True, False])
    x = np.array([1, 2, 3, 4])
    y = np.array([10, 20, 30, 40])
    np.testing.assert_array_equal(ferray.where(cond, x, y), np.where(cond, x, y))


def test_where_broadcasts():
    cond = np.array([True, False, True])
    x = 0
    y = np.array([10, 20, 30])
    np.testing.assert_array_equal(ferray.where(cond, x, y), np.where(cond, x, y))
