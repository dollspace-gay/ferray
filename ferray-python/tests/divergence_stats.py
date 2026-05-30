"""Adversarial divergence pins for ferray-python/src/stats.rs vs NumPy 2.4.

Each test computes the NumPy oracle LIVE and asserts ferray matches value,
dtype, and exception type. Every test here is EXPECTED TO FAIL on the current
ferray build; that failure pins a real divergence for the fixer.

Oracle: ``import numpy as np`` (installed numpy = source of truth).
Target: ``import ferray as fr``.

NumPy emits RuntimeWarnings for empty / degenerate reductions: that is numpy
RETURNING a value (nan / inf), NOT raising. ferray must return the value too.
"""

import warnings

import numpy as np
import pytest

import ferray as fr


def _np_value(fn):
    """Evaluate a numpy reduction, swallowing the RuntimeWarning numpy emits
    while still capturing the value it returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn()


# ---------------------------------------------------------------------------
# Empty-array reductions: numpy returns nan (+ RuntimeWarning), never raises.
# numpy/_core/_methods.py:122 — `warnings.warn("Mean of empty slice", ...)`
# (it WARNS and proceeds to return nan; it does NOT raise).
# ---------------------------------------------------------------------------


def test_mean_empty_returns_nan_not_valueerror():
    """numpy/_core/_methods.py:122 warns 'Mean of empty slice' and returns nan.
    Expected: nan (float64 0-d). Actual: ferray raises ValueError."""
    expected = _np_value(lambda: np.mean(np.array([])))
    assert np.isnan(expected)
    got = fr.mean(fr.array([]))
    assert np.isnan(np.asarray(got))


def test_var_empty_returns_nan_not_valueerror():
    """numpy/_core/_methods.py:154-156 warns 'Degrees of freedom <= 0' and
    returns nan for an empty array. Expected: nan. Actual: ferray raises."""
    expected = _np_value(lambda: np.var(np.array([])))
    assert np.isnan(expected)
    got = fr.var(fr.array([]))
    assert np.isnan(np.asarray(got))


def test_std_empty_returns_nan_not_valueerror():
    """numpy/_core/_methods.py:217 (_std calls _var) returns nan for empty.
    Expected: nan. Actual: ferray raises ValueError."""
    expected = _np_value(lambda: np.std(np.array([])))
    assert np.isnan(expected)
    got = fr.std(fr.array([]))
    assert np.isnan(np.asarray(got))


def test_median_empty_returns_nan_not_valueerror():
    """numpy/lib/_function_base_impl.py:4003 (_median) returns nan + warns for
    an empty array. Expected: nan. Actual: ferray raises ValueError."""
    expected = _np_value(lambda: np.median(np.array([])))
    assert np.isnan(expected)
    got = fr.median(fr.array([]))
    assert np.isnan(np.asarray(got))


# ---------------------------------------------------------------------------
# ddof >= n: numpy clamps rcount-ddof to 0 and divides -> +inf (+ warning),
# numpy/_core/_methods.py:154-156 (warn) and :204 (`maximum(rcount-ddof, 0)`).
# ferray raises ValueError instead.
# ---------------------------------------------------------------------------


def test_var_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:204 `rcount = um.maximum(rcount - ddof, 0)`
    then divides -> inf when ddof >= n. Expected: inf. Actual: ferray raises."""
    src = np.array([1.0, 2.0, 3.0])
    expected = _np_value(lambda: np.var(src, ddof=3))
    assert np.isinf(expected)
    got = fr.var(fr.array([1.0, 2.0, 3.0]), ddof=3)
    assert np.isinf(np.asarray(got))


def test_std_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:217-227 (_std sqrt of _var) -> inf for ddof>=n.
    Expected: inf. Actual: ferray raises ValueError."""
    src = np.array([1.0, 2.0, 3.0])
    expected = _np_value(lambda: np.std(src, ddof=5))
    assert np.isinf(expected)
    got = fr.std(fr.array([1.0, 2.0, 3.0]), ddof=5)
    assert np.isinf(np.asarray(got))


# ---------------------------------------------------------------------------
# Integer accumulator promotion: numpy promotes sub-platform-int dtypes to the
# default platform integer for the accumulator.
# numpy/_core/fromnumeric.py:2323-2327 — "unless `a` has an integer dtype of
# less precision than the default platform integer.  In that case, if `a` is
# signed then the platform integer is used..."
# ferray keeps the narrow dtype -> WRONG VALUE on overflow.
# ---------------------------------------------------------------------------


def test_sum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2324 — int8 sum accumulates in platform int
    (int64). Expected value 200, dtype int64. ferray keeps int8 -> -56 (overflow)."""
    src = np.array([100, 100], dtype=np.int8)
    expected = np.sum(src)
    assert int(expected) == 200
    assert expected.dtype == np.int64
    got = fr.sum(fr.array([100, 100], dtype="int8"))
    assert int(np.asarray(got)) == 200
    assert np.asarray(got).dtype == np.int64


def test_prod_int8_promotes_to_platform_int():
    """numpy/_core/fromnumeric.py:2692-2693 — prod of sub-platform int promotes
    accumulator to platform int. Expected dtype int64. ferray keeps int8."""
    src = np.array([2, 3, 4], dtype=np.int8)
    expected = np.prod(src)
    assert expected.dtype == np.int64
    got = fr.prod(fr.array([2, 3, 4], dtype="int8"))
    assert np.asarray(got).dtype == np.int64
    assert int(np.asarray(got)) == int(expected)


def test_cumsum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2854 — cumsum of sub-platform int promotes to
    platform int. Expected [100, 200] int64. ferray keeps int8 -> [100, -56]."""
    src = np.array([100, 100], dtype=np.int8)
    expected = np.cumsum(src)
    assert expected.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(expected), [100, 200])
    got = np.asarray(fr.cumsum(fr.array([100, 100], dtype="int8")))
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [100, 200])


def test_sum_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2323-2327 — bool sum accumulates in platform
    int. Expected int64 value 3. ferray raises TypeError (bool unsupported)."""
    src = np.array([True, True, False, True])
    expected = np.sum(src)
    assert int(expected) == 3
    assert expected.dtype == np.int64
    got = fr.sum(fr.array([True, True, False, True]))
    assert int(np.asarray(got)) == 3
    assert np.asarray(got).dtype == np.int64


def test_prod_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2692-2693 — bool prod accumulates in platform
    int. Expected int64 value 1. ferray raises TypeError (bool unsupported)."""
    src = np.array([True, True])
    expected = np.prod(src)
    assert int(expected) == 1
    assert expected.dtype == np.int64
    got = fr.prod(fr.array([True, True]))
    assert int(np.asarray(got)) == 1
    assert np.asarray(got).dtype == np.int64


# ---------------------------------------------------------------------------
# percentile / quantile accept array-like q.
# numpy/lib/_function_base_impl.py:4083-4085 — "q : array_like of float ...
# Percentage or sequence of percentages". ferray's signature only accepts a
# scalar f64 -> TypeError on a list/array q.
# ---------------------------------------------------------------------------


def test_percentile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4083 'q : array_like of float'. A list q
    returns an array of percentiles. Expected [1.75, 2.5, 3.25]. ferray raises
    TypeError ('argument q: must be real number, not list')."""
    src = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.percentile(src, [25, 50, 75])
    got = np.asarray(fr.percentile(fr.array([1.0, 2.0, 3.0, 4.0]), [25, 50, 75]))
    np.testing.assert_allclose(got, expected)


def test_quantile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4268 quantile mirrors percentile's
    'q : array_like of float'. A list q returns an array. Expected
    [1.75, 2.5, 3.25]. ferray raises TypeError on a list q."""
    src = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.quantile(src, [0.25, 0.5, 0.75])
    got = np.asarray(fr.quantile(fr.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75]))
    np.testing.assert_allclose(got, expected)
