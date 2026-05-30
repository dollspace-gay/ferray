"""ACToR critic pins — DynMa dtype-preservation refactor (commit 07179028, #853).

These tests pin DIVERGENCES between ferray.ma and numpy.ma surfaced while
auditing the DynMa enum refactor that made PyMaskedArray dtype-preserving.

R-CHAR-3: every expected value is derived LIVE from numpy.ma in the test body
(never literal-copied from the ferray side). Each assertion FAILS against the
current ferray build.

CONFIRMED divergence pinned here: the integer/bool REDUCTION surface
(sum / prod / cumsum / cumprod / min / max) collapses the native integer dtype
to float64. numpy.ma keeps the integer dtype (delegating to
`self.filled(0).sum(...)` — numpy/ma/core.py:5244, `result = self.filled(0).sum(...)`;
docstring example `x.sum()` -> `25` int — numpy/ma/core.py:5224). The ferray
binding routes these through `to_f64_ma()` (ferray-python/src/ma.rs:426,
`fn to_f64_ma`), so the result egresses as float64.

This is the builder's DISCLOSED int-reduction-f64 spillover (doc-comment on
`fn sum` in ma.rs: "dtype-faithful integer reductions are a follow-up tracked
under #853"). Pinned here so it is tracked as a failing test; classified as a
binding-fixable divergence (the ferray-ma library MaskedArray<T> is already
generic — the f64 funnel lives entirely in the binding's `to_f64_ma`).
"""

import numpy as np
import pytest

import ferray as fr


# --- full reductions: sum / min / max on integer input ---------------------

def test_int_sum_keeps_int_dtype():
    """numpy.ma sum of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py:5244 `result = self.filled(0).sum(...)`.
    """
    n = np.ma.array([1, 2, 3], dtype=np.int64).sum()
    f = fr.ma.array([1, 2, 3], dtype=np.int64).sum()
    expected_dtype = np.asarray(n).dtype  # numpy says int64
    assert str(np.asarray(f).dtype) == str(expected_dtype)


def test_bool_sum_promotes_to_int64():
    """numpy.ma sum of a bool array -> int64; ferray collapses to float64.

    Upstream: bool reduces through `filled(0).sum()` which numpy promotes
    bool->intp/int64 (numpy/ma/core.py:5244).
    """
    n = np.ma.array([True, True, False]).sum()
    f = fr.ma.array([True, True, False]).sum()
    expected_dtype = np.asarray(n).dtype  # numpy says int64
    assert str(np.asarray(f).dtype) == str(expected_dtype)


def test_int_min_keeps_int_dtype():
    """numpy.ma min of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py:5861 `def min` -> int result for int input.
    """
    n = np.ma.min(np.ma.array([3, 1, 2], dtype=np.int64))
    f = fr.ma.min(fr.ma.array([3, 1, 2], dtype=np.int64))
    assert str(np.asarray(f).dtype) == str(np.asarray(n).dtype)


def test_int_max_keeps_int_dtype():
    """numpy.ma max of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py:5959 `def max` -> int result for int input.
    """
    n = np.ma.max(np.ma.array([3, 1, 2], dtype=np.int64))
    f = fr.ma.max(fr.ma.array([3, 1, 2], dtype=np.int64))
    assert str(np.asarray(f).dtype) == str(np.asarray(n).dtype)


# --- free-fn reductions: prod / cumsum / cumprod ---------------------------

def test_int_prod_keeps_int_dtype():
    """numpy.ma prod of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py:5303 `def prod`.
    """
    n = np.ma.prod(np.ma.array([1, 2, 3], dtype=np.int64))
    f = fr.ma.prod(fr.ma.array([1, 2, 3], dtype=np.int64))
    assert str(np.asarray(f).dtype) == str(np.asarray(n).dtype)


def test_int_cumsum_keeps_int_dtype():
    """numpy.ma cumsum of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py:5261 `def cumsum`.
    """
    n = np.ma.cumsum(np.ma.array([1, 2, 3], dtype=np.int64))
    f = fr.ma.cumsum(fr.ma.array([1, 2, 3], dtype=np.int64))
    assert str(np.asarray(f).dtype) == str(np.asarray(n).dtype)


def test_int_cumprod_keeps_int_dtype():
    """numpy.ma cumprod of int64 stays int64; ferray collapses to float64.

    Upstream: numpy/ma/core.py (`def cumprod`), mirrors cumsum.
    """
    n = np.ma.cumprod(np.ma.array([1, 2, 3], dtype=np.int64))
    f = fr.ma.cumprod(fr.ma.array([1, 2, 3], dtype=np.int64))
    assert str(np.asarray(f).dtype) == str(np.asarray(n).dtype)


# --- KNOWN/disclosed out-of-scope gaps (still pinned for tracking) ---------

def test_KNOWN_complex_masked_array_supported_by_numpy():
    """numpy.ma SUPPORTS complex128 masked arrays; ferray raises TypeError.

    KNOWN/disclosed: builder's build_dynma rejects complex as out-of-scope.
    numpy: `np.ma.array([1+2j]).dtype` == complex128 (numpy/ma/core.py:2885
    `_data = np.array(data, dtype=dtype, ...)` keeps the complex dtype).
    This pins that the rejection IS a divergence vs numpy.
    """
    expected_dtype = str(np.ma.array([1 + 2j, 3 + 4j]).dtype)  # complex128
    f = fr.ma.array([1 + 2j, 3 + 4j])
    assert str(np.asarray(f).dtype) == expected_dtype
