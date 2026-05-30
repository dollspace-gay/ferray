"""Adversarial divergence pins for ferray-python/src/stats.rs vs NumPy 2.4.

Authored by the acto-critic re-audit of `ferray-python/src/stats.rs`. Each
test computes the NumPy oracle LIVE (R-CHAR-3: no expected value is copied
from the ferray side) and asserts that ferray matches value, dtype, or
exception type. Every test here is EXPECTED TO FAIL on the current ferray
build; that failure pins a real NumPy divergence for the fixer.

Oracle: ``import numpy as np`` (installed numpy 2.4.x = source of truth).
Target: ``import ferray as fr``.

NOTE ON #764 (empty reductions): a prior fix made ``mean/var/std/median``
of an empty array return ``nan`` (matching numpy's warn-and-return, not
raise). That fix is VERIFIED CORRECT by this re-audit and is therefore NOT
re-pinned here (re-pinning already-fixed behavior would be a tautology).
``sum([]) -> 0.0`` and ``prod([]) -> 1.0`` were also verified correct.

NumPy emits RuntimeWarnings for degenerate reductions: that is numpy
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
# ddof >= n: numpy clamps `rcount - ddof` to 0 and divides -> +inf (+ warning);
# it does NOT raise. numpy/_core/_methods.py:204
#   `rcount = um.maximum(rcount - ddof, 0)`  then  true_divide -> inf.
# (the warning is numpy/_core/_methods.py:154-156). ferray raises ValueError.
# ---------------------------------------------------------------------------


def test_var_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:204 clamps dof to 0 then divides -> inf when
    ddof >= n. Expected: inf. Actual: ferray raises ValueError
    ('ddof >= number of elements')."""
    expected = _np_value(lambda: np.var(np.array([1.0, 2.0, 3.0]), ddof=3))
    assert np.isinf(expected)
    got = fr.var(fr.array([1.0, 2.0, 3.0]), ddof=3)
    assert np.isinf(np.asarray(got))


def test_std_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:217-227 (_std = sqrt of _var) -> inf for ddof>=n.
    Expected: inf. Actual: ferray raises ValueError."""
    expected = _np_value(lambda: np.std(np.array([1.0, 2.0, 3.0]), ddof=5))
    assert np.isinf(expected)
    got = fr.std(fr.array([1.0, 2.0, 3.0]), ddof=5)
    assert np.isinf(np.asarray(got))


# ---------------------------------------------------------------------------
# Integer accumulator promotion. numpy promotes a sub-platform-int input to
# the default platform integer (int64 signed / uint64 unsigned) for the
# accumulator. numpy/_core/fromnumeric.py:2325-2327 (sum):
#   "if `a` is signed then the platform integer is used while if `a` is
#    unsigned then an unsigned integer of the same precision ... is used".
#
# ferray keeps the narrow input dtype. Two observable consequences:
#   (a) WRONG RESULT DTYPE even when no overflow occurs, and
#   (b) on overflow ferray PANICS (debug overflow check) where numpy returns
#       the correct widened value.
# ---------------------------------------------------------------------------


def test_sum_int8_result_dtype_is_int64():
    """numpy/_core/fromnumeric.py:2325 — int8 sum accumulates in the platform
    integer (int64). No overflow here (1+2+3=6); only the dtype diverges.
    Expected dtype int64. ferray returns int8."""
    expected = np.sum(np.array([1, 2, 3], dtype=np.int8))
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([1, 2, 3], dtype="int8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_sum_uint8_result_dtype_is_uint64():
    """numpy/_core/fromnumeric.py:2326-2327 — unsigned sub-platform int sum
    accumulates in uint64. Expected dtype uint64. ferray returns uint8."""
    expected = np.sum(np.array([1, 2, 3], dtype=np.uint8))
    assert expected.dtype == np.uint64
    got = np.asarray(fr.sum(fr.array([1, 2, 3], dtype="uint8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_sum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2325 — int8 sum [100,100] accumulates in
    int64 -> 200. Expected 200 (int64). ferray PANICS (attempt to add with
    overflow, ferray-stats/src/parallel.rs) because it keeps int8."""
    expected = np.sum(np.array([100, 100], dtype=np.int8))
    assert int(expected) == 200
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([100, 100], dtype="int8")))
    assert int(got) == 200
    assert got.dtype == np.int64


def test_prod_int8_result_dtype_is_int64():
    """numpy/_core/fromnumeric.py:2692-2693 — prod of a sub-platform int
    promotes the accumulator to the platform int. Expected dtype int64.
    ferray keeps int8 (value 24 is correct but dtype is wrong)."""
    expected = np.prod(np.array([2, 3, 4], dtype=np.int8))
    assert expected.dtype == np.int64
    got = np.asarray(fr.prod(fr.array([2, 3, 4], dtype="int8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_cumsum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2854-2855 — cumsum of a sub-platform int
    promotes to the platform int. Expected [100, 200] int64. ferray keeps
    int8 and PANICS (attempt to add with overflow,
    ferray-ufunc/src/ops/arithmetic.rs)."""
    expected = np.cumsum(np.array([100, 100], dtype=np.int8))
    assert expected.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(expected), [100, 200])
    got = np.asarray(fr.cumsum(fr.array([100, 100], dtype="int8")))
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [100, 200])


# ---------------------------------------------------------------------------
# bool reductions: numpy sums/products bools in the platform integer.
# numpy/_core/fromnumeric.py:2325-2327 (the same promotion rule covers bool,
# treated as an integer of less precision than the platform integer).
# ferray raises TypeError ('unsupported dtype for numeric op: "bool"').
# ---------------------------------------------------------------------------


def test_sum_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2325 — bool sum accumulates in the platform
    integer. Expected int64 value 3. ferray raises TypeError (bool unsupported)."""
    expected = np.sum(np.array([True, True, False, True]))
    assert int(expected) == 3
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([True, True, False, True])))
    assert int(got) == 3
    assert got.dtype == np.int64


def test_prod_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2692-2693 — bool prod accumulates in the
    platform integer. Expected int64 value 1. ferray raises TypeError."""
    expected = np.prod(np.array([True, True]))
    assert int(expected) == 1
    assert expected.dtype == np.int64
    got = np.asarray(fr.prod(fr.array([True, True])))
    assert int(got) == 1
    assert got.dtype == np.int64


# ---------------------------------------------------------------------------
# percentile / quantile accept array-like q and return an array of results.
# numpy/lib/_function_base_impl.py:4083 (percentile) and :4284 (quantile):
#   "q : array_like of float".
# ferray's binding signature only accepts a scalar f64 -> TypeError on list q.
# ---------------------------------------------------------------------------


def test_percentile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4083 'q : array_like of float'. A list q
    returns an array of percentiles. Expected [1.75, 2.5, 3.25]. ferray raises
    TypeError ('argument q: must be real number, not list')."""
    expected = np.percentile(np.array([1.0, 2.0, 3.0, 4.0]), [25, 50, 75])
    got = np.asarray(fr.percentile(fr.array([1.0, 2.0, 3.0, 4.0]), [25, 50, 75]))
    np.testing.assert_allclose(got, expected)


def test_quantile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4284 'q : array_like of float' mirrors
    percentile. A list q returns an array. Expected [1.75, 2.5, 3.25]. ferray
    raises TypeError on a list q."""
    expected = np.quantile(np.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75])
    got = np.asarray(fr.quantile(fr.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75]))
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# Axis out of bounds: numpy raises numpy.exceptions.AxisError, which is a
# subclass of BOTH ValueError and IndexError.
# numpy/exceptions.py:108 — `class AxisError(ValueError, IndexError)`.
# ferray raises a plain builtins.ValueError, so a caller doing
# `except IndexError` or `except np.exceptions.AxisError` would NOT catch it.
# ---------------------------------------------------------------------------


def test_sum_axis_out_of_bounds_raises_axiserror():
    """numpy/exceptions.py:108 — out-of-bounds axis raises AxisError
    (subclass of ValueError AND IndexError). ferray raises plain ValueError,
    which fails `isinstance(err, IndexError)`."""
    src = np.array([1, 2, 3])
    with pytest.raises(np.exceptions.AxisError):
        np.sum(src, axis=5)
    with pytest.raises(np.exceptions.AxisError):
        fr.sum(fr.array([1, 2, 3]), axis=5)
