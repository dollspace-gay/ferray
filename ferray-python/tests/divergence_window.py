"""Adversarial divergence tests for the ferray window-function surface.

ferray-python is meant to be a drop-in replacement for NumPy 2.4.x. These
tests pin places where `ferray`'s window functions diverge from the *live*
`numpy` oracle (numpy 2.4.x). Every expected value is produced by calling
numpy at test time (R-CHAR-3 — never literal-copied from the ferray side).

Source under audit: ferray-python/src/window.rs
NumPy oracle: numpy/lib/_function_base_impl.py

NOTE (2026-05-31 critic pass): divergences 1-3 below (negative M, integral-
float M, `M` keyword) were FIXED by the current binding, which now coerces M
via `coerce_window_m` (src/conv.rs) and names the parameter `M`. Those tests
now PASS and stand as regression guards. The NEW open divergence is
test_noninteger_M_* (divergence 4) — fractional M is truncated.
"""

import numpy as np
import pytest

import ferray as fr

WINDOWS = ["bartlett", "blackman", "hamming", "hanning"]


# ---------------------------------------------------------------------------
# Divergence 1 — M < 1 must return an empty float64 array, NOT raise.
#
# Each window guards `if M < 1: return array([], dtype=values.dtype)`:
#   numpy/lib/_function_base_impl.py:3140-3141 (blackman)
#   :3247-3248 (bartlett), :3349-3350 (hanning), :3448-3449 (hamming)
# M is signed (`values = np.array([0.0, M])`, e.g. :3346 hanning), so a
# negative M yields the empty array. kaiser: :3733 `n = arange(0, M)` over a
# negative M is empty too.
#
# STATUS: FIXED — now a regression guard (passes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("M", [-1, -5])
def test_negative_M_returns_empty_array(name, M):
    expected = getattr(np, name)(M)  # live oracle: array([], dtype=float64)
    got = getattr(fr, name)(M)
    assert got.shape == expected.shape == (0,)
    assert got.dtype == expected.dtype == np.float64


def test_kaiser_negative_M_returns_empty_array():
    expected = np.kaiser(-3, 14.0)
    got = fr.kaiser(-3, 14.0)
    assert got.shape == expected.shape == (0,)
    assert got.dtype == expected.dtype == np.float64


# ---------------------------------------------------------------------------
# Divergence 2 — Float M must be accepted (numpy types M as _FloatLike_co).
#
# numpy casts M to float64 via `values = np.array([0.0, M])`
# (numpy/lib/_function_base_impl.py:3346 hanning, :3244 bartlett, :3137
# blackman, :3445 hamming, :3727 kaiser), so `np.hanning(8.0)` is valid and
# equal to `np.hanning(8)`. A negative *float* (-1.0) likewise routes through
# the `M < 1` guard to the empty array.
#
# STATUS: FIXED — now a regression guard (passes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WINDOWS)
def test_integral_float_M_accepted(name):
    expected = getattr(np, name)(8.0)
    got = getattr(fr, name)(8.0)
    assert got.shape == expected.shape == (8,)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_kaiser_integral_float_M_accepted():
    expected = np.kaiser(8.0, 14.0)
    got = fr.kaiser(8.0, 14.0)
    assert got.shape == expected.shape == (8,)
    np.testing.assert_allclose(got, expected, atol=1e-12)


@pytest.mark.parametrize("name", WINDOWS)
def test_negative_float_M_returns_empty_array(name):
    # -1.0 routes through numpy's `M < 1` guard to array([], float64);
    # ferray rejects the float before the guard can run.
    expected = getattr(np, name)(-1.0)
    got = getattr(fr, name)(-1.0)
    assert got.shape == expected.shape == (0,)
    assert got.dtype == expected.dtype == np.float64


# ---------------------------------------------------------------------------
# Divergence 3 — The first parameter must be named `M`, not `m`.
#
# numpy/lib/_function_base_impl.py:3256 `def hanning(M):` (also :3149
# bartlett, :3050 blackman, :3358 hamming, :3608 `def kaiser(M, beta):`).
# numpy accepts `np.hanning(M=8)` / `np.kaiser(M=8, beta=14.0)`.
#
# STATUS: FIXED — now a regression guard (passes).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WINDOWS)
def test_M_keyword_argument(name):
    expected = getattr(np, name)(M=8)
    got = getattr(fr, name)(M=8)
    assert got.shape == expected.shape == (8,)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_kaiser_M_keyword_argument():
    expected = np.kaiser(M=8, beta=14.0)
    got = fr.kaiser(M=8, beta=14.0)
    assert got.shape == expected.shape == (8,)
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Divergence 4 (OPEN — tracking #1012) — non-integer M is truncated.
#
# NumPy keeps M as a *float64* through the whole formula. For hanning:
#   numpy/lib/_function_base_impl.py:3346  values = np.array([0.0, M])
#   :3347                                  M = values[1]      # stays 2.7
#   :3353                                  n = arange(1 - M, M, 2)
#   :3354                                  return 0.5 + 0.5 * cos(pi * n / (M - 1))
# (identical structure: blackman :3137/:3144/:3145, bartlett :3244/:3251/:3252,
#  hamming :3445/:3452/:3453, kaiser :3727/:3733/:3734.)
# So `arange(1 - M, M, 2)` for M=2.7 yields len-3 [-1.7, 0.3, 2.3] and the
# denominator `M - 1` is 1.7 — np.hanning(2.7) has 3 elements, not 2.
#
# ferray's `coerce_window_m` (src/conv.rs:346-354) does `f.trunc() as isize`,
# so M=2.7 becomes integer 2 BEFORE any window math: wrong length AND wrong
# values. This affects all five canonical windows.
#
# Expected values are taken from the LIVE oracle at test time (R-CHAR-3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("M", [2.7, 3.5, 4.9])
def test_noninteger_M_uses_float_in_formula(name, M):
    expected = getattr(np, name)(M)  # live numpy oracle, float64 M preserved
    got = np.asarray(getattr(fr, name)(M))
    # numpy's length is len(arange(1 - M, M, 2)); ferray truncates M first.
    assert got.shape == expected.shape, (
        f"{name}({M}): ferray shape {got.shape} != numpy {expected.shape} "
        f"(non-integer M must not be truncated)"
    )
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("M", [2.7, 4.9])
def test_kaiser_noninteger_M_uses_float_in_formula(M):
    expected = np.kaiser(M, 14.0)  # live numpy oracle
    got = np.asarray(fr.kaiser(M, 14.0))
    assert got.shape == expected.shape, (
        f"kaiser({M}, 14.0): ferray shape {got.shape} != numpy {expected.shape}"
    )
    # numpy emits NaN for the out-of-domain tail element (invalid sqrt); compare
    # with equal_nan so the pin tracks the value divergence, not the NaN itself.
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15, equal_nan=True)
