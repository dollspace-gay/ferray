"""Adversarial divergence tests for the ferray window-function surface.

ferray-python is meant to be a drop-in replacement for NumPy 2.4.x. These
tests pin places where `ferray`'s window functions diverge from the *live*
`numpy` oracle (numpy 2.4.5). Every expected value is produced by calling
numpy at test time (R-CHAR-3 — never literal-copied from the ferray side).

Source under audit: ferray-python/src/window.rs
NumPy oracle: numpy/lib/_function_base_impl.py

Root cause of all three pins: the PyO3 bindings type the first parameter as
`m: usize` (window.rs `bind_window_n!` macro, lines 27-35, and
`pub fn kaiser(py, m: usize, beta: f64)` line 51). NumPy types `M` as a
*signed float* (`_FloatLike_co`) and names it `M`. So `m: usize`:
  (1) rejects negative M with OverflowError instead of returning array([]),
  (2) rejects integral-float M (e.g. 8.0) with TypeError, and
  (3) names the kwarg `m`, breaking `np.hanning(M=8)`.
The fix belongs in ferray-python (the marshalling boundary): accept a signed
float-coercible M named `M`, guard `M < 1 -> empty float64 array`.
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
# ferray's `m: usize` rejects any negative int at the PyO3 boundary with
# OverflowError("can't convert negative int to unsigned").
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
# ferray's `m: usize` rejects any Python float with
#   TypeError: argument 'm': 'float' object cannot be interpreted as an integer
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
# ferray names the parameter `m` (window.rs `bind_window_n!` and
# `pub fn kaiser(... m: usize ...)`), so `fr.hanning(M=8)` raises
#   TypeError: hanning() got an unexpected keyword argument 'M'.
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
