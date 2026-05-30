"""Adversarial divergence tests for the ferray window-function surface.

ferray-python is meant to be a drop-in replacement for NumPy 2.4.x.  These
tests pin places where `ferray`'s window functions diverge from the live
`numpy` oracle.  Each test cites the upstream NumPy definition by
file:line in /home/doll/numpy-ref.

Source under audit: ferray-python/src/window.rs
NumPy oracle: numpy/lib/_function_base_impl.py
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Divergence 1 — Negative M must return an empty float64 array, not raise.
#
# Every numpy window function guards `if M < 1: return array([], ...)`:
#   numpy/lib/_function_base_impl.py:3140-3141 (blackman)
#       "    if M < 1:\n        return array([], dtype=values.dtype)"
#   :3247-3248 (bartlett), :3349-3350 (hanning), :3448-3449 (hamming)
# M is a *signed* float there (`values = np.array([0.0, M]); M = values[1]`),
# so a negative M yields the empty array.
#
# ferray's binding `bind_window_n!`/`kaiser` in window.rs takes `m: usize`,
# so any negative argument is rejected at the PyO3 boundary with
# OverflowError ("can't convert negative int to unsigned") instead of
# returning the empty array numpy returns.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bartlett", "blackman", "hamming", "hanning"])
@pytest.mark.parametrize("M", [-1, -5])
def test_negative_M_returns_empty_array(name, M):
    np_fn = getattr(np, name)
    fr_fn = getattr(fr, name)
    expected = np_fn(M)  # array([], dtype=float64)
    got = fr_fn(M)
    assert got.shape == expected.shape == (0,)
    assert got.dtype == expected.dtype == np.float64


def test_kaiser_negative_M_returns_empty_array():
    # kaiser: numpy/lib/_function_base_impl.py:3733 `n = arange(0, M)` with a
    # negative M produces an empty array; ferray's `kaiser(m: usize, ...)`
    # rejects the negative int with OverflowError.
    expected = np.kaiser(-3, 14.0)
    got = fr.kaiser(-3, 14.0)
    assert got.shape == expected.shape == (0,)
    assert got.dtype == expected.dtype == np.float64


# ---------------------------------------------------------------------------
# Divergence 2 — Float M must be accepted (numpy types M as _FloatLike_co).
#
# numpy/lib/_function_base_impl.pyi:1305-1309 declares every window taking
# `M: _FloatLike_co`.  The implementation does
#   numpy/lib/_function_base_impl.py:3346 "values = np.array([0.0, M])"
# which casts M to float64 — so `np.hanning(8.0)` is valid and identical to
# `np.hanning(8)`.
#
# ferray's `m: usize` rejects any Python float with
#   TypeError: 'float' object cannot be interpreted as an integer
# even for an integral float like 8.0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bartlett", "blackman", "hamming", "hanning"])
def test_integral_float_M_accepted(name):
    np_fn = getattr(np, name)
    fr_fn = getattr(fr, name)
    expected = np_fn(8.0)
    got = fr_fn(8.0)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_kaiser_integral_float_M_accepted():
    expected = np.kaiser(8.0, 14.0)
    got = fr.kaiser(8.0, 14.0)
    np.testing.assert_allclose(got, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Divergence 3 — The first parameter must be named `M`, not `m`.
#
# numpy/lib/_function_base_impl.py:3256 `def hanning(M):` (and :3149 bartlett,
# :3050 blackman, :3358 hamming, :3608 `def kaiser(M, beta):`).  numpy accepts
# the keyword call `np.hanning(M=8)` / `np.kaiser(M=8, beta=14.0)`.
#
# ferray's binding names the parameter `m` (window.rs `bind_window_n!` and
# `pub fn kaiser(... m: usize ...)`), so `fr.hanning(M=8)` raises
#   TypeError: ... got an unexpected keyword argument 'M'
# breaking drop-in keyword compatibility.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["bartlett", "blackman", "hamming", "hanning"])
def test_M_keyword_argument(name):
    np_fn = getattr(np, name)
    fr_fn = getattr(fr, name)
    expected = np_fn(M=8)
    got = fr_fn(M=8)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_kaiser_M_keyword_argument():
    expected = np.kaiser(M=8, beta=14.0)
    got = fr.kaiser(M=8, beta=14.0)
    np.testing.assert_allclose(got, expected, atol=1e-12)
