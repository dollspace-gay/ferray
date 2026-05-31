"""Phase-4 parity tests for the ferray window-function surface."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Top-level window functions (NumPy puts these at the top level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["hanning", "hamming", "blackman", "bartlett"])
def test_top_level_window_matches_numpy(name):
    fr_fn = getattr(ferray, name)
    np_fn = getattr(np, name)
    np.testing.assert_allclose(fr_fn(16), np_fn(16), atol=1e-12)


def test_kaiser_top_level_matches_numpy():
    np.testing.assert_allclose(ferray.kaiser(16, 14.0), np.kaiser(16, 14.0), atol=1e-12)


def test_window_returns_float64():
    a = ferray.hanning(8)
    assert a.dtype == np.float64


def test_window_has_correct_length():
    for n in (1, 4, 16, 51):
        assert ferray.hanning(n).shape == (n,)


# ---------------------------------------------------------------------------
# ferray.window submodule (full set including SciPy extras)
# ---------------------------------------------------------------------------


def test_window_submodule_exposes_all_canonical_names():
    for name in ("hanning", "hamming", "blackman", "bartlett", "kaiser"):
        assert hasattr(ferray.window, name)


def test_window_submodule_extras_callable():
    # Just exercise the SciPy extras — exact values are validated by
    # ferray-window's own oracle tests.
    assert ferray.window.cosine(8).shape == (8,)
    assert ferray.window.nuttall(8).shape == (8,)
    assert ferray.window.parzen(8).shape == (8,)
    assert ferray.window.gaussian(8, 1.0).shape == (8,)
    assert ferray.window.exponential(8).shape == (8,)
    assert ferray.window.tukey(8, 0.5).shape == (8,)
    assert ferray.window.general_cosine(8, [0.5, 0.5]).shape == (8,)
    assert ferray.window.general_hamming(8, 0.54).shape == (8,)
    assert ferray.window.taylor(16).shape == (16,)


def test_kaiser_endpoints_are_close_to_zero():
    a = ferray.kaiser(64, 14.0)
    assert a[0] < 1e-3
    assert a[-1] < 1e-3
