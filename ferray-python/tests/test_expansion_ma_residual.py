"""numpy.ma residual slice (refs #837): polyfit / convolve / correlate +
the two-variable ``y`` form of cov / corrcoef.

Each case asserts ``ferray.ma`` against the live ``numpy.ma`` oracle
(numpy 2.4.x) for BOTH the data and the mask (R-CHAR-3 — expected values
come from numpy, never literal-copied from the ferray side).
"""

import numpy as np
import pytest

import ferray as fr


def _data(ma):
    """Underlying data buffer of a ferray.ma masked array as an ndarray."""
    return np.asarray(ma.data)


def _mask(ma):
    """Full bool mask of a ferray.ma masked array as an ndarray."""
    return np.asarray(fr.ma.getmaskarray(ma))


def _assert_ma_equal(got, want, *, float_data=True):
    """Compare a ferray.ma result against a numpy.ma result, value + mask."""
    want_mask = np.ma.getmaskarray(want)
    np.testing.assert_array_equal(_mask(got), want_mask)
    gd = _data(got)
    wd = np.asarray(want.data)
    if float_data:
        np.testing.assert_allclose(gd, wd, rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_array_equal(gd, wd)


# ---------------------------------------------------------------------------
# polyfit
# ---------------------------------------------------------------------------


def test_polyfit_linear_drops_masked_pair():
    x = np.ma.array([0.0, 1.0, 2.0, 3.0], mask=[0, 0, 1, 0])
    y = np.ma.array([1.0, 3.0, 5.0, 7.0], mask=[0, 0, 0, 0])
    want = np.ma.polyfit(x, y, 1)
    got = np.asarray(fr.ma.polyfit(x, y, 1))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)


def test_polyfit_unmasked_matches_plain():
    x = np.ma.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.ma.array([1.0, 2.1, 3.9, 6.2, 8.1])
    want = np.ma.polyfit(x, y, 1)
    got = np.asarray(fr.ma.polyfit(x, y, 1))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_polyfit_quadratic_y_mask_propagates():
    x = np.ma.array([0.0, 1.0, 2.0, 3.0, 4.0], mask=[0, 0, 1, 0, 0])
    y = np.ma.array([1.0, 2.0, 3.0, 4.0, 9.0], mask=[0, 0, 0, 0, 1])
    want = np.ma.polyfit(x, y, 2)
    got = np.asarray(fr.ma.polyfit(x, y, 2))
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9)


# ---------------------------------------------------------------------------
# convolve
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_modes(mode):
    a = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    v = [1.0, 1.0]
    want = np.ma.convolve(a, v, mode=mode)
    got = fr.ma.convolve(a, v, mode=mode)
    _assert_ma_equal(got, want)


@pytest.mark.parametrize("propagate", [True, False])
def test_convolve_propagate_mask(propagate):
    a = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    v = np.ma.array([0.0, 1.0, 0.0], mask=[0, 0, 1])
    want = np.ma.convolve(a, v, propagate_mask=propagate)
    got = fr.ma.convolve(a, v, propagate_mask=propagate)
    _assert_ma_equal(got, want)


def test_convolve_unmasked_matches_plain():
    a = np.array([1.0, 2.0, 3.0])
    v = np.array([0.0, 1.0, 0.5])
    want = np.ma.convolve(a, v)
    got = fr.ma.convolve(a, v)
    _assert_ma_equal(got, want)


# ---------------------------------------------------------------------------
# correlate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["valid", "same", "full"])
def test_correlate_modes(mode):
    a = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    v = [1.0, 1.0]
    want = np.ma.correlate(a, v, mode=mode)
    got = fr.ma.correlate(a, v, mode=mode)
    _assert_ma_equal(got, want)


def test_correlate_default_mode_is_valid():
    # numpy.ma.correlate defaults to 'valid' (unlike convolve's 'full').
    a = np.ma.array([1.0, 2.0, 3.0])
    v = np.ma.array([0.0, 1.0, 0.0])
    want = np.ma.correlate(a, v)
    got = fr.ma.correlate(a, v)
    assert _data(got).shape == np.asarray(want.data).shape
    _assert_ma_equal(got, want)


@pytest.mark.parametrize("propagate", [True, False])
def test_correlate_propagate_mask(propagate):
    a = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    v = np.ma.array([0.0, 1.0, 0.0], mask=[0, 0, 1])
    want = np.ma.correlate(a, v, propagate_mask=propagate)
    got = fr.ma.correlate(a, v, propagate_mask=propagate)
    _assert_ma_equal(got, want)


# ---------------------------------------------------------------------------
# cov / corrcoef two-variable y form
# ---------------------------------------------------------------------------


def test_cov_two_var_unmasked():
    x = np.ma.array([1.0, 2.0, 3.0, 4.0])
    y = np.ma.array([1.0, 3.0, 2.0, 5.0])
    want = np.ma.cov(x, y)
    got = fr.ma.cov(x, y)
    np.testing.assert_allclose(_data(got), np.asarray(want.data), rtol=1e-9)


def test_cov_two_var_common_mask():
    x = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    y = np.ma.array([1.0, 3.0, 2.0, 5.0], mask=[0, 0, 0, 1])
    want = np.ma.cov(x, y)
    got = fr.ma.cov(x, y)
    np.testing.assert_allclose(_data(got), np.asarray(want.data), rtol=1e-9)
    np.testing.assert_array_equal(_mask(got), np.ma.getmaskarray(want))


def test_corrcoef_two_var_unmasked():
    x = np.ma.array([1.0, 2.0, 3.0, 4.0])
    y = np.ma.array([1.0, 3.0, 2.0, 5.0])
    want = np.ma.corrcoef(x, y)
    got = fr.ma.corrcoef(x, y)
    np.testing.assert_allclose(_data(got), np.asarray(want.data), rtol=1e-9)


def test_corrcoef_two_var_common_mask():
    x = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    y = np.ma.array([1.0, 3.0, 2.0, 5.0], mask=[0, 0, 0, 1])
    want = np.ma.corrcoef(x, y)
    got = fr.ma.corrcoef(x, y)
    np.testing.assert_allclose(_data(got), np.asarray(want.data), rtol=1e-9)
    np.testing.assert_array_equal(_mask(got), np.ma.getmaskarray(want))


def test_cov_single_var_unchanged():
    # y=None path is unchanged by the residual slice.
    m = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 0], [0, 1, 0]])
    want = np.ma.cov(m)
    got = fr.ma.cov(m)
    np.testing.assert_allclose(_data(got), np.asarray(want.data), rtol=1e-9)
