"""Regression pins for divergence cluster #972 — complex raise/accept mismatches.

Class D (RAISE-where-numpy-COMPUTES): ediff1d / diagonal / cross / choose /
polysub on complex input — numpy computes a genuinely-complex result; ferray
previously raised TypeError (real-only-sealed dispatch) or silently discarded
the imaginary part. Fix: delegate the complex case to numpy → complex result.

Class X (ACCEPT-where-numpy-RAISES): percentile / quantile on complex input —
numpy raises TypeError("a must be an array of real numbers"); ferray previously
coerced to float64 and returned a bogus real-only quantile. Fix: raise the
matching TypeError instead of computing a silent-wrong real.

histogram on complex input — numpy 2.4 does NOT raise: it casts to real with a
ComplexWarning and computes. ferray previously raised a bogus IndexError. Fix:
delegate the complex case to numpy → cast-and-compute.

Every expected value is derived LIVE from the numpy 2.4 oracle (R-CHAR-3),
never literal-copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr

CZ = [1 + 2j, 3 - 4j, -5 + 0.5j, 2 + 0j]


def _np(x):
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Class D — numpy COMPUTES complex; ferray must compute (delegate) complex.
# ---------------------------------------------------------------------------
def test_D_ediff1d_complex_computes():
    expected = np.ediff1d(np.array(CZ))
    got = _np(fr.ediff1d(fr.array(CZ)))
    assert np.iscomplexobj(got), f"fr.ediff1d dropped imag / wrong dtype: {got.dtype}"
    assert np.allclose(expected, got)


def test_D_diagonal_complex_computes():
    cm = [[1 + 1j, 2 - 1j], [0 + 2j, 3 + 0j]]
    expected = np.diagonal(np.array(cm))
    got = _np(fr.diagonal(fr.array(cm)))
    assert np.iscomplexobj(got), f"fr.diagonal dropped imag / wrong dtype: {got.dtype}"
    assert np.allclose(expected, got)


def test_D_cross_complex_computes():
    cv = [1 + 2j, 3 - 4j, 5 + 0j]
    expected = np.cross(np.array(cv), np.array(cv))
    got = _np(fr.cross(fr.array(cv), fr.array(cv)))
    assert np.iscomplexobj(got), f"fr.cross dropped imag / wrong dtype: {got.dtype}"
    assert np.allclose(expected, got)


def test_D_choose_complex_computes():
    idx = [0, 1, 0, 1]
    expected = np.choose(np.array(idx), [np.array(CZ), np.array(CZ) * 2])
    got = _np(fr.choose(fr.array(idx), [fr.array(CZ), fr.array(CZ) * 2]))
    assert np.iscomplexobj(got), f"fr.choose dropped imag / wrong dtype: {got.dtype}"
    assert np.allclose(expected, got)


def test_D_polysub_complex_computes():
    a1 = np.array(CZ)
    a2 = np.array(CZ) * 0.5
    expected = np.polysub(a1, a2)
    got = _np(fr.polysub(fr.array(CZ), fr.array([c * 0.5 for c in CZ])))
    assert np.iscomplexobj(expected)
    assert np.iscomplexobj(got), f"fr.polysub dropped imag / wrong dtype: {got.dtype}"
    assert np.allclose(expected, got)


def test_D_histogram_complex_matches_numpy():
    # numpy 2.4 casts complex -> real (ComplexWarning) and computes; it does NOT
    # raise. ferray must match: same integer bin counts.
    with pytest.warns(np.exceptions.ComplexWarning):
        expected_counts = np.histogram(np.array(CZ))[0]
    got_counts = _np(fr.histogram(fr.array(CZ))[0])
    assert np.array_equal(expected_counts, got_counts)


# ---------------------------------------------------------------------------
# Class X — numpy RAISES; ferray must RAISE (not silently compute a real-only).
# ---------------------------------------------------------------------------
def test_X_percentile_complex_raises():
    with pytest.raises(TypeError):
        np.percentile(np.array(CZ), 50)
    with pytest.raises(TypeError):
        fr.percentile(fr.array(CZ), 50)


def test_X_quantile_complex_raises():
    with pytest.raises(TypeError):
        np.quantile(np.array(CZ), 0.5)
    with pytest.raises(TypeError):
        fr.quantile(fr.array(CZ), 0.5)
