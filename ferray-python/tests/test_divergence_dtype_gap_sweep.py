"""Divergence pins: dtype-handling gap sweep (fr.<fn> vs np.<fn>).

ACToR critic artifact. Each test PINS a divergence between ferray and the
numpy 2.4 oracle for complex128 / int64 inputs. Expected values are derived
LIVE from numpy in every assertion (R-CHAR-3) — never literal-copied from the
ferray side.

Classification:
  C  = SILENT-CAST: numpy returns complex, ferray drops the imaginary part
       (returns real / float64). DATA CORRUPTION. R-CODE-4 CRITICAL.
  E  = REJECTS-int: numpy computes on int64, ferray raises (float-only-sealed).
  D  = RAISE-where-numpy-computes: numpy returns a value, ferray raises.
  X  = ACCEPTS-where-numpy-raises: numpy raises TypeError on complex, ferray
       silently computes a (wrong, real-only) answer.

Each test FAILS under the current ferray build; that failure IS the pin.
Owning crates: ferray-stats (trapezoid/percentile/quantile),
ferray-polynomial (polyval/polymul/polyadd/polyder/polyint),
ferray-linalg (kron/tensordot/vdot/inner/dot/matmul/trace/diagonal/cross/
matrix_power), ferray-core/ufunc (ediff1d/interp/choose/histogram).
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    return np.asarray(x)


# ---------------------------------------------------------------------------
# Class C — SILENT COMPLEX-CAST (drops imaginary part). CRITICAL data corruption.
# ---------------------------------------------------------------------------
CZ = [1 + 2j, 3 - 4j, -5 + 0.5j, 2 + 0j]


def test_C_trapezoid_complex_drops_imag():
    """np.trapezoid keeps imag; fr.trapezoid returns float64."""
    expected = np.trapezoid(np.array(CZ))  # complex128 scalar
    got = _np(fr.trapezoid(fr.array(CZ)))
    assert np.iscomplexobj(np.asarray(expected))
    assert np.iscomplexobj(got), f"fr dropped imag: got {got.dtype}, expected complex"
    assert np.allclose(np.asarray(expected), got)


def test_C_interp_complex_drops_imag():
    """np.interp supports complex fp; fr.interp returns real."""
    x = np.array([1.5])
    xp = np.array([1.0, 2.0, 3.0])
    fp = np.array(CZ[:3])
    expected = np.interp(x, xp, fp)
    got = _np(fr.interp(fr.array([1.5]), fr.array([1.0, 2.0, 3.0]), fr.array(CZ[:3])))
    assert np.iscomplexobj(np.asarray(expected))
    assert np.iscomplexobj(got), f"fr.interp dropped imag: got {got.dtype}"
    assert np.allclose(np.asarray(expected), got)


def test_C_polyval_complex_drops_imag():
    expected = np.polyval(np.array(CZ), 2.0)
    got = _np(fr.polyval(fr.array(CZ), 2.0))
    assert np.iscomplexobj(np.asarray(expected))
    assert np.iscomplexobj(got), f"fr.polyval dropped imag: got {got.dtype}"
    assert np.allclose(np.asarray(expected), got)


def test_C_polymul_complex_drops_imag():
    expected = np.polymul(np.array(CZ), np.array(CZ))
    got = _np(fr.polymul(fr.array(CZ), fr.array(CZ)))
    assert np.iscomplexobj(expected)
    assert np.iscomplexobj(got), f"fr.polymul dropped imag: got {got.dtype}"
    assert np.allclose(expected, got)


def test_C_polyadd_complex_drops_imag():
    expected = np.polyadd(np.array(CZ), np.array(CZ))
    got = _np(fr.polyadd(fr.array(CZ), fr.array(CZ)))
    assert np.iscomplexobj(expected)
    assert np.iscomplexobj(got), f"fr.polyadd dropped imag: got {got.dtype}"
    assert np.allclose(expected, got)


def test_C_polyder_complex_drops_imag():
    expected = np.polyder(np.array(CZ))
    got = _np(fr.polyder(fr.array(CZ)))
    assert np.iscomplexobj(expected)
    assert np.iscomplexobj(got), f"fr.polyder dropped imag: got {got.dtype}"
    assert np.allclose(expected, got)


def test_C_polyint_complex_drops_imag():
    expected = np.polyint(np.array(CZ))
    got = _np(fr.polyint(fr.array(CZ)))
    assert np.iscomplexobj(expected)
    assert np.iscomplexobj(got), f"fr.polyint dropped imag: got {got.dtype}"
    assert np.allclose(expected, got)


# ---------------------------------------------------------------------------
# Class E — REJECTS int64 where numpy computes (float-only-sealed linalg).
# ---------------------------------------------------------------------------
IV = [1, 2, 3]
IM = [[1, 2], [3, 4]]


def test_E_kron_int_rejected():
    expected = np.kron(np.array(IV), np.array(IV))
    got = _np(fr.kron(fr.array(IV), fr.array(IV)))
    assert np.array_equal(expected, got)


def test_E_tensordot_int_rejected():
    expected = np.tensordot(np.array(IM), np.array(IM))
    got = _np(fr.tensordot(fr.array(IM), fr.array(IM)))
    assert np.allclose(np.asarray(expected, dtype=float), got.astype(float))


def test_E_vdot_int_rejected():
    expected = np.vdot(np.array(IV), np.array(IV))
    got = _np(fr.vdot(fr.array(IV), fr.array(IV)))
    assert np.allclose(float(expected), float(np.asarray(got)))


def test_E_inner_int_rejected():
    expected = np.inner(np.array(IV), np.array(IV))
    got = _np(fr.inner(fr.array(IV), fr.array(IV)))
    assert np.allclose(float(expected), float(np.asarray(got)))


def test_E_dot_int_rejected():
    expected = np.dot(np.array(IV), np.array(IV))
    got = _np(fr.dot(fr.array(IV), fr.array(IV)))
    assert np.allclose(float(expected), float(np.asarray(got)))


def test_E_matmul_int_rejected():
    expected = np.matmul(np.array(IM), np.array(IM))
    got = _np(fr.matmul(fr.array(IM), fr.array(IM)))
    assert np.array_equal(np.asarray(expected, dtype=float), got.astype(float))


def test_E_trace_int_rejected():
    expected = np.trace(np.array(IM))
    got = _np(fr.trace(fr.array(IM)))
    assert np.allclose(float(expected), float(np.asarray(got)))


def test_E_matrix_power_int_rejected():
    expected = np.linalg.matrix_power(np.array(IM), 2)
    got = _np(fr.linalg.matrix_power(fr.array(IM), 2))
    assert np.array_equal(np.asarray(expected, dtype=float), got.astype(float))


# ---------------------------------------------------------------------------
# Class D — RAISE where numpy computes.
# ---------------------------------------------------------------------------
def test_D_ediff1d_complex_rejected():
    expected = np.ediff1d(np.array(CZ))
    got = _np(fr.ediff1d(fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert np.allclose(expected, got)


def test_D_diagonal_complex_rejected():
    cm = np.array([[1 + 1j, 2 - 1j], [0 + 2j, 3 + 0j]])
    expected = np.diagonal(cm)
    got = _np(fr.diagonal(fr.array([[1 + 1j, 2 - 1j], [0 + 2j, 3 + 0j]])))
    assert np.iscomplexobj(got)
    assert np.allclose(expected, got)


def test_D_cross_complex_rejected():
    cv = np.array([1 + 2j, 3 - 4j, 5 + 0j])
    expected = np.cross(cv, cv)
    got = _np(fr.cross(fr.array([1 + 2j, 3 - 4j, 5 + 0j]),
                       fr.array([1 + 2j, 3 - 4j, 5 + 0j])))
    assert np.iscomplexobj(got)
    assert np.allclose(expected, got)


def test_D_choose_complex_rejected():
    idx = np.array([0, 1, 0, 1])
    expected = np.choose(idx, [np.array(CZ), np.array(CZ) * 2])
    got = _np(fr.choose(fr.array([0, 1, 0, 1]),
                        [fr.array(CZ), fr.array(CZ) * 2]))
    assert np.iscomplexobj(got)
    assert np.allclose(expected, got)


def test_D_histogram_complex_rejected():
    # numpy raises on complex? Confirm live, then pin observed numpy behavior.
    try:
        expected = np.histogram(np.array(CZ))[0]
    except Exception:
        pytest.skip("numpy itself raises on complex histogram; not a fr divergence")
    got = _np(fr.histogram(fr.array(CZ))[0])
    assert np.array_equal(expected, got)


# ---------------------------------------------------------------------------
# Class X — ferray ACCEPTS where numpy RAISES (silent wrong-answer on complex).
# numpy raises TypeError("a must be an array of real numbers"); ferray returns
# a real-only quantile. Pin the numpy contract: it must raise.
# ---------------------------------------------------------------------------
def test_X_percentile_complex_should_raise():
    with pytest.raises(Exception):
        np.percentile(np.array(CZ), 50)
    # ferray must mirror: raising, not silently computing a real-only answer.
    with pytest.raises(Exception):
        fr.percentile(fr.array(CZ), 50)


def test_X_quantile_complex_should_raise():
    with pytest.raises(Exception):
        np.quantile(np.array(CZ), 0.5)
    with pytest.raises(Exception):
        fr.quantile(fr.array(CZ), 0.5)
