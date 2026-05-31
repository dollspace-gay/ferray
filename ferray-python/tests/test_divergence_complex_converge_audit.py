"""Convergence-gate audit: remaining complex-dtype divergences on the MAIN `fr` array surface.

Run after epic #919 (#920-936). These pins FAIL against current ferray; each asserts the
numpy 2.4 oracle value (R-CHAR-3: expected values are live numpy calls, never copied from
ferray). The SILENT-CAST class is R-CODE-4 critical: ferray drops the imaginary part and
returns a WRONG real result where numpy returns complex.

VERDICT: NOT CONVERGED. Run:
    cd ferray-python && PYTHONPATH=python python3 -m pytest tests/test_divergence_complex_converge_audit.py -q
"""
import numpy as np
import ferray as fr
import pytest

Z = [3 + 4j, 1 - 2j, -5 + 12j]
M = [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]]


def _np(x):
    return np.asarray(x)


# ============================================================================
# CLASS C — SILENT-CAST (R-CODE-4 CRITICAL): numpy complex -> ferray WRONG real
# ============================================================================

def test_silentcast_nanmin():
    # numpy lexicographic min of complex -> (-5+12j); ferray returns real -5.0.
    exp = np.nanmin(_np(Z))
    got = fr.nanmin(fr.array(Z))
    assert np.iscomplexobj(_np(got)), f"SILENT-CAST: ferray nanmin returned real {got}"
    assert _np(got).item() == exp


def test_silentcast_nanmax():
    exp = np.nanmax(_np(Z))  # (3+4j)
    got = fr.nanmax(fr.array(Z))
    assert np.iscomplexobj(_np(got)), f"SILENT-CAST: ferray nanmax returned real {got}"
    assert _np(got).item() == exp


def test_silentcast_nanmedian():
    exp = np.nanmedian(_np(Z))  # (1-2j)
    got = fr.nanmedian(fr.array(Z))
    assert np.iscomplexobj(_np(got)), f"SILENT-CAST: ferray nanmedian returned real {got}"
    assert _np(got).item() == exp


def test_silentcast_nancumsum():
    exp = np.nancumsum(_np(Z))  # [3+4j, 4+2j, -1+14j]
    got = _np(fr.nancumsum(fr.array(Z)))
    assert np.iscomplexobj(got), f"SILENT-CAST: ferray nancumsum returned real {got}"
    np.testing.assert_allclose(got, exp)


def test_silentcast_nancumprod():
    # Doubly broken: drops imaginary AND mis-multiplies. numpy: [3+4j, 11-2j, -31+142j]
    exp = np.nancumprod(_np(Z))
    got = _np(fr.nancumprod(fr.array(Z)))
    assert np.iscomplexobj(got), f"SILENT-CAST: ferray nancumprod returned real {got}"
    np.testing.assert_allclose(got, exp)


def test_silentcast_meshgrid():
    exp = np.meshgrid(_np(Z), _np(Z))[0]  # complex128
    got = _np(fr.meshgrid(fr.array(Z), fr.array(Z))[0])
    assert np.iscomplexobj(got), f"SILENT-CAST: ferray meshgrid dropped imaginary -> {got.dtype}"
    np.testing.assert_allclose(got, exp)


def test_silentcast_nanvar():
    # numpy var of complex = mean(|x-mean|^2) = 44.444; ferray drops imaginary -> 11.555.
    exp = np.nanvar(_np(Z))
    got = float(_np(fr.nanvar(fr.array(Z))))
    assert got == pytest.approx(exp), f"SILENT-CAST: nanvar dropped imaginary (got {got}, want {exp})"


def test_silentcast_nanstd():
    exp = np.nanstd(_np(Z))  # 6.666
    got = float(_np(fr.nanstd(fr.array(Z))))
    assert got == pytest.approx(exp), f"SILENT-CAST: nanstd dropped imaginary (got {got}, want {exp})"


def test_silentcast_linalg_eigvals():
    # numpy eigvals are complex [-0.34+1.76j, 5.34+1.24j]; ferray returns eigvals of M.real.
    exp = np.sort_complex(np.linalg.eigvals(_np(M)))
    got = np.sort_complex(_np(fr.linalg.eigvals(fr.array(M))))
    np.testing.assert_allclose(got, exp), f"SILENT-CAST: eigvals dropped imaginary input"


def test_silentcast_linalg_eig_values():
    exp = np.sort_complex(np.linalg.eig(_np(M)).eigenvalues)
    fr_eig = fr.linalg.eig(fr.array(M))
    vals = fr_eig.eigenvalues if hasattr(fr_eig, "eigenvalues") else fr_eig[0]
    got = np.sort_complex(_np(vals))
    np.testing.assert_allclose(got, exp)


# E-class also silent-casts (numpy raises TypeError, ferray drops imaginary and computes):
def test_silentcast_unwrap():
    # numpy raises TypeError on complex unwrap; ferray silently unwraps real parts.
    with pytest.raises(TypeError):
        fr.unwrap(fr.array(Z))


def test_silentcast_digitize():
    with pytest.raises(TypeError):
        fr.digitize(fr.array(Z), fr.array([0.0, 1.0, 2.0]))


# ============================================================================
# CLASS D — raise-where-numpy-computes: ferray rejects complex numpy supports.
# ============================================================================

def test_D_take():
    exp = np.take(_np(Z), [0, 2])
    np.testing.assert_array_equal(_np(fr.take(fr.array(Z), [0, 2])), exp)


def test_D_trace():
    exp = np.trace(_np(M))  # (5+3j)
    assert _np(fr.trace(fr.array(M))).item() == exp


def test_D_nonzero():
    exp = np.nonzero(_np([0j, 1 + 1j, 2 + 0j]))
    got = fr.nonzero(fr.array([0j, 1 + 1j, 2 + 0j]))
    np.testing.assert_array_equal(_np(got[0]), exp[0])


def test_D_kron():
    exp = np.kron(_np(M), _np(M))
    np.testing.assert_allclose(_np(fr.kron(fr.array(M), fr.array(M))), exp)


def test_D_tensordot():
    exp = np.tensordot(_np(M), _np(M))
    np.testing.assert_allclose(_np(fr.tensordot(fr.array(M), fr.array(M))), exp)


def test_D_vander():
    exp = np.vander(_np(Z))
    np.testing.assert_allclose(_np(fr.vander(fr.array(Z))), exp)


def test_D_pad():
    exp = np.pad(_np(Z), (1, 1))
    np.testing.assert_allclose(_np(fr.pad(fr.array(Z), (1, 1))), exp)


def test_D_broadcast_to():
    exp = np.broadcast_to(_np(Z), (2, 3))
    np.testing.assert_allclose(_np(fr.broadcast_to(fr.array(Z), (2, 3))), exp)


def test_D_count_nonzero():
    exp = np.count_nonzero(_np(Z))  # 3
    assert int(fr.count_nonzero(fr.array(Z))) == exp


def test_D_searchsorted():
    base = [1 + 0j, 2 + 0j, 3 + 0j]
    exp = np.searchsorted(_np(base), 1 + 1j)
    assert int(fr.searchsorted(fr.array(base), 1 + 1j)) == int(exp)


def test_D_partition():
    exp = np.partition(_np(Z), 1)
    np.testing.assert_array_equal(_np(fr.partition(fr.array(Z), 1)), exp)


def test_D_geomspace():
    exp = np.geomspace(1 + 1j, 8 + 8j, 3)
    np.testing.assert_allclose(_np(fr.geomspace(1 + 1j, 8 + 8j, 3)), exp)


def test_D_eye_complex():
    exp = np.eye(2, dtype=complex)
    np.testing.assert_allclose(_np(fr.eye(2, dtype=complex)), exp)


def test_D_identity_complex():
    exp = np.identity(2, dtype=complex)
    np.testing.assert_allclose(_np(fr.identity(2, dtype=complex)), exp)


def test_D_linalg_svd():
    exp = np.linalg.svd(_np(M)).S
    s = fr.linalg.svd(fr.array(M))
    got = s.S if hasattr(s, "S") else s[1]
    np.testing.assert_allclose(np.sort(_np(got))[::-1], np.sort(exp)[::-1])


def test_D_linalg_qr():
    qn, rn = np.linalg.qr(_np(M))
    res = fr.linalg.qr(fr.array(M))
    qf = res.Q if hasattr(res, "Q") else res[0]
    rf = res.R if hasattr(res, "R") else res[1]
    np.testing.assert_allclose(_np(qf) @ _np(rf), _np(M))


def test_D_linalg_pinv():
    exp = np.linalg.pinv(_np(M))
    np.testing.assert_allclose(_np(fr.linalg.pinv(fr.array(M))), exp)


def test_D_linalg_matrix_rank():
    exp = np.linalg.matrix_rank(_np(M))
    assert int(fr.linalg.matrix_rank(fr.array(M))) == int(exp)


def test_D_linalg_slogdet():
    sgn, ld = np.linalg.slogdet(_np(M))
    res = fr.linalg.slogdet(fr.array(M))
    fsgn = res.sign if hasattr(res, "sign") else res[0]
    fld = res.logabsdet if hasattr(res, "logabsdet") else res[1]
    assert _np(fsgn).item() == pytest.approx(sgn)
    assert float(_np(fld)) == pytest.approx(ld)


def test_D_linalg_matrix_power():
    exp = np.linalg.matrix_power(_np(M), 2)
    np.testing.assert_allclose(_np(fr.linalg.matrix_power(fr.array(M), 2)), exp)


def test_D_intersect1d():
    exp = np.intersect1d(_np(Z), [1 - 2j])
    np.testing.assert_array_equal(_np(fr.intersect1d(fr.array(Z), [1 - 2j])), exp)


def test_D_union1d():
    exp = np.union1d(_np(Z), [1 - 2j])
    np.testing.assert_array_equal(np.sort_complex(_np(fr.union1d(fr.array(Z), [1 - 2j]))),
                                  np.sort_complex(exp))


def test_D_setdiff1d():
    exp = np.setdiff1d(_np(Z), [1 - 2j])
    np.testing.assert_array_equal(np.sort_complex(_np(fr.setdiff1d(fr.array(Z), [1 - 2j]))),
                                  np.sort_complex(exp))


def test_D_isin():
    exp = np.isin(_np(Z), [1 - 2j])
    np.testing.assert_array_equal(_np(fr.isin(fr.array(Z), [1 - 2j])), exp)
