"""#937 — complex-dtype corruption fixes (R-CODE-4): nan-reductions, meshgrid,
linalg.eig/eigvals compute complex (no imaginary discard); unwrap/digitize/
histogram RAISE on complex exactly where numpy raises.

Every expected value is a LIVE numpy 2.4 call (R-CHAR-3 — never copied from the
ferray side). Run:
    cd ferray-python && maturin develop
    python -m pytest tests/test_expansion_complex_corruption2.py -q
"""
import numpy as np
import ferray as fr
import pytest

Z = [3 + 4j, 1 - 2j, -5 + 12j]
# A set with explicit NaN-complex to exercise the nan-skip path.
ZN = [3 + 4j, complex(np.nan, 0.0), -5 + 12j, complex(np.nan, np.nan)]
M = [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]]


def _np(x):
    return np.asarray(x)


def _sortc(a):
    return np.sort_complex(_np(a))


# ---------------------------------------------------------------------------
# nan-reductions: complex compute (no imaginary discard)
# ---------------------------------------------------------------------------

def test_nanmin_complex():
    exp = np.nanmin(_np(Z))  # (-5+12j)
    got = _np(fr.nanmin(fr.array(Z)))
    assert np.iscomplexobj(got), f"nanmin dropped imaginary -> {got.dtype}"
    assert got.item() == exp


def test_nanmax_complex():
    exp = np.nanmax(_np(Z))  # (3+4j)
    got = _np(fr.nanmax(fr.array(Z)))
    assert np.iscomplexobj(got), f"nanmax dropped imaginary -> {got.dtype}"
    assert got.item() == exp


def test_nanmin_complex_skips_nan():
    exp = np.nanmin(_np(ZN))  # (-5+12j), NaN-complex ignored
    got = _np(fr.nanmin(fr.array(ZN)))
    assert np.iscomplexobj(got)
    assert got.item() == exp


def test_nanmax_complex_skips_nan():
    exp = np.nanmax(_np(ZN))  # (3+4j)
    got = _np(fr.nanmax(fr.array(ZN)))
    assert np.iscomplexobj(got)
    assert got.item() == exp


def test_nanmedian_complex():
    exp = np.nanmedian(_np(Z))  # (1-2j)
    got = _np(fr.nanmedian(fr.array(Z)))
    assert np.iscomplexobj(got), f"nanmedian dropped imaginary -> {got.dtype}"
    assert got.item() == exp


def test_nanmedian_complex_skips_nan():
    exp = np.nanmedian(_np(ZN))  # (-1+8j), even count over the 2 non-nan values
    got = _np(fr.nanmedian(fr.array(ZN)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got.item(), exp)


def test_nanvar_complex():
    # Real variance over COMPLEX data = mean(|x-mean|^2) = 44.444 (NOT var of reals).
    exp = np.nanvar(_np(Z))
    got = float(_np(fr.nanvar(fr.array(Z))))
    assert got == pytest.approx(exp), f"nanvar dropped imaginary (got {got}, want {exp})"


def test_nanstd_complex():
    exp = np.nanstd(_np(Z))  # 6.666
    got = float(_np(fr.nanstd(fr.array(Z))))
    assert got == pytest.approx(exp)


def test_nanvar_complex_skips_nan():
    exp = np.nanvar(_np(ZN))  # 32.0 over the 2 non-nan values
    got = float(_np(fr.nanvar(fr.array(ZN))))
    assert got == pytest.approx(exp)


def test_nancumsum_complex():
    exp = np.nancumsum(_np(Z))  # [3+4j, 4+2j, -1+14j]
    got = _np(fr.nancumsum(fr.array(Z)))
    assert np.iscomplexobj(got), f"nancumsum dropped imaginary -> {got.dtype}"
    np.testing.assert_allclose(got, exp)


def test_nancumprod_complex():
    exp = np.nancumprod(_np(Z))  # [3+4j, 11-2j, -31+142j]
    got = _np(fr.nancumprod(fr.array(Z)))
    assert np.iscomplexobj(got), f"nancumprod dropped imaginary -> {got.dtype}"
    np.testing.assert_allclose(got, exp)


def test_nancumsum_complex_skips_nan():
    exp = np.nancumsum(_np(ZN))  # NaN positions carry the running value forward
    got = _np(fr.nancumsum(fr.array(ZN)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp)


def test_nancumprod_complex_skips_nan():
    exp = np.nancumprod(_np(ZN))
    got = _np(fr.nancumprod(fr.array(ZN)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp)


def test_nancumsum_complex64_width():
    a = np.asarray(Z, dtype=np.complex64)
    exp = np.nancumsum(a)
    got = _np(fr.nancumsum(fr.array(a)))
    assert got.dtype == np.complex64, f"width not preserved: {got.dtype}"
    np.testing.assert_allclose(got, exp, rtol=1e-6)


# ---------------------------------------------------------------------------
# meshgrid: dtype-passthrough (complex grid)
# ---------------------------------------------------------------------------

def test_meshgrid_complex():
    exp0, exp1 = np.meshgrid(_np(Z), _np(Z))
    g = fr.meshgrid(fr.array(Z), fr.array(Z))
    g0, g1 = _np(g[0]), _np(g[1])
    assert np.iscomplexobj(g0), f"meshgrid dropped imaginary -> {g0.dtype}"
    np.testing.assert_allclose(g0, exp0)
    np.testing.assert_allclose(g1, exp1)


def test_meshgrid_complex_ij():
    exp0, exp1 = np.meshgrid(_np(Z), _np([1 + 1j, 2 - 2j]), indexing="ij")
    g = fr.meshgrid(fr.array(Z), fr.array([1 + 1j, 2 - 2j]), indexing="ij")
    np.testing.assert_allclose(_np(g[0]), exp0)
    np.testing.assert_allclose(_np(g[1]), exp1)


def test_meshgrid_complex64_width():
    a = np.asarray(Z, dtype=np.complex64)
    exp0 = np.meshgrid(a, a)[0]
    g0 = _np(fr.meshgrid(fr.array(a), fr.array(a))[0])
    assert g0.dtype == np.complex64, f"width not preserved: {g0.dtype}"
    np.testing.assert_allclose(g0, exp0, rtol=1e-6)


# ---------------------------------------------------------------------------
# linalg.eig / eigvals: complex eigendecomposition (order not canonical -> sort)
# ---------------------------------------------------------------------------

def test_eigvals_complex():
    exp = _sortc(np.linalg.eigvals(_np(M)))
    got = _sortc(fr.linalg.eigvals(fr.array(M)))
    np.testing.assert_allclose(got, exp)


def test_eig_eigenvalues_complex():
    exp = _sortc(np.linalg.eig(_np(M)).eigenvalues)
    fr_eig = fr.linalg.eig(fr.array(M))
    vals = fr_eig.eigenvalues if hasattr(fr_eig, "eigenvalues") else fr_eig[0]
    got = _sortc(vals)
    np.testing.assert_allclose(got, exp)


def test_eig_av_equals_lambda_v():
    # A v == λ v for each returned eigenpair (independent of numpy's vector phase).
    Mn = _np(M)
    fr_eig = fr.linalg.eig(fr.array(M))
    w = _np(fr_eig.eigenvalues if hasattr(fr_eig, "eigenvalues") else fr_eig[0])
    v = _np(fr_eig.eigenvectors if hasattr(fr_eig, "eigenvectors") else fr_eig[1])
    for j in range(w.shape[0]):
        np.testing.assert_allclose(Mn @ v[:, j], w[j] * v[:, j], atol=1e-9)


# ---------------------------------------------------------------------------
# RAISE-where-numpy-raises: unwrap / digitize (TypeError), histogram (IndexError)
# ---------------------------------------------------------------------------

def test_unwrap_complex_raises():
    with pytest.raises(TypeError):
        np.unwrap(_np(Z))  # confirm numpy raises TypeError (oracle)
    with pytest.raises(TypeError):
        fr.unwrap(fr.array(Z))


def test_digitize_complex_raises():
    with pytest.raises(TypeError):
        np.digitize(_np(Z), _np([0.0, 1.0, 2.0]))
    with pytest.raises(TypeError):
        fr.digitize(fr.array(Z), fr.array([0.0, 1.0, 2.0]))


def test_histogram_complex_raises():
    with pytest.raises(IndexError):
        np.histogram(_np(Z))
    with pytest.raises(IndexError):
        fr.histogram(fr.array(Z))


# ---------------------------------------------------------------------------
# real path UNCHANGED (no regression from the complex branches)
# ---------------------------------------------------------------------------

def test_real_nan_reductions_unchanged():
    a = [1.0, np.nan, 3.0, 5.0]
    assert float(_np(fr.nanmin(fr.array(a)))) == np.nanmin(a)
    assert float(_np(fr.nanmax(fr.array(a)))) == np.nanmax(a)
    assert float(_np(fr.nanmedian(fr.array(a)))) == np.nanmedian(a)
    assert float(_np(fr.nanvar(fr.array(a)))) == pytest.approx(np.nanvar(a))
    np.testing.assert_allclose(_np(fr.nancumsum(fr.array(a))), np.nancumsum(a))


def test_real_meshgrid_unchanged():
    exp0, exp1 = np.meshgrid([1.0, 2.0, 3.0], [4.0, 5.0])
    g = fr.meshgrid(fr.array([1.0, 2.0, 3.0]), fr.array([4.0, 5.0]))
    np.testing.assert_allclose(_np(g[0]), exp0)
    np.testing.assert_allclose(_np(g[1]), exp1)


def test_real_eigvals_unchanged():
    R = [[2.0, 0.0], [0.0, 3.0]]
    exp = _sortc(np.linalg.eigvals(_np(R)))
    got = _sortc(fr.linalg.eigvals(fr.array(R)))
    np.testing.assert_allclose(got, exp)
