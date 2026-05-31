"""Expansion suite (#939): complex linalg decompositions — svd / qr / pinv /
matrix_rank / slogdet / matrix_power over a complex-valued matrix.

ferray previously raised TypeError for these (the real `match_dtype_float!`
path had no complex arm); numpy computes complex. Every expected value is a
LIVE numpy 2.4 call (R-CHAR-3), never copied from ferray.

SVD U/Vh phase and QR Q/R sign are NOT canonical, so those are verified by
RECONSTRUCTION (`U @ diag(s) @ Vh ≈ A`, `Q @ R ≈ A`, `Q^H @ Q ≈ I`) and the
canonical singular values `s` are compared directly. matrix_rank / slogdet /
matrix_power / pinv are canonical and compared value-by-value.

Real arrays must be UNCHANGED (regression guard at the bottom).

    cd ferray-python && maturin develop && \
      PYTHONPATH=python python3 -m pytest tests/test_expansion_complex_decomp.py -q
"""
import numpy as np
import ferray as fr
import pytest


def _np(x):
    return np.asarray(x)


# A square non-Hermitian complex matrix, plus a tall and a rank-deficient one.
M = [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]]
TALL = [[1 + 1j, 2 + 0j], [0 + 1j, 3 - 1j], [1 + 0j, 0 + 2j]]
RANK1 = [[1 + 1j, 2 + 2j], [2 + 2j, 4 + 4j]]  # row 2 == 2*row 1 -> rank 1


# --------------------------------------------------------------------------
# SVD — complex U / Vh, REAL singular values; verify by reconstruction + s.
# --------------------------------------------------------------------------

def _svd_parts(res):
    if hasattr(res, "U"):
        return res.U, res.S, res.Vh
    return res[0], res[1], res[2]


@pytest.mark.parametrize("A", [M, TALL])
def test_svd_singular_values_and_reconstruct(A):
    An = _np(A)
    exp_s = np.linalg.svd(An).S
    u, s, vh = _svd_parts(fr.linalg.svd(fr.array(A)))
    u, s, vh = _np(u), _np(s), _np(vh)
    # Singular values are canonical (descending, real).
    np.testing.assert_allclose(np.sort(s)[::-1], np.sort(exp_s)[::-1], rtol=1e-6, atol=1e-9)
    assert not np.iscomplexobj(s), "singular values must be REAL"
    # Reconstruction (full_matrices=True default): A == U[:, :k] diag(s) Vh[:k, :]
    k = min(An.shape)
    recon = u[:, :k] @ np.diag(s) @ vh[:k, :]
    np.testing.assert_allclose(recon, An, rtol=1e-6, atol=1e-9)


def test_svd_u_unitary():
    u, _s, vh = _svd_parts(fr.linalg.svd(fr.array(M)))
    u, vh = _np(u), _np(vh)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(u.shape[1]), atol=1e-9)
    np.testing.assert_allclose(vh @ vh.conj().T, np.eye(vh.shape[0]), atol=1e-9)


def test_svd_thin_shapes_and_reconstruct():
    An = _np(TALL)
    res = fr.linalg.svd(fr.array(TALL), full_matrices=False)
    u, s, vh = _svd_parts(res)
    u, s, vh = _np(u), _np(s), _np(vh)
    k = min(An.shape)
    assert u.shape == (An.shape[0], k)
    assert vh.shape == (k, An.shape[1])
    np.testing.assert_allclose(u @ np.diag(s) @ vh, An, rtol=1e-6, atol=1e-9)


def test_svd_compute_uv_false():
    exp_s = np.linalg.svd(_np(M), compute_uv=False)
    got = _np(fr.linalg.svd(fr.array(M), compute_uv=False))
    assert not np.iscomplexobj(got)
    np.testing.assert_allclose(np.sort(got)[::-1], np.sort(exp_s)[::-1], rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# QR — complex unitary Q, complex R; verify by reconstruction + Q^H Q == I.
# --------------------------------------------------------------------------

def _qr_parts(res):
    if hasattr(res, "Q"):
        return res.Q, res.R
    return res[0], res[1]


@pytest.mark.parametrize("A", [M, TALL])
def test_qr_reconstruct_and_unitary(A):
    An = _np(A)
    res = fr.linalg.qr(fr.array(A))
    q, r = _qr_parts(res)
    q, r = _np(q), _np(r)
    np.testing.assert_allclose(q @ r, An, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(q.conj().T @ q, np.eye(q.shape[1]), atol=1e-9)
    # R upper-triangular.
    np.testing.assert_allclose(np.tril(r, -1), np.zeros_like(r), atol=1e-9)


def test_qr_mode_r():
    rn = np.linalg.qr(_np(M), mode="r")
    rf = _np(fr.linalg.qr(fr.array(M), mode="r"))
    # |R| diagonal magnitudes are canonical even though sign/phase isn't.
    np.testing.assert_allclose(np.abs(np.diag(rf)), np.abs(np.diag(rn)), rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# pinv / matrix_rank / slogdet / matrix_power — canonical, compare values.
# --------------------------------------------------------------------------

def test_pinv_matches_numpy():
    exp = np.linalg.pinv(_np(M))
    got = _np(fr.linalg.pinv(fr.array(M)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-9)


def test_pinv_pseudoinverse_property():
    # A @ pinv(A) @ A == A for any A.
    An = _np(TALL)
    p = _np(fr.linalg.pinv(fr.array(TALL)))
    np.testing.assert_allclose(An @ p @ An, An, rtol=1e-6, atol=1e-9)


def test_matrix_rank_full():
    assert int(fr.linalg.matrix_rank(fr.array(M))) == int(np.linalg.matrix_rank(_np(M)))


def test_matrix_rank_deficient():
    exp = np.linalg.matrix_rank(_np(RANK1))  # 1
    assert int(fr.linalg.matrix_rank(fr.array(RANK1))) == int(exp)


def _slog_parts(res):
    if hasattr(res, "sign"):
        return res.sign, res.logabsdet
    return res[0], res[1]


def test_slogdet_matches_numpy():
    sgn, ld = np.linalg.slogdet(_np(M))
    fsgn, fld = _slog_parts(fr.linalg.slogdet(fr.array(M)))
    fsgn = _np(fsgn).item()
    assert np.iscomplexobj(_np(_slog_parts(fr.linalg.slogdet(fr.array(M)))[0]))
    assert fsgn == pytest.approx(sgn)
    assert float(_np(fld)) == pytest.approx(ld)
    # Cross-check: sign * exp(logabsdet) == det.
    det = np.linalg.det(_np(M))
    assert fsgn * np.exp(float(_np(fld))) == pytest.approx(det)


@pytest.mark.parametrize("n", [0, 1, 2, 3, -1, -2])
def test_matrix_power_matches_numpy(n):
    exp = np.linalg.matrix_power(_np(M), n)
    got = _np(fr.linalg.matrix_power(fr.array(M), n))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# Regression: real arrays unchanged.
# --------------------------------------------------------------------------

R = [[4.0, 1.0], [1.0, 3.0]]


def test_real_svd_unchanged():
    exp = np.linalg.svd(_np(R)).S
    u, s, vh = _svd_parts(fr.linalg.svd(fr.array(R)))
    np.testing.assert_allclose(np.sort(_np(s))[::-1], np.sort(exp)[::-1], rtol=1e-9)
    assert not np.iscomplexobj(_np(s))


def test_real_qr_unchanged():
    q, r = _qr_parts(fr.linalg.qr(fr.array(R)))
    np.testing.assert_allclose(_np(q) @ _np(r), _np(R), atol=1e-9)
    assert not np.iscomplexobj(_np(q))


def test_real_pinv_rank_slogdet_power_unchanged():
    np.testing.assert_allclose(_np(fr.linalg.pinv(fr.array(R))), np.linalg.pinv(_np(R)), atol=1e-9)
    assert int(fr.linalg.matrix_rank(fr.array(R))) == int(np.linalg.matrix_rank(_np(R)))
    sgn, ld = np.linalg.slogdet(_np(R))
    fsgn, fld = _slog_parts(fr.linalg.slogdet(fr.array(R)))
    assert float(_np(fsgn)) == pytest.approx(sgn)
    assert float(_np(fld)) == pytest.approx(ld)
    np.testing.assert_allclose(
        _np(fr.linalg.matrix_power(fr.array(R), 3)), np.linalg.matrix_power(_np(R), 3), atol=1e-9
    )
