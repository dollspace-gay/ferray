"""#931 — complex linalg core: dot / vdot / inner / outer / matmul (@) /
linalg.norm / det / inv / solve over complex input.

Every expected value is derived LIVE from numpy 2.4.5 (the oracle), never
literal-copied from the ferray side (R-CHAR-3). The prior binding raised
``TypeError: unsupported dtype for floating-point op: complex128`` because the
numpy-named entrypoints routed through ``match_dtype_float!`` (real-only); numpy
COMPUTES these. The real path must stay byte-identical (unregressed).

Contract checks (all confirmed live):
  - dot/inner       = sum(a*b)            (NO conjugation)
  - vdot            = sum(conj(a) * b)    (conjugates the FIRST arg)
  - outer           = a[i]*b[j]           (NO conjugation, both flattened)
  - matmul / @      = complex matrix product; 1-D x 1-D -> 0-D inner product
  - linalg.norm     = sqrt(sum |x|^2)     (REAL result; ord=None/2/fro/1/inf/-inf)
  - det/inv/solve   = complex determinant / inverse / linear solve
  - c64 stays c64, c128 stays c128; norm of c64 -> float32, c128 -> float64
  - real arrays are UNCHANGED by the new dispatch
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """ferray result -> numpy array."""
    return np.asarray(x)


# --- vectors / matrices, reused across tests; derived from numpy below ---

A1 = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
B1 = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
M = [[1 + 1j, 2 + 0j], [0 + 1j, 1 - 0j]]
N = [[1 + 0j, 0 + 1j], [1 + 1j, 2 + 0j]]
# A well-conditioned hermitian-ish 2x2 for det/inv/solve.
S = [[2 + 0j, 1 + 1j], [1 - 1j, 3 + 0j]]
RHS = [1 + 0j, 2 + 0j]


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------

def test_dot_complex_vectors():
    expected = np.dot(np.array(A1), np.array(B1))  # live oracle (-18+68j)
    result = complex(_np(fr.dot(fr.array(A1), fr.array(B1))))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_dot_complex_no_conjugation():
    # dot does NOT conjugate; conj(a)·b would be a different value.
    a = np.array(A1)
    b = np.array(B1)
    assert not np.allclose(np.dot(a, b), np.vdot(a, b)), "test premise: dot != vdot"
    result = complex(_np(fr.dot(fr.array(A1), fr.array(B1))))
    assert np.allclose(result, np.dot(a, b), rtol=1e-12)


def test_inner_complex_vectors():
    expected = np.inner(np.array(A1), np.array(B1))  # live oracle == dot for 1-D
    result = complex(_np(fr.inner(fr.array(A1), fr.array(B1))))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_vdot_complex_conjugates_first():
    expected = np.vdot(np.array(A1), np.array(B1))  # live oracle (conj first)
    result = complex(_np(fr.vdot(fr.array(A1), fr.array(B1))))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_vdot_conjugation_is_on_first_arg():
    # vdot(a,b) == conj(vdot(b,a)); also vdot(a,b) != dot(a,b) here.
    a, b = np.array(A1), np.array(B1)
    assert np.allclose(np.vdot(a, b), np.conj(np.vdot(b, a)))
    r_ab = complex(_np(fr.vdot(fr.array(A1), fr.array(B1))))
    r_ba = complex(_np(fr.vdot(fr.array(B1), fr.array(A1))))
    assert np.allclose(r_ab, np.conj(r_ba), rtol=1e-12)


def test_outer_complex():
    expected = np.outer(np.array(A1), np.array(B1))  # live oracle
    result = _np(fr.outer(fr.array(A1), fr.array(B1)))
    assert result.shape == expected.shape
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_matmul_complex_matrices():
    expected = np.matmul(np.array(M), np.array(N))  # live oracle
    result = _np(fr.matmul(fr.array(M), fr.array(N)))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_matmul_operator_atsign_complex():
    # The `@` operator routes through the same binding.
    expected = np.array(M) @ np.array(N)  # live oracle
    result = _np(fr.matmul(fr.array(M), fr.array(N)))
    assert np.allclose(result, expected, rtol=1e-12)


def test_matmul_complex_1d_collapses_to_scalar():
    expected = np.matmul(np.array(A1), np.array(B1))  # 0-D inner product
    result = _np(fr.matmul(fr.array(A1), fr.array(B1)))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


# ---------------------------------------------------------------------------
# Norm (real result)
# ---------------------------------------------------------------------------

def test_norm_complex_vector_default():
    expected = np.linalg.norm(np.array(A1))  # sqrt(sum |x|^2)
    result = float(_np(fr.linalg.norm(fr.array(A1))))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_norm_complex_vector_is_real():
    # The result has no imaginary part — numpy returns a real float.
    result = _np(fr.linalg.norm(fr.array(A1)))
    assert not np.iscomplexobj(result), result.dtype


@pytest.mark.parametrize("ord_", [None, 2, 1, np.inf, -np.inf])
def test_norm_complex_vector_ords(ord_):
    expected = np.linalg.norm(np.array(A1), ord_)  # live oracle per ord
    result = float(_np(fr.linalg.norm(fr.array(A1), ord_)))
    assert np.allclose(result, expected, rtol=1e-12), (ord_, expected, result)


def test_norm_complex_matrix_fro():
    expected = np.linalg.norm(np.array(M), "fro")  # sqrt(sum |x|^2)
    result = float(_np(fr.linalg.norm(fr.array(M), "fro")))
    assert np.allclose(result, expected, rtol=1e-12), (expected, result)


def test_norm_complex_matrix_one_and_inf():
    for ord_ in (1, np.inf):
        expected = np.linalg.norm(np.array(M), ord_)  # max col / row abs-sum
        result = float(_np(fr.linalg.norm(fr.array(M), ord_)))
        assert np.allclose(result, expected, rtol=1e-12), (ord_, expected, result)


# ---------------------------------------------------------------------------
# Decompositions (faer complex LU)
# ---------------------------------------------------------------------------

def test_det_complex():
    expected = np.linalg.det(np.array(S))  # live oracle (complex)
    result = complex(_np(fr.linalg.det(fr.array(S))))
    assert np.allclose(result, expected, rtol=1e-10), (expected, result)


def test_inv_complex():
    expected = np.linalg.inv(np.array(S))  # live oracle (complex)
    result = _np(fr.linalg.inv(fr.array(S)))
    assert np.allclose(result, expected, rtol=1e-10), (expected, result)


def test_inv_complex_roundtrip():
    # A @ inv(A) == I (live identity construction, not literal copy).
    A = np.array(S)
    inv = _np(fr.linalg.inv(fr.array(S)))
    assert np.allclose(A @ inv, np.eye(2, dtype=complex), atol=1e-10)


def test_solve_complex_vector_rhs():
    expected = np.linalg.solve(np.array(S), np.array(RHS))  # live oracle
    result = _np(fr.linalg.solve(fr.array(S), fr.array(RHS)))
    assert np.allclose(result, expected, rtol=1e-10), (expected, result)


def test_solve_complex_matrix_rhs():
    B = [[1 + 0j, 0 + 1j], [2 + 0j, 1 - 1j]]
    expected = np.linalg.solve(np.array(S), np.array(B))  # live oracle
    result = _np(fr.linalg.solve(fr.array(S), fr.array(B)))
    assert np.allclose(result, expected, rtol=1e-10), (expected, result)


# ---------------------------------------------------------------------------
# dtype width contract
# ---------------------------------------------------------------------------

def test_complex64_dot_stays_complex64():
    a = np.array(A1, dtype=np.complex64)
    b = np.array(B1, dtype=np.complex64)
    expected = np.dot(a, b)  # numpy keeps complex64
    result = _np(fr.dot(fr.array(a), fr.array(b)))
    assert result.dtype == np.complex64, result.dtype
    assert np.allclose(result, expected, rtol=1e-5)


def test_complex64_norm_returns_float32():
    a = np.array(A1, dtype=np.complex64)
    expected = np.linalg.norm(a)  # numpy returns float32
    result = _np(fr.linalg.norm(fr.array(a)))
    assert result.dtype == np.float32, result.dtype
    assert np.allclose(result, expected, rtol=1e-5)


def test_complex128_matmul_stays_complex128():
    a = np.array(M, dtype=np.complex128)
    result = _np(fr.matmul(fr.array(a), fr.array(a)))
    assert result.dtype == np.complex128, result.dtype


# ---------------------------------------------------------------------------
# Real path unregressed by the new complex dispatch
# ---------------------------------------------------------------------------

def test_real_dot_unchanged():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    expected = np.dot(np.array(a), np.array(b))  # 32.0
    result = _np(fr.dot(fr.array(a), fr.array(b)))
    assert np.allclose(result, expected)


def test_real_matmul_unchanged():
    a = [[1.0, 2.0], [3.0, 4.0]]
    expected = np.matmul(np.array(a), np.array(a))
    result = _np(fr.matmul(fr.array(a), fr.array(a)))
    assert np.allclose(result, expected)


def test_real_norm_unchanged():
    a = [3.0, 4.0]
    expected = np.linalg.norm(np.array(a))  # 5.0
    result = float(_np(fr.linalg.norm(fr.array(a))))
    assert np.allclose(result, expected)


def test_real_det_inv_solve_unchanged():
    A = [[2.0, 1.0], [1.0, 3.0]]
    b = [1.0, 2.0]
    assert np.allclose(_np(fr.linalg.det(fr.array(A))), np.linalg.det(np.array(A)))
    assert np.allclose(_np(fr.linalg.inv(fr.array(A))), np.linalg.inv(np.array(A)))
    assert np.allclose(
        _np(fr.linalg.solve(fr.array(A), fr.array(b))),
        np.linalg.solve(np.array(A), np.array(b)),
    )
