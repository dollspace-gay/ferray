"""Adversarial divergence tests: ferray.linalg vs the numpy oracle.

Each test pins a CONTRACT divergence (not a ULP difference) where
``import ferray as fr`` fails to match ``numpy`` 2.4.x. Oracle is the live
installed numpy. Values use np.allclose; shapes/dtypes/exception types are
exact. Upstream citations reference /home/doll/numpy-ref/numpy/linalg/_linalg.py.

These tests are EXPECTED TO FAIL on current ferray (they pin bugs).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Top-level vector/matrix product API surface.
# numpy/linalg/_linalg.py imports `dot`, `matmul`, `outer`, `tensordot`,
# `vecdot` into numpy._core; `np.dot`, `np.vdot`, `np.matmul`, `np.inner`,
# `np.outer` are top-level numpy functions a drop-in replacement must expose.
# (numpy public API: numpy.dot / numpy.vdot / numpy.matmul / numpy.inner /
#  numpy.outer — all top-level, not under numpy.linalg.)
# ---------------------------------------------------------------------------


def test_top_level_dot_exists_and_matches():
    """`np.dot` is a top-level numpy function; `fr.dot` must mirror it."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert hasattr(fr, "dot"), "ferray is missing top-level fr.dot"
    out = float(fr.dot(a, b))
    assert np.isclose(out, float(np.dot(a, b)))


def test_top_level_matmul_exists_and_matches():
    """`np.matmul` is top-level; `fr.matmul` must exist and match."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert hasattr(fr, "matmul"), "ferray is missing top-level fr.matmul"
    np.testing.assert_allclose(np.asarray(fr.matmul(a, b)), np.matmul(a, b))


def test_top_level_vdot_exists_and_matches():
    """`np.vdot` is a top-level numpy function; `fr.vdot` must mirror it."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert hasattr(fr, "vdot"), "ferray is missing top-level fr.vdot"
    assert np.isclose(float(fr.vdot(a, b)), float(np.vdot(a, b)))


# ---------------------------------------------------------------------------
# qr mode='r' contract.
# _linalg.py:1142-1145:
#     if mode == 'r':
#         r = triu(a[..., :mn, :])
#         r = r.astype(result_t, copy=False)
#         return wrap(r)
# mode='r' returns a SINGLE ndarray R, NOT a (Q, R) tuple.
# ---------------------------------------------------------------------------


def test_qr_mode_r_returns_single_r_matrix():
    """qr(a, mode='r') must return just R (an ndarray), per _linalg.py:1142."""
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_r = np.linalg.qr(a, mode="r")  # ndarray, shape (2, 2)
    got = fr.linalg.qr(a, mode="r")
    # numpy returns a bare ndarray here, not a tuple.
    assert not isinstance(got, tuple), (
        "qr(mode='r') must return a single R ndarray, not a (Q, R) tuple"
    )
    got_arr = np.asarray(got)
    assert got_arr.shape == expected_r.shape
    # R from QR is unique only up to row signs; compare |R| to avoid sign LARP.
    np.testing.assert_allclose(np.abs(got_arr), np.abs(expected_r), atol=1e-10)


# ---------------------------------------------------------------------------
# svd compute_uv=False contract.
# _linalg.py:1668: def svd(a, full_matrices=True, compute_uv=True, hermitian=False)
# _linalg.py:1846-... compute_uv=False returns ONLY the singular values (1-D).
# ---------------------------------------------------------------------------


def test_svd_compute_uv_false_returns_singular_values_only():
    """svd(compute_uv=False) returns the 1-D singular values, per _linalg.py:1668."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_s = np.linalg.svd(a, compute_uv=False)
    got = fr.linalg.svd(a, compute_uv=False)
    got_arr = np.asarray(got)
    assert got_arr.ndim == 1, "compute_uv=False must yield a 1-D singular-value array"
    np.testing.assert_allclose(got_arr, expected_s, atol=1e-10)


# ---------------------------------------------------------------------------
# Singular-matrix exception type.
# _linalg.py:115: class LinAlgError(ValueError)
# _linalg.py:145: raise LinAlgError("Singular matrix")
# inv/solve on a singular matrix raise numpy.linalg.LinAlgError, which a
# drop-in replacement must expose as fr.linalg.LinAlgError and raise.
# ---------------------------------------------------------------------------


def test_linalg_error_symbol_exists():
    """numpy.linalg.LinAlgError is public (_linalg.py:115); ferray must expose it."""
    assert hasattr(fr.linalg, "LinAlgError"), "fr.linalg.LinAlgError is missing"


def test_inv_singular_raises_linalgerror():
    """inv of a singular matrix raises LinAlgError, per _linalg.py:145."""
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1, singular
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.inv(sing)


def test_solve_singular_raises_linalgerror():
    """solve with a singular A raises LinAlgError, per _linalg.py:145."""
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.solve(sing, b)


def test_inv_nonsquare_raises_linalgerror():
    """inv of a non-square matrix raises LinAlgError, per _linalg.py:259."""
    ns = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.inv(ns)


# ---------------------------------------------------------------------------
# Integer-input dtype promotion.
# _linalg.py:2356 det / det dispatch goes through _commonType, which promotes
# integer inputs to float64. det/trace/inv of an int array must NOT raise;
# they return float64 results.
# ---------------------------------------------------------------------------


def test_det_integer_input_promotes_to_float64():
    """det of an int matrix promotes to float64 (via _commonType), not raise."""
    ai = np.array([[1, 2], [3, 4]])
    expected = np.linalg.det(ai)
    got = fr.linalg.det(ai)
    assert np.isclose(float(got), float(expected))
    assert np.asarray(got).dtype == np.float64


def test_inv_integer_input_promotes_to_float64():
    """inv of an int matrix promotes to float64, per _commonType."""
    ai = np.array([[1, 2], [3, 5]])  # det = -1, invertible
    expected = np.linalg.inv(ai)
    got = np.asarray(fr.linalg.inv(ai))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Matrix norm with negative integer ord.
# _linalg.py norm(): ord=-1 on a 2-D matrix is the MIN column abs-sum, a valid
# matrix norm — not a vector p-norm. numpy returns a finite scalar.
# ---------------------------------------------------------------------------


def test_matrix_norm_ord_neg1():
    """norm(A, ord=-1) on a 2-D matrix = min column abs-sum (numpy norm())."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.linalg.norm(a, ord=-1)
    got = float(fr.linalg.norm(a, ord=-1))
    assert np.isclose(got, float(expected))


def test_matrix_norm_ord_neg2():
    """norm(A, ord=-2) on a 2-D matrix = smallest singular value (numpy norm())."""
    a = np.array([[3.0, 0.0], [0.0, 1.0]])
    expected = np.linalg.norm(a, ord=-2)
    got = float(fr.linalg.norm(a, ord=-2))
    assert np.isclose(got, float(expected))
