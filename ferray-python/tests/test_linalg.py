"""Phase-3 parity tests for ferray.linalg."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Norms / measures
# ---------------------------------------------------------------------------


def test_norm_default_is_l2():
    a = np.array([3.0, 4.0])
    assert float(ferray.linalg.norm(a)) == pytest.approx(5.0)


@pytest.mark.parametrize("ord_val", [1, 2, np.inf, -np.inf, 'fro'])
def test_norm_matches_numpy(ord_val):
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4))
    if ord_val == 'fro':
        np.testing.assert_allclose(
            float(ferray.linalg.norm(a, ord=ord_val)),
            float(np.linalg.norm(a, ord=ord_val)),
        )
    else:
        np.testing.assert_allclose(
            float(ferray.linalg.norm(a, ord=ord_val)),
            float(np.linalg.norm(a, ord=ord_val)),
            rtol=1e-10,
        )


def test_det_simple():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert float(ferray.linalg.det(a)) == pytest.approx(-2.0)


def test_det_identity():
    for n in (2, 3, 4):
        assert float(ferray.linalg.det(np.eye(n))) == pytest.approx(1.0)


def test_slogdet_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 5))
    sign, logdet = ferray.linalg.slogdet(a)
    e_sign, e_log = np.linalg.slogdet(a)
    np.testing.assert_allclose(float(sign), float(e_sign))
    np.testing.assert_allclose(float(logdet), float(e_log), rtol=1e-10)


def test_matrix_rank_full():
    a = np.eye(4)
    assert ferray.linalg.matrix_rank(a) == 4


def test_matrix_rank_deficient():
    a = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1
    assert ferray.linalg.matrix_rank(a) == 1


def test_trace_main_diag():
    a = np.arange(16).reshape(4, 4).astype(np.float64)
    assert float(ferray.linalg.trace(a)) == pytest.approx(float(np.trace(a)))


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------


def test_cholesky_round_trip():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 4))
    a = m @ m.T + 4 * np.eye(4)  # symmetric positive definite
    l = ferray.linalg.cholesky(a)
    np.testing.assert_allclose(l @ l.T, a, rtol=1e-10)


def test_qr_round_trip():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 3))
    q, r = ferray.linalg.qr(a)
    np.testing.assert_allclose(q @ r, a, rtol=1e-10)
    # Q has orthonormal columns
    np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10)


def test_qr_complete_mode():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 3))
    q, r = ferray.linalg.qr(a, mode="complete")
    assert q.shape == (5, 5)


def test_svd_reconstructs():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 3))
    u, s, vt = ferray.linalg.svd(a, full_matrices=False)
    reconstructed = u @ np.diag(s) @ vt
    np.testing.assert_allclose(reconstructed, a, rtol=1e-10)


def test_eigh_symmetric():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 4))
    a = (m + m.T) / 2  # symmetric
    w, v = ferray.linalg.eigh(a)
    # Eigenvalues are real and ascending for symmetric input.
    assert np.all(np.diff(w) >= -1e-10)
    # A v == diag(w) v
    np.testing.assert_allclose(a @ v, v @ np.diag(w), atol=1e-10)


def test_eigvalsh_matches_eigh_first_output():
    rng = np.random.default_rng(42)
    m = rng.standard_normal((4, 4))
    a = (m + m.T) / 2
    w_only = ferray.linalg.eigvalsh(a)
    w_full, _ = ferray.linalg.eigh(a)
    np.testing.assert_allclose(w_only, w_full, rtol=1e-10)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def test_solve_basic():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 10.0])
    x = ferray.linalg.solve(a, b)
    np.testing.assert_allclose(a @ x, b, rtol=1e-10)


def test_inv_round_trip():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    a_inv = ferray.linalg.inv(a)
    np.testing.assert_allclose(a @ a_inv, np.eye(4), atol=1e-10)


def test_pinv_for_square_invertible():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4)) + 4 * np.eye(4)
    pinv_a = ferray.linalg.pinv(a)
    inv_a = ferray.linalg.inv(a)
    np.testing.assert_allclose(pinv_a, inv_a, rtol=1e-8)


def test_pinv_rectangular():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((5, 3))
    pinv_a = ferray.linalg.pinv(a)
    # Defining property: A @ pinv(A) @ A == A.
    np.testing.assert_allclose(a @ pinv_a @ a, a, atol=1e-8)


def test_matrix_power_zero_is_identity():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(
        ferray.linalg.matrix_power(a, 0), np.eye(2), atol=1e-12
    )


def test_matrix_power_three():
    a = np.array([[2.0, 1.0], [1.0, 3.0]])
    np.testing.assert_allclose(
        ferray.linalg.matrix_power(a, 3), a @ a @ a, rtol=1e-10
    )


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def test_matmul_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(ferray.linalg.matmul(a, b), a @ b)


def test_dot_1d_is_scalar():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    out = ferray.linalg.dot(a, b)
    # ferray returns 0-D ndarray; numpy returns scalar
    assert float(out) == pytest.approx(32.0)


def test_vdot_flat():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    assert float(ferray.linalg.vdot(a, b)) == pytest.approx(float(np.vdot(a, b)))


def test_inner_1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    np.testing.assert_allclose(ferray.linalg.inner(a, b), np.inner(a, b))


def test_outer_1d():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0])
    np.testing.assert_allclose(ferray.linalg.outer(a, b), np.outer(a, b))


def test_kron_2d():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[0.0, 5.0], [6.0, 7.0]])
    np.testing.assert_allclose(ferray.linalg.kron(a, b), np.kron(a, b))


def test_tensordot_axes_2():
    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(120.0).reshape(4, 5, 6)
    np.testing.assert_allclose(
        ferray.linalg.tensordot(a, b, axes=2), np.tensordot(a, b, axes=2)
    )
