"""Phase-4 parity tests for ferray.polynomial (function-style API)."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# polyvalfromroots / polyval2d / polyval3d
# ---------------------------------------------------------------------------


def test_polyvalfromroots_scalar():
    # (x - 1)(x - 2)(x - 3) at x=4 = 3 * 2 * 1 = 6
    assert ferray.polynomial.polyvalfromroots(4.0, [1.0, 2.0, 3.0]) == pytest.approx(6.0)


def test_polyval2d_simple():
    # f(x, y) = 1 + 2x + 3y + 4xy → coeffs (nx=2, ny=2) row-major
    # At (1, 1): 1 + 2 + 3 + 4 = 10.
    out = ferray.polynomial.polyval2d(
        x=[1.0],
        y=[1.0],
        coeffs=[1.0, 3.0, 2.0, 4.0],
        nx=2,
        ny=2,
    )
    assert out.shape == (1,)
    assert out[0] == pytest.approx(10.0)


def test_polyval3d_returns_correct_length():
    out = ferray.polynomial.polyval3d(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0, 1.0],
        coeffs=[1.0] * 8,
        nx=2,
        ny=2,
        nz=2,
    )
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# polygrid2d / polygrid3d
# ---------------------------------------------------------------------------


def test_polygrid2d_shape():
    out = ferray.polynomial.polygrid2d(
        x=[0.0, 0.5, 1.0],
        y=[0.0, 1.0],
        coeffs=[1.0, 0.0, 0.0, 1.0],
        nx=2,
        ny=2,
    )
    assert out.shape == (3, 2)


def test_polygrid3d_shape():
    out = ferray.polynomial.polygrid3d(
        x=[0.0, 1.0],
        y=[0.0, 1.0],
        z=[0.0, 1.0],
        coeffs=[1.0] * 8,
        nx=2,
        ny=2,
        nz=2,
    )
    assert out.shape == (2, 2, 2)


# ---------------------------------------------------------------------------
# polyvander2d / polyvander3d
# ---------------------------------------------------------------------------


def test_polyvander2d_shape():
    n = 5
    degx, degy = 2, 3
    out = ferray.polynomial.polyvander2d(
        x=list(np.linspace(0.0, 1.0, n)),
        y=list(np.linspace(0.0, 1.0, n)),
        degx=degx,
        degy=degy,
    )
    assert out.shape == (n, (degx + 1) * (degy + 1))


def test_polyvander3d_shape():
    n = 4
    degx, degy, degz = 2, 2, 2
    out = ferray.polynomial.polyvander3d(
        x=list(np.linspace(0.0, 1.0, n)),
        y=list(np.linspace(0.0, 1.0, n)),
        z=list(np.linspace(0.0, 1.0, n)),
        degx=degx,
        degy=degy,
        degz=degz,
    )
    assert out.shape == (n, (degx + 1) * (degy + 1) * (degz + 1))


# ---------------------------------------------------------------------------
# Chebyshev points / weights
# ---------------------------------------------------------------------------


def test_chebpts1_matches_numpy():
    np.testing.assert_allclose(
        ferray.polynomial.chebpts1(5),
        np.polynomial.chebyshev.chebpts1(5),
        atol=1e-12,
    )


def test_chebpts2_matches_numpy():
    np.testing.assert_allclose(
        ferray.polynomial.chebpts2(5),
        np.polynomial.chebyshev.chebpts2(5),
        atol=1e-12,
    )


def test_chebweight_at_zero():
    # 1 / sqrt(1 - 0) = 1.
    assert ferray.polynomial.chebweight(0.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gauss quadrature
# ---------------------------------------------------------------------------


def test_chebgauss_matches_numpy():
    pts, weights = ferray.polynomial.chebgauss(5)
    e_pts, e_weights = np.polynomial.chebyshev.chebgauss(5)
    np.testing.assert_allclose(np.sort(pts), np.sort(e_pts), atol=1e-12)
    np.testing.assert_allclose(weights, e_weights, atol=1e-12)


def test_leggauss_matches_numpy():
    pts, weights = ferray.polynomial.leggauss(5)
    e_pts, e_weights = np.polynomial.legendre.leggauss(5)
    np.testing.assert_allclose(np.sort(pts), np.sort(e_pts), atol=1e-10)
    np.testing.assert_allclose(np.sort(weights), np.sort(e_weights), atol=1e-10)


def test_hermgauss_returns_tuple_of_n():
    pts, weights = ferray.polynomial.hermgauss(4)
    assert pts.shape == (4,)
    assert weights.shape == (4,)


def test_laggauss_returns_tuple_of_n():
    pts, weights = ferray.polynomial.laggauss(4)
    assert pts.shape == (4,)
    assert weights.shape == (4,)


# ---------------------------------------------------------------------------
# Basis conversions: power ↔ chebyshev / hermite / hermite-e / laguerre / legendre
# ---------------------------------------------------------------------------


def test_poly2cheb_round_trips():
    src = [1.0, 2.0, 3.0, 4.0]
    cheb = ferray.polynomial.poly2cheb(src)
    back = ferray.polynomial.cheb2poly(cheb.tolist())
    np.testing.assert_allclose(back, src, atol=1e-10)


def test_poly2leg_round_trips():
    src = [1.0, -1.0, 2.0]
    leg = ferray.polynomial.poly2leg(src)
    back = ferray.polynomial.leg2poly(leg.tolist())
    np.testing.assert_allclose(back, src, atol=1e-10)


def test_poly2herm_round_trips():
    src = [0.5, 1.0, -1.0]
    herm = ferray.polynomial.poly2herm(src)
    back = ferray.polynomial.herm2poly(herm.tolist())
    np.testing.assert_allclose(back, src, atol=1e-10)


def test_poly2herme_round_trips():
    src = [1.0, 0.0, -2.0]
    he = ferray.polynomial.poly2herme(src)
    back = ferray.polynomial.herme2poly(he.tolist())
    np.testing.assert_allclose(back, src, atol=1e-10)


def test_poly2lag_round_trips():
    src = [1.0, 2.0, 1.0]
    lag = ferray.polynomial.poly2lag(src)
    back = ferray.polynomial.lag2poly(lag.tolist())
    np.testing.assert_allclose(back, src, atol=1e-10)
