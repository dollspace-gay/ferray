"""Parity tests for the top-level numpy poly1d family in ferray.

These cover the CLASSIC numpy 1-D polynomial functions on highest-degree-first
coefficient arrays (numpy/lib/_polynomial_impl.py): np.polyval / poly / roots /
polyadd / polysub / polymul / polyder / polyint / polyfit / polydiv. They are
distinct from numpy.polynomial.polynomial.* which is lowest-first.

Every expected value is produced by a LIVE numpy call (R-CHAR-3); nothing is
literal-copied from the ferray side. numpy.roots returns roots in raw
eigenvalue order (it does NOT sort), so root tests compare as a sorted set.
"""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# polyval — highest-first Horner evaluation
# ---------------------------------------------------------------------------


def test_polyval_scalar():
    # np.polyval([1,2,3], 2) == 2**2 + 2*2 + 3 == 11
    assert ferray.polyval([1, 2, 3], 2) == pytest.approx(np.polyval([1, 2, 3], 2))


def test_polyval_array_x():
    expected = np.polyval([1, 2, 3], [0, 1, 2, 3])
    got = ferray.polyval([1, 2, 3], [0, 1, 2, 3])
    assert np.allclose(got, expected)
    assert np.asarray(got).shape == expected.shape


def test_polyval_negative_and_float():
    p = [2.5, -1.0, 0.0, 3.0]
    for x in (-2.0, 0.0, 1.5, 10.0):
        assert ferray.polyval(p, x) == pytest.approx(np.polyval(p, x))


def test_polyval_2d_x_shape():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    expected = np.polyval([1, 0, -1], x)
    got = ferray.polyval([1, 0, -1], x)
    assert np.allclose(got, expected)
    assert np.asarray(got).shape == expected.shape


# ---------------------------------------------------------------------------
# poly — monic coefficients from (real) roots
# ---------------------------------------------------------------------------


def test_poly_real_roots():
    expected = np.poly([1, 2])
    got = ferray.poly([1, 2])
    assert np.allclose(got, expected)
    assert np.asarray(got).shape == expected.shape


def test_poly_three_roots():
    expected = np.poly([1.0, -2.0, 3.0])
    got = ferray.poly([1.0, -2.0, 3.0])
    assert np.allclose(got, expected)


def test_poly_empty_returns_one():
    expected = np.poly([])
    got = ferray.poly([])
    assert np.asarray(got) == pytest.approx(np.asarray(expected))


# ---------------------------------------------------------------------------
# roots — companion-matrix eigenvalues (compared as a sorted set)
# ---------------------------------------------------------------------------


def _sorted_complex(a):
    a = np.asarray(a, dtype=complex)
    return a[np.lexsort((a.imag, a.real))]


def test_roots_quadratic_real():
    expected = np.roots([1, -3, 2])
    got = ferray.roots([1, -3, 2])
    assert np.allclose(_sorted_complex(got), _sorted_complex(expected))


def test_roots_complex_pair():
    expected = np.roots([1, 0, 1])  # x**2 + 1 -> +/- i
    got = ferray.roots([1, 0, 1])
    assert np.allclose(_sorted_complex(got), _sorted_complex(expected))


def test_roots_cubic():
    expected = np.roots([1, -6, 11, -6])  # (x-1)(x-2)(x-3)
    got = ferray.roots([1, -6, 11, -6])
    assert np.allclose(_sorted_complex(got), _sorted_complex(expected))


def test_roots_trailing_zeros_add_zero_roots():
    # np.roots([1,-3,2,0]) == roots of (x^2-3x+2)*x -> {0,1,2}
    expected = np.roots([1, -3, 2, 0])
    got = ferray.roots([1, -3, 2, 0])
    assert np.allclose(_sorted_complex(got), _sorted_complex(expected))


def test_roots_all_zero_is_empty():
    expected = np.roots([0, 0, 0])
    got = ferray.roots([0, 0, 0])
    assert np.asarray(got).shape == expected.shape == (0,)


def test_roots_linear():
    expected = np.roots([1, 2])
    got = ferray.roots([1, 2])
    assert np.allclose(_sorted_complex(got), _sorted_complex(expected))


# ---------------------------------------------------------------------------
# polyadd / polysub / polymul — highest-first, trimmed
# ---------------------------------------------------------------------------


def test_polyadd():
    assert np.allclose(ferray.polyadd([1, 2], [1, 1]), np.polyadd([1, 2], [1, 1]))


def test_polyadd_trims_leading_cancellation():
    expected = np.polyadd([1, 2, 3], [-1, -2])
    got = ferray.polyadd([1, 2, 3], [-1, -2])
    assert np.allclose(got, expected)
    assert np.asarray(got).shape == expected.shape


def test_polysub():
    assert np.allclose(ferray.polysub([1, 2], [1, 1]), np.polysub([1, 2], [1, 1]))


def test_polymul():
    expected = np.polymul([1, 2], [1, 3])
    got = ferray.polymul([1, 2], [1, 3])
    assert np.allclose(got, expected)
    assert np.asarray(got).shape == expected.shape


def test_polymul_longer():
    a, b = [2.0, -1.0, 3.0], [1.0, 0.0, -2.0, 1.0]
    assert np.allclose(ferray.polymul(a, b), np.polymul(a, b))


# ---------------------------------------------------------------------------
# polyder / polyint — derivative and antiderivative
# ---------------------------------------------------------------------------


def test_polyder():
    expected = np.polyder([1, 2, 3])
    got = ferray.polyder([1, 2, 3])
    assert np.allclose(got, expected)


def test_polyder_m2():
    expected = np.polyder([1, 2, 3, 4], 2)
    got = ferray.polyder([1, 2, 3, 4], 2)
    assert np.allclose(got, expected)


def test_polyint_default():
    expected = np.polyint([3, 2, 1])
    got = ferray.polyint([3, 2, 1])
    assert np.allclose(got, expected)


def test_polyint_with_k():
    expected = np.polyint([1, 1, 1], m=1, k=[5])
    got = ferray.polyint([1, 1, 1], m=1, k=[5])
    assert np.allclose(got, expected)


def test_polyint_m2_with_k():
    expected = np.polyint([2], m=2, k=[1, 2])
    got = ferray.polyint([2], m=2, k=[1, 2])
    assert np.allclose(got, expected)


def test_polyder_then_polyint_roundtrip():
    p = [2.0, -3.0, 0.0, 4.0, 1.0]
    # d/dx of the antiderivative recovers p (up to numpy's own roundtrip).
    expected = np.polyder(np.polyint(p))
    got = ferray.polyder(ferray.polyint(p))
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# polyfit — least-squares fit (highest-first)
# ---------------------------------------------------------------------------


def test_polyfit_quadratic():
    x = [0.0, 1.0, 2.0]
    y = [1.0, 2.0, 5.0]
    expected = np.polyfit(x, y, 2)
    got = ferray.polyfit(x, y, 2)
    assert np.allclose(got, expected, atol=1e-9)


def test_polyfit_cubic_recovers_polynomial():
    x = np.linspace(-3, 3, 25)
    true = [0.5, -1.0, 2.0, 3.0]
    y = np.polyval(true, x)
    expected = np.polyfit(x, y, 3)
    got = ferray.polyfit(list(x), list(y), 3)
    assert np.allclose(got, expected, atol=1e-6)
    assert np.allclose(got, true, atol=1e-6)


# ---------------------------------------------------------------------------
# polydiv — (quotient, remainder) tuple
# ---------------------------------------------------------------------------


def test_polydiv():
    eq, er = np.polydiv([1, 0, -1], [1, 1])
    gq, gr = ferray.polydiv([1, 0, -1], [1, 1])
    assert np.allclose(gq, eq)
    assert np.allclose(gr, er)


def test_polydiv_exact():
    # (x^2 - 3x + 2) / (x - 1) = (x - 2) remainder 0
    eq, er = np.polydiv([1, -3, 2], [1, -1])
    gq, gr = ferray.polydiv([1, -3, 2], [1, -1])
    assert np.allclose(gq, eq)
    assert np.allclose(gr, er)


def test_polydiv_with_remainder():
    eq, er = np.polydiv([1, 2, 3, 4], [1, 1])
    gq, gr = ferray.polydiv([1, 2, 3, 4], [1, 1])
    assert np.allclose(gq, eq)
    assert np.allclose(gr, er)


# ---------------------------------------------------------------------------
# convention guard: top-level poly* is highest-first, opposite to
# numpy.polynomial.polynomial.* (lowest-first)
# ---------------------------------------------------------------------------


def test_highest_first_convention_differs_from_lowest_first():
    # np.polyval([1,2,3], 2) -> 1*4 + 2*2 + 3 = 11 (highest-first)
    # np.polynomial.polynomial.polyval(2, [1,2,3]) -> 1 + 2*2 + 3*4 = 17
    assert ferray.polyval([1, 2, 3], 2) == pytest.approx(11.0)
    assert ferray.polyval([1, 2, 3], 2) != pytest.approx(17.0)
