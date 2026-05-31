"""Tests for #717 — polynomial class API.

Cover Polynomial, Chebyshev, Hermite, HermiteE, Laguerre, Legendre.
Each class shares the same Python interface; the tests for each are
parametric so they run uniformly.
"""

import numpy as np
import pytest

from ferray.polynomial import (
    Chebyshev,
    Hermite,
    HermiteE,
    Laguerre,
    Legendre,
    Polynomial,
)

ALL_CLASSES = [
    pytest.param(Polynomial, id="Polynomial"),
    pytest.param(Chebyshev, id="Chebyshev"),
    pytest.param(Hermite, id="Hermite"),
    pytest.param(HermiteE, id="HermiteE"),
    pytest.param(Laguerre, id="Laguerre"),
    pytest.param(Legendre, id="Legendre"),
]


# ---------------------------------------------------------------------------
# Construction / accessors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_construction_preserves_coefs(cls):
    p = cls([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(p.coef, np.array([1.0, 2.0, 3.0]))


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_default_construction_is_zero(cls):
    p = cls()
    assert p.coef.shape == (1,)
    assert float(p.coef[0]) == 0.0


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_degree_matches_len_minus_one(cls):
    p = cls([1.0, 2.0, 3.0, 4.0])
    assert p.degree == 3


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_repr_includes_class_name_and_coefs(cls):
    p = cls([1.0, 2.0])
    s = repr(p)
    assert cls.__name__ in s
    assert "1" in s and "2" in s


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_domain_and_window_are_length_2_arrays(cls):
    p = cls([1.0])
    assert p.domain.shape == (2,)
    assert p.window.shape == (2,)


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_len_is_coef_length(cls):
    p = cls([1.0, 2.0, 3.0])
    assert len(p) == 3


# ---------------------------------------------------------------------------
# __call__: scalar and array
# ---------------------------------------------------------------------------


def test_polynomial_call_scalar():
    # 1 + 2x + 3x^2 at x=2  =  1 + 4 + 12 = 17
    p = Polynomial([1.0, 2.0, 3.0])
    assert p(2.0) == pytest.approx(17.0)


def test_polynomial_call_array():
    p = Polynomial([1.0, 2.0, 3.0])
    out = p(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(out, [1.0, 6.0, 17.0])


def test_polynomial_call_2d_array_shape_preserved():
    p = Polynomial([0.0, 1.0])  # f(x) = x
    inp = np.arange(6.0).reshape(2, 3)
    out = p(inp)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, inp)


def test_polynomial_call_matches_numpy():
    p = Polynomial([1.0, -2.0, 3.0])
    np_p = np.polynomial.Polynomial([1.0, -2.0, 3.0])
    xs = np.linspace(-2, 2, 11)
    np.testing.assert_allclose(p(xs), np_p(xs), rtol=1e-12)


def test_chebyshev_call_matches_numpy():
    fr_c = Chebyshev([1.0, 2.0, 3.0])
    np_c = np.polynomial.Chebyshev([1.0, 2.0, 3.0])
    xs = np.linspace(-1, 1, 21)
    np.testing.assert_allclose(fr_c(xs), np_c(xs), rtol=1e-12)


# ---------------------------------------------------------------------------
# Calculus
# ---------------------------------------------------------------------------


def test_polynomial_deriv():
    # d/dx (1 + 2x + 3x^2) = 2 + 6x
    p = Polynomial([1.0, 2.0, 3.0])
    d = p.deriv()
    np.testing.assert_allclose(d.coef, [2.0, 6.0])


def test_polynomial_deriv_twice():
    p = Polynomial([1.0, 2.0, 3.0, 4.0])  # 1 + 2x + 3x² + 4x³
    d2 = p.deriv(2)
    # d²/dx² = 6 + 24x
    np.testing.assert_allclose(d2.coef, [6.0, 24.0])


def test_polynomial_integ_round_trips():
    p = Polynomial([2.0, 6.0])  # 2 + 6x
    integrated = p.integ()
    # ∫(2 + 6x)dx = 2x + 3x² (constant 0)
    np.testing.assert_allclose(integrated.coef, [0.0, 2.0, 3.0])
    # Differentiating gets us back.
    np.testing.assert_allclose(integrated.deriv().coef, p.coef)


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


def test_polynomial_real_roots():
    # x² - 1 = (x-1)(x+1). numpy returns a REAL float64 array when every
    # root is real (numpy/polynomial/polynomial.py:1606 _to_real_if_imag_zero);
    # ferray matches after the roots real-dtype fix (#815).
    p = Polynomial([-1.0, 0.0, 1.0])
    roots = p.roots()
    assert roots.dtype == np.float64
    real = sorted(roots.tolist())
    np.testing.assert_allclose(real, [-1.0, 1.0])


def test_polynomial_complex_roots():
    # x² + 1 = 0 has roots ±i
    p = Polynomial([1.0, 0.0, 1.0])
    roots = sorted(p.roots(), key=lambda c: c.imag)
    np.testing.assert_allclose([roots[0].imag, roots[1].imag], [-1.0, 1.0])
    np.testing.assert_allclose([roots[0].real, roots[1].real], [0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Arithmetic dunders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_addition_in_class(cls):
    a = cls([1.0, 2.0])
    b = cls([3.0, 4.0])
    s = a + b
    np.testing.assert_allclose(s.coef, [4.0, 6.0])


@pytest.mark.parametrize("cls", ALL_CLASSES)
def test_subtraction_in_class(cls):
    a = cls([5.0, 7.0])
    b = cls([3.0, 4.0])
    d = a - b
    np.testing.assert_allclose(d.coef, [2.0, 3.0])


def test_polynomial_multiplication():
    # (1 + x) * (1 + x) = 1 + 2x + x²
    a = Polynomial([1.0, 1.0])
    out = a * a
    np.testing.assert_allclose(out.coef, [1.0, 2.0, 1.0])


def test_polynomial_pow():
    a = Polynomial([1.0, 1.0])
    sq = a ** 2
    np.testing.assert_allclose(sq.coef, [1.0, 2.0, 1.0])


def test_polynomial_divmod():
    # (x² - 1) / (x - 1) = x + 1, remainder 0
    a = Polynomial([-1.0, 0.0, 1.0])
    b = Polynomial([-1.0, 1.0])
    q, r = a.divmod(b)
    np.testing.assert_allclose(q.coef, [1.0, 1.0])
    # Trim trailing zeros for remainder comparison.
    assert all(abs(v) < 1e-12 for v in r.coef.tolist())


# ---------------------------------------------------------------------------
# from_power_basis / convert_to_power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [Chebyshev, Hermite, HermiteE, Laguerre, Legendre],
    ids=lambda c: c.__name__,
)
def test_from_power_basis_round_trip(cls):
    src_coefs = [1.0, 2.0, 3.0]
    converted = cls.from_power_basis(src_coefs)
    back = converted.convert_to_power()
    np.testing.assert_allclose(back, src_coefs, atol=1e-10)


@pytest.mark.parametrize(
    "cls",
    [Chebyshev, Hermite, HermiteE, Laguerre, Legendre],
    ids=lambda c: c.__name__,
)
def test_from_power_basis_evaluates_same(cls):
    """A polynomial in any basis evaluates to the same values."""
    src = [1.0, -2.0, 3.0]
    converted = cls.from_power_basis(src)
    p_power = Polynomial(src)
    xs = np.linspace(-0.9, 0.9, 5)  # avoid edge case at ±1 for Legendre
    np.testing.assert_allclose(converted(xs), p_power(xs), rtol=1e-10)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------


def test_polynomial_fit_recovers_quadratic():
    rng = np.random.default_rng(42)
    x = np.linspace(-1.0, 1.0, 50)
    true_coefs = [1.0, -0.5, 2.0]
    y = 1.0 - 0.5 * x + 2.0 * x ** 2
    fit = Polynomial.fit(x.tolist(), y.tolist(), 2)
    np.testing.assert_allclose(fit.coef, true_coefs, atol=1e-10)


# ---------------------------------------------------------------------------
# trim / truncate
# ---------------------------------------------------------------------------


def test_polynomial_trim_drops_small_trailing():
    p = Polynomial([1.0, 2.0, 1e-15])
    trimmed = p.trim(1e-10)
    assert len(trimmed) == 2


def test_polynomial_truncate_drops_excess():
    p = Polynomial([1.0, 2.0, 3.0, 4.0])
    t = p.truncate(2)
    assert len(t) == 2
    np.testing.assert_allclose(t.coef, [1.0, 2.0])
