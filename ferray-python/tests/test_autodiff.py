"""Tests for #718 — ferray.autodiff DualNumber + helpers."""

import math

import numpy as np
import pytest

from ferray.autodiff import (
    DualNumber,
    cos,
    derivative,
    exp,
    gradient,
    jacobian,
    ln,
    log10,
    log2,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)


# ---------------------------------------------------------------------------
# DualNumber construction and accessors
# ---------------------------------------------------------------------------


def test_constructor_default_dual_is_zero():
    d = DualNumber(3.0)
    assert d.real == 3.0
    assert d.dual == 0.0


def test_constructor_explicit_dual():
    d = DualNumber(3.0, 2.0)
    assert d.real == 3.0
    assert d.dual == 2.0


def test_variable_seeds_dual_to_one():
    d = DualNumber.variable(7.0)
    assert d.real == 7.0
    assert d.dual == 1.0


def test_constant_seeds_dual_to_zero():
    d = DualNumber.constant(7.0)
    assert d.real == 7.0
    assert d.dual == 0.0


def test_repr_includes_real_and_dual():
    d = DualNumber(1.5, 2.5)
    s = repr(d)
    assert "1.5" in s and "2.5" in s


# ---------------------------------------------------------------------------
# Arithmetic dunders
# ---------------------------------------------------------------------------


def test_add_two_duals():
    a = DualNumber(1.0, 2.0)
    b = DualNumber(3.0, 4.0)
    c = a + b
    assert c.real == 4.0 and c.dual == 6.0


def test_add_with_python_float():
    a = DualNumber(1.0, 2.0)
    c = a + 5.0
    assert c.real == 6.0 and c.dual == 2.0  # constant has dual=0


def test_radd_with_python_float():
    a = DualNumber(1.0, 2.0)
    c = 5.0 + a
    assert c.real == 6.0 and c.dual == 2.0


def test_subtract():
    a = DualNumber(10.0, 1.0)
    b = DualNumber(3.0, 1.0)
    c = a - b
    assert c.real == 7.0 and c.dual == 0.0


def test_multiply_chain_rule():
    # d/dx (x * x) = 2x; at x=3 the derivative is 6.
    x = DualNumber.variable(3.0)
    y = x * x
    assert y.real == 9.0 and y.dual == 6.0


def test_truediv_quotient_rule():
    # d/dx (1/x) = -1/x^2; at x=2 derivative is -0.25.
    x = DualNumber.variable(2.0)
    y = DualNumber.constant(1.0) / x
    assert y.real == 0.5
    assert y.dual == pytest.approx(-0.25)


def test_negate():
    x = DualNumber(2.0, 3.0)
    y = -x
    assert y.real == -2.0 and y.dual == -3.0


def test_pow_integer():
    # d/dx x^4 at x=2 is 4*2^3 = 32.
    x = DualNumber.variable(2.0)
    y = x ** 4
    assert y.real == 16.0
    assert y.dual == pytest.approx(32.0)


def test_pow_zero_is_one():
    x = DualNumber.variable(5.0)
    y = x ** 0
    assert y.real == 1.0
    assert y.dual == 0.0


def test_pow_negative():
    x = DualNumber.variable(2.0)
    y = x ** -1
    # 1/x at x=2 → 0.5; derivative -1/x^2 = -0.25.
    assert y.real == pytest.approx(0.5)
    assert y.dual == pytest.approx(-0.25)


# ---------------------------------------------------------------------------
# Elementary functions check chain rule
# ---------------------------------------------------------------------------


def test_sin_derivative_is_cos():
    x = DualNumber.variable(0.5)
    y = sin(x)
    assert y.real == pytest.approx(math.sin(0.5))
    assert y.dual == pytest.approx(math.cos(0.5))


def test_cos_derivative_is_neg_sin():
    x = DualNumber.variable(0.5)
    y = cos(x)
    assert y.real == pytest.approx(math.cos(0.5))
    assert y.dual == pytest.approx(-math.sin(0.5))


def test_exp_derivative_equals_value():
    x = DualNumber.variable(1.0)
    y = exp(x)
    assert y.real == pytest.approx(math.e)
    assert y.dual == pytest.approx(math.e)


def test_ln_derivative_is_inverse():
    x = DualNumber.variable(5.0)
    y = ln(x)
    assert y.real == pytest.approx(math.log(5.0))
    assert y.dual == pytest.approx(0.2)


def test_sqrt_derivative():
    x = DualNumber.variable(4.0)
    y = sqrt(x)
    # d/dx sqrt(x) = 1/(2*sqrt(x)) = 0.25 at x=4.
    assert y.real == 2.0
    assert y.dual == pytest.approx(0.25)


@pytest.mark.parametrize("fn, math_fn, math_deriv", [
    (tan, math.tan, lambda x: 1.0 / math.cos(x) ** 2),
    (sinh, math.sinh, math.cosh),
    (tanh, math.tanh, lambda x: 1.0 - math.tanh(x) ** 2),
    (log2, math.log2, lambda x: 1.0 / (x * math.log(2.0))),
    (log10, math.log10, lambda x: 1.0 / (x * math.log(10.0))),
])
def test_elementary_function_chain_rule(fn, math_fn, math_deriv):
    x_val = 0.6
    x = DualNumber.variable(x_val)
    y = fn(x)
    assert y.real == pytest.approx(math_fn(x_val))
    assert y.dual == pytest.approx(math_deriv(x_val))


# ---------------------------------------------------------------------------
# derivative / gradient / jacobian helpers
# ---------------------------------------------------------------------------


def test_derivative_helper_polynomial():
    f = lambda x: x ** 3 + 2 * x ** 2 + x + 5
    # f'(2) = 3*4 + 4*2 + 1 = 21.
    assert derivative(f, 2.0) == pytest.approx(21.0)


def test_derivative_helper_with_elementary():
    f = lambda x: sin(x) * exp(x)
    expected = math.cos(2.0) * math.exp(2.0) + math.sin(2.0) * math.exp(2.0)
    assert derivative(f, 2.0) == pytest.approx(expected)


def test_gradient_quadratic():
    # f(x, y) = x^2 + y^2 + xy
    # ∇f = (2x + y, 2y + x); at (3, 4) → (10, 11)
    f = lambda args: args[0] ** 2 + args[1] ** 2 + args[0] * args[1]
    g = gradient(f, [3.0, 4.0])
    np.testing.assert_allclose(g, [10.0, 11.0])


def test_gradient_returns_ndarray():
    f = lambda args: args[0] + args[1]
    g = gradient(f, [1.0, 2.0])
    assert isinstance(g, np.ndarray)
    assert g.dtype == np.float64


def test_jacobian_2x2():
    # f(x, y) = (x + y, x * y); J = [[1, 1], [y, x]] = [[1, 1], [4, 3]] at (3, 4)
    def f(args):
        x, y = args
        return [x + y, x * y]
    j = jacobian(f, [3.0, 4.0])
    np.testing.assert_allclose(j, np.array([[1.0, 1.0], [4.0, 3.0]]))


def test_jacobian_shape():
    # f: R^3 -> R^2.
    def f(args):
        x, y, z = args
        return [x + y + z, x * y * z]
    j = jacobian(f, [1.0, 2.0, 3.0])
    assert j.shape == (2, 3)
