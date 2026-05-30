"""Adversarial divergence tests: ferray.polynomial vs numpy 2.4.x oracle.

Audit target: ferray-python/src/polynomial.rs (the numpy.polynomial
class API: Polynomial / Chebyshev / Legendre / ...).

Each test pins a CONFIRMED divergence from the live numpy oracle. They
are expected to FAIL against the current ferray build; when the binding
(or an underlying ferray-polynomial crate) is fixed to match numpy, they
flip to passing.

Upstream references are cited per-test as `file:line` into
/home/doll/numpy-ref.
"""

import numpy as np
import pytest

import ferray as fr

P = fr.polynomial.Polynomial
NP_P = np.polynomial.Polynomial
C = fr.polynomial.Chebyshev
NP_C = np.polynomial.Chebyshev


# ---------------------------------------------------------------------------
# 1. roots() dtype: real polynomial -> real array, not complex
# ---------------------------------------------------------------------------
def test_roots_real_polynomial_returns_real_dtype():
    """numpy/polynomial/polynomial.py:17-18 (polyroots docstring):
    "If all the roots are real, then `out` is also real, otherwise it is
    complex." numpy's `_polybase.roots` returns float64 for an all-real
    root set; ferray hard-codes complex128.

    x^2 - 5x + 6 = (x-2)(x-3): roots {2, 3}, all real.
    """
    p = P([6.0, -5.0, 1.0])
    np_p = NP_P([6.0, -5.0, 1.0])
    fr_roots = p.roots()
    np_roots = np_p.roots()
    assert fr_roots.dtype == np_roots.dtype  # numpy: float64; ferray: complex128


# ---------------------------------------------------------------------------
# 2. roots() ordering: numpy returns sorted roots
# ---------------------------------------------------------------------------
def test_roots_are_sorted_like_numpy():
    """numpy/polynomial/_polybase.py:900-913 `roots()` returns the
    eigenvalues of the companion matrix via numpy's root solver, which
    yields ascending-sorted roots for this case. ferray returns an
    unsorted complex array, so an *exact-order* comparison (not a
    set comparison) diverges.

    x^2 - 5x + 6 -> numpy [2., 3.]; ferray [3.+0j, 2.+0j].
    """
    p = P([6.0, -5.0, 1.0])
    np_roots = NP_P([6.0, -5.0, 1.0]).roots()
    fr_roots = np.asarray(p.roots())
    # Compare in the exact order numpy emits (real part), no pre-sorting.
    np.testing.assert_allclose(fr_roots.real, np.asarray(np_roots).real)


# ---------------------------------------------------------------------------
# 3. scalar __call__ return type: numpy.float64, not Python float
# ---------------------------------------------------------------------------
def test_scalar_call_returns_numpy_scalar():
    """numpy/polynomial/_polybase.py:510-511 `__call__` runs the value
    through `polyval`, which returns a numpy scalar (np.float64) for a
    scalar argument. ferray's binding returns a bare Python float, so
    `.dtype` / numpy-scalar identity is lost across the boundary
    (goal.md R-CODE-4 boundary discipline).
    """
    p = P([1.0, 2.0, 3.0])
    np_p = NP_P([1.0, 2.0, 3.0])
    fr_val = p(2.0)
    np_val = np_p(2.0)
    assert isinstance(fr_val, type(np_val))  # numpy.float64


# ---------------------------------------------------------------------------
# 4. fit(): default domain is mapped to [x.min, x.max]; coef + domain diverge
# ---------------------------------------------------------------------------
def test_fit_maps_domain_and_coef_to_numpy():
    """numpy/polynomial/_polybase.py:1014-1034 `fit`: when `domain is
    None` it picks `pu.getdomain(x)` (= [x.min, x.max]), maps x into the
    window, and stores the fit coefficients in that mapped basis. ferray
    ignores domain mapping: it returns power-basis coef with the default
    domain [-1, 1].

    Quadratic over x in [0, 10].
    """
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 + 2.0 * x + x**2
    fr_fit = P.fit(x.tolist(), y.tolist(), 2)
    np_fit = NP_P.fit(x, y, 2)
    np.testing.assert_allclose(np.asarray(fr_fit.coef), np.asarray(np_fit.coef))


def test_fit_domain_matches_numpy():
    """numpy/polynomial/_polybase.py:1015 `domain = pu.getdomain(x)`.
    numpy's fitted polynomial reports domain [0., 10.]; ferray reports
    the unmapped default [-1., 1.].
    """
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 + 2.0 * x + x**2
    fr_fit = P.fit(x.tolist(), y.tolist(), 2)
    np_fit = NP_P.fit(x, y, 2)
    np.testing.assert_array_equal(
        np.asarray(fr_fit.domain), np.asarray(np_fit.domain)
    )


# ---------------------------------------------------------------------------
# 5. constructor: domain / window kwargs accepted (numpy public ABI)
# ---------------------------------------------------------------------------
def test_constructor_accepts_domain_kwarg():
    """numpy/polynomial/_polybase.py: every series class constructor takes
    `domain`, `window`, `symbol` kwargs (see __init__ signature). ferray's
    #[new] only accepts `coef`, so the public constructor ABI diverges and
    domain-mapped polynomials cannot be expressed at all.
    """
    # numpy accepts this; if ferray also accepts it the divergence is gone.
    NP_P([1.0, 2.0], domain=[0.0, 10.0], window=[-1.0, 1.0])
    P([1.0, 2.0], domain=[0.0, 10.0], window=[-1.0, 1.0])


# ---------------------------------------------------------------------------
# 6. __repr__ includes domain / window / symbol
# ---------------------------------------------------------------------------
def test_repr_matches_numpy_format():
    """numpy/polynomial/_polybase.py `__repr__` emits e.g.
    `Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.],
    symbol='x')`. ferray emits `Polynomial([1.0, 2.0, 3.0])` — missing
    the domain/window/symbol fields.
    """
    p = P([1.0, 2.0, 3.0])
    np_p = NP_P([1.0, 2.0, 3.0])
    assert repr(p) == repr(np_p)
