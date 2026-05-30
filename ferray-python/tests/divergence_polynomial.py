"""Adversarial divergence tests: ferray.polynomial vs numpy 2.4.5 oracle.

Audit target: ferray-python/src/polynomial.rs — the numpy.polynomial class
API (Polynomial / Chebyshev / Legendre / ...) and the macro-generated
`#[pymethods]` surface (poly_class!).

Every expected value below comes from a LIVE numpy call (the imported
`numpy` 2.4.5 is the oracle) or a numpy `file:line` symbolic contract —
never literal-copied from the ferray side (goal.md R-CHAR-3).

These tests are EXPECTED TO FAIL against the current ferray build. They
flip green only when the binding (or the underlying ferray-polynomial
crate) is fixed to match numpy. Upstream cites are into /home/doll/numpy-ref.
"""

import numpy as np
import pytest

import ferray as fr

P = fr.polynomial.Polynomial
NP_P = np.polynomial.Polynomial


# ---------------------------------------------------------------------------
# 1. roots() dtype: an all-real polynomial returns a REAL array, not complex.
# ---------------------------------------------------------------------------
def test_roots_real_polynomial_returns_real_dtype():
    """numpy/polynomial/polynomial.py:1562 (polyroots docstring):
        "then `out` is also real, otherwise it is complex."
    numpy/polynomial/polynomial.py:1603 `r.sort()` then a real-cast when
    all imag parts vanish. numpy returns float64 for
    x^2 - 5x + 6 = (x-2)(x-3); ferray hard-codes complex128 in
    poly_class!::roots (complex_vec_to_pyarray).
    """
    fr_dtype = P([6.0, -5.0, 1.0]).roots().dtype
    np_dtype = NP_P([6.0, -5.0, 1.0]).roots().dtype
    assert fr_dtype == np_dtype  # numpy: float64; ferray: complex128


# ---------------------------------------------------------------------------
# 2. roots() ordering: numpy returns ascending-sorted roots.
# ---------------------------------------------------------------------------
def test_roots_are_ascending_sorted_like_numpy():
    """numpy/polynomial/polynomial.py:1603 `r.sort()` — polyroots sorts the
    companion-matrix eigenvalues ascending. For x^2 - 5x + 6 numpy yields
    [2., 3.]; ferray emits the unsorted [3.+0j, 2.+0j]. Compared in
    numpy's exact emit order (no pre-sorting).
    """
    np_roots = np.asarray(NP_P([6.0, -5.0, 1.0]).roots())
    fr_roots = np.asarray(P([6.0, -5.0, 1.0]).roots()).real
    np.testing.assert_allclose(fr_roots, np_roots)


# ---------------------------------------------------------------------------
# 3. scalar __call__ return type: numpy.float64, not a bare Python float.
# ---------------------------------------------------------------------------
def test_scalar_call_returns_numpy_scalar():
    """numpy/polynomial/_polybase.py:510-512 `__call__` routes a scalar
    through `self._val`, returning a numpy scalar (np.float64). ferray's
    poly_class!::__call__ does `v.into_pyobject(py)` on an f64, yielding a
    bare Python float — the numpy-scalar contract is lost across the PyO3
    boundary (goal.md R-CODE-4).
    """
    fr_val = P([1.0, 2.0, 3.0])(2.0)
    np_val = NP_P([1.0, 2.0, 3.0])(2.0)
    assert isinstance(fr_val, type(np_val))  # numpy.float64


# ---------------------------------------------------------------------------
# 4. fit(): coefficients are stored in the mapped (domain->window) basis.
# ---------------------------------------------------------------------------
def test_fit_coef_matches_numpy_mapped_basis():
    """numpy/polynomial/_polybase.py:946 `def fit(...)` with
    numpy/polynomial/_polybase.py:1015 `domain = pu.getdomain(x)`: numpy
    maps x into `window` and stores the fit coefficients in that MAPPED
    basis. ferray.fit returns raw power-basis coef with the default domain.

    Quadratic 3 + 2x + x^2 over x in [0, 10]:
    numpy coef -> [38., 60., 25.]; ferray -> [3., 2., 1.].
    """
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 + 2.0 * x + x ** 2
    fr_coef = np.asarray(P.fit(x.tolist(), y.tolist(), 2).coef)
    np_coef = np.asarray(NP_P.fit(x, y, 2).coef)
    np.testing.assert_allclose(fr_coef, np_coef)


def test_fit_domain_matches_numpy():
    """numpy/polynomial/_polybase.py:1015 `domain = pu.getdomain(x)`.
    numpy reports domain [0., 10.] for a fit over x in [0, 10];
    ferray reports the unmapped default [-1., 1.].
    """
    x = np.linspace(0.0, 10.0, 50)
    y = 3.0 + 2.0 * x + x ** 2
    fr_dom = np.asarray(P.fit(x.tolist(), y.tolist(), 2).domain)
    np_dom = np.asarray(NP_P.fit(x, y, 2).domain)
    np.testing.assert_array_equal(fr_dom, np_dom)


# ---------------------------------------------------------------------------
# 5. constructor: domain / window / symbol kwargs (numpy public ABI).
# ---------------------------------------------------------------------------
def test_constructor_accepts_domain_window_kwargs():
    """numpy/polynomial/_polybase.py:292 `def __init__(self, coef,
    domain=None, window=None, symbol='x')`. ferray's poly_class!::py_new
    has signature `(coef = vec![0.0])` only, so the public constructor ABI
    diverges and domain-mapped polynomials cannot be expressed.
    """
    NP_P([1.0, 2.0], domain=[0.0, 10.0], window=[-1.0, 1.0])  # numpy: OK
    P([1.0, 2.0], domain=[0.0, 10.0], window=[-1.0, 1.0])  # ferray: TypeError


def test_symbol_attribute_present():
    """numpy/polynomial/_polybase.py:320 `self._symbol = symbol` (default
    'x' from the __init__ at line 292). ferray's poly_class! exposes no
    `symbol` attribute.
    """
    np_sym = NP_P([1.0, 2.0]).symbol
    assert P([1.0, 2.0]).symbol == np_sym


# ---------------------------------------------------------------------------
# 6. __repr__ includes domain / window / symbol.
# ---------------------------------------------------------------------------
def test_repr_matches_numpy_format():
    """numpy/polynomial/_polybase.py:322-328 `__repr__` emits
    `Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.],
    symbol='x')`. ferray's poly_class!::__repr__ emits
    `Polynomial([1.0, 2.0, 3.0])` (Rust Debug formatting of the coef Vec)
    — wrong float formatting AND missing domain/window/symbol.
    """
    assert repr(P([1.0, 2.0, 3.0])) == repr(NP_P([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# 7. degree is a METHOD returning len(coef)-1 (counts trailing zeros).
# ---------------------------------------------------------------------------
def test_degree_is_callable_method():
    """numpy/polynomial/_polybase.py:670 `def degree(self):` — numpy's
    `degree` is a callable method (returns `len(self) - 1`). ferray exposes
    it as a non-callable #[getter] property, so `p.degree()` raises
    TypeError ('int' object is not callable).
    """
    npp = NP_P([1.0, 2.0, 3.0])
    assert callable(npp.degree)  # oracle: numpy degree is a method
    p = P([1.0, 2.0, 3.0])
    assert p.degree() == npp.degree()  # ferray: property -> not callable


def test_degree_counts_trailing_zeros():
    """numpy/polynomial/_polybase.py:670 `degree()` returns `len(self) - 1`
    and does NOT trim trailing zeros, so [1, 2, 0, 0] has degree 3. ferray's
    getter delegates to the trimming `inner.degree()`, returning 1.
    """
    npp = NP_P([1.0, 2.0, 0.0, 0.0])
    p = P([1.0, 2.0, 0.0, 0.0])
    # Read fr.degree as a property value to isolate the value divergence
    # (the call-shape divergence is pinned separately above).
    assert p.degree == npp.degree()  # numpy: 3; ferray: 1


# ---------------------------------------------------------------------------
# 8. builtin divmod(a, b) — numpy implements __divmod__.
# ---------------------------------------------------------------------------
def test_builtin_divmod_supported():
    """numpy/polynomial/_polybase.py:577 `def __divmod__(self, other):` —
    the Python builtin `divmod(a, b)` returns (quotient, remainder). ferray
    exposes a named `divmod` method but no `__divmod__` dunder, so the
    builtin raises TypeError (unsupported operand type).

    (x^2 - 1) / (x - 1) = x + 1, remainder 0.
    """
    a, b = P([-1.0, 0.0, 1.0]), P([-1.0, 1.0])
    na, nb = NP_P([-1.0, 0.0, 1.0]), NP_P([-1.0, 1.0])
    nq, nr = divmod(na, nb)
    q, r = divmod(a, b)  # ferray: TypeError (no __divmod__)
    np.testing.assert_allclose(np.asarray(q.coef), np.asarray(nq.coef))
