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


# ===========================================================================
# poly1d family — narrow-float dtype preservation (ACToR adversarial sweep,
# 2026-05-31). The verified behaviors (all-integer / int+float / complex)
# match the oracle exactly — see report "NO DIVERGENCE" verdicts. The seam
# in ferray-python/src/polynomial.rs delegates to numpy ONLY for complex and
# integer-kind dtypes; it MISSES narrow-float (float32/float16) inputs, which
# numpy preserves but ferray upcasts to float64.
# ===========================================================================

# ---------------------------------------------------------------------------
# 9. polyadd(float32, float32) — numpy keeps float32; ferray upcasts to float64.
# ---------------------------------------------------------------------------
def test_polyadd_float32_preserves_dtype():
    """numpy/lib/_polynomial_impl.py:850-857 — polyadd does `a1 + a2` over
    `atleast_1d(a1)`/`atleast_1d(a2)`; float32 + float32 -> float32 by numpy
    promotion. ferray's poly_binop (polynomial.rs:875) coerces both operands
    to Vec<f64>, so the result is float64. Values agree; dtype diverges.
    Expected dtype from the LIVE oracle.
    """
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    np_dtype = np.polyadd(a, b).dtype          # oracle: float32
    fr_dtype = np.asarray(fr.polyadd(a, b)).dtype
    assert fr_dtype == np_dtype


# ---------------------------------------------------------------------------
# 10. polysub(float32, float32) — same float32-preservation divergence.
# ---------------------------------------------------------------------------
def test_polysub_float32_preserves_dtype():
    """numpy/lib/_polynomial_impl.py:906-918 — polysub mirrors polyadd's
    `a1 - a2` over `atleast_1d`, preserving float32. ferray upcasts to
    float64 via the Vec<f64> seam (polynomial.rs:875,951).
    """
    a = np.array([5.0, 7.0], dtype=np.float32)
    b = np.array([1.0, 2.0], dtype=np.float32)
    np_dtype = np.polysub(a, b).dtype          # oracle: float32
    fr_dtype = np.asarray(fr.polysub(a, b)).dtype
    assert fr_dtype == np_dtype


# ---------------------------------------------------------------------------
# 11. polymul(float32, float32) — convolution preserves float32 in numpy.
# ---------------------------------------------------------------------------
def test_polymul_float32_preserves_dtype():
    """numpy/lib/_polynomial_impl.py:979-982 — polymul = `NX.convolve(a1, a2)`
    over `atleast_1d`; float32 convolution stays float32. ferray's poly_binop
    (polynomial.rs:875,965) upcasts to float64.
    """
    a = np.array([1.0, 2.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    np_dtype = np.polymul(a, b).dtype          # oracle: float32
    fr_dtype = np.asarray(fr.polymul(a, b)).dtype
    assert fr_dtype == np_dtype


# ---------------------------------------------------------------------------
# 12. polyval(float32 p, float32 x) — Horner over asarray(p) keeps float32.
# ---------------------------------------------------------------------------
def test_polyval_float32_preserves_dtype():
    """numpy/lib/_polynomial_impl.py:782-790 — polyval evaluates Horner
    `y = y*x + pv` over `NX.asarray(p)` and `NX.asanyarray(x)`; float32 p and
    float32 x keep a float32 result. ferray's polyval (polynomial.rs:753,755)
    coerces p to Vec<f64> and x through as_eval_input's `asarray(_, float64)`
    cast, yielding float64. The seam delegates only for complex / integer-kind
    inputs (polynomial.rs:740-752), not narrow floats.
    """
    p = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    x = np.float32(2.0)
    np_dtype = np.asarray(np.polyval(p, x)).dtype   # oracle: float32
    fr_dtype = np.asarray(fr.polyval(p, x)).dtype
    assert fr_dtype == np_dtype


# ---------------------------------------------------------------------------
# 13. polyadd(float16, float16) — float16 preserved by numpy promotion.
# ---------------------------------------------------------------------------
def test_polyadd_float16_preserves_dtype():
    """numpy/lib/_polynomial_impl.py:850-857 — float16 + float16 -> float16.
    Generalises the narrow-float divergence below float32 (the seam upcasts
    ALL non-int/non-complex dtypes to float64).
    """
    a = np.array([1.0, 2.0], dtype=np.float16)
    b = np.array([3.0, 4.0], dtype=np.float16)
    np_dtype = np.polyadd(a, b).dtype          # oracle: float16
    fr_dtype = np.asarray(fr.polyadd(a, b)).dtype
    assert fr_dtype == np_dtype


# ===========================================================================
# CLASS-API adversarial sweep #2 (ACToR critic, 2026-05-31). The first batch
# (tests 1-16 above) was fixed by the generator and is now GREEN. This batch
# targets the class methods/dunders the audit task enumerates that the macro
# `poly_class!` (ferray-python/src/polynomial.rs:403-674) does NOT implement,
# and the domain/window-Jacobian scaling that deriv/integ/roots omit.
#
# Every expected value comes from a LIVE numpy 2.4.x call (the oracle) or a
# numpy `file:line` symbolic contract (goal.md R-CHAR-3). Cites are into
# /home/doll/numpy-ref/numpy/polynomial/_polybase.py.
# ===========================================================================

CLASSES = [
    (fr.polynomial.Polynomial, np.polynomial.Polynomial),
    (fr.polynomial.Chebyshev, np.polynomial.Chebyshev),
    (fr.polynomial.Hermite, np.polynomial.Hermite),
    (fr.polynomial.HermiteE, np.polynomial.HermiteE),
    (fr.polynomial.Laguerre, np.polynomial.Laguerre),
    (fr.polynomial.Legendre, np.polynomial.Legendre),
]


# ---------------------------------------------------------------------------
# 17. roots() must map window-space roots back to DOMAIN space.
# ---------------------------------------------------------------------------
def test_roots_mapped_back_through_domain():
    """numpy/polynomial/_polybase.py:900-913 `def roots(self):` ends with
    `return pu.mapdomain(roots, self.window, self.domain)` — the companion
    eigenvalues live in WINDOW space and are mapped back to DOMAIN space.
    For Polynomial([1,0,-1], domain=[0,4]) (x^2-1 in window coords -> roots
    +-1 in window [-1,1]) numpy maps those to [0., 4.] in the domain; ferray
    returns the unmapped [-1., 1.].
    """
    fr_r = np.sort(np.asarray(fr.polynomial.Polynomial(
        [1.0, 0.0, -1.0], domain=[0.0, 4.0]).roots()).real)
    np_r = np.sort(np.asarray(np.polynomial.Polynomial(
        [1.0, 0.0, -1.0], domain=[0.0, 4.0]).roots()).real)
    np.testing.assert_allclose(fr_r, np_r)  # numpy [0,4]; ferray [-1,1]


# ---------------------------------------------------------------------------
# 18. deriv() must scale by the domain->window Jacobian (mapparms scl).
# ---------------------------------------------------------------------------
def test_deriv_applies_domain_jacobian():
    """numpy/polynomial/_polybase.py:878-901 `def deriv(self, m=1):` does
    `off, scl = self.mapparms(); coef = self._der(self.coef, m, scl)`. With a
    non-default domain the derivative is scaled by
    scl = (window[1]-window[0])/(domain[1]-domain[0]). For
    Polynomial([1,2,3,4], domain=[0,10]) numpy gives deriv coef
    [0.4, 1.2, 2.4]; ferray omits the scl factor and gives [2., 6., 12.].
    """
    fp = fr.polynomial.Polynomial([1.0, 2.0, 3.0, 4.0], domain=[0.0, 10.0])
    npp = np.polynomial.Polynomial([1.0, 2.0, 3.0, 4.0], domain=[0.0, 10.0])
    np.testing.assert_allclose(
        np.asarray(fp.deriv().coef), np.asarray(npp.deriv().coef))


# ---------------------------------------------------------------------------
# 19. integ() must scale by the domain->window Jacobian.
# ---------------------------------------------------------------------------
def test_integ_applies_domain_jacobian():
    """numpy/polynomial/_polybase.py:845-877 `def integ(self, m=1, k=[],
    lbnd=None):` does `off, scl = self.mapparms()` and integrates with that
    scl. For Polynomial([1,2,3], domain=[0,10]) numpy integ coef is
    [0., 5., 5., 5.]; ferray omits scl and gives [0., 1., 1., 1.].
    """
    fp = fr.polynomial.Polynomial([1.0, 2.0, 3.0], domain=[0.0, 10.0])
    npp = np.polynomial.Polynomial([1.0, 2.0, 3.0], domain=[0.0, 10.0])
    np.testing.assert_allclose(
        np.asarray(fp.integ().coef), np.asarray(npp.integ().coef))


# ---------------------------------------------------------------------------
# 20. integ() must accept the `lbnd` keyword argument.
# ---------------------------------------------------------------------------
def test_integ_accepts_lbnd_kwarg():
    """numpy/polynomial/_polybase.py:845 `def integ(self, m=1, k=[],
    lbnd=None):` — `lbnd` sets the lower integration bound so the constant of
    integration makes the antiderivative vanish at `lbnd`. ferray's
    poly_class!::integ signature is `(m = 1, k = vec![])` only, so the kwarg
    raises TypeError. numpy integ(1, k=[5], lbnd=2) on [1,2,3] gives
    [-9., 1., 1., 1.].
    """
    npp = np.polynomial.Polynomial([1.0, 2.0, 3.0])
    exp = np.asarray(npp.integ(1, k=[5.0], lbnd=2.0).coef)
    fp = fr.polynomial.Polynomial([1.0, 2.0, 3.0])
    got = np.asarray(fp.integ(1, k=[5.0], lbnd=2.0).coef)  # ferray: TypeError
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 21. __eq__: two equal series compare equal.
# ---------------------------------------------------------------------------
def test_eq_equal_series_are_equal():
    """numpy/polynomial/_polybase.py:643-651 `def __eq__(self, other):`
    compares class, domain, window, coef.shape, coef values, and symbol.
    Two Polynomial([1,2,3]) instances are equal. ferray's poly_class! defines
    no `__eq__`, so it falls back to object identity and returns False.
    """
    np_eq = (np.polynomial.Polynomial([1.0, 2.0, 3.0])
             == np.polynomial.Polynomial([1.0, 2.0, 3.0]))
    fr_eq = (fr.polynomial.Polynomial([1.0, 2.0, 3.0])
             == fr.polynomial.Polynomial([1.0, 2.0, 3.0]))
    assert fr_eq == np_eq  # numpy True; ferray False


# ---------------------------------------------------------------------------
# 22. __neg__: unary negation of a series.
# ---------------------------------------------------------------------------
def test_neg_negates_coefficients():
    """numpy/polynomial/_polybase.py:522 `def __neg__(self):` returns a series
    with `-self.coef`. ferray's poly_class! defines no `__neg__`, so `-p`
    raises TypeError (bad operand type for unary -).
    """
    exp = np.asarray((-np.polynomial.Polynomial([1.0, 2.0, 3.0])).coef)
    got = np.asarray((-fr.polynomial.Polynomial([1.0, 2.0, 3.0])).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 23. scalar __add__: p + scalar broadcasts onto the constant term.
# ---------------------------------------------------------------------------
def test_scalar_add_supported():
    """numpy/polynomial/_polybase.py:530-536 `def __add__(self, other):` runs
    `othercoef = self._get_coefficients(other)`, which wraps a bare scalar as
    a constant series. numpy: Polynomial([1,2,3]) + 1 -> coef [2,2,3]. ferray
    only accepts another instance (poly_class!::__add__ takes `other: &Self`),
    so `p + 1` raises TypeError.
    """
    exp = np.asarray((np.polynomial.Polynomial([1.0, 2.0, 3.0]) + 1).coef)
    got = np.asarray((fr.polynomial.Polynomial([1.0, 2.0, 3.0]) + 1).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 24. scalar __mul__: p * scalar scales every coefficient.
# ---------------------------------------------------------------------------
def test_scalar_mul_supported():
    """numpy/polynomial/_polybase.py:546-552 `def __mul__(self, other):`.
    numpy: Polynomial([1,2,3]) * 2 -> coef [2,4,6]. ferray's __mul__ takes
    `other: &Self` only -> TypeError on a scalar.
    """
    exp = np.asarray((np.polynomial.Polynomial([1.0, 2.0, 3.0]) * 2).coef)
    got = np.asarray((fr.polynomial.Polynomial([1.0, 2.0, 3.0]) * 2).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 25. __floordiv__ (//) and __mod__ (%) dunders.
# ---------------------------------------------------------------------------
def test_floordiv_supported():
    """numpy/polynomial/_polybase.py:565 `def __floordiv__(self, other):` ->
    the quotient of polynomial division. numpy: (x^2-1)//(x-1) -> coef [1,1].
    ferray's poly_class! defines __divmod__ but no __floordiv__, so `a // b`
    raises TypeError.
    """
    a = fr.polynomial.Polynomial([-1.0, 0.0, 1.0])
    b = fr.polynomial.Polynomial([-1.0, 1.0])
    na = np.polynomial.Polynomial([-1.0, 0.0, 1.0])
    nb = np.polynomial.Polynomial([-1.0, 1.0])
    exp = np.asarray((na // nb).coef)
    got = np.asarray((a // b).coef)  # ferray: TypeError
    np.testing.assert_allclose(got, exp)


def test_mod_supported():
    """numpy/polynomial/_polybase.py:571 `def __mod__(self, other):` -> the
    remainder of polynomial division. numpy: (x^2-1) % (x-1) -> coef [0].
    ferray defines no __mod__ -> TypeError.
    """
    a = fr.polynomial.Polynomial([-1.0, 0.0, 1.0])
    b = fr.polynomial.Polynomial([-1.0, 1.0])
    na = np.polynomial.Polynomial([-1.0, 0.0, 1.0])
    nb = np.polynomial.Polynomial([-1.0, 1.0])
    exp = np.asarray((na % nb).coef)
    got = np.asarray((a % b).coef)  # ferray: TypeError
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 26. convert(kind=...) — cross-basis conversion preserving the value.
# ---------------------------------------------------------------------------
def test_convert_kind_method():
    """numpy/polynomial/_polybase.py:779 `def convert(self, domain=None,
    kind=None, window=None):` returns an equivalent series in another basis.
    numpy: Chebyshev([1,2,3]).convert(kind=Polynomial).coef -> [-2., 2., 6.].
    ferray's poly_class! exposes no `convert` method (only the raw
    `convert_to_power` returning a bare coef array).
    """
    exp = np.asarray(np.polynomial.Chebyshev([1.0, 2.0, 3.0])
                     .convert(kind=np.polynomial.Polynomial).coef)
    got = np.asarray(fr.polynomial.Chebyshev([1.0, 2.0, 3.0])
                     .convert(kind=fr.polynomial.Polynomial).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 27. fromroots classmethod.
# ---------------------------------------------------------------------------
def test_fromroots_classmethod():
    """numpy/polynomial/_polybase.py:1037 `def fromroots(cls, roots,
    domain=[], window=None, symbol='x'):` builds the monic series with the
    given roots. numpy: Polynomial.fromroots([1,2,3]).coef ->
    [-6., 11., -6., 1.]. ferray's poly_class! exposes no `fromroots`.
    """
    exp = np.asarray(np.polynomial.Polynomial.fromroots([1.0, 2.0, 3.0]).coef)
    got = np.asarray(fr.polynomial.Polynomial.fromroots([1.0, 2.0, 3.0]).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 28. basis classmethod.
# ---------------------------------------------------------------------------
def test_basis_classmethod():
    """numpy/polynomial/_polybase.py:1115 `def basis(cls, deg, domain=None,
    window=None, symbol='x'):` returns the degree-`deg` basis series. numpy:
    Polynomial.basis(3).coef -> [0., 0., 0., 1.]. ferray exposes no `basis`.
    """
    exp = np.asarray(np.polynomial.Polynomial.basis(3).coef)
    got = np.asarray(fr.polynomial.Polynomial.basis(3).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 29. identity classmethod.
# ---------------------------------------------------------------------------
def test_identity_classmethod():
    """numpy/polynomial/_polybase.py:1080 `def identity(cls, domain=None,
    window=None, symbol='x'):` returns the series p(x)=x. numpy:
    Polynomial.identity()(5.0) == 5.0. ferray exposes no `identity`.
    """
    npid = np.polynomial.Polynomial.identity()
    frid = fr.polynomial.Polynomial.identity()
    np.testing.assert_allclose(float(frid(5.0)), float(npid(5.0)))


# ---------------------------------------------------------------------------
# 30. mapparms method.
# ---------------------------------------------------------------------------
def test_mapparms_method():
    """numpy/polynomial/_polybase.py:816 `def mapparms(self):` returns
    (off, scl) mapping domain -> window. numpy: Polynomial([1,2],
    domain=[0,10]).mapparms() -> (-1.0, 0.2). ferray exposes no `mapparms`.
    """
    npp = np.polynomial.Polynomial([1.0, 2.0], domain=[0.0, 10.0])
    fp = fr.polynomial.Polynomial([1.0, 2.0], domain=[0.0, 10.0])
    np.testing.assert_allclose(np.asarray(fp.mapparms()),
                               np.asarray(npp.mapparms()))


# ---------------------------------------------------------------------------
# 31. cutdeg method.
# ---------------------------------------------------------------------------
def test_cutdeg_method():
    """numpy/polynomial/_polybase.py:704 `def cutdeg(self, deg):` returns the
    series truncated to degree `deg`. numpy: Polynomial([1,2,3,4]).cutdeg(1)
    .coef -> [1., 2.]. ferray exposes `truncate(size)` but no `cutdeg(deg)`.
    """
    exp = np.asarray(np.polynomial.Polynomial([1.0, 2.0, 3.0, 4.0])
                     .cutdeg(1).coef)
    got = np.asarray(fr.polynomial.Polynomial([1.0, 2.0, 3.0, 4.0])
                     .cutdeg(1).coef)
    np.testing.assert_allclose(got, exp)


# ---------------------------------------------------------------------------
# 32. linspace method.
# ---------------------------------------------------------------------------
def test_linspace_method():
    """numpy/polynomial/_polybase.py:915 `def linspace(self, n=100,
    domain=None):` returns (x, p(x)) over the domain. numpy:
    Polynomial([0,1]).linspace(5) -> x=[-1,-0.5,0,0.5,1], y same. ferray
    exposes no `linspace`.
    """
    nx, ny = np.polynomial.Polynomial([0.0, 1.0]).linspace(5)
    fx, fy = fr.polynomial.Polynomial([0.0, 1.0]).linspace(5)
    np.testing.assert_allclose(np.asarray(fx), np.asarray(nx))
    np.testing.assert_allclose(np.asarray(fy), np.asarray(ny))


# ---------------------------------------------------------------------------
# 33. fit() must accept domain=/window=/full= keyword arguments.
# ---------------------------------------------------------------------------
def test_fit_accepts_domain_window_full_kwargs():
    """numpy/polynomial/_polybase.py:946 `def fit(cls, x, y, deg, domain=None,
    rcond=None, full=False, w=None, window=None, symbol='x'):`. ferray's
    poly_class!::fit signature is `(x, y, deg)` only, so passing `domain=`,
    `window=`, or `full=` raises TypeError. Here we pin the explicit-domain
    fit: numpy fitted coef with domain=[0,10], window=[-1,1].
    """
    x = np.linspace(0.0, 10.0, 30)
    y = 3.0 + 2.0 * x + x ** 2
    exp = np.asarray(np.polynomial.Polynomial.fit(
        x, y, 2, domain=[0.0, 10.0], window=[-1.0, 1.0]).coef)
    got = np.asarray(fr.polynomial.Polynomial.fit(
        x.tolist(), y.tolist(), 2,
        domain=[0.0, 10.0], window=[-1.0, 1.0]).coef)  # ferray: TypeError
    np.testing.assert_allclose(got, exp)
