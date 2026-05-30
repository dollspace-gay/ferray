"""Conformance tests for the ``ferray.emath`` (a.k.a. ``numpy.lib.scimath``)
submodule against the live numpy oracle.

``numpy.emath`` is the complex-aware family: each function returns a COMPLEX
result for inputs where the plain real-valued version would be a domain error
(``x < 0`` for sqrt/log/power-base; ``|x| > 1`` for arccos/arcsin/arctanh) and
a REAL result of the same dtype numpy would otherwise produce. The mechanism
mirrors ``numpy/lib/_scimath_impl.py`` (``_fix_real_lt_zero`` /
``_fix_int_lt_zero`` / ``_fix_real_abs_gt_1``).

Every expected value is produced by a live call into ``numpy.emath`` (the
oracle), never literal-copied from the ferray side (R-CHAR-3). Each assertion
checks VALUE (``np.allclose``), DTYPE (complex where numpy is complex, real
where real), and SHAPE (scalar-vs-array collapse).
"""

import warnings

import numpy as np
import pytest

import ferray as fr


def _assert_matches(got, exp):
    """Assert ferray result matches the numpy-oracle result in value, dtype,
    and shape — the full emath output-object contract."""
    ga = np.asarray(got)
    ea = np.asarray(exp)
    assert ga.dtype == ea.dtype, f"dtype: ferray {ga.dtype} != numpy {ea.dtype}"
    assert ga.shape == ea.shape, f"shape: ferray {ga.shape} != numpy {ea.shape}"
    assert np.allclose(ga, ea, equal_nan=True), f"value: ferray {got!r} != numpy {exp!r}"


def _oracle(fn_name, *args):
    """Call ``numpy.emath.<fn_name>`` (the oracle), silencing the benign
    divide-by-zero RuntimeWarning numpy itself emits for ``log(0)`` etc."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(np.emath, fn_name)(*args)


def _ferray(fn_name, *args):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return getattr(fr.emath, fn_name)(*args)


def _check(fn_name, *args):
    _assert_matches(_ferray(fn_name, *args), _oracle(fn_name, *args))


# ---------------------------------------------------------------------------
# Submodule registration
# ---------------------------------------------------------------------------


def test_emath_submodule_exposed():
    """``import ferray as fr; fr.emath`` exists and exposes all 9 functions,
    mirroring ``numpy.emath.__all__`` (``_scimath_impl.py:23-26``)."""
    assert hasattr(fr, "emath")
    for name in ("sqrt", "log", "log2", "log10", "logn", "power", "arccos",
                 "arcsin", "arctanh"):
        assert hasattr(fr.emath, name), f"fr.emath.{name} missing"


def test_emath_importable_as_submodule():
    """``from ferray.emath import sqrt`` works (sys.modules registration)."""
    from ferray.emath import sqrt as emath_sqrt

    _assert_matches(emath_sqrt(-1), _oracle("sqrt", -1))


# ---------------------------------------------------------------------------
# sqrt — _fix_real_lt_zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [-1, 4, 0, -4.0, 2.0, 0.25])
def test_sqrt_scalar(x):
    _check("sqrt", x)


@pytest.mark.parametrize("x", [[4, -1], [4, 9], [-1, -4, 9], [0.0, -0.0, 2.0]])
def test_sqrt_array(x):
    _check("sqrt", x)


def test_sqrt_neg_is_complex_pos_is_real():
    assert np.asarray(_ferray("sqrt", -1)).dtype == np.complex128
    assert np.asarray(_ferray("sqrt", 4)).dtype == np.float64


# ---------------------------------------------------------------------------
# log / log2 / log10 — _fix_real_lt_zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["log", "log2", "log10"])
@pytest.mark.parametrize("x", [-1, np.e, 1, -4, 8, 100, -100, 0.5])
def test_log_family_scalar(fn, x):
    _check(fn, x)


@pytest.mark.parametrize("fn", ["log", "log2", "log10"])
@pytest.mark.parametrize("x", [[-4, -8, 8], [1, -1], [4, 9, 16], [-1, 2, -3]])
def test_log_family_array(fn, x):
    _check(fn, x)


def test_log_e_is_real_one():
    assert np.asarray(_ferray("log", np.e)).dtype == np.float64
    assert np.asarray(_ferray("log", -1)).dtype == np.complex128


# ---------------------------------------------------------------------------
# logn — _fix_real_lt_zero on both n and x, then log(x)/log(n)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,x", [
    (2, 8), (2, 4), (3, 27), (10, 1000),
    (2, -4), (-2, 8), (2, [-4, -8, 8]), (2, [4, 8]),
])
def test_logn(n, x):
    _check("logn", n, x)


def test_logn_real_vs_complex_dtype():
    assert np.asarray(_ferray("logn", 2, 8)).dtype == np.float64
    assert np.asarray(_ferray("logn", 2, -4)).dtype == np.complex128


# ---------------------------------------------------------------------------
# power — _fix_real_lt_zero(base) + _fix_int_lt_zero(exp)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x,p", [
    (2, 2), (2, -2), (-1, 0.5), (-2, 2), (4, 0.5), (-4, 0.5), (3, 3),
])
def test_power_scalar(x, p):
    _check("power", x, p)


@pytest.mark.parametrize("x,p", [
    ([2, 4], 2), ([2, 4], -2), ([-2, 4], 2), ([2, 4], [2, 4]),
    ([-1, -2], 0.5), ([-1, -2], [0.5, 0.5]),
])
def test_power_array(x, p):
    _check("power", x, p)


def test_power_negexp_promotes_to_float_not_intpow():
    # _fix_int_lt_zero: power([2,4], -2) is float [0.25, 0.0625], NOT int [0,0].
    got = _ferray("power", [2, 4], -2)
    _assert_matches(got, _oracle("power", [2, 4], -2))
    assert np.asarray(got).dtype == np.float64


def test_power_negbase_is_complex():
    assert np.asarray(_ferray("power", -1, 0.5)).dtype == np.complex128
    assert np.asarray(_ferray("power", 2, 2)).dtype == np.int64


# ---------------------------------------------------------------------------
# arccos / arcsin / arctanh — _fix_real_abs_gt_1 (branch-cut sensitive)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", ["arccos", "arcsin", "arctanh"])
@pytest.mark.parametrize("x", [0.5, -0.5, 0.0, 2, -2, 1.5, -1.5, 5, -5, 3, -3])
def test_inverse_trig_scalar(fn, x):
    _check(fn, x)


@pytest.mark.parametrize("fn", ["arccos", "arcsin", "arctanh"])
@pytest.mark.parametrize("x", [
    [0.5, 2, -3, 1],          # mixed in/out of domain, both signs
    [2, 5, -2, -5],           # all out of domain, both signs
    [0.5, -0.5, 0.0],         # all in domain -> stays real
    [1.5, -1.5],
])
def test_inverse_trig_array_branch_cut(fn, x):
    """The out-of-domain branch differs by sign of the real part — exercises
    the per-element branch correction (numpy's ``x + 0.0j`` branch)."""
    _check(fn, x)


def test_inverse_trig_in_domain_real_out_of_domain_complex():
    assert np.asarray(_ferray("arccos", 0.5)).dtype == np.float64
    assert np.asarray(_ferray("arccos", 2)).dtype == np.complex128
    assert np.asarray(_ferray("arcsin", 0.5)).dtype == np.float64
    assert np.asarray(_ferray("arcsin", 2)).dtype == np.complex128
    assert np.asarray(_ferray("arctanh", 0.5)).dtype == np.float64
    assert np.asarray(_ferray("arctanh", 2)).dtype == np.complex128


# ---------------------------------------------------------------------------
# Genuinely complex inputs — pass straight through the complex kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fn", [
    "sqrt", "log", "log2", "log10", "arccos", "arcsin", "arctanh",
])
@pytest.mark.parametrize("arr", [
    [2 + 1j, -3 - 2j, 0.5 + 0.5j],
    [1j, -1j],
])
def test_complex_input_passthrough(fn, arr):
    _check(fn, np.asarray(arr))


def test_power_complex_input():
    _check("power", np.asarray([2 + 1j, -1 + 0.5j]), 2)
