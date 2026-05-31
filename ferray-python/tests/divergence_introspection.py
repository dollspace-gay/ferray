"""Adversarial divergence tests for ferray's dtype/array introspection surface.

Audited surface (registered in ferray-python/src/aliases.rs):
  result_type, promote_types, can_cast, min_scalar_type, common_type,
  issubdtype, isdtype, astype, ndim, shape, size, isscalar, isfortran, divmod.

Most of these delegate straight to the live numpy install
(``py.import("numpy")?.call_method1(...)`` in aliases.rs) and are therefore
bit-for-bit identical to numpy by construction — NO DIVERGENCE there.

The genuine divergences live in ``numpy.divmod``, which aliases.rs does NOT
delegate to numpy. Instead ``aliases::divmod`` (aliases.rs:252) *composes* the
ferray ``floor_divide`` + ``mod`` ufuncs (ufunc.rs), which dispatch through
``match_dtype_numeric!`` (conv.rs:967, no ``bool`` arm) and through the
ferray-ufunc integer floor-division kernel (ferray-ufunc/src/ops/arithmetic.rs)
which uses a *checked* ``/`` that panics on the ``INT_MIN / -1`` overflow.

Each test pins a CONFIRMED divergence. The oracle value/type is computed LIVE
(R-CHAR-3 — never literal-copied from the ferray side). These tests are
EXPECTED TO FAIL on the current build; they document work the generator must
fix. Written by the adversarial critic; contain no fixes.

Layer tags:
  [binding] — fix lives in ferray-python (aliases::divmod dtype dispatch /
              bool promotion before delegating to the ufuncs).
  [library] — fix lives down in ferray-ufunc (integer floor_divide / mod
              kernel: bool->int8 promotion, wrapping INT_MIN/-1).
"""

import warnings

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# divmod(bool, bool): numpy promotes bool -> int8 and computes the integer
# divmod; ferray's floor_divide/mod dispatch (match_dtype_numeric!, conv.rs:967)
# has no bool arm, so aliases::divmod (aliases.rs:252) raises TypeError.
# [binding]/[library]
# ---------------------------------------------------------------------------


def test_divmod_bool_bool_promotes_to_int8():
    """numpy's ``divmod`` ufunc has no bool loop, so bool operands promote to
    ``int8`` before the integer divmod runs (NEP-50 / ufunc type resolution;
    numpy/_core/code_generators/generate_umath.py 'divmod' = floor_divide +
    remainder, integer loops only). So
    ``np.divmod(array([True,False,True,False]), array([True,True,False,False]))``
    returns two ``int8`` arrays.

    Expected (numpy oracle, live below): q.dtype==int8, q==[1,0,0,0];
                                          r.dtype==int8, r==[0,0,0,0].
    Actual (ferray): raises ``TypeError`` ("unsupported dtype for numeric op:
                     'bool'") — divmod rejects the bool dtype entirely.
    """
    a = np.array([True, False, True, False])
    b = np.array([True, True, False, False])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy warns on the True//False = 1//0
        oq, orr = np.divmod(a, b)  # live oracle

    fq, frr = fr.divmod(a, b)
    fq = np.asarray(fq)
    frr = np.asarray(frr)

    assert fq.dtype == oq.dtype, f"q dtype: numpy {oq.dtype} vs ferray {fq.dtype}"
    assert frr.dtype == orr.dtype, f"r dtype: numpy {orr.dtype} vs ferray {frr.dtype}"
    assert fq.tolist() == oq.tolist()
    assert frr.tolist() == orr.tolist()


def test_divmod_scalar_bool_bool_returns_int8():
    """``np.divmod(True, True)`` returns ``(np.int8(1), np.int8(0))`` — bool
    promotes to int8. ferray's ``fr.divmod(True, True)`` raises ``TypeError``.

    Expected (numpy oracle, live): both elements int8, values (1, 0).
    Actual (ferray): TypeError.
    """
    oq, orr = np.divmod(True, True)  # live oracle: (int8(1), int8(0))

    fq, frr = fr.divmod(True, True)
    fq = np.asarray(fq)
    frr = np.asarray(frr)

    assert fq.dtype == np.asarray(oq).dtype
    assert (int(fq), int(frr)) == (int(oq), int(orr))


# ---------------------------------------------------------------------------
# divmod(INT_MIN, -1): integer overflow. numpy WRAPS (INT_MIN // -1 == INT_MIN,
# remainder 0) with a RuntimeWarning — it does NOT raise. ferray's integer
# floor_divide kernel (ferray-ufunc/src/ops/arithmetic.rs:492) uses a checked
# division that PANICS, surfacing as pyo3_runtime.PanicException across the FFI
# boundary. A panic crossing the boundary is a release blocker.
# [library]
# ---------------------------------------------------------------------------


def test_divmod_int8_min_over_neg1_wraps_not_panic():
    """numpy computes ``INT_MIN // -1`` by C wraparound: for int8,
    ``-128 // -1 == -128`` and ``-128 % -1 == 0`` (with a RuntimeWarning, NOT
    an exception; numpy/_core/src/umath/loops.c.src integer divmod loops wrap
    like C signed division). So ``np.divmod(np.int8(-128), np.int8(-1))`` is
    ``(np.int8(-128), np.int8(0))``.

    Expected (numpy oracle, live below): q==-128 (int8), r==0 (int8).
    Actual (ferray): raises ``PanicException`` — ferray-ufunc's integer
                     floor_divide kernel (arithmetic.rs:492) panics with
                     "attempt to divide with overflow" instead of wrapping.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # numpy: "overflow encountered in divmod"
        oq, orr = np.divmod(np.int8(-128), np.int8(-1))  # live oracle

    fq, frr = fr.divmod(np.int8(-128), np.int8(-1))
    fq = np.asarray(fq)
    frr = np.asarray(frr)

    assert fq.dtype == np.asarray(oq).dtype
    assert int(fq) == int(oq)  # -128
    assert int(frr) == int(orr)  # 0


def test_divmod_int64_min_over_neg1_wraps_not_panic():
    """Same INT_MIN/-1 wraparound at int64 width:
    ``np.divmod(np.int64(np.iinfo(np.int64).min), np.int64(-1))`` ==
    ``(INT64_MIN, 0)`` (numpy wraps; warns; never raises).

    Expected (numpy oracle, live): q == iinfo(int64).min, r == 0.
    Actual (ferray): PanicException from the int64 floor_divide kernel.
    """
    imin = np.int64(np.iinfo(np.int64).min)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oq, orr = np.divmod(imin, np.int64(-1))  # live oracle

    fq, frr = fr.divmod(imin, np.int64(-1))
    fq = np.asarray(fq)
    frr = np.asarray(frr)

    assert int(fq) == int(oq)
    assert int(frr) == int(orr)
