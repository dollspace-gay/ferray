"""Parity tests for ``ferray.ma.MaskedArray`` comparison operators
(#863, REQ-2 R-B + REQ-7 R-G) against ``numpy.ma`` (numpy 2.4.x oracle).

Covers the six rich-comparison operators ``< <= > >= == !=`` over
{scalar, list, ndarray, numpy.ma, fr.ma} operands, the eq/ne mask-aware
special-case (``numpy/ma/core.py:4245``), the ``numpy.ma.masked`` singleton
operand, the ``nomask`` result sentinel, broadcasting, and every real dtype.

A comparison always yields a BOOL ``MaskedArray`` masked where either operand
is masked (``mask_or(smask, omask)``, ``numpy/ma/core.py:4206``). For ``==`` /
``!=`` ONLY, the underlying bool at masked positions is ``compare(smask,
omask)`` — equal iff both masked, unequal iff exactly one masked
(``numpy/ma/core.py:4245``). The four ordering ops keep the raw comparison
data under the mask.

Expected values come from live ``numpy.ma`` calls (R-CHAR-3 — never
literal-copied from ferray): each ``_assert_cmp`` compares ferray's result
against numpy.ma's result for the SAME inputs.
"""

import warnings

import numpy as np
import pytest

import ferray as fr

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fr(data, mask=None, dtype=None):
    return fr.ma.array(data, mask=mask, dtype=dtype)


def _np(data, mask=None, dtype=None):
    return np.ma.array(data, mask=mask, dtype=dtype)


def _mask_full(x):
    """Full bool mask of a fr.ma / numpy.ma result (never the sentinel)."""
    if isinstance(x, fr.ma.MaskedArray):
        return np.asarray(fr.ma.getmaskarray(x))
    return np.ma.getmaskarray(x)


def _is_nomask(x):
    """True iff the result carries the ``nomask`` sentinel (no real mask)."""
    if isinstance(x, fr.ma.MaskedArray):
        return x.mask is np.ma.nomask
    return x.mask is np.ma.nomask


def _assert_cmp(desc, fr_res, np_res):
    """Assert dtype(bool) + nomask-identity + mask + full data parity."""
    assert str(fr_res.dtype) == str(np_res.dtype) == "bool", (
        f"{desc}: dtype {fr_res.dtype} != {np_res.dtype} (want bool)"
    )
    # nomask result identity must match numpy (a comparison of two nomask
    # operands keeps `mask is nomask`, not a materialized all-False array).
    assert _is_nomask(fr_res) == _is_nomask(np_res), (
        f"{desc}: nomask identity {_is_nomask(fr_res)} != {_is_nomask(np_res)}"
    )
    np.testing.assert_array_equal(
        _mask_full(fr_res), _mask_full(np_res), err_msg=f"{desc}: mask"
    )
    # The eq/ne override makes the FULL data buffer deterministic for every op
    # (ordering ops keep the raw comparison; eq/ne overwrite masked slots), so
    # we can compare the whole buffer, not just the unmasked positions.
    np.testing.assert_array_equal(
        np.asarray(fr_res.data), np.asarray(np_res.data), err_msg=f"{desc}: data"
    )


def _all_ops(a, b):
    """Yield (label, fr_result, np_result) for the six ops; a is (fr, np)."""
    fa, na = a
    fb, nb = b
    yield "<", fa < fb, na < nb
    yield "<=", fa <= fb, na <= nb
    yield ">", fa > fb, na > nb
    yield ">=", fa >= fb, na >= nb
    yield "==", fa == fb, na == nb
    yield "!=", fa != fb, na != nb


# ---------------------------------------------------------------------------
# Scalar operand (R-B / R-G) — the core gap (`a > 2`, `a == 2`, …).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scalar", [0, 1, 2, 3, 4])
def test_scalar_all_ops(scalar):
    a = (_fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0]))
    for label, fr_res, np_res in _all_ops(a, (scalar, scalar)):
        _assert_cmp(f"a {label} {scalar}", fr_res, np_res)


def test_scalar_eq_ne_masked_position_data():
    # The worst prior bug: `a == 2` returned Python `False`. Now it is a bool
    # MaskedArray; the masked slot (value 2) carries `compare(smask=True,
    # omask=False)` -> `==` gives False, `!=` gives True (the eq/ne override).
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    _assert_cmp("a==2", a == 2, na == 2)
    _assert_cmp("a!=2", a != 2, na != 2)
    # Explicit byte-for-byte derivation from numpy (R-CHAR-3):
    np.testing.assert_array_equal((a == 2).data, (na == 2).data)
    np.testing.assert_array_equal((a != 2).data, (na != 2).data)


def test_reflected_scalar():
    # Python rewrites `2 < a` as `a.__gt__(2)`; PyO3 delivers it already
    # oriented, so the result must equal `a > 2` and numpy's `2 < na`.
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    _assert_cmp("2<a", 2 < a, 2 < na)
    _assert_cmp("2>a", 2 > a, 2 > na)
    _assert_cmp("2==a", 2 == a, 2 == na)
    _assert_cmp("2!=a", 2 != a, 2 != na)
    _assert_cmp("2<=a", 2 <= a, 2 <= na)
    _assert_cmp("2>=a", 2 >= a, 2 >= na)


# ---------------------------------------------------------------------------
# Array / list / numpy.ma / fr.ma operands (broadcast + mask union).
# ---------------------------------------------------------------------------

def test_list_operand():
    a = (_fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0]))
    b = ([1, 5, 3], [1, 5, 3])
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"a {label} list", fr_res, np_res)


def test_ndarray_operand():
    a = (_fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0]))
    arr = np.array([3, 2, 1])
    b = (arr, arr)
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"a {label} ndarray", fr_res, np_res)


def test_numpy_ma_operand():
    a = (_fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0]))
    nb = _np([1, 2, 3], [1, 1, 0])
    # Both operands use the SAME numpy.ma RHS; ferray reads its mask via
    # numpy.ma.getmaskarray, so the union mask must match.
    b = (nb, nb)
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"a {label} np.ma", fr_res, np_res)


def test_fr_ma_operand():
    a = (_fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0]))
    b = (_fr([1, 2, 3], [1, 1, 0]), _np([1, 2, 3], [1, 1, 0]))
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"a {label} fr.ma", fr_res, np_res)


def test_eq_ne_both_masked_special_case():
    # idx0: a unmasked, b masked -> compare(False,True): == False, != True.
    # idx1: both masked          -> compare(True,True):  == True,  != False.
    # idx2: both unmasked        -> raw 3==3 / 3!=3.
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    b, nb = _fr([1, 2, 3], [1, 1, 0]), _np([1, 2, 3], [1, 1, 0])
    _assert_cmp("a==b", a == b, na == nb)
    _assert_cmp("a!=b", a != b, na != nb)
    np.testing.assert_array_equal((a == b).data, (na == nb).data)
    np.testing.assert_array_equal((a != b).data, (na != nb).data)


# ---------------------------------------------------------------------------
# Broadcasting (1-D scalar, 2-D, row-vs-col).
# ---------------------------------------------------------------------------

def test_broadcast_2d_scalar():
    e = (_fr([[1, 2], [3, 4]], [[0, 1], [0, 0]]),
         _np([[1, 2], [3, 4]], [[0, 1], [0, 0]]))
    for label, fr_res, np_res in _all_ops(e, (2, 2)):
        _assert_cmp(f"e {label} 2", fr_res, np_res)


def test_broadcast_row_col():
    a = (_fr([[1, 2, 3]], [[0, 1, 0]]), _np([[1, 2, 3]], [[0, 1, 0]]))
    b = (_fr([[1], [3]], [[0], [1]]), _np([[1], [3]], [[0], [1]]))
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"row {label} col", fr_res, np_res)


def test_broadcast_mismatch_raises():
    a = _fr([1, 2, 3])
    with pytest.raises((ValueError, Exception)):
        _ = a == _fr([1, 2])


# ---------------------------------------------------------------------------
# nomask result sentinel (both operands nomask -> `mask is nomask`).
# ---------------------------------------------------------------------------

def test_nomask_result_sentinel():
    c, nc = _fr([1, 2, 3]), _np([1, 2, 3])
    for label, fr_res, np_res in _all_ops((c, nc), (2, 2)):
        _assert_cmp(f"nomask c {label} 2", fr_res, np_res)
        # numpy.ma.getmask returns the nomask singleton here.
        assert fr.ma.getmask(fr_res) is np.ma.nomask
        assert np.ma.getmask(np_res) is np.ma.nomask


def test_nomask_vs_masked_operand_materializes():
    # A nomask receiver compared with a MASKED operand DOES carry a real mask.
    c, nc = _fr([1, 2, 3]), _np([1, 2, 3])
    b, nb = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    for label, fr_res, np_res in _all_ops((c, nc), (b, nb)):
        _assert_cmp(f"nomask c {label} masked", fr_res, np_res)
        assert not _is_nomask(fr_res)


# ---------------------------------------------------------------------------
# The numpy.ma.masked singleton operand -> all positions masked.
# ---------------------------------------------------------------------------

def test_compare_to_masked_singleton():
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    m = np.ma.masked
    for label, op in (("==", lambda x: x == m), ("!=", lambda x: x != m)):
        fr_res, np_res = op(a), op(na)
        _assert_cmp(f"a {label} masked", fr_res, np_res)
        # Every position is masked (getmask(masked) is True scalar -> union).
        assert _mask_full(fr_res).all()
    # Ordering vs the masked singleton is also all-masked in numpy.ma.
    for label, op in ((">", lambda x: x > m), ("<", lambda x: x < m)):
        fr_res, np_res = op(a), op(na)
        _assert_cmp(f"a {label} masked", fr_res, np_res)


# ---------------------------------------------------------------------------
# Dtype coverage — every real dtype compares to a scalar -> bool result.
# ---------------------------------------------------------------------------

REAL_DTYPES = [
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64", "float32", "float64",
]


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_all_dtypes_scalar_compare(dt):
    base = [0, 1, 1] if dt == "bool" else [1, 2, 3]
    a = (_fr(base, [0, 1, 0], dtype=dt), _np(base, [0, 1, 0], dtype=np.dtype(dt)))
    scalar = True if dt == "bool" else 2
    for label, fr_res, np_res in _all_ops(a, (scalar, scalar)):
        _assert_cmp(f"{dt} {label} {scalar}", fr_res, np_res)


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_all_dtypes_array_eq(dt):
    base = [0, 1, 1] if dt == "bool" else [1, 2, 3]
    other = [1, 1, 0] if dt == "bool" else [1, 5, 3]
    a = (_fr(base, [0, 1, 0], dtype=dt), _np(base, [0, 1, 0], dtype=np.dtype(dt)))
    b = (_fr(other, dtype=dt), _np(other, dtype=np.dtype(dt)))
    for label, fr_res, np_res in _all_ops(a, b):
        _assert_cmp(f"{dt} {label} arr", fr_res, np_res)


def test_cross_dtype_int_vs_float():
    # int receiver vs float operand: the comparison runs over the promoted
    # common dtype (float64), result still bool.
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    for label, fr_res, np_res in _all_ops((a, na), (2.5, 2.5)):
        _assert_cmp(f"int {label} 2.5", fr_res, np_res)


# ---------------------------------------------------------------------------
# Float NaN under the mask (NaN comparisons are always False; mask carried).
# ---------------------------------------------------------------------------

def test_float_nan_comparison():
    a = (_fr([1.0, np.nan, 3.0], [0, 0, 1]),
         _np([1.0, np.nan, 3.0], [0, 0, 1]))
    for label, fr_res, np_res in _all_ops(a, (2.0, 2.0)):
        _assert_cmp(f"nan {label} 2.0", fr_res, np_res)
