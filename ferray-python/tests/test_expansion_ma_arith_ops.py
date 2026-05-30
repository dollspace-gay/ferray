"""Parity tests for ``ferray.ma.MaskedArray`` binary arithmetic operators
(#862, REQ-1 R-A + REQ-6 R-F) against ``numpy.ma`` (numpy 2.4.5 oracle).

Covers the 7 binary ops ``+ - * / // % **`` and their reflected forms over
{scalar, list, ndarray, numpy.ma, fr.ma} operands, asserting parity of:

* the result dtype (NEP-50 promotion: ``int+int -> int`` with wrap,
  ``int/int -> float64``, ``int + python_int -> int64`` per numpy.ma
  materializing the scalar as ``np.asarray(1)``),
* the result mask (union of operand masks, plus divisor-zero / non-finite
  domain masking for ``/ // %``),
* the result data.

Expected values are taken from live ``numpy.ma`` calls (R-CHAR-3 — never
literal-copied from ferray). For the *domained* ops (``/ // %``) numpy
explicitly does not guarantee the data at masked positions
(``numpy/ma/core.py:1229`` "impossible to guarantee masked values"), so for
those we assert mask + dtype + the UNMASKED (compressed) data, while the
additive/multiplicative ops assert the full data buffer (numpy reverts masked
positions to the left operand deterministically, ``core.py:1096``).
"""

import warnings

import numpy as np
import pytest

import ferray as fr

warnings.simplefilter("ignore")  # numpy emits div-by-zero RuntimeWarnings


# ---------------------------------------------------------------------------
# Helpers — build matching fr.ma / numpy.ma operands and compare results.
# ---------------------------------------------------------------------------

def _fr(data, mask=None, dtype=None):
    return fr.ma.array(data, mask=mask, dtype=dtype)


def _np(data, mask=None, dtype=None):
    return np.ma.array(data, mask=mask, dtype=dtype)


def _mask_of(x):
    """Full bool mask array of a fr.ma / numpy.ma result, never the sentinel."""
    if isinstance(x, fr.ma.MaskedArray):
        return np.asarray(fr.ma.getmaskarray(x))
    return np.ma.getmaskarray(x)


def _assert_full(desc, fr_res, np_res):
    """Assert dtype + mask + FULL data parity (non-domained ops)."""
    assert str(fr_res.dtype) == str(np_res.dtype), (
        f"{desc}: dtype {fr_res.dtype} != {np_res.dtype}"
    )
    np.testing.assert_array_equal(
        _mask_of(fr_res), _mask_of(np_res), err_msg=f"{desc}: mask"
    )
    np.testing.assert_array_equal(
        np.asarray(fr_res.data), np.asarray(np_res.data), err_msg=f"{desc}: data"
    )


def _assert_domained(desc, fr_res, np_res):
    """Assert dtype + mask + UNMASKED data parity (domained ops `/ // %`)."""
    assert str(fr_res.dtype) == str(np_res.dtype), (
        f"{desc}: dtype {fr_res.dtype} != {np_res.dtype}"
    )
    fm, nm = _mask_of(fr_res), _mask_of(np_res)
    np.testing.assert_array_equal(fm, nm, err_msg=f"{desc}: mask")
    # Compare only the unmasked positions (numpy does not guarantee masked
    # data for domained ops).
    fd = np.asarray(fr_res.data)
    nd = np.asarray(np_res.data)
    np.testing.assert_allclose(
        fd[~nm], nd[~nm], rtol=1e-12, atol=0, err_msg=f"{desc}: unmasked data"
    )


# ---------------------------------------------------------------------------
# Scalar operands (R-F) + NEP-50-ish dtype (numpy.ma materializes scalars).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scalar", [1, 2, 3])
def test_scalar_add_sub_mul(scalar):
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    _assert_full(f"a+{scalar}", a + scalar, na + scalar)
    _assert_full(f"a-{scalar}", a - scalar, na - scalar)
    _assert_full(f"a*{scalar}", a * scalar, na * scalar)
    _assert_full(f"a**{scalar}", a ** scalar, na ** scalar)


def test_scalar_reflected():
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    _assert_full("1+a", 1 + a, 1 + na)
    _assert_full("10-a", 10 - a, 10 - na)
    _assert_full("3*a", 3 * a, 3 * na)
    _assert_domained("100//a", 100 // a, 100 // na)
    _assert_domained("100/a", 100 / a, 100 / na)
    _assert_domained("100%a", 100 % a, 100 % na)


def test_scalar_dtype_nep50():
    # int array + python int -> int64 (numpy.ma materializes the scalar as
    # np.asarray(1) == int64, NOT an NEP-50 weak scalar).
    a8, na8 = _fr([1, 2], dtype="int8"), _np([1, 2], dtype=np.int8)
    assert str((a8 + 1).dtype) == str((na8 + 1).dtype) == "int64"
    # int array + python float -> float64.
    assert str((a8 + 1.0).dtype) == str((na8 + 1.0).dtype) == "float64"
    # default int64 array + python int stays int64.
    a, na = _fr([1, 2, 3]), _np([1, 2, 3])
    assert str((a + 1).dtype) == str((na + 1).dtype) == "int64"
    # int / scalar -> float64 (true division).
    assert str((a / 2).dtype) == str((na / 2).dtype) == "float64"


# ---------------------------------------------------------------------------
# Array / list / numpy.ma / fr.ma operands (R-F) — mask union.
# ---------------------------------------------------------------------------

OPERAND_BUILDERS = [
    ("list", lambda: ([10, 20, 30], [10, 20, 30])),
    ("ndarray", lambda: (np.array([10, 20, 30]), np.array([10, 20, 30]))),
    (
        "numpy.ma",
        lambda: (
            np.ma.array([10, 20, 30], mask=[1, 0, 0]),
            np.ma.array([10, 20, 30], mask=[1, 0, 0]),
        ),
    ),
    (
        "fr.ma",
        lambda: (
            fr.ma.array([10, 20, 30], mask=[1, 0, 0]),
            np.ma.array([10, 20, 30], mask=[1, 0, 0]),
        ),
    ),
]


@pytest.mark.parametrize("kind,builder", OPERAND_BUILDERS)
def test_array_operands_additive(kind, builder):
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    fb, nb = builder()
    _assert_full(f"a+{kind}", a + fb, na + nb)
    _assert_full(f"a-{kind}", a - fb, na - nb)
    _assert_full(f"a*{kind}", a * fb, na * nb)


@pytest.mark.parametrize("kind,builder", OPERAND_BUILDERS)
def test_array_operands_domained(kind, builder):
    a, na = _fr([12, 24, 36], [0, 1, 0]), _np([12, 24, 36], [0, 1, 0])
    fb, nb = builder()
    _assert_domained(f"a/{kind}", a / fb, na / nb)
    _assert_domained(f"a//{kind}", a // fb, na // nb)
    _assert_domained(f"a%{kind}", a % fb, na % nb)


# ---------------------------------------------------------------------------
# Domain masking on divisor zero (R-E subset folded into #862).
# ---------------------------------------------------------------------------

def test_domain_zero_int():
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    z, nz = _fr([0, 2, 0]), _np([0, 2, 0])
    _assert_domained("int a/[0,2,0]", a / z, na / nz)
    _assert_domained("int a//[0,2,0]", a // z, na // nz)
    _assert_domained("int a%[0,2,0]", a % z, na % nz)
    # The two zero-divisor positions (0 and 2) are masked.
    assert _mask_of(a / z).tolist() == [True, True, True]


def test_domain_zero_float():
    a, na = _fr([1.0, 2.0, 3.0]), _np([1.0, 2.0, 3.0])
    z, nz = _fr([0.0, 2.0, 0.0]), _np([0.0, 2.0, 0.0])
    _assert_domained("float a/[0,2,0]", a / z, na / nz)
    _assert_domained("float a//[0,2,0]", a // z, na // nz)
    _assert_domained("float a%[0,2,0]", a % z, na % nz)


def test_domain_scalar_zero():
    a, na = _fr([1.0, 2.0]), _np([1.0, 2.0])
    _assert_domained("a/0", a / 0, na / 0)
    assert _mask_of(a / 0).tolist() == [True, True]


# ---------------------------------------------------------------------------
# Mask-materialization parity: non-domained both-nomask keeps nomask;
# domained both-nomask materializes a real mask (`numpy/ma/core.py:1077`).
# ---------------------------------------------------------------------------

def test_nomask_materialization_additive():
    r = fr.ma.array([1, 2, 3]) + 1
    assert fr.ma.getmask(r) is fr.ma.nomask
    assert np.ma.getmask(np.ma.array([1, 2, 3]) + 1) is np.ma.nomask


def test_nomask_one_masked_materializes():
    # One masked operand -> result carries a real (all-but) mask array.
    r = fr.ma.array([1, 2, 3]) + fr.ma.array([4, 5, 6], mask=[1, 0, 0])
    nr = np.ma.array([1, 2, 3]) + np.ma.array([4, 5, 6], mask=[1, 0, 0])
    assert fr.ma.getmask(r) is not fr.ma.nomask
    np.testing.assert_array_equal(_mask_of(r), nr.mask)


def test_domained_nomask_materializes_real_mask():
    r = fr.ma.array([4.0, 6.0]) / fr.ma.array([2.0, 3.0])
    assert fr.ma.getmask(r) is not fr.ma.nomask
    assert np.ma.getmask(np.ma.array([4.0, 6.0]) / np.ma.array([2.0, 3.0])) \
        is not np.ma.nomask


# ---------------------------------------------------------------------------
# dtype matrix: signed / unsigned / float operand-common promotion.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["int8", "int16", "int32", "int64",
                                "uint8", "uint16", "uint32", "uint64",
                                "float32", "float64"])
def test_dtype_preservation_additive(dt):
    a = _fr([3, 4, 5], [0, 1, 0], dtype=dt)
    na = _np([3, 4, 5], [0, 1, 0], dtype=dt)
    b = _fr([1, 2, 3], dtype=dt)
    nb = _np([1, 2, 3], dtype=dt)
    _assert_full(f"{dt} a+b", a + b, na + nb)
    _assert_full(f"{dt} a*b", a * b, na * nb)
    if dt.startswith("int") or dt.startswith("float"):
        _assert_full(f"{dt} a-b", a - b, na - nb)
    # true division always promotes to float for these dtypes.
    _assert_domained(f"{dt} a/b", a / b, na / nb)


@pytest.mark.parametrize("dt", ["int8", "int32", "int64", "uint8",
                                "float32", "float64"])
def test_dtype_preservation_floordiv_mod_pow(dt):
    a = _fr([7, 8, 9], [0, 1, 0], dtype=dt)
    na = _np([7, 8, 9], [0, 1, 0], dtype=dt)
    b = _fr([2, 3, 4], dtype=dt)
    nb = _np([2, 3, 4], dtype=dt)
    _assert_domained(f"{dt} a//b", a // b, na // nb)
    _assert_domained(f"{dt} a%b", a % b, na % nb)
    _assert_full(f"{dt} a**b", a ** _fr([2, 1, 2], dtype=dt),
                 na ** _np([2, 1, 2], dtype=dt))


# ---------------------------------------------------------------------------
# Broadcasting + shape-mismatch (R-F shape rule).
# ---------------------------------------------------------------------------

def test_broadcast_row_plus_col():
    a = _fr([[1, 2, 3]], [[0, 1, 0]])
    na = _np([[1, 2, 3]], [[0, 1, 0]])
    b = _fr([[10], [20]])
    nb = _np([[10], [20]])
    _assert_full("broadcast a+b", a + b, na + nb)
    _assert_domained("broadcast a/b", a / b, na / nb)


def test_broadcast_scalar_array():
    a = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    na = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    _assert_full("2d a+10", a + 10, na + 10)
    _assert_full("2d a*2", a * 2, na * 2)


def test_shape_mismatch_raises():
    a = _fr([1, 2, 3])
    with pytest.raises((ValueError, Exception)):
        _ = a + fr.ma.array([1, 2])


# ---------------------------------------------------------------------------
# int + float cross-dtype promotion (NEP-50 via numpy.result_type).
# ---------------------------------------------------------------------------

def test_int_plus_float_array_promotes():
    a, na = _fr([1, 2, 3], [0, 1, 0]), _np([1, 2, 3], [0, 1, 0])
    b, nb = _fr([1.5, 2.5, 3.5]), _np([1.5, 2.5, 3.5])
    _assert_full("int+float", a + b, na + nb)
    assert str((a + b).dtype) == "float64"
    _assert_domained("int/float", a / b, na / nb)


def test_int32_plus_int64_promotes():
    a = _fr([1, 2, 3], [0, 1, 0], dtype="int32")
    na = _np([1, 2, 3], [0, 1, 0], dtype=np.int32)
    b, nb = _fr([10, 20, 30], dtype="int64"), _np([10, 20, 30], dtype=np.int64)
    _assert_full("int32+int64", a + b, na + nb)
    assert str((a + b).dtype) == str((na + nb).dtype) == "int64"
