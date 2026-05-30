"""Parity tests for ``ferray.ma.MaskedArray`` bitwise / shift operators
(#865, REQ-3 R-C) against ``numpy.ma`` (numpy 2.4.x oracle).

Covers the 5 ops ``& | ^ << >>`` and their reflected forms over
{scalar, list, ndarray, numpy.ma, fr.ma} operands, asserting parity of:

* the result dtype (NEP-50 *weak* scalar promotion: ``int8 & 1 -> int8``,
  ``bool & bool -> bool``, ``bool & 1 -> int64``, shift ``bool << 1 -> int64``
  — numpy.ma inherits the bitwise dunders from ``ndarray`` and routes them
  through ``numpy.bitwise_and`` / ``numpy.left_shift`` etc. on the subclass,
  preserving weak-scalar promotion),
* the result mask (union of operand masks; ``nomask`` sentinel preserved
  when both operands are nomask — bitwise is NON-domained),
* the result data (raw ufunc result; numpy does not revert masked positions
  for the ndarray-inherited path).

Bitwise ops are defined for INTEGER + BOOL dtypes only; a FLOAT operand raises
``TypeError`` (numpy has no float ``bitwise_and`` / ``left_shift`` loop) — we
assert ``fr`` raises ``TypeError`` exactly where ``numpy.ma`` does.

Expected values come from live ``numpy.ma`` calls (R-CHAR-3 — never
literal-copied from ferray).
"""

import warnings

import numpy as np
import pytest

import ferray as fr

warnings.simplefilter("ignore")  # numpy may emit shift/overflow RuntimeWarnings


# ---------------------------------------------------------------------------
# Helpers — build matching fr.ma / numpy.ma operands and compare results.
# ---------------------------------------------------------------------------

INT_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


# `mask=None` (explicit) builds a REAL all-False mask in both numpy and ferray
# (#886); these helpers intend a genuine *nomask* array when no mask is given, so
# they OMIT `mask=` entirely (the absent path -> nomask) rather than forwarding
# `mask=None`.
def _fr(data, mask=None, dtype=None):
    if mask is None:
        return fr.ma.array(data, dtype=dtype)
    return fr.ma.array(data, mask=mask, dtype=dtype)


def _np(data, mask=None, dtype=None):
    if mask is None:
        return np.ma.array(data, dtype=dtype)
    return np.ma.array(data, mask=mask, dtype=dtype)


def _mask_of(x):
    """Full bool mask array of a fr.ma / numpy.ma result, never the sentinel."""
    if isinstance(x, fr.ma.MaskedArray):
        return np.asarray(fr.ma.getmaskarray(x))
    return np.ma.getmaskarray(x)


def _is_nomask(x):
    """True iff the masked array carries the nomask sentinel (no real mask)."""
    if isinstance(x, fr.ma.MaskedArray):
        return fr.ma.getmask(x) is fr.ma.nomask or fr.ma.getmask(x) is False
    return x.mask is np.ma.nomask


def _assert_parity(desc, fr_res, np_res):
    """Assert dtype + mask + FULL data parity (bitwise is non-domained)."""
    assert str(fr_res.dtype) == str(np_res.dtype), (
        f"{desc}: dtype {fr_res.dtype} != {np_res.dtype}"
    )
    np.testing.assert_array_equal(
        _mask_of(fr_res), _mask_of(np_res), err_msg=f"{desc}: mask"
    )
    np.testing.assert_array_equal(
        np.asarray(fr_res.data), np.asarray(np_res.data), err_msg=f"{desc}: data"
    )


# ---------------------------------------------------------------------------
# 1. Scalar operand, every integer dtype, all 5 ops + reflected.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", INT_DTYPES)
def test_scalar_all_int_dtypes(dt):
    fa = _fr([1, 2, 3, 12], mask=[0, 1, 0, 0], dtype=dt)
    na = _np([1, 2, 3, 12], mask=[0, 1, 0, 0], dtype=dt)
    for op in ("__and__", "__or__", "__xor__"):
        _assert_parity(
            f"{dt.__name__}.{op}(2)",
            getattr(fa, op)(2),
            getattr(na, op)(2),
        )
    for op in ("__lshift__", "__rshift__"):
        _assert_parity(
            f"{dt.__name__}.{op}(1)",
            getattr(fa, op)(1),
            getattr(na, op)(1),
        )


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_scalar_reflected(dt):
    fa = _fr([1, 2, 3], mask=[0, 1, 0], dtype=dt)
    na = _np([1, 2, 3], mask=[0, 1, 0], dtype=dt)
    _assert_parity("2 & a", 2 & fa, 2 & na)
    _assert_parity("2 | a", 2 | fa, 2 | na)
    _assert_parity("2 ^ a", 2 ^ fa, 2 ^ na)
    _assert_parity("2 << a", 2 << fa, 2 << na)
    _assert_parity("8 >> a", 8 >> fa, 8 >> na)


# ---------------------------------------------------------------------------
# 2. Bool operands — &|^ stay bool, & with int promote, shifts -> int.
# ---------------------------------------------------------------------------

def test_bool_and_or_xor():
    fa = _fr([True, False, True, False], mask=[0, 1, 0, 0])
    na = _np([True, False, True, False], mask=[0, 1, 0, 0])
    fb = _fr([True, True, False, False])
    nb = _np([True, True, False, False])
    _assert_parity("bool & bool", fa & fb, na & nb)
    _assert_parity("bool | bool", fa | fb, na | nb)
    _assert_parity("bool ^ bool", fa ^ fb, na ^ nb)


def test_bool_scalar_promote():
    # bool & python int -> int64 (NEP-50); bool & int8 -> int8.
    fa = _fr([True, False, True], mask=[0, 1, 0])
    na = _np([True, False, True], mask=[0, 1, 0])
    _assert_parity("bool & 1", fa & 1, na & 1)
    _assert_parity(
        "bool & int8", fa & _fr([1, 2, 3], dtype=np.int8), na & _np([1, 2, 3], dtype=np.int8)
    )


def test_bool_shift_promotes_to_int():
    fa = _fr([True, False, True], mask=[0, 1, 0])
    na = _np([True, False, True], mask=[0, 1, 0])
    _assert_parity("bool << 1", fa << 1, na << 1)
    _assert_parity("bool >> 1", fa >> 1, na >> 1)
    # bool << bool -> int8 per numpy
    _assert_parity(
        "bool << bool",
        fa << _fr([True, True, False]),
        na << _np([True, True, False]),
    )


# ---------------------------------------------------------------------------
# 3. Array / ndarray / numpy.ma / fr.ma operands; mask union.
# ---------------------------------------------------------------------------

def test_array_operands_mask_union():
    fa = _fr([1, 2, 3, 4], mask=[1, 0, 0, 0], dtype=np.int32)
    na = _np([1, 2, 3, 4], mask=[1, 0, 0, 0], dtype=np.int32)
    # fr.ma other operand
    fb = _fr([5, 6, 7, 8], mask=[0, 1, 0, 0], dtype=np.int32)
    nb = _np([5, 6, 7, 8], mask=[0, 1, 0, 0], dtype=np.int32)
    _assert_parity("ma & ma", fa & fb, na & nb)
    _assert_parity("ma | ma", fa | fb, na | nb)
    _assert_parity("ma ^ ma", fa ^ fb, na ^ nb)
    # plain list operand (nomask other)
    _assert_parity("ma & list", fa & [1, 1, 1, 1], na & [1, 1, 1, 1])
    # numpy.ma operand into fr; numpy.ndarray operand
    _assert_parity(
        "ma & np.ndarray",
        fa & np.array([3, 3, 3, 3], dtype=np.int32),
        na & np.array([3, 3, 3, 3], dtype=np.int32),
    )


def test_numpy_ma_operand():
    fa = _fr([1, 2, 3, 4], mask=[0, 0, 1, 0], dtype=np.int64)
    na = _np([1, 2, 3, 4], mask=[0, 0, 1, 0], dtype=np.int64)
    other = np.ma.array([5, 6, 7, 8], mask=[1, 0, 0, 0], dtype=np.int64)
    _assert_parity("ma & numpy.ma", fa & other, na & other)
    _assert_parity("ma ^ numpy.ma", fa ^ other, na ^ other)


# ---------------------------------------------------------------------------
# 4. Mixed integer dtype promotion (NEP-50 common dtype).
# ---------------------------------------------------------------------------

def test_mixed_int_dtype_promotion():
    fa = _fr([1, 2, 3], dtype=np.int8, mask=[0, 1, 0])
    na = _np([1, 2, 3], dtype=np.int8, mask=[0, 1, 0])
    fb = _fr([4, 5, 6], dtype=np.int32)
    nb = _np([4, 5, 6], dtype=np.int32)
    _assert_parity("int8 & int32", fa & fb, na & nb)  # -> int32
    _assert_parity("int8 | int32", fa | fb, na | nb)


# ---------------------------------------------------------------------------
# 5. nomask materialization — two nomask operands keep the nomask sentinel.
# ---------------------------------------------------------------------------

def test_nomask_preserved():
    fa = _fr([1, 2, 3], dtype=np.int32)
    na = _np([1, 2, 3], dtype=np.int32)
    fr_r = fa & 1
    np_r = na & 1
    # numpy keeps mask is nomask for nomask & scalar
    assert _is_nomask(np_r), "oracle precondition: numpy keeps nomask"
    assert _is_nomask(fr_r), "fr should keep nomask sentinel too"
    _assert_parity("nomask & 1", fr_r, np_r)


def test_real_mask_materialized_when_operand_masked():
    fa = _fr([1, 2, 3], dtype=np.int32, mask=[0, 1, 0])
    na = _np([1, 2, 3], dtype=np.int32, mask=[0, 1, 0])
    fr_r = fa & 1
    np_r = na & 1
    assert not _is_nomask(np_r)
    assert not _is_nomask(fr_r)
    _assert_parity("masked & 1", fr_r, np_r)


# ---------------------------------------------------------------------------
# 6. Float operand raises TypeError (no float bitwise loop).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expr",
    [
        lambda a: a & 1,
        lambda a: a | 1,
        lambda a: a ^ 1,
        lambda a: a << 1,
        lambda a: a >> 1,
        lambda a: 1 & a,
        lambda a: 1 << a,
    ],
)
def test_float_operand_raises_typeerror(expr):
    fa = _fr([1.0, 2.0, 3.0], dtype=np.float64, mask=[0, 1, 0])
    na = _np([1.0, 2.0, 3.0], dtype=np.float64, mask=[0, 1, 0])
    with pytest.raises(TypeError):
        expr(na)  # oracle: numpy.ma also raises TypeError
    with pytest.raises(TypeError):
        expr(fa)


def test_int_with_float_operand_raises():
    # int receiver, float other -> float common dtype -> TypeError
    fa = _fr([1, 2, 3], dtype=np.int32)
    na = _np([1, 2, 3], dtype=np.int32)
    with pytest.raises(TypeError):
        na & np.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        fa & np.array([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 7. Shift edge cases — negative counts, overflow (>= bit-width), per dtype.
# ---------------------------------------------------------------------------

SIGNED_DTYPES = [np.int8, np.int16, np.int32, np.int64]
UNSIGNED_DTYPES = [np.uint8, np.uint16, np.uint32, np.uint64]


@pytest.mark.parametrize("dt", SIGNED_DTYPES)
def test_shift_negative_count_signed(dt):
    # For signed dtypes numpy clamps a negative shift count to 0 (result 0).
    fa = _fr([1, 2, 4, 8], dtype=dt, mask=[0, 1, 0, 0])
    na = _np([1, 2, 4, 8], dtype=dt, mask=[0, 1, 0, 0])
    _assert_parity(f"{dt.__name__} << -1", fa << -1, na << -1)
    _assert_parity(f"{dt.__name__} >> -1", fa >> -1, na >> -1)


@pytest.mark.parametrize("dt", UNSIGNED_DTYPES)
def test_shift_negative_count_unsigned_raises(dt):
    # For unsigned dtypes a negative Python shift count is out of bounds:
    # numpy raises OverflowError; fr must raise the same.
    fa = _fr([1, 2, 4, 8], dtype=dt)
    na = _np([1, 2, 4, 8], dtype=dt)
    with pytest.raises(OverflowError):
        na << -1
    with pytest.raises(OverflowError):
        fa << -1


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_shift_overflow_count(dt):
    # Shift by a count >= bit-width: << -> 0; >> -> sign-fill (signed) / 0.
    big = 100
    fa = _fr([1, 2, 4, 8], dtype=dt, mask=[0, 1, 0, 0])
    na = _np([1, 2, 4, 8], dtype=dt, mask=[0, 1, 0, 0])
    _assert_parity(f"{dt.__name__} << {big}", fa << big, na << big)
    _assert_parity(f"{dt.__name__} >> {big}", fa >> big, na >> big)


def test_shift_negative_value_arithmetic():
    # signed >> keeps the sign (arithmetic shift); numpy oracle.
    fa = _fr([-1, -2, -8, 7], dtype=np.int32, mask=[0, 1, 0, 0])
    na = _np([-1, -2, -8, 7], dtype=np.int32, mask=[0, 1, 0, 0])
    _assert_parity("neg >> 1", fa >> 1, na >> 1)
    _assert_parity("neg >> 100", fa >> 100, na >> 100)


def test_shift_array_of_counts():
    fa = _fr([1, 2, 4, 8], dtype=np.int32, mask=[0, 0, 1, 0])
    na = _np([1, 2, 4, 8], dtype=np.int32, mask=[0, 0, 1, 0])
    counts_fr = _fr([0, 1, 2, 3], mask=[0, 0, 0, 1])
    counts_np = _np([0, 1, 2, 3], mask=[0, 0, 0, 1])
    _assert_parity("a << counts", fa << counts_fr, na << counts_np)
    _assert_parity("a >> counts", fa >> counts_fr, na >> counts_np)


# ---------------------------------------------------------------------------
# 8. Broadcasting + 2-D.
# ---------------------------------------------------------------------------

def test_broadcast_2d():
    fa = _fr([[1, 2, 3], [4, 5, 6]], dtype=np.int32, mask=[[0, 1, 0], [0, 0, 1]])
    na = _np([[1, 2, 3], [4, 5, 6]], dtype=np.int32, mask=[[0, 1, 0], [0, 0, 1]])
    fb = _fr([1, 2, 3], dtype=np.int32)
    nb = _np([1, 2, 3], dtype=np.int32)
    _assert_parity("2d & 1d (broadcast)", fa & fb, na & nb)
    _assert_parity("2d ^ scalar", fa ^ 7, na ^ 7)
    _assert_parity("2d << 1d", fa << fb, na << nb)


def test_broadcast_mismatch_raises():
    fa = _fr([1, 2, 3], dtype=np.int32)
    with pytest.raises(ValueError):
        fa & _fr([1, 2], dtype=np.int32)


# ---------------------------------------------------------------------------
# 9. uint-specific behaviour (logical >> on unsigned).
# ---------------------------------------------------------------------------

def test_uint_shift_logical():
    fa = _fr([255, 128, 1], dtype=np.uint8, mask=[0, 1, 0])
    na = _np([255, 128, 1], dtype=np.uint8, mask=[0, 1, 0])
    _assert_parity("uint8 >> 1", fa >> 1, na >> 1)
    _assert_parity("uint8 << 1", fa << 1, na << 1)  # wraps within uint8
    _assert_parity("uint8 & 0x0F", fa & 0x0F, na & 0x0F)
