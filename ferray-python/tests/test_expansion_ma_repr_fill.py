"""Expansion: numpy.ma-faithful ``__repr__``/``__str__`` (#884) + binary-op /
unary fill_value propagation (#885).

Every expected value is derived LIVE from ``numpy.ma`` (numpy 2.4.x is the
oracle, R-CHAR-3) — the ferray result is compared byte-for-byte against the
equivalent ``numpy.ma`` operation, never against a literal copied from the
ferray side.

#884: ``repr(fr.ma.array(...))`` must equal ``repr`` of the equivalent
``numpy.ma.MaskedArray`` (``masked_array(data=..., mask=..., fill_value=...)``
with ``--`` for masked positions), and ``str(...)`` must equal numpy's
``str`` (the bare data array with ``--`` for masked).

#885: ``(a OP b).fill_value`` carries the RECEIVER operand's fill when the
receiver's fill is MATERIALIZED, else the per-dtype default — matching numpy's
``_update_from(self)`` carry (``numpy/ma/core.py:3040``).
"""

import numpy as np
import pytest

import ferray as fr


# --------------------------------------------------------------------------
# Helpers: build a matched (ferray, numpy) masked-array pair from the SAME
# data + mask, so the numpy side is the live oracle for the ferray side.
# --------------------------------------------------------------------------
def _pair(data, mask=None, dtype=None):
    # NOTE: `mask=None` is NOT passed through to the constructors. numpy treats
    # an EXPLICIT `mask=None` as a request for a REAL all-False mask, while
    # omitting the kwarg yields the `nomask` singleton (verified live numpy
    # 2.4.4); ferray's `array(mask=None)` yields nomask. To compare matched
    # mask STATES (the subject of #884's repr), the nomask cases omit the kwarg
    # on BOTH sides — keeping the construction divergence (a separate, spillover
    # concern) out of the repr/str comparison.
    if mask is None:
        fa = fr.ma.array(data, dtype=dtype)
        na = np.ma.array(data, dtype=dtype)
    else:
        fa = fr.ma.array(data, mask=mask, dtype=dtype)
        na = np.ma.array(data, mask=mask, dtype=dtype)
    return fa, na


# --------------------------------------------------------------------------
# #884 — __repr__ across dtypes / shapes / mask states.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "data,mask,dtype",
    [
        # masked 1-d float
        ([1.0, 2.0, 3.0], [0, 1, 0], None),
        # unmasked 1-d float (nomask -> repr shows `mask=False`)
        ([1.0, 2.0, 3.0], None, None),
        # explicit all-False mask (real mask -> `mask=[False, ...]`)
        ([1.0, 2.0, 3.0], [0, 0, 0], None),
        # int masked
        ([1, 2, 3], [0, 1, 0], None),
        # int unmasked
        ([10, 20, 30], None, None),
        # bool masked
        ([True, False, True], [0, 1, 0], None),
        # 2-d masked
        ([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [1, 0]], None),
        # 2-d unmasked
        ([[1, 2], [3, 4]], None, None),
        # all-masked
        ([1.0, 2.0, 3.0], [1, 1, 1], None),
        # empty
        ([], None, "float64"),
        ([], None, "int64"),
    ],
)
def test_repr_matches_numpy(data, mask, dtype):
    fa, na = _pair(data, mask, dtype)
    assert repr(fa) == repr(na), f"\nferray:\n{repr(fa)}\nnumpy:\n{repr(na)}"


@pytest.mark.parametrize(
    "data,mask,dtype",
    [
        ([1.0, 2.0, 3.0], [0, 1, 0], None),
        ([1.0, 2.0, 3.0], None, None),
        ([1, 2, 3], [0, 1, 0], None),
        ([True, False, True], [0, 1, 0], None),
        ([[1.0, 2.0], [3.0, 4.0]], [[0, 1], [1, 0]], None),
        ([1.0, 2.0, 3.0], [1, 1, 1], None),
        ([], None, "float64"),
    ],
)
def test_str_matches_numpy(data, mask, dtype):
    fa, na = _pair(data, mask, dtype)
    assert str(fa) == str(na), f"\nferray:\n{str(fa)}\nnumpy:\n{str(na)}"


def test_repr_complex_masked():
    fa = fr.ma.array([1 + 2j, 3 + 4j, 5 - 6j], mask=[0, 1, 0])
    na = np.ma.array([1 + 2j, 3 + 4j, 5 - 6j], mask=[0, 1, 0])
    assert repr(fa) == repr(na)
    assert str(fa) == str(na)


def test_repr_0d_masked():
    fa = fr.ma.array(5.0, mask=True)
    na = np.ma.array(5.0, mask=True)
    assert repr(fa) == repr(na)
    assert str(fa) == str(na)


def test_repr_0d_unmasked_scalar_array():
    fa = fr.ma.array(7.0)
    na = np.ma.array(7.0)
    assert repr(fa) == repr(na)
    assert str(fa) == str(na)


def test_repr_uses_effective_fill_value():
    # A materialized fill_value must appear in the repr block (#882/#883
    # overlay flows into the numpy-delegated repr).
    fa = fr.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    fr.ma.set_fill_value(fa, 9.0)
    na = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    na.fill_value = 9.0
    assert repr(fa) == repr(na)
    assert "fill_value=9.0" in repr(fa)


def test_repr_large_array_summarized():
    # numpy summarizes large arrays (`...`); the delegation inherits that.
    data = list(range(1000))
    mask = [i % 7 == 0 for i in range(1000)]
    fa, na = _pair(data, mask)
    assert repr(fa) == repr(na)
    assert str(fa) == str(na)


# --------------------------------------------------------------------------
# #885 — binary-op fill_value propagation (receiver's materialized fill).
# --------------------------------------------------------------------------
def _fr_with_fill(data, mask, fill):
    a = fr.ma.array(data, mask=mask)
    fr.ma.set_fill_value(a, fill)
    return a


def _np_with_fill(data, mask, fill):
    a = np.ma.array(data, mask=mask)
    a.fill_value = fill
    return a


def test_default_fill_not_propagated_as_nondefault():
    # Default (non-materialized) operand -> result keeps the per-dtype default.
    fa = fr.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    na = np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0])
    assert (fa + 1).fill_value == (na + 1).fill_value
    assert (fa + 1).fill_value == na.fill_value  # default 1e20


@pytest.mark.parametrize(
    "op",
    [
        lambda a: a + 1,
        lambda a: 1 + a,  # reflected: self is still the masked receiver
        lambda a: a - 2,
        lambda a: 2 - a,
        lambda a: a * 3,
        lambda a: 3 * a,
        lambda a: a / 2,
        lambda a: a // 2,
        lambda a: a % 2,
        lambda a: a**2,
    ],
)
def test_materialized_fill_propagates_scalar(op):
    fa = _fr_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    na = _np_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    assert op(fa).fill_value == op(na).fill_value


@pytest.mark.parametrize("uop", [lambda a: -a, lambda a: +a, lambda a: abs(a)])
def test_unary_fill_propagates(uop):
    fa = _fr_with_fill([1.0, -2.0, 3.0], [0, 1, 0], 9.0)
    na = _np_with_fill([1.0, -2.0, 3.0], [0, 1, 0], 9.0)
    assert uop(fa).fill_value == uop(na).fill_value


def test_unary_default_fill():
    fa = fr.ma.array([1.0, -2.0, 3.0], mask=[0, 1, 0])
    na = np.ma.array([1.0, -2.0, 3.0], mask=[0, 1, 0])
    assert (-fa).fill_value == (-na).fill_value


def test_two_masked_operands_left_materialized():
    # Receiver (left) materialized -> result carries left fill.
    fa = _fr_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    fb = fr.ma.array([4.0, 5.0, 6.0], mask=[1, 0, 0])  # default fill
    na = _np_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    nb = np.ma.array([4.0, 5.0, 6.0], mask=[1, 0, 0])
    assert (fa + fb).fill_value == (na + nb).fill_value


def test_two_masked_operands_right_materialized_only():
    # Receiver (left) is default; right is materialized -> result is DEFAULT
    # (numpy ignores the other operand's fill).
    fa = fr.ma.array([4.0, 5.0, 6.0], mask=[1, 0, 0])  # default fill
    fb = _fr_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    na = np.ma.array([4.0, 5.0, 6.0], mask=[1, 0, 0])
    nb = _np_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    assert (fa + fb).fill_value == (na + nb).fill_value


def test_two_masked_operands_both_materialized():
    fa = _fr_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    fb = _fr_with_fill([4.0, 5.0, 6.0], [1, 0, 0], 7.0)
    na = _np_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    nb = _np_with_fill([4.0, 5.0, 6.0], [1, 0, 0], 7.0)
    assert (fa + fb).fill_value == (na + nb).fill_value  # left -> 9
    assert (fb + fa).fill_value == (nb + na).fill_value  # left -> 7


def test_int_fill_propagates_through_truediv_to_float():
    # `/` promotes int operands to float64; the materialized int fill is
    # re-coerced to the float result dtype (numpy: 7 -> 7.0).
    fa = _fr_with_fill([1, 2, 3], [0, 1, 0], 7)
    na = _np_with_fill([1, 2, 3], [0, 1, 0], 7)
    r_fr = fa / 2
    r_np = na / 2
    assert str(r_fr.dtype) == str(r_np.dtype) == "float64"
    assert r_fr.fill_value == r_np.fill_value


def test_comparison_does_not_propagate_fill():
    # Comparisons yield a bool result with the DEFAULT bool fill (numpy does
    # NOT carry the operand's float fill through a comparison).
    fa = _fr_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    na = _np_with_fill([1.0, 2.0, 3.0], [0, 1, 0], 9.0)
    assert (fa > 1).fill_value == (na > 1).fill_value
