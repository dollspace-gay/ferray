"""ACToR critic pins — comprehensive audit of fr.ma.MaskedArray operator surface.

Audited commits (one pipeline in ferray-python/src/ma.rs):
  284696f1 (#861) unary __neg__/__pos__/__abs__/__invert__ + carry_unary_mask
  7c811431 (#862) arithmetic __add__..__pow__ + reflected + binary_op/compute_binary/BinOp
  96f892d8 (#863) comparison __richcmp__ + compare_op/compute_compare/CmpOp
  93a708b4 (#865) bitwise __and__/__or__/__xor__/__lshift__/__rshift__ + bitwise_op/compute_bitwise/BitOp

Every expected value is derived LIVE from numpy.ma (numpy 2.4.4 installed oracle),
per R-CHAR-3 — never literal-copied from the ferray side.

CONFIRMED DIVERGENCE pinned here:
  * `%` (remainder/__mod__) nomask identity — numpy.ma keeps `mask is nomask`
    when both operands are nomask and no zero divisor; ferray materializes a real
    all-False mask (BinOp::Mod is in `is_domained()` -> `want_real` forced true).

The non-divergent probe matrix (mask-union broadcast, domain masking on div/mod by
zero, NEP-50 dtype matrix, reflected ops, eq/ne vs masked, operand kinds, float
bitwise / bool unary exception types, shift edges incl. negative/>=width/array,
chained, in-place +=) all MATCHED numpy.ma and is recorded in the critic report,
not pinned (no failing assertion is warranted where behavior matches).
"""

import numpy as np
import numpy.ma as nma
import pytest

import ferray as fr

np.seterr(all="ignore")


def _fr_is_nomask(r):
    """ferray mirrors numpy's `nomask` sentinel exactly: `MaskedArray.mask`
    returns the `numpy.ma.nomask` singleton (a 0-d ``np.False_``) when the array
    carries no real mask (ferray-python ma.rs ``fn mask`` -> ``ma_nomask`` ->
    ``numpy.ma.nomask``), and a real bool ndarray otherwise. So the nomask
    identity check is ``r.mask is numpy.ma.nomask`` — the same identity numpy's
    own ``getmask(...) is nomask`` uses on the oracle side (R-CHAR-3)."""
    return r.mask is nma.nomask


# --------------------------------------------------------------------------
# CONFIRMED DIVERGENCE: `%` nomask identity.
#
# numpy.ma.MaskedArray has NO __mod__ override (numpy/ma/core.py:4355-4385 define
# __truediv__/__floordiv__ which route to the always-materializing
# _DomainedBinaryOperation, but there is NO __mod__). So `%` falls through to
# ndarray.__mod__ -> np.remainder ufunc, which keeps `mask is nomask` for two
# nomask operands with no zero divisor.
#
# ferray's BinOp::Mod is in `is_domained()` (ma.rs `fn is_domained`), so
# `want_real = op.is_domained() || ...` (ma.rs binary_op) FORCES a real mask for
# every `%`, diverging from numpy's nomask sentinel.
#
# Tracking: crosslink blocker under #860.
# --------------------------------------------------------------------------


def test_mod_two_nomask_int_keeps_nomask_like_numpy():
    """numpy: `ma([5,2,7]) % ma([2,3,4])` -> `mask is nomask` (True)."""
    np_is_nomask = nma.getmask(nma.array([5, 2, 7]) % nma.array([2, 3, 4])) is nma.nomask
    assert np_is_nomask is True  # live numpy contract anchor
    fr_res = fr.ma.array([5, 2, 7]) % fr.ma.array([2, 3, 4])
    assert _fr_is_nomask(fr_res) == np_is_nomask


def test_mod_two_nomask_float_keeps_nomask_like_numpy():
    """numpy: `ma([5.0,2.0]) % ma([2.0,3.0])` -> `mask is nomask` (True)."""
    np_is_nomask = nma.getmask(nma.array([5.0, 2.0]) % nma.array([2.0, 3.0])) is nma.nomask
    assert np_is_nomask is True
    fr_res = fr.ma.array([5.0, 2.0]) % fr.ma.array([2.0, 3.0])
    assert _fr_is_nomask(fr_res) == np_is_nomask


def test_rmod_scalar_over_nomask_keeps_nomask_like_numpy():
    """numpy reflected: `10 % ma([3,4,7])` -> `mask is nomask` (True)."""
    np_is_nomask = nma.getmask(10 % nma.array([3, 4, 7])) is nma.nomask
    assert np_is_nomask is True
    fr_res = 10 % fr.ma.array([3, 4, 7])
    assert _fr_is_nomask(fr_res) == np_is_nomask


def test_mod_by_zero_domain_still_masks_like_numpy():
    """Control: the domain DOES still fire at a zero divisor (must keep matching).

    numpy: `ma([5,2,7]) % ma([2,0,3])` -> mask [F,T,F]; ferray must agree.
    This guards the fix from over-correcting (dropping the zero-divisor domain).
    """
    npr = nma.array([5, 2, 7]) % nma.array([2, 0, 3])
    np_mask = np.asarray(nma.getmaskarray(npr)).tolist()
    assert np_mask == [False, True, False]
    fr_res = fr.ma.array([5, 2, 7]) % fr.ma.array([2, 0, 3])
    assert np.asarray(fr_res.mask).tolist() == np_mask
