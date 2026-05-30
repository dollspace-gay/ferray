"""ACToR-critic audit pins for commit 798495f3 (#869): complex masked arithmetic.

Every expected value is taken from a LIVE numpy 2.4.4 oracle call in-test
(R-CHAR-3 — never literal-copied from the ferray side). Each test asserts the
numpy-observable contract and FAILS against the current ferray build, pinning a
real divergence for the fixer.

Oracle: numpy.ma (`numpy/ma/core.py:1207` `_DomainedBinaryOperation.__call__`;
`numpy/ma/core.py:955` complex `negative`).
Target: ferray-python `src/ma.rs` complex_arm! + `__neg__`; ferray-ufunc
`src/ops/arithmetic.rs` `TrueDivide for Complex` / `power_complex`.
"""

import numpy as np
import pytest

try:
    import ferray as fr
except ImportError:  # pragma: no cover
    fr = None

pytestmark = pytest.mark.skipif(fr is None, reason="ferray not built")


# ---------------------------------------------------------------------------
# DIVERGENCE 1 — complex true-division spuriously masks a FINITE quotient.
#
# numpy.ma's `_DomainedBinaryOperation.__call__` masks a `/` slot iff
#   m = ~isfinite(result) | domain(da, db)
# with `_DomainSafeDivide`: domain == (|a|*tiny >= |b|)  (numpy/ma/core.py:900).
# For a = 2+0j, b = 6.675e-308+0j:  |a|*tiny = 4.45e-308 < |b|  → domain FALSE,
# and the true quotient 2/6.675e-308 ≈ 2.996e+307 is FINITE → numpy does NOT mask.
#
# ferray's complex_arm! computes the quotient with num_complex's Smith division,
# which OVERFLOWS this case to (inf + NaN*i) — non-finite — so ferray's
# `!r.is_finite()` domain check fires and ferray spuriously masks the slot.
# Root cause is in ferray-ufunc `TrueDivide for Complex` (num_complex Div).
# ---------------------------------------------------------------------------
def test_complex_divide_finite_quotient_not_masked():
    a, b = complex(2.0, 0.0), complex(6.675e-308, 0.0)
    an = np.ma.array([a], dtype=np.complex128)
    bn = np.ma.array([b], dtype=np.complex128)
    rn = an / bn
    np_mask = bool(np.asarray(rn.mask)[0])
    np_val = complex(np.asarray(rn.data)[0])

    af = fr.ma.array([a], dtype=np.complex128)
    bf = fr.ma.array([b], dtype=np.complex128)
    rf = af / bf
    fr_mask = bool(np.asarray(rf.mask)[0])
    fr_val = complex(np.asarray(rf.data)[0])

    # numpy leaves this UNMASKED with a finite quotient.
    assert np_mask is False
    assert np.isfinite(np_val)
    # ferray must match: unmasked, same finite quotient.
    assert fr_mask == np_mask, f"ferray masks a finite quotient numpy keeps: {fr_mask=}"
    assert fr_val == pytest.approx(np_val, rel=1e-12)


# ---------------------------------------------------------------------------
# DIVERGENCE 2 (tracked #10) — complex unary `__neg__` (`-a`) RAISES.
#
# numpy.ma computes `-(1+2j) = -1-2j`, mask-preserving (numpy/ma/core.py:955
# `negative`). ferray-python `ma.rs:1676` returns `complex_arith_pending`,
# raising TypeError. The follow-up note in code points at #869, now CLOSED, so
# the gap is currently UNTRACKED.
# ---------------------------------------------------------------------------
def test_complex_unary_negate_computes():
    src = [1 + 2j, 3 - 1j]
    an = np.ma.array(src, dtype=np.complex128)
    np_neg = np.asarray((-an).data)

    af = fr.ma.array(src, dtype=np.complex128)
    fr_neg = np.asarray((-af).data)  # currently raises TypeError → pin fails here

    np.testing.assert_array_equal(fr_neg, np_neg)


def test_complex_unary_negate_preserves_mask():
    src = [1 + 2j, 3 - 1j]
    am = np.ma.array(src, mask=[False, True], dtype=np.complex128)
    np_data = np.asarray((-am).data)
    np_mask = np.asarray((-am).mask)

    af = fr.ma.array(src, mask=[False, True], dtype=np.complex128)
    neg = -af  # currently raises TypeError → pin fails here
    np.testing.assert_array_equal(np.asarray(neg.data), np_data)
    np.testing.assert_array_equal(np.asarray(neg.mask), np_mask)
