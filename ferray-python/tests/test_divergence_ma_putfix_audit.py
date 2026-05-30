"""ACToR critic RE-AUDIT pins — numpy.ma put/putmask fix (commit 4dc7aea1).

Re-audit of the acto-fixer commit 4dc7aea1 which rewrote the hard-mask branch
of `ferray-ma/src/put.rs` (MaskedArray::put) to zero-pad short `values`
(matching numpy's `values.resize(indices.shape)`), gated on
`is_hard_mask() && has_real_mask()`.

The rewrite introduced a NEW divergence in the hard-branch GATE. numpy keys the
hard-mask branch off `self._mask is not nomask` (numpy/ma/core.py:4899). The
zero-pad branch runs only when `self._hardmask and self._mask is not nomask`;
otherwise control falls through to `self._data.put(indices, values, mode=mode)`
(numpy/ma/core.py:4907), which CYCLES a short `values`.

So a *hardened array with NO explicit mask* (`_mask is nomask`) skips the
zero-pad branch and falls through to the plain `ndarray.put`, which CYCLES a
short `values`. ferray's Python binding (`ferray-python/src/ma.rs:200-214`,
`PyMaskedArray::py_new`) ALWAYS constructs an explicit all-False mask when
`mask=None` and calls `RustMa::new` (`ferray-ma/src/masked_array.rs:189`,
`real_mask: true`). So every ferray.ma array has `real_mask == true`, the
hard-branch gate `has_real_mask()` is always satisfied, and a hardened
no-mask array wrongly ZERO-PADS where numpy CYCLES.

Expected values are computed from `numpy.ma` directly at runtime (R-CHAR-3 —
never literal-copied from the ferray side).

Run:
    cd ferray-python && .venv/bin/python -m pytest \
        tests/test_divergence_ma_putfix_audit.py -q
"""

import numpy as np
import numpy.ma as ma

import ferray as fr


def _fdata(a):
    return [float(x) for x in np.asarray(a.data)]


def _fmask(a, n):
    m = a.mask
    if m is ma.nomask:
        return [False] * n
    return [bool(b) for b in np.asarray(m)]


def _ndata(a):
    return [float(x) for x in np.asarray(a.data)]


def _nmask(a):
    return [bool(b) for b in ma.getmaskarray(a)]


# ---------------------------------------------------------------------------
# NEW DIVERGENCE D-NEW-1 — hard-mask put on an array with NO explicit mask
# CYCLES in numpy but ZERO-PADS in ferray.
#
# numpy/ma/core.py:4899 gates the zero-pad branch on
# `self._hardmask and self._mask is not nomask`; a `ma.array([...])` with no
# mask has `_mask is nomask`, so harden_mask() does NOT enable that branch and
# control reaches `self._data.put(...)` (numpy/ma/core.py:4907) which CYCLES.
#
# ferray (ferray-python/src/ma.rs:200-214) forces an explicit all-False mask
# for `mask=None`, so `has_real_mask()` is always true and put.rs:150/162
# takes the zero-pad branch — `data[2]` becomes 0.0 instead of the cycled 10.0.
#
# numpy oracle: data -> [10, 20, 10, 4, 5]   (CYCLE of [10,20])
# ferray today: data -> [10, 20,  0, 4, 5]   (ZERO-PAD)
# Tracking: see crosslink blocker issue referenced in the audit report.
# ---------------------------------------------------------------------------
def test_divergence_hard_put_nomask_array_cycles_not_zeropad():
    # --- numpy oracle (the source of truth) ---
    na = ma.array([1.0, 2, 3, 4, 5])  # NO mask -> _mask is nomask
    assert na._mask is ma.nomask  # precondition: numpy stays on nomask path
    na.harden_mask()
    ma.put(na, [0, 1, 2], [10.0, 20.0])
    expected_data = _ndata(na)
    expected_mask = _nmask(na)

    # --- ferray under test ---
    fa = fr.ma.array([1.0, 2, 3, 4, 5])  # NO mask
    fr.ma.harden_mask(fa)
    fr.ma.put(fa, [0, 1, 2], [10.0, 20.0])

    assert _fdata(fa) == expected_data, (
        f"hard+nomask put must CYCLE like numpy ndarray.put: "
        f"ferray={_fdata(fa)} numpy={expected_data}"
    )
    assert _fmask(fa, 5) == expected_mask


# ---------------------------------------------------------------------------
# NON-REGRESSION CONTROL — hard-mask put WITH an explicit all-False mask MUST
# zero-pad (this is the case numpy's `_mask is not nomask` branch covers and
# the D1 fix correctly handles). Kept here so the pin file documents the exact
# boundary: explicit-mask -> zero-pad (passes today), nomask -> cycle (fails).
# ---------------------------------------------------------------------------
def test_control_hard_put_explicit_allfalse_mask_zeropads():
    na = ma.array([1.0, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0])  # explicit -> not nomask
    assert na._mask is not ma.nomask
    na.harden_mask()
    ma.put(na, [0, 1, 2], [10.0, 20.0])
    expected = _ndata(na)  # numpy zero-pads -> [10, 20, 0, 4, 5]

    fa = fr.ma.array([1.0, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0])
    fr.ma.harden_mask(fa)
    fr.ma.put(fa, [0, 1, 2], [10.0, 20.0])
    assert _fdata(fa) == expected
