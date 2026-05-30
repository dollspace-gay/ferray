"""Divergence pins from the re-audit of acto-builder commit 89cb6207 (#851).

The #851 fix carries a source mask over on a ``mask=None`` round-trip. This
re-audit (Objective A) walked every construction variant and found ONE residual
mask divergence — the ``keep_mask`` default precedence — and confirmed the
spillover ``__array__`` protocol divergence the builder flagged (#854,
Objective B).

Expected values come from a LIVE ``numpy.ma`` oracle in the same process
(R-CHAR-3) — never literal-copied from ferray. Each pin FAILS against the
current ferray build and pins the exact upstream contract.

The verified-matching cases (no pin needed) are recorded in the audit report:
A1 (all 5 claimed cases), A2 (2-D mask shape carry-over), A3 (0-d masked True /
False, empty masked), A5 (fr-nomask stays nomask), A6 (mask aligns after the
out-of-scope f64 data cast). A7 (keep_mask/shrink/hard_mask kwargs) is a noted
NOT-pinned divergence: ferray does not CLAIM to accept those kwargs (it raises
TypeError on an unexpected keyword), so per the audit scope it is recorded but
not pinned here.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Objective A.4 — explicit mask= given together with an already-masked source.
#
# numpy's MaskedArray.__new__ default is keep_mask=True. With a real mask
# ALREADY on the source AND an explicit mask= argument, numpy ORs the two:
#   numpy/ma/core.py:3008  `_data._mask = np.logical_or(mask, _data._mask)`
# (reached because `_data._mask is not nomask` and keep_mask is True, the
# default — core.py:2992 only REPLACES when `not keep_mask`).
#
# ferray's py_new (ferray-python/src/ma.rs) takes the `Some(m)` branch and
# builds the mask from the explicit `m` ALONE, discarding the source's mask —
# i.e. it behaves like numpy's keep_mask=False. Divergence -> tracking #855.
# ---------------------------------------------------------------------------


def test_explicit_mask_with_masked_npma_source_ors_keep_mask_default():
    """np.ma source + explicit mask -> numpy ORs (keep_mask=True default).

    Divergence: ferray.ma.array(masked_obj, mask=explicit) uses only the
    explicit mask; numpy ORs the source mask in (ma/core.py:3008).
    Tracking: #855
    """
    base = np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    explicit = [0, 1, 0]

    expected = np.ma.array(base, mask=explicit).mask  # live oracle: [T, T, F]
    actual = fr.ma.array(base, mask=explicit).mask

    np.testing.assert_array_equal(actual, expected)


def test_explicit_mask_with_masked_fr_source_ors_keep_mask_default():
    """fr.ma source + explicit mask -> numpy ORs (keep_mask=True default).

    Same divergence on the ferray.ma source path (py_new's early
    PyMaskedArray.extract branch, mask=Some). numpy's analog ORs the source
    mask with the explicit one (ma/core.py:3008).
    Tracking: #855
    """
    fr_src = fr.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    np_src = np.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    explicit = [0, 1, 0]

    expected = np.ma.array(np_src, mask=explicit).mask  # live oracle: [T, T, F]
    actual = fr.ma.array(fr_src, mask=explicit).mask

    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Objective B — numpy 2.x __array__ protocol (spillover #854, CONFIRMED LIVE).
#
# numpy 2.x calls obj.__array__(dtype=None, copy=None) on any object exposing
# the protocol (NEP 56 / numpy 2.0 added the `copy=` kwarg). ferray's
# PyMaskedArray.__array__(&self, py) takes NO args (ferray-python/src/ma.rs),
# so:
#   - np.asarray(fr_ma, dtype=float)  -> TypeError (1 given)
#   - np.array(fr_ma)                 -> TypeError (no keyword arguments)
#   - np.array(fr_ma, dtype=float)    -> TypeError
# This blocks ALL numpy interop egress of a fr.ma array. numpy.ma's own
# __array__ accepts these kwargs (m.__array__(dtype=..., copy=...) works).
# Tracking: #854.
# ---------------------------------------------------------------------------


def test_array_protocol_accepts_dtype_copy_kwargs():
    """np.array / np.asarray must drive a fr.ma array through __array__.

    Divergence: PyMaskedArray.__array__ takes no args; numpy 2.x passes
    dtype=/copy= (NEP 56). Every np.array(fr_ma...) raises TypeError, breaking
    numpy interop egress. numpy.ma.__array__ accepts both kwargs.
    Tracking: #854
    """
    fr_ma = fr.ma.array([1.0, 2.0, 3.0])

    # Oracle: numpy.ma round-trips cleanly through the same protocol calls.
    expected = np.asarray(np.ma.array([1.0, 2.0, 3.0]), dtype=float)

    # Each of these is how a numpy consumer would ingest a fr.ma array; under
    # the current binding they raise TypeError instead of returning an ndarray.
    np.testing.assert_array_equal(np.array(fr_ma), expected)
    np.testing.assert_array_equal(np.array(fr_ma, dtype=float), expected)
    np.testing.assert_array_equal(np.asarray(fr_ma, dtype=float), expected)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
