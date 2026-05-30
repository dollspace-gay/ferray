"""ACToR critic pins: ferray.ma basic-index VIEW semantics vs numpy.ma (#857).

Re-audit of acto-builder commit 8fd3b874 (Option B: enum Storage {Owned, View}
with snapshot/with_buffer_mut/set_flat_through). The basic-index writeback,
read-reflects-base, fancy-still-copies, 2-D/chained, and soft/hard-mask cases
the builder shipped all match numpy in the broad probe matrix. These two
edge cases DIVERGE.

Every expected value is computed by running the SAME operation against
``numpy.ma`` (the live oracle, numpy 2.4.x installed) — never literal-copied
from the ferray side (R-CHAR-3 / goal.md). Each test asserts the numpy value
and FAILS against the current ferray implementation, pinning the divergence.

Tracking issues filed from repo root via crosslink (see commit body / report).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Divergence A — set_fill_value on a basic-index VIEW is routed to the BASE
# ROOT, but numpy keeps `_fill_value` PER-OBJECT.
#
# numpy/ma/core.py: `fill_value.setter` does, for a fresh view whose
# `_fill_value is None`, `self._fill_value = target` — assigning a NEW array to
# the VIEW ONLY (the base, never having materialized its own `_fill_value`,
# stays at the default 1e20). `numpy.ma.set_fill_value(b, v)` calls that setter
# on `b`. So `b.fill_value` changes and `a.fill_value` is UNCHANGED.
#
# ferray's `set_fill_value` (src/ma.rs `set_fill_value_owned`) recurses through
# the View base and sets the fill on the ROOT owner — the exact opposite: the
# base changes and the view's own fill is left at the default. The builder's own
# code comment flags this as "the conservative ... choice" — it is a divergence.
# ---------------------------------------------------------------------------


def _numpy_view_fill():
    """Live numpy oracle: (view_fill, base_fill) after set_fill_value(view)."""
    a = np.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    np.ma.set_fill_value(b, 7.0)
    return float(b.fill_value), float(a.fill_value)


def test_divergence_set_fill_value_on_view_is_per_object():
    """Divergence: ferray.ma.set_fill_value on a slice VIEW writes the BASE
    root; numpy writes the VIEW's own per-object `_fill_value` and leaves the
    base unchanged. Oracle: numpy/ma/core.py `fill_value.setter`
    (`self._fill_value = target` for a fresh `_fill_value is None`).
    """
    np_view_fill, np_base_fill = _numpy_view_fill()
    # numpy: view fill == 7.0, base fill == default 1e20 (independent).
    assert (np_view_fill, np_base_fill) == (7.0, 1e20)

    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    fr.ma.set_fill_value(b, 7.0)
    assert float(b.fill_value) == np_view_fill, (
        "view's own fill_value must change to 7.0 (numpy per-object setter)"
    )
    assert float(a.fill_value) == np_base_fill, (
        "base fill_value must stay at default 1e20 (numpy never touches base)"
    )


def test_divergence_set_fill_value_on_chained_view_is_per_object():
    """Same per-object divergence through a CHAIN of views: set_fill_value on
    the leaf view `c = a[1:4][0:2]` changes only `c`, leaving `b` and `a` at the
    default. Oracle: live numpy.ma.
    """
    an = np.ma.array([1.0, 2, 3, 4, 5])
    bn = an[1:4]
    cn = bn[0:2]
    np.ma.set_fill_value(cn, 7.0)
    exp = (float(cn.fill_value), float(bn.fill_value), float(an.fill_value))
    assert exp == (7.0, 1e20, 1e20)

    a = fr.ma.array([1.0, 2, 3, 4, 5])
    b = a[1:4]
    c = b[0:2]
    fr.ma.set_fill_value(c, 7.0)
    got = (float(c.fill_value), float(b.fill_value), float(a.fill_value))
    assert got == exp


# ---------------------------------------------------------------------------
# Divergence B — masking a position through a VIEW of a NOMASK base must NOT
# mask the base.
#
# numpy: when the base array carries no real mask (mask is the `nomask`
# sentinel scalar), a basic-index view shares that `nomask`. Assigning
# `b[i] = numpy.ma.masked` materializes a FRESH local mask array on the VIEW
# only (there is no shared mask buffer to write through), so the base stays
# completely unmasked. (When the base already carries a REAL mask buffer the
# mask IS shared and the assignment DOES propagate — ferray matches that case.)
#
# ferray's `with_buffer_mut` scatters the after-mask bit through
# `set_flat_through` into the base unconditionally, so masking via a view of a
# nomask base wrongly masks the base position.
#
# Data writes (`b[i] = 99`) still correctly propagate on a nomask base in BOTH
# numpy and ferray; only the MASK-bit write-through diverges for nomask bases.
# ---------------------------------------------------------------------------


def test_divergence_mask_set_through_view_of_nomask_base():
    """Divergence: ferray masks the base when `b[0] = masked` is applied to a
    slice view of a NOMASK base; numpy materializes the mask on the view only
    and leaves the base unmasked. Oracle: live numpy.ma — base mask stays all
    False, view mask becomes [True, False].
    """
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    bn[0] = np.ma.masked
    np_base_masked = bool(np.ma.getmaskarray(an)[1])
    np_view_masked = bool(np.ma.getmaskarray(bn)[0])
    assert (np_base_masked, np_view_masked) == (False, True)

    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    b[0] = fr.ma.masked
    assert bool(fr.ma.getmaskarray(b)[0]) == np_view_masked, (
        "the view's own position must read as masked"
    )
    assert bool(fr.ma.getmaskarray(a)[1]) == np_base_masked, (
        "masking through a view of a NOMASK base must NOT mask the base"
    )


def test_mask_set_through_view_of_real_mask_base_propagates():
    """Control (must already PASS): when the base carries a REAL mask buffer,
    `b[0] = masked` DOES propagate to the base (shared mask buffer). This pins
    the boundary of divergence B so a fix does not over-correct and break the
    real-mask propagation path. Oracle: live numpy.ma.
    """
    an = np.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    bn = an[1:3]
    bn[0] = np.ma.masked
    np_base_masked = bool(np.ma.getmaskarray(an)[1])
    assert np_base_masked is True

    a = fr.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    b = a[1:3]
    b[0] = fr.ma.masked
    assert bool(fr.ma.getmaskarray(a)[1]) == np_base_masked


def test_divergence_mask_set_through_2d_column_view_of_nomask_base():
    """Same nomask-base divergence through a non-contiguous 2-D COLUMN view:
    `a[:, 1][0] = masked` must not mask the base when the base is nomask.
    Oracle: live numpy.ma.
    """
    an = np.ma.array([[1.0, 2], [3, 4]])
    cn = an[:, 1]
    cn[0] = np.ma.masked
    np_base = bool(np.ma.getmaskarray(an)[0, 1])
    assert np_base is False

    a = fr.ma.array([[1.0, 2], [3, 4]])
    c = a[:, 1]
    c[0] = fr.ma.masked
    assert bool(fr.ma.getmaskarray(a)[0, 1]) == np_base


def test_divergence_mask_set_through_chained_view_of_nomask_base():
    """Same nomask-base divergence through a CHAIN of views:
    `c = a[1:4][0:2]; c[0] = masked` must not mask the base when nomask.
    Oracle: live numpy.ma.
    """
    an = np.ma.array([1.0, 2, 3, 4, 5])
    cn = an[1:4][0:2]
    cn[0] = np.ma.masked
    np_base = bool(np.ma.getmaskarray(an)[1])
    assert np_base is False

    a = fr.ma.array([1.0, 2, 3, 4, 5])
    c = a[1:4][0:2]
    c[0] = fr.ma.masked
    assert bool(fr.ma.getmaskarray(a)[1]) == np_base
