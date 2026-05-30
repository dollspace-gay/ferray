"""ACToR critic FINAL-AUDIT pins: ferray.ma view per-object fill_value vs numpy.ma.

Re-audit of acto-fixer commit 7e660ae1 (#880/#881): per-object view overlays
``own_fill: Option<Py<PyAny>>`` and ``own_mask: Option<Vec<bool>>`` on
``Storage::View``. The own_mask overlay (#881) is CORRECT across the full probe
matrix (nomask-base mask stays view-local, overlapping siblings independent,
own_mask vs later real-base-mask read-merge, soft-unmask, data/base-mutation
interplay, real-mask-base regression guard). The own_fill GETTER's per-object
read is also CORRECT once a base's ``_fill_value`` has materialized (numpy
shares that 0-d ``_fill_value`` array by reference, so a view that snapshotted a
materialized base value tracks later in-place base changes -- ferray's
recurse-to-base getter matches).

The own_fill overlay (#880) DIVERGES in two ways:

  * Divergence A — a view of a DEFAULT base (whose ``_fill_value is None``)
    keeps ``_fill_value is None`` and returns the per-dtype default FOREVER,
    even after the base later sets its fill. numpy's
    ``__array_finalize__`` only shares the base's ``_fill_value`` if it is
    already materialized; a None base leaves the view None, and a later
    ``set_fill_value(base, v)`` allocates a FRESH ``_fill_value`` on the base
    only -- the view never sees it. ferray's getter recurses to
    ``base.fill_value`` (the base's value NOW), so a fresh view erroneously
    tracks the base's later change.

  * Divergence C — ``view.filled()`` ignores the view's own_fill overlay. The
    getter ``view.fill_value`` reports the per-object fill (e.g. 7), but
    ``view.filled()`` substitutes the base/default filler (e.g. 999999) because
    it calls ``filled_default()`` on the gathered snapshot DynMa, never
    consulting ``own_fill``. numpy ``filled()`` uses the object's ``_fill_value``.

Every expected value is computed by running the SAME operation against
``numpy.ma`` (the live oracle, numpy 2.4.x installed), never literal-copied
from the ferray side (R-CHAR-3 / goal.md). Each test asserts the numpy value
and FAILS against the current ferray implementation.

Tracking issues filed from repo root via crosslink (see report).
"""

import numpy as np
import numpy.ma as nma
import pytest

import ferray as fr

fma = fr.ma


# ---------------------------------------------------------------------------
# Divergence A — fresh view of a DEFAULT base; base sets its fill AFTER the
# view exists. numpy: the view never materialized its own _fill_value (stays
# None), so it returns the per-dtype default 1e20 -- it does NOT track the
# base's later change. ferray: getter recurses to base.fill_value (now 99).
# ---------------------------------------------------------------------------
def test_view_fill_does_not_track_base_set_after_creation():
    # numpy oracle
    na = nma.array([1.0, 2.0, 3.0, 4.0])
    nb = na[1:3]
    nma.set_fill_value(na, 99)
    expected = nb.fill_value  # live numpy value (1e20)

    a = fma.array([1.0, 2.0, 3.0, 4.0])
    b = a[1:3]
    fma.set_fill_value(a, 99)
    assert b.fill_value == expected


# ---------------------------------------------------------------------------
# Divergence C — view.filled() must use the view's per-object fill overlay.
# Set own_fill=7 on the view, mask one element, then filled() -> numpy fills
# the masked slot with 7. ferray reports b.fill_value==7 but filled() uses the
# default int filler 999999.
# ---------------------------------------------------------------------------
def test_view_filled_uses_own_fill_overlay():
    na = nma.array([1, 2, 3, 4])
    nb = na[1:3]
    nma.set_fill_value(nb, 7)
    nb[0] = nma.masked
    expected = list(np.asarray(nb.filled()))  # live numpy ([7, 3])

    a = fma.array([1, 2, 3, 4])
    b = a[1:3]
    fma.set_fill_value(b, 7)
    b[0] = fma.masked
    got = list(np.asarray(b.filled()))
    assert got == expected


# ---------------------------------------------------------------------------
# Divergence C variant — own_fill set AFTER masking, then filled().
# ---------------------------------------------------------------------------
def test_view_filled_uses_own_fill_overlay_mask_first():
    na = nma.array([1, 2, 3, 4])
    nb = na[1:3]
    nb[0] = nma.masked
    nma.set_fill_value(nb, 7)
    expected = list(np.asarray(nb.filled()))  # live numpy ([7, 3])

    a = fma.array([1, 2, 3, 4])
    b = a[1:3]
    b[0] = fma.masked
    fma.set_fill_value(b, 7)
    got = list(np.asarray(b.filled()))
    assert got == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
