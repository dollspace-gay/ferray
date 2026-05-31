"""ACToR critic re-audit of acto-builder commit 976732ae (#896/#898):
numpy-backed `.data` buffer for fr.ma.MaskedArray.

This file pins divergences found while adversarially probing the new
buffer-aliasing / refresh / writeback machinery against numpy.ma 2.4.x.

All expected values are derived LIVE from `numpy.ma` in-test (R-CHAR-3) —
never literal-copied from ferray.

Audit coverage that CONVERGED (no pin needed, recorded here for the report):
  - real-dtype `.data` aliasing: owned write-through, identity-view base
    sharing, foreign ndarray sharing, held-`d=a.data` round-trip, 2-D.
  - stale-buffer sync: data-write-then-reduction, setitem-then-data,
    interleaved data/setitem/arithmetic, bulk `data[:]=...`, put-then-data.
  - borrow-overlap: `a[:]=a.data*2`, `ma.dot(a,a)`, held-data during sum.
  - copy semantics: copy=True / dtype-recast / a.copy() / (a+1).data all
    independent; default copy=False aliases foreign.
  - dtype fidelity (int8/uint16/float32/bool/int64/uint64) through buffer.
  - keep_mask-OR (#855), mask= identity-view base sharing, repr/filled
    after data mutation (#884), complex ops/reductions/repr.
"""

import numpy as np
import ferray as fr


# ---------------------------------------------------------------------------
# Divergence: complex `.data` does NOT alias the shared buffer in ferray.
#
# numpy.ma stores complex `_data` exactly like real dtypes, so `.data` is a
# WRITABLE alias of the shared buffer: `cm.data[i] = x` writes the buffer and
# `cm[i]`/`cm.data[i]` observe it, and `np.shares_memory(cm, cm.data)` is True.
#
# The buffered-data builder (976732ae) carries NO DataBuf for complex
# (`DynMa::Complex32/64 => None`, the legacy copy path), so ferray's complex
# `.data` is a COPY: a `cm.data[i] = x` write is dropped. This is the
# known complex-mode gap called out in the commit's MODEL note
# ("Complex dtypes carry NO `DataBuf`").
#
# Pin is left UN-`@pytest.mark.xfail` so the failing assertion IS the
# release-block signal per the goal's no-skip discipline (R-DEFER-3).
# ---------------------------------------------------------------------------


def test_complex_data_is_aliasing_writeback_divergence():
    """numpy.ma aliases complex `.data`; ferray copies it (write dropped)."""
    # Live numpy oracle: a complex .data write lands in the array.
    nm = np.ma.array([1 + 2j, 3 + 4j, 5 + 6j])
    nm.data[0] = 9 + 9j
    np_after = complex(nm[0])
    assert np_after == 9 + 9j  # numpy contract: write-through

    fm = fr.ma.array([1 + 2j, 3 + 4j, 5 + 6j])
    fm.data[0] = 9 + 9j
    fr_after = complex(fm[0])

    # ferray must match numpy's aliasing write-through.
    assert fr_after == np_after


def test_complex_data_shares_memory_divergence():
    """numpy.ma: `np.shares_memory(cm, cm.data)` is True; ferray returns False."""
    nm = np.ma.array([1 + 2j, 3 + 4j])
    np_shares = np.shares_memory(nm, nm.data)
    assert np_shares is True or bool(np_shares) is True  # numpy contract

    fm = fr.ma.array([1 + 2j, 3 + 4j])
    fr_shares = bool(np.shares_memory(np.asarray(fm), fm.data))

    assert fr_shares == bool(np_shares)
