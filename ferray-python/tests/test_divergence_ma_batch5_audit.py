"""ACToR critic pins — numpy.ma stateful batch (commit a9a51b11).

Each test asserts the LIVE numpy.ma oracle result and FAILS against the
current ferray.ma implementation, pinning a real semantic divergence in
`ferray-ma/src/put.rs` (MaskedArray::put / putmask). Expected values are
computed from `numpy.ma` directly at runtime (R-CHAR-3 — never literal-copied
from the ferray side).

Run:
    cd ferray-python && PYTHONPATH=python python3 -m pytest \
        tests/test_divergence_ma_batch5_audit.py -q
"""

import numpy as np
import numpy.ma as ma
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# DIVERGENCE 1 — hard-mask put with values shorter than indices.
#
# numpy MaskedArray.put (numpy/ma/core.py:4899-4905):
#     values = narray(values, ...); values.resize(indices.shape)
#     indices = indices[~mask]; values = values[~mask]
# `ndarray.resize` ZERO-PADS when growing (it does NOT cycle), and the
# resize+filter happens on the FULL index set. ferray's put.rs cycles values
# with `values[n % values.len()]` in every branch (put.rs:161), so the
# hard-mask path writes a cycled value where numpy writes the zero-pad.
#
# Oracle: a=[1,2,3,4,5] mask=[F,T,F,F,F] hardened; a.put([0,1,2],[10,20]):
#   resize([10,20]->(3,)) = [10,20,0]; ~mask over idx[0,1,2]=[F,T,F]
#   -> keep idx[0,2], values[10,0] -> data[2] = 0, NOT 10.
# ---------------------------------------------------------------------------
def test_divergence_hard_mask_put_short_values_zeropad():
    na = ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 0, 0])
    na.harden_mask()
    na.put([0, 1, 2], [10.0, 20.0])
    expected_data = na.data.tolist()
    expected_mask = ma.getmaskarray(na).tolist()

    fa = fr.ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 0, 0])
    fr.ma.harden_mask(fa)
    fr.ma.put(fa, [0, 1, 2], [10.0, 20.0])

    assert np.asarray(fa.data).tolist() == expected_data
    assert np.asarray(fa.mask).tolist() == expected_mask


# ---------------------------------------------------------------------------
# DIVERGENCE 2 — putmask with a broadcastable (length-1) boolean mask.
#
# numpy.ma.putmask ends with `np.copyto(a._data, valdata, where=mask)`
# (numpy/ma/core.py:7590). copyto BROADCASTS `where` against `a`, so a
# length-1 boolean mask broadcasts across every position. ferray's
# putmask (put.rs:212) rejects any `mask.len() != self.size()` with a
# ShapeMismatch, so the broadcastable mask raises instead of writing.
#
# Oracle: a=[1,2,3,4]; putmask(a, [True], [9]) -> [9,9,9,9].
# ---------------------------------------------------------------------------
def test_divergence_putmask_length1_mask_broadcasts():
    na = ma.array([1.0, 2, 3, 4])
    ma.putmask(na, [True], [9.0])
    expected = na.data.tolist()

    fa = fr.ma.array([1.0, 2, 3, 4])
    fr.ma.putmask(fa, [True], [9.0])

    assert np.asarray(fa.data).tolist() == expected


# ---------------------------------------------------------------------------
# DIVERGENCE 3 — put with empty values and non-empty indices.
#
# numpy MaskedArray.put delegates to `self._data.put(indices, values, ...)`
# (numpy/ma/core.py:4907). ndarray.put with an empty `values` is a SILENT
# no-op (the C loop has nothing to iterate). ferray's put.rs (put.rs:127)
# instead raises InvalidValue -> ValueError when `values.is_empty()`.
#
# Oracle: a=[1,2,3]; a.put([0], []) -> data unchanged [1,2,3], no error.
# ---------------------------------------------------------------------------
def test_divergence_put_empty_values_is_noop():
    na = ma.array([1.0, 2, 3])
    na.put([0], [])  # numpy: silent no-op
    expected = na.data.tolist()

    fa = fr.ma.array([1.0, 2, 3])
    fr.ma.put(fa, [0], [])  # ferray currently raises ValueError

    assert np.asarray(fa.data).tolist() == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
