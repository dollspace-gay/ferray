"""ACToR critic re-audit of e59a8b9e (#904): slice/basic-index VIEW `.data`
aliasing in ferray-python/src/ma.rs.

The audited change (ViewDesc.basic_index + aliasing_data_pyarray) was found
CLEAN across ~40 probes (writeback breadth, chained composition, reverse/column
non-contiguous landing at the correct base index, fancy/bool copy semantics,
shares_memory parity, read-through, borrow-panic, overlays, dtype, identity-view
and __setitem__ non-regression).

The two divergences pinned below were surfaced by the audit's probe sweep but
live in code paths the commit did NOT touch (verified via
`git diff e59a8b9e^ e59a8b9e -- ferray-python/src/ma.rs`): the view-`__setitem__`
masked-assignment path and the integer-scalar egress path. They are pre-existing
divergences on `main` (goal R-DEFER-5: "every divergence on main is something WE
broke"), so they are pinned here as real work.

Every expected value is derived LIVE from numpy.ma (R-CHAR-3), never
literal-copied from the ferray side.

Upstream contract:
  - numpy/ma/core.py:3349 `MaskedArray.__setitem__` — assigning
    `numpy.ma.masked` to a VIEW element sets that element's mask in the view's
    own mask; numpy materialises a per-view mask (the view's `_mask`) so the
    view reports the masked element even though the base array stays unmasked.
  - numpy/ma/core.py:3288 `dout = self.data[indx]` — a fully-resolved scalar
    index returns the underlying numpy scalar (e.g. `numpy.int64`), preserving
    the array dtype; ferray collapses it to a Python `int`.
"""

import numpy as np
import pytest

import ferray as fr


def test_view_setitem_masked_sets_view_mask():
    """`v = b[1:4]; v[0] = ma.masked` must mark the view's element 0 masked.

    Divergence: ferray's view `__setitem__` ignores the masked assignment — the
    view's mask stays all-False — while numpy materialises a per-view mask.
    numpy/ma/core.py:3349. Upstream returns [True, False, False]; ferray
    returns [False, False, False].
    """
    nb = np.ma.array([10, 20, 30, 40, 50])
    nv = nb[1:4]
    nv[0] = np.ma.masked
    expected = np.asarray(np.ma.getmaskarray(nv)).tolist()  # live numpy: [True, False, False]

    fb = fr.ma.array([10, 20, 30, 40, 50])
    fv = fb[1:4]
    fv[0] = fr.ma.masked
    actual = np.asarray(np.ma.getmaskarray(fv)).tolist()

    assert actual == expected


def test_scalar_index_egress_preserves_numpy_dtype():
    """`b[0, 1]` on an int array must egress as a numpy scalar (int64), not a
    Python `int`.

    Divergence: ferray returns the builtin Python `int`, losing the numpy dtype
    contract. numpy/ma/core.py:3288 (`dout = self.data[indx]` -> numpy scalar).
    Upstream type is `numpy.int64`; ferray type is `int`.
    """
    nb = np.ma.array([[1, 2, 3], [4, 5, 6]])
    expected_type = type(nb[0, 1]).__name__  # live numpy: 'int64'

    fb = fr.ma.array([[1, 2, 3], [4, 5, 6]])
    actual_type = type(fb[0, 1]).__name__

    assert actual_type == expected_type
