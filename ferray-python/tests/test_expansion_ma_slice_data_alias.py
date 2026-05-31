"""Slice/basic-index VIEW `.data` is an ALIASING buffer view for fr.ma vs
numpy.ma (#904, refs #850).

numpy's basic indexing of a masked array reuses `ndarray.__getitem__`
(`numpy/ma/core.py:3288` `dout = self.data[indx]`), so a slice/int/ellipsis/
newaxis sub-array's `.data` is an ALIASING VIEW of the base buffer:
`v.data[i] = x` writes the shared store and thus the base element. A FANCY/bool
view copies (numpy copies fancy-indexed `.data`), so its `.data` write does NOT
reach the base. This suite mirrors every assertion on numpy.ma AND ferray.ma:
each EXPECTED value is derived live from numpy (R-CHAR-3 — never literal-copied
from the ferray side), or compared against `numpy.shares_memory`.

Preserves: identity-view `.data` alias (#896), `__setitem__` write-through
(#857). Covers 1-D slice/step/negative, int-index row, 2-D row/column/sub-block,
chained `b[1:4][1:2]`, complex dtype, and the FANCY-still-copies contract.
"""

import numpy as np
import ferray as fr


# ---------------------------------------------------------------------------
# 1-D slice view: `.data` aliases the base buffer (write-through).
# ---------------------------------------------------------------------------

def test_slice_view_data_write_through_float():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nv = nb[1:3]
    nv.data[0] = 99.0
    np_after = nb[1]  # numpy contract: slice .data aliases the base buffer

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fv = fb[1:3]
    fv.data[0] = 99.0
    assert fb[1] == np_after


def test_slice_view_data_write_through_int():
    nb = np.ma.array([1, 2, 3, 4])
    nv = nb[2:4]
    nv.data[1] = 77
    np_after = nb[3]

    fb = fr.ma.array([1, 2, 3, 4])
    fv = fb[2:4]
    fv.data[1] = 77
    assert fb[3] == np_after


def test_slice_view_step_data_write_through():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nv = nb[::2]
    nv.data[1] = 999.0
    np_after = nb[2]  # step-slice view aliases base[2]

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fv = fb[::2]
    fv.data[1] = 999.0
    assert fb[2] == np_after


def test_slice_view_negative_data_write_through():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nv = nb[-2:]
    nv.data[0] = -1.0
    np_after = nb[2]

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fv = fb[-2:]
    fv.data[0] = -1.0
    assert fb[2] == np_after


def test_slice_view_data_read_reflects_base_write():
    # READ direction: a base `.data[i] = x` is visible through the slice view.
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nv = nb[1:3]
    nb.data[1] = 42.0
    np_seen = nv.data[0]

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fv = fb[1:3]
    fb.data[1] = 42.0
    assert fv.data[0] == np_seen


def test_slice_view_shares_memory_with_base():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nv = nb[1:3]
    np_shares = np.shares_memory(nv.data, nb.data)

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fv = fb[1:3]
    assert np.shares_memory(fv.data, fb.data) == np_shares


# ---------------------------------------------------------------------------
# Integer-index view (a row of a 2-D array): basic, aliases.
# ---------------------------------------------------------------------------

def test_int_index_row_view_data_write_through():
    nb = np.ma.array([[10.0, 20.0], [30.0, 40.0]])
    nv = nb[1]
    nv.data[0] = -7.0
    np_after = nb[1, 0]

    fb = fr.ma.array([[10.0, 20.0], [30.0, 40.0]])
    fv = fb[1]
    fv.data[0] = -7.0
    assert fb[1, 0] == np_after


# ---------------------------------------------------------------------------
# 2-D row / column / sub-block slice views.
# ---------------------------------------------------------------------------

def test_2d_row_slice_view_data_write_through():
    nb = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nv = nb[0]
    nv.data[2] = 88.0
    np_after = nb[0, 2]

    fb = fr.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fv = fb[0]
    fv.data[2] = 88.0
    assert fb[0, 2] == np_after


def test_2d_column_slice_view_data_write_through():
    nb = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nv = nb[:, 1]
    nv.data[0] = 77.0
    np_after = nb[0, 1]

    fb = fr.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fv = fb[:, 1]
    fv.data[0] = 77.0
    assert fb[0, 1] == np_after


def test_2d_subblock_slice_view_data_write_through():
    nb = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    nv = nb[0:2, 1:3]
    nv.data[0, 0] = -5.0
    np_after = nb[0, 1]

    fb = fr.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    fv = fb[0:2, 1:3]
    fv.data[0, 0] = -5.0
    assert fb[0, 1] == np_after


# ---------------------------------------------------------------------------
# Chained basic-index views: `b[1:4][1:2]` composes through to the root buffer.
# ---------------------------------------------------------------------------

def test_chained_slice_view_data_write_through():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nv = nb[1:4][1:2]
    nv.data[0] = 55.0
    np_after = nb[2]  # b[1:4] -> [2,3,4]; [1:2] -> [3] aliases base[2]

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fv = fb[1:4][1:2]
    fv.data[0] = 55.0
    assert fb[2] == np_after


def test_chained_int_then_slice_view_data_write_through():
    nb = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    nv = nb[1][1:3]
    nv.data[0] = -3.0
    np_after = nb[1, 1]

    fb = fr.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fv = fb[1][1:3]
    fv.data[0] = -3.0
    assert fb[1, 1] == np_after


# ---------------------------------------------------------------------------
# Complex dtype slice view.
# ---------------------------------------------------------------------------

def test_complex_slice_view_data_write_through():
    nb = np.ma.array([1 + 2j, 3 + 4j, 5 + 6j])
    nv = nb[1:3]
    nv.data[0] = 9 + 9j
    np_after = nb[1]

    fb = fr.ma.array([1 + 2j, 3 + 4j, 5 + 6j])
    fv = fb[1:3]
    fv.data[0] = 9 + 9j
    assert fb[1] == np_after


# ---------------------------------------------------------------------------
# FANCY / bool view: `.data` COPIES (numpy parity) — write does NOT reach base.
# ---------------------------------------------------------------------------

def test_fancy_index_view_data_does_not_write_base():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nf = nb[[0, 2]]
    nf.data[0] = -9.0
    np_base0 = nb[0]  # numpy: fancy index copies; base[0] unchanged (1.0)

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    ff = fb[[0, 2]]
    ff.data[0] = -9.0
    assert fb[0] == np_base0


def test_bool_index_view_data_does_not_write_base():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    nf = nb[mask]
    nf.data[0] = -9.0
    np_base0 = nb[0]  # bool index copies; base[0] unchanged

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fmask = np.array([True, False, True, False])
    ff = fb[fmask]
    ff.data[0] = -9.0
    assert fb[0] == np_base0


# ---------------------------------------------------------------------------
# Slice-view `.data` coexists with __setitem__ write-through (#857).
# ---------------------------------------------------------------------------

def test_slice_view_data_and_setitem_consistent():
    nb = np.ma.array([1.0, 2.0, 3.0, 4.0])
    nv = nb[1:3]
    nv.data[0] = 11.0
    nv[1] = 22.0
    np_data = np.asarray(nb.data)

    fb = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fv = fb[1:3]
    fv.data[0] = 11.0
    fv[1] = 22.0
    np.testing.assert_array_equal(np.asarray(fb.data), np_data)


# ---------------------------------------------------------------------------
# Identity view `.data` alias preserved (#896 regression guard).
# ---------------------------------------------------------------------------

def test_identity_view_data_alias_preserved():
    nb = np.ma.array([1.0, 2.0, 3.0])
    nv = nb[:]
    nv.data[0] = 11.0
    np_after = nb[0]

    fb = fr.ma.array([1.0, 2.0, 3.0])
    fv = fb[:]
    fv.data[0] = 11.0
    assert fb[0] == np_after
