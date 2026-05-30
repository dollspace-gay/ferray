"""Subscripting parity for ferray.ma.MaskedArray (#852, REQ-1).

Every expected value is derived LIVE from numpy.ma (the installed 2.4.x
oracle), never literal-copied from the ferray side (R-CHAR-3). Mirrors
numpy/ma/core.py __getitem__ (core.py:3277) / __setitem__ (core.py:3406).
"""

import numpy as np
import numpy.ma as nma
import pytest

import ferray as fr


def _mask_list(a):
    """Mask of a ferray MaskedArray as a python list (nomask -> all-False)."""
    m = a.mask
    if m is fr.ma.nomask:
        return [False] * int(np.asarray(a.data).size)
    return np.asarray(m).ravel().tolist()


# ---------------------------------------------------------------------------
# __getitem__ — scalar
# ---------------------------------------------------------------------------

def test_getitem_int_masked_returns_masked_singleton():
    npa = nma.array([1, 2, 3], mask=[1, 0, 0])
    fra = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    assert npa[0] is nma.masked  # oracle
    assert fra[0] is fr.ma.masked


def test_getitem_int_unmasked_returns_scalar():
    npa = nma.array([1, 2, 3], mask=[1, 0, 0])
    fra = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    assert fra[1] == float(npa[1])
    assert fra[2] == float(npa[2])
    assert not isinstance(fra[1], fr.ma.MaskedArray)


def test_getitem_nomask_int_is_scalar():
    npa = nma.array([10.0, 20.0, 30.0])
    fra = fr.ma.array([10.0, 20.0, 30.0])
    assert fra[0] == float(npa[0])
    assert fra[0] is not fr.ma.masked


def test_getitem_negative_index():
    npa = nma.array([1.0, 2.0, 3.0], mask=[0, 0, 1])
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[0, 0, 1])
    assert npa[-1] is nma.masked
    assert fra[-1] is fr.ma.masked
    assert fra[-2] == float(npa[-2])


@pytest.mark.parametrize("idx", [10, -10])
def test_getitem_out_of_bounds_raises_indexerror(idx):
    npa = nma.array([1.0, 2.0, 3.0])
    fra = fr.ma.array([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        npa[idx]
    with pytest.raises(IndexError):
        fra[idx]


# ---------------------------------------------------------------------------
# __getitem__ — slice / fancy / bool
# ---------------------------------------------------------------------------

def test_getitem_slice_carries_data_and_mask():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 1])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 1])
    nsub = npa[1:3]
    fsub = fra[1:3]
    assert isinstance(fsub, fr.ma.MaskedArray)
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()
    assert _mask_list(fsub) == nma.getmaskarray(nsub).tolist()


def test_getitem_fancy_int_array():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[1, 0, 1, 0])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[1, 0, 1, 0])
    nsub = npa[[0, 2, 3]]
    fsub = fra[[0, 2, 3]]
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()
    assert _mask_list(fsub) == nma.getmaskarray(nsub).tolist()


def test_getitem_bool_array():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 1])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 1])
    cond = np.array([True, False, True, True])
    nsub = npa[cond]
    fsub = fra[cond]
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()
    assert _mask_list(fsub) == nma.getmaskarray(nsub).tolist()


def test_getitem_bool_from_comparison():
    data = [1.0, 2.0, 3.0, 4.0]
    npa = nma.array(data)
    fra = fr.ma.array(data)
    cond = np.asarray(data) > 1
    nsub = npa[cond]
    fsub = fra[cond]
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()


def test_getitem_slice_from_nomask_stays_nomask():
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fsub = fra[1:3]
    assert fsub.mask is fr.ma.nomask


# ---------------------------------------------------------------------------
# __getitem__ — 2-D
# ---------------------------------------------------------------------------

def test_getitem_2d_row():
    d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    m = [[0, 1, 0], [1, 0, 0]]
    npa = nma.array(d, mask=m)
    fra = fr.ma.array(d, mask=m)
    nsub = npa[0]
    fsub = fra[0]
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()
    assert _mask_list(fsub) == nma.getmaskarray(nsub).ravel().tolist()
    assert list(fsub.shape) == list(nsub.shape)


def test_getitem_2d_element():
    d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    m = [[0, 1, 0], [1, 0, 0]]
    npa = nma.array(d, mask=m)
    fra = fr.ma.array(d, mask=m)
    assert npa[0, 1] is nma.masked
    assert fra[0, 1] is fr.ma.masked
    assert fra[0, 0] == float(npa[0, 0])
    assert fra[1, 1] == float(npa[1, 1])


def test_getitem_2d_column():
    d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    m = [[0, 1, 0], [1, 0, 0]]
    npa = nma.array(d, mask=m)
    fra = fr.ma.array(d, mask=m)
    nsub = npa[:, 1]
    fsub = fra[:, 1]
    assert np.asarray(fsub.data).tolist() == np.asarray(nsub.data).tolist()
    assert _mask_list(fsub) == nma.getmaskarray(nsub).ravel().tolist()


# ---------------------------------------------------------------------------
# __setitem__ — soft mask (default)
# ---------------------------------------------------------------------------

def test_setitem_scalar_sets_data():
    npa = nma.array([1.0, 2.0, 3.0])
    fra = fr.ma.array([1.0, 2.0, 3.0])
    npa[0] = 5.0
    fra[0] = 5.0
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()


def test_setitem_soft_mask_unmasks_on_assign():
    # numpy: assigning a value under a soft mask UNMASKS the position.
    npa = nma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    npa[0] = 9.0
    fra[0] = 9.0
    assert npa[0] is not nma.masked  # oracle: now unmasked
    assert fra[0] == 9.0
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_assign_masked_singleton_masks_position():
    npa = nma.array([1.0, 2.0, 3.0], mask=[0, 0, 0])
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[0, 0, 0])
    npa[1] = nma.masked
    fra[1] = fr.ma.masked
    assert npa[1] is nma.masked
    assert fra[1] is fr.ma.masked
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_masked_singleton_materializes_mask_on_nomask():
    # Assigning masked to a nomask array materializes a real mask (#848/#849).
    npa = nma.array([1.0, 2.0, 3.0])
    fra = fr.ma.array([1.0, 2.0, 3.0])
    assert npa.mask is nma.nomask
    assert fra.mask is fr.ma.nomask
    npa[2] = nma.masked
    fra[2] = fr.ma.masked
    assert fra.mask is not fr.ma.nomask
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_slice_assign_list():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 1, 0])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 1, 0])
    npa[1:3] = [7.0, 8.0]
    fra[1:3] = [7.0, 8.0]
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_bool_assign_scalar():
    data = [1.0, 2.0, 3.0, 4.0]
    npa = nma.array(data)
    fra = fr.ma.array(data)
    npa[np.asarray(data) > 1] = 0.0
    fra[np.asarray(data) > 1] = 0.0
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()


def test_setitem_fancy_assign_masked():
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    npa[[0, 2]] = nma.masked
    fra[[0, 2]] = fr.ma.masked
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_masked_array_rhs():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 0, 0])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 0, 0])
    npa[0:2] = nma.array([5.0, 6.0], mask=[1, 0])
    fra[0:2] = fr.ma.array([5.0, 6.0], mask=[1, 0])
    # masked RHS position re-masks the target; unmasked RHS writes data.
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()
    # data: numpy writes dval everywhere (even at masked positions) for soft mask
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()


def test_setitem_2d_row_assign():
    d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    npa = nma.array(d, mask=[[0, 1, 0], [0, 0, 0]])
    fra = fr.ma.array(d, mask=[[0, 1, 0], [0, 0, 0]])
    npa[0] = [7.0, 8.0, 9.0]
    fra[0] = [7.0, 8.0, 9.0]
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()
    assert _mask_list(fra) == nma.getmaskarray(npa).ravel().tolist()


# ---------------------------------------------------------------------------
# __setitem__ — hard mask
# ---------------------------------------------------------------------------

def test_setitem_hard_mask_suppresses_assignment():
    # Under a hard mask, assigning a value to a masked position is SUPPRESSED:
    # data unchanged, stays masked.
    npa = nma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    npa.harden_mask()
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    fra.harden_mask()
    npa[0] = 9.0
    fra[0] = 9.0
    assert npa[0] is nma.masked  # oracle: still masked
    assert fra[0] is fr.ma.masked
    # data at masked position unchanged
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()
    assert _mask_list(fra) == nma.getmaskarray(npa).tolist()


def test_setitem_hard_mask_unmasked_position_writes():
    npa = nma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    npa.harden_mask()
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    fra.harden_mask()
    npa[1] = 99.0
    fra[1] = 99.0
    assert fra[1] == float(npa[1])
    assert np.asarray(fra.data).tolist() == np.asarray(npa.data).tolist()


def test_setitem_soft_after_harden_then_soften():
    npa = nma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    npa.harden_mask()
    npa.soften_mask()
    fra = fr.ma.array([1.0, 2.0, 3.0], mask=[1, 0, 0])
    fra.harden_mask()
    fra.soften_mask()
    npa[0] = 9.0
    fra[0] = 9.0
    assert npa[0] is not nma.masked
    assert fra[0] == 9.0
