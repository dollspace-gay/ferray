"""numpy.ma stateful-mask batch (#842 #843 #844 #845): harden_mask /
soften_mask / put / putmask.

Every expected value is derived from a live ``numpy.ma`` call (R-CHAR-3): the
oracle constructs the same operation in numpy.ma and the test asserts ferray.ma
matches data + mask. The central concern is that the ferray ``&mut self`` /
``Py<PyMaskedArray>`` mutation actually sticks on the caller's object, exactly
as numpy mutates the array in place.
"""

import numpy as np
import numpy.ma as nma
import pytest

import ferray as fr


def _fr_data_mask(a):
    """(data list, mask list) for a ferray.ma.MaskedArray."""
    data = np.asarray(fr.ma.getdata(a)).tolist()
    mask = np.asarray(fr.ma.getmaskarray(a)).tolist()
    return data, mask


def _np_data_mask(a):
    data = nma.getdata(a).tolist()
    mask = nma.getmaskarray(a).tolist()
    return data, mask


# ---------------------------------------------------------------------------
# harden_mask / soften_mask — flag flip + in-place mutation that STICKS
# ---------------------------------------------------------------------------


def test_harden_mask_returns_same_object():
    a = fr.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    b = fr.ma.harden_mask(a)
    # numpy returns the same (mutated) array; ferray returns the same object.
    assert b is a
    assert a.hardmask is True


def test_soften_mask_returns_same_object():
    a = fr.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    fr.ma.harden_mask(a)
    b = fr.ma.soften_mask(a)
    assert b is a
    assert a.hardmask is False


def test_harden_mask_method_mutates_in_place():
    # The #1 risk: PyMaskedArray is #[derive(Clone)]; ensure the &mut self
    # method mutates the wrapped MaskedArray in the pyclass cell, not a clone.
    a = fr.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    a.harden_mask()
    assert a.hardmask is True
    # The flag persists across a subsequent attribute read.
    assert a.hardmask is True


def test_harden_then_put_suppresses_masked_assignment_like_numpy():
    # numpy: after harden, putting to a masked position is suppressed.
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    npa.harden_mask()
    npa.put([1, 3], [10.0, 30.0])
    exp_data, exp_mask = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    fra.harden_mask()
    fra.put([1, 3], [10.0, 30.0])
    got_data, got_mask = _fr_data_mask(fra)

    assert got_data == exp_data
    assert got_mask == exp_mask
    # The masked position 1 was neither overwritten nor unmasked.
    assert got_mask[1] is True
    assert got_data[1] == 2.0


def test_soften_then_put_clears_mask_like_numpy():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    npa.harden_mask()
    npa.soften_mask()
    npa.put([1, 3], [10.0, 30.0])
    exp_data, exp_mask = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    fra.harden_mask()
    fra.soften_mask()
    fra.put([1, 3], [10.0, 30.0])
    got_data, got_mask = _fr_data_mask(fra)

    assert got_data == exp_data
    assert got_mask == exp_mask


# ---------------------------------------------------------------------------
# put — data + mask, modes, masked values, hard-mask suppression
# ---------------------------------------------------------------------------


def test_put_basic_unmasks_written():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    npa.put([1, 3], [10.0, 30.0])
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, True, False, False])
    fr.ma.put(fra, [1, 3], [10.0, 30.0])
    assert _fr_data_mask(fra) == exp


def test_put_method_form():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[False, False, False, False])
    npa.put([0, 2], [10.0, 30.0])
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fra.put([0, 2], [10.0, 30.0])
    assert _fr_data_mask(fra) == exp


def test_put_values_repeat():
    npa = nma.array([1.0, 2.0, 3.0, 4.0, 5.0])
    npa.put([0, 1, 2, 3], [9.0])
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fr.ma.put(fra, [0, 1, 2, 3], [9.0])
    assert _fr_data_mask(fra) == exp


def test_put_masked_values_propagate():
    vals = nma.array([10.0, 20.0], mask=[True, False])
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    npa.put([0, 1], vals)
    exp = _np_data_mask(npa)

    fr_vals = fr.ma.array([10.0, 20.0], mask=[True, False])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fr.ma.put(fra, [0, 1], fr_vals)
    assert _fr_data_mask(fra) == exp


def test_put_masked_values_from_numpy_ma():
    # values may be a numpy.ma.MaskedArray; its mask must survive the boundary.
    vals = nma.array([10.0, 20.0], mask=[True, False])
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    npa.put([0, 1], vals)
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fr.ma.put(fra, [0, 1], vals)
    assert _fr_data_mask(fra) == exp


@pytest.mark.parametrize("mode", ["wrap", "clip"])
def test_put_mode_wrap_clip(mode):
    npa = nma.array([1.0, 2.0, 3.0])
    npa.put([5], [99.0], mode=mode)
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0])
    fr.ma.put(fra, [5], [99.0], mode=mode)
    assert _fr_data_mask(fra) == exp


def test_put_mode_raise_out_of_bounds_indexerror():
    fra = fr.ma.array([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        fr.ma.put(fra, [5], [99.0], mode="raise")
    # numpy raises IndexError too.
    npa = nma.array([1.0, 2.0, 3.0])
    with pytest.raises(IndexError):
        npa.put([5], [99.0], mode="raise")


def test_put_negative_index():
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    npa.put([-1, -2], [10.0, 20.0])
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fr.ma.put(fra, [-1, -2], [10.0, 20.0])
    assert _fr_data_mask(fra) == exp


def test_put_bad_mode_raises():
    fra = fr.ma.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        fr.ma.put(fra, [0], [1.0], mode="nonsense")


# ---------------------------------------------------------------------------
# putmask — conditional in-place assignment
# ---------------------------------------------------------------------------


def test_putmask_basic_clears_mask():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[True, False, False, False])
    nma.putmask(npa, np.array([True, False, True, False]),
                np.array([10.0, 20.0, 30.0, 40.0]))
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[True, False, False, False])
    fr.ma.putmask(fra, [True, False, True, False], [10.0, 20.0, 30.0, 40.0])
    assert _fr_data_mask(fra) == exp


def test_putmask_scalar_broadcast():
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    nma.putmask(npa, np.array([True, False, True, False]), 99.0)
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fr.ma.putmask(fra, [True, False, True, False], 99.0)
    assert _fr_data_mask(fra) == exp


def test_putmask_hard_keeps_mask_overwrites_data():
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[True, False, False, False])
    npa.harden_mask()
    nma.putmask(npa, np.array([True, True, False, False]),
                np.array([10.0, 20.0, 30.0, 40.0]))
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[True, False, False, False])
    fra.harden_mask()
    fr.ma.putmask(fra, [True, True, False, False], [10.0, 20.0, 30.0, 40.0])
    assert _fr_data_mask(fra) == exp


def test_putmask_soft_masked_values_propagate():
    vals = nma.array([10.0, 20.0, 30.0, 40.0], mask=[True, False, False, False])
    npa = nma.array([1.0, 2.0, 3.0, 4.0], mask=[False, False, False, True])
    nma.putmask(npa, np.array([True, True, False, False]), vals)
    exp = _np_data_mask(npa)

    fr_vals = fr.ma.array([10.0, 20.0, 30.0, 40.0], mask=[True, False, False, False])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0], mask=[False, False, False, True])
    fr.ma.putmask(fra, [True, True, False, False], fr_vals)
    assert _fr_data_mask(fra) == exp


def test_putmask_hard_masked_values_union():
    vals = nma.array([10.0, 20.0, 30.0, 40.0], mask=[True, False, False, False])
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    npa.harden_mask()
    nma.putmask(npa, np.array([True, True, False, False]), vals)
    exp = _np_data_mask(npa)

    fr_vals = fr.ma.array([10.0, 20.0, 30.0, 40.0], mask=[True, False, False, False])
    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fra.harden_mask()
    fr.ma.putmask(fra, [True, True, False, False], fr_vals)
    assert _fr_data_mask(fra) == exp


def test_putmask_method_form():
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    nma.putmask(npa, np.array([False, True, False, True]), 7.0)
    exp = _np_data_mask(npa)

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    fra.putmask([False, True, False, True], 7.0)
    assert _fr_data_mask(fra) == exp


def test_putmask_mismatched_values_raises():
    # numpy.ma.putmask broadcasts (copyto) rather than cycling: a length-2
    # values into a length-4 array raises ValueError.
    npa = nma.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        nma.putmask(npa, np.array([True, True, True, True]), np.array([9.0, 8.0]))

    fra = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        fr.ma.putmask(fra, [True, True, True, True], [9.0, 8.0])
