"""numpy.ma library expansion slice 3 (refs #835): mask-structure run-length
analysis (clump / notmasked / flatnotmasked edges + contiguous), the 2-D masked
matrix product ``ma.dot``, and multi-axis ``ma.argsort``.

Every expected value is produced by calling ``numpy.ma`` directly on the same
inputs (R-CHAR-3) — never literal-copied from the ferray side. Slice lists are
compared against numpy's ``slice`` objects directly.
"""

import numpy as np
import pytest

import ferray.ma as fma


def _fr(npma):
    """A numpy masked array -> equivalent ferray.ma.MaskedArray."""
    return fma.masked_array(np.asarray(npma.data, dtype=float), np.ma.getmaskarray(npma))


def _to_np(m):
    """A ferray.ma.MaskedArray -> (data ndarray, mask ndarray), broadcasting a
    scalar `nomask` to an all-False mask of the data shape."""
    data = np.asarray(m.data, dtype=float)
    mask = np.asarray(m.mask, dtype=bool)
    if mask.shape != data.shape:
        mask = np.zeros(data.shape, dtype=bool)
    return data, mask


def _assert_slices_equal(got, expected):
    """Compare a list of Python slices to numpy's list of slices."""
    assert len(got) == len(expected), f"len {len(got)} != {len(expected)}"
    for g, e in zip(got, expected):
        assert isinstance(g, slice), f"expected slice, got {type(g)}"
        assert int(g.start) == int(e.start), f"start {g.start} != {e.start}"
        assert int(g.stop) == int(e.stop), f"stop {g.stop} != {e.stop}"
        assert g.step is None and e.step is None


# ---------------------------------------------------------------------------
# clump_masked / clump_unmasked
# ---------------------------------------------------------------------------

# numpy: ma.masked_array(arange(10)); a[[0,1,2,6,8,9]] = masked
_CLUMP = np.ma.masked_array(np.arange(10, dtype=float))
_CLUMP[[0, 1, 2, 6, 8, 9]] = np.ma.masked


def test_clump_masked_matches_numpy():
    got = fma.clump_masked(_fr(_CLUMP))
    _assert_slices_equal(got, np.ma.clump_masked(_CLUMP))


def test_clump_unmasked_matches_numpy():
    got = fma.clump_unmasked(_fr(_CLUMP))
    _assert_slices_equal(got, np.ma.clump_unmasked(_CLUMP))


def test_clump_no_mask():
    m = np.ma.masked_array(np.arange(5, dtype=float))
    _assert_slices_equal(fma.clump_unmasked(_fr(m)), np.ma.clump_unmasked(m))
    _assert_slices_equal(fma.clump_masked(_fr(m)), np.ma.clump_masked(m))


def test_clump_all_masked():
    m = np.ma.masked_array(np.arange(5, dtype=float), mask=[1, 1, 1, 1, 1])
    _assert_slices_equal(fma.clump_unmasked(_fr(m)), np.ma.clump_unmasked(m))
    _assert_slices_equal(fma.clump_masked(_fr(m)), np.ma.clump_masked(m))


# ---------------------------------------------------------------------------
# flatnotmasked_contiguous / flatnotmasked_edges / notmasked_*
# ---------------------------------------------------------------------------

_M = np.ma.array(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], mask=[0, 0, 1, 1, 0, 0]
)


def test_flatnotmasked_contiguous_matches_numpy():
    got = fma.flatnotmasked_contiguous(_fr(_M))
    _assert_slices_equal(got, np.ma.flatnotmasked_contiguous(_M))


def test_flatnotmasked_edges_matches_numpy():
    got = fma.flatnotmasked_edges(_fr(_M))
    exp = np.ma.flatnotmasked_edges(_M)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))


def test_flatnotmasked_edges_all_masked_is_none():
    m = np.ma.array([1.0, 2.0, 3.0], mask=[1, 1, 1])
    assert fma.flatnotmasked_edges(_fr(m)) is None
    assert np.ma.flatnotmasked_edges(m) is None


def test_notmasked_edges_flat_matches_numpy():
    got = fma.notmasked_edges(_fr(_M))
    exp = np.ma.notmasked_edges(_M)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))


def test_notmasked_contiguous_flat_matches_numpy():
    got = fma.notmasked_contiguous(_fr(_M))
    _assert_slices_equal(got, np.ma.notmasked_contiguous(_M))


def test_notmasked_contiguous_axis_2d_matches_numpy():
    # numpy: a = arange(12).reshape(3,4) with a hand-built mask.
    a = np.arange(12).reshape((3, 4)).astype(float)
    mask = np.zeros_like(a, dtype=bool)
    mask[1:, :-1] = True
    mask[0, 1] = True
    mask[-1, 0] = False
    nm = np.ma.array(a, mask=mask)
    frm = fma.masked_array(a, mask)

    for axis in (0, 1):
        got = fma.notmasked_contiguous(frm, axis)
        exp = np.ma.notmasked_contiguous(nm, axis=axis)
        assert len(got) == len(exp)
        for glane, elane in zip(got, exp):
            _assert_slices_equal(glane, elane)


# ---------------------------------------------------------------------------
# 2-D ma.dot
# ---------------------------------------------------------------------------


def test_dot_2d_matches_numpy():
    a = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    exp = np.ma.dot(a, a)
    gd, gm = _to_np(fma.dot(_fr(a), _fr(a)))
    em = np.ma.getmaskarray(exp)
    np.testing.assert_array_equal(gm, em)
    np.testing.assert_allclose(gd[~gm], np.asarray(exp.data)[~em], atol=1e-12)


def test_dot_2d_rectangular_matches_numpy():
    a = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 0, 0]]).astype(float)
    b = np.ma.array([[1, 2], [3, 4], [5, 6]], mask=[[1, 0], [0, 0], [0, 0]]).astype(float)
    exp = np.ma.dot(a, b)
    gd, gm = _to_np(fma.dot(_fr(a), _fr(b)))
    em = np.ma.getmaskarray(exp)
    np.testing.assert_array_equal(gm, em)
    np.testing.assert_allclose(gd[~gm], np.asarray(exp.data)[~em], atol=1e-12)


def test_dot_1d_still_scalar():
    a = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 0, 1, 0])
    b = np.ma.array([5.0, 6.0, 7.0, 8.0], mask=[0, 0, 0, 0])
    got = fma.dot(_fr(a), _fr(b))
    assert isinstance(got, float)
    assert got == pytest.approx(float(np.ma.dot(a, b)))


# ---------------------------------------------------------------------------
# multi-axis ma.argsort
# ---------------------------------------------------------------------------

_B = np.ma.array([[3, 1, 2], [6, 5, 4]], mask=[[0, 0, 1], [0, 0, 0]]).astype(float)


def test_argsort_axis1_matches_numpy():
    got = np.asarray(fma.argsort(_fr(_B), 1))
    exp = np.ma.argsort(_B, axis=1)
    np.testing.assert_array_equal(got, exp)


def test_argsort_axis0_matches_numpy():
    got = np.asarray(fma.argsort(_fr(_B), 0))
    exp = np.ma.argsort(_B, axis=0)
    np.testing.assert_array_equal(got, exp)


def test_argsort_axis_none_flat_matches_numpy():
    got = np.asarray(fma.argsort(_fr(_B)))
    exp = np.ma.argsort(_B, axis=None)
    np.testing.assert_array_equal(got, exp)


def test_argsort_axis_negative_matches_numpy():
    got = np.asarray(fma.argsort(_fr(_B), -1))
    exp = np.ma.argsort(_B, axis=-1)
    np.testing.assert_array_equal(got, exp)
