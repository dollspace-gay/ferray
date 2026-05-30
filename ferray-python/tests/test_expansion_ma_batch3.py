"""numpy.ma composable batch 3 (refs #835 #818).

Asserts the newly bound `ferray.ma` array-of-masked constructors
(`masked_all`, `masked_all_like`, `fromfunction`, `indices`), masked iteration
(`apply_along_axis`, `apply_over_axes`, `ndenumerate`), masked split
(`hsplit`), whole-row/col masking (`mask_rows`, `mask_cols`), multi-axis
compress (`compress_nd`), the `bool_` re-export, and `ids` against `numpy.ma`
directly (R-CHAR-3 — every expected value is constructed from numpy.ma, never
literal-copied from the ferray side).

Each masked result is compared on BOTH its data and its mask, mirroring
numpy.ma's `(getdata, getmaskarray)` observable contract.
"""

import numpy as np
import pytest

import ferray as fr


def _fr(data, mask=None):
    return fr.ma.array(np.asarray(data, dtype=float), mask=mask)


def _np(data, mask=None):
    return np.ma.array(np.asarray(data, dtype=float), mask=mask)


def _getmask(obj):
    """Mask of either a ferray.ma.MaskedArray or a numpy.ma object as a full
    bool array (numpy.ma.getmaskarray can't read the ferray class)."""
    if isinstance(obj, fr.ma.MaskedArray):
        m = np.asarray(obj.mask, dtype=bool)
        if m.shape == ():
            m = np.zeros(np.asarray(obj.data).shape, dtype=bool)
        return m
    return np.ma.getmaskarray(obj)


def _getdata(obj):
    if isinstance(obj, fr.ma.MaskedArray):
        return np.asarray(obj.data)
    return np.ma.getdata(obj)


def assert_ma_equal(got, expected):
    """Compare a masked result to a numpy.ma result on data + mask."""
    gd = _getdata(got)
    gm = _getmask(got)
    ed = _getdata(expected)
    em = _getmask(expected)
    if gm.shape != ed.shape:
        gm = np.broadcast_to(gm, ed.shape)
    assert gd.shape == ed.shape
    np.testing.assert_array_equal(gm, em)
    keep = ~em
    np.testing.assert_allclose(
        np.asarray(gd, dtype=float)[keep],
        np.asarray(ed, dtype=float)[keep],
        rtol=1e-12,
        atol=0,
        equal_nan=True,
    )


# ---------------------------------------------------------------------------
# masked_all / masked_all_like
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2, 2), (3,), (2, 3, 1)])
def test_masked_all(shape):
    assert_ma_equal(fr.ma.masked_all(shape), np.ma.masked_all(shape))


def test_masked_all_all_masked():
    got = fr.ma.masked_all((2, 2))
    assert np.ma.getmaskarray(got).all()


def test_masked_all_like():
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert_ma_equal(fr.ma.masked_all_like(base), np.ma.masked_all_like(base))


def test_masked_all_like_from_masked():
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    na = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.masked_all_like(a), np.ma.masked_all_like(na))


# ---------------------------------------------------------------------------
# fromfunction / indices
# ---------------------------------------------------------------------------


def test_fromfunction():
    assert_ma_equal(
        fr.ma.fromfunction(lambda i, j: i + j, (2, 3)),
        np.ma.fromfunction(lambda i, j: i + j, (2, 3)),
    )


def test_fromfunction_1d():
    assert_ma_equal(
        fr.ma.fromfunction(lambda i: i * 2.0, (4,)),
        np.ma.fromfunction(lambda i: i * 2.0, (4,)),
    )


@pytest.mark.parametrize("dims", [(2, 2), (3, 4)])
def test_indices(dims):
    got = fr.ma.indices(dims)
    exp = np.ma.indices(dims)
    np.testing.assert_array_equal(np.ma.getdata(got), np.ma.getdata(exp))
    assert got.dtype == exp.dtype


# ---------------------------------------------------------------------------
# apply_along_axis / apply_over_axes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_along_axis(axis):
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    na = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert_ma_equal(
        fr.ma.apply_along_axis(np.ma.sum, axis, a),
        np.ma.apply_along_axis(np.ma.sum, axis, na),
    )


def test_apply_over_axes():
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    na = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert_ma_equal(
        fr.ma.apply_over_axes(np.ma.sum, a, [0]),
        np.ma.apply_over_axes(np.ma.sum, na, [0]),
    )


# ---------------------------------------------------------------------------
# ndenumerate
# ---------------------------------------------------------------------------


def test_ndenumerate_compressed_skips_masked():
    b = _fr([1.0, 2.0, 3.0], mask=[0, 1, 0])
    nb = _np([1.0, 2.0, 3.0], mask=[0, 1, 0])
    got = list(fr.ma.ndenumerate(b))
    exp = list(np.ma.ndenumerate(nb))
    assert [i for i, _ in got] == [i for i, _ in exp]
    np.testing.assert_array_equal(
        [float(v) for _, v in got], [float(v) for _, v in exp]
    )


def test_ndenumerate_full_yields_masked():
    b = _fr([1.0, 2.0, 3.0], mask=[0, 1, 0])
    nb = _np([1.0, 2.0, 3.0], mask=[0, 1, 0])
    got = list(fr.ma.ndenumerate(b, compressed=False))
    exp = list(np.ma.ndenumerate(nb, compressed=False))
    assert [i for i, _ in got] == [i for i, _ in exp]
    assert [v is np.ma.masked for _, v in got] == [
        v is np.ma.masked for _, v in exp
    ]


def test_ndenumerate_2d():
    c = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    nc = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert [i for i, _ in fr.ma.ndenumerate(c)] == [
        i for i, _ in np.ma.ndenumerate(nc)
    ]


# ---------------------------------------------------------------------------
# hsplit
# ---------------------------------------------------------------------------


def test_hsplit():
    a = _fr([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], mask=[[0, 1, 0, 0], [0, 0, 0, 0]])
    na = _np([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], mask=[[0, 1, 0, 0], [0, 0, 0, 0]])
    got = fr.ma.hsplit(a, 2)
    exp = np.ma.hsplit(na, 2)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert_ma_equal(g, e)


def test_hsplit_1d():
    a = _fr([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    na = _np([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    got = fr.ma.hsplit(a, 2)
    exp = np.ma.hsplit(na, 2)
    for g, e in zip(got, exp):
        assert_ma_equal(g, e)


# ---------------------------------------------------------------------------
# mask_rows / mask_cols
# ---------------------------------------------------------------------------


def test_mask_rows():
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    na = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.mask_rows(a), np.ma.mask_rows(na))


def test_mask_cols():
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    na = _np([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.mask_cols(a), np.ma.mask_cols(na))


def test_mask_rows_equals_mask_rowcols_axis0():
    a = _fr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [0, 0, 0]])
    assert_ma_equal(fr.ma.mask_rows(a), fr.ma.mask_rowcols(a, 0))


def test_mask_cols_equals_mask_rowcols_axis1():
    a = _fr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [0, 0, 0]])
    assert_ma_equal(fr.ma.mask_cols(a), fr.ma.mask_rowcols(a, 1))


# ---------------------------------------------------------------------------
# compress_nd
# ---------------------------------------------------------------------------


def test_compress_nd_all_axes():
    x = _fr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 1, 0], [0, 0, 0]])
    nx = _np([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(fr.ma.compress_nd(x), np.ma.compress_nd(nx))


@pytest.mark.parametrize("axis", [0, 1])
def test_compress_nd_axis(axis):
    x = _fr([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 1, 0], [0, 0, 0]])
    nx = _np([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 1, 0], [0, 0, 0]])
    np.testing.assert_array_equal(
        fr.ma.compress_nd(x, axis), np.ma.compress_nd(nx, axis)
    )


# ---------------------------------------------------------------------------
# bool_ / ids
# ---------------------------------------------------------------------------


def test_bool_is_numpy_bool():
    assert fr.ma.bool_ is np.bool_
    assert fr.ma.bool_ is fr.bool_


def test_ids_returns_pair():
    a = _fr([[1.0, 2.0], [3.0, 4.0]], mask=[[0, 1], [0, 0]])
    result = fr.ma.ids(a)
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert all(isinstance(x, int) for x in result)
