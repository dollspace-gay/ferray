"""numpy.ma specialized masked algorithms — ferray.ma vs numpy.ma (refs #835).

Every expected value is produced by the live ``numpy.ma.*`` oracle (numpy
2.4.5), never literal-copied from the ferray side (R-CHAR-3). Covers:
where / choose / diff / ediff1d / nonzero, plus the corrected vander / isin /
in1d mask semantics.
"""

import numpy as np
import pytest

import ferray as fr


def _ma(data, mask=None):
    """ferray masked array."""
    if mask is None:
        return fr.ma.masked_array(np.asarray(data, dtype=float))
    return fr.ma.masked_array(
        np.asarray(data, dtype=float), mask=np.asarray(mask, dtype=bool)
    )


def _np(data, mask=None):
    if mask is None:
        return np.ma.array(np.asarray(data, dtype=float))
    return np.ma.array(np.asarray(data, dtype=float), mask=np.asarray(mask, dtype=bool))


def _assert_ma_equal(got, expected):
    """Compare ferray masked result to a numpy.ma expected (data + mask)."""
    np.testing.assert_array_equal(np.asarray(got.mask, dtype=bool),
                                  np.ma.getmaskarray(expected))
    # Compare data only at unmasked positions (masked data is unobservable).
    gm = np.asarray(got.mask, dtype=bool)
    gd = np.asarray(got.data, dtype=float)
    ed = np.ma.getdata(expected).astype(float)
    np.testing.assert_allclose(gd[~gm], ed[~gm])


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------


def test_where_scalar_branches():
    cond = _ma([1.0, 0.0, 1.0], [0, 1, 0])
    got = fr.ma.where(cond, _ma([10.0, 10.0, 10.0]), _ma([20.0, 20.0, 20.0]))
    exp = np.ma.where(
        _np([1.0, 0.0, 1.0], [0, 1, 0]) > 0, 10.0, 20.0
    )
    # numpy's `>0` masks where cond masked; our cond is the bool directly.
    # Recreate the exact oracle: condition is the masked bool.
    exp = np.ma.where(
        np.ma.array([True, False, True], mask=[0, 1, 0]),
        np.ma.array([10.0, 10.0, 10.0]),
        np.ma.array([20.0, 20.0, 20.0]),
    )
    _assert_ma_equal(got, exp)


def test_where_propagates_source_mask():
    got = fr.ma.where(
        _ma([1.0, 1.0, 0.0]),
        _ma([1.0, 2.0, 3.0], [0, 1, 0]),
        _ma([4.0, 5.0, 6.0], [1, 0, 1]),
    )
    exp = np.ma.where(
        np.array([True, True, False]),
        np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]),
        np.ma.array([4.0, 5.0, 6.0], mask=[1, 0, 1]),
    )
    _assert_ma_equal(got, exp)


# ---------------------------------------------------------------------------
# choose
# ---------------------------------------------------------------------------


def test_choose_masked_index():
    got = fr.ma.choose(
        _ma([0.0, 1.0, 0.0], [0, 1, 0]),
        [_ma([10.0, 20.0, 30.0]), _ma([40.0, 50.0, 60.0])],
    )
    exp = np.ma.choose(
        np.ma.array([0, 1, 0], mask=[0, 1, 0]),
        ([10.0, 20.0, 30.0], [40.0, 50.0, 60.0]),
    )
    _assert_ma_equal(got, exp)


def test_choose_masked_choice():
    got = fr.ma.choose(
        _ma([0.0, 1.0, 0.0]),
        [_ma([10.0, 20.0, 30.0], [0, 0, 1]), _ma([40.0, 50.0, 60.0])],
    )
    exp = np.ma.choose(
        np.array([0, 1, 0]),
        (np.ma.array([10.0, 20.0, 30.0], mask=[0, 0, 1]), np.ma.array([40.0, 50.0, 60.0])),
    )
    _assert_ma_equal(got, exp)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_diff_n1():
    a = [1.0, 2.0, 3.0, 4.0, 7.0, 0.0, 2.0, 3.0]
    msk = [v < 2 for v in a]
    got = fr.ma.diff(_ma(a, msk))
    exp = np.ma.diff(np.ma.masked_where(np.array(a) < 2, np.array(a)))
    _assert_ma_equal(got, exp)


def test_diff_n2():
    a = [1.0, 2.0, 3.0, 4.0, 7.0, 0.0, 2.0, 3.0]
    msk = [v < 2 for v in a]
    got = fr.ma.diff(_ma(a, msk), n=2)
    exp = np.ma.diff(np.ma.masked_where(np.array(a) < 2, np.array(a)), n=2)
    _assert_ma_equal(got, exp)


def test_diff_axis0_2d():
    a = np.array([[1.0, 3.0, 1.0, 5.0, 10.0], [0.0, 1.0, 5.0, 6.0, 8.0]])
    msk = (a == 1.0)
    got = fr.ma.diff(_ma(a, msk), axis=0)
    exp = np.ma.diff(np.ma.masked_equal(a, 1.0), axis=0)
    _assert_ma_equal(got, exp)


def test_diff_n0_returns_input():
    a = [1.0, 2.0, 3.0]
    got = fr.ma.diff(_ma(a), n=0)
    exp = np.ma.diff(np.ma.array(a), n=0)
    _assert_ma_equal(got, exp)


# ---------------------------------------------------------------------------
# ediff1d
# ---------------------------------------------------------------------------


def test_ediff1d_basic():
    got = fr.ma.ediff1d(_ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0]))
    exp = np.ma.ediff1d(np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]))
    _assert_ma_equal(got, exp)


def test_ediff1d_to_begin_end():
    got = fr.ma.ediff1d(
        _ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0]),
        to_begin=np.array([99.0]),
        to_end=np.array([88.0]),
    )
    exp = np.ma.ediff1d(
        np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]),
        to_begin=99.0,
        to_end=88.0,
    )
    _assert_ma_equal(got, exp)


# ---------------------------------------------------------------------------
# nonzero
# ---------------------------------------------------------------------------


def test_nonzero_1d():
    got = fr.ma.nonzero(_ma([0.0, 1.0, 0.0, 2.0], [0, 0, 1, 0]))
    exp = np.ma.nonzero(np.ma.array([0.0, 1.0, 0.0, 2.0], mask=[0, 0, 1, 0]))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        np.testing.assert_array_equal(np.asarray(g), e)


def test_nonzero_2d():
    a = np.array([[0.0, 1.0], [2.0, 0.0]])
    msk = np.array([[0, 0], [1, 0]], dtype=bool)
    got = fr.ma.nonzero(_ma(a, msk))
    exp = np.ma.nonzero(np.ma.array(a, mask=msk))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        np.testing.assert_array_equal(np.asarray(g), e)


# ---------------------------------------------------------------------------
# count / getdata / getmaskarray (already bound — assert numpy parity)
# ---------------------------------------------------------------------------


def test_count_unmasked():
    m = _ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0])
    exp = np.ma.count(np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]))
    assert fr.ma.count(m) == int(exp)


def test_getdata_getmaskarray():
    m = _ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0])
    npm = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
    np.testing.assert_array_equal(np.asarray(fr.ma.getdata(m)), np.ma.getdata(npm))
    np.testing.assert_array_equal(
        np.asarray(fr.ma.getmaskarray(m)), np.ma.getmaskarray(npm)
    )


# ---------------------------------------------------------------------------
# vander (corrected: masked rows -> all zeros, nomask)
# ---------------------------------------------------------------------------


def test_vander_masked_rows_zeroed_nomask():
    got = fr.ma.vander(_ma([1.0, 2.0, 3.0], [0, 1, 0]), 3)
    exp = np.ma.vander(np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]), 3)
    np.testing.assert_array_equal(
        np.asarray(got.mask, dtype=bool), np.ma.getmaskarray(exp)
    )
    np.testing.assert_allclose(np.asarray(got.data), np.asarray(exp))


def test_vander_no_mask():
    got = fr.ma.vander(_ma([1.0, 2.0, 3.0]), 3)
    exp = np.ma.vander(np.ma.array([1.0, 2.0, 3.0]), 3)
    np.testing.assert_allclose(np.asarray(got.data), np.asarray(exp))
    assert not np.asarray(got.mask, dtype=bool).any()


# ---------------------------------------------------------------------------
# isin / in1d (corrected: masked positions report False, mask preserved)
# ---------------------------------------------------------------------------


def test_isin_masked_position_false():
    got = fr.ma.isin(_ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0]), np.array([2.0, 3.0]))
    exp = np.ma.isin(
        np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]), [2.0, 3.0]
    )
    np.testing.assert_array_equal(np.asarray(got, dtype=bool), np.ma.getdata(exp))


def test_in1d_membership():
    got = fr.ma.in1d(_ma([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 0]), np.array([2.0, 3.0]))
    exp = np.ma.in1d(
        np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0]), [2.0, 3.0]
    )
    np.testing.assert_array_equal(np.asarray(got, dtype=bool), np.ma.getdata(exp))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
