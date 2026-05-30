"""Specialized numpy.ma algorithm bindings (refs #835, #818).

Every expected value is produced by a *live* ``numpy.ma`` call (R-CHAR-3):
the oracle is ``numpy`` 2.4.x. Each binding is checked on both the unmasked
values (via ``np.ma.getdata`` on the unmasked positions) and the resulting
``mask`` — masked positions carry no observable data contract, so we never
assert masked data values.
"""

import numpy as np
import pytest

import ferray as fr


def _fr(data, mask=None):
    return fr.ma.masked_array(data, mask=mask)


def _np(data, mask=None):
    return np.ma.array(np.asarray(data, dtype=float), mask=mask)


def _assert_ma_equal(got, expected):
    """Compare a ferray.ma result against a numpy.ma result.

    Asserts identical mask, and identical *unmasked* data values.
    """
    g_data = np.asarray(got.data)
    g_mask = np.asarray(got.mask)
    e_data = np.ma.getdata(expected)
    e_mask = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(g_mask, e_mask)
    # Only unmasked positions have an observable value.
    unmasked = ~e_mask
    np.testing.assert_allclose(
        np.asarray(g_data)[unmasked], np.asarray(e_data)[unmasked]
    )


# ---------------------------------------------------------------------------
# sort / argsort — masked values trail
# ---------------------------------------------------------------------------


def test_sort_1d_masked_last():
    g = fr.ma.sort(_fr([3.0, 1.0, 2.0], mask=[0, 1, 0]))
    e = np.ma.sort(_np([3.0, 1.0, 2.0], mask=[0, 1, 0]))
    _assert_ma_equal(g, e)


def test_sort_axis_none_flattens():
    data = [[3.0, 1.0], [2.0, 4.0]]
    mask = [[0, 1], [0, 0]]
    g = fr.ma.sort(_fr(data, mask=mask), axis=None)
    e = np.ma.sort(_np(data, mask=mask), axis=None)
    _assert_ma_equal(g, e)


def test_sort_default_axis_last():
    data = [[3.0, 1.0], [2.0, 4.0]]
    mask = [[0, 1], [0, 0]]
    g = fr.ma.sort(_fr(data, mask=mask))
    e = np.ma.sort(_np(data, mask=mask), axis=-1)
    _assert_ma_equal(g, e)


def test_sort_axis0():
    data = [[3.0, 1.0], [2.0, 4.0]]
    mask = [[0, 1], [0, 0]]
    g = fr.ma.sort(_fr(data, mask=mask), axis=0)
    e = np.ma.sort(_np(data, mask=mask), axis=0)
    _assert_ma_equal(g, e)


def test_argsort_1d_masked_last():
    g = fr.ma.argsort(_fr([3.0, 1.0, 2.0], mask=[0, 1, 0]))
    e = np.ma.argsort(_np([3.0, 1.0, 2.0], mask=[0, 1, 0]), axis=None)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_argsort_no_mask():
    g = fr.ma.argsort(_fr([5.0, 2.0, 8.0, 1.0]))
    e = np.ma.argsort(_np([5.0, 2.0, 8.0, 1.0]), axis=None)
    np.testing.assert_array_equal(np.asarray(g), np.asarray(e))


def test_argsort_multidim_axis_rejected():
    with pytest.raises(ValueError):
        fr.ma.argsort(_fr([[1.0, 2.0], [3.0, 4.0]]), axis=1)


# ---------------------------------------------------------------------------
# take
# ---------------------------------------------------------------------------


def test_take_carries_mask():
    src = _fr([3.0, 1.0, 2.0, 9.0], mask=[0, 1, 0, 1])
    g = fr.ma.take(src, [0, 1, 3])
    e = np.ma.take(_np([3.0, 1.0, 2.0, 9.0], mask=[0, 1, 0, 1]), [0, 1, 3])
    _assert_ma_equal(g, e)


def test_take_out_of_bounds_raises():
    with pytest.raises((IndexError, ValueError)):
        fr.ma.take(_fr([1.0, 2.0]), [5])


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------


def test_trace_square():
    g = fr.ma.trace(_fr([[1.0, 2.0], [3.0, 4.0]]))
    e = np.ma.trace(_np([[1.0, 2.0], [3.0, 4.0]]))
    assert g == pytest.approx(float(e))


def test_trace_masked_diagonal_skipped():
    data = [[1.0, 2.0], [3.0, 4.0]]
    mask = [[1, 0], [0, 0]]
    g = fr.ma.trace(_fr(data, mask=mask))
    e = np.ma.trace(_np(data, mask=mask))
    assert g == pytest.approx(float(e))


def test_trace_offset():
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    g = fr.ma.trace(_fr(data), offset=1)
    e = np.ma.trace(_np(data), offset=1)
    assert g == pytest.approx(float(e))


def test_trace_non_2d_raises():
    with pytest.raises(ValueError):
        fr.ma.trace(_fr([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# dot (1-D inner product)
# ---------------------------------------------------------------------------


def test_dot_1d_inner():
    g = fr.ma.dot(_fr([1.0, 2.0, 3.0]), _fr([4.0, 5.0, 6.0]))
    e = np.ma.dot(_np([1.0, 2.0, 3.0]), _np([4.0, 5.0, 6.0]))
    assert g == pytest.approx(float(e))


def test_dot_masked_positions_zero():
    a = _fr([1.0, 2.0, 3.0], mask=[0, 1, 0])
    b = _fr([4.0, 5.0, 6.0], mask=[0, 0, 0])
    g = fr.ma.dot(a, b)
    e = np.ma.dot(_np([1.0, 2.0, 3.0], mask=[0, 1, 0]), _np([4.0, 5.0, 6.0]))
    assert g == pytest.approx(float(e))


def test_dot_2d_rejected():
    with pytest.raises(ValueError):
        fr.ma.dot(_fr([[1.0, 2.0], [3.0, 4.0]]), _fr([[1.0, 0.0], [0.0, 1.0]]))


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------


def test_unique_no_mask():
    g = fr.ma.unique(_fr([3.0, 1.0, 2.0, 1.0]))
    e = np.ma.unique(_np([3.0, 1.0, 2.0, 1.0]))
    _assert_ma_equal(g, e)


def test_unique_with_mask_trailing_masked_entry():
    g = fr.ma.unique(_fr([1.0, 1.0, 2.0], mask=[0, 1, 0]))
    e = np.ma.unique(_np([1.0, 1.0, 2.0], mask=[0, 1, 0]))
    _assert_ma_equal(g, e)


def test_unique_multiple_masked_collapse_to_one():
    g = fr.ma.unique(_fr([9.0, 8.0, 2.0, 2.0], mask=[1, 1, 0, 0]))
    e = np.ma.unique(_np([9.0, 8.0, 2.0, 2.0], mask=[1, 1, 0, 0]))
    _assert_ma_equal(g, e)


# ---------------------------------------------------------------------------
# vander / isin / in1d ARE now bound (#835), with mask semantics corrected to
# match numpy.ma: vander zeros masked rows + nomask; isin/in1d preserve the
# input mask and report masked positions as False. Full parity coverage lives
# in test_expansion_ma_lib.py; this guards the binding surface stays present.
# ---------------------------------------------------------------------------


def test_vander_isin_in1d_now_bound_and_match_numpy():
    assert hasattr(fr.ma, "vander")
    assert hasattr(fr.ma, "isin")
    assert hasattr(fr.ma, "in1d")
    # vander: masked rows -> all zeros, nomask (numpy/ma/extras.py:2216).
    got = fr.ma.vander(
        fr.ma.masked_array(np.array([1.0, 2.0, 3.0]), mask=np.array([0, 1, 0], dtype=bool)),
        3,
    )
    exp = np.ma.vander(np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]), 3)
    np.testing.assert_allclose(np.asarray(got.data), np.asarray(exp))
    assert not np.asarray(got.mask, dtype=bool).any()
