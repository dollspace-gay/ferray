"""numpy.ma library expansion slice 2 (refs #835): masked set operations,
row/column suppression, and masked cov/corrcoef.

Every expected value is produced by calling ``numpy.ma`` directly on the same
inputs (R-CHAR-3) — never literal-copied from the ferray side. For masked set
ops the trailing masked slot's underlying *data* value is not observable, so we
compare only the unmasked positions' values plus the full mask vector.
"""

import numpy as np
import pytest

import ferray
import ferray.ma as fma


def _to_np(m):
    """A ferray.ma.MaskedArray -> (data ndarray, mask ndarray)."""
    data = np.asarray(m.data, dtype=float)
    mask = np.asarray(m.mask, dtype=bool)
    if mask.shape != data.shape:  # nomask -> broadcast to all-False
        mask = np.zeros(data.shape, dtype=bool)
    return data, mask


def _assert_ma_equal(got, expected, atol=1e-12):
    """Compare a ferray masked array to a numpy masked array: masks must match
    exactly, and unmasked data values must match within tolerance."""
    gd, gm = _to_np(got)
    ed = np.asarray(expected.data, dtype=float)
    em = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(gm, em)
    # Compare only unmasked positions (masked slot data is unobservable).
    np.testing.assert_allclose(gd[~gm], ed[~em], atol=atol)


# ---------------------------------------------------------------------------
# Set operations
# ---------------------------------------------------------------------------

A = np.ma.array([1.0, 2.0, 3.0, 4.0], mask=[0, 1, 0, 0])
B = np.ma.array([3.0, 4.0, 5.0], mask=[0, 0, 1])


def _fr(npma):
    return fma.masked_array(np.asarray(npma.data), np.ma.getmaskarray(npma))


def test_intersect1d():
    got = fma.intersect1d(_fr(A), _fr(B))
    _assert_ma_equal(got, np.ma.intersect1d(A, B))


def test_union1d():
    got = fma.union1d(_fr(A), _fr(B))
    _assert_ma_equal(got, np.ma.union1d(A, B))


def test_setdiff1d():
    got = fma.setdiff1d(_fr(A), _fr(B))
    _assert_ma_equal(got, np.ma.setdiff1d(A, B))


def test_setxor1d():
    got = fma.setxor1d(_fr(A), _fr(B))
    _assert_ma_equal(got, np.ma.setxor1d(A, B))


def test_setdiff1d_no_rhs_mask_keeps_masked_slot():
    # When ar2 has no masked element, ar1's masked slot survives.
    a = np.ma.array([1.0, 2.0, 9.0], mask=[0, 0, 1])
    b = np.ma.array([2.0], mask=[0])
    got = fma.setdiff1d(_fr(a), _fr(b))
    _assert_ma_equal(got, np.ma.setdiff1d(a, b))


# ---------------------------------------------------------------------------
# compress_rowcols / compress_rows / compress_cols
# ---------------------------------------------------------------------------

M = np.ma.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mask=[[0, 0, 1], [0, 0, 0]])


def _frm(npma):
    return fma.masked_array(np.asarray(npma.data), np.ma.getmaskarray(npma))


def test_compress_rowcols_none():
    got = np.asarray(fma.compress_rowcols(_frm(M)))
    np.testing.assert_allclose(got, np.ma.compress_rowcols(M))


def test_compress_rowcols_axis0():
    got = np.asarray(fma.compress_rowcols(_frm(M), 0))
    np.testing.assert_allclose(got, np.ma.compress_rowcols(M, 0))


def test_compress_rowcols_axis1():
    got = np.asarray(fma.compress_rowcols(_frm(M), 1))
    np.testing.assert_allclose(got, np.ma.compress_rowcols(M, 1))


def test_compress_rows():
    got = np.asarray(fma.compress_rows(_frm(M)))
    np.testing.assert_allclose(got, np.ma.compress_rows(M))


def test_compress_cols():
    got = np.asarray(fma.compress_cols(_frm(M)))
    np.testing.assert_allclose(got, np.ma.compress_cols(M))


def test_compress_rowcols_not_2d_raises():
    v = fma.masked_array(np.array([1.0, 2.0, 3.0]), np.array([0, 1, 0]))
    with pytest.raises((ValueError, Exception)):
        fma.compress_rowcols(v)


# ---------------------------------------------------------------------------
# mask_rowcols
# ---------------------------------------------------------------------------


def test_mask_rowcols_none():
    got = fma.mask_rowcols(_frm(M))
    _assert_ma_equal(got, np.ma.mask_rowcols(M.copy()))


def test_mask_rowcols_axis0():
    got = fma.mask_rowcols(_frm(M), 0)
    _assert_ma_equal(got, np.ma.mask_rowcols(M.copy(), 0))


def test_mask_rowcols_axis1():
    got = fma.mask_rowcols(_frm(M), 1)
    _assert_ma_equal(got, np.ma.mask_rowcols(M.copy(), 1))


# ---------------------------------------------------------------------------
# cov / corrcoef
# ---------------------------------------------------------------------------


def test_cov_basic():
    got = fma.cov(_frm(M))
    _assert_ma_equal(got, np.ma.cov(M))


def test_cov_rowvar_false():
    got = fma.cov(_frm(M), None, False)  # rowvar=False
    _assert_ma_equal(got, np.ma.cov(M, rowvar=False))


def test_cov_bias():
    got = fma.cov(_frm(M), None, True, True)  # rowvar=True, bias=True
    _assert_ma_equal(got, np.ma.cov(M, bias=True))


def test_corrcoef_basic():
    got = fma.corrcoef(_frm(M))
    _assert_ma_equal(got, np.ma.corrcoef(M), atol=1e-10)


def test_cov_no_mask_matches_plain():
    # With no masked values, ma.cov reduces to the ordinary covariance.
    x = np.ma.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    got = fma.cov(_frm(x))
    _assert_ma_equal(got, np.ma.cov(x))


def test_y_argument_unsupported():
    with pytest.raises((ValueError, Exception)):
        fma.cov(_frm(M), _frm(M))
