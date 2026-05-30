"""numpy.ma composable batch 2 (refs #835 #818).

Asserts the newly bound `ferray.ma` manipulation / elementwise / alias /
fill-value / error-class surface against `numpy.ma` directly (R-CHAR-3 — every
expected value is constructed from numpy.ma, never literal-copied from ferray).

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


def assert_ma_equal(got, expected):
    """Compare a ferray.ma result to a numpy.ma result on data + mask."""
    gd = np.asarray(got.data, dtype=float)
    gm = np.asarray(got.mask, dtype=bool)
    if gm.shape == () or gm.ndim == 0:
        gm = np.zeros(gd.shape, dtype=bool)
    ed = np.ma.getdata(expected).astype(float)
    em = np.ma.getmaskarray(expected)
    # numpy may broadcast nomask to all-False; align shapes.
    if gm.shape != ed.shape:
        gm = np.broadcast_to(gm, ed.shape)
    np.testing.assert_array_equal(gd.shape, ed.shape)
    # Compare unmasked data positions only (masked slots are don't-care).
    keep = ~em
    np.testing.assert_allclose(gd[keep], ed[keep], rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_array_equal(gm, em)


# ---------------------------------------------------------------------------
# Mask-propagating manipulation
# ---------------------------------------------------------------------------


def test_atleast_1d_2d_3d():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.atleast_1d(m), np.ma.atleast_1d(n))
    assert_ma_equal(fr.ma.atleast_2d(m), np.ma.atleast_2d(n))
    assert_ma_equal(fr.ma.atleast_3d(m), np.ma.atleast_3d(n))


def test_column_stack_dstack():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.column_stack([m, m]), np.ma.column_stack([n, n]))
    assert_ma_equal(fr.ma.dstack([m, m]), np.ma.dstack([n, n]))


def test_vstack_hstack_stack():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.vstack([m, m]), np.ma.vstack([n, n]))
    assert_ma_equal(fr.ma.hstack([m, m]), np.ma.hstack([n, n]))
    assert_ma_equal(fr.ma.stack([m, m]), np.ma.stack([n, n]))
    assert_ma_equal(fr.ma.row_stack([m, m]), np.ma.row_stack([n, n]))


def test_diagonal_diagflat():
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.diagonal(m2), np.ma.diagonal(n2))
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.diagflat(m), np.ma.diagflat(n))


def test_swapaxes_resize():
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.swapaxes(m2, 0, 1), np.ma.swapaxes(n2, 0, 1))
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.resize(m, (2, 3)), np.ma.resize(n, (2, 3)))


def test_append_flatten_and_axis():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.append(m, [4.0, 5.0]), np.ma.append(n, [4.0, 5.0]))
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert_ma_equal(
        fr.ma.append(m2, [[5.0, 6.0]], axis=0),
        np.ma.append(n2, [[5.0, 6.0]], axis=0),
    )


def test_compress_1d_and_axis():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.compress([1, 0, 1], m), np.ma.compress([1, 0, 1], n))
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert_ma_equal(
        fr.ma.compress([1, 0], m2, axis=0), np.ma.compress([1, 0], n2, axis=0)
    )


def test_empty_like_ones_like_zeros_like():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert fr.ma.empty_like(m).shape == np.ma.empty_like(n).shape
    assert_ma_equal(fr.ma.ones_like(m), np.ma.ones_like(n))
    assert_ma_equal(fr.ma.zeros_like(m), np.ma.zeros_like(n))


# ---------------------------------------------------------------------------
# Cumulative ops (masked → identity, accumulate, mask kept)
# ---------------------------------------------------------------------------


def test_cumsum_cumprod():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.cumsum(m), np.ma.cumsum(n))
    assert_ma_equal(fr.ma.cumprod(m), np.ma.cumprod(n))


def test_cumsum_axis():
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert_ma_equal(fr.ma.cumsum(m2, axis=0), np.ma.cumsum(n2, axis=0))


# ---------------------------------------------------------------------------
# Rounding + alias
# ---------------------------------------------------------------------------


def test_round_and_round_alias():
    m = _fr([1.234, 2.5, 3.456], [0, 1, 0])
    n = _np([1.234, 2.5, 3.456], [0, 1, 0])
    assert_ma_equal(fr.ma.round(m, 1), np.ma.round(n, 1))
    assert_ma_equal(fr.ma.round_(m, 2), np.ma.round_(n, 2))


# ---------------------------------------------------------------------------
# Masked comparisons + logical ops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["equal", "not_equal", "greater", "greater_equal", "less", "less_equal"],
)
def test_comparisons(name):
    m = _fr([1, 2, 3], [0, 1, 0])
    m2 = _fr([1, 2, 4], [1, 0, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    n2 = _np([1, 2, 4], [1, 0, 0])
    got = getattr(fr.ma, name)(m, m2)
    exp = getattr(np.ma, name)(n, n2)
    assert_ma_equal(got, exp)


@pytest.mark.parametrize("name", ["logical_and", "logical_or", "logical_xor"])
def test_logical_binary(name):
    m = _fr([1, 0, 3], [0, 1, 0])
    m2 = _fr([1, 1, 0], [0, 0, 0])
    n = _np([1, 0, 3], [0, 1, 0])
    n2 = _np([1, 1, 0], [0, 0, 0])
    assert_ma_equal(getattr(fr.ma, name)(m, m2), getattr(np.ma, name)(n, n2))


def test_logical_not():
    m = _fr([1, 0, 3], [0, 1, 0])
    n = _np([1, 0, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.logical_not(m), np.ma.logical_not(n))


# ---------------------------------------------------------------------------
# Reductions as free functions + aliases
# ---------------------------------------------------------------------------


def test_reduction_free_functions():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert fr.ma.amax(m) == np.ma.amax(n)
    assert fr.ma.amin(m) == np.ma.amin(n)
    assert fr.ma.max(m) == np.ma.max(n)
    assert fr.ma.min(m) == np.ma.min(n)
    assert fr.ma.sum(m) == np.ma.sum(n)
    assert fr.ma.mean(m) == np.ma.mean(n)
    assert fr.ma.product(m) == np.ma.product(n)
    np.testing.assert_allclose(fr.ma.std(m), np.ma.std(n))
    np.testing.assert_allclose(fr.ma.var(m), np.ma.var(n))


def test_alltrue_sometrue_aliases():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert fr.ma.alltrue(m) == bool(np.ma.alltrue(n))
    assert fr.ma.sometrue(m) == bool(np.ma.sometrue(n))
    z = _fr([0, 0, 0], [0, 1, 0])
    nz = _np([0, 0, 0], [0, 1, 0])
    assert fr.ma.alltrue(z) == bool(np.ma.alltrue(nz))
    assert fr.ma.sometrue(z) == bool(np.ma.sometrue(nz))


def test_anom_anomalies():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert_ma_equal(fr.ma.anom(m), np.ma.anom(n))
    assert_ma_equal(fr.ma.anomalies(m), np.ma.anomalies(n))


# ---------------------------------------------------------------------------
# Inner / outer products + shape inspectors
# ---------------------------------------------------------------------------


def test_outer_inner():
    a = _fr([1, 2], [0, 1])
    b = _fr([1, 1], None)
    na = _np([1, 2], [0, 1])
    nb = _np([1, 1], None)
    assert_ma_equal(fr.ma.outer(a, b), np.ma.outer(na, nb))
    assert_ma_equal(fr.ma.outerproduct(a, b), np.ma.outerproduct(na, nb))
    x = _fr([1, 2, 3], None)
    y = _fr([1, 1, 1], None)
    assert fr.ma.inner(x, y) == np.ma.inner(_np([1, 2, 3]), _np([1, 1, 1]))
    assert fr.ma.innerproduct(x, y) == np.ma.innerproduct(
        _np([1, 2, 3]), _np([1, 1, 1])
    )


def test_inner_skips_masked():
    a = _fr([1, 2, 3], [0, 1, 0])
    b = _fr([2, 2, 2], None)
    na = _np([1, 2, 3], [0, 1, 0])
    nb = _np([2, 2, 2], None)
    assert fr.ma.inner(a, b) == np.ma.inner(na, nb)


def test_shape_inspectors():
    m2 = _fr([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    n2 = _np([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert fr.ma.ndim(m2) == np.ma.ndim(n2)
    assert tuple(fr.ma.shape(m2)) == np.ma.shape(n2)
    assert fr.ma.size(m2) == np.ma.size(n2)


# ---------------------------------------------------------------------------
# Angle
# ---------------------------------------------------------------------------


def test_angle_real():
    m = _fr([1.0, -1.0, 2.0], [0, 1, 0])
    n = _np([1.0, -1.0, 2.0], [0, 1, 0])
    assert_ma_equal(fr.ma.angle(m), np.ma.angle(n))


# ---------------------------------------------------------------------------
# Predicates / fill-value helpers
# ---------------------------------------------------------------------------


def test_allequal_allclose():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert fr.ma.allequal(m, m) == bool(np.ma.allequal(n, n))
    assert fr.ma.allclose(m, m) == bool(np.ma.allclose(n, n))
    m2 = _fr([1, 2, 9], [0, 1, 0])
    n2 = _np([1, 2, 9], [0, 1, 0])
    assert fr.ma.allequal(m, m2) == bool(np.ma.allequal(n, n2))
    assert fr.ma.allclose(m, m2) == bool(np.ma.allclose(n, n2))


def test_allequal_masked_pair_counts_equal():
    # With fill_value=True (default), a masked position in either operand is
    # treated as equal (numpy.ma.allequal).
    a = _fr([1, 5, 3], [0, 1, 0])
    b = _fr([1, 7, 3], [0, 1, 0])
    na = _np([1, 5, 3], [0, 1, 0])
    nb = _np([1, 7, 3], [0, 1, 0])
    assert fr.ma.allequal(a, b) == bool(np.ma.allequal(na, nb))


def test_fill_value_helpers():
    m = _fr([1, 2, 3], [0, 1, 0])
    n = _np([1, 2, 3], [0, 1, 0])
    assert fr.ma.maximum_fill_value(m) == np.ma.maximum_fill_value(n)
    assert fr.ma.minimum_fill_value(m) == np.ma.minimum_fill_value(n)
    assert fr.ma.common_fill_value(m, m) == np.ma.common_fill_value(n, n)


def test_common_fill_value_differ_is_none():
    a = fr.ma.array(np.array([1.0]))
    b = fr.ma.array(np.array([2.0]))
    fr.ma.set_fill_value(a, 5.0)
    fr.ma.set_fill_value(b, 9.0)
    assert fr.ma.common_fill_value(a, b) is None


# ---------------------------------------------------------------------------
# Error classes / type vocabulary + singletons (re-exported from numpy)
# ---------------------------------------------------------------------------


def test_error_classes_are_numpy_classes():
    assert fr.ma.MAError is np.ma.MAError
    assert fr.ma.MaskError is np.ma.MaskError
    assert issubclass(fr.ma.MaskError, fr.ma.MAError)
    assert issubclass(fr.ma.MAError, Exception)


def test_mask_type_and_singletons():
    assert fr.ma.MaskType is np.ma.MaskType
    assert fr.ma.masked is np.ma.masked
    assert fr.ma.masked_singleton is np.ma.masked
    assert fr.ma.nomask is np.ma.nomask
