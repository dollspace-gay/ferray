"""Expansion coverage for `searchsorted` (scalar/array `v`, `side`, `sorter`)
and `linalg.norm` (`ord`/`axis`/`keepdims`, vector + matrix norms, complex).

Every expected value is derived LIVE from numpy 2.4.x (R-CHAR-3) — the test
asserts ferray matches numpy, never a literal copied from the ferray side.
Pins divergence #974: `fr.searchsorted(a, SCALAR v)` raised TypeError and
`fr.linalg.norm` lacked `axis=`/`ord=`/`keepdims=`.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# searchsorted — scalar v (the #974 regression), array v, side, sorter, dtypes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, v",
    [
        ([1.0, 2.0, 3.0], 2.5),          # float a, scalar float v  (#974 core)
        ([1.0, 2.0, 3.0], 0.5),
        ([1.0, 2.0, 3.0], 5.0),
        ([1, 2, 3], 2.5),                # int a, scalar float v (no lossy cast)
        ([1, 2, 3], 2),                  # int a, scalar int v
        ([1, 2, 3, 4, 5], 3),
    ],
)
def test_searchsorted_scalar_v(a, v):
    expected = np.searchsorted(a, v)
    got = fr.searchsorted(a, v)
    assert np.asarray(got) == np.asarray(expected)
    # numpy returns a scalar / 0-d for a scalar v.
    assert np.ndim(got) == np.ndim(expected)


@pytest.mark.parametrize(
    "a, v",
    [
        ([1, 2, 3], [2.5, 0.5]),         # int a, float v: must NOT lossily cast
        ([1.0, 2.0, 3.0], [2.5, 0.5]),
        ([1, 2, 3, 4], [0, 4, 2, 5]),
    ],
)
def test_searchsorted_array_v(a, v):
    expected = np.searchsorted(a, v)
    got = fr.searchsorted(a, v)
    np.testing.assert_array_equal(np.asarray(got), expected)


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_side(side):
    a = [1, 2, 2, 2, 3]
    v = 2
    expected = np.searchsorted(a, v, side=side)
    got = fr.searchsorted(a, v, side=side)
    assert np.asarray(got) == np.asarray(expected)


def test_searchsorted_sorter():
    a = [3, 1, 2]
    sorter = [1, 2, 0]
    v = [2.5, 0.5]
    expected = np.searchsorted(a, v, sorter=sorter)
    got = fr.searchsorted(a, v, sorter=sorter)
    np.testing.assert_array_equal(np.asarray(got), expected)


def test_searchsorted_bad_side_raises():
    with pytest.raises(ValueError):
        fr.searchsorted([1, 2, 3], 2, side="middle")


# ---------------------------------------------------------------------------
# linalg.norm — vector ord × axis, matrix ord, keepdims, complex
# ---------------------------------------------------------------------------

MAT = np.array([[3.0, 4.0], [5.0, 12.0]])


def _check_norm(x, **kw):
    expected = np.linalg.norm(x, **kw)
    got = fr.linalg.norm(x, **kw)
    np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-12, atol=1e-12)
    assert np.shape(got) == np.shape(expected)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_norm_axis_default_ord(axis):
    _check_norm(MAT, axis=axis)


@pytest.mark.parametrize("ord", [1, 2, np.inf, -np.inf, 0, 3])
@pytest.mark.parametrize("axis", [0, 1])
def test_norm_vector_ord_axis(ord, axis):
    _check_norm(MAT, ord=ord, axis=axis)


def test_norm_axis_keepdims():
    _check_norm(MAT, axis=1, keepdims=True)


@pytest.mark.parametrize("ord", [None, "fro", "nuc", 1, 2, np.inf, -1, -2])
def test_norm_matrix_ord_fullarray(ord):
    # axis=None full-array matrix norm (the existing real path), still matching.
    _check_norm(MAT, ord=ord)


@pytest.mark.parametrize("ord", [None, "fro", 1, 2, np.inf])
def test_norm_matrix_ord_tuple_axis(ord):
    _check_norm(MAT, ord=ord, axis=(0, 1))


def test_norm_vector_1d():
    v = np.array([3.0, 4.0])
    _check_norm(v)
    _check_norm(v, ord=1)
    _check_norm(v, ord=np.inf)


def test_norm_complex_axis():
    z = np.array([[3.0 + 4.0j, 0.0], [0.0, 5.0 + 12.0j]])
    _check_norm(z, axis=1)


def test_norm_complex_fullarray_unchanged():
    # The #931 complex full-array norm path stays green.
    z = np.array([3.0 + 4.0j, 0.0, 0.0])
    _check_norm(z)
