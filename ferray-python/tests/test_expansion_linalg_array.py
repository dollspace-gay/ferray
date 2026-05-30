"""Expansion batch: numpy 2.0 array-API linalg gaps + top-level array
helpers (refs #818).

Every expected value comes from a *live* numpy call (R-CHAR-3): the test
asserts `ferray.<fn>(...)` equals `numpy.<fn>(...)` on identical inputs,
so the oracle is numpy itself, never a literal copied from the ferray
side.

Group A — numpy.linalg: matrix_norm / vector_norm / svdvals / cross.
Group B — top-level array helpers: empty_like / copyto / ix_ /
fill_diagonal / *_indices_from / put_along_axis / asarray_chkfinite /
may_share_memory / shares_memory / iterable / unique_all/counts/
inverse/values.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Group A — numpy.linalg array-API norms / singular values / cross
# ---------------------------------------------------------------------------


def test_matrix_norm_fro_default():
    x = [[1.0, 2.0], [3.0, 4.0]]
    assert np.allclose(fr.linalg.matrix_norm(x), np.linalg.matrix_norm(x))


def test_matrix_norm_explicit_ords():
    x = [[1.0, -2.0], [-3.0, 4.0]]
    for ord_ in ("fro", "nuc", 1, np.inf, -1, 2, -2):
        assert np.allclose(
            fr.linalg.matrix_norm(x, ord=ord_),
            np.linalg.matrix_norm(x, ord=ord_),
        ), ord_


def test_matrix_norm_keepdims():
    x = [[1.0, 2.0], [3.0, 4.0]]
    a = fr.linalg.matrix_norm(x, keepdims=True)
    b = np.linalg.matrix_norm(x, keepdims=True)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_norm_2d_default_is_frobenius():
    # numpy.linalg.norm(x) with ord=None on a 2-D matrix is the Frobenius
    # norm, NOT the ord=2 largest-singular-value (_linalg.py:2650 norm
    # table: "None -> Frobenius norm / 2-norm"). Regression for #826:
    # ferray previously returned the ord=2 spectral norm (~5.465) as the
    # default instead of Frobenius (~5.477).
    x = [[1.0, 2.0], [3.0, 4.0]]
    assert np.allclose(fr.linalg.norm(x), np.linalg.norm(x))
    # Frobenius (default) and the explicit ord=2 spectral norm must differ.
    assert not np.allclose(fr.linalg.norm(x), fr.linalg.norm(x, 2))
    assert np.allclose(fr.linalg.norm(x, 2), np.linalg.norm(x, 2))


def test_norm_1d_default_is_2norm():
    # For a 1-D vector ord=None stays the 2-norm (unchanged by #826).
    x = [3.0, 4.0]
    assert np.allclose(fr.linalg.norm(x), np.linalg.norm(x))


def test_norm_2d_default_int_input_frobenius():
    # Integer 2-D input is promoted to float64 (_commonType) and still
    # defaults to Frobenius.
    x = [[1, 2], [3, 4]]
    assert np.allclose(fr.linalg.norm(x), np.linalg.norm(x))


def test_vector_norm_default_flatten():
    x = [3.0, 4.0]
    assert np.allclose(fr.linalg.vector_norm(x), np.linalg.vector_norm(x))


def test_vector_norm_2d_flattens():
    x = [[3.0, 4.0], [0.0, 0.0]]
    assert np.allclose(fr.linalg.vector_norm(x), np.linalg.vector_norm(x))


def test_vector_norm_axis():
    x = [[3.0, 4.0], [6.0, 8.0]]
    assert np.allclose(
        fr.linalg.vector_norm(x, axis=1),
        np.linalg.vector_norm(x, axis=1),
    )


def test_vector_norm_ord():
    x = [1.0, -2.0, 3.0, -4.0]
    for ord_ in (1, 2, np.inf, -np.inf):
        assert np.allclose(
            fr.linalg.vector_norm(x, ord=ord_),
            np.linalg.vector_norm(x, ord=ord_),
        ), ord_


def test_vector_norm_keepdims():
    x = [[3.0, 4.0]]
    a = fr.linalg.vector_norm(x, keepdims=True)
    b = np.linalg.vector_norm(x, keepdims=True)
    assert a.shape == b.shape
    assert np.allclose(a, b)


def test_svdvals_matches():
    x = [[1.0, 0.0], [0.0, 2.0]]
    assert np.allclose(fr.linalg.svdvals(x), np.linalg.svdvals(x))


def test_svdvals_rectangular():
    x = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert np.allclose(fr.linalg.svdvals(x), np.linalg.svdvals(x))


def test_svdvals_int_input_promotes():
    x = [[1, 0], [0, 2]]
    assert np.allclose(fr.linalg.svdvals(x), np.linalg.svdvals(x))


def test_cross_basis_vectors():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    assert np.allclose(fr.linalg.cross(a, b), np.linalg.cross(a, b))


def test_cross_general():
    a, b = [2.0, 3.0, 4.0], [5.0, 6.0, 7.0]
    assert np.allclose(fr.linalg.cross(a, b), np.linalg.cross(a, b))


# ---------------------------------------------------------------------------
# Group B — top-level array helpers
# ---------------------------------------------------------------------------


def test_empty_like_shape_dtype():
    proto = np.arange(6).reshape(2, 3)
    a = fr.empty_like(proto)
    b = np.empty_like(proto)
    assert a.shape == b.shape
    assert a.dtype == b.dtype


def test_empty_like_dtype_override():
    proto = [1, 2, 3]
    a = fr.empty_like(proto, dtype="float64")
    b = np.empty_like(proto, dtype="float64")
    assert a.shape == b.shape
    assert a.dtype == b.dtype


def test_copyto_inplace():
    a = np.zeros(3)
    ref = np.zeros(3)
    fr.copyto(a, [1.0, 2.0, 3.0])
    np.copyto(ref, [1.0, 2.0, 3.0])
    assert np.array_equal(a, ref)


def test_ix_open_mesh():
    fa = fr.ix_([0, 1], [2, 4])
    na = np.ix_([0, 1], [2, 4])
    assert len(fa) == len(na)
    for f, n in zip(fa, na):
        assert np.array_equal(f, n)


def test_fill_diagonal_inplace():
    a = np.zeros((3, 3))
    ref = np.zeros((3, 3))
    fr.fill_diagonal(a, 5)
    np.fill_diagonal(ref, 5)
    assert np.array_equal(a, ref)


def test_diag_indices_from():
    m = np.zeros((4, 4))
    fa = fr.diag_indices_from(m)
    na = np.diag_indices_from(m)
    for f, n in zip(fa, na):
        assert np.array_equal(f, n)


def test_tril_indices_from():
    m = np.zeros((3, 4))
    fa = fr.tril_indices_from(m)
    na = np.tril_indices_from(m)
    for f, n in zip(fa, na):
        assert np.array_equal(f, n)


def test_triu_indices_from():
    m = np.zeros((4, 3))
    fa = fr.triu_indices_from(m)
    na = np.triu_indices_from(m)
    for f, n in zip(fa, na):
        assert np.array_equal(f, n)


def test_put_along_axis_inplace():
    a = np.zeros((2, 3))
    ref = np.zeros((2, 3))
    idx = np.array([[0], [1]])
    fr.put_along_axis(a, idx, 99, axis=1)
    np.put_along_axis(ref, idx, 99, axis=1)
    assert np.array_equal(a, ref)


def test_asarray_chkfinite_ok():
    assert np.array_equal(
        fr.asarray_chkfinite([1.0, 2.0, 3.0]),
        np.asarray_chkfinite([1.0, 2.0, 3.0]),
    )


def test_asarray_chkfinite_raises_on_nan():
    with pytest.raises(ValueError):
        fr.asarray_chkfinite([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        np.asarray_chkfinite([1.0, np.nan, 3.0])


def test_asarray_chkfinite_raises_on_inf():
    with pytest.raises(ValueError):
        fr.asarray_chkfinite([1.0, np.inf, 3.0])


def test_may_share_memory():
    a = np.arange(5)
    assert fr.may_share_memory(a, a) == np.may_share_memory(a, a)
    b = np.arange(5)
    assert fr.may_share_memory(a, b) == np.may_share_memory(a, b)


def test_shares_memory():
    a = np.arange(5)
    assert fr.shares_memory(a, a) == np.shares_memory(a, a)
    b = np.arange(5)
    assert fr.shares_memory(a, b) == np.shares_memory(a, b)


def test_iterable():
    for obj in ([1, 2], (1,), "abc", 5, 3.14, None):
        assert fr.iterable(obj) == np.iterable(obj)


def test_unique_all():
    x = [[1, 1], [2, 3]]
    fa = fr.unique_all(x)
    na = np.unique_all(x)
    assert np.array_equal(fa.values, na.values)
    assert np.array_equal(fa.indices, na.indices)
    assert np.array_equal(fa.inverse_indices, na.inverse_indices)
    assert np.array_equal(fa.counts, na.counts)
    assert fa.inverse_indices.shape == np.asarray(x).shape


def test_unique_counts():
    x = [1, 1, 2, 3, 3, 3]
    fa = fr.unique_counts(x)
    na = np.unique_counts(x)
    assert np.array_equal(fa.values, na.values)
    assert np.array_equal(fa.counts, na.counts)


def test_unique_inverse():
    x = [[1, 1], [2, 2]]
    fa = fr.unique_inverse(x)
    na = np.unique_inverse(x)
    assert np.array_equal(fa.values, na.values)
    assert np.array_equal(fa.inverse_indices, na.inverse_indices)
    assert fa.inverse_indices.shape == np.asarray(x).shape


def test_unique_values():
    x = [3, 1, 1, 2, 3]
    # The array-API leaves unique_values ordering unspecified; compare as
    # sets (numpy 2.4 returns them unsorted; ferray returns them sorted).
    assert set(fr.unique_values(x).tolist()) == set(np.unique_values(x).tolist())
