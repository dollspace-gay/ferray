"""Phase-1 parity tests for the ferray indexing surface."""

import numpy as np

import ferray


def test_indices_basic_2d():
    a = ferray.indices([3, 4])
    expected = np.indices([3, 4])
    # ferray returns int64; numpy default is intp (int64 on 64-bit Linux)
    assert a.shape == expected.shape
    assert np.array_equal(a, expected)


def test_indices_3d():
    a = ferray.indices([2, 3, 4])
    assert a.shape == (3, 2, 3, 4)
    assert np.array_equal(a, np.indices([2, 3, 4]))


def test_diag_indices_default_ndim():
    a = ferray.diag_indices(5)
    expected = np.diag_indices(5)
    assert len(a) == len(expected)
    for got, exp in zip(a, expected):
        assert np.array_equal(got, exp)


def test_diag_indices_3d():
    a = ferray.diag_indices(4, ndim=3)
    expected = np.diag_indices(4, ndim=3)
    for got, exp in zip(a, expected):
        assert np.array_equal(got, exp)


def test_tril_indices_default():
    rows, cols = ferray.tril_indices(4)
    e_rows, e_cols = np.tril_indices(4)
    assert np.array_equal(rows, e_rows)
    assert np.array_equal(cols, e_cols)


def test_tril_indices_with_k_and_m():
    rows, cols = ferray.tril_indices(4, k=1, m=5)
    e_rows, e_cols = np.tril_indices(4, k=1, m=5)
    assert np.array_equal(rows, e_rows)
    assert np.array_equal(cols, e_cols)


def test_triu_indices_default():
    rows, cols = ferray.triu_indices(4)
    e_rows, e_cols = np.triu_indices(4)
    assert np.array_equal(rows, e_rows)
    assert np.array_equal(cols, e_cols)


def test_mask_indices_tril():
    mi = ferray.mask_indices(5, "tril")
    expected = np.mask_indices(5, np.tril)
    # numpy returns a tuple of two flat-index arrays; ferray returns a
    # single flat-indices array. Convert numpy's to flat for comparison.
    expected_flat = np.ravel_multi_index(expected, (5, 5))
    assert np.array_equal(mi, expected_flat)


def test_ravel_multi_index_basic():
    rows = [0, 1, 2, 3]
    cols = [0, 1, 2, 3]
    flat = ferray.ravel_multi_index([rows, cols], [4, 4])
    expected = np.ravel_multi_index([rows, cols], (4, 4))
    assert np.array_equal(flat, expected)


def test_unravel_index_round_trips():
    flat_idx = [0, 5, 11]
    coords = ferray.unravel_index(flat_idx, [3, 4])
    expected = np.unravel_index(flat_idx, (3, 4))
    for got, exp in zip(coords, expected):
        assert np.array_equal(got, exp)


def test_flatnonzero_matches():
    src = np.array([0, 1, 0, 2, 0, 3])
    assert np.array_equal(ferray.flatnonzero(src), np.flatnonzero(src))


def test_flatnonzero_2d():
    src = np.array([[0, 1, 0], [2, 0, 3]])
    assert np.array_equal(ferray.flatnonzero(src), np.flatnonzero(src))


def test_nonzero_2d():
    src = np.array([[0, 1, 0], [2, 0, 3]])
    rows, cols = ferray.nonzero(src)
    e_rows, e_cols = np.nonzero(src)
    assert np.array_equal(rows, e_rows)
    assert np.array_equal(cols, e_cols)


def test_argwhere_2d():
    src = np.array([[0, 1, 0], [2, 0, 3]])
    a = ferray.argwhere(src)
    expected = np.argwhere(src)
    assert a.shape == expected.shape
    assert np.array_equal(a, expected)


def test_argwhere_no_nonzero():
    src = np.zeros((3, 3))
    a = ferray.argwhere(src)
    assert a.shape == (0, 2)
