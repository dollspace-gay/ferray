"""Binding-gap cluster #969: histogram2d / histogramdd / outer / kron.

Every expected value is derived LIVE from numpy 2.4.x (the installed oracle),
never literal-copied from the ferray side (R-CHAR-3). ferray (`fr`) must match
numpy (`np`) on:

  1. histogram2d — bins int / [int,int] / [array,array], default bins=10,
     float64 counts, edges contract.
  2. histogramdd — bins optional (default 10), (hist, edges) contract.
  3. outer — int / float / complex inputs, dtype preserved.
  4. kron 1-D complex — value + complex dtype; real outer/kron unchanged.
"""

import numpy as np
import pytest

import ferray as fr


# --------------------------------------------------------------------------- #
# histogram2d
# --------------------------------------------------------------------------- #
def test_histogram2d_default_bins_float64_counts():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    h_np, ex_np, ey_np = np.histogram2d(x, y)
    h_fr, ex_fr, ey_fr = fr.histogram2d(x, y)
    # numpy default is bins=10 -> 10x10 float64 counts.
    assert h_np.shape == (10, 10)
    assert h_np.dtype == np.float64
    assert h_fr.shape == h_np.shape
    assert h_fr.dtype == h_np.dtype
    np.testing.assert_array_equal(h_fr, h_np)
    np.testing.assert_allclose(ex_fr, ex_np)
    np.testing.assert_allclose(ey_fr, ey_np)


def test_histogram2d_int_bins():
    x = [1.0, 2.0, 3.0]
    y = [1.0, 2.0, 3.0]
    h_np, ex_np, ey_np = np.histogram2d(x, y, bins=2)
    h_fr, ex_fr, ey_fr = fr.histogram2d(x, y, bins=2)
    assert h_np.dtype == np.float64
    assert h_fr.dtype == h_np.dtype
    np.testing.assert_array_equal(h_fr, h_np)
    np.testing.assert_allclose(ex_fr, ex_np)
    np.testing.assert_allclose(ey_fr, ey_np)


def test_histogram2d_seq_bins():
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 2.0, 3.0, 4.0]
    h_np, ex_np, ey_np = np.histogram2d(x, y, bins=[2, 3])
    h_fr, ex_fr, ey_fr = fr.histogram2d(x, y, bins=[2, 3])
    assert h_np.shape == (2, 3)
    np.testing.assert_array_equal(h_fr, h_np)
    np.testing.assert_allclose(ex_fr, ex_np)
    np.testing.assert_allclose(ey_fr, ey_np)


def test_histogram2d_array_bins():
    x = [0.5, 1.5, 2.5]
    y = [0.5, 1.5, 2.5]
    xedges = np.array([0.0, 1.0, 2.0, 3.0])
    yedges = np.array([0.0, 1.5, 3.0])
    h_np, ex_np, ey_np = np.histogram2d(x, y, bins=[xedges, yedges])
    h_fr, ex_fr, ey_fr = fr.histogram2d(x, y, bins=[xedges, yedges])
    np.testing.assert_array_equal(h_fr, h_np)
    np.testing.assert_allclose(ex_fr, ex_np)
    np.testing.assert_allclose(ey_fr, ey_np)


def test_histogram2d_density():
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 3.0, 4.0]
    h_np, _, _ = np.histogram2d(x, y, bins=2, density=True)
    h_fr, _, _ = fr.histogram2d(x, y, bins=2, density=True)
    np.testing.assert_allclose(h_fr, h_np)


# --------------------------------------------------------------------------- #
# histogramdd
# --------------------------------------------------------------------------- #
def test_histogramdd_default_bins():
    sample = [[1.0, 2.0], [3.0, 4.0]]
    h_np, edges_np = np.histogramdd(sample)
    h_fr, edges_fr = fr.histogramdd(sample)
    assert h_np.shape == (10, 10)
    assert h_np.dtype == np.float64
    assert h_fr.shape == h_np.shape
    assert h_fr.dtype == h_np.dtype
    np.testing.assert_array_equal(h_fr, h_np)
    assert len(edges_fr) == len(edges_np)
    for ef, en in zip(edges_fr, edges_np):
        np.testing.assert_allclose(ef, en)


def test_histogramdd_int_bins():
    # 2-D ndarray sample: N=3 points, D=2 dims (an unambiguous shape so numpy
    # bins per-dimension rather than treating a ragged list as D-dimensional).
    sample = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    h_np, edges_np = np.histogramdd(sample, bins=3)
    h_fr, edges_fr = fr.histogramdd(sample, bins=3)
    assert h_np.shape == (3, 3)
    np.testing.assert_array_equal(h_fr, h_np)
    for ef, en in zip(edges_fr, edges_np):
        np.testing.assert_allclose(ef, en)


def test_histogramdd_seq_bins():
    sample = np.array([[1.0, 2.0], [3.0, 4.0], [2.0, 3.0]])
    h_np, edges_np = np.histogramdd(sample, bins=[2, 3])
    h_fr, edges_fr = fr.histogramdd(sample, bins=[2, 3])
    assert h_np.shape == (2, 3)
    np.testing.assert_array_equal(h_fr, h_np)
    for ef, en in zip(edges_fr, edges_np):
        np.testing.assert_allclose(ef, en)


# --------------------------------------------------------------------------- #
# outer
# --------------------------------------------------------------------------- #
def test_outer_int_preserves_dtype():
    r_np = np.outer([1, 2], [3, 4])
    r_fr = fr.outer([1, 2], [3, 4])
    assert r_np.dtype == np.int64
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_array_equal(r_fr, r_np)


def test_outer_int8_preserves_dtype():
    a = np.array([1, 2, 3], dtype=np.int8)
    b = np.array([4, 5], dtype=np.int8)
    r_np = np.outer(a, b)
    r_fr = fr.outer(a, b)
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_array_equal(r_fr, r_np)


def test_outer_uint_preserves_dtype():
    a = np.array([1, 2], dtype=np.uint32)
    b = np.array([3, 4], dtype=np.uint32)
    r_np = np.outer(a, b)
    r_fr = fr.outer(a, b)
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_array_equal(r_fr, r_np)


def test_outer_float_unchanged():
    r_np = np.outer([1.0, 2.0], [3.0, 4.0])
    r_fr = fr.outer([1.0, 2.0], [3.0, 4.0])
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_allclose(r_fr, r_np)


def test_outer_complex_unchanged():
    a = [1 + 1j, 2]
    b = [1, 1j]
    r_np = np.outer(a, b)
    r_fr = fr.outer(a, b)
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_allclose(r_fr, r_np)


# --------------------------------------------------------------------------- #
# kron
# --------------------------------------------------------------------------- #
def test_kron_1d_complex():
    a = [1 + 1j, 2]
    b = [1, 1j]
    r_np = np.kron(a, b)
    r_fr = fr.kron(a, b)
    assert r_np.dtype == np.complex128
    assert r_fr.dtype == r_np.dtype
    assert r_fr.shape == r_np.shape
    np.testing.assert_allclose(r_fr, r_np)


def test_kron_1d_complex64():
    a = np.array([1 + 1j, 2], dtype=np.complex64)
    b = np.array([1, 1j], dtype=np.complex64)
    r_np = np.kron(a, b)
    r_fr = fr.kron(a, b)
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_allclose(r_fr, r_np)


def test_kron_1d_real_unchanged():
    # The real (float) kron path is the genuinely-unchanged route; #969 only
    # touches the complex 1-D branch. (Integer kron is a separate, pre-existing
    # gap — `fr.kron([1,2],[3,4])` still raises — filed as spillover.)
    r_np = np.kron([1.0, 2.0], [3.0, 4.0])
    r_fr = fr.kron([1.0, 2.0], [3.0, 4.0])
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_allclose(r_fr, r_np)


def test_kron_2d_complex_unchanged():
    a = [[1 + 1j, 2]]
    b = [[1, 1j]]
    r_np = np.kron(a, b)
    r_fr = fr.kron(a, b)
    assert r_fr.dtype == r_np.dtype
    np.testing.assert_allclose(r_fr, r_np)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
