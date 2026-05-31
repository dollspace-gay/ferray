"""Phase-1 parity tests for the ferray manipulation surface."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Shape transforms
# ---------------------------------------------------------------------------


def test_reshape_2d_to_1d():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.reshape(src, 12), np.reshape(src, 12))


def test_reshape_to_higher_rank():
    src = np.arange(24)
    assert np.array_equal(
        ferray.reshape(src, (2, 3, 4)), np.reshape(src, (2, 3, 4))
    )


def test_ravel_matches_numpy():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.ravel(src), np.ravel(src))


def test_flatten_matches_numpy():
    src = np.arange(8).reshape(2, 2, 2)
    assert np.array_equal(ferray.flatten(src), src.flatten())


def test_squeeze_all_singletons():
    src = np.zeros((1, 3, 1, 4))
    assert ferray.squeeze(src).shape == np.squeeze(src).shape


def test_squeeze_specific_axis():
    src = np.zeros((1, 3, 1, 4))
    assert ferray.squeeze(src, axis=0).shape == np.squeeze(src, axis=0).shape


def test_expand_dims_inserts_axis():
    src = np.array([1, 2, 3])
    a = ferray.expand_dims(src, 0)
    assert a.shape == (1, 3)


def test_broadcast_to_shapes():
    src = np.array([1, 2, 3])
    a = ferray.broadcast_to(src, (4, 3))
    assert np.array_equal(a, np.broadcast_to(src, (4, 3)))


# ---------------------------------------------------------------------------
# Axis transforms
# ---------------------------------------------------------------------------


def test_transpose_default_reverses_axes():
    src = np.arange(24).reshape(2, 3, 4)
    assert np.array_equal(ferray.transpose(src), np.transpose(src))


def test_transpose_with_explicit_axes():
    src = np.arange(24).reshape(2, 3, 4)
    a = ferray.transpose(src, axes=[2, 0, 1])
    assert np.array_equal(a, np.transpose(src, axes=(2, 0, 1)))


def test_swapaxes_matches_numpy():
    src = np.arange(24).reshape(2, 3, 4)
    a = ferray.swapaxes(src, 0, 2)
    assert np.array_equal(a, np.swapaxes(src, 0, 2))


def test_moveaxis_matches_numpy():
    src = np.arange(24).reshape(2, 3, 4)
    a = ferray.moveaxis(src, 0, 2)
    assert np.array_equal(a, np.moveaxis(src, 0, 2))


def test_rollaxis_matches_numpy():
    src = np.arange(24).reshape(2, 3, 4)
    a = ferray.rollaxis(src, 2, 0)
    assert np.array_equal(a, np.rollaxis(src, 2, 0))


# ---------------------------------------------------------------------------
# Flips and rotations
# ---------------------------------------------------------------------------


def test_flip_axis_0():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.flip(src, 0), np.flip(src, axis=0))


def test_flip_axis_1():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.flip(src, 1), np.flip(src, axis=1))


def test_fliplr_matches():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.fliplr(src), np.fliplr(src))


def test_flipud_matches():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.flipud(src), np.flipud(src))


@pytest.mark.parametrize("k", [-2, -1, 0, 1, 2, 3, 5])
def test_rot90_matches(k):
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(ferray.rot90(src, k=k), np.rot90(src, k=k))


def test_roll_no_axis():
    src = np.arange(10)
    assert np.array_equal(ferray.roll(src, 3), np.roll(src, 3))


def test_roll_with_axis():
    src = np.arange(12).reshape(3, 4)
    assert np.array_equal(
        ferray.roll(src, 1, axis=1), np.roll(src, 1, axis=1)
    )


# ---------------------------------------------------------------------------
# atleast_*d
# ---------------------------------------------------------------------------


def test_atleast_1d_promotes_scalar():
    a = ferray.atleast_1d(np.float64(3.5))
    assert a.shape == (1,)


def test_atleast_2d_promotes_1d():
    src = np.array([1, 2, 3])
    a = ferray.atleast_2d(src)
    assert a.shape == (1, 3)


def test_atleast_3d_promotes_2d():
    src = np.array([[1, 2], [3, 4]])
    a = ferray.atleast_3d(src)
    assert a.ndim == 3


# ---------------------------------------------------------------------------
# Triangular and diagonal
# ---------------------------------------------------------------------------


def test_tril_matches_numpy():
    src = np.arange(16).reshape(4, 4)
    for k in (-1, 0, 1):
        assert np.array_equal(ferray.tril(src, k=k), np.tril(src, k=k))


def test_triu_matches_numpy():
    src = np.arange(16).reshape(4, 4)
    for k in (-1, 0, 1):
        assert np.array_equal(ferray.triu(src, k=k), np.triu(src, k=k))


def test_diag_extracts_main_diagonal():
    src = np.arange(16).reshape(4, 4)
    a = ferray.diag(src)
    assert np.array_equal(a, np.diag(src))


def test_diag_builds_from_1d():
    src = np.array([1, 2, 3])
    a = ferray.diag(src)
    assert np.array_equal(a, np.diag(src))


def test_diagflat_from_2d():
    src = np.array([[1, 2], [3, 4]])
    a = ferray.diagflat(src)
    assert np.array_equal(a, np.diagflat(src))


# ---------------------------------------------------------------------------
# Concatenation / stacking
# ---------------------------------------------------------------------------


def test_concatenate_axis_0():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    assert np.array_equal(
        ferray.concatenate([a, b], axis=0), np.concatenate([a, b], axis=0)
    )


def test_concatenate_axis_1():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])
    assert np.array_equal(
        ferray.concatenate([a, b], axis=1), np.concatenate([a, b], axis=1)
    )


def test_stack_creates_new_axis():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.array_equal(ferray.stack([a, b]), np.stack([a, b]))


def test_stack_with_axis():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    assert np.array_equal(
        ferray.stack([a, b], axis=1), np.stack([a, b], axis=1)
    )


def test_vstack_matches_numpy():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
    assert np.array_equal(ferray.vstack([a, b]), np.vstack([a, b]))


def test_hstack_matches_numpy_2d():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5], [6]])
    assert np.array_equal(ferray.hstack([a, b]), np.hstack([a, b]))


def test_dstack_matches_numpy():
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    assert np.array_equal(ferray.dstack([a, b]), np.dstack([a, b]))


def test_concatenate_promotes_mixed_dtypes_to_first():
    a = np.array([1, 2], dtype=np.float64)
    b = np.array([3, 4], dtype=np.int32)
    out = ferray.concatenate([a, b])
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, np.array([1.0, 2.0, 3.0, 4.0]))
