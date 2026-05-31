"""N-D + axis + kth-sequence coverage for `partition` / `argpartition` (#963).

The prior bindings accepted only a 1-D array and a single integer `kth`, so
`fr.partition([[3,1,2]], 1)` raised
`TypeError: 'ndarray' object is not an instance of 'ndarray'`. numpy supports
N-D input with `axis=-1` default (partition along the last axis), `axis=None`
flatten, negative axes, and `kth` as int OR sequence.

Every expected value is derived LIVE from numpy (R-CHAR-3) — the assertions
compare `fr.<fn>(...)` against `np.<fn>(...)` on the same input, never against a
literal copied from the ferray side. A partition is only defined up to the
ordering of the two sides of each pivot, so the structural invariant
(`np.sort == np.sort(partition(...))` along the axis, and the pivot element in
final position) is checked rather than exact element identity — which is exactly
what numpy guarantees and how numpy's own test-suite checks partition.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    return np.asarray(x)


def _assert_partitioned(got, ref_input, kth, axis):
    """Structural partition invariant, axis-aware, derived from numpy.

    The pivot value(s) at `kth` must equal numpy's, and the full sort of the
    result along `axis` must equal the full sort of the input along `axis`
    (a partition is a permutation per lane that fixes the kth position).
    """
    got = _np(got)
    ref = _np(ref_input)
    # Same shape/dtype as numpy's own partition.
    np_part = np.partition(ref, kth, axis=axis)
    assert got.shape == np_part.shape
    assert got.dtype == np_part.dtype
    # Sorting both along axis must agree (permutation-per-lane invariant).
    if axis is None:
        np.testing.assert_array_equal(np.sort(got, axis=None), np.sort(ref, axis=None))
    else:
        np.testing.assert_array_equal(np.sort(got, axis=axis), np.sort(ref, axis=axis))
    # The element(s) at `kth` along `axis` must match numpy's partition exactly.
    kths = [kth] if np.isscalar(kth) else list(kth)
    for k in kths:
        if axis is None:
            np.testing.assert_array_equal(
                np.take(got, k), np.take(np_part, k)
            )
        else:
            np.testing.assert_array_equal(
                np.take(got, k, axis=axis), np.take(np_part, k, axis=axis)
            )


# ---------------------------------------------------------------------------
# partition: the divergence that #963 fixes — 2-D input no longer a TypeError.
# ---------------------------------------------------------------------------


def test_partition_2d_default_axis_last():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.partition(a, 1)
    _assert_partitioned(got, a, 1, axis=-1)


def test_partition_2d_axis0():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.partition(a, 1, axis=0)
    _assert_partitioned(got, a, 1, axis=0)


def test_partition_2d_axis_none_flatten():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.partition(a, 1, axis=None)
    _assert_partitioned(got, a, 1, axis=None)


def test_partition_single_row_2d():
    # The exact divergence reproduction from the dispatch.
    a = [[3, 1, 2]]
    got = fr.partition(a, 1)
    _assert_partitioned(got, a, 1, axis=-1)


def test_partition_3d_default_axis():
    a = np.arange(24)[::-1].reshape(2, 3, 4)
    got = fr.partition(a, 2)
    _assert_partitioned(got, a, 2, axis=-1)


def test_partition_3d_axis1():
    a = np.arange(24)[::-1].reshape(2, 3, 4)
    got = fr.partition(a, 1, axis=1)
    _assert_partitioned(got, a, 1, axis=1)


def test_partition_negative_axis():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.partition(a, 1, axis=-2)
    _assert_partitioned(got, a, 1, axis=-2)


def test_partition_negative_kth():
    a = [5, 2, 8, 1, 9]
    got = fr.partition(a, -2)
    # numpy normalises negative kth to n+kth; compare on that normalised pivot.
    _assert_partitioned(got, a, len(a) - 2, axis=-1)


# ---------------------------------------------------------------------------
# kth as a sequence (multiple pivots).
# ---------------------------------------------------------------------------


def test_partition_kth_sequence_1d():
    a = [5, 2, 8, 1, 9, 3, 7]
    got = fr.partition(a, [1, 3])
    _assert_partitioned(got, a, [1, 3], axis=-1)


def test_partition_kth_sequence_2d():
    a = [[3, 1, 2, 7], [6, 4, 5, 0]]
    got = fr.partition(a, [0, 2])
    _assert_partitioned(got, a, [0, 2], axis=-1)


# ---------------------------------------------------------------------------
# dtype preservation across int / float / complex.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["int32", "int64", "float32", "float64"])
def test_partition_dtype_preserved(dt):
    a = np.array([[3, 1, 2], [6, 4, 5]], dtype=dt)
    got = _np(fr.partition(a, 1))
    assert got.dtype == np.partition(a, 1).dtype == np.dtype(dt)
    _assert_partitioned(got, a, 1, axis=-1)


def test_partition_complex_2d():
    a = np.array([[3 + 1j, 1 + 2j, 1 + 1j], [2 + 0j, 0 + 0j, 5 - 1j]])
    got = _np(fr.partition(a, 1))
    assert got.dtype == np.partition(a, 1).dtype
    _assert_partitioned(got, a, 1, axis=-1)


# ---------------------------------------------------------------------------
# 1-D real path must remain unregressed (the pre-#963 behaviour).
# ---------------------------------------------------------------------------


def test_partition_1d_real_unchanged():
    a = [5, 2, 8, 1, 9]
    got = _np(fr.partition(a, 2))
    np.testing.assert_array_equal(got, np.partition(a, 2))


def test_partition_kth_out_of_bounds_raises():
    # numpy raises ValueError for an out-of-range kth; the binding must too.
    with pytest.raises((ValueError, IndexError)):
        fr.partition([1, 2, 3], 5)


# ---------------------------------------------------------------------------
# argpartition: same N-D / axis / kth-sequence surface, int64 index output.
# ---------------------------------------------------------------------------


def _assert_argpartitioned(got_idx, ref_input, kth, axis):
    """The indices, applied along `axis`, must reproduce a valid partition."""
    got_idx = _np(got_idx)
    ref = _np(ref_input)
    np_idx = np.argpartition(ref, kth, axis=axis)
    assert got_idx.shape == np_idx.shape
    assert got_idx.dtype == np_idx.dtype  # int64 on this platform
    # Reconstruct the partitioned array via take_along_axis and check invariant.
    if axis is None:
        got_part = ref.reshape(-1)[got_idx]
        _assert_partitioned(got_part, ref.reshape(-1), kth, axis=-1)
    else:
        got_part = np.take_along_axis(ref, got_idx, axis=axis)
        _assert_partitioned(got_part, ref, kth, axis=axis)


def test_argpartition_2d_default_axis():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.argpartition(a, 1)
    _assert_argpartitioned(got, a, 1, axis=-1)


def test_argpartition_2d_axis0():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.argpartition(a, 1, axis=0)
    _assert_argpartitioned(got, a, 1, axis=0)


def test_argpartition_axis_none_flatten():
    a = [[3, 1, 2], [6, 4, 5]]
    got = fr.argpartition(a, 1, axis=None)
    _assert_argpartitioned(got, a, 1, axis=None)


def test_argpartition_3d_axis1():
    a = np.arange(24)[::-1].reshape(2, 3, 4)
    got = fr.argpartition(a, 1, axis=1)
    _assert_argpartitioned(got, a, 1, axis=1)


def test_argpartition_kth_sequence():
    a = [5, 2, 8, 1, 9, 3, 7]
    got = fr.argpartition(a, [1, 3])
    _assert_argpartitioned(got, a, [1, 3], axis=-1)


def test_argpartition_dtype_is_int64():
    a = np.array([[3, 1, 2], [6, 4, 5]], dtype="float32")
    got = _np(fr.argpartition(a, 1))
    assert got.dtype == np.argpartition(a, 1).dtype


def test_argpartition_1d_real_unchanged():
    a = [5, 2, 8, 1, 9]
    got = _np(fr.argpartition(a, 2))
    _assert_argpartitioned(got, a, 2, axis=-1)
