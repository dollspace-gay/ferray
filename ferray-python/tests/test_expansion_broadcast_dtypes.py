"""Divergence #962 — `fr.broadcast_to` (and `fr.broadcast_arrays`) must preserve
every dtype numpy preserves.

`np.broadcast_to` / `np.broadcast_arrays` are pure shape/view ops: they change
only the shape, never the dtype or values, for EVERY dtype (datetime64,
timedelta64, fixed-width string `<U`/`<S`, float16, complex, object, structured).
The prior ferray binding dispatched only over the real+complex `Element` set and
raised ``TypeError: unsupported dtype`` for the rest.

R-CHAR-3: every expected value comes from a live numpy call (the oracle), never
copied from the ferray side.
"""

import ferray as fr
import numpy as np
import pytest


# (array, target-shape) pairs spanning the dtypes the real-only dispatch rejects.
_CASES = [
    (np.array(["2020-01-01", "2021-06-15"], dtype="datetime64[D]"), (3, 2)),
    (np.array([1, 2], dtype="timedelta64[s]"), (3, 2)),
    (np.array(["ab", "cde"], dtype="<U3"), (3, 2)),
    (np.array([b"ab", b"cdef"], dtype="<S4"), (3, 2)),
    (np.array([1.5, -2.25], dtype="float16"), (3, 2)),
    (np.array([1 + 2j, -3 - 4j], dtype="complex64"), (3, 2)),
    (np.array([1 + 2j, -3 - 4j], dtype="complex128"), (3, 2)),
    (np.array([{"a": 1}, (2, 3)], dtype=object), (3, 2)),
]


@pytest.mark.parametrize("arr,shape", _CASES)
def test_broadcast_to_preserves_dtype(arr, shape):
    want = np.broadcast_to(arr, shape)
    got = fr.broadcast_to(arr, shape)
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    assert got.shape == want.shape == tuple(shape)
    # Values identical (object arrays compared elementwise via array_equal).
    assert np.array_equal(np.asarray(got), want)
    # numpy returns a read-only view (`_broadcast_to(..., readonly=True)`).
    assert got.flags.writeable is False


@pytest.mark.parametrize("arr,shape", _CASES)
def test_broadcast_arrays_preserves_dtype(arr, shape):
    other = np.ones(shape, dtype="int64")
    want = np.broadcast_arrays(arr, other)
    got = fr.broadcast_arrays(arr, other)
    assert isinstance(got, tuple)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g.dtype == w.dtype, (g.dtype, w.dtype)
        assert g.shape == w.shape == tuple(shape)
        assert np.array_equal(np.asarray(g), w)


def test_broadcast_to_real_path_unchanged():
    """Real/numeric broadcast_to must be byte-for-byte unchanged."""
    arr = np.array([1, 2, 3], dtype="int64")
    want = np.broadcast_to(arr, (4, 3))
    got = fr.broadcast_to(arr, (4, 3))
    assert got.dtype == want.dtype == np.dtype("int64")
    assert got.shape == want.shape
    assert np.array_equal(np.asarray(got), want)
    assert got.flags.writeable is False
