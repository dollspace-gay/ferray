"""Adversarial divergence tests for the ferray indexing surface.

ACToR-CRITIC (re-audit, authored from scratch) of
``ferray-python/src/indexing.rs`` against the numpy 2.4.5 live oracle.
Every test below is EXPECTED TO FAIL on current ferray; the divergence
it pins is real. Each docstring cites the upstream numpy source the
target violates (file:line, read this iteration). Expected values are
produced by LIVE numpy calls (R-CHAR-3), never literal-copied from the
ferray side.

Run:
    PYTHONPATH=python python3 -m pytest tests/divergence_indexing.py -q
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# take — default axis is None => operate on the FLATTENED array
# ---------------------------------------------------------------------------
def test_take_default_axis_flattens():
    """numpy/_core/fromnumeric.py:107 ``def take(a, indices, axis=None, ...)``
    and :135-137 "axis : int, optional ... By default, the flattened input
    array is used." With a 2-D source and no axis, numpy gathers from the
    flattened array. ferray's binding hard-defaults ``axis=0`` (signature
    ``(a, indices, axis = 0)`` in `take`, indexing.rs), so it indexes axis 0
    and an index >= nrows raises instead of flattening.
    """
    m = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.take(m, [0, 4])  # flattened -> [1, 5]
    got = fr.take(m, [0, 4])
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# take — out-of-bounds index raises IndexError (numpy), not ValueError
# ---------------------------------------------------------------------------
def test_take_out_of_bounds_raises_indexerror():
    """numpy/_core/fromnumeric.py:145 mode='raise' (default) "raise an error".
    numpy raises ``IndexError`` for an out-of-bounds take index. A plain
    ValueError is NOT an IndexError in numpy's hierarchy, so a caller catching
    IndexError breaks. ferray's `take` raises ``ValueError``.
    """
    a = np.array([10, 20, 30, 40])
    with pytest.raises(IndexError):
        np.take(a, [5])
    with pytest.raises(IndexError):
        fr.take(a, [5])


# ---------------------------------------------------------------------------
# take — axis out of bounds raises AxisError (not plain ValueError)
# ---------------------------------------------------------------------------
def test_take_bad_axis_raises_axiserror():
    """numpy/_core/fromnumeric.py:200 ``take`` dispatches to ndarray.take,
    which normalizes axis and raises ``numpy.exceptions.AxisError`` for an
    out-of-range axis (R-DEV-2). ferray's `take` raises a plain ``ValueError``,
    which is NOT an AxisError instance.
    """
    a = np.array([10, 20, 30, 40])
    with pytest.raises(np.exceptions.AxisError):
        np.take(a, [0], axis=5)
    with pytest.raises(np.exceptions.AxisError):
        fr.take(a, [0], axis=5)


# ---------------------------------------------------------------------------
# take — mode kwarg ('raise'/'wrap'/'clip') is part of the public signature
# ---------------------------------------------------------------------------
def test_take_mode_clip():
    """numpy/_core/fromnumeric.py:107 ``def take(a, indices, axis=None,
    out=None, mode='raise')`` and :142-151 mode {'raise','wrap','clip'};
    'clip' replaces too-large indices with the last index.
    ``np.take(a, [5], mode='clip')`` -> [a[-1]]. ferray's `take` binding
    exposes no `mode` kwarg, so this raises TypeError.
    """
    a = np.array([10, 20, 30])
    expected = np.take(a, [5], mode="clip")  # -> [30]
    got = fr.take(a, [5], mode="clip")
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


def test_take_mode_wrap():
    """numpy/_core/fromnumeric.py:146 'wrap' wraps OOB indices around.
    ``np.take([10,20,30],[4],mode='wrap')`` -> [a[4 % 3]] = [20]. ferray
    exposes no `mode` kwarg -> TypeError.
    """
    a = np.array([10, 20, 30])
    expected = np.take(a, [4], mode="wrap")  # -> [20]
    got = fr.take(a, [4], mode="wrap")
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# take — multi-dimensional index array preserves the index shape
# ---------------------------------------------------------------------------
def test_take_2d_index_shape():
    """numpy/_core/fromnumeric.py:194-198 "If `indices` is not one
    dimensional, the output also has these dimensions." ``np.take([4,3,5,7,
    6,8], [[0,1],[2,3]])`` returns a (2,2) array. ferray's `take` binding
    accepts only a flat ``Vec<isize>``, so a 2-D index list is a TypeError and
    the output-shape contract is unmet.
    """
    a = np.array([4, 3, 5, 7, 6, 8])
    expected = np.take(a, [[0, 1], [2, 3]])
    got = fr.take(a, [[0, 1], [2, 3]])
    assert got.shape == expected.shape
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# take — scalar index returns a 0-d result (scalar), not a 1-element array
# ---------------------------------------------------------------------------
def test_take_scalar_index():
    """numpy/_core/fromnumeric.py:132-134 "Also allow scalars for indices."
    ``np.take([10,20,30], 2)`` returns the 0-d scalar 30 (shape ``()``).
    ferray's `take` binding requires a Sequence for `indices` (``Vec<isize>``),
    so a scalar int raises TypeError.
    """
    a = np.array([10, 20, 30])
    expected = np.take(a, 2)
    got = fr.take(a, 2)
    assert np.shape(got) == np.shape(expected) == ()
    assert got == expected


# ---------------------------------------------------------------------------
# take_along_axis — element-wise N-d index array along an axis
# ---------------------------------------------------------------------------
def test_take_along_axis_2d():
    """numpy/lib/_shape_base_impl.py take_along_axis: the index array has the
    same ndim as `a` and is matched element-wise along `axis`. Sorting each row
    via ``idx = argsort(a, axis=1)`` then ``take_along_axis(a, idx, axis=1)``
    yields the per-row sorted array. ferray's `take_along_axis` binding accepts
    only a flat ``Vec<isize>``, so a 2-D index array raises TypeError.
    """
    a = np.array([[10, 30, 20], [60, 40, 50]])
    idx = np.argsort(a, axis=1)
    expected = np.take_along_axis(a, idx, axis=1)  # [[10,20,30],[40,50,60]]
    got = fr.take_along_axis(a, idx, axis=1)
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# compress — default axis=None operates on the FLATTENED array
# ---------------------------------------------------------------------------
def test_compress_default_axis_flattens():
    """numpy/_core/fromnumeric.py:2138 ``def compress(condition, a, axis=None,
    out=None)``; "axis : int, optional ... If None (default), work on the
    flattened array." ``np.compress([False, True], m)`` selects the flat
    element at index 1. ferray's `compress` hard-defaults ``axis=0`` (signature
    ``(condition, a, axis = 0)``), so it returns a row-slice instead.
    """
    m = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.compress([False, True], m)  # flattened -> [2]
    got = fr.compress([False, True], m)
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# putmask — values cycle by GLOBAL flat position n (values[n % len])
# ---------------------------------------------------------------------------
def test_putmask_values_cycle_by_flat_index():
    """numpy putmask sets ``a.flat[n] = values[n % len(values)]`` for each n
    where ``mask.flat[n]`` is True. Doc example on ``arange(5)``:
    ``np.putmask(x, x>1, [-33,-44])`` -> [0, 1, -33, -44, -33] (masked
    positions 2,3,4 take values[2],values[3],values[4] = -33,-44,-33 by GLOBAL
    flat index, NOT mask-relative). ferray's `putmask` requires
    ``len(values)`` to equal the array size, raising ValueError otherwise.
    """
    x = np.arange(5)
    xc = x.copy()
    np.putmask(xc, x > 1, [-33, -44])  # live oracle: [0,1,-33,-44,-33]
    got = fr.putmask(x.astype(np.int64), (x > 1), [-33.0, -44.0])
    assert np.array_equal(got, xc), f"{got!r} != {xc!r}"


# ---------------------------------------------------------------------------
# indices — dtype keyword controls the output dtype
# ---------------------------------------------------------------------------
def test_indices_dtype_kwarg():
    """numpy/_core/numeric.py indices ``def indices(dimensions, dtype=int,
    sparse=False)``. ``np.indices((2,2), dtype=np.float64)`` returns a float64
    grid. ferray's `indices` binding exposes neither `dtype` nor `sparse`
    (signature ``indices(py, dimensions: Vec<usize>)``), so `dtype` is a
    TypeError and float grids are unreachable.
    """
    expected = np.indices((2, 2), dtype=np.float64)
    got = fr.indices([2, 2], dtype=np.float64)
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# indices — sparse=True returns a TUPLE of open grids, not a dense array
# ---------------------------------------------------------------------------
def test_indices_sparse_kwarg():
    """numpy/_core/numeric.py indices: with ``sparse=True`` the result is a
    tuple of arrays each shaped with 1s on all but its own axis. ferray's
    `indices` exposes no `sparse` kwarg, so this raises TypeError.
    """
    expected = np.indices((2, 3), sparse=True)
    got = fr.indices([2, 3], sparse=True)
    assert isinstance(got, tuple)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g.shape == e.shape
        assert np.array_equal(g, e)


# ---------------------------------------------------------------------------
# nonzero — on a 0-d array numpy RAISES ValueError; ferray returns ()
# ---------------------------------------------------------------------------
def test_nonzero_0d_raises():
    """numpy/_core/fromnumeric.py:1994 ``nonzero``: "Calling nonzero on 0d
    arrays is not allowed." numpy raises ``ValueError``. ferray's `nonzero`
    silently returns an empty tuple ``()`` for a 0-d input.
    """
    z = np.array(5)
    with pytest.raises(ValueError):
        np.nonzero(z)
    with pytest.raises(ValueError):
        fr.nonzero(z)


# ---------------------------------------------------------------------------
# put — numpy exposes np.put; ferray's indexing surface omits it entirely
# ---------------------------------------------------------------------------
def test_put_function_exists():
    """numpy/_core/fromnumeric.py ``def put(a, ind, v, mode='raise')`` sets
    ``a.flat[ind] = v``. ``np.put(arange(5),[0,2],[-44,-55])`` -> [-44,1,-55,
    3,4]. ferray exposes no `put` (`hasattr(fr,'put')` is False), so the
    public surface is incomplete.
    """
    base = np.arange(5)
    expected = base.copy()
    np.put(expected, [0, 2], [-44, -55])  # live oracle
    assert hasattr(fr, "put"), "ferray.put is missing from the public surface"
    got = fr.put(base.astype(np.int64), [0, 2], [-44.0, -55.0])
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"
