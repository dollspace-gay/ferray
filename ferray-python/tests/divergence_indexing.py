"""Adversarial divergence tests for the ferray indexing surface.

ACToR-CRITIC audit of ``ferray-python/src/indexing.rs`` against the
numpy 2.4.5 oracle. Every test below is EXPECTED TO FAIL on current
ferray (the divergence it pins is real). Each docstring cites the
upstream numpy source the target violates.

Run:
    PYTHONPATH=python python3 -m pytest tests/divergence_indexing.py -q
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# take — default axis (numpy: axis=None => operate on flattened array)
# ---------------------------------------------------------------------------
def test_take_default_axis_flattens():
    """numpy/_core/fromnumeric.py:107 `def take(a, indices, axis=None, ...)`
    and :134-136 "axis : int, optional ... By default, the flattened input
    array is used." With a 2-D source and no axis, numpy gathers from the
    flattened array. ferray's binding defaults `axis=0` (signature
    `(a, indices, axis = 0)` in `take` in indexing.rs), producing a wrong
    result / IndexError instead of flattening.
    """
    m = np.array([[1, 2, 3], [4, 5, 6]])
    expected = np.take(m, [0, 4])  # flattened -> [1, 5]
    got = fr.take(m, [0, 4])
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"
    assert got.dtype == expected.dtype


# ---------------------------------------------------------------------------
# take — out-of-bounds index raises IndexError (not ValueError)
# ---------------------------------------------------------------------------
def test_take_out_of_bounds_raises_indexerror():
    """numpy/_core/fromnumeric.py:142 mode='raise' (default) "raise an error".
    numpy raises ``IndexError`` for an out-of-bounds take index; ferray's
    `take` raises ``ValueError`` ("index 5 is out of bounds ..."). IndexError
    and ValueError are unrelated in numpy's hierarchy (a plain ValueError is
    NOT an IndexError), so a caller catching IndexError breaks.
    """
    a = np.array([10, 20, 30, 40])
    with pytest.raises(Exception) as np_exc:
        np.take(a, [5])
    assert isinstance(np_exc.value, IndexError)
    with pytest.raises(IndexError):
        fr.take(a, [5])


# ---------------------------------------------------------------------------
# take — axis out of bounds raises AxisError (not plain ValueError)
# ---------------------------------------------------------------------------
def test_take_bad_axis_raises_axiserror():
    """numpy/_core/fromnumeric.py:107 take dispatches to ndarray.take, which
    normalizes axis via normalize_axis_index and raises
    ``numpy.exceptions.AxisError`` for an out-of-range axis. ferray's `take`
    raises a plain ``ValueError`` ("axis 5 is out of bounds ..."), which is
    NOT an AxisError instance.
    """
    a = np.array([10, 20, 30, 40])
    with pytest.raises(np.exceptions.AxisError):
        np.take(a, [0], axis=5)
    with pytest.raises(np.exceptions.AxisError):
        fr.take(a, [0], axis=5)


# ---------------------------------------------------------------------------
# take — mode kwarg ('clip'/'wrap'/'raise') is part of the public signature
# ---------------------------------------------------------------------------
def test_take_mode_clip():
    """numpy/_core/fromnumeric.py:107 `def take(a, indices, axis=None,
    out=None, mode='raise')` and :142-146 mode {'raise','wrap','clip'}.
    `np.take(a, [5], mode='clip')` clips to the last element -> [a[-1]].
    ferray's `take` binding exposes no `mode` kwarg (signature
    `(a, indices, axis = 0)`), so passing mode='clip' is a TypeError.
    """
    a = np.array([10, 20, 30])
    expected = np.take(a, [5], mode="clip")  # -> [30]
    got = fr.take(a, [5], mode="clip")
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# take — multi-dimensional index array preserves index shape
# ---------------------------------------------------------------------------
def test_take_2d_index_shape():
    """numpy/_core/fromnumeric.py:200-204 "If `indices` is not one dimensional,
    the output also has these dimensions." `np.take([4,3,5,7,6,8],
    [[0,1],[2,3]])` returns a (2,2) array. ferray's `take` binding accepts
    only a flat `Vec<isize>` of indices, so a 2-D index list is a TypeError
    and the output-shape contract is unmet.
    """
    a = np.array([4, 3, 5, 7, 6, 8])
    expected = np.take(a, [[0, 1], [2, 3]])
    got = fr.take(a, [[0, 1], [2, 3]])
    assert got.shape == expected.shape == (2, 2)
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# take_along_axis — element-wise N-d index array along an axis
# ---------------------------------------------------------------------------
def test_take_along_axis_2d():
    """numpy take_along_axis matches the source and index arrays element-wise
    along `axis`; the index array has the same ndim as `a`
    (numpy/lib/_shape_base_impl.py take_along_axis). Sorting each row via
    `idx = argsort(a, axis=1)` then `take_along_axis(a, idx, axis=1)` yields
    the per-row sorted array. ferray's `take_along_axis` binding accepts only
    a flat `Vec<isize>`, so a 2-D index array is a TypeError.
    """
    a = np.array([[10, 30, 20], [60, 40, 50]])
    idx = np.argsort(a, axis=1)
    expected = np.take_along_axis(a, idx, axis=1)  # [[10,20,30],[40,50,60]]
    got = fr.take_along_axis(a, idx, axis=1)
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# compress — default axis=None operates on the flattened array
# ---------------------------------------------------------------------------
def test_compress_default_axis_flattens():
    """numpy/_core/fromnumeric.py:2138 `def compress(condition, a, axis=None,
    out=None)` and :2156-2157 "axis : int, optional ... If None (default),
    work on the flattened array." `np.compress([False, True], m)` selects the
    flat element at index 1. ferray's `compress` binding defaults `axis=0`
    (signature `(condition, a, axis = 0)` in `compress` in indexing.rs), so it
    returns a row-slice instead of a flattened-element selection.
    """
    m = np.array([[1, 2], [3, 4], [5, 6]])
    expected = np.compress([False, True], m)  # flattened -> [2]
    got = fr.compress([False, True], m)
    assert np.array_equal(got, expected), f"{got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# putmask — values cycle by GLOBAL flat position, not mask-relative position
# ---------------------------------------------------------------------------
def test_putmask_values_cycle_by_flat_index():
    """numpy/_core/multiarray.py:1145 "Sets ``a.flat[n] = values[n]`` for each
    n where ``mask.flat[n]==True``" and :1175-1178 example
    `np.putmask(x, x>1, [-33, -44])` on `arange(5)` -> [0, 1, -33, -44, -33]:
    values are indexed by the GLOBAL flat position n (values[n % len]), so
    masked positions 2,3,4 take values[2],values[3],values[4] =
    -33,-44,-33. ferray's `putmask` requires `len(values)` to equal the number
    of masked elements (raises ValueError "values length 2 must be 1 or
    equal ...") and would cycle by mask-relative index, both wrong.
    """
    x = np.arange(5)
    xc = x.copy()
    np.putmask(xc, x > 1, [-33, -44])  # [0, 1, -33, -44, -33]
    got = fr.putmask(x.astype(np.int64), (x > 1), [-33.0, -44.0])
    assert np.array_equal(got, xc), f"{got!r} != {xc.tolist()!r}"


# ---------------------------------------------------------------------------
# indices — dtype keyword controls the output dtype
# ---------------------------------------------------------------------------
def test_indices_dtype_kwarg():
    """numpy/_core/numeric.py:1726 `def indices(dimensions, dtype=int,
    sparse=False)`. `np.indices((2, 2), dtype=np.float64)` returns a float64
    grid. ferray's `indices` binding exposes neither `dtype` nor `sparse`
    (signature `indices(py, dimensions: Vec<usize>)` in indexing.rs), so the
    dtype kwarg is a TypeError and float grids are unreachable.
    """
    expected = np.indices((2, 2), dtype=np.float64)
    got = fr.indices([2, 2], dtype=np.float64)
    assert got.dtype == expected.dtype == np.float64
    assert np.array_equal(got, expected)
