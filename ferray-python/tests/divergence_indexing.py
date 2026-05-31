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


# ===========================================================================
# ACToR-CRITIC re-audit (2026-05) — NEW divergences found on the CURRENT
# indexing.rs. The 14 tests above are now GREEN (those divergences were
# fixed); the pins below are RED and pin REAL, currently-unfixed divergences.
# numpy oracle: live 2.4.4. Expected values come from live numpy (R-CHAR-3).
# ===========================================================================


# ---------------------------------------------------------------------------
# put — int64 values are round-tripped through f64 (lossy above 2**53)
# ---------------------------------------------------------------------------
def test_put_int64_value_not_lossy():
    """numpy/_core/fromnumeric.py:489 ``def put(a, ind, v, mode='raise')`` —
    ``a.flat[ind] = v``. For an int64 array numpy stores the integer value
    exactly. ferray's `put` binding types ``v: Vec<f64>`` (indexing.rs `put`)
    and casts each value through f64; an int64 value above 2**53 loses its low
    bits. Oracle: np.put(arange(5,int64),[0],[2**60+1])[0] == 2**60+1
    (1152921504606846977); ferray yields 1152921504606846976.
    """
    big = 2**60 + 1
    a = np.arange(5, dtype=np.int64)
    expected = a.copy()
    np.put(expected, [0], [big])  # live oracle: exact int64
    got = fr.put(a.copy(), [0], [big])
    assert int(got[0]) == int(expected[0]) == big, f"{int(got[0])} != {big}"


# ---------------------------------------------------------------------------
# putmask — int64 values are round-tripped through f64 (lossy above 2**53)
# ---------------------------------------------------------------------------
def test_putmask_int64_value_not_lossy():
    """numpy putmask sets ``a.flat[n] = values[n % len(values)]`` exactly for
    an int64 array. ferray's `putmask` binding types ``values: Vec<f64>``
    (indexing.rs `putmask`) and casts through f64, losing low bits of large
    int64 values. Oracle on arange(5,int64), mask a>1, value 2**60+1:
    [0,1,1152921504606846977,...]; ferray yields ...976.
    """
    big = 2**60 + 1
    a = np.arange(5, dtype=np.int64)
    expected = a.copy()
    np.putmask(expected, expected > 1, [big])  # live oracle
    got = fr.putmask(a.copy(), a > 1, [big])
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# place — int64 values are round-tripped through f64 (lossy above 2**53)
# ---------------------------------------------------------------------------
def test_place_int64_value_not_lossy():
    """numpy place sets ``a[mask] = vals`` (cycled) exactly for an int64 array.
    ferray's `place` binding types ``vals: Vec<f64>`` (indexing.rs `place`) and
    casts through f64, losing low bits of large int64 values. Oracle on
    arange(5,int64), mask a>1, value 2**60+1:
    [0,1,1152921504606846977,1152921504606846977,1152921504606846977].
    """
    big = 2**60 + 1
    a = np.arange(5, dtype=np.int64)
    expected = a.copy()
    np.place(expected, expected > 1, [big])  # live oracle
    got = fr.place(a.copy(), a > 1, [big])
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# select — the `default` value is round-tripped through f64 (lossy above 2**53)
# ---------------------------------------------------------------------------
def test_select_default_int64_not_lossy():
    """numpy/_core/_function_base_impl.py select(condlist, choicelist,
    default=0): when no condition matches, the default fills with the choice
    dtype (here int64) exactly. ferray's `select` binding types
    ``default: f64`` (indexing.rs `select` signature `default = 0.0`) and casts
    through f64, losing the low bits of a large int64 default. Oracle: all-False
    cond on int64 choices with default 2**60+1 -> [1152921504606846977, ...];
    ferray yields ...976.
    """
    big = 2**60 + 1
    cond = [np.array([False, False])]
    ch = [np.array([1, 2], dtype=np.int64)]
    expected = np.select(cond, ch, default=big)  # live oracle
    got = fr.select(cond, ch, default=big)
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# put — complex values are rejected (numpy supports complex arrays)
# ---------------------------------------------------------------------------
def test_put_complex_supported():
    """numpy/_core/fromnumeric.py:489 ``put`` works on a complex128 array:
    np.put([1+1j,2+2j,3+3j],[0],[9+9j]) -> [9+9j, 2+2j, 3+3j]. ferray's `put`
    binding types ``v: Vec<f64>``, which rejects complex with
    ``TypeError: must be real number, not complex`` — the imaginary part can
    never be written (R-CODE-4).
    """
    a = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    expected = a.copy()
    np.put(expected, [0], [9 + 9j])  # live oracle
    got = fr.put(a.copy(), [0], [9 + 9j])
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# choose — the `mode` kwarg ('wrap'/'clip') is part of the public signature
# ---------------------------------------------------------------------------
def test_choose_mode_wrap():
    """numpy/_core/fromnumeric.py choose(a, choices, out=None, mode='raise');
    mode {'raise','wrap','clip'} resolves out-of-bounds index entries.
    np.choose([0,3,1,2], [[0,1,2,3],[10,11,12,13]], mode='wrap') -> [0,11,12,3]
    (index 3 -> 3 % 2 = 1, index 2 -> 0). ferray's `choose` binding exposes no
    `mode` kwarg -> TypeError, and out-of-range entries always raise.
    """
    idx = np.array([0, 3, 1, 2])
    chs = [np.array([0, 1, 2, 3]), np.array([10, 11, 12, 13])]
    expected = np.choose(idx, chs, mode="wrap")  # live oracle: [0,11,12,3]
    got = fr.choose(idx, chs, mode="wrap")
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


def test_choose_mode_clip():
    """numpy choose mode='clip' clamps OOB index entries into range.
    np.choose([0,3,1,2], [[0,1,2,3],[10,11,12,13]], mode='clip') -> [0,11,12,13]
    (index 3 and 2 both clip to last choice 1). ferray's `choose` exposes no
    `mode` kwarg -> TypeError.
    """
    idx = np.array([0, 3, 1, 2])
    chs = [np.array([0, 1, 2, 3]), np.array([10, 11, 12, 13])]
    expected = np.choose(idx, chs, mode="clip")  # live oracle: [0,11,12,13]
    got = fr.choose(idx, chs, mode="clip")
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"


# ---------------------------------------------------------------------------
# unravel_index — a scalar integer `indices` is accepted (returns scalar tuple)
# ---------------------------------------------------------------------------
def test_unravel_index_scalar():
    """numpy/_core/multiarray ``unravel_index(indices, shape)`` accepts a
    scalar int index and returns a tuple of scalar coordinates:
    np.unravel_index(3, (2,2)) -> (1, 1). ferray's `unravel_index` binding types
    ``indices: Vec<usize>`` (indexing.rs), so a scalar int raises
    ``TypeError: 'int' object is not an instance of 'Sequence'``.
    """
    expected = np.unravel_index(3, (2, 2))  # live oracle: (1, 1)
    got = fr.unravel_index(3, (2, 2))
    assert tuple(int(np.asarray(x)) for x in got) == tuple(int(x) for x in expected)


# ---------------------------------------------------------------------------
# ravel_multi_index — the `order` kwarg ('C'/'F') is part of the public ABI
# ---------------------------------------------------------------------------
def test_ravel_multi_index_order_f():
    """numpy ``ravel_multi_index(multi_index, dims, mode='raise', order='C')``.
    With order='F' the flat index is computed in Fortran (column-major) order:
    np.ravel_multi_index([[0,1],[1,1]], (2,2), order='F') -> [2, 3]
    (vs C-order [1, 3]). ferray's `ravel_multi_index` binding exposes no `order`
    kwarg -> TypeError, so column-major raveling is unreachable.
    """
    mi = [[0, 1], [1, 1]]
    expected = np.ravel_multi_index(mi, (2, 2), order="F")  # live oracle: [2,3]
    got = fr.ravel_multi_index(mi, (2, 2), order="F")
    assert np.array_equal(got, expected), f"{got.tolist()} != {expected.tolist()}"
