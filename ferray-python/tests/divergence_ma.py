"""Adversarial divergence pins: ferray.ma vs the live numpy.ma oracle.

numpy 2.4.x is the source of truth (goal.md verification model B). Every
expected value here is produced by a LIVE numpy.ma call or by an identity
check against a numpy singleton (`np.ma.masked`, `np.ma.nomask`) — never
literal-copied from the ferray side (goal.md R-CHAR-3). Where the oracle is a
singleton, an `is`-identity sanity assert proves non-tautology before the
ferray assertion runs.

Each test is EXPECTED TO FAIL against current ferray. The critic writes the
pin; the fixer lands the fix in the owning crate (ferray-ma) and the test goes
green (R-DEFER-3). The critic never edits production code.

Upstream cites are into /home/doll/numpy-ref/numpy/ma/core.py.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. Default fill value for float64 is 1e20, not 0.0.
#    numpy/ma/core.py:166  ->  default_filler['f'] = 1.e20
#    numpy/ma/core.py:3857 def filled(self, fill_value=None) uses it when None.
#    ferray's filled_default substitutes 0.0.
# ---------------------------------------------------------------------------


def test_filled_default_matches_numpy_float64_fill():
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    oracle = np.ma.masked_array(data, mask=mask).filled()
    # Non-tautology: the masked slot must read back as numpy's 1e20 default.
    assert oracle[1] == 1e20
    got = fr.ma.masked_array(data, mask=mask).filled()
    np.testing.assert_array_equal(got, oracle)


def test_module_filled_default_matches_numpy_float64_fill():
    """numpy.ma.filled(a) free function uses the same 1e20 default."""
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    oracle = np.ma.filled(np.ma.masked_array(data, mask=mask))
    assert oracle[1] == 1e20
    got = fr.ma.filled(fr.ma.masked_array(data, mask=mask))
    np.testing.assert_array_equal(got, oracle)


# ---------------------------------------------------------------------------
# 2. fill_value property: numpy exposes it (1e20 for float64). ferray's
#    //! doc-comment lists `fill_value` as a property but the pyclass omits it.
#    numpy/ma/core.py:3793 @property def fill_value(self).
# ---------------------------------------------------------------------------


def test_fill_value_property_exists_and_matches_numpy():
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    oracle = np.ma.masked_array(data, mask=mask).fill_value
    assert oracle == 1e20  # non-tautology against numpy default_filler['f']
    fm = fr.ma.masked_array(data, mask=mask)
    assert fm.fill_value == oracle


# ---------------------------------------------------------------------------
# 3. masked_equal sets fill_value to the compared value.
#    numpy/ma/core.py:2171-2172  output = masked_where(...); output.fill_value = value
# ---------------------------------------------------------------------------


def test_masked_equal_sets_fill_value_to_compared_value():
    data = np.array([1.0, 2.0, 3.0, 2.0])
    oracle = np.ma.masked_equal(data, 2.0).fill_value
    assert oracle == 2.0  # non-tautology: numpy assigns the compared value
    fm = fr.ma.masked_equal(data, 2.0)
    assert fm.fill_value == oracle


# ---------------------------------------------------------------------------
# 4. All-masked full reductions return the `masked` singleton, NOT a finite/NaN
#    scalar and NOT an exception.
#    sum  -> numpy/ma/core.py:5250  result = masked
#    mean -> numpy/ma/core.py:5417  result = masked
#    min  -> numpy/ma/core.py:5942  result = masked
#    max  -> numpy/ma/core.py:6047  result = masked
#    var/std go through _var and yield masked when count==0.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "var", "std"])
def test_all_masked_reduction_returns_masked_singleton(op):
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, True, True])
    oracle = getattr(np.ma.masked_array(data, mask=mask), op)()
    # Non-tautology: the live numpy oracle really is the masked singleton.
    assert oracle is np.ma.masked
    got = getattr(fr.ma.masked_array(data, mask=mask), op)()
    assert got is np.ma.masked


# ---------------------------------------------------------------------------
# 5. getmask of an unmasked array is the `nomask` scalar (np.False_), not a
#    full bool ndarray. getmaskarray is the full-array variant.
#    numpy/ma/core.py:1468  return getattr(a, '_mask', nomask)
# ---------------------------------------------------------------------------


def test_getmask_no_mask_returns_nomask_scalar():
    data = np.array([1.0, 2.0, 3.0])
    oracle = np.ma.getmask(np.ma.masked_array(data))
    assert oracle is np.ma.nomask  # non-tautology: numpy returns the singleton
    got = fr.ma.getmask(fr.ma.masked_array(data))
    assert got is np.ma.nomask or (np.ndim(got) == 0 and bool(got) is False)


# ---------------------------------------------------------------------------
# 6. masked_where with a mismatched-shape condition raises IndexError
#    (boolean-index broadcast failure), not ValueError.
#    numpy/ma/core.py:1885 def masked_where(condition, a, copy=True).
# ---------------------------------------------------------------------------


def test_masked_where_shape_mismatch_raises_indexerror():
    data = np.array([1.0, 2.0, 3.0])
    bad_cond = np.array([True, False])
    # Non-tautology: confirm the live oracle raises IndexError on this input.
    with pytest.raises(IndexError):
        np.ma.masked_where(bad_cond, data)
    with pytest.raises(IndexError):
        fr.ma.masked_where(bad_cond, data)


# ---------------------------------------------------------------------------
# 7. The NumPy `__array__` protocol returns the underlying DATA verbatim, NOT
#    the fill-substituted array. `np.asarray(ma)` therefore equals `ma.data`,
#    with the original values at masked positions. ferray's __array__ calls
#    filled() and so substitutes the fill value at masked slots.
#    numpy/ma/core.py: MaskedArray.__array__ yields self._data (asarray==data).
# ---------------------------------------------------------------------------


def test_array_protocol_returns_data_not_filled():
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    nm = np.ma.masked_array(data, mask=mask)
    oracle = np.asarray(nm)
    # Non-tautology: numpy's asarray equals the underlying data, keeping 2.0.
    np.testing.assert_array_equal(oracle, nm.data)
    assert oracle[1] == 2.0
    got = np.asarray(fr.ma.masked_array(data, mask=mask))
    np.testing.assert_array_equal(got, oracle)


# ---------------------------------------------------------------------------
# 8. The `.mask` property of an unmasked array is the `nomask` scalar, mirroring
#    getmask. ferray returns a full array([False, ...]).
#    numpy/ma/core.py: MaskedArray.mask getter returns self._mask (nomask here).
# ---------------------------------------------------------------------------


def test_mask_property_no_mask_is_nomask_scalar():
    data = np.array([1.0, 2.0, 3.0])
    oracle = np.ma.masked_array(data).mask
    assert oracle is np.ma.nomask  # non-tautology against numpy singleton
    got = fr.ma.masked_array(data).mask
    assert got is np.ma.nomask or (np.ndim(got) == 0 and bool(got) is False)


# ===========================================================================
# NEW PINS (acto-critic batch 2): dtype-aware fill values, dtype preservation
# through join/manipulation ops, and missing MaskedArray methods.
# ===========================================================================


# ---------------------------------------------------------------------------
# 9. Module-level numpy.ma.default_fill_value(obj) keys off the dtype KIND:
#    integer -> 999999, bool -> True, float -> 1e20. ferray's standalone
#    default_fill_value ignores the dtype kind and always returns the float
#    filler (1e20) regardless of integer/bool input.
#    numpy/ma/core.py:260 default_fill_value; :163-171 default_filler table
#    ('i'->999999, 'u'->999999, 'b'->True, 'f'->1.e20).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.int32),
        np.array([1, 2], dtype=np.int8),
        np.array([1, 2], dtype=np.uint8),
        np.array([True, False], dtype=np.bool_),
    ],
)
def test_module_default_fill_value_keys_off_dtype_kind(arr):
    oracle = np.ma.default_fill_value(arr)
    # Non-tautology: integers map to 999999, bool to True (NOT the 1e20 float
    # filler that ferray emits for every kind).
    if arr.dtype.kind in "iu":
        assert int(np.asarray(oracle)) == 999999
    elif arr.dtype.kind == "b":
        assert bool(np.asarray(oracle)) is True
    got = fr.ma.default_fill_value(arr)
    assert int(np.asarray(got)) == int(np.asarray(oracle))


# ---------------------------------------------------------------------------
# 10. numpy.ma.minimum_fill_value(obj) returns the per-dtype MAXIMUM (the value
#     that loses every min comparison): +inf for float, iinfo(dtype).max for
#     integer. ferray returns +inf for integer dtypes (and raises OverflowError
#     for narrow ints), diverging from the integer typemax.
#     numpy/ma/core.py:331 minimum_fill_value -> _extremum_fill_value(min_filler)
#     numpy/ma/core.py:213-214 min_filler = _maxvals (iinfo max for integers).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2], dtype=np.int8),
        np.array([1, 2], dtype=np.int16),
        np.array([1, 2], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.uint8),
    ],
)
def test_minimum_fill_value_is_dtype_max_for_integers(arr):
    oracle = np.ma.minimum_fill_value(arr)
    # Non-tautology: numpy returns the integer typemax, not +inf.
    assert int(np.asarray(oracle)) == np.iinfo(arr.dtype).max
    got = fr.ma.minimum_fill_value(arr)
    assert int(np.asarray(got)) == int(np.asarray(oracle))


# ---------------------------------------------------------------------------
# 11. numpy.ma.maximum_fill_value(obj) returns the per-dtype MINIMUM: -inf for
#     float, iinfo(dtype).min for integer. ferray returns -inf for integers.
#     numpy/ma/core.py:383 maximum_fill_value -> _extremum_fill_value(max_filler)
#     numpy/ma/core.py:209-210 max_filler = _minvals (iinfo min for integers).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "arr",
    [
        np.array([1, 2], dtype=np.int8),
        np.array([1, 2], dtype=np.int16),
        np.array([1, 2], dtype=np.int32),
        np.array([1, 2], dtype=np.int64),
        np.array([1, 2], dtype=np.uint8),
    ],
)
def test_maximum_fill_value_is_dtype_min_for_integers(arr):
    oracle = np.ma.maximum_fill_value(arr)
    # Non-tautology: numpy returns the integer typemin, not -inf.
    assert int(np.asarray(oracle)) == np.iinfo(arr.dtype).min
    got = fr.ma.maximum_fill_value(arr)
    assert int(np.asarray(got)) == int(np.asarray(oracle))


# ---------------------------------------------------------------------------
# 12. The `.fill_value` ATTRIBUTE for a narrow integer array reports the
#     UNCAST default 999999 (numpy stores it as int64, not cast to int8), while
#     ferray reports the dtype-cast value (999999 % 256 == 63 for int8).
#     numpy/ma/core.py:260 default_fill_value returns the raw Python int 999999
#     (the _recursive_fill_value scalar branch does NOT down-cast the scalar).
# ---------------------------------------------------------------------------


def test_int8_array_fill_value_attribute_is_uncast_999999():
    arr = np.array([1, 2], dtype=np.int8)
    oracle = np.ma.masked_array(arr).fill_value
    # Non-tautology: numpy reports the uncast 999999, not the int8-wrapped 63.
    assert int(np.asarray(oracle)) == 999999
    got = fr.ma.masked_array(arr).fill_value
    assert int(np.asarray(got)) == int(np.asarray(oracle))


# ---------------------------------------------------------------------------
# 13. Integer dtype is PRESERVED through join / manipulation ops. numpy keeps
#     int64 across concatenate/stack/vstack/hstack/column_stack/append; ferray's
#     f64-only model upcasts every result to float64.
#     numpy/ma/core.py:7299 def concatenate(arrays, axis=0) -> dtype from inputs
#     numpy/ma/extras.py stack/vstack/hstack delegate to concatenate.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        lambda m, a, b: m.concatenate([a, b]),
        lambda m, a, b: m.stack([a, b]),
        lambda m, a, b: m.vstack([a, b]),
        lambda m, a, b: m.hstack([a, b]),
        lambda m, a, b: m.column_stack([a, b]),
        lambda m, a, b: m.append(a, b),
    ],
    ids=["concatenate", "stack", "vstack", "hstack", "column_stack", "append"],
)
def test_join_ops_preserve_integer_dtype(fn):
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    na = np.ma.masked_array(a, mask=[0, 1])
    nb = np.ma.masked_array(b, mask=[1, 0])
    oracle = fn(np.ma, na, nb)
    # Non-tautology: numpy's joined result is genuinely int64.
    assert oracle.dtype == np.int64
    fa = fr.ma.masked_array(a, mask=[0, 1])
    fb = fr.ma.masked_array(b, mask=[1, 0])
    got = fn(fr.ma, fa, fb)
    assert np.asarray(got.data).dtype == oracle.dtype


# ---------------------------------------------------------------------------
# 14. numpy.ma.outer preserves integer dtype (int64 x int64 -> int64). ferray
#     upcasts to float64.
#     numpy/ma/core.py:8317 def outer(a, b) -> data = np.outer(filled a, filled b)
#     np.outer of two int64 inputs yields int64.
# ---------------------------------------------------------------------------


def test_outer_preserves_integer_dtype():
    a = np.array([1, 2], dtype=np.int64)
    b = np.array([3, 4], dtype=np.int64)
    oracle = np.ma.outer(np.ma.masked_array(a), np.ma.masked_array(b))
    assert oracle.dtype == np.int64  # non-tautology
    got = fr.ma.outer(fr.ma.masked_array(a), fr.ma.masked_array(b))
    assert np.asarray(got.data).dtype == oracle.dtype


# ---------------------------------------------------------------------------
# 15. MaskedArray.anom() method (deviation from the masked mean) exists on the
#     array object. ferray exposes ma.anom() as a module function but the
#     pyclass METHOD is missing (AttributeError).
#     numpy/ma/core.py:5432 def anom(self, axis=None, dtype=None).
# ---------------------------------------------------------------------------


def test_masked_array_has_anom_method():
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    oracle = np.ma.masked_array(data, mask=mask).anom().filled(-99.0)
    # Non-tautology: numpy's anomalies are data - mean(=2.0) at unmasked slots.
    np.testing.assert_array_equal(oracle, np.array([-1.0, -99.0, 1.0]))
    got = fr.ma.masked_array(data, mask=mask).anom().filled(-99.0)
    np.testing.assert_array_equal(np.asarray(got), oracle)


# ---------------------------------------------------------------------------
# 16. MaskedArray.compress(condition) method exists on the array object.
#     ferray exposes ma.compress as a module function but the pyclass METHOD is
#     missing (AttributeError).
#     numpy/ma/core.py:3974 def compress(self, condition, axis=None, out=None).
# ---------------------------------------------------------------------------


def test_masked_array_has_compress_method():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([False, True, False, False])
    cond = [True, False, True, True]
    oracle = np.ma.masked_array(data, mask=mask).compress(cond).filled(-99.0)
    # Non-tautology: numpy keeps positions 0,2,3 (-> 1.0, 3.0, 4.0).
    np.testing.assert_array_equal(oracle, np.array([1.0, 3.0, 4.0]))
    got = fr.ma.masked_array(data, mask=mask).compress(cond).filled(-99.0)
    np.testing.assert_array_equal(np.asarray(got), oracle)


# ---------------------------------------------------------------------------
# 17. MaskedArray.dot(b) method exists on the array object (2-D masked matmul).
#     ferray exposes ma.dot as a module function but the pyclass METHOD is
#     missing (AttributeError).
#     numpy/ma/core.py:5163 def dot(self, b, out=None, strict=False).
# ---------------------------------------------------------------------------


def test_masked_array_has_dot_method():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.ma.masked_array(data).dot(np.ma.masked_array(data)).filled(-99.0)
    # Non-tautology: numpy's [[1,2],[3,4]] @ itself = [[7,10],[15,22]].
    np.testing.assert_array_equal(oracle, np.array([[7.0, 10.0], [15.0, 22.0]]))
    got = fr.ma.masked_array(data).dot(fr.ma.masked_array(data)).filled(-99.0)
    np.testing.assert_array_equal(np.asarray(got), oracle)
