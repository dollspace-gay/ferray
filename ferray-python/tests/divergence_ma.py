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
