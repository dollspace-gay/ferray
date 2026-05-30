"""Adversarial divergence tests for ferray.ma vs numpy.ma (numpy 2.4.x oracle).

Each test pins an OBSERVABLE divergence between ferray.ma and the live
numpy.ma oracle. The numpy result is computed inline (not hardcoded) so the
oracle stays honest across numpy patch releases.

These tests are EXPECTED TO FAIL on current ferray until the divergence is
fixed. They are written by the critic and never carry fixes.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. Default fill value (numpy float64 default is 1e20, not 0.0)
#    numpy/ma/core.py:166  ->  'f': 1.e20  (default_filler table)
#    numpy/ma/core.py:3857 def filled(self, fill_value=None): uses default.
# ---------------------------------------------------------------------------


def test_filled_default_matches_numpy_float64_fill():
    """numpy.ma fills masked float64 with 1e20 by default.

    numpy/ma/core.py:166 -- default_filler['f'] = 1.e20
    ferray substitutes 0.0, so masked positions read back wrong.
    """
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    oracle = np.ma.masked_array(data, mask=mask).filled()
    got = fr.ma.masked_array(data, mask=mask).filled()
    np.testing.assert_array_equal(got, oracle)


# ---------------------------------------------------------------------------
# 2. fill_value property is documented but absent.
#    The module docstring lists `fill_value` as a property; numpy exposes it.
#    numpy/ma/core.py: MaskedArray.fill_value property (default 1e20 for f8).
# ---------------------------------------------------------------------------


def test_fill_value_property_exists_and_matches_numpy():
    """numpy exposes MaskedArray.fill_value (1e20 for float64).

    ferray's ma.rs //! doc-comment lists `fill_value` as a property but the
    pyclass never defines it.
    """
    data = np.array([1.0, 2.0, 3.0])
    oracle = np.ma.masked_array(data, mask=[False, True, False]).fill_value
    fm = fr.ma.masked_array(data, mask=np.array([False, True, False]))
    assert fm.fill_value == oracle


# ---------------------------------------------------------------------------
# 3. All-masked full reductions: numpy returns the `masked` singleton, NOT a
#    finite/NaN scalar and NOT an exception.
#    numpy/ma/core.py:5249-5250  elif newmask: result = masked
# ---------------------------------------------------------------------------


def test_all_masked_sum_returns_masked_singleton():
    """numpy: x.sum() on a fully masked array is `np.ma.masked`.

    numpy/ma/core.py:5249 -- 'elif newmask: result = masked'
    ferray returns 0.0 instead.
    """
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, True, True])
    oracle = np.ma.masked_array(data, mask=mask).sum()
    assert oracle is np.ma.masked  # sanity: oracle really is masked
    got = fr.ma.masked_array(data, mask=mask).sum()
    assert got is np.ma.masked


def test_all_masked_mean_returns_masked_singleton():
    """numpy: x.mean() on a fully masked array is `np.ma.masked` (not NaN).

    numpy/ma/core.py:5376 def mean -- division yields masked when count==0.
    ferray returns nan.
    """
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, True, True])
    oracle = np.ma.masked_array(data, mask=mask).mean()
    assert oracle is np.ma.masked
    got = fr.ma.masked_array(data, mask=mask).mean()
    assert got is np.ma.masked


def test_all_masked_min_returns_masked_singleton():
    """numpy: x.min() on a fully masked array is `np.ma.masked`.

    numpy/ma/core.py MaskedArray.min returns `masked` when all masked.
    ferray raises ValueError('all elements are masked').
    """
    data = np.array([1.0, 2.0])
    mask = np.array([True, True])
    oracle = np.ma.masked_array(data, mask=mask).min()
    assert oracle is np.ma.masked
    got = fr.ma.masked_array(data, mask=mask).min()
    assert got is np.ma.masked


def test_all_masked_max_returns_masked_singleton():
    """numpy: x.max() on a fully masked array is `np.ma.masked`.

    numpy/ma/core.py MaskedArray.max returns `masked` when all masked.
    ferray raises ValueError('all elements are masked').
    """
    data = np.array([1.0, 2.0])
    mask = np.array([True, True])
    oracle = np.ma.masked_array(data, mask=mask).max()
    assert oracle is np.ma.masked
    got = fr.ma.masked_array(data, mask=mask).max()
    assert got is np.ma.masked


# ---------------------------------------------------------------------------
# 4. getmask of an unmasked array: numpy returns the `nomask` scalar (False),
#    not a full bool array. getmaskarray is the full-array variant.
#    numpy/ma/core.py:1468  return getattr(a, '_mask', nomask)
# ---------------------------------------------------------------------------


def test_getmask_no_mask_returns_nomask_scalar():
    """numpy.ma.getmask of an unmasked array is the `nomask` scalar (False).

    numpy/ma/core.py:1460-1463 -- ma.getmask(b) == ma.nomask is True.
    ferray returns a full array([False, False, False]); compare against
    getmaskarray which is the full-array variant.
    """
    data = np.array([1.0, 2.0, 3.0])
    m = np.ma.masked_array(data)  # no mask -> nomask
    oracle = np.ma.getmask(m)
    assert oracle is np.ma.nomask
    got = fr.ma.getmask(fr.ma.masked_array(data))
    # numpy returns the scalar nomask (False), ferray returns an ndarray.
    assert got is np.ma.nomask or (np.ndim(got) == 0 and got == np.ma.nomask)


# ---------------------------------------------------------------------------
# 5. masked_where with a condition of mismatched length: numpy raises
#    IndexError (boolean index broadcast), ferray raises ValueError.
#    numpy/ma/core.py:1885 def masked_where -- mask = make_mask(condition)
#    then indexing raises IndexError on shape mismatch.
# ---------------------------------------------------------------------------


def test_masked_where_shape_mismatch_exception_type():
    """numpy.ma.masked_where raises IndexError on mismatched condition length.

    numpy/ma/core.py:1885 def masked_where(condition, a, ...).
    ferray raises ValueError instead.
    """
    data = np.array([1.0, 2.0, 3.0])
    bad_cond = np.array([True, False])

    with pytest.raises(IndexError):
        np.ma.masked_where(bad_cond, data)  # oracle: IndexError

    with pytest.raises(IndexError):
        fr.ma.masked_where(bad_cond, data)


# ---------------------------------------------------------------------------
# 6. masked_equal sets fill_value to the comparison value in numpy.
#    numpy/ma/core.py:2143 def masked_equal(x, value, ...):
#        result.fill_value = value
# ---------------------------------------------------------------------------


def test_masked_equal_sets_fill_value_to_compared_value():
    """numpy.ma.masked_equal sets fill_value to the compared value.

    numpy/ma/core.py: masked_equal sets `output.fill_value = value`.
    ferray has no fill_value at all, and would not track the value.
    """
    data = np.array([1.0, 2.0, 3.0, 2.0])
    oracle = np.ma.masked_equal(data, 2.0).fill_value
    fm = fr.ma.masked_equal(data, 2.0)
    assert fm.fill_value == oracle
