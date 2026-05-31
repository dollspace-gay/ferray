"""numpy.ma PRIVATE interop-attribute parity (#907).

`numpy.ma` free functions read MaskedArray's private underscore attributes
(`_data`, `_fill_value`, `_hardmask`, `_sharedmask`, `_mask`) directly. The
`masked_array(obj)` constructor that backs `fix_invalid`/`getdata`/... reads
`getattr(obj, '_fill_value', None)` (numpy/ma/core.py:3013) and
`getattr(obj, '_hardmask', False)` (:3019). Before #907 `fr.ma.MaskedArray`
exposed only `_mask` (#905), so those free functions mis-read ferray arrays —
most observably `np.ma.fix_invalid(a).fill_value` dropped to the per-dtype
default 1e20 instead of propagating `a`'s materialized fill.

Every expected value comes from a live numpy.ma call (R-CHAR-3), never a
literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


def _pair(maker):
    """Build the same masked array in ferray and numpy from a constructor."""
    return maker(fr.ma), maker(np.ma)


def test_fill_value_private_is_0d_array_matching_numpy():
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]))
    a.fill_value = 7
    na.fill_value = 7
    assert isinstance(a._fill_value, np.ndarray)
    assert a._fill_value.shape == na._fill_value.shape == ()
    assert a._fill_value.dtype == na._fill_value.dtype
    assert np.array_equal(a._fill_value, na._fill_value)


def test_fill_value_private_default_matches_numpy_check_fill_value():
    # numpy's _fill_value is None when never set; the constructor falls back to
    # the per-dtype default. ferray reports the materialized default as a 0-d
    # array equal to numpy's check_fill_value(default, dtype).
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 0, 0]))
    expected = np.array(na.fill_value, dtype=na.dtype)
    assert np.array_equal(a._fill_value, expected)
    assert a._fill_value.dtype == expected.dtype


@pytest.mark.parametrize(
    "maker",
    [
        lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]),
        lambda m: m.array([1, 2, 3], mask=[1, 0, 0]),
        lambda m: m.array([True, False, True], mask=[0, 1, 0]),
    ],
)
def test_data_private_matches_getdata(maker):
    a, na = _pair(maker)
    assert np.array_equal(a._data, na._data)
    assert np.array_equal(a._data, np.ma.getdata(na))
    assert a._data.dtype == na._data.dtype


def test_hardmask_private_matches_numpy():
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]))
    assert a._hardmask == na._hardmask is False
    a.harden_mask()
    na.harden_mask()
    assert a._hardmask == na._hardmask is True


def test_sharedmask_private_is_bool():
    a = fr.ma.array([1.0, 2, 3], mask=[0, 1, 0])
    # numpy's typical post-construction value for an explicit mask= is True.
    na = np.ma.array([1.0, 2, 3], mask=[0, 1, 0])
    assert isinstance(a._sharedmask, bool)
    assert a._sharedmask == na._sharedmask is True


def test_mask_private_still_present_no_regression_905():
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]))
    assert np.array_equal(a._mask, na._mask)


def test_fix_invalid_propagates_fill_value_907_divergence():
    # The pinned divergence: fix_invalid reads _fill_value to propagate it.
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]))
    a.fill_value = 7
    na.fill_value = 7
    assert np.ma.fix_invalid(a).fill_value == np.ma.fix_invalid(na).fill_value
    assert np.ma.fix_invalid(a).fill_value == 7.0


def test_fix_invalid_masks_nonfinite_matching_numpy():
    a, na = _pair(lambda m: m.array([1.0, np.nan, np.inf, 4.0], mask=[1, 0, 0, 0]))
    fa = np.ma.fix_invalid(a)
    fna = np.ma.fix_invalid(na)
    assert np.array_equal(np.ma.getmaskarray(fa), np.ma.getmaskarray(fna))
    assert np.array_equal(fa.filled(), fna.filled())


def test_getdata_getmask_filled_is_masked_parity():
    a, na = _pair(lambda m: m.array([1.0, 2, 3], mask=[0, 1, 0]))
    a.fill_value = 5
    na.fill_value = 5
    assert np.array_equal(np.ma.getdata(a), np.ma.getdata(na))
    assert np.array_equal(np.ma.getmask(a), np.ma.getmask(na))
    assert np.array_equal(np.ma.filled(a), np.ma.filled(na))
    assert np.ma.is_masked(a) == np.ma.is_masked(na)


def test_int_dtype_fill_value_private():
    a, na = _pair(lambda m: m.array([1, 2, 3], mask=[0, 1, 0]))
    a.fill_value = 99
    na.fill_value = 99
    assert np.array_equal(a._fill_value, na._fill_value)
    assert a._fill_value.dtype == na._fill_value.dtype
