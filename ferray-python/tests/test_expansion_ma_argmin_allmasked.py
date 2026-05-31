"""All-masked + delegation parity for fr.ma.argmin / fr.ma.argmax module forms.

Pins divergence #903: the module-level `fr.ma.argmin`/`argmax` wrappers used to
take the native `to_f64_ma().argmin()` path, which RAISED `ValueError` on an
all-masked array. numpy.ma.argmin/argmax instead return `numpy.int64(0)` (they
fill masked entries with the dtype max/min and never raise). The fix delegates
the module wrappers to numpy.ma — the same path the METHOD forms already use.

All expected values are derived LIVE from numpy.ma (R-CHAR-3): nothing is
literal-copied from the ferray side.
"""

import ferray as fr
import numpy as np


def test_module_argmin_all_masked_returns_int64_zero():
    data, mask = [1, 2, 3], [1, 1, 1]
    np_res = np.ma.argmin(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmin(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res) == 0
    assert hasattr(fr_res, "dtype")
    assert str(fr_res.dtype) == str(np_res.dtype)


def test_module_argmax_all_masked_returns_int64_zero():
    data, mask = [1, 2, 3], [1, 1, 1]
    np_res = np.ma.argmax(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmax(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res) == 0
    assert hasattr(fr_res, "dtype")
    assert str(fr_res.dtype) == str(np_res.dtype)


def test_module_argmin_partial_mask_matches_numpy():
    data, mask = [5, 2, 3], [0, 1, 0]
    np_res = np.ma.argmin(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmin(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res)
    assert hasattr(fr_res, "dtype")


def test_module_argmax_partial_mask_matches_numpy():
    data, mask = [5, 2, 3], [0, 1, 0]
    np_res = np.ma.argmax(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmax(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res)
    assert hasattr(fr_res, "dtype")


def test_module_argmin_non_masked_unchanged():
    data, mask = [5, 2, 3], [0, 0, 0]
    np_res = np.ma.argmin(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmin(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res)


def test_module_argmin_2d_axis_returns_int_array():
    data = [[1, 2, 3], [4, 5, 6]]
    mask = [[0, 1, 0], [1, 0, 0]]
    np_res = np.ma.argmin(np.ma.array(data, mask=mask), axis=0)
    fr_res = fr.ma.argmin(fr.ma.array(data, mask=mask), axis=0)
    np.testing.assert_array_equal(np.asarray(fr_res), np.asarray(np_res))
    assert str(np.asarray(fr_res).dtype) == str(np_res.dtype)


def test_module_argmax_2d_axis_returns_int_array():
    data = [[1, 2, 3], [4, 5, 6]]
    mask = [[0, 1, 0], [1, 0, 0]]
    np_res = np.ma.argmax(np.ma.array(data, mask=mask), axis=1)
    fr_res = fr.ma.argmax(fr.ma.array(data, mask=mask), axis=1)
    np.testing.assert_array_equal(np.asarray(fr_res), np.asarray(np_res))
