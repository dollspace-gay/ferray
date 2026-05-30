"""Tests for #716 — numpy.ma masked array bindings (f64 main slice)."""

import numpy as np
import pytest

import ferray
from ferray.ma import MaskedArray


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_no_mask_defaults_to_unmasked():
    ma = MaskedArray(np.array([1.0, 2.0, 3.0]))
    assert ma.count() == 3
    np.testing.assert_array_equal(ma.mask, np.zeros(3, dtype=bool))


def test_construction_with_mask():
    ma = MaskedArray(np.array([1.0, 2.0, 3.0]), mask=np.array([False, True, False]))
    assert ma.count() == 2
    np.testing.assert_array_equal(ma.compressed(), np.array([1.0, 3.0]))


def test_construction_from_python_list():
    ma = MaskedArray([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ma.data, np.array([1.0, 2.0, 3.0]))


def test_construction_preserves_input_dtype():
    # #853 (REQ-2): the boundary preserves the input's native dtype across the
    # 11 real dtypes — NO silent int->f64 cast (R-CODE-4). Expected dtype is
    # derived from the numpy.ma oracle (R-CHAR-3).
    data = np.array([1, 2, 3], dtype=np.int64)
    ma = MaskedArray(data)
    assert ma.dtype == np.ma.array(data).dtype.name == "int64"


def test_repr_matches_numpy_ma_format():
    # #884: `repr` is byte-identical to numpy.ma's `masked_array(data=...,
    # mask=..., fill_value=...)` block (masked shown as `--`), NOT the old
    # `MaskedArray(shape=..., masked=..., dtype=...)` debug string that pinned
    # a non-numpy format. Expected value is the live numpy.ma repr (R-CHAR-3).
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, True, False])
    ma = MaskedArray(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    assert repr(ma) == repr(na)
    assert str(ma) == str(na)


def test_array_alias_for_masked_array():
    a = ferray.ma.array(np.array([1.0, 2.0]))
    b = ferray.ma.masked_array(np.array([1.0, 2.0]))
    np.testing.assert_array_equal(a.data, b.data)


# ---------------------------------------------------------------------------
# masked_where / masked_invalid
# ---------------------------------------------------------------------------


def test_masked_where_basic():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cond = np.array([True, False, True, False, True])
    ma = ferray.ma.masked_where(cond, data)
    np.testing.assert_array_equal(ma.compressed(), np.array([2.0, 4.0]))


def test_masked_invalid_handles_nan_and_inf():
    data = np.array([1.0, np.nan, 3.0, np.inf, -np.inf, 5.0])
    ma = ferray.ma.masked_invalid(data)
    assert ma.count() == 3
    np.testing.assert_array_equal(ma.compressed(), np.array([1.0, 3.0, 5.0]))


# ---------------------------------------------------------------------------
# masked_<comparison> family
# ---------------------------------------------------------------------------


def test_masked_equal():
    data = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    ma = ferray.ma.masked_equal(data, 2.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([1.0, 3.0, 1.0]))


def test_masked_not_equal():
    data = np.array([1.0, 2.0, 3.0])
    ma = ferray.ma.masked_not_equal(data, 2.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([2.0]))


def test_masked_greater():
    data = np.array([1.0, 5.0, 10.0, 3.0])
    ma = ferray.ma.masked_greater(data, 4.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([1.0, 3.0]))


def test_masked_less():
    data = np.array([1.0, 5.0, 10.0, 3.0])
    ma = ferray.ma.masked_less(data, 4.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([5.0, 10.0]))


def test_masked_inside():
    data = np.array([0.0, 5.0, 10.0, 15.0])
    ma = ferray.ma.masked_inside(data, 4.0, 11.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([0.0, 15.0]))


def test_masked_outside():
    data = np.array([0.0, 5.0, 10.0, 15.0])
    ma = ferray.ma.masked_outside(data, 4.0, 11.0)
    np.testing.assert_array_equal(ma.compressed(), np.array([5.0, 10.0]))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_sum_ignores_masked():
    data = np.array([1.0, 2.0, 100.0, 4.0])
    ma = ferray.ma.masked_array(data, mask=np.array([False, False, True, False]))
    assert ma.sum() == pytest.approx(7.0)


def test_mean_ignores_masked():
    data = np.array([1.0, 2.0, 100.0, 4.0])
    ma = ferray.ma.masked_array(data, mask=np.array([False, False, True, False]))
    # mean of [1, 2, 4] = 7/3
    assert ma.mean() == pytest.approx(7.0 / 3.0)


def test_min_max_ignore_masked():
    data = np.array([1.0, 0.5, 100.0, 4.0])
    ma = ferray.ma.masked_array(data, mask=np.array([False, False, True, False]))
    assert ma.min() == pytest.approx(0.5)
    assert ma.max() == pytest.approx(4.0)


def test_var_std_ignore_masked():
    data = np.array([1.0, 2.0, 100.0, 3.0, 4.0])
    mask = np.array([False, False, True, False, False])
    ma = ferray.ma.masked_array(data, mask=mask)
    expected = np.array([1.0, 2.0, 3.0, 4.0])
    assert ma.var() == pytest.approx(float(np.var(expected)))
    assert ma.std() == pytest.approx(float(np.std(expected)))


def test_sum_axis_returns_masked_array():
    data = np.arange(12.0).reshape(3, 4)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    ma = ferray.ma.masked_array(data, mask=mask)
    out = ma.sum_axis(0)
    assert isinstance(out, MaskedArray)
    assert out.shape == (4,)


def test_count_full():
    ma = MaskedArray(
        np.array([1.0, 2.0, 3.0, 4.0]),
        mask=np.array([False, True, True, False]),
    )
    assert ma.count() == 2


# ---------------------------------------------------------------------------
# filled / compressed / mask access
# ---------------------------------------------------------------------------


def test_filled_with_explicit_value():
    data = np.array([1.0, 2.0, 3.0])
    ma = ferray.ma.masked_array(data, mask=np.array([False, True, False]))
    out = ma.filled(-99.0)
    np.testing.assert_array_equal(out, np.array([1.0, -99.0, 3.0]))


def test_filled_default_uses_default_fill():
    data = np.array([1.0, 2.0, 3.0])
    ma = ferray.ma.masked_array(data, mask=np.array([False, True, False]))
    out = ma.filled()
    # Default float64 fill value matches numpy's (1e20).
    assert out[0] == 1.0
    assert out[2] == 3.0
    # Position 1 has whatever the fill is — just verify it's a real number.
    assert np.isfinite(out[1])


def test_compressed_returns_unmasked_only():
    data = np.array([1.0, 2.0, 3.0, 4.0])
    ma = ferray.ma.masked_array(data, mask=np.array([True, False, True, False]))
    np.testing.assert_array_equal(ma.compressed(), np.array([2.0, 4.0]))


def test_mask_property_returns_bool_ndarray():
    ma = MaskedArray(
        np.array([1.0, 2.0, 3.0]),
        mask=np.array([False, True, False]),
    )
    m = ma.mask
    assert m.dtype == np.bool_
    np.testing.assert_array_equal(m, [False, True, False])


def test_data_property_is_numpy_ndarray():
    ma = MaskedArray(
        np.array([1.0, 2.0, 3.0]),
        mask=np.array([False, True, False]),
    )
    d = ma.data
    assert isinstance(d, np.ndarray)
    assert d.dtype == np.float64
    np.testing.assert_array_equal(d, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def test_module_count_masked():
    ma = MaskedArray(
        np.array([1.0, 2.0, 3.0, 4.0]),
        mask=np.array([True, False, True, False]),
    )
    assert ferray.ma.count_masked(ma) == 2


def test_module_is_masked_true():
    ma = MaskedArray(np.array([1.0, 2.0]), mask=np.array([False, True]))
    assert ferray.ma.is_masked(ma) is True


def test_module_is_masked_false_when_no_masks():
    ma = MaskedArray(np.array([1.0, 2.0]))
    assert ferray.ma.is_masked(ma) is False


def test_module_getmask_getdata():
    ma = MaskedArray(np.array([1.0, 2.0, 3.0]), mask=np.array([False, True, False]))
    np.testing.assert_array_equal(
        ferray.ma.getdata(ma), np.array([1.0, 2.0, 3.0])
    )
    np.testing.assert_array_equal(
        ferray.ma.getmask(ma), np.array([False, True, False])
    )


def test_array_protocol_works_with_numpy_asarray():
    ma = MaskedArray(np.array([1.0, 2.0, 3.0]), mask=np.array([False, False, False]))
    arr = np.asarray(ma)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# 2-D shapes
# ---------------------------------------------------------------------------


def test_2d_construction_and_count():
    data = np.arange(12.0).reshape(3, 4)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 2] = True
    ma = MaskedArray(data, mask=mask)
    assert ma.count() == 10
    assert ma.shape == (3, 4)


def test_2d_filled():
    data = np.arange(6.0).reshape(2, 3)
    mask = np.array([[True, False, False], [False, True, False]])
    ma = MaskedArray(data, mask=mask)
    out = ma.filled(-1.0)
    np.testing.assert_array_equal(out, np.array([[-1.0, 1.0, 2.0], [3.0, -1.0, 5.0]]))
