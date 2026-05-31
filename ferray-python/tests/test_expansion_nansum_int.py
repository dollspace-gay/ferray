"""nansum preserves the native integer/bool accumulator dtype, not float64 (#957).

Integers and bool have no NaN, so `nansum == sum`, and numpy keeps the `sum`
accumulator dtype (int8/16/32->int64, uint*->uint64, bool->int64) — it does NOT
coerce to float64. Cross-checked live against numpy 2.4.x (the oracle), the same
upcast as `np.sum` (#858) and mirroring the #950 nanmin/nanmax/nanprod
native-dtype path. Float input with NaN keeps the NaN-skipping float path
(unchanged). Reference: numpy/lib/_nanfunctions_impl.py:725 — nansum is
`_replace_nan(a, 0)` then `np.sum(..., dtype=None)`.
"""

import numpy as np
import pytest

import ferray as fr

INT_DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_nansum_int_preserves_accumulator_dtype(dt):
    data = [1, 2, 3, 4]
    fr_r = fr.nansum(np.array(data, dtype=dt))
    np_r = np.nansum(np.array(data, dtype=dt))
    assert np.asarray(fr_r).dtype == np_r.dtype, (
        f"{dt}: ferray dtype {np.asarray(fr_r).dtype} != numpy {np_r.dtype}"
    )
    assert int(np.asarray(fr_r)) == int(np_r) == 10


def test_nansum_bool_accumulator_is_int64():
    data = [True, False, True, True]
    fr_r = fr.nansum(np.array(data, dtype="bool"))
    np_r = np.nansum(np.array(data, dtype="bool"))
    assert np.asarray(fr_r).dtype == np_r.dtype == np.dtype("int64")
    assert int(np.asarray(fr_r)) == int(np_r) == 3


@pytest.mark.parametrize("dt", INT_DTYPES + ["bool"])
def test_nansum_int_axis_preserves_dtype(dt):
    src = [[1, 0, 1], [1, 1, 0]] if dt == "bool" else [[1, 2, 3], [4, 5, 6]]
    a = np.array(src, dtype=dt)
    fr_r = np.asarray(fr.nansum(a, axis=0))
    np_r = np.nansum(a, axis=0)
    assert fr_r.dtype == np_r.dtype
    np.testing.assert_array_equal(fr_r, np_r)


def test_nansum_float_nan_skip_unchanged():
    a = np.array([1.0, np.nan, 3.0])
    fr_r = np.asarray(fr.nansum(a))
    np_r = np.nansum(a)
    assert fr_r.dtype == np_r.dtype == np.dtype("float64")
    assert float(fr_r) == float(np_r) == 4.0


def test_nansum_float32_nan_skip_unchanged():
    a = np.array([1.0, np.nan, 3.0], dtype="float32")
    fr_r = np.asarray(fr.nansum(a))
    np_r = np.nansum(a)
    assert fr_r.dtype == np_r.dtype == np.dtype("float32")
    assert float(fr_r) == float(np_r) == 4.0


def test_nanmean_int_stays_float64_unchanged():
    # Guard: the #957 fix must NOT leak into nanmean (always float64).
    a = np.array([1, 2, 3], dtype="int8")
    fr_r = np.asarray(fr.nanmean(a))
    np_r = np.nanmean(a)
    assert fr_r.dtype == np_r.dtype == np.dtype("float64")
    assert float(fr_r) == float(np_r)
