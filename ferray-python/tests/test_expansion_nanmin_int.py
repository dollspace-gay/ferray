"""#950 — nanmin/nanmax/nanprod preserve the integer/bool dtype per numpy.

Integers and bool have no NaN, so `nanmin == min`, `nanmax == max`,
`nanprod == prod` for those dtypes, and numpy keeps the native dtype rather
than coercing to float64. The prior `bind_nan_reduction!` macro coerced every
non-float input to float64, so `fr.nanmin([3,1,2])` returned `1.0` (float64)
instead of `1` (int64). Expected values come from live numpy 2.4.x (R-CHAR-3).

nanmean stays float64 (numpy `nanmean` is always float) — guarded here so the
fix does not regress it. The float NaN-skip path is unchanged.
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
def test_nanmin_nanmax_preserve_int_dtype(dt):
    a = np.array([3, 1, 2], dtype=dt)
    for name in ("nanmin", "nanmax"):
        fr_r = getattr(fr, name)(a)
        np_r = getattr(np, name)(a)
        assert str(fr_r.dtype) == str(np_r.dtype), name
        assert int(fr_r) == int(np_r), name


@pytest.mark.parametrize("dt", INT_DTYPES)
def test_nanprod_int_accumulator_dtype(dt):
    # numpy upcasts the prod accumulator: int8/16/32 -> int64, uint* -> uint64.
    a = np.array([3, 1, 2], dtype=dt)
    fr_r = fr.nanprod(a)
    np_r = np.nanprod(a)
    assert str(fr_r.dtype) == str(np_r.dtype)
    assert int(fr_r) == int(np_r)


def test_nanmin_nanmax_bool_dtype():
    b = np.array([True, False, True])
    assert str(fr.nanmin(b).dtype) == str(np.nanmin(b).dtype) == "bool"
    assert bool(fr.nanmin(b)) == bool(np.nanmin(b))
    assert str(fr.nanmax(b).dtype) == str(np.nanmax(b).dtype) == "bool"
    assert bool(fr.nanmax(b)) == bool(np.nanmax(b))


def test_nanprod_bool_dtype():
    b = np.array([True, False, True])
    # numpy nanprod(bool) -> int64.
    assert str(fr.nanprod(b).dtype) == str(np.nanprod(b).dtype) == "int64"
    assert int(fr.nanprod(b)) == int(np.nanprod(b))


def test_float_nan_skip_unchanged():
    f = np.array([3.0, np.nan, 1.0])
    assert float(fr.nanmin(f)) == float(np.nanmin(f)) == 1.0
    assert str(fr.nanmin(f).dtype) == str(np.nanmin(f).dtype) == "float64"
    assert float(fr.nanmax(f)) == float(np.nanmax(f)) == 3.0
    f32 = f.astype("float32")
    assert str(fr.nanmin(f32).dtype) == str(np.nanmin(f32).dtype) == "float32"


def test_nanmean_int_bool_stays_float64():
    # nanmean is always float64 in numpy — must NOT be touched by this fix.
    a = np.array([3, 1, 2], dtype="int64")
    assert str(fr.nanmean(a).dtype) == str(np.nanmean(a).dtype) == "float64"
    b = np.array([True, False, True])
    assert str(fr.nanmean(b).dtype) == str(np.nanmean(b).dtype) == "float64"


def test_axis_variants_preserve_dtype():
    m = np.array([[3, 1, 2], [6, 5, 4]], dtype="int32")
    assert str(fr.nanmin(m, axis=0).dtype) == str(np.nanmin(m, axis=0).dtype)
    assert fr.nanmin(m, axis=0).tolist() == np.nanmin(m, axis=0).tolist()
    assert str(fr.nanmax(m, axis=1).dtype) == str(np.nanmax(m, axis=1).dtype)
    assert fr.nanmax(m, axis=1).tolist() == np.nanmax(m, axis=1).tolist()
    assert str(fr.nanprod(m, axis=1).dtype) == str(np.nanprod(m, axis=1).dtype)
    assert fr.nanprod(m, axis=1).tolist() == np.nanprod(m, axis=1).tolist()
