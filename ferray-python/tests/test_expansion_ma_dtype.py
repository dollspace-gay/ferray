"""dtype-preserving PyMaskedArray (#853 / REQ-2, refs #850).

Verifies that `ferray.ma.MaskedArray` preserves the input's native dtype across
the 11 REAL numpy dtypes (bool, i8/16/32/64, u8/16/32/64, f32/f64) at the PyO3
boundary — no silent int->f64 cast (R-CODE-4). Every expected value is derived
from the `numpy.ma` oracle (numpy 2.4.x), NOT literal-copied from ferray
(R-CHAR-3).

Out of scope (kept on the existing delegation/f64 paths, unchanged): complex
(c64/c128) and structured/datetime masked arrays.
"""

import numpy as np
import pytest

import ferray as fr

# The 11 real dtypes ferray.ma now preserves end-to-end.
REAL_DTYPES = [
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float32",
    "float64",
]


def _sample(dt):
    """A small, dtype-faithful sample array for `dt` (oracle-derived shape)."""
    if dt == "bool":
        return np.array([True, False, True], dtype=dt)
    return np.array([1, 2, 3], dtype=dt)


# ---------------------------------------------------------------------------
# Construction + .dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_construction_preserves_dtype(dt):
    data = _sample(dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    assert fa.dtype == na.dtype.name == dt


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_explicit_dtype_kwarg(dt):
    # `fr.ma.array(x, dtype=...)` casts like numpy's `array(x, dtype=...)`.
    src = [1, 0, 1]
    fa = fr.ma.array(src, dtype=np.dtype(dt))
    na = np.ma.array(src, dtype=np.dtype(dt))
    assert fa.dtype == na.dtype.name == dt


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_masked_array_alias_preserves_dtype(dt):
    data = _sample(dt)
    assert fr.ma.masked_array(data).dtype == np.ma.masked_array(data).dtype.name


# ---------------------------------------------------------------------------
# filled() egresses the native dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_filled_dtype_and_values(dt):
    data = _sample(dt)
    mask = [False, True, False]
    fa = fr.ma.array(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    fr_filled = fa.filled()
    np_filled = na.filled()
    assert fr_filled.dtype == np_filled.dtype
    np.testing.assert_array_equal(fr_filled, np_filled)


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32", "float64"])
def test_filled_explicit_value_coerced_to_dtype(dt):
    data = _sample(dt)
    mask = [False, True, False]
    fa = fr.ma.array(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    # The fill value is coerced to the array's dtype (numpy casts to self.dtype).
    fr_filled = fa.filled(7)
    np_filled = na.filled(7)
    assert fr_filled.dtype == np_filled.dtype
    np.testing.assert_array_equal(fr_filled, np_filled)


@pytest.mark.parametrize("dt", ["bool", "int64", "uint64", "float64"])
def test_default_fill_value_matches_numpy(dt):
    # Headline dtypes: ferray-ma's `default_fill_value` matches numpy's
    # per-dtype default_filler (1e20 float64 / 999999 int64 / True bool). The
    # NARROW int/float fill defaults (int8..int16, float32) are a SEPARATE
    # ferray-ma library divergence (the 999999/1e20 literal is truncated to the
    # narrow type) — out of scope for this binding build; filed as a spillover.
    data = _sample(dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    assert fa.fill_value == na.fill_value


# ---------------------------------------------------------------------------
# __array__ egress in native dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_array_protocol_native_dtype(dt):
    data = _sample(dt)
    fa = fr.ma.array(data)
    assert np.asarray(fa).dtype == np.asarray(np.ma.array(data)).dtype


# ---------------------------------------------------------------------------
# getitem / setitem preserve the native dtype
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["int64", "int32", "uint16", "float32", "bool"])
def test_getitem_scalar_and_masked_singleton(dt):
    data = _sample(dt)
    mask = [False, True, False]
    fa = fr.ma.array(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    # Unmasked scalar equals numpy's; masked element is the masked singleton.
    assert fa[0] == na[0]
    assert fa[1] is np.ma.masked


@pytest.mark.parametrize("dt", ["int64", "int32", "uint16", "float32"])
def test_getitem_slice_keeps_dtype(dt):
    data = np.array([1, 2, 3, 4], dtype=dt)
    mask = [False, True, False, True]
    fa = fr.ma.array(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    sub = fa[1:]
    nsub = na[1:]
    assert sub.dtype == nsub.dtype.name
    np.testing.assert_array_equal(np.asarray(sub), np.asarray(nsub))
    np.testing.assert_array_equal(fr.ma.getmaskarray(sub), np.ma.getmaskarray(nsub))


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32"])
def test_setitem_keeps_dtype(dt):
    data = _sample(dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    fa[0] = 9
    na[0] = 9
    assert fa.dtype == na.dtype.name
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(na))


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32"])
def test_setitem_masked_singleton(dt):
    data = _sample(dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    fa[1] = fr.ma.masked
    na[1] = np.ma.masked
    assert fa.dtype == na.dtype.name
    np.testing.assert_array_equal(fr.ma.getmaskarray(fa), np.ma.getmaskarray(na))


# ---------------------------------------------------------------------------
# put / putmask coerce values to the native dtype (no f64 round-trip)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32"])
def test_put_keeps_dtype(dt):
    data = _sample(dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    # Values typed to the array dtype so numpy's own `put` cast is well-defined
    # (oracle parity, R-CHAR-3).
    vals = np.array([7, 8], dtype=dt)
    fr.ma.put(fa, [0, 2], vals)
    na.put([0, 2], vals)
    assert fa.dtype == na.dtype.name
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(na))


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32"])
def test_putmask_keeps_dtype(dt):
    data = np.array([1, 2, 3, 4], dtype=dt)
    fa = fr.ma.array(data)
    na = np.ma.array(data)
    cond = np.array([True, False, True, False])
    vals = np.array([5, 6, 7, 8], dtype=dt)
    fr.ma.putmask(fa, cond, vals)
    np.ma.putmask(na, cond, vals)
    assert fa.dtype == na.dtype.name
    np.testing.assert_array_equal(np.asarray(fa), np.asarray(na))


# ---------------------------------------------------------------------------
# arithmetic-free reductions: count is dtype-independent and matches numpy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", REAL_DTYPES)
def test_count_matches_numpy(dt):
    data = _sample(dt)
    mask = [False, True, False]
    fa = fr.ma.array(data, mask=mask)
    na = np.ma.array(data, mask=mask)
    assert fr.ma.count(fa) == int(na.count())
    assert fr.ma.count_masked(fa) == int(np.ma.count_masked(na))


# ---------------------------------------------------------------------------
# mask round-trip + keep_mask OR across dtypes (must survive the refactor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["int64", "int32", "uint8", "float32", "bool"])
def test_mask_roundtrip_from_numpy_ma(dt):
    data = _sample(dt)
    src = np.ma.array(data, mask=[1, 0, 1])
    fa = fr.ma.array(src)
    assert fa.dtype == src.dtype.name
    np.testing.assert_array_equal(fr.ma.getmaskarray(fa), np.ma.getmaskarray(src))


@pytest.mark.parametrize("dt", ["int64", "float32"])
def test_keep_mask_or_with_explicit_mask(dt):
    data = _sample(dt)
    src = np.ma.array(data, mask=[1, 0, 0])
    # keep_mask=True default: explicit mask ORs with the source mask.
    fa = fr.ma.array(src, mask=[0, 0, 1])
    na = np.ma.array(src, mask=[0, 0, 1])
    np.testing.assert_array_equal(fr.ma.getmaskarray(fa), np.ma.getmaskarray(na))


def test_nomask_vs_explicit_all_false_preserved():
    # A plain list -> nomask sentinel; an explicit all-False mask -> real mask
    # (the #848/#849 distinction must survive the DynMa refactor).
    a = fr.ma.array([1, 2, 3])
    assert fr.ma.getmask(a) is np.ma.nomask
    b = fr.ma.array([1, 2, 3], mask=[0, 0, 0])
    np.testing.assert_array_equal(fr.ma.getmask(b), np.array([False, False, False]))


def test_int_input_filled_dtype_is_int_not_float():
    # The headline #853 divergence: fr.ma.array([1,2,3]).filled() must be int64,
    # matching numpy — NOT the old float64 cast.
    a = fr.ma.array([1, 2, 3])
    assert a.dtype == "int64"
    assert a.filled().dtype == np.ma.array([1, 2, 3]).filled().dtype == np.dtype("int64")
