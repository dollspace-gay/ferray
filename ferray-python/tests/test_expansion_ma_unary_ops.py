"""Tests for #861 (REQ-4, R-D) — ferray.ma.MaskedArray unary operators.

`__neg__` (`-a`), `__pos__` (`+a`), `__abs__` (`abs(a)`), `__invert__` (`~a`)
on a masked array, mask-preserving and dtype-faithful, verified against the
LIVE numpy.ma oracle (R-CHAR-3): every expected value is produced by calling
the same operator on a `numpy.ma.MaskedArray` built from the same data/mask —
NEVER literal-copied from ferray. numpy 2.4.x is the oracle.

Mirrors numpy's `_MaskedUnaryOperation` (`numpy/ma/core.py:955`/`:1010`):
the op transforms the FULL data buffer (masked positions included) and the
result carries a REAL mask materialized from `getmaskarray(a)` — so even a
nomask operand yields a real all-False result mask (`getmask(-n) is nomask`
is `False`).
"""

import numpy as np
import pytest

from ferray.ma import MaskedArray

# The 11 real dtypes the DynMa binding carries.
SIGNED = ["int8", "int16", "int32", "int64"]
UNSIGNED = ["uint8", "uint16", "uint32", "uint64"]
FLOAT = ["float32", "float64"]
INTEGER = SIGNED + UNSIGNED


def _fr(data, mask, dtype):
    """A ferray masked array of the given dtype (native, no f64 collapse)."""
    return MaskedArray(np.asarray(data, dtype=dtype), mask=np.asarray(mask, dtype=bool))


def _np(data, mask, dtype):
    return np.ma.array(np.asarray(data, dtype=dtype), mask=np.asarray(mask, dtype=bool))


def _assert_like_numpy(got, want):
    """`got` is a ferray MaskedArray, `want` a numpy.ma — compare data, mask,
    dtype exactly (including masked-position data, which numpy transforms)."""
    assert got.dtype == want.dtype.name, f"dtype {got.dtype!r} != {want.dtype.name!r}"
    np.testing.assert_array_equal(got.data, np.ma.getdata(want))
    np.testing.assert_array_equal(
        np.asarray(got.mask, dtype=bool), np.ma.getmaskarray(want)
    )


# ---------------------------------------------------------------------------
# __neg__  (-a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SIGNED + FLOAT)
def test_neg_signed_and_float(dtype):
    data, mask = [-1, 2, -3], [0, 1, 0]
    _assert_like_numpy(-_fr(data, mask, dtype), -_np(data, mask, dtype))


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_neg_unsigned_wraps_c_semantics(dtype):
    # numpy negates unsigned with C two's-complement wrap (-uint8(2) == 254).
    data, mask = [1, 2, 0], [1, 0, 0]
    _assert_like_numpy(-_fr(data, mask, dtype), -_np(data, mask, dtype))


def test_neg_masked_position_data_is_transformed():
    # numpy transforms the data under the mask: (-a).data == [1, -2, 3].
    a = _fr([-1, 2, -3], [0, 1, 0], "int8")
    expected = (-_np([-1, 2, -3], [0, 1, 0], "int8")).data
    np.testing.assert_array_equal((-a).data, expected)
    np.testing.assert_array_equal((-a).data, np.array([1, -2, 3], dtype=np.int8))


def test_neg_bool_raises_typeerror_like_numpy():
    a = _fr([True, False], [1, 0], "bool")
    with pytest.raises(TypeError):
        -a
    # confirm numpy raises the same family.
    with pytest.raises(TypeError):
        -_np([True, False], [1, 0], "bool")


# ---------------------------------------------------------------------------
# __pos__  (+a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", INTEGER + FLOAT)
def test_pos_is_mask_preserving_copy(dtype):
    data, mask = [-1, 2, -3] if dtype in SIGNED + FLOAT else [1, 2, 3], [0, 1, 0]
    _assert_like_numpy(+_fr(data, mask, dtype), +_np(data, mask, dtype))


def test_pos_bool_raises_typeerror_like_numpy():
    a = _fr([True, False], [1, 0], "bool")
    with pytest.raises(TypeError):
        +a
    with pytest.raises(TypeError):
        +_np([True, False], [1, 0], "bool")


# ---------------------------------------------------------------------------
# __abs__  (abs(a))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SIGNED + FLOAT)
def test_abs_signed_and_float(dtype):
    data, mask = [-1, 2, -3], [0, 1, 0]
    _assert_like_numpy(abs(_fr(data, mask, dtype)), abs(_np(data, mask, dtype)))


@pytest.mark.parametrize("dtype", UNSIGNED)
def test_abs_unsigned_is_identity(dtype):
    data, mask = [1, 2, 3], [0, 1, 0]
    _assert_like_numpy(abs(_fr(data, mask, dtype)), abs(_np(data, mask, dtype)))


def test_abs_bool_is_identity_bool():
    # abs(bool) stays bool (abs(True)==True), verified live.
    data, mask = [True, False], [1, 0]
    _assert_like_numpy(abs(_fr(data, mask, "bool")), abs(_np(data, mask, "bool")))


def test_abs_masked_position_data_is_transformed():
    a = _fr([-1, 2, -3], [0, 1, 0], "int16")
    expected = abs(_np([-1, 2, -3], [0, 1, 0], "int16")).data
    np.testing.assert_array_equal(abs(a).data, expected)
    np.testing.assert_array_equal(abs(a).data, np.array([1, 2, 3], dtype=np.int16))


# ---------------------------------------------------------------------------
# __invert__  (~a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", INTEGER)
def test_invert_integer(dtype):
    # signed: two's-complement (~int8(-1)==0); unsigned: bit-flip.
    data = [1, 2, 3] if dtype in UNSIGNED else [-1, 2, -3]
    mask = [1, 0, 0]
    _assert_like_numpy(~_fr(data, mask, dtype), ~_np(data, mask, dtype))


def test_invert_bool_flips():
    data, mask = [True, False], [1, 0]
    _assert_like_numpy(~_fr(data, mask, "bool"), ~_np(data, mask, "bool"))


@pytest.mark.parametrize("dtype", FLOAT)
def test_invert_float_raises_typeerror_like_numpy(dtype):
    a = _fr([1.0, 2.0], [0, 1], dtype)
    with pytest.raises(TypeError):
        ~a
    with pytest.raises(TypeError):
        ~_np([1.0, 2.0], [0, 1], dtype)


def test_invert_masked_position_data_is_transformed():
    a = _fr([-1, 2, -3], [0, 1, 0], "int8")
    expected = (~_np([-1, 2, -3], [0, 1, 0], "int8")).data
    np.testing.assert_array_equal((~a).data, expected)
    np.testing.assert_array_equal((~a).data, np.array([0, -3, 2], dtype=np.int8))


# ---------------------------------------------------------------------------
# nomask preservation contract (numpy materializes a REAL all-False mask)
# ---------------------------------------------------------------------------


def test_nomask_operand_yields_real_all_false_mask_like_numpy():
    # A nomask operand: numpy's _MaskedUnaryOperation materializes
    # getmaskarray(a) -> a real all-False mask, NOT the nomask singleton.
    a = MaskedArray(np.array([1, 2, 3], dtype=np.int16))  # no mask= -> nomask
    n = np.ma.array(np.array([1, 2, 3], dtype=np.int16))
    for fr_res, np_res in [(-a, -n), (+a, +n), (abs(a), abs(n)), (~a, ~n)]:
        # numpy: getmask(result) is NOT nomask (a real bool array).
        assert np.ma.getmask(np_res) is not np.ma.nomask
        np.testing.assert_array_equal(
            np.asarray(fr_res.mask, dtype=bool), np.zeros(3, dtype=bool)
        )
        _assert_like_numpy(fr_res, np_res)


def test_real_mask_carried_verbatim():
    a = _fr([10, 20, 30, 40], [0, 1, 1, 0], "int32")
    n = _np([10, 20, 30, 40], [0, 1, 1, 0], "int32")
    for fr_res, np_res in [(-a, -n), (+a, +n), (abs(a), abs(n)), (~a, ~n)]:
        _assert_like_numpy(fr_res, np_res)
