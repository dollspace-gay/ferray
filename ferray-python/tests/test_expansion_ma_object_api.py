"""Expansion suite for the fr.ma.MaskedArray OBJECT API (#887 #888 #890).

Covers the bound object methods (copy/reshape/flatten/ravel/transpose/T/
astype/tolist/count(axis)), the attribute setters (.mask / .fill_value /
set_fill_value), and the `fill_value=` constructor kwarg — each asserted
against the LIVE numpy.ma oracle (R-CHAR-3): the expected value is computed
from numpy.ma in-test and ferray is required to match it. No expected value
is literal-copied from the ferray side.

Coverage spans multiple dtypes (int / float / bool / complex), multiple
shapes (1-D / 2-D / 3-D), the nomask-vs-real-mask distinction, and the
view-vs-owned receiver distinction (a method/setter invoked on a basic-index
view must produce numpy-matching VALUES).
"""

import numpy as np
import ferray as fr
import pytest


def _mask_list(a):
    """The mask of a (numpy.ma OR fr.ma) array as a nested bool list, with the
    nomask sentinel normalized to the literal string ``"nomask"`` so the two
    libraries compare on identical ground."""
    m = a.mask
    if m is np.ma.nomask or m is fr.ma.nomask:
        return "nomask"
    return m.tolist()


# ---------------------------------------------------------------------------
# Parametrization fixtures: (data, mask, dtype) triples spanning shapes/dtypes.
# ---------------------------------------------------------------------------

CASES = [
    ([1, 2, 3], [1, 0, 1], np.int64),
    ([1, 2, 3, 4], [0, 0, 0, 0], np.int64),          # real all-False mask
    ([1, 2, 3], None, np.int64),                      # nomask
    ([1.5, 2.5, 3.5], [0, 1, 0], np.float64),
    ([True, False, True], [1, 0, 0], np.bool_),
    ([[1, 2], [3, 4]], [[1, 0], [0, 1]], np.int64),
    ([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [1, 0, 1]], np.int32),
    ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
     [[[1, 0], [0, 1]], [[0, 0], [1, 1]]], np.int64),  # 3-D
    ([1 + 2j, 3 + 4j, 5 + 6j], [0, 1, 0], np.complex128),
]


def _pair(data, mask, dtype):
    if mask is None:
        return (np.ma.array(data, dtype=dtype),
                fr.ma.array(data, dtype=dtype))
    return (np.ma.array(data, mask=mask, dtype=dtype),
            fr.ma.array(data, mask=mask, dtype=dtype))


# =====================================================================
# #888 — object methods
# =====================================================================

@pytest.mark.parametrize("data,mask,dtype", CASES)
def test_copy_matches_numpy(data, mask, dtype):
    a_np, a_fr = _pair(data, mask, dtype)
    r_np, r_fr = a_np.copy(), a_fr.copy()
    assert str(r_fr.dtype) == str(r_np.dtype)
    assert _mask_list(r_fr) == _mask_list(r_np)
    assert r_fr.data.tolist() == r_np.data.tolist()


@pytest.mark.parametrize("data,mask,dtype", CASES)
def test_flatten_matches_numpy(data, mask, dtype):
    a_np, a_fr = _pair(data, mask, dtype)
    r_np, r_fr = a_np.flatten(), a_fr.flatten()
    assert str(r_fr.dtype) == str(r_np.dtype)
    assert _mask_list(r_fr) == _mask_list(r_np)
    assert r_fr.data.tolist() == r_np.data.tolist()


@pytest.mark.parametrize("data,mask,dtype", CASES)
def test_ravel_matches_numpy(data, mask, dtype):
    a_np, a_fr = _pair(data, mask, dtype)
    r_np, r_fr = a_np.ravel(), a_fr.ravel()
    assert str(r_fr.dtype) == str(r_np.dtype)
    assert _mask_list(r_fr) == _mask_list(r_np)
    assert r_fr.data.tolist() == r_np.data.tolist()


def test_reshape_varargs_and_tuple_match_numpy():
    a_np = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 1])
    a_fr = fr.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 1])
    # varargs form a.reshape(2, 2)
    assert a_fr.reshape(2, 2).mask.tolist() == a_np.reshape(2, 2).mask.tolist()
    # single-tuple form a.reshape((2, 2))
    assert a_fr.reshape((2, 2)).mask.tolist() == a_np.reshape((2, 2)).mask.tolist()
    # data preserved
    assert a_fr.reshape(4, 1).data.tolist() == a_np.reshape(4, 1).data.tolist()


def test_reshape_preserves_dtype_and_nomask():
    a_np = np.ma.array([1, 2, 3, 4], dtype=np.int32)  # nomask
    a_fr = fr.ma.array([1, 2, 3, 4], dtype=np.int32)
    r_np, r_fr = a_np.reshape(2, 2), a_fr.reshape(2, 2)
    assert str(r_fr.dtype) == str(r_np.dtype) == "int32"
    assert _mask_list(r_fr) == _mask_list(r_np) == "nomask"


@pytest.mark.parametrize("data,mask,dtype", [
    ([[1, 2], [3, 4]], [[1, 0], [0, 0]], np.int64),
    ([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [1, 0, 1]], np.float64),
    ([[1 + 1j, 2], [3, 4 + 4j]], [[1, 0], [0, 1]], np.complex128),
])
def test_transpose_and_T_match_numpy(data, mask, dtype):
    a_np, a_fr = _pair(data, mask, dtype)
    assert a_fr.transpose().mask.tolist() == a_np.transpose().mask.tolist()
    assert a_fr.T.mask.tolist() == a_np.T.mask.tolist()
    assert a_fr.T.data.tolist() == a_np.T.data.tolist()
    assert str(a_fr.T.dtype) == str(a_np.T.dtype)


def test_transpose_explicit_axes_matches_numpy():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    assert a_fr.transpose(1, 0).mask.tolist() == a_np.transpose(1, 0).mask.tolist()
    assert a_fr.transpose((1, 0)).mask.tolist() == a_np.transpose((1, 0)).mask.tolist()


@pytest.mark.parametrize("target", [np.float64, np.float32, np.int32, np.int64])
def test_astype_real_matches_numpy(target):
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 0], dtype=np.int64)
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 0], dtype=np.int64)
    r_np, r_fr = a_np.astype(target), a_fr.astype(target)
    assert str(r_fr.dtype) == str(r_np.dtype)
    assert r_fr.mask.tolist() == r_np.mask.tolist()
    assert r_fr.data.tolist() == r_np.data.tolist()


def test_astype_complex_to_real_discards_imag_like_numpy():
    """numpy casts complex->real discarding the imaginary part with a
    ComplexWarning (not an error). ferray must match the resulting real data
    and dtype, mask preserved."""
    a_np = np.ma.array([1 + 2j, 3 + 4j], mask=[1, 0], dtype=np.complex128)
    a_fr = fr.ma.array([1 + 2j, 3 + 4j], mask=[1, 0], dtype=np.complex128)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r_np, r_fr = a_np.astype(np.float64), a_fr.astype(np.float64)
    assert str(r_fr.dtype) == str(r_np.dtype) == "float64"
    assert r_fr.mask.tolist() == r_np.mask.tolist()
    assert r_fr.data.tolist() == r_np.data.tolist()


@pytest.mark.parametrize("data,mask,dtype", CASES)
def test_tolist_masked_to_none_matches_numpy(data, mask, dtype):
    a_np, a_fr = _pair(data, mask, dtype)
    assert a_fr.tolist() == a_np.tolist()


def test_count_no_axis_matches_numpy():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    assert int(a_fr.count()) == int(a_np.count())


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_count_axis_matches_numpy(axis):
    a_np = np.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 1, 0]])
    a_fr = fr.ma.array([[1, 2, 3], [4, 5, 6]], mask=[[1, 0, 0], [0, 1, 0]])
    assert a_fr.count(axis=axis).tolist() == a_np.count(axis=axis).tolist()


def test_count_axis_3d_matches_numpy():
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    mask = [[[1, 0], [0, 1]], [[0, 0], [1, 1]]]
    a_np = np.ma.array(data, mask=mask)
    a_fr = fr.ma.array(data, mask=mask)
    for axis in (0, 1, 2):
        assert a_fr.count(axis=axis).tolist() == a_np.count(axis=axis).tolist()


# =====================================================================
# #890 — attribute setters
# =====================================================================

def test_mask_setter_list_matches_numpy():
    a_np = np.ma.array([1, 2, 3]); a_np.mask = [1, 0, 1]
    a_fr = fr.ma.array([1, 2, 3]); a_fr.mask = [1, 0, 1]
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_mask_setter_true_masks_all_matches_numpy():
    a_np = np.ma.array([1, 2, 3]); a_np.mask = True
    a_fr = fr.ma.array([1, 2, 3]); a_fr.mask = True
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_mask_setter_false_gives_real_all_false_matches_numpy():
    """numpy: a.mask = False yields a REAL all-False mask (NOT nomask)."""
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 1]); a_np.mask = False
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 1]); a_fr.mask = False
    assert (a_np.mask is np.ma.nomask) == (a_fr.mask is fr.ma.nomask)
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_mask_setter_nomask_matches_numpy():
    """numpy: a.mask = nomask leaves the mask as the nomask sentinel."""
    a_np = np.ma.array([1, 2, 3]); a_np.mask = np.ma.nomask
    a_fr = fr.ma.array([1, 2, 3]); a_fr.mask = fr.ma.nomask
    assert (a_np.mask is np.ma.nomask) == (a_fr.mask is fr.ma.nomask)


def test_mask_setter_2d_matches_numpy():
    a_np = np.ma.array([[1, 2], [3, 4]]); a_np.mask = [[1, 0], [0, 1]]
    a_fr = fr.ma.array([[1, 2], [3, 4]]); a_fr.mask = [[1, 0], [0, 1]]
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_mask_setter_hardmask_adds_only_matches_numpy():
    """Under a hardmask the setter only ADDS masks (current_mask |= mask)."""
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 0]); a_np.harden_mask()
    a_np.mask = [False, True, False]
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 0]); a_fr.harden_mask()
    a_fr.mask = [False, True, False]
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_mask_setter_on_view_writes_through_matches_numpy():
    a_np = np.ma.array([1, 2, 3, 4], mask=[0, 0, 0, 0]); v_np = a_np[1:3]
    v_np.mask = [True, False]
    a_fr = fr.ma.array([1, 2, 3, 4], mask=[0, 0, 0, 0]); v_fr = a_fr[1:3]
    v_fr.mask = [True, False]
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_fill_value_setter_matches_numpy():
    a_np = np.ma.array([1, 2, 3]); a_np.fill_value = 99
    a_fr = fr.ma.array([1, 2, 3]); a_fr.fill_value = 99
    assert int(a_fr.fill_value) == int(a_np.fill_value)


def test_fill_value_setter_float_matches_numpy():
    a_np = np.ma.array([1.0, 2.0, 3.0]); a_np.fill_value = 3.5
    a_fr = fr.ma.array([1.0, 2.0, 3.0]); a_fr.fill_value = 3.5
    assert float(a_fr.fill_value) == float(a_np.fill_value)


def test_set_fill_value_method_matches_numpy():
    a_np = np.ma.array([1, 2, 3]); a_np.set_fill_value(5)
    a_fr = fr.ma.array([1, 2, 3]); a_fr.set_fill_value(5)
    assert int(a_fr.fill_value) == int(a_np.fill_value)


# =====================================================================
# #887 — fill_value= constructor kwarg
# =====================================================================

@pytest.mark.parametrize("ctor", ["array", "masked_array"])
def test_fill_value_kwarg_sets_fill(ctor):
    exp = int(getattr(np.ma, ctor)([1, 2, 3], fill_value=42).fill_value)
    got = int(getattr(fr.ma, ctor)([1, 2, 3], fill_value=42).fill_value)
    assert got == exp


def test_fill_value_kwarg_with_mask():
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 1], fill_value=7)
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 1], fill_value=7)
    assert int(a_fr.fill_value) == int(a_np.fill_value)
    assert a_fr.mask.tolist() == a_np.mask.tolist()


def test_fill_value_kwarg_float_dtype():
    a_np = np.ma.array([1.0, 2.0], fill_value=2.5)
    a_fr = fr.ma.array([1.0, 2.0], fill_value=2.5)
    assert float(a_fr.fill_value) == float(a_np.fill_value)


def test_fill_value_kwarg_carries_through_binary_op():
    """Once fill_value is settable, the #885 left-operand carry is testable."""
    a_np = np.ma.array([1, 2, 3], fill_value=7); b_np = np.ma.array([4, 5, 6])
    a_fr = fr.ma.array([1, 2, 3], fill_value=7); b_fr = fr.ma.array([4, 5, 6])
    assert int((a_fr + b_fr).fill_value) == int((a_np + b_np).fill_value)


def test_fill_value_kwarg_carries_through_slice():
    """A basic-index slice of a materialized-fill base carries the fill (#880)."""
    a_np = np.ma.array([1, 2, 3, 4], fill_value=7)
    a_fr = fr.ma.array([1, 2, 3, 4], fill_value=7)
    assert int(a_fr[1:3].fill_value) == int(a_np[1:3].fill_value)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
