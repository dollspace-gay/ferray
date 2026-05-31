"""Convergence-sweep divergence pins for fr.ma.MaskedArray vs numpy.ma.

Each test derives its EXPECTED value LIVE from numpy.ma (R-CHAR-3): the numpy
oracle is computed in-test and ferray is asserted to match it. No expected
value is literal-copied from the ferray side.

These are FAILING pins enumerating the remaining divergence set after the
fr.ma faithfulness campaign (view fill #883, repr #884, binary-op fill #885,
mask=None tri-state #886). Grouped by sweep area. A passing test means that
sub-area has converged; a failing test pins live divergence.

Owning layers:
  - binding (ferray-python/src/ma.rs): kwarg surface, method registration,
    attribute setters, mask-normalization on result construction.
  - library (ferray-ma): mask-collapse semantics, fill propagation logic.
"""

import numpy as np
import ferray as fr
import pytest


# =====================================================================
# AREA 1 — mask normalization through ops
# numpy collapses an all-False RESULT mask to nomask (scalar np.False_)
# after COMPARISON and BITWISE ufuncs, but keeps the real all-False
# array after ARITHMETIC ops. ferray keeps a real all-False array for
# comparison/bitwise too.  Layer: library (ferray-ma mask_ops / result
# construction) surfaced via binding.
# =====================================================================

def test_comparison_result_mask_collapses_to_nomask():
    """numpy: (all-False-masked == scalar).mask is nomask (scalar False)."""
    a_np = np.ma.array([1, 2, 3], mask=[False, False, False])
    a_fr = fr.ma.array([1, 2, 3], mask=[False, False, False])
    exp_is_nomask = (a_np == 1).mask is np.ma.nomask  # True (live oracle)
    got_is_nomask = (a_fr == 1).mask is fr.ma.nomask
    assert got_is_nomask == exp_is_nomask


def test_less_than_result_mask_collapses_to_nomask():
    a_np = np.ma.array([1, 2, 3], mask=[False, False, False])
    a_fr = fr.ma.array([1, 2, 3], mask=[False, False, False])
    exp_is_nomask = (a_np < 2).mask is np.ma.nomask  # True
    got_is_nomask = (a_fr < 2).mask is fr.ma.nomask
    assert got_is_nomask == exp_is_nomask


def test_bitwise_and_result_mask_collapses_to_nomask():
    a_np = np.ma.array([1, 2, 3], mask=[False, False, False])
    a_fr = fr.ma.array([1, 2, 3], mask=[False, False, False])
    exp_is_nomask = (a_np & 1).mask is np.ma.nomask  # True
    got_is_nomask = (a_fr & 1).mask is fr.ma.nomask
    assert got_is_nomask == exp_is_nomask


def test_bitwise_or_result_mask_collapses_to_nomask():
    a_np = np.ma.array([1, 2, 3], mask=[False, False, False])
    a_fr = fr.ma.array([1, 2, 3], mask=[False, False, False])
    exp_is_nomask = (a_np | 1).mask is np.ma.nomask  # True
    got_is_nomask = (a_fr | 1).mask is fr.ma.nomask
    assert got_is_nomask == exp_is_nomask


# =====================================================================
# AREA 4 — construction: fill_value= kwarg and copy semantics
# fr.ma.array signature is (data, mask=Ellipsis, dtype=None). numpy's is
# (data, dtype, copy=False, ..., fill_value=None, ...). The fill_value=
# kwarg is entirely unsupported -> TypeError. Layer: binding (ma.rs
# pyfunction signature) + library (constructor must thread fill_value).
# =====================================================================

def test_array_accepts_fill_value_kwarg():
    """numpy ma.array(data, fill_value=42) sets the fill_value."""
    exp = np.ma.array([1, 2, 3], fill_value=42).fill_value  # np.int64(42)
    got = fr.ma.array([1, 2, 3], fill_value=42).fill_value
    assert int(got) == int(exp)


def test_masked_array_accepts_fill_value_kwarg():
    exp = np.ma.masked_array([1, 2, 3], fill_value=42).fill_value
    got = fr.ma.masked_array([1, 2, 3], fill_value=42).fill_value
    assert int(got) == int(exp)


def test_array_from_maskedarray_is_view_by_default():
    """numpy ma.array(other) has copy=False -> a view; mutating one mutates both."""
    a_np = np.ma.array([1, 2, 3]); b_np = np.ma.array(a_np); b_np[0] = 99
    exp = int(a_np[0])  # 99 (view: base changed)
    a_fr = fr.ma.array([1, 2, 3]); b_fr = fr.ma.array(a_fr); b_fr[0] = 99
    got = int(a_fr[0])
    assert got == exp


# =====================================================================
# AREA 3 — fill propagation requires a settable / kwarg fill_value
# Once fill_value= is supported these pin the propagation contract.
# numpy: (a+b).fill_value takes a's fill if a was materialized.
# Layer: library (ferray-ma arithmetic fill propagation).
# =====================================================================

def test_binary_op_carries_left_fill_value():
    a_np = np.ma.array([1, 2, 3], fill_value=7)
    b_np = np.ma.array([4, 5, 6])
    exp = int((a_np + b_np).fill_value)  # 7
    a_fr = fr.ma.array([1, 2, 3], fill_value=7)
    b_fr = fr.ma.array([4, 5, 6])
    got = int((a_fr + b_fr).fill_value)
    assert got == exp


def test_slice_carries_fill_value():
    a_np = np.ma.array([1, 2, 3, 4], fill_value=7)
    exp = int(a_np[1:3].fill_value)  # 7
    a_fr = fr.ma.array([1, 2, 3, 4], fill_value=7)
    got = int(a_fr[1:3].fill_value)
    assert got == exp


# =====================================================================
# AREA 5 — .mask and .fill_value attribute setters
# numpy supports a.mask = [...] / True / False and a.fill_value = x.
# ferray makes both read-only -> AttributeError. Layer: binding (ma.rs
# #[setter]) + library (mask/fill mutation).
# =====================================================================

def test_mask_setter_list():
    a_np = np.ma.array([1, 2, 3]); a_np.mask = [1, 0, 1]
    exp = a_np.mask.tolist()  # [True, False, True]
    a_fr = fr.ma.array([1, 2, 3]); a_fr.mask = [1, 0, 1]
    got = a_fr.mask.tolist()
    assert got == exp


def test_mask_setter_scalar_true():
    a_np = np.ma.array([1, 2, 3]); a_np.mask = True
    exp = a_np.mask.tolist()  # [True, True, True]
    a_fr = fr.ma.array([1, 2, 3]); a_fr.mask = True
    got = a_fr.mask.tolist()
    assert got == exp


def test_fill_value_setter():
    a_np = np.ma.array([1, 2, 3]); a_np.fill_value = 99
    exp = int(a_np.fill_value)  # 99
    a_fr = fr.ma.array([1, 2, 3]); a_fr.fill_value = 99
    got = int(a_fr.fill_value)
    assert got == exp


# =====================================================================
# AREA 6 — methods present on numpy.ma.MaskedArray but missing on ferray.
# These are commonly used and module-level equivalents already exist in
# fr.ma; only the bound methods are absent. Layer: binding (ma.rs
# #[pymethods]) delegating to existing ferray-ma functions.
# =====================================================================

def test_method_copy_exists():
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    exp = a_np.copy().mask.tolist()  # [True, False, False]
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    got = a_fr.copy().mask.tolist()
    assert got == exp


def test_method_reshape_exists():
    a_np = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 1])
    exp = a_np.reshape(2, 2).mask.tolist()  # [[True,False],[False,True]]
    a_fr = fr.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 1])
    got = a_fr.reshape(2, 2).mask.tolist()
    assert got == exp


def test_method_flatten_exists():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 1]])
    exp = a_np.flatten().mask.tolist()  # [True,False,False,True]
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 1]])
    got = a_fr.flatten().mask.tolist()
    assert got == exp


def test_method_ravel_exists():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 1]])
    exp = a_np.ravel().mask.tolist()
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 1]])
    got = a_fr.ravel().mask.tolist()
    assert got == exp


def test_method_transpose_exists():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    exp = a_np.transpose().mask.tolist()  # [[True,False],[False,False]]
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    got = a_fr.transpose().mask.tolist()
    assert got == exp


def test_property_T_exists():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    exp = a_np.T.mask.tolist()
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    got = a_fr.T.mask.tolist()
    assert got == exp


def test_method_astype_exists_preserves_mask():
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    res_np = a_np.astype(np.float64)
    exp = (str(res_np.dtype), res_np.mask.tolist())  # ('float64',[True,False,False])
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    res_fr = a_fr.astype(np.float64)
    got = (str(res_fr.dtype), res_fr.mask.tolist())
    assert got == exp


def test_method_tolist_masked_to_none():
    a_np = np.ma.array([1, 2, 3], mask=[1, 0, 1])
    exp = a_np.tolist()  # [None, 2, None]
    a_fr = fr.ma.array([1, 2, 3], mask=[1, 0, 1])
    got = a_fr.tolist()
    assert got == exp


def test_method_count_axis_kwarg():
    a_np = np.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    exp = a_np.count(axis=0).tolist()  # [1, 2]
    a_fr = fr.ma.array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]])
    got = a_fr.count(axis=0).tolist()
    assert got == exp


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
