"""FINAL convergence-sweep divergence pins for fr.ma.MaskedArray vs numpy.ma.

HEAD b89a90c7. Each test derives its EXPECTED value live from numpy.ma
(R-CHAR-3 — never literal-copied from the ferray side) and FAILS against the
current ferray build. Grouped by divergence class.

Classes:
  - binding   : fixable in ferray-python/src/ma.rs (method/attr/kwarg surface)
  - library   : root cause in ferray-ma (dtype preservation)
  - arch-limit: cannot fix without foreign-buffer sharing

Tracking issues: #892 (object methods), #893 (axis/keepdims kwargs),
#894 (int->float upcast), #895 (attributes), #896 (ctor view), #897 (where
scalar broadcast), #898 (arch-limit foreign-buffer view).
"""

import numpy as np
import ferray as fr
import pytest


# ---------------------------------------------------------------------------
# GROUP 1 — missing MaskedArray object methods (binding; tracking #892)
# numpy.ma.MaskedArray exposes these as bound methods; ferray lacks them.
# ---------------------------------------------------------------------------

def _ma2():
    data = [[1, 2, 3], [4, 5, 6]]
    mask = [[0, 1, 0], [0, 0, 1]]
    return fr.ma.array(data, mask=mask), np.ma.array(data, mask=mask)


def test_method_all_binding_892():
    fa, na = _ma2()
    assert bool(fa.all()) == bool(na.all())


def test_method_any_binding_892():
    fa, na = _ma2()
    assert bool(fa.any()) == bool(na.any())


def test_method_take_binding_892():
    fa, na = _ma2()
    assert fa.take([0, 2]).tolist() == na.take([0, 2]).tolist()


def test_method_squeeze_binding_892():
    f = fr.ma.array([[1, 2, 3]])
    n = np.ma.array([[1, 2, 3]])
    assert f.squeeze().tolist() == n.squeeze().tolist()


def test_method_ptp_binding_892():
    fa, na = _ma2()
    assert fa.ptp() == na.ptp()


def test_method_nonzero_binding_892():
    f = fr.ma.array([0, 2, 0, 3], mask=[0, 0, 0, 0])
    n = np.ma.array([0, 2, 0, 3], mask=[0, 0, 0, 0])
    assert [x.tolist() for x in f.nonzero()] == [x.tolist() for x in n.nonzero()]


def test_method_argmin_binding_892():
    fa, na = _ma2()
    assert fa.argmin() == na.argmin()


def test_method_argmax_binding_892():
    fa, na = _ma2()
    assert fa.argmax() == na.argmax()


def test_method_prod_binding_892():
    fa, na = _ma2()
    assert fa.prod() == na.prod()


def test_method_cumsum_binding_892():
    fa, na = _ma2()
    # masked positions render as masked; compare filled views + masks
    assert fa.cumsum().filled(0).tolist() == na.cumsum().filled(0).tolist()


def test_method_cumprod_binding_892():
    f = fr.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 0])
    n = np.ma.array([1, 2, 3, 4], mask=[0, 1, 0, 0])
    assert f.cumprod().filled(1).tolist() == n.cumprod().filled(1).tolist()


def test_method_clip_binding_892():
    fa, na = _ma2()
    assert fa.clip(2, 5).filled(0).tolist() == na.clip(2, 5).filled(0).tolist()


def test_method_round_binding_892():
    f = fr.ma.array([1.4, 2.6, 3.5])
    n = np.ma.array([1.4, 2.6, 3.5])
    assert f.round().tolist() == n.round().tolist()


def test_method_swapaxes_binding_892():
    fa, na = _ma2()
    assert fa.swapaxes(0, 1).shape == na.swapaxes(0, 1).shape


def test_method_trace_binding_892():
    fa, na = _ma2()
    assert fa.trace() == na.trace()


def test_method_diagonal_binding_892():
    fa, na = _ma2()
    assert fa.diagonal().tolist() == na.diagonal().tolist()


def test_method_item_binding_892():
    f = fr.ma.array([5])
    n = np.ma.array([5])
    assert f.item() == n.item()


def test_method_repeat_binding_892():
    f = fr.ma.array([1, 2], mask=[0, 1])
    n = np.ma.array([1, 2], mask=[0, 1])
    assert f.repeat(2).filled(0).tolist() == n.repeat(2).filled(0).tolist()


def test_method_sort_binding_892():
    f = fr.ma.array([3, 1, 2], mask=[0, 1, 0])
    n = np.ma.array([3, 1, 2], mask=[0, 1, 0])
    n.sort()
    fs = fr.ma.array([3, 1, 2], mask=[0, 1, 0])
    fs.sort()
    assert fs.filled(0).tolist() == n.filled(0).tolist()


def test_method_argsort_binding_892():
    f = fr.ma.array([3, 1, 2], mask=[0, 1, 0])
    n = np.ma.array([3, 1, 2], mask=[0, 1, 0])
    assert f.argsort().tolist() == n.argsort().tolist()


# ---------------------------------------------------------------------------
# GROUP 2 — reduction methods reject axis=/keepdims=/dtype= (binding; #893)
# numpy.ma min/max/sum/mean/std accept these kwargs; ferray only has *_axis.
# ---------------------------------------------------------------------------

def test_min_axis_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.min(axis=0).tolist() == na.min(axis=0).tolist()


def test_max_axis_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.max(axis=1).tolist() == na.max(axis=1).tolist()


def test_sum_axis_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.sum(axis=0).tolist() == na.sum(axis=0).tolist()


def test_sum_keepdims_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.sum(keepdims=True).shape == na.sum(keepdims=True).shape


def test_mean_axis_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.mean(axis=0).tolist() == na.mean(axis=0).tolist()


def test_mean_dtype_kwarg_binding_893():
    fa, na = _ma2()
    assert str(fa.mean(dtype="float32").dtype) == str(na.mean(dtype="float32").dtype)


def test_std_axis_kwarg_binding_893():
    fa, na = _ma2()
    assert fa.std(axis=0).filled(0).tolist() == na.std(axis=0).filled(0).tolist()


# ---------------------------------------------------------------------------
# GROUP 3 — module fns upcast integer input to float64 (library; #894)
# numpy.ma sort/clip/dot/where preserve int dtype; ferray returns float64.
# ---------------------------------------------------------------------------

def test_module_sort_preserves_int_dtype_library_894():
    f = fr.ma.sort(fr.ma.array([3, 1, 2], mask=[0, 1, 0]))
    n = np.ma.sort(np.ma.array([3, 1, 2], mask=[0, 1, 0]))
    assert str(f.dtype) == str(n.dtype)


def test_module_clip_preserves_int_dtype_library_894():
    f = fr.ma.clip(fr.ma.array([1, 5, 2]), 2, 4)
    n = np.ma.clip(np.ma.array([1, 5, 2]), 2, 4)
    assert str(f.dtype) == str(n.dtype)


def test_module_dot_preserves_int_dtype_library_894():
    f = fr.ma.dot(fr.ma.array([[1, 2]]), fr.ma.array([[3], [4]]))
    n = np.ma.dot(np.ma.array([[1, 2]]), np.ma.array([[3], [4]]))
    assert str(f.dtype) == str(n.dtype)


def test_module_where_preserves_int_dtype_library_894():
    cond = np.array([True, False])
    f = fr.ma.where(cond, np.array([1, 1]), np.array([2, 2]))
    n = np.ma.where(cond, np.array([1, 1]), np.array([2, 2]))
    assert str(f.dtype) == str(n.dtype)


# ---------------------------------------------------------------------------
# GROUP 4 — missing MaskedArray attributes (binding; #895)
# numpy.ma exposes itemsize/nbytes/strides/flat/base; ferray lacks them.
# ---------------------------------------------------------------------------

def test_attr_itemsize_binding_895():
    f = fr.ma.array([1.0, 2.0, 3.0])
    n = np.ma.array([1.0, 2.0, 3.0])
    assert f.itemsize == n.itemsize


def test_attr_nbytes_binding_895():
    f = fr.ma.array([1.0, 2.0, 3.0])
    n = np.ma.array([1.0, 2.0, 3.0])
    assert f.nbytes == n.nbytes


def test_attr_strides_binding_895():
    f = fr.ma.array([[1.0, 2.0], [3.0, 4.0]])
    n = np.ma.array([[1.0, 2.0], [3.0, 4.0]])
    assert tuple(f.strides) == tuple(n.strides)


def test_attr_flat_iter_binding_895():
    f = fr.ma.array([1, 2, 3], mask=[0, 1, 0])
    n = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    fvals = [(x is fr.ma.masked) for x in f.flat]
    nvals = [(x is np.ma.masked) for x in n.flat]
    assert fvals == nvals


def test_attr_base_binding_895():
    # numpy: a base array's .base is None; a view's .base is the parent.
    n = np.ma.array([1, 2, 3])
    f = fr.ma.array([1, 2, 3])
    assert (f.base is None) == (n.base is None)


# ---------------------------------------------------------------------------
# GROUP 5 — constructor view semantics (binding; #896)
# numpy ma.array(existing_ma, mask=/fill_value=) shares base _data.
# ---------------------------------------------------------------------------

def test_ctor_mask_kwarg_is_view_binding_896():
    nbase = np.ma.array([1, 2, 3, 4])
    nview = np.ma.array(nbase, mask=[1, 0, 0, 0])
    nview.data[1] = 99
    np_shared = nbase.data[1]  # numpy contract: 99 (shared)

    fbase = fr.ma.array([1, 2, 3, 4])
    fview = fr.ma.array(fbase, mask=[1, 0, 0, 0])
    fview.data[1] = 99
    assert fbase.data[1] == np_shared


def test_ctor_fill_value_kwarg_is_view_binding_896():
    nbase = np.ma.array([1, 2, 3, 4])
    nview = np.ma.array(nbase, fill_value=7)
    nview.data[2] = 88
    np_shared = nbase.data[2]  # numpy contract: 88 (shared)

    fbase = fr.ma.array([1, 2, 3, 4])
    fview = fr.ma.array(fbase, fill_value=7)
    fview.data[2] = 88
    assert fbase.data[2] == np_shared


# ---------------------------------------------------------------------------
# GROUP 6 — fr.ma.where scalar broadcasting (binding; #897)
# numpy.ma.where broadcasts scalar x/y against an array condition.
# ---------------------------------------------------------------------------

def test_where_scalar_broadcast_binding_897():
    cond = np.array([True, False])
    n = np.ma.where(cond, 1, 2)
    f = fr.ma.where(cond, 1, 2)
    assert f.tolist() == n.tolist()


# ---------------------------------------------------------------------------
# GROUP 7 — foreign-buffer view (ARCHITECTURAL-LIMIT; #898)
# numpy ma.array(ndarray) shares the foreign ndarray buffer; ferray copies.
# Cannot fix without foreign-buffer sharing across the PyO3 boundary.
# ---------------------------------------------------------------------------

def test_ctor_ndarray_is_view_arch_limit_898():
    nnd = np.array([1, 2, 3])
    nm = np.ma.array(nnd)
    nm.data[0] = 77
    np_shared = nnd[0]  # numpy contract: 77 (shared foreign buffer)

    fnd = np.array([1, 2, 3])
    fm = fr.ma.array(fnd)
    fm.data[0] = 77
    assert fnd[0] == np_shared
