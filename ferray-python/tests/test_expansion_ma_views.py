"""ferray.ma basic-index VIEW semantics vs numpy.ma (#857, refs #850).

numpy.ma.MaskedArray basic indexing (int-row / slice / ellipsis / newaxis /
int-slice tuples) returns a VIEW that shares the parent's data+mask buffer:
a write through the sub-array propagates back to the parent, and a read through
it reflects the parent's current state (R-DEV-1 view-vs-copy contract).
Advanced indexing (integer-array, boolean-mask, list-of-indices) returns a COPY
— writes do NOT propagate. ferray must match numpy for BOTH.

Every expected value is constructed by running the SAME operation against
``numpy.ma`` (the live oracle, numpy 2.4.x) — never literal-copied from the
ferray side (R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Helpers: run an identical scenario against numpy.ma and ferray.ma, compare.
# ---------------------------------------------------------------------------


def _both(make):
    """Return (numpy_result, ferray_result) of ``make(module_array_ctor)``."""
    return make(np.ma), make(fr.ma)


# ---------------------------------------------------------------------------
# 1-D basic slice: writeback (the #857 pin scenario, generalized).
# ---------------------------------------------------------------------------


def test_slice_writeback_propagates_to_parent():
    def scenario(ma):
        a = ma.array([1.0, 2, 3, 4])
        b = a[1:3]
        b[0] = 99.0
        return float(a[1]), float(b[0])

    assert _both(scenario)[0] == _both(scenario)[1]
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    b[0] = 99.0
    # numpy oracle (live): the parent sees the write.
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    bn[0] = 99.0
    assert float(a[1]) == float(an[1]) == 99.0
    assert float(b[0]) == float(bn[0]) == 99.0


def test_slice_view_read_reflects_base_mutation():
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    a[2] = 50.0  # mutate the PARENT
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    an[2] = 50.0
    assert float(b[1]) == float(bn[1]) == 50.0
    assert np.asarray(b.data).tolist() == np.asarray(bn.data).tolist()


def test_parent_setitem_without_child_unchanged():
    # `a[1:3] = x` (setitem on the PARENT, no child) must still work.
    a = fr.ma.array([1.0, 2, 3, 4])
    a[1:3] = [7.0, 8.0]
    an = np.ma.array([1.0, 2, 3, 4])
    an[1:3] = [7.0, 8.0]
    assert np.asarray(a.data).tolist() == np.asarray(an.data).tolist()


# ---------------------------------------------------------------------------
# 2-D: int-row view, column view, scalar (NOT a view).
# ---------------------------------------------------------------------------


def test_2d_int_row_is_view_writeback():
    a = fr.ma.array([[1.0, 2], [3, 4]])
    r = a[0]
    r[1] = 9.0
    an = np.ma.array([[1.0, 2], [3, 4]])
    rn = an[0]
    rn[1] = 9.0
    assert float(a[0, 1]) == float(an[0, 1]) == 9.0


def test_2d_column_view_writeback():
    a = fr.ma.array([[1.0, 2], [3, 4]])
    c = a[:, 1]
    c[0] = 7.0
    an = np.ma.array([[1.0, 2], [3, 4]])
    cn = an[:, 1]
    cn[0] = 7.0
    assert float(a[0, 1]) == float(an[0, 1]) == 7.0
    assert np.asarray(a.data).tolist() == np.asarray(an.data).tolist()


def test_2d_scalar_index_is_not_view():
    # An N-D index resolving to a 0-d scalar returns a value, not a view.
    a = fr.ma.array([[1.0, 2], [3, 4]])
    an = np.ma.array([[1.0, 2], [3, 4]])
    assert float(a[0, 1]) == float(an[0, 1]) == 2.0


# ---------------------------------------------------------------------------
# Advanced indexing STILL COPIES (numpy copies these too — ferray matches).
# ---------------------------------------------------------------------------


def test_fancy_list_index_is_copy():
    a = fr.ma.array([1.0, 2, 3, 4])
    c = a[[0, 2]]
    c[0] = 7.0
    an = np.ma.array([1.0, 2, 3, 4])
    cn = an[[0, 2]]
    cn[0] = 7.0
    # numpy COPIES — parent[0] is unchanged. ferray must match.
    assert float(a[0]) == float(an[0]) == 1.0


def test_bool_mask_index_is_copy():
    a = fr.ma.array([1.0, 2, 3, 4])
    c = a[a > 1]
    c[0] = 99.0
    an = np.ma.array([1.0, 2, 3, 4])
    cn = an[an > 1]
    cn[0] = 99.0
    # numpy COPIES a boolean-mask gather — parent unchanged.
    assert np.asarray(a.data).tolist() == np.asarray(an.data).tolist()


# ---------------------------------------------------------------------------
# Step / negative / chained slices.
# ---------------------------------------------------------------------------


def test_step_slice_writeback():
    a = fr.ma.array([1.0, 2, 3, 4, 5])
    s = a[::2]
    s[0] = 100.0
    an = np.ma.array([1.0, 2, 3, 4, 5])
    sn = an[::2]
    sn[0] = 100.0
    assert float(a[0]) == float(an[0]) == 100.0


def test_negative_step_slice_writeback():
    a = fr.ma.array([1.0, 2, 3, 4])
    s = a[::-1]
    s[0] = 42.0  # reversed-view position 0 == base position 3
    an = np.ma.array([1.0, 2, 3, 4])
    sn = an[::-1]
    sn[0] = 42.0
    assert float(a[3]) == float(an[3]) == 42.0
    assert np.asarray(a.data).tolist() == np.asarray(an.data).tolist()


def test_chained_view_writeback():
    a = fr.ma.array([1.0, 2, 3, 4])
    v = a[1:3][0:1]  # view-of-a-view → base position 1
    v[0] = 42.0
    an = np.ma.array([1.0, 2, 3, 4])
    vn = an[1:3][0:1]
    vn[0] = 42.0
    assert float(a[1]) == float(an[1]) == 42.0


def test_newaxis_view():
    # `a[None]` (newaxis) is a basic index — a view of shape (1, n).
    a = fr.ma.array([1.0, 2, 3])
    v = a[None]
    an = np.ma.array([1.0, 2, 3])
    vn = an[None]
    assert list(v.shape) == list(vn.shape) == [1, 3]
    v[0, 1] = 55.0
    vn[0, 1] = 55.0
    assert float(a[1]) == float(an[1]) == 55.0


def test_ellipsis_view():
    a = fr.ma.array([1.0, 2, 3])
    v = a[...]
    v[0] = 9.0
    an = np.ma.array([1.0, 2, 3])
    vn = an[...]
    vn[0] = 9.0
    assert float(a[0]) == float(an[0]) == 9.0


# ---------------------------------------------------------------------------
# Masked views: view sees + writes the mask; `= masked` masks the base.
# ---------------------------------------------------------------------------


def test_view_sees_base_mask():
    a = fr.ma.array([1.0, 2, 3, 4], mask=[0, 1, 0, 0])
    b = a[1:3]
    an = np.ma.array([1.0, 2, 3, 4], mask=[0, 1, 0, 0])
    bn = an[1:3]
    assert np.asarray(b.mask).tolist() == np.asarray(bn.mask).tolist() == [True, False]


def test_view_setitem_masked_masks_base():
    a = fr.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    b = a[1:3]
    b[0] = fr.ma.masked
    an = np.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    bn = an[1:3]
    bn[0] = np.ma.masked
    assert np.asarray(a.mask).tolist() == np.asarray(an.mask).tolist()
    assert bool(np.asarray(a.mask).tolist()[1]) is True


def test_view_setitem_unmasks_base_soft():
    a = fr.ma.array([1.0, 2, 3, 4], mask=[0, 1, 0, 0])
    b = a[1:3]
    b[0] = 99.0  # soft mask: write unmasks
    an = np.ma.array([1.0, 2, 3, 4], mask=[0, 1, 0, 0])
    bn = an[1:3]
    bn[0] = 99.0
    assert np.asarray(a.mask).tolist() == np.asarray(an.mask).tolist()
    assert float(a[1]) == float(an[1]) == 99.0


def test_hard_view_writeback_keeps_masked_data():
    a = fr.ma.array([1.0, 2, 3, 4], mask=[1, 1, 0, 0])
    b = a[0:2]
    b.harden_mask()
    b[0] = 99.0  # hard mask → masked position keeps its data
    an = np.ma.array([1.0, 2, 3, 4], mask=[1, 1, 0, 0])
    bn = an[0:2]
    bn.harden_mask()
    bn[0] = 99.0
    assert np.asarray(a.data).tolist() == np.asarray(an.data).tolist()
    assert np.asarray(a.mask).tolist() == np.asarray(an.mask).tolist()
    # numpy: a.hardmask stays False (harden_mask is per-object on the view).
    assert a.hardmask == an.hardmask


# ---------------------------------------------------------------------------
# View attributes / reductions / operators reflect current base state.
# ---------------------------------------------------------------------------


def test_view_data_attribute_reflects_base():
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    a[1] = 88.0
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    an[1] = 88.0
    assert np.asarray(b.data).tolist() == np.asarray(bn.data).tolist()


def test_view_dtype_and_shape():
    a = fr.ma.array([1, 2, 3, 4], dtype=np.int32)
    b = a[1:3]
    an = np.ma.array([1, 2, 3, 4], dtype=np.int32)
    bn = an[1:3]
    assert b.dtype == bn.dtype.name == "int32"
    assert list(b.shape) == list(bn.shape) == [2]
    assert b.ndim == bn.ndim == 1
    assert b.size == bn.size == 2


def test_view_sum_reflects_base():
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    a[1] = 10.0  # base mutation visible to the view's reduction
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    an[1] = 10.0
    assert float(b.sum()) == float(bn.sum()) == 13.0


def test_view_operator_yields_owned_result():
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    r = b + 10.0  # arithmetic on a view → a NEW owned array (numpy too)
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    rn = bn + 10.0
    assert np.asarray(r.data).tolist() == np.asarray(rn.data).tolist()
    # The owned result does NOT alias the base: mutating it leaves a untouched.
    r[0] = -1.0
    rn[0] = -1.0
    assert float(a[1]) == float(an[1]) == 2.0


def test_view_count_reflects_base_mask():
    a = fr.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    b = a[1:3]
    a[1] = fr.ma.masked
    an = np.ma.array([1.0, 2, 3, 4], mask=[0, 0, 0, 0])
    bn = an[1:3]
    an[1] = np.ma.masked
    assert b.count() == bn.count() == 1


def test_view_put_writes_through():
    a = fr.ma.array([1.0, 2, 3, 4])
    b = a[1:3]
    b.put([0], [77.0])
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    bn.put([0], [77.0])
    assert float(a[1]) == float(an[1]) == 77.0


def test_integer_dtype_view_writeback():
    a = fr.ma.array([1, 2, 3, 4], dtype=np.int64)
    b = a[1:3]
    b[0] = 99
    an = np.ma.array([1, 2, 3, 4], dtype=np.int64)
    bn = an[1:3]
    bn[0] = 99
    assert int(a[1]) == int(an[1]) == 99
    assert a.dtype == an.dtype.name == "int64"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
