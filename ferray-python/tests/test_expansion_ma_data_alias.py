"""`.data` shared-buffer aliasing for fr.ma.MaskedArray vs numpy.ma (#896 #898).

Builds on the numpy-backed data buffer: an OWNED masked array's `.data` aliases
a numpy-owned `Py<PyArray>`, a constructor identity-VIEW (`ma.array(other,
mask=/fill_value=)`) aliases the base buffer, and `ma.array(ndarray)` shares the
foreign ndarray buffer. Every EXPECTED value is derived live from numpy.ma
(R-CHAR-3 — never literal-copied from the ferray side); each assertion mirrors
the same operation on numpy and on ferray, or compares against
`numpy.shares_memory`.
"""

import numpy as np
import ferray as fr


# ---------------------------------------------------------------------------
# OWNED array: `.data` aliases its own buffer (`a.data[i] = x` => a[i] == x).
# ---------------------------------------------------------------------------

def test_owned_data_write_through_int():
    na = np.ma.array([1, 2, 3, 4])
    na.data[2] = 50
    np_after = na[2]  # numpy contract: writing .data mutates the array

    fa = fr.ma.array([1, 2, 3, 4])
    fa.data[2] = 50
    assert fa[2] == np_after


def test_owned_data_write_through_float():
    na = np.ma.array([1.5, 2.5, 3.5])
    na.data[0] = 9.25
    fa = fr.ma.array([1.5, 2.5, 3.5])
    fa.data[0] = 9.25
    assert fa.data[0] == na.data[0]


def test_owned_data_dtype_preserved():
    for dt in ("int8", "int16", "int32", "int64",
               "uint8", "uint16", "uint32", "uint64", "float32", "float64"):
        na = np.ma.array([1, 2, 3], dtype=dt)
        fa = fr.ma.array([1, 2, 3], dtype=dt)
        assert str(fa.data.dtype) == str(na.data.dtype)
        fa.data[1] = 7
        assert fa.data[1] == 7


def test_owned_data_self_shares_memory():
    # numpy: a.data and a.data view the SAME buffer.
    na = np.ma.array([1, 2, 3])
    assert np.shares_memory(na.data, na.data)
    fa = fr.ma.array([1, 2, 3])
    assert np.shares_memory(fa.data, fa.data)


# ---------------------------------------------------------------------------
# Constructor identity VIEW shares the base buffer (#896).
# ---------------------------------------------------------------------------

def test_ctor_mask_view_shares_base_data():
    nbase = np.ma.array([1, 2, 3, 4])
    nview = np.ma.array(nbase, mask=[1, 0, 0, 0])
    nview.data[3] = 40
    np_shared = nbase.data[3]

    fbase = fr.ma.array([1, 2, 3, 4])
    fview = fr.ma.array(fbase, mask=[1, 0, 0, 0])
    fview.data[3] = 40
    assert fbase.data[3] == np_shared


def test_ctor_fill_view_shares_base_data():
    nbase = np.ma.array([10, 20, 30])
    nview = np.ma.array(nbase, fill_value=5)
    nview.data[0] = 11
    np_shared = nbase.data[0]

    fbase = fr.ma.array([10, 20, 30])
    fview = fr.ma.array(fbase, fill_value=5)
    fview.data[0] = 11
    assert fbase.data[0] == np_shared


def test_ctor_view_base_write_visible_through_view():
    # The READ direction: a base `.data[i] = x` is visible through the view.
    nbase = np.ma.array([1, 2, 3])
    nview = np.ma.array(nbase, mask=[0, 1, 0])
    nbase.data[2] = 99
    np_seen = nview.data[2]

    fbase = fr.ma.array([1, 2, 3])
    fview = fr.ma.array(fbase, mask=[0, 1, 0])
    fbase.data[2] = 99
    assert fview.data[2] == np_seen


def test_ctor_mask_keep_mask_or_preserved_with_view():
    # The keep_mask=True OR must survive the new view path (#855 regression
    # guard): explicit mask ORs the source mask.
    nsrc = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    nres = np.ma.array(nsrc, mask=[0, 1, 0])

    fsrc = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    fres = fr.ma.array(fsrc, mask=[0, 1, 0])
    np.testing.assert_array_equal(np.asarray(fres.mask), np.asarray(nres.mask))


# ---------------------------------------------------------------------------
# Foreign ndarray buffer sharing (#898).
# ---------------------------------------------------------------------------

def test_ctor_ndarray_shares_foreign_buffer():
    nnd = np.array([1, 2, 3])
    nm = np.ma.array(nnd)
    nm.data[1] = 22
    np_shared = nnd[1]

    fnd = np.array([1, 2, 3])
    fm = fr.ma.array(fnd)
    fm.data[1] = 22
    assert fnd[1] == np_shared


def test_ctor_ndarray_shares_memory():
    # numpy: ma.array(ndarray).data shares memory with the source ndarray.
    nnd = np.array([1.0, 2.0, 3.0])
    nm = np.ma.array(nnd)
    np_shares = np.shares_memory(nm.data, nnd)

    fnd = np.array([1.0, 2.0, 3.0])
    fm = fr.ma.array(fnd)
    assert np.shares_memory(fm.data, fnd) == np_shares


# ---------------------------------------------------------------------------
# Non-aliasing after copy=True (the buffer must be independent).
# ---------------------------------------------------------------------------

def test_copy_true_does_not_alias_foreign():
    nnd = np.array([1, 2, 3])
    nm = np.ma.array(nnd, copy=True)
    nm.data[0] = 77
    np_src_unchanged = nnd[0]  # numpy: copy=True => source untouched

    fnd = np.array([1, 2, 3])
    fm = fr.ma.array(fnd, copy=True)
    fm.data[0] = 77
    assert fnd[0] == np_src_unchanged


def test_dtype_recast_does_not_alias_source():
    # An explicit dtype= recast reallocates _data: the recast view does not
    # write back through to the differently-typed source.
    nbase = np.ma.array([1, 2, 3], dtype="int32")
    nview = np.ma.array(nbase, dtype="int64")
    nview.data[0] = 55
    np_src = nbase.data[0]  # numpy: recast => source unchanged

    fbase = fr.ma.array([1, 2, 3], dtype="int32")
    fview = fr.ma.array(fbase, dtype="int64")
    fview.data[0] = 55
    assert fbase.data[0] == np_src


# ---------------------------------------------------------------------------
# `.data` aliasing coexists with __setitem__ write-through and masking.
# ---------------------------------------------------------------------------

def test_setitem_then_data_read_consistent():
    fa = fr.ma.array([1, 2, 3, 4])
    fa[1] = 200
    na = np.ma.array([1, 2, 3, 4])
    na[1] = 200
    assert fa.data[1] == na.data[1]


def test_data_write_then_setitem_consistent():
    fa = fr.ma.array([1, 2, 3, 4])
    fa.data[0] = 11
    fa[2] = 33
    na = np.ma.array([1, 2, 3, 4])
    na.data[0] = 11
    na[2] = 33
    np.testing.assert_array_equal(np.asarray(fa.data), np.asarray(na.data))


def test_data_write_survives_mask_assignment():
    fa = fr.ma.array([1, 2, 3])
    fa.data[1] = 88
    fa.mask = [True, False, False]
    na = np.ma.array([1, 2, 3])
    na.data[1] = 88
    na.mask = [True, False, False]
    np.testing.assert_array_equal(np.asarray(fa.data), np.asarray(na.data))


def test_sort_inplace_updates_data_buffer():
    fa = fr.ma.array([3, 1, 2])
    fa.sort()
    na = np.ma.array([3, 1, 2])
    na.sort()
    np.testing.assert_array_equal(np.asarray(fa.data), np.asarray(na.data))
