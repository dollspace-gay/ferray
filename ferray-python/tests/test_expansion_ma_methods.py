"""Expansion suite: fr.ma.MaskedArray method/kwarg/attribute surface vs numpy.ma.

Covers #892 (20 object methods), #893 (reduction axis/keepdims/dtype/ddof
kwargs), #895 (itemsize/nbytes/strides/flat/base) across dtypes, shapes and
masked patterns.

Every expected value is derived LIVE from numpy.ma (R-CHAR-3) — never
literal-copied from the ferray side. Each test builds the SAME data+mask in
both libraries and asserts ferray matches numpy's own method/attribute.
"""

import numpy as np
import ferray as fr
import pytest


def _pair(data, mask=None, dtype=None):
    """Build an (ferray, numpy) MaskedArray pair with identical data+mask."""
    if mask is None:
        return fr.ma.array(data, dtype=dtype), np.ma.array(data, dtype=dtype)
    return (
        fr.ma.array(data, mask=mask, dtype=dtype),
        np.ma.array(data, mask=mask, dtype=dtype),
    )


def _mask_list(x):
    """The full per-element mask as a nested list (nomask -> all-False)."""
    m = x.mask
    if m is fr.ma.nomask or m is np.ma.nomask:
        return np.zeros(x.shape, dtype=bool).tolist()
    return np.asarray(m).tolist()


def _mt(x):
    """A masked-aware snapshot: filled-with-0 data list + the per-element mask."""
    return x.filled(0).tolist(), _mask_list(x)


# ---------------------------------------------------------------------------
# #892 — object methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data,mask", [
    ([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]]),
    ([1, 0, 3], [0, 0, 1]),
    ([1, 1, 1], None),
])
def test_all_any_match(data, mask):
    f, n = _pair(data, mask)
    assert bool(f.all()) == bool(n.all())
    assert bool(f.any()) == bool(n.any())


def test_all_any_axis_keepdims():
    f, n = _pair([[1, 0, 3], [4, 5, 0]], [[0, 1, 0], [0, 0, 0]])
    assert f.all(axis=0).filled(0).tolist() == n.all(axis=0).filled(0).tolist()
    assert f.any(axis=1).filled(0).tolist() == n.any(axis=1).filled(0).tolist()
    assert f.all(keepdims=True).shape == n.all(keepdims=True).shape


@pytest.mark.parametrize("dtype", ["int32", "int64", "float64", "uint8"])
def test_take(dtype):
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]], dtype=dtype)
    fr_t, np_t = f.take([0, 2, 4]), n.take([0, 2, 4])
    assert fr_t.filled(0).tolist() == np_t.filled(0).tolist()
    assert str(fr_t.dtype) == str(np_t.dtype)


def test_take_axis():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.take([0, 2], axis=1).filled(0).tolist() == n.take([0, 2], axis=1).filled(0).tolist()


def test_squeeze():
    f, n = _pair([[1, 2, 3]])
    assert f.squeeze().tolist() == n.squeeze().tolist()
    assert f.squeeze().shape == n.squeeze().shape


def test_ptp():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.ptp() == n.ptp()


def test_nonzero():
    f = fr.ma.array([0, 2, 0, 3, 5], mask=[0, 0, 0, 1, 0])
    n = np.ma.array([0, 2, 0, 3, 5], mask=[0, 0, 0, 1, 0])
    assert [x.tolist() for x in f.nonzero()] == [x.tolist() for x in n.nonzero()]


@pytest.mark.parametrize("data,mask", [
    ([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]]),
    ([7, 3, 9, 1], [0, 0, 1, 0]),
])
def test_argmin_argmax(data, mask):
    f, n = _pair(data, mask)
    assert f.argmin() == n.argmin()
    assert f.argmax() == n.argmax()


def test_argmin_axis():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.argmin(axis=1).tolist() == n.argmin(axis=1).tolist()


@pytest.mark.parametrize("dtype", ["int64", "float64", "uint8"])
def test_prod_method(dtype):
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]], dtype=dtype)
    assert f.prod() == n.prod()
    assert str(np.asarray(f.prod()).dtype) == str(np.asarray(n.prod()).dtype)


def test_cumsum_cumprod():
    f, n = _pair([1, 2, 3, 4], [0, 1, 0, 0])
    assert _mt(f.cumsum()) == _mt(n.cumsum())
    assert _mt(f.cumprod()) == _mt(n.cumprod())


def test_clip():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert _mt(f.clip(2, 5)) == _mt(n.clip(2, 5))


def test_round():
    f, n = _pair([1.4, 2.6, 3.5, -1.55], [0, 1, 0, 0])
    assert f.round().filled(0).tolist() == n.round().filled(0).tolist()
    assert f.round(1).filled(0).tolist() == n.round(1).filled(0).tolist()


def test_swapaxes():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.swapaxes(0, 1).shape == n.swapaxes(0, 1).shape
    assert f.swapaxes(0, 1).filled(0).tolist() == n.swapaxes(0, 1).filled(0).tolist()


def test_trace():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.trace() == n.trace()


def test_diagonal():
    f, n = _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])
    assert f.diagonal().filled(0).tolist() == n.diagonal().filled(0).tolist()
    assert f.diagonal(offset=1).filled(0).tolist() == n.diagonal(offset=1).filled(0).tolist()


def test_item():
    f, n = _pair([5])
    assert f.item() == n.item()
    f2, n2 = _pair([[7]])
    assert f2.item() == n2.item()


def test_fill_inplace():
    f = fr.ma.array([1, 2, 3], mask=[0, 1, 0])
    n = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    assert f.fill(9) is None
    n.fill(9)
    # fill sets DATA only; mask unchanged
    assert f.data.tolist() == n.data.tolist()
    assert f.mask.tolist() == n.mask.tolist()


def test_repeat():
    f, n = _pair([1, 2], [0, 1])
    assert _mt(f.repeat(2)) == _mt(n.repeat(2))


def test_repeat_axis():
    f, n = _pair([[1, 2], [3, 4]], [[0, 1], [0, 0]])
    assert _mt(f.repeat(2, axis=0)) == _mt(n.repeat(2, axis=0))


def test_sort_inplace_masked_to_end():
    f = fr.ma.array([3, 1, 2, 5], mask=[0, 1, 0, 1])
    n = np.ma.array([3, 1, 2, 5], mask=[0, 1, 0, 1])
    assert f.sort() is None
    n.sort()
    assert f.filled(0).tolist() == n.filled(0).tolist()
    assert f.mask.tolist() == n.mask.tolist()


def test_argsort_masked_to_end():
    f = fr.ma.array([3, 1, 2, 5], mask=[0, 1, 0, 1])
    n = np.ma.array([3, 1, 2, 5], mask=[0, 1, 0, 1])
    assert f.argsort().tolist() == n.argsort().tolist()


# ---------------------------------------------------------------------------
# #893 — reduction axis/keepdims/dtype/ddof kwargs
# ---------------------------------------------------------------------------

def _md():
    return _pair([[1, 2, 3], [4, 5, 6]], [[0, 1, 0], [0, 0, 1]])


def test_sum_axis_keepdims():
    f, n = _md()
    assert f.sum(axis=0).filled(0).tolist() == n.sum(axis=0).filled(0).tolist()
    assert f.sum(axis=1).filled(0).tolist() == n.sum(axis=1).filled(0).tolist()
    assert f.sum(keepdims=True).shape == n.sum(keepdims=True).shape


def test_min_max_axis():
    f, n = _md()
    assert f.min(axis=0).filled(0).tolist() == n.min(axis=0).filled(0).tolist()
    assert f.max(axis=1).filled(0).tolist() == n.max(axis=1).filled(0).tolist()


def test_mean_axis_dtype():
    f, n = _md()
    assert f.mean(axis=0).filled(0).tolist() == n.mean(axis=0).filled(0).tolist()
    assert str(f.mean(dtype="float32").dtype) == str(n.mean(dtype="float32").dtype)


def test_std_var_axis_ddof():
    f, n = _md()
    assert f.std(axis=0).filled(0).tolist() == n.std(axis=0).filled(0).tolist()
    assert f.var(axis=1).filled(0).tolist() == n.var(axis=1).filled(0).tolist()
    assert f.var(ddof=1) == n.var(ddof=1)


def test_prod_axis_keepdims():
    f, n = _md()
    assert f.prod(axis=0).filled(0).tolist() == n.prod(axis=0).filled(0).tolist()
    assert f.prod(keepdims=True).shape == n.prod(keepdims=True).shape


def test_no_kwarg_reductions_unchanged():
    # #858/#873 no-regression: no-kwarg int sum/prod keep native accumulator
    # dtype; the no-arg fast path must match numpy exactly.
    f, n = _pair([1, 2, 3, 4], [0, 1, 0, 0], dtype="int32")
    assert f.sum() == n.sum()
    assert str(np.asarray(f.sum()).dtype) == str(np.asarray(n.sum()).dtype)
    assert f.prod() == n.prod()
    assert str(np.asarray(f.prod()).dtype) == str(np.asarray(n.prod()).dtype)


# ---------------------------------------------------------------------------
# #895 — attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "int32", "int64", "uint8", "complex128"])
def test_itemsize_nbytes(dtype):
    f, n = _pair([1, 2, 3, 4], dtype=dtype)
    assert f.itemsize == n.itemsize
    assert f.nbytes == n.nbytes


def test_strides():
    f, n = _pair([[1.0, 2.0], [3.0, 4.0]])
    assert tuple(f.strides) == tuple(n.strides)


def test_flat_iter_masked_singleton():
    f = fr.ma.array([1, 2, 3], mask=[0, 1, 0])
    n = np.ma.array([1, 2, 3], mask=[0, 1, 0])
    fvals = [(x is fr.ma.masked) for x in f.flat]
    nvals = [(x is np.ma.masked) for x in n.flat]
    assert fvals == nvals


def test_base_owned_is_none_matches_numpy():
    n = np.ma.array([1, 2, 3])
    f = fr.ma.array([1, 2, 3])
    assert (f.base is None) == (n.base is None)
