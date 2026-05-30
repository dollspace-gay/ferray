"""ferray.ma `mask=` tri-state vs numpy.ma (#886).

numpy distinguishes three `mask=` cases (verified live, numpy 2.4.x):
  * ABSENT (`ma.array(data)`)          -> `_mask is nomask`     (np.False_)
  * `mask=numpy.ma.nomask`             -> `_mask is nomask`     (np.False_)
  * `mask=None` (EXPLICIT Python None) -> a REAL all-False mask (make_mask_none)
plus an explicit mask array carries that real mask. ferray previously conflated
ABSENT and `mask=None` (both -> nomask) because PyO3's `Option<_>` collapses an
omitted argument and an explicit `None`. Each assertion's expected value is read
from a LIVE numpy.ma call in the same test (R-CHAR-3), never literal-copied.
"""

import numpy as np
import pytest

import ferray as fr


def _mask_repr(arr):
    """Normalise `.mask` to a comparable form: the nomask sentinel -> the
    string 'nomask'; a real bool array -> a nested Python-bool list."""
    m = arr.mask
    if m is np.ma.nomask or m is np.False_ or (np.isscalar(m) and m is np.False_):
        # numpy's nomask is np.False_ (a 0-d bool scalar that is the sentinel).
        return "nomask"
    return np.asarray(m, dtype=bool).tolist()


def _is_nomask_fr(arr):
    m = arr.mask
    # ferray returns the genuine numpy.ma.nomask singleton for a nomask array.
    return m is np.ma.nomask


# ---------------------------------------------------------------------------
# The four canonical cases, derived live from numpy.ma.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ctor", ["array", "masked_array", "class"])
def test_mask_tristate_1d(ctor):
    data = [1, 2, 3]

    def make(mask_kw):
        if mask_kw == "ABSENT":
            args, kwargs = (data,), {}
        else:
            args, kwargs = (data,), {"mask": mask_kw}
        if ctor == "array":
            return fr.ma.array(*args, **kwargs), np.ma.array(*args, **kwargs)
        if ctor == "masked_array":
            return fr.ma.masked_array(*args, **kwargs), np.ma.masked_array(
                *args, **kwargs
            )
        return fr.ma.MaskedArray(*args, **kwargs), np.ma.MaskedArray(*args, **kwargs)

    # ABSENT -> nomask
    f, n = make("ABSENT")
    assert n.mask is np.ma.nomask
    assert _is_nomask_fr(f)

    # mask=numpy.ma.nomask -> nomask
    f, n = make(np.ma.nomask)
    assert n.mask is np.ma.nomask
    assert _is_nomask_fr(f)

    # mask=None (EXPLICIT) -> a REAL all-False mask
    f, n = make(None)
    assert _mask_repr(n) == [False, False, False]
    assert _is_nomask_fr(f) is False
    assert _mask_repr(f) == _mask_repr(n)

    # mask=[1,0,0] -> a real mask [True, False, False]
    f, n = make([1, 0, 0])
    assert _mask_repr(n) == [True, False, False]
    assert _mask_repr(f) == _mask_repr(n)


def test_mask_none_2d():
    data = [[1, 2], [3, 4]]
    n = np.ma.array(data, mask=None)
    f = fr.ma.array(data, mask=None)
    assert _mask_repr(n) == [[False, False], [False, False]]
    assert _mask_repr(f) == _mask_repr(n)
    assert _is_nomask_fr(f) is False


def test_mask_none_dtypes():
    for dt in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        data = np.array([1, 2, 3], dtype=dt)
        n = np.ma.array(data, mask=None)
        f = fr.ma.array(data, mask=None)
        assert _mask_repr(n) == [False, False, False]
        assert _mask_repr(f) == _mask_repr(n)
        assert _is_nomask_fr(f) is False


def test_mask_none_on_masked_source_keeps_source_mask():
    # numpy: mask=None on a masked source does NOT clear the source mask — the
    # all-False make_mask_none is ORed (keep_mask=True) with the source mask, so
    # the result equals the source mask [True, False, False] (verified live).
    n_src = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    f_src = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    n = np.ma.array(n_src, mask=None)
    f = fr.ma.array(f_src, mask=None)
    assert _mask_repr(n) == [True, False, False]
    assert _mask_repr(f) == _mask_repr(n)


def test_absent_on_masked_source_carries_mask_unregressed():
    # #851: an ABSENT mask on a masked source carries the source's real mask.
    n_src = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    f_src = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    n = np.ma.array(n_src)
    f = fr.ma.array(f_src)
    assert _mask_repr(n) == [True, False, False]
    assert _mask_repr(f) == _mask_repr(n)


def test_absent_plain_data_is_nomask_unregressed():
    # #848/#849: an ABSENT mask on plain (unmasked) data stays nomask.
    n = np.ma.array([1, 2, 3])
    f = fr.ma.array([1, 2, 3])
    assert n.mask is np.ma.nomask
    assert _is_nomask_fr(f)


def test_explicit_mask_keep_mask_or_unregressed():
    # #855: an explicit mask= ORs with a masked source's own mask.
    n_src = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 0])
    f_src = fr.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 0])
    n = np.ma.array(n_src, mask=[0, 1, 0, 0])
    f = fr.ma.array(f_src, mask=[0, 1, 0, 0])
    assert _mask_repr(n) == [True, True, False, False]
    assert _mask_repr(f) == _mask_repr(n)
