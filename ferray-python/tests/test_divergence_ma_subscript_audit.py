"""ACToR critic pins — re-audit of acto-builder commit ef1e3286 (#852):
``PyMaskedArray.__getitem__`` / ``__setitem__`` in ``ferray-python/src/ma.rs``.

Adversarial divergence pins for ``ferray.ma.MaskedArray`` subscripting vs
``numpy.ma`` (numpy 2.4.4 = oracle). Every expected value is derived LIVE from
``numpy.ma`` in-test (R-CHAR-3) — none is literal-copied from the ferray side.

CONFIRMED DIVERGENCE (this file): ``__getitem__`` on a source carrying a REAL
mask collapses the result's ``.mask`` to the ``nomask`` singleton whenever the
gathered/sliced positions happen to be all-unmasked, instead of preserving a
real bool mask array of the result shape.

    numpy/ma/core.py:3317  `if _mask is not nomask: mout = _mask[indx]`

When ``_mask`` is a real array, ``_mask[indx]`` is ALWAYS a real array (an
all-False one if no selected position is masked), never ``nomask``. ferray's
``__getitem__`` Gather branch short-circuits to the ``from_data`` nomask
sentinel via ``if sub_mask.iter().any(|&b| b)`` (ma.rs ``__getitem__``), so the
result reports ``.mask is nomask`` and a 0-d ``np.False_`` scalar of shape
``()`` rather than numpy's real array of the result shape.

dtype f64-pinning is OUT OF SCOPE (#853); these pins assert MASK STRUCTURE /
nomask identity only, never value dtype.
"""

import numpy as np
import ferray as fr


# ---------------------------------------------------------------------------
# CONFIRMED divergence: real-masked source -> getitem collapses to nomask
# ---------------------------------------------------------------------------


def test_getitem_slice_realmask_source_keeps_real_mask_array():
    """numpy: a slice of a REAL-masked array keeps a real bool ``.mask`` array
    (here all-False because the slice selects only unmasked positions), and
    ``.mask is nomask`` is False. ferray collapses to the nomask singleton.

    Divergence: ferray ``__getitem__`` Gather branch (ma.rs) routes through the
    ``from_data`` nomask sentinel when no gathered position is masked; numpy
    ``core.py:3317`` ``mout = _mask[indx]`` keeps a real array.
    """
    an = np.ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])
    af = fr.ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])

    # Live numpy contract: a[0:1] selects only the unmasked element 0.
    np_mask = an[0:1].mask
    assert np_mask is not np.ma.nomask
    assert np_mask.shape == (1,)
    assert np_mask.tolist() == [False]

    fr_slice = af[0:1]
    assert fr_slice.mask is not fr.ma.nomask
    assert np.asarray(fr_slice.mask).shape == np_mask.shape
    assert np.asarray(fr_slice.mask).tolist() == np_mask.tolist()


def test_getitem_fancy_realmask_source_keeps_real_mask_array():
    """numpy: fancy-indexing a real-masked source to all-unmasked positions
    keeps a real all-False ``.mask`` of the result shape; ferray collapses to
    the nomask scalar (shape ``()``)."""
    an = np.ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])
    af = fr.ma.array([1.0, 2, 3, 4, 5], mask=[0, 1, 0, 1, 0])

    np_mask = an[[0, 2, 4]].mask  # all selected positions unmasked
    assert np_mask is not np.ma.nomask
    assert np_mask.shape == (3,)

    fr_mask = np.asarray(af[[0, 2, 4]].mask)
    assert af[[0, 2, 4]].mask is not fr.ma.nomask
    assert fr_mask.shape == np_mask.shape
    assert fr_mask.tolist() == np_mask.tolist()


def test_getitem_diagonal_gather_realmask_mask_is_array():
    """numpy: ``a[[0,1],[0,1]]`` diagonal gather of a real-masked 2-D source
    yields a length-2 real ``.mask`` array; iterating it works. ferray returns
    a 0-d ``np.False_`` scalar — ``list(result.mask)`` raises ``TypeError:
    'numpy.bool' object is not iterable``.

    This is the same nomask-collapse root cause surfacing as a hard crash on a
    perfectly ordinary diagonal gather.
    """
    an = np.ma.array([[1.0, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 0]])
    af = fr.ma.array([[1.0, 2, 3], [4, 5, 6]], mask=[[0, 1, 0], [1, 0, 0]])

    np_mask = an[[0, 1], [0, 1]].mask  # picks elems (0,0),(1,1): both unmasked
    assert np_mask.shape == (2,)
    np_mask_list = np_mask.tolist()

    fr_mask = np.asarray(af[[0, 1], [0, 1]].mask)
    assert fr_mask.shape == np_mask.shape
    assert fr_mask.tolist() == np_mask_list


def test_getitem_2d_row_realmask_source_keeps_real_mask_array():
    """numpy: a row ``a[0]`` of a real-masked 2-D source whose row is all
    unmasked keeps a real all-False ``.mask`` of shape (ncols,); ferray
    collapses to the nomask scalar."""
    an = np.ma.array([[1.0, 2, 3], [4, 5, 6]], mask=[[0, 0, 0], [1, 0, 0]])
    af = fr.ma.array([[1.0, 2, 3], [4, 5, 6]], mask=[[0, 0, 0], [1, 0, 0]])

    np_mask = an[0].mask
    assert np_mask is not np.ma.nomask
    assert np_mask.shape == (3,)

    fr_mask = np.asarray(af[0].mask)
    assert af[0].mask is not fr.ma.nomask
    assert fr_mask.shape == np_mask.shape
    assert fr_mask.tolist() == np_mask.tolist()


def test_getitem_empty_index_realmask_source_keeps_empty_mask_array():
    """numpy: ``a[[]]`` on a real-masked source yields an empty (shape (0,))
    real bool ``.mask`` array, not the nomask singleton. ferray returns the
    nomask scalar ``np.False_`` of shape ``()``."""
    an = np.ma.array([1.0, 2, 3], mask=[0, 1, 0])
    af = fr.ma.array([1.0, 2, 3], mask=[0, 1, 0])

    np_mask = an[[]].mask
    assert np_mask is not np.ma.nomask
    assert np_mask.shape == (0,)

    fr_mask = np.asarray(af[[]].mask)
    assert af[[]].mask is not fr.ma.nomask
    assert fr_mask.shape == np_mask.shape


def test_getitem_explicit_allfalse_source_slice_keeps_real_mask():
    """#849 consistency: an explicit all-False ``mask=[0,0,0]`` source carries a
    REAL mask. numpy: slicing it keeps a real all-False ``.mask`` array
    (``mask is nomask`` False). ferray collapses to the nomask singleton."""
    an = np.ma.array([1.0, 2, 3], mask=[0, 0, 0])
    af = fr.ma.array([1.0, 2, 3], mask=[0, 0, 0])

    assert an[0:2].mask is not np.ma.nomask
    assert an[0:2].mask.tolist() == [False, False]

    assert af[0:2].mask is not fr.ma.nomask
    assert np.asarray(af[0:2].mask).tolist() == [False, False]


# ---------------------------------------------------------------------------
# KNOWN / lower-severity caveat (builder-disclosed): slice returns a COPY,
# numpy returns a VIEW. Tracked separately; needs a library view abstraction,
# not a binding tweak. Data + mask READS are correct — only writeback differs.
# ---------------------------------------------------------------------------


def test_getitem_slice_is_view_writeback_KNOWN_copy_divergence():
    """KNOWN (lower-sev, builder-disclosed): numpy ``a[1:3]`` is a VIEW — a
    write through the sub-array propagates back to the parent. ferray returns a
    COPY, so the writeback is lost.

    numpy slice-view semantics (R-DEV-1 view-vs-copy): ``b = a[1:3]; b[0] = 99``
    sets ``a[1] == 99``. Implementing this in ferray needs a MaskedArray view
    abstraction (shared buffer / strided window) in the ferray-ma library, not
    just a binding change — hence lower severity than the nomask-collapse pins
    above. Data + mask READS through the sub-array are already correct.
    """
    an = np.ma.array([1.0, 2, 3, 4])
    bn = an[1:3]
    bn[0] = 99.0
    # Live numpy contract: the parent sees the write (view semantics).
    assert an[1] == 99.0

    af = fr.ma.array([1.0, 2, 3, 4])
    bf = af[1:3]
    bf[0] = 99.0
    assert af[1] == 99.0  # FAILS under ferray: slice is a copy, a[1] stays 2.0
