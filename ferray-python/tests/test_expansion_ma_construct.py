"""Mask-preserving construction from a masked input (#851, REQ-3).

Constructing a ``ferray.ma.MaskedArray`` from an already-masked input with
``mask=None`` must carry over the source mask, matching numpy's
``MaskedArray.__new__`` ``_data``/``_mask`` copy (``numpy/ma/core.py:2820``).

Expected values are derived LIVE from numpy 2.4.x (R-CHAR-3) — every
assertion compares ``ferray.ma`` against the corresponding ``numpy.ma`` call
on the same input; no ferray-side literal is hard-coded.

The fix must NOT regress the #848 nomask-vs-explicit distinction nor the #849
getter behaviour:
- a nomask input round-trips as ``nomask`` (scalar ``np.False_``);
- an explicit all-False mask round-trips as a real ``array([False, ...])``;
- an any-True mask round-trips with its True bits intact.
"""

import numpy as np
import pytest

import ferray as fr


def _mask_repr(m):
    """Normalise a ``.mask`` to a comparable form: the ``nomask`` sentinel
    (a 0-d ``np.False_``) -> the string ``"nomask"``, a real bool array -> the
    array. Lets us compare ferray vs numpy without ``is np.ma.nomask``
    identity (ferray returns numpy's own ``nomask`` singleton via ``ma_nomask``,
    but normalising by ndim is oracle-independent)."""
    if np.ndim(m) == 0:
        # scalar -> the nomask sentinel; assert it's the False scalar
        assert bool(m) is False
        return "nomask"
    return np.asarray(m, dtype=bool)


def _assert_mask_matches(fr_ma, np_ma):
    fr_m = _mask_repr(fr_ma.mask)
    np_m = _mask_repr(np_ma.mask)
    if isinstance(np_m, str) or isinstance(fr_m, str):
        assert fr_m == np_m, f"ferray mask {fr_ma.mask!r} != numpy mask {np_ma.mask!r}"
    else:
        np.testing.assert_array_equal(
            fr_m, np_m, err_msg=f"ferray {fr_ma.mask!r} != numpy {np_ma.mask!r}"
        )


def test_anytrue_mask_carried_over():
    """np.ma input with an any-True mask -> mask preserved (the divergence)."""
    src = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    got = fr.ma.array(src)
    want = np.ma.array(src)  # numpy oracle: copies _data/_mask
    _assert_mask_matches(got, want)
    # explicit live expectation: numpy yields array([True, False, False])
    np.testing.assert_array_equal(got.mask, np.array([True, False, False]))


def test_explicit_all_false_mask_preserved():
    """np.ma input with an explicit all-False mask -> real all-False (#849)."""
    src = np.ma.array([1, 2, 3], mask=[0, 0, 0])
    got = fr.ma.array(src)
    want = np.ma.array(src)
    _assert_mask_matches(got, want)
    # numpy: a real array([False, False, False]), NOT the nomask sentinel.
    assert np.ndim(want.mask) == 1  # oracle: real mask, not scalar
    np.testing.assert_array_equal(got.mask, np.array([False, False, False]))


def test_nomask_input_stays_nomask():
    """np.ma input with NO mask= -> nomask sentinel preserved (#848)."""
    src = np.ma.array([1, 2, 3])
    got = fr.ma.array(src)
    want = np.ma.array(src)
    _assert_mask_matches(got, want)
    # numpy: scalar nomask (np.False_).
    assert np.ndim(want.mask) == 0
    assert np.ndim(got.mask) == 0
    assert bool(got.mask) is False


def test_plain_list_stays_nomask():
    """A plain Python list -> nomask (unchanged)."""
    got = fr.ma.array([1, 2, 3])
    want = np.ma.array([1, 2, 3])
    _assert_mask_matches(got, want)
    assert np.ndim(got.mask) == 0
    assert bool(got.mask) is False


def test_fr_to_fr_round_trip():
    """ferray.ma -> ferray.ma round-trip carries the mask over."""
    inner = fr.ma.array([1, 2, 3], mask=[1, 0, 0])
    got = fr.ma.array(inner)
    # numpy oracle for the same logical operation:
    want = np.ma.array(np.ma.array([1, 2, 3], mask=[1, 0, 0]))
    _assert_mask_matches(got, want)
    np.testing.assert_array_equal(got.mask, np.array([True, False, False]))


def test_masked_array_alias_carries_mask():
    """The `masked_array` constructor alias also carries the mask over."""
    src = np.ma.array([1, 2, 3], mask=[1, 0, 0])
    got = fr.ma.masked_array(src)
    want = np.ma.masked_array(src)
    _assert_mask_matches(got, want)


def test_explicit_mask_kwarg_unchanged():
    """Passing mask= explicitly is unchanged by the fix."""
    got = fr.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
    want = np.ma.array([1, 2, 3, 4], mask=[1, 0, 1, 0])
    _assert_mask_matches(got, want)
    np.testing.assert_array_equal(got.mask, want.mask)


@pytest.mark.parametrize(
    "data,mask",
    [
        ([1, 2, 3, 4, 5], [1, 0, 0, 1, 0]),
        ([10, 20, 30], [0, 1, 0]),
        ([[1, 2], [3, 4]], [[1, 0], [0, 1]]),  # 2-D mask round-trip
    ],
)
def test_carryover_array_equal_vs_numpy(data, mask):
    """Across several shapes/masks, fr.ma.array(np.ma.array(..)) mask-equals
    numpy's own round-trip (AC-3)."""
    src = np.ma.array(data, mask=mask)
    got = fr.ma.array(src)
    want = np.ma.array(src)
    # `got.mask` is the ferray getter; numpy's getmaskarray on `want` is the
    # oracle full-bool mask. Compare the two as bool arrays.
    np.testing.assert_array_equal(
        np.asarray(got.mask, dtype=bool), np.ma.getmaskarray(want)
    )
