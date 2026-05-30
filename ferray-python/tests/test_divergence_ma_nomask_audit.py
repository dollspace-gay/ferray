"""Divergence pins for the `numpy.ma` nomask/explicit-mask audit (re #848).

Objective-B pin: the `PyMaskedArray::mask` getter keys off `any_masked()`
(`count_masked > 0`), so an EXPLICIT all-False mask array collapses to the
`nomask` scalar `False`. numpy preserves an explicit mask array even when it
is all-False, because the user passed `mask=`. The #848 ctor fix correctly
preserved the data-path distinction (nomask vs explicit-all-False) for `put`,
but the `.mask` GETTER still collapses explicit-all-False -> nomask.

Live numpy oracle (numpy 2.4.4), source contract:
  numpy/ma/core.py:1468 — `getmask` returns `nomask` only when `_mask is nomask`.
  For `mask=[0,0,0]` the user-passed mask is a real bool array, so `_mask`
  is that all-False ndarray and `.mask` returns `array([False, False, False])`,
  NOT the `nomask` singleton.

Expected values are taken from a live numpy call below, never copied from the
ferray side (R-CHAR-3).
"""

import numpy as np
import ferray as fr


def test_explicit_all_false_mask_is_preserved_not_collapsed_to_nomask():
    # Live numpy oracle: explicit all-False mask -> real bool array, NOT nomask.
    n = np.ma.array([1, 2, 3], mask=[0, 0, 0])
    assert n.mask is not np.ma.nomask  # user passed a mask -> real array
    assert isinstance(n.mask, np.ndarray)
    assert n.mask.shape == (3,)
    np.testing.assert_array_equal(n.mask, np.array([False, False, False]))

    # ferray must mirror: explicit mask= (even all-False) is preserved.
    f = fr.ma.array([1, 2, 3], mask=[0, 0, 0])
    fm = f.mask
    # FAILS today: the getter returns the np.False_ nomask scalar instead of
    # an array([False, False, False]).
    assert isinstance(fm, np.ndarray), (
        f"explicit all-False mask collapsed to {fm!r} ({type(fm).__name__}); "
        "numpy preserves array([False, False, False])"
    )
    assert fm.shape == (3,)
    np.testing.assert_array_equal(fm, np.array([False, False, False]))


def test_default_ctor_mask_is_nomask_scalar_no_regression():
    """Control: default (no mask=) ctor must STILL yield the nomask scalar.

    This is the behavior #848 introduced and must not regress when the getter
    divergence above is fixed.
    """
    n = np.ma.array([1, 2, 3])
    assert n.mask is np.ma.nomask

    f = fr.ma.array([1, 2, 3])
    fm = f.mask
    # nomask is mirrored as the scalar np.False_ (not an ndarray).
    assert not isinstance(fm, np.ndarray)
    assert bool(fm) is False
