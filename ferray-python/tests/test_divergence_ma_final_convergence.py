"""Final-convergence-gate divergence pins for fr.ma vs numpy.ma.

Authored by the ACToR critic during the DEFINITIVE FINAL convergence sweep of
fr.ma.MaskedArray (HEAD c73ba7df). Each test derives its expected value LIVE
from numpy.ma (R-CHAR-3): no literal-copied expectations.

IMPORTANT methodology note for future critics:
    fr.ma.MaskedArray is NOT a subclass of numpy.ma.MaskedArray. Therefore
    `isinstance(fr_obj, np.ma.MaskedArray)` is False and
    `np.ma.getmaskarray(fr_obj)` MISREADS a ferray object as an unmasked plain
    array. Use ferray's OWN `.mask` / `.filled()` accessors when comparing, or
    you will produce a cascade of false-positive "DIFF"s. The vast majority of
    the fr.ma surface CONVERGES; only the divergences pinned below are real.

Confirmed CONVERGED in this sweep (NOT pinned — they match numpy exactly under
native accessors): module-form sort/clip/take/repeat/squeeze/round/cumsum/
cumprod/ptp/prod/nonzero/all/any; domain-masking ufuncs log/log10/log2/sqrt/
arcsin/arccos/arccosh/arctanh/divide/tan; axis reductions sum/prod/mean/min/
max/std/var (incl all-masked-along-axis -> masked slot, keepdims+axis);
average/median/diff/ediff1d/unique/getmask/getdata/getmaskarray/count/
count_masked/compressed/masked_invalid/fix_invalid/masked_where/masked_equal;
masked-constant arithmetic; a[bool]=masked and a[int_list]=masked setitem;
all-masked scalar reductions -> masked.
"""

import ferray as fr
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Group A — module-form fr.ma.argsort returns the wrong integer dtype.
#
# numpy: np.ma.argsort returns an int64 index array (matching np.argsort).
# ferray method form fr.ma.MaskedArray.argsort() ALSO returns int64 and is
# correct; only the MODULE wrapper fr.ma.argsort(...) returns uint64.
# Index arrays that are uint64 break round-trips through code that expects a
# signed index dtype (e.g. negative-index arithmetic, np.int64-typed APIs).
#
# Divergence: ferray-python ma module argsort wrapper diverges from
# numpy/ma/core.py argsort (returns int64). Tracking: file a blocker.
# ---------------------------------------------------------------------------

def test_ma_module_argsort_index_dtype_is_int64():
    data = [1.5, 2.5, 3.5, 0.5]
    mask = [0, 1, 0, 0]
    expected_dtype = np.ma.argsort(np.ma.array(data, mask=mask)).dtype  # live oracle: int64
    got = fr.ma.argsort(fr.ma.array(data, mask=mask))
    assert str(got.dtype) == str(expected_dtype), (
        f"fr.ma.argsort dtype {got.dtype} != numpy {expected_dtype}"
    )


def test_ma_module_argsort_index_dtype_2d_matches_method():
    data = [[3, 1], [2, 4]]
    mask = [[0, 1], [0, 0]]
    # method form (already correct) is itself derived live from the same family,
    # but the authoritative oracle is numpy:
    expected_dtype = np.ma.argsort(np.ma.array(data, mask=mask)).dtype
    got = fr.ma.argsort(fr.ma.array(data, mask=mask))
    assert str(got.dtype) == str(expected_dtype), (
        f"fr.ma.argsort 2-D dtype {got.dtype} != numpy {expected_dtype}"
    )


# ---------------------------------------------------------------------------
# Group B — module-form fr.ma.argmin / argmax return a Python int, not a
# numpy integer scalar.
#
# numpy: np.ma.argmin / np.ma.argmax of a 1-D array return a numpy.int64
# (a 0-d numpy scalar with .shape == () and a .dtype). ferray returns a bare
# Python int, losing the numpy-scalar ABI (no .dtype, breaks code that does
# `result.dtype` or feeds it back into a numpy-typed slot).
#
# Divergence: ferray-python ma module argmin/argmax wrappers diverge from
# numpy/ma/core.py argmin/argmax (return numpy integer scalar).
# ---------------------------------------------------------------------------

def test_ma_module_argmin_returns_numpy_scalar():
    data = [1.5, 2.5, 3.5, 0.5]
    mask = [0, 1, 0, 0]
    np_res = np.ma.argmin(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmin(fr.ma.array(data, mask=mask))
    # value must match (sanity)
    assert int(fr_res) == int(np_res)
    # ABI: numpy yields a numpy integer scalar (has .dtype); ferray yields int.
    assert hasattr(np_res, "dtype")  # documents the oracle's contract
    assert hasattr(fr_res, "dtype"), (
        f"fr.ma.argmin returned {type(fr_res).__name__}; numpy returns a "
        f"numpy integer scalar with .dtype ({type(np_res).__name__})"
    )


def test_ma_module_argmax_returns_numpy_scalar():
    data = [1.5, 2.5, 3.5, 0.5]
    mask = [0, 1, 0, 0]
    np_res = np.ma.argmax(np.ma.array(data, mask=mask))
    fr_res = fr.ma.argmax(fr.ma.array(data, mask=mask))
    assert int(fr_res) == int(np_res)
    assert hasattr(np_res, "dtype")
    assert hasattr(fr_res, "dtype"), (
        f"fr.ma.argmax returned {type(fr_res).__name__}; numpy returns a "
        f"numpy integer scalar with .dtype ({type(np_res).__name__})"
    )
