"""Divergence #973 — `fr.pad` reflect correctness + all modes/kwargs via numpy-delegate.

Prior to #973 `fr.pad(..., mode='reflect')` mirrored WITH the edge element
(i.e. it computed `symmetric`: `[2,1,1,2,3,4,3,2]`) where numpy reflects
WITHOUT the edge (`[3,2,1,2,3,4,3,2]`), and the `linear_ramp`/`maximum`/
`mean`/`minimum`/`median`/`empty` modes raised TypeError. The fix delegates
the numeric/complex path to `numpy.pad`, which owns every mode + kwarg.

Every expected value is derived LIVE from numpy (R-CHAR-3) — never copied
from the ferray side.
"""

import numpy as np
import ferray as fr
import pytest


def _assert_pad(arr, pad_width, **kwargs):
    """fr.pad must match np.pad in values, shape and dtype."""
    got = np.asarray(fr.pad(arr, pad_width, **kwargs))
    exp = np.pad(np.asarray(arr), pad_width, **kwargs)
    assert got.shape == exp.shape, (kwargs, got.shape, exp.shape)
    assert got.dtype == exp.dtype, (kwargs, got.dtype, exp.dtype)
    assert np.array_equal(got, exp) or np.allclose(got, exp, equal_nan=True), (
        kwargs,
        got,
        exp,
    )


BASE = [1.0, 2.0, 3.0, 4.0]


# --- reflect: the headline divergence -------------------------------------
def test_reflect_does_not_repeat_edge():
    # numpy reflect mirrors WITHOUT the edge element.
    got = np.asarray(fr.pad(BASE, 2, mode="reflect"))
    exp = np.pad(np.asarray(BASE), 2, mode="reflect")
    assert list(got) == list(exp)
    assert list(exp) == [3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]


def test_symmetric_repeats_edge():
    # symmetric mirrors WITH the edge element (distinct from reflect).
    got = np.asarray(fr.pad(BASE, 2, mode="symmetric"))
    exp = np.pad(np.asarray(BASE), 2, mode="symmetric")
    assert list(got) == list(exp)
    assert list(exp) == [2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 4.0, 3.0]


@pytest.mark.parametrize(
    "mode",
    ["reflect", "symmetric", "edge", "wrap", "constant"],
)
def test_existing_modes_unregressed(mode):
    _assert_pad(BASE, 2, mode=mode)


# --- previously-missing modes ---------------------------------------------
@pytest.mark.parametrize(
    "mode",
    ["linear_ramp", "maximum", "mean", "minimum", "median"],
)
def test_new_stat_and_ramp_modes(mode):
    _assert_pad(BASE, 2, mode=mode)


def test_empty_mode_shape():
    # 'empty' leaves the pad region uninitialised; only shape/dtype are
    # contractual.
    got = np.asarray(fr.pad(BASE, 2, mode="empty"))
    exp = np.pad(np.asarray(BASE), 2, mode="empty")
    assert got.shape == exp.shape
    assert got.dtype == exp.dtype


# --- mode-specific kwargs --------------------------------------------------
def test_reflect_type_odd():
    _assert_pad(BASE, 2, mode="reflect", reflect_type="odd")
    _assert_pad(BASE, 2, mode="symmetric", reflect_type="odd")


def test_linear_ramp_end_values():
    _assert_pad(BASE, 2, mode="linear_ramp", end_values=(10, 20))


def test_stat_length():
    _assert_pad(BASE, 2, mode="maximum", stat_length=2)
    _assert_pad(BASE, 3, mode="mean", stat_length=1)


def test_constant_values():
    _assert_pad(BASE, 2, mode="constant", constant_values=9)
    _assert_pad(BASE, 2, mode="constant", constant_values=(5, 7))


# --- pad_width forms -------------------------------------------------------
def test_pad_width_int_and_tuple():
    _assert_pad(BASE, 3)
    _assert_pad(BASE, (1, 2))


def test_pad_width_per_axis_2d():
    m = np.arange(6).reshape(2, 3).astype(float)
    _assert_pad(m, [(1, 1), (2, 2)], mode="reflect")
    _assert_pad(m, [(0, 2), (1, 0)], mode="symmetric")


# --- dtype preservation ----------------------------------------------------
def test_int_dtype_preserved():
    _assert_pad(np.array([1, 2, 3], dtype=np.int64), 1, mode="reflect")
    _assert_pad(np.array([1, 2, 3], dtype=np.int32), 2, mode="wrap")


def test_complex_dtype_preserved():
    _assert_pad(np.array([1 + 2j, 3 + 4j]), 1, mode="wrap")
    _assert_pad(np.array([1 + 2j, 3 + 4j]), 2, mode="reflect")


def test_2d_stat_mode():
    m = np.arange(6).reshape(2, 3).astype(float)
    _assert_pad(m, 1, mode="mean")
    _assert_pad(m, 1, mode="maximum")
