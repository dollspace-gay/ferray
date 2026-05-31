"""Parity tests for the default `axis` of fr.sort / fr.argsort vs numpy (#949).

numpy.sort / numpy.argsort default to ``axis=-1`` (sort along the LAST axis);
an explicit ``axis=None`` flattens. A prior ferray pass defaulted the binding
to ``axis=None`` (flatten), diverging from numpy for any 2-D+ input. These
tests pin the numpy-faithful default (R-CHAR-3: every expected value comes from
a live ``numpy`` call, never literal-copied from ferray).
"""

import ferray as fr
import numpy as np
import pytest


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


def test_sort_2d_default_is_last_axis():
    a = np.array([[3, 1], [2, 4]])
    # numpy default axis=-1 -> per-row sort, shape preserved.
    assert _eq(fr.sort(a), np.sort(a))
    assert np.asarray(fr.sort(a)).shape == a.shape


def test_sort_2d_axis0_column():
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    assert _eq(fr.sort(a, axis=0), np.sort(a, axis=0))


def test_sort_2d_axis_minus1_explicit():
    a = np.array([[3, 1, 4], [1, 5, 9]])
    assert _eq(fr.sort(a, axis=-1), np.sort(a, axis=-1))


def test_sort_2d_axis_none_flattens():
    a = np.array([[3, 1], [2, 4]])
    got = np.asarray(fr.sort(a, axis=None))
    assert _eq(got, np.sort(a, axis=None))
    assert got.shape == (a.size,)


def test_sort_1d_unchanged():
    a = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    assert _eq(fr.sort(a), np.sort(a))


def test_sort_3d_default_last_axis():
    a = np.arange(24).reshape(2, 3, 4)[:, :, ::-1].copy()
    assert _eq(fr.sort(a), np.sort(a))


def test_sort_3d_middle_axis():
    a = np.arange(24).reshape(2, 3, 4)[:, ::-1, :].copy()
    assert _eq(fr.sort(a, axis=1), np.sort(a, axis=1))


# ---------------------------------------------------------------------------
# argsort
# ---------------------------------------------------------------------------


def test_argsort_2d_default_is_last_axis():
    a = np.array([[3, 1], [2, 4]])
    assert _eq(fr.argsort(a), np.argsort(a))


def test_argsort_2d_axis0():
    a = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5]])
    assert _eq(fr.argsort(a, axis=0), np.argsort(a, axis=0))


def test_argsort_2d_axis_none_flattens():
    a = np.array([[3, 1], [2, 4]])
    got = np.asarray(fr.argsort(a, axis=None))
    assert _eq(got, np.argsort(a, axis=None))
    assert got.shape == (a.size,)


def test_argsort_1d_unchanged():
    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert _eq(fr.argsort(a), np.argsort(a))


def test_argsort_3d_default_last_axis():
    a = np.arange(24).reshape(2, 3, 4)[:, :, ::-1].copy()
    assert _eq(fr.argsort(a), np.argsort(a))


# ---------------------------------------------------------------------------
# inherited dispatch arms (complex / datetime) get the default too
# ---------------------------------------------------------------------------


def test_complex_sort_2d_default_last_axis():
    a = np.array([[3 + 1j, 1 + 2j], [1 + 1j, 2 - 1j]])
    assert _eq(fr.sort(a), np.sort(a))
    assert _eq(fr.argsort(a), np.argsort(a))


def test_datetime_sort_2d_default_last_axis():
    a = np.array(
        [["2020-01-03", "2020-01-01"], ["2020-01-04", "2020-01-02"]],
        dtype="datetime64[D]",
    )
    assert _eq(fr.sort(a), np.sort(a))
    assert _eq(fr.argsort(a), np.argsort(a))


# ---------------------------------------------------------------------------
# axis validation matches numpy's AxisError
# ---------------------------------------------------------------------------


def test_sort_axis_out_of_bounds_raises_axiserror():
    a = np.array([[3, 1], [2, 4]])
    with pytest.raises(np.exceptions.AxisError):
        fr.sort(a, axis=5)
    with pytest.raises(np.exceptions.AxisError):
        fr.argsort(a, axis=5)
