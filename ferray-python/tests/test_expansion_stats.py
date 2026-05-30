"""Parity tests for the stats/ufunc expansion batch (refs #818).

Each test compares `ferray` against the live `numpy` oracle (numpy 2.4.x
installed), so the expected values are never literal-copied from the ferray
side (R-CHAR-3). The functions exercised here are the top-level numpy
functions whose ferray-library impls existed but were unbound through
ferray-python, plus the nan-aware reductions:

    all / any (+ axis), average, nancumsum / nancumprod,
    nanpercentile / nanquantile, around(decimals) / round(decimals),
    nan_to_num, sort_complex, cross, trace, unwrap.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# all / any
# ---------------------------------------------------------------------------

def test_all_true():
    assert bool(fr.all([1, 1, 1]).item()) == bool(np.all([1, 1, 1]))


def test_all_false():
    assert bool(fr.all([1, 0, 1]).item()) == bool(np.all([1, 0, 1]))


def test_any_true():
    assert bool(fr.any([0, 1, 0]).item()) == bool(np.any([0, 1, 0]))


def test_any_false():
    assert bool(fr.any([0, 0, 0]).item()) == bool(np.any([0, 0, 0]))


def test_all_bool_dtype():
    # numpy returns a bool_ scalar; ferray returns a 0-d bool array.
    assert str(np.asarray(fr.all([1, 1])).dtype) == "bool"


def test_all_axis0():
    a = [[1, 0], [1, 1]]
    np.testing.assert_array_equal(np.asarray(fr.all(a, axis=0)), np.all(a, axis=0))


def test_any_axis1():
    a = [[1, 0], [0, 0]]
    np.testing.assert_array_equal(np.asarray(fr.any(a, axis=1)), np.any(a, axis=1))


def test_all_float_input():
    a = [1.0, 0.0, 2.0]
    assert bool(fr.all(a).item()) == bool(np.all(a))


# ---------------------------------------------------------------------------
# average
# ---------------------------------------------------------------------------

def test_average_no_weights():
    a = [1, 2, 3, 4]
    assert fr.average(a).item() == pytest.approx(np.average(a))


def test_average_weighted():
    a = [1, 2, 3]
    w = [1, 2, 3]
    assert fr.average(a, weights=w).item() == pytest.approx(np.average(a, weights=w))


def test_average_axis():
    a = [[1, 2], [3, 4]]
    np.testing.assert_allclose(np.asarray(fr.average(a, axis=0)), np.average(a, axis=0))


def test_average_weighted_axis():
    a = [[1.0, 2.0], [3.0, 4.0]]
    w = [1.0, 3.0]
    np.testing.assert_allclose(
        np.asarray(fr.average(a, axis=0, weights=w)),
        np.average(a, axis=0, weights=w),
    )


# ---------------------------------------------------------------------------
# nancumsum / nancumprod
# ---------------------------------------------------------------------------

def test_nancumsum():
    a = [1.0, np.nan, 2.0, np.nan, 3.0]
    np.testing.assert_allclose(np.asarray(fr.nancumsum(a)), np.nancumsum(a))


def test_nancumprod():
    a = [1.0, np.nan, 2.0, 3.0]
    np.testing.assert_allclose(np.asarray(fr.nancumprod(a)), np.nancumprod(a))


def test_nancumsum_int_promotes():
    a = [1, 2, 3]
    np.testing.assert_allclose(np.asarray(fr.nancumsum(a)), np.nancumsum(a))


# ---------------------------------------------------------------------------
# nanpercentile / nanquantile
# ---------------------------------------------------------------------------

def test_nanpercentile_scalar():
    a = [1.0, 2.0, np.nan, 4.0]
    assert fr.nanpercentile(a, 50).item() == pytest.approx(np.nanpercentile(a, 50))


def test_nanpercentile_sequence():
    a = [1.0, 2.0, np.nan, 4.0]
    np.testing.assert_allclose(
        np.asarray(fr.nanpercentile(a, [25, 50, 75])),
        np.nanpercentile(a, [25, 50, 75]),
    )


def test_nanquantile_scalar():
    a = [1.0, 2.0, np.nan, 4.0]
    assert fr.nanquantile(a, 0.5).item() == pytest.approx(np.nanquantile(a, 0.5))


def test_nanquantile_sequence():
    a = [1.0, 2.0, np.nan, 4.0]
    np.testing.assert_allclose(
        np.asarray(fr.nanquantile(a, [0.25, 0.75])),
        np.nanquantile(a, [0.25, 0.75]),
    )


# ---------------------------------------------------------------------------
# around / round with decimals
# ---------------------------------------------------------------------------

def test_around_decimals_scalar():
    assert fr.around(3.14159, 2).item() == pytest.approx(np.around(3.14159, 2))


def test_around_decimals_array():
    a = [1.234, 5.678, -2.345]
    np.testing.assert_allclose(np.asarray(fr.around(a, 2)), np.around(a, 2))


def test_around_banker_rounding():
    # half-to-even: 2.5 -> 2, 3.5 -> 4 (matches numpy)
    a = [0.5, 1.5, 2.5, 3.5]
    np.testing.assert_allclose(np.asarray(fr.around(a, 0)), np.around(a, 0))


def test_around_negative_decimals():
    np.testing.assert_allclose(
        np.asarray(fr.around([123.456, 789.0], -1)),
        np.around([123.456, 789.0], -1),
    )


def test_round_alias_decimals():
    a = [1.111, 2.222]
    np.testing.assert_allclose(np.asarray(fr.round(a, 2)), np.round(a, 2))


def test_around_int_identity():
    a = np.array([1, 2, 3])
    out = fr.around(a, 0)
    np.testing.assert_array_equal(np.asarray(out), np.around(a, 0))
    assert str(np.asarray(out).dtype) == str(np.around(a, 0).dtype)


# ---------------------------------------------------------------------------
# nan_to_num
# ---------------------------------------------------------------------------

def test_nan_to_num_default():
    a = np.array([np.nan, np.inf, -np.inf, 1.0])
    np.testing.assert_array_equal(np.asarray(fr.nan_to_num(a)), np.nan_to_num(a))


def test_nan_to_num_custom():
    a = np.array([np.nan, np.inf, -np.inf, 2.0])
    np.testing.assert_allclose(
        np.asarray(fr.nan_to_num(a, nan=-1.0, posinf=99.0, neginf=-99.0)),
        np.nan_to_num(a, nan=-1.0, posinf=99.0, neginf=-99.0),
    )


def test_nan_to_num_int_unchanged():
    a = np.array([1, 2, 3])
    np.testing.assert_array_equal(np.asarray(fr.nan_to_num(a)), np.nan_to_num(a))


# ---------------------------------------------------------------------------
# sort_complex
# ---------------------------------------------------------------------------

def test_sort_complex():
    a = [3 + 1j, 1 + 2j, 1 + 1j, 2 + 0j]
    np.testing.assert_array_equal(np.asarray(fr.sort_complex(a)), np.sort_complex(a))


def test_sort_complex_real_input():
    a = [3.0, 1.0, 2.0]
    np.testing.assert_array_equal(np.asarray(fr.sort_complex(a)), np.sort_complex(a))


# ---------------------------------------------------------------------------
# cross (1-D 3-vector)
# ---------------------------------------------------------------------------

def test_cross_unit_vectors():
    np.testing.assert_array_equal(
        np.asarray(fr.cross([1, 0, 0], [0, 1, 0])), np.cross([1, 0, 0], [0, 1, 0])
    )


def test_cross_float():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    np.testing.assert_allclose(np.asarray(fr.cross(a, b)), np.cross(a, b))


# ---------------------------------------------------------------------------
# trace
# ---------------------------------------------------------------------------

def test_trace():
    a = [[1.0, 2.0], [3.0, 4.0]]
    assert fr.trace(a).item() == pytest.approx(np.trace(a))


def test_trace_rectangular():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert fr.trace(a).item() == pytest.approx(np.trace(a))


# ---------------------------------------------------------------------------
# unwrap
# ---------------------------------------------------------------------------

def test_unwrap():
    a = [0.0, np.pi, 2 * np.pi, 4 * np.pi]
    np.testing.assert_allclose(np.asarray(fr.unwrap(a)), np.unwrap(a))


def test_unwrap_wrapped_phase():
    phase = np.linspace(0, 6 * np.pi, 20)
    wrapped = (phase + np.pi) % (2 * np.pi) - np.pi
    np.testing.assert_allclose(np.asarray(fr.unwrap(wrapped)), np.unwrap(wrapped))
