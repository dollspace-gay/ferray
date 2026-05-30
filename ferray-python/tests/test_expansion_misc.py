"""Expansion-misc parity tests (issue #836).

GROUP A — stacked / 2-component cross product (np.cross + np.linalg.cross).
GROUP B — np.poly with complex roots.

Every expected value comes from the live numpy 2.4.5 oracle (`import numpy as
np`), never copied from the ferray side (goal.md R-CHAR-3). 2-component cross
is deprecated-but-supported in numpy 2.4.5, so those calls are wrapped to
silence the DeprecationWarning on the numpy side.
"""

import warnings

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# GROUP A — stacked / 2-component cross
# ---------------------------------------------------------------------------


def test_cross_single_3vec():
    a, b = [1, 2, 3], [4, 5, 6]
    np.testing.assert_array_equal(np.asarray(fr.cross(a, b)), np.cross(a, b))


def test_cross_stacked_3vec():
    a = [[1, 0, 0], [0, 1, 0]]
    b = [[0, 1, 0], [0, 0, 1]]
    got = np.asarray(fr.cross(a, b))
    np.testing.assert_array_equal(got, np.cross(a, b))
    assert got.shape == (2, 3)


def test_cross_stacked_float():
    a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    b = [[7.0, 8.0, 9.0], [1.0, 0.0, -1.0]]
    np.testing.assert_allclose(np.asarray(fr.cross(a, b)), np.cross(a, b))


def test_cross_broadcast_single_against_stack():
    a = [1.0, 0.0, 0.0]
    b = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(np.asarray(fr.cross(a, b)), np.cross(a, b))


def test_cross_2component_scalar():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = np.cross([1, 2], [3, 4])
    got = np.asarray(fr.cross([1.0, 2.0], [3.0, 4.0]))
    np.testing.assert_allclose(got, expected)


def test_cross_stacked_2component():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = np.cross(a, b)
    np.testing.assert_allclose(np.asarray(fr.cross(a, b)), expected)


# ---------------------------------------------------------------------------
# GROUP A — np.linalg.cross (array-API, 3-component only, stacked)
# ---------------------------------------------------------------------------


def test_linalg_cross_single():
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    np.testing.assert_allclose(
        np.asarray(fr.linalg.cross(a, b)), np.linalg.cross(a, b)
    )


def test_linalg_cross_stacked():
    a = [[1.0, 0.0, 0.0]]
    b = [[0.0, 1.0, 0.0]]
    got = np.asarray(fr.linalg.cross(a, b))
    np.testing.assert_allclose(got, np.linalg.cross(a, b))
    assert got.shape == (1, 3)


def test_linalg_cross_stacked_multi():
    a = [[2.0, 3.0, 4.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    b = [[5.0, 6.0, 7.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    np.testing.assert_allclose(
        np.asarray(fr.linalg.cross(a, b)), np.linalg.cross(a, b)
    )


def test_linalg_cross_axis0():
    # Component axis is the FIRST axis.
    a = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(
        np.asarray(fr.linalg.cross(a, b, axis=0)),
        np.linalg.cross(a, b, axis=0),
    )


def test_linalg_cross_rejects_2component():
    # np.linalg.cross requires exactly 3 components.
    with pytest.raises(ValueError):
        fr.linalg.cross([1.0, 2.0], [3.0, 4.0])


# ---------------------------------------------------------------------------
# GROUP B — np.poly with complex roots
# ---------------------------------------------------------------------------


def test_poly_conjugate_pair_returns_real():
    roots = [1 + 2j, 1 - 2j]
    got = np.asarray(fr.poly(roots))
    expected = np.poly(roots)
    np.testing.assert_allclose(got, expected)
    # numpy returns a REAL array when roots are conjugate-closed.
    assert not np.iscomplexobj(got)


def test_poly_imaginary_conjugates_real():
    roots = [1j, -1j]
    got = np.asarray(fr.poly(roots))
    np.testing.assert_allclose(got, np.poly(roots))
    assert not np.iscomplexobj(got)


def test_poly_real_roots():
    roots = [1, 2, 3]
    np.testing.assert_allclose(np.asarray(fr.poly(roots)), np.poly(roots))


def test_poly_unpaired_complex_stays_complex():
    roots = [1 + 2j]
    got = np.asarray(fr.poly(roots))
    expected = np.poly(roots)
    np.testing.assert_allclose(got, expected)
    assert np.iscomplexobj(got)


def test_poly_complex_quartic():
    # Two conjugate pairs -> real quartic.
    roots = [1 + 1j, 1 - 1j, 2 + 3j, 2 - 3j]
    got = np.asarray(fr.poly(roots))
    np.testing.assert_allclose(got, np.poly(roots))
    assert not np.iscomplexobj(got)


def test_poly_empty():
    got = np.asarray(fr.poly([]))
    np.testing.assert_allclose(got, np.poly([]))
