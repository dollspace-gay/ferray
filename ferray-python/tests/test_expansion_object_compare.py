"""#967 — object/structured comparison (equal/not_equal/less/greater).

ferray's comparison ufunc gate was string-only (`dtype.kind ∈ {'U','S'}`), so an
object (`'O'`) or structured/void (`'V'`) operand fell through to the real-only
numeric body and was rejected with ferray's `unsupported dtype for numeric op`
TypeError. numpy, by contrast:

- COMPUTES object comparison element-wise via the Python `==`/`<`/… on the stored
  objects, returning a bool array (verified live numpy 2.4.x).
- registers NO comparison loop for structured/void dtypes, raising its own
  `UFuncTypeError` / `_UFuncNoLoopError` (both `TypeError` subclasses).

These tests assert ferray now matches numpy per op per kind. Expected values come
from a live `np.<op>` call (R-CHAR-3 — never literal-copied from the ferray side).
"""

import numpy as np
import pytest

import ferray as fr

OPS = ["equal", "not_equal", "less", "greater"]


@pytest.fixture
def obj_pair():
    a = np.array([1, "a", 2], dtype=object)
    b = np.array([2, "a", 3], dtype=object)
    return a, b


@pytest.mark.parametrize("op", OPS)
def test_object_compare_matches_numpy(op, obj_pair):
    a, b = obj_pair
    expected = getattr(np, op)(a, b)
    got = getattr(fr, op)(a, b)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == np.bool_


@pytest.mark.parametrize("op", ["equal", "not_equal"])
def test_object_equality_self(op, obj_pair):
    a, _ = obj_pair
    expected = getattr(np, op)(a, a)
    got = getattr(fr, op)(a, a)
    np.testing.assert_array_equal(got, expected)


def test_object_vs_int_array_uses_object_loop():
    # numpy promotes object-vs-int through the object loop (Python ==), not a
    # numeric loop: np.equal(obj, int_arr) computes element-wise.
    o = np.array([1, 2, 3], dtype=object)
    other = np.array([1, 9, 3])
    np.testing.assert_array_equal(fr.equal(o, other), np.equal(o, other))
    np.testing.assert_array_equal(fr.not_equal(o, other), np.not_equal(o, other))


@pytest.mark.parametrize("op", OPS)
def test_structured_compare_raises_like_numpy(op):
    # numpy registers no comparison loop for structured/void dtypes; every op
    # raises a TypeError subclass (UFuncTypeError / _UFuncNoLoopError). ferray
    # must propagate numpy's exception, not its own "unsupported dtype" text.
    s = np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f4")])

    np_exc = None
    try:
        getattr(np, op)(s, s)
    except Exception as e:  # noqa: BLE001 - capture numpy's exact type
        np_exc = type(e)
    assert np_exc is not None and issubclass(np_exc, TypeError)

    with pytest.raises(TypeError):
        getattr(fr, op)(s, s)


@pytest.mark.parametrize("op", OPS)
def test_string_compare_unregressed(op):
    # #961 string comparison must stay green (delegated to numpy → bool).
    a = np.array(["a", "b", "c"])
    b = np.array(["b", "a", "c"])
    np.testing.assert_array_equal(getattr(fr, op)(a, b), getattr(np, op)(a, b))
