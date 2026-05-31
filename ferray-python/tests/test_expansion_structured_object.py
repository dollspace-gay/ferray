"""Structured/void (``V``) + object (``O``) dtype construction and manipulation
passthrough (blocker #964, the last flexible-dtype gaps).

``fr.array``/``asarray``/``zeros``/``ones``/``empty``/``full``/``*_like`` accept
structured dtype specs (``dtype=[('a','i4'),('b','f4')]`` -> kind ``'V'``) and
the object dtype (``dtype=object`` -> kind ``'O'``, incl. mixed / ragged input),
delegating construction to numpy (which owns the tuple-spec parse, the field
layout, and the object/ragged passthrough) and returning the numpy ndarray
unchanged. The data-move / select / order ops (reshape/transpose/concatenate/
flip/roll/repeat/tile/take/where/sort/...) preserve the structured/object dtype
by the same detect-and-delegate seam (broadened from the string ``<U``/``<S``
seam, #960, to kind ``∈ {U,S,V,O}``). Structured/object never enter the Rust
library — they ride the binding boundary as numpy ndarrays.

Every expected value is derived LIVE from numpy in-test (R-CHAR-3): each case
compares ``fr.<fn>(...)`` against the corresponding ``np.<fn>(...)`` call (values
+ dtype), never a literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr

STRUCT_SPEC = [("a", "i4"), ("b", "f4")]


def _same(got, expected):
    """ferray result equals the numpy oracle in both values and dtype."""
    g = np.asarray(got)
    assert np.array_equal(g, expected), f"values: {g!r} != {expected!r}"
    assert g.dtype == expected.dtype, f"dtype: {g.dtype} != {expected.dtype}"


# ---------------------------------------------------------------------------
# Structured (void) construction
# ---------------------------------------------------------------------------


def test_array_structured_from_tuples():
    _same(
        fr.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC),
        np.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC),
    )


def test_array_structured_kind_is_void():
    a = np.asarray(fr.array([(1, 2.0)], dtype=STRUCT_SPEC))
    assert a.dtype.kind == "V"
    assert str(a.dtype) == str(np.array([(1, 2.0)], dtype=STRUCT_SPEC).dtype)


def test_array_structured_field_access_a():
    a = np.asarray(fr.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC))
    exp = np.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC)
    assert a["a"].tolist() == exp["a"].tolist()
    assert a["a"].dtype == exp["a"].dtype  # int32


def test_array_structured_field_access_b():
    a = np.asarray(fr.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC))
    exp = np.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC)
    assert a["b"].tolist() == exp["b"].tolist()
    assert a["b"].dtype == exp["b"].dtype  # float32


def test_asarray_structured():
    _same(
        fr.asarray([(5, 6.0)], dtype=STRUCT_SPEC),
        np.asarray([(5, 6.0)], dtype=STRUCT_SPEC),
    )


def test_zeros_structured():
    _same(fr.zeros(2, dtype=STRUCT_SPEC), np.zeros(2, dtype=STRUCT_SPEC))


def test_zeros_structured_tuple_shape():
    _same(fr.zeros((2, 3), dtype=STRUCT_SPEC), np.zeros((2, 3), dtype=STRUCT_SPEC))


def test_ones_structured():
    _same(fr.ones(2, dtype=STRUCT_SPEC), np.ones(2, dtype=STRUCT_SPEC))


def test_empty_structured_defined_buffer():
    # `empty` -> defined (zero) buffer per the binding contract; compare to zeros.
    _same(fr.empty(2, dtype=STRUCT_SPEC), np.zeros(2, dtype=STRUCT_SPEC))


def test_full_structured():
    _same(
        fr.full(2, (7, 8.0), dtype=STRUCT_SPEC),
        np.full(2, (7, 8.0), dtype=STRUCT_SPEC),
    )


def test_zeros_structured_from_dtype_instance():
    dt = np.dtype(STRUCT_SPEC)
    _same(fr.zeros(3, dtype=dt), np.zeros(3, dtype=dt))


# ---------------------------------------------------------------------------
# Object construction
# ---------------------------------------------------------------------------


def test_array_object_mixed():
    _same(fr.array([1, "a", [2, 3]], dtype=object), np.array([1, "a", [2, 3]], dtype=object))


def test_array_object_kind():
    a = np.asarray(fr.array([1, "a"], dtype=object))
    assert a.dtype.kind == "O"


def test_array_object_ragged():
    _same(fr.array([1, [2, 3]], dtype=object), np.array([1, [2, 3]], dtype=object))


def test_array_object_values_preserved():
    a = np.asarray(fr.array([1, "a", [2, 3]], dtype=object))
    exp = np.array([1, "a", [2, 3]], dtype=object)
    assert a.tolist() == exp.tolist()


def test_asarray_object():
    _same(fr.asarray([1, "x"], dtype=object), np.asarray([1, "x"], dtype=object))


def test_zeros_object():
    # numpy `np.zeros(2, object)` is `[0, 0]` (ints), not None.
    _same(fr.zeros(2, dtype=object), np.zeros(2, dtype=object))


def test_ones_object():
    _same(fr.ones(2, dtype=object), np.ones(2, dtype=object))


def test_empty_object_defined_buffer():
    _same(fr.empty(2, dtype=object), np.zeros(2, dtype=object))


def test_full_object():
    _same(fr.full(3, "hi", dtype=object), np.full(3, "hi", dtype=object))


# ---------------------------------------------------------------------------
# *_like preserves structured/object
# ---------------------------------------------------------------------------


def test_zeros_like_structured():
    src = np.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC)
    _same(fr.zeros_like(src), np.zeros_like(src))


def test_ones_like_structured():
    src = np.array([(1, 2.0)], dtype=STRUCT_SPEC)
    _same(fr.ones_like(src), np.ones_like(src))


def test_full_like_structured():
    src = np.array([(1, 2.0), (3, 4.0)], dtype=STRUCT_SPEC)
    _same(fr.full_like(src, (9, 9.0)), np.full_like(src, (9, 9.0)))


def test_zeros_like_object():
    src = np.array([1, "a", [2, 3]], dtype=object)
    _same(fr.zeros_like(src), np.zeros_like(src))


def test_full_like_object():
    src = np.array([1, 2, 3], dtype=object)
    _same(fr.full_like(src, "z"), np.full_like(src, "z"))


# ---------------------------------------------------------------------------
# Manipulation passthrough — structured preserves dtype + field layout
# ---------------------------------------------------------------------------

_STRUCT = np.array([(1, 2.0), (3, 4.0), (5, 6.0)], dtype=STRUCT_SPEC)
_STRUCT2D = np.array([[(1, 2.0), (3, 4.0)], [(5, 6.0), (7, 8.0)]], dtype=STRUCT_SPEC)


def test_reshape_structured():
    _same(fr.reshape(_STRUCT2D, (4,)), np.reshape(_STRUCT2D, (4,)))


def test_ravel_structured():
    _same(fr.ravel(_STRUCT2D), np.ravel(_STRUCT2D))


def test_transpose_structured():
    _same(fr.transpose(_STRUCT2D), np.transpose(_STRUCT2D))


def test_flip_structured():
    _same(fr.flip(_STRUCT), np.flip(_STRUCT))


def test_roll_structured():
    _same(fr.roll(_STRUCT, 1), np.roll(_STRUCT, 1))


def test_repeat_structured():
    _same(fr.repeat(_STRUCT, 2), np.repeat(_STRUCT, 2))


def test_tile_structured():
    _same(fr.tile(_STRUCT, 2), np.tile(_STRUCT, 2))


def test_concatenate_structured():
    _same(fr.concatenate([_STRUCT, _STRUCT]), np.concatenate([_STRUCT, _STRUCT]))


def test_stack_structured():
    _same(fr.stack([_STRUCT, _STRUCT]), np.stack([_STRUCT, _STRUCT]))


def test_squeeze_structured():
    src = np.array([[(1, 2.0)]], dtype=STRUCT_SPEC)
    _same(fr.squeeze(src), np.squeeze(src))


def test_take_structured():
    _same(fr.take(_STRUCT, [0, 2]), np.take(_STRUCT, [0, 2]))


def test_structured_field_after_reshape():
    g = np.asarray(fr.reshape(_STRUCT2D, (4,)))
    exp = np.reshape(_STRUCT2D, (4,))
    assert g["a"].tolist() == exp["a"].tolist()


# ---------------------------------------------------------------------------
# Manipulation passthrough — object preserves dtype + payload
# ---------------------------------------------------------------------------

_OBJ = np.array([3, 1, 2], dtype=object)
_OBJ_MIX = np.array([1, "a", [2, 3]], dtype=object)
_OBJ2D = np.array([[1, 2], [3, 4]], dtype=object)


def test_reshape_object():
    _same(fr.reshape(_OBJ2D, (4,)), np.reshape(_OBJ2D, (4,)))


def test_transpose_object():
    _same(fr.transpose(_OBJ2D), np.transpose(_OBJ2D))


def test_flip_object():
    _same(fr.flip(_OBJ_MIX), np.flip(_OBJ_MIX))


def test_roll_object():
    _same(fr.roll(_OBJ_MIX, 1), np.roll(_OBJ_MIX, 1))


def test_repeat_object():
    _same(fr.repeat(_OBJ, 2), np.repeat(_OBJ, 2))


def test_tile_object():
    _same(fr.tile(_OBJ, 2), np.tile(_OBJ, 2))


def test_concatenate_object():
    _same(fr.concatenate([_OBJ, _OBJ]), np.concatenate([_OBJ, _OBJ]))


def test_sort_object_by_comparison():
    # numpy sorts object arrays by Python comparison.
    _same(fr.sort(_OBJ), np.sort(_OBJ))


def test_argsort_object():
    g = np.asarray(fr.argsort(_OBJ))
    exp = np.argsort(_OBJ)
    assert g.tolist() == exp.tolist()


def test_unique_object():
    _same(fr.unique(np.array([1, 1, 2, 2, 3], dtype=object)),
          np.unique(np.array([1, 1, 2, 2, 3], dtype=object)))


def test_nonzero_object():
    g = fr.nonzero(np.array([0, 1, 0, 2], dtype=object))
    exp = np.nonzero(np.array([0, 1, 0, 2], dtype=object))
    assert [np.asarray(x).tolist() for x in g] == [x.tolist() for x in exp]


def test_where_object():
    cond = np.array([True, False, True])
    x = np.array([1, 2, 3], dtype=object)
    y = np.array(["a", "b", "c"], dtype=object)
    _same(fr.where(cond, x, y), np.where(cond, x, y))


def test_take_object():
    _same(fr.take(_OBJ, [0, 2]), np.take(_OBJ, [0, 2]))


# ---------------------------------------------------------------------------
# Regression guard — real / numeric / string / complex paths UNCHANGED
# ---------------------------------------------------------------------------


def test_real_float_unchanged():
    _same(fr.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    _same(fr.zeros(3), np.zeros(3))
    _same(fr.reshape(np.arange(6.0), (2, 3)), np.reshape(np.arange(6.0), (2, 3)))


def test_real_int_unchanged():
    _same(fr.array([1, 2, 3], dtype="int64"), np.array([1, 2, 3], dtype="int64"))
    _same(fr.sort(np.array([3, 1, 2])), np.sort(np.array([3, 1, 2])))


def test_string_unchanged():
    _same(fr.array(["ab", "cd"]), np.array(["ab", "cd"]))
    _same(fr.zeros(2, dtype="U5"), np.zeros(2, dtype="U5"))
    _same(fr.concatenate([np.array(["ab"]), np.array(["cde"])]),
          np.concatenate([np.array(["ab"]), np.array(["cde"])]))


def test_complex_unchanged():
    _same(fr.array([1 + 2j, 3 - 4j]), np.array([1 + 2j, 3 - 4j]))
    _same(fr.zeros(3, dtype="complex128"), np.zeros(3, dtype="complex128"))


def test_float16_unchanged():
    _same(fr.array([1.0, 2.0], dtype="float16"), np.array([1.0, 2.0], dtype="float16"))


def test_datetime_unchanged():
    src = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    _same(fr.array(src), src)
    _same(fr.reshape(src, (2,)), np.reshape(src, (2,)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
