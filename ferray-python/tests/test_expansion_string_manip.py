"""String (``<U`` unicode / ``<S`` bytes) manipulation + select/order passthrough
for the main-array surface (REQ-2, blocker #960, epic #958).

Every ``fr.*`` data-move op (reshape/ravel/transpose/flip/roll/repeat/tile/
concatenate/stack/...) and select/order op (where/unique/sort/argsort/take/
compress/nonzero/setops/...) detects a fixed-width string array (dtype kind in
``{'U','S'}``) ahead of its real-only dtype-dispatch macro and delegates to
numpy's own op, returning the numpy string result directly (no transport —
strings never enter the Rust library, .design/ferray-core-string.md). numpy owns
the dtype passthrough, the concatenate/stack width-promotion (output ``<U{max}``)
and the lexicographic sort.

Every expected value is derived LIVE from numpy in-test (R-CHAR-3): each case
compares ``fr.<op>(...)`` against the corresponding ``np.<op>(...)`` call (values
+ dtype), never a literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


def _same(got, expected):
    """ferray result equals the numpy oracle in both values and dtype."""
    g = np.asarray(got)
    e = np.asarray(expected)
    assert np.array_equal(g, e), f"values: {g!r} != {e!r}"
    assert g.dtype == e.dtype, f"dtype: {g.dtype} != {e.dtype}"


def _same_list(got, expected):
    got = list(got)
    expected = list(expected)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        _same(g, e)


def _U(*items):
    return np.array(list(items))


def _S(*items):
    return np.array([s.encode() for s in items])


# ---------------------------------------------------------------------------
# Shape transforms — reshape / ravel / flatten / squeeze / expand_dims
# ---------------------------------------------------------------------------


def test_reshape_preserves_string_dtype():
    a = _U("a", "b", "c", "d")
    _same(fr.reshape(a, (2, 2)), np.reshape(a, (2, 2)))
    assert str(np.asarray(fr.reshape(a, (2, 2))).dtype) == "<U1"


def test_reshape_bytes_preserves_dtype():
    a = _S("ab", "cd", "ef", "gh")
    _same(fr.reshape(a, (2, 2)), np.reshape(a, (2, 2)))


def test_ravel_preserves_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.ravel(a), np.ravel(a))


def test_flatten_preserves_string_dtype():
    a = np.array([["ab", "cd"]])
    _same(fr.flatten(a), a.flatten())


def test_squeeze_preserves_string_dtype():
    a = np.array([["a", "b"]])
    _same(fr.squeeze(a), np.squeeze(a))


def test_expand_dims_preserves_string_dtype():
    a = _U("a", "b")
    _same(fr.expand_dims(a, 0), np.expand_dims(a, 0))


# ---------------------------------------------------------------------------
# Axis transforms — transpose / swapaxes / moveaxis
# ---------------------------------------------------------------------------


def test_transpose_preserves_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.transpose(a), np.transpose(a))


def test_swapaxes_preserves_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.swapaxes(a, 0, 1), np.swapaxes(a, 0, 1))


def test_moveaxis_preserves_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.moveaxis(a, 0, 1), np.moveaxis(a, 0, 1))


# ---------------------------------------------------------------------------
# Flips / rotation / roll
# ---------------------------------------------------------------------------


def test_flip_preserves_string_dtype():
    a = _U("a", "b", "c")
    _same(fr.flip(a), np.flip(a))


def test_fliplr_flipud_preserve_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.fliplr(a), np.fliplr(a))
    _same(fr.flipud(a), np.flipud(a))


def test_rot90_preserves_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same(fr.rot90(a), np.rot90(a))


def test_roll_preserves_string_dtype():
    a = _U("a", "b", "c")
    _same(fr.roll(a, 1), np.roll(a, 1))


# ---------------------------------------------------------------------------
# atleast_* / tile / repeat
# ---------------------------------------------------------------------------


def test_atleast_1d_2d_3d_preserve_string_dtype():
    a = _U("a")
    _same(fr.atleast_1d(a), np.atleast_1d(a))
    _same(fr.atleast_2d(a), np.atleast_2d(a))
    _same(fr.atleast_3d(a), np.atleast_3d(a))


def test_tile_preserves_string_dtype():
    a = _U("a", "b")
    _same(fr.tile(a, 2), np.tile(a, 2))


def test_repeat_preserves_string_dtype():
    a = _U("a", "b")
    _same(fr.repeat(a, 2), np.repeat(a, 2))


# ---------------------------------------------------------------------------
# concatenate / stack — width-promotion to max (numpy owns it)
# ---------------------------------------------------------------------------


def test_concatenate_width_promotes_to_max():
    out = fr.concatenate([_U("ab"), _U("cde")])
    exp = np.concatenate([_U("ab"), _U("cde")])
    _same(out, exp)
    # numpy promotes the output to the max input width.
    assert str(np.asarray(out).dtype) == "<U3"


def test_concatenate_bytes_width_promotes():
    out = fr.concatenate([_S("ab"), _S("cde")])
    exp = np.concatenate([_S("ab"), _S("cde")])
    _same(out, exp)
    assert str(np.asarray(out).dtype) == "|S3"


def test_stack_preserves_string_dtype():
    a, b = _U("a", "b"), _U("c", "d")
    _same(fr.stack([a, b]), np.stack([a, b]))


def test_vstack_hstack_column_dstack_preserve_string_dtype():
    a, b = _U("a", "b"), _U("c", "d")
    _same(fr.vstack([a, b]), np.vstack([a, b]))
    _same(fr.hstack([a, b]), np.hstack([a, b]))
    _same(fr.column_stack([a, b]), np.column_stack([a, b]))
    _same(fr.dstack([a, b]), np.dstack([a, b]))


def test_block_preserves_string_dtype():
    a, b = _U("a", "b"), _U("c", "d")
    _same(fr.block([[a], [b]]), np.block([[a], [b]]))


# ---------------------------------------------------------------------------
# delete / insert / append / resize / pad
# ---------------------------------------------------------------------------


def test_delete_preserves_string_dtype():
    a = _U("a", "b", "c")
    _same(fr.delete(a, 1, 0), np.delete(a, 1, 0))


def test_insert_width_promotes():
    a = _U("a", "c")
    _same(fr.insert(a, 1, "b", 0), np.insert(a, 1, "b", 0))


def test_insert_wider_value_promotes_width():
    a = _U("a", "c")
    _same(fr.insert(a, 1, "longer", 0), np.insert(a, 1, "longer", 0))


def test_append_preserves_string_dtype():
    a = _U("a", "b")
    _same(fr.append(a, _U("c")), np.append(a, _U("c")))


def test_resize_cycles_and_preserves_dtype():
    a = _U("a", "b")
    _same(fr.resize(a, (4,)), np.resize(a, (4,)))


def test_pad_fills_empty_string():
    a = _U("a", "b")
    _same(fr.pad(a, (1, 1)), np.pad(a, (1, 1)))


# ---------------------------------------------------------------------------
# split family — list of string arrays
# ---------------------------------------------------------------------------


def test_split_preserves_string_dtype():
    a = _U("a", "b", "c", "d")
    _same_list(fr.split(a, 2), np.split(a, 2))


def test_array_split_uneven_preserves_string_dtype():
    a = _U("a", "b", "c", "d", "e")
    _same_list(fr.array_split(a, 2), np.array_split(a, 2))


def test_vsplit_hsplit_preserve_string_dtype():
    a = np.array([["a", "b"], ["c", "d"]])
    _same_list(fr.vsplit(a, 2), np.vsplit(a, 2))
    _same_list(fr.hsplit(a, 2), np.hsplit(a, 2))


# ---------------------------------------------------------------------------
# Order — sort / argsort / partition / argpartition / lexsort / searchsorted
# ---------------------------------------------------------------------------


def test_sort_lexicographic():
    a = _U("c", "a", "b")
    _same(fr.sort(a), np.sort(a))


def test_argsort_returns_int64_indices():
    a = _U("c", "a", "b")
    _same(fr.argsort(a), np.argsort(a))


def test_partition_lexicographic():
    a = _U("c", "a", "b")
    _same(fr.partition(a, 1), np.partition(a, 1))


def test_argpartition_lexicographic():
    a = _U("c", "a", "b")
    _same(fr.argpartition(a, 1), np.argpartition(a, 1))


def test_lexsort_preserves_indices():
    keys = [_U("b", "a")]
    _same(fr.lexsort(keys), np.lexsort(keys))


def test_searchsorted_lexicographic_insertion():
    a = _U("a", "b", "c")
    _same(fr.searchsorted(a, "b"), np.searchsorted(a, "b"))


# ---------------------------------------------------------------------------
# unique / count_nonzero
# ---------------------------------------------------------------------------


def test_unique_sorted_unique():
    a = _U("b", "a", "b")
    _same(fr.unique(a), np.unique(a))


def test_unique_extended_returns_counts():
    a = _U("b", "a", "b")
    got = fr.unique_extended(a, return_counts=True)
    exp = np.unique(a, return_counts=True)
    _same(got[0], exp[0])
    _same(got[1], exp[1])


def test_count_nonzero_counts_non_empty():
    a = _U("", "a", "")
    _same(fr.count_nonzero(a), np.count_nonzero(a))


# ---------------------------------------------------------------------------
# Set operations — intersect1d / union1d / setdiff1d / setxor1d / isin / in1d
# ---------------------------------------------------------------------------


def test_intersect1d_lexicographic():
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.intersect1d(a, b), np.intersect1d(a, b))


def test_union1d_lexicographic():
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.union1d(a, b), np.union1d(a, b))


def test_setdiff1d_lexicographic():
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.setdiff1d(a, b), np.setdiff1d(a, b))


def test_setxor1d_lexicographic():
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.setxor1d(a, b), np.setxor1d(a, b))


def test_isin_membership_bool():
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.isin(a, b), np.isin(a, b))


def test_in1d_membership_bool():
    # numpy.in1d removed in 2.x — ferray's in1d delegates to numpy.isin.
    a, b = _U("a", "b"), _U("b", "c")
    _same(fr.in1d(a, b), np.isin(a, b))


# ---------------------------------------------------------------------------
# where — selection + width-promotion
# ---------------------------------------------------------------------------


def test_where_selects_strings():
    mask = np.array([True, False])
    _same(
        fr.where(mask, _U("a", "b"), _U("c", "d")),
        np.where(mask, _U("a", "b"), _U("c", "d")),
    )


def test_where_broadcasts_scalar_and_promotes_width():
    mask = np.array([True, False])
    _same(fr.where(mask, _U("a", "b"), "zz"), np.where(mask, _U("a", "b"), "zz"))


# ---------------------------------------------------------------------------
# Select / index — take / take_along_axis / compress / nonzero /
# flatnonzero / extract / argwhere
# ---------------------------------------------------------------------------


def test_take_preserves_string_dtype():
    a = _U("ab", "cde")
    _same(fr.take(a, [0, 1]), np.take(a, [0, 1]))


def test_take_along_axis_preserves_string_dtype():
    a = _U("a", "b", "c")
    idx = np.array([2, 0])
    _same(fr.take_along_axis(a, idx, 0), np.take_along_axis(a, idx, 0))


def test_compress_preserves_string_dtype():
    a = _U("a", "b")
    _same(fr.compress([True, False], a), np.compress([True, False], a))


def test_nonzero_of_non_empty_strings():
    a = _U("", "a", "")
    got = fr.nonzero(a)
    exp = np.nonzero(a)
    _same(got[0], exp[0])


def test_flatnonzero_of_non_empty_strings():
    a = _U("", "a", "")
    _same(fr.flatnonzero(a), np.flatnonzero(a))


def test_extract_preserves_string_dtype():
    a = _U("a", "b")
    mask = np.array([True, False])
    _same(fr.extract(mask, a), np.extract(mask, a))


def test_argwhere_of_non_empty_strings():
    a = _U("", "a")
    _same(fr.argwhere(a), np.argwhere(a))


# ---------------------------------------------------------------------------
# Regression guards — numeric / datetime real paths UNCHANGED
# ---------------------------------------------------------------------------


def test_numeric_sort_unregressed():
    a = np.array([3, 1, 2])
    _same(fr.sort(a), np.sort(a))


def test_numeric_concatenate_promotion_unregressed():
    out = fr.concatenate([np.array([1, 2]), np.array([3.0])])
    _same(out, np.concatenate([np.array([1, 2]), np.array([3.0])]))


def test_numeric_where_unregressed():
    mask = np.array([True, False, True])
    _same(
        fr.where(mask, np.array([1, 2, 3]), np.array([4, 5, 6])),
        np.where(mask, np.array([1, 2, 3]), np.array([4, 5, 6])),
    )


def test_datetime_reshape_unregressed():
    a = np.array(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"], dtype="datetime64[D]")
    _same(fr.reshape(a, (2, 2)), np.reshape(a, (2, 2)))


def test_complex_sort_unregressed():
    a = np.array([3 + 1j, 1 + 2j, 1 + 1j])
    _same(fr.sort(a), np.sort(a))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
