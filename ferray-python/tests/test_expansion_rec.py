"""ferray.rec submodule + structured-array (record) data-model vocabulary.

These assert that `import ferray as fr` exposes the `numpy.rec` submodule and
the structured-array TYPES (recarray / record / void) and that they produce
results matching numpy. The structured dtype + recarray model is numpy's
SHARED boundary representation that ferray operates over (ferray arrays/dtypes
ARE numpy's), so re-exposing numpy's rec callables under `fr.rec` is the
correct drop-in behavior — the same rationale as the dtype-scalar types (#829)
and the ufunc/matrix/memmap vocabulary (#839). Refs #833.

Oracle: the live `numpy.rec` API. Every expected value is constructed by a
parallel `numpy.rec` call (R-CHAR-3 — never literal-copied from the ferray
side).
"""

import numpy as np
import pytest

import ferray as fr


REC_CALLABLES = (
    "array",
    "fromarrays",
    "fromrecords",
    "fromstring",
    "fromfile",
    "find_duplicate",
    "format_parser",
    "recarray",
    "record",
)


def test_rec_submodule_exists_and_registered():
    import types
    import sys

    assert isinstance(fr.rec, types.ModuleType)
    assert fr.rec.__name__ == "ferray.rec"
    # `from ferray.rec import fromarrays` must work via sys.modules.
    assert sys.modules["ferray.rec"] is fr.rec
    from ferray.rec import fromarrays as _fa  # noqa: F401


@pytest.mark.parametrize("name", REC_CALLABLES)
def test_rec_callable_present_and_is_numpy(name):
    assert hasattr(fr.rec, name), f"ferray.rec missing {name}"
    # ferray.rec re-exposes numpy's callables (shared data-model vocabulary).
    assert getattr(fr.rec, name) is getattr(np.rec, name)


def test_rec_all_surface_matches_numpy_subset():
    # Every callable numpy.rec advertises that we declared must be present.
    for name in REC_CALLABLES:
        assert name in fr.rec.__all__


def test_recarray_record_void_toplevel_are_numpy():
    assert fr.recarray is np.recarray
    assert fr.record is np.record
    assert fr.void is np.void


def test_fromarrays_matches_numpy():
    cols = [[1, 2, 3], [4.0, 5.0, 6.0]]
    got = fr.rec.fromarrays(cols, names="a,b")
    want = np.rec.fromarrays(cols, names="a,b")
    assert got.dtype == want.dtype
    assert isinstance(got, np.recarray)
    # field access by attribute (the defining recarray feature)
    assert np.array_equal(got.a, want.a)
    assert np.array_equal(got.b, want.b)
    assert got.a.dtype == np.int64
    assert got.b.dtype == np.float64


def test_fromrecords_matches_numpy():
    rows = [(1, "x"), (2, "y"), (3, "z")]
    got = fr.rec.fromrecords(rows, names="n,s")
    want = np.rec.fromrecords(rows, names="n,s")
    assert got.dtype == want.dtype
    assert np.array_equal(got.n, want.n)
    assert np.array_equal(got.s, want.s)


def test_recarray_construction_matches_numpy():
    dt = np.dtype([("x", np.int32), ("y", np.float64)])
    got = fr.recarray((2,), dtype=dt)
    want = np.recarray((2,), dtype=dt)
    assert isinstance(got, np.recarray)
    assert got.dtype == want.dtype
    assert got.shape == want.shape == (2,)
    # populate by field attribute and read it back
    got.x = [10, 20]
    got.y = [1.5, 2.5]
    assert np.array_equal(got.x, np.asarray([10, 20], dtype=np.int32))
    assert np.array_equal(got.y, np.asarray([1.5, 2.5]))


def test_record_scalar_is_numpy_record():
    r = fr.rec.fromarrays([[1, 2], [3.0, 4.0]], names="a,b")
    elem = r[0]
    want_elem = np.rec.fromarrays([[1, 2], [3.0, 4.0]], names="a,b")[0]
    assert isinstance(elem, np.record)
    assert isinstance(elem, fr.record)
    assert elem.a == want_elem.a
    assert elem.b == want_elem.b


def test_format_parser_matches_numpy():
    got = fr.rec.format_parser(
        formats=["i4", "f8"], names=["a", "b"], titles=None
    )
    want = np.rec.format_parser(
        formats=["i4", "f8"], names=["a", "b"], titles=None
    )
    assert got.dtype == want.dtype
    assert got.dtype.names == want.dtype.names == ("a", "b")


def test_find_duplicate_matches_numpy():
    seq = ["a", "b", "a", "c", "b"]
    assert fr.rec.find_duplicate(seq) == np.rec.find_duplicate(seq)
