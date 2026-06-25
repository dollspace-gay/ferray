"""Drop-in vocabulary expansion (refs #818).

Asserts the remaining shared-vocabulary types/iterators, pure text-parse /
protocol-import helpers, array-protocol inspection, and the busdaycalendar
class are exposed on ``ferray`` (and the right submodules) AND that each IS /
matches numpy's object. ferray arrays are numpy.ndarray at the boundary, so
re-exposing numpy's object is the correct drop-in behavior. Expected values
come from a live ``numpy`` call (R-CHAR-3), never literal-copied.
"""

import importlib

import numpy as np
import pytest

import ferray as fr


AUDIT_TOP_LEVEL_NAMES = [
    "False_", "ScalarType", "True_", "core", "ctypeslib", "dtypes",
    "e", "euler_gamma", "exceptions", "f2py", "index_exp", "inf",
    "little_endian", "nan", "newaxis", "pi", "s_", "sctypeDict",
    "testing", "typecodes", "typing",
]

AUDIT_TOP_LEVEL_MODULES = [
    "core", "ctypeslib", "dtypes", "exceptions", "f2py", "testing", "typing",
]


# --- shared-vocabulary type/iterator objects (must BE numpy's) ---------------

@pytest.mark.parametrize("name", [
    "ufunc", "matrix", "asmatrix", "bmat", "memmap", "nested_iters",
    "__array_namespace_info__", "fromregex", "from_dlpack", "busdaycalendar",
])
def test_top_level_name_is_numpy_object(name):
    assert hasattr(fr, name), f"ferray.{name} missing"
    assert getattr(fr, name) is getattr(np, name)


def test_in_dunder_all():
    for name in ["ufunc", "matrix", "asmatrix", "bmat", "memmap",
                 "nested_iters", "__array_namespace_info__", "fromregex",
                 "from_dlpack", "busdaycalendar"]:
        assert name in fr.__all__, f"{name} not in ferray.__all__"


@pytest.mark.parametrize("name", AUDIT_TOP_LEVEL_NAMES)
def test_audit_top_level_name_exists_and_exports(name):
    assert hasattr(fr, name), f"ferray.{name} missing"
    assert name in fr.__all__, f"ferray.{name} missing from __all__"


@pytest.mark.parametrize("name", AUDIT_TOP_LEVEL_NAMES)
def test_audit_top_level_name_matches_numpy(name):
    got = getattr(fr, name)
    want = getattr(np, name)
    if name == "nan":
        assert np.isnan(got)
    elif isinstance(want, float):
        assert got == want
    else:
        assert got is want


@pytest.mark.parametrize("name", AUDIT_TOP_LEVEL_MODULES)
def test_audit_top_level_module_import_path(name):
    assert importlib.import_module(f"ferray.{name}") is getattr(fr, name)


def test_audit_ma_namespace_has_no_numpy_all_gap():
    missing = sorted(name for name in np.ma.__all__ if not hasattr(fr.ma, name))
    assert missing == []
    for name in ("core", "extras", "masked_print_option", "mr_"):
        assert name in fr.ma.__all__
        assert getattr(fr.ma, name) is getattr(np.ma, name)


@pytest.mark.parametrize("name", ["core", "extras"])
def test_audit_ma_module_import_path(name):
    assert importlib.import_module(f"ferray.ma.{name}") is getattr(fr.ma, name)


def test_audit_polynomial_namespace_has_no_numpy_all_gap():
    missing = sorted(
        name for name in np.polynomial.__all__ if not hasattr(fr.polynomial, name)
    )
    assert missing == []
    for name in (
        "chebyshev", "hermite", "hermite_e", "laguerre", "legendre",
        "polynomial", "set_default_printstyle",
    ):
        assert name in fr.polynomial.__all__
        assert getattr(fr.polynomial, name) is getattr(np.polynomial, name)


@pytest.mark.parametrize(
    "name",
    ["chebyshev", "hermite", "hermite_e", "laguerre", "legendre", "polynomial"],
)
def test_audit_polynomial_module_import_path(name):
    assert importlib.import_module(f"ferray.polynomial.{name}") is getattr(
        fr.polynomial, name
    )


def test_audit_char_constructor_exports():
    for name in ("array", "asarray", "chararray"):
        assert hasattr(fr.char, name)
        assert name in fr.char.__all__
        assert getattr(fr.char, name) is getattr(np.char, name)


def test_audit_lib_namespace_has_no_numpy_all_gap():
    missing = sorted(name for name in np.lib.__all__ if not hasattr(fr.lib, name))
    assert missing == []
    for name in np.lib.__all__:
        assert name in fr.lib.__all__, f"ferray.lib.{name} missing from __all__"


@pytest.mark.parametrize(
    "name", ["array_utils", "format", "introspect", "mixins", "npyio", "scimath"]
)
def test_audit_lib_module_import_path(name):
    assert importlib.import_module(f"ferray.lib.{name}") is getattr(fr.lib, name)


# --- ufunc: the type of every universal function -----------------------------

def test_ufunc_type():
    # The deliverable is the TYPE object `np.ufunc` exposed as `fr.ufunc`.
    # (ferray's own ufuncs are PyO3 Rust-backed callables, not numpy.ufunc
    # instances, so `isinstance(fr.add, fr.ufunc)` is intentionally NOT
    # asserted — that is a separate binding concern.)
    assert fr.ufunc is np.ufunc
    assert isinstance(np.add, fr.ufunc)


# --- matrix / asmatrix / bmat ------------------------------------------------

def test_matrix_construct():
    m = fr.matrix([[1, 2], [3, 4]])
    assert isinstance(m, np.matrix)
    assert np.array_equal(m, np.matrix([[1, 2], [3, 4]]))


def test_asmatrix():
    a = fr.array([[1, 2], [3, 4]])
    m = fr.asmatrix(a)
    assert isinstance(m, np.matrix)
    assert np.array_equal(m, np.asmatrix(np.array([[1, 2], [3, 4]])))


def test_bmat():
    a = np.matrix([[1, 2]])
    b = np.matrix([[3, 4]])
    got = fr.bmat([[a, b]])
    assert np.array_equal(got, np.bmat([[a, b]]))


# --- memmap ------------------------------------------------------------------

def test_memmap_roundtrip(tmp_path):
    path = tmp_path / "mm.dat"
    mm = fr.memmap(str(path), dtype="float64", mode="w+", shape=(3,))
    mm[:] = [1.0, 2.0, 3.0]
    mm.flush()
    ref = np.memmap(str(path), dtype="float64", mode="r", shape=(3,))
    assert isinstance(mm, np.memmap)
    assert np.array_equal(np.asarray(ref), np.array([1.0, 2.0, 3.0]))


# --- nested_iters ------------------------------------------------------------

def test_nested_iters():
    a = fr.arange(12).reshape(2, 3, 2)
    fr_i, fr_j = fr.nested_iters(a, [[0], [1, 2]])
    np_a = np.arange(12).reshape(2, 3, 2)
    np_i, np_j = np.nested_iters(np_a, [[0], [1, 2]])
    fr_vals = []
    for _ in fr_i:
        for _ in fr_j:
            fr_vals.append(int(fr_j.value))
    np_vals = []
    for _ in np_i:
        for _ in np_j:
            np_vals.append(int(np_j.value))
    assert fr_vals == np_vals


# --- __array_namespace_info__ ------------------------------------------------

def test_array_namespace_info():
    info = fr.__array_namespace_info__()
    assert info.default_dtypes() == np.__array_namespace_info__().default_dtypes()


# --- fromregex (pure text parse) ---------------------------------------------

def test_fromregex(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("1312 foo\n1534 bar\n444 baz\n")
    dt = [("num", np.int64), ("key", "S3")]
    got = fr.fromregex(str(path), r"(\d+)\s+(...)", dt)
    ref = np.fromregex(str(path), r"(\d+)\s+(...)", dt)
    assert np.array_equal(got, ref)


# --- from_dlpack (protocol import) -------------------------------------------

def test_from_dlpack():
    src = np.arange(5, dtype=np.float64)
    got = fr.from_dlpack(src)
    assert np.array_equal(np.asarray(got), src)


# --- busdaycalendar ----------------------------------------------------------

def test_busdaycalendar_defaults():
    cal = fr.busdaycalendar()
    ref = np.busdaycalendar()
    assert np.array_equal(cal.weekmask, ref.weekmask)
    assert np.array_equal(cal.holidays, ref.holidays)


def test_busdaycalendar_weekmask():
    cal = fr.busdaycalendar("1111100")
    ref = np.busdaycalendar("1111100")
    assert np.array_equal(cal.weekmask, ref.weekmask)


def test_busdaycalendar_holidays():
    hols = ["2026-01-01", "2026-07-04"]
    cal = fr.busdaycalendar(weekmask="1111100", holidays=hols)
    ref = np.busdaycalendar(weekmask="1111100", holidays=hols)
    assert np.array_equal(cal.holidays, ref.holidays)


def test_busdaycalendar_weekmask_count():
    # The weekmask carries the 7-day business-day pattern; for the standard
    # Mon-Fri mask there are exactly 5 business days.
    cal = fr.busdaycalendar(weekmask="1111100")
    ref = np.busdaycalendar(weekmask="1111100")
    assert int(np.count_nonzero(cal.weekmask)) == int(np.count_nonzero(ref.weekmask)) == 5


# --- char.chararray / char.array / char.asarray ------------------------------

def test_char_chararray_is_numpy():
    assert fr.char.chararray is np.char.chararray


def test_char_array():
    got = fr.char.array(["abc", "de"])
    ref = np.char.array(["abc", "de"])
    assert isinstance(got, np.char.chararray)
    assert np.array_equal(got, ref)


def test_char_asarray():
    got = fr.char.asarray(["abc", "de"])
    ref = np.char.asarray(["abc", "de"])
    assert isinstance(got, np.char.chararray)
    assert np.array_equal(got, ref)


# --- polynomial.set_default_printstyle ---------------------------------------

def test_set_default_printstyle_exists():
    assert fr.polynomial.set_default_printstyle is np.polynomial.set_default_printstyle


def test_set_default_printstyle_applies():
    # Pure-Python config; round-trip ascii<->unicode to confirm it runs and
    # matches numpy's effect on a polynomial repr.
    fr.polynomial.set_default_printstyle("ascii")
    p = fr.polynomial.Polynomial([1, 2, 3])
    np.polynomial.set_default_printstyle("ascii")
    q = np.polynomial.Polynomial([1, 2, 3])
    assert repr(p) == repr(q)
    fr.polynomial.set_default_printstyle("unicode")
    np.polynomial.set_default_printstyle("unicode")
