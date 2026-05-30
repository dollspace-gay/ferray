"""Drop-in vocabulary parity: ferray must expose every numpy dtype-scalar
type object and array-protocol iterator/info class so that `import ferray as
np` is a true drop-in for these shared-boundary names.

ferray arrays ARE numpy.ndarray at the Python boundary, and ferray's
creation/dtype functions already accept and produce numpy dtypes. The
dtype-scalar TYPE OBJECTS (np.float64, np.int8, np.complex128, np.bool_, ...)
and the array-protocol classes (np.ndarray, np.nditer, np.ndindex,
np.ndenumerate, np.broadcast, np.flatiter, np.poly1d) are the shared
vocabulary of that boundary — exposing them as the numpy objects is the
correct drop-in behavior (the Rust *computation* is what ferray translates;
the dtype type-objects are the shared boundary vocabulary). Every expected
value here comes from a live numpy call (R-CHAR-3).

refs #818.
"""

import numpy as np
import pytest

import ferray as fr


# Concrete dtype-scalar type objects (sized + C-named aliases).
CONCRETE_SCALAR_TYPES = [
    "bool", "bool_", "byte", "bytes_", "cdouble", "clongdouble",
    "complex64", "complex128", "complex256", "csingle", "datetime64",
    "double", "float16", "float32", "float64", "float128", "half",
    "int8", "int16", "int32", "int64", "int_", "intc", "intp",
    "long", "longdouble", "longlong", "object_", "short", "single", "str_",
    "timedelta64", "ubyte", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintc", "uintp", "ulong", "ulonglong", "ushort", "void",
]

# Abstract scalar-type bases (the numpy type hierarchy).
ABSTRACT_BASES = [
    "generic", "number", "integer", "signedinteger", "unsignedinteger",
    "inexact", "floating", "complexfloating", "flexible", "character",
]

# Array type + array-protocol iterator/info classes.
PROTOCOL_CLASSES = [
    "ndarray", "broadcast", "nditer", "ndindex", "ndenumerate", "flatiter",
    "poly1d",
]

ALL_TYPE_NAMES = CONCRETE_SCALAR_TYPES + ABSTRACT_BASES + PROTOCOL_CLASSES


@pytest.mark.parametrize("name", ALL_TYPE_NAMES)
def test_name_exists_and_is_numpy_object(name):
    """Each exposed name must exist on ferray and BE the numpy object — the
    boundary type is the numpy type (oracle: getattr(np, name))."""
    assert hasattr(fr, name), f"ferray missing {name!r}"
    assert getattr(fr, name) is getattr(np, name), (
        f"fr.{name} is not np.{name} (drop-in identity broken)"
    )


@pytest.mark.parametrize("name", ALL_TYPE_NAMES)
def test_name_in_all(name):
    """Every exposed type/class name is re-exported via __all__."""
    assert name in fr.__all__


# --- concrete scalar types behave as numpy dtype keys -----------------------

# dtype kinds that np.dtype() can construct directly (excludes object_/void/
# flexible-base which need special handling but still resolve via np.dtype).
DTYPE_CONSTRUCTIBLE = [
    "bool_", "byte", "cdouble", "complex64", "complex128", "complex256",
    "csingle", "double", "float16", "float32", "float64", "float128", "half",
    "int8", "int16", "int32", "int64", "int_", "intc", "intp",
    "long", "longdouble", "longlong", "short", "single",
    "ubyte", "uint", "uint8", "uint16", "uint32", "uint64",
    "uintc", "uintp", "ulong", "ulonglong", "ushort",
]


@pytest.mark.parametrize("name", DTYPE_CONSTRUCTIBLE)
def test_scalar_type_is_dtype_key(name):
    """np.dtype(fr.<scalar>) resolves identically to np.dtype(np.<scalar>) —
    the type object is a usable dtype key on the shared boundary."""
    assert np.dtype(getattr(fr, name)) == np.dtype(getattr(np, name))


def test_zeros_accepts_string_dtype_from_scalar_type():
    """ferray's creation functions take the dtype *name* string; the scalar
    type object names match numpy's, so a zeros() call keyed by that dtype
    name produces the matching dtype (oracle: numpy's np.dtype name).

    Restricted to the dtypes ferray's Rust creation kernels currently
    support (bool/int/uint/float32/64). Complex creation + accepting the
    numpy *type object* directly (not just its name string) are Rust-crate
    boundary gaps tracked separately — see the spillover blockers filed
    under epic #818 — not part of this Python-vocabulary exposure."""
    for name in ["float64", "int8", "uint16", "float32", "int64", "bool_"]:
        dt_name = np.dtype(getattr(fr, name)).name
        a = fr.zeros(3, dtype=dt_name)
        assert a.dtype == np.dtype(getattr(np, name))


def test_bool_is_numpy_bool_not_builtin():
    """fr.bool / fr.bool_ are numpy's bool scalar type, matching numpy 2.x
    where np.bool is numpy.bool_ (NOT Python's builtins.bool)."""
    import builtins
    assert fr.bool is np.bool_
    assert fr.bool_ is np.bool_
    assert fr.bool is not builtins.bool


# --- abstract base hierarchy -------------------------------------------------

def test_abstract_hierarchy_matches_numpy():
    """The numpy scalar-type subclass relationships hold through ferray's
    re-exposed bases (oracle: numpy issubclass results)."""
    assert issubclass(fr.float64, fr.floating)
    assert issubclass(fr.floating, fr.inexact)
    assert issubclass(fr.complex128, fr.complexfloating)
    assert issubclass(fr.int32, fr.signedinteger)
    assert issubclass(fr.uint32, fr.unsignedinteger)
    assert issubclass(fr.int32, fr.integer)
    assert issubclass(fr.integer, fr.number)
    assert issubclass(fr.number, fr.generic)
    assert issubclass(fr.float64, np.floating)  # cross-check vs numpy itself


def test_issubdtype_via_exposed_bases():
    """ferray.issubdtype agrees with numpy when keyed by the exposed bases."""
    assert fr.issubdtype(fr.int64, fr.integer) == np.issubdtype(np.int64, np.integer)
    assert fr.issubdtype(fr.float32, fr.floating) == np.issubdtype(np.float32, np.floating)


# --- array type + protocol classes ------------------------------------------

def test_ndarray_isinstance():
    """fr.ndarray is np.ndarray; ferray arrays ARE numpy.ndarray, so
    isinstance holds (oracle: numpy isinstance)."""
    a = fr.zeros((2, 3))
    assert fr.ndarray is np.ndarray
    assert isinstance(a, fr.ndarray)
    assert isinstance(a, np.ndarray)


def test_ndindex_iteration():
    """fr.ndindex(2,3) yields the same index tuples numpy does."""
    assert list(fr.ndindex(2, 3)) == list(np.ndindex(2, 3))


def test_ndenumerate_over_ferray_array():
    """fr.ndenumerate iterates a ferray array identically to numpy."""
    a = fr.arange(6).reshape(2, 3)
    got = list(fr.ndenumerate(a))
    expected = list(np.ndenumerate(np.arange(6).reshape(2, 3)))
    assert [idx for idx, _ in got] == [idx for idx, _ in expected]
    assert [int(v) for _, v in got] == [int(v) for _, v in expected]


def test_broadcast_object():
    """fr.broadcast over ferray arrays gives numpy's broadcast shape/nd."""
    b = fr.broadcast(fr.zeros((2, 1)), fr.zeros((1, 3)))
    ref = np.broadcast(np.zeros((2, 1)), np.zeros((1, 3)))
    assert b.shape == ref.shape == (2, 3)
    assert b.nd == ref.nd


def test_nditer_iteration():
    """fr.nditer iterates a ferray array element-for-element like numpy."""
    a = fr.arange(4).reshape(2, 2)
    assert [int(x) for x in fr.nditer(a)] == [
        int(x) for x in np.nditer(np.arange(4).reshape(2, 2))
    ]


def test_flatiter_is_array_flat_type():
    """fr.flatiter is the type of a ferray array's .flat iterator."""
    a = fr.arange(3)
    assert fr.flatiter is np.flatiter
    assert isinstance(a.flat, fr.flatiter)


def test_poly1d_class():
    """fr.poly1d is numpy's poly1d class and evaluates highest-degree-first
    coefficient arrays identically (oracle: numpy poly1d)."""
    assert fr.poly1d is np.poly1d
    p = fr.poly1d([1, 2, 3])      # x^2 + 2x + 3
    ref = np.poly1d([1, 2, 3])
    assert p(2) == ref(2) == 1 * 4 + 2 * 2 + 3
    assert list(p.coeffs) == list(ref.coeffs)
    # derivative parity
    assert list(p.deriv().coeffs) == list(ref.deriv().coeffs)


def test_total_exposed_count():
    """Guard: the full set of newly-exposed drop-in type/class names is
    present (regression tripwire for accidental removal)."""
    missing = [n for n in ALL_TYPE_NAMES if not hasattr(fr, n)]
    assert not missing, f"missing exposed names: {missing}"
    assert len(set(ALL_TYPE_NAMES)) == 61
