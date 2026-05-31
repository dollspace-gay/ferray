"""#900 — fr.ma.MaskedArray.dtype returns a numpy.dtype OBJECT (not a str).

numpy.ma.array(...).dtype is a `numpy.dtype` instance, so `a.dtype == np.float64`
is True and `a.dtype.kind`/`.itemsize`/`.name`/`.char` are queryable. Prior to
the fix ferray's `fr.ma` getter returned the dtype *name* string, so
`a.dtype == np.float64` was False. Every expectation here comes from a live
numpy.ma oracle call (R-CHAR-3), never literal-copied from ferray.
"""

import numpy as np
import pytest

import ferray as fr

# (constructor data, numpy scalar type) across the real + complex dtypes.
_REAL = [
    ("bool", [True, False, True], np.bool_),
    ("int8", [1, 2, 3], np.int8),
    ("int16", [1, 2, 3], np.int16),
    ("int32", [1, 2, 3], np.int32),
    ("int64", [1, 2, 3], np.int64),
    ("uint8", [1, 2, 3], np.uint8),
    ("uint16", [1, 2, 3], np.uint16),
    ("uint32", [1, 2, 3], np.uint32),
    ("uint64", [1, 2, 3], np.uint64),
    ("float32", [1.0, 2.0, 3.0], np.float32),
    ("float64", [1.0, 2.0, 3.0], np.float64),
    ("complex64", [1 + 2j, 3 + 4j], np.complex64),
    ("complex128", [1 + 2j, 3 + 4j], np.complex128),
]


@pytest.mark.parametrize("name,data,sctype", _REAL)
def test_dtype_is_numpy_dtype_object(name, data, sctype):
    fa = fr.ma.array(data, dtype=name)
    na = np.ma.array(data, dtype=name)

    # It is a numpy.dtype instance, matching numpy.ma exactly.
    assert isinstance(fa.dtype, np.dtype)
    assert fa.dtype == na.dtype

    # Equality against the scalar type, the dtype object, and the name string
    # all behave like numpy's dtype.__eq__.
    assert (fa.dtype == sctype) == (na.dtype == sctype) is True
    assert fa.dtype == np.dtype(name)
    assert fa.dtype == name  # numpy.dtype == 'float64' is True

    # Introspection attributes match numpy's dtype object.
    assert fa.dtype.kind == na.dtype.kind
    assert fa.dtype.itemsize == na.dtype.itemsize
    assert fa.dtype.name == na.dtype.name
    assert fa.dtype.char == na.dtype.char

    # str()/repr() match numpy's dtype repr.
    assert str(fa.dtype) == str(na.dtype)
    assert repr(fa.dtype) == repr(na.dtype)


def test_default_float_dtype_eq_np_float64():
    # The headline divergence: a.dtype == np.float64 must be True.
    a = fr.ma.array([1.0, 2.0, 3.0])
    n = np.ma.array([1.0, 2.0, 3.0])
    assert a.dtype == np.float64
    assert a.dtype == n.dtype
    assert isinstance(a.dtype, np.dtype)


def test_default_int_dtype_eq_np_int64():
    a = fr.ma.array([1, 2, 3])
    n = np.ma.array([1, 2, 3])
    assert a.dtype == np.int64
    assert a.dtype == n.dtype
    # A cross-dtype compare is False, like numpy.
    assert (a.dtype == np.dtype("int32")) == (n.dtype == np.dtype("int32"))


def test_view_dtype_object():
    a = fr.ma.array([1.0, 2.0, 3.0, 4.0])
    n = np.ma.array([1.0, 2.0, 3.0, 4.0])
    av, nv = a[1:3], n[1:3]
    assert isinstance(av.dtype, np.dtype)
    assert av.dtype == nv.dtype
    assert av.dtype == np.float64


def test_astype_dtype_object():
    a = fr.ma.array([1, 2, 3]).astype("float32")
    n = np.ma.array([1, 2, 3]).astype("float32")
    assert isinstance(a.dtype, np.dtype)
    assert a.dtype == n.dtype
    assert a.dtype == np.float32
