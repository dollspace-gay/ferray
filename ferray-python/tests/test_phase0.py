"""Phase-0 smoke tests for the ferray Python bindings.

These verify the marshalling pipeline end-to-end: shape coercion, dtype
dispatch, and round-tripping into a real `numpy.ndarray` that NumPy can
operate on. They are intentionally lightweight — the proper API
parity tests will follow phase-by-phase as functions are bound.
"""

import numpy as np
import pytest

import ferray


def test_version_present():
    assert isinstance(ferray.__version__, str)
    assert ferray.__version__.count(".") >= 2


def test_zeros_default_dtype_is_float64():
    a = ferray.zeros((3, 4))
    assert isinstance(a, np.ndarray)
    assert a.shape == (3, 4)
    assert a.dtype == np.float64
    assert np.array_equal(a, np.zeros((3, 4)))


def test_zeros_scalar_shape():
    a = ferray.zeros(5)
    assert a.shape == (5,)
    assert np.array_equal(a, np.zeros(5))


@pytest.mark.parametrize(
    "dtype",
    [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float32", "float64",
    ],
)
def test_zeros_all_supported_dtypes(dtype):
    a = ferray.zeros((2, 3), dtype=dtype)
    assert a.shape == (2, 3)
    assert a.dtype == np.dtype(dtype)
    assert np.array_equal(a, np.zeros((2, 3), dtype=dtype))


def test_zeros_unknown_dtype_raises_typeerror():
    # ferray now routes dtype parsing through `numpy.dtype(obj)` (so it can
    # also accept type objects / dtype instances), which means an unparseable
    # dtype raises numpy's OWN TypeError verbatim — "data type 'banana' not
    # understood" — exactly matching `np.zeros((3,), dtype='banana')`
    # (R-DEV-2). The exception TYPE (TypeError) is unchanged.
    with pytest.raises(TypeError, match="not understood"):
        ferray.zeros((3,), dtype="banana")
    # Same message numpy itself produces for the identical call.
    with pytest.raises(TypeError) as fr_exc:
        ferray.zeros((3,), dtype="banana")
    with pytest.raises(TypeError) as np_exc:
        np.zeros((3,), dtype="banana")
    assert str(fr_exc.value) == str(np_exc.value)


def test_ones_default_dtype():
    a = ferray.ones((4,))
    assert a.dtype == np.float64
    assert np.array_equal(a, np.ones(4))


def test_ones_int32():
    a = ferray.ones((2, 2), dtype="int32")
    assert a.dtype == np.int32
    assert np.array_equal(a, np.ones((2, 2), dtype="int32"))


def test_arange_one_arg_is_stop():
    a = ferray.arange(5)
    assert a.dtype == np.int64
    assert np.array_equal(a, np.arange(5, dtype=np.int64))


def test_arange_two_args_start_stop():
    a = ferray.arange(2, 7)
    assert np.array_equal(a, np.arange(2, 7, dtype=np.int64))


def test_arange_three_args_start_stop_step():
    a = ferray.arange(0, 10, 2)
    assert np.array_equal(a, np.arange(0, 10, 2, dtype=np.int64))


def test_arange_float_promotes_to_float64():
    a = ferray.arange(0, 1, 0.25)
    assert a.dtype == np.float64
    np.testing.assert_allclose(a, np.arange(0, 1, 0.25))


def test_arange_explicit_float32_dtype():
    a = ferray.arange(0, 5, dtype="float32")
    assert a.dtype == np.float32


def test_dtype_namespace_has_known_names():
    assert ferray.dtype.float64 == "float64"
    assert ferray.dtype.int32 == "int32"
    assert ferray.dtype.bool == "bool"


def test_dtype_strings_round_trip_through_creation():
    a = ferray.zeros((2,), dtype=ferray.dtype.uint16)
    assert a.dtype == np.uint16


def test_returned_array_is_writable_and_mutable():
    a = ferray.zeros((3,), dtype="float64")
    a[1] = 42.0
    assert a[1] == 42.0


def test_returned_array_supports_numpy_ops():
    a = ferray.ones((4,), dtype="float64")
    b = ferray.arange(4, dtype="float64")
    c = a + b
    assert np.array_equal(c, np.array([1.0, 2.0, 3.0, 4.0]))
