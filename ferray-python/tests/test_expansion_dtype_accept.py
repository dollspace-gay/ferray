"""Expansion parity: creation functions must accept numpy dtype OBJECTS.

`numpy` accepts three equivalent forms in any `dtype=` argument — a string
(`"float64"`), a numpy scalar *type* object (`np.float64`, which ferray now
re-exposes as `fr.float64`), and a `numpy.dtype` instance
(`np.dtype("float64")`) — because every creation front-end funnels the
argument through `numpy.dtype(obj)` (numpy/_core/numeric.py `def full` ->
`empty(shape, dtype, ...)`; the C `PyArray_DescrConverter`). A drop-in
replacement that exposes `fr.float64` MUST accept it in `dtype=`.

Also covers the complex creation kernel: `fr.zeros/ones/empty/full` with a
`complex64`/`complex128` dtype, and `fr.full((2,), 1+2j)` defaulting to
complex.

Every expected value is taken from a LIVE numpy call (R-CHAR-3): the test
asserts ferray's result dtype/value equals numpy's for the identical call.
"""

import numpy as np
import pytest

import ferray as fr


# The three equivalent dtype spellings numpy accepts, paired with the
# canonical name each must normalize to. The "type object" column is the
# np.<scalar> type (which ferray re-exposes), exercised both via numpy and via
# the ferray-exposed attribute.
DTYPE_CASES = [
    (np.float64, np.dtype("float64"), "float64"),
    (np.float32, np.dtype("float32"), "float32"),
    (np.int64, np.dtype("int64"), "int64"),
    (np.int32, np.dtype("int32"), "int32"),
    (np.int16, np.dtype("int16"), "int16"),
    (np.int8, np.dtype("int8"), "int8"),
    (np.uint8, np.dtype("uint8"), "uint8"),
    (np.bool_, np.dtype("bool"), "bool"),
]


def _spellings(type_obj, dtype_inst, name):
    """All three numpy-accepted forms for one dtype."""
    return [type_obj, dtype_inst, name]


@pytest.mark.parametrize("type_obj,dtype_inst,name", DTYPE_CASES)
def test_zeros_accepts_all_dtype_spellings(type_obj, dtype_inst, name):
    for spelling in _spellings(type_obj, dtype_inst, name):
        got = fr.zeros(4, dtype=spelling)
        ref = np.zeros(4, dtype=spelling)
        assert got.dtype == ref.dtype, (spelling, got.dtype, ref.dtype)
        assert got.dtype.name == name
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("type_obj,dtype_inst,name", DTYPE_CASES)
def test_ones_accepts_all_dtype_spellings(type_obj, dtype_inst, name):
    for spelling in _spellings(type_obj, dtype_inst, name):
        got = fr.ones(3, dtype=spelling)
        ref = np.ones(3, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("type_obj,dtype_inst,name", DTYPE_CASES)
def test_empty_accepts_all_dtype_spellings(type_obj, dtype_inst, name):
    for spelling in _spellings(type_obj, dtype_inst, name):
        got = fr.empty((2, 2), dtype=spelling)
        ref = np.empty((2, 2), dtype=spelling)
        assert got.dtype == ref.dtype


@pytest.mark.parametrize("type_obj,dtype_inst,name", DTYPE_CASES)
def test_eye_accepts_all_dtype_spellings(type_obj, dtype_inst, name):
    for spelling in _spellings(type_obj, dtype_inst, name):
        got = fr.eye(3, dtype=spelling)
        ref = np.eye(3, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_identity_accepts_dtype_object():
    for spelling in (np.int32, np.dtype("int32"), "int32"):
        got = fr.identity(3, dtype=spelling)
        ref = np.identity(3, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_arange_accepts_dtype_object():
    for spelling in (np.int16, np.dtype("int16"), "int16"):
        got = fr.arange(5, dtype=spelling)
        ref = np.arange(5, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_full_accepts_dtype_object():
    for spelling in (np.float32, np.dtype("float32"), "float32"):
        got = fr.full((2, 2), 7, dtype=spelling)
        ref = np.full((2, 2), 7, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_linspace_accepts_dtype_object():
    for spelling in (np.int64, np.dtype("int64"), "int64"):
        got = fr.linspace(0, 10, 6, dtype=spelling)
        ref = np.linspace(0, 10, 6, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_zeros_like_accepts_dtype_object():
    src = np.arange(6, dtype=np.float64)
    for spelling in (np.int32, np.dtype("int32"), "int32"):
        got = fr.zeros_like(src, dtype=spelling)
        ref = np.zeros_like(src, dtype=spelling)
        assert got.dtype == ref.dtype


def test_ferray_reexposed_type_object_is_numpy_type():
    # `fr.float64` IS `np.float64` (ferray re-exposes the numpy scalar types),
    # so passing it must behave identically to passing `np.float64`.
    assert fr.float64 is np.float64
    assert fr.int8 is np.int8
    got = fr.zeros(3, dtype=fr.float64)
    assert got.dtype == np.zeros(3, dtype=np.float64).dtype


# ---------------------------------------------------------------------------
# Complex creation kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cdt", ["complex64", "complex128"])
def test_zeros_complex_dtype(cdt):
    got = fr.zeros(4, dtype=cdt)
    ref = np.zeros(4, dtype=cdt)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("cdt", ["complex64", "complex128"])
def test_ones_complex_dtype(cdt):
    got = fr.ones(3, dtype=cdt)
    ref = np.ones(3, dtype=cdt)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("cdt", ["complex64", "complex128"])
def test_empty_complex_dtype(cdt):
    got = fr.empty((2, 2), dtype=cdt)
    ref = np.empty((2, 2), dtype=cdt)
    assert got.dtype == ref.dtype


def test_zeros_complex_via_type_object_and_dtype_instance():
    for spelling in (np.complex128, np.dtype("complex128"), "complex128"):
        got = fr.zeros(3, dtype=spelling)
        ref = np.zeros(3, dtype=spelling)
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got, ref)


def test_full_complex_fill_default_dtype():
    # numpy defaults dtype to np.array(fill_value).dtype, so a complex fill
    # yields complex128 (numpy/_core/numeric.py:382-384).
    got = fr.full((2,), 1 + 2j)
    ref = np.full((2,), 1 + 2j)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("cdt", ["complex64", "complex128"])
def test_full_complex_explicit_dtype(cdt):
    got = fr.full((3,), 1 + 2j, dtype=cdt)
    ref = np.full((3,), 1 + 2j, dtype=cdt)
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)


def test_full_real_fill_complex_dtype():
    # A real fill with an explicit complex dtype lands at imag==0.
    got = fr.full((2,), 5, dtype="complex128")
    ref = np.full((2,), 5, dtype="complex128")
    assert got.dtype == ref.dtype
    np.testing.assert_array_equal(got, ref)
