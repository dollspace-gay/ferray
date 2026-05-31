"""Phase-1 parity tests for the ferray creation surface.

Each test asserts numeric / shape / dtype equivalence against the
canonical NumPy implementation.
"""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Shape + dtype creation
# ---------------------------------------------------------------------------


def test_empty_returns_zeroed_for_now():
    a = ferray.empty((3, 4))
    assert a.shape == (3, 4)
    assert a.dtype == np.float64
    # We currently zero — the public docstring documents this.
    assert np.all(a == 0)


def test_full_default_dtype_int_for_int_value():
    a = ferray.full((2, 3), 7)
    assert a.dtype == np.int64
    assert np.array_equal(a, np.full((2, 3), 7, dtype=np.int64))


def test_full_default_dtype_float_for_float_value():
    a = ferray.full((2, 3), 1.5)
    assert a.dtype == np.float64
    assert np.array_equal(a, np.full((2, 3), 1.5))


def test_full_explicit_dtype():
    a = ferray.full((4,), -1, dtype="int32")
    assert a.dtype == np.int32
    assert np.array_equal(a, np.full((4,), -1, dtype=np.int32))


def test_identity_matches_numpy():
    for n in (1, 3, 5):
        a = ferray.identity(n)
        assert np.array_equal(a, np.identity(n))


def test_eye_default_args():
    a = ferray.eye(4)
    assert np.array_equal(a, np.eye(4))


def test_eye_with_offset():
    for k in (-2, -1, 0, 1, 2):
        a = ferray.eye(4, m=5, k=k)
        assert np.array_equal(a, np.eye(4, M=5, k=k))


def test_eye_int_dtype():
    a = ferray.eye(3, dtype="int32")
    assert a.dtype == np.int32
    assert np.array_equal(a, np.eye(3, dtype=np.int32))


def test_tri_default_dtype_and_shape():
    a = ferray.tri(4)
    assert np.array_equal(a, np.tri(4))


def test_tri_offset_and_rect():
    a = ferray.tri(4, m=5, k=1)
    assert np.array_equal(a, np.tri(4, M=5, k=1))


# ---------------------------------------------------------------------------
# Range
# ---------------------------------------------------------------------------


def test_linspace_default():
    a = ferray.linspace(0.0, 1.0, num=11)
    np.testing.assert_allclose(a, np.linspace(0.0, 1.0, num=11))


def test_linspace_no_endpoint():
    a = ferray.linspace(0.0, 1.0, num=4, endpoint=False)
    np.testing.assert_allclose(a, np.linspace(0.0, 1.0, num=4, endpoint=False))


def test_linspace_float32():
    a = ferray.linspace(0.0, 1.0, num=11, dtype="float32")
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, np.linspace(0.0, 1.0, num=11, dtype=np.float32))


def test_logspace_default():
    a = ferray.logspace(0.0, 3.0, num=4)
    np.testing.assert_allclose(a, np.logspace(0.0, 3.0, num=4))


def test_logspace_custom_base():
    a = ferray.logspace(0.0, 4.0, num=5, base=2.0)
    np.testing.assert_allclose(a, np.logspace(0.0, 4.0, num=5, base=2.0))


def test_geomspace_matches_numpy():
    a = ferray.geomspace(1.0, 1000.0, num=4)
    np.testing.assert_allclose(a, np.geomspace(1.0, 1000.0, num=4))


# ---------------------------------------------------------------------------
# Array → array
# ---------------------------------------------------------------------------


def test_zeros_like_inherits_dtype_and_shape():
    src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    a = ferray.zeros_like(src)
    assert a.shape == src.shape
    assert a.dtype == src.dtype
    assert np.all(a == 0)


def test_zeros_like_dtype_override():
    src = np.array([1, 2, 3], dtype=np.int64)
    a = ferray.zeros_like(src, dtype="float32")
    assert a.dtype == np.float32


def test_ones_like_inherits():
    src = np.zeros((2, 3), dtype=np.uint16)
    a = ferray.ones_like(src)
    assert a.dtype == np.uint16
    assert np.all(a == 1)


def test_full_like_basic():
    src = np.zeros((4,), dtype=np.float32)
    a = ferray.full_like(src, 3.5)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a, np.full((4,), 3.5, dtype=np.float32))


def test_copy_round_trips():
    src = np.array([[1.5, -2.0], [3.25, 0.0]])
    a = ferray.copy(src)
    assert np.array_equal(a, src)
    assert a is not src  # different buffer


def test_ascontiguousarray_returns_c_contig():
    src = np.asfortranarray(np.arange(12).reshape(3, 4))
    assert not src.flags["C_CONTIGUOUS"]
    a = ferray.ascontiguousarray(src)
    assert a.flags["C_CONTIGUOUS"]
    assert np.array_equal(a, src)


def test_asfortranarray_returns_f_contig():
    # #698 closed: ferray-numpy-interop's IntoNumPy detects F-contiguous
    # inputs and routes them through reshape_with_order=F so the
    # F_CONTIGUOUS flag is preserved end-to-end. ferray-core's
    # asfortranarray was also fixed to actually produce F-layout
    # (previously it just cloned).
    src = np.ascontiguousarray(np.arange(12).reshape(3, 4))
    a = ferray.asfortranarray(src)
    assert a.flags["F_CONTIGUOUS"]
    assert np.array_equal(a, src)


def test_asanyarray_passes_through_ndarray():
    src = np.array([1, 2, 3])
    a = ferray.asanyarray(src)
    assert np.array_equal(a, src)


def test_array_from_python_list():
    a = ferray.array([1, 2, 3, 4])
    assert isinstance(a, np.ndarray)
    assert np.array_equal(a, np.array([1, 2, 3, 4]))


def test_array_with_dtype_cast():
    a = ferray.array([1.5, 2.5, 3.5], dtype="int32")
    assert a.dtype == np.int32
    assert np.array_equal(a, np.array([1, 2, 3], dtype=np.int32))


def test_asarray_alias():
    a = ferray.asarray([[1, 2], [3, 4]])
    assert np.array_equal(a, np.asarray([[1, 2], [3, 4]]))


@pytest.mark.parametrize("dtype", ["bool", "int8", "int16", "uint16", "float32", "float64"])
def test_zeros_like_round_trip_all_dtypes(dtype):
    src = np.zeros((2, 2), dtype=dtype)
    a = ferray.zeros_like(src)
    assert a.dtype == np.dtype(dtype)


# ---------------------------------------------------------------------------
# meshgrid dtype preservation (#978)
# ---------------------------------------------------------------------------
# numpy.meshgrid preserves EACH input's own dtype (no promotion) and supports
# sparse=/copy=. The prior binding coerced every input to float64 — widening
# int/float32, dropping the imaginary part of complex inputs, and CRASHING on
# string input. These pin per-input dtype + sparseness parity (live numpy 2.4.5).


def test_meshgrid_preserves_int_dtype():
    fg = ferray.meshgrid([1, 2], [3, 4])
    ng = np.meshgrid([1, 2], [3, 4])
    for f, n in zip(fg, ng):
        assert np.asarray(f).dtype == n.dtype == np.int64
        assert np.array_equal(f, n)


def test_meshgrid_float64_native_path_unchanged():
    fg = ferray.meshgrid([1.0, 2.0], [3.0, 4.0])
    ng = np.meshgrid([1.0, 2.0], [3.0, 4.0])
    for f, n in zip(fg, ng):
        assert np.asarray(f).dtype == n.dtype == np.float64
        assert np.array_equal(f, n)


def test_meshgrid_mixed_dtype_per_input():
    fg = ferray.meshgrid([1, 2], [3.0, 4.0])
    ng = np.meshgrid([1, 2], [3.0, 4.0])
    assert np.asarray(fg[0]).dtype == ng[0].dtype == np.int64
    assert np.asarray(fg[1]).dtype == ng[1].dtype == np.float64


def test_meshgrid_float32_preserved():
    x = np.array([1, 2], np.float32)
    fg = ferray.meshgrid(x, [3.0, 4.0])
    assert np.asarray(fg[0]).dtype == np.float32


def test_meshgrid_string_inputs():
    fg = ferray.meshgrid(["a", "b"], ["c", "d"])
    ng = np.meshgrid(["a", "b"], ["c", "d"])
    for f, n in zip(fg, ng):
        assert np.asarray(f).dtype == n.dtype
        assert np.array_equal(f, n)


def test_meshgrid_complex_preserved():
    fg = ferray.meshgrid([1 + 2j, 3], [4, 5])
    ng = np.meshgrid([1 + 2j, 3], [4, 5])
    assert np.asarray(fg[0]).dtype == ng[0].dtype == np.complex128
    assert np.array_equal(fg[0], ng[0])


def test_meshgrid_indexing_ij():
    fg = ferray.meshgrid([1, 2, 3], [4, 5], indexing="ij")
    ng = np.meshgrid([1, 2, 3], [4, 5], indexing="ij")
    for f, n in zip(fg, ng):
        assert np.array_equal(f, n)
        assert np.asarray(f).shape == n.shape


def test_meshgrid_sparse():
    fg = ferray.meshgrid([1, 2, 3], [4, 5], sparse=True)
    ng = np.meshgrid([1, 2, 3], [4, 5], sparse=True)
    for f, n in zip(fg, ng):
        assert np.asarray(f).shape == n.shape
        assert np.array_equal(f, n)
