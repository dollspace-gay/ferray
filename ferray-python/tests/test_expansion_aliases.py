"""Verification artifact for the top-level numpy alias / introspection
expansion batch (refs #818).

Every expected value is produced by a *live* `numpy` call (numpy is the
installed oracle — R-CHAR-3, no literal-copied ferray values). Each test
asserts `ferray.<name>` matches `numpy.<name>` in value AND, where
relevant, dtype.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# numpy-2.0 trig aliases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, arg",
    [
        ("acos", 0.5),
        ("acosh", 2.0),
        ("asin", 0.5),
        ("asinh", 1.0),
        ("atan", 1.0),
        ("atanh", 0.5),
    ],
)
def test_trig_unary_aliases(name, arg):
    fr_fn = getattr(fr, name)
    np_fn = getattr(np, name)
    got = np.asarray(fr_fn(arg))
    want = np.asarray(np_fn(arg))
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_atan2_alias():
    got = np.asarray(fr.atan2(1.0, 1.0))
    want = np.asarray(np.atan2(1.0, 1.0))
    np.testing.assert_allclose(got, want, rtol=1e-12)


# ---------------------------------------------------------------------------
# bitwise aliases
# ---------------------------------------------------------------------------

def test_bitwise_invert():
    x = np.array([1, 2, -3], dtype=np.int8)
    got = fr.bitwise_invert(x)
    want = np.bitwise_invert(x)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == want.dtype


def test_bitwise_left_shift():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    got = fr.bitwise_left_shift(a, b)
    want = np.bitwise_left_shift(a, b)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == want.dtype


def test_bitwise_right_shift():
    a = np.array([8, 16, 32], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    got = fr.bitwise_right_shift(a, b)
    want = np.bitwise_right_shift(a, b)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == want.dtype


# ---------------------------------------------------------------------------
# misc ufunc / manipulation aliases
# ---------------------------------------------------------------------------

def test_concat_alias():
    a = np.array([1, 2])
    b = np.array([3, 4])
    np.testing.assert_array_equal(fr.concat([a, b]), np.concat([a, b]))


def test_permute_dims_alias():
    x = np.arange(6).reshape(2, 3)
    got = fr.permute_dims(x, (1, 0))
    want = np.permute_dims(x, (1, 0))
    np.testing.assert_array_equal(got, want)
    assert got.shape == want.shape


def test_pow_alias():
    a = np.array([2, 3, 4])
    b = np.array([3, 2, 1])
    got = fr.pow(a, b)
    want = np.pow(a, b)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == want.dtype


def test_divmod_int():
    a = np.array([7, 8, 9])
    b = np.array([3, 3, 4])
    got_q, got_r = fr.divmod(a, b)
    want_q, want_r = np.divmod(a, b)
    np.testing.assert_array_equal(got_q, want_q)
    np.testing.assert_array_equal(got_r, want_r)
    assert got_q.dtype == want_q.dtype
    assert got_r.dtype == want_r.dtype


def test_divmod_scalar():
    got_q, got_r = fr.divmod(7, 3)
    want_q, want_r = np.divmod(7, 3)
    assert int(np.asarray(got_q)) == int(want_q)
    assert int(np.asarray(got_r)) == int(want_r)


# ---------------------------------------------------------------------------
# reduction aliases
# ---------------------------------------------------------------------------

def test_amax_amin():
    x = np.array([[1, 5], [3, 2]])
    np.testing.assert_array_equal(fr.amax(x), np.amax(x))
    np.testing.assert_array_equal(fr.amin(x), np.amin(x))
    np.testing.assert_array_equal(fr.amax(x, axis=0), np.amax(x, axis=0))
    np.testing.assert_array_equal(fr.amin(x, axis=1), np.amin(x, axis=1))


def test_cumulative_sum_prod():
    x = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(fr.cumulative_sum(x), np.cumsum(x))
    np.testing.assert_array_equal(fr.cumulative_prod(x), np.cumprod(x))


# ---------------------------------------------------------------------------
# array introspection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obj", [5, 3.14, [[1, 2], [3, 4]], [1, 2, 3], np.arange(6).reshape(2, 3)])
def test_ndim(obj):
    assert fr.ndim(obj) == np.ndim(obj)


@pytest.mark.parametrize("obj", [5, [[1, 2], [3, 4]], [1, 2, 3], np.arange(6).reshape(2, 3)])
def test_shape(obj):
    assert tuple(fr.shape(obj)) == np.shape(obj)


@pytest.mark.parametrize("obj", [5, [[1, 2], [3, 4]], [1, 2, 3]])
def test_size(obj):
    assert fr.size(obj) == np.size(obj)


def test_size_axis():
    x = np.arange(6).reshape(2, 3)
    assert fr.size(x, axis=0) == np.size(x, axis=0)
    assert fr.size(x, axis=1) == np.size(x, axis=1)
    assert fr.size(x, axis=-1) == np.size(x, axis=-1)


@pytest.mark.parametrize(
    "obj", [5, 3.14, "hello", b"x", np.int64(3), [1, 2], np.array([1, 2]), np.array(5)]
)
def test_isscalar(obj):
    assert fr.isscalar(obj) == np.isscalar(obj)


def test_isfortran():
    f = np.asfortranarray(np.arange(6).reshape(2, 3))
    c = np.ascontiguousarray(np.arange(6).reshape(2, 3))
    assert fr.isfortran(f) == np.isfortran(f)
    assert fr.isfortran(c) == np.isfortran(c)


# ---------------------------------------------------------------------------
# dtype introspection
# ---------------------------------------------------------------------------

def test_astype():
    x = np.array([1, 2, 3], dtype=np.int32)
    got = fr.astype(x, np.float64)
    want = np.astype(x, np.float64)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == want.dtype


@pytest.mark.parametrize(
    "frm, to, expect",
    [("int8", "int16", True), ("float64", "int8", False), ("int32", "float64", True)],
)
def test_can_cast(frm, to, expect):
    assert fr.can_cast(frm, to) == np.can_cast(frm, to) == expect


def test_promote_types():
    assert fr.promote_types("int8", "int16") == np.promote_types("int8", "int16")
    assert fr.promote_types("float32", "int32") == np.promote_types("float32", "int32")


def test_result_type():
    assert fr.result_type("int8", "float32") == np.result_type("int8", "float32")
    assert fr.result_type(np.int16, np.uint8) == np.result_type(np.int16, np.uint8)


def test_min_scalar_type():
    assert fr.min_scalar_type(300) == np.min_scalar_type(300)
    assert fr.min_scalar_type(-1) == np.min_scalar_type(-1)


def test_issubdtype():
    assert fr.issubdtype(np.int32, np.integer) == np.issubdtype(np.int32, np.integer)
    assert fr.issubdtype(np.float64, np.integer) == np.issubdtype(np.float64, np.integer)


def test_isdtype():
    assert fr.isdtype(np.dtype("int32"), "integral") == np.isdtype(np.dtype("int32"), "integral")
    assert fr.isdtype(np.dtype("float64"), "integral") == np.isdtype(
        np.dtype("float64"), "integral"
    )


def test_common_type():
    a = np.array([1.0], dtype=np.float32)
    b = np.array([2.0], dtype=np.float64)
    assert fr.common_type(a, b) == np.common_type(a, b)
