"""#975 — top-level reduction kwargs (where=/dtype=/keepdims=/initial=).

Every expected value is derived LIVE from numpy (R-CHAR-3): the test calls the
numpy reduction with the SAME kwargs and asserts ferray matches value + dtype +
shape. Also pins that the no-kwarg fast path is byte-for-byte unchanged and that
the #858 int / #925 complex / #952 f16 dtype contract survives a kwarg.
"""

import numpy as np
import ferray as fr
import pytest


def _match(got, exp):
    assert got.dtype == exp.dtype, f"dtype {got.dtype} != {exp.dtype}"
    assert got.shape == exp.shape, f"shape {got.shape} != {exp.shape}"
    np.testing.assert_array_equal(np.asarray(got), np.asarray(exp))


def _close(got, exp):
    assert got.dtype == exp.dtype, f"dtype {got.dtype} != {exp.dtype}"
    assert got.shape == exp.shape, f"shape {got.shape} != {exp.shape}"
    np.testing.assert_allclose(np.asarray(got), np.asarray(exp), rtol=1e-6)


# --------------------------------------------------------------------------
# where= on sum / prod / min / max
# --------------------------------------------------------------------------

def test_sum_where():
    a = [1, 2, 3, 4]
    w = [True, False, True, False]
    _match(np.asarray(fr.sum(a, where=w)), np.asarray(np.sum(a, where=w)))


def test_sum_where_initial():
    a = [1, 2, 3, 4]
    w = [True, False, True, False]
    _match(
        np.asarray(fr.sum(a, where=w, initial=10)),
        np.asarray(np.sum(a, where=w, initial=10)),
    )


def test_prod_where_initial():
    a = [1, 2, 3, 4]
    w = [True, False, True, False]
    _match(
        np.asarray(fr.prod(a, where=w, initial=1)),
        np.asarray(np.prod(a, where=w, initial=1)),
    )


def test_min_where_initial():
    a = [3, 1, 2]
    w = [True, False, True]
    _match(
        np.asarray(fr.min(a, where=w, initial=100)),
        np.asarray(np.min(a, where=w, initial=100)),
    )


def test_max_where_initial():
    a = [3, 1, 2]
    w = [True, False, True]
    _match(
        np.asarray(fr.max(a, where=w, initial=-100)),
        np.asarray(np.max(a, where=w, initial=-100)),
    )


# --------------------------------------------------------------------------
# initial= alone
# --------------------------------------------------------------------------

def test_sum_initial():
    a = [1, 2, 3]
    _match(np.asarray(fr.sum(a, initial=10)), np.asarray(np.sum(a, initial=10)))


# --------------------------------------------------------------------------
# dtype= on mean / sum
# --------------------------------------------------------------------------

def test_mean_dtype_float32():
    a = [1, 2, 3, 4]
    _close(
        np.asarray(fr.mean(a, dtype="float32")),
        np.asarray(np.mean(a, dtype="float32")),
    )


def test_sum_dtype_float32():
    a = [1, 2, 3, 4]
    _match(
        np.asarray(fr.sum(a, dtype="float32")),
        np.asarray(np.sum(a, dtype="float32")),
    )


def test_sum_dtype_int64_on_int32():
    a = np.array([1, 2, 3], dtype="int32")
    _match(
        np.asarray(fr.sum(a, dtype="int64")),
        np.asarray(np.sum(a, dtype="int64")),
    )


# --------------------------------------------------------------------------
# keepdims= with axis on std / var / mean / sum
# --------------------------------------------------------------------------

def test_std_axis_keepdims():
    a = [[1.0, 2], [3, 4]]
    _close(
        np.asarray(fr.std(a, axis=1, keepdims=True)),
        np.asarray(np.std(a, axis=1, keepdims=True)),
    )


def test_var_axis_keepdims_ddof():
    a = [[1.0, 2], [3, 4]]
    _close(
        np.asarray(fr.var(a, axis=1, keepdims=True, ddof=1)),
        np.asarray(np.var(a, axis=1, keepdims=True, ddof=1)),
    )


def test_mean_axis_keepdims():
    a = [[1.0, 2], [3, 4]]
    _close(
        np.asarray(fr.mean(a, axis=1, keepdims=True)),
        np.asarray(np.mean(a, axis=1, keepdims=True)),
    )


def test_sum_axis_keepdims():
    a = [[1, 2], [3, 4]]
    _match(
        np.asarray(fr.sum(a, axis=1, keepdims=True)),
        np.asarray(np.sum(a, axis=1, keepdims=True)),
    )


# --------------------------------------------------------------------------
# keepdims= on argmin / argmax
# --------------------------------------------------------------------------

def test_argmax_axis_keepdims():
    a = [[1, 5], [3, 2]]
    _match(
        np.asarray(fr.argmax(a, axis=1, keepdims=True)),
        np.asarray(np.argmax(a, axis=1, keepdims=True)),
    )


def test_argmin_axis_keepdims():
    a = [[1, 5], [3, 2]]
    _match(
        np.asarray(fr.argmin(a, axis=1, keepdims=True)),
        np.asarray(np.argmin(a, axis=1, keepdims=True)),
    )


# --------------------------------------------------------------------------
# dtype/shape preserved through a kwarg: #925 complex, #952 f16, #858 int
# --------------------------------------------------------------------------

def test_complex_sum_keepdims_stays_complex():
    a = np.array([[1 + 2j, 3 + 4j]], dtype="complex128")
    _match(
        np.asarray(fr.sum(a, axis=1, keepdims=True)),
        np.asarray(np.sum(a, axis=1, keepdims=True)),
    )


def test_complex64_sum_keepdims_stays_complex64():
    a = np.array([[1 + 2j, 3 + 4j]], dtype="complex64")
    _match(
        np.asarray(fr.sum(a, axis=1, keepdims=True)),
        np.asarray(np.sum(a, axis=1, keepdims=True)),
    )


def test_float16_mean_keepdims_stays_float16():
    a = np.array([[1, 2, 3, 4]], dtype="float16")
    _close(
        np.asarray(fr.mean(a, axis=1, keepdims=True)),
        np.asarray(np.mean(a, axis=1, keepdims=True)),
    )


def test_int_sum_keepdims_dtype():
    a = np.array([[1, 2], [3, 4]], dtype="int32")
    _match(
        np.asarray(fr.sum(a, axis=1, keepdims=True)),
        np.asarray(np.sum(a, axis=1, keepdims=True)),
    )


# --------------------------------------------------------------------------
# all / any where= + keepdims=
# --------------------------------------------------------------------------

def test_all_where():
    a = [True, False, True]
    w = [True, False, True]
    _match(np.asarray(fr.all(a, where=w)), np.asarray(np.all(a, where=w)))


def test_any_keepdims():
    a = [[False, True], [False, False]]
    _match(
        np.asarray(fr.any(a, axis=1, keepdims=True)),
        np.asarray(np.any(a, axis=1, keepdims=True)),
    )


# --------------------------------------------------------------------------
# no-kwarg fast path UNCHANGED (regression guard for #858/#925/#952)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn,a",
    [
        ("sum", np.array([1, 2, 3], dtype="int32")),
        ("prod", np.array([1, 2, 3], dtype="int64")),
        ("mean", np.array([1.0, 2, 3], dtype="float32")),
        ("std", np.array([1.0, 2, 3], dtype="float64")),
        ("var", np.array([1.0, 2, 3], dtype="float64")),
        ("min", np.array([3, 1, 2], dtype="int16")),
        ("max", np.array([3, 1, 2], dtype="int16")),
        ("sum", np.array([1 + 2j, 3 + 4j], dtype="complex64")),
        ("mean", np.array([1, 2, 3], dtype="float16")),
    ],
)
def test_no_kwarg_fast_path_unchanged(fn, a):
    got = np.asarray(getattr(fr, fn)(a))
    exp = np.asarray(getattr(np, fn)(a))
    assert got.dtype == exp.dtype
    np.testing.assert_allclose(got, exp, rtol=1e-6)
