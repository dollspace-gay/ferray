"""Divergence #909 regression: numpy.ma reductions return numpy SCALARS.

`MaskedArray.mean`/`std`/`var` (no-axis) return a ``numpy.float64`` scalar
(``.dtype == float64``, ``.shape == ()``) and ``count`` returns a
``numpy.int64`` scalar -- NOT a bare Python ``float``/``int``. Complex ``mean``
returns ``numpy.complex128``; complex ``var``/``std`` return ``numpy.float64``
(real). The oracle is live numpy.ma; every expected type/value is taken from a
numpy call in the same test (R-CHAR-3, no literal-copied expectations).
"""

import numpy as np
import pytest

import ferray as fr


def _assert_scalar_match(label, fv, nv):
    """ferray result must match numpy: same python type, dtype, shape, value."""
    assert type(fv) is type(nv), (
        f"{label}: type {type(fv).__name__} != numpy {type(nv).__name__}"
    )
    assert fv.dtype == nv.dtype, f"{label}: dtype {fv.dtype} != numpy {nv.dtype}"
    assert fv.shape == nv.shape == (), f"{label}: shape {fv.shape} != ()"
    assert np.allclose(np.asarray(fv), np.asarray(nv)), (
        f"{label}: value {fv} != numpy {nv}"
    )


def _pair(data, mask, dtype=None):
    arr = np.asarray(data, dtype=dtype) if dtype is not None else data
    return (
        fr.ma.array(arr, mask=mask),
        np.ma.array(np.asarray(arr), mask=mask),
    )


def test_mean_returns_numpy_float64_scalar():
    fa, na = _pair([1.0, 2, 3, 4], [0, 1, 0, 0])
    _assert_scalar_match("mean", fa.mean(), na.mean())
    assert isinstance(fa.mean(), np.float64)


def test_std_returns_numpy_float64_scalar():
    fa, na = _pair([1.0, 2, 3, 4], [0, 1, 0, 0])
    _assert_scalar_match("std", fa.std(), na.std())
    _assert_scalar_match("std_ddof1", fa.std(ddof=1), na.std(ddof=1))


def test_var_returns_numpy_float64_scalar():
    fa, na = _pair([1.0, 2, 3, 4], [0, 1, 0, 0])
    _assert_scalar_match("var", fa.var(), na.var())
    _assert_scalar_match("var_ddof1", fa.var(ddof=1), na.var(ddof=1))


def test_count_returns_numpy_int64_scalar():
    fa, na = _pair([1.0, 2, 3, 4], [0, 1, 0, 0])
    _assert_scalar_match("count", fa.count(), na.count())
    assert isinstance(fa.count(), np.int64)


def test_float32_mean_promotes_to_float64():
    # numpy.ma promotes float32 input to float64 for mean/std/var.
    fa, na = _pair([1, 2, 3, 4], [0, 1, 0, 0], dtype=np.float32)
    _assert_scalar_match("f32_mean", fa.mean(), na.mean())
    _assert_scalar_match("f32_std", fa.std(), na.std())
    _assert_scalar_match("f32_var", fa.var(), na.var())


def test_integer_mean_promotes_to_float64():
    fa, na = _pair([1, 2, 3, 4], [0, 1, 0, 0], dtype=np.int64)
    _assert_scalar_match("int_mean", fa.mean(), na.mean())
    _assert_scalar_match("int_count", fa.count(), na.count())
    assert isinstance(fa.count(), np.int64)


def test_complex_mean_returns_complex128_scalar():
    fa, na = _pair([1 + 2j, 3 + 4j, 5 + 6j], [0, 1, 0])
    _assert_scalar_match("cmean", fa.mean(), na.mean())
    assert isinstance(fa.mean(), np.complex128)


def test_complex_var_std_return_float64_scalar():
    fa, na = _pair([1 + 2j, 3 + 4j, 5 + 6j], [0, 1, 0])
    _assert_scalar_match("cvar", fa.var(), na.var())
    _assert_scalar_match("cstd", fa.std(), na.std())


@pytest.mark.parametrize("name", ["mean", "std", "var", "count"])
def test_module_forms_inherit_scalar_type(name):
    fa, na = _pair([1.0, 2, 3, 4], [0, 1, 0, 0])
    fv = getattr(fr.ma, name)(fa)
    nv = getattr(np.ma, name)(na)
    _assert_scalar_match(f"mod.{name}", fv, nv)
