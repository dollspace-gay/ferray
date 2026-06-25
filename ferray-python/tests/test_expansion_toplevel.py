"""Expansion batch: remaining top-level numpy functions (refs #818).

Each test asserts a ferray top-level function against the live numpy oracle
(numpy 2.4.x installed) — pure-Python composables over already-bound ferray
primitives, plus dtype/format/meta helpers. Config/meta functions that have
no array-math behavior are asserted to exist and behave numpy-compatibly.
"""

import numpy as np
import pytest

import ferray as fr


def _eq(got, want):
    g = np.asarray(got)
    w = np.asarray(want)
    assert g.shape == w.shape
    if g.dtype.kind in "fc":
        assert np.allclose(g, w, equal_nan=True)
    else:
        assert np.array_equal(g, w)


# --- existence sweep ---------------------------------------------------------

EXPANSION_NAMES = [
    "apply_along_axis", "apply_over_axes", "piecewise", "select", "unstack",
    "require", "vectorize", "frompyfunc", "base_repr", "binary_repr",
    "mintypecode", "typename", "finfo", "iinfo", "packbits", "unpackbits",
    "format_float_positional", "format_float_scientific",
    "array2string", "array_repr", "array_str",
    "get_printoptions", "set_printoptions", "printoptions", "errstate",
    "geterr", "seterr", "geterrcall", "seterrcall", "getbufsize", "setbufsize",
    "get_include", "show_config", "show_runtime", "info", "test", "einsum_path",
]


@pytest.mark.parametrize("name", EXPANSION_NAMES)
def test_name_exists(name):
    assert hasattr(fr, name), f"ferray.{name} is missing"
    assert name in fr.__all__


# --- functional composables --------------------------------------------------

def test_apply_along_axis_scalar_func():
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    f = lambda a: (a[0] + a[-1]) * 0.5  # noqa: E731
    _eq(fr.apply_along_axis(f, 0, b), np.apply_along_axis(f, 0, b))
    _eq(fr.apply_along_axis(f, 1, b), np.apply_along_axis(f, 1, b))


def test_apply_along_axis_1d_return():
    b = np.array([[8, 1, 7], [4, 3, 9], [5, 2, 6]])
    _eq(fr.apply_along_axis(sorted, 1, b), np.apply_along_axis(sorted, 1, b))


def test_apply_along_axis_higher_dim_return():
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    _eq(fr.apply_along_axis(np.diag, -1, b), np.apply_along_axis(np.diag, -1, b))


def test_apply_along_axis_3d():
    a = np.arange(24).reshape(2, 3, 4)
    f = lambda v: v.sum()  # noqa: E731
    for ax in (0, 1, 2, -1):
        _eq(fr.apply_along_axis(f, ax, a), np.apply_along_axis(f, ax, a))


def test_apply_over_axes():
    a = np.arange(24).reshape(2, 3, 4)
    _eq(fr.apply_over_axes(np.sum, a, [0, 2]),
        np.apply_over_axes(np.sum, a, [0, 2]))
    _eq(fr.apply_over_axes(np.sum, a, 1), np.apply_over_axes(np.sum, a, 1))


def test_piecewise():
    x = np.linspace(-2.5, 2.5, 6)
    _eq(fr.piecewise(x, [x < 0, x >= 0], [-1, 1]),
        np.piecewise(x, [x < 0, x >= 0], [-1, 1]))
    _eq(fr.piecewise(x, [x < 0], [lambda v: -v, lambda v: v]),
        np.piecewise(x, [x < 0], [lambda v: -v, lambda v: v]))
    # implied "otherwise" function (len(condlist) == len(funclist) - 1)
    _eq(fr.piecewise(x, [x < -1, x > 1], [1, 2, 99]),
        np.piecewise(x, [x < -1, x > 1], [1, 2, 99]))


def test_select():
    cond = [np.array([True, False, True]), np.array([False, True, False])]
    choice = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    _eq(fr.select(cond, choice, default=9), np.select(cond, choice, default=9))


def test_unstack():
    b = np.arange(12).reshape(3, 4)
    fparts = fr.unstack(b, axis=0)
    nparts = np.unstack(b, axis=0)
    assert len(fparts) == len(nparts)
    for fp, npp in zip(fparts, nparts):
        _eq(fp, npp)
    _eq(np.stack(fr.unstack(b, axis=1)), np.stack(np.unstack(b, axis=1)))


def test_require():
    a = [1, 2, 3]
    fa = fr.require(a, dtype=np.float64, requirements=["C"])
    na = np.require(a, dtype=np.float64, requirements=["C"])
    _eq(fa, na)
    assert fa.flags["C_CONTIGUOUS"]


def test_vectorize():
    vf = fr.vectorize(lambda a, b: a + b)
    _eq(vf([1, 2, 3], [4, 5, 6]), np.array([5, 7, 9]))
    nvf = np.vectorize(lambda a, b: a + b)
    _eq(vf([1, 2, 3], [4, 5, 6]), nvf([1, 2, 3], [4, 5, 6]))


def test_frompyfunc():
    pf = fr.frompyfunc(lambda a: a * 2, 1, 1)
    npf = np.frompyfunc(lambda a: a * 2, 1, 1)
    assert list(pf([1, 2, 3])) == list(npf([1, 2, 3]))


# --- integer string representations ------------------------------------------

def test_base_repr():
    assert fr.base_repr(255, 16) == np.base_repr(255, 16)
    assert fr.base_repr(5, 2, padding=3) == np.base_repr(5, 2, padding=3)
    assert fr.base_repr(-12, 5) == np.base_repr(-12, 5)
    assert fr.base_repr(0) == np.base_repr(0)


def test_base_repr_errors():
    with pytest.raises(ValueError):
        fr.base_repr(1, 37)
    with pytest.raises(ValueError):
        fr.base_repr(1, 1)


def test_binary_repr():
    assert fr.binary_repr(5) == np.binary_repr(5)
    assert fr.binary_repr(5, width=8) == np.binary_repr(5, width=8)
    assert fr.binary_repr(-3, width=8) == np.binary_repr(-3, width=8)


# --- dtype / typecode helpers ------------------------------------------------

def test_mintypecode():
    assert fr.mintypecode("fd") == np.mintypecode("fd")
    assert fr.mintypecode("GDF") == np.mintypecode("GDF")


def test_typename():
    for c in ("d", "f", "i", "?"):
        assert fr.typename(c) == np.typename(c)


def test_finfo():
    assert fr.finfo(np.float64).eps == np.finfo(np.float64).eps
    assert fr.finfo(np.float32).max == np.finfo(np.float32).max


def test_iinfo():
    assert fr.iinfo(np.int32).max == np.iinfo(np.int32).max
    assert fr.iinfo(np.uint8).min == np.iinfo(np.uint8).min


# --- bit packing -------------------------------------------------------------

def test_packbits():
    a = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    _eq(fr.packbits(a), np.packbits(a))
    m = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    _eq(fr.packbits(m, axis=1), np.packbits(m, axis=1))
    _eq(fr.packbits(a, bitorder="little"), np.packbits(a, bitorder="little"))


def test_unpackbits():
    a = np.array([7, 128], dtype=np.uint8)
    _eq(fr.unpackbits(a), np.unpackbits(a))
    _eq(fr.unpackbits(a, count=4), np.unpackbits(a, count=4))
    _eq(fr.unpackbits(a, bitorder="little"),
        np.unpackbits(a, bitorder="little"))


# --- float formatting --------------------------------------------------------

def test_format_float_positional():
    for v in (1.5, 0.0001, 1234.0, np.pi):
        assert (fr.format_float_positional(np.float64(v))
                == np.format_float_positional(np.float64(v)))
    assert (fr.format_float_positional(np.float64(np.pi), precision=4)
            == np.format_float_positional(np.float64(np.pi), precision=4))


def test_format_float_scientific():
    for v in (1234.5, 0.0001, np.pi):
        assert (fr.format_float_scientific(np.float64(v))
                == np.format_float_scientific(np.float64(v)))


# --- array string representation ---------------------------------------------

def test_array2string():
    b = np.arange(6).reshape(2, 3)
    assert fr.array2string(b) == np.array2string(b)
    assert (fr.array2string(b, separator=", ")
            == np.array2string(b, separator=", "))


def test_array_repr_str():
    b = np.arange(6).reshape(2, 3)
    assert fr.array_repr(b) == np.array_repr(b)
    assert fr.array_str(b) == np.array_str(b)


# --- print options / fp error state ------------------------------------------

def test_get_set_printoptions():
    opts = fr.get_printoptions()
    assert "precision" in opts and "threshold" in opts
    old = opts["precision"]
    try:
        fr.set_printoptions(precision=2)
        assert fr.get_printoptions()["precision"] == 2
    finally:
        fr.set_printoptions(precision=old)


def test_printoptions_context():
    with fr.printoptions(precision=3):
        assert fr.get_printoptions()["precision"] == 3


def test_errstate_and_seterr():
    assert fr.errstate is np.errstate
    old = fr.geterr()
    assert isinstance(old, dict)
    try:
        with fr.errstate(divide="ignore"):
            np.float64(1.0) / np.float64(0.0)
        fr.seterr(over="warn")
        assert fr.geterr()["over"] == "warn"
    finally:
        fr.seterr(**old)


def test_errcall():
    assert fr.geterrcall() == np.geterrcall()
    old = fr.geterrcall()
    try:
        fr.seterrcall(None)
        assert fr.geterrcall() is None
    finally:
        fr.seterrcall(old)


def test_bufsize():
    assert isinstance(fr.getbufsize(), int)
    old = fr.getbufsize()
    try:
        fr.setbufsize(8192)
        assert fr.getbufsize() == 8192
    finally:
        fr.setbufsize(old)


# --- meta / introspection ----------------------------------------------------

def test_get_include():
    assert fr.get_include() == np.get_include()


def test_show_config_returns_dict():
    cfg = fr.show_config(mode="dicts")
    assert isinstance(cfg, dict)


def test_info_runs(capsys):
    fr.info(np.ndarray.sum)
    out = capsys.readouterr().out
    assert isinstance(out, str)


def test_test_matches_numpy_tester():
    assert fr.test is np.test
    assert callable(fr.test)


def test_einsum_path():
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    fp = fr.einsum_path("ij,jk->ik", a, b)
    npth = np.einsum_path("ij,jk->ik", a, b)
    assert fp[0] == npth[0]
