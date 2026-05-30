"""numpy.ma expansion parity tests (refs #818).

Each test asserts a ferray.ma function against the live numpy.ma oracle
(numpy 2.4.x installed). Expected values are constructed by calling
numpy.ma directly (R-CHAR-3 — never literal-copied from the ferray side).

Comparison protocol for masked results: a ferray.ma.MaskedArray exposes
`.data` (ndarray) and `.mask` (ndarray-or-nomask). We compare the mask
positions exactly, and the *unmasked* data values against numpy.ma's
unmasked data — numpy leaves arbitrary values under the mask, so masked
slots are excluded from the value comparison.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def mask_of(arr):
    """Full bool mask array for a numpy.ma or ferray.ma masked array."""
    m = arr.mask
    if m is np.ma.nomask or m is None or (isinstance(m, np.bool_) and not m):
        return np.zeros(np.shape(np.asarray(arr.data)), dtype=bool)
    return np.asarray(m, dtype=bool)


def assert_masked_equal(got, expected, *, rtol=1e-12, atol=1e-12):
    """Compare a ferray.ma result against a numpy.ma result."""
    gm = mask_of(got)
    em = np.ma.getmaskarray(expected)
    np.testing.assert_array_equal(gm, em, err_msg="mask mismatch")
    gd = np.asarray(got.data, dtype=float)
    ed = np.ma.getdata(expected).astype(float)
    assert gd.shape == ed.shape, f"shape {gd.shape} != {ed.shape}"
    # compare only unmasked positions
    unmasked = ~em
    np.testing.assert_allclose(
        gd[unmasked], ed[unmasked], rtol=rtol, atol=atol,
        err_msg="unmasked data mismatch",
    )


# ---------------------------------------------------------------------------
# unary elementwise ufuncs
# ---------------------------------------------------------------------------

UNARY_DATA = [1.0, 2.0, 3.0]
UNARY_MASK = [0, 1, 0]


@pytest.mark.parametrize("name", [
    "sin", "cos", "tan", "arctan", "sinh", "cosh", "tanh", "arcsinh",
    "exp", "negative", "absolute", "abs", "fabs", "conjugate",
])
def test_unary_basic(name):
    data = [-1.5, 2.0, 0.5]
    mask = [0, 1, 0]
    got = getattr(fr.ma, name)(fr.ma.masked_array(data, mask))
    exp = getattr(np.ma, name)(np.ma.array(data, mask=mask))
    assert_masked_equal(got, exp)


@pytest.mark.parametrize("name", ["floor", "ceil", "around"])
def test_unary_rounding(name):
    data = [1.7, -2.3, 2.5]
    mask = [0, 1, 0]
    got = getattr(fr.ma, name)(fr.ma.masked_array(data, mask))
    exp = getattr(np.ma, name)(np.ma.array(data, mask=mask))
    assert_masked_equal(got, exp)


def test_around_decimals():
    data = [1.234, 5.678, 2.345]
    mask = [0, 1, 0]
    got = fr.ma.around(fr.ma.masked_array(data, mask), 1)
    exp = np.ma.around(np.ma.array(data, mask=mask), 1)
    assert_masked_equal(got, exp)


@pytest.mark.parametrize("name,data", [
    ("sqrt", [4.0, -1.0, 9.0]),
    ("log", [1.0, -1.0, np.e]),
    ("log2", [1.0, -1.0, 8.0]),
    ("log10", [1.0, -1.0, 1000.0]),
    ("arcsin", [0.5, 2.0, -0.5]),
    ("arccos", [0.5, 2.0, -0.5]),
    ("arccosh", [2.0, 0.5, 3.0]),
    ("arctanh", [0.5, 2.0, -0.5]),
])
def test_unary_domain(name, data):
    """Domain-out values get masked, matching numpy.ma."""
    got = getattr(fr.ma, name)(fr.ma.masked_array(data, [0, 0, 0]))
    exp = getattr(np.ma, name)(np.ma.array(data, mask=[0, 0, 0]))
    assert_masked_equal(got, exp)


def test_unary_accepts_plain_list():
    got = fr.ma.sqrt([4.0, 9.0, 16.0])
    exp = np.ma.sqrt(np.ma.array([4.0, 9.0, 16.0]))
    assert_masked_equal(got, exp)


# ---------------------------------------------------------------------------
# binary elementwise ufuncs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "add", "subtract", "multiply", "power", "arctan2", "hypot",
    "fmod", "remainder", "mod", "floor_divide", "maximum", "minimum",
])
def test_binary_basic(name):
    a = [6.0, 7.0, 8.0]
    am = [0, 1, 0]
    b = [3.0, 3.0, 5.0]
    bm = [0, 0, 1]
    got = getattr(fr.ma, name)(fr.ma.masked_array(a, am), fr.ma.masked_array(b, bm))
    exp = getattr(np.ma, name)(np.ma.array(a, mask=am), np.ma.array(b, mask=bm))
    assert_masked_equal(got, exp)


def test_binary_scalar_second_arg():
    a = [1.0, 2.0, 3.0]
    am = [0, 1, 0]
    got = fr.ma.add(fr.ma.masked_array(a, am), 10.0)
    exp = np.ma.add(np.ma.array(a, mask=am), 10.0)
    assert_masked_equal(got, exp)


def test_divide_domain_zero_denominator():
    a = [1.0, 2.0, 3.0]
    b = [0.0, 2.0, 0.0]
    got = fr.ma.divide(fr.ma.masked_array(a, [0, 0, 0]), fr.ma.masked_array(b, [0, 0, 0]))
    exp = np.ma.divide(np.ma.array(a, mask=[0, 0, 0]), np.ma.array(b, mask=[0, 0, 0]))
    assert_masked_equal(got, exp)


def test_true_divide_alias():
    a = [10.0, 20.0]
    got = fr.ma.true_divide(fr.ma.masked_array(a, [0, 0]), fr.ma.masked_array([2.0, 4.0], [0, 0]))
    exp = np.ma.true_divide(np.ma.array(a, mask=[0, 0]), np.ma.array([2.0, 4.0], mask=[0, 0]))
    assert_masked_equal(got, exp)


@pytest.mark.parametrize("name", ["bitwise_and", "bitwise_or", "bitwise_xor"])
def test_binary_bitwise(name):
    a = [5.0, 6.0, 7.0]
    b = [3.0, 3.0, 3.0]
    got = getattr(fr.ma, name)(fr.ma.masked_array(a, [0, 1, 0]), fr.ma.masked_array(b, [0, 0, 0]))
    exp = getattr(np.ma, name)(np.ma.array(a, mask=[0, 1, 0]).astype(int),
                               np.ma.array(b, mask=[0, 0, 0]).astype(int))
    assert_masked_equal(got, exp)


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------

REDUC_DATA = [1.0, 2.0, 3.0, 4.0]
REDUC_MASK = [0, 1, 0, 0]


@pytest.mark.parametrize("name", ["prod", "median", "ptp", "average", "count"])
def test_reduction_scalar(name):
    m = fr.ma.masked_array(REDUC_DATA, REDUC_MASK)
    npm = np.ma.array(REDUC_DATA, mask=REDUC_MASK)
    got = getattr(fr.ma, name)(m)
    exp = getattr(np.ma, name)(npm)
    assert float(got) == pytest.approx(float(exp))


@pytest.mark.parametrize("name", ["argmin", "argmax"])
def test_reduction_argindex(name):
    data = [3.0, 1.0, 9.0, 2.0]
    mask = [0, 1, 0, 0]
    got = getattr(fr.ma, name)(fr.ma.masked_array(data, mask))
    exp = getattr(np.ma, name)(np.ma.array(data, mask=mask))
    assert int(got) == int(exp)


def test_average_weighted():
    data = [1.0, 2.0, 3.0]
    got = fr.ma.average(fr.ma.masked_array(data, [0, 0, 0]), weights=[1.0, 2.0, 3.0])
    exp = np.ma.average(np.ma.array(data, mask=[0, 0, 0]), weights=[1.0, 2.0, 3.0])
    assert float(got) == pytest.approx(float(exp))


@pytest.mark.parametrize("data,mask", [
    ([1.0, 2.0, 3.0], [0, 0, 0]),
    ([0.0, 1.0, 2.0], [1, 0, 0]),
    ([0.0, 0.0, 0.0], [0, 0, 0]),
])
def test_all_any(data, mask):
    m = fr.ma.masked_array(data, mask)
    npm = np.ma.array(data, mask=mask)
    assert bool(fr.ma.all(m)) == bool(np.ma.all(npm))
    assert bool(fr.ma.any(m)) == bool(np.ma.any(npm))


def test_anom():
    data = [1.0, 2.0, 3.0, 4.0]
    got = fr.ma.anom(fr.ma.masked_array(data, [0, 0, 0, 0]))
    exp = np.ma.anom(np.ma.array(data, mask=[0, 0, 0, 0]))
    assert_masked_equal(got, exp)


def test_all_masked_returns_singleton():
    m = fr.ma.masked_array([1.0, 2.0], [1, 1])
    assert fr.ma.prod(m) is np.ma.masked
    assert fr.ma.median(m) is np.ma.masked
    assert fr.ma.ptp(m) is np.ma.masked


# ---------------------------------------------------------------------------
# creation
# ---------------------------------------------------------------------------

def test_zeros():
    assert_masked_equal(fr.ma.zeros(4), np.ma.zeros(4))
    assert_masked_equal(fr.ma.zeros((2, 3)), np.ma.zeros((2, 3)))


def test_ones():
    assert_masked_equal(fr.ma.ones(3), np.ma.ones(3))


def test_empty_shape():
    got = fr.ma.empty(5)
    assert got.shape == (5,)
    assert fr.ma.count(got) == 5  # nomask


def test_arange():
    assert_masked_equal(fr.ma.arange(5), np.ma.arange(5))
    assert_masked_equal(fr.ma.arange(2, 10, 2), np.ma.arange(2, 10, 2))


def test_identity():
    assert_masked_equal(fr.ma.identity(3), np.ma.identity(3))


def test_asarray():
    got = fr.ma.asarray([1.0, 2.0, 3.0])
    exp = np.ma.asarray([1.0, 2.0, 3.0])
    assert_masked_equal(got, exp)


def test_asarray_passthrough_keeps_mask():
    m = fr.ma.masked_array([1.0, 2.0, 3.0], [0, 1, 0])
    got = fr.ma.asarray(m)
    assert_masked_equal(got, np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]))


def test_copy():
    m = fr.ma.masked_array([1.0, 2.0], [0, 1])
    got = fr.ma.copy(m)
    assert_masked_equal(got, np.ma.array([1.0, 2.0], mask=[0, 1]))


# ---------------------------------------------------------------------------
# manipulation
# ---------------------------------------------------------------------------

def _md():
    return [[1.0, 2.0], [3.0, 4.0]], [[0, 1], [0, 0]]


def test_reshape():
    d, mk = _md()
    got = fr.ma.reshape(fr.ma.masked_array(d, mk), (4,))
    exp = np.ma.reshape(np.ma.array(d, mask=mk), (4,))
    assert_masked_equal(got, exp)


def test_ravel():
    d, mk = _md()
    got = fr.ma.ravel(fr.ma.masked_array(d, mk))
    exp = np.ma.ravel(np.ma.array(d, mask=mk))
    assert_masked_equal(got, exp)


def test_transpose():
    d, mk = _md()
    got = fr.ma.transpose(fr.ma.masked_array(d, mk))
    exp = np.ma.transpose(np.ma.array(d, mask=mk))
    assert_masked_equal(got, exp)


def test_squeeze():
    got = fr.ma.squeeze(fr.ma.masked_array([[1.0, 2.0, 3.0]], [[0, 1, 0]]))
    exp = np.ma.squeeze(np.ma.array([[1.0, 2.0, 3.0]], mask=[[0, 1, 0]]))
    assert_masked_equal(got, exp)


def test_expand_dims():
    got = fr.ma.expand_dims(fr.ma.masked_array([1.0, 2.0], [0, 1]), 0)
    exp = np.ma.expand_dims(np.ma.array([1.0, 2.0], mask=[0, 1]), 0)
    assert got.shape == exp.shape
    assert_masked_equal(got, exp)


def test_concatenate():
    got = fr.ma.concatenate([
        fr.ma.masked_array([1.0, 2.0], [0, 1]),
        fr.ma.masked_array([3.0, 4.0], [1, 0]),
    ])
    exp = np.ma.concatenate([
        np.ma.array([1.0, 2.0], mask=[0, 1]),
        np.ma.array([3.0, 4.0], mask=[1, 0]),
    ])
    assert_masked_equal(got, exp)


def test_diag():
    got = fr.ma.diag(fr.ma.masked_array([1.0, 2.0], [0, 1]))
    exp = np.ma.diag(np.ma.array([1.0, 2.0], mask=[0, 1]))
    assert_masked_equal(got, exp)


def test_repeat():
    got = fr.ma.repeat(fr.ma.masked_array([1.0, 2.0], [0, 1]), 2)
    exp = np.ma.repeat(np.ma.array([1.0, 2.0], mask=[0, 1]), 2)
    assert_masked_equal(got, exp)


def test_clip():
    got = fr.ma.clip(fr.ma.masked_array([1.0, 5.0, 9.0], [0, 1, 0]), 2.0, 8.0)
    exp = np.ma.clip(np.ma.array([1.0, 5.0, 9.0], mask=[0, 1, 0]), 2.0, 8.0)
    assert_masked_equal(got, exp)


# ---------------------------------------------------------------------------
# mask helpers + predicates
# ---------------------------------------------------------------------------

def test_getmaskarray():
    got = fr.ma.getmaskarray(fr.ma.masked_array([1.0, 2.0, 3.0], [0, 1, 0]))
    exp = np.ma.getmaskarray(np.ma.array([1.0, 2.0, 3.0], mask=[0, 1, 0]))
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_getmaskarray_nomask_is_full_false():
    got = fr.ma.getmaskarray(fr.ma.masked_array([1.0, 2.0, 3.0]))
    exp = np.ma.getmaskarray(np.ma.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_make_mask():
    got = fr.ma.make_mask([0, 1, 0])
    exp = np.ma.make_mask([0, 1, 0])
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_make_mask_all_false_is_nomask():
    got = fr.ma.make_mask([0, 0, 0])
    assert got is np.ma.nomask


def test_make_mask_none():
    got = fr.ma.make_mask_none((3,))
    exp = np.ma.make_mask_none((3,))
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_mask_or():
    got = fr.ma.mask_or([1, 0, 1], [0, 1, 1])
    exp = np.ma.mask_or(np.array([1, 0, 1], dtype=bool), np.array([0, 1, 1], dtype=bool))
    np.testing.assert_array_equal(np.asarray(got), exp)


def test_masked_values():
    data = [1.0, 1.1, 2.0, 1.0, 3.0]
    got = fr.ma.masked_values(data, 1.0)
    exp = np.ma.masked_values(data, 1.0)
    assert_masked_equal(got, exp)


def test_masked_object():
    data = [1.0, 2.0, 3.0, 2.0]
    got = fr.ma.masked_object(data, 2.0)
    exp = np.ma.masked_object(data, 2.0)
    assert_masked_equal(got, exp)


def test_fix_invalid():
    data = [1.0, np.nan, np.inf, 4.0]
    got = fr.ma.fix_invalid(fr.ma.masked_array(data, [0, 0, 0, 0]))
    exp = np.ma.fix_invalid(np.ma.array(data, mask=[0, 0, 0, 0]))
    assert_masked_equal(got, exp)


def test_is_mask():
    assert fr.ma.is_mask(np.array([True, False])) is True
    assert fr.ma.is_mask(np.ma.nomask) is True
    assert fr.ma.is_mask([1, 2, 3]) is False
    assert fr.ma.is_mask(fr.ma.masked_array([1.0, 2.0], [0, 1])) is False
    # parity with numpy
    assert fr.ma.is_mask(np.array([True, False])) == np.ma.is_mask(np.array([True, False]))
    assert fr.ma.is_mask([1, 2, 3]) == np.ma.is_mask([1, 2, 3])


def test_is_masked_array():
    m = fr.ma.masked_array([1.0, 2.0], [0, 1])
    assert fr.ma.isMaskedArray(m) is True
    assert fr.ma.isMaskedArray([1, 2, 3]) is False
    assert fr.ma.isMaskedArray(m) == np.ma.isMaskedArray(np.ma.array([1.0, 2.0]))
    assert fr.ma.isMA(m) is True
    assert fr.ma.isarray(m) is True


def test_set_and_default_fill_value():
    m = fr.ma.masked_array([1.0, 2.0], [0, 1])
    fr.ma.set_fill_value(m, 7.0)
    assert m.fill_value == 7.0
    assert fr.ma.default_fill_value([1.0, 2.0]) == np.ma.default_fill_value(np.array([1.0]))


def test_is_masked():
    assert fr.ma.is_masked(fr.ma.masked_array([1.0, 2.0], [1, 0])) is True
    assert fr.ma.is_masked(fr.ma.masked_array([1.0, 2.0])) is False
