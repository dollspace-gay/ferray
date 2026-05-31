"""Expansion tests: datetime64/timedelta64 elementwise + selection ops (#947).

The LAST datetime blocker — `diff` / `ediff1d` / `abs` / `negative` / `sign`
/ `maximum` / `minimum` / `where` / `clip` / `count_nonzero` / `nonzero` over
datetime64 / timedelta64 inputs.

Every expected value is derived LIVE from numpy (R-CHAR-3 — no literal-copied
target values). ferray (`import ferray as fr`) is compared against numpy
(`import numpy as np`, the 2.4 oracle) for dtype AND values, including the NaT
/ cross-unit / 2-D edge cases.
"""
import numpy as np
import pytest
import ferray as fr


def _dt(vals, unit='D'):
    return fr.array(vals, dtype=f'datetime64[{unit}]')


def _ndt(vals, unit='D'):
    return np.array(vals, dtype=f'datetime64[{unit}]')


def _td(vals, unit='D'):
    return fr.array(vals, dtype=f'timedelta64[{unit}]')


def _ntd(vals, unit='D'):
    return np.array(vals, dtype=f'timedelta64[{unit}]')


def _same(got, exp):
    assert str(np.asarray(got).dtype) == str(np.asarray(exp).dtype), (
        f"dtype: numpy {np.asarray(exp).dtype}, ferray {np.asarray(got).dtype}")
    # NaT != NaT, so compare on the int64 tick view (NaT == i64::MIN).
    g = np.asarray(got)
    e = np.asarray(exp)
    if g.dtype.kind in ('M', 'm'):
        assert np.array_equal(g.view('int64'), e.view('int64'))
    else:
        assert np.array_equal(g, e)


# ---------------------------------------------------------------------------
# diff / ediff1d
# ---------------------------------------------------------------------------

def test_diff_datetime():
    src = ['2020-01-01', '2020-01-05', '2020-01-03']
    _same(fr.diff(_dt(src)), np.diff(_ndt(src)))


def test_diff_timedelta():
    _same(fr.diff(_td([5, -2, 8])), np.diff(_ntd([5, -2, 8])))


def test_diff_timedelta_unit_h():
    _same(fr.diff(_td([5, -2, 8], 'h')), np.diff(_ntd([5, -2, 8], 'h')))


def test_diff_datetime_n2():
    src = ['2020-01-01', '2020-01-05', '2020-01-03', '2020-01-10']
    _same(fr.diff(_dt(src), 2), np.diff(_ndt(src), n=2))


def test_diff_datetime_nat():
    src = ['2020-01-01', 'NaT', '2020-01-03']
    _same(fr.diff(_dt(src)), np.diff(_ndt(src)))


def test_ediff1d_datetime():
    src = ['2020-01-01', '2020-01-05', '2020-01-03']
    _same(fr.ediff1d(_dt(src)), np.ediff1d(_ndt(src)))


def test_ediff1d_timedelta():
    _same(fr.ediff1d(_td([5, -2, 8])), np.ediff1d(_ntd([5, -2, 8])))


# ---------------------------------------------------------------------------
# abs / negative / sign  (timedelta computes; datetime RAISES)
# ---------------------------------------------------------------------------

def test_abs_timedelta():
    _same(fr.abs(_td([-1, 3, -10])), np.abs(_ntd([-1, 3, -10])))


def test_abs_timedelta_nat():
    src = ['5', '-2', 'NaT']
    _same(fr.abs(_td(src)), np.abs(_ntd(src)))


def test_negative_timedelta():
    _same(fr.negative(_td([5, -2, 8])), np.negative(_ntd([5, -2, 8])))


def test_negative_timedelta_nat():
    _same(fr.negative(_td(['5', 'NaT'])), np.negative(_ntd(['5', 'NaT'])))


def test_sign_timedelta():
    # sign(td) -> timedelta64 ({-1,0,1} ticks AS a timedelta), NOT int.
    exp = np.sign(_ntd([-2, 0, 6]))
    assert str(exp.dtype) == 'timedelta64[D]'
    _same(fr.sign(_td([-2, 0, 6])), exp)


def test_sign_timedelta_nat():
    src = ['-2', '0', '6', 'NaT']
    _same(fr.sign(_td(src)), np.sign(_ntd(src)))


@pytest.mark.parametrize('op', ['abs', 'negative', 'sign'])
def test_unary_datetime_raises(op):
    # numpy: abs/negative/sign over datetime64 RAISE; ferray must match (raise).
    dt = _ndt(['2020-01-01'])
    with pytest.raises(Exception):
        getattr(np, op)(dt)
    with pytest.raises(Exception):
        getattr(fr, op)(_dt(['2020-01-01']))


# ---------------------------------------------------------------------------
# maximum / minimum  (same-kind time pair)
# ---------------------------------------------------------------------------

def test_maximum_datetime():
    a = ['2020-01-01', '2020-01-05', '2020-01-09']
    b = ['2020-01-05', '2020-01-05', '2020-01-05']
    _same(fr.maximum(_dt(a), _dt(b)), np.maximum(_ndt(a), _ndt(b)))


def test_minimum_timedelta():
    _same(fr.minimum(_td([2, 4, 6]), _td([5, 5, 5])),
          np.minimum(_ntd([2, 4, 6]), _ntd([5, 5, 5])))


def test_maximum_datetime_nat_propagates():
    a = ['2020-01-01', 'NaT', '2020-01-03']
    b = ['2020-01-05', '2020-01-02', 'NaT']
    _same(fr.maximum(_dt(a), _dt(b)), np.maximum(_ndt(a), _ndt(b)))
    _same(fr.minimum(_dt(a), _dt(b)), np.minimum(_ndt(a), _ndt(b)))


def test_maximum_datetime_cross_unit():
    a = _ndt(['2020-01-01T05', '2020-01-05T00'], 'h')
    b = _ndt(['2020-01-01', '2020-01-04'], 'D')
    fa = _dt(['2020-01-01T05', '2020-01-05T00'], 'h')
    fb = _dt(['2020-01-01', '2020-01-04'], 'D')
    _same(fr.maximum(fa, fb), np.maximum(a, b))


# ---------------------------------------------------------------------------
# where
# ---------------------------------------------------------------------------

def test_where_datetime():
    cond = [True, False, True]
    a = ['2020-01-01', '2020-01-05', '2020-01-09']
    b = ['1999-01-01', '1999-01-01', '1999-01-01']
    _same(fr.where(fr.array(cond), _dt(a), _dt(b)),
          np.where(np.array(cond), _ndt(a), _ndt(b)))


def test_where_timedelta():
    cond = [True, False, True]
    _same(fr.where(fr.array(cond), _td([5, -2, 8]), _td([0, 0, 0])),
          np.where(np.array(cond), _ntd([5, -2, 8]), _ntd([0, 0, 0])))


# ---------------------------------------------------------------------------
# clip  (composes maximum + minimum)
# ---------------------------------------------------------------------------

def test_clip_datetime():
    a = ['2020-01-01', '2020-01-05', '2020-01-09']
    exp = np.clip(_ndt(a), np.datetime64('2020-01-03'), np.datetime64('2020-01-07'))
    got = fr.clip(_dt(a), fr.datetime64('2020-01-03'), fr.datetime64('2020-01-07'))
    _same(got, exp)


def test_clip_timedelta():
    exp = np.clip(_ntd([5, -2, 8]), np.timedelta64(0, 'D'), np.timedelta64(6, 'D'))
    got = fr.clip(_td([5, -2, 8]), fr.timedelta64(0, 'D'), fr.timedelta64(6, 'D'))
    _same(got, exp)


# ---------------------------------------------------------------------------
# count_nonzero / nonzero  (NaT is nonzero)
# ---------------------------------------------------------------------------

def test_count_nonzero_timedelta():
    exp = np.count_nonzero(_ntd([0, 2, 0]))
    got = fr.count_nonzero(_td([0, 2, 0]))
    assert int(np.asarray(got)) == int(exp)


def test_count_nonzero_timedelta_nat():
    # NaT (i64::MIN) is nonzero -> counted.
    src = ['0', '2', '0', 'NaT']
    exp = np.count_nonzero(_ntd(src))
    got = fr.count_nonzero(_td(src))
    assert int(np.asarray(got)) == int(exp)


def test_count_nonzero_timedelta_axis():
    src = _ntd([0, 2, 0, 5, 0, 7]).reshape(2, 3)
    exp = np.count_nonzero(src, axis=1)
    got = fr.count_nonzero(_td([0, 2, 0, 5, 0, 7]).reshape(2, 3), axis=1)
    assert np.array_equal(np.asarray(got), exp)


def test_nonzero_timedelta():
    exp = np.nonzero(_ntd([0, 2, 0]))[0]
    got = fr.nonzero(_td([0, 2, 0]))[0]
    assert np.array_equal(np.asarray(got), exp)


def test_nonzero_timedelta_nat():
    src = ['0', '2', '0', 'NaT']
    exp = np.nonzero(_ntd(src))[0]
    got = fr.nonzero(_td(src))[0]
    assert np.array_equal(np.asarray(got), exp)


def test_nonzero_timedelta_2d():
    src = _ntd([0, 2, 0, 5]).reshape(2, 2)
    exp = np.nonzero(src)
    got = fr.nonzero(_td([0, 2, 0, 5]).reshape(2, 2))
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        assert np.array_equal(np.asarray(g), e)


# ---------------------------------------------------------------------------
# Real / complex paths UNREGRESSED (the time arms must not perturb numeric).
# ---------------------------------------------------------------------------

def test_real_diff_unregressed():
    assert np.array_equal(np.asarray(fr.diff(fr.array([1.0, 4.0, 9.0]))),
                          np.diff(np.array([1.0, 4.0, 9.0])))


def test_real_maximum_unregressed():
    assert np.array_equal(np.asarray(fr.maximum(fr.array([1, 5, 3]), fr.array([4, 2, 3]))),
                          np.maximum(np.array([1, 5, 3]), np.array([4, 2, 3])))


def test_real_sign_unregressed():
    assert np.array_equal(np.asarray(fr.sign(fr.array([-3.0, 0.0, 7.0]))),
                          np.sign(np.array([-3.0, 0.0, 7.0])))


def test_real_count_nonzero_unregressed():
    assert int(np.asarray(fr.count_nonzero(fr.array([0, 1, 0, 2])))) == \
        int(np.count_nonzero(np.array([0, 1, 0, 2])))
