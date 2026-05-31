"""ACToR critic pins: datetime64/timedelta64 divergences after epic #940 (HEAD 5d9a4ddc).

Each test derives its expected value LIVE from numpy (R-CHAR-3 — no literal-copied
target values). They are written to FAIL against the current ferray implementation;
they pin remaining divergences for the generator to fix.

Two divergence classes:
  * R-CODE-4 SILENT-FLOAT/INT (CRITICAL): ferray returns a bare float64/int64 where
    numpy returns a datetime64/timedelta64 scalar/array. Datetime semantics are
    silently corrupted into a unit-less number.
  * RAISE-vs-COMPUTE (HIGH/MEDIUM): numpy computes a datetime/timedelta result for a
    common op; ferray rejects the dtype.

Run: pytest tests/test_divergence_datetime_audit.py -q
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


# ---------------------------------------------------------------------------
# R-CODE-4 SILENT-FLOAT/INT class (CRITICAL)
# ---------------------------------------------------------------------------

def test_median_timedelta_silent_float():
    """SILENT-FLOAT: np.median(td) -> timedelta64; ferray -> float64."""
    exp = np.median(_ntd([2, 4, 6]))                # timedelta64(4,'D')
    got = fr.median(_td([2, 4, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"median(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_nanmedian_timedelta_silent_float():
    """SILENT-FLOAT: np.nanmedian(td) -> timedelta64; ferray -> float64."""
    exp = np.nanmedian(_ntd([2, 4, 6]))
    got = fr.nanmedian(_td([2, 4, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"nanmedian(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_nanmax_timedelta_silent_float():
    """SILENT-FLOAT: np.nanmax(td) -> timedelta64; ferray -> float64."""
    exp = np.nanmax(_ntd([2, 4, 6]))
    got = fr.nanmax(_td([2, 4, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"nanmax(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_nanmin_datetime_silent_float():
    """SILENT-FLOAT: np.nanmin(datetime) -> datetime64; ferray -> float64."""
    exp = np.nanmin(_ndt(['2020-01-01', '2020-01-09']))
    got = fr.nanmin(_dt(['2020-01-01', '2020-01-09']))
    assert str(got.dtype) == str(exp.dtype), (
        f"nanmin(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_nansum_timedelta_silent_float():
    """SILENT-FLOAT: np.nansum(td) -> timedelta64; ferray -> float64."""
    exp = np.nansum(_ntd([2, 4, 6]))
    got = fr.nansum(_td([2, 4, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"nansum(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_percentile_timedelta_silent_float():
    """SILENT-FLOAT: np.percentile(td,50) -> timedelta64; ferray -> float64."""
    exp = np.percentile(_ntd([2, 4, 6]), 50)
    got = fr.percentile(_td([2, 4, 6]), 50)
    assert str(got.dtype) == str(exp.dtype), (
        f"percentile(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_quantile_timedelta_silent_float():
    """SILENT-FLOAT: np.quantile(td,.5) -> timedelta64; ferray -> float64."""
    exp = np.quantile(_ntd([2, 4, 6]), 0.5)
    got = fr.quantile(_td([2, 4, 6]), 0.5)
    assert str(got.dtype) == str(exp.dtype), (
        f"quantile(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_percentile_datetime_silent_float():
    """SILENT-FLOAT: np.percentile(datetime,50) -> datetime64; ferray -> float64."""
    exp = np.percentile(_ndt(['2020-01-01', '2020-01-05', '2020-01-09']), 50)
    got = fr.percentile(_dt(['2020-01-01', '2020-01-05', '2020-01-09']), 50)
    assert str(got.dtype) == str(exp.dtype), (
        f"percentile(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_gradient_timedelta_silent_float():
    """SILENT-FLOAT: np.gradient(td) -> timedelta64 array; ferray -> float64."""
    exp = np.gradient(_ntd([2, 4, 6]))
    got = fr.gradient(_td([2, 4, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"gradient(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


# ---------------------------------------------------------------------------
# RAISE-vs-COMPUTE: np computes datetime/timedelta result; ferray rejects.
# diff is the highest-value common op.
# ---------------------------------------------------------------------------

def test_diff_datetime_returns_timedelta():
    """np.diff(datetime) -> timedelta64; ferray raises TypeError."""
    exp = np.diff(_ndt(['2020-01-01', '2020-01-03', '2020-01-10']))
    got = fr.diff(_dt(['2020-01-01', '2020-01-03', '2020-01-10']))
    assert str(got.dtype) == str(exp.dtype), (
        f"diff(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")
    assert np.array_equal(np.asarray(got), exp)


def test_diff_timedelta_returns_timedelta():
    """np.diff(td) -> timedelta64; ferray raises TypeError."""
    exp = np.diff(_ntd([1, 3, 10]))
    got = fr.diff(_td([1, 3, 10]))
    assert str(got.dtype) == str(exp.dtype), (
        f"diff(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_ediff1d_datetime_returns_timedelta():
    """np.ediff1d(datetime) -> timedelta64; ferray raises TypeError."""
    exp = np.ediff1d(_ndt(['2020-01-01', '2020-01-03']))
    got = fr.ediff1d(_dt(['2020-01-01', '2020-01-03']))
    assert str(got.dtype) == str(exp.dtype), (
        f"ediff1d(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_abs_timedelta_returns_timedelta():
    """np.abs(td) -> timedelta64 (magnitude); ferray raises TypeError."""
    exp = np.abs(_ntd([-1, 3, -10]))
    got = fr.abs(_td([-1, 3, -10]))
    assert str(got.dtype) == str(exp.dtype), (
        f"abs(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_negative_timedelta_returns_timedelta():
    """np.negative(td) -> timedelta64; ferray raises TypeError."""
    exp = np.negative(_ntd([1, 3, 10]))
    got = fr.negative(_td([1, 3, 10]))
    assert str(got.dtype) == str(exp.dtype), (
        f"negative(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_sign_timedelta_returns_timedelta():
    """np.sign(td) -> timedelta64 ({-1,0,1} as td); ferray raises TypeError."""
    exp = np.sign(_ntd([-2, 0, 6]))
    got = fr.sign(_td([-2, 0, 6]))
    assert str(got.dtype) == str(exp.dtype), (
        f"sign(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_maximum_datetime_lexicographic():
    """np.maximum(datetime,datetime) -> datetime64; ferray raises TypeError."""
    a = _ndt(['2020-01-01', '2020-01-05', '2020-01-09'])
    b = _ndt(['2020-01-05', '2020-01-05', '2020-01-05'])
    exp = np.maximum(a, b)
    got = fr.maximum(_dt(['2020-01-01', '2020-01-05', '2020-01-09']),
                     _dt(['2020-01-05', '2020-01-05', '2020-01-05']))
    assert str(got.dtype) == str(exp.dtype), (
        f"maximum(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_minimum_timedelta():
    """np.minimum(td,td) -> timedelta64; ferray raises TypeError."""
    exp = np.minimum(_ntd([2, 4, 6]), _ntd([5, 5, 5]))
    got = fr.minimum(_td([2, 4, 6]), _td([5, 5, 5]))
    assert str(got.dtype) == str(exp.dtype), (
        f"minimum(timedelta) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_where_datetime_preserves_dtype():
    """np.where(cond,datetime,datetime) -> datetime64; ferray raises TypeError."""
    cond = np.array([True, False, True])
    a = _ndt(['2020-01-01', '2020-01-05', '2020-01-09'])
    b = _ndt(['1999-01-01', '1999-01-01', '1999-01-01'])
    exp = np.where(cond, a, b)
    got = fr.where(fr.array([True, False, True]),
                   _dt(['2020-01-01', '2020-01-05', '2020-01-09']),
                   _dt(['1999-01-01', '1999-01-01', '1999-01-01']))
    assert str(got.dtype) == str(exp.dtype), (
        f"where(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_clip_datetime_preserves_dtype():
    """np.clip(datetime,lo,hi) -> datetime64; ferray raises TypeError."""
    a = _ndt(['2020-01-01', '2020-01-05', '2020-01-09'])
    exp = np.clip(a, np.datetime64('2020-01-03'), np.datetime64('2020-01-07'))
    got = fr.clip(_dt(['2020-01-01', '2020-01-05', '2020-01-09']),
                  fr.datetime64('2020-01-03'), fr.datetime64('2020-01-07'))
    assert str(got.dtype) == str(exp.dtype), (
        f"clip(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


# ---------------------------------------------------------------------------
# Manipulation (pure data-movement) MUST preserve datetime dtype.
# ---------------------------------------------------------------------------

_MANIP = ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04']


def test_reshape_datetime_preserves_dtype():
    exp = _ndt(_MANIP).reshape(2, 2)
    got = fr.reshape(_dt(_MANIP), (2, 2))
    assert str(got.dtype) == str(exp.dtype), (
        f"reshape(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_transpose_datetime_preserves_dtype():
    exp = _ndt(_MANIP).reshape(2, 2).T
    got = fr.transpose(fr.reshape(_dt(_MANIP), (2, 2)))
    assert str(got.dtype) == str(exp.dtype), (
        f"transpose(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_flip_datetime_preserves_dtype():
    exp = np.flip(_ndt(_MANIP))
    got = fr.flip(_dt(_MANIP))
    assert str(got.dtype) == str(exp.dtype), (
        f"flip(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_roll_datetime_preserves_dtype():
    exp = np.roll(_ndt(_MANIP), 1)
    got = fr.roll(_dt(_MANIP), 1)
    assert str(got.dtype) == str(exp.dtype), (
        f"roll(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_repeat_datetime_preserves_dtype():
    exp = np.repeat(_ndt(_MANIP), 2)
    got = fr.repeat(_dt(_MANIP), 2)
    assert str(got.dtype) == str(exp.dtype), (
        f"repeat(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_tile_datetime_preserves_dtype():
    exp = np.tile(_ndt(_MANIP), 2)
    got = fr.tile(_dt(_MANIP), 2)
    assert str(got.dtype) == str(exp.dtype), (
        f"tile(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_stack_datetime_preserves_dtype():
    exp = np.stack([_ndt(_MANIP), _ndt(_MANIP)])
    got = fr.stack([_dt(_MANIP), _dt(_MANIP)])
    assert str(got.dtype) == str(exp.dtype), (
        f"stack(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_concatenate_datetime_preserves_dtype():
    exp = np.concatenate([_ndt(['2020-01-01']), _ndt(['2020-01-02'])])
    got = fr.concatenate([_dt(['2020-01-01']), _dt(['2020-01-02'])])
    assert str(got.dtype) == str(exp.dtype), (
        f"concatenate(datetime) dtype: numpy {exp.dtype}, ferray {got.dtype}")


def test_count_nonzero_timedelta():
    """np.count_nonzero(td) -> int; ferray raises TypeError."""
    exp = np.count_nonzero(_ntd([0, 2, 0]))
    got = fr.count_nonzero(_td([0, 2, 0]))
    assert int(np.asarray(got)) == int(exp)


def test_nonzero_timedelta():
    """np.nonzero(td) works on td; ferray raises TypeError."""
    exp = np.nonzero(_ntd([0, 2, 0]))[0]
    got = fr.nonzero(_td([0, 2, 0]))[0]
    assert np.array_equal(np.asarray(got), exp)
