"""R-CODE-4 corruption fix (#946): datetime64/timedelta64 order-statistic
reductions compute datetime/timedelta results (or raise numpy's exact
exception) instead of silently corrupting into float64.

Covered ops: median, nanmedian, nanmax, nanmin, nansum, percentile, quantile,
gradient, average, histogram.

Every expected value is derived LIVE from numpy 2.4 (R-CHAR-3) — never
literal-copied from the ferray side. Each `_eq` compares dtype + shape + the
int64 tick buffer (so NaT == NaT compares equal), which is exactly numpy's
observable contract for datetime64/timedelta64.
"""
import numpy as np
import pytest
import ferray as fr


def _td(vals, unit="D"):
    return fr.array(vals, dtype=f"timedelta64[{unit}]")


def _ntd(vals, unit="D"):
    return np.array(vals, dtype=f"timedelta64[{unit}]")


def _dt(vals, unit="D"):
    return fr.array(vals, dtype=f"datetime64[{unit}]")


def _ndt(vals, unit="D"):
    return np.array(vals, dtype=f"datetime64[{unit}]")


def _eq(got, exp):
    """Assert a ferray time result matches numpy's: same dtype, shape, and
    int64 ticks (NaT-aware)."""
    ga = np.asarray(got)
    ea = np.asarray(exp)
    assert str(ga.dtype) == str(ea.dtype), f"dtype: ferray {ga.dtype} vs numpy {ea.dtype}"
    assert ga.shape == ea.shape, f"shape: ferray {ga.shape} vs numpy {ea.shape}"
    gi = ga.astype("int64")
    ei = ea.astype("int64")
    assert (gi == ei).all(), f"ticks: ferray {gi.tolist()} vs numpy {ei.tolist()}"


# ---------------------------------------------------------------------------
# median / nanmedian
# ---------------------------------------------------------------------------

def test_median_timedelta_odd():
    _eq(fr.median(_td([5, 2, 8])), np.median(_ntd([5, 2, 8])))


def test_median_timedelta_even_truncates():
    # two-middle average truncates toward zero: (2+5)/2 -> 3
    _eq(fr.median(_td([1, 2, 5, 8])), np.median(_ntd([1, 2, 5, 8])))


def test_median_timedelta_nat_propagates():
    # plain median: any NaT in the lane -> NaT
    _eq(fr.median(_td([5, 2, "NaT", 8])), np.median(_ntd([5, 2, "NaT", 8])))


def test_median_timedelta_axis():
    a = [[5, 2, 8], [1, 9, 3]]
    _eq(fr.median(_td(a), axis=1), np.median(_ntd(a), axis=1))


def test_median_datetime_raises():
    # numpy adds two datetimes (mean of two middles) -> undefined -> raises.
    with pytest.raises(Exception) as ei:
        fr.median(_dt(["2020-01-01", "2020-01-05"]))
    with pytest.raises(Exception) as en:
        np.median(_ndt(["2020-01-01", "2020-01-05"]))
    assert type(ei.value).__name__ == type(en.value).__name__


def test_nanmedian_timedelta_skips_nat():
    _eq(fr.nanmedian(_td([5, 2, "NaT", 8])), np.nanmedian(_ntd([5, 2, "NaT", 8])))


def test_nanmedian_datetime_raises():
    with pytest.raises(Exception):
        fr.nanmedian(_dt(["2020-01-01", "2020-01-05"]))


# ---------------------------------------------------------------------------
# nanmax / nanmin / nansum
# ---------------------------------------------------------------------------

def test_nanmax_timedelta_skips_nat():
    _eq(fr.nanmax(_td([5, 2, "NaT", 8])), np.nanmax(_ntd([5, 2, "NaT", 8])))


def test_nanmin_timedelta_skips_nat():
    _eq(fr.nanmin(_td([5, 2, "NaT", 8])), np.nanmin(_ntd([5, 2, "NaT", 8])))


def test_nanmax_datetime():
    a = ["2020-01-01", "2020-01-05", "2020-01-03"]
    _eq(fr.nanmax(_dt(a)), np.nanmax(_ndt(a)))


def test_nanmin_datetime():
    a = ["2020-01-09", "2020-01-01", "2020-01-05"]
    _eq(fr.nanmin(_dt(a)), np.nanmin(_ndt(a)))


def test_nanmin_datetime_axis():
    a = [["2020-01-09", "2020-01-01"], ["2020-01-03", "2020-01-07"]]
    _eq(fr.nanmin(_dt(a), axis=1), np.nanmin(_ndt(a), axis=1))


def test_nansum_timedelta():
    _eq(fr.nansum(_td([5, 2, 8])), np.nansum(_ntd([5, 2, 8])))


def test_nansum_timedelta_nat_propagates():
    # nansum is plain sum for time arrays: NaT is NOT skipped -> NaT.
    _eq(fr.nansum(_td([5, 2, "NaT", 8])), np.nansum(_ntd([5, 2, "NaT", 8])))


def test_nansum_datetime_raises():
    with pytest.raises(Exception):
        fr.nansum(_dt(["2020-01-01", "2020-01-05"]))


# ---------------------------------------------------------------------------
# percentile / quantile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0, 10, 20, 25, 30, 40, 50, 60, 75, 90, 100])
def test_percentile_timedelta_interpolation(p):
    a = [2, 5, 8]
    _eq(fr.percentile(_td(a), p), np.percentile(_ntd(a), p))


def test_percentile_datetime():
    a = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-10"]
    _eq(fr.percentile(_dt(a), 30), np.percentile(_ndt(a), 30))


def test_quantile_timedelta():
    a = [2, 5, 8]
    _eq(fr.quantile(_td(a), 0.3), np.quantile(_ntd(a), 0.3))


def test_quantile_datetime():
    a = ["2020-01-01", "2020-01-05", "2020-01-09"]
    _eq(fr.quantile(_dt(a), 0.5), np.quantile(_ndt(a), 0.5))


def test_percentile_timedelta_sequence_q():
    a = [2, 5, 8]
    _eq(fr.percentile(_td(a), [25, 50, 75]), np.percentile(_ntd(a), [25, 50, 75]))


def test_percentile_timedelta_nat_propagates():
    _eq(fr.percentile(_td([5, 2, "NaT", 8]), 25),
        np.percentile(_ntd([5, 2, "NaT", 8]), 25))


def test_percentile_timedelta_axis():
    a = [[5, 2, 8], [1, 9, 3]]
    _eq(fr.percentile(_td(a), 50, axis=1), np.percentile(_ntd(a), 50, axis=1))


def test_percentile_q_out_of_range_raises():
    with pytest.raises(ValueError):
        fr.percentile(_td([2, 5, 8]), 150)


def test_quantile_q_out_of_range_raises():
    with pytest.raises(ValueError):
        fr.quantile(_td([2, 5, 8]), 1.5)


def test_percentile_finer_unit():
    a = [5, 2, 8]
    _eq(fr.percentile(_td(a, "h"), 25), np.percentile(_ntd(a, "h"), 25))


# ---------------------------------------------------------------------------
# gradient
# ---------------------------------------------------------------------------

def test_gradient_timedelta():
    _eq(fr.gradient(_td([5, 2, 8, 1])), np.gradient(_ntd([5, 2, 8, 1])))


def test_gradient_timedelta_dx():
    _eq(fr.gradient(_td([5, 2, 8]), 2.0), np.gradient(_ntd([5, 2, 8]), 2.0))


def test_gradient_datetime_is_timedelta():
    a = ["2020-01-01", "2020-01-05", "2020-01-03"]
    _eq(fr.gradient(_dt(a)), np.gradient(_ndt(a)))


def test_gradient_timedelta_nat_propagates():
    _eq(fr.gradient(_td([5, 2, "NaT", 8])), np.gradient(_ntd([5, 2, "NaT", 8])))


# ---------------------------------------------------------------------------
# average / histogram — numpy RAISES on time inputs (no silent float)
# ---------------------------------------------------------------------------

def test_average_timedelta_raises():
    with pytest.raises(Exception) as ei:
        fr.average(_td([5, 2, 8]))
    with pytest.raises(Exception) as en:
        np.average(_ntd([5, 2, 8]))
    assert type(ei.value).__name__ == type(en.value).__name__


def test_average_datetime_raises():
    with pytest.raises(Exception):
        fr.average(_dt(["2020-01-01", "2020-01-05"]))


def test_histogram_timedelta_raises():
    with pytest.raises(Exception) as ei:
        fr.histogram(_td([5, 2, 8]))
    with pytest.raises(Exception) as en:
        np.histogram(_ntd([5, 2, 8]))
    assert type(ei.value).__name__ == type(en.value).__name__


def test_histogram_datetime_raises():
    with pytest.raises(Exception):
        fr.histogram(_dt(["2020-01-01", "2020-01-05"]))


# ---------------------------------------------------------------------------
# corruption-gone guard: NO bare float ever returned for a computed time op
# ---------------------------------------------------------------------------

def test_no_silent_float_corruption():
    td = _td([5, 2, 8])
    for fn in (fr.median, fr.nanmedian, fr.nanmax, fr.nanmin, fr.nansum):
        assert np.asarray(fn(td)).dtype.kind == "m", fn.__name__
    assert np.asarray(fr.percentile(td, 50)).dtype.kind == "m"
    assert np.asarray(fr.quantile(td, 0.5)).dtype.kind == "m"
    assert np.asarray(fr.gradient(td)).dtype.kind == "m"
