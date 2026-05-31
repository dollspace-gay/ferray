"""float16-preserving reductions, batch 2 (#956).

numpy computes these reductions at float32 width internally then narrows the
reduced result back to float16, so the OUTPUT dtype stays ``float16`` (verified
live numpy 2.4.4). A prior translation pass widened the ferray output to
``float64`` by coercing the non-f32/f64 input to ``float64`` ahead of the
real-only reduction macro. This pins the eleven still-widening ops --
``median`` / ``percentile`` / ``quantile`` / ``nanmin`` / ``nanmax`` /
``nansum`` / ``nanmean`` / ``nanstd`` / ``nanvar`` / ``nanmedian`` /
``average`` -- against numpy (value + dtype), the #954 follow-up to the
``sum`` / ``mean`` / ``std`` / ``var`` / ``min`` / ``max`` / ``ptp`` /
``cumsum`` batch.

Expected values come from the live numpy call (R-CHAR-3) -- never literal-copied
from the ferray side.
"""

import numpy as np
import ferray as fr
import pytest


def _f16(*vals):
    return np.array(vals, dtype="float16")


A = _f16(1, 2, 3)
B = np.array([1, np.nan, 3], dtype="float16")
A2 = np.array([[1, 2, 3], [4, 5, 6]], dtype="float16")
B2 = np.array([[1, np.nan, 3], [4, 5, np.nan]], dtype="float16")


def _assert_match(got, expected):
    got = np.asarray(got)
    expected = np.asarray(expected)
    assert got.dtype == expected.dtype, f"{got.dtype} != {expected.dtype}"
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, equal_nan=True)


# --- whole-array float16 preservation (value + dtype) ----------------------

def test_median_f16():
    _assert_match(fr.median(A), np.median(A))


def test_percentile_scalar_q_f16():
    _assert_match(fr.percentile(A, 50), np.percentile(A, 50))


def test_quantile_scalar_q_f16():
    _assert_match(fr.quantile(A, 0.5), np.quantile(A, 0.5))


def test_nanmin_f16():
    _assert_match(fr.nanmin(B), np.nanmin(B))


def test_nanmax_f16():
    _assert_match(fr.nanmax(B), np.nanmax(B))


def test_nansum_f16():
    _assert_match(fr.nansum(B), np.nansum(B))


def test_nanmean_f16():
    _assert_match(fr.nanmean(B), np.nanmean(B))


def test_nanstd_f16():
    _assert_match(fr.nanstd(B), np.nanstd(B))


def test_nanvar_f16():
    _assert_match(fr.nanvar(B), np.nanvar(B))


def test_nanmedian_f16():
    _assert_match(fr.nanmedian(B), np.nanmedian(B))


def test_average_f16():
    _assert_match(fr.average(A), np.average(A))


# --- axis variants keep float16 --------------------------------------------

def test_median_axis_f16():
    _assert_match(fr.median(A2, axis=0), np.median(A2, axis=0))


def test_nanmean_axis_f16():
    _assert_match(fr.nanmean(B2, axis=1), np.nanmean(B2, axis=1))


def test_nansum_axis_f16():
    _assert_match(fr.nansum(B2, axis=0), np.nansum(B2, axis=0))


def test_nanmin_axis_f16():
    _assert_match(fr.nanmin(B2, axis=1), np.nanmin(B2, axis=1))


def test_average_axis_f16():
    _assert_match(fr.average(A2, axis=0), np.average(A2, axis=0))


# --- ddof forwarded -------------------------------------------------------

def test_nanstd_ddof_f16():
    _assert_match(fr.nanstd(B2, ddof=1), np.nanstd(B2, ddof=1))


def test_nanvar_ddof_f16():
    _assert_match(fr.nanvar(B2, ddof=1), np.nanvar(B2, ddof=1))


# --- q-list: numpy itself returns float64; delegating reproduces it --------

def test_percentile_list_q_f16():
    _assert_match(fr.percentile(A2, [25, 50, 75]), np.percentile(A2, [25, 50, 75]))


def test_quantile_list_q_f16():
    _assert_match(fr.quantile(A2, [0.25, 0.5]), np.quantile(A2, [0.25, 0.5]))


# --- weighted average keeps float16 ----------------------------------------

def test_average_weights_f16():
    w = np.array([1, 2, 3, 4, 5, 6], dtype="float16").reshape(2, 3)
    _assert_match(fr.average(A2, weights=w), np.average(A2, weights=w))


# --- nan handling: nan-variants skip NaN per numpy -------------------------

def test_nan_variants_skip_nan_f16():
    for ff, nf in [
        (fr.nansum, np.nansum),
        (fr.nanmean, np.nanmean),
        (fr.nanmin, np.nanmin),
        (fr.nanmax, np.nanmax),
        (fr.nanmedian, np.nanmedian),
        (fr.nanstd, np.nanstd),
        (fr.nanvar, np.nanvar),
    ]:
        _assert_match(ff(B), nf(B))


# --- real f32/f64/int paths UNCHANGED (non-regression) ---------------------

def test_real_paths_unchanged():
    f64 = np.array([1.0, 2.0, 3.0])
    f32 = np.array([1, 2, 3], dtype="float32")
    i32 = np.array([1, 2, 3], dtype="int32")
    _assert_match(fr.median(f64), np.median(f64))
    _assert_match(fr.median(f32), np.median(f32))
    _assert_match(fr.median(i32), np.median(i32))
    _assert_match(fr.nansum(f64), np.nansum(f64))
    _assert_match(fr.nanstd(f32), np.nanstd(f32))
    _assert_match(fr.average(i32), np.average(i32))
    _assert_match(fr.percentile(f64, 50), np.percentile(f64, 50))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
