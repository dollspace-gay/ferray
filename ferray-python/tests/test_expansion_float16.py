"""float16 input coercion + reductions: `import ferray as fr` vs `import numpy as np`.

Pins the float16 divergences closed by #952 (input coercion — `fr.array`/
`fr.asarray`/`fr.full` accept `dtype='float16'`) and #954 (reductions —
`fr.sum`/`prod`/`mean`/`std`/`var`/`min`/`max`/`ptp`/`cumsum` over float16 return
a float16 result, eliminating the prior `fr.mean(f16) -> float64` silent
widening).

Every expected value is derived LIVE from numpy (the installed numpy 2.4.x is the
oracle), never literal-copied from the ferray side (goal.md R-CHAR-3). numpy
computes float16 reductions at float32 width internally and narrows the result
back to float16 (`.design/ferray-core-float16.md` compute contract), so delegating
to numpy is bit-for-bit correct.
"""

import numpy as np
import pytest

import ferray as fr


def _eq(frv, npv):
    """float16 in -> float16 out: dtype AND value must match numpy exactly."""
    fa = np.asarray(frv)
    na = np.asarray(npv)
    assert str(fa.dtype) == str(na.dtype), f"dtype {fa.dtype} != numpy {na.dtype}"
    assert np.array_equal(fa, na, equal_nan=True), f"value {fa!r} != numpy {na!r}"


# ---------------------------------------------------------------------------
# REQ-1 (#952): input coercion — array / asarray / full accept float16
# ---------------------------------------------------------------------------


def test_array_float16_dtype_supported():
    # numpy: np.array([1.,2,3], dtype='float16').dtype == float16
    _eq(fr.array([1.0, 2, 3], dtype="float16"), np.array([1.0, 2, 3], dtype="float16"))


def test_array_float16_2d_preserves_shape():
    src = [[1.0, 2], [3, 4]]
    _eq(fr.array(src, dtype="float16"), np.array(src, dtype="float16"))


def test_array_float16_rounds_like_numpy():
    # 0.1 is not exactly representable in float16; numpy rounds to nearest-even.
    # The expected value is numpy's own narrow, so this is not tautological.
    _eq(fr.array([0.1, 0.2, 0.3], dtype="float16"),
        np.array([0.1, 0.2, 0.3], dtype="float16"))


def test_asarray_of_numpy_float16_array():
    src = np.array([1.0, 2, 3], dtype="float16")
    _eq(fr.asarray(src), np.asarray(src))


def test_asarray_list_with_float16_dtype():
    _eq(fr.asarray([4.0, 5, 6], dtype="float16"),
        np.asarray([4.0, 5, 6], dtype="float16"))


def test_full_float16_dtype_supported():
    # numpy: np.full(3, 1.5, dtype='float16').dtype == float16
    _eq(fr.full(3, 1.5, dtype="float16"), np.full(3, 1.5, dtype="float16"))


def test_full_float16_2d_shape():
    _eq(fr.full((2, 3), 2.0, dtype="float16"), np.full((2, 3), 2.0, dtype="float16"))


def test_array_float16_is_a_copy_not_a_view():
    # numpy.array always copies: mutating the source must not change the result.
    src = np.array([1.0, 2, 3], dtype="float16")
    out = np.asarray(fr.array(src, dtype="float16"))
    src[0] = 9.0
    _eq(out, np.array([1.0, 2, 3], dtype="float16"))


# ---------------------------------------------------------------------------
# REQ-4 (#954): reductions over float16 return float16 (no float64 widening)
# ---------------------------------------------------------------------------

F16 = np.array([1.0, 2, 3], dtype="float16")


@pytest.mark.parametrize(
    "fr_fn, np_fn",
    [
        (fr.sum, np.sum),
        (fr.prod, np.prod),
        (fr.mean, np.mean),
        (fr.std, np.std),
        (fr.var, np.var),
        (fr.min, np.min),
        (fr.max, np.max),
        (fr.ptp, np.ptp),
        (fr.cumsum, np.cumsum),
    ],
)
def test_reduction_float16_returns_float16(fr_fn, np_fn):
    _eq(fr_fn(F16), np_fn(F16))


def test_mean_float16_not_widened_to_float64():
    # The active divergence this REQ eliminates: fr.mean(f16) USED to return
    # float64. numpy returns float16.
    out = np.asarray(fr.mean(F16))
    assert str(out.dtype) == "float16", f"mean widened to {out.dtype}"
    _eq(out, np.mean(F16))


def test_std_float16_not_widened_to_float64():
    out = np.asarray(fr.std(F16))
    assert str(out.dtype) == "float16", f"std widened to {out.dtype}"
    _eq(out, np.std(F16))


def test_var_float16_not_widened_to_float64():
    out = np.asarray(fr.var(F16))
    assert str(out.dtype) == "float16", f"var widened to {out.dtype}"
    _eq(out, np.var(F16))


def test_sum_float16_overflow_to_inf():
    # f32-compute then narrow: a sum exceeding float16's max overflows to inf,
    # exactly as numpy (which warns + narrows). The expected value is numpy's.
    big = np.array([60000.0, 60000.0], dtype="float16")
    with np.errstate(over="ignore"):
        expected = np.sum(big)
    _eq(fr.sum(big), expected)


def test_prod_float16_overflow_to_inf():
    big = np.array([300.0, 300.0], dtype="float16")
    with np.errstate(over="ignore"):
        expected = np.prod(big)
    _eq(fr.prod(big), expected)


M16 = np.array([[1.0, 2], [3, 4]], dtype="float16")


def test_sum_float16_axis0():
    _eq(fr.sum(M16, axis=0), np.sum(M16, axis=0))


def test_sum_float16_axis1():
    _eq(fr.sum(M16, axis=1), np.sum(M16, axis=1))


def test_mean_float16_axis1_stays_float16():
    out = np.asarray(fr.mean(M16, axis=1))
    assert str(out.dtype) == "float16"
    _eq(out, np.mean(M16, axis=1))


def test_max_float16_axis0():
    _eq(fr.max(M16, axis=0), np.max(M16, axis=0))


def test_min_float16_axis1():
    _eq(fr.min(M16, axis=1), np.min(M16, axis=1))


def test_cumsum_float16_axis0():
    _eq(fr.cumsum(M16, axis=0), np.cumsum(M16, axis=0))


def test_var_float16_ddof1():
    _eq(fr.var(F16, ddof=1), np.var(F16, ddof=1))


def test_std_float16_ddof1():
    _eq(fr.std(F16, ddof=1), np.std(F16, ddof=1))


def test_negative_float16_min_max():
    a = np.array([-2.0, -1.0, 3.0], dtype="float16")
    _eq(fr.min(a), np.min(a))
    _eq(fr.max(a), np.max(a))


def test_float16_nan_propagates_in_max():
    a = np.array([1.0, np.nan, 3.0], dtype="float16")
    _eq(fr.max(a), np.max(a))
    _eq(fr.sum(a), np.sum(a))


# ---------------------------------------------------------------------------
# Real-path regression guard: non-float16 reductions are UNCHANGED
# ---------------------------------------------------------------------------


def test_sum_float64_unchanged():
    a = np.array([1.0, 2, 3])
    _eq(fr.sum(a), np.sum(a))


def test_mean_int64_still_float64():
    a = np.array([1, 2, 3])
    out = np.asarray(fr.mean(a))
    assert str(out.dtype) == "float64"
    _eq(out, np.mean(a))


def test_std_float32_unchanged():
    a = np.array([1.0, 2, 3], dtype="float32")
    _eq(fr.std(a), np.std(a))


def test_array_float64_unchanged():
    _eq(fr.array([1.0, 2, 3]), np.array([1.0, 2, 3]))


def test_full_int32_unchanged():
    _eq(fr.full(3, 7, dtype="int32"), np.full(3, 7, dtype="int32"))
