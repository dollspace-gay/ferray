"""float16 comparison + manipulation/data-move/unary ops: `import ferray as fr`
vs `numpy`.

Pins the float16 divergences closed by #955 (REQ-5 of
`.design/ferray-core-float16.md` — the LAST float16 blocker, converging epic
#951): the comparison ufuncs (`less`/`greater`/`less_equal`/`greater_equal`/
`equal`/`not_equal`), the unary numeric ops (`sign`/`absolute`/`negative`/
`floor`/`ceil`/`trunc`/`round`/`square`/`around`), the float predicates
(`isnan`/`isinf`/`isfinite`/`signbit`), and the entire data-move / view / set /
gather surface (`reshape`/`transpose`/`flip`/`roll`/`repeat`/`tile`/`sort`/
`argsort`/`unique`/`where`/`concatenate`/`stack`/`split`/`diff`/`ediff1d`/
`nonzero`/`take`/`compress`/`extract`/`partition`/`count_nonzero`/`searchsorted`/
`intersect1d`/`union1d`/`setdiff1d`/`setxor1d`/`isin`/`in1d`/…) over float16 —
all of which the real-only dispatch macros (`match_dtype_numeric!` /
`match_dtype_orderable!` / `match_dtype_all!` / `match_dtype_all_complex!` /
`match_dtype_float!`) previously REJECTED with `TypeError: unsupported dtype ...
"float16"`.

The binding now detects a float16 operand and DELEGATES the whole op to numpy's
own top-level function on the ORIGINAL operand(s) via the shared
`crate::conv::is_float16_dtype` + `crate::conv::f16_delegate` seam
(`ferray-python/src/conv.rs`), wired ahead of each rejecting macro in
`ufunc.rs` / `manipulation.rs` / `stats.rs` / `indexing.rs`. Since `fr.ndarray`
IS `numpy.ndarray`, numpy's result is a valid boundary return as-is — numpy
preserves the float16 dtype for data-move ops, computes bool for comparisons,
and applies its exact f32-compute + round-to-f16 narrow for the unary numeric
ops. This keeps `half::f16` out of the Rust boundary entirely — the same
detect-and-delegate pattern as the float16 arithmetic (#953), reductions (#954),
and creation coercion (#952).

Every expected value is derived LIVE from numpy (installed numpy 2.4.x = the
oracle), never literal-copied from the ferray side (goal.md R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr


def _f16(*vals):
    return np.array(list(vals), dtype="float16")


A = _f16(3, 1, 2, 4)
B = _f16(2, 2, 2, 2)
MASK = np.array([True, False, True, False])


# ---------------------------------------------------------------------------
# Comparisons (float16 -> bool). numpy generate_umath.py:567 `less`
# `TD(inexact + times, out='?')` registers a float16 compare loop.
# ---------------------------------------------------------------------------

COMPARISONS = ["less", "greater", "less_equal", "greater_equal", "equal", "not_equal"]


@pytest.mark.parametrize("op", COMPARISONS)
def test_comparison_float16_returns_bool(op):
    a = _f16(1, 2, 3, 4)
    b = _f16(4, 2, 1, 4)
    got = np.asarray(getattr(fr, op)(a, b))
    want = getattr(np, op)(a, b)  # live numpy oracle
    assert got.dtype == want.dtype == np.bool_
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("op", COMPARISONS)
def test_comparison_float16_vs_float32_promotes(op):
    # NEP-50: f16 vs f32 compare still yields bool; numpy owns the promotion.
    a = _f16(1, 2, 3)
    b = np.array([1, 5, 3], dtype="float32")
    got = np.asarray(getattr(fr, op)(a, b))
    want = getattr(np, op)(a, b)
    assert got.dtype == want.dtype == np.bool_
    np.testing.assert_array_equal(got, want)


def test_comparison_float16_nan_unordered():
    a = _f16(np.nan, 1.0, 2.0)
    b = _f16(1.0, 1.0, 1.0)
    for op in COMPARISONS:
        got = np.asarray(getattr(fr, op)(a, b))
        want = getattr(np, op)(a, b)
        np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Unary numeric ops (float16 -> float16, dtype-preserving via f32-compute).
# ---------------------------------------------------------------------------

UNARY_F16_PRESERVING = [
    "sign",
    "absolute",
    "negative",
    "positive",
    "floor",
    "ceil",
    "trunc",
    "rint",
    "square",
    "reciprocal",
    "fabs",
]


@pytest.mark.parametrize("op", UNARY_F16_PRESERVING)
def test_unary_float16_preserves_dtype(op):
    a = _f16(-1.5, 0.0, 2.5, 3.0)
    got = np.asarray(getattr(fr, op)(a))
    want = getattr(np, op)(a)  # live numpy oracle
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, np.asarray(want))


def test_round_float16_preserves_dtype():
    a = _f16(0.5, 1.5, 2.5, -0.5)
    got = np.asarray(fr.round(a))
    want = np.round(a)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_around_decimals_float16_preserves_dtype():
    a = _f16(1.234, 2.345, 3.456)
    got = np.asarray(fr.around(a, 1))
    want = np.around(a, 1)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Float predicates (float16 -> bool).
# ---------------------------------------------------------------------------

PREDICATES = ["isnan", "isinf", "isfinite", "signbit"]


@pytest.mark.parametrize("op", PREDICATES)
def test_predicate_float16_returns_bool(op):
    a = _f16(np.nan, np.inf, -np.inf, 1.0, -0.0)
    got = np.asarray(getattr(fr, op)(a))
    want = getattr(np, op)(a)  # live numpy oracle
    assert got.dtype == want.dtype == np.bool_
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Data-move / view ops (float16 -> float16, dtype preserved through).
# ---------------------------------------------------------------------------


def test_reshape_float16_preserves_dtype():
    got = np.asarray(fr.reshape(A, (2, 2)))
    want = np.reshape(A, (2, 2))
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_transpose_float16_preserves_dtype():
    src = np.reshape(A, (2, 2))
    got = np.asarray(fr.transpose(src))
    want = np.transpose(src)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(
    "name,call_fr,call_np",
    [
        ("flip", lambda x: fr.flip(x), lambda x: np.flip(x)),
        ("roll", lambda x: fr.roll(x, 1), lambda x: np.roll(x, 1)),
        ("repeat", lambda x: fr.repeat(x, 2), lambda x: np.repeat(x, 2)),
        ("tile", lambda x: fr.tile(x, 2), lambda x: np.tile(x, 2)),
        ("ravel", lambda x: fr.ravel(x), lambda x: np.ravel(x)),
        ("squeeze", lambda x: fr.squeeze(x), lambda x: np.squeeze(x)),
        ("expand_dims", lambda x: fr.expand_dims(x, 0), lambda x: np.expand_dims(x, 0)),
        ("sort", lambda x: fr.sort(x), lambda x: np.sort(x)),
        ("unique", lambda x: fr.unique(x), lambda x: np.unique(x)),
        ("clip", lambda x: fr.clip(x, 1.5, 3.5), lambda x: np.clip(x, 1.5, 3.5)),
        ("diff", lambda x: fr.diff(x), lambda x: np.diff(x)),
        ("ediff1d", lambda x: fr.ediff1d(x), lambda x: np.ediff1d(x)),
        ("partition", lambda x: fr.partition(x, 1), lambda x: np.partition(x, 1)),
        ("cumsum", lambda x: fr.cumsum(x), lambda x: np.cumsum(x)),
        ("cumprod", lambda x: fr.cumprod(x), lambda x: np.cumprod(x)),
    ],
)
def test_data_move_float16_preserves_dtype(name, call_fr, call_np):
    got = np.asarray(call_fr(A))
    want = np.asarray(call_np(A))
    assert got.dtype == want.dtype == np.float16, name
    np.testing.assert_array_equal(got, want)


def test_concatenate_float16_preserves_dtype():
    got = np.asarray(fr.concatenate([A, B]))
    want = np.concatenate([A, B])
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_stack_float16_preserves_dtype():
    got = np.asarray(fr.stack([A, B]))
    want = np.stack([A, B])
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_split_float16_preserves_dtype():
    got = fr.split(A, 2)
    want = np.split(A, 2)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        g = np.asarray(g)
        assert g.dtype == w.dtype == np.float16
        np.testing.assert_array_equal(g, w)


def test_where_float16_preserves_dtype():
    got = np.asarray(fr.where(MASK, A, B))
    want = np.where(MASK, A, B)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Index / gather ops over float16 (index outputs are int; gathered values f16).
# ---------------------------------------------------------------------------


def test_argsort_float16_returns_int_indices():
    got = np.asarray(fr.argsort(A))
    want = np.argsort(A)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_take_float16_preserves_dtype():
    got = np.asarray(fr.take(A, [0, 2]))
    want = np.take(A, [0, 2])
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_nonzero_float16_returns_int_indices():
    got = np.asarray(fr.nonzero(A)[0])
    want = np.nonzero(A)[0]
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_flatnonzero_float16_returns_int_indices():
    got = np.asarray(fr.flatnonzero(A))
    want = np.flatnonzero(A)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_argwhere_float16_returns_int_indices():
    got = np.asarray(fr.argwhere(A))
    want = np.argwhere(A)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_count_nonzero_float16():
    assert fr.count_nonzero(A) == np.count_nonzero(A)


def test_compress_float16_preserves_dtype():
    cond = [True, False, True, False]
    got = np.asarray(fr.compress(cond, A))
    want = np.compress(cond, A)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_extract_float16_preserves_dtype():
    got = np.asarray(fr.extract(MASK, A))
    want = np.extract(MASK, A)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_searchsorted_float16():
    sorted_a = np.array([1, 2, 3, 4], dtype="float16")
    got = np.asarray(fr.searchsorted(sorted_a, 2.5))
    want = np.searchsorted(sorted_a, 2.5)
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Set operations over float16 (dtype-preserving for value-returning ops).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op", ["intersect1d", "union1d", "setdiff1d", "setxor1d"]
)
def test_set_op_float16_preserves_dtype(op):
    a = _f16(1, 2, 3, 4)
    b = _f16(2, 4, 6)
    got = np.asarray(getattr(fr, op)(a, b))
    want = getattr(np, op)(a, b)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_isin_float16_returns_bool():
    a = _f16(1, 2, 3, 4)
    b = _f16(2, 4)
    got = np.asarray(fr.isin(a, b))
    want = np.isin(a, b)
    assert got.dtype == want.dtype == np.bool_
    np.testing.assert_array_equal(got, want)


def test_in1d_float16_returns_bool():
    # `numpy.in1d` was removed in numpy 2.x; `fr.in1d` mirrors the legacy
    # behavior, which equals `numpy.isin` over 1-D inputs (the live oracle).
    a = _f16(1, 2, 3, 4)
    b = _f16(2, 4)
    got = np.asarray(fr.in1d(a, b))
    want = np.isin(a, b)
    assert got.dtype == want.dtype == np.bool_
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------
# Overflow / NaN behavior matches numpy exactly (f32-compute + f16-narrow).
# ---------------------------------------------------------------------------


def test_square_float16_overflow_to_inf():
    a = _f16(300.0)  # 300**2 = 90000 overflows float16 (max ~65504) -> inf
    got = np.asarray(fr.square(a))
    want = np.square(a)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)


def test_sort_float16_nan_last():
    a = _f16(np.nan, 3.0, 1.0, np.nan, 2.0)
    got = np.asarray(fr.sort(a))
    want = np.sort(a)
    assert got.dtype == want.dtype == np.float16
    np.testing.assert_array_equal(got, want)  # NaN ordering matches numpy


# ---------------------------------------------------------------------------
# Real-path regression guards: f32/f64/int unchanged by the float16 arms.
# ---------------------------------------------------------------------------


def test_real_float32_path_unregressed():
    a = np.array([3.0, 1.0, 2.0, 4.0], dtype="float32")
    assert np.asarray(fr.sort(a)).dtype == np.float32
    assert np.asarray(fr.where(MASK, a, a)).dtype == np.float32
    assert np.asarray(fr.less(a, a + 1)).dtype == np.bool_
    np.testing.assert_array_equal(np.asarray(fr.sort(a)), np.sort(a))


def test_real_float64_and_int_path_unregressed():
    a = np.array([3.0, 1.0, 2.0, 4.0], dtype="float64")
    i = np.array([3, 1, 2, 4], dtype="int32")
    assert np.asarray(fr.sort(a)).dtype == np.float64
    assert np.asarray(fr.reshape(i, (2, 2))).dtype == np.int32
    np.testing.assert_array_equal(np.asarray(fr.sort(i)), np.sort(i))
    np.testing.assert_array_equal(
        np.asarray(fr.concatenate([a, a])), np.concatenate([a, a])
    )
