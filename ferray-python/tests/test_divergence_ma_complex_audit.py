"""ACToR critic re-audit of #868 (complex masked-array construction, 208db934).

Two objectives were audited:

  A. The `match_ma_real!` / `match_ma!` macro split must NOT have regressed the
     REAL-dtype surface (~20 dispatch sites re-threaded).
  B. Complex construction + the compute-vs-raise matrix must be numpy-faithful,
     with the R-CODE-4 silent-imaginary-drop hunt as the priority.

VERDICT SUMMARY (full prose in the critic report):

  * Objective A: NO REGRESSION found. Real-dtype operators, reductions,
    construction/getitem/setitem/filled/.mask/.dtype all still match numpy.ma
    after the macro split (probed live; see the green guard tests below).

  * Objective B / R-CODE-4 imaginary-drop hunt: NO silent imaginary-part drop
    found. Every complex method that numpy returns complex for either computes
    correctly (sum, prod via `fr.ma.prod`, compressed, getitem, setitem, ==/!=)
    or raises a clean TypeError (no panic / process-crash). The reductions that
    route through `to_f64_ma` (mean/var/std/median) are guarded by an explicit
    `complex => return complex_reduction_pending(...)` arm and raise instead of
    silently dropping the imaginary part. This is the doc's flagged BIGGEST RISK
    and it is handled correctly.

  * DIVERGENCE FOUND (the ordering ops): the design doc
    (`.design/ferray-ma-complex.md`) and the #868 commit classify complex
    ordering `<`, `>`, `<=`, `>=` as PERMANENT raises ("numpy raises too —
    complex is unordered"). This is FALSE against the live oracle: numpy 2.4.4
    COMPUTES complex ordering LEXICOGRAPHICALLY with full mask propagation
    (numpy/ma/core.py `MaskedArray.__lt__` etc. delegate to `_comparison`,
    which calls `np.less` on the complex data — and `np.less` on complex
    performs a lexicographic (real, then imag) compare; it does NOT raise).
    ferray raises TypeError where numpy computes. Because the doc froze this as
    "permanent / correct", it is an UNTRACKED gap (unlike the #869/#873 pending
    raises which ARE tracked) and will never be fixed under the current plan.

These pins assert the LIVE numpy.ma ordering result (R-CHAR-3: expected values
derived from a live numpy.ma call, never literal-copied from ferray). They FAIL
today because ferray raises TypeError. They are NOT `@pytest.mark.xfail`'d: this
is a release-blocker-class mis-specification (a translation unit declaring a
behavior "matches numpy" when the oracle contradicts it). The orchestrator
decides tracking; the failing test IS the audit artifact.
"""

import numpy as np
import pytest

import ferray as fr


# Mixed-mask operands so the masked-position propagation is also exercised.
_DATA_A = [1 + 2j, 3 - 1j, 1 + 5j, 2 + 0j]
_DATA_B = [1 + 3j, 3 - 1j, 1 + 5j, 9 + 9j]
_MASK_A = [0, 1, 0, 0]
_MASK_B = [0, 0, 1, 0]


def _np_pair():
    return (
        np.ma.array(_DATA_A, mask=_MASK_A),
        np.ma.array(_DATA_B, mask=_MASK_B),
    )


def _fr_pair():
    return (
        fr.ma.array(_DATA_A, mask=_MASK_A),
        fr.ma.array(_DATA_B, mask=_MASK_B),
    )


# ---------------------------------------------------------------------------
# DIVERGENCE: complex ordering computes in numpy, ferray raises.
# Doc/commit call these "permanent raises"; the live oracle disagrees.
# numpy/ma/core.py MaskedArray ordering -> np.less/greater on complex data
# (lexicographic, NO TypeError). Tracking: filed as a blocker by the critic.
# ---------------------------------------------------------------------------

def test_complex_lt_computes_lexicographically_like_numpy():
    na, nb = _np_pair()
    nr = na < nb
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    fa, fb = _fr_pair()
    fr_res = fa < fb  # ferray RAISES TypeError here -> test fails (divergence)
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


def test_complex_le_computes_lexicographically_like_numpy():
    na, nb = _np_pair()
    nr = na <= nb
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    fa, fb = _fr_pair()
    fr_res = fa <= fb
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


def test_complex_gt_computes_lexicographically_like_numpy():
    na, nb = _np_pair()
    nr = na > nb
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    fa, fb = _fr_pair()
    fr_res = fa > fb
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


def test_complex_ge_computes_lexicographically_like_numpy():
    na, nb = _np_pair()
    nr = na >= nb
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    fa, fb = _fr_pair()
    fr_res = fa >= fb
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


def test_complex_lt_scalar_computes_like_numpy():
    # Scalar RHS path (compare complex masked array to a complex scalar).
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    nr = n < (2 + 0j)
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()

    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    fr_res = f < (2 + 0j)
    assert np.asarray(fr_res.data).tolist() == exp_data


# ---------------------------------------------------------------------------
# GUARD (must stay GREEN): the PERMANENT raises are genuinely permanent —
# numpy ALSO raises for complex bitwise / floor-div / mod. These confirm the
# raise matrix is right where the doc says "numpy raises too" AND the oracle
# agrees, so the divergence above is isolated to the four ordering ops.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "op",
    ["and", "or", "xor", "floordiv", "mod", "invert"],
)
def test_complex_permanent_raise_matches_numpy(op):
    """numpy ALSO raises TypeError for these on complex -> ferray correct."""
    nd = np.ma.array(_DATA_A, mask=_MASK_A)
    fd = fr.ma.array(_DATA_A, mask=_MASK_A)
    ops = {
        "and": lambda a: a & a,
        "or": lambda a: a | a,
        "xor": lambda a: a ^ a,
        "floordiv": lambda a: a // (1 + 0j),
        "mod": lambda a: a % (1 + 0j),
        "invert": lambda a: ~a,
    }
    with pytest.raises(TypeError):
        _ = ops[op](nd)  # confirm numpy raises (oracle)
    with pytest.raises(TypeError):
        _ = ops[op](fd)  # ferray matches


# ---------------------------------------------------------------------------
# COMPUTE (#873 landed): the complex reductions no longer route through
# `to_f64_ma` (which would drop the imaginary part). `mean` keeps the imaginary
# part (complex sum / count, -> complex128); `var`/`std` are the REAL
# `mean(|x - mean|**2)` / its sqrt (-> float64), which is numpy's complex
# variance/std (a REAL quantity, NOT an imag-drop). Was: asserted a TypeError.
# This stays as the R-CODE-4 imag-drop guard: `mean` must match numpy's COMPLEX
# value (not a real collapse), and var/std must match numpy's REAL value.
# ---------------------------------------------------------------------------

def test_complex_mean_keeps_imag_matches_numpy():
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    # The imaginary part is non-zero here; a real collapse would drop it.
    assert n.mean().imag != 0.0
    assert complex(f.mean()) == pytest.approx(complex(n.mean()))


@pytest.mark.parametrize("name", ["var", "std"])
def test_complex_var_std_real_matches_numpy(name):
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    # numpy complex var/std are REAL float64 (mean(|x-mean|**2) / its sqrt).
    assert float(getattr(f, name)()) == pytest.approx(float(getattr(n, name)()))
    assert str(np.asarray(getattr(f, name)()).dtype) == "float64"


# ---------------------------------------------------------------------------
# GUARD (must stay GREEN): complex egress preserves the imaginary part
# everywhere it computes (sum / prod / compressed / __array__ / getitem).
# These confirm no imaginary-part drop on the COMPUTE paths.
# ---------------------------------------------------------------------------

def test_complex_sum_preserves_imag():
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    assert complex(f.sum()) == complex(n.sum())


def test_complex_prod_preserves_imag():
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    assert complex(fr.ma.prod(f)) == complex(n.prod())


def test_complex_array_egress_preserves_imag_and_dtype():
    n = np.ma.array(_DATA_A, mask=_MASK_A)
    f = fr.ma.array(_DATA_A, mask=_MASK_A)
    fe = np.asarray(f)
    assert fe.dtype == np.asarray(n).dtype == np.complex128
    assert fe.tolist() == np.asarray(n).tolist()


def test_complex_eq_masked_position_override_matches_numpy():
    """Control: complex ==/!= mask override IS correct (probed via `.mask`)."""
    na, nb = _np_pair()
    nr = na == nb
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    fa, fb = _fr_pair()
    fr_res = fa == fb
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask
